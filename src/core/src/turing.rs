//! Turing morphogenesis field computed from the DC grid.
//!
//! Implements Difference of Gaussians (DoG) on the DC (block-level) grid to
//! produce a reaction-diffusion field that identifies structural ridges
//! (activator > inhibitor zones). The field is fully reconstructible at the
//! decoder from the decoded DC values — it costs zero bits.

#[allow(unused_imports)]
use crate::calibration;
use crate::golden::PHI;
use rayon::prelude::*;

// ============================================================================
// Public structs
// ============================================================================

/// A predicted contour extracted from the Turing field ridges.
/// Coordinates are in DC-grid space (integer positions).
#[derive(Debug, Clone)]
pub struct PredictedContour {
    /// Grid positions (y, x) along the ridge.
    pub positions: Vec<(usize, usize)>,
    /// Local orientation angle in radians at each position.
    pub orientations: Vec<f64>,
    /// |∇I| at each position (for magnitude prediction).
    pub magnitudes: Vec<f64>,
}

/// Turing morphogenesis field computed from the DC grid.
/// Reconstructible at the decoder from decoded DC — costs zero bits.
#[derive(Debug, Clone)]
pub struct TuringField {
    /// T(x,y) normalized to [0, 1]. Size: grid_h × grid_w.
    pub field: Vec<f64>,
    /// Step modulation factor φ^(-k·T_norm). Size: grid_h × grid_w.
    pub step_modulation: Vec<f64>,
    /// Gradient of the inhibitor field (magnitude). Size: grid_h × grid_w.
    pub inhibitor_gradient: Vec<f64>,
    pub grid_h: usize,
    pub grid_w: usize,
}

// ============================================================================
// Sobel gradient magnitude
// ============================================================================

/// Sobel 3×3 gradient magnitude.
///
/// Interior pixels use the standard Sobel kernel (result divided by 4 for
/// normalization). Border pixels use forward/backward differences scaled ×0.5
/// for parity with the interior normalization.
pub fn sobel_magnitude(grid: &[f64], h: usize, w: usize) -> Vec<f64> {
    assert_eq!(grid.len(), h * w, "sobel_magnitude: grid size mismatch");
    let mut out = vec![0.0f64; h * w];

    out.par_chunks_mut(w).enumerate().for_each(|(r, row_out)| {
        let get = |row: usize, col: usize| -> f64 { grid[row * w + col] };
        let interior_r = r > 0 && r < h - 1;

        for c in 0..w {
            let gx: f64;
            let gy: f64;
            let interior_c = c > 0 && c < w - 1;

            if interior_r && interior_c {
                gx = (get(r - 1, c + 1) + 2.0 * get(r, c + 1) + get(r + 1, c + 1)
                    - get(r - 1, c - 1) - 2.0 * get(r, c - 1) - get(r + 1, c - 1))
                    / 4.0;
                gy = (get(r + 1, c - 1) + 2.0 * get(r + 1, c) + get(r + 1, c + 1)
                    - get(r - 1, c - 1) - 2.0 * get(r - 1, c) - get(r - 1, c + 1))
                    / 4.0;
            } else {
                let r_next = if r + 1 < h { r + 1 } else { r };
                let r_prev = if r > 0 { r - 1 } else { r };
                let c_next = if c + 1 < w { c + 1 } else { c };
                let c_prev = if c > 0 { c - 1 } else { c };
                gx = (get(r, c_next) - get(r, c_prev)) * 0.5;
                gy = (get(r_next, c) - get(r_prev, c)) * 0.5;
            }

            row_out[c] = (gx * gx + gy * gy).sqrt();
        }
    });
    out
}

// ============================================================================
// Separable Gaussian blur
// ============================================================================

/// Separable Gaussian blur with clamp-to-edge border handling.
///
/// The kernel is truncated at radius = ceil(3σ). If sigma < 0.01, the input
/// is returned unchanged (identity pass).
pub fn gaussian_blur_separable(input: &[f64], h: usize, w: usize, sigma: f64) -> Vec<f64> {
    assert_eq!(input.len(), h * w, "gaussian_blur_separable: input size mismatch");

    if sigma < 0.01 {
        return input.to_vec();
    }

    let radius = (3.0 * sigma).ceil() as usize;
    let kernel_size = 2 * radius + 1;

    // Build normalized 1-D Gaussian kernel
    let mut kernel = vec![0.0f64; kernel_size];
    for i in 0..kernel_size {
        let x = i as f64 - radius as f64;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
    }
    let ksum: f64 = kernel.iter().sum();
    for v in kernel.iter_mut() {
        *v /= ksum;
    }

    // Horizontal pass (along columns) — parallel per row
    let mut tmp = vec![0.0f64; h * w];
    tmp.par_chunks_mut(w).enumerate().for_each(|(r, row_out)| {
        for c in 0..w {
            let mut acc = 0.0f64;
            for (ki, &kv) in kernel.iter().enumerate() {
                let src_c = (c as isize + ki as isize - radius as isize)
                    .clamp(0, w as isize - 1) as usize;
                acc += input[r * w + src_c] * kv;
            }
            row_out[c] = acc;
        }
    });

    // Vertical pass (along rows) — parallel per row
    let mut out = vec![0.0f64; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(r, row_out)| {
        for c in 0..w {
            let mut acc = 0.0f64;
            for (ki, &kv) in kernel.iter().enumerate() {
                let src_r = (r as isize + ki as isize - radius as isize)
                    .clamp(0, h as isize - 1) as usize;
                acc += tmp[src_r * w + c] * kv;
            }
            row_out[c] = acc;
        }
    });
    out
}

// ============================================================================
// Normalize to [0, 1]
// ============================================================================

/// Normalize a slice to [0, 1].
///
/// If all values are equal (flat input), returns all zeros to avoid NaN.
pub fn normalize_01(input: &[f64]) -> Vec<f64> {
    let min = input.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < f64::EPSILON {
        return vec![0.0; input.len()];
    }
    let range = max - min;
    input.iter().map(|&v| (v - min) / range).collect()
}

// ============================================================================
// Full Turing field pipeline
// ============================================================================

/// Compute the Turing morphogenesis field from the decoded DC grid.
///
/// Pipeline:
/// 1. Sobel magnitude on raw DC → edge map
/// 2. Gaussian(σ_a) → activator A
/// 3. Gaussian(σ_a × σ_ratio) → inhibitor I
/// 4. DoG = ReLU(A − I)
/// 5. Normalize DoG → T_norm ∈ [0, 1]
/// 6. step_modulation = φ^(−k × T_norm)
/// 7. inhibitor_gradient = Sobel(I)
pub fn compute_turing_field(
    decoded_dc: &[f64],
    grid_h: usize,
    grid_w: usize,
    sigma_a: f64,
    sigma_ratio: f64,
    k_step: f64,
) -> TuringField {
    assert_eq!(
        decoded_dc.len(),
        grid_h * grid_w,
        "compute_turing_field: DC grid size mismatch"
    );

    // Step 1: Sobel edge map from raw DC
    let edge_map = sobel_magnitude(decoded_dc, grid_h, grid_w);

    // Step 2: Activator — blur edge map with σ_a
    let activator = gaussian_blur_separable(&edge_map, grid_h, grid_w, sigma_a);

    // Step 3: Inhibitor — blur edge map with σ_a × σ_ratio
    let sigma_i = sigma_a * sigma_ratio;
    let inhibitor = gaussian_blur_separable(&edge_map, grid_h, grid_w, sigma_i);

    // Step 4: DoG = ReLU(A − I)
    let dog: Vec<f64> = activator
        .iter()
        .zip(inhibitor.iter())
        .map(|(&a, &i)| (a - i).max(0.0))
        .collect();

    // Step 5: Normalize → T_norm
    let t_norm = normalize_01(&dog);

    // Step 6: Step modulation = φ^(−k × T_norm)
    let step_modulation: Vec<f64> = t_norm
        .iter()
        .map(|&t| PHI.powf(-k_step * t))
        .collect();

    // Step 7: Inhibitor gradient
    let inhibitor_gradient = sobel_magnitude(&inhibitor, grid_h, grid_w);

    TuringField {
        field: t_norm,
        step_modulation,
        inhibitor_gradient,
        grid_h,
        grid_w,
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Map T_norm to a bucket index 0–3.
///
/// Buckets:
/// - 0: [0.0, 0.2)
/// - 1: [0.2, 0.5)
/// - 2: [0.5, 0.8)
/// - 3: [0.8, 1.0]
pub fn turing_bucket(t_norm: f64) -> usize {
    if t_norm < 0.2 {
        0
    } else if t_norm < 0.5 {
        1
    } else if t_norm < 0.8 {
        2
    } else {
        3
    }
}

/// Predict the coefficient magnitude from the inhibitor gradient.
///
/// Returns `inhibitor_grad × mag_calibration`.
pub fn predicted_magnitude(inhibitor_grad: f64, mag_calibration: f64) -> f64 {
    inhibitor_grad * mag_calibration
}

// ============================================================================
// Ridge tracing
// ============================================================================

/// Walk along the T_norm ridge from a seed in one direction.
///
/// - `sign`: +1.0 for forward, -1.0 for backward.
/// - Gradient direction = perpendicular to ∇T (i.e. along the ridge).
/// - Continuity: flip direction if dot product with previous step is negative.
/// - Stops at borders, already-visited pixels, T < threshold, or 500 steps.
fn trace_half(
    t_norm: &[f64],
    h: usize,
    w: usize,
    start_y: usize,
    start_x: usize,
    sign: f64,
    visited: &mut [bool],
    threshold: f64,
) -> Vec<(usize, usize)> {
    let mut path: Vec<(usize, usize)> = Vec::new();
    let mut cy = start_y as isize;
    let mut cx = start_x as isize;
    let mut prev_dy = 0.0f64;
    let mut prev_dx = 0.0f64;

    for _ in 0..500 {
        if cy < 0 || cy >= h as isize || cx < 0 || cx >= w as isize {
            break;
        }
        let uy = cy as usize;
        let ux = cx as usize;
        let idx = uy * w + ux;

        if t_norm[idx] < threshold {
            break;
        }
        if visited[idx] {
            break;
        }

        visited[idx] = true;
        path.push((uy, ux));

        // Central differences for ∇T at current position (forward/backward at borders)
        let r_next = if uy + 1 < h { uy + 1 } else { uy };
        let r_prev = if uy > 0 { uy - 1 } else { uy };
        let c_next = if ux + 1 < w { ux + 1 } else { ux };
        let c_prev = if ux > 0 { ux - 1 } else { ux };

        let grad_y = (t_norm[r_next * w + ux] - t_norm[r_prev * w + ux])
            / (if r_next != r_prev { 2.0 } else { 1.0 });
        let grad_x = (t_norm[uy * w + c_next] - t_norm[uy * w + c_prev])
            / (if c_next != c_prev { 2.0 } else { 1.0 });

        // Perpendicular to ∇T = ridge direction: (-grad_x * sign, grad_y * sign)
        let mut dy = -grad_x * sign;
        let mut dx = grad_y * sign;

        // Normalize
        let len = (dy * dy + dx * dx).sqrt();
        if len < 1e-12 {
            break;
        }
        dy /= len;
        dx /= len;

        // Continuity: avoid backtracking
        if prev_dy != 0.0 || prev_dx != 0.0 {
            let dot = dy * prev_dy + dx * prev_dx;
            if dot < 0.0 {
                dy = -dy;
                dx = -dx;
            }
        }
        prev_dy = dy;
        prev_dx = dx;

        // Step to nearest integer grid cell
        cy += dy.round() as isize;
        cx += dx.round() as isize;
    }

    path
}

/// Extract predicted contours from the Turing field by following ridges.
///
/// Algorithm:
/// 1. NMS: find pixels where T_norm > threshold AND T_norm >= all 8 neighbours.
/// 2. Sort seeds in raster order (y, x) for determinism.
/// 3. For each unvisited seed, trace bidirectionally.
/// 4. Combine both halves (backward reversed + forward).
/// 5. Keep only contours with ≥ 3 positions.
/// 6. Compute orientations from finite differences along the path.
/// 7. Magnitudes from inhibitor_gradient if provided, else from t_norm.
pub fn trace_ridges(
    t_norm: &[f64],
    h: usize,
    w: usize,
    inhibitor_gradient: Option<&[f64]>,
) -> Vec<PredictedContour> {
    use crate::calibration::TURING_RIDGE_THRESHOLD;

    assert_eq!(t_norm.len(), h * w);
    if let Some(ig) = inhibitor_gradient {
        assert_eq!(ig.len(), h * w);
    }

    let threshold = TURING_RIDGE_THRESHOLD;

    // Step 1: NMS — find local maxima over 8-neighbourhood
    let mut seeds: Vec<(usize, usize)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let v = t_norm[y * w + x];
            if v <= threshold {
                continue;
            }
            let mut is_max = true;
            'outer: for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let ny = y as isize + dy;
                    let nx = x as isize + dx;
                    if ny < 0 || ny >= h as isize || nx < 0 || nx >= w as isize {
                        continue;
                    }
                    if t_norm[ny as usize * w + nx as usize] > v {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            if is_max {
                seeds.push((y, x));
            }
        }
    }
    // Seeds are already in raster order (y-major iteration above)

    // Step 2–4: trace each seed bidirectionally
    let mut visited = vec![false; h * w];
    let mut contours: Vec<PredictedContour> = Vec::new();

    for (sy, sx) in seeds {
        if visited[sy * w + sx] {
            continue;
        }

        // Forward half
        let fwd = trace_half(t_norm, h, w, sy, sx, 1.0, &mut visited, threshold);
        // Backward half: re-mark the seed as unvisited so trace_half can visit it
        // (forward already visited it; we start from the same seed going opposite way)
        // Actually, forward marked the seed visited. We need to restart from seed for backward.
        // We allow the seed to be re-entered by the backward trace by not checking it again —
        // instead start the backward trace at the seed without re-marking it.
        // Simpler: mark seed unvisited temporarily for the backward pass, then re-mark.
        visited[sy * w + sx] = false;
        let bwd = trace_half(t_norm, h, w, sy, sx, -1.0, &mut visited, threshold);
        // bwd[0] == seed, fwd[0] == seed — combine: reverse bwd (skipping seed) + fwd
        let mut positions: Vec<(usize, usize)> = Vec::new();
        for &p in bwd.iter().skip(1).rev() {
            positions.push(p);
        }
        for &p in &fwd {
            positions.push(p);
        }

        if positions.len() < 3 {
            continue;
        }

        let n = positions.len();

        // Orientations: finite differences along path
        let mut orientations = vec![0.0f64; n];
        for i in 0..n {
            let (y0, x0) = if i > 0 { positions[i - 1] } else { positions[0] };
            let (y1, x1) = if i + 1 < n { positions[i + 1] } else { positions[n - 1] };
            let dy = y1 as f64 - y0 as f64;
            let dx = x1 as f64 - x0 as f64;
            orientations[i] = dy.atan2(dx);
        }

        // Magnitudes
        let magnitudes: Vec<f64> = positions
            .iter()
            .map(|&(py, px)| {
                let idx = py * w + px;
                if let Some(ig) = inhibitor_gradient {
                    ig[idx]
                } else {
                    t_norm[idx]
                }
            })
            .collect();

        contours.push(PredictedContour {
            positions,
            orientations,
            magnitudes,
        });
    }

    contours
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // 1. Sobel on a flat grid → all zeros
    // -------------------------------------------------------------------------
    #[test]
    fn test_sobel_flat_grid() {
        let h = 5;
        let w = 5;
        let grid = vec![42.0f64; h * w];
        let mag = sobel_magnitude(&grid, h, w);
        for &v in &mag {
            assert!(
                v.abs() < 1e-10,
                "flat grid must produce zero gradient, got {}",
                v
            );
        }
    }

    // -------------------------------------------------------------------------
    // 2. Sobel on horizontal step edge → strong gradient at centre row
    // -------------------------------------------------------------------------
    #[test]
    fn test_sobel_horizontal_edge() {
        let h = 5;
        let w = 5;
        // Top 2 rows = 0, bottom 3 rows = 100
        let mut grid = vec![0.0f64; h * w];
        for r in 2..h {
            for c in 0..w {
                grid[r * w + c] = 100.0;
            }
        }
        let mag = sobel_magnitude(&grid, h, w);

        // Row 2 (interior, straddles the step) should have the strongest gradient
        let row2_max = (0..w).map(|c| mag[2 * w + c]).fold(0.0f64, f64::max);
        let row0_max = (0..w).map(|c| mag[0 * w + c]).fold(0.0f64, f64::max);

        assert!(
            row2_max > row0_max,
            "row 2 (edge) gradient ({}) should exceed row 0 gradient ({})",
            row2_max,
            row0_max
        );
        assert!(
            row2_max > 1.0,
            "row 2 gradient ({}) should be notably non-zero",
            row2_max
        );
    }

    // -------------------------------------------------------------------------
    // 3. Gaussian blur with near-zero sigma → identity
    // -------------------------------------------------------------------------
    #[test]
    fn test_gaussian_blur_identity_sigma_zero() {
        let h = 5;
        let w = 5;
        let input: Vec<f64> = (0..(h * w)).map(|i| i as f64).collect();
        let out = gaussian_blur_separable(&input, h, w, 0.005);
        for (orig, blurred) in input.iter().zip(out.iter()) {
            assert!(
                (orig - blurred).abs() < 1e-10,
                "near-zero sigma should preserve input: {} vs {}",
                orig,
                blurred
            );
        }
    }

    // -------------------------------------------------------------------------
    // 4. Gaussian blur smooths a centre spike; total energy conserved
    // -------------------------------------------------------------------------
    #[test]
    fn test_gaussian_blur_smooths() {
        let h = 7;
        let w = 7;
        let mut input = vec![0.0f64; h * w];
        // Single spike at centre
        input[3 * w + 3] = 100.0;

        let out = gaussian_blur_separable(&input, h, w, 1.0);

        // Centre must be reduced
        assert!(
            out[3 * w + 3] < 100.0,
            "centre should be reduced after blur, got {}",
            out[3 * w + 3]
        );

        // Immediate neighbours must gain energy
        let neighbour = out[3 * w + 4]; // right of centre
        assert!(
            neighbour > 0.0,
            "neighbour should gain energy, got {}",
            neighbour
        );

        // Total energy must be conserved (sum unchanged)
        let energy_in: f64 = input.iter().sum();
        let energy_out: f64 = out.iter().sum();
        assert!(
            (energy_in - energy_out).abs() < 1e-6,
            "energy must be conserved: in={} out={}",
            energy_in,
            energy_out
        );
    }

    // -------------------------------------------------------------------------
    // 5. normalize_01 on flat input → all zeros (not NaN)
    // -------------------------------------------------------------------------
    #[test]
    fn test_normalize_01_flat() {
        let input = vec![7.0f64; 10];
        let out = normalize_01(&input);
        for &v in &out {
            assert!(
                v.abs() < 1e-10 && !v.is_nan(),
                "flat input should normalize to 0, got {}",
                v
            );
        }
    }

    // -------------------------------------------------------------------------
    // 6. normalize_01 on [10, 20, 30, 40] → [0, 1/3, 2/3, 1]
    // -------------------------------------------------------------------------
    #[test]
    fn test_normalize_01_range() {
        let input = vec![10.0f64, 20.0, 30.0, 40.0];
        let out = normalize_01(&input);
        let expected = [0.0f64, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "index {}: expected {}, got {}",
                i,
                exp,
                got
            );
        }
    }

    // -------------------------------------------------------------------------
    // 7. compute_turing_field on 8×8 with horizontal edge → structural ridge
    // -------------------------------------------------------------------------
    #[test]
    fn test_compute_turing_field_basic() {
        let h = 8usize;
        let w = 8usize;
        // Top half = 0, bottom half = 200 — clear horizontal edge at row 4
        let mut dc = vec![0.0f64; h * w];
        for r in 4..h {
            for c in 0..w {
                dc[r * w + c] = 200.0;
            }
        }

        let tf = compute_turing_field(
            &dc,
            h,
            w,
            calibration::TURING_SIGMA_A,
            calibration::TURING_SIGMA_RATIO,
            calibration::TURING_K_STEP,
        );

        assert_eq!(tf.grid_h, h);
        assert_eq!(tf.grid_w, w);
        assert_eq!(tf.field.len(), h * w);
        assert_eq!(tf.step_modulation.len(), h * w);
        assert_eq!(tf.inhibitor_gradient.len(), h * w);

        // step_modulation must be ≤ 1.0 everywhere (PHI^(-k*T) ≤ 1 when T ≥ 0)
        for &sm in &tf.step_modulation {
            assert!(
                sm <= 1.0 + 1e-10,
                "step_modulation must be ≤ 1.0, got {}",
                sm
            );
            assert!(sm > 0.0, "step_modulation must be positive, got {}", sm);
        }

        // T_norm must be in [0, 1]
        for &t in &tf.field {
            assert!((0.0..=1.0 + 1e-10).contains(&t), "T_norm out of range: {}", t);
        }

        // The edge region (rows 3–4) should have higher average T than the
        // uniform interior (row 0 or row 7)
        let edge_t_avg: f64 = (0..w).map(|c| tf.field[4 * w + c]).sum::<f64>() / w as f64;
        let top_t_avg: f64 = (0..w).map(|c| tf.field[0 * w + c]).sum::<f64>() / w as f64;
        let bot_t_avg: f64 =
            (0..w).map(|c| tf.field[(h - 1) * w + c]).sum::<f64>() / w as f64;

        assert!(
            edge_t_avg >= top_t_avg,
            "edge row should have T ≥ flat top row: edge={} top={}",
            edge_t_avg,
            top_t_avg
        );
        assert!(
            edge_t_avg >= bot_t_avg,
            "edge row should have T ≥ flat bottom row: edge={} bot={}",
            edge_t_avg,
            bot_t_avg
        );
    }

    // -------------------------------------------------------------------------
    // 8. trace_ridges on a 12×12 grid with a horizontal edge → detects ridge
    // -------------------------------------------------------------------------
    #[test]
    fn test_trace_ridges_horizontal_edge() {
        // 12×12 DC grid: top=50, bottom=200 → strong horizontal edge at row 6
        let mut dc = vec![50.0; 144];
        for y in 6..12 {
            for x in 0..12 {
                dc[y * 12 + x] = 200.0;
            }
        }
        let tf = compute_turing_field(&dc, 12, 12, 1.5, 2.618, 0.5);
        let ridges = trace_ridges(&tf.field, 12, 12, Some(&tf.inhibitor_gradient));
        assert!(!ridges.is_empty(), "should detect a ridge at the horizontal edge");
        let r = &ridges[0];
        assert!(r.positions.len() >= 3, "ridge should span at least 3 grid cells");
    }

    // -------------------------------------------------------------------------
    // 9. trace_ridges on a flat image → no ridges
    // -------------------------------------------------------------------------
    #[test]
    fn test_trace_ridges_flat_image() {
        let dc = vec![100.0; 64]; // 8×8
        let tf = compute_turing_field(&dc, 8, 8, 1.5, 2.618, 0.5);
        let ridges = trace_ridges(&tf.field, 8, 8, None);
        assert!(ridges.is_empty(), "flat image should produce no ridges");
    }

    // -------------------------------------------------------------------------
    // 10. trace_ridges is deterministic (same input → identical output)
    // -------------------------------------------------------------------------
    #[test]
    fn test_trace_ridges_deterministic() {
        let mut dc = vec![80.0; 100]; // 10×10
        for y in 4..6 {
            for x in 0..10 {
                dc[y * 10 + x] = 220.0;
            }
        }
        let tf = compute_turing_field(&dc, 10, 10, 1.5, 2.618, 0.5);
        let r1 = trace_ridges(&tf.field, 10, 10, None);
        let r2 = trace_ridges(&tf.field, 10, 10, None);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.positions.len(), b.positions.len());
            for (pa, pb) in a.positions.iter().zip(b.positions.iter()) {
                assert_eq!(pa.0, pb.0);
                assert_eq!(pa.1, pb.1);
            }
        }
    }
}
