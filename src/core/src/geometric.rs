/// Geometric module: primitives (segments, arcs) for wavelet detail band prediction.
/// Encodes contours using geometric primitives + residual.
///
/// Fibonacci/phi ratios structure the computations:
/// - Force: quantized at Fibonacci levels (Weber-Fechner)
/// - Profile: golden bell along primitives (not constant force)
/// - Arcs: sampling at the golden angle (137.5 degrees)
use ndarray::Array2;

/// Fibonacci force levels (natural logarithmic quantization).
/// Perception follows Weber-Fechner: delta_perception = k * log(stimulus).
/// Fibonacci spacing approximates a geometric progression with ratio phi.
const FIB_FORCE_LEVELS: [i8; 8] = [1, 2, 3, 5, 8, 13, 21, 34];

/// Snap a continuous force to the nearest Fibonacci level.
#[inline]
fn snap_to_fibonacci(force: f64) -> i8 {
    let abs_f = force.abs();
    let sign = if force >= 0.0 { 1i8 } else { -1i8 };
    if abs_f < 0.5 { return 0; }
    let mut best = FIB_FORCE_LEVELS[0];
    let mut best_dist = (abs_f - best as f64).abs();
    for &level in &FIB_FORCE_LEVELS[1..] {
        let dist = (abs_f - level as f64).abs();
        if dist < best_dist {
            best = level;
            best_dist = dist;
        }
        if level as f64 > abs_f * 2.0 { break; } // no need to search further
    }
    sign * best
}

/// Calibrated CDF 9/7 profile: localized spike (FWHM=1 pixel).
/// The CDF 9/7 filter produces a damped Dirac, not a golden bell.
/// Empirically calibrated CDF 9/7 lookup table.
/// Index = angle in degrees / 5 (0..18 -> 0 to 90 degrees).
/// Values = amplitude fraction (sqrt(energy)) per band.
const CALIB_LH: [f64; 19] = [
    1.0000, 0.9569, 0.9113, 0.8746, 0.8188, 0.7652, 0.7319, 0.6405,
    0.5098, 0.3977, 0.4291, 0.4101, 0.4402, 0.4433, 0.4304, 0.3564,
    0.2945, 0.2078, 0.1828,
];
const CALIB_HL: [f64; 19] = [
    0.0000, 0.2078, 0.2945, 0.3564, 0.4304, 0.4433, 0.4402, 0.4101,
    0.4291, 0.4030, 0.5098, 0.6405, 0.7319, 0.7652, 0.8188, 0.8746,
    0.9113, 0.9569, 0.9750,
];
const CALIB_HH: [f64; 19] = [
    0.0000, 0.2028, 0.2878, 0.3289, 0.3800, 0.4669, 0.5201, 0.6493,
    0.7457, 0.8242, 0.7457, 0.6493, 0.5201, 0.4669, 0.3800, 0.3289,
    0.2878, 0.2028, 0.1259,
];

/// Interpolate the calibrated lookup table for a given angle.
#[inline]
fn calibrated_weights(theta: f64) -> (f64, f64, f64) {
    // Normalize theta to [0, pi/2] by symmetry
    let angle_deg = theta.abs().to_degrees() % 180.0;
    let angle_norm = if angle_deg > 90.0 { 180.0 - angle_deg } else { angle_deg };

    // Fractional index in the table (5-degree step)
    let idx_f = angle_norm / 5.0;
    let idx0 = (idx_f.floor() as usize).min(17);
    let idx1 = idx0 + 1;
    let frac = idx_f - idx0 as f64;

    // Linear interpolation
    let w_lh = CALIB_LH[idx0] * (1.0 - frac) + CALIB_LH[idx1] * frac;
    let w_hl = CALIB_HL[idx0] * (1.0 - frac) + CALIB_HL[idx1] * frac;
    let w_hh = CALIB_HH[idx0] * (1.0 - frac) + CALIB_HH[idx1] * frac;

    (w_lh, w_hl, w_hh)
}

// ============================================================
// Data structures
// ============================================================

#[derive(Debug, Clone)]
pub enum Primitive {
    /// Superstring segment: straight contour with phi oscillation.
    /// The force oscillates at 1/(2*phi) cycles/pixel along the segment.
    Segment { x1: i16, y1: i16, x2: i16, y2: i16, amplitude: i8, phase: i8 },
    /// Superstring arc: curved contour with phi oscillation.
    Arc { cx: i16, cy: i16, radius: u16, theta_start: i8, theta_end: i8, amplitude: i8, phase: i8 },
}

/// Natural oscillation frequency of the CDF 9/7 filter: 1/(2*phi) cycles/pixel.
/// Empirical discovery: coefficients along contours oscillate at this frequency.
const PHI_FREQ: f64 = 0.30901699437494742; // 1/(2*phi)

impl Primitive {
    /// Serialization cost in bytes of the primitive.
    pub fn byte_cost(&self) -> usize {
        match self {
            Primitive::Segment { .. } => 11, // 1(type) + 4*i16 + amplitude(i8) + phase(i8)
            Primitive::Arc { .. } => 13,     // 1(type) + 2*i16 + u16 + 2*i8 + amplitude(i8) + phase(i8)
        }
    }
}

/// Compute the oscillating force at pixel i along a primitive.
/// force(i) = amplitude * sin(2*pi * PHI_FREQ * i + phase)
#[inline]
fn oscillating_force(amplitude: f64, phase: f64, pixel_index: usize) -> f64 {
    amplitude * (2.0 * std::f64::consts::PI * PHI_FREQ * pixel_index as f64 + phase).sin()
}

// ============================================================
// Renderer: projection of primitives into LH/HL/HH bands
// ============================================================

/// Project a list of primitives into the three wavelet detail bands.
/// CDF 9/7 conventions:
///   LH (horizontal contours, vertical gradient): force * |cos(theta)|
///   HL (vertical contours, horizontal gradient): force * |sin(theta)|
///   HH (diagonals):                             force * |sin(theta) * cos(theta)|
pub fn render_primitives(
    primitives: &[Primitive],
    h_lh: usize, w_lh: usize,
    h_hl: usize, w_hl: usize,
    h_hh: usize, w_hh: usize,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let mut lh = Array2::<f64>::zeros((h_lh, w_lh));
    let mut hl = Array2::<f64>::zeros((h_hl, w_hl));
    let mut hh = Array2::<f64>::zeros((h_hh, w_hh));

    for prim in primitives {
        match prim {
            Primitive::Segment { x1, y1, x2, y2, amplitude, phase } => {
                let dx = (*x2 - *x1) as f64;
                let dy = (*y2 - *y1) as f64;
                let theta = dy.atan2(dx);
                let amp = *amplitude as f64;
                let ph = *phase as f64 * std::f64::consts::PI / 128.0;

                // Calibrated CDF 9/7 weights
                let (w_lh_c, w_hl_c, w_hh_c) = calibrated_weights(theta);

                // Bresenham rasterization with phi oscillation
                let pixels = bresenham_pixels(*x1, *y1, *x2, *y2);
                for (i, &(px, py)) in pixels.iter().enumerate() {
                    // Oscillating force: amp * sin(2*pi * PHI_FREQ * i + phase)
                    let f = oscillating_force(amp, ph, i);
                    accumulate(&mut lh, py, px, f * w_lh_c, h_lh, w_lh);
                    accumulate(&mut hl, py, px, f * w_hl_c, h_hl, w_hl);
                    accumulate(&mut hh, py, px, f * w_hh_c, h_hh, w_hh);
                }
            }
            Primitive::Arc { cx, cy, radius, theta_start, theta_end, amplitude, phase } => {
                let amp = *amplitude as f64;
                let ph = *phase as f64 * std::f64::consts::PI / 128.0;
                let ts = *theta_start as f64 * std::f64::consts::PI / 128.0;
                let te = *theta_end as f64 * std::f64::consts::PI / 128.0;
                let r = *radius as f64;
                let cx_f = *cx as f64;
                let cy_f = *cy as f64;

                let arc_len = (te - ts).abs() * r;
                let n_steps = ((arc_len * 2.0).ceil() as usize).max(8).min(2000);

                for i in 0..=n_steps {
                    let frac = i as f64 / n_steps as f64;
                    let t = ts + (te - ts) * frac;
                    let px = (cx_f + r * t.cos()).round() as i32;
                    let py = (cy_f + r * t.sin()).round() as i32;
                    let tangent = t + std::f64::consts::FRAC_PI_2;
                    let (w_lh_c, w_hl_c, w_hh_c) = calibrated_weights(tangent);
                    // Oscillating force along the arc
                    let f = oscillating_force(amp, ph, i);
                    accumulate(&mut lh, py, px, f * w_lh_c, h_lh, w_lh);
                    accumulate(&mut hl, py, px, f * w_hl_c, h_hl, w_hl);
                    accumulate(&mut hh, py, px, f * w_hh_c, h_hh, w_hh);
                }
            }
        }
    }

    (lh, hl, hh)
}

/// Accumulate a value in an Array2 if coordinates are valid.
#[inline]
fn accumulate(band: &mut Array2<f64>, row: i32, col: i32, val: f64, h: usize, w: usize) {
    if row >= 0 && row < h as i32 && col >= 0 && col < w as i32 {
        band[[row as usize, col as usize]] += val;
    }
}

/// Rasterize a segment into pixels (Bresenham algorithm).
fn bresenham_pixels(x1: i16, y1: i16, x2: i16, y2: i16) -> Vec<(i32, i32)> {
    let mut x1 = x1 as i32;
    let mut y1 = y1 as i32;
    let x2 = x2 as i32;
    let y2 = y2 as i32;

    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    let sx = if x2 > x1 { 1i32 } else { -1 };
    let sy = if y2 > y1 { 1i32 } else { -1 };
    let mut err = dx - dy;
    let mut pixels = Vec::new();

    loop {
        pixels.push((x1, y1));
        if x1 == x2 && y1 == y2 { break; }
        let e2 = 2 * err;
        if e2 > -dy { err -= dy; x1 += sx; }
        if e2 < dx  { err += dx; y1 += sy; }
    }
    pixels
}

// ============================================================
// Connected components (8-connectivity) via Union-Find
// ============================================================

/// 3x3 morphological dilation, repeated `passes` times.
/// Fills gaps of 1-2 pixels between adjacent clusters.
fn dilate_mask(mask: &[bool], h: usize, w: usize, passes: usize) -> Vec<bool> {
    let mut current = mask.to_vec();
    let mut next = vec![false; h * w];

    for _ in 0..passes {
        for r in 0..h {
            for c in 0..w {
                if current[r * w + c] {
                    next[r * w + c] = true;
                    continue;
                }
                // Check 8 neighbors
                let mut has_neighbor = false;
                for dr in -1i32..=1 {
                    for dc in -1i32..=1 {
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr >= 0 && nr < h as i32 && nc >= 0 && nc < w as i32 {
                            if current[nr as usize * w + nc as usize] {
                                has_neighbor = true;
                            }
                        }
                    }
                }
                next[r * w + c] = has_neighbor;
            }
        }
        std::mem::swap(&mut current, &mut next);
        next.fill(false);
    }

    current
}

/// Compute connected components with 8-connectivity.
/// Returns (label_map, n_components).
/// label_map[i] = u32::MAX if mask[i] = false.
fn connected_components_8(mask: &[bool], h: usize, w: usize) -> (Vec<u32>, usize) {
    let n = h * w;
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path compression (path halving)
            x = parent[x];
        }
        x
    }

    fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb { parent[rb] = ra; }
    }

    // 8-neighborhood connections: right, down, down-right, down-left
    for r in 0..h {
        for c in 0..w {
            let idx = r * w + c;
            if !mask[idx] { continue; }

            // Right
            if c + 1 < w && mask[idx + 1] { union(&mut parent, idx, idx + 1); }
            // Down
            if r + 1 < h && mask[(r + 1) * w + c] { union(&mut parent, idx, (r + 1) * w + c); }
            // Down-right
            if r + 1 < h && c + 1 < w && mask[(r + 1) * w + c + 1] {
                union(&mut parent, idx, (r + 1) * w + c + 1);
            }
            // Down-left
            if r + 1 < h && c > 0 && mask[(r + 1) * w + c - 1] {
                union(&mut parent, idx, (r + 1) * w + c - 1);
            }
        }
    }

    // Full compression
    for i in 0..n { find(&mut parent, i); }

    // Compact renumbering of roots
    let mut root_to_label = vec![u32::MAX; n];
    let mut next_label = 0u32;
    let mut label_map = vec![u32::MAX; n];

    for i in 0..n {
        if !mask[i] { continue; }
        let root = parent[i];
        if root_to_label[root] == u32::MAX {
            root_to_label[root] = next_label;
            next_label += 1;
        }
        label_map[i] = root_to_label[root];
    }

    (label_map, next_label as usize)
}

// ============================================================
// Weighted PCA segment fitting
// ============================================================

/// Fit a segment onto a point cloud (row, col, amplitude).
/// Uses weighted PCA to find the principal direction.
/// Returns (x1, y1, x2, y2, force) or None if too few points.
fn fit_segment(points: &[(usize, usize, f64)]) -> Option<(i16, i16, i16, i16, f64, f64)> {
    if points.len() < 3 { return None; }

    // Weights = absolute value of amplitude
    let weights: Vec<f64> = points.iter().map(|(_, _, a)| a.abs().max(1e-9)).collect();
    let w_sum: f64 = weights.iter().sum();

    // Weighted center of mass (col = x, row = y)
    let cx: f64 = points.iter().zip(&weights).map(|((_, c, _), w)| *c as f64 * w).sum::<f64>() / w_sum;
    let cy: f64 = points.iter().zip(&weights).map(|((r, _, _), w)| *r as f64 * w).sum::<f64>() / w_sum;

    // Weighted covariance matrix
    let mut cxx = 0.0f64;
    let mut cxy = 0.0f64;
    let mut cyy = 0.0f64;
    for ((r, c, _), w) in points.iter().zip(&weights) {
        let dx = *c as f64 - cx;
        let dy = *r as f64 - cy;
        cxx += w * dx * dx;
        cxy += w * dx * dy;
        cyy += w * dy * dy;
    }
    cxx /= w_sum;
    cxy /= w_sum;
    cyy /= w_sum;

    // Principal eigenvector (2x2 eigendecomposition)
    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = ((trace * trace / 4.0) - det).max(0.0).sqrt();
    let lambda_max = trace / 2.0 + disc;

    // Principal direction
    let dir_x;
    let dir_y;
    if cxy.abs() > 1e-12 {
        let norm = ((lambda_max - cyy).powi(2) + cxy * cxy).sqrt();
        dir_x = (lambda_max - cyy) / norm;
        dir_y = cxy / norm;
    } else {
        if cxx >= cyy { dir_x = 1.0; dir_y = 0.0; }
        else          { dir_x = 0.0; dir_y = 1.0; }
    }

    // Projection of points onto principal direction
    let projections: Vec<f64> = points.iter()
        .map(|(r, c, _)| (*c as f64 - cx) * dir_x + (*r as f64 - cy) * dir_y)
        .collect();

    let t_min = projections.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = projections.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let x1 = (cx + t_min * dir_x).round() as i16;
    let y1 = (cy + t_min * dir_y).round() as i16;
    let x2 = (cx + t_max * dir_x).round() as i16;
    let y2 = (cy + t_max * dir_y).round() as i16;

    // Amplitude et phase par correlation avec le template d'oscillation phi.
    // On cherche la phase qui maximise la correlation entre les coefficients
    // reels et le pattern sin(2*pi * PHI_FREQ * i + phase).
    let n = points.len();
    let mut sum_sin = 0.0f64;
    let mut sum_cos = 0.0f64;
    for (i, &(_, _, amp)) in points.iter().enumerate() {
        let angle = 2.0 * std::f64::consts::PI * PHI_FREQ * i as f64;
        sum_sin += amp * angle.sin();
        sum_cos += amp * angle.cos();
    }
    let raw_amplitude = (sum_sin * sum_sin + sum_cos * sum_cos).sqrt() * 2.0 / n as f64;
    let raw_phase = sum_cos.atan2(sum_sin);

    Some((x1, y1, x2, y2, raw_amplitude, raw_phase))
}

// ============================================================
// Arc fitting: Taubin method
// ============================================================

/// Fit a circular arc using the Taubin algebraic method.
/// theta encoded as i8 * pi/128.
/// Returns (cx, cy, radius, theta_start, theta_end, force) or None on failure.
fn fit_arc(points: &[(usize, usize, f64)]) -> Option<(i16, i16, u16, i8, i8, f64)> {
    if points.len() < 5 { return None; }

    let n = points.len() as f64;

    // Means
    let mx: f64 = points.iter().map(|(_, c, _)| *c as f64).sum::<f64>() / n;
    let my: f64 = points.iter().map(|(r, _, _)| *r as f64).sum::<f64>() / n;

    // Centered variables u = x - mx, v = y - my
    let us: Vec<f64> = points.iter().map(|(_, c, _)| *c as f64 - mx).collect();
    let vs: Vec<f64> = points.iter().map(|(r, _, _)| *r as f64 - my).collect();
    let zs: Vec<f64> = us.iter().zip(&vs).map(|(u, v)| u * u + v * v).collect();

    // Moments for the Taubin method
    let zmean = zs.iter().sum::<f64>() / n;

    let mut mxx = 0.0f64; let mut myy = 0.0f64; let mut mxy = 0.0f64;
    let mut mxz = 0.0f64; let mut myz = 0.0f64; let mut mzz = 0.0f64;
    for i in 0..points.len() {
        let u = us[i]; let v = vs[i]; let z = zs[i];
        mxx += u * u; myy += v * v; mxy += u * v;
        mxz += u * z; myz += v * z; mzz += z * z;
    }
    mxx /= n; myy /= n; mxy /= n;
    mxz /= n; myz /= n; mzz /= n;

    // Matrix M and Taubin constraint
    // Find the minimal eigenvector of M - eta * N
    // Using Newton iteration on the characteristic polynomial

    // Characteristic polynomial coefficients of M - eta*N (simplified version)
    // We solve the Taubin algebraic problem directly
    let _cov_xy = mxx + myy;
    let _det_xy = mxx * myy - mxy * mxy;
    let _cov_xz = mxz;
    let _cov_yz = myz;

    // Normal equations (algebraic least squares + Taubin)
    // Direct solution: (B, C, D) by least squares in Bi*xi + Ci*yi + Di = -(xi^2+yi^2)
    // then center = (-B/2, -C/2), radius = sqrt(B^2/4 + C^2/4 - D)
    // This formulation is the simple "algebraic fit" (Pratt/Bookstein)

    // 3x3 normal matrix
    let s11 = mxx;
    let s12 = mxy;
    let s13 = mxz / 2.0;  // sum(u * z) / n
    let s22 = myy;
    let s23 = myz / 2.0;
    let s33 = mzz;

    // RHS vector for eq B*u + C*v + D = -(u^2+v^2)/2 => no; direct formula:
    // We use the formulation: find (a,b,c) minimizing sum((a*u+b*v+c - z)^2)
    // => a = sum(u*z)/sum(u^2), with the moments above

    // Direct approach Coope 1993 / circleFit simple
    // [sum(u^2), sum(u*v), sum(u)] [a]   [sum(u*z)/2]
    // [sum(u*v), sum(v^2), sum(v)] [b] = [sum(v*z)/2]
    // [sum(u),   sum(v),   n     ] [c]   [sum(z)/2  ]
    // centre = (a, b) + (mx, my), rayon = sqrt(a^2+b^2+c)

    let su  = us.iter().sum::<f64>() / n;  // 0 car centrees
    let sv  = vs.iter().sum::<f64>() / n;  // 0
    let _su2 = mxx; let _sv2 = myy; let _suv = mxy;
    let suz = mxz;
    let svz = myz;
    let sz2 = zmean;

    // With perfect centering su=0, sv=0:
    // [mxx, mxy, 0] [a]   [suz/2]
    // [mxy, myy, 0] [b] = [svz/2]
    // [0,   0,   1] [c]   [sz2/2]
    // => c = sz2/2
    // [mxx, mxy][a] = [suz/2]
    // [mxy, myy][b]   [svz/2]

    let _ = (s11, s12, s13, s22, s23, s33, su, sv, sz2); // silence warnings

    let det2 = mxx * myy - mxy * mxy;
    if det2.abs() < 1e-12 { return None; }

    let a = (myy * suz / 2.0 - mxy * svz / 2.0) / det2;
    let b = (mxx * svz / 2.0 - mxy * suz / 2.0) / det2;
    let c = zmean / 2.0;

    // Rayon^2 = a^2 + b^2 + c
    let r2 = a * a + b * b + c;
    if r2 <= 0.0 { return None; }
    let radius = r2.sqrt();

    // Center in original coordinates
    let center_x = a + mx;
    let center_y = b + my;

    // Validation
    if radius > 10000.0 || radius < 1.0 { return None; }

    // Compute theta_start and theta_end angles
    let angles: Vec<f64> = points.iter()
        .map(|(r, c, _)| {
            let dy = *r as f64 - center_y;
            let dx = *c as f64 - center_x;
            dy.atan2(dx)
        })
        .collect();

    // Find angular extent (min/max)
    let theta_start_f = angles.iter().copied().fold(f64::INFINITY, f64::min);
    let theta_end_f   = angles.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Encode as i8 * pi/128
    let encode_angle = |a: f64| -> i8 {
        (a * 128.0 / std::f64::consts::PI).round().clamp(-128.0, 127.0) as i8
    };

    let theta_start = encode_angle(theta_start_f);
    let theta_end   = encode_angle(theta_end_f);

    let cx = center_x.round() as i16;
    let cy = center_y.round() as i16;
    let rad = radius.round() as u16;

    // Force = median of amplitudes
    let mut amps: Vec<f64> = points.iter().map(|(_, _, a)| *a).collect();
    amps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let force = amps[amps.len() / 2];

    Some((cx, cy, rad, theta_start, theta_end, force))
}

// ============================================================
// Greedy primitive extraction
// ============================================================

/// Extract geometric primitives from the three detail bands.
/// Works on all three bands simultaneously (cross-band).
/// Returns (primitives, residual_lh, residual_hl, residual_hh).
pub fn extract_primitives(
    q_lh: &Array2<f64>,
    q_hl: &Array2<f64>,
    q_hh: &Array2<f64>,
) -> (Vec<Primitive>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let h_lh = q_lh.nrows(); let w_lh = q_lh.ncols();
    let h_hl = q_hl.nrows(); let w_hl = q_hl.ncols();
    let h_hh = q_hh.nrows(); let w_hh = q_hh.ncols();

    // Skip if bands are too small
    if h_lh * w_lh < 64 || h_hl * w_hl < 64 || h_hh * w_hh < 64 {
        return (Vec::new(), q_lh.clone(), q_hl.clone(), q_hh.clone());
    }

    let mut r_lh = q_lh.clone();
    let mut r_hl = q_hl.clone();
    let mut r_hh = q_hh.clone();

    // Common dimensions
    let h_c = h_lh.min(h_hl).min(h_hh);
    let w_c = w_lh.min(w_hl).min(w_hh);

    // === SINGLE-PASS with dilation + minimum threshold ===

    // 1. Combined energy
    let mut energy = vec![0.0f64; h_c * w_c];
    for idx in 0..h_c * w_c {
        let r = idx / w_c;
        let c = idx % w_c;
        energy[idx] = r_lh[[r, c]].abs() + r_hl[[r, c]].abs() + r_hh[[r, c]].abs();
    }

    // 2. Dilated mask: fill gaps between clusters
    //    Adaptive number of passes: larger images need more dilation
    //    (HD contours are more fragmented by quantization)
    let raw_mask: Vec<bool> = energy.iter().map(|&e| e > 0.5).collect();
    let n_passes = if h_c * w_c > 500_000 { 4 }
                   else if h_c * w_c > 100_000 { 3 }
                   else { 2 };
    let dilated = dilate_mask(&raw_mask, h_c, w_c, n_passes);

    // 3. Connected components on the dilated mask
    let (labels, n_comp) = connected_components_8(&dilated, h_c, w_c);
    if n_comp == 0 {
        return (Vec::new(), r_lh, r_hl, r_hh);
    }

    // 4. Group REAL pixels (not dilated) by dilated cluster
    //    We keep coordinates and energies of real pixels,
    //    but grouped according to dilated clusters (larger)
    let mut clusters: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); n_comp];
    for idx in 0..h_c * w_c {
        let lbl = labels[idx];
        if lbl == u32::MAX { continue; }
        // Only keep pixels with real energy
        if energy[idx] > 0.5 {
            let r = idx / w_c;
            let c = idx % w_c;
            clusters[lbl as usize].push((r, c, energy[idx]));
        }
    }

    // 5. Fit clusters that are long enough (adaptive threshold)
    //    Break-even threshold: byte_cost * 8 / bits_saved_per_pixel
    //    Larger images have more long clusters dominating -> stricter threshold
    let min_cluster_size: usize = if h_c * w_c > 500_000 { 34 } // Fibonacci !
                                  else if h_c * w_c > 100_000 { 21 } // Fibonacci !
                                  else { 13 }; // Fibonacci !
    let mut primitives: Vec<Primitive> = Vec::new();

    for cluster in &clusters {
        if cluster.len() < min_cluster_size { continue; }

        // Fit superstring segment (Fibonacci amplitude + phi phase)
        let seg_result = fit_segment(cluster).and_then(|(x1, y1, x2, y2, amp, phase)| {
            let a = snap_to_fibonacci(amp);
            if a == 0 { return None; }
            let ph = (phase * 128.0 / std::f64::consts::PI).round().clamp(-128.0, 127.0) as i8;
            let prim = Primitive::Segment { x1, y1, x2, y2, amplitude: a, phase: ph };
            let profit = compute_profit_fast(&prim, cluster);
            if profit > 0.0 { Some((prim, profit)) } else { None }
        });

        // Fit arc (>= 5 points needed)
        let arc_result = if cluster.len() >= 5 {
            fit_arc(cluster).and_then(|(cx, cy, radius, ts, te, _force)| {
                // Phi phase via correlation on arc points
                let n = cluster.len();
                let mut ss = 0.0f64;
                let mut sc = 0.0f64;
                for (i, &(_, _, amp)) in cluster.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * PHI_FREQ * i as f64;
                    ss += amp * angle.sin();
                    sc += amp * angle.cos();
                }
                let amp_val = (ss * ss + sc * sc).sqrt() * 2.0 / n as f64;
                let phase_val = sc.atan2(ss);
                let a = snap_to_fibonacci(amp_val);
                if a == 0 { return None; }
                let ph = (phase_val * 128.0 / std::f64::consts::PI).round().clamp(-128.0, 127.0) as i8;
                let prim = Primitive::Arc { cx, cy, radius, theta_start: ts, theta_end: te, amplitude: a, phase: ph };
                let profit = compute_profit_fast(&prim, cluster);
                if profit > 0.0 { Some((prim, profit)) } else { None }
            })
        } else {
            None
        };

        // Choose the best between segment and arc
        let best = match (seg_result, arc_result) {
            (Some((sp, sprof)), Some((ap, aprof))) => {
                if sprof >= aprof { Some(sp) } else { Some(ap) }
            }
            (Some((sp, _)), None) => Some(sp),
            (None, Some((ap, _))) => Some(ap),
            (None, None) => None,
        };

        if let Some(prim) = best {
            primitives.push(prim);
        }
    }

    // 5. Subtract ALL predictions in a single pass
    if !primitives.is_empty() {
        let (pred_lh, pred_hl, pred_hh) = render_primitives(
            &primitives, h_lh, w_lh, h_hl, w_hl, h_hh, w_hh,
        );
        for r in 0..h_lh { for c in 0..w_lh { r_lh[[r, c]] -= pred_lh[[r, c]]; } }
        for r in 0..h_hl { for c in 0..w_hl { r_hl[[r, c]] -= pred_hl[[r, c]]; } }
        for r in 0..h_hh { for c in 0..w_hh { r_hh[[r, c]] -= pred_hh[[r, c]]; } }
    }

    (primitives, r_lh, r_hl, r_hh)
}

/// Fast profit: estimation without full render.
/// bits_saved ~ n_points_cluster * 2 (sigmap) + n_large * 8 - cost_primitive
#[inline]
fn compute_profit_fast(prim: &Primitive, cluster: &[(usize, usize, f64)]) -> f64 {
    let n = cluster.len() as f64;
    let n_large = cluster.iter().filter(|(_, _, e)| e.abs() >= 2.0).count() as f64;
    // Conservative estimate: ~60% of cluster points are effectively predicted
    let bits_saved = n * 0.6 * 2.0 + n_large * 0.3 * 8.0;
    let cost_bits = prim.byte_cost() as f64 * 8.0;
    bits_saved - cost_bits
}

// ============================================================
// Serialization / Deserialization
// ============================================================

/// Serialize a list of primitives.
/// Format: [u32 section_size LE][u16 n_prims LE][type(u8) + params] ...
pub fn serialize_primitives(primitives: &[Primitive]) -> Vec<u8> {
    let mut body: Vec<u8> = Vec::new();

    // n_prims (u16 LE)
    let n = primitives.len() as u16;
    body.extend_from_slice(&n.to_le_bytes());

    for prim in primitives {
        match prim {
            Primitive::Segment { x1, y1, x2, y2, amplitude, phase } => {
                body.push(0x00);
                body.extend_from_slice(&x1.to_le_bytes());
                body.extend_from_slice(&y1.to_le_bytes());
                body.extend_from_slice(&x2.to_le_bytes());
                body.extend_from_slice(&y2.to_le_bytes());
                body.push(*amplitude as u8);
                body.push(*phase as u8);
            }
            Primitive::Arc { cx, cy, radius, theta_start, theta_end, amplitude, phase } => {
                body.push(0x01);
                body.extend_from_slice(&cx.to_le_bytes());
                body.extend_from_slice(&cy.to_le_bytes());
                body.extend_from_slice(&radius.to_le_bytes());
                body.push(*theta_start as u8);
                body.push(*theta_end as u8);
                body.push(*amplitude as u8);
                body.push(*phase as u8);
            }
        }
    }

    // section_size prefix (u32 LE) = body size (without the prefix itself)
    let section_size = body.len() as u32;
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&section_size.to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// Deserialize a list of primitives from a byte slice.
/// Returns (primitives, total_bytes_consumed).
pub fn deserialize_primitives(data: &[u8]) -> (Vec<Primitive>, usize) {
    if data.len() < 6 { return (Vec::new(), 0); }

    let section_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let total_bytes = 4 + section_size;

    if data.len() < total_bytes { return (Vec::new(), 0); }

    let n_prims = u16::from_le_bytes([data[4], data[5]]) as usize;
    let mut p = 6usize;
    let mut primitives = Vec::with_capacity(n_prims);

    for _ in 0..n_prims {
        if p >= total_bytes { break; }
        let ptype = data[p]; p += 1;
        match ptype {
            0x00 => {
                // Superstring segment: 4*i16 + amplitude(i8) + phase(i8) = 10 bytes
                if p + 10 > total_bytes { break; }
                let x1 = i16::from_le_bytes([data[p], data[p + 1]]); p += 2;
                let y1 = i16::from_le_bytes([data[p], data[p + 1]]); p += 2;
                let x2 = i16::from_le_bytes([data[p], data[p + 1]]); p += 2;
                let y2 = i16::from_le_bytes([data[p], data[p + 1]]); p += 2;
                let amplitude = data[p] as i8; p += 1;
                let phase = data[p] as i8; p += 1;
                primitives.push(Primitive::Segment { x1, y1, x2, y2, amplitude, phase });
            }
            0x01 => {
                // Superstring arc: 2*i16 + u16 + 2*i8 + amplitude(i8) + phase(i8) = 12 bytes
                if p + 12 > total_bytes { break; }
                let cx = i16::from_le_bytes([data[p], data[p + 1]]); p += 2;
                let cy = i16::from_le_bytes([data[p], data[p + 1]]); p += 2;
                let radius = u16::from_le_bytes([data[p], data[p + 1]]); p += 2;
                let theta_start = data[p] as i8; p += 1;
                let theta_end   = data[p] as i8; p += 1;
                let amplitude = data[p] as i8; p += 1;
                let phase = data[p] as i8; p += 1;
                primitives.push(Primitive::Arc { cx, cy, radius, theta_start, theta_end, amplitude, phase });
            }
            _ => break, // unknown type
        }
    }

    (primitives, total_bytes)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    const EPS: f64 = 1e-9;

    // ----------------------------------------------------------
    // Tests du renderer
    // ----------------------------------------------------------

    /// Un segment horizontal (theta=0) doit emettre uniquement dans LH.
    /// cos(0)=1, sin(0)=0 => LH>0, HL=0, HH=0
    #[test]
    fn test_segment_horizontal_emits_lh_only() {
        let prim = Primitive::Segment { x1: 2, y1: 5, x2: 10, y2: 5, amplitude: 10, phase: 0 };
        let (lh, hl, hh) = render_primitives(&[prim], 20, 20, 20, 20, 20, 20);

        let lh_sum: f64 = lh.iter().copied().sum();
        let hl_sum: f64 = hl.iter().copied().sum();
        let hh_sum: f64 = hh.iter().copied().sum();

        assert!(lh_sum > EPS, "LH doit etre non nul pour segment horizontal");
        assert!(hl_sum.abs() < EPS, "HL doit etre nul pour segment horizontal, got {}", hl_sum);
        assert!(hh_sum.abs() < EPS, "HH doit etre nul pour segment horizontal, got {}", hh_sum);
    }

    /// Un segment vertical (theta=pi/2) doit emettre uniquement dans HL.
    /// cos(pi/2)=0, sin(pi/2)=1 => LH=0, HL>0, HH=0
    #[test]
    fn test_segment_vertical_emits_hl_only() {
        let prim = Primitive::Segment { x1: 5, y1: 2, x2: 5, y2: 10, amplitude: 10, phase: 0 };
        let (lh, hl, hh) = render_primitives(&[prim], 20, 20, 20, 20, 20, 20);

        let lh_sum: f64 = lh.iter().copied().sum();
        let hl_sum: f64 = hl.iter().copied().sum();
        let hh_sum: f64 = hh.iter().copied().sum();

        // HL must dominate for a vertical segment; LH/HH can have small cross-talk
        assert!(hl_sum > lh_sum.abs() * 2.0, "HL doit dominer pour segment vertical (HL={}, LH={})", hl_sum, lh_sum);
        assert!(hl_sum > hh_sum.abs() * 2.0, "HL doit dominer pour segment vertical (HL={}, HH={})", hl_sum, hh_sum);
    }

    /// Un segment diagonal a 45 degres doit emettre dans les trois bandes.
    /// theta=pi/4 : cos=sin=1/sqrt(2), HH = |sin*cos| = 0.5
    #[test]
    fn test_segment_diagonal_emits_all_bands() {
        let prim = Primitive::Segment { x1: 2, y1: 2, x2: 10, y2: 10, amplitude: 10, phase: 0 };
        let (lh, hl, hh) = render_primitives(&[prim], 20, 20, 20, 20, 20, 20);

        let lh_sum: f64 = lh.iter().copied().sum();
        let hl_sum: f64 = hl.iter().copied().sum();
        let hh_sum: f64 = hh.iter().copied().sum();

        assert!(lh_sum > EPS, "LH doit etre non nul pour segment diagonal");
        assert!(hl_sum > EPS, "HL doit etre non nul pour segment diagonal");
        assert!(hh_sum > EPS, "HH doit etre non nul pour segment diagonal");
    }

    // ----------------------------------------------------------
    // Tests des composantes connexes
    // ----------------------------------------------------------

    /// Des pixels en diagonale forment une seule composante (8-connexite).
    #[test]
    fn test_connected_components_8() {
        // Pixels (0,0), (1,1), (2,2) — diagonale
        let mut mask = vec![false; 3 * 3];
        mask[0 * 3 + 0] = true; // (0,0)
        mask[1 * 3 + 1] = true; // (1,1)
        mask[2 * 3 + 2] = true; // (2,2)

        let (labels, n_comp) = connected_components_8(&mask, 3, 3);
        assert_eq!(n_comp, 1, "La diagonale doit former une seule composante");
        assert_eq!(labels[0 * 3 + 0], labels[1 * 3 + 1]);
        assert_eq!(labels[1 * 3 + 1], labels[2 * 3 + 2]);
    }

    /// Deux clusters separes doivent former deux composantes distinctes.
    #[test]
    fn test_connected_components_8_two_clusters() {
        // Cluster A : (0,0), (0,1)
        // Cluster B : (2,2), (2,3)
        // Separees par une ligne vide
        let mut mask = vec![false; 4 * 5];
        mask[0 * 5 + 0] = true;
        mask[0 * 5 + 1] = true;
        mask[2 * 5 + 2] = true;
        mask[2 * 5 + 3] = true;

        let (labels, n_comp) = connected_components_8(&mask, 4, 5);
        assert_eq!(n_comp, 2, "Doit avoir 2 composantes distinctes");
        assert_ne!(labels[0 * 5 + 0], labels[2 * 5 + 2]);
        assert_eq!(labels[0 * 5 + 0], labels[0 * 5 + 1]);
        assert_eq!(labels[2 * 5 + 2], labels[2 * 5 + 3]);
    }

    // ----------------------------------------------------------
    // Tests de l'ajustement de segment
    // ----------------------------------------------------------

    /// Ajuster un segment sur des points horizontaux.
    #[test]
    fn test_fit_segment_horizontal() {
        let points: Vec<(usize, usize, f64)> = (0..10).map(|c| (5usize, c, 5.0)).collect();
        let result = fit_segment(&points);
        assert!(result.is_some(), "fit_segment doit reussir sur 10 points horizontaux");
        let (x1, y1, x2, y2, amp, _phase) = result.unwrap();
        // Les deux extremites doivent etre sur la meme ligne (row=5)
        assert_eq!(y1, y2, "y1 et y2 doivent etre egaux pour un segment horizontal");
        assert!(x1 < x2 || x1 > x2, "x1 != x2 pour un segment non trivial");
        assert!(amp.abs() > 0.1, "Amplitude doit etre non nulle, got {}", amp);
        let _ = (x1, x2);
    }

    // ----------------------------------------------------------
    // Tests de l'ajustement d'arc
    // ----------------------------------------------------------

    /// Ajuster un arc sur des points d'un demi-cercle.
    #[test]
    fn test_fit_arc_semicircle() {
        // Semi-cercle de centre (50, 50), rayon 20
        let cx_true = 50.0f64;
        let cy_true = 50.0f64;
        let r_true  = 20.0f64;
        let points: Vec<(usize, usize, f64)> = (0..=10)
            .map(|i| {
                let t = i as f64 * std::f64::consts::PI / 10.0;
                let row = (cy_true + r_true * t.sin()).round() as usize;
                let col = (cx_true + r_true * t.cos()).round() as usize;
                (row, col, 8.0f64)
            })
            .collect();

        let result = fit_arc(&points);
        assert!(result.is_some(), "fit_arc doit reussir sur un demi-cercle");
        let (cx, cy, radius, _ts, _te, _force) = result.unwrap();
        assert!((cx as f64 - cx_true).abs() < 5.0, "Centre X approximatif : {} vs {}", cx, cx_true);
        assert!((cy as f64 - cy_true).abs() < 5.0, "Centre Y approximatif : {} vs {}", cy, cy_true);
        assert!((radius as f64 - r_true).abs() < 5.0, "Rayon approximatif : {} vs {}", radius, r_true);
    }

    // ----------------------------------------------------------
    // Tests de serialisation
    // ----------------------------------------------------------

    /// Serialiser puis deserialiser doit retrouver les memes primitives.
    #[test]
    fn test_serialize_roundtrip() {
        let primitives = vec![
            Primitive::Segment { x1: 10, y1: 20, x2: 30, y2: 40, amplitude: 5, phase: 0 },
            Primitive::Arc { cx: 100, cy: 200, radius: 50, theta_start: -64, theta_end: 64, amplitude: -3, phase: 1 },
            Primitive::Segment { x1: -10, y1: -5, x2: 15, y2: 25, amplitude: 127, phase: -1 },
        ];

        let bytes = serialize_primitives(&primitives);
        let (decoded, consumed) = deserialize_primitives(&bytes);

        assert_eq!(consumed, bytes.len(), "Tous les bytes doivent etre consommes");
        assert_eq!(decoded.len(), primitives.len(), "Meme nombre de primitives");

        match (&decoded[0], &primitives[0]) {
            (Primitive::Segment { x1: a1, y1: b1, x2: c1, y2: d1, amplitude: f1, phase: p1 },
             Primitive::Segment { x1: a2, y1: b2, x2: c2, y2: d2, amplitude: f2, phase: p2 }) => {
                assert_eq!(a1, a2); assert_eq!(b1, b2);
                assert_eq!(c1, c2); assert_eq!(d1, d2);
                assert_eq!(f1, f2); assert_eq!(p1, p2);
            }
            _ => panic!("Type incorrect pour primitive[0]"),
        }

        match (&decoded[1], &primitives[1]) {
            (Primitive::Arc { cx: a1, cy: b1, radius: r1, theta_start: ts1, theta_end: te1, amplitude: f1, phase: p1 },
             Primitive::Arc { cx: a2, cy: b2, radius: r2, theta_start: ts2, theta_end: te2, amplitude: f2, phase: p2 }) => {
                assert_eq!(a1, a2); assert_eq!(b1, b2); assert_eq!(r1, r2);
                assert_eq!(ts1, ts2); assert_eq!(te1, te2); assert_eq!(f1, f2); assert_eq!(p1, p2);
            }
            _ => panic!("Type incorrect pour primitive[1]"),
        }
    }

    // ----------------------------------------------------------
    // Tests de l'extracteur greedy
    // ----------------------------------------------------------

    /// Une ligne horizontale claire dans LH doit produire au moins une primitive.
    #[test]
    fn test_extract_horizontal_line() {
        let h = 32usize; let w = 32usize;
        let mut q_lh = Array2::<f64>::zeros((h, w));
        let q_hl = Array2::<f64>::zeros((h, w));
        let q_hh = Array2::<f64>::zeros((h, w));

        // Ligne horizontale d'energie forte sur la ligne 16
        for c in 2..30 {
            q_lh[[16, c]] = 20.0;
        }

        let (prims, _r_lh, _r_hl, _r_hh) = extract_primitives(&q_lh, &q_hl, &q_hh);
        assert!(!prims.is_empty(), "Doit trouver au moins une primitive pour une ligne horizontale claire");
    }

    /// Des bandes vides ne doivent produire aucune primitive.
    #[test]
    fn test_extract_empty_band() {
        let q_lh = Array2::<f64>::zeros((32, 32));
        let q_hl = Array2::<f64>::zeros((32, 32));
        let q_hh = Array2::<f64>::zeros((32, 32));

        let (prims, _r_lh, _r_hl, _r_hh) = extract_primitives(&q_lh, &q_hl, &q_hh);
        assert!(prims.is_empty(), "Bandes vides => aucune primitive");
    }
}
