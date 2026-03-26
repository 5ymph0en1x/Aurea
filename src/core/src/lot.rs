//! Lapped Orthogonal Transform (LOT) module.
//!
//! DCT-II based block transform with sine-window lapping for overlap-add
//! reconstruction. Supports fixed-size and variable-size (quadtree) blocking.

use std::f64::consts::PI;
const PHI: f64 = 1.6180339887498948482; // nombre d'or

// QMAT parametric constants (derived from pi and phi, LPIPS-optimized)
// Q_A=17.708 (scale), Q_ALPHA=0.846 (anisotropy), Q_BETA=0.899 (exponent), Q_GAMMA=15.795 (diagonal boost)
const Q_A: f64 = 5.0 * PI + 2.0;
const Q_ALPHA: f64 = 11.0 / 13.0;
const Q_BETA: f64 = 5.0 * PHI / 9.0;
const Q_GAMMA: f64 = 176.0 * PI / 35.0;

use std::sync::LazyLock;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// 0. Precomputed cosine LUT — eliminates cos() from DCT hot loops
// ---------------------------------------------------------------------------

/// Precomputed cosine table for DCT-II / IDCT-II of a fixed size N.
/// table[k*N + i] = cos(PI/N * (i + 0.5) * k)
struct DctLut {
    table: Vec<f64>,
    n: usize,
}

impl DctLut {
    fn new(n: usize) -> Self {
        let nf = n as f64;
        let mut table = vec![0.0; n * n];
        for k in 0..n {
            for i in 0..n {
                table[k * n + i] = (PI / nf * (i as f64 + 0.5) * k as f64).cos();
            }
        }
        Self { table, n }
    }

    /// Forward DCT-II using precomputed cosines.
    #[inline]
    fn dct(&self, input: &[f64], out: &mut [f64]) {
        let n = self.n;
        for k in 0..n {
            let base = k * n;
            let mut sum = 0.0;
            for i in 0..n {
                // SAFETY: base + i = k*n + i < n*n = table.len(); i < n = input.len()
                unsafe { sum += *input.get_unchecked(i) * *self.table.get_unchecked(base + i); }
            }
            unsafe { *out.get_unchecked_mut(k) = sum; }
        }
    }

    /// Inverse DCT-II (DCT-III with normalization) using precomputed cosines.
    #[inline]
    fn idct(&self, input: &[f64], out: &mut [f64]) {
        let n = self.n;
        let inv_n = 1.0 / n as f64;
        for i in 0..n {
            let mut sum = unsafe { *input.get_unchecked(0) };
            for k in 1..n {
                unsafe {
                    sum += 2.0 * *input.get_unchecked(k) * *self.table.get_unchecked(k * n + i);
                }
            }
            unsafe { *out.get_unchecked_mut(i) = sum * inv_n; }
        }
    }
}

// SAFETY: DctLut is immutable after construction, safe to share across threads.
unsafe impl Sync for DctLut {}

static DCT_LUT_8: LazyLock<DctLut> = LazyLock::new(|| DctLut::new(8));
static DCT_LUT_16: LazyLock<DctLut> = LazyLock::new(|| DctLut::new(16));
static DCT_LUT_32: LazyLock<DctLut> = LazyLock::new(|| DctLut::new(32));

#[inline]
fn get_dct_lut(n: usize) -> Option<&'static DctLut> {
    match n {
        8 => Some(&DCT_LUT_8),
        16 => Some(&DCT_LUT_16),
        32 => Some(&DCT_LUT_32),
        _ => None,
    }
}

// Cached sine windows (avoid repeated sin() calls per block)
static SINE_WIN_8: LazyLock<Vec<f64>> = LazyLock::new(|| sine_window(8));
static SINE_WIN_16: LazyLock<Vec<f64>> = LazyLock::new(|| sine_window(16));
static SINE_WIN_32: LazyLock<Vec<f64>> = LazyLock::new(|| sine_window(32));

fn cached_sine_window(n: usize) -> &'static [f64] {
    match n {
        8 => &SINE_WIN_8,
        16 => &SINE_WIN_16,
        32 => &SINE_WIN_32,
        _ => panic!("unsupported sine window size {n}"),
    }
}

// ---------------------------------------------------------------------------
// 1. DCT-II / IDCT-II  (1-D)
// ---------------------------------------------------------------------------

/// Forward DCT-II of length N.
///
/// X[k] = sum_{n=0}^{N-1} x[n] * cos( pi/N * (n + 0.5) * k )
///
/// Uses precomputed cosine LUT for N ∈ {8, 16, 32}.
pub fn dct_ii(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0; n];
    if let Some(lut) = get_dct_lut(n) {
        lut.dct(input, &mut out);
    } else {
        let nf = n as f64;
        for k in 0..n {
            let mut sum = 0.0;
            for i in 0..n {
                sum += input[i] * (PI / nf * (i as f64 + 0.5) * k as f64).cos();
            }
            out[k] = sum;
        }
    }
    out
}

/// Inverse DCT-II (DCT-III with normalization) of length N.
///
/// x[n] = (1/N) * X[0] + (2/N) * sum_{k=1}^{N-1} X[k] * cos( pi/N * k * (n + 0.5) )
///
/// Uses precomputed cosine LUT for N ∈ {8, 16, 32}.
pub fn idct_ii(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0; n];
    if let Some(lut) = get_dct_lut(n) {
        lut.idct(input, &mut out);
    } else {
        let nf = n as f64;
        for i in 0..n {
            let mut sum = input[0];
            for k in 1..n {
                sum += 2.0 * input[k] * (PI / nf * k as f64 * (i as f64 + 0.5)).cos();
            }
            out[i] = sum / nf;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 2. DCT-II / IDCT-II  (2-D separable)
// ---------------------------------------------------------------------------

/// Forward 2-D DCT-II (separable: rows then columns).
/// Uses in-place LUT transforms to avoid per-transform allocation.
pub fn dct_2d(block: &[f64], h: usize, w: usize) -> Vec<f64> {
    assert_eq!(block.len(), h * w);
    let lut_w = get_dct_lut(w);
    let lut_h = get_dct_lut(h);

    // Transform rows
    let mut tmp = vec![0.0; h * w];
    if let Some(lut) = lut_w {
        for r in 0..h {
            lut.dct(&block[r * w..(r + 1) * w], &mut tmp[r * w..(r + 1) * w]);
        }
    } else {
        for r in 0..h {
            let t = dct_ii(&block[r * w..(r + 1) * w]);
            tmp[r * w..(r + 1) * w].copy_from_slice(&t);
        }
    }

    // Transform columns
    let mut out = vec![0.0; h * w];
    let mut col_in = vec![0.0; h];
    let mut col_out = vec![0.0; h];
    if let Some(lut) = lut_h {
        for c in 0..w {
            for r in 0..h { col_in[r] = tmp[r * w + c]; }
            lut.dct(&col_in, &mut col_out);
            for r in 0..h { out[r * w + c] = col_out[r]; }
        }
    } else {
        for c in 0..w {
            for r in 0..h { col_in[r] = tmp[r * w + c]; }
            let t = dct_ii(&col_in);
            for r in 0..h { out[r * w + c] = t[r]; }
        }
    }
    out
}

/// Inverse 2-D DCT-II (separable: columns then rows).
/// Uses in-place LUT transforms to avoid per-transform allocation.
pub fn idct_2d(coeffs: &[f64], h: usize, w: usize) -> Vec<f64> {
    assert_eq!(coeffs.len(), h * w);
    let lut_w = get_dct_lut(w);
    let lut_h = get_dct_lut(h);

    // Inverse-transform columns first
    let mut tmp = vec![0.0; h * w];
    let mut col_in = vec![0.0; h];
    let mut col_out = vec![0.0; h];
    if let Some(lut) = lut_h {
        for c in 0..w {
            for r in 0..h { col_in[r] = coeffs[r * w + c]; }
            lut.idct(&col_in, &mut col_out);
            for r in 0..h { tmp[r * w + c] = col_out[r]; }
        }
    } else {
        for c in 0..w {
            for r in 0..h { col_in[r] = coeffs[r * w + c]; }
            let t = idct_ii(&col_in);
            for r in 0..h { tmp[r * w + c] = t[r]; }
        }
    }

    // Inverse-transform rows
    let mut out = vec![0.0; h * w];
    if let Some(lut) = lut_w {
        for r in 0..h {
            lut.idct(&tmp[r * w..(r + 1) * w], &mut out[r * w..(r + 1) * w]);
        }
    } else {
        for r in 0..h {
            let t = idct_ii(&tmp[r * w..(r + 1) * w]);
            out[r * w..(r + 1) * w].copy_from_slice(&t);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 3. Sine window
// ---------------------------------------------------------------------------

/// Sine window: w(i) = sin(pi * (i + 0.5) / n).
pub fn sine_window(n: usize) -> Vec<f64> {
    (0..n).map(|i| (PI * (i as f64 + 0.5) / n as f64).sin()).collect()
}

/// Apply a separable 2-D window in-place.
/// `win_h` has length `h`, `win_w` has length `w`.
pub fn apply_window_2d(block: &mut [f64], h: usize, w: usize, win_h: &[f64], win_w: &[f64]) {
    assert_eq!(block.len(), h * w);
    assert_eq!(win_h.len(), h);
    assert_eq!(win_w.len(), w);
    for r in 0..h {
        let wh = win_h[r];
        for c in 0..w {
            block[r * w + c] *= wh * win_w[c];
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Boundary reflection helper
// ---------------------------------------------------------------------------

/// Mirror-reflect index `i` into [0, n).
fn reflect_idx(i: i64, n: i64) -> i64 {
    if n <= 1 {
        return 0;
    }
    let mut j = i;
    // Bring into range [-(n-1), 2*(n-1)]
    let period = 2 * (n - 1);
    if period > 0 {
        j = j % period;
        if j < 0 {
            j += period;
        }
    }
    // Fold back from the far side
    if j >= n {
        j = period - j;
    }
    j
}

// ---------------------------------------------------------------------------
// 5. Single-block LOT analyze / synthesize
// ---------------------------------------------------------------------------

/// Extract, window, and DCT a single block_size x block_size region centred
/// at pixel (block_y, block_x) with mirror-boundary extension.
pub fn lot_analyze_block(
    image: &[f64],
    img_h: usize,
    img_w: usize,
    block_y: usize,
    block_x: usize,
    block_size: usize,
    use_overlap: bool,
) -> Vec<f64> {
    let stride = lot_stride(block_size, use_overlap);
    let use_window = stride < block_size; // window only with overlap
    let offset = if use_window { (block_size / 2) as i64 } else { 0 };
    let ih = img_h as i64;
    let iw = img_w as i64;

    let mut block = vec![0.0; block_size * block_size];
    for r in 0..block_size {
        let sy = block_y as i64 - offset + r as i64;
        let ry = reflect_idx(sy, ih) as usize;
        for c in 0..block_size {
            let sx = block_x as i64 - offset + c as i64;
            let rx = reflect_idx(sx, iw) as usize;
            block[r * block_size + c] = image[ry * img_w + rx];
        }
    }

    if use_window {
        let win = cached_sine_window(block_size);
        apply_window_2d(&mut block, block_size, block_size, win, win);
    }
    dct_2d(&block, block_size, block_size)
}

/// IDCT, window, and overlap-add a single block into the output buffer.
/// `weight` accumulates squared window values for later normalization.
pub fn lot_synthesize_block(
    coeffs: &[f64],
    block_size: usize,
    output: &mut [f64],
    weight: &mut [f64],
    out_h: usize,
    out_w: usize,
    block_y: usize,
    block_x: usize,
    use_overlap: bool,
) {
    let stride = lot_stride(block_size, use_overlap);
    let use_window = stride < block_size;
    let offset = if use_window { block_size / 2 } else { 0 };
    let win = if use_window { cached_sine_window(block_size) } else { &[] };

    let mut spatial = idct_2d(coeffs, block_size, block_size);
    if use_window {
        apply_window_2d(&mut spatial, block_size, block_size, win, win);
    }

    for r in 0..block_size {
        let py = block_y as i64 - offset as i64 + r as i64;
        if py < 0 || py >= out_h as i64 {
            continue;
        }
        let py = py as usize;
        let wh = if use_window { win[r] } else { 1.0 };
        for c in 0..block_size {
            let px = block_x as i64 - offset as i64 + c as i64;
            if px < 0 || px >= out_w as i64 {
                continue;
            }
            let px = px as usize;
            let ww = if use_window { win[c] } else { 1.0 };
            let w2 = wh * wh * ww * ww; // window^2 (separable)
            output[py * out_w + px] += spatial[r * block_size + c];
            weight[py * out_w + px] += w2;
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Full-image fixed-size LOT
// ---------------------------------------------------------------------------

/// Analyze a full image with fixed block size.
///
/// Returns `(dc_grid, ac_blocks, grid_h, grid_w)` where `dc_grid` holds one
/// DC coefficient per block and `ac_blocks` holds the remaining block_size^2 - 1
/// AC coefficients per block.
pub fn lot_analyze_image(
    image: &[f64],
    h: usize,
    w: usize,
    block_size: usize,
    use_overlap: bool,
) -> (Vec<f64>, Vec<Vec<f64>>, usize, usize) {
    let stride = lot_stride(block_size, use_overlap);
    let grid_h = (h + stride - 1) / stride;
    let grid_w = (w + stride - 1) / stride;
    let n_blocks = grid_h * grid_w;

    // Parallel block analysis: each block's DCT is independent
    let results: Vec<Vec<f64>> = (0..n_blocks)
        .into_par_iter()
        .map(|idx| {
            let gy = idx / grid_w;
            let gx = idx % grid_w;
            lot_analyze_block(image, h, w, gy * stride, gx * stride, block_size, use_overlap)
        })
        .collect();

    let mut dc_grid = Vec::with_capacity(n_blocks);
    let mut ac_blocks = Vec::with_capacity(n_blocks);
    for coeffs in results {
        dc_grid.push(coeffs[0]);
        ac_blocks.push(coeffs[1..].to_vec());
    }

    (dc_grid, ac_blocks, grid_h, grid_w)
}

/// Synthesize a full image from DC grid + AC blocks.
/// Normalizes by accumulated window weight.
pub fn lot_synthesize_image(
    dc_grid: &[f64],
    ac_blocks: &[Vec<f64>],
    grid_h: usize,
    grid_w: usize,
    h: usize,
    w: usize,
    block_size: usize,
    use_overlap: bool,
) -> Vec<f64> {
    let stride = lot_stride(block_size, use_overlap);
    let use_window = stride < block_size;
    let n = h * w;
    let n_blocks = grid_h * grid_w;

    // Phase 1: Parallel IDCT + windowing for all blocks
    let spatial_blocks: Vec<Vec<f64>> = (0..n_blocks)
        .into_par_iter()
        .map(|idx| {
            let mut coeffs = Vec::with_capacity(block_size * block_size);
            coeffs.push(dc_grid[idx]);
            coeffs.extend_from_slice(&ac_blocks[idx]);
            let mut spatial = idct_2d(&coeffs, block_size, block_size);
            if use_window {
                let win = cached_sine_window(block_size);
                apply_window_2d(&mut spatial, block_size, block_size, win, win);
            }
            spatial
        })
        .collect();

    // Phase 2: Sequential overlap-add (cheap: only additions)
    let mut output = vec![0.0; n];
    let mut weight = vec![0.0; n];
    let offset = if use_window { block_size / 2 } else { 0 };
    let win = if use_window { cached_sine_window(block_size) } else { &[] };

    for idx in 0..n_blocks {
        let gy = idx / grid_w;
        let gx = idx % grid_w;
        let by = gy * stride;
        let bx = gx * stride;
        let spatial = &spatial_blocks[idx];

        for r in 0..block_size {
            let py = by as i64 - offset as i64 + r as i64;
            if py < 0 || py >= h as i64 { continue; }
            let py = py as usize;
            let wh = if use_window { win[r] } else { 1.0 };
            for c in 0..block_size {
                let px = bx as i64 - offset as i64 + c as i64;
                if px < 0 || px >= w as i64 { continue; }
                let px = px as usize;
                let ww = if use_window { win[c] } else { 1.0 };
                let w2 = wh * wh * ww * ww;
                output[py * w + px] += spatial[r * block_size + c];
                weight[py * w + px] += w2;
            }
        }
    }

    // Phase 3: Parallel normalization
    output.par_iter_mut().zip(weight.par_iter()).for_each(|(o, &wt)| {
        if wt > 1e-15 { *o /= wt; }
    });
    output
}

// ---------------------------------------------------------------------------
// 7. Block classification (quadtree)
// ---------------------------------------------------------------------------

/// Allowed LOT block sizes (must be powers of 2).
pub const BLOCK_SIZES: [usize; 3] = [8, 16, 32];

/// Precompute the structural codon map for an entire DC grid.
/// Returns one factor per block: low = preserve (structural), high = compress (smooth).
///
/// Uses DC gradient (max absolute difference with 4 neighbors) as the measure
/// of structural importance. The gradient captures the ESSENCE of each block:
/// is this block part of an edge, a texture transition, or a smooth region?
///
/// Identical in encoder and decoder (both use the same reconstructed DC grid).
/// No encoder/decoder mismatch: pure DC-derived, no AC dependency.
///
/// Factors are in [CODON_STRUCT_MIN, CODON_STRUCT_MAX]:
/// - Low gradient (smooth) → high factor (compress aggressively)
/// - High gradient (structure) → low factor (preserve detail)
pub fn codon_structural_map(
    dc_grid: &[f64], grid_h: usize, grid_w: usize,
) -> Vec<f64> {
    use crate::calibration;

    let n = grid_h * grid_w;
    if n == 0 { return Vec::new(); }

    // Phase 1: compute per-block DC gradient (max |DC_neighbor - DC_self|)
    let mut gradients = vec![0.0f64; n];
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let idx = gy * grid_w + gx;
            let dc = dc_grid[idx];
            let mut max_grad = 0.0f64;
            if gy > 0        { max_grad = max_grad.max((dc_grid[(gy-1)*grid_w + gx] - dc).abs()); }
            if gy+1 < grid_h { max_grad = max_grad.max((dc_grid[(gy+1)*grid_w + gx] - dc).abs()); }
            if gx > 0        { max_grad = max_grad.max((dc_grid[gy*grid_w + gx-1] - dc).abs()); }
            if gx+1 < grid_w { max_grad = max_grad.max((dc_grid[gy*grid_w + gx+1] - dc).abs()); }
            gradients[idx] = max_grad;
        }
    }

    // Phase 2: compute median gradient as the reference "typical structure"
    let mut sorted = gradients.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median_grad = sorted[sorted.len() / 2].max(1.0); // floor at 1.0 to avoid div/0

    // Phase 3: map gradient to codon factor
    // grad_ratio = gradient / median: <1 = smoother than typical, >1 = more structured
    // factor curve: smooth → CODON_STRUCT_MAX (compress), structured → CODON_STRUCT_MIN (preserve)
    let f_min = calibration::CODON_STRUCT_FACTOR_MIN; // 0.8 (preserve)
    let f_max = calibration::CODON_STRUCT_FACTOR_MAX; // 1.3 (compress)

    gradients.iter().map(|&g| {
        let ratio = (g / median_grad).clamp(0.0, 3.0);
        // Linear ramp: ratio=0 → f_max, ratio=2 → f_min, clamp both ends
        let t = (ratio / 2.0).clamp(0.0, 1.0); // 0=smooth, 1=structured
        let factor = f_max + t * (f_min - f_max); // lerp from max to min
        factor.clamp(f_min, f_max)
    }).collect()
}

/// Compute per-block foveal saliency map from DC grid.
/// Replaces the stepped codon_structural_map + luminance_trna with a single
/// continuous field S ∈ [0,1] that modulates quantization via step * φ^(-k*S).
///
/// Sources (all from reconstructed DC, zero bpp):
/// - Weber contrast of DC gradient (edge importance)
/// - Luminance sensitivity (Weber-Fechner: dark regions more sensitive)
///
/// Returns: per-block foveal step factor = φ^(-k * S_norm).
/// Crests (edges, dark details): factor < 1 → finer step → preserve
/// Valleys (smooth, bright): factor > 1 → coarser step → compress
pub fn foveal_saliency_map(
    dc_grid: &[f64], grid_h: usize, grid_w: usize,
) -> Vec<f64> {
    use crate::golden::PHI;

    let n = grid_h * grid_w;
    if n == 0 { return Vec::new(); }
    let k = *crate::calibration::TUNABLE_FOVEAL_K;

    // Source 1: DC gradient Weber contrast
    let mut weber_contrast = vec![0.0f64; n];
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let idx = gy * grid_w + gx;
            let dc = dc_grid[idx];
            let mut max_grad = 0.0f64;
            if gy > 0        { max_grad = max_grad.max((dc_grid[(gy-1)*grid_w + gx] - dc).abs()); }
            if gy+1 < grid_h { max_grad = max_grad.max((dc_grid[(gy+1)*grid_w + gx] - dc).abs()); }
            if gx > 0        { max_grad = max_grad.max((dc_grid[gy*grid_w + gx-1] - dc).abs()); }
            if gx+1 < grid_w { max_grad = max_grad.max((dc_grid[gy*grid_w + gx+1] - dc).abs()); }
            // Weber: gradient normalized by local luminance
            weber_contrast[idx] = max_grad / (dc.abs() + 10.0);
        }
    }

    // Source 2: Luminance sensitivity (dark = more perceptible = higher saliency)
    // Inverse of luminance_trna: dark → high S, bright → low S
    let lum_sensitivity: Vec<f64> = dc_grid.iter().map(|&dc| {
        let lum_norm = (dc / 255.0).clamp(0.0, 1.0);
        1.0 - lum_norm // dark=1.0, bright=0.0
    }).collect();

    // Normalize Weber contrast to [0, 1]
    let max_weber = weber_contrast.iter().cloned().fold(0.0f64, f64::max).max(1e-6);

    // Combine: S = 0.6 * weber + 0.4 * luminance_sensitivity
    // Then map to step factor: φ^(-k * S)
    (0..n).map(|i| {
        let weber_norm = (weber_contrast[i] / max_weber).min(1.0);
        let s = 0.6 * weber_norm + 0.4 * lum_sensitivity[i];
        PHI.powf(-k * s)
    }).collect()
}

/// Legacy codon_dc_factor — kept for spectral spin local_steps computation.
/// Returns 1.0 (neutral) since the structural map is the new primary codon.
pub fn codon_dc_factor(_dc_l: f64, _dc_c1: f64, _dc_c2: f64) -> f64 {
    1.0
}

/// Legacy 3D codon — delegates to structural map when available.
/// Called from encoder/decoder per-block loops.
pub fn codon_3d_factor(_dc_l: f64, _dc_c1: f64, _dc_c2: f64, _ac_energy: f64, _n_ac: usize) -> f64 {
    1.0 // neutral: actual modulation comes from codon_structural_map
}

/// Legacy 4D codon — same as 3D.
pub fn codon_4d_factor(
    _dc_l: f64, _dc_c1: f64, _dc_c2: f64,
    _ac_block: &[f64], _block_size: usize,
) -> f64 {
    1.0
}

/// LOT stride as fraction of block_size. Controls overlap amount.
/// block_size/2 = 50% overlap (4× redundancy, best quality, high bitrate)
/// block_size*3/4 = 25% overlap (~1.78× redundancy, good quality, lower bitrate)
/// block_size = 0% overlap (no redundancy, blocking artifacts)
pub fn lot_stride(block_size: usize, use_overlap: bool) -> usize {
    if use_overlap {
        block_size / 2
    } else {
        block_size  // No overlap: critical sampling
    }
}

// ======================================================================
// Weber-Fechner luminance tRNA (DNA5)
// ======================================================================

/// Luminance-adaptive step factor based on Weber-Fechner law.
/// Dark blocks get finer quantization (factor < 1), bright blocks coarser.
/// Uses the same tRNA table as the wavelet codon_step_map.
///
/// Returns a multiplicative factor for the AC step: [0.5, 0.75, 1.0, 1.5].
pub fn luminance_trna(dc_luminance: f64) -> f64 {
    let thresholds = crate::calibration::CODON_LUM_THRESHOLDS;
    // Use runtime-tunable tRNA (env var overridable)
    let trna = *crate::calibration::TUNABLE_TRNA;
    if dc_luminance < thresholds[0] { trna[0] }       // dark
    else if dc_luminance < thresholds[1] { trna[1] }   // mid-dark
    else if dc_luminance < thresholds[2] { trna[2] }   // mid-bright (=1.0)
    else { trna[3] }                                    // bright
}

// ======================================================================
// Point 5: CSF (Contrast Sensitivity Function) frequency-dependent
// ======================================================================

/// CSF-modulated QMAT factor (Point 5).
/// In dark blocks, high-frequency positions get LARGER factors (coarser quant)
/// because the human visual system is less sensitive to HF at low luminance.
///
/// Returns a multiplicative factor >= 1.0 to apply on top of QMAT.
/// `row`, `col`: position in the block (0..block_size-1)
/// `dc_luminance`: block DC value in PTF space (~0-255)
pub fn csf_qmat_factor(row: usize, col: usize, block_size: usize, dc_luminance: f64) -> f64 {
    let bs_max = (block_size - 1).max(1) as f64;
    let freq_norm = ((row * row + col * col) as f64).sqrt() / (bs_max * std::f64::consts::SQRT_2);
    let lum_norm = (dc_luminance / 255.0).clamp(0.0, 1.0);

    // Dark regions: boost HF quantization (less visible)
    1.0 + crate::calibration::CSF_DARK_BOOST * (1.0 - lum_norm) * freq_norm * freq_norm
}

// ======================================================================
// Point 4: Structural coherence (polymerase-inspired analysis)
// ======================================================================

/// Structural coherence factor (Point 4, polymerase-inspired).
/// Measures directional organization of AC energy in a block.
/// High coherence = structured pattern (lines, curves) -> finer quantization.
/// Low coherence = noise/isotropic texture -> standard quantization.
///
/// Returns a factor in [0.8, 1.0]. Lower = finer quantization.
pub fn structural_coherence_factor(ac: &[f64], block_size: usize) -> f64 {
    if ac.len() < 4 { return 1.0; }

    // Directional energy decomposition in frequency domain
    let mut h_energy = 0.0f64; // horizontal frequencies (first row, r=0)
    let mut v_energy = 0.0f64; // vertical frequencies (first col, c=0)
    let mut d_energy = 0.0f64; // diagonal frequencies (r>0, c>0)

    for r in 0..block_size {
        for c in 0..block_size {
            let idx = r * block_size + c;
            if idx == 0 { continue; } // skip DC
            let ac_idx = idx - 1;
            if ac_idx >= ac.len() { break; }
            let val = ac[ac_idx].abs();

            if r == 0 { h_energy += val; }
            else if c == 0 { v_energy += val; }
            else { d_energy += val; }
        }
    }

    let total = h_energy + v_energy + d_energy;
    if total < 1.0 { return 1.0; }

    // Anisotropy: how dominant is one direction?
    let max_dir = h_energy.max(v_energy).max(d_energy);
    let anisotropy = max_dir / total; // [0.33, 1.0]

    // High anisotropy = structural -> finer quantization
    let factor = 1.0 - 0.2 * (anisotropy - 0.33).max(0.0) / 0.67;
    factor.clamp(crate::calibration::CODON_STRUCT_FACTOR_MIN, 1.0)
}

/// QMAT value for arbitrary block size (Point 2: variable blocks).
/// Drop-in replacement for the old bilinear-interpolation version.
/// Now uses the optimized parametric formula directly — works natively
/// for any block size (8, 16, 32) without interpolation artifacts.
pub fn qmat_for_block_size(row: usize, col: usize, block_size: usize) -> f64 {
    if row == 0 && col == 0 {
        return 0.0;
    }
    let max_index = (block_size - 1) as f64;
    let r = row as f64 / max_index;
    let c = col as f64 / max_index;
    let d = (r * r + Q_ALPHA * c * c).sqrt();
    let q = Q_A * d.powf(Q_BETA) + Q_GAMMA * r * c;
    q.max(0.1)
}

/// Optimized quantization matrix for 16x16 LOT blocks (row-major, 256 entries).
/// In-the-loop LPIPS-optimized on 6 diverse images × 2 quality levels (120 trials).
/// Parametric: scale=17.6652, exp=0.8997, aniso=0.8479, diag=15.7975
/// DC (index 0) = 0.0 (handled separately).
pub static QMAT_16: LazyLock<[f64; 256]> = LazyLock::new(|| {
    let mut table = [0.0; 256];
    let size = 16;
    let n = size as f64;
    let max_index = (size - 1) as f64;
    for row in 0..size {
        for col in 0..size {
            if row == 0 && col == 0 {
                table[row * size + col] = 0.0;
                continue;
            }
            let r = row as f64 / max_index;
            let c = col as f64 / max_index;
            let d = (r * r + Q_ALPHA * c * c).sqrt();
            let q = Q_A * d.powf(Q_BETA) + Q_GAMMA * r * c;
            table[row * size + col] = q.max(0.1);
        }
    }
    table
});

/// Classify every 8x8 cell into the largest block size whose local variance
/// stays below `merge_thresh = step^2 * 0.25`.  Returns `(size_grid, grid_h,
/// grid_w)` — one byte per 8x8 cell encoding the chosen block size.
pub fn classify_blocks(
    image: &[f64],
    h: usize,
    w: usize,
    step: f64,
) -> (Vec<u8>, usize, usize) {
    let cell = 8usize;
    let grid_h = (h + cell - 1) / cell;
    let grid_w = (w + cell - 1) / cell;
    let merge_thresh = step * step * 0.25;

    // Compute per-cell variance
    let mut var_grid = vec![0.0f64; grid_h * grid_w];
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let y0 = gy * cell;
            let x0 = gx * cell;
            let y1 = (y0 + cell).min(h);
            let x1 = (x0 + cell).min(w);
            let mut sum = 0.0;
            let mut sum2 = 0.0;
            let mut cnt = 0.0;
            for r in y0..y1 {
                for c in x0..x1 {
                    let v = image[r * w + c];
                    sum += v;
                    sum2 += v * v;
                    cnt += 1.0;
                }
            }
            if cnt > 0.0 {
                let mean = sum / cnt;
                var_grid[gy * grid_w + gx] = sum2 / cnt - mean * mean;
            }
        }
    }

    // Start at 16×16 (proven default), merge up to 32 if smooth.
    // The 16×16 LOT captures longer-range correlations than 8×8 and compresses
    // better on the majority of content (gradients, sky, architecture).
    let mut size_grid = vec![16u8; grid_h * grid_w];

    // Supercordes structural classification on the DC grid.
    let dc_gh = (h + 15) / 16;
    let dc_gw = (w + 15) / 16;
    let mut dc_means = vec![0.0f64; dc_gh * dc_gw];
    for dgy in 0..dc_gh {
        for dgx in 0..dc_gw {
            let y0 = dgy * 16;
            let x0 = dgx * 16;
            let y1 = (y0 + 16).min(h);
            let x1 = (x0 + 16).min(w);
            let mut sum = 0.0;
            let mut cnt = 0.0;
            for r in y0..y1 {
                for c in x0..x1 {
                    sum += image[r * w + c];
                    cnt += 1.0;
                }
            }
            if cnt > 0.0 { dc_means[dgy * dc_gw + dgx] = sum / cnt; }
        }
    }
    let supercordes = crate::dsp::supercordes_classify(&dc_means, dc_gh, dc_gw);

    let g16 = 2usize; // used by 32-merge below

    // Try 32: merge 4x4 groups of 8-cells
    let g32 = 4usize; // 32/8
    for gy in (0..grid_h).step_by(g32) {
        for gx in (0..grid_w).step_by(g32) {
            let mut max_var = 0.0f64;
            let mut all_16 = true;
            let mut all_present = true;
            for dy in 0..g32 {
                for dx in 0..g32 {
                    let cy = gy + dy;
                    let cx = gx + dx;
                    if cy >= grid_h || cx >= grid_w {
                        all_present = false;
                        continue;
                    }
                    max_var = max_var.max(var_grid[cy * grid_w + cx]);
                    if size_grid[cy * grid_w + cx] != 16 {
                        all_16 = false;
                    }
                }
            }
            // Supercordes gate: ALL underlying 16×16 blocks must be "Rien"
            // (no geometric structure). Segments and Arcs stay at 16×16
            // even if variance is low (roads, subtle gradients, reflections).
            let mut all_rien = true;
            if all_present && all_16 {
                // Each 32×32 covers 2×2 blocks in the 16×16 DC grid
                let dc_gy = gy / 2;
                let dc_gx = gx / 2;
                for ddy in 0..2 {
                    for ddx in 0..2 {
                        let sy = dc_gy + ddy;
                        let sx = dc_gx + ddx;
                        if sy < dc_gh && sx < dc_gw {
                            if supercordes[sy * dc_gw + sx] != crate::dsp::Supercorde::Rien {
                                all_rien = false;
                            }
                        }
                    }
                }
            }

            if all_present && all_16 && all_rien && max_var < merge_thresh {
                for dy in 0..g32 {
                    for dx in 0..g32 {
                        size_grid[(gy + dy) * grid_w + (gx + dx)] = 32;
                    }
                }
            }
        }
    }

    (size_grid, grid_h, grid_w)
}

/// Encode the block-size grid as 2 bits per cell (packed, MSB first).
/// Mapping: 8 → 0b00, 16 → 0b01, 32 → 0b10.
pub fn encode_block_map(size_grid: &[u8]) -> Vec<u8> {
    let n = size_grid.len();
    let n_bytes = (n * 2 + 7) / 8;
    let mut bytes = vec![0u8; n_bytes];
    for (i, &s) in size_grid.iter().enumerate() {
        let code: u8 = match s {
            8 => 0,
            16 => 1,
            32 => 2,
            _ => 0,
        };
        let byte_idx = (i * 2) / 8;
        let bit_offset = (i * 2) % 8;
        bytes[byte_idx] |= code << (6 - bit_offset);
    }
    bytes
}

/// Decode the block-size grid from packed 2-bit representation.
pub fn decode_block_map(bytes: &[u8], n_cells: usize) -> Vec<u8> {
    let mut grid = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        let byte_idx = (i * 2) / 8;
        let bit_offset = (i * 2) % 8;
        let code = (bytes[byte_idx] >> (6 - bit_offset)) & 0x03;
        let size = match code {
            0 => 8,
            1 => 16,
            2 => 32,
            _ => 8,
        };
        grid.push(size);
    }
    grid
}

// ---------------------------------------------------------------------------
// 8. Variable-size LOT
// ---------------------------------------------------------------------------

/// Iterate over the size grid and return unique blocks as `(block_y, block_x,
/// block_size)`.  Each block is emitted once — its top-left 8x8 cell is used
/// as the anchor and the centre is placed at `anchor + block_size/2`.
pub fn iter_blocks(
    size_grid: &[u8],
    gh: usize,
    gw: usize,
    use_overlap: bool,
) -> Vec<(usize, usize, usize)> {
    let cell = 8usize;
    let mut visited = vec![false; gh * gw];
    let mut blocks = Vec::new();

    for gy in 0..gh {
        for gx in 0..gw {
            if visited[gy * gw + gx] {
                continue;
            }
            let bs = size_grid[gy * gw + gx] as usize;
            let cells_span = bs / cell;
            // Mark all cells covered by this block as visited
            for dy in 0..cells_span {
                for dx in 0..cells_span {
                    let cy = gy + dy;
                    let cx = gx + dx;
                    if cy < gh && cx < gw {
                        visited[cy * gw + cx] = true;
                    }
                }
            }
            // Block position: centre for overlap, top-left for no-overlap
            let stride = lot_stride(bs, use_overlap);
            let (by, bx) = if stride < bs {
                (gy * cell + bs / 2, gx * cell + bs / 2)
            } else {
                (gy * cell, gx * cell)
            };
            blocks.push((by, bx, bs));
        }
    }
    blocks
}

/// Analyze a full image with variable block sizes determined by `size_grid`.
pub fn lot_analyze_variable(
    image: &[f64],
    h: usize,
    w: usize,
    size_grid: &[u8],
    gh: usize,
    gw: usize,
    use_overlap: bool,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<(usize, usize, usize)>) {
    let blocks = iter_blocks(size_grid, gh, gw, use_overlap);

    // Parallel block analysis: each block's DCT is independent
    let results: Vec<(f64, Vec<f64>)> = blocks.par_iter()
        .map(|&(by, bx, bs)| {
            let coeffs = lot_analyze_block(image, h, w, by, bx, bs, use_overlap);
            let dc_scale = 256.0 / (bs * bs) as f64;
            let ac_scale = 16.0 / bs as f64;
            let dc = coeffs[0] * dc_scale;
            let ac: Vec<f64> = coeffs[1..].iter().map(|&c| c * ac_scale).collect();
            (dc, ac)
        })
        .collect();

    let mut dc_grid = Vec::with_capacity(blocks.len());
    let mut ac_blocks = Vec::with_capacity(blocks.len());
    for (dc, ac) in results {
        dc_grid.push(dc);
        ac_blocks.push(ac);
    }

    (dc_grid, ac_blocks, blocks)
}

/// Synthesize a full image from variable-size LOT blocks.
pub fn lot_synthesize_variable(
    dc_grid: &[f64],
    ac_blocks: &[Vec<f64>],
    blocks: &[(usize, usize, usize)],
    h: usize,
    w: usize,
    use_overlap: bool,
) -> Vec<f64> {
    let n = h * w;

    // Phase 1: Parallel IDCT + windowing for all blocks
    let spatial_data: Vec<(usize, usize, usize, Vec<f64>)> = blocks.par_iter().enumerate()
        .map(|(i, &(by, bx, bs))| {
            let dc_inv_scale = (bs * bs) as f64 / 256.0;
            let ac_inv_scale = bs as f64 / 16.0;

            let mut coeffs = Vec::with_capacity(bs * bs);
            coeffs.push(dc_grid[i] * dc_inv_scale);
            for &c in &ac_blocks[i] {
                coeffs.push(c * ac_inv_scale);
            }

            let stride = lot_stride(bs, use_overlap);
            let use_window = stride < bs;
            let mut spatial = idct_2d(&coeffs, bs, bs);
            if use_window {
                let win = cached_sine_window(bs);
                apply_window_2d(&mut spatial, bs, bs, win, win);
            }
            (by, bx, bs, spatial)
        })
        .collect();

    // Phase 2: Sequential overlap-add
    let mut output = vec![0.0; n];
    let mut weight = vec![0.0; n];

    for &(by, bx, bs, ref spatial) in &spatial_data {
        let stride = lot_stride(bs, use_overlap);
        let use_window = stride < bs;
        let offset = if use_window { bs / 2 } else { 0 };
        let win = if use_window { cached_sine_window(bs) } else { &[] };

        for r in 0..bs {
            let py = by as i64 - offset as i64 + r as i64;
            if py < 0 || py >= h as i64 { continue; }
            let py = py as usize;
            let wh = if use_window { win[r] } else { 1.0 };
            for c in 0..bs {
                let px = bx as i64 - offset as i64 + c as i64;
                if px < 0 || px >= w as i64 { continue; }
                let px = px as usize;
                let ww = if use_window { win[c] } else { 1.0 };
                let w2 = wh * wh * ww * ww;
                output[py * w + px] += spatial[r * bs + c];
                weight[py * w + px] += w2;
            }
        }
    }

    // Phase 3: Parallel normalization
    output.par_iter_mut().zip(weight.par_iter()).for_each(|(o, &wt)| {
        if wt > 1e-15 { *o /= wt; }
    });
    output
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_roundtrip_1d() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let freq = dct_ii(&input);
        let back = idct_ii(&freq);
        for (a, b) in input.iter().zip(back.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "1-D roundtrip mismatch: {} vs {} (err {})",
                a,
                b,
                (a - b).abs()
            );
        }
    }

    #[test]
    fn test_dct_roundtrip_2d() {
        let h = 8;
        let w = 8;
        let mut block = vec![0.0; h * w];
        for (i, v) in block.iter_mut().enumerate() {
            *v = (i as f64 * 0.37).sin() * 100.0 + 50.0;
        }
        let freq = dct_2d(&block, h, w);
        let back = idct_2d(&freq, h, w);
        for (a, b) in block.iter().zip(back.iter()) {
            assert!(
                (a - b).abs() < 1e-8,
                "2-D roundtrip mismatch: {} vs {} (err {})",
                a,
                b,
                (a - b).abs()
            );
        }
    }

    #[test]
    fn test_lot_image_roundtrip() {
        let h = 64;
        let w = 64;
        let mut image = vec![0.0; h * w];
        for r in 0..h {
            for c in 0..w {
                // Gradient + sine pattern
                image[r * w + c] = 50.0
                    + 100.0 * (r as f64 / h as f64)
                    + 30.0 * (PI * c as f64 / 8.0).sin();
            }
        }

        let block_size = 16;
        let (dc, ac, gh, gw) = lot_analyze_image(&image, h, w, block_size, false);
        let recon = lot_synthesize_image(&dc, &ac, gh, gw, h, w, block_size, false);

        let mut max_err = 0.0f64;
        for (a, b) in image.iter().zip(recon.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 0.1,
            "LOT image roundtrip max error = {} (should be < 0.1)",
            max_err
        );
    }

    #[test]
    fn test_lot_image_roundtrip_overlap() {
        let h = 64;
        let w = 64;
        let mut image = vec![0.0; h * w];
        for r in 0..h {
            for c in 0..w {
                image[r * w + c] = 50.0
                    + 100.0 * (r as f64 / h as f64)
                    + 30.0 * (std::f64::consts::PI * c as f64 / 8.0).sin();
            }
        }
        let block_size = 16;
        let (dc, ac, gh, gw) = lot_analyze_image(&image, h, w, block_size, true);
        let recon = lot_synthesize_image(&dc, &ac, gh, gw, h, w, block_size, true);
        let mut max_err = 0.0f64;
        for (a, b) in image.iter().zip(recon.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 0.1, "LOT overlap roundtrip max error = {} (should be < 0.1)", max_err);
    }

    #[test]
    fn test_block_map_roundtrip() {
        let grid: Vec<u8> = vec![8, 16, 32, 8, 16, 32, 8, 8, 16, 32];
        let encoded = encode_block_map(&grid);
        let decoded = decode_block_map(&encoded, grid.len());
        assert_eq!(grid, decoded, "Block map roundtrip mismatch");
    }

    #[test]
    fn test_lot_variable_roundtrip() {
        let h = 128;
        let w = 128;
        let mut image = vec![0.0; h * w];
        for r in 0..h {
            for c in 0..w {
                if r < 64 && c < 64 {
                    // Smooth quadrant — should get large blocks
                    image[r * w + c] = 128.0 + 10.0 * (r as f64 / 64.0);
                } else {
                    // Detailed quadrant — should get small blocks
                    image[r * w + c] = 100.0
                        + 50.0 * (PI * r as f64 / 4.0).sin()
                        + 30.0 * (PI * c as f64 / 3.0).cos();
                }
            }
        }

        let step = 20.0; // moderate quantization step
        let (size_grid, gh, gw) = classify_blocks(&image, h, w, step);

        let (dc, ac, blocks) = lot_analyze_variable(&image, h, w, &size_grid, gh, gw, false);
        let recon = lot_synthesize_variable(&dc, &ac, &blocks, h, w, false);

        let mut max_err = 0.0f64;
        for (a, b) in image.iter().zip(recon.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 0.5,
            "Variable LOT roundtrip max error = {} (should be < 0.5)",
            max_err
        );
    }
}
