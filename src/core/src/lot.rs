//! Lapped Orthogonal Transform (LOT) module.
//!
//! DCT-II based block transform with sine-window lapping for overlap-add
//! reconstruction. Supports fixed-size and variable-size (quadtree) blocking.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// 1. DCT-II / IDCT-II  (1-D)
// ---------------------------------------------------------------------------

/// Forward DCT-II of length N.
///
/// X[k] = sum_{n=0}^{N-1} x[n] * cos( pi/N * (n + 0.5) * k )
pub fn dct_ii(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    let mut out = vec![0.0; n];
    for k in 0..n {
        let mut sum = 0.0;
        for i in 0..n {
            sum += input[i] * (PI / nf * (i as f64 + 0.5) * k as f64).cos();
        }
        out[k] = sum;
    }
    out
}

/// Inverse DCT-II (DCT-III with normalization) of length N.
///
/// x[n] = (1/N) * X[0] + (2/N) * sum_{k=1}^{N-1} X[k] * cos( pi/N * k * (n + 0.5) )
pub fn idct_ii(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut sum = input[0]; // k=0 term (cos(0) = 1)
        for k in 1..n {
            sum += 2.0 * input[k] * (PI / nf * k as f64 * (i as f64 + 0.5)).cos();
        }
        out[i] = sum / nf;
    }
    out
}

// ---------------------------------------------------------------------------
// 2. DCT-II / IDCT-II  (2-D separable)
// ---------------------------------------------------------------------------

/// Forward 2-D DCT-II (separable: rows then columns).
pub fn dct_2d(block: &[f64], h: usize, w: usize) -> Vec<f64> {
    assert_eq!(block.len(), h * w);
    // Transform rows
    let mut tmp = vec![0.0; h * w];
    for r in 0..h {
        let row = &block[r * w..(r + 1) * w];
        let t = dct_ii(row);
        tmp[r * w..(r + 1) * w].copy_from_slice(&t);
    }
    // Transform columns
    let mut out = vec![0.0; h * w];
    let mut col_buf = vec![0.0; h];
    for c in 0..w {
        for r in 0..h {
            col_buf[r] = tmp[r * w + c];
        }
        let t = dct_ii(&col_buf);
        for r in 0..h {
            out[r * w + c] = t[r];
        }
    }
    out
}

/// Inverse 2-D DCT-II (separable: columns then rows).
pub fn idct_2d(coeffs: &[f64], h: usize, w: usize) -> Vec<f64> {
    assert_eq!(coeffs.len(), h * w);
    // Inverse-transform columns first
    let mut tmp = vec![0.0; h * w];
    let mut col_buf = vec![0.0; h];
    for c in 0..w {
        for r in 0..h {
            col_buf[r] = coeffs[r * w + c];
        }
        let t = idct_ii(&col_buf);
        for r in 0..h {
            tmp[r * w + c] = t[r];
        }
    }
    // Inverse-transform rows
    let mut out = vec![0.0; h * w];
    for r in 0..h {
        let row = &tmp[r * w..(r + 1) * w];
        let t = idct_ii(row);
        out[r * w..(r + 1) * w].copy_from_slice(&t);
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
) -> Vec<f64> {
    let stride = lot_stride(block_size);
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
        let win = sine_window(block_size);
        apply_window_2d(&mut block, block_size, block_size, &win, &win);
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
) {
    let stride = lot_stride(block_size);
    let use_window = stride < block_size;
    let offset = if use_window { block_size / 2 } else { 0 };
    let win = if use_window { sine_window(block_size) } else { vec![1.0; block_size] };

    let mut spatial = idct_2d(coeffs, block_size, block_size);
    if use_window {
        apply_window_2d(&mut spatial, block_size, block_size, &win, &win);
    }

    for r in 0..block_size {
        let py = block_y as i64 - offset as i64 + r as i64;
        if py < 0 || py >= out_h as i64 {
            continue;
        }
        let py = py as usize;
        let wh = win[r];
        for c in 0..block_size {
            let px = block_x as i64 - offset as i64 + c as i64;
            if px < 0 || px >= out_w as i64 {
                continue;
            }
            let px = px as usize;
            let ww = win[c];
            let w2 = wh * wh * ww * ww; // window^2 (separable)
            output[py * out_w + px] += spatial[r * block_size + c];
            weight[py * out_w + px] += w2;
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Full-image fixed-size LOT
// ---------------------------------------------------------------------------

/// Analyze a full image with fixed block size and 50 % overlap.
///
/// Returns `(dc_grid, ac_blocks, grid_h, grid_w)` where `dc_grid` holds one
/// DC coefficient per block and `ac_blocks` holds the remaining block_size^2 - 1
/// AC coefficients per block.
pub fn lot_analyze_image(
    image: &[f64],
    h: usize,
    w: usize,
    block_size: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, usize, usize) {
    let stride = lot_stride(block_size);
    let grid_h = (h + stride - 1) / stride;
    let grid_w = (w + stride - 1) / stride;

    let mut dc_grid = Vec::with_capacity(grid_h * grid_w);
    let mut ac_blocks = Vec::with_capacity(grid_h * grid_w);

    for gy in 0..grid_h {
        let by = gy * stride;
        for gx in 0..grid_w {
            let bx = gx * stride;
            let coeffs = lot_analyze_block(image, h, w, by, bx, block_size);
            dc_grid.push(coeffs[0]);
            ac_blocks.push(coeffs[1..].to_vec());
        }
    }

    (dc_grid, ac_blocks, grid_h, grid_w)
}

/// Synthesize a full image from DC grid + AC blocks (fixed block size, 50 %
/// overlap).  Normalizes by accumulated window weight.
pub fn lot_synthesize_image(
    dc_grid: &[f64],
    ac_blocks: &[Vec<f64>],
    grid_h: usize,
    grid_w: usize,
    h: usize,
    w: usize,
    block_size: usize,
) -> Vec<f64> {
    let stride = lot_stride(block_size);
    let n = h * w;
    let mut output = vec![0.0; n];
    let mut weight = vec![0.0; n];

    for gy in 0..grid_h {
        let by = gy * stride;
        for gx in 0..grid_w {
            let bx = gx * stride;
            let idx = gy * grid_w + gx;
            let mut coeffs = Vec::with_capacity(block_size * block_size);
            coeffs.push(dc_grid[idx]);
            coeffs.extend_from_slice(&ac_blocks[idx]);
            lot_synthesize_block(
                &coeffs, block_size, &mut output, &mut weight, h, w, by, bx,
            );
        }
    }

    // Normalize by accumulated window weight
    for i in 0..n {
        if weight[i] > 1e-15 {
            output[i] /= weight[i];
        }
    }
    output
}

// ---------------------------------------------------------------------------
// 7. Block classification (quadtree)
// ---------------------------------------------------------------------------

/// Allowed LOT block sizes (must be powers of 2).
pub const BLOCK_SIZES: [usize; 3] = [8, 16, 32];

/// 3D Codon: luminance × saturation × local detail → step factor.
/// "Gravitational lens" effect: perceptually important zones (high saturation,
/// high detail, dark regions) bend the quantization field toward finer steps.
///
/// Returns a factor in [0.15, 2.0]. Lower = finer quantization = more bits.
///
/// Arguments:
/// - `dc_l`: L channel DC value (PTF space, ~0-255)
/// - `dc_c1`: C1 channel DC value (blue chroma)
/// - `dc_c2`: C2 channel DC value (red chroma)
/// - `ac_energy`: sum of |AC coefficients| in this block (from L channel)
/// - `n_ac`: number of AC coefficients
/// Codon factor using only DC-derived dimensions (luminance + saturation).
/// CRITICAL: uses only information available identically to encoder AND decoder
/// (DC values). AC-based dimensions were removed because encoder uses original
/// AC energy while decoder uses quantized AC energy → systematic mismatch.
pub fn codon_3d_factor(dc_l: f64, dc_c1: f64, dc_c2: f64, _ac_energy: f64, _n_ac: usize) -> f64 {
    codon_dc_factor(dc_l, dc_c1, dc_c2)
}

/// Core codon factor from DC values only (no encoder/decoder mismatch).
pub fn codon_dc_factor(dc_l: f64, dc_c1: f64, dc_c2: f64) -> f64 {
    use crate::calibration;

    // Dimension 1: Luminance (Weber-Fechner, calibrated thresholds)
    let lum_factor = if dc_l < calibration::CODON_LUM_THRESHOLDS[0] {
        calibration::CODON_TRNA[0]
    } else if dc_l < calibration::CODON_LUM_THRESHOLDS[1] {
        calibration::CODON_TRNA[1]
    } else if dc_l < calibration::CODON_LUM_THRESHOLDS[2] {
        calibration::CODON_TRNA[2]
    } else {
        calibration::CODON_TRNA[3]
    };

    // Dimension 2: Saturation (chroma energy)
    let chroma_energy = (dc_c1 * dc_c1 + dc_c2 * dc_c2).sqrt();
    let sat_factor = if chroma_energy > calibration::CODON_SAT_THRESHOLD {
        calibration::CODON_SAT_FACTOR
    } else {
        1.0
    };

    let combined: f64 = lum_factor * sat_factor;
    combined.clamp(0.15, 2.0)
}

/// Codon 4D factor: adds structural coherence dimension (Point 4).
/// NOTE: structural coherence also removed from step computation due to
/// encoder/decoder mismatch. Kept as analysis function.
pub fn codon_4d_factor(
    dc_l: f64, dc_c1: f64, dc_c2: f64,
    _ac_block: &[f64], _block_size: usize,
) -> f64 {
    codon_dc_factor(dc_l, dc_c1, dc_c2)
}

/// LOT stride as fraction of block_size. Controls overlap amount.
/// block_size/2 = 50% overlap (4× redundancy, best quality, high bitrate)
/// block_size*3/4 = 25% overlap (~1.78× redundancy, good quality, lower bitrate)
/// block_size = 0% overlap (no redundancy, blocking artifacts)
pub fn lot_stride(block_size: usize) -> usize {
    block_size  // No overlap: critical sampling
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
/// Maps (row, col) in a block of any size to the equivalent QMAT_16 value.
/// Uses bilinear interpolation into the calibrated 16x16 matrix.
pub fn qmat_for_block_size(row: usize, col: usize, block_size: usize) -> f64 {
    if block_size == 16 {
        return QMAT_16[row * 16 + col];
    }
    // Map to float position in QMAT_16
    let nr = row as f64 * 15.0 / (block_size - 1).max(1) as f64;
    let nc = col as f64 * 15.0 / (block_size - 1).max(1) as f64;
    let r0 = (nr as usize).min(14);
    let c0 = (nc as usize).min(14);
    let dr = nr - r0 as f64;
    let dc = nc - c0 as f64;
    // Bilinear interpolation
    let v00 = QMAT_16[r0 * 16 + c0];
    let v01 = QMAT_16[r0 * 16 + c0 + 1];
    let v10 = QMAT_16[(r0 + 1) * 16 + c0];
    let v11 = QMAT_16[(r0 + 1) * 16 + c0 + 1];
    v00 * (1.0 - dr) * (1.0 - dc) + v01 * (1.0 - dr) * dc
        + v10 * dr * (1.0 - dc) + v11 * dr * dc
}

/// Calibrated quantization matrix for 16x16 LOT blocks (row-major, 256 entries).
/// Factor per position. Multiply by base_step for actual quantization step.
/// DC (index 0) = 0.0 (handled separately).
/// Calibrated from 12 HD images (248,880 blocks), energy-inverse weighting.
#[rustfmt::skip]
pub const QMAT_16: [f64; 256] = [
    0.0,  1.1,  2.4,  3.8,  5.2,  6.4,  7.8,  9.7, 12.1, 14.0, 16.0, 18.0, 20.6, 24.2, 28.0, 30.5,
    1.0,  2.2,  3.5,  5.1,  6.5,  7.9,  9.4, 11.6, 14.3, 16.5, 18.6, 20.9, 23.4, 27.4, 31.8, 36.6,
    2.1,  3.2,  4.3,  5.7,  7.0,  8.4,  9.9, 12.3, 15.0, 17.1, 19.3, 21.6, 24.3, 28.2, 33.2, 38.0,
    3.2,  4.4,  5.3,  6.5,  7.7,  9.0, 10.7, 13.0, 15.7, 17.8, 19.9, 22.3, 24.9, 28.6, 33.4, 37.3,
    4.4,  5.5,  6.3,  7.4,  8.5, 10.0, 11.7, 13.8, 16.4, 18.6, 20.8, 22.9, 25.3, 28.5, 33.2, 34.2,
    5.4,  6.6,  7.5,  8.5,  9.7, 11.1, 12.7, 14.7, 17.4, 19.6, 21.5, 23.7, 26.2, 29.3, 33.5, 34.8,
    6.7,  8.0,  8.9,  9.9, 11.2, 12.6, 14.2, 16.2, 18.6, 20.6, 22.6, 24.7, 27.2, 30.4, 34.4, 38.2,
    8.3,  9.7, 10.8, 11.9, 13.0, 14.4, 16.0, 17.9, 20.0, 22.1, 23.9, 25.8, 28.6, 32.2, 36.3, 39.6,
   10.5, 12.2, 13.4, 14.3, 15.5, 16.8, 18.3, 20.1, 21.1, 23.6, 25.6, 28.0, 31.1, 34.9, 39.0, 41.8,
   12.4, 13.9, 15.1, 16.2, 17.3, 18.4, 20.0, 21.4, 22.3, 24.5, 26.9, 29.5, 32.8, 36.7, 40.9, 44.0,
   14.4, 15.9, 17.2, 18.4, 19.3, 20.4, 21.6, 23.1, 24.1, 26.5, 29.3, 32.0, 35.5, 39.4, 43.4, 46.9,
   16.3, 18.1, 19.3, 20.5, 21.5, 22.6, 23.8, 24.9, 26.2, 28.8, 31.7, 35.0, 38.6, 42.3, 46.6, 49.6,
   18.8, 20.7, 21.9, 23.3, 24.2, 25.3, 26.4, 27.5, 29.4, 31.9, 34.8, 38.0, 41.8, 45.8, 49.8, 52.2,
   21.3, 23.3, 24.7, 26.1, 27.3, 27.6, 29.1, 27.7, 28.6, 33.4, 38.4, 41.2, 44.9, 48.2, 50.7, 51.7,
   24.6, 26.7, 28.7, 30.0, 31.3, 31.2, 32.9, 30.8, 31.2, 36.6, 42.1, 44.9, 48.4, 50.9, 53.1, 52.9,
   27.4, 32.1, 33.9, 34.9, 35.7, 36.2, 37.2, 37.2, 38.2, 41.9, 45.5, 47.9, 51.1, 52.0, 54.0, 48.7,
];

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

    // Start with smallest size, then try to merge into larger blocks
    let mut size_grid = vec![8u8; grid_h * grid_w];

    // Try 16: merge 2x2 groups of 8-cells
    let g16 = 2usize; // 16/8
    for gy in (0..grid_h).step_by(g16) {
        for gx in (0..grid_w).step_by(g16) {
            let mut max_var = 0.0f64;
            let mut all_present = true;
            for dy in 0..g16 {
                for dx in 0..g16 {
                    let cy = gy + dy;
                    let cx = gx + dx;
                    if cy >= grid_h || cx >= grid_w {
                        all_present = false;
                        continue;
                    }
                    max_var = max_var.max(var_grid[cy * grid_w + cx]);
                }
            }
            if all_present && max_var < merge_thresh {
                for dy in 0..g16 {
                    for dx in 0..g16 {
                        size_grid[(gy + dy) * grid_w + (gx + dx)] = 16;
                    }
                }
            }
        }
    }

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
            if all_present && all_16 && max_var < merge_thresh {
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
            let stride = lot_stride(bs);
            let (by, bx) = if stride < bs {
                (gy * cell + bs / 2, gx * cell + bs / 2) // overlap: centre
            } else {
                (gy * cell, gx * cell) // no overlap: top-left
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
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<(usize, usize, usize)>) {
    let blocks = iter_blocks(size_grid, gh, gw);
    let mut dc_grid = Vec::with_capacity(blocks.len());
    let mut ac_blocks = Vec::with_capacity(blocks.len());

    for &(by, bx, bs) in &blocks {
        let coeffs = lot_analyze_block(image, h, w, by, bx, bs);
        dc_grid.push(coeffs[0]);
        ac_blocks.push(coeffs[1..].to_vec());
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
) -> Vec<f64> {
    let n = h * w;
    let mut output = vec![0.0; n];
    let mut weight = vec![0.0; n];

    for (i, &(by, bx, bs)) in blocks.iter().enumerate() {
        let mut coeffs = Vec::with_capacity(bs * bs);
        coeffs.push(dc_grid[i]);
        coeffs.extend_from_slice(&ac_blocks[i]);
        lot_synthesize_block(&coeffs, bs, &mut output, &mut weight, h, w, by, bx);
    }

    for i in 0..n {
        if weight[i] > 1e-15 {
            output[i] /= weight[i];
        }
    }
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
        let (dc, ac, gh, gw) = lot_analyze_image(&image, h, w, block_size);
        let recon = lot_synthesize_image(&dc, &ac, gh, gw, h, w, block_size);

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

        let (dc, ac, blocks) = lot_analyze_variable(&image, h, w, &size_grid, gh, gw);
        let recon = lot_synthesize_variable(&dc, &ac, &blocks, h, w);

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
