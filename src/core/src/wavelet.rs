//! CDF 9/7 wavelet transform (same as JPEG 2000).
//!
//! Lifting scheme with 4 steps + normalization.
//! Used in v6 pipeline: multi-level decomposition on Y/Cb/Cr channels.

use ndarray::{Array2, s};

// CDF 9/7 lifting coefficients
const ALPHA: f64 = -1.58613434205992;
const BETA: f64 = -0.05298011857296;
const GAMMA: f64 = 0.88291107553093;
const DELTA: f64 = 0.44350685204397;
const K: f64 = 1.149604398863410;

/// Forward CDF 9/7 on all rows simultaneously.
/// Input: (H, W) → Output: (even columns, odd columns) each (H, ne) and (H, no)
fn cdf97_fwd_rows(data: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let (_h, w) = (data.nrows(), data.ncols());
    let ne = (w + 1) / 2;
    let no = w / 2;

    // Split even/odd columns
    let mut even = Array2::<f64>::zeros((data.nrows(), ne));
    let mut odd = Array2::<f64>::zeros((data.nrows(), no));

    for j in 0..ne {
        even.column_mut(j).assign(&data.column(2 * j));
    }
    for j in 0..no {
        odd.column_mut(j).assign(&data.column(2 * j + 1));
    }

    // Step 1: odd += alpha * (even[j] + even[j+1])
    for j in 0..no {
        let j_right = (j + 1).min(ne - 1);
        for i in 0..data.nrows() {
            odd[[i, j]] += ALPHA * (even[[i, j.min(ne - 1)]] + even[[i, j_right]]);
        }
    }

    // Step 2: even += beta * (odd[j-1] + odd[j])
    for j in 0..ne {
        let j_left = if j == 0 { 0 } else { (j - 1).min(no - 1) };
        let j_cur = j.min(no - 1);
        for i in 0..data.nrows() {
            even[[i, j]] += BETA * (odd[[i, j_left]] + odd[[i, j_cur]]);
        }
    }

    // Step 3: odd += gamma * (even[j] + even[j+1])
    for j in 0..no {
        let j_right = (j + 1).min(ne - 1);
        for i in 0..data.nrows() {
            odd[[i, j]] += GAMMA * (even[[i, j.min(ne - 1)]] + even[[i, j_right]]);
        }
    }

    // Step 4: even += delta * (odd[j-1] + odd[j])
    for j in 0..ne {
        let j_left = if j == 0 { 0 } else { (j - 1).min(no - 1) };
        let j_cur = j.min(no - 1);
        for i in 0..data.nrows() {
            even[[i, j]] += DELTA * (odd[[i, j_left]] + odd[[i, j_cur]]);
        }
    }

    // Normalize
    even.mapv_inplace(|v| v / K);
    odd.mapv_inplace(|v| v * K);

    (even, odd)
}

/// Inverse CDF 9/7 on all rows simultaneously.
/// Input: (even, odd) → Output: (H, W) with W = ne + no
fn cdf97_inv_rows(even_in: &Array2<f64>, odd_in: &Array2<f64>) -> Array2<f64> {
    let ne = even_in.ncols();
    let no = odd_in.ncols();
    let w = ne + no;
    let h = even_in.nrows();

    let mut even = even_in.mapv(|v| v * K);
    let mut odd = odd_in.mapv(|v| v / K);

    // Step 4 inverse
    for j in 0..ne {
        let j_left = if j == 0 { 0 } else { (j - 1).min(no - 1) };
        let j_cur = j.min(no - 1);
        for i in 0..h {
            even[[i, j]] -= DELTA * (odd[[i, j_left]] + odd[[i, j_cur]]);
        }
    }

    // Step 3 inverse
    for j in 0..no {
        let j_right = (j + 1).min(ne - 1);
        for i in 0..h {
            odd[[i, j]] -= GAMMA * (even[[i, j.min(ne - 1)]] + even[[i, j_right]]);
        }
    }

    // Step 2 inverse
    for j in 0..ne {
        let j_left = if j == 0 { 0 } else { (j - 1).min(no - 1) };
        let j_cur = j.min(no - 1);
        for i in 0..h {
            even[[i, j]] -= BETA * (odd[[i, j_left]] + odd[[i, j_cur]]);
        }
    }

    // Step 1 inverse
    for j in 0..no {
        let j_right = (j + 1).min(ne - 1);
        for i in 0..h {
            odd[[i, j]] -= ALPHA * (even[[i, j.min(ne - 1)]] + even[[i, j_right]]);
        }
    }

    // Interleave
    let mut result = Array2::<f64>::zeros((h, w));
    for j in 0..ne {
        result.column_mut(2 * j).assign(&even.column(j));
    }
    for j in 0..no {
        result.column_mut(2 * j + 1).assign(&odd.column(j));
    }

    result
}

/// 2D forward CDF 9/7 transform.
/// Returns (LL, LH, HL, HH) subbands.
pub fn cdf97_forward_2d(image: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    // Rows
    let (row_lo, row_hi) = cdf97_fwd_rows(image);

    // Columns on row_lo (transpose, apply rows, transpose back)
    let row_lo_t = row_lo.t().to_owned();
    let (ll_t, lh_t) = cdf97_fwd_rows(&row_lo_t);
    let ll = ll_t.t().to_owned();
    let lh = lh_t.t().to_owned();

    // Columns on row_hi
    let row_hi_t = row_hi.t().to_owned();
    let (hl_t, hh_t) = cdf97_fwd_rows(&row_hi_t);
    let hl = hl_t.t().to_owned();
    let hh = hh_t.t().to_owned();

    (ll, lh, hl, hh)
}

/// 2D inverse CDF 9/7 transform.
/// Reconstructs from (LL, LH, HL, HH), clips to (orig_h, orig_w).
pub fn cdf97_inverse_2d(
    ll: &Array2<f64>, lh: &Array2<f64>,
    hl: &Array2<f64>, hh: &Array2<f64>,
    orig_h: usize, orig_w: usize,
) -> Array2<f64> {
    // Inverse columns
    let row_lo = cdf97_inv_rows(&ll.t().to_owned(), &lh.t().to_owned()).t().to_owned();
    let row_hi = cdf97_inv_rows(&hl.t().to_owned(), &hh.t().to_owned()).t().to_owned();

    // Inverse rows
    let result = cdf97_inv_rows(&row_lo, &row_hi);
    result.slice(s![..orig_h, ..orig_w]).to_owned()
}

/// Subband sizes for CDF 9/7 decomposition of input (h, w).
/// Returns [(lh_h, lh_w), (hl_h, hl_w), (hh_h, hh_w)]
pub fn detail_band_sizes(h: usize, w: usize) -> [(usize, usize); 3] {
    [
        (h / 2, (w + 1) / 2),       // LH
        ((h + 1) / 2, w / 2),       // HL
        (h / 2, w / 2),             // HH
    ]
}

/// Multi-level wavelet decomposition.
/// Returns (LL, subbands, sizes) where:
///   subbands[level] = (LH, HL, HH)
///   sizes[level] = (input_h, input_w) at that level
pub fn wavelet_decompose(
    channel: &Array2<f64>, levels: usize,
) -> (Array2<f64>, Vec<(Array2<f64>, Array2<f64>, Array2<f64>)>, Vec<(usize, usize)>) {
    let mut current = channel.clone();
    let mut subbands = Vec::with_capacity(levels);
    let mut sizes = Vec::with_capacity(levels);

    for _lv in 0..levels {
        let (h, w) = (current.nrows(), current.ncols());
        sizes.push((h, w));
        let (ll, lh, hl, hh) = cdf97_forward_2d(&current);
        subbands.push((lh, hl, hh));
        current = ll;
    }

    (current, subbands, sizes)
}

/// Multi-level wavelet recomposition (inverse).
pub fn wavelet_recompose(
    ll: &Array2<f64>,
    subbands: &[(Array2<f64>, Array2<f64>, Array2<f64>)],
    sizes: &[(usize, usize)],
) -> Array2<f64> {
    let mut current = ll.clone();
    for lv in (0..subbands.len()).rev() {
        let (h, w) = sizes[lv];
        let (ref lh, ref hl, ref hh) = subbands[lv];
        current = cdf97_inverse_2d(&current, lh, hl, hh, h, w);
    }
    current
}

/// Dead zone factor: effective threshold = (0.5 + DEAD_ZONE) * step
pub const DEAD_ZONE: f64 = 0.22;

/// Weber-Fechner luminance-adaptive factor for v9.
/// Dark pixels (L=0) get finer quantization (factor=0.5),
/// bright pixels (L=1) get base quantization (factor=1.0).
#[inline]
pub fn weber_factor(l_norm: f64) -> f64 {
    use crate::golden::PHI_INV2;
    let raw = l_norm.powf(0.4) * (1.0 - PHI_INV2) + PHI_INV2;
    raw.clamp(0.5, 1.0)
}

/// Sidelobe suppression: zero out small coefficients adjacent to large ones.
/// CDF 9/7 creates oscillating sidelobes (support ~4px) around edges.
/// After quantization these become structured "golf ball" artifacts.
/// Suppressing them BEFORE quantization eliminates the pattern.
pub fn suppress_sidelobes(band: &mut Array2<f64>, step: f64) {
    let h = band.nrows();
    let w = band.ncols();
    if h < 5 || w < 5 { return; }

    const RADIUS: usize = 3;         // tight: first sidelobe lobe only
    const SMALL_THRESH: f64 = 0.4;   // only the very smallest
    const LARGE_THRESH: f64 = 3.0;   // only near very strong edges

    let small_abs = SMALL_THRESH * step;
    let large_abs = LARGE_THRESH * step;

    // First pass: find large coefficients (edges)
    let mut is_large = vec![false; h * w];
    for i in 0..h {
        for j in 0..w {
            if band[[i, j]].abs() > large_abs {
                is_large[i * w + j] = true;
            }
        }
    }

    // Dilate the large mask by RADIUS (any pixel within RADIUS of a large coeff)
    let mut near_large = vec![false; h * w];
    for i in 0..h {
        for j in 0..w {
            if !is_large[i * w + j] { continue; }
            let i_lo = i.saturating_sub(RADIUS);
            let i_hi = (i + RADIUS).min(h - 1);
            let j_lo = j.saturating_sub(RADIUS);
            let j_hi = (j + RADIUS).min(w - 1);
            for ii in i_lo..=i_hi {
                for jj in j_lo..=j_hi {
                    near_large[ii * w + jj] = true;
                }
            }
        }
    }

    // Second pass: zero small coefficients that are near large ones (sidelobes)
    for i in 0..h {
        for j in 0..w {
            let idx = i * w + j;
            if near_large[idx] && !is_large[idx] && band[[i, j]].abs() < small_abs {
                band[[i, j]] = 0.0;
            }
        }
    }
}

/// Quantize a detail band with dead zone: sign(x) * max(0, floor(|x|/step + 0.5 - dz))
pub fn quantize_band(band: &Array2<f64>, step: f64) -> Array2<f64> {
    band.mapv(|v| {
        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
        let q = (v.abs() / step + 0.5 - DEAD_ZONE).floor();
        if q > 0.0 { sign * q } else { 0.0 }
    })
}

/// Quantize with a per-pixel step map (same dead-zone formula).
pub fn quantize_band_map(band: &Array2<f64>, step_map: &[f64]) -> Array2<f64> {
    let h = band.nrows();
    let w = band.ncols();
    let mut q = Array2::<f64>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            let v = band[[i, j]];
            let s = step_map[i * w + j];
            let sign = if v >= 0.0 { 1.0 } else { -1.0 };
            let qv = (v.abs() / s + 0.5 - DEAD_ZONE).floor();
            q[[i, j]] = if qv > 0.0 { sign * qv } else { 0.0 };
        }
    }
    q
}

/// Quantize with per-pixel step map AND per-pixel dead zone (Weber-Fechner).
/// dz_map[i] in [0, 0.15]: dark regions get smaller dead zone -> more coefficients survive.
pub fn quantize_band_weber(band: &Array2<f64>, step_map: &[f64], dz_map: &[f64]) -> Array2<f64> {
    let h = band.nrows();
    let w = band.ncols();
    let mut q = Array2::<f64>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            let v = band[[i, j]];
            let idx = i * w + j;
            let s = step_map[idx];
            let dz = dz_map[idx];
            let sign = if v >= 0.0 { 1.0 } else { -1.0 };
            let qv = (v.abs() / s + 0.5 - dz).floor();
            q[[i, j]] = if qv > 0.0 { sign * qv } else { 0.0 };
        }
    }
    q
}

/// Sigma-delta noise-shaped quantization (telecom/ADC technique).
/// Feeds back quantization error to the next coefficient, converting
/// structured banding (low-freq error) into invisible high-freq noise.
/// Encoder-only: the decoder just dequantizes normally.
pub fn quantize_band_noise_shaped(band: &Array2<f64>, step: f64) -> Array2<f64> {
    let h = band.nrows();
    let w = band.ncols();
    let mut q = Array2::<f64>::zeros((h, w));
    const FEEDBACK: f64 = 0.7;

    for i in 0..h {
        let mut err = 0.0;
        for j in 0..w {
            // Inject previous error into current coefficient
            let c = band[[i, j]] + FEEDBACK * err;
            let sign = if c >= 0.0 { 1.0 } else { -1.0 };
            let qv = (c.abs() / step + 0.5 - DEAD_ZONE).floor();
            let quantized = if qv > 0.0 { sign * qv } else { 0.0 };
            q[[i, j]] = quantized;
            // Error = what we wanted minus what we got
            err = band[[i, j]] - quantized * step;
        }
    }
    q
}

/// Dequantize: multiply by step
pub fn dequantize_band(qband: &Array2<f64>, step: f64) -> Array2<f64> {
    qband.mapv(|v| v * step)
}

/// Compute a per-pixel dead zone map from a luminance reference (downsampled L).
/// Dark pixels get near-zero dead zone (preserve detail), bright pixels get wider.
/// `l_ref` must have exactly `band_h * band_w` elements, values in [0, 255].
pub fn luminance_dz_map(l_ref: &[f64], band_h: usize, band_w: usize) -> Vec<f64> {
    let n = band_h * band_w;
    // Robust normalization (5th / 95th percentile)
    let mut sorted: Vec<f64> = l_ref.iter().copied().collect();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx_05 = ((n as f64 * 0.05) as usize).min(n.saturating_sub(1));
    let idx_95 = ((n as f64 * 0.95) as usize).min(n.saturating_sub(1));
    let l_min = sorted[idx_05];
    let l_max = sorted[idx_95].max(l_min + 1.0);

    const DZ_DARK: f64 = 0.02;  // near-zero: preserve dark detail
    const DZ_BRIGHT: f64 = 0.25; // wider than default 0.15: save bits in brights

    l_ref.iter().map(|&v| {
        let l_norm = ((v - l_min) / (l_max - l_min)).clamp(0.0, 1.0);
        DZ_DARK + (DZ_BRIGHT - DZ_DARK) * l_norm.sqrt()
    }).collect()
}

/// Dequantize with a per-pixel step map.
pub fn dequantize_band_map(qband: &Array2<f64>, step_map: &[f64]) -> Array2<f64> {
    let h = qband.nrows();
    let w = qband.ncols();
    let mut out = Array2::<f64>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            out[[i, j]] = qband[[i, j]] * step_map[i * w + j];
        }
    }
    out
}

/// Build a block AQ map from the L channel (PTF space).
/// Each 32x32 block stores its mean luminance as u8.
/// Returns (map, blocks_h, blocks_w).
pub const AQ_BLOCK: usize = 32;

pub fn build_block_aq_map(l_ch: &[f64], h: usize, w: usize) -> (Vec<u8>, usize, usize) {
    let bh = (h + AQ_BLOCK - 1) / AQ_BLOCK;
    let bw = (w + AQ_BLOCK - 1) / AQ_BLOCK;
    let mut map = Vec::with_capacity(bh * bw);

    for by in 0..bh {
        for bx in 0..bw {
            let y0 = by * AQ_BLOCK;
            let x0 = bx * AQ_BLOCK;
            let y1 = (y0 + AQ_BLOCK).min(h);
            let x1 = (x0 + AQ_BLOCK).min(w);
            let mut sum = 0.0;
            let mut count = 0u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += l_ch[y * w + x];
                    count += 1;
                }
            }
            let mean = sum / count.max(1) as f64;
            map.push(mean.clamp(0.0, 255.0) as u8);
        }
    }
    (map, bh, bw)
}

/// Convert a block AQ map entry (mean luminance u8) to a step factor.
/// Weber-Fechner: factor = (L / 128)^0.5, clamped [0.25, 4.0].
/// Dark blocks get finer step, bright blocks get coarser step.
#[inline]
pub fn aq_block_factor(lum: u8) -> f64 {
    let ratio = lum.max(1) as f64 / 128.0;
    ratio.sqrt().clamp(0.25, 4.0)
}

/// Build a per-pixel step map for a detail subband from the block AQ map.
/// The subband at level `lv` has dimensions (band_h, band_w).
/// The block map covers the full image at 32x32 granularity.
/// Level 0 subbands are ~half image size, level 1 ~quarter, etc.
pub fn aq_step_map_for_band(
    aq_map: &[u8], map_bh: usize, map_bw: usize,
    band_h: usize, band_w: usize,
    img_h: usize, img_w: usize,
    base_step: f64,
) -> Vec<f64> {
    let mut step_map = Vec::with_capacity(band_h * band_w);
    // Spatial scale: band pixel (i,j) corresponds to image pixel (i*sy, j*sx)
    let sy = if band_h > 1 { (img_h - 1) as f64 / (band_h - 1) as f64 } else { 0.0 };
    let sx = if band_w > 1 { (img_w - 1) as f64 / (band_w - 1) as f64 } else { 0.0 };

    for i in 0..band_h {
        for j in 0..band_w {
            let img_y = (i as f64 * sy) as usize;
            let img_x = (j as f64 * sx) as usize;
            let by = (img_y / AQ_BLOCK).min(map_bh.saturating_sub(1));
            let bx = (img_x / AQ_BLOCK).min(map_bw.saturating_sub(1));
            let lum = aq_map[by * map_bw + bx];
            step_map.push(base_step * aq_block_factor(lum));
        }
    }
    step_map
}

/// Codon-adaptive step map: 4-zone tRNA table.
/// Maps local luminance from decoded LL to a step multiplier.
/// Dark regions get finer quantization (Weber-Fechner).
pub fn codon_step_map(
    ll_decoded: &[f64], ll_h: usize, ll_w: usize,
    band_h: usize, band_w: usize,
    _img_h: usize, _img_w: usize,
    base_step: f64,
) -> Vec<f64> {
    use crate::calibration;
    let trna = calibration::CODON_TRNA;
    let thresholds = calibration::CODON_LUM_THRESHOLDS;

    let mut step_map = Vec::with_capacity(band_h * band_w);
    let sy = if band_h > 1 { (ll_h as f64 - 1.0) / (band_h as f64 - 1.0) } else { 0.0 };
    let sx = if band_w > 1 { (ll_w as f64 - 1.0) / (band_w as f64 - 1.0) } else { 0.0 };

    for i in 0..band_h {
        for j in 0..band_w {
            let ll_y = (i as f64 * sy).round() as usize;
            let ll_x = (j as f64 * sx).round() as usize;
            let ll_y = ll_y.min(ll_h.saturating_sub(1));
            let ll_x = ll_x.min(ll_w.saturating_sub(1));
            let lum = ll_decoded[ll_y * ll_w + ll_x];

            let zone = if lum < thresholds[0] { 0 }
                       else if lum < thresholds[1] { 1 }
                       else if lum < thresholds[2] { 2 }
                       else { 3 };

            step_map.push(base_step * trna[zone]);
        }
    }
    step_map
}

/// Chroma detail factor (v6+: 2.0x coarser than Y)
pub const CHROMA_DETAIL_FACTOR: f64 = 1.5;

/// Perceptual band weights (QGA-optimized)
pub const PERCEPTUAL_BAND_WEIGHTS: [f64; 3] = [1.3, 0.65, 1.3]; // LH, HL, HH

/// v9: symmetric LH/HL to eliminate horizontal banding asymmetry.
pub const PERCEPTUAL_BAND_WEIGHTS_V9: [f64; 3] = [1.0, 1.0, 1.3];

/// Level scales (degressive: deeper levels quantized more finely)
pub const LEVEL_SCALES: [f64; 4] = [1.0, 0.9, 0.5, 0.3];

/// Flags for v6+ improvements
pub const FLAG_ZIGZAG: u8 = 0x01;
pub const FLAG_INTERSCALE: u8 = 0x02;
pub const FLAG_ADAPTIVE_QUANT: u8 = 0x04;
pub const FLAG_MORTON_2BIT: u8 = 0x08;
pub const FLAG_CONTEXT: u8 = 0x10;
pub const DEFAULT_FLAGS: u8 = FLAG_ZIGZAG | FLAG_INTERSCALE | FLAG_ADAPTIVE_QUANT | FLAG_MORTON_2BIT;

// ======================================================================
// Morton Z-order scan (better 2D locality for LZMA)
// ======================================================================

/// Compute Z-order for a pixel (x, y) via bit interleaving.
#[inline]
fn spread_bits(v: u32) -> u32 {
    let mut v = v;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    v
}

/// Build Morton Z-order for an h x w grid.
/// Returns a vector of raster indices sorted by their Z value.
pub fn morton_order(h: usize, w: usize) -> Vec<usize> {
    let n = h * w;
    let mut z_vals: Vec<(u32, usize)> = Vec::with_capacity(n);
    for y in 0..h {
        for x in 0..w {
            let z = spread_bits(x as u32) | (spread_bits(y as u32) << 1);
            z_vals.push((z, y * w + x));
        }
    }
    z_vals.sort_unstable_by_key(|&(z, _)| z);
    z_vals.iter().map(|&(_, idx)| idx).collect()
}

/// Inverse of Morton order: for each raster position, gives the Morton position.
fn morton_inverse(h: usize, w: usize) -> Vec<usize> {
    let order = morton_order(h, w);
    let n = h * w;
    let mut inv = vec![0usize; n];
    for (morton_pos, &raster_pos) in order.iter().enumerate() {
        inv[raster_pos] = morton_pos;
    }
    inv
}

/// Auto wavelet levels based on image dimensions.
/// 2 levels is the sweet spot for most photographic images.
/// 3 levels only for very large images (>= 4096px min dimension).
pub fn auto_wv_levels(h: usize, w: usize) -> usize {
    let min_dim = h.min(w);
    if min_dim < 256 { 1 }
    else if min_dim < 4096 { 2 }
    else { 3 }
}

/// v10: deeper wavelet decomposition for primitives-first pipeline.
pub fn auto_wv_levels_v10(h: usize, w: usize) -> usize {
    let min_dim = h.min(w);
    if min_dim < 64 { 1 }
    else if min_dim < 256 { 2 }
    else if min_dim < 1024 { 3 }
    else if min_dim < 4096 { 4 }
    else { 5 }
}

/// Zigzag diagonal scan order for an (h, w) grid.
/// Returns a permutation vector: zigzag[k] = raster index of the k-th element.
pub fn zigzag_order(h: usize, w: usize) -> Vec<usize> {
    let n = h * w;
    let mut indices = Vec::with_capacity(n);

    for s in 0..(h + w - 1) {
        if s % 2 == 0 {
            // Up-right
            let i_start = s.min(h - 1);
            let i_end = if s >= w { s - w + 1 } else { 0 };
            let mut i = i_start as i64;
            while i >= i_end as i64 {
                let j = s - i as usize;
                indices.push(i as usize * w + j);
                i -= 1;
            }
        } else {
            // Down-left
            let i_start = if s >= w { s - w + 1 } else { 0 };
            let i_end = s.min(h - 1);
            for i in i_start..=i_end {
                let j = s - i;
                indices.push(i * w + j);
            }
        }
    }

    indices
}

/// Inverse of zigzag: inv[raster_idx] = zigzag_position.
pub fn zigzag_inverse(h: usize, w: usize) -> Vec<usize> {
    let order = zigzag_order(h, w);
    let n = h * w;
    let mut inv = vec![0usize; n];
    for (k, &raster_idx) in order.iter().enumerate() {
        inv[raster_idx] = k;
    }
    inv
}





/// Simple bilinear upsample for inter-scale prediction.
pub fn upsample_band(band: &Array2<f64>, target_h: usize, target_w: usize) -> Array2<f64> {
    let sh = band.nrows();
    let sw = band.ncols();
    let mut result = Array2::<f64>::zeros((target_h, target_w));

    let scale_y = if target_h > 1 { (sh as f64 - 1.0) / (target_h as f64 - 1.0) } else { 0.0 };
    let scale_x = if target_w > 1 { (sw as f64 - 1.0) / (target_w as f64 - 1.0) } else { 0.0 };

    for y in 0..target_h {
        let sy = y as f64 * scale_y;
        let iy0 = (sy as usize).min(sh.saturating_sub(1));
        let iy1 = (iy0 + 1).min(sh - 1);
        let dy = sy - iy0 as f64;

        for x in 0..target_w {
            let sx = x as f64 * scale_x;
            let ix0 = (sx as usize).min(sw.saturating_sub(1));
            let ix1 = (ix0 + 1).min(sw - 1);
            let dx = sx - ix0 as f64;

            let v = band[[iy0, ix0]] * (1.0 - dy) * (1.0 - dx)
                  + band[[iy0, ix1]] * (1.0 - dy) * dx
                  + band[[iy1, ix0]] * dy * (1.0 - dx)
                  + band[[iy1, ix1]] * dy * dx;
            result[[y, x]] = v;
        }
    }

    result
}

/// Build the gas/solid dead zone map from the parent sigmap.
/// Gas (parent zero) -> DZ=dz_gas, Solid (parent non-zero) -> DZ=dz_solid.
/// The parent is upsampled 2x nearest-neighbor to match child size.
pub fn build_gas_dz_map(
    parent_band: &Array2<f64>,
    child_h: usize, child_w: usize,
    dz_solid: f64, dz_gas: f64,
) -> Vec<f64> {
    let ph = parent_band.nrows();
    let pw = parent_band.ncols();
    let mut dz_map = vec![dz_solid; child_h * child_w];

    for y in 0..child_h {
        // Nearest-neighbor upsample: child pixel (y,x) -> parent (y/2, x/2)
        let py = (y / 2).min(ph.saturating_sub(1));
        for x in 0..child_w {
            let px = (x / 2).min(pw.saturating_sub(1));
            if parent_band[[py, px]] == 0.0 {
                // Gas zone: parent was zero -> aggressive dead zone
                dz_map[y * child_w + x] = dz_gas;
            }
        }
    }

    dz_map
}

/// Compute auto detail_step from gradient complexity and quality parameter.
pub fn auto_detail_step(y_channel: &Array2<f64>, quality: u8) -> f64 {
    let h = y_channel.nrows();
    let w = y_channel.ncols();

    // Gradient energy
    let mut dy_energy = 0.0;
    let mut dx_energy = 0.0;
    let mut n_dy = 0usize;
    let mut n_dx = 0usize;

    for i in 0..h - 1 {
        for j in 0..w {
            let d = y_channel[[i + 1, j]] - y_channel[[i, j]];
            dy_energy += d * d;
            n_dy += 1;
        }
    }
    for i in 0..h {
        for j in 0..w - 1 {
            let d = y_channel[[i, j + 1]] - y_channel[[i, j]];
            dx_energy += d * d;
            n_dx += 1;
        }
    }

    let grad_energy = dy_energy / n_dy.max(1) as f64 + dx_energy / n_dx.max(1) as f64;
    let complexity = (grad_energy + 1.0).log2();
    let c_norm = ((complexity - 6.0) / 5.0).clamp(0.0, 1.0);

    let q_frac = (quality as f64 / 100.0).clamp(0.01, 1.0);
    let base_step = 2.0 + (1.0 - q_frac) * 28.0;
    let factor = 1.0 - 0.4 * c_norm;
    let step = base_step * factor;

    step.clamp(1.0, 48.0)
}

/// Auto detail_step calibrated for echolot (no spectral spin/anti-ring at decoder).
/// Base = echolot formula (101-q)/8, modulated by gradient complexity.
/// Simple images: +20% step (compression without visible loss).
/// Complex images: step nearly unchanged (preserve details).
pub fn auto_detail_step_echo(y_channel: &Array2<f64>, quality: u8) -> f64 {
    let h = y_channel.nrows();
    let w = y_channel.ncols();

    let mut dy_energy = 0.0;
    let mut dx_energy = 0.0;
    let mut n_dy = 0usize;
    let mut n_dx = 0usize;

    for i in 0..h - 1 {
        for j in 0..w {
            let d = y_channel[[i + 1, j]] - y_channel[[i, j]];
            dy_energy += d * d;
            n_dy += 1;
        }
    }
    for i in 0..h {
        for j in 0..w - 1 {
            let d = y_channel[[i, j + 1]] - y_channel[[i, j]];
            dx_energy += d * d;
            n_dx += 1;
        }
    }

    let grad_energy = dy_energy / n_dy.max(1) as f64 + dx_energy / n_dx.max(1) as f64;
    let complexity = (grad_energy + 1.0).log2();
    let c_norm = ((complexity - 6.0) / 5.0).clamp(0.0, 1.0);

    // Classic echolot base
    let echo_base = ((101.0 - quality as f64) / 8.0).max(1.0);
    // Soft modulation: simple images +20%, complex ~unchanged
    let adapt = 1.0 + 0.2 * (1.0 - c_norm);

    (echo_base * adapt).clamp(1.0, 30.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_even() {
        // Test with even dimensions
        let h = 64;
        let w = 64;
        let mut img = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                img[[i, j]] = (i * 7 + j * 3) as f64 % 256.0;
            }
        }

        let (ll, lh, hl, hh) = cdf97_forward_2d(&img);
        let recon = cdf97_inverse_2d(&ll, &lh, &hl, &hh, h, w);

        let max_diff = img.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-10, "max_diff = {}", max_diff);
    }

    #[test]
    fn test_roundtrip_odd() {
        // Test with odd dimensions
        let h = 63;
        let w = 65;
        let mut img = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                img[[i, j]] = (i * 11 + j * 5) as f64 % 256.0;
            }
        }

        let (ll, lh, hl, hh) = cdf97_forward_2d(&img);
        let recon = cdf97_inverse_2d(&ll, &lh, &hl, &hh, h, w);

        let max_diff = img.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-10, "max_diff = {}", max_diff);
    }

    #[test]
    fn test_multilevel_roundtrip() {
        let h = 128;
        let w = 128;
        let mut img = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                img[[i, j]] = ((i + j) as f64 * 0.7).sin() * 100.0 + 128.0;
            }
        }

        let (ll, subs, sizes) = wavelet_decompose(&img, 3);
        let recon = wavelet_recompose(&ll, &subs, &sizes);

        let max_diff = img.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-8, "max_diff = {}", max_diff);
    }

    #[test]
    fn test_weber_factor() {
        let f0 = super::weber_factor(0.0);
        assert!((f0 - 0.5).abs() < 1e-10, "dark: clamped to 0.5, got {}", f0);
        let f05 = super::weber_factor(0.5);
        let phi_inv2 = crate::golden::PHI_INV2;
        let expected = 0.5f64.powf(0.4) * (1.0 - phi_inv2) + phi_inv2;
        assert!((f05 - expected).abs() < 1e-6, "mid: got {}", f05);
        let f1 = super::weber_factor(1.0);
        assert!((f1 - 1.0).abs() < 1e-10, "bright: got {}", f1);
        // Monotone increasing
        for i in 0..100 {
            let a = i as f64 / 100.0;
            let b = (i + 1) as f64 / 100.0;
            assert!(super::weber_factor(b) >= super::weber_factor(a), "not monotone at {}", a);
        }
    }
}


