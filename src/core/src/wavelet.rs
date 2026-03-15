//! CDF 9/7 wavelet transform (same as JPEG 2000).
//!
//! Lifting scheme with 4 steps + normalization.
//! Used in v6 pipeline: multi-level decomposition on Y/Cb/Cr channels.

use ndarray::{Array2, s};
use rayon::prelude::*;

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
pub const DEAD_ZONE: f64 = 0.15;

/// Quantize a detail band with dead zone: sign(x) * max(0, floor(|x|/step + 0.5 - dz))
pub fn quantize_band(band: &Array2<f64>, step: f64) -> Array2<f64> {
    band.mapv(|v| {
        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
        let q = (v.abs() / step + 0.5 - DEAD_ZONE).floor();
        if q > 0.0 { sign * q } else { 0.0 }
    })
}

/// Dequantize: multiply by step
pub fn dequantize_band(qband: &Array2<f64>, step: f64) -> Array2<f64> {
    qband.mapv(|v| v * step)
}

/// Entropy threshold: zero out ±1 coefficients in low-density regions.
/// density_cutoff: fraction of non-zero neighbors below which ±1 are zeroed.
pub fn entropy_threshold(q: &Array2<f64>, block_size: usize, density_cutoff: f64) -> Array2<f64> {
    let h = q.nrows();
    let w = q.ncols();
    let mut result = q.clone();

    // Compute local density of non-zero coefficients
    let half = block_size / 2;
    for i in 0..h {
        for j in 0..w {
            if result[[i, j]].abs() != 1.0 {
                continue;
            }
            // Count non-zero in block
            let i0 = i.saturating_sub(half);
            let i1 = (i + half + 1).min(h);
            let j0 = j.saturating_sub(half);
            let j1 = (j + half + 1).min(w);
            let mut nz = 0usize;
            let mut total = 0usize;
            for ii in i0..i1 {
                for jj in j0..j1 {
                    total += 1;
                    if q[[ii, jj]] != 0.0 {
                        nz += 1;
                    }
                }
            }
            let density = nz as f64 / total as f64;
            if density < density_cutoff {
                result[[i, j]] = 0.0;
            }
        }
    }

    result
}

/// Encode a detail band: quantize + entropy threshold + significance map packing.
/// Returns (encoded_bytes, dequantized_band).
pub fn encode_detail_band(band: &Array2<f64>, step: f64) -> (Vec<u8>, Array2<f64>) {
    let q = quantize_band(band, step);
    let q = entropy_threshold(&q, 8, 0.10);

    let h = q.nrows();
    let w = q.ncols();
    let n = h * w;
    let flat: Vec<i16> = q.iter().map(|&v| v as i16).collect();

    let mut stream = Vec::new();

    // Significance map: pack as bits (8 coeffs per byte, little-endian bit order)
    let n_sig_bytes = (n + 7) / 8;
    let mut sig_packed = vec![0u8; n_sig_bytes];
    for i in 0..n {
        if flat[i] != 0 {
            sig_packed[i / 8] |= 1 << (i % 8);
        }
    }
    stream.extend_from_slice(&sig_packed);

    // Non-zero values only
    let nz_vals: Vec<i16> = flat.iter().copied().filter(|&v| v != 0).collect();
    let n_nz = nz_vals.len() as u32;
    stream.extend_from_slice(&n_nz.to_le_bytes());

    if !nz_vals.is_empty() {
        let max_abs = nz_vals.iter().map(|v| v.abs()).max().unwrap_or(0);
        if max_abs <= 127 {
            stream.push(0x08);
            for &v in &nz_vals {
                stream.push(v as i8 as u8);
            }
        } else {
            stream.push(0x10);
            for &v in &nz_vals {
                stream.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    let deq = dequantize_band(&q, step);
    (stream, deq)
}

/// Decode a detail band from byte stream.
/// Returns (band as f64, new position).
pub fn decode_detail_band(stream: &[u8], pos: usize, h: usize, w: usize) -> (Array2<f64>, usize) {
    let n = h * w;
    let n_sig_bytes = (n + 7) / 8;
    let mut p = pos;

    // Unpack significance map
    let sig_packed = &stream[p..p + n_sig_bytes];
    p += n_sig_bytes;

    let mut sig = vec![false; n];
    for i in 0..n {
        sig[i] = (sig_packed[i / 8] >> (i % 8)) & 1 != 0;
    }

    // Number of non-zero values
    let n_nz = u32::from_le_bytes([stream[p], stream[p + 1], stream[p + 2], stream[p + 3]]) as usize;
    p += 4;

    let mut flat = vec![0.0f64; n];

    if n_nz > 0 {
        let marker = stream[p];
        p += 1;

        let mut nz_idx = 0;
        if marker == 0x08 {
            // int8
            for i in 0..n {
                if sig[i] {
                    flat[i] = stream[p + nz_idx] as i8 as f64;
                    nz_idx += 1;
                }
            }
            p += n_nz;
        } else {
            // int16
            for i in 0..n {
                if sig[i] {
                    let v = i16::from_le_bytes([stream[p + nz_idx * 2], stream[p + nz_idx * 2 + 1]]);
                    flat[i] = v as f64;
                    nz_idx += 1;
                }
            }
            p += n_nz * 2;
        }
    }

    let band = Array2::from_shape_vec((h, w), flat).unwrap();
    (band, p)
}

/// Chroma detail factor (v6+: 2.0x coarser than Y)
pub const CHROMA_DETAIL_FACTOR: f64 = 1.5;

/// Perceptual band weights (QGA-optimized)
pub const PERCEPTUAL_BAND_WEIGHTS: [f64; 3] = [1.3, 0.65, 1.3]; // LH, HL, HH

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

/// Smooth block mask for adaptive quantization.
/// Returns (bh, bw, mask) where mask[b] = true if the block is smooth.
pub fn block_smooth_mask(band: &Array2<f64>, bs: usize) -> (usize, usize, Vec<bool>) {
    let h = band.nrows();
    let w = band.ncols();
    let bh = (h + bs - 1) / bs;
    let bw = (w + bs - 1) / bs;

    // Per-block energy
    let mut energies = Vec::with_capacity(bh * bw);
    for bi in 0..bh {
        for bj in 0..bw {
            let i0 = bi * bs;
            let i1 = (i0 + bs).min(h);
            let j0 = bj * bs;
            let j1 = (j0 + bs).min(w);
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for i in i0..i1 {
                for j in j0..j1 {
                    sum += band[[i, j]].abs();
                    count += 1;
                }
            }
            energies.push(sum / count.max(1) as f64);
        }
    }

    // Threshold = median / 2
    let mut sorted_e = energies.clone();
    sorted_e.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_e[sorted_e.len() / 2];
    let threshold = median / 2.0;

    let mask: Vec<bool> = energies.iter().map(|&e| e < threshold).collect();
    (bh, bw, mask)
}

/// Build a per-pixel step map from the smooth block mask.
/// smooth blocks: base_step * 1.5, normal blocks: base_step.
pub fn build_step_map(h: usize, w: usize, bh: usize, bw: usize,
                       mask: &[bool], base_step: f64, bs: usize) -> Vec<f64> {
    let mut step_map = vec![base_step; h * w];
    for bi in 0..bh {
        for bj in 0..bw {
            if mask[bi * bw + bj] {
                let i0 = bi * bs;
                let i1 = (i0 + bs).min(h);
                let j0 = bj * bs;
                let j1 = (j0 + bs).min(w);
                for i in i0..i1 {
                    for j in j0..j1 {
                        step_map[i * w + j] = base_step * 1.5;
                    }
                }
            }
        }
    }
    step_map
}

/// Encode a detail band with zigzag + adaptive quantization.
/// Returns (encoded_bytes, dequantized_band).
pub fn encode_detail_band_v2(
    band: &Array2<f64>, base_step: f64, flags: u8,
) -> (Vec<u8>, Array2<f64>) {
    encode_detail_band_v2_with_dz(band, base_step, flags, None)
}

/// Encode with gas-first reordering (Minesweeper).
/// If `gas_mask` is provided (raster order, true=gas/parent_zero),
/// gas pixels are grouped at the head of the Morton stream,
/// followed by solid pixels. LZMA compresses long zero sequences better.
/// Quantization identical to v2 -> zero quality loss.
pub fn encode_detail_band_v2_gas(
    band: &Array2<f64>, base_step: f64, flags: u8,
    gas_mask: Option<&[bool]>,
) -> (Vec<u8>, Array2<f64>) {
    // Without gas mask, encode normally
    if gas_mask.is_none() || flags & FLAG_MORTON_2BIT == 0 {
        return encode_detail_band_v2(band, base_step, flags);
    }
    let gas = gas_mask.unwrap();

    let h = band.nrows();
    let w = band.ncols();
    let n = h * w;
    let bs = 16usize;

    // Quantization (identical to v2 -- no DZ modification)
    let (bh, bw, smooth_mask) = if flags & FLAG_ADAPTIVE_QUANT != 0 {
        block_smooth_mask(band, bs)
    } else {
        (0, 0, vec![])
    };

    let q: Array2<f64>;
    let deq: Array2<f64>;

    if flags & FLAG_ADAPTIVE_QUANT != 0 {
        let step_map = build_step_map(h, w, bh, bw, &smooth_mask, base_step, bs);
        let mut q_arr = Array2::<f64>::zeros((h, w));
        let mut deq_arr = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                let s = step_map[i * w + j];
                let v = band[[i, j]];
                let sign = if v >= 0.0 { 1.0 } else { -1.0 };
                let qv_raw = (v.abs() / s + 0.5 - DEAD_ZONE).floor();
                let qv = if qv_raw > 0.0 { sign * qv_raw } else { 0.0 };
                q_arr[[i, j]] = qv;
                deq_arr[[i, j]] = qv * s;
            }
        }
        q = entropy_threshold(&q_arr, 8, 0.10);
        let mut d2 = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                d2[[i, j]] = q[[i, j]] * step_map[i * w + j];
            }
        }
        deq = d2;
    } else {
        let q0 = quantize_band(band, base_step);
        q = entropy_threshold(&q0, 8, 0.10);
        deq = dequantize_band(&q, base_step);
    };

    let flat: Vec<i16> = q.iter().map(|&v| v as i16).collect();

    // Morton order → puis gas-first reordering
    let morton = morton_order(h, w);
    // Separer les indices Morton en gas et solid
    let mut gas_indices = Vec::new();
    let mut solid_indices = Vec::new();
    for &mi in &morton {
        if gas[mi] {
            gas_indices.push(mi);
        } else {
            solid_indices.push(mi);
        }
    }
    let n_gas = gas_indices.len() as u32;

    // Build flat_ordered: gas first, solid second
    let mut flat_ordered = Vec::with_capacity(n);
    for &idx in &gas_indices {
        flat_ordered.push(flat[idx]);
    }
    for &idx in &solid_indices {
        flat_ordered.push(flat[idx]);
    }

    let mut stream = Vec::new();

    // Gas-split header: store n_gas so the decoder knows the cutoff point
    stream.extend_from_slice(&n_gas.to_le_bytes());

    // Adaptive quant mask header
    if flags & FLAG_ADAPTIVE_QUANT != 0 {
        stream.extend_from_slice(&(bh as u16).to_le_bytes());
        stream.extend_from_slice(&(bw as u16).to_le_bytes());
        let n_mask_bytes = (bh * bw + 7) / 8;
        let mut mask_packed = vec![0u8; n_mask_bytes];
        for i in 0..(bh * bw) {
            if smooth_mask[i] {
                mask_packed[i / 8] |= 1 << (i % 8);
            }
        }
        stream.extend_from_slice(&mask_packed);
    }

    // 2-bit sigmap encoding (gas-first order)
    let n_2bit_bytes = (n + 3) / 4;
    let mut packed_2bit = vec![0u8; n_2bit_bytes];
    for i in 0..n {
        let code: u8 = match flat_ordered[i] {
            0 => 0,
            1 => 1,
            -1 => 2,
            _ => 3,
        };
        packed_2bit[i / 4] |= code << ((i % 4) * 2);
    }
    stream.extend_from_slice(&packed_2bit);

    // Remaining values: |val| >= 2 (gas-first order)
    let rest: Vec<i16> = flat_ordered.iter().copied()
        .filter(|&v| v != 0 && v != 1 && v != -1).collect();
    let n_rest = rest.len() as u32;
    stream.extend_from_slice(&n_rest.to_le_bytes());

    if !rest.is_empty() {
        let max_abs = rest.iter().map(|v| v.abs()).max().unwrap_or(0);
        if max_abs <= 127 {
            stream.push(0x08);
            for &v in &rest {
                stream.push(v as i8 as u8);
            }
        } else {
            stream.push(0x10);
            for &v in &rest {
                stream.extend_from_slice(&v.to_le_bytes());
            }
        }
    }

    (stream, deq)
}

/// Encode with spatially variable dead zone (gas/solid prediction).
/// If `dz_map` is provided, each pixel uses its own dead zone
/// instead of the DEAD_ZONE constant.
pub fn encode_detail_band_v2_with_dz(
    band: &Array2<f64>, base_step: f64, flags: u8,
    dz_map: Option<&[f64]>,
) -> (Vec<u8>, Array2<f64>) {
    let h = band.nrows();
    let w = band.ncols();
    let n = h * w;
    let bs = 16usize;

    // Adaptive quantization: smooth block mask
    let (bh, bw, smooth_mask) = if flags & FLAG_ADAPTIVE_QUANT != 0 {
        block_smooth_mask(band, bs)
    } else {
        (0, 0, vec![])
    };

    // Quantization (uniform or adaptive)
    let q: Array2<f64>;
    let deq: Array2<f64>;

    if flags & FLAG_ADAPTIVE_QUANT != 0 {
        let step_map = build_step_map(h, w, bh, bw, &smooth_mask, base_step, bs);
        let mut q_arr = Array2::<f64>::zeros((h, w));
        let mut deq_arr = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                let s = step_map[i * w + j];
                let v = band[[i, j]];
                let dz = if let Some(dm) = dz_map { dm[i * w + j] } else { DEAD_ZONE };
                let sign = if v >= 0.0 { 1.0 } else { -1.0 };
                let qv_raw = (v.abs() / s + 0.5 - dz).floor();
                let qv = if qv_raw > 0.0 { sign * qv_raw } else { 0.0 };
                q_arr[[i, j]] = qv;
                deq_arr[[i, j]] = qv * s;
            }
        }
        // Entropy threshold
        q = entropy_threshold(&q_arr, 8, 0.10);
        // Recompute deq after threshold
        let mut d2 = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                d2[[i, j]] = q[[i, j]] * step_map[i * w + j];
            }
        }
        deq = d2;
    } else if let Some(dm) = dz_map {
        // Spatially variable dead zone without adaptive quant
        let mut q_arr = Array2::<f64>::zeros((h, w));
        let mut deq_arr = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                let v = band[[i, j]];
                let dz = dm[i * w + j];
                let sign = if v >= 0.0 { 1.0 } else { -1.0 };
                let qv_raw = (v.abs() / base_step + 0.5 - dz).floor();
                let qv = if qv_raw > 0.0 { sign * qv_raw } else { 0.0 };
                q_arr[[i, j]] = qv;
                deq_arr[[i, j]] = qv * base_step;
            }
        }
        q = entropy_threshold(&q_arr, 8, 0.10);
        deq = dequantize_band(&q, base_step);
    } else {
        let q0 = quantize_band(band, base_step);
        q = entropy_threshold(&q0, 8, 0.10);
        deq = dequantize_band(&q, base_step);
    };

    let flat: Vec<i16> = q.iter().map(|&v| v as i16).collect();

    // Scan reorder: Morton (priority) or Zigzag
    let flat_ordered: Vec<i16> = if flags & FLAG_MORTON_2BIT != 0 {
        let order = morton_order(h, w);
        order.iter().map(|&idx| flat[idx]).collect()
    } else if flags & FLAG_ZIGZAG != 0 {
        let zz = zigzag_order(h, w);
        zz.iter().map(|&idx| flat[idx]).collect()
    } else {
        flat
    };

    let mut stream = Vec::new();

    // Adaptive quant mask header (u16 LE for Python compatibility)
    if flags & FLAG_ADAPTIVE_QUANT != 0 {
        stream.extend_from_slice(&(bh as u16).to_le_bytes());
        stream.extend_from_slice(&(bw as u16).to_le_bytes());
        let n_mask_bytes = (bh * bw + 7) / 8;
        let mut mask_packed = vec![0u8; n_mask_bytes];
        for i in 0..(bh * bw) {
            if smooth_mask[i] {
                mask_packed[i / 8] |= 1 << (i % 8);
            }
        }
        stream.extend_from_slice(&mask_packed);
    }

    if flags & FLAG_MORTON_2BIT != 0 {
        // Encodage 2-bit sigmap : 0=zero, 1=+1, 2=-1, 3=other
        let n_2bit_bytes = (n + 3) / 4;
        let mut packed_2bit = vec![0u8; n_2bit_bytes];
        for i in 0..n {
            let code: u8 = match flat_ordered[i] {
                0 => 0,
                1 => 1,
                -1 => 2,
                _ => 3,
            };
            packed_2bit[i / 4] |= code << ((i % 4) * 2);
        }
        stream.extend_from_slice(&packed_2bit);

        // Remaining values: |val| >= 2
        let rest: Vec<i16> = flat_ordered.iter().copied()
            .filter(|&v| v != 0 && v != 1 && v != -1).collect();
        let n_rest = rest.len() as u32;
        stream.extend_from_slice(&n_rest.to_le_bytes());

        if !rest.is_empty() {
            let max_abs = rest.iter().map(|v| v.abs()).max().unwrap_or(0);
            if max_abs <= 127 {
                stream.push(0x08);
                for &v in &rest {
                    stream.push(v as i8 as u8);
                }
            } else {
                stream.push(0x10);
                for &v in &rest {
                    stream.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    } else {
        // Format classique: sigmap + NZ values
        let n_sig_bytes = (n + 7) / 8;
        let mut sig_packed = vec![0u8; n_sig_bytes];
        for i in 0..n {
            if flat_ordered[i] != 0 {
                sig_packed[i / 8] |= 1 << (i % 8);
            }
        }
        stream.extend_from_slice(&sig_packed);

        let nz_vals: Vec<i16> = flat_ordered.iter().copied().filter(|&v| v != 0).collect();
        let n_nz = nz_vals.len() as u32;
        stream.extend_from_slice(&n_nz.to_le_bytes());

        if !nz_vals.is_empty() {
            let max_abs = nz_vals.iter().map(|v| v.abs()).max().unwrap_or(0);
            if max_abs <= 127 {
                stream.push(0x08);
                for &v in &nz_vals {
                    stream.push(v as i8 as u8);
                }
            } else {
                stream.push(0x10);
                for &v in &nz_vals {
                    stream.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }

    (stream, deq)
}

/// Decode a v2 detail band (zigzag + adaptive quant).
/// Returns (dequantized_band, new_position).
pub fn decode_detail_band_v2(
    stream: &[u8], pos: usize, h: usize, w: usize,
    base_step: f64, flags: u8,
) -> (Array2<f64>, usize) {
    let n = h * w;
    let bs = 16usize;
    let mut p = pos;

    // Adaptive quant mask
    let smooth_mask: Vec<bool>;
    let bh: usize;
    let bw: usize;

    if flags & FLAG_ADAPTIVE_QUANT != 0 {
        bh = u16::from_le_bytes([stream[p], stream[p + 1]]) as usize; p += 2;
        bw = u16::from_le_bytes([stream[p], stream[p + 1]]) as usize; p += 2;
        let n_mask_bytes = (bh * bw + 7) / 8;
        let mask_packed = &stream[p..p + n_mask_bytes];
        p += n_mask_bytes;
        smooth_mask = (0..(bh * bw)).map(|i| {
            (mask_packed[i / 8] >> (i % 8)) & 1 != 0
        }).collect();
    } else {
        bh = 0; bw = 0;
        smooth_mask = vec![];
    }

    let flat: Vec<f64> = if flags & FLAG_MORTON_2BIT != 0 {
        // 2-bit sigmap decoding: 0=zero, 1=+1, 2=-1, 3=read rest
        let n_2bit_bytes = (n + 3) / 4;
        let packed_2bit = &stream[p..p + n_2bit_bytes];
        p += n_2bit_bytes;

        let mut flat_morton = vec![0.0f64; n];
        for i in 0..n {
            let code = (packed_2bit[i / 4] >> ((i % 4) * 2)) & 0x03;
            match code {
                1 => flat_morton[i] = 1.0,
                2 => flat_morton[i] = -1.0,
                _ => {} // 0 (zero) et 3 (rest) traites apres
            }
        }

        // Remaining values (|val| >= 2)
        let n_rest = u32::from_le_bytes([stream[p], stream[p+1], stream[p+2], stream[p+3]]) as usize;
        p += 4;

        if n_rest > 0 {
            let marker = stream[p]; p += 1;
            let mut rest_idx = 0;
            for i in 0..n {
                let code = (packed_2bit[i / 4] >> ((i % 4) * 2)) & 0x03;
                if code == 3 {
                    if marker == 0x08 {
                        flat_morton[i] = stream[p + rest_idx] as i8 as f64;
                        rest_idx += 1;
                    } else {
                        let v = i16::from_le_bytes([stream[p + rest_idx*2], stream[p + rest_idx*2 + 1]]);
                        flat_morton[i] = v as f64;
                        rest_idx += 1;
                    }
                }
            }
            if marker == 0x08 { p += n_rest; } else { p += n_rest * 2; }
        }

        // Inverse Morton order
        let inv = morton_inverse(h, w);
        let mut flat_raster = vec![0.0f64; n];
        for i in 0..n {
            flat_raster[i] = flat_morton[inv[i]];
        }
        flat_raster
    } else {
        // Format classique: sigmap + NZ values
        let n_sig_bytes = (n + 7) / 8;
        let sig_packed = &stream[p..p + n_sig_bytes];
        p += n_sig_bytes;

        let mut sig = vec![false; n];
        for i in 0..n {
            sig[i] = (sig_packed[i / 8] >> (i % 8)) & 1 != 0;
        }

        let n_nz = u32::from_le_bytes([stream[p], stream[p+1], stream[p+2], stream[p+3]]) as usize;
        p += 4;

        let mut flat_ordered = vec![0.0f64; n];

        if n_nz > 0 {
            let marker = stream[p]; p += 1;
            let mut nz_idx = 0;
            if marker == 0x08 {
                for i in 0..n {
                    if sig[i] {
                        flat_ordered[i] = stream[p + nz_idx] as i8 as f64;
                        nz_idx += 1;
                    }
                }
                p += n_nz;
            } else {
                for i in 0..n {
                    if sig[i] {
                        let v = i16::from_le_bytes([stream[p + nz_idx*2], stream[p + nz_idx*2 + 1]]);
                        flat_ordered[i] = v as f64;
                        nz_idx += 1;
                    }
                }
                p += n_nz * 2;
            }
        }

        // Inverse zigzag
        if flags & FLAG_ZIGZAG != 0 {
            let zz = zigzag_order(h, w);
            let mut out = vec![0.0f64; n];
            for k in 0..n {
                out[zz[k]] = flat_ordered[k];
            }
            out
        } else {
            flat_ordered
        }
    };

    // Dequantize
    let mut band = if flags & FLAG_ADAPTIVE_QUANT != 0 {
        let step_map = build_step_map(h, w, bh, bw, &smooth_mask, base_step, bs);
        let mut arr = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                arr[[i, j]] = flat[i * w + j] * step_map[i * w + j];
            }
        }
        arr
    } else {
        let arr_q = Array2::from_shape_vec((h, w), flat).unwrap();
        dequantize_band(&arr_q, base_step)
    };

    // Spectral Spin : raffinement median decoder-only
    {
        let band_slice = band.as_slice_mut().unwrap();
        spectral_spin(band_slice, h, w, base_step);
    }

    (band, p)
}

/// Decode a band encoded with gas-first reordering.
/// `gas_mask` (raster order, true=gas) allows reconstructing the original order.
/// If gas_mask is None, delegates to the standard decoder.
pub fn decode_detail_band_v2_gas(
    stream: &[u8], pos: usize, h: usize, w: usize,
    base_step: f64, flags: u8,
    gas_mask: Option<&[bool]>,
) -> (Array2<f64>, usize) {
    // Without gas mask or without Morton, standard decoding
    if gas_mask.is_none() || flags & FLAG_MORTON_2BIT == 0 {
        return decode_detail_band_v2(stream, pos, h, w, base_step, flags);
    }
    let gas = gas_mask.unwrap();

    let n = h * w;
    let bs = 16usize;
    let mut p = pos;

    // Read n_gas header
    let n_gas = u32::from_le_bytes([stream[p], stream[p+1], stream[p+2], stream[p+3]]) as usize;
    p += 4;

    // Adaptive quant mask
    let smooth_mask: Vec<bool>;
    let bh: usize;
    let bw: usize;
    if flags & FLAG_ADAPTIVE_QUANT != 0 {
        bh = u16::from_le_bytes([stream[p], stream[p + 1]]) as usize; p += 2;
        bw = u16::from_le_bytes([stream[p], stream[p + 1]]) as usize; p += 2;
        let n_mask_bytes = (bh * bw + 7) / 8;
        let mask_packed = &stream[p..p + n_mask_bytes];
        p += n_mask_bytes;
        smooth_mask = (0..(bh * bw)).map(|i| {
            (mask_packed[i / 8] >> (i % 8)) & 1 != 0
        }).collect();
    } else {
        bh = 0; bw = 0;
        smooth_mask = vec![];
    }

    // Decode the 2-bit sigmap (in gas-first order)
    let n_2bit_bytes = (n + 3) / 4;
    let packed_2bit = &stream[p..p + n_2bit_bytes];
    p += n_2bit_bytes;

    let mut flat_gasfirst = vec![0.0f64; n];
    for i in 0..n {
        let code = (packed_2bit[i / 4] >> ((i % 4) * 2)) & 0x03;
        match code {
            1 => flat_gasfirst[i] = 1.0,
            2 => flat_gasfirst[i] = -1.0,
            _ => {}
        }
    }

    // Remaining values
    let n_rest = u32::from_le_bytes([stream[p], stream[p+1], stream[p+2], stream[p+3]]) as usize;
    p += 4;

    if n_rest > 0 {
        let marker = stream[p]; p += 1;
        let mut rest_idx = 0;
        for i in 0..n {
            let code = (packed_2bit[i / 4] >> ((i % 4) * 2)) & 0x03;
            if code == 3 {
                if marker == 0x08 {
                    flat_gasfirst[i] = stream[p + rest_idx] as i8 as f64;
                    rest_idx += 1;
                } else {
                    let v = i16::from_le_bytes([stream[p + rest_idx*2], stream[p + rest_idx*2 + 1]]);
                    flat_gasfirst[i] = v as f64;
                    rest_idx += 1;
                }
            }
        }
        if marker == 0x08 { p += n_rest; } else { p += n_rest * 2; }
    }

    // Reconstruct original Morton order from gas-first order
    // Same logic as the encoder: compute gas_indices and solid_indices in Morton order
    let morton = morton_order(h, w);
    let mut gas_indices = Vec::with_capacity(n_gas);
    let mut solid_indices = Vec::with_capacity(n - n_gas);
    for &mi in &morton {
        if gas[mi] {
            gas_indices.push(mi);
        } else {
            solid_indices.push(mi);
        }
    }

    // flat_gasfirst[0..n_gas] = gas values, flat_gasfirst[n_gas..] = solid values
    // Place in raster order
    let mut flat_raster = vec![0.0f64; n];
    for (gi, &raster_idx) in gas_indices.iter().enumerate() {
        flat_raster[raster_idx] = flat_gasfirst[gi];
    }
    for (si, &raster_idx) in solid_indices.iter().enumerate() {
        flat_raster[raster_idx] = flat_gasfirst[n_gas + si];
    }

    // Dequantize
    let mut band = if flags & FLAG_ADAPTIVE_QUANT != 0 {
        let step_map = build_step_map(h, w, bh, bw, &smooth_mask, base_step, bs);
        let mut arr = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                arr[[i, j]] = flat_raster[i * w + j] * step_map[i * w + j];
            }
        }
        arr
    } else {
        let arr_q = Array2::from_shape_vec((h, w), flat_raster).unwrap();
        dequantize_band(&arr_q, base_step)
    };

    // Spectral Spin
    {
        let band_slice = band.as_slice_mut().unwrap();
        spectral_spin(band_slice, h, w, base_step);
    }

    (band, p)
}

// ======================================================================
// Spectral Spin: median refinement of wavelet coefficients
// ======================================================================

/// Index with boundary reflection (scipy 'reflect' mode).
#[inline]
fn reflect(i: i32, n: i32) -> usize {
    if i < 0 {
        (-i).min(n - 1) as usize
    } else if i >= n {
        (2 * (n - 1) - i).max(0) as usize
    } else {
        i as usize
    }
}

/// 5x5 median filter with boundary reflection, parallelized per row.
fn median_filter_5x5(plane: &[f64], h: usize, w: usize) -> Vec<f64> {
    let hi = h as i32;
    let wi = w as i32;
    let mut result = vec![0.0f64; h * w];

    result.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let iy = y as i32;
        let mut buf = [0.0f64; 25];
        for x in 0..w {
            let ix = x as i32;
            let mut k = 0;
            for dy in -2..=2i32 {
                let ry = reflect(iy + dy, hi);
                for dx in -2..=2i32 {
                    let rx = reflect(ix + dx, wi);
                    buf[k] = plane[ry * w + rx];
                    k += 1;
                }
            }
            buf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            row[x] = buf[12]; // median de 25 elements
        }
    });

    result
}

/// Spectral spin: local median regularization of wavelet coefficients.
/// Analogous to Fibonacci spin for LL flats:
/// the 5x5 median exploits spatial correlation to predict the position
/// of each coefficient within its quantization bin.
/// Formula: band * 0.92 + median_5x5 * 0.08
pub fn spectral_spin(band: &mut [f64], h: usize, w: usize, _step: f64) {
    let med = median_filter_5x5(band, h, w);
    let n = h * w;
    for i in 0..n {
        band[i] = band[i] * 0.92 + med[i] * 0.08;
    }
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
}


// ======================================================================
// ADN coding: enumerative combinatorial by blocks (AUREA v5)
// ======================================================================

pub const FLAG_ADN: u8 = 0x20;
const ADN_BLOCK_SIZE: usize = 64;
const ADN_FORMAT_MARKER: u8 = 0xAD;
const ADN_EMPTY_THRESHOLD: f64 = 0.30;

use std::sync::LazyLock;
use crate::zeckendorf::{BitWriter, BitReader};

/// Pascal's triangle C(n,k) for n=0..64, k=0..64.
static PASCAL: LazyLock<[[u64; 65]; 65]> = LazyLock::new(|| {
    let mut p = [[0u64; 65]; 65];
    for n in 0..65 {
        p[n][0] = 1;
        for k in 1..=n {
            p[n][k] = p[n - 1][k - 1].saturating_add(p[n - 1][k]);
        }
    }
    p
});

/// Number of bits for combinatorial rank C(b, k).
#[inline]
fn rank_bits(k: usize, b: usize) -> usize {
    if k == 0 || k >= b { return 0; }
    let total = PASCAL[b][k];
    if total <= 1 { return 0; }
    64 - (total - 1).leading_zeros() as usize
}

/// Combinadic rank (sorted ascending positions).
fn comb_rank(positions: &[usize], _b: usize) -> u64 {
    let p = &*PASCAL;
    let mut rank: u64 = 0;
    for (i, &pos) in positions.iter().enumerate() {
        if pos >= i + 1 {
            rank += p[pos][i + 1];
        }
    }
    rank
}

/// Inverse of combinadic rank.
fn comb_unrank(mut rank: u64, k: usize, b: usize) -> Vec<usize> {
    let p = &*PASCAL;
    let mut positions = Vec::with_capacity(k);
    for i in (0..k).rev() {
        let mut c = i;
        while c + 1 < b && p[c + 1][i + 1] <= rank {
            c += 1;
        }
        positions.push(c);
        rank -= p[c][i + 1];
    }
    positions.reverse();
    positions
}

/// Quantize a band and reorder in Morton (shared code).
fn quantize_and_reorder(
    band: &Array2<f64>, base_step: f64, flags: u8,
) -> (Vec<i16>, Array2<f64>, usize, usize, Vec<bool>) {
    let h = band.nrows();
    let w = band.ncols();
    let bs = 16usize;

    let (bh, bw, smooth_mask) = if flags & FLAG_ADAPTIVE_QUANT != 0 {
        block_smooth_mask(band, bs)
    } else {
        (0, 0, vec![])
    };

    if flags & FLAG_ADAPTIVE_QUANT != 0 {
        let step_map = build_step_map(h, w, bh, bw, &smooth_mask, base_step, bs);
        let mut q_arr = Array2::<f64>::zeros((h, w));
        let mut deq_arr = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                let s = step_map[i * w + j];
                let v = band[[i, j]];
                let sign = if v >= 0.0 { 1.0 } else { -1.0 };
                let qv_raw = (v.abs() / s + 0.5 - DEAD_ZONE).floor();
                let qv = if qv_raw > 0.0 { sign * qv_raw } else { 0.0 };
                q_arr[[i, j]] = qv;
                deq_arr[[i, j]] = qv * s;
            }
        }
        let q2 = entropy_threshold(&q_arr, 8, 0.10);
        let mut d2 = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                d2[[i, j]] = q2[[i, j]] * step_map[i * w + j];
            }
        }
        let flat: Vec<i16> = q2.iter().map(|&v| v as i16).collect();
        let order = morton_order(h, w);
        let flat_ordered = order.iter().map(|&idx| flat[idx]).collect();
        (flat_ordered, d2, bh, bw, smooth_mask)
    } else {
        let q0 = quantize_band(band, base_step);
        let q2 = entropy_threshold(&q0, 8, 0.10);
        let deq2 = dequantize_band(&q2, base_step);
        let flat: Vec<i16> = q2.iter().map(|&v| v as i16).collect();
        let order = morton_order(h, w);
        let flat_ordered = order.iter().map(|&idx| flat[idx]).collect();
        (flat_ordered, deq2, bh, bw, smooth_mask)
    }
}

/// Encode a detail band with ADN enumerative coding.
/// Adaptive: if > 30% of blocks are empty, uses ADN; otherwise fallback to v2.
pub fn encode_detail_band_adn(
    band: &Array2<f64>, base_step: f64, flags: u8, _is_chroma: bool,
) -> (Vec<u8>, Array2<f64>) {
    let h = band.nrows();
    let w = band.ncols();
    let n = h * w;

    let (flat_ordered, deq, bh, bw, smooth_mask) =
        quantize_and_reorder(band, base_step, flags);

    // Count empty blocks for adaptive decision
    let n_blocks = (n + ADN_BLOCK_SIZE - 1) / ADN_BLOCK_SIZE;
    let n_empty: usize = (0..n_blocks).filter(|&b| {
        let start = b * ADN_BLOCK_SIZE;
        let end = (start + ADN_BLOCK_SIZE).min(n);
        flat_ordered[start..end].iter().all(|&v| v == 0)
    }).count();

    if (n_empty as f64) < ADN_EMPTY_THRESHOLD * n_blocks as f64 {
        // Dense band -> fallback sigmap v2, prefixed by 0x00 marker
        let (mut v2_data, v2_deq) = encode_detail_band_v2(band, base_step, flags & !FLAG_ADN);
        let mut prefixed = Vec::with_capacity(1 + v2_data.len());
        prefixed.push(0x00); // marqueur legacy
        prefixed.append(&mut v2_data);
        return (prefixed, v2_deq);
    }

    // --- ADN encoding ---
    let mut stream = Vec::new();
    stream.push(ADN_FORMAT_MARKER);

    // Header adaptive quant
    if flags & FLAG_ADAPTIVE_QUANT != 0 {
        stream.extend_from_slice(&(bh as u16).to_le_bytes());
        stream.extend_from_slice(&(bw as u16).to_le_bytes());
        let n_mask_bytes = (bh * bw + 7) / 8;
        let mut mask_packed = vec![0u8; n_mask_bytes];
        for i in 0..(bh * bw) {
            if smooth_mask[i] {
                mask_packed[i / 8] |= 1 << (i % 8);
            }
        }
        stream.extend_from_slice(&mask_packed);
    }

    stream.extend_from_slice(&(n_blocks as u16).to_le_bytes());

    // Bit-pack the blocks
    let mut bw_out = BitWriter::new();
    let mut rest_vals: Vec<i16> = Vec::new();

    for b in 0..n_blocks {
        let start = b * ADN_BLOCK_SIZE;
        let end = (start + ADN_BLOCK_SIZE).min(n);
        let block = &flat_ordered[start..end];
        let block_b = end - start;

        let nz_pos: Vec<usize> = block.iter().enumerate()
            .filter(|(_, v)| **v != 0)
            .map(|(i, _)| i)
            .collect();
        let k = nz_pos.len();

        if k == 0 {
            bw_out.write_bit(true); // vide
        } else {
            bw_out.write_bit(false);
            bw_out.write_bits(k as u64, 6);

            let nbits = rank_bits(k, block_b);
            if nbits > 0 {
                let rank = comb_rank(&nz_pos, block_b);
                bw_out.write_bits(rank, nbits);
            }

            for &pos in &nz_pos {
                let v = block[pos];
                let is_pm1 = v.abs() == 1;
                bw_out.write_bit(is_pm1);
                bw_out.write_bit(v < 0);
                if !is_pm1 {
                    rest_vals.push(v);
                }
            }
        }
    }

    let bit_data = bw_out.finish();
    stream.extend_from_slice(&bit_data);

    // Remaining values raw
    let n_rest = rest_vals.len() as u32;
    stream.extend_from_slice(&n_rest.to_le_bytes());
    if !rest_vals.is_empty() {
        let max_abs = rest_vals.iter().map(|v| v.abs()).max().unwrap_or(0);
        if max_abs <= 127 {
            stream.push(0x08);
            for &v in &rest_vals { stream.push(v as i8 as u8); }
        } else {
            stream.push(0x10);
            for &v in &rest_vals { stream.extend_from_slice(&v.to_le_bytes()); }
        }
    }

    (stream, deq)
}

/// Decode an ADN band (or legacy fallback).
pub fn decode_detail_band_adn(
    stream: &[u8], pos: usize, h: usize, w: usize,
    base_step: f64, flags: u8, _is_chroma: bool,
) -> (Array2<f64>, usize) {
    let n = h * w;
    let bs = 16usize;
    let mut p = pos;

    let marker = stream[p]; p += 1;
    if marker != ADN_FORMAT_MARKER {
        // Fallback v2
        return decode_detail_band_v2(stream, p, h, w, base_step, flags & !FLAG_ADN);
    }

    // Adaptive quant mask
    let smooth_mask: Vec<bool>;
    let aq_bh: usize;
    let aq_bw: usize;
    if flags & FLAG_ADAPTIVE_QUANT != 0 {
        aq_bh = u16::from_le_bytes([stream[p], stream[p + 1]]) as usize; p += 2;
        aq_bw = u16::from_le_bytes([stream[p], stream[p + 1]]) as usize; p += 2;
        let n_mask_bytes = (aq_bh * aq_bw + 7) / 8;
        let mask_packed = &stream[p..p + n_mask_bytes];
        p += n_mask_bytes;
        smooth_mask = (0..(aq_bh * aq_bw)).map(|i| {
            (mask_packed[i / 8] >> (i % 8)) & 1 != 0
        }).collect();
    } else {
        aq_bh = 0; aq_bw = 0;
        smooth_mask = vec![];
    }

    let n_blocks = u16::from_le_bytes([stream[p], stream[p + 1]]) as usize; p += 2;

    // Read bit-packed blocks
    let mut reader = BitReader::new(&stream[p..]);
    let mut flat_morton = vec![0i16; n];
    let mut rest_positions: Vec<usize> = Vec::new();

    for b in 0..n_blocks {
        let start = b * ADN_BLOCK_SIZE;
        let end = (start + ADN_BLOCK_SIZE).min(n);
        let block_b = end - start;

        let is_empty = reader.read_bit().unwrap_or(true);
        if is_empty { continue; }

        let k = reader.read_bits(6).unwrap_or(0) as usize;
        let nbits = rank_bits(k, block_b);
        let rank = if nbits > 0 { reader.read_bits(nbits).unwrap_or(0) } else { 0 };
        let positions = comb_unrank(rank, k, block_b);

        for &lpos in &positions {
            let gpos = start + lpos;
            let is_pm1 = reader.read_bit().unwrap_or(true);
            let negative = reader.read_bit().unwrap_or(false);

            if is_pm1 {
                flat_morton[gpos] = if negative { -1 } else { 1 };
            } else {
                rest_positions.push(gpos);
            }
        }
    }

    // Advance p
    p += (reader.bit_position() + 7) / 8;

    // Remaining values
    let n_rest = u32::from_le_bytes([stream[p], stream[p+1], stream[p+2], stream[p+3]]) as usize;
    p += 4;
    if n_rest > 0 {
        let rest_marker = stream[p]; p += 1;
        for (ri, &gpos) in rest_positions.iter().enumerate() {
            if ri >= n_rest { break; }
            flat_morton[gpos] = if rest_marker == 0x08 {
                stream[p + ri] as i8 as i16
            } else {
                i16::from_le_bytes([stream[p + ri * 2], stream[p + ri * 2 + 1]])
            };
        }
        if rest_marker == 0x08 { p += n_rest; } else { p += n_rest * 2; }
    }

    // Morton inverse → raster
    let inv = morton_inverse(h, w);
    let mut flat_raster = vec![0.0f64; n];
    for i in 0..n {
        flat_raster[i] = flat_morton[inv[i]] as f64;
    }

    // Dequantize
    let mut band_out = if flags & FLAG_ADAPTIVE_QUANT != 0 {
        let step_map = build_step_map(h, w, aq_bh, aq_bw, &smooth_mask, base_step, bs);
        let mut arr = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                arr[[i, j]] = flat_raster[i * w + j] * step_map[i * w + j];
            }
        }
        arr
    } else {
        let arr_q = Array2::from_shape_vec((h, w), flat_raster).unwrap();
        dequantize_band(&arr_q, base_step)
    };

    // Spectral Spin
    {
        let s = band_out.as_slice_mut().unwrap();
        spectral_spin(s, h, w, base_step);
    }

    (band_out, p)
}
