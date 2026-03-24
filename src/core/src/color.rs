/// YCbCr <-> RGB conversion (ITU-R BT.601) and 4:2:0 subsampling.

use rayon::prelude::*;

/// Convert a YCbCr pixel to RGB, returns (R, G, B) clipped to [0, 255].
#[inline]
pub fn ycbcr_to_rgb_pixel(y: f64, cb: f64, cr: f64) -> (u8, u8, u8) {
    let cb = cb - 128.0;
    let cr = cr - 128.0;
    let r = (y + 1.402 * cr).clamp(0.0, 255.0) as u8;
    let g = (y - 0.344136 * cb - 0.714136 * cr).clamp(0.0, 255.0) as u8;
    let b = (y + 1.772 * cb).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

/// Convert a YCbCr image (H, W, 3) to RGB (H, W, 3).
/// The Y, Cb, Cr planes are passed separately as f64.
pub fn ycbcr_to_rgb(y_plane: &[f64], cb_plane: &[f64], cr_plane: &[f64], len: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; len * 3];
    rgb.par_chunks_mut(3).enumerate().for_each(|(i, chunk)| {
        let (r, g, b) = ycbcr_to_rgb_pixel(y_plane[i], cb_plane[i], cr_plane[i]);
        chunk[0] = r;
        chunk[1] = g;
        chunk[2] = b;
    });
    rgb
}

/// Bilinear upsampling of a chroma plane (Hc, Wc) to (H, W).
/// Reproduces exactly scipy.ndimage.zoom(channel, (fy, fx), order=1).
/// scipy mapping: in_coord = out_coord * (in_size - 1) / (out_size - 1)
pub fn upsample_420(channel: &[f64], hc: usize, wc: usize, target_h: usize, target_w: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; target_h * target_w];

    let scale_y = if target_h > 1 { (hc as f64 - 1.0) / (target_h as f64 - 1.0) } else { 0.0 };
    let scale_x = if target_w > 1 { (wc as f64 - 1.0) / (target_w as f64 - 1.0) } else { 0.0 };

    result.par_chunks_mut(target_w).enumerate().for_each(|(y, row)| {
        let sy = y as f64 * scale_y;
        let iy0 = (sy as usize).min(hc - 1);
        let iy1 = (iy0 + 1).min(hc - 1);
        let dy = sy - iy0 as f64;

        for x in 0..target_w {
            let sx = x as f64 * scale_x;
            let ix0 = (sx as usize).min(wc - 1);
            let ix1 = (ix0 + 1).min(wc - 1);
            let dx = sx - ix0 as f64;

            let v00 = channel[iy0 * wc + ix0];
            let v01 = channel[iy0 * wc + ix1];
            let v10 = channel[iy1 * wc + ix0];
            let v11 = channel[iy1 * wc + ix1];

            row[x] = v00 * (1.0 - dy) * (1.0 - dx)
                   + v01 * (1.0 - dy) * dx
                   + v10 * dy * (1.0 - dx)
                   + v11 * dy * dx;
        }
    });

    result
}

/// RGB -> YCbCr BT.601 conversion. Returns 3 planes (Y, Cb, Cr) as f64.
pub fn rgb_to_ycbcr_planes(rgb: &[u8], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut y = vec![0.0f64; n];
    let mut cb = vec![0.0f64; n];
    let mut cr = vec![0.0f64; n];

    // Process in chunks for rayon
    let chunk_size = (n / rayon::current_num_threads().max(1)).max(4096);
    y.par_chunks_mut(chunk_size)
        .zip(cb.par_chunks_mut(chunk_size))
        .zip(cr.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(chunk_idx, ((y_chunk, cb_chunk), cr_chunk))| {
            let start = chunk_idx * chunk_size;
            for j in 0..y_chunk.len() {
                let i = start + j;
                let r = rgb[i * 3] as f64;
                let g = rgb[i * 3 + 1] as f64;
                let b = rgb[i * 3 + 2] as f64;
                y_chunk[j] = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0);
                cb_chunk[j] = (-0.169 * r - 0.331 * g + 0.500 * b + 128.0).clamp(0.0, 255.0);
                cr_chunk[j] = (0.500 * r - 0.419 * g - 0.081 * b + 128.0).clamp(0.0, 255.0);
            }
        });

    (y, cb, cr)
}

/// RGB (separate f64 planes) -> YCbCr BT.601 conversion. Returns 3 planes (Y, Cb, Cr).
pub fn rgb_to_ycbcr_from_f64(r: &[f64], g: &[f64], b: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut y = Vec::with_capacity(n);
    let mut cb = Vec::with_capacity(n);
    let mut cr = Vec::with_capacity(n);

    for i in 0..n {
        y.push((0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]).clamp(0.0, 255.0));
        cb.push((-0.169 * r[i] - 0.331 * g[i] + 0.500 * b[i] + 128.0).clamp(0.0, 255.0));
        cr.push((0.500 * r[i] - 0.419 * g[i] - 0.081 * b[i] + 128.0).clamp(0.0, 255.0));
    }

    (y, cb, cr)
}

/// 4:2:0 subsampling by averaging 2x2 blocks.
/// Returns (channel_sub, hc, wc).
pub fn subsample_420_encode(channel: &[f64], h: usize, w: usize) -> (Vec<f64>, usize, usize) {
    let hc = (h + 1) / 2;
    let wc = (w + 1) / 2;
    let mut result = vec![0.0f64; hc * wc];

    for cy in 0..hc {
        for cx in 0..wc {
            let y0 = cy * 2;
            let y1 = (y0 + 1).min(h - 1);
            let x0 = cx * 2;
            let x1 = (x0 + 1).min(w - 1);

            let v00 = channel[y0 * w + x0];
            let v01 = channel[y0 * w + x1];
            let v10 = channel[y1 * w + x0];
            let v11 = channel[y1 * w + x1];

            result[cy * wc + cx] = (v00 + v01 + v10 + v11) / 4.0;
        }
    }

    (result, hc, wc)
}

/// Golden Color Transform (GCT) -- golden rotation of the color space.
///
/// Golden luminance: L_phi = (R + phi*G + phi^-1*B) / (2*phi)
///   Weights: R=0.309, G=0.500, B=0.191 (close to BT.601, naturally!)
/// Chromas: C1 = B - L_phi, C2 = R - L_phi (centered on zero for gray)
///
/// Inverse: R = L + C2, G = L - phi^-2*C1 - phi^-1*C2, B = L + C1
///
/// Properties: luma/chroma decorrelation, gray -> (L, 0, 0),
/// the inverse uses only phi^-1 and phi^-2 (Fibonacci sequence).
pub fn golden_rotate_forward(rgb: &[u8], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use crate::golden::{PHI, PHI_INV};
    let norm = 1.0 / (2.0 * PHI);

    let mut l = vec![0.0f64; n];
    let mut c1 = vec![0.0f64; n];
    let mut c2 = vec![0.0f64; n];

    let chunk_size = (n / rayon::current_num_threads().max(1)).max(4096);
    l.par_chunks_mut(chunk_size)
        .zip(c1.par_chunks_mut(chunk_size))
        .zip(c2.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(chunk_idx, ((l_chunk, c1_chunk), c2_chunk))| {
            let start = chunk_idx * chunk_size;
            for j in 0..l_chunk.len() {
                let i = start + j;
                let r = rgb[i * 3] as f64;
                let g = rgb[i * 3 + 1] as f64;
                let b = rgb[i * 3 + 2] as f64;
                let lum = (r + PHI * g + PHI_INV * b) * norm;
                l_chunk[j] = lum;
                c1_chunk[j] = b - lum;
                c2_chunk[j] = r - lum;
            }
        });

    (l, c1, c2)
}

/// Inverse of GCT: (L_phi, C1, C2) -> (R, G, B) as f64.
pub fn golden_rotate_inverse(l: &[f64], c1: &[f64], c2: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use crate::golden::{PHI_INV, PHI_INV2};

    let mut r = vec![0.0f64; n];
    let mut g = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];

    let chunk_size = (n / rayon::current_num_threads().max(1)).max(4096);
    r.par_chunks_mut(chunk_size)
        .zip(g.par_chunks_mut(chunk_size))
        .zip(b.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(chunk_idx, ((r_chunk, g_chunk), b_chunk))| {
            let start = chunk_idx * chunk_size;
            for j in 0..r_chunk.len() {
                let i = start + j;
                r_chunk[j] = l[i] + c2[i];
                g_chunk[j] = l[i] - PHI_INV2 * c1[i] - PHI_INV * c2[i];
                b_chunk[j] = l[i] + c1[i];
            }
        });

    (r, g, b)
}

// ======================================================================
// 4:2:2 subsampling (horizontal only, for C2 red chroma)
// ======================================================================

/// 4:2:2 subsampling: average pairs horizontally, keep full vertical resolution.
/// Returns (channel_sub, h, wc) where wc = (w+1)/2.
pub fn subsample_422_encode(channel: &[f64], h: usize, w: usize) -> (Vec<f64>, usize, usize) {
    let wc = (w + 1) / 2;
    let mut result = vec![0.0f64; h * wc];

    for y in 0..h {
        for cx in 0..wc {
            let x0 = cx * 2;
            let x1 = (x0 + 1).min(w - 1);
            result[y * wc + cx] = (channel[y * w + x0] + channel[y * w + x1]) / 2.0;
        }
    }

    (result, h, wc)
}

/// Bilinear upsample from 4:2:2 (h, wc) to (h, w) — horizontal only.
pub fn upsample_422(channel: &[f64], h: usize, wc: usize, target_w: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; h * target_w];
    let scale_x = if target_w > 1 { (wc as f64 - 1.0) / (target_w as f64 - 1.0) } else { 0.0 };

    for y in 0..h {
        for x in 0..target_w {
            let sx = x as f64 * scale_x;
            let ix0 = (sx as usize).min(wc.saturating_sub(1));
            let ix1 = (ix0 + 1).min(wc.saturating_sub(1));
            let dx = sx - ix0 as f64;
            result[y * target_w + x] =
                channel[y * wc + ix0] * (1.0 - dx) + channel[y * wc + ix1] * dx;
        }
    }

    result
}

// ======================================================================
// Phi-Chroma: saturation map + adaptive chroma factor + chroma residual
// ======================================================================

/// Compute normalized saturation map from two chroma LL planes.
/// Returns S_norm in [0, 1] per pixel: 0 = achromatic, 1 = fully saturated.
/// Both encoder and decoder call this on reconstructed LL data (identical).
pub fn saturation_map(c1: &[f64], c2: &[f64], h: usize, w: usize) -> Vec<f64> {
    let n = h * w;
    let raw: Vec<f64> = (0..n)
        .map(|i| (c1[i] * c1[i] + c2[i] * c2[i]).sqrt())
        .collect();

    // 95th percentile (robust normalization)
    let mut sorted = raw.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx_95 = ((n as f64 * 0.95) as usize).min(n.saturating_sub(1));
    let s_95th = sorted[idx_95].max(1.0); // floor at 1.0 for achromatic images

    raw.iter().map(|&s| (s / s_95th).clamp(0.0, 1.0)).collect()
}

/// Phi-adaptive chroma factor from normalized saturation.
/// factor(S) = phi^-2 + (1 - phi^-2) * (1 - S)^2.
/// Range: [phi^-2, 1.0] = [0.382, 1.0].
/// Saturated zones get FINER quantization than luma (factor < 1.0).
/// Desaturated zones get same as luma (factor = 1.0).
#[inline]
pub fn phi_chroma_factor(s_norm: f64) -> f64 {
    use crate::golden::PHI_INV2;
    PHI_INV2 + (1.0 - PHI_INV2) * (1.0 - s_norm) * (1.0 - s_norm)
}

/// Perceptual dead zone map: DNA bathtub curve + texture modulation.
///
/// The dead zone balances two competing needs:
/// - At edges/contours in dark/bright zones: SMALL dz (preserve detail)
/// - In smooth/flat dark zones: LARGE dz (kill structured ±1 noise = Morton diamond grid)
///
/// Formula:
///   sensitivity = cos(pi * L_norm)^2        [bathtub: 1 at extremes, 0 at mid]
///   texture = local gradient magnitude       [high at edges, low in flat zones]
///   dz = dz_base * (1 - phi^-1 * sensitivity * texture_norm)
///
/// - Dark/bright EDGES: small dz (sensitivity=1, texture=1) → preserves contours
/// - Dark/bright FLATS: large dz (sensitivity=1, texture=0) → kills Morton diamonds
/// - Mid-tones: dz = base regardless of texture
pub fn perceptual_dz_map(luma: &[f64], h: usize, w: usize, base_dz: f64) -> Vec<f64> {
    use crate::golden::PHI_INV;
    let n = h * w;
    if n == 0 { return vec![]; }

    // Robust luminance range
    let mut sorted: Vec<f64> = luma.iter().copied().collect();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx_05 = ((n as f64 * 0.05) as usize).min(n.saturating_sub(1));
    let idx_95 = ((n as f64 * 0.95) as usize).min(n.saturating_sub(1));
    let l_min = sorted[idx_05];
    let l_max = sorted[idx_95].max(l_min + 1.0);

    // Local gradient magnitude (texture detector)
    let mut grad = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let gy = if y + 1 < h {
                (luma[(y + 1) * w + x] - luma[y * w + x]).abs()
            } else { 0.0 };
            let gx = if x + 1 < w {
                (luma[y * w + x + 1] - luma[y * w + x]).abs()
            } else { 0.0 };
            grad[y * w + x] = gy.max(gx);
        }
    }
    // Normalize gradient: 95th percentile
    let mut g_sorted = grad.clone();
    g_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let g_95 = g_sorted[((n as f64 * 0.95) as usize).min(n.saturating_sub(1))].max(1.0);

    (0..n).map(|i| {
        let l_norm = ((luma[i] - l_min) / (l_max - l_min)).clamp(0.0, 1.0);
        let cos_val = (std::f64::consts::PI * l_norm).cos();
        let sensitivity = cos_val * cos_val;

        let texture_norm = (grad[i] / g_95).clamp(0.0, 1.0);

        // At edges (texture=1): reduce dz (preserve detail)
        // At flats (texture=0): keep dz at base (kill noise)
        base_dz * (1.0 - PHI_INV * sensitivity * texture_norm)
    }).collect()
}

/// v9 compensatory dead zone: mid-tones get wider dead zone to offset
/// Weber-Fechner bpp cost in dark areas. Replaces `perceptual_dz_map`.
///
/// The compensator uses sin(pi * L_norm)^2 which peaks at 0.5 (mid-tone)
/// and is zero at extremes (dark/bright). This widens mid-tone dz by up to 50%,
/// saving bits where the eye is least sensitive to quantization noise.
pub fn perceptual_dz_map_v9(luma: &[f64], h: usize, w: usize, base_dz: f64) -> Vec<f64> {
    use crate::golden::PHI_INV;
    let n = h * w;
    if n == 0 { return vec![]; }

    // Robust luminance range
    let mut sorted: Vec<f64> = luma.iter().copied().collect();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let idx_05 = ((n as f64 * 0.05) as usize).min(n.saturating_sub(1));
    let idx_95 = ((n as f64 * 0.95) as usize).min(n.saturating_sub(1));
    let l_min_robust = sorted[idx_05];
    let l_max_robust = sorted[idx_95];
    // If dynamic range is narrow (< 32 levels), use absolute [0, 255] normalization
    // so mid-tones are recognized as mid-tones even in flat patches.
    let (l_min, l_max) = if (l_max_robust - l_min_robust) < 32.0 {
        (0.0, 255.0)
    } else {
        (l_min_robust, l_max_robust.max(l_min_robust + 1.0))
    };

    // Local gradient magnitude (texture detector)
    let mut grad = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let gy = if y + 1 < h { (luma[(y + 1) * w + x] - luma[y * w + x]).abs() } else { 0.0 };
            let gx = if x + 1 < w { (luma[y * w + x + 1] - luma[y * w + x]).abs() } else { 0.0 };
            grad[y * w + x] = gy.max(gx);
        }
    }
    // Normalize gradient: 95th percentile
    let mut g_sorted = grad.clone();
    g_sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let g_95 = g_sorted[((n as f64 * 0.95) as usize).min(n.saturating_sub(1))].max(1.0);

    (0..n).map(|i| {
        let l_norm = ((luma[i] - l_min) / (l_max - l_min)).clamp(0.0, 1.0);
        let cos_val = (std::f64::consts::PI * l_norm).cos();
        let sensitivity = cos_val * cos_val;
        let texture_norm = (grad[i] / g_95).clamp(0.0, 1.0);

        // Compensator: sin^2 peaks at mid-tone (l_norm=0.5), zero at extremes
        let sin_val = (std::f64::consts::PI * l_norm).sin();
        let compensator = 1.0 + 0.5 * sin_val * sin_val;

        base_dz * compensator * (1.0 - PHI_INV * sensitivity * texture_norm)
    }).collect()
}

/// Bilinear downsample from (src_h, src_w) to (dst_h, dst_w).
/// Same mapping as upsample_420: in = out * (in_size-1)/(out_size-1).
pub fn downsample_2d(
    data: &[f64], src_h: usize, src_w: usize, dst_h: usize, dst_w: usize,
) -> Vec<f64> {
    let mut result = vec![0.0f64; dst_h * dst_w];
    let scale_y = if dst_h > 1 { (src_h as f64 - 1.0) / (dst_h as f64 - 1.0) } else { 0.0 };
    let scale_x = if dst_w > 1 { (src_w as f64 - 1.0) / (dst_w as f64 - 1.0) } else { 0.0 };

    result.par_chunks_mut(dst_w).enumerate().for_each(|(y, row)| {
        let sy = y as f64 * scale_y;
        let iy0 = (sy as usize).min(src_h.saturating_sub(1));
        let iy1 = (iy0 + 1).min(src_h.saturating_sub(1));
        let dy = sy - iy0 as f64;
        for x in 0..dst_w {
            let sx = x as f64 * scale_x;
            let ix0 = (sx as usize).min(src_w.saturating_sub(1));
            let ix1 = (ix0 + 1).min(src_w.saturating_sub(1));
            let dx = sx - ix0 as f64;
            row[x] = data[iy0 * src_w + ix0] * (1.0 - dy) * (1.0 - dx)
                   + data[iy0 * src_w + ix1] * (1.0 - dy) * dx
                   + data[iy1 * src_w + ix0] * dy * (1.0 - dx)
                   + data[iy1 * src_w + ix1] * dy * dx;
        }
    });
    result
}

/// Encode the chroma subsampling residual for one channel.
///
/// Computes: residual = original_fullres - upsample(subsampled)
/// Then masks by block 8x8: only blocks where sat_norm > phi^-1 are kept.
/// Returns (block_mask, quantized_residuals_for_active_blocks).
///
/// block_mask: 1 bit per 8x8 block (row-major, packed into bytes)
/// residuals: i8 values for active blocks only (row-major within block, blocks in scan order)
pub fn encode_chroma_residual(
    original: &[f64],     // full-res channel (H x W)
    subsampled: &[f64],   // 4:2:0 channel (Hc x Wc)
    sat_map_full: &[f64], // saturation S_norm at full resolution (H x W)
    h: usize, w: usize,
    hc: usize, wc: usize,
    quant_step: f64,
) -> (Vec<u8>, Vec<i8>) {
    use crate::golden::PHI_INV;
    const BLOCK: usize = 8;

    // Upsample subsampled back to full res
    let upsampled = upsample_420(subsampled, hc, wc, h, w);

    // Residual = original - upsampled
    let residual: Vec<f64> = original.iter().zip(&upsampled)
        .map(|(&o, &u)| o - u)
        .collect();

    let bh = (h + BLOCK - 1) / BLOCK;
    let bw = (w + BLOCK - 1) / BLOCK;
    let n_blocks = bh * bw;
    let mask_bytes = (n_blocks + 7) / 8;
    let mut block_mask = vec![0u8; mask_bytes];
    let mut active_residuals: Vec<i8> = Vec::new();

    for by in 0..bh {
        for bx in 0..bw {
            let block_idx = by * bw + bx;
            let y0 = by * BLOCK;
            let x0 = bx * BLOCK;
            let y1 = (y0 + BLOCK).min(h);
            let x1 = (x0 + BLOCK).min(w);

            // Average saturation in this block
            let mut sat_sum = 0.0;
            let mut count = 0;
            for y in y0..y1 {
                for x in x0..x1 {
                    sat_sum += sat_map_full[y * w + x];
                    count += 1;
                }
            }
            let sat_avg = sat_sum / count.max(1) as f64;

            // Only activate block if saturation > phi^-1
            if sat_avg > PHI_INV {
                block_mask[block_idx / 8] |= 1 << (block_idx % 8);

                // Quantize and store residuals for this block
                for y in y0..y1 {
                    for x in x0..x1 {
                        let r = residual[y * w + x];
                        let q = (r / quant_step).round().clamp(-127.0, 127.0) as i8;
                        active_residuals.push(q);
                    }
                }
            }
        }
    }

    (block_mask, active_residuals)
}

/// Decode the chroma residual and add it to the upsampled chroma.
/// Modifies chroma_full in-place.
pub fn decode_chroma_residual(
    chroma_full: &mut [f64],  // upsampled chroma (H x W), modified in-place
    block_mask: &[u8],
    residuals: &[i8],
    h: usize, w: usize,
    quant_step: f64,
) {
    const BLOCK: usize = 8;
    let bh = (h + BLOCK - 1) / BLOCK;
    let bw = (w + BLOCK - 1) / BLOCK;
    let mut res_idx = 0;

    for by in 0..bh {
        for bx in 0..bw {
            let block_idx = by * bw + bx;
            let active = (block_mask[block_idx / 8] >> (block_idx % 8)) & 1 != 0;

            if active {
                let y0 = by * BLOCK;
                let x0 = bx * BLOCK;
                let y1 = (y0 + BLOCK).min(h);
                let x1 = (x0 + BLOCK).min(w);

                for y in y0..y1 {
                    for x in x0..x1 {
                        if res_idx < residuals.len() {
                            chroma_full[y * w + x] += residuals[res_idx] as f64 * quant_step;
                            res_idx += 1;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_chroma_factor() {
        use crate::golden::PHI_INV2;
        // phi_chroma_factor(0.0) = PHI_INV2 + (1-PHI_INV2)*1 = 1.0 (desaturated → same as luma)
        let f0 = phi_chroma_factor(0.0);
        assert!((f0 - 1.0).abs() < 1e-10, "f(0) should be 1.0, got {}", f0);
        // phi_chroma_factor(1.0) = PHI_INV2 = 0.382 (fully saturated → finest chroma quant)
        let f1 = phi_chroma_factor(1.0);
        assert!((f1 - PHI_INV2).abs() < 1e-10, "f(1) should be phi^-2, got {}", f1);
        // Monotone decreasing
        for i in 0..100 {
            assert!(phi_chroma_factor(i as f64 / 100.0) >= phi_chroma_factor((i + 1) as f64 / 100.0));
        }
    }

    #[test]
    fn test_saturation_map_grey() {
        let c1 = vec![0.1, 0.2, 0.0, 0.1];
        let c2 = vec![0.0, 0.1, 0.1, 0.0];
        let sat = saturation_map(&c1, &c2, 2, 2);
        for &s in &sat { assert!(s < 0.3); }
    }

    #[test]
    fn test_ycbcr_white() {
        // Pure white: Y=255, Cb=128, Cr=128 -> R=255, G=255, B=255
        let (r, g, b) = ycbcr_to_rgb_pixel(255.0, 128.0, 128.0);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn test_ycbcr_black() {
        let (r, g, b) = ycbcr_to_rgb_pixel(0.0, 128.0, 128.0);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn test_gct_roundtrip() {
        // GCT forward + inverse should give original values
        let rgb: Vec<u8> = vec![200, 100, 50, 0, 0, 0, 255, 255, 255, 128, 128, 128];
        let n = 4;
        let (l, c1, c2) = golden_rotate_forward(&rgb, n);

        // Gray (128,128,128) -> C1=0, C2=0
        assert!((c1[3]).abs() < 1e-10, "gray C1 should be 0, got {}", c1[3]);
        assert!((c2[3]).abs() < 1e-10, "gray C2 should be 0, got {}", c2[3]);

        // Inverse
        let (r, g, b) = golden_rotate_inverse(&l, &c1, &c2, n);
        for i in 0..n {
            let ro = rgb[i * 3] as f64;
            let go = rgb[i * 3 + 1] as f64;
            let bo = rgb[i * 3 + 2] as f64;
            assert!((r[i] - ro).abs() < 1e-10, "R mismatch at {}: {} vs {}", i, r[i], ro);
            assert!((g[i] - go).abs() < 1e-10, "G mismatch at {}: {} vs {}", i, g[i], go);
            assert!((b[i] - bo).abs() < 1e-10, "B mismatch at {}: {} vs {}", i, b[i], bo);
        }
    }

    #[test]
    fn test_upsample_identity() {
        // 2x2 -> 4x4
        let ch = vec![10.0, 20.0, 30.0, 40.0];
        let up = upsample_420(&ch, 2, 2, 4, 4);
        assert_eq!(up.len(), 16);
        // Corners should be close to source values
        assert!((up[0] - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_perceptual_dz_map_v9_midtone_wider() {
        let luma = vec![128.0; 64]; // 8x8 flat mid-tone
        let dz = perceptual_dz_map_v9(&luma, 8, 8, 0.15);
        let avg: f64 = dz.iter().sum::<f64>() / dz.len() as f64;
        // Mid-tone flat: compensator=1.5, sensitivity=0, so dz = 0.15 * 1.5 = 0.225
        assert!(avg > 0.20 && avg < 0.25, "mid-tone dz should be ~0.225, got {}", avg);
    }

    #[test]
    fn test_perceptual_dz_map_v9_dark_unchanged() {
        let luma = vec![10.0; 64]; // 8x8 flat dark
        let dz = perceptual_dz_map_v9(&luma, 8, 8, 0.15);
        let avg: f64 = dz.iter().sum::<f64>() / dz.len() as f64;
        assert!(avg > 0.12 && avg < 0.18, "dark dz should be ~0.15, got {}", avg);
    }
}
