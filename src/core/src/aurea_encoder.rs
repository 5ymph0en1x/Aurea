/// Native AUREA encoder: RGB -> .aur file
///
/// v1: RGB -> YCbCr 4:2:0 -> CDF 9/7 wavelet -> LL: VQ+Paeth | Details: Fibonacci/Zeckendorf
/// v2: RGB -> GCT 4:2:0 -> CDF 9/7 wavelet -> LL: VQ+Paeth | Details: uniform quant + Morton+2bit
///   - Adaptive auto_detail_step (gradient energy + quality)
///   - FLAG_ADAPTIVE_QUANT (smooth blocks x1.5)
///   - Spectral spin decoder-side (+0.35 dB)
///   - detail_step stored in header

use byteorder::{LittleEndian, WriteBytesExt};
use ndarray::Array2;

use crate::aurea::AUREA_MAGIC;
use crate::color;
use crate::geometric;
use crate::paeth;
use crate::rans;
use crate::vq;
use crate::wavelet;

/// AUREA encoding parameters.
pub struct AureaEncoderParams {
    pub quality: u8,
    pub n_representatives: usize,
    pub geometric: bool,
}

/// AUREA encoding result.
pub struct AureaEncoderResult {
    pub aurea_data: Vec<u8>,
    pub compressed_size: usize,
}


/// Compute the auto-adaptive detail_step (identical to Python _auto_detail_step).
/// Uses the gradient energy of the luminance channel to adapt the step
/// to the image complexity.
fn auto_detail_step(l_channel: &[f64], height: usize, width: usize, quality: u8) -> f64 {
    // Gradient energy (finite differences vertical + horizontal)
    let mut sum_dy2 = 0.0;
    let mut count_dy = 0usize;
    let mut sum_dx2 = 0.0;
    let mut count_dx = 0usize;

    // Vertical gradients: dy = L[i+1,j] - L[i,j]
    for i in 0..height - 1 {
        for j in 0..width {
            let dy = l_channel[(i + 1) * width + j] - l_channel[i * width + j];
            sum_dy2 += dy * dy;
            count_dy += 1;
        }
    }
    // Horizontal gradients: dx = L[i,j+1] - L[i,j]
    for i in 0..height {
        for j in 0..width - 1 {
            let dx = l_channel[i * width + j + 1] - l_channel[i * width + j];
            sum_dx2 += dx * dx;
            count_dx += 1;
        }
    }
    // mean(dy^2) + mean(dx^2) -- identical to Python np.mean(dy**2) + np.mean(dx**2)
    let mean_energy = sum_dy2 / count_dy.max(1) as f64
                    + sum_dx2 / count_dx.max(1) as f64;

    // Normalized complexity [0, 1]
    let complexity = (mean_energy + 1.0).log2(); // ~6.5 (smooth) to ~11 (texture)
    let c_norm = ((complexity - 6.0) / 5.0).clamp(0.0, 1.0);

    // Base step: quality 100 -> 2.0, quality 1 -> 30.0
    let q_frac = (quality as f64 / 100.0).clamp(0.01, 1.0);
    let base_step = 2.0 + (1.0 - q_frac) * 28.0;

    // Adjustment by complexity
    let factor = 1.0 - 0.4 * c_norm; // [0.6, 1.0]
    let step = base_step * factor;

    step.clamp(1.0, 48.0)
}




/// Encode an RGB image as an .aur file (AUREA v2 "Golden Fusion").
///
/// Golden rotation of the color space (GCT):
///   L_phi = (R + phi*G + phi^-1*B) / (2*phi)   [golden luminance]
///   C_phi1 = B - L_phi                          [blue chroma]
///   C_phi2 = R - L_phi                          [red chroma]
/// The luminance weights (0.309R + 0.500G + 0.191B) naturally approximate
/// human sensitivity (BT.601: 0.299R + 0.587G + 0.114B).
///
/// Each channel is encoded independently. The C1/C2 chromas receive
/// a step multiplied by phi (more aggressive quantization).
pub fn encode_aurea_v2(
    rgb: &[u8],
    width: usize,
    height: usize,
    params: &AureaEncoderParams,
) -> Result<AureaEncoderResult, Box<dyn std::error::Error>> {
    let n = height * width;

    // 1. Golden Color Transform: RGB -> (L_phi, C_phi1, C_phi2)
    let (l_ch, c1_ch, c2_ch) = color::golden_rotate_forward(rgb, n);

    // Auto-adaptive detail step (must be computed before l_ch is moved)
    let detail_step = auto_detail_step(&l_ch, height, width, params.quality);

    // 2. 4:2:0 subsampling of chromas C1/C2
    let hc = (height + 1) / 2;
    let wc = (width + 1) / 2;
    let (c1_sub, _, _) = color::subsample_420_encode(&c1_ch, height, width);
    let (c2_sub, _, _) = color::subsample_420_encode(&c2_ch, height, width);

    // 3. CDF 9/7 wavelet decomposition
    let wv_levels = wavelet::auto_wv_levels(height, width);
    let l_arr = Array2::from_shape_vec((height, width), l_ch)?;
    let c1_arr = Array2::from_shape_vec((hc, wc), c1_sub)?;
    let c2_arr = Array2::from_shape_vec((hc, wc), c2_sub)?;

    let (l_ll, l_subs, _) = wavelet::wavelet_decompose(&l_arr, wv_levels);
    let (c1_ll, c1_subs, _) = wavelet::wavelet_decompose(&c1_arr, wv_levels);
    let (c2_ll, c2_subs, _) = wavelet::wavelet_decompose(&c2_arr, wv_levels);

    let ll_lh = l_ll.nrows();
    let ll_lw = l_ll.ncols();
    let ll_ch = c1_ll.nrows();
    let ll_cw = c1_ll.ncols();

    // 4. Normalize LL to [0, 255] for VQ + fibonacci
    let l_ll_flat: Vec<f64> = l_ll.iter().copied().collect();
    let c1_ll_flat: Vec<f64> = c1_ll.iter().copied().collect();
    let c2_ll_flat: Vec<f64> = c2_ll.iter().copied().collect();

    let (l_ll_norm, l_ll_min, l_ll_max) = crate::normalize_ll(&l_ll_flat);
    let (c1_ll_norm, c1_ll_min, c1_ll_max) = crate::normalize_ll(&c1_ll_flat);
    let (c2_ll_norm, c2_ll_min, c2_ll_max) = crate::normalize_ll(&c2_ll_flat);

    // 5. VQ quantization + refinement
    let n_l = params.n_representatives.min(255);
    let n_c = 16usize.max(n_l * 3 / 4).min(255);

    let centroids_l = vq::kmeans_1d(&l_ll_norm, n_l, 10);
    let centroids_c1 = vq::kmeans_1d(&c1_ll_norm, n_c, 10);
    let centroids_c2 = vq::kmeans_1d(&c2_ll_norm, n_c, 10);

    let centroids_l = vq::refine_centroids(&l_ll_norm, &centroids_l, ll_lh, ll_lw, 5);
    let centroids_c1 = vq::refine_centroids(&c1_ll_norm, &centroids_c1, ll_ch, ll_cw, 5);
    let centroids_c2 = vq::refine_centroids(&c2_ll_norm, &centroids_c2, ll_ch, ll_cw, 5);

    let cl_u8: Vec<f64> = centroids_l.iter().map(|&c| c.round().clamp(0.0, 255.0)).collect();
    let cc1_u8: Vec<f64> = centroids_c1.iter().map(|&c| c.round().clamp(0.0, 255.0)).collect();
    let cc2_u8: Vec<f64> = centroids_c2.iter().map(|&c| c.round().clamp(0.0, 255.0)).collect();

    // 6. Label assignment + Paeth prediction
    let labels_l = vq::assign_nearest(&l_ll_norm, &cl_u8);
    let labels_c1 = vq::assign_nearest(&c1_ll_norm, &cc1_u8);
    let labels_c2 = vq::assign_nearest(&c2_ll_norm, &cc2_u8);

    let pred_l = paeth::paeth_predict_2d(&labels_l, ll_lh, ll_lw);
    let pred_c1 = paeth::paeth_predict_2d(&labels_c1, ll_ch, ll_cw);
    let pred_c2 = paeth::paeth_predict_2d(&labels_c2, ll_ch, ll_cw);

    // 7. Encode detail bands
    let flags = wavelet::FLAG_MORTON_2BIT | wavelet::FLAG_ADAPTIVE_QUANT;
    let mut det_stream: Vec<u8> = Vec::new();

    if params.geometric {
        // v6 geometric: level -> channel -> (primitives + 3 residuals)
        for lv_idx in 0..wv_levels {
            let lv = wv_levels - 1 - lv_idx; // deepest first
            let level_scale = if lv < wavelet::LEVEL_SCALES.len() {
                wavelet::LEVEL_SCALES[lv]
            } else {
                0.3
            };

            let steps_lh = detail_step * level_scale * wavelet::PERCEPTUAL_BAND_WEIGHTS[0];
            let steps_hl = detail_step * level_scale * wavelet::PERCEPTUAL_BAND_WEIGHTS[1];
            let steps_hh = detail_step * level_scale * wavelet::PERCEPTUAL_BAND_WEIGHTS[2];

            // For each channel (L, C1, C2)
            let channel_subs = [&l_subs[lv], &c1_subs[lv], &c2_subs[lv]];
            let channel_chroma = [false, true, true];

            for (ch_idx, subs) in channel_subs.iter().enumerate() {
                let chroma_mul = if channel_chroma[ch_idx] { 1.5 } else { 1.0 };
                let step_lh = steps_lh * chroma_mul;
                let step_hl = steps_hl * chroma_mul;
                let step_hh = steps_hh * chroma_mul;

                // Quantize the 3 bands for extract_primitives
                let q_lh = wavelet::quantize_band(&subs.0, step_lh);
                let q_hl = wavelet::quantize_band(&subs.1, step_hl);
                let q_hh = wavelet::quantize_band(&subs.2, step_hh);

                let q_lh = wavelet::entropy_threshold(&q_lh, 8, 0.10);
                let q_hl = wavelet::entropy_threshold(&q_hl, 8, 0.10);
                let q_hh = wavelet::entropy_threshold(&q_hh, 8, 0.10);

                // Extract primitives + residuals (in quantized space)
                let (prims, res_lh, res_hl, res_hh) =
                    geometric::extract_primitives(&q_lh, &q_hl, &q_hh);

                // Serialize primitives
                let prim_data = geometric::serialize_primitives(&prims);
                det_stream.extend_from_slice(&prim_data);

                // Encode residuals: rANS-band with wavelet context model
                // Each residual band is quantized integers — encode directly with rANS
                for res in [&res_lh, &res_hl, &res_hh] {
                    // Flatten to i16 in Morton order (already quantized)
                    let h = res.nrows();
                    let w = res.ncols();
                    let flat: Vec<i16> = res.iter().map(|&v| v as i16).collect();
                    let order = wavelet::morton_order(h, w);
                    let morton_ordered: Vec<i16> = order.iter().map(|&idx| flat[idx]).collect();
                    // rANS-encode with native wavelet context model
                    let encoded = rans::rans_encode_band(&morton_ordered);
                    det_stream.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
                    det_stream.extend_from_slice(&encoded);
                }
            }
        }
    } else {
        // v2 classic: level -> band_type -> channel
        for lv_idx in 0..wv_levels {
            let lv = wv_levels - 1 - lv_idx; // deepest first
            let level_scale = if lv < wavelet::LEVEL_SCALES.len() {
                wavelet::LEVEL_SCALES[lv]
            } else {
                0.3
            };

            for bi in 0..3 {
                let step_l = detail_step * level_scale * wavelet::PERCEPTUAL_BAND_WEIGHTS[bi];
                let step_c = step_l * 1.5;

                for (band, step) in [
                    (band_ref(&l_subs[lv], bi), step_l),
                    (band_ref(&c1_subs[lv], bi), step_c),
                    (band_ref(&c2_subs[lv], bi), step_c),
                ] {
                    let (data, _deq) = wavelet::encode_detail_band_v2(band, step, flags);
                    det_stream.extend_from_slice(&(data.len() as u32).to_le_bytes());
                    det_stream.extend_from_slice(&data);
                }
            }
        }
    }

    // 8. Assemble the AUREA v2 bitstream
    let mut raw: Vec<u8> = Vec::new();

    raw.push(if params.geometric { 7 } else { 2 }); // v7 superstring+rANS or v2 classic
    raw.push(params.quality);
    raw.write_u16::<LittleEndian>(width as u16)?;
    raw.write_u16::<LittleEndian>(height as u16)?;
    raw.push(wv_levels as u8);
    // detail_step stored in header (auto-adaptive, not reconstructable)
    raw.write_f32::<LittleEndian>(detail_step as f32)?;

    // LL ranges: L_phi, C_phi1, C_phi2
    raw.write_f32::<LittleEndian>(l_ll_min as f32)?;
    raw.write_f32::<LittleEndian>(l_ll_max as f32)?;
    raw.write_f32::<LittleEndian>(c1_ll_min as f32)?;
    raw.write_f32::<LittleEndian>(c1_ll_max as f32)?;
    raw.write_f32::<LittleEndian>(c2_ll_min as f32)?;
    raw.write_f32::<LittleEndian>(c2_ll_max as f32)?;

    // Centroid counts
    raw.push(n_l as u8);
    raw.push(n_c as u8);

    // Centroids (L, C1, C2)
    for &c in &cl_u8 { raw.push(c as u8); }
    for &c in &cc1_u8 { raw.push(c as u8); }
    for &c in &cc2_u8 { raw.push(c as u8); }

    // 9. Per-stream rANS encoding — each data stream gets its native model.
    //
    // Format: AURA + header(raw) + paeth_L(rANS) + paeth_C1(rANS) + paeth_C2(rANS) + det_stream
    // Each rANS section: [u32 compressed_size][rANS bytes]
    // Detail bands: primitives(raw) + residuals encoded via rANS-band

    // Paeth residuals — rANS with Laplacian context model (per-stream)
    let paeth_l_enc = rans::rans_encode_paeth(&pred_l);
    let paeth_c1_enc = rans::rans_encode_paeth(&pred_c1);
    let paeth_c2_enc = rans::rans_encode_paeth(&pred_c2);

    raw.extend_from_slice(&(paeth_l_enc.len() as u32).to_le_bytes());
    raw.extend_from_slice(&paeth_l_enc);
    raw.extend_from_slice(&(paeth_c1_enc.len() as u32).to_le_bytes());
    raw.extend_from_slice(&paeth_c1_enc);
    raw.extend_from_slice(&(paeth_c2_enc.len() as u32).to_le_bytes());
    raw.extend_from_slice(&paeth_c2_enc);

    // Detail bands — already encoded in det_stream
    // The residual bands inside det_stream use encode_detail_band_v2 (sigmap format).
    // We rANS-compress each residual band's raw bytes for better compression.
    // For now, include det_stream as-is — the geometric primitives are raw,
    // and the sigmap residuals will be compressed per-band below.
    raw.extend_from_slice(&det_stream);

    // No LZMA wrapper — raw payload directly
    let mut aurea_data = Vec::with_capacity(4 + raw.len());
    aurea_data.extend_from_slice(AUREA_MAGIC);
    aurea_data.extend(raw);

    let compressed_size = aurea_data.len();

    Ok(AureaEncoderResult {
        aurea_data,
        compressed_size,
    })
}

/// Map quality to n_representatives.
pub fn quality_to_n_repr(quality: u8) -> usize {
    match quality {
        1..=15 => 16,
        16..=30 => 24,
        31..=50 => 32,
        51..=70 => 48,
        71..=85 => 64,
        86..=95 => 96,
        _ => 128,
    }
}

/// Helper: access a band by index in the (LH, HL, HH) tuple.
fn band_ref(subs: &(Array2<f64>, Array2<f64>, Array2<f64>), idx: usize) -> &Array2<f64> {
    match idx {
        0 => &subs.0,
        1 => &subs.1,
        2 => &subs.2,
        _ => panic!("band index out of range"),
    }
}
