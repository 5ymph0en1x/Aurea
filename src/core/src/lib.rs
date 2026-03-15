pub mod aurea;
pub mod aurea_encoder;
pub mod bitstream;
pub mod color;
pub mod dsp;
pub mod encoder;
pub mod fibonacci;
pub mod geometric;
pub mod golden;
pub mod paeth;
pub mod rans;
pub mod vq;
pub mod wavelet;
pub mod zeckendorf;

use bitstream::X267V6Stream;
use ndarray::Array2;

/// Decoded RGB image.
pub struct DecodedImage {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// Decode a complete AUREA file (raw bytes) and return the RGB image.
/// Supports both legacy LZMA format and new rANS format.
pub fn decode_aurea(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    if file_data.len() < 4 || &file_data[0..4] != aurea::AUREA_MAGIC {
        return Err("Not an AUREA file".into());
    }
    let after_magic = &file_data[4..];

    // Detect payload format by inspecting the first bytes:
    // - LZMA: starts with 0xFD 0x37 (legacy v2-v6 with LZMA wrapper)
    // - Raw v2-v7: first byte is a valid version number (1-7)
    // - rANS byte-level: anything else (v6 with rANS wrapper)
    let payload = if after_magic.len() >= 2 && after_magic[0] == 0xFD && after_magic[1] == 0x37 {
        // Legacy LZMA format
        bitstream::decompress_xts_payload(after_magic)?
    } else if !after_magic.is_empty() && after_magic[0] >= 1 && after_magic[0] <= 7 {
        // Raw payload: version byte directly present (v2+ with per-stream rANS)
        after_magic.to_vec()
    } else {
        // Try rANS byte-level decompression
        rans::rans_decompress_bytes(after_magic)
    };

    let v6 = aurea::parse_aurea_payload(&payload)?;
    decode_v6_from_parsed(&v6)
}

/// Reconstruct an image from a parsed AUREA v6 stream.
/// CDF 9/7 wavelets + VQ/fibonacci on LL.
/// Supports flags: zigzag, adaptive quant, inter-scale prediction.
fn decode_v6_from_parsed(v6: &X267V6Stream) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let hdr = &v6.header;
    let height = hdr.h;
    let width = hdr.w;
    let (ll_yh, ll_yw) = hdr.ll_y_size;
    let (ll_ch, ll_cw) = hdr.ll_c_size;
    let (ll_crh, ll_crw) = hdr.ll_cr_size;
    let flags = hdr.flags;
    let use_interscale = flags & wavelet::FLAG_INTERSCALE != 0;

    // --- 1. Reconstruct LL from labels (VQ + fibonacci + denormalize) ---
    let labels_y = paeth::paeth_unpredict_2d(&v6.pred_y, ll_yh, ll_yw);
    let labels_cb = paeth::paeth_unpredict_2d(&v6.pred_cb, ll_ch, ll_cw);
    let labels_cr = paeth::paeth_unpredict_2d(&v6.pred_cr, ll_crh, ll_crw);

    // VQ lookup
    let mut y_ll_flat = vec![0.0f64; ll_yh * ll_yw];
    for i in 0..y_ll_flat.len() {
        let k = (labels_y[i] as usize).min(v6.centroids_y.len() - 1);
        y_ll_flat[i] = v6.centroids_y[k];
    }
    let mut cb_ll_flat = vec![0.0f64; ll_ch * ll_cw];
    for i in 0..cb_ll_flat.len() {
        let k = (labels_cb[i] as usize).min(v6.centroids_cb.len() - 1);
        cb_ll_flat[i] = v6.centroids_cb[k];
    }
    let mut cr_ll_flat = vec![0.0f64; ll_crh * ll_crw];
    for i in 0..cr_ll_flat.len() {
        let k = (labels_cr[i] as usize).min(v6.centroids_cr.len() - 1);
        cr_ll_flat[i] = v6.centroids_cr[k];
    }

    // Fibonacci correction (parallel)
    let (y_ll_fib, (cb_ll_fib, cr_ll_fib)) = rayon::join(
        || fibonacci::fibonacci_correction(&y_ll_flat, &labels_y, &v6.centroids_y, ll_yh, ll_yw),
        || rayon::join(
            || fibonacci::fibonacci_correction(&cb_ll_flat, &labels_cb, &v6.centroids_cb, ll_ch, ll_cw),
            || fibonacci::fibonacci_correction(&cr_ll_flat, &labels_cr, &v6.centroids_cr, ll_crh, ll_crw),
        ),
    );

    // Denormalize LL
    let y_ll = denormalize_ll(&y_ll_fib, hdr.ll_ranges[0].0, hdr.ll_ranges[0].1);
    let cb_ll = denormalize_ll(&cb_ll_fib, hdr.ll_ranges[1].0, hdr.ll_ranges[1].1);
    let cr_ll = denormalize_ll(&cr_ll_fib, hdr.ll_ranges[2].0, hdr.ll_ranges[2].1);

    let y_ll_arr = Array2::from_shape_vec((ll_yh, ll_yw), y_ll)?;
    let cb_ll_arr = Array2::from_shape_vec((ll_ch, ll_cw), cb_ll)?;
    let cr_ll_arr = Array2::from_shape_vec((ll_crh, ll_crw), cr_ll)?;

    // --- 2. Decode detail bands ---
    let det = &v6.detail_data;
    let mut p = 0usize;

    // Pre-allocate slots (filled in reading order)
    let mut y_subs_opt: Vec<Option<(Array2<f64>, Array2<f64>, Array2<f64>)>> =
        (0..hdr.wv_levels).map(|_| None).collect();
    let mut cb_subs_opt: Vec<Option<(Array2<f64>, Array2<f64>, Array2<f64>)>> =
        (0..hdr.wv_levels).map(|_| None).collect();
    let mut cr_subs_opt: Vec<Option<(Array2<f64>, Array2<f64>, Array2<f64>)>> =
        (0..hdr.wv_levels).map(|_| None).collect();

    // Determine reading order: if interscale, deepest-first; otherwise, shallowest-first
    let level_order: Vec<usize> = if use_interscale || v6.band_size_prefixed {
        (0..hdr.wv_levels).rev().collect()
    } else {
        (0..hdr.wv_levels).collect()
    };

    if v6.geometric {
        // --- Geometric decode path (AUREA v6) ---
        // Order: level (deepest first) -> channel (L, C1, C2) -> primitives + 3 residual bands
        for &lv in &level_order {
            let (y_h, y_w) = hdr.y_sizes[lv];
            let (c_h, c_w) = hdr.cb_sizes[lv];
            let (cr_h, cr_w) = hdr.cr_sizes[lv];
            let y_bsizes = wavelet::detail_band_sizes(y_h, y_w);
            let c_bsizes = wavelet::detail_band_sizes(c_h, c_w);
            let cr_bsizes = wavelet::detail_band_sizes(cr_h, cr_w);

            let all_bsizes = [y_bsizes, c_bsizes, cr_bsizes];
            let all_steps = [hdr.steps_y[lv], hdr.steps_c[lv], hdr.steps_cr[lv]];
            let mut all_bands_out: [Vec<Array2<f64>>; 3] = [Vec::new(), Vec::new(), Vec::new()];

            for ch_idx in 0..3 {
                let bsizes = all_bsizes[ch_idx];
                let steps = all_steps[ch_idx];

                // 1. Read serialized primitives
                let (prims, prim_bytes) = geometric::deserialize_primitives(&det[p..]);
                p += prim_bytes;

                // 2. Project primitives into the 3 prediction bands
                let (h_lh, w_lh) = bsizes[0];
                let (h_hl, w_hl) = bsizes[1];
                let (h_hh, w_hh) = bsizes[2];
                let (pred_lh, pred_hl, pred_hh) = geometric::render_primitives(
                    &prims, h_lh, w_lh, h_hl, w_hl, h_hh, w_hh,
                );

                // 3. Read 3 residual bands (LH, HL, HH)
                let preds = [&pred_lh, &pred_hl, &pred_hh];
                for bi in 0..3 {
                    let (bh, bw) = bsizes[bi];
                    let step = steps[bi];
                    let n_coeffs = bh * bw;
                    let band_size = u32::from_le_bytes([
                        det[p], det[p + 1], det[p + 2], det[p + 3],
                    ]) as usize;
                    let data_start = p + 4;

                    let res_band = if v6.rans_bands {
                        // v7: rANS-band encoded residuals (quantized i16 in Morton order)
                        let (morton_coeffs, _) = rans::rans_decode_band(
                            &det[data_start..data_start + band_size], n_coeffs,
                        );
                        // Morton inverse -> raster order, then dequantize
                        let inv = wavelet::morton_order(bh, bw);
                        let mut flat = vec![0.0f64; n_coeffs];
                        for (morton_pos, &raster_pos) in inv.iter().enumerate() {
                            if morton_pos < morton_coeffs.len() {
                                flat[raster_pos] = morton_coeffs[morton_pos] as f64 * step;
                            }
                        }
                        // Apply spectral spin
                        wavelet::spectral_spin(&mut flat, bh, bw, step);
                        Array2::from_shape_vec((bh, bw), flat).unwrap()
                    } else {
                        // v6: sigmap v2 format
                        let (band, _) = wavelet::decode_detail_band_v2(
                            det, data_start, bh, bw, step, flags,
                        );
                        band
                    };
                    p = data_start + band_size;

                    // 4. Final band = residual (dequantized) + prediction * step
                    let pred = preds[bi];
                    let mut final_band = res_band;
                    let rh = bh.min(pred.nrows());
                    let rw = bw.min(pred.ncols());
                    for i in 0..rh {
                        for j in 0..rw {
                            final_band[[i, j]] += pred[[i, j]] * step;
                        }
                    }
                    all_bands_out[ch_idx].push(final_band);
                }
            }

            // Assign Y, Cb, Cr
            y_subs_opt[lv] = Some((
                all_bands_out[0].remove(0),
                all_bands_out[0].remove(0),
                all_bands_out[0].remove(0),
            ));
            cb_subs_opt[lv] = Some((
                all_bands_out[1].remove(0),
                all_bands_out[1].remove(0),
                all_bands_out[1].remove(0),
            ));
            cr_subs_opt[lv] = Some((
                all_bands_out[2].remove(0),
                all_bands_out[2].remove(0),
                all_bands_out[2].remove(0),
            ));
        }
    } else {
        // --- Legacy decode path (AUREA v2) ---
        for &lv in &level_order {
            let (y_h, y_w) = hdr.y_sizes[lv];
            let (c_h, c_w) = hdr.cb_sizes[lv];
            let (cr_h, cr_w) = hdr.cr_sizes[lv];
            let y_bsizes = wavelet::detail_band_sizes(y_h, y_w);
            let c_bsizes = wavelet::detail_band_sizes(c_h, c_w);
            let cr_bsizes = wavelet::detail_band_sizes(cr_h, cr_w);

            let mut y_bands = Vec::with_capacity(3);
            let mut cb_bands = Vec::with_capacity(3);
            let mut cr_bands = Vec::with_capacity(3);

            for bi in 0..3 {
                let (bh_y, bw_y) = y_bsizes[bi];
                let (bh_c, bw_c) = c_bsizes[bi];
                let (bh_cr, bw_cr) = cr_bsizes[bi];

                let (band_y, new_p) = decode_one_band(v6, det, p, bh_y, bw_y, hdr.steps_y[lv][bi], flags);
                p = new_p;
                let (band_cb, new_p) = decode_one_band(v6, det, p, bh_c, bw_c, hdr.steps_c[lv][bi], flags);
                p = new_p;
                let (band_cr, new_p) = decode_one_band(v6, det, p, bh_cr, bw_cr, hdr.steps_cr[lv][bi], flags);
                p = new_p;

                let band_y = if use_interscale && lv + 1 < hdr.wv_levels {
                    if let Some(ref deeper) = y_subs_opt[lv + 1] {
                        let deeper_b = band_ref(deeper, bi);
                        let pred = wavelet::upsample_band(deeper_b, bh_y, bw_y);
                        band_y + &pred
                    } else { band_y }
                } else { band_y };

                let band_cb = if use_interscale && lv + 1 < hdr.wv_levels {
                    if let Some(ref deeper) = cb_subs_opt[lv + 1] {
                        let deeper_b = band_ref(deeper, bi);
                        let pred = wavelet::upsample_band(deeper_b, bh_c, bw_c);
                        band_cb + &pred
                    } else { band_cb }
                } else { band_cb };

                let band_cr = if use_interscale && lv + 1 < hdr.wv_levels {
                    if let Some(ref deeper) = cr_subs_opt[lv + 1] {
                        let deeper_b = band_ref(deeper, bi);
                        let pred = wavelet::upsample_band(deeper_b, bh_cr, bw_cr);
                        band_cr + &pred
                    } else { band_cr }
                } else { band_cr };

                y_bands.push(band_y);
                cb_bands.push(band_cb);
                cr_bands.push(band_cr);
            }

            y_subs_opt[lv] = Some((y_bands.remove(0), y_bands.remove(0), y_bands.remove(0)));
            cb_subs_opt[lv] = Some((cb_bands.remove(0), cb_bands.remove(0), cb_bands.remove(0)));
            cr_subs_opt[lv] = Some((cr_bands.remove(0), cr_bands.remove(0), cr_bands.remove(0)));
        }
    }

    // Convert Option -> value (all levels are filled)
    let y_subs: Vec<_> = y_subs_opt.into_iter().map(|o| o.unwrap()).collect();
    let cb_subs: Vec<_> = cb_subs_opt.into_iter().map(|o| o.unwrap()).collect();
    let cr_subs: Vec<_> = cr_subs_opt.into_iter().map(|o| o.unwrap()).collect();

    // MERA inverse disentanglers (if present)
    let mut y_subs = y_subs;
    let mut cb_subs = cb_subs;
    let mut cr_subs = cr_subs;
    if let Some(ref mera) = v6.mera_angles {
        undo_post_disentanglers_intra(&mut y_subs, hdr.wv_levels, &mera[0]);
        undo_post_disentanglers_intra(&mut cb_subs, hdr.wv_levels, &mera[1]);
        undo_post_disentanglers_intra(&mut cr_subs, hdr.wv_levels, &mera[2]);
    }

    // --- 3. Inverse wavelet recomposition ---
    let y_sizes_vec: Vec<(usize, usize)> = hdr.y_sizes.clone();
    let cb_sizes_vec: Vec<(usize, usize)> = hdr.cb_sizes.clone();
    let cr_sizes_vec: Vec<(usize, usize)> = hdr.cr_sizes.clone();

    let y_recon_2d = wavelet::wavelet_recompose(&y_ll_arr, &y_subs, &y_sizes_vec);
    let cb_recon_2d = wavelet::wavelet_recompose(&cb_ll_arr, &cb_subs, &cb_sizes_vec);
    let cr_recon_2d = wavelet::wavelet_recompose(&cr_ll_arr, &cr_subs, &cr_sizes_vec);

    // --- 3b. Anti-ringing sigma filter (smooths halos near edges) ---
    let y_h = y_recon_2d.nrows();
    let y_w = y_recon_2d.ncols();
    let mut y_flat: Vec<f64> = y_recon_2d.iter().copied().collect();
    let step_y_base = hdr.steps_y[0].iter().copied().fold(f64::INFINITY, f64::min);
    dsp::anti_ring_sigma(&mut y_flat, y_h, y_w, step_y_base);

    let cb_h = cb_recon_2d.nrows();
    let cb_w = cb_recon_2d.ncols();
    let cr_h = cr_recon_2d.nrows();
    let cr_w = cr_recon_2d.ncols();
    let mut cb_flat: Vec<f64> = cb_recon_2d.iter().copied().collect();
    let mut cr_flat: Vec<f64> = cr_recon_2d.iter().copied().collect();
    let step_c_base = hdr.steps_c[0].iter().copied().fold(f64::INFINITY, f64::min);
    let step_cr_base = hdr.steps_cr[0].iter().copied().fold(f64::INFINITY, f64::min);
    dsp::anti_ring_sigma(&mut cb_flat, cb_h, cb_w, step_c_base);
    dsp::anti_ring_sigma(&mut cr_flat, cr_h, cr_w, step_cr_base);

    // --- 4. Final conversion to RGB ---
    let n = height * width;

    if v6.aurea_v2 {
        // GCT: upsample chromas C1/C2 (4:2:0) then inverse Golden Color Transform
        let c1_up = color::upsample_420(&cb_flat, cb_h, cb_w, height, width);
        let c2_up = color::upsample_420(&cr_flat, cr_h, cr_w, height, width);
        let (r_f, g_f, b_f) = color::golden_rotate_inverse(&y_flat, &c1_up, &c2_up, n);
        let mut rgb = vec![0u8; n * 3];
        for i in 0..n {
            rgb[i * 3]     = r_f[i].clamp(0.0, 255.0).round() as u8;
            rgb[i * 3 + 1] = g_f[i].clamp(0.0, 255.0).round() as u8;
            rgb[i * 3 + 2] = b_f[i].clamp(0.0, 255.0).round() as u8;
        }
        Ok(DecodedImage { rgb, width, height })
    } else {
        // YCbCr -> RGB with chroma upsample
        for v in y_flat.iter_mut() {
            *v = v.clamp(0.0, 255.0);
        }

        let cb_up = color::upsample_420(&cb_flat, cb_h, cb_w, height, width);
        let cr_up = color::upsample_420(&cr_flat, cb_h, cb_w, height, width);

        let cb_clipped: Vec<f64> = cb_up.iter().map(|v| v.clamp(0.0, 255.0)).collect();
        let cr_clipped: Vec<f64> = cr_up.iter().map(|v| v.clamp(0.0, 255.0)).collect();

        let rgb = color::ycbcr_to_rgb(&y_flat, &cb_clipped, &cr_clipped, n);

        Ok(DecodedImage { rgb, width, height })
    }
}

/// Decode a detail band with size prefix (u32 LE).
fn decode_band_with_size_prefix(
    det: &[u8], pos: usize, h: usize, w: usize, step: f64, flags: u8,
) -> (Array2<f64>, usize) {
    let band_size = u32::from_le_bytes([
        det[pos], det[pos + 1], det[pos + 2], det[pos + 3],
    ]) as usize;
    let data_start = pos + 4;
    let (band, _consumed_p) = wavelet::decode_detail_band_v2(
        det, data_start, h, w, step, flags,
    );
    (band, data_start + band_size)
}

/// Decode a detail band according to the stream format.
fn decode_one_band(
    v6: &X267V6Stream, det: &[u8], p: usize,
    h: usize, w: usize, step: f64, flags: u8,
) -> (Array2<f64>, usize) {
    if v6.band_size_prefixed {
        if v6.aurea_bands {
            aurea::decode_aurea_band_with_prefix(det, p, h, w, step)
        } else {
            decode_band_with_size_prefix(det, p, h, w, step, flags)
        }
    } else {
        wavelet::decode_detail_band_v2(det, p, h, w, step, flags)
    }
}

/// Access band bi from a (LH, HL, HH) tuple.
fn band_ref(subs: &(Array2<f64>, Array2<f64>, Array2<f64>), bi: usize) -> &Array2<f64> {
    match bi {
        0 => &subs.0,
        1 => &subs.1,
        _ => &subs.2,
    }
}

/// Inverse in-place Givens rotation between two subbands.
fn cross_subband_rotate_inv(a: &mut Array2<f64>, b: &mut Array2<f64>, theta: f64) {
    let c = theta.cos();
    let s = (-theta).sin(); // inverse rotation = -theta
    let c_inv = c; // cos(-theta) = cos(theta)
    let s_inv = s; // sin(-theta) = -sin(theta)
    let h = a.nrows().min(b.nrows());
    let w = a.ncols().min(b.ncols());
    for i in 0..h {
        for j in 0..w {
            let av = a[[i, j]];
            let bv = b[[i, j]];
            a[[i, j]] = c_inv * av - s_inv * bv;
            b[[i, j]] = s_inv * av + c_inv * bv;
        }
    }
}

/// Inverse of intra-scale MERA post-disentanglers.
fn undo_post_disentanglers_intra(
    subs: &mut Vec<(Array2<f64>, Array2<f64>, Array2<f64>)>,
    wv_levels: usize,
    angles: &[[f64; 6]],
) {
    for lv in (0..wv_levels).rev() {
        let (ref mut lh, ref mut hl, ref mut hh) = subs[lv];
        cross_subband_rotate_inv(hl, hh, angles[lv][5]);
        cross_subband_rotate_inv(lh, hh, angles[lv][4]);
        cross_subband_rotate_inv(lh, hl, angles[lv][3]);
    }
}

/// Denormalize an LL plane from [0,255] to [min, max].
fn denormalize_ll(data: &[f64], ll_min: f32, ll_max: f32) -> Vec<f64> {
    let mut range = ll_max as f64 - ll_min as f64;
    if range.abs() < 1e-6 {
        range = 1.0;
    }
    data.iter().map(|&v| v * range / 255.0 + ll_min as f64).collect()
}

/// Normalize an LL plane to [0, 255]. Returns (normalized, min, max).
pub fn normalize_ll(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let ll_min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let ll_max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut range = ll_max - ll_min;
    if range < 1e-6 {
        range = 1.0;
    }
    let norm: Vec<f64> = data.iter().map(|&v| (v - ll_min) * 255.0 / range).collect();
    (norm, ll_min, ll_max)
}
