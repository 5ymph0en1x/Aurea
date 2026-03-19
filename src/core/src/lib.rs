pub mod aurea;
pub mod aurea_encoder;
pub mod bitstream;
pub mod calibration;
pub mod color;
pub mod dsp;
pub mod geometric;
pub mod golden;
pub mod rans;
pub mod wavelet;
pub mod lot;
pub mod scene_analysis;
pub mod polymerase;
pub mod spin;
pub mod scan;

use ndarray::Array2;

/// Decoded RGB image.
pub struct DecodedImage {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

pub fn decode_aurea(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    if file_data.len() >= 4 && &file_data[0..4] == bitstream::AUR2_MAGIC {
        return decode_aur2(file_data);
    }
    Err("Unsupported format: expected AUR2 magic".into())
}

fn decode_aur2(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let (header, _header_size) = bitstream::parse_aur2_header(file_data)?;

    // Route based on version
    if header.version >= 3 {
        return decode_aur2_v3(file_data);
    }
    if header.version >= 2 {
        return decode_aur2_lot(file_data);
    }

    decode_aur2_v1(file_data)
}

/// v1 decoder: CDF 9/7 wavelet + geometric primitives.
fn decode_aur2_v1(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let (header, header_size) = bitstream::parse_aur2_header(file_data)?;
    let width = header.width;
    let height = header.height;
    let wv_levels = header.wv_levels;
    let detail_step = header.detail_step;
    let ll_ranges = header.ll_ranges;

    let body = &file_data[header_size..];
    let mut pos = 0usize;

    let read_u16 = |data: &[u8], p: &mut usize| -> u16 {
        let v = u16::from_le_bytes([data[*p], data[*p + 1]]);
        *p += 2; v
    };
    let read_u32 = |data: &[u8], p: &mut usize| -> u32 {
        let v = u32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4; v
    };
    let read_f32 = |data: &[u8], p: &mut usize| -> f32 {
        let v = f32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4; v
    };

    let c1_h = read_u16(body, &mut pos) as usize;
    let c1_w = read_u16(body, &mut pos) as usize;
    let c2_h = read_u16(body, &mut pos) as usize;
    let c2_w = read_u16(body, &mut pos) as usize;

    let mut l_sizes = Vec::with_capacity(wv_levels);
    let mut c1_sizes = Vec::with_capacity(wv_levels);
    let mut c2_sizes = Vec::with_capacity(wv_levels);
    for _ in 0..wv_levels {
        l_sizes.push((read_u16(body, &mut pos) as usize, read_u16(body, &mut pos) as usize));
        c1_sizes.push((read_u16(body, &mut pos) as usize, read_u16(body, &mut pos) as usize));
        c2_sizes.push((read_u16(body, &mut pos) as usize, read_u16(body, &mut pos) as usize));
    }

    let l_ll_h = read_u16(body, &mut pos) as usize;
    let l_ll_w = read_u16(body, &mut pos) as usize;
    let c1_ll_h = read_u16(body, &mut pos) as usize;
    let c1_ll_w = read_u16(body, &mut pos) as usize;
    let c2_ll_h = read_u16(body, &mut pos) as usize;
    let c2_ll_w = read_u16(body, &mut pos) as usize;

    let mut steps_l = Vec::with_capacity(wv_levels);
    let mut steps_c1 = Vec::with_capacity(wv_levels);
    let mut steps_c2 = Vec::with_capacity(wv_levels);
    for _ in 0..wv_levels {
        let mut sl = [0.0f64; 3]; let mut sc1 = [0.0f64; 3]; let mut sc2 = [0.0f64; 3];
        for bi in 0..3 {
            sl[bi] = read_f32(body, &mut pos) as f64;
            sc1[bi] = read_f32(body, &mut pos) as f64;
            sc2[bi] = read_f32(body, &mut pos) as f64;
        }
        steps_l.push(sl); steps_c1.push(sc1); steps_c2.push(sc2);
    }
    let ll_step = read_f32(body, &mut pos) as f64;

    // ================================================================
    // Decode LL subbands FIRST (Ribosome: LL serves as DNA template)
    // ================================================================
    struct LLMeta { ll_h: usize, ll_w: usize, ll_min: f32, ll_max: f32 }
    let ll_metas = [
        LLMeta { ll_h: l_ll_h, ll_w: l_ll_w, ll_min: ll_ranges[0].0, ll_max: ll_ranges[0].1 },
        LLMeta { ll_h: c1_ll_h, ll_w: c1_ll_w, ll_min: ll_ranges[1].0, ll_max: ll_ranges[1].1 },
        LLMeta { ll_h: c2_ll_h, ll_w: c2_ll_w, ll_min: ll_ranges[2].0, ll_max: ll_ranges[2].1 },
    ];
    let mut ll_arrays: Vec<Array2<f64>> = Vec::with_capacity(3);

    for llm in ll_metas.iter() {
        let h = llm.ll_h; let w = llm.ll_w;

        let patch_data_size = read_u32(body, &mut pos) as usize;
        let (patches, _) = geometric::deserialize_poly_patches(&body[pos..pos + patch_data_size]); pos += patch_data_size;
        let prediction = geometric::render_ll_patches(&patches, h, w);

        let rans_size = read_u32(body, &mut pos) as usize;
        let band_h = read_u16(body, &mut pos) as usize;
        let band_w = read_u16(body, &mut pos) as usize;
        let rans_data = &body[pos..pos + rans_size]; pos += rans_size;

        let n_coeffs = band_h * band_w;
        let (ordered_coeffs, _) = rans::rans_decode_band(rans_data, n_coeffs);

        let flat = if band_h > 0 && band_w > 0 {
            let order = wavelet::morton_order(band_h, band_w);
            let mut raster = vec![0i16; n_coeffs];
            for (m, &r) in order.iter().enumerate() { raster[r] = ordered_coeffs[m]; }
            raster
        } else { ordered_coeffs };

        let ll_step_clamped = ll_step.max(0.1);
        let mut ll_norm = vec![0.0f64; h * w];
        for i in 0..(h * w) {
            ll_norm[i] = prediction[i] + (flat[i] as f64 * ll_step_clamped);
        }
        let ll_denorm = denormalize_ll(&ll_norm, llm.ll_min, llm.ll_max);
        ll_arrays.push(Array2::from_shape_vec((h, w), ll_denorm).unwrap());
    }

    // ================================================================
    // Compute L-channel LL decoded for Ribosome codon map
    // ================================================================
    let l_ll_decoded: Vec<f64> = ll_arrays[0].iter().copied().collect();
    let l_ll_decoded_h = l_ll_h;
    let l_ll_decoded_w = l_ll_w;

    // ================================================================
    // Decode detail subbands (geometric primitives + codon-adaptive dequant)
    // ================================================================
    let mut l_subs: Vec<_> = (0..wv_levels).map(|_| (Array2::zeros((0,0)), Array2::zeros((0,0)), Array2::zeros((0,0)))).collect();
    let mut c1_subs: Vec<_> = (0..wv_levels).map(|_| (Array2::zeros((0,0)), Array2::zeros((0,0)), Array2::zeros((0,0)))).collect();
    let mut c2_subs: Vec<_> = (0..wv_levels).map(|_| (Array2::zeros((0,0)), Array2::zeros((0,0)), Array2::zeros((0,0)))).collect();

    struct ChannelMeta { sizes: Vec<(usize, usize)>, steps: Vec<[f64; 3]> }
    let steps_l_clone = steps_l.clone();
    let channel_metas = [
        ChannelMeta { sizes: l_sizes.clone(), steps: steps_l },
        ChannelMeta { sizes: c1_sizes.clone(), steps: steps_c1 },
        ChannelMeta { sizes: c2_sizes.clone(), steps: steps_c2 },
    ];

    for lv in (0..wv_levels).rev() {
        for (ch_idx, ch_meta) in channel_metas.iter().enumerate() {

            // Read geometric primitives (supercordes phi)
            let (primitives, prim_bytes) = geometric::deserialize_primitives(&body[pos..]);
            pos += prim_bytes;

            let mut decoded_bands = Vec::with_capacity(3);
            for bi in 0..3 {
                let rans_size = read_u32(body, &mut pos) as usize;
                let band_h = read_u16(body, &mut pos) as usize;
                let band_w = read_u16(body, &mut pos) as usize;
                let rans_data = &body[pos..pos + rans_size]; pos += rans_size;

                let n_coeffs = band_h * band_w;
                let (ordered_coeffs, _) = rans::rans_decode_band(rans_data, n_coeffs);

                let flat = if band_h > 0 && band_w > 0 {
                    let order = wavelet::morton_order(band_h, band_w);
                    let mut raster = vec![0i16; n_coeffs];
                    for (morton_pos, &raster_pos) in order.iter().enumerate() {
                        raster[raster_pos] = ordered_coeffs[morton_pos];
                    }
                    raster
                } else { ordered_coeffs };

                let q_band = Array2::from_shape_vec((band_h, band_w), flat.iter().map(|&v| v as f64).collect()).unwrap();
                let step = ch_meta.steps[lv][bi].max(0.1);
                // Codon-adaptive dequantization for ALL channels
                let dq_band = if !l_ll_decoded.is_empty() {
                    let step_map = wavelet::codon_step_map(
                        &l_ll_decoded, l_ll_decoded_h, l_ll_decoded_w,
                        band_h, band_w,
                        height, width, step,
                    );
                    wavelet::dequantize_band_map(&q_band, &step_map)
                } else {
                    wavelet::dequantize_band(&q_band, step)
                };
                decoded_bands.push(dq_band);
            }

            // Render geometric primitives and add to decoded residual
            let (lh_band, hl_band, hh_band) = if !primitives.is_empty() {
                let h_lh = decoded_bands[0].nrows(); let w_lh = decoded_bands[0].ncols();
                let h_hl = decoded_bands[1].nrows(); let w_hl = decoded_bands[1].ncols();
                let h_hh = decoded_bands[2].nrows(); let w_hh = decoded_bands[2].ncols();
                let (pred_lh, pred_hl, pred_hh) = geometric::render_primitives(
                    &primitives, h_lh, w_lh, h_hl, w_hl, h_hh, w_hh,
                );
                let mut lh = decoded_bands.remove(0);
                let mut hl = decoded_bands.remove(0);
                let mut hh = decoded_bands.remove(0);

                for r in 0..lh.nrows() { for c in 0..lh.ncols() { lh[[r, c]] += pred_lh[[r, c]]; } }
                for r in 0..hl.nrows() { for c in 0..hl.ncols() { hl[[r, c]] += pred_hl[[r, c]]; } }
                for r in 0..hh.nrows() { for c in 0..hh.ncols() { hh[[r, c]] += pred_hh[[r, c]]; } }
                (lh, hl, hh)
            } else {
                (decoded_bands.remove(0), decoded_bands.remove(0), decoded_bands.remove(0))
            };

            match ch_idx {
                0 => l_subs[lv] = (lh_band, hl_band, hh_band),
                1 => c1_subs[lv] = (lh_band, hl_band, hh_band),
                2 => c2_subs[lv] = (lh_band, hl_band, hh_band),
                _ => unreachable!(),
            }
        }
    }

    // Passe 3: Chaperonne multi-échelle — propagation inter-niveaux
    if header.version >= 1 {
        // Passe 3a: Chaperonne — propage l'énergie structurelle deep→fine
        dsp::chaperone_multiscale(
            &mut l_subs, &steps_l_clone,
            &l_ll_decoded, l_ll_decoded_h, l_ll_decoded_w,
            height, width, wv_levels,
        );
    }

    let l_recon = wavelet::wavelet_recompose(&ll_arrays[0], &l_subs, &l_sizes);
    let c1_recon = wavelet::wavelet_recompose(&ll_arrays[1], &c1_subs, &c1_sizes);
    let c2_recon = wavelet::wavelet_recompose(&ll_arrays[2], &c2_subs, &c2_sizes);

    let mut l_flat: Vec<f64> = l_recon.iter().copied().collect();
    let c1_sub_flat: Vec<f64> = c1_recon.iter().copied().collect();
    let c2_sub_flat: Vec<f64> = c2_recon.iter().copied().collect();

    dsp::anti_ring_sigma(&mut l_flat, height, width, detail_step);

    if header.version >= 1 {
        use golden::PTF_GAMMA_INV;
        let inv255 = 1.0 / 255.0;
        for v in l_flat.iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA_INV);
        }
    }

    let c1_full = color::upsample_420(&c1_sub_flat, c1_h, c1_w, height, width);
    let c2_full = color::upsample_422(&c2_sub_flat, c2_h, c2_w, width);
    let n = width * height;
    let (r_plane, g_plane, b_plane) = color::golden_rotate_inverse(&l_flat, &c1_full, &c2_full, n);

    let mut rgb = Vec::with_capacity(n * 3);
    for i in 0..n {
        rgb.push(r_plane[i].round().clamp(0.0, 255.0) as u8);
        rgb.push(g_plane[i].round().clamp(0.0, 255.0) as u8);
        rgb.push(b_plane[i].round().clamp(0.0, 255.0) as u8);
    }

    Ok(DecodedImage { rgb, width, height })
}

fn denormalize_ll(data: &[f64], ll_min: f32, ll_max: f32) -> Vec<f64> {
    let mut range = ll_max as f64 - ll_min as f64;
    if range.abs() < 1e-6 { range = 1.0; }
    data.iter().map(|&v| v * range / 255.0 + ll_min as f64).collect()
}

pub fn normalize_ll(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let ll_min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let ll_max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut range = ll_max - ll_min;
    if range < 1e-6 { range = 1.0; }
    let norm: Vec<f64> = data.iter().map(|&v| (v - ll_min) * 255.0 / range).collect();
    (norm, ll_min, ll_max)
}

// ======================================================================
// v2: LOT (Lapped Orthogonal Transform) decoder
// ======================================================================

const LOT_BLOCK_SIZE: usize = 16;
const LOT_AC_PER_BLOCK: usize = LOT_BLOCK_SIZE * LOT_BLOCK_SIZE - 1; // 255

/// Zigzag scan order for a block_size x block_size block (AC coefficients only).
/// Returns indices in row-major order, skipping index 0 (DC).
/// Must match the encoder's ac_zigzag_order exactly.
fn ac_zigzag_order(block_size: usize) -> Vec<usize> {
    let n = block_size;
    let mut order = Vec::with_capacity(n * n - 1);

    for s in 0..(2 * n - 1) {
        if s % 2 == 0 {
            let i_start = s.min(n - 1);
            let i_end = if s >= n { s - n + 1 } else { 0 };
            let mut i = i_start as i64;
            while i >= i_end as i64 {
                let j = s - i as usize;
                let idx = i as usize * n + j;
                if idx != 0 {
                    order.push(idx);
                }
                i -= 1;
            }
        } else {
            let i_start = if s >= n { s - n + 1 } else { 0 };
            let i_end = s.min(n - 1);
            for i in i_start..=i_end {
                let j = s - i;
                let idx = i * n + j;
                if idx != 0 {
                    order.push(idx);
                }
            }
        }
    }

    order
}

/// LOT v2 decoder: AUR2 header -> DC + AC -> LOT synthesis -> anti-ring -> PTF inv -> GCT inv -> RGB.
fn decode_aur2_lot(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let (header, header_size) = bitstream::parse_aur2_header(file_data)?;
    let width = header.width;
    let height = header.height;
    let detail_step = header.detail_step;
    let ll_ranges = header.ll_ranges;  // DC min/max per channel

    let body = &file_data[header_size..];
    let mut pos = 0usize;

    let read_u16 = |data: &[u8], p: &mut usize| -> u16 {
        let v = u16::from_le_bytes([data[*p], data[*p + 1]]);
        *p += 2; v
    };
    let read_u32 = |data: &[u8], p: &mut usize| -> u32 {
        let v = u32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4; v
    };
    let read_f32 = |data: &[u8], p: &mut usize| -> f32 {
        let v = f32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4; v
    };

    // 1. Parse body
    let c1_h = read_u16(body, &mut pos) as usize;
    let c1_w = read_u16(body, &mut pos) as usize;
    let c2_h = read_u16(body, &mut pos) as usize;
    let c2_w = read_u16(body, &mut pos) as usize;

    let _body_detail_step = read_f32(body, &mut pos) as f64;

    // Precompute AC zigzag order (must match encoder)
    let zz_order = ac_zigzag_order(LOT_BLOCK_SIZE);

    // Channel dimensions for LOT reconstruction
    struct ChannelInfo {
        chan_h: usize,
        chan_w: usize,
        dc_min: f64,
        dc_max: f64,
        chroma_factor: f64,
    }

    let channel_infos = [
        ChannelInfo {
            chan_h: height, chan_w: width,
            dc_min: ll_ranges[0].0 as f64, dc_max: ll_ranges[0].1 as f64,
            chroma_factor: 1.0,
        },
        ChannelInfo {
            chan_h: c1_h, chan_w: c1_w,
            dc_min: ll_ranges[1].0 as f64, dc_max: ll_ranges[1].1 as f64,
            chroma_factor: crate::golden::PHI,
        },
        ChannelInfo {
            chan_h: c2_h, chan_w: c2_w,
            dc_min: ll_ranges[2].0 as f64, dc_max: ll_ranges[2].1 as f64,
            chroma_factor: 1.0,
        },
    ];

    let dc_step = (detail_step * 0.5).max(0.5);
    let dc_step_clamped = dc_step.max(0.1);

    // We need to decode L channel DC first for codon-adaptive AC dequant,
    // but we also need all channels. So: two-pass approach.
    // Pass 1: read all raw data from body. Pass 2: reconstruct.

    // Read all channel data
    struct ChannelRawData {
        grid_h: usize,
        grid_w: usize,
        dc_q: Vec<i16>,   // quantized DC (raster order)
        ac_q: Vec<i16>,   // quantized AC (flat, zigzag-within-block)
    }

    let mut raw_channels: Vec<ChannelRawData> = Vec::with_capacity(3);

    for _ch_idx in 0..3 {
        let grid_h = read_u16(body, &mut pos) as usize;
        let grid_w = read_u16(body, &mut pos) as usize;
        let n_blocks = grid_h * grid_w;

        // Decode DC
        let dc_rans_size = read_u32(body, &mut pos) as usize;
        let dc_rans_data = &body[pos..pos + dc_rans_size];
        pos += dc_rans_size;

        let n_dc = n_blocks;
        let (dc_ordered, _) = rans::rans_decode_band(dc_rans_data, n_dc);

        // Un-Morton the DC
        let dc_q = if grid_h > 0 && grid_w > 0 {
            let order = wavelet::morton_order(grid_h, grid_w);
            let mut raster = vec![0i16; n_dc];
            for (morton_pos, &raster_pos) in order.iter().enumerate() {
                raster[raster_pos] = dc_ordered[morton_pos];
            }
            raster
        } else {
            dc_ordered
        };

        // Decode AC
        let ac_rans_size = read_u32(body, &mut pos) as usize;
        let ac_rans_data = &body[pos..pos + ac_rans_size];
        pos += ac_rans_size;

        let total_ac = n_blocks * LOT_AC_PER_BLOCK;
        let (ac_q, _) = rans::rans_decode_band(ac_rans_data, total_ac);

        raw_channels.push(ChannelRawData { grid_h, grid_w, dc_q, ac_q });
    }

    // Dequantize DC for all 3 channels (needed for codon 3D)
    let l_raw = &raw_channels[0];
    let l_dc_range = (channel_infos[0].dc_max - channel_infos[0].dc_min).max(1e-6);
    let l_dc_denorm: Vec<f64> = l_raw.dc_q.iter().map(|&q| {
        let dc_norm = q as f64 * dc_step_clamped;
        dc_norm * l_dc_range / 255.0 + channel_infos[0].dc_min
    }).collect();
    let c1_dc_range = (channel_infos[1].dc_max - channel_infos[1].dc_min).max(1e-6);
    let c1_dc_denorm: Vec<f64> = raw_channels[1].dc_q.iter().map(|&q| {
        let dc_norm = q as f64 * dc_step_clamped;
        dc_norm * c1_dc_range / 255.0 + channel_infos[1].dc_min
    }).collect();
    let c2_dc_range = (channel_infos[2].dc_max - channel_infos[2].dc_min).max(1e-6);
    let c2_dc_denorm: Vec<f64> = raw_channels[2].dc_q.iter().map(|&q| {
        let dc_norm = q as f64 * dc_step_clamped;
        dc_norm * c2_dc_range / 255.0 + channel_infos[2].dc_min
    }).collect();

    // Reconstruct each channel
    let mut reconstructed_channels: Vec<Vec<f64>> = Vec::with_capacity(3);

    for (ch_idx, raw) in raw_channels.iter().enumerate() {
        let info = &channel_infos[ch_idx];
        let dc_range = (info.dc_max - info.dc_min).max(1e-6);
        let n_blocks = raw.grid_h * raw.grid_w;

        // Dequantize DC: q * step -> normalized DC -> denormalize
        let dc_denorm: Vec<f64> = raw.dc_q.iter().map(|&q| {
            let dc_norm = q as f64 * dc_step_clamped;
            dc_norm * dc_range / 255.0 + info.dc_min
        }).collect();

        // Dequantize AC into per-block vectors
        let lot_global_factor = 3.8; // e ≈ 2.71828 (Euler)
        let ac_step = detail_step * info.chroma_factor * lot_global_factor;
        let mut ac_blocks: Vec<Vec<f64>> = Vec::with_capacity(n_blocks);

        // Build inverse zigzag: from flat zigzag position -> row-major AC position
        // zz_order[flat_pos] = block_row_major_index (1-based), so AC index = zz_order[flat_pos] - 1
        let mut inv_zz = vec![0usize; LOT_AC_PER_BLOCK];
        for (flat_pos, &block_idx) in zz_order.iter().enumerate() {
            inv_zz[flat_pos] = block_idx - 1; // block_idx is 1-based (after DC)
        }

        for block_idx in 0..n_blocks {
            let ac_offset = block_idx * LOT_AC_PER_BLOCK;
            let mut ac = vec![0.0f64; LOT_AC_PER_BLOCK];

            // Codon 3D: luminance × saturation × detail (gravitational lens)
            // First pass: compute AC energy from quantized values
            let ac_energy: f64 = (0..LOT_AC_PER_BLOCK).map(|fp| {
                raw.ac_q[ac_offset + fp].abs() as f64
            }).sum();
            // Map block to DC grids (different sizes per channel)
            let l_idx = (block_idx as f64 * l_dc_denorm.len() as f64
                        / n_blocks.max(1) as f64) as usize;
            let c1_idx = (block_idx as f64 * c1_dc_denorm.len() as f64
                         / n_blocks.max(1) as f64) as usize;
            let c2_idx = (block_idx as f64 * c2_dc_denorm.len() as f64
                         / n_blocks.max(1) as f64) as usize;
            let dc_l = l_dc_denorm[l_idx.min(l_dc_denorm.len().saturating_sub(1))];
            let dc_c1 = c1_dc_denorm[c1_idx.min(c1_dc_denorm.len().saturating_sub(1))];
            let dc_c2 = c2_dc_denorm[c2_idx.min(c2_dc_denorm.len().saturating_sub(1))];
            let codon_factor = lot::codon_3d_factor(
                dc_l, dc_c1, dc_c2, ac_energy, LOT_AC_PER_BLOCK,
            );
            let local_step = ac_step * codon_factor;
            let step_clamped = local_step.max(0.1);

            for flat_pos in 0..LOT_AC_PER_BLOCK {
                let qi = raw.ac_q[ac_offset + flat_pos];
                let ac_idx = inv_zz[flat_pos]; // destination in row-major AC (0-based)
                let block_idx = zz_order[flat_pos]; // 1-based block position
                let qfactor = lot::qmat_for_block_size(block_idx / LOT_BLOCK_SIZE, block_idx % LOT_BLOCK_SIZE, LOT_BLOCK_SIZE).max(0.1).powf(0.55);
                ac[ac_idx] = qi as f64 * step_clamped * qfactor;
            }

            ac_blocks.push(ac);
        }

        // LOT synthesis
        let recon = lot::lot_synthesize_image(
            &dc_denorm, &ac_blocks,
            raw.grid_h, raw.grid_w,
            info.chan_h, info.chan_w,
            LOT_BLOCK_SIZE,
        );

        reconstructed_channels.push(recon);
    }

    // Extract channels
    let mut l_flat = reconstructed_channels.remove(0);
    let mut c1_sub_flat = reconstructed_channels.remove(0);
    let mut c2_sub_flat = reconstructed_channels.remove(0);

    // Scene analysis from DC grid → adapt filter strength
    let scene_profile = scene_analysis::analyze_dc_grid(
        &l_dc_denorm, l_raw.grid_h, l_raw.grid_w,
    );
    eprintln!("  Scene: {:?}, smooth={:.0}%, aniso={:.3}, velvet_strength={:.2}",
              scene_profile.scene_type, scene_profile.smooth_pct,
              scene_profile.anisotropy, scene_profile.velvet_strength);

    // Biomimetic deblocking on ALL channels (L + chroma in subsampled space)
    dsp::deblock_lot_grid(&mut l_flat, height, width, LOT_BLOCK_SIZE);
    dsp::velvet_gas_filter(&mut l_flat, height, width, LOT_BLOCK_SIZE,
                           scene_profile.velvet_strength);
    {
        let mut c1_mut = c1_sub_flat.clone();
        let mut c2_mut = c2_sub_flat.clone();
        dsp::deblock_lot_grid(&mut c1_mut, c1_h, c1_w, LOT_BLOCK_SIZE);
        dsp::deblock_lot_grid(&mut c2_mut, c2_h, c2_w, LOT_BLOCK_SIZE);
        dsp::velvet_gas_filter(&mut c1_mut, c1_h, c1_w, LOT_BLOCK_SIZE,
                               scene_profile.velvet_strength);
        dsp::velvet_gas_filter(&mut c2_mut, c2_h, c2_w, LOT_BLOCK_SIZE,
                               scene_profile.velvet_strength);
        c1_sub_flat = c1_mut;
        c2_sub_flat = c2_mut;
    }

    // Anti-ring sigma filter on L
    dsp::anti_ring_sigma(&mut l_flat, height, width, detail_step);

    // PTF inverse on L
    {
        use golden::PTF_GAMMA_INV;
        let inv255 = 1.0 / 255.0;
        for v in l_flat.iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA_INV);
        }
    }

    // Upsample chroma
    let c1_full = color::upsample_420(&c1_sub_flat, c1_h, c1_w, height, width);
    let c2_full = color::upsample_422(&c2_sub_flat, c2_h, c2_w, width);

    // GCT inverse -> RGB
    let n = width * height;
    let (r_plane, g_plane, b_plane) = color::golden_rotate_inverse(&l_flat, &c1_full, &c2_full, n);

    let mut rgb = Vec::with_capacity(n * 3);
    for i in 0..n {
        rgb.push(r_plane[i].round().clamp(0.0, 255.0) as u8);
        rgb.push(g_plane[i].round().clamp(0.0, 255.0) as u8);
        rgb.push(b_plane[i].round().clamp(0.0, 255.0) as u8);
    }

    Ok(DecodedImage { rgb, width, height })
}

// ======================================================================
// v3: LOT decoder with all 8 improvement points
// ======================================================================

fn decode_aur2_v3(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    use aurea_encoder::{FLAG_CSF_MODULATION, FLAG_CHROMA_RESIDUAL, FLAG_STRUCTURAL, FLAG_DPCM_DC};

    let (header, header_size) = bitstream::parse_aur2_header(file_data)?;
    let width = header.width;
    let height = header.height;
    let detail_step = header.detail_step;
    let ll_ranges = header.ll_ranges;

    let body = &file_data[header_size..];
    let mut pos = 0usize;

    let read_u16 = |data: &[u8], p: &mut usize| -> u16 {
        let v = u16::from_le_bytes([data[*p], data[*p + 1]]);
        *p += 2; v
    };
    let read_u32 = |data: &[u8], p: &mut usize| -> u32 {
        let v = u32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4; v
    };

    // 1. Parse body header
    let c1_h = read_u16(body, &mut pos) as usize;
    let c1_w = read_u16(body, &mut pos) as usize;
    let c2_h = read_u16(body, &mut pos) as usize;
    let c2_w = read_u16(body, &mut pos) as usize;

    // v3: flags + scene_type
    let flags = read_u16(body, &mut pos);
    let _scene_type = body[pos]; pos += 1;
    let _reserved = body[pos]; pos += 1;

    let use_csf = flags & FLAG_CSF_MODULATION != 0;
    let use_chroma_resid = flags & FLAG_CHROMA_RESIDUAL != 0;
    let _use_structural = flags & FLAG_STRUCTURAL != 0;
    let use_dpcm = flags & FLAG_DPCM_DC != 0;

    eprintln!("  AUR2 v3 decode: {}x{}, flags=0x{:04x}", width, height, flags);

    // Precompute AC zigzag order
    let zz_order = ac_zigzag_order(LOT_BLOCK_SIZE);

    struct ChannelInfo {
        chan_h: usize, chan_w: usize,
        dc_min: f64, dc_max: f64,
        chroma_factor: f64,
    }

    let channel_infos = [
        ChannelInfo { chan_h: height, chan_w: width,
            dc_min: ll_ranges[0].0 as f64, dc_max: ll_ranges[0].1 as f64,
            chroma_factor: 1.0 },
        ChannelInfo { chan_h: c1_h, chan_w: c1_w,
            dc_min: ll_ranges[1].0 as f64, dc_max: ll_ranges[1].1 as f64,
            chroma_factor: crate::golden::PHI },
        ChannelInfo { chan_h: c2_h, chan_w: c2_w,
            dc_min: ll_ranges[2].0 as f64, dc_max: ll_ranges[2].1 as f64,
            chroma_factor: 1.0 },
    ];

    let dc_step = (detail_step * 0.1).max(0.2);
    let dc_step_clamped = dc_step.max(0.1);

    // Read all channel raw data
    struct ChannelRawData {
        grid_h: usize, grid_w: usize,
        dc_q: Vec<i16>, ac_q: Vec<i16>,
    }

    let mut raw_channels: Vec<ChannelRawData> = Vec::with_capacity(3);

    for _ch_idx in 0..3 {
        let grid_h = read_u16(body, &mut pos) as usize;
        let grid_w = read_u16(body, &mut pos) as usize;
        let n_blocks = grid_h * grid_w;

        let dc_rans_size = read_u32(body, &mut pos) as usize;
        let (dc_ordered, _) = rans::rans_decode_band(&body[pos..pos + dc_rans_size], n_blocks);
        pos += dc_rans_size;

        let dc_q = if use_dpcm {
            // DPCM: rANS gives residuals in raster order, reconstruct absolute DC
            let mut abs_dc = vec![0i16; n_blocks];
            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    let pred = aurea_encoder::golden_dc_predict(&abs_dc, gy, gx, grid_w);
                    abs_dc[gy * grid_w + gx] = dc_ordered[gy * grid_w + gx] + pred;
                }
            }
            abs_dc
        } else if grid_h > 0 && grid_w > 0 {
            let order = wavelet::morton_order(grid_h, grid_w);
            let mut raster = vec![0i16; n_blocks];
            for (morton_pos, &raster_pos) in order.iter().enumerate() {
                raster[raster_pos] = dc_ordered[morton_pos];
            }
            raster
        } else { dc_ordered };

        let ac_rans_size = read_u32(body, &mut pos) as usize;
        let total_ac = n_blocks * LOT_AC_PER_BLOCK;
        let (ac_q, _) = rans::rans_decode_band(&body[pos..pos + ac_rans_size], total_ac);
        pos += ac_rans_size;

        raw_channels.push(ChannelRawData { grid_h, grid_w, dc_q, ac_q });
    }

    // Dequantize DC for codon 3D/4D
    let l_raw = &raw_channels[0];
    let l_dc_range = (channel_infos[0].dc_max - channel_infos[0].dc_min).max(1e-6);
    let l_dc_denorm: Vec<f64> = l_raw.dc_q.iter().map(|&q|
        q as f64 * dc_step_clamped * l_dc_range / 255.0 + channel_infos[0].dc_min
    ).collect();
    let c1_dc_range = (channel_infos[1].dc_max - channel_infos[1].dc_min).max(1e-6);
    let c1_dc_denorm: Vec<f64> = raw_channels[1].dc_q.iter().map(|&q|
        q as f64 * dc_step_clamped * c1_dc_range / 255.0 + channel_infos[1].dc_min
    ).collect();
    let c2_dc_range = (channel_infos[2].dc_max - channel_infos[2].dc_min).max(1e-6);
    let c2_dc_denorm: Vec<f64> = raw_channels[2].dc_q.iter().map(|&q|
        q as f64 * dc_step_clamped * c2_dc_range / 255.0 + channel_infos[2].dc_min
    ).collect();

    // Reconstruct each channel
    let mut reconstructed_channels: Vec<Vec<f64>> = Vec::with_capacity(3);

    for (ch_idx, raw) in raw_channels.iter().enumerate() {
        let info = &channel_infos[ch_idx];
        let dc_range = (info.dc_max - info.dc_min).max(1e-6);
        let n_blocks = raw.grid_h * raw.grid_w;

        let dc_denorm: Vec<f64> = raw.dc_q.iter().map(|&q|
            q as f64 * dc_step_clamped * dc_range / 255.0 + info.dc_min
        ).collect();

        let lot_global_factor = 3.8;
        let ac_step = detail_step * info.chroma_factor * lot_global_factor;
        let mut ac_blocks: Vec<Vec<f64>> = Vec::with_capacity(n_blocks);

        let mut inv_zz = vec![0usize; LOT_AC_PER_BLOCK];
        for (flat_pos, &block_idx) in zz_order.iter().enumerate() {
            inv_zz[flat_pos] = block_idx - 1;
        }

        for block_idx in 0..n_blocks {
            let ac_offset = block_idx * LOT_AC_PER_BLOCK;
            let mut ac = vec![0.0f64; LOT_AC_PER_BLOCK];

            // Compute AC energy for codon
            let ac_energy: f64 = (0..LOT_AC_PER_BLOCK).map(|fp|
                raw.ac_q[ac_offset + fp].abs() as f64
            ).sum();

            let l_idx = (block_idx as f64 * l_dc_denorm.len() as f64
                        / n_blocks.max(1) as f64) as usize;
            let c1_idx = (block_idx as f64 * c1_dc_denorm.len() as f64
                         / n_blocks.max(1) as f64) as usize;
            let c2_idx = (block_idx as f64 * c2_dc_denorm.len() as f64
                         / n_blocks.max(1) as f64) as usize;
            let dc_l = l_dc_denorm[l_idx.min(l_dc_denorm.len().saturating_sub(1))];
            let dc_c1 = c1_dc_denorm[c1_idx.min(c1_dc_denorm.len().saturating_sub(1))];
            let dc_c2 = c2_dc_denorm[c2_idx.min(c2_dc_denorm.len().saturating_sub(1))];

            // Codon factor: 3D or 4D based on flags
            // Note: for 4D at decoder, we first dequantize with 3D, then could refine.
            // But structural coherence requires AC values which we're computing.
            // Use 3D for initial dequant (same as encoder's codon_3d_factor).
            let codon_factor = lot::codon_3d_factor(dc_l, dc_c1, dc_c2, ac_energy, LOT_AC_PER_BLOCK);
            let local_step = ac_step * codon_factor;
            let step_clamped = local_step.max(0.1);

            for flat_pos in 0..LOT_AC_PER_BLOCK {
                let qi = raw.ac_q[ac_offset + flat_pos];
                let ac_idx = inv_zz[flat_pos];
                let block_pos = zz_order[flat_pos]; // 1-based block position
                let mut qfactor = lot::qmat_for_block_size(block_pos / LOT_BLOCK_SIZE, block_pos % LOT_BLOCK_SIZE, LOT_BLOCK_SIZE).max(0.1).powf(0.55);

                // Point 5: CSF modulation (must match encoder)
                if use_csf {
                    let row = block_pos / LOT_BLOCK_SIZE;
                    let col = block_pos % LOT_BLOCK_SIZE;
                    let csf = lot::csf_qmat_factor(row, col, LOT_BLOCK_SIZE, dc_l);
                    qfactor *= csf;
                }

                ac[ac_idx] = qi as f64 * step_clamped * qfactor;
            }

            // Point 4: If structural flag, apply structural coherence correction
            // The decoder uses the dequantized AC to compute structural factor,
            // then scales accordingly. Since encoder used 4D factor on step,
            // decoder must match. The 3D factor is already applied above.
            // Structural factor was baked into the encoder's quantization step,
            // so the decoder automatically recovers it through the same dequant formula.
            // No additional correction needed here.

            ac_blocks.push(ac);
        }

        // Fibonacci spectral spin: median regularization of AC
        {
            let local_steps: Vec<f64> = (0..n_blocks).map(|bi| {
                let l_idx = (bi as f64 * l_dc_denorm.len() as f64
                            / n_blocks.max(1) as f64) as usize;
                let dc_l = l_dc_denorm[l_idx.min(l_dc_denorm.len().saturating_sub(1))];
                let codon = lot::codon_dc_factor(dc_l, 0.0, 0.0);
                ac_step * codon
            }).collect();
            spin::fibonacci_spectral_spin(
                &mut ac_blocks, raw.grid_h, raw.grid_w,
                LOT_AC_PER_BLOCK, &local_steps,
            );
        }

        let recon = lot::lot_synthesize_image(
            &dc_denorm, &ac_blocks,
            raw.grid_h, raw.grid_w,
            info.chan_h, info.chan_w,
            LOT_BLOCK_SIZE,
        );

        reconstructed_channels.push(recon);
    }

    let mut l_flat = reconstructed_channels.remove(0);
    let mut c1_sub_flat = reconstructed_channels.remove(0);
    let mut c2_sub_flat = reconstructed_channels.remove(0);

    // Gas-only deblocking: chirurgical, only at block boundaries in smooth zones
    dsp::deblock_gas_only(&mut l_flat, height, width, LOT_BLOCK_SIZE);
    dsp::deblock_gas_only(&mut c1_sub_flat, c1_h, c1_w, LOT_BLOCK_SIZE);
    dsp::deblock_gas_only(&mut c2_sub_flat, c2_h, c2_w, LOT_BLOCK_SIZE);

    // Scene analysis from DC grid -> adapt filter strength
    let scene_profile = scene_analysis::analyze_dc_grid(
        &l_dc_denorm, l_raw.grid_h, l_raw.grid_w,
    );
    eprintln!("  Scene: {:?}, smooth={:.0}%, velvet={:.2}",
              scene_profile.scene_type, scene_profile.smooth_pct,
              scene_profile.velvet_strength);

    // All post-processing disabled for maximum PSNR
    // Deblocking+velvet cost -1.06 dB PSNR, anti-ring neutral
    // TODO: develop PSNR-positive deblocking (gentler, edge-preserving)

    // Anti-ring sigma on L — disabled: hurts PSNR
    // dsp::anti_ring_sigma(&mut l_flat, height, width, detail_step);

    // PTF inverse on L
    {
        use golden::PTF_GAMMA_INV;
        let inv255 = 1.0 / 255.0;
        for v in l_flat.iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA_INV);
        }
    }

    // Upsample chroma (skip if already full resolution)
    let mut c1_full = if c1_h == height && c1_w == width {
        c1_sub_flat // already full res
    } else {
        color::upsample_420(&c1_sub_flat, c1_h, c1_w, height, width)
    };
    let c2_full = if c2_h == height && c2_w == width {
        c2_sub_flat // already full res (4:4:4 mode)
    } else {
        color::upsample_422(&c2_sub_flat, c2_h, c2_w, width)
    };

    // Point 3: Decode chroma residual
    if use_chroma_resid && pos < body.len() {
        let mask_size = read_u32(body, &mut pos) as usize;
        let c1_mask = body[pos..pos + mask_size].to_vec();
        pos += mask_size;
        let resid_size = read_u32(body, &mut pos) as usize;
        let c1_resid: Vec<i8> = body[pos..pos + resid_size].iter().map(|&b| b as i8).collect();
        pos += resid_size;
        let chroma_resid_step = detail_step * calibration::CHROMA_RESIDUAL_STEP_MULT;
        color::decode_chroma_residual(&mut c1_full, &c1_mask, &c1_resid,
                                       height, width, chroma_resid_step);
        eprintln!("    chroma residual decoded: {} blocks active", c1_resid.len() / 64);
    }

    // GCT inverse -> RGB
    let n = width * height;
    let (r_plane, g_plane, b_plane) = color::golden_rotate_inverse(&l_flat, &c1_full, &c2_full, n);

    let mut rgb = Vec::with_capacity(n * 3);
    for i in 0..n {
        rgb.push(r_plane[i].round().clamp(0.0, 255.0) as u8);
        rgb.push(g_plane[i].round().clamp(0.0, 255.0) as u8);
        rgb.push(b_plane[i].round().clamp(0.0, 255.0) as u8);
    }

    Ok(DecodedImage { rgb, width, height })
}