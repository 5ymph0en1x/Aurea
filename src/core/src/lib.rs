// v13 active modules
pub mod aurea_encoder;
pub mod bitstream;
pub mod calibration;
pub mod cfl;
pub mod codec_params;
pub mod color;
pub mod dsp;
pub mod error;
pub mod geometric;
pub mod golden;
pub mod hierarchy;
pub mod lot;
pub mod postprocess;
pub mod rans;
pub mod scan;
pub mod scene_analysis;
pub mod trellis;
pub mod turing;

// wavelet.rs kept for morton_order() and DEAD_ZONE (used by v13 encoder)
pub mod wavelet;

pub use error::AureaError;

/// Decoded RGB image.
pub struct DecodedImage {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// Apply CAS sharpening on the luminance plane (f64 [0,255]).
/// Converts to f32 [0,1] for postprocess::apply_cas_sharpening, then back.
fn sharpen_luminance(l_plane: &mut [f64], h: usize, w: usize) {
    let inv255 = 1.0 / 255.0;
    let l_f32: Vec<f32> = l_plane.iter().map(|&v| (v * inv255).clamp(0.0, 1.0) as f32).collect();
    let sharpened = postprocess::apply_cas_sharpening(&l_f32, w, h, 0.8);
    for (dst, &src) in l_plane.iter_mut().zip(sharpened.iter()) {
        *dst = (src as f64 * 255.0).clamp(0.0, 255.0);
    }
}

/// Apply CAS sharpening on the luminance plane (f64 [0,255]) stored in output_planes[0].
fn sharpen_luminance_planes(output_planes: &mut [Vec<f64>], h: usize, w: usize) {
    let inv255 = 1.0 / 255.0;
    let l_f32: Vec<f32> = output_planes[0].iter().map(|&v| (v * inv255).clamp(0.0, 1.0) as f32).collect();
    let sharpened = postprocess::apply_cas_sharpening(&l_f32, w, h, 0.8);
    for (dst, &src) in output_planes[0].iter_mut().zip(sharpened.iter()) {
        *dst = (src as f64 * 255.0).clamp(0.0, 255.0);
    }
}

pub fn decode_aurea(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    if file_data.len() < 4 || &file_data[0..4] != bitstream::AUR2_MAGIC {
        return Err("Unsupported format: expected AUR2 magic".into());
    }
    let (header, _) = bitstream::parse_aur2_header(file_data)?;
    if header.version == 12 {
        return decode_aur2_v12(file_data);
    }
    Err(format!("Unsupported AUR2 version {}. This build only supports v12/v13.", header.version).into())
}

pub fn normalize_ll(data: &[f64]) -> (Vec<f64>, f64, f64) {
    let ll_min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let ll_max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut range = ll_max - ll_min;
    if range < 1e-6 { range = 1.0; }
    let norm: Vec<f64> = data.iter().map(|&v| (v - ll_min) * 255.0 / range).collect();
    (norm, ll_min, ll_max)
}

const LOT_BLOCK_SIZE: usize = 16;

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

fn decode_aur2_v12(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    use aurea_encoder::{
        FLAG_CSF_MODULATION, FLAG_CHROMA_RESIDUAL, FLAG_DPCM_DC,
        FLAG_QUALITY_ADAPTIVE, FLAG_LOT_OVERLAP, FLAG_VARIABLE_BLOCKS,
        FLAG_BAYESIAN_HIERARCHY, FLAG_WEBER_TRNA, FLAG_CFL,
    };

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

    // 1. Parse body header: chroma dims (8 bytes)
    let c1_h = read_u16(body, &mut pos) as usize;
    let c1_w = read_u16(body, &mut pos) as usize;
    let c2_h = read_u16(body, &mut pos) as usize;
    let c2_w = read_u16(body, &mut pos) as usize;

    // Flags (u16) + scene_type (u8) + reserved (u8)
    let flags = read_u16(body, &mut pos);
    let _scene_type = body[pos]; pos += 1;
    let _reserved = body[pos]; pos += 1;

    let use_csf = flags & FLAG_CSF_MODULATION != 0;
    let use_chroma_resid = flags & FLAG_CHROMA_RESIDUAL != 0;
    let use_dpcm = flags & FLAG_DPCM_DC != 0;
    let use_quality_adaptive = flags & FLAG_QUALITY_ADAPTIVE != 0;
    let use_overlap = flags & FLAG_LOT_OVERLAP != 0;
    let use_variable = flags & FLAG_VARIABLE_BLOCKS != 0;
    let use_bayesian = flags & FLAG_BAYESIAN_HIERARCHY != 0;
    let _use_weber_trna = flags & FLAG_WEBER_TRNA != 0;
    let use_cfl = flags & FLAG_CFL != 0;

    // 2. Parse TuringHeader (16 bytes) when FLAG_BAYESIAN_HIERARCHY is set
    let turing_hdr = if use_bayesian {
        let (hdr, consumed) = bitstream::TuringHeader::parse(&body[pos..])?;
        pos += consumed;
        Some(hdr)
    } else {
        None
    };

    // 3. Read block map if variable blocks — rANS compressed
    let block_map = if use_variable {
        let bgh = read_u16(body, &mut pos) as usize;
        let bgw = read_u16(body, &mut pos) as usize;
        let map_len = read_u16(body, &mut pos) as usize;
        let n_cells = bgh * bgw;
        let map_ctx = vec![0u8; n_cells];
        let (map_symbols, _) = rans::rans_decode_band_v12(
            &body[pos..pos + map_len], n_cells, &map_ctx);
        pos += map_len;
        let size_grid: Vec<u8> = map_symbols.iter().map(|&s| match s {
            0 => 8u8, 1 => 16, 2 => 32, _ => 16,
        }).collect();
        Some((size_grid, bgh, bgw))
    } else {
        None
    };

    // Precompute block info for variable mode
    let var_blocks: Option<Vec<(usize, usize, usize)>> = block_map.as_ref().map(|(sg, bgh, bgw)| {
        lot::iter_blocks(sg, *bgh, *bgw, use_overlap)
    });

    let lot_global_factor = if use_quality_adaptive {
        calibration::lot_factor_for_quality(header.quality)
    } else {
        3.8
    };
    let qmat_power = if use_quality_adaptive {
        calibration::qmat_power_for_quality(header.quality)
    } else {
        0.55
    };

    eprintln!("  AUR2 v12 decode: {}x{}, flags=0x{:04x}, bayesian={}",
              width, height, flags, use_bayesian);

    // Precompute AC zigzag orders for each block size
    let zz_8 = ac_zigzag_order(8);
    let zz_16 = ac_zigzag_order(16);
    let zz_32 = ac_zigzag_order(32);

    struct V12ChannelInfo {
        chan_h: usize, chan_w: usize,
        dc_min: f64, dc_max: f64,
        chroma_factor: f64,
    }

    let channel_infos = [
        V12ChannelInfo { chan_h: height, chan_w: width,
            dc_min: ll_ranges[0].0 as f64, dc_max: ll_ranges[0].1 as f64,
            chroma_factor: 1.0 },
        V12ChannelInfo { chan_h: c1_h, chan_w: c1_w,
            dc_min: ll_ranges[1].0 as f64, dc_max: ll_ranges[1].1 as f64,
            chroma_factor: crate::golden::PHI },
        V12ChannelInfo { chan_h: c2_h, chan_w: c2_w,
            dc_min: ll_ranges[2].0 as f64, dc_max: ll_ranges[2].1 as f64,
            chroma_factor: 1.0 },
    ];

    let dc_step = (detail_step * 0.1).max(0.2);
    let dc_step_clamped = dc_step.max(0.1);

    // === SINGLE-PASS: Read DC+AC per channel, decode, dequantize, synthesize ===
    //
    // Channel 0 (L) is processed first to compute the TuringField needed for
    // AC decoding of all channels. The TuringField is derived from the L-channel
    // decoded DC, so it is available after reading ch0's DC.

    struct V12ChannelRaw {
        grid_h: usize, grid_w: usize,
        dc_q: Vec<i16>, ac_q: Vec<i16>,
    }

    let mut raw_channels: Vec<V12ChannelRaw> = Vec::with_capacity(3);
    let mut turing_field: Option<crate::turing::TuringField> = None;

    // CfL per-chroma-channel data: [ch1_flags, ch1_alphas], [ch2_flags, ch2_alphas]
    struct CflChannelData {
        flags: Vec<bool>,
        alpha_indices: Vec<usize>,
    }
    let mut cfl_data: Vec<CflChannelData> = Vec::new(); // index 0 = ch1, index 1 = ch2

    for ch_idx in 0..3usize {
        // v12: n_blocks stored as u32 (supports >65535 blocks for large images)
        let n_blocks = read_u32(body, &mut pos) as usize;
        let (grid_h, grid_w) = if let Some(ref vb) = var_blocks {
            (vb.len(), 1)
        } else {
            // Reconstruct grid dims from n_blocks for non-variable mode
            let gw = (width + LOT_BLOCK_SIZE - 1) / LOT_BLOCK_SIZE;
            let gh = if gw > 0 { n_blocks / gw } else { n_blocks };
            (gh, gw)
        };

        // DC via rANS v12 (Exp-Golomb)
        let dc_rans_size = read_u32(body, &mut pos) as usize;
        let dc_ctx = vec![0u8; n_blocks];
        let (dc_ordered, _) = rans::rans_decode_band_v12(&body[pos..pos + dc_rans_size], n_blocks, &dc_ctx);
        pos += dc_rans_size;

        // Golden DPCM reconstruction (same as v3)
        let dc_q = if use_dpcm {
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

        // v12: After decoding ch0 DC, compute TuringField for all channels
        if ch_idx == 0 {
            if let Some(ref hdr) = turing_hdr {
                let l_info = &channel_infos[0];
                let l_dc_range = (l_info.dc_max - l_info.dc_min).max(1e-6);
                let l_dc_denorm: Vec<f64> = dc_q.iter().map(|&q|
                    q as f64 * dc_step_clamped * l_dc_range / 255.0 + l_info.dc_min
                ).collect();
                let params = hierarchy::HierarchyParams::from(hdr);
                turing_field = Some(hierarchy::compute_level1(&l_dc_denorm, grid_h, grid_w, &params));
            }
        }

        // v12: Parse match section for L channel only (ch_idx == 0).
        if ch_idx == 0 {
            let match_section_len = read_u32(body, &mut pos) as usize;
            pos += match_section_len;
        }

        // Read CfL metadata for chroma channels (ch_idx == 1, 2) — rANS compressed
        if use_cfl && ch_idx > 0 {
            let cfl_n_symbols = read_u32(body, &mut pos) as usize;
            let cfl_rans_size = read_u32(body, &mut pos) as usize;
            let cfl_ctx = vec![0u8; cfl_n_symbols];
            let (cfl_symbols, _) = rans::rans_decode_band_v12(
                &body[pos..pos + cfl_rans_size], cfl_n_symbols, &cfl_ctx);
            pos += cfl_rans_size;

            let mut cfl_flags = vec![false; n_blocks];
            for i in 0..n_blocks {
                if i < cfl_symbols.len() {
                    cfl_flags[i] = cfl_symbols[i] != 0;
                }
            }
            let n_active: usize = cfl_flags.iter().filter(|&&f| f).count();
            let mut alpha_indices = vec![3usize; n_blocks];
            let mut alpha_cursor = 0usize;
            for block_idx in 0..n_blocks {
                if cfl_flags[block_idx] {
                    let sym_idx = n_blocks + alpha_cursor;
                    if sym_idx < cfl_symbols.len() {
                        alpha_indices[block_idx] = (cfl_symbols[sym_idx] & 0x07) as usize;
                    }
                    alpha_cursor += 1;
                }
            }

            cfl_data.push(CflChannelData { flags: cfl_flags, alpha_indices });
        }

        // Read EOB positions: DPCM + rANS compressed (Turing-contextualized)
        let eob_count = read_u32(body, &mut pos) as usize;
        let eob_rans_size = read_u32(body, &mut pos) as usize;
        let eob_data = &body[pos..pos + eob_rans_size];
        pos += eob_rans_size;
        let mut eob_positions: Vec<u16> = Vec::with_capacity(eob_count);
        {
            // v13: per-block Turing buckets for EOB context
            let eob_turing: Vec<u8> = if let Some(ref tf) = turing_field {
                let buckets = hierarchy::build_turing_buckets(tf, grid_h, grid_w, LOT_BLOCK_SIZE);
                buckets.iter().map(|&b| b as u8).collect()
            } else {
                vec![0u8; eob_count]
            };
            let (eob_deltas, _) = rans::rans_decode_band_v12(eob_data, eob_count, &eob_turing);
            let mut prev_eob: i16 = 0;
            for &delta in &eob_deltas {
                prev_eob += delta;
                eob_positions.push(prev_eob.max(0) as u16);
            }
        }

        // Read AC rANS data
        let ac_rans_size = read_u32(body, &mut pos) as usize;
        let ac_data = &body[pos..pos + ac_rans_size];
        pos += ac_rans_size;

        // Compute total_ac from eob_positions (sum of per-block truncated lengths)
        let total_ac: usize = eob_positions.iter().map(|&e| e as usize).sum();

        // v13: per-block Turing buckets for AC context (repeated per-coefficient)
        let ac_turing_vec: Vec<u8> = if let Some(ref tf) = turing_field {
            let buckets = hierarchy::build_turing_buckets(tf, grid_h, grid_w, LOT_BLOCK_SIZE);
            let mut v = Vec::with_capacity(total_ac);
            for (bi, &eob) in eob_positions.iter().enumerate() {
                let b = if bi < buckets.len() { buckets[bi] as u8 } else { 0 };
                for _ in 0..eob as usize {
                    v.push(b);
                }
            }
            v
        } else {
            vec![0u8; total_ac]
        };

        // Decode via rANS v12 (or v11 fallback if no turing field)
        let (ac_truncated, _) = if turing_field.is_some() {
            rans::rans_decode_band_v12(ac_data, total_ac, &ac_turing_vec)
        } else {
            rans::rans_decode_band_v11(ac_data, total_ac)
        };

        // Scatter into full blocks using eob_positions (pad EOB tail with zeros)
        let mut ac_q: Vec<i16> = Vec::new();
        let mut ac_cursor = 0usize;

        for block_idx in 0..n_blocks {
            let block_size = if let Some(ref blocks) = var_blocks {
                if block_idx < blocks.len() { blocks[block_idx].2 } else { LOT_BLOCK_SIZE }
            } else {
                LOT_BLOCK_SIZE
            };
            let ac_per_block = block_size * block_size - 1;
            let eob = if block_idx < eob_positions.len() {
                eob_positions[block_idx] as usize
            } else { 0 };

            // Copy truncated portion
            for _ in 0..eob {
                if ac_cursor < ac_truncated.len() {
                    ac_q.push(ac_truncated[ac_cursor]);
                    ac_cursor += 1;
                } else {
                    ac_q.push(0);
                }
            }
            // Fill EOB tail with zeros
            for _ in eob..ac_per_block {
                ac_q.push(0);
            }
        }

        raw_channels.push(V12ChannelRaw { grid_h, grid_w, dc_q, ac_q });
    }

    // === Dequantize L channel DC for TuringField and foveal map ===
    let l_raw = &raw_channels[0];
    let l_dc_range = (channel_infos[0].dc_max - channel_infos[0].dc_min).max(1e-6);
    let l_dc_denorm: Vec<f64> = l_raw.dc_q.iter().map(|&q|
        q as f64 * dc_step_clamped * l_dc_range / 255.0 + channel_infos[0].dc_min
    ).collect();

    let bs2 = (LOT_BLOCK_SIZE * LOT_BLOCK_SIZE) as f64;
    let l_dc_pixel: Vec<f64> = l_dc_denorm.iter().map(|&v| v / bs2).collect();

    // Foveal saliency map (computed from decoded DC, same as encoder)
    let l_foveal_map = lot::foveal_saliency_map(&l_dc_denorm, l_raw.grid_h, l_raw.grid_w);

    eprintln!("  v12 Turing decode: field {}x{}, modulation active={}",
              l_raw.grid_h, l_raw.grid_w, turing_field.is_some());

    // === Dequantize AC with Turing step modulation, then synthesize ===
    // Two-pass for CfL: dequantize luma first, then chroma with CfL reconstruction.
    let mut reconstructed_channels: Vec<Vec<f64>> = Vec::with_capacity(3);

    // Shared dequantization logic for one channel, returning (ac_blocks, l_rec_blocks_opt).
    // When `collect_l_rec` is true, also returns per-block dequantized AC in natural order.
    let dequantize_channel = |
        raw: &V12ChannelRaw,
        info: &V12ChannelInfo,
        turing_field: &Option<crate::turing::TuringField>,
        collect_l_rec: bool,
    | -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let _dc_range = (info.dc_max - info.dc_min).max(1e-6);
        let n_blocks = raw.grid_h * raw.grid_w;

        let ac_step = detail_step * info.chroma_factor * lot_global_factor;
        let mut ac_blocks: Vec<Vec<f64>> = Vec::with_capacity(n_blocks);
        let mut l_rec_blocks: Vec<Vec<f64>> = if collect_l_rec {
            Vec::with_capacity(n_blocks)
        } else {
            Vec::new()
        };
        let mut ac_cursor = 0usize;

        // Directional QMAT: mirror encoder's gradient-based rotation
        let gradient_angles: Vec<(f64, f64)> = if let Some(tf) = turing_field.as_ref() {
            hierarchy::build_gradient_angles(tf, raw.grid_h, raw.grid_w, LOT_BLOCK_SIZE)
        } else {
            vec![(0.0, 0.0); n_blocks]
        };
        let grad_threshold = *calibration::TUNABLE_GRADIENT_THRESHOLD;

        for block_idx in 0..n_blocks {
            let block_size = if let Some(ref blocks) = var_blocks {
                if block_idx < blocks.len() { blocks[block_idx].2 } else { LOT_BLOCK_SIZE }
            } else {
                LOT_BLOCK_SIZE
            };
            let ac_per_block = block_size * block_size - 1;

            let zz = match block_size {
                8 => &zz_8,
                32 => &zz_32,
                _ => &zz_16,
            };
            let mut inv_zz = vec![0usize; ac_per_block];
            for (fp, &bi) in zz.iter().enumerate() {
                if fp < ac_per_block { inv_zz[fp] = bi - 1; }
            }

            let mut ac = vec![0.0f64; ac_per_block];

            let l_fov_idx = (block_idx as f64 * l_foveal_map.len() as f64
                            / n_blocks.max(1) as f64) as usize;
            let foveal_factor = l_foveal_map[l_fov_idx.min(l_foveal_map.len().saturating_sub(1))];

            let l_idx = (block_idx as f64 * l_dc_pixel.len() as f64
                        / n_blocks.max(1) as f64) as usize;
            let dc_l = l_dc_pixel[l_idx.min(l_dc_pixel.len().saturating_sub(1))];

            let raw_turing_mod = if let Some(tf) = turing_field.as_ref() {
                if block_idx < tf.step_modulation.len() {
                    tf.step_modulation[block_idx]
                } else {
                    1.0
                }
            } else {
                1.0
            };

            let turing_mod = crate::aurea_encoder::psychovisual_turing_pivot(raw_turing_mod, header.quality);
            let local_step = ac_step * foveal_factor * turing_mod;
            let step_clamped = local_step.max(0.1);

            // Select rotated or standard QMAT (must match encoder exactly)
            let (blk_angle, blk_strength) = if block_idx < gradient_angles.len() {
                gradient_angles[block_idx]
            } else {
                (0.0, 0.0)
            };
            let use_rotated = blk_strength > grad_threshold;
            // Quantize angle to same 8 steps as encoder for bit-exact match
            let n_angle_steps = 8usize;
            let angle_step = std::f64::consts::PI / n_angle_steps as f64;
            let quantized_angle = if use_rotated {
                let a = blk_angle.rem_euclid(std::f64::consts::PI);
                let idx = ((a / angle_step + 0.5) as usize) % n_angle_steps;
                idx as f64 * angle_step
            } else {
                0.0
            };

            for flat_pos in 0..ac_per_block {
                let qi = if ac_cursor + flat_pos < raw.ac_q.len() {
                    raw.ac_q[ac_cursor + flat_pos]
                } else { 0 };
                let ac_idx = inv_zz[flat_pos];
                let block_pos = zz[flat_pos];
                let row = block_pos / block_size;
                let col = block_pos % block_size;
                let mut qfactor = if use_rotated {
                    lot::qmat_rotated_for_block_size(row, col, block_size, quantized_angle)
                        .powf(qmat_power)
                } else {
                    lot::qmat_for_block_size(row, col, block_size).powf(qmat_power)
                };

                if use_csf {
                    let csf = lot::csf_qmat_factor(row, col, block_size, dc_l);
                    qfactor *= csf;
                }

                ac[ac_idx] = qi as f64 * step_clamped * qfactor;
            }
            ac_cursor += ac_per_block;

            if collect_l_rec {
                l_rec_blocks.push(ac.clone());
            }
            ac_blocks.push(ac);
        }

        (ac_blocks, l_rec_blocks)
    };

    // --- Pass 1: Dequantize and synthesize luma (ch_idx=0) ---
    let l_rec_ac_blocks: Vec<Vec<f64>>;
    {
        let raw = &raw_channels[0];
        let info = &channel_infos[0];
        let dc_range = (info.dc_max - info.dc_min).max(1e-6);

        let dc_denorm: Vec<f64> = raw.dc_q.iter().map(|&q|
            q as f64 * dc_step_clamped * dc_range / 255.0 + info.dc_min
        ).collect();

        let (ac_blocks, l_rec) = dequantize_channel(raw, info, &turing_field, use_cfl);
        l_rec_ac_blocks = l_rec;

        let recon = if let Some(ref blocks) = var_blocks {
            lot::lot_synthesize_variable(
                &dc_denorm, &ac_blocks, blocks,
                info.chan_h, info.chan_w, use_overlap,
            )
        } else {
            lot::lot_synthesize_image(
                &dc_denorm, &ac_blocks,
                raw.grid_h, raw.grid_w,
                info.chan_h, info.chan_w,
                LOT_BLOCK_SIZE, use_overlap,
            )
        };
        reconstructed_channels.push(recon);
    }

    // --- Pass 2: Dequantize and synthesize chroma (ch_idx=1,2) with CfL ---
    for ch_idx in 1..3usize {
        let raw = &raw_channels[ch_idx];
        let info = &channel_infos[ch_idx];
        let dc_range = (info.dc_max - info.dc_min).max(1e-6);
        let n_blocks = raw.grid_h * raw.grid_w;

        let dc_denorm: Vec<f64> = raw.dc_q.iter().map(|&q|
            q as f64 * dc_step_clamped * dc_range / 255.0 + info.dc_min
        ).collect();

        let (mut ac_blocks, _) = dequantize_channel(raw, info, &turing_field, false);

        // CfL reconstruction: add back alpha * L_rec for active blocks
        if use_cfl && !l_rec_ac_blocks.is_empty() {
            let cfl_idx = ch_idx - 1; // cfl_data[0] = ch1, cfl_data[1] = ch2
            if cfl_idx < cfl_data.len() {
                let cfd = &cfl_data[cfl_idx];
                for block_idx in 0..n_blocks {
                    if cfd.flags[block_idx] && block_idx < l_rec_ac_blocks.len() {
                        let l_rec_block = &l_rec_ac_blocks[block_idx];
                        let residual = &ac_blocks[block_idx];
                        let len = l_rec_block.len().min(residual.len());
                        let reconstructed = cfl::reconstruct(
                            &l_rec_block[..len], &residual[..len], cfd.alpha_indices[block_idx],
                        );
                        ac_blocks[block_idx] = reconstructed;
                    }
                }
            }
        }

        let recon = if let Some(ref blocks) = var_blocks {
            lot::lot_synthesize_variable(
                &dc_denorm, &ac_blocks, blocks,
                info.chan_h, info.chan_w, use_overlap,
            )
        } else {
            lot::lot_synthesize_image(
                &dc_denorm, &ac_blocks,
                raw.grid_h, raw.grid_w,
                info.chan_h, info.chan_w,
                LOT_BLOCK_SIZE, use_overlap,
            )
        };
        reconstructed_channels.push(recon);
    }

    let mut l_flat = reconstructed_channels.remove(0);
    let mut c1_sub_flat = reconstructed_channels.remove(0);
    let mut c2_sub_flat = reconstructed_channels.remove(0);

    // v12: gas-only deblocking (smooth-zone block boundaries only)
    dsp::deblock_gas_only(&mut l_flat, height, width, LOT_BLOCK_SIZE);
    dsp::deblock_gas_only(&mut c1_sub_flat, c1_h, c1_w, LOT_BLOCK_SIZE);
    dsp::deblock_gas_only(&mut c2_sub_flat, c2_h, c2_w, LOT_BLOCK_SIZE);

    // v12: anti-ring sigma filter — cleans mixed-block artifacts near edges.
    // Sky pixels in blocks straddling tree/sky boundary get pulled toward correct
    // sky values by excluding dark tree pixels from the local sigma average.
    dsp::anti_ring_sigma(&mut l_flat, height, width, detail_step);

    // PTF inverse on L channel
    {
        use golden::PTF_GAMMA_INV;
        let inv255 = 1.0 / 255.0;
        for v in l_flat.iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA_INV);
        }
    }

    // Upsample chroma (skip if already full resolution — v12 uses 4:4:4)
    let mut c1_full = if c1_h == height && c1_w == width {
        c1_sub_flat
    } else {
        color::upsample_420(&c1_sub_flat, c1_h, c1_w, height, width)
    };
    let c2_full = if c2_h == height && c2_w == width {
        c2_sub_flat
    } else {
        color::upsample_422(&c2_sub_flat, c2_h, c2_w, width)
    };

    // Decode chroma residual if present
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
    }

    let _ = pos; // suppress unused-variable warning

    // GCT inverse -> RGB (no CAS sharpening for v12)
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
