pub mod aurea_encoder;
pub mod bitstream;
pub mod calibration;
pub mod color;
pub mod dsp;
pub mod error;
pub mod geometric;
pub mod golden;
pub mod hex;
pub mod hex_decoder;
pub mod hex_edge;
pub mod hex_encoder;
pub mod rans;
pub mod wavelet;
pub mod lot;
pub mod scene_analysis;
pub mod polymerase;
pub mod spin;
pub mod scan;
pub mod codec_params;
pub mod photon;
pub mod capillary;
pub mod turing;
pub mod hierarchy;
pub mod postprocess;
pub mod cfl;

pub use error::AureaError;

use ndarray::Array2;

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
    if file_data.len() >= 4 && &file_data[0..4] == bitstream::AUR2_MAGIC {
        return decode_aur2(file_data);
    }
    Err("Unsupported format: expected AUR2 magic".into())
}

fn decode_aur2(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let (header, _header_size) = bitstream::parse_aur2_header(file_data)?;

    // Route based on version
    if header.version == 11 {
        return decode_optica(file_data);
    }
    if header.version == 8 {
        return decode_edge_energy(file_data);
    }
    if header.version == 7 {
        return decode_aur2_hex_pyramid(file_data);
    }
    if header.version == 6 {
        return decode_aur2_hex_multiscale(file_data);
    }
    if header.version == 5 {
        return decode_aur2_hex(file_data);
    }
    if header.version == 12 {
        return decode_aur2_v12(file_data);
    }
    if header.version >= 3 {
        return decode_aur2_v3(file_data);
    }
    if header.version >= 2 {
        return decode_aur2_lot(file_data);
    }

    decode_aur2_v1(file_data)
}

/// Optica decoder (AUR2 version 11).
///
/// Pipeline: parse header -> rANS v11 decode path-DPCM (L, C1, C2) ->
/// inverse PTF on L -> inverse GCT -> RGB -> Schrodinger collapse.
fn decode_optica(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    let (header, header_size) = bitstream::parse_aur2_header(file_data)?;
    let width = header.width;
    let height = header.height;
    let n_pixels = width * height;

    let body = &file_data[header_size..];
    let mut pos = 0usize;

    let read_f32_le = |data: &[u8], p: &mut usize| -> f32 {
        let v = f32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4;
        v
    };
    let read_u32 = |data: &[u8], p: &mut usize| -> u32 {
        let v = u32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4;
        v
    };
    let read_u16_le = |data: &[u8], p: &mut usize| -> u16 {
        let v = u16::from_le_bytes([data[*p], data[*p + 1]]);
        *p += 2; v
    };

    // 1. Header: reference values + step sizes + mode
    let _ref_l = read_f32_le(body, &mut pos) as f64;
    let _ref_c1 = read_f32_le(body, &mut pos) as f64;
    let _ref_c2 = read_f32_le(body, &mut pos) as f64;
    let edge_step_l = read_f32_le(body, &mut pos) as f64;
    let edge_step_c = read_f32_le(body, &mut pos) as f64;
    let _mode = body[pos]; pos += 1; // mode=11

    // Residual step sizes (finer, for independent quantization)
    let res_step_l = read_f32_le(body, &mut pos) as f64;
    let res_step_c = read_f32_le(body, &mut pos) as f64;

    let dc_steps = [edge_step_l, edge_step_c, edge_step_c];
    let res_steps = [res_step_l, res_step_c, res_step_c];

    // 2. Hex grid dimensions
    let hex_cols = read_u16_le(body, &mut pos) as usize;
    let hex_rows = read_u16_le(body, &mut pos) as usize;
    let n_hexes = read_u32(body, &mut pos) as usize;

    // 4. Decode hex DC grids
    let grid = hex::HexGrid { cols: hex_cols, rows: hex_rows, img_w: width, img_h: height };
    let shape = hex::compute_hex_shape();

    let mut recon_dcs_all: Vec<Vec<f64>> = Vec::with_capacity(3);
    for ch in 0..3usize {
        let stream_size = read_u32(body, &mut pos) as usize;
        let (dc_deltas, _) = rans::rans_decode_band_v11(&body[pos..pos + stream_size], n_hexes);
        pos += stream_size;

        // Golden 2D predictor: same as encoder (φ-weighted left/top/diag)
        let fine_step = 1.0_f64;
        let cols = hex_cols;
        let rows = hex_rows;
        let mut recon_dcs = vec![0.0f64; n_hexes];

        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let pred = if row == 0 && col == 0 {
                    0.0
                } else if row == 0 {
                    recon_dcs[idx - 1]
                } else if col == 0 {
                    recon_dcs[(row - 1) * cols + col]
                } else {
                    let left = recon_dcs[idx - 1];
                    let top = recon_dcs[(row - 1) * cols + col];
                    let diag = recon_dcs[(row - 1) * cols + (col - 1)];
                    let w_sum = golden::PHI_INV + golden::PHI_INV2 + golden::PHI_INV3;
                    (left * golden::PHI_INV + top * golden::PHI_INV2 + diag * golden::PHI_INV3) / w_sum
                };

                let d = if idx < dc_deltas.len() { dc_deltas[idx] } else { 0 };
                recon_dcs[idx] = pred + d as f64 * fine_step;
            }
        }
        recon_dcs_all.push(recon_dcs);
    }

    // Build smooth prediction maps (bilinear interpolation of DC grids)
    let empty_aa = vec![124u8; n_hexes];
    let pred_maps: Vec<Vec<f64>> = (0..3).map(|ch| {
        hex_edge::build_hex_gradient_prediction(
            &recon_dcs_all[ch], &empty_aa, 1.0,
            width, height, &grid, &shape,
        )
    }).collect();

    // 5. Reconstruct saliency S(x,y) from transmitted DCs — same computation as encoder.
    // This is FREE: zero bpp, the decoder derives it from already-transmitted data.
    let saliency_maps: Vec<Vec<f64>> = (0..3).map(|ch| {
        let hex_sal = hex_edge::compute_hex_saliency(&recon_dcs_all[ch], &grid);
        hex_edge::build_saliency_map(&hex_sal, width, height, &grid)
    }).collect();

    // 6. Decode residual streams with foveal quantization
    let path = hex_edge::hilbert_path(width, height);
    let hex_id_map = hex_edge::build_hex_ownership_map(width, height, &grid, &shape);
    let mut output_planes: Vec<Vec<f64>> = Vec::with_capacity(3);

    for ch in 0..3usize {
        let stream_size = read_u32(body, &mut pos) as usize;
        let (deltas, _) = rans::rans_decode_band_v11(&body[pos..pos + stream_size], n_pixels);
        pos += stream_size;

        let plane = hex_edge::decode_hex_predicted_residuals(
            &deltas, &pred_maps[ch], &path, res_steps[ch], n_pixels, &saliency_maps[ch], &hex_id_map,
        );
        output_planes.push(plane);
    }

    // 6. Photon map (read but injection disabled for now)
    let photon_block_size = read_u32(body, &mut pos) as usize;
    let (_photon_map, _) = photon::decode_photon_map(&body[pos..pos + photon_block_size])?;

    // 5. Inverse PTF on L channel
    {
        use golden::PTF_GAMMA_INV;
        let inv255 = 1.0 / 255.0;
        for v in output_planes[0].iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA_INV);
        }
    }

    // 6. CAS sharpening on luminance only
    sharpen_luminance_planes(&mut output_planes, height, width);

    // 7. Inverse GCT -> RGB
    let (mut r_plane, mut g_plane, mut b_plane) = color::golden_rotate_inverse(
        &output_planes[0], &output_planes[1], &output_planes[2], n_pixels,
    );

    // 7. Schrodinger collapse
    let r_out = dsp::schrodinger_collapse(&r_plane, height, width);
    let g_out = dsp::schrodinger_collapse(&g_plane, height, width);
    let b_out = dsp::schrodinger_collapse(&b_plane, height, width);

    let mut rgb = Vec::with_capacity(n_pixels * 3);
    for i in 0..n_pixels {
        rgb.push(r_out[i]);
        rgb.push(g_out[i]);
        rgb.push(b_out[i]);
    }

    Ok(DecodedImage { rgb, width, height })
}

/// Edge-energy decoder (AUR2 version 8).
///
/// Pipeline: parse header -> read reference pixel values -> rANS decode 9 edge
/// planes (3 dirs x 3 channels) -> dequantize -> Poisson reconstruction from
/// edges + reference value -> inverse PTF on L -> inverse GCT -> RGB ->
/// Schrodinger collapse (Floyd-Steinberg dither).
fn decode_edge_energy(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    // 1. Parse AUR2 header (version=8)
    let (header, header_size) = bitstream::parse_aur2_header(file_data)?;
    let width = header.width;
    let height = header.height;
    let n_pixels = width * height;

    let body = &file_data[header_size..];
    let mut pos = 0usize;

    let read_f32_le = |data: &[u8], p: &mut usize| -> f32 {
        let v = f32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4;
        v
    };
    let read_u32 = |data: &[u8], p: &mut usize| -> u32 {
        let v = u32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4;
        v
    };

    // 2. Read edge-energy header: reference values + step sizes (20 bytes)
    let ref_l = read_f32_le(body, &mut pos) as f64;
    let ref_c1 = read_f32_le(body, &mut pos) as f64;
    let ref_c2 = read_f32_le(body, &mut pos) as f64;
    let edge_step_l = read_f32_le(body, &mut pos) as f64;
    let edge_step_c = read_f32_le(body, &mut pos) as f64;

    // 2b. Read mode byte
    let mode = body[pos]; pos += 1;

    let steps = [edge_step_l, edge_step_c, edge_step_c];
    let mut output_planes: Vec<Vec<f64>> = Vec::with_capacity(3);

    if mode == 2 {
        // DNA-guided path-DPCM mode: codons + deltas
        let path = hex_edge::hilbert_path(width, height);

        // Read codon streams first (3 channels)
        let mut codon_streams: Vec<Vec<u8>> = Vec::with_capacity(3);
        for _ch in 0..3usize {
            let stream_size = read_u32(body, &mut pos) as usize;
            let amino_acids = rans::rans_decompress_bytes(&body[pos..pos + stream_size]);
            pos += stream_size;
            codon_streams.push(amino_acids);
        }

        // Read DPCM delta streams (3 channels) and reconstruct
        let empty_solid = vec![false; n_pixels]; // DNA mode has its own step logic
        for ch in 0..3usize {
            let stream_size = read_u32(body, &mut pos) as usize;
            let (deltas, _) = rans::rans_decode_band(&body[pos..pos + stream_size], n_pixels);
            pos += stream_size;

            let plane = hex_edge::decode_path_dpcm_dna(
                &deltas, &path, steps[ch], n_pixels, &codon_streams[ch],
            );
            output_planes.push(plane);
        }
    } else if mode == 3 {
        // Hex-path polymerase mode: hex-level DPCM with gas/solid classification

        // Read interior step sizes
        let interior_step_l = read_f32_le(body, &mut pos) as f64;
        let interior_step_c = read_f32_le(body, &mut pos) as f64;
        let int_steps = [interior_step_l, interior_step_c, interior_step_c];

        // Read number of hexes
        let n_hexes = read_u32(body, &mut pos) as usize;

        // Read gas map (packed bits, rANS compressed)
        let gas_map_size = read_u32(body, &mut pos) as usize;
        let gas_packed = rans::rans_decompress_bytes(&body[pos..pos + gas_map_size]);
        pos += gas_map_size;

        // Unpack gas flags
        let mut gas_flags = Vec::with_capacity(n_hexes);
        for i in 0..n_hexes {
            let byte_idx = i / 8;
            let bit_idx = 7 - i % 8;
            let is_gas = if byte_idx < gas_packed.len() {
                (gas_packed[byte_idx] >> bit_idx) & 1 == 1
            } else {
                true // default to gas
            };
            gas_flags.push(is_gas);
        }

        // Read codon streams (3 channels)
        let mut codon_streams: Vec<Vec<u8>> = Vec::with_capacity(3);
        for _ch in 0..3usize {
            let stream_size = read_u32(body, &mut pos) as usize;
            let amino_acids = rans::rans_decompress_bytes(&body[pos..pos + stream_size]);
            pos += stream_size;
            codon_streams.push(amino_acids);
        }

        // Read DC delta streams (3 channels)
        let mut dc_streams: Vec<Vec<i16>> = Vec::with_capacity(3);
        for _ch in 0..3usize {
            let stream_size = read_u32(body, &mut pos) as usize;
            let (deltas, _) = rans::rans_decode_band(&body[pos..pos + stream_size], n_hexes);
            pos += stream_size;
            dc_streams.push(deltas);
        }

        // Read interior delta streams (3 channels): count + rANS data
        let mut interior_streams: Vec<Vec<i16>> = Vec::with_capacity(3);
        for _ch in 0..3usize {
            let n_interior = read_u32(body, &mut pos) as usize;
            let stream_size = read_u32(body, &mut pos) as usize;
            let (deltas, _) = rans::rans_decode_band(&body[pos..pos + stream_size], n_interior);
            pos += stream_size;
            interior_streams.push(deltas);
        }

        // Build hex path (same as encoder)
        let hex_path = hex_edge::hex_spiral_path(width, height);
        let shape = hex::compute_hex_shape();

        // Decode each channel
        for ch in 0..3usize {
            let plane = hex_edge::decode_hex_path_channel(
                &dc_streams[ch],
                &interior_streams[ch],
                &codon_streams[ch],
                &gas_flags,
                &hex_path,
                &shape,
                width, height,
                steps[ch],  // DC step
                int_steps[ch], // interior step
            );
            output_planes.push(plane);
        }
    } else if mode == 1 {
        // Unified path-DPCM with hex supercordes oracle
        let path = hex_edge::hilbert_path(width, height);

        // Read hex solid map
        let read_u16_le = |data: &[u8], p: &mut usize| -> u16 {
            let v = u16::from_le_bytes([data[*p], data[*p + 1]]);
            *p += 2; v
        };
        let solid_cols = read_u16_le(body, &mut pos) as usize;
        let solid_rows = read_u16_le(body, &mut pos) as usize;
        let solid_size = read_u32(body, &mut pos) as usize;
        let solid_packed = rans::rans_decompress_bytes(&body[pos..pos + solid_size]);
        pos += solid_size;
        let solid_map = hex_edge::unpack_solid_map(&solid_packed, solid_cols, solid_rows, width, height);

        // Decode all 3 channels: L, C1, C2 (no relativistic modulation)
        for ch in 0..3usize {
            let stream_size = read_u32(body, &mut pos) as usize;
            let (deltas, _) = rans::rans_decode_band(&body[pos..pos + stream_size], n_pixels);
            pos += stream_size;

            let plane = hex_edge::decode_path_dpcm(
                &deltas, &path, steps[ch], n_pixels, &solid_map, None,
            );
            output_planes.push(plane);
        }
    } else {
        // Legacy 9-edge mode (backward compat)
        return Err("Legacy 9-edge mode not supported in path-DPCM decoder".into());
    }

    // 6. Inverse PTF on L channel
    {
        use golden::PTF_GAMMA_INV;
        let inv255 = 1.0 / 255.0;
        for v in output_planes[0].iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA_INV);
        }
    }

    // CAS sharpening on luminance only
    sharpen_luminance_planes(&mut output_planes, height, width);

    // 7. Inverse GCT -> RGB
    let (mut r_plane, mut g_plane, mut b_plane) = color::golden_rotate_inverse(
        &output_planes[0],
        &output_planes[1],
        &output_planes[2],
        n_pixels,
    );

    // Post-decode CASP sharpening

    // 8. Schrodinger collapse: continuous -> certain via error diffusion (Floyd-Steinberg)
    let r_out = dsp::schrodinger_collapse(&r_plane, height, width);
    let g_out = dsp::schrodinger_collapse(&g_plane, height, width);
    let b_out = dsp::schrodinger_collapse(&b_plane, height, width);

    let mut rgb_out = Vec::with_capacity(n_pixels * 3);
    for i in 0..n_pixels {
        rgb_out.push(r_out[i]);
        rgb_out.push(g_out[i]);
        rgb_out.push(b_out[i]);
    }

    eprintln!("  Edge-energy decoded: {}x{}", width, height);

    Ok(DecodedImage {
        rgb: rgb_out,
        width,
        height,
    })
}

/// DNA5 hexagonal decoder: full pipeline.
///
/// Pipeline: parse headers -> rANS decode DC/AC/codons -> DPCM inverse ->
/// IDCT-II -> harmonic background + texture field -> scatter to pixels ->
/// inverse PTF -> inverse GCT -> RGB.
fn decode_aur2_hex(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    use hex::{compute_hex_shape, HexGrid, HexSubdiv, HEX_R};
    use hex_decoder::{HarmonicWeights, reconstruct_hex_interior_subdivided};
    use hex_encoder::{idct_ii_1d, unpack_codon, unpack_subdivisions, VertexCodon};

    // 1. Parse AUR2 header
    let (header, header_size) = bitstream::parse_aur2_header(file_data)?;
    let width = header.width;
    let height = header.height;
    let detail_step = header.detail_step;

    let body = &file_data[header_size..];
    let mut pos = 0usize;

    // Helper closures for reading from body
    let read_u16 = |data: &[u8], p: &mut usize| -> u16 {
        let v = u16::from_le_bytes([data[*p], data[*p + 1]]);
        *p += 2;
        v
    };
    let read_u32 = |data: &[u8], p: &mut usize| -> u32 {
        let v = u32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4;
        v
    };
    let read_f32_le = |data: &[u8], p: &mut usize| -> f32 {
        let v = f32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4;
        v
    };

    // 2. Parse hex header (6 bytes)
    let hex_hdr = bitstream::parse_hex_header(&body[pos..])?;
    pos += 6;

    let _hex_radius = hex_hdr.hex_radius as usize;
    let hex_cols = hex_hdr.hex_cols as usize;
    let hex_rows = hex_hdr.hex_rows as usize;
    let n_hexes = hex_cols * hex_rows;

    // DC step
    let dc_step = read_f32_le(body, &mut pos) as f64;

    // Perimeter and coefficients counts
    let n_perimeter = read_u16(body, &mut pos) as usize;
    let n_coeffs = read_u16(body, &mut pos) as usize;

    // max_harmonics: number of AC coefficients actually encoded per hex
    let max_harmonics = body[pos] as usize;
    pos += 1;
    // codons_per_hex: 0 = variable mode (Whole=2, subdivided=6), nonzero = fixed count
    let codons_per_hex_flag = body[pos] as usize;
    pos += 1;

    let ac_per_hex = max_harmonics; // only max_harmonics AC coefficients were encoded

    // 3. rANS decode DC streams (3 channels)
    let mut dc_residuals: Vec<Vec<i16>> = Vec::with_capacity(3);
    for _ in 0..3 {
        let stream_size = read_u32(body, &mut pos) as usize;
        let (decoded, _) = rans::rans_decode_band(&body[pos..pos + stream_size], n_hexes);
        pos += stream_size;
        dc_residuals.push(decoded);
    }

    // 4. rANS decode AC streams (3 channels)
    let mut ac_flat: Vec<Vec<i16>> = Vec::with_capacity(3);
    for _ in 0..3 {
        let stream_size = read_u32(body, &mut pos) as usize;
        let total_ac = n_hexes * ac_per_hex;
        let (decoded, _) = rans::rans_decode_band(&body[pos..pos + stream_size], total_ac);
        pos += stream_size;
        ac_flat.push(decoded);
    }

    // 5. rANS decode codons stream
    let codon_stream_size = read_u32(body, &mut pos) as usize;
    let codon_bytes = rans::rans_decompress_bytes(&body[pos..pos + codon_stream_size]);
    pos += codon_stream_size;

    // 5b. rANS decode subdivision stream (2 bits per hex, packed)
    let subdivisions: Vec<HexSubdiv> = if pos + 4 <= body.len() {
        let subdiv_stream_size = read_u32(body, &mut pos) as usize;
        if subdiv_stream_size > 0 && pos + subdiv_stream_size <= body.len() {
            let subdiv_packed = rans::rans_decompress_bytes(&body[pos..pos + subdiv_stream_size]);
            pos += subdiv_stream_size;
            unpack_subdivisions(&subdiv_packed, n_hexes)
        } else {
            pos += subdiv_stream_size;
            vec![HexSubdiv::Whole; n_hexes]
        }
    } else {
        // Backward compatibility: old files without subdivision stream
        vec![HexSubdiv::Whole; n_hexes]
    };
    // 5c. Build per-hex codon offset table for variable codon mode.
    // codons_per_hex_flag == 0: variable (Whole=2, subdivided=6).
    // codons_per_hex_flag > 0: fixed count (backward compat with older files).
    let codon_offsets: Vec<usize> = {
        let mut offsets = Vec::with_capacity(n_hexes);
        let mut cursor = 0usize;
        for i in 0..n_hexes {
            offsets.push(cursor);
            let n_codons = if codons_per_hex_flag == 0 {
                match subdivisions[i] {
                    HexSubdiv::Whole => 2,
                    _ => 6, // Bisect/Trisect/Full: all 6 vertex codons
                }
            } else {
                codons_per_hex_flag
            };
            cursor += n_codons;
        }
        offsets
    };

    // 6. DC DPCM inverse (cumulative sum)
    let mut dc_values: Vec<Vec<f64>> = Vec::with_capacity(3);
    for ch_idx in 0..3 {
        let mut dc_ch: Vec<f64> = Vec::with_capacity(n_hexes);
        let mut prev = 0.0f64;
        for &q in &dc_residuals[ch_idx] {
            let dc_val = prev + q as f64 * dc_step;
            dc_ch.push(dc_val);
            prev = dc_val;
        }
        dc_values.push(dc_ch);
    }

    // 7. Precompute hex shape and harmonic weights
    let shape = compute_hex_shape();
    let hw = HarmonicWeights::precompute();
    let grid = HexGrid::new(width, height);

    // Vertex positions (6 vertices of the hex) relative to center
    let hex_vertex_offsets: Vec<(f64, f64)> = (0..6)
        .map(|i| {
            let angle = std::f64::consts::PI / 3.0 * i as f64;
            let r = HEX_R as f64;
            (r * angle.cos(), r * angle.sin())
        })
        .collect();

    // Interior positions from hex shape (as f64)
    let interior_positions: Vec<(f64, f64)> = shape
        .interior
        .iter()
        .map(|&(x, y)| (x as f64, y as f64))
        .collect();

    // Build lookup maps for perimeter and interior (relative offsets)
    let interior_lookup: std::collections::HashMap<(i32, i32), usize> = shape
        .interior
        .iter()
        .enumerate()
        .map(|(i, &pt)| (pt, i))
        .collect();
    let peri_lookup: std::collections::HashMap<(i32, i32), usize> = shape
        .perimeter
        .iter()
        .enumerate()
        .map(|(i, &pt)| (pt, i))
        .collect();

    // 8. Reconstruct per channel: allocate output planes
    let n_pixels = width * height;
    let mut output_planes: Vec<Vec<f64>> = vec![vec![0.0f64; n_pixels]; 3];
    let mut pixel_written: Vec<bool> = vec![false; n_pixels];

    // Hex-scaled quantization step (must match encoder)
    let hex_step = detail_step * 8.0;
    let phi = golden::PHI;

    // Iterate hexes in raster order
    let mut hex_idx = 0usize;
    for (col, row) in grid.iter_raster() {
        let (cx, cy) = grid.center(col, row);

        // Reconstruct DCT coefficients for all 3 channels
        let mut perimeter_values_ch: Vec<Vec<f64>> = Vec::with_capacity(3);
        let mut interior_vals_ch: Vec<Vec<f64>> = Vec::with_capacity(3);

        for ch_idx in 0..3 {
            // Start with all-zero coefficients (zero-pad truncated harmonics)
            let mut coeffs = vec![0.0f64; n_coeffs];
            coeffs[0] = dc_values[ch_idx][hex_idx];

            // Only max_harmonics AC coefficients were encoded; the rest stay zero
            for k in 1..=max_harmonics {
                if k >= n_coeffs { break; }
                let ac_idx = hex_idx * ac_per_hex + (k - 1);
                if ac_idx < ac_flat[ch_idx].len() {
                    let golden_step = hex_step * (k as f64 * phi).powf(0.55);
                    let step = golden_step.max(0.1);
                    coeffs[k] = ac_flat[ch_idx][ac_idx] as f64 * step;
                }
            }

            // IDCT-II: reconstruct perimeter values (with zero-padded high harmonics)
            let perimeter_values = idct_ii_1d(&coeffs, n_perimeter);

            // Reconstruct interior: harmonic background + texture
            let vertex_positions: Vec<(f64, f64)> = hex_vertex_offsets
                .iter()
                .map(|&(dx, dy)| (cx + dx, cy + dy))
                .collect();

            // Reconstruct 6 codons from the variable-length codon stream.
            // Whole hexes: 2 codons stored (vertices 0,1), others default (Rien).
            // Subdivided hexes: all 6 codons stored.
            let codons: Vec<VertexCodon> = if ch_idx == 0 {
                let base = codon_offsets[hex_idx];
                let n_stored = if codons_per_hex_flag == 0 {
                    match subdivisions[hex_idx] {
                        HexSubdiv::Whole => 2,
                        _ => 6,
                    }
                } else {
                    codons_per_hex_flag
                };
                (0..6)
                    .map(|i| {
                        if i < n_stored {
                            let ci = base + i;
                            if ci < codon_bytes.len() {
                                unpack_codon(codon_bytes[ci])
                            } else {
                                VertexCodon::default()
                            }
                        } else {
                            // Non-stored vertex: default (Rien)
                            VertexCodon::default()
                        }
                    })
                    .collect()
            } else {
                vec![VertexCodon::default(); 6]
            };

            let subdiv = subdivisions[hex_idx];
            let interior_vals = reconstruct_hex_interior_subdivided(
                &hw,
                &perimeter_values,
                &interior_positions,
                &vertex_positions,
                &codons,
                subdiv,
            );

            perimeter_values_ch.push(perimeter_values);
            interior_vals_ch.push(interior_vals);
        }

        // Compute mean perimeter value per channel (for fallback on voronoi-only pixels)
        let mean_peri: Vec<f64> = (0..3)
            .map(|ch| {
                let pv = &perimeter_values_ch[ch];
                if pv.is_empty() {
                    0.0
                } else {
                    pv.iter().sum::<f64>() / pv.len() as f64
                }
            })
            .collect();

        // Scatter all voronoi pixels: use perimeter value if on perimeter,
        // interior harmonic value if in interior, mean value otherwise.
        for &(dx, dy) in &shape.voronoi {
            let px = (cx + dx as f64).round() as isize;
            let py = (cy + dy as f64).round() as isize;
            if px < 0 || px >= width as isize || py < 0 || py >= height as isize {
                continue;
            }
            let idx = py as usize * width + px as usize;

            // Skip if already written by a previous hex (first-write wins for voronoi)
            if pixel_written[idx] {
                continue;
            }

            for ch in 0..3 {
                let val = if let Some(&peri_idx) = peri_lookup.get(&(dx, dy)) {
                    // Perimeter pixel: use DCT-reconstructed value
                    if peri_idx < perimeter_values_ch[ch].len() {
                        perimeter_values_ch[ch][peri_idx]
                    } else {
                        mean_peri[ch]
                    }
                } else if let Some(&int_idx) = interior_lookup.get(&(dx, dy)) {
                    // Interior pixel: use harmonic reconstruction
                    if int_idx < interior_vals_ch[ch].len() {
                        interior_vals_ch[ch][int_idx]
                    } else {
                        mean_peri[ch]
                    }
                } else {
                    // Voronoi pixel not in perimeter or interior: use mean value
                    mean_peri[ch]
                };

                output_planes[ch][idx] = val;
            }
            pixel_written[idx] = true;
        }

        hex_idx += 1;
    }

    // 8b. Apply interior residuals for ALL hexes, ALL channels (L, C1, C2).
    //     Read 3 residual streams (one per channel), dequantize and add to output.
    {
        let resid_step = detail_step * 0.8; // must match encoder (finer = quality carrier)

        // Total residual pixels per channel = ALL hexes × interior pixels
        let n_interior = shape.interior.len();
        let total_resid_pixels: usize = n_hexes * n_interior;

        for ch_idx in 0..3 {
            if pos + 4 > body.len() { break; }
            let resid_stream_size = read_u32(body, &mut pos) as usize;
            if resid_stream_size == 0 || pos + resid_stream_size > body.len() {
                pos += resid_stream_size;
                continue;
            }

            let (resid_decoded, _) = rans::rans_decode_band(
                &body[pos..pos + resid_stream_size],
                total_resid_pixels,
            );
            pos += resid_stream_size;

            // Walk hexes in raster order, applying residuals to this channel
            let mut resid_idx = 0usize;
            let mut hex_i2 = 0usize;
            for (col, row) in grid.iter_raster() {
                let (cx, cy) = grid.center(col, row);

                for &(dx, dy) in &shape.interior {
                    let px = (cx + dx as f64).round() as isize;
                    let py = (cy + dy as f64).round() as isize;
                    if px >= 0 && px < width as isize && py >= 0 && py < height as isize {
                        let idx = py as usize * width + px as usize;
                        if resid_idx < resid_decoded.len() {
                            output_planes[ch_idx][idx] += resid_decoded[resid_idx] as f64 * resid_step;
                        }
                    }
                    resid_idx += 1;
                }

                hex_i2 += 1;
            }
            let _ = hex_i2; // suppress unused warning
        }
    }
    let _ = pos; // suppress unused warning

    // Fill any remaining uncovered pixels with neighbor average
    // (edge pixels that no voronoi cell claimed)
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if !pixel_written[idx] {
                // Find nearest written pixel via simple search
                let mut best_dist = f64::MAX;
                let mut best_val = [0.0f64; 3];
                let search_r = 10isize;
                for dy in -search_r..=search_r {
                    for dx in -search_r..=search_r {
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                            let nidx = ny as usize * width + nx as usize;
                            if pixel_written[nidx] {
                                let d = (dx * dx + dy * dy) as f64;
                                if d < best_dist {
                                    best_dist = d;
                                    for ch in 0..3 {
                                        best_val[ch] = output_planes[ch][nidx];
                                    }
                                }
                            }
                        }
                    }
                }
                for ch in 0..3 {
                    output_planes[ch][idx] = best_val[ch];
                }
            }
        }
    }

    // 9. Inverse PTF on L channel
    {
        use golden::PTF_GAMMA_INV;
        let inv255 = 1.0 / 255.0;
        for v in output_planes[0].iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA_INV);
        }
    }

    // CAS sharpening on luminance only
    sharpen_luminance_planes(&mut output_planes, height, width);

    // 10. Inverse GCT -> RGB
    let (mut r_plane, mut g_plane, mut b_plane) = color::golden_rotate_inverse(
        &output_planes[0],
        &output_planes[1],
        &output_planes[2],
        n_pixels,
    );

    // Post-decode CASP sharpening

    let mut rgb_out = Vec::with_capacity(n_pixels * 3);
    for i in 0..n_pixels {
        rgb_out.push(r_plane[i].round().clamp(0.0, 255.0) as u8);
        rgb_out.push(g_plane[i].round().clamp(0.0, 255.0) as u8);
        rgb_out.push(b_plane[i].round().clamp(0.0, 255.0) as u8);
    }

    Ok(DecodedImage {
        rgb: rgb_out,
        width,
        height,
    })
}

/// DNA7 hierarchical hex pyramid decoder: 4-level multi-resolution tessellation.
///
/// Pipeline: parse headers -> for each level (R=13,8,5,3): decode hex positions,
/// DC/AC/codons -> reconstruct interiors -> ADD to accumulator -> inverse PTF ->
/// inverse GCT -> RGB.
///
/// The final image is the SUM of all 4 pyramid levels.
fn decode_aur2_hex_pyramid(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    use hex::compute_hex_shape_r;
    use hex::HexSubdiv;
    use hex_decoder::{MultiScaleWeights, reconstruct_hex_interior_subdivided_r};
    use hex_encoder::{
        idct_ii_1d, unpack_codon, unpack_subdivisions, VertexCodon,
        max_harmonics_for_radius, encode_codons_for_radius, hex_radius_factor,
        build_regular_hex_grid,
    };

    // 1. Parse AUR2 header
    let (header, header_size) = bitstream::parse_aur2_header(file_data)?;
    let width = header.width;
    let height = header.height;

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
    let read_f32_le = |data: &[u8], p: &mut usize| -> f32 {
        let v = f32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4; v
    };

    // 2. Parse hex header (6 bytes)
    let _hex_hdr = bitstream::parse_hex_header(&body[pos..])?;
    pos += 6;

    // 3. Pyramid header
    let n_levels = body[pos] as usize;
    pos += 1;
    let mode = body[pos];
    pos += 1;
    let _detail_step_header = read_f32_le(body, &mut pos) as f64;

    // Precompute shapes and weights for all Fibonacci radii
    let msw = MultiScaleWeights::precompute();

    // Accumulator planes (additive: sum of all levels)
    let n_pixels = width * height;
    let mut output_planes: Vec<Vec<f64>> = vec![vec![0.0f64; n_pixels]; 3];
    // Track pixel coverage (from level 0 primarily)
    let mut pixel_written: Vec<bool> = vec![false; n_pixels];

    let phi = golden::PHI;

    // 4. Decode level 0: hex perimeter reconstruction
    {
        let r = body[pos] as usize;
        pos += 1;
        let n_active = read_u32(body, &mut pos) as usize;
        let _grid_cols = read_u16(body, &mut pos) as usize;
        let _grid_rows = read_u16(body, &mut pos) as usize;
        let dc_step = read_f32_le(body, &mut pos) as f64;

        let shape = compute_hex_shape_r(r);
        let hw = msw.for_radius(r);
        let n_peri = shape.perimeter.len();
        let n_coeffs = n_peri / 2;
        let mh = max_harmonics_for_radius(r, header.quality).min(n_coeffs.saturating_sub(1));
        let hex_step = _detail_step_header * hex_radius_factor(r);
        let rf = r as f64;

        // Rebuild the full grid to map indices back to centers
        let (all_centers, _gc, _gr) = build_regular_hex_grid(width, height, r);

        // Read active hex indices
        let idx_stream_size = read_u32(body, &mut pos) as usize;
        let idx_bytes = rans::rans_decompress_bytes(&body[pos..pos + idx_stream_size]);
        pos += idx_stream_size;

        let active_indices: Vec<u32> = idx_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(active_indices.len(), n_active,
            "level 0 active index count mismatch: {} vs {}", active_indices.len(), n_active);

        let active_centers: Vec<(f64, f64)> = active_indices
            .iter()
            .map(|&idx| all_centers[idx as usize])
            .collect();

        // DC streams (3 channels)
        let mut dc_residuals: Vec<Vec<i16>> = Vec::with_capacity(3);
        for _ in 0..3 {
            let stream_size = read_u32(body, &mut pos) as usize;
            let (decoded, _) = rans::rans_decode_band(&body[pos..pos + stream_size], n_active);
            pos += stream_size;
            dc_residuals.push(decoded);
        }

        // AC streams (3 channels)
        let total_ac = n_active * mh;
        let mut ac_flat: Vec<Vec<i16>> = Vec::with_capacity(3);
        for _ in 0..3 {
            let stream_size = read_u32(body, &mut pos) as usize;
            let (decoded, _) = rans::rans_decode_band(&body[pos..pos + stream_size], total_ac);
            pos += stream_size;
            ac_flat.push(decoded);
        }

        // Codon stream
        let codon_stream_size = read_u32(body, &mut pos) as usize;
        let codon_bytes = rans::rans_decompress_bytes(&body[pos..pos + codon_stream_size]);
        pos += codon_stream_size;

        // Subdivision stream
        let subdiv_stream_size = read_u32(body, &mut pos) as usize;
        let subdivisions: Vec<HexSubdiv> = if subdiv_stream_size > 0 && pos + subdiv_stream_size <= body.len() {
            let subdiv_packed = rans::rans_decompress_bytes(&body[pos..pos + subdiv_stream_size]);
            pos += subdiv_stream_size;
            unpack_subdivisions(&subdiv_packed, n_active)
        } else {
            pos += subdiv_stream_size;
            vec![HexSubdiv::Whole; n_active]
        };

        // Build codon offsets
        let codon_offsets: Vec<usize> = {
            let mut offsets = Vec::with_capacity(n_active);
            let mut cursor = 0usize;
            for i in 0..n_active {
                offsets.push(cursor);
                let n_codons = if encode_codons_for_radius(r) {
                    match subdivisions[i] {
                        HexSubdiv::Whole => 2,
                        _ => 6,
                    }
                } else {
                    0
                };
                cursor += n_codons;
            }
            offsets
        };

        // DC DPCM inverse
        let mut dc_values: Vec<Vec<f64>> = Vec::with_capacity(3);
        for ch_idx in 0..3 {
            let mut dc_ch: Vec<f64> = Vec::with_capacity(n_active);
            let mut prev_norm = 0.0f64;
            for &q in &dc_residuals[ch_idx] {
                let dc_norm = prev_norm + q as f64 * dc_step;
                dc_ch.push(dc_norm * n_peri as f64);
                prev_norm = dc_norm;
            }
            dc_values.push(dc_ch);
        }

        // Build lookup maps for this shape
        let interior_lookup: std::collections::HashMap<(i32, i32), usize> = shape
            .interior
            .iter()
            .enumerate()
            .map(|(i, &pt)| (pt, i))
            .collect();
        let peri_lookup: std::collections::HashMap<(i32, i32), usize> = shape
            .perimeter
            .iter()
            .enumerate()
            .map(|(i, &pt)| (pt, i))
            .collect();

        // Reconstruct each active hex
        for (ai, &(cx, cy)) in active_centers.iter().enumerate() {
            let subdiv = subdivisions[ai];

            let codons: Vec<VertexCodon> = if encode_codons_for_radius(r) {
                let base = codon_offsets[ai];
                let n_stored = match subdivisions[ai] {
                    HexSubdiv::Whole => 2,
                    _ => 6,
                };
                (0..6)
                    .map(|i| {
                        if i < n_stored {
                            let ci = base + i;
                            if ci < codon_bytes.len() {
                                unpack_codon(codon_bytes[ci])
                            } else {
                                VertexCodon::default()
                            }
                        } else {
                            VertexCodon::default()
                        }
                    })
                    .collect()
            } else {
                vec![VertexCodon::default(); 6]
            };

            let interior_positions: Vec<(f64, f64)> = shape
                .interior
                .iter()
                .map(|&(x, y)| (x as f64, y as f64))
                .collect();

            let vertex_positions: Vec<(f64, f64)> = (0..6)
                .map(|i| {
                    let angle = std::f64::consts::PI / 3.0 * i as f64;
                    (cx + rf * angle.cos(), cy + rf * angle.sin())
                })
                .collect();

            let mut perimeter_values_ch: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            let mut interior_vals_ch: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            let mut mean_peri_ch = [0.0f64; 3];

            for ch_idx in 0..3 {
                let mut coeffs = vec![0.0f64; n_coeffs];
                coeffs[0] = dc_values[ch_idx][ai];

                let ac_offset = ai * mh;
                for k in 1..=mh {
                    if k >= n_coeffs { break; }
                    let ac_idx = ac_offset + (k - 1);
                    if ac_idx < ac_flat[ch_idx].len() {
                        let golden_step = hex_step * (k as f64 * phi).powf(0.55);
                        let step = golden_step.max(0.1);
                        coeffs[k] = ac_flat[ch_idx][ac_idx] as f64 * step;
                    }
                }

                let perimeter_values = idct_ii_1d(&coeffs, n_peri);

                let ch_codons = if ch_idx == 0 { &codons } else {
                    &codons
                };

                let interior_vals = reconstruct_hex_interior_subdivided_r(
                    hw,
                    &perimeter_values,
                    &interior_positions,
                    &vertex_positions,
                    ch_codons,
                    subdiv,
                    r,
                );

                mean_peri_ch[ch_idx] = if perimeter_values.is_empty() {
                    0.0
                } else {
                    perimeter_values.iter().sum::<f64>() / perimeter_values.len() as f64
                };

                perimeter_values_ch[ch_idx] = perimeter_values;
                interior_vals_ch[ch_idx] = interior_vals;
            }

            // Scatter to pixel buffer (first-write-wins for level 0)
            for &(ddx, ddy) in &shape.voronoi {
                let px = (cx + ddx as f64).round() as isize;
                let py = (cy + ddy as f64).round() as isize;
                if px < 0 || px >= width as isize || py < 0 || py >= height as isize {
                    continue;
                }
                let idx = py as usize * width + px as usize;

                if pixel_written[idx] {
                    continue;
                }

                for ch_idx in 0..3 {
                    let val = if let Some(&peri_idx) = peri_lookup.get(&(ddx, ddy)) {
                        if peri_idx < perimeter_values_ch[ch_idx].len() {
                            perimeter_values_ch[ch_idx][peri_idx]
                        } else { mean_peri_ch[ch_idx] }
                    } else if let Some(&int_idx) = interior_lookup.get(&(ddx, ddy)) {
                        if int_idx < interior_vals_ch[ch_idx].len() {
                            interior_vals_ch[ch_idx][int_idx]
                        } else { mean_peri_ch[ch_idx] }
                    } else {
                        mean_peri_ch[ch_idx]
                    };

                    output_planes[ch_idx][idx] = val;
                }
                pixel_written[idx] = true;
            }
        }

        eprintln!("  Decoded pyramid level 0 (R={}): {} hexes", r, n_active);
    }

    // 5. Decode refinement levels 1..n_levels-1
    if mode == 1 {
        // Mode 1: per-pixel residual for levels 1+
        for level in 1..n_levels {
            let resid_step = read_f32_le(body, &mut pos) as f64;

            for ch in 0..3 {
                let stream_size = read_u32(body, &mut pos) as usize;
                let (quantized, _) = rans::rans_decode_band(
                    &body[pos..pos + stream_size],
                    n_pixels,
                );
                pos += stream_size;

                for i in 0..n_pixels {
                    output_planes[ch][i] += quantized[i] as f64 * resid_step;
                }
            }

            eprintln!("  Decoded pyramid level {} (pixel residual, step={:.3})", level, resid_step);
        }
    } else {
        // Mode 0 (legacy): all levels are hex perimeter
        for level in 1..n_levels {
            let r = body[pos] as usize;
            pos += 1;
            let n_active = read_u32(body, &mut pos) as usize;
            let _grid_cols = read_u16(body, &mut pos) as usize;
            let _grid_rows = read_u16(body, &mut pos) as usize;
            let dc_step = read_f32_le(body, &mut pos) as f64;

            let shape = compute_hex_shape_r(r);
            let hw = msw.for_radius(r);
            let n_peri = shape.perimeter.len();
            let n_coeffs = n_peri / 2;
            let mh = max_harmonics_for_radius(r, header.quality).min(n_coeffs.saturating_sub(1));
            let hex_step = _detail_step_header * hex_radius_factor(r);
            let rf = r as f64;

            let (all_centers, _gc, _gr) = build_regular_hex_grid(width, height, r);

            let idx_stream_size = read_u32(body, &mut pos) as usize;
            let idx_bytes = rans::rans_decompress_bytes(&body[pos..pos + idx_stream_size]);
            pos += idx_stream_size;

            let active_indices: Vec<u32> = idx_bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let active_centers: Vec<(f64, f64)> = active_indices
                .iter()
                .map(|&idx| all_centers[idx as usize])
                .collect();

            // DC streams
            let mut dc_residuals: Vec<Vec<i16>> = Vec::with_capacity(3);
            for _ in 0..3 {
                let stream_size = read_u32(body, &mut pos) as usize;
                let (decoded, _) = rans::rans_decode_band(&body[pos..pos + stream_size], n_active);
                pos += stream_size;
                dc_residuals.push(decoded);
            }

            // AC streams
            let total_ac = n_active * mh;
            let mut ac_flat: Vec<Vec<i16>> = Vec::with_capacity(3);
            for _ in 0..3 {
                let stream_size = read_u32(body, &mut pos) as usize;
                let (decoded, _) = rans::rans_decode_band(&body[pos..pos + stream_size], total_ac);
                pos += stream_size;
                ac_flat.push(decoded);
            }

            // Codon stream
            let codon_stream_size = read_u32(body, &mut pos) as usize;
            let _codon_bytes = rans::rans_decompress_bytes(&body[pos..pos + codon_stream_size]);
            pos += codon_stream_size;

            // Subdivision stream
            let subdiv_stream_size = read_u32(body, &mut pos) as usize;
            let subdivisions: Vec<HexSubdiv> = if subdiv_stream_size > 0 && pos + subdiv_stream_size <= body.len() {
                let subdiv_packed = rans::rans_decompress_bytes(&body[pos..pos + subdiv_stream_size]);
                pos += subdiv_stream_size;
                unpack_subdivisions(&subdiv_packed, n_active)
            } else {
                pos += subdiv_stream_size;
                vec![HexSubdiv::Whole; n_active]
            };

            // DC DPCM inverse
            let mut dc_values: Vec<Vec<f64>> = Vec::with_capacity(3);
            for ch_idx in 0..3 {
                let mut dc_ch: Vec<f64> = Vec::with_capacity(n_active);
                let mut prev_norm = 0.0f64;
                for &q in &dc_residuals[ch_idx] {
                    let dc_norm = prev_norm + q as f64 * dc_step;
                    dc_ch.push(dc_norm * n_peri as f64);
                    prev_norm = dc_norm;
                }
                dc_values.push(dc_ch);
            }

            let interior_lookup: std::collections::HashMap<(i32, i32), usize> = shape
                .interior
                .iter()
                .enumerate()
                .map(|(i, &pt)| (pt, i))
                .collect();
            let peri_lookup: std::collections::HashMap<(i32, i32), usize> = shape
                .perimeter
                .iter()
                .enumerate()
                .map(|(i, &pt)| (pt, i))
                .collect();

            for (ai, &(cx, cy)) in active_centers.iter().enumerate() {
                let subdiv = subdivisions[ai];
                let codons: Vec<VertexCodon> = vec![VertexCodon::default(); 6];

                let interior_positions: Vec<(f64, f64)> = shape
                    .interior
                    .iter()
                    .map(|&(x, y)| (x as f64, y as f64))
                    .collect();

                let vertex_positions: Vec<(f64, f64)> = (0..6)
                    .map(|i| {
                        let angle = std::f64::consts::PI / 3.0 * i as f64;
                        (cx + rf * angle.cos(), cy + rf * angle.sin())
                    })
                    .collect();

                let mut perimeter_values_ch: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
                let mut interior_vals_ch: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
                let mut mean_peri_ch = [0.0f64; 3];

                for ch_idx in 0..3 {
                    let mut coeffs = vec![0.0f64; n_coeffs];
                    coeffs[0] = dc_values[ch_idx][ai];

                    let ac_offset = ai * mh;
                    for k in 1..=mh {
                        if k >= n_coeffs { break; }
                        let ac_idx = ac_offset + (k - 1);
                        if ac_idx < ac_flat[ch_idx].len() {
                            let golden_step = hex_step * (k as f64 * phi).powf(0.55);
                            let step = golden_step.max(0.1);
                            coeffs[k] = ac_flat[ch_idx][ac_idx] as f64 * step;
                        }
                    }

                    let perimeter_values = idct_ii_1d(&coeffs, n_peri);

                    let interior_vals = reconstruct_hex_interior_subdivided_r(
                        hw,
                        &perimeter_values,
                        &interior_positions,
                        &vertex_positions,
                        &codons,
                        subdiv,
                        r,
                    );

                    mean_peri_ch[ch_idx] = if perimeter_values.is_empty() {
                        0.0
                    } else {
                        perimeter_values.iter().sum::<f64>() / perimeter_values.len() as f64
                    };

                    perimeter_values_ch[ch_idx] = perimeter_values;
                    interior_vals_ch[ch_idx] = interior_vals;
                }

                for &(ddx, ddy) in &shape.voronoi {
                    let px = (cx + ddx as f64).round() as isize;
                    let py = (cy + ddy as f64).round() as isize;
                    if px < 0 || px >= width as isize || py < 0 || py >= height as isize {
                        continue;
                    }
                    let idx = py as usize * width + px as usize;

                    for ch_idx in 0..3 {
                        let val = if let Some(&peri_idx) = peri_lookup.get(&(ddx, ddy)) {
                            if peri_idx < perimeter_values_ch[ch_idx].len() {
                                perimeter_values_ch[ch_idx][peri_idx]
                            } else { mean_peri_ch[ch_idx] }
                        } else if let Some(&int_idx) = interior_lookup.get(&(ddx, ddy)) {
                            if int_idx < interior_vals_ch[ch_idx].len() {
                                interior_vals_ch[ch_idx][int_idx]
                            } else { mean_peri_ch[ch_idx] }
                        } else {
                            mean_peri_ch[ch_idx]
                        };

                        output_planes[ch_idx][idx] += val;
                    }
                }
            }

            eprintln!("  Decoded pyramid level {} (R={}): {} hexes", level, r, n_active);
        }
    }
    let _ = pos;

    // 6. Fill uncovered pixels (edge pixels no voronoi cell claimed)
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if !pixel_written[idx] {
                let mut best_dist = f64::MAX;
                let mut best_val = [0.0f64; 3];
                let search_r = 15isize;
                for dy in -search_r..=search_r {
                    for ddx in -search_r..=search_r {
                        let nx = x as isize + ddx;
                        let ny = y as isize + dy;
                        if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                            let nidx = ny as usize * width + nx as usize;
                            if pixel_written[nidx] {
                                let d = (ddx * ddx + dy * dy) as f64;
                                if d < best_dist {
                                    best_dist = d;
                                    for ch in 0..3 {
                                        best_val[ch] = output_planes[ch][nidx];
                                    }
                                }
                            }
                        }
                    }
                }
                for ch in 0..3 {
                    output_planes[ch][idx] = best_val[ch];
                }
            }
        }
    }

    // 7. Inverse PTF on L channel
    {
        use golden::PTF_GAMMA_INV;
        let inv255 = 1.0 / 255.0;
        for v in output_planes[0].iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA_INV);
        }
    }

    // CAS sharpening on luminance only
    sharpen_luminance_planes(&mut output_planes, height, width);

    // 8. Inverse GCT -> RGB
    let (mut r_plane, mut g_plane, mut b_plane) = color::golden_rotate_inverse(
        &output_planes[0],
        &output_planes[1],
        &output_planes[2],
        n_pixels,
    );

    // Post-decode CASP sharpening

    // 9. Schrodinger collapse: continuous -> certain via error diffusion.
    let r_out = dsp::schrodinger_collapse(&r_plane, height, width);
    let g_out = dsp::schrodinger_collapse(&g_plane, height, width);
    let b_out = dsp::schrodinger_collapse(&b_plane, height, width);

    let mut rgb_out = Vec::with_capacity(n_pixels * 3);
    for i in 0..n_pixels {
        rgb_out.push(r_out[i]);
        rgb_out.push(g_out[i]);
        rgb_out.push(b_out[i]);
    }

    Ok(DecodedImage {
        rgb: rgb_out,
        width,
        height,
    })
}

/// DNA6 multi-scale hexagonal decoder: fractal granulometry with Fibonacci radii.
///
/// Pipeline: parse headers -> read radius map -> read per-cell data ->
/// rANS decode DC/AC/codons -> DPCM inverse -> IDCT-II per cell radius ->
/// harmonic background + texture field -> scatter to pixels -> inverse PTF -> inverse GCT -> RGB.
fn decode_aur2_hex_multiscale(file_data: &[u8]) -> Result<DecodedImage, Box<dyn std::error::Error>> {
    use hex::{compute_hex_shape_r, unpack_radius_map, MultiScaleHexGrid, FIB_RADII};
    use hex_decoder::{MultiScaleWeights, reconstruct_hex_interior_subdivided_r};
    use hex::HexSubdiv;
    use hex_encoder::{idct_ii_1d, unpack_codon, unpack_subdivisions, VertexCodon,
        max_harmonics_for_radius, encode_codons_for_radius, encode_residual_for_radius,
        hex_radius_factor, resid_step_factor};

    // 1. Parse AUR2 header
    let (header, header_size) = bitstream::parse_aur2_header(file_data)?;
    let width = header.width;
    let height = header.height;
    let detail_step = header.detail_step;

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
    let read_f32_le = |data: &[u8], p: &mut usize| -> f32 {
        let v = f32::from_le_bytes([data[*p], data[*p + 1], data[*p + 2], data[*p + 3]]);
        *p += 4; v
    };

    // 2. Parse hex header (6 bytes)
    let _hex_hdr = bitstream::parse_hex_header(&body[pos..])?;
    pos += 6;

    // 3. Multi-scale header extension
    let n_hexes = read_u32(body, &mut pos) as usize;
    let n_rx = read_u16(body, &mut pos) as usize;
    let n_ry = read_u16(body, &mut pos) as usize;

    // DC step
    let dc_step = read_f32_le(body, &mut pos) as f64;

    // codons_per_hex flag
    let codons_per_hex_flag = body[pos] as usize;
    pos += 1;

    // 4. Read radius map and rebuild cell grid (no per-cell coordinates in bitstream)
    let radius_map_stream_size = read_u32(body, &mut pos) as usize;
    let cells: Vec<(f64, f64, usize)> = if radius_map_stream_size > 0 && pos + radius_map_stream_size <= body.len() {
        let decompressed = rans::rans_decompress_bytes(&body[pos..pos + radius_map_stream_size]);
        pos += radius_map_stream_size;
        let n_regions = n_rx * n_ry;
        let radius_map = unpack_radius_map(&decompressed, n_regions);
        // Rebuild the exact same grid the encoder built from the radius map
        let grid = MultiScaleHexGrid::build(&radius_map, n_rx, n_ry, width, height);
        assert_eq!(grid.cells.len(), n_hexes,
            "grid rebuild cell count mismatch: {} vs expected {}", grid.cells.len(), n_hexes);
        grid.cells
    } else {
        pos += radius_map_stream_size;
        return Err("missing radius map in v6 hex bitstream".into());
    };

    // 6. Precompute shapes and weights for all Fibonacci radii
    let shapes: std::collections::HashMap<usize, hex::HexShape> = FIB_RADII
        .iter()
        .map(|&r| (r, compute_hex_shape_r(r)))
        .collect();
    let msw = MultiScaleWeights::precompute();

    // Compute max_harmonics per cell (radius-scaled, must match encoder)
    let all_max_harmonics: Vec<usize> = cells.iter().map(|&(_, _, r)| {
        let shape = &shapes[&r];
        let n_peri = shape.perimeter.len();
        let n_coeffs = n_peri / 2;
        let mh = max_harmonics_for_radius(r, header.quality);
        mh.min(n_coeffs.saturating_sub(1))
    }).collect();

    // Total AC count per channel
    let total_ac: usize = all_max_harmonics.iter().sum();

    // 7. Decode DC streams (3 channels)
    let mut dc_residuals: Vec<Vec<i16>> = Vec::with_capacity(3);
    for _ in 0..3 {
        let stream_size = read_u32(body, &mut pos) as usize;
        let (decoded, _) = rans::rans_decode_band(&body[pos..pos + stream_size], n_hexes);
        pos += stream_size;
        dc_residuals.push(decoded);
    }

    // 8. Decode AC streams (3 channels)
    let mut ac_flat: Vec<Vec<i16>> = Vec::with_capacity(3);
    for _ in 0..3 {
        let stream_size = read_u32(body, &mut pos) as usize;
        let (decoded, _) = rans::rans_decode_band(&body[pos..pos + stream_size], total_ac);
        pos += stream_size;
        ac_flat.push(decoded);
    }

    // 9. Decode codons
    let codon_stream_size = read_u32(body, &mut pos) as usize;
    let codon_bytes = rans::rans_decompress_bytes(&body[pos..pos + codon_stream_size]);
    pos += codon_stream_size;

    // 10. Decode subdivisions
    let subdivisions: Vec<HexSubdiv> = if pos + 4 <= body.len() {
        let subdiv_stream_size = read_u32(body, &mut pos) as usize;
        if subdiv_stream_size > 0 && pos + subdiv_stream_size <= body.len() {
            let subdiv_packed = rans::rans_decompress_bytes(&body[pos..pos + subdiv_stream_size]);
            pos += subdiv_stream_size;
            unpack_subdivisions(&subdiv_packed, n_hexes)
        } else {
            pos += subdiv_stream_size;
            vec![HexSubdiv::Whole; n_hexes]
        }
    } else {
        vec![HexSubdiv::Whole; n_hexes]
    };

    // Build per-hex codon offsets (R=3 hexes have no codons)
    let codon_offsets: Vec<usize> = {
        let mut offsets = Vec::with_capacity(n_hexes);
        let mut cursor = 0usize;
        for i in 0..n_hexes {
            offsets.push(cursor);
            let (_, _, r) = cells[i];
            let n_codons = if !encode_codons_for_radius(r) {
                0 // R=3: no codons
            } else if codons_per_hex_flag == 0 {
                match subdivisions[i] {
                    HexSubdiv::Whole => 2,
                    _ => 6,
                }
            } else {
                codons_per_hex_flag
            };
            cursor += n_codons;
        }
        offsets
    };

    // 11. DC DPCM inverse with denormalization
    //     DPCM was done on DC/N_perimeter values; denormalize back to DCT DC = val * N_perimeter
    let mut dc_values: Vec<Vec<f64>> = Vec::with_capacity(3);
    for ch_idx in 0..3 {
        let mut dc_ch: Vec<f64> = Vec::with_capacity(n_hexes);
        let mut prev_norm = 0.0f64;
        for (cell_idx, &q) in dc_residuals[ch_idx].iter().enumerate() {
            let dc_norm = prev_norm + q as f64 * dc_step;
            let (_, _, r) = cells[cell_idx];
            let n_peri = shapes[&r].perimeter.len() as f64;
            dc_ch.push(dc_norm * n_peri);
            prev_norm = dc_norm;
        }
        dc_values.push(dc_ch);
    }

    // 12. Reconstruct per channel
    let n_pixels = width * height;
    let phi = golden::PHI;
    let mut output_planes: Vec<Vec<f64>> = vec![vec![0.0f64; n_pixels]; 3];
    let mut pixel_written: Vec<bool> = vec![false; n_pixels];

    // Iterate cells in order
    let mut ac_cursor = [0usize; 3];

    for (hex_idx, &(cx, cy, r)) in cells.iter().enumerate() {
        let shape = &shapes[&r];
        let hw = msw.for_radius(r);
        let n_peri = shape.perimeter.len();
        let n_coeffs = n_peri / 2;
        let max_harmonics = all_max_harmonics[hex_idx];
        let rf = r as f64;

        // Per-cell hex step based on radius (must match encoder)
        let hex_step = detail_step * hex_radius_factor(r);

        let mut perimeter_values_ch: Vec<Vec<f64>> = Vec::with_capacity(3);
        let mut interior_vals_ch: Vec<Vec<f64>> = Vec::with_capacity(3);

        for ch_idx in 0..3 {
            let mut coeffs = vec![0.0f64; n_coeffs];
            coeffs[0] = dc_values[ch_idx][hex_idx];

            for k in 1..=max_harmonics {
                if k >= n_coeffs { break; }
                let ac_idx = ac_cursor[ch_idx] + (k - 1);
                if ac_idx < ac_flat[ch_idx].len() {
                    let golden_step = hex_step * (k as f64 * phi).powf(0.55);
                    let step = golden_step.max(0.1);
                    coeffs[k] = ac_flat[ch_idx][ac_idx] as f64 * step;
                }
            }
            ac_cursor[ch_idx] += max_harmonics;

            let perimeter_values = idct_ii_1d(&coeffs, n_peri);

            let interior_positions: Vec<(f64, f64)> = shape
                .interior
                .iter()
                .map(|&(x, y)| (x as f64, y as f64))
                .collect();

            let vertex_positions: Vec<(f64, f64)> = (0..6)
                .map(|i| {
                    let angle = std::f64::consts::PI / 3.0 * i as f64;
                    (cx + rf * angle.cos(), cy + rf * angle.sin())
                })
                .collect();

            let codons: Vec<VertexCodon> = if ch_idx == 0 && encode_codons_for_radius(r) {
                let base = codon_offsets[hex_idx];
                let n_stored = if codons_per_hex_flag == 0 {
                    match subdivisions[hex_idx] {
                        HexSubdiv::Whole => 2,
                        _ => 6,
                    }
                } else {
                    codons_per_hex_flag
                };
                (0..6)
                    .map(|i| {
                        if i < n_stored {
                            let ci = base + i;
                            if ci < codon_bytes.len() {
                                unpack_codon(codon_bytes[ci])
                            } else {
                                VertexCodon::default()
                            }
                        } else {
                            VertexCodon::default()
                        }
                    })
                    .collect()
            } else {
                vec![VertexCodon::default(); 6]
            };

            let subdiv = subdivisions[hex_idx];
            let interior_vals = reconstruct_hex_interior_subdivided_r(
                hw,
                &perimeter_values,
                &interior_positions,
                &vertex_positions,
                &codons,
                subdiv,
                r,
            );

            perimeter_values_ch.push(perimeter_values);
            interior_vals_ch.push(interior_vals);
        }

        // Build lookup maps for this shape
        let interior_lookup: std::collections::HashMap<(i32, i32), usize> = shape
            .interior
            .iter()
            .enumerate()
            .map(|(i, &pt)| (pt, i))
            .collect();
        let peri_lookup: std::collections::HashMap<(i32, i32), usize> = shape
            .perimeter
            .iter()
            .enumerate()
            .map(|(i, &pt)| (pt, i))
            .collect();

        let mean_peri: Vec<f64> = (0..3)
            .map(|ch| {
                let pv = &perimeter_values_ch[ch];
                if pv.is_empty() { 0.0 }
                else { pv.iter().sum::<f64>() / pv.len() as f64 }
            })
            .collect();

        for &(dx, dy) in &shape.voronoi {
            let px = (cx + dx as f64).round() as isize;
            let py = (cy + dy as f64).round() as isize;
            if px < 0 || px >= width as isize || py < 0 || py >= height as isize {
                continue;
            }
            let idx = py as usize * width + px as usize;
            if pixel_written[idx] {
                continue;
            }
            for ch in 0..3 {
                let val = if let Some(&peri_idx) = peri_lookup.get(&(dx, dy)) {
                    if peri_idx < perimeter_values_ch[ch].len() {
                        perimeter_values_ch[ch][peri_idx]
                    } else { mean_peri[ch] }
                } else if let Some(&int_idx) = interior_lookup.get(&(dx, dy)) {
                    if int_idx < interior_vals_ch[ch].len() {
                        interior_vals_ch[ch][int_idx]
                    } else { mean_peri[ch] }
                } else {
                    mean_peri[ch]
                };
                output_planes[ch][idx] = val;
            }
            pixel_written[idx] = true;
        }
    }

    // 13. Apply interior residuals (only for hexes with encode_residual_for_radius)
    {
        for ch_idx in 0..3 {
            if pos + 4 > body.len() { break; }
            let resid_stream_size = read_u32(body, &mut pos) as usize;
            if resid_stream_size == 0 || pos + resid_stream_size > body.len() {
                pos += resid_stream_size;
                continue;
            }

            // Total residual pixels = sum of interior pixels per cell (only cells with residuals)
            let total_resid: usize = cells.iter()
                .filter(|&&(_, _, r)| encode_residual_for_radius(r))
                .map(|&(_, _, r)| shapes[&r].interior.len())
                .sum();
            let (resid_decoded, _) = rans::rans_decode_band(
                &body[pos..pos + resid_stream_size],
                total_resid,
            );
            pos += resid_stream_size;

            let mut resid_idx = 0usize;
            for &(cx, cy, r) in &cells {
                // Skip residuals for large hexes (R=13) — must match encoder
                if !encode_residual_for_radius(r) {
                    continue;
                }
                let resid_step = detail_step * resid_step_factor(r);
                let shape = &shapes[&r];
                for &(dx, dy) in &shape.interior {
                    let px = (cx + dx as f64).round() as isize;
                    let py = (cy + dy as f64).round() as isize;
                    if px >= 0 && px < width as isize && py >= 0 && py < height as isize {
                        let idx = py as usize * width + px as usize;
                        if resid_idx < resid_decoded.len() {
                            output_planes[ch_idx][idx] += resid_decoded[resid_idx] as f64 * resid_step;
                        }
                    }
                    resid_idx += 1;
                }
            }
        }
    }
    let _ = pos;

    // 14. Fill uncovered pixels
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if !pixel_written[idx] {
                let mut best_dist = f64::MAX;
                let mut best_val = [0.0f64; 3];
                let search_r = 15isize;
                for dy in -search_r..=search_r {
                    for ddx in -search_r..=search_r {
                        let nx = x as isize + ddx;
                        let ny = y as isize + dy;
                        if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
                            let nidx = ny as usize * width + nx as usize;
                            if pixel_written[nidx] {
                                let d = (ddx * ddx + dy * dy) as f64;
                                if d < best_dist {
                                    best_dist = d;
                                    for ch in 0..3 {
                                        best_val[ch] = output_planes[ch][nidx];
                                    }
                                }
                            }
                        }
                    }
                }
                for ch in 0..3 {
                    output_planes[ch][idx] = best_val[ch];
                }
            }
        }
    }

    // 15. Inverse PTF on L channel
    {
        use golden::PTF_GAMMA_INV;
        let inv255 = 1.0 / 255.0;
        for v in output_planes[0].iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA_INV);
        }
    }

    // CAS sharpening on luminance only
    sharpen_luminance_planes(&mut output_planes, height, width);

    // 16. Inverse GCT -> RGB
    let (mut r_plane, mut g_plane, mut b_plane) = color::golden_rotate_inverse(
        &output_planes[0],
        &output_planes[1],
        &output_planes[2],
        n_pixels,
    );

    // Post-decode CASP sharpening

    // 17. Schrödinger collapse: continuous → certain via error diffusion.
    // The wavefunction of possible pixel values collapses into definite u8.
    // Floyd-Steinberg propagates rounding error → smooth gradients, no banding.
    let r_out = dsp::schrodinger_collapse(&r_plane, height, width);
    let g_out = dsp::schrodinger_collapse(&g_plane, height, width);
    let b_out = dsp::schrodinger_collapse(&b_plane, height, width);

    let mut rgb_out = Vec::with_capacity(n_pixels * 3);
    for i in 0..n_pixels {
        rgb_out.push(r_out[i]);
        rgb_out.push(g_out[i]);
        rgb_out.push(b_out[i]);
    }

    Ok(DecodedImage {
        rgb: rgb_out,
        width,
        height,
    })
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

    // Pass 3: Multi-scale chaperone — inter-level propagation
    if header.version >= 1 {
        // Pass 3a: Chaperone — propagate structural energy deep->fine
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
    sharpen_luminance(&mut l_flat, height, width);
    let n = width * height;
    let (mut r_plane, mut g_plane, mut b_plane) = color::golden_rotate_inverse(&l_flat, &c1_full, &c2_full, n);

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
            LOT_BLOCK_SIZE, false, // v2: no overlap
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

    // CAS sharpening on luminance only, then GCT inverse -> RGB
    sharpen_luminance(&mut l_flat, height, width);
    let n = width * height;
    let (mut r_plane, mut g_plane, mut b_plane) = color::golden_rotate_inverse(&l_flat, &c1_full, &c2_full, n);

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
    let use_quality_adaptive = flags & aurea_encoder::FLAG_QUALITY_ADAPTIVE != 0;
    let use_overlap = flags & aurea_encoder::FLAG_LOT_OVERLAP != 0;
    let use_variable = flags & aurea_encoder::FLAG_VARIABLE_BLOCKS != 0;
    let use_weber_trna = flags & aurea_encoder::FLAG_WEBER_TRNA != 0;

    // Read block map if variable blocks
    let block_map = if use_variable {
        let bgh = read_u16(body, &mut pos) as usize;
        let bgw = read_u16(body, &mut pos) as usize;
        let map_len = read_u16(body, &mut pos) as usize;
        let map_bytes = &body[pos..pos + map_len];
        pos += map_len;
        let size_grid = lot::decode_block_map(map_bytes, bgh * bgw);
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
            // Golden DPCM in raster order (must match encoder).
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
        let total_ac = if let Some(ref blocks) = var_blocks {
            blocks.iter().map(|&(_, _, bs)| bs * bs - 1).sum()
        } else {
            n_blocks * LOT_AC_PER_BLOCK
        };
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

    // Structural codon map (must match encoder exactly).
    // For variable blocks, use the block-map grid dimensions for codon.
    let codon_gh = if let Some((_, bgh, _)) = block_map { bgh } else { l_raw.grid_h };
    let codon_gw = if let Some((_, _, bgw)) = block_map { bgw } else { l_raw.grid_w };
    // Average block size for DC→pixel conversion (approximate for variable)
    let avg_bs2 = (LOT_BLOCK_SIZE * LOT_BLOCK_SIZE) as f64;
    let l_dc_pixel: Vec<f64> = l_dc_denorm.iter().map(|&v| v / avg_bs2).collect();
    let l_codon_map = lot::codon_structural_map(&l_dc_pixel, l_raw.grid_h, l_raw.grid_w);
    // Foveal saliency map: same computation as encoder (zero bpp)
    let l_foveal_map = lot::foveal_saliency_map(&l_dc_pixel, l_raw.grid_h, l_raw.grid_w);

    // Precompute zigzag orders for each block size
    let zz_8 = ac_zigzag_order(8);
    let zz_16 = ac_zigzag_order(16);
    let zz_32 = ac_zigzag_order(32);

    // Reconstruct each channel
    let mut reconstructed_channels: Vec<Vec<f64>> = Vec::with_capacity(3);

    for (ch_idx, raw) in raw_channels.iter().enumerate() {
        let info = &channel_infos[ch_idx];
        let dc_range = (info.dc_max - info.dc_min).max(1e-6);
        let n_blocks = raw.grid_h * raw.grid_w;

        let dc_denorm: Vec<f64> = raw.dc_q.iter().map(|&q|
            q as f64 * dc_step_clamped * dc_range / 255.0 + info.dc_min
        ).collect();

        let ac_step = detail_step * info.chroma_factor * lot_global_factor;
        let mut ac_blocks: Vec<Vec<f64>> = Vec::with_capacity(n_blocks);

        let mut ac_cursor = 0usize; // position in flat AC stream

        for block_idx in 0..n_blocks {
            // Determine block size
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
            // Build inverse zigzag for this block size
            let mut inv_zz = vec![0usize; ac_per_block];
            for (fp, &bi) in zz.iter().enumerate() {
                if fp < ac_per_block { inv_zz[fp] = bi - 1; }
            }

            let mut ac = vec![0.0f64; ac_per_block];

            // Foveal saliency factor (matches encoder)
            let l_fov_idx = (block_idx as f64 * l_foveal_map.len() as f64
                            / n_blocks.max(1) as f64) as usize;
            let foveal_factor = l_foveal_map[l_fov_idx.min(l_foveal_map.len().saturating_sub(1))];

            let l_idx = (block_idx as f64 * l_dc_pixel.len() as f64
                        / n_blocks.max(1) as f64) as usize;
            let dc_l = l_dc_pixel[l_idx.min(l_dc_pixel.len().saturating_sub(1))];

            let local_step = ac_step * foveal_factor;
            let step_clamped = local_step.max(0.1);

            for flat_pos in 0..ac_per_block {
                let qi = if ac_cursor + flat_pos < raw.ac_q.len() {
                    raw.ac_q[ac_cursor + flat_pos]
                } else { 0 };
                let ac_idx = inv_zz[flat_pos];
                let block_pos = zz[flat_pos];
                let row = block_pos / block_size;
                let col = block_pos % block_size;
                let mut qfactor = lot::qmat_for_block_size(row, col, block_size).max(0.1).powf(qmat_power);

                if use_csf {
                    let csf = lot::csf_qmat_factor(row, col, block_size, dc_l);
                    qfactor *= csf;
                }

                ac[ac_idx] = qi as f64 * step_clamped * qfactor;
            }
            ac_cursor += ac_per_block;

            // Point 4: If structural flag, apply structural coherence correction
            // The decoder uses the dequantized AC to compute structural factor,
            // then scales accordingly. Since encoder used 4D factor on step,
            // decoder must match. The 3D factor is already applied above.
            // Structural factor was baked into the encoder's quantization step,
            // so the decoder automatically recovers it through the same dequant formula.
            // No additional correction needed here.

            ac_blocks.push(ac);
        }

        // Fibonacci spectral spin (only for fixed-size blocks — variable AC sizes are incompatible)
        if !use_variable {
            let local_steps: Vec<f64> = (0..n_blocks).map(|bi| {
                let l_map_idx = (bi as f64 * l_codon_map.len() as f64
                                / n_blocks.max(1) as f64) as usize;
                let codon = l_codon_map[l_map_idx.min(l_codon_map.len().saturating_sub(1))];
                let l_idx = (bi as f64 * l_dc_pixel.len() as f64
                            / n_blocks.max(1) as f64) as usize;
                let dc_l = l_dc_pixel[l_idx.min(l_dc_pixel.len().saturating_sub(1))];
                let trna = if use_weber_trna { lot::luminance_trna(dc_l) } else { 1.0 };
                ac_step * codon * trna
            }).collect();
            let spin_w = if use_quality_adaptive {
                calibration::spin_weight_for_quality(header.quality)
            } else {
                0.382
            };
            spin::fibonacci_spectral_spin(
                &mut ac_blocks, raw.grid_h, raw.grid_w,
                LOT_AC_PER_BLOCK, &local_steps, spin_w,
            );
        }

        // LOT synthesis: variable or fixed
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

    // Gas-only deblocking at actual block boundaries.
    // With 16/32 adaptive mesh: deblock at 16px (catches both 16×16 and 32×32 boundaries).
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

    // DC interpolation: disabled with variable blocks (incompatible grid mapping).
    // The 32×32 blocks on smooth areas already provide naturally smoother DC
    // (larger averaging = less quantization noise per pixel).

    // DNA4: Ternary supercordes sharpening — decoder-side, zero bpp.
    // Sharpens Segment/Arc blocks, leaves Rien untouched.
    dsp::supercordes_sharpen(
        &mut l_flat, height, width,
        &l_dc_pixel, l_raw.grid_h, l_raw.grid_w,
        LOT_BLOCK_SIZE,
        crate::golden::PHI_INV2,
    );

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

    // CAS sharpening on luminance only, then GCT inverse -> RGB
    sharpen_luminance(&mut l_flat, height, width);
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
// v12 Bayesian hierarchy decoder
// ======================================================================

/// v12 Bayesian hierarchy decoder (AUR2 version 12).
///
/// Single-pass decoder (like v3) with v12-specific additions:
/// - TuringHeader parsed after flags+scene_type+reserved
/// - TuringField reconstructed from decoded DC (zero-bit, same computation as encoder)
/// - step_modulation applied to AC dequantization
/// - Match section parsed (L channel only)
/// - EOB positions per block + rans_decode_band_v12 with turing buckets
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
        let grid_h = read_u16(body, &mut pos) as usize;
        let grid_w = read_u16(body, &mut pos) as usize;
        let n_blocks = grid_h * grid_w;

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

        // Read EOB positions: DPCM + rANS compressed
        let eob_count = read_u32(body, &mut pos) as usize;
        let eob_rans_size = read_u32(body, &mut pos) as usize;
        let eob_data = &body[pos..pos + eob_rans_size];
        pos += eob_rans_size;
        let mut eob_positions: Vec<u16> = Vec::with_capacity(eob_count);
        {
            let eob_turing = vec![0u8; eob_count];
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

        // Uniform bucket 0 — matches encoder, avoids 128-context fragmentation
        let ac_turing_vec: Vec<u8> = if turing_field.is_some() {
            vec![0u8; total_ac]
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

            for flat_pos in 0..ac_per_block {
                let qi = if ac_cursor + flat_pos < raw.ac_q.len() {
                    raw.ac_q[ac_cursor + flat_pos]
                } else { 0 };
                let ac_idx = inv_zz[flat_pos];
                let block_pos = zz[flat_pos];
                let row = block_pos / block_size;
                let col = block_pos % block_size;
                let mut qfactor = lot::qmat_for_block_size(row, col, block_size)
                    .max(0.1).powf(qmat_power);

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