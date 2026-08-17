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
pub mod spectral_envelope;
pub mod timing;
pub mod trellis;
pub mod turing;

// wavelet.rs kept for morton_order() and DEAD_ZONE (used by v13 encoder)
pub mod wavelet;

pub use error::AureaError;

const MAX_IMAGE_PIXELS: usize = 100_000_000;

struct CheckedReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> CheckedReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn position(&self) -> usize {
        self.pos
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_u8(&mut self, section: &str) -> Result<u8, AureaError> {
        Ok(self.take(1, section)?[0])
    }

    fn read_u16(&mut self, section: &str) -> Result<u16, AureaError> {
        let b = self.take(2, section)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self, section: &str) -> Result<u32, AureaError> {
        let b = self.take(4, section)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn take(&mut self, len: usize, section: &str) -> Result<&'a [u8], AureaError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| AureaError::InvalidFormat(format!("{section} length overflows")))?;
        if end > self.data.len() {
            return Err(AureaError::InvalidFormat(format!(
                "truncated {section}: need {len} bytes, only {} remain",
                self.remaining(),
            )));
        }
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }
}

fn invalid_format(message: impl Into<String>) -> AureaError {
    AureaError::InvalidFormat(message.into())
}

fn decode_rans_v12(
    data: &[u8],
    n_symbols: usize,
    contexts: &[u8],
    section: &str,
) -> Result<Vec<i16>, AureaError> {
    rans::rans_decode_band_v12(data, n_symbols, contexts)
        .map(|(symbols, _)| symbols)
        .map_err(|e| AureaError::DecodeFailed(format!("invalid {section}: {e}")))
}

/// Decoded RGB image.
pub struct DecodedImage {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub progressive: Option<ProgressiveDecodeInfo>,
}

/// Location and extent of the progressive prefix used for a reconstruction.
pub struct ProgressiveDecodeInfo {
    pub total_layers: usize,
    pub decoded_layers: usize,
    pub prefix_bytes: usize,
}

/// Apply CAS sharpening on the luminance plane (f64 [0,255]).
/// Converts to f32 [0,1] for postprocess::apply_cas_sharpening, then back.
fn sharpen_luminance(l_plane: &mut [f64], h: usize, w: usize) {
    let inv255 = 1.0 / 255.0;
    let l_f32: Vec<f32> = l_plane
        .iter()
        .map(|&v| (v * inv255).clamp(0.0, 1.0) as f32)
        .collect();
    let sharpened = postprocess::apply_cas_sharpening(&l_f32, w, h, 0.8);
    for (dst, &src) in l_plane.iter_mut().zip(sharpened.iter()) {
        *dst = (src as f64 * 255.0).clamp(0.0, 255.0);
    }
}

/// Apply CAS sharpening on the luminance plane (f64 [0,255]) stored in output_planes[0].
fn sharpen_luminance_planes(output_planes: &mut [Vec<f64>], h: usize, w: usize) {
    let inv255 = 1.0 / 255.0;
    let l_f32: Vec<f32> = output_planes[0]
        .iter()
        .map(|&v| (v * inv255).clamp(0.0, 1.0) as f32)
        .collect();
    let sharpened = postprocess::apply_cas_sharpening(&l_f32, w, h, 0.8);
    for (dst, &src) in output_planes[0].iter_mut().zip(sharpened.iter()) {
        *dst = (src as f64 * 255.0).clamp(0.0, 255.0);
    }
}

pub fn decode_aurea(file_data: &[u8]) -> Result<DecodedImage, AureaError> {
    decode_aurea_with_progressive_layers(file_data, None)
}

/// Decode an AUREA image, optionally retaining only the first `n`
/// progressive AC packets. `Some(0)` produces the common DC base layer.
pub fn decode_aurea_with_progressive_layers(
    file_data: &[u8],
    progressive_layers: Option<usize>,
) -> Result<DecodedImage, AureaError> {
    if file_data.len() < 4 || &file_data[0..4] != bitstream::AUR2_MAGIC {
        return Err(invalid_format("expected AUR2 magic"));
    }
    let (header, _) = bitstream::parse_aur2_header(file_data)?;
    // AUR2 deliberately has no legacy decoder path. v21 adds the spectral
    // amplitude/phase envelope on top of the v20 semantics: joint color-codon
    // RDO, C1 Fourier phase continuation, the C1->C2 phase residual and the
    // 1.75 luma exchange. v20 stays readable because the envelope's neutral
    // state is exactly that grammar.
    // v25 keys the AC entropy contexts on intra-block activity (the EOB
    // position) instead of the Turing bucket.
    // v28 relieves the luma pivot and the chroma exchange slope on the
    // diffuse-dark field; v29 adds the quality-keyed perceptual texture law
    // (encoder allocation + decoder post chain). The decoder applies both
    // from the version alone, never from the process environment.
    if (20..=32).contains(&header.version) {
        return decode_aur2_v13(file_data, progressive_layers);
    }
    Err(AureaError::UnsupportedVersion(header.version))
}

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

fn spectral_parent_positions(zigzag: &[usize], block_size: usize) -> Vec<Option<usize>> {
    let mut rank = vec![usize::MAX; block_size * block_size];
    for (position, &natural_idx) in zigzag.iter().enumerate() {
        rank[natural_idx] = position;
    }
    zigzag
        .iter()
        .map(|&natural_idx| {
            let row = natural_idx / block_size;
            let col = natural_idx % block_size;
            let parent_row = row / 2;
            let parent_col = col / 2;
            if parent_row == 0 && parent_col == 0 {
                None
            } else {
                let position = rank[parent_row * block_size + parent_col];
                (position != usize::MAX).then_some(position)
            }
        })
        .collect()
}

fn validate_adaptive_block_map(
    size_grid: &[u8],
    grid_h: usize,
    grid_w: usize,
) -> Result<(), AureaError> {
    let expected = grid_h
        .checked_mul(grid_w)
        .ok_or_else(|| invalid_format("block-map dimensions overflow"))?;
    if size_grid.len() != expected {
        return Err(invalid_format("block-map symbol count mismatch"));
    }
    let mut visited = vec![false; expected];
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let anchor = gy * grid_w + gx;
            if visited[anchor] {
                continue;
            }
            let size = size_grid[anchor];
            if !matches!(size, 8 | 16 | 32) {
                return Err(invalid_format(format!("unsupported block size {size}")));
            }
            let span = size as usize / 8;
            for cy in gy..(gy + span).min(grid_h) {
                for cx in gx..(gx + span).min(grid_w) {
                    let idx = cy * grid_w + cx;
                    if visited[idx] || size_grid[idx] != size {
                        return Err(invalid_format("inconsistent adaptive block tiling"));
                    }
                    visited[idx] = true;
                }
            }
        }
    }
    if visited.iter().any(|&v| !v) {
        return Err(invalid_format(
            "adaptive block map contains uncovered cells",
        ));
    }
    Ok(())
}

fn decode_aur2_v13(
    file_data: &[u8],
    progressive_layers: Option<usize>,
) -> Result<DecodedImage, AureaError> {
    use aurea_encoder::{
        FLAG_BAYESIAN_HIERARCHY, FLAG_CFL, FLAG_CHROMA_BREATHING, FLAG_CHROMA_RESIDUAL,
        FLAG_CSF_MODULATION, FLAG_DPCM_DC, FLAG_FOLDED_SPECTRAL_SCAN, FLAG_LOT_OVERLAP,
        FLAG_PHYLLOTACTIC_SCAN, FLAG_PROGRESSIVE_PACKETS, FLAG_QUALITY_ADAPTIVE,
        FLAG_SPIRAL_ARC_SCAN, FLAG_VARIABLE_BLOCKS, FLAG_WEBER_TRNA,
    };

    let (header, header_size) = bitstream::parse_aur2_header(file_data)?;
    let width = header.width;
    let height = header.height;
    let detail_step = header.detail_step;
    let ll_ranges = header.ll_ranges;

    if width == 0 || height == 0 {
        return Err(invalid_format("image dimensions must be non-zero"));
    }
    let n_pixels = width
        .checked_mul(height)
        .ok_or_else(|| invalid_format("image dimensions overflow"))?;
    if n_pixels > MAX_IMAGE_PIXELS {
        return Err(invalid_format(format!(
            "image has {n_pixels} pixels; decoder limit is {MAX_IMAGE_PIXELS}",
        )));
    }
    if !(1..=100).contains(&header.quality) {
        return Err(invalid_format("quality must be in the range 1..=100"));
    }
    if !detail_step.is_finite() || !(0.01..=1024.0).contains(&detail_step) {
        return Err(invalid_format(
            "detail_step is outside the safe range 0.01..=1024",
        ));
    }
    for &(min, max) in &ll_ranges {
        if !min.is_finite()
            || !max.is_finite()
            || min > max
            || min.abs() > 1_000_000.0
            || max.abs() > 1_000_000.0
        {
            return Err(invalid_format("invalid DC range in AUR2 header"));
        }
    }

    let body = &file_data[header_size..];
    let mut reader = CheckedReader::new(body);

    // 1. Parse body header: chroma dims (8 bytes)
    let c1_h = reader.read_u16("C1 height")? as usize;
    let c1_w = reader.read_u16("C1 width")? as usize;
    let c2_h = reader.read_u16("C2 height")? as usize;
    let c2_w = reader.read_u16("C2 width")? as usize;
    if c1_h == 0
        || c1_w == 0
        || c2_h == 0
        || c2_w == 0
        || c1_h > height
        || c1_w > width
        || c2_h > height
        || c2_w > width
    {
        return Err(invalid_format("invalid chroma dimensions"));
    }

    // Flags (u16) + scene_type/bright-compensation (u8) + chroma-breathing (u8)
    let flags = reader.read_u16("feature flags")?;
    // Lower 3 bits: scene type. Upper 5 bits: bright-compensation strength
    // (« contre-don de Weber »), zero in legacy v13 files = neutral.
    let scene_and_bright = reader.read_u8("scene type")?;
    let _scene_type = scene_and_bright & 0x07;
    let bright_strength = (scene_and_bright >> 3) as f64 / 31.0;
    let chroma_breathing_code = reader.read_u8("chroma-breathing strength")?;
    if flags & FLAG_CHROMA_BREATHING == 0 && chroma_breathing_code != 0 {
        return Err(invalid_format(
            "chroma-breathing strength requires its feature flag",
        ));
    }

    let use_csf = flags & FLAG_CSF_MODULATION != 0;
    let use_chroma_resid = flags & FLAG_CHROMA_RESIDUAL != 0;
    let use_dpcm = flags & FLAG_DPCM_DC != 0;
    let use_quality_adaptive = flags & FLAG_QUALITY_ADAPTIVE != 0;
    let use_overlap = flags & FLAG_LOT_OVERLAP != 0;
    let use_variable = flags & FLAG_VARIABLE_BLOCKS != 0;
    let use_bayesian = flags & FLAG_BAYESIAN_HIERARCHY != 0;
    let _use_weber_trna = flags & FLAG_WEBER_TRNA != 0;
    let use_cfl = flags & FLAG_CFL != 0;
    let use_phyllotactic_scan = flags & FLAG_PHYLLOTACTIC_SCAN != 0;
    let use_spiral_arc_scan = flags & FLAG_SPIRAL_ARC_SCAN != 0;
    let use_folded_spectral_scan = flags & FLAG_FOLDED_SPECTRAL_SCAN != 0;
    let use_progressive_packets = flags & FLAG_PROGRESSIVE_PACKETS != 0;
    let use_chroma_breathing = flags & FLAG_CHROMA_BREATHING != 0;
    if use_chroma_breathing && chroma_breathing_code == 0 {
        return Err(invalid_format(
            "chroma-breathing flag requires a non-zero strength",
        ));
    }
    let chroma_breathing_strength = chroma_breathing_code as f64 / 255.0;
    if use_phyllotactic_scan && use_spiral_arc_scan {
        return Err(invalid_format("multiple spatial AC scans are enabled"));
    }
    if use_progressive_packets && (use_phyllotactic_scan || use_spiral_arc_scan) {
        return Err(invalid_format(
            "progressive packets cannot be combined with a spatial AC scan",
        ));
    }
    if progressive_layers.is_some() && !use_progressive_packets {
        return Err(invalid_format(
            "a progressive layer limit requires a packetized AUREA file",
        ));
    }
    if !use_bayesian || !use_variable {
        return Err(invalid_format(
            "v13 requires Bayesian rANS and adaptive blocks",
        ));
    }

    // 2. Parse TuringHeader (16 bytes) when FLAG_BAYESIAN_HIERARCHY is set
    let turing_hdr = if use_bayesian {
        let bytes = reader.take(16, "Turing header")?;
        let (hdr, _) = bitstream::TuringHeader::parse(bytes)?;
        if !hdr.sigma_a.is_finite()
            || !hdr.sigma_ratio.is_finite()
            || !hdr.k_step.is_finite()
            || !hdr.mag_calibration.is_finite()
            || !(0.01..=16.0).contains(&hdr.sigma_a)
            || !(0.1..=16.0).contains(&hdr.sigma_ratio)
            || !(0.0..=4.0).contains(&hdr.k_step)
            || !(0.0..=100.0).contains(&hdr.mag_calibration)
        {
            return Err(invalid_format("invalid Turing parameters"));
        }
        Some(hdr)
    } else {
        None
    };

    // 3. Read block map if variable blocks — rANS compressed
    let block_map = if use_variable {
        if c1_h != height || c1_w != width || c2_h != height || c2_w != width {
            return Err(invalid_format("v13 requires 4:4:4 chroma dimensions"));
        }
        let bgh = reader.read_u16("block-map height")? as usize;
        let bgw = reader.read_u16("block-map width")? as usize;
        let expected_bgh = height.div_ceil(8);
        let expected_bgw = width.div_ceil(8);
        if bgh != expected_bgh || bgw != expected_bgw {
            return Err(invalid_format(format!(
                "block-map dimensions {bgh}x{bgw} do not match expected {expected_bgh}x{expected_bgw}",
            )));
        }
        let map_len = reader.read_u16("block-map length")? as usize;
        let n_cells = bgh
            .checked_mul(bgw)
            .ok_or_else(|| invalid_format("block-map dimensions overflow"))?;
        let map_data = reader.take(map_len, "compressed block map")?;
        let map_ctx = vec![0u8; n_cells];
        let map_symbols = decode_rans_v12(map_data, n_cells, &map_ctx, "block map")?;
        let mut size_grid = Vec::with_capacity(n_cells);
        for symbol in map_symbols {
            size_grid.push(match symbol {
                0 => 8,
                1 => 16,
                2 => 32,
                _ => return Err(invalid_format("block map contains an invalid symbol")),
            });
        }
        validate_adaptive_block_map(&size_grid, bgh, bgw)?;
        Some((size_grid, bgh, bgw))
    } else {
        None
    };

    // Precompute block info for variable mode
    let var_blocks: Option<Vec<(usize, usize, usize)>> = block_map
        .as_ref()
        .map(|(sg, bgh, bgw)| lot::iter_blocks(sg, *bgh, *bgw, use_overlap));

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

    eprintln!(
        "  AUR2 v{} decode: {}x{}, flags=0x{:04x}, bayesian={}",
        header.version, width, height, flags, use_bayesian
    );

    // Precompute AC zigzag orders for each block size
    let zz_8 = if use_folded_spectral_scan {
        scan::fixed_pitch_folded_spiral_order(8)
    } else {
        ac_zigzag_order(8)
    };
    let zz_16 = if use_folded_spectral_scan {
        scan::fixed_pitch_folded_spiral_order(16)
    } else {
        ac_zigzag_order(16)
    };
    let zz_32 = if use_folded_spectral_scan {
        scan::fixed_pitch_folded_spiral_order(32)
    } else {
        ac_zigzag_order(32)
    };
    let spectral_parents_8 = spectral_parent_positions(&zz_8, 8);
    let spectral_parents_16 = spectral_parent_positions(&zz_16, 16);
    let spectral_parents_32 = spectral_parent_positions(&zz_32, 32);

    // Cycle 20 spectral envelope. A v20 stream is decoded with the neutral
    // configuration and never evaluates a neighbourhood; a v21 stream carries
    // the laboratory grammar and therefore requires the same configuration on
    // both sides until the experiment is promoted to fixed semantics.
    let spectral_config = if header.version >= 21 {
        let mut config = *calibration::TUNABLE_SPECTRAL_ENVELOPE;
        if header.version < 22 {
            config.mix_rate = 0;
        } else if header.version < 23 {
            // A v22 stream carries exactly one secondary opinion, on P(zero).
            config.mix_inputs = spectral_envelope::MIX_INPUT_FOREIGN;
            config.mix_pm1 = false;
            config.apm_rate = 0;
            config.mix_sign = false;
        } else if header.version < 24 {
            // v23 mixes three opinions and leaves the sign to its own counters.
            config.mix_inputs = spectral_envelope::MIX_INPUT_FOREIGN
                | spectral_envelope::MIX_INPUT_SCALED
                | spectral_envelope::MIX_INPUT_BAND;
            config.mix_sign = false;
        }
        config
    } else {
        spectral_envelope::SpectralEnvelopeConfig::neutral()
    };
    let use_spectral_envelope = !spectral_config.is_neutral();
    if header.version >= 21 && !use_spectral_envelope {
        return Err(invalid_format(
            "AUR2 v21 requires the spectral envelope configuration used at encode time \
             (AUREA_SPECTRAL_ENVELOPE_*)",
        ));
    }
    // The mixed grammar is a different engine, not a different context: reading
    // a v22+ stream without it, or a v21 stream with it, cannot half-work.
    if (header.version >= 22) != (spectral_config.mix_rate > 0) {
        return Err(invalid_format(
            "AUR2 v22+ requires the same logistic mixing rate as the encoder \
             (AUREA_SPECTRAL_ENVELOPE_MIX)",
        ));
    }
    let spectral_tables = spectral_envelope::SpectralNeighborTables::new(&zz_8, &zz_16, &zz_32);

    struct V13ChannelInfo {
        chan_h: usize,
        chan_w: usize,
        dc_min: f64,
        dc_max: f64,
        chroma_factor: f64,
    }

    let channel_infos = [
        V13ChannelInfo {
            chan_h: height,
            chan_w: width,
            dc_min: ll_ranges[0].0 as f64,
            dc_max: ll_ranges[0].1 as f64,
            chroma_factor: 1.0,
        },
        V13ChannelInfo {
            chan_h: c1_h,
            chan_w: c1_w,
            dc_min: ll_ranges[1].0 as f64,
            dc_max: ll_ranges[1].1 as f64,
            chroma_factor: *calibration::TUNABLE_CHROMA_PHI,
        },
        V13ChannelInfo {
            chan_h: c2_h,
            chan_w: c2_w,
            dc_min: ll_ranges[2].0 as f64,
            dc_max: ll_ranges[2].1 as f64,
            chroma_factor: 1.0,
        },
    ];

    let dc_step = (detail_step * 0.1).max(0.2);
    let dc_step_clamped = dc_step.max(0.1);

    // === SINGLE-PASS: Read DC+AC per channel, decode, dequantize, synthesize ===
    //
    // Channel 0 (L) is processed first to compute the TuringField needed for
    // AC decoding of all channels. The TuringField is derived from the L-channel
    // decoded DC, so it is available after reading ch0's DC.

    struct V13ChannelRaw {
        grid_h: usize,
        grid_w: usize,
        dc_q: Vec<i16>,
        ac_q: Vec<i16>,
    }

    let mut raw_channels: Vec<V13ChannelRaw> = Vec::with_capacity(3);
    let mut turing_field: Option<crate::turing::TuringField> = None;
    let mut block_turing_buckets: Option<Vec<u8>> = None;
    let mut block_scan_order: Option<Vec<usize>> = if use_phyllotactic_scan || use_spiral_arc_scan {
        None
    } else {
        Some((0..var_blocks.as_ref().map_or(0, Vec::len)).collect())
    };
    let mut progressive_packets: Option<Vec<Vec<usize>>> = None;

    // CfL per-chroma-channel data: [ch1_flags, ch1_alphas], [ch2_flags, ch2_alphas]
    struct CflChannelData {
        flags: Vec<bool>,
        alpha_indices: Vec<usize>,
    }
    let mut cfl_data: Vec<CflChannelData> = Vec::new(); // index 0 = ch1, index 1 = ch2
    struct PendingProgressiveRaw {
        grid_h: usize,
        grid_w: usize,
        dc_q: Vec<i16>,
    }
    let mut pending_progressive: Vec<PendingProgressiveRaw> = Vec::new();
    let mut partial_progressive = false;
    let mut progressive_info: Option<ProgressiveDecodeInfo> = None;

    for ch_idx in 0..3usize {
        // v13 stores n_blocks as u32 (supports >65535 blocks for large images).
        let n_blocks = reader.read_u32("channel block count")? as usize;
        let info = &channel_infos[ch_idx];
        let stride = lot::lot_stride(LOT_BLOCK_SIZE, use_overlap);
        let fixed_grid_h = info.chan_h.div_ceil(stride);
        let fixed_grid_w = info.chan_w.div_ceil(stride);
        let expected_n_blocks = var_blocks.as_ref().map_or_else(
            || fixed_grid_h.saturating_mul(fixed_grid_w),
            |blocks| blocks.len(),
        );
        if n_blocks != expected_n_blocks {
            return Err(invalid_format(format!(
                "channel {ch_idx} block count {n_blocks} does not match expected {expected_n_blocks}",
            )));
        }
        let (grid_h, grid_w) = if let Some(ref vb) = var_blocks {
            (vb.len(), 1)
        } else {
            (fixed_grid_h, fixed_grid_w)
        };

        // DC via rANS v12 (Exp-Golomb)
        let t_dc = crate::timing::stage("dec:rans_dc");
        let dc_rans_size = reader.read_u32("DC rANS length")? as usize;
        let dc_data = reader.take(dc_rans_size, "DC rANS stream")?;
        let dc_ctx = vec![0u8; n_blocks];
        let dc_ordered = decode_rans_v12(dc_data, n_blocks, &dc_ctx, "DC stream")?;

        // Golden DPCM reconstruction (same as v3)
        let dc_q = if use_dpcm {
            let mut abs_dc = vec![0i16; n_blocks];
            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    let pred = aurea_encoder::golden_dc_predict(&abs_dc, gy, gx, grid_w);
                    let value = dc_ordered[gy * grid_w + gx] as i32 + pred as i32;
                    abs_dc[gy * grid_w + gx] = i16::try_from(value)
                        .map_err(|_| invalid_format("DC prediction overflows i16"))?;
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
        } else {
            dc_ordered
        };

        drop(t_dc);
        // After decoding luma DC, compute the shared 2-D Turing field.
        if ch_idx == 0 {
            let _t_tf = crate::timing::stage("dec:turing_field");
            if let Some(ref hdr) = turing_hdr {
                let l_info = &channel_infos[0];
                let l_dc_range = (l_info.dc_max - l_info.dc_min).max(1e-6);
                let l_dc_denorm: Vec<f64> = dc_q
                    .iter()
                    .map(|&q| q as f64 * dc_step_clamped * l_dc_range / 255.0 + l_info.dc_min)
                    .collect();
                let params = hierarchy::HierarchyParams::from(hdr);
                let (_, bgh, bgw) = block_map
                    .as_ref()
                    .ok_or_else(|| invalid_format("missing v13 block map"))?;
                let blocks = var_blocks
                    .as_ref()
                    .ok_or_else(|| invalid_format("missing v13 block layout"))?;
                let dense_dc = hierarchy::expand_adaptive_dc_grid(
                    &l_dc_denorm,
                    blocks,
                    *bgh,
                    *bgw,
                    use_overlap,
                )
                .map_err(|e| invalid_format(format!("invalid adaptive DC layout: {e}")))?;
                let tf = hierarchy::compute_level1(&dense_dc, *bgh, *bgw, &params);
                block_turing_buckets = Some(hierarchy::build_adaptive_turing_buckets(
                    &tf,
                    blocks,
                    use_overlap,
                ));
                if use_phyllotactic_scan || use_spiral_arc_scan {
                    block_scan_order = Some(if use_spiral_arc_scan {
                        scan::adaptive_spiral_arc_order(
                            blocks,
                            *bgh,
                            *bgw,
                            use_overlap,
                            &tf.inhibitor_gradient,
                        )
                    } else {
                        scan::adaptive_phyllotactic_order(
                            blocks,
                            *bgh,
                            *bgw,
                            use_overlap,
                            &tf.inhibitor_gradient,
                        )
                    });
                }
                if use_progressive_packets {
                    let spiral_order = scan::adaptive_spiral_arc_order(
                        blocks,
                        *bgh,
                        *bgw,
                        use_overlap,
                        &tf.inhibitor_gradient,
                    );
                    progressive_packets = Some(scan::progressive_spiral_packets(&spiral_order));
                }
                turing_field = Some(tf);
            }
        }

        // v13 stores the primitive-match section for the L channel only.
        if ch_idx == 0 {
            let match_section_len = reader.read_u32("primitive-match length")? as usize;
            reader.take(match_section_len, "primitive-match section")?;
        }

        // Read CfL metadata for chroma channels (ch_idx == 1, 2) — rANS compressed
        if use_cfl && ch_idx > 0 {
            let cfl_n_symbols = reader.read_u32("CfL symbol count")? as usize;
            let max_cfl_symbols = n_blocks
                .checked_mul(2)
                .ok_or_else(|| invalid_format("CfL symbol count overflow"))?;
            if cfl_n_symbols < n_blocks || cfl_n_symbols > max_cfl_symbols {
                return Err(invalid_format("invalid CfL symbol count"));
            }
            let cfl_rans_size = reader.read_u32("CfL rANS length")? as usize;
            let cfl_data_bytes = reader.take(cfl_rans_size, "CfL rANS stream")?;
            let cfl_ctx = vec![0u8; cfl_n_symbols];
            let cfl_symbols =
                decode_rans_v12(cfl_data_bytes, cfl_n_symbols, &cfl_ctx, "CfL stream")?;

            let mut cfl_flags = vec![false; n_blocks];
            for i in 0..n_blocks {
                match cfl_symbols[i] {
                    0 => {}
                    1 => cfl_flags[i] = true,
                    _ => return Err(invalid_format("CfL flag must be 0 or 1")),
                }
            }
            let n_active: usize = cfl_flags.iter().filter(|&&f| f).count();
            if cfl_n_symbols != n_blocks + n_active {
                return Err(invalid_format(
                    "CfL alpha count does not match active flags",
                ));
            }
            let mut alpha_indices = vec![3usize; n_blocks];
            let mut alpha_cursor = 0usize;
            for block_idx in 0..n_blocks {
                if cfl_flags[block_idx] {
                    let sym_idx = n_blocks + alpha_cursor;
                    let alpha = cfl_symbols[sym_idx];
                    if !(0..=7).contains(&alpha) {
                        return Err(invalid_format("CfL alpha index is outside 0..=7"));
                    }
                    alpha_indices[block_idx] = alpha as usize;
                    alpha_cursor += 1;
                }
            }

            cfl_data.push(CflChannelData {
                flags: cfl_flags,
                alpha_indices,
            });
        }

        if use_progressive_packets {
            pending_progressive.push(PendingProgressiveRaw {
                grid_h,
                grid_w,
                dc_q,
            });
            continue;
        }

        let _t_ac = crate::timing::stage("dec:rans_ac");
        // Read EOB positions: DPCM + rANS compressed (Turing-contextualized)
        let eob_count = reader.read_u32("EOB count")? as usize;
        if eob_count != n_blocks {
            return Err(invalid_format("EOB count does not match block count"));
        }
        let eob_rans_size = reader.read_u32("EOB rANS length")? as usize;
        let eob_data = reader.take(eob_rans_size, "EOB rANS stream")?;
        let mut eob_positions: Vec<u16> = Vec::with_capacity(eob_count);
        {
            let eob_turing = block_turing_buckets
                .as_ref()
                .ok_or_else(|| invalid_format("missing Turing contexts"))?;
            if eob_turing.len() != eob_count {
                return Err(invalid_format(
                    "Turing context count does not match EOB count",
                ));
            }
            let eob_deltas = decode_rans_v12(eob_data, eob_count, eob_turing, "EOB stream")?;
            let mut prev_eob: i32 = 0;
            for (block_idx, &delta) in eob_deltas.iter().enumerate() {
                prev_eob = prev_eob
                    .checked_add(delta as i32)
                    .ok_or_else(|| invalid_format("EOB DPCM overflow"))?;
                let block_size = var_blocks
                    .as_ref()
                    .map_or(LOT_BLOCK_SIZE, |blocks| blocks[block_idx].2);
                let max_eob = block_size * block_size - 1;
                if prev_eob < 0 || prev_eob as usize > max_eob {
                    return Err(invalid_format(
                        "EOB position lies outside its transform block",
                    ));
                }
                eob_positions.push(prev_eob as u16);
            }
        }

        let mut ac_q_by_block: Vec<Vec<i16>> = vec![Vec::new(); n_blocks];
        let ac_per_block = |block_idx: usize| {
            let block_size = if let Some(ref blocks) = var_blocks {
                if block_idx < blocks.len() {
                    blocks[block_idx].2
                } else {
                    LOT_BLOCK_SIZE
                }
            } else {
                LOT_BLOCK_SIZE
            };
            block_size * block_size - 1
        };

        if use_progressive_packets {
            let packets = progressive_packets
                .as_ref()
                .ok_or_else(|| invalid_format("missing progressive spiral packet order"))?;
            let packet_count = reader.read_u32("progressive AC packet count")? as usize;
            if packet_count != packets.len() {
                return Err(invalid_format(
                    "progressive AC packet count does not match the image layout",
                ));
            }

            let buckets = block_turing_buckets
                .as_ref()
                .ok_or_else(|| invalid_format("missing Turing contexts"))?;
            for (packet_idx, packet) in packets.iter().enumerate() {
                let symbol_count = reader.read_u32("progressive AC symbol count")? as usize;
                let encoded_size = reader.read_u32("progressive AC rANS length")? as usize;
                let encoded = reader.take(encoded_size, "progressive AC rANS stream")?;
                let expected_symbols = packet.iter().try_fold(0usize, |sum, &block_idx| {
                    sum.checked_add(eob_positions[block_idx] as usize)
                        .ok_or_else(|| invalid_format("AC symbol count overflow"))
                })?;
                if symbol_count != expected_symbols {
                    return Err(invalid_format(
                        "progressive AC symbol count does not match packet EOBs",
                    ));
                }

                let retain_packet = progressive_layers.is_none_or(|limit| packet_idx < limit);
                let decoded = if retain_packet {
                    let mut contexts = Vec::with_capacity(symbol_count);
                    for &block_idx in packet {
                        contexts.extend(std::iter::repeat_n(
                            buckets[block_idx],
                            eob_positions[block_idx] as usize,
                        ));
                    }
                    Some(decode_rans_v12(
                        encoded,
                        symbol_count,
                        &contexts,
                        "progressive AC stream",
                    )?)
                } else {
                    None
                };

                let mut cursor = 0usize;
                for &block_idx in packet {
                    let block_len = ac_per_block(block_idx);
                    let eob = eob_positions[block_idx] as usize;
                    let mut block_ac = vec![0i16; block_len];
                    if let Some(ref symbols) = decoded {
                        block_ac[..eob].copy_from_slice(&symbols[cursor..cursor + eob]);
                    }
                    cursor += eob;
                    ac_q_by_block[block_idx] = block_ac;
                }
                if cursor != symbol_count {
                    return Err(invalid_format(
                        "progressive AC packet symbol count mismatch",
                    ));
                }
            }
        } else {
            // Read and decode the monolithic AC rANS data.
            let ac_rans_size = reader.read_u32("AC rANS length")? as usize;
            let ac_data = reader.take(ac_rans_size, "AC rANS stream")?;
            let total_ac = eob_positions.iter().try_fold(0usize, |acc, &e| {
                acc.checked_add(e as usize)
                    .ok_or_else(|| invalid_format("AC symbol count overflow"))
            })?;
            let buckets = block_turing_buckets
                .as_ref()
                .ok_or_else(|| invalid_format("missing Turing contexts"))?;
            let scan_order = block_scan_order
                .as_ref()
                .ok_or_else(|| invalid_format("missing spatial AC block order"))?;
            // v25 keys the AC contexts on intra-block activity rather than the
            // Turing bucket. Every EOB is decoded by now, and the encoder
            // derived its buckets from exactly these numbers.
            let ac_buckets: Vec<u8> = if header.version >= 25 {
                rans::activity_buckets(&eob_positions, &ac_per_block)
            } else {
                buckets.to_vec()
            };
            let mut contexts = Vec::with_capacity(total_ac);
            for &block_idx in scan_order {
                contexts.extend(std::iter::repeat_n(
                    ac_buckets[block_idx],
                    eob_positions[block_idx] as usize,
                ));
            }
            let decoded = decode_rans_v12(ac_data, total_ac, &contexts, "AC stream")?;
            let mut cursor = 0usize;
            for &block_idx in scan_order {
                let block_len = ac_per_block(block_idx);
                let eob = eob_positions[block_idx] as usize;
                let mut block_ac = vec![0i16; block_len];
                block_ac[..eob].copy_from_slice(&decoded[cursor..cursor + eob]);
                cursor += eob;
                ac_q_by_block[block_idx] = block_ac;
            }

            if cursor != decoded.len() {
                return Err(invalid_format("AC stream symbol count mismatch"));
            }
        }
        let ac_q = ac_q_by_block.into_iter().flatten().collect();

        raw_channels.push(V13ChannelRaw {
            grid_h,
            grid_w,
            dc_q,
            ac_q,
        });
    }

    let t_prog = crate::timing::stage("dec:rans_ac");
    // Hoisted for the v30/v31 EOB-driven step fields: the transmitted EOB
    // positions of all three channels, read before any AC coefficient.
    let mut eob_positions_by_channel: Vec<Vec<u16>> = Vec::new();
    if use_progressive_packets {
        if pending_progressive.len() != 3 {
            return Err(invalid_format(
                "progressive base layer does not contain three channels",
            ));
        }
        let packets = progressive_packets
            .as_ref()
            .ok_or_else(|| invalid_format("missing progressive spiral packet order"))?;
        let buckets = block_turing_buckets
            .as_ref()
            .ok_or_else(|| invalid_format("missing Turing contexts"))?;
        let n_blocks = pending_progressive[0].dc_q.len();
        if pending_progressive
            .iter()
            .any(|channel| channel.dc_q.len() != n_blocks)
        {
            return Err(invalid_format(
                "progressive channel block counts do not agree",
            ));
        }

        let block_ac_len = |block_idx: usize| {
            let block_size = var_blocks
                .as_ref()
                .map_or(LOT_BLOCK_SIZE, |blocks| blocks[block_idx].2);
            block_size * block_size - 1
        };
        let mut eob_by_channel: Vec<Vec<u16>> = Vec::with_capacity(3);
        for _ in 0..3 {
            let eob_count = reader.read_u32("progressive EOB count")? as usize;
            if eob_count != n_blocks || buckets.len() != eob_count {
                return Err(invalid_format(
                    "progressive EOB count does not match block contexts",
                ));
            }
            let encoded_size = reader.read_u32("progressive EOB rANS length")? as usize;
            let encoded = reader.take(encoded_size, "progressive EOB rANS stream")?;
            let deltas = decode_rans_v12(encoded, eob_count, buckets, "progressive EOB stream")?;
            let mut positions = Vec::with_capacity(eob_count);
            let mut previous = 0i32;
            for (block_idx, &delta) in deltas.iter().enumerate() {
                previous = previous
                    .checked_add(delta as i32)
                    .ok_or_else(|| invalid_format("progressive EOB DPCM overflow"))?;
                let max_eob = block_ac_len(block_idx);
                if previous < 0 || previous as usize > max_eob {
                    return Err(invalid_format(
                        "progressive EOB lies outside its transform block",
                    ));
                }
                positions.push(previous as u16);
            }
            eob_by_channel.push(positions);
        }
        eob_positions_by_channel = eob_by_channel.clone();

        // Cycle 27 — v25 keys the AC contexts on intra-block activity instead
        // of the Turing bucket, which is a Sobel of the DC grid and therefore
        // blind to what a block contains. Every channel's EOB is decoded by
        // now, so the decoder derives the buckets from exactly the numbers the
        // encoder used. The EOB stream itself keeps the Turing bucket: a symbol
        // cannot be its own context.
        let ac_buckets_by_channel: Vec<Vec<u8>> = eob_by_channel
            .iter()
            .map(|eob| {
                if header.version >= 25 {
                    rans::activity_buckets(eob, &block_ac_len)
                } else {
                    buckets.to_vec()
                }
            })
            .collect();

        let layer_count = reader.read_u32("progressive AC layer count")? as usize;
        if layer_count != packets.len() + 1 {
            return Err(invalid_format(
                "progressive AC layer count does not match the spiral layout",
            ));
        }
        let retained_layers = progressive_layers.unwrap_or(layer_count).min(layer_count);
        partial_progressive = retained_layers < layer_count;
        let mut ac_by_channel: Vec<Vec<Vec<i16>>> = (0..3)
            .map(|_| {
                (0..n_blocks)
                    .map(|block_idx| vec![0i16; block_ac_len(block_idx)])
                    .collect()
            })
            .collect();
        let fourier_top_blocks: Vec<Option<usize>> = if let Some(ref blocks) = var_blocks {
            let mut by_geometry = std::collections::HashMap::with_capacity(blocks.len());
            for (index, &(y, x, size)) in blocks.iter().enumerate() {
                by_geometry.insert((y, x, size), index);
            }
            blocks
                .iter()
                .map(|&(y, x, size)| {
                    y.checked_sub(size)
                        .and_then(|top_y| by_geometry.get(&(top_y, x, size)).copied())
                })
                .collect()
        } else {
            vec![None; n_blocks]
        };
        let mut progressive_block_rank = vec![usize::MAX; n_blocks];
        let mut rank = 0usize;
        for packet in packets {
            for &block_idx in packet {
                progressive_block_rank[block_idx] = rank;
                rank += 1;
            }
        }

        let skeleton_coeffs = scan::PROGRESSIVE_SKELETON_COEFFS;
        // v27 — layer 0 ionizes one channel per component; every detail layer
        // re-enters it rather than breaking down flat air again.
        let mut layer0_memory: Vec<rans::EnvelopeTables> =
            (0..3).map(|_| rans::EnvelopeTables::default()).collect();
        // Speed campaign (bit-exact): every packet is its own rANS stream at
        // a known offset, so all byte slices are read up front and the
        // packets are scheduled in three phases — L details fully parallel
        // (they read only frozen skeleton data), C1 sequential in arc order
        // (its Fourier phase predictor may reference the top neighbour from
        // an earlier arc), C2 fully parallel (it reads only its own arc's
        // decoded L/C1). The visited values are identical to the historical
        // (layer × channel) loop, hence the reconstruction is too.
        let mut packet_streams: Vec<Vec<(usize, &[u8])>> = Vec::with_capacity(retained_layers);
        for _layer in 0..retained_layers {
            let mut row = Vec::with_capacity(3);
            for _channel in 0..3 {
                let symbol_count = reader.read_u32("progressive AC symbol count")? as usize;
                let encoded_size = reader.read_u32("progressive AC rANS length")? as usize;
                row.push((
                    symbol_count,
                    reader.take(encoded_size, "progressive AC rANS stream")?,
                ));
            }
            packet_streams.push(row);
        }
        let decode_packet = |layer_idx: usize,
                             channel_idx: usize,
                             ac_by_channel: &Vec<Vec<Vec<i16>>>,
                             layer_memory: &mut rans::EnvelopeTables|
         -> Result<Vec<i16>, AureaError> {
            let (symbol_count, encoded) = packet_streams[layer_idx][channel_idx];
            let layer_blocks: Vec<usize> = if layer_idx == 0 {
                (0..n_blocks).collect()
            } else {
                packets[layer_idx - 1].clone()
            };
            {
                let expected_symbols =
                    layer_blocks.iter().try_fold(0usize, |sum, &block_idx| {
                        let eob = eob_by_channel[channel_idx][block_idx] as usize;
                        let count = if layer_idx == 0 {
                            eob.min(skeleton_coeffs)
                        } else {
                            eob.saturating_sub(skeleton_coeffs)
                        };
                        sum.checked_add(count)
                            .ok_or_else(|| invalid_format("progressive AC symbol count overflow"))
                    })?;
                if symbol_count != expected_symbols {
                    return Err(invalid_format(format!(
                        "progressive AC layer {layer_idx} symbol count does not match EOBs",
                    )));
                }
                let mut contexts = Vec::with_capacity(symbol_count);
                for &block_idx in &layer_blocks {
                    let eob = eob_by_channel[channel_idx][block_idx] as usize;
                    let count = if layer_idx == 0 {
                        eob.min(skeleton_coeffs)
                    } else {
                        eob.saturating_sub(skeleton_coeffs)
                    };
                    contexts.extend(std::iter::repeat_n(
                        ac_buckets_by_channel[channel_idx][block_idx],
                        count,
                    ));
                }
                let decoded = if header.version >= 17
                    || (header.version >= 16 && header.quality <= rans::WAVELET_PARENT_MAX_QUALITY)
                {
                    let mut parent_refs = Vec::with_capacity(symbol_count);
                    let mut layer_offsets = vec![None; n_blocks];
                    let mut running_offset = 0usize;
                    for &block_idx in &layer_blocks {
                        layer_offsets[block_idx] = Some(running_offset);
                        let eob = eob_by_channel[channel_idx][block_idx] as usize;
                        let count = if layer_idx == 0 {
                            eob.min(skeleton_coeffs)
                        } else {
                            eob.saturating_sub(skeleton_coeffs)
                        };
                        running_offset += count;
                    }

                    // Cycle 20: one segment per block, describing where its
                    // coefficients live in this flat sequence and which of them
                    // an earlier layer already delivered.
                    let mut spectral_segments = Vec::new();
                    // The radar echo: how loud this block's already-decoded
                    // skeleton came back. Aligned with `spectral_segments`,
                    // pushed under exactly the same condition.
                    let mut block_echo: Vec<u32> = Vec::new();
                    if use_spectral_envelope {
                        spectral_segments.reserve(layer_blocks.len());
                        for &block_idx in &layer_blocks {
                            let eob = eob_by_channel[channel_idx][block_idx] as usize;
                            let start = if layer_idx == 0 {
                                0
                            } else {
                                eob.min(skeleton_coeffs)
                            };
                            let end = if layer_idx == 0 {
                                eob.min(skeleton_coeffs)
                            } else {
                                eob
                            };
                            if end <= start {
                                continue;
                            }
                            let block_size = var_blocks
                                .as_ref()
                                .map_or(LOT_BLOCK_SIZE, |blocks| blocks[block_idx].2);
                            let table =
                                spectral_envelope::SpectralNeighborTables::table_class(block_size);
                            let offset = layer_offsets[block_idx]
                                .ok_or_else(|| invalid_format("missing v21 AC block offset"))?;
                            if header.version >= 26 && layer_idx > 0 {
                                block_echo.push(rans::skeleton_echo(
                                    &ac_by_channel[channel_idx][block_idx][..start],
                                ));
                            }
                            spectral_segments.push(if layer_idx == 0 {
                                spectral_envelope::SpectralSegment::root(offset, end - start, table)
                            } else {
                                spectral_envelope::SpectralSegment::detail(
                                    offset,
                                    start,
                                    end - start,
                                    table,
                                    &ac_by_channel[channel_idx][block_idx][..start],
                                )
                            });
                        }
                    }

                    for &block_idx in &layer_blocks {
                        let eob = eob_by_channel[channel_idx][block_idx] as usize;
                        let start = if layer_idx == 0 {
                            0
                        } else {
                            eob.min(skeleton_coeffs)
                        };
                        let end = if layer_idx == 0 {
                            eob.min(skeleton_coeffs)
                        } else {
                            eob
                        };
                        let block_size = var_blocks
                            .as_ref()
                            .map_or(LOT_BLOCK_SIZE, |blocks| blocks[block_idx].2);
                        let parent_positions = match block_size {
                            8 => &spectral_parents_8,
                            32 => &spectral_parents_32,
                            _ => &spectral_parents_16,
                        };
                        let block_offset = layer_offsets[block_idx]
                            .ok_or_else(|| invalid_format("missing v16 AC block offset"))?;
                        for position in start..end {
                            let Some(parent_position) = parent_positions[position] else {
                                parent_refs.push(rans::ParentSupportRef::Known(false));
                                continue;
                            };
                            if parent_position >= position {
                                return Err(invalid_format(
                                    "non-causal v16 spectral parent ordering",
                                ));
                            }
                            if parent_position >= start {
                                parent_refs.push(rans::ParentSupportRef::Decoded(
                                    block_offset + parent_position - start,
                                ));
                            } else {
                                parent_refs.push(rans::ParentSupportRef::Known(
                                    ac_by_channel[channel_idx][block_idx][parent_position] != 0,
                                ));
                            }
                        }
                    }
                    let decoded_result = if header.version >= 17 {
                        let mut codon_classes = Vec::with_capacity(symbol_count);
                        let mut phase_refs = Vec::with_capacity(symbol_count);
                        // Cycle 22: Lphi's neighbourhood, kept as its own
                        // witness for the mixer instead of being folded in.
                        let mut foreign_amplitudes = if spectral_config.mix_rate > 0 {
                            Vec::with_capacity(symbol_count)
                        } else {
                            Vec::new()
                        };
                        for &block_idx in &layer_blocks {
                            let eob = eob_by_channel[channel_idx][block_idx] as usize;
                            let start = if layer_idx == 0 {
                                0
                            } else {
                                eob.min(skeleton_coeffs)
                            };
                            let end = if layer_idx == 0 {
                                eob.min(skeleton_coeffs)
                            } else {
                                eob
                            };
                            let block_size = var_blocks
                                .as_ref()
                                .map_or(LOT_BLOCK_SIZE, |blocks| blocks[block_idx].2);
                            let cross_table = spectral_tables.table(
                                spectral_envelope::SpectralNeighborTables::table_class(block_size),
                            );
                            for position in start..end {
                                let l_value = ac_by_channel[0][block_idx][position];
                                let c1_value = ac_by_channel[1][block_idx][position];
                                let base = rans::vector_codon_class(
                                    header.quality,
                                    channel_idx,
                                    l_value,
                                    c1_value,
                                );
                                // Cycle 21/22 — Lphi's neighbourhood, read from
                                // coefficients this layer already decoded.
                                let needs_foreign = channel_idx > 0
                                    && (spectral_config.cross_codon_mode
                                        != spectral_envelope::CROSS_CODON_NONE
                                        || spectral_config.mix_rate > 0);
                                let foreign = if needs_foreign {
                                    let luma = &ac_by_channel[0][block_idx];
                                    spectral_envelope::foreign_amplitude(
                                        cross_table,
                                        position,
                                        |q| luma.get(q).copied().unwrap_or(0),
                                    )
                                } else {
                                    0
                                };
                                if spectral_config.mix_rate > 0 {
                                    foreign_amplitudes.push(foreign);
                                }
                                codon_classes.push(spectral_envelope::combine_codon_class(
                                    spectral_config.cross_codon_mode,
                                    base,
                                    foreign,
                                ));
                                let phase_ref = if channel_idx == 2 && c1_value != 0 {
                                    rans::PhasePredictorRef::Known(c1_value < 0)
                                } else if channel_idx == 1 {
                                    let Some(top_idx) =
                                        fourier_top_blocks.get(block_idx).copied().flatten()
                                    else {
                                        phase_refs.push(rans::PhasePredictorRef::None);
                                        continue;
                                    };
                                    let causal = if layer_idx == 0 {
                                        top_idx < block_idx
                                    } else {
                                        progressive_block_rank
                                            .get(top_idx)
                                            .copied()
                                            .unwrap_or(usize::MAX)
                                            < progressive_block_rank
                                                .get(block_idx)
                                                .copied()
                                                .unwrap_or(usize::MAX)
                                    };
                                    if !causal
                                        || eob_by_channel[channel_idx][top_idx] as usize <= position
                                    {
                                        rans::PhasePredictorRef::None
                                    } else {
                                        let block_size = var_blocks
                                            .as_ref()
                                            .map_or(LOT_BLOCK_SIZE, |blocks| blocks[block_idx].2);
                                        let zigzag = match block_size {
                                            8 => &zz_8,
                                            32 => &zz_32,
                                            _ => &zz_16,
                                        };
                                        let invert = zigzag[position] / block_size & 1 != 0;
                                        if let Some(top_offset) = layer_offsets[top_idx] {
                                            let top_eob =
                                                eob_by_channel[channel_idx][top_idx] as usize;
                                            let top_start = if layer_idx == 0 {
                                                0
                                            } else {
                                                top_eob.min(skeleton_coeffs)
                                            };
                                            let predictor_index = top_offset + position - top_start;
                                            if predictor_index >= phase_refs.len() {
                                                return Err(invalid_format(
                                                    "non-causal v20 Fourier phase reference",
                                                ));
                                            }
                                            rans::PhasePredictorRef::Decoded {
                                                index: predictor_index,
                                                invert,
                                            }
                                        } else {
                                            let value =
                                                ac_by_channel[channel_idx][top_idx][position];
                                            if value == 0 {
                                                rans::PhasePredictorRef::None
                                            } else {
                                                rans::PhasePredictorRef::Known((value < 0) ^ invert)
                                            }
                                        }
                                    }
                                } else {
                                    rans::PhasePredictorRef::None
                                };
                                phase_refs.push(phase_ref);
                            }
                        }
                        let (phase_initial, phase_shift) = if channel_idx == 1 {
                            (
                                rans::FOURIER_PHASE_INITIAL_P_SAME,
                                rans::FOURIER_PHASE_ADAPT_SHIFT,
                            )
                        } else {
                            (rans::PHASE_INITIAL_P_SAME, rans::PHASE_ADAPT_SHIFT)
                        };
                        if use_spectral_envelope {
                            rans::rans_decode_band_v21_envelope(
                                encoded,
                                symbol_count,
                                &contexts,
                                &codon_classes,
                                &phase_refs,
                                &foreign_amplitudes,
                                &spectral_segments,
                                &spectral_tables,
                                &spectral_config,
                                channel_idx,
                                phase_initial,
                                phase_shift,
                                header.version >= 25,
                                &block_echo,
                                layer_memory,
                            )
                        } else {
                            rans::rans_decode_band_v20_phase(
                                encoded,
                                symbol_count,
                                &contexts,
                                &parent_refs,
                                &codon_classes,
                                &phase_refs,
                                phase_initial,
                                phase_shift,
                            )
                        }
                    } else {
                        rans::rans_decode_band_v12_parent(
                            encoded,
                            symbol_count,
                            &contexts,
                            &parent_refs,
                        )
                    };
                    decoded_result
                        .map(|(symbols, _)| symbols)
                        .map_err(|error| {
                            AureaError::DecodeFailed(format!(
                                "invalid progressive AC stream: {error}"
                            ))
                        })?
                } else {
                    decode_rans_v12(encoded, symbol_count, &contexts, "progressive AC stream")?
                };
                Ok(decoded)
            }
        };
        let merge_packet = |layer_idx: usize,
                           channel_idx: usize,
                           decoded: Vec<i16>,
                           ac_by_channel: &mut Vec<Vec<Vec<i16>>>|
         -> Result<(), AureaError> {
            let layer_blocks: Vec<usize> = if layer_idx == 0 {
                (0..n_blocks).collect()
            } else {
                packets[layer_idx - 1].clone()
            };
            let mut cursor = 0usize;
            for &block_idx in &layer_blocks {
                let eob = eob_by_channel[channel_idx][block_idx] as usize;
                let start = if layer_idx == 0 {
                    0
                } else {
                    eob.min(skeleton_coeffs)
                };
                let end = if layer_idx == 0 {
                    eob.min(skeleton_coeffs)
                } else {
                    eob
                };
                let count = end - start;
                ac_by_channel[channel_idx][block_idx][start..end]
                    .copy_from_slice(&decoded[cursor..cursor + count]);
                cursor += count;
            }
            if cursor != decoded.len() {
                return Err(invalid_format("progressive AC layer symbol count mismatch"));
            }
            Ok(())
        };

        // Layer 0 — the skeleton's codon chain is channel-sequential, but
        // C1's skeleton only reads L's (its own future chroma values are the
        // zeros the historical sweep also read), so its *decode* overlaps
        // the parallel L-detail phase below; the merges stay ordered.
        if retained_layers > 0 {
            let mut memory = rans::EnvelopeTables::default();
            let decoded = decode_packet(0, 0, &ac_by_channel, &mut memory)?;
            merge_packet(0, 0, decoded, &mut ac_by_channel)?;
            if header.version >= 27 {
                layer0_memory[0] = memory;
            }
            if retained_layers == 1 {
                // Skeleton-only prefix: finish the codon chain sequentially.
                for channel_idx in 1..3 {
                    let mut memory = rans::EnvelopeTables::default();
                    let decoded = decode_packet(0, channel_idx, &ac_by_channel, &mut memory)?;
                    merge_packet(0, channel_idx, decoded, &mut ac_by_channel)?;
                    if header.version >= 27 {
                        layer0_memory[channel_idx] = memory;
                    }
                }
            }
        }
        if retained_layers > 1 {
            use rayon::prelude::*;
            let phase_t0 = std::time::Instant::now();
            // Phase L ∥ C1-skeleton — both read only channel 0's frozen data.
            let l_layer0 = &layer0_memory[0];
            let (l_decoded, c1_skeleton): (
                Result<Vec<Vec<i16>>, AureaError>,
                Result<(Vec<i16>, rans::EnvelopeTables), AureaError>,
            ) = rayon::join(
                || {
                    (1..retained_layers)
                        .into_par_iter()
                        .map(|layer_idx| {
                            let mut memory = if header.version >= 27 {
                                l_layer0.clone()
                            } else {
                                rans::EnvelopeTables::default()
                            };
                            decode_packet(layer_idx, 0, &ac_by_channel, &mut memory)
                        })
                        .collect()
                },
                || {
                    let mut memory = rans::EnvelopeTables::default();
                    decode_packet(0, 1, &ac_by_channel, &mut memory)
                        .map(|decoded| (decoded, memory))
                },
            );
            for (offset, decoded) in l_decoded?.into_iter().enumerate() {
                merge_packet(offset + 1, 0, decoded, &mut ac_by_channel)?;
            }
            let (c1_skeleton_decoded, c1_memory) = c1_skeleton?;
            merge_packet(0, 1, c1_skeleton_decoded, &mut ac_by_channel)?;
            if header.version >= 27 {
                layer0_memory[1] = c1_memory;
            }
            // C2 skeleton — needs C1's merged skeleton for its codon classes.
            {
                let mut memory = rans::EnvelopeTables::default();
                let decoded = decode_packet(0, 2, &ac_by_channel, &mut memory)?;
                merge_packet(0, 2, decoded, &mut ac_by_channel)?;
                if header.version >= 27 {
                    layer0_memory[2] = memory;
                }
            }
            let fresh_memory = |channel_idx: usize| -> rans::EnvelopeTables {
                if header.version >= 27 {
                    layer0_memory[channel_idx].clone()
                } else {
                    rans::EnvelopeTables::default()
                }
            };
            let phase_l = phase_t0.elapsed();
            // Phase C1 — arc order, for the cross-arc Fourier phase reference.
            // AUREA_DAG_STATS=1 measures the *actual* cross-arc dependency
            // graph: which earlier arcs each C1 packet really reads through
            // the Known-branch of its top-neighbour phase predictor.
            let dag_stats = std::env::var("AUREA_DAG_STATS")
                .map(|v| v == "1")
                .unwrap_or(false);
            if dag_stats {
                let arc_of = {
                    let mut arc = vec![usize::MAX; n_blocks];
                    for (packet_idx, packet) in packets.iter().enumerate() {
                        for &b in packet {
                            arc[b] = packet_idx + 1;
                        }
                    }
                    arc
                };
                let mut chain_deps = 0usize;
                let mut max_span = 0usize;
                let mut dep_arcs = 0usize;
                for layer_idx in 1..retained_layers {
                    let mut deepest: Option<usize> = None;
                    for &block_idx in &packets[layer_idx - 1] {
                        if let Some(top_idx) = fourier_top_blocks[block_idx] {
                            let causal = progressive_block_rank[top_idx]
                                < progressive_block_rank[block_idx];
                            let top_arc = arc_of[top_idx];
                            if causal && top_arc != layer_idx && top_arc != usize::MAX {
                                deepest =
                                    Some(deepest.map_or(top_arc, |d: usize| d.max(top_arc)));
                            }
                        }
                    }
                    if let Some(d) = deepest {
                        dep_arcs += 1;
                        max_span = max_span.max(layer_idx - d);
                        if d + 1 == layer_idx {
                            chain_deps += 1;
                        }
                    }
                }
                eprintln!(
                    "[AUREA_DAG_STATS] arcs={} avec_dep_transarc={} dep_adjacente={} portee_max={}",
                    retained_layers.saturating_sub(1),
                    dep_arcs,
                    chain_deps,
                    max_span,
                );
            }
            let phase_t1 = std::time::Instant::now();
            for layer_idx in 1..retained_layers {
                let mut memory = fresh_memory(1);
                let decoded = decode_packet(layer_idx, 1, &ac_by_channel, &mut memory)?;
                merge_packet(layer_idx, 1, decoded, &mut ac_by_channel)?;
            }
            let phase_c1 = phase_t1.elapsed();
            let phase_t2 = std::time::Instant::now();
            // Phase C2 — reads only its own arc's decoded L and C1.
            let c2_decoded: Result<Vec<Vec<i16>>, AureaError> = (1..retained_layers)
                .into_par_iter()
                .map(|layer_idx| {
                    let mut memory = fresh_memory(2);
                    decode_packet(layer_idx, 2, &ac_by_channel, &mut memory)
                })
                .collect();
            for (offset, decoded) in c2_decoded?.into_iter().enumerate() {
                merge_packet(offset + 1, 2, decoded, &mut ac_by_channel)?;
            }
            if std::env::var("AUREA_TIMINGS").map(|v| v == "1").unwrap_or(false) {
                eprintln!(
                    "[AUREA_TIMINGS]   dec:rans_phases  L={:.3}s C1={:.3}s C2={:.3}s",
                    phase_l.as_secs_f64(),
                    phase_c1.as_secs_f64(),
                    phase_t2.elapsed().as_secs_f64(),
                );
            }
        }
        progressive_info = Some(ProgressiveDecodeInfo {
            total_layers: layer_count,
            decoded_layers: retained_layers,
            prefix_bytes: header_size + reader.position(),
        });

        for (channel_idx, pending) in pending_progressive.into_iter().enumerate() {
            raw_channels.push(V13ChannelRaw {
                grid_h: pending.grid_h,
                grid_w: pending.grid_w,
                dc_q: pending.dc_q,
                ac_q: ac_by_channel[channel_idx].drain(..).flatten().collect(),
            });
        }
    }

    drop(t_prog);
    let t_fields = crate::timing::stage("dec:hierarchy_fields");
    // === Dequantize L channel DC for TuringField and foveal map ===
    let l_raw = &raw_channels[0];
    let l_dc_range = (channel_infos[0].dc_max - channel_infos[0].dc_min).max(1e-6);
    let l_dc_denorm: Vec<f64> = l_raw
        .dc_q
        .iter()
        .map(|&q| q as f64 * dc_step_clamped * l_dc_range / 255.0 + channel_infos[0].dc_min)
        .collect();

    // v13 derives all zero-bit spatial factors from the dense 2-D cell grid,
    // then averages them back to one value per adaptive transform block.
    let (_, bgh, bgw) = block_map
        .as_ref()
        .ok_or_else(|| invalid_format("missing v13 block map"))?;
    let blocks = var_blocks
        .as_ref()
        .ok_or_else(|| invalid_format("missing v13 block layout"))?;
    let tf = turing_field
        .as_ref()
        .ok_or_else(|| invalid_format("missing v13 Turing field"))?;
    let dense_dc =
        hierarchy::expand_adaptive_dc_grid(&l_dc_denorm, blocks, *bgh, *bgw, use_overlap)
            .map_err(|e| invalid_format(format!("invalid adaptive DC layout: {e}")))?;
    let foveal_cells = lot::foveal_saliency_map(&dense_dc, *bgh, *bgw);
    let l_foveal_map =
        hierarchy::sample_scalar_for_blocks(&foveal_cells, *bgh, *bgw, blocks, use_overlap);
    let local_block_turing_mods = hierarchy::sample_scalar_for_blocks(
        &tf.step_modulation,
        tf.grid_h,
        tf.grid_w,
        blocks,
        use_overlap,
    );
    let multiscale_strength = *calibration::TUNABLE_TURING_MULTISCALE;
    let block_turing_mods = if multiscale_strength > 0.0 {
        let local_params = hierarchy::HierarchyParams::from(
            turing_hdr
                .as_ref()
                .ok_or_else(|| invalid_format("missing v13 Turing parameters"))?,
        );
        let wider_params = hierarchy::HierarchyParams {
            sigma_a: local_params.sigma_a * crate::golden::PHI,
            sigma_ratio: local_params.sigma_ratio,
            k_step: local_params.k_step,
            mag_calibration: local_params.mag_calibration,
        };
        let wider_field = hierarchy::compute_level1(&dense_dc, *bgh, *bgw, &wider_params);
        let wider_mods = hierarchy::sample_scalar_for_blocks(
            &wider_field.step_modulation,
            wider_field.grid_h,
            wider_field.grid_w,
            blocks,
            use_overlap,
        );
        hierarchy::phi_multiscale_step_modulations(
            &local_block_turing_mods,
            &wider_mods,
            multiscale_strength,
        )
    } else {
        local_block_turing_mods
    };
    let pivoted_turing_mods: Vec<f64> = block_turing_mods
        .iter()
        .map(|&raw_modulation| {
            crate::aurea_encoder::psychovisual_turing_pivot(raw_modulation, header.quality)
        })
        .collect();
    let turing_block_sizes: Vec<usize> = blocks
        .iter()
        .map(|&(_, _, block_size)| block_size)
        .collect();
    let bs2 = (LOT_BLOCK_SIZE * LOT_BLOCK_SIZE) as f64;
    let l_dc_pixel: Vec<f64> = l_dc_denorm.iter().map(|&v| v / bs2).collect();
    let homeostatic_turing_mods = hierarchy::homeostatic_step_modulations(
        &pivoted_turing_mods,
        &turing_block_sizes,
        *calibration::TUNABLE_TURING_HOMEOSTASIS,
    );
    let spiral_field_strength = *calibration::TUNABLE_SPIRAL_FIELD_STRENGTH;
    let homeostatic_turing_mods = if spiral_field_strength > 0.0 {
        let spiral_order = crate::scan::adaptive_spiral_arc_order(
            blocks,
            *bgh,
            *bgw,
            use_overlap,
            &tf.inhibitor_gradient,
        );
        hierarchy::spiral_surprise_step_modulations(
            &homeostatic_turing_mods,
            &spiral_order,
            &turing_block_sizes,
            spiral_field_strength,
        )
    } else {
        homeostatic_turing_mods
    };
    let block_gradient_angles = hierarchy::build_adaptive_directional_structures(
        tf,
        blocks,
        use_overlap,
        *calibration::TUNABLE_DIRECTIONAL_COHERENCE,
    );
    let subsonic_wave_strength = *calibration::TUNABLE_SUBSONIC_WAVE_STRENGTH;
    let subsonic_waves = if subsonic_wave_strength > 0.0 {
        hierarchy::subsonic_wave_descriptors(&dense_dc, *bgh, *bgw, blocks, use_overlap)
    } else {
        Vec::new()
    };

    // Zero-bit bright compensation on luma steps. The strength is read from
    // the file (self-described), mirroring the encoder exactly.
    let bright_mods = hierarchy::bright_compensation_step_modulations(
        &l_dc_pixel,
        &turing_block_sizes,
        bright_strength * *calibration::TUNABLE_BRIGHT_COMPENSATION_GAIN,
    );

    // AUR2 v18+ fixes the zero-bit exchange in the format. Gamma remains
    // image-adaptive, but neither its luma/chroma strengths nor the color
    // basis depend on environment variables.
    let spectral_exchange_eta = 1.0;
    let mut chroma_breathing_mods = vec![vec![1.0; blocks.len()]; 2];
    let mut chroma_dense_grids: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    if use_chroma_breathing || spectral_exchange_eta > 0.0 {
        for cfl_idx in 0..2usize {
            let ch_idx = cfl_idx + 1;
            let raw = &raw_channels[ch_idx];
            let info = &channel_infos[ch_idx];
            let dc_range = (info.dc_max - info.dc_min).max(1e-6);
            let dc_denorm: Vec<f64> = raw
                .dc_q
                .iter()
                .map(|&q| q as f64 * dc_step_clamped * dc_range / 255.0 + info.dc_min)
                .collect();
            let chroma_dense =
                hierarchy::expand_adaptive_dc_grid(&dc_denorm, blocks, *bgh, *bgw, use_overlap)
                    .map_err(|e| {
                        invalid_format(format!("invalid adaptive chroma DC layout: {e}"))
                    })?;
            if use_chroma_breathing {
                chroma_breathing_mods[cfl_idx] = hierarchy::chroma_breathing_step_modulations(
                    &dense_dc,
                    &chroma_dense,
                    *bgh,
                    *bgw,
                    blocks,
                    use_overlap,
                    chroma_breathing_strength,
                );
            }
            chroma_dense_grids[cfl_idx] = chroma_dense;
        }
    }

    // Zero-bit spectral exchange, mirroring the encoder exactly.
    let exchange_gamma = if spectral_exchange_eta > 0.0 {
        hierarchy::spectral_exchange_gamma(
            &dense_dc,
            &chroma_dense_grids[0],
            &chroma_dense_grids[1],
            *bgh,
            *bgw,
            spectral_exchange_eta,
        )
    } else {
        0.0
    };
    // Diffuse-chroma relief field — exact mirror of the encoder, from the
    // same decoded DC grids (zero-bit). The relief strengths are a v28 format
    // property: files v27 and older always decode with both at zero.
    let (luma_pivot_relief, spectral_chroma_relief) = if header.version >= 28 {
        (
            *calibration::TUNABLE_LUMA_PIVOT_RELIEF,
            *calibration::TUNABLE_SPECTRAL_CHROMA_RELIEF,
        )
    } else {
        (0.0, 0.0)
    };
    let chroma_diffusion = hierarchy::chroma_diffusion_signature(
        &chroma_dense_grids[0],
        &chroma_dense_grids[1],
        exchange_gamma,
    );
    let luma_relief_gate = hierarchy::luma_relief_dark_gate(&dense_dc);
    let relief_field = chroma_diffusion.repair * luma_relief_gate;
    let chroma_exchange_relief =
        (1.0 - spectral_chroma_relief * relief_field).clamp(0.0, 1.0);
    let exchange_chroma_gamma =
        exchange_gamma * *calibration::TUNABLE_SPECTRAL_CHROMA * chroma_exchange_relief;
    // Cycle 21 local redistribution — mirrors the encoder from the same
    // already-decoded DC grids, so it stays strictly zero-bit.
    let (exchange_local_levels, exchange_local_ladder) = if exchange_gamma > 0.0 {
        hierarchy::spectral_exchange_local_levels(
            &dense_dc,
            &chroma_dense_grids[0],
            &chroma_dense_grids[1],
            *bgh,
            *bgw,
            blocks,
            use_overlap,
            *calibration::TUNABLE_SPECTRAL_LOCAL,
        )
    } else {
        (Vec::new(), {
            let mut table = [0.0f64; hierarchy::SPECTRAL_LOCAL_LEVELS];
            table[0] = 1.0;
            table
        })
    };
    let exchange_local_on =
        !exchange_local_levels.is_empty() && exchange_local_levels.iter().any(|&level| level != 0);
    // Adaptive relief of the luma counter-gift — exact mirror of the encoder.
    // The balancing delta follows the *relieved* chroma slope, so the
    // exchange stays log-neutral at luma strength 1.
    let luma_strength =
        (*calibration::TUNABLE_SPECTRAL_LUMA - luma_pivot_relief * relief_field).max(1.0);
    let exchange_delta = hierarchy::spectral_exchange_delta(
        exchange_gamma * chroma_exchange_relief * luma_strength,
        &turing_block_sizes,
    );
    let chroma_repair = chroma_diffusion.repair * *calibration::TUNABLE_CHROMA_DIFFUSION_REPAIR;
    let c1_chroma_factor =
        hierarchy::chroma_diffusion_step_factor(*calibration::TUNABLE_CHROMA_PHI, chroma_repair);
    let luma_exchange_scale = if exchange_delta > 0.0 {
        golden::PHI.powf(-exchange_delta)
    } else {
        1.0
    };

    eprintln!(
        "  v13 Turing decode: field {}x{}, modulation active=true",
        tf.grid_h, tf.grid_w
    );

    drop(t_fields);
    // === Dequantize AC with Turing step modulation, then synthesize ===
    // Two-pass for CfL: dequantize luma first, then chroma with CfL reconstruction.
    let mut reconstructed_channels: Vec<Vec<f64>> = Vec::with_capacity(3);

    // Shared dequantization logic for one channel, returning (ac_blocks, l_rec_blocks_opt).
    // When `collect_l_rec` is true, also returns per-block dequantized AC in natural order.
    // Per-size LUTs shared by the three dequantize_channel calls. Bit-exact
    // mirror of the per-coefficient path: same functions, same powf, cached
    // once instead of being recomputed for every coefficient.
    struct DequantSizeLuts {
        block_size: usize,
        /// zigzag position -> natural AC index (zz value - 1)
        inv_zz: Vec<usize>,
        /// natural (row, col) factors: qmat_for_block_size(...).powf(qmat_power),
        /// indexed by zigzag position for cache-friendly scanning.
        standard_pow: Vec<f64>,
        /// 8 rotated variants (angles k * PI/8), same powf, same indexing.
        rotated_pow: Vec<Vec<f64>>,
        /// freq_norm of csf_qmat_factor per zigzag position (same sqrt formula).
        csf_freq: Vec<f64>,
    }
    let n_angle_steps = 8usize;
    let angle_step = std::f64::consts::PI / n_angle_steps as f64;
    let build_dequant_luts = |block_size: usize, zz: &[usize]| -> DequantSizeLuts {
        let ac_per_block = block_size * block_size - 1;
        let mut inv_zz = vec![0usize; ac_per_block];
        let mut standard_pow = vec![0.0f64; ac_per_block];
        let mut csf_freq = vec![0.0f64; ac_per_block];
        let bs_max = (block_size - 1).max(1) as f64;
        for (fp, &bi) in zz.iter().enumerate().take(ac_per_block) {
            inv_zz[fp] = bi - 1;
            let row = bi / block_size;
            let col = bi % block_size;
            standard_pow[fp] = lot::qmat_for_block_size(row, col, block_size).powf(qmat_power);
            csf_freq[fp] =
                ((row * row + col * col) as f64).sqrt() / (bs_max * std::f64::consts::SQRT_2);
        }
        let rotated_pow = (0..n_angle_steps)
            .map(|a| {
                let full =
                    lot::precompute_rotated_qmat(block_size, a as f64 * angle_step, qmat_power);
                (0..ac_per_block)
                    .map(|fp| full[zz[fp]])
                    .collect::<Vec<f64>>()
            })
            .collect();
        DequantSizeLuts {
            block_size,
            inv_zz,
            standard_pow,
            rotated_pow,
            csf_freq,
        }
    };
    let dequant_luts = [
        build_dequant_luts(8, &zz_8),
        build_dequant_luts(16, &zz_16),
        build_dequant_luts(32, &zz_32),
    ];

    // Spectral exchange chroma tables (phi^(gamma * freq_norm) per zigzag
    // position), None in the neutral state — exact mirror of the encoder.
    let exchange_tables: Option<Vec<[Vec<f64>; 3]>> = if exchange_gamma > 0.0 {
        let build = |luts: &DequantSizeLuts, gamma: f64| -> Vec<f64> {
            luts.csf_freq
                .iter()
                .map(|&f| golden::PHI.powf(gamma * f))
                .collect()
        };
        let used = if exchange_local_on {
            hierarchy::SPECTRAL_LOCAL_LEVELS
        } else {
            1
        };
        Some(
            (0..used)
                .map(|level| {
                    let gamma = exchange_chroma_gamma * exchange_local_ladder[level];
                    [
                        build(&dequant_luts[0], gamma),
                        build(&dequant_luts[1], gamma),
                        build(&dequant_luts[2], gamma),
                    ]
                })
                .collect(),
        )
    } else {
        None
    };

    let dequant_offset = *calibration::TUNABLE_DEQUANT_OFFSET;
    let dequantize_channel = |raw: &V13ChannelRaw,
                              info: &V13ChannelInfo,
                              ch_idx: usize,
                              _turing_field: &Option<crate::turing::TuringField>,
                              collect_l_rec: bool,
                              chroma_step_mods: Option<&[f64]>|
     -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let _t_dq = crate::timing::stage("dec:dequant");
        let _dc_range = (info.dc_max - info.dc_min).max(1e-6);
        let n_blocks = raw.grid_h * raw.grid_w;

        let chroma_factor = if ch_idx == 1 {
            c1_chroma_factor
        } else {
            info.chroma_factor
        };
        let ac_step = detail_step * chroma_factor * lot_global_factor;
        // Spectral exchange: finer luma steps, coarser chroma HF steps.
        let ac_step = if ch_idx == 0 {
            ac_step * luma_exchange_scale
        } else {
            ac_step
        };

        // Directional QMAT: mirror encoder's gradient-based rotation.
        let gradient_angles = &block_gradient_angles;
        let grad_threshold = *calibration::TUNABLE_GRADIENT_THRESHOLD;

        let block_size_of = |block_idx: usize| -> usize {
            if let Some(ref blocks) = var_blocks {
                if block_idx < blocks.len() {
                    blocks[block_idx].2
                } else {
                    LOT_BLOCK_SIZE
                }
            } else {
                LOT_BLOCK_SIZE
            }
        };

        // Prefix offsets into raw.ac_q so each block can be processed independently.
        let mut ac_offsets = Vec::with_capacity(n_blocks + 1);
        let mut acc = 0usize;
        ac_offsets.push(0);
        for block_idx in 0..n_blocks {
            let bs = block_size_of(block_idx);
            acc += bs * bs - 1;
            ac_offsets.push(acc);
        }

        use rayon::prelude::*;
        let ac_blocks: Vec<Vec<f64>> = (0..n_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let block_size = block_size_of(block_idx);
                let ac_per_block = block_size * block_size - 1;
                let luts = match block_size {
                    8 => &dequant_luts[0],
                    32 => &dequant_luts[2],
                    _ => &dequant_luts[1],
                };
                debug_assert_eq!(luts.block_size, block_size);
                let ac_cursor = ac_offsets[block_idx];

                let mut ac = vec![0.0f64; ac_per_block];

                let l_fov_idx = (block_idx as f64 * l_foveal_map.len() as f64
                    / n_blocks.max(1) as f64) as usize;
                let foveal_factor =
                    l_foveal_map[l_fov_idx.min(l_foveal_map.len().saturating_sub(1))];

                let l_idx =
                    (block_idx as f64 * l_dc_pixel.len() as f64 / n_blocks.max(1) as f64) as usize;
                let dc_l = l_dc_pixel[l_idx.min(l_dc_pixel.len().saturating_sub(1))];
                // Same operation order as lot::csf_qmat_factor:
                // csf = 1.0 + ((CSF_DARK_BOOST * (1 - lum_norm)) * f) * f
                let lum_norm = (dc_l / 255.0).clamp(0.0, 1.0);
                let csf_ka = calibration::CSF_DARK_BOOST * (1.0 - lum_norm);

                let turing_mod = homeostatic_turing_mods
                    .get(block_idx)
                    .copied()
                    .unwrap_or(1.0);
                let chroma_breathing_mod = chroma_step_mods
                    .and_then(|mods| mods.get(block_idx))
                    .copied()
                    .unwrap_or(1.0);
                let local_step = ac_step * foveal_factor * turing_mod * chroma_breathing_mod;
                let step_clamped = local_step.max(0.1);

                // Select rotated or standard QMAT (must match encoder exactly)
                let (blk_angle, blk_strength, blk_qmat_blend) = if block_idx < gradient_angles.len()
                {
                    gradient_angles[block_idx]
                } else {
                    (0.0, 0.0, 0.0)
                };
                let use_rotated = blk_strength > grad_threshold && blk_qmat_blend > 0.0;
                // Quantize angle to same 8 steps as encoder for bit-exact match
                let rotated_lut: Option<&[f64]> = if use_rotated {
                    let a = blk_angle.rem_euclid(std::f64::consts::PI);
                    let idx = ((a / angle_step + 0.5) as usize) % n_angle_steps;
                    Some(&luts.rotated_pow[idx])
                } else {
                    None
                };
                let wave_qfactors = if ch_idx == 0 && subsonic_wave_strength > 0.0 {
                    Some(hierarchy::subsonic_wave_qfactors(
                        subsonic_waves.get(block_idx).copied().unwrap_or_default(),
                        block_size,
                        subsonic_wave_strength,
                    ))
                } else {
                    None
                };

                for flat_pos in 0..ac_per_block {
                    let qi = if ac_cursor + flat_pos < raw.ac_q.len() {
                        raw.ac_q[ac_cursor + flat_pos]
                    } else {
                        0
                    };
                    let ac_idx = luts.inv_zz[flat_pos];
                    let standard_qfactor = luts.standard_pow[flat_pos];
                    let mut qfactor = if let Some(rot) = rotated_lut {
                        hierarchy::blend_qmat_factors(
                            standard_qfactor,
                            rot[flat_pos],
                            blk_qmat_blend,
                        )
                    } else {
                        standard_qfactor
                    };

                    if use_csf {
                        let f = luts.csf_freq[flat_pos];
                        let csf = 1.0 + csf_ka * f * f;
                        qfactor *= csf;
                    }
                    if ch_idx > 0 {
                        if let Some(ref tables) = exchange_tables {
                            let level =
                                exchange_local_levels.get(block_idx).copied().unwrap_or(0) as usize;
                            let set = &tables[level.min(tables.len() - 1)];
                            let exch = match block_size {
                                8 => &set[0],
                                32 => &set[2],
                                _ => &set[1],
                            };
                            qfactor *= exch[flat_pos];
                        }
                    }
                    if let Some(ref wave) = wave_qfactors {
                        qfactor *= wave[luts.inv_zz[flat_pos] + 1];
                    }

                    // Dequant offset: nonzero coefficients reconstruct away
                    // from the bin center, restoring quantized-away texture
                    // energy (mirrored by the encoder's own reconstruction).
                    let q_off = qi as f64 + dequant_offset * (qi as f64).signum();
                    ac[ac_idx] = q_off * step_clamped * qfactor;
                }
                ac
            })
            .collect();

        let l_rec_blocks: Vec<Vec<f64>> = if collect_l_rec {
            ac_blocks.clone()
        } else {
            Vec::new()
        };

        (ac_blocks, l_rec_blocks)
    };

    // Cycle 32/33 — EOB-driven step fields (AUR2 v30 luma, v31 chroma): the
    // exact mirror of the encoder's second pass, derived from the very EOB
    // positions the stream transmitted (zero-bit). Merges the field into a
    // channel's existing per-block step modulations.
    let merge_eob_step_mods = |ch_idx: usize, base: &[f64]| -> Vec<f64> {
        let eob_step_min_q = calibration::env_override("AUREA_EOB_STEP_MIN_Q")
            .map(|v| v as u8)
            .unwrap_or(calibration::V30_EOB_STEP_MIN_QUALITY);
        let eta = if header.quality >= eob_step_min_q {
            if ch_idx == 0 {
                // v32: softened on the diffuse-dark field (SSIM protection).
                // Only below the anti-ring boundary: at fine steps the field
                // does not collapse variance, and its full strength is what
                // wins the high-quality texture metrics.
                let luma_relief = if header.version >= 32
                    && header.quality < calibration::V29_ANTI_RING_OFF_QUALITY
                {
                    1.0 - calibration::V32_EOB_LUMA_RELIEF * relief_field
                } else {
                    1.0
                };
                calibration::env_override("AUREA_EOB_STEP")
                    .map(|v| v.clamp(0.0, 2.0))
                    .unwrap_or(if header.version >= 30 {
                        calibration::V30_EOB_STEP_ETA * luma_relief
                    } else {
                        0.0
                    })
            } else {
                calibration::env_override("AUREA_EOB_STEP_CHROMA")
                    .map(|v| v.clamp(0.0, 2.0))
                    .unwrap_or(if header.version >= 31 {
                        calibration::V31_EOB_STEP_CHROMA_ETA
                            * calibration::eob_chroma_gate(exchange_gamma)
                    } else {
                        0.0
                    })
            }
        } else {
            0.0
        };
        let channel_eobs = eob_positions_by_channel.get(ch_idx);
        match channel_eobs {
            Some(positions) if eta > 0.0 && !positions.is_empty() => {
                let eobs: Vec<usize> = positions.iter().map(|&e| e as usize).collect();
                let capacities: Vec<f64> = (0..eobs.len())
                    .map(|i| {
                        let block_size = var_blocks
                            .as_ref()
                            .map_or(LOT_BLOCK_SIZE, |blocks| blocks[i].2);
                        (block_size * block_size - 1) as f64
                    })
                    .collect();
                let mods = hierarchy::eob_step_mods(&eobs, &capacities, eta);
                base.iter().zip(&mods).map(|(b, m)| b * m).collect()
            }
            _ => base.to_vec(),
        }
    };

    // --- Pass 1: Dequantize and synthesize luma (ch_idx=0) ---
    let l_rec_ac_blocks: Vec<Vec<f64>>;
    {
        let raw = &raw_channels[0];
        let info = &channel_infos[0];
        let dc_range = (info.dc_max - info.dc_min).max(1e-6);

        let dc_denorm: Vec<f64> = raw
            .dc_q
            .iter()
            .map(|&q| q as f64 * dc_step_clamped * dc_range / 255.0 + info.dc_min)
            .collect();

        let (ac_blocks, l_rec) = dequantize_channel(
            raw,
            info,
            0,
            &turing_field,
            use_cfl,
            Some(&merge_eob_step_mods(0, &bright_mods)),
        );
        l_rec_ac_blocks = l_rec;

        let _t_synth = crate::timing::stage("dec:idct_overlap");
        let recon = if let Some(ref blocks) = var_blocks {
            lot::lot_synthesize_variable(
                &dc_denorm,
                &ac_blocks,
                blocks,
                info.chan_h,
                info.chan_w,
                use_overlap,
            )
        } else {
            lot::lot_synthesize_image(
                &dc_denorm,
                &ac_blocks,
                raw.grid_h,
                raw.grid_w,
                info.chan_h,
                info.chan_w,
                LOT_BLOCK_SIZE,
                use_overlap,
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

        let dc_denorm: Vec<f64> = raw
            .dc_q
            .iter()
            .map(|&q| q as f64 * dc_step_clamped * dc_range / 255.0 + info.dc_min)
            .collect();

        let (mut ac_blocks, _) = dequantize_channel(
            raw,
            info,
            ch_idx,
            &turing_field,
            false,
            Some(&merge_eob_step_mods(
                ch_idx,
                &chroma_breathing_mods[ch_idx - 1],
            )),
        );

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
                            &l_rec_block[..len],
                            &residual[..len],
                            cfd.alpha_indices[block_idx],
                        );
                        ac_blocks[block_idx] = reconstructed;
                    }
                }
            }
        }

        let _t_synth = crate::timing::stage("dec:idct_overlap");
        let recon = if let Some(ref blocks) = var_blocks {
            lot::lot_synthesize_variable(
                &dc_denorm,
                &ac_blocks,
                blocks,
                info.chan_h,
                info.chan_w,
                use_overlap,
            )
        } else {
            lot::lot_synthesize_image(
                &dc_denorm,
                &ac_blocks,
                raw.grid_h,
                raw.grid_w,
                info.chan_h,
                info.chan_w,
                LOT_BLOCK_SIZE,
                use_overlap,
            )
        };
        reconstructed_channels.push(recon);
    }

    let t_deblock = crate::timing::stage("dec:deblock_antiring");
    let mut l_flat = reconstructed_channels.remove(0);
    let mut c1_sub_flat = reconstructed_channels.remove(0);
    let mut c2_sub_flat = reconstructed_channels.remove(0);

    // Gas-only deblocking (smooth-zone block boundaries only). Laboratory
    // gates: the post chain is pure decoder output shaping — the encoder
    // never simulates it — so these knobs are measurement-only and any
    // default change must ride a version bump.
    let post_t0 = std::time::Instant::now();
    if *calibration::TUNABLE_DEBLOCK {
        // Three independent planes — deblock them concurrently.
        rayon::join(
            || dsp::deblock_gas_only(&mut l_flat, height, width, LOT_BLOCK_SIZE),
            || {
                rayon::join(
                    || dsp::deblock_gas_only(&mut c1_sub_flat, c1_h, c1_w, LOT_BLOCK_SIZE),
                    || dsp::deblock_gas_only(&mut c2_sub_flat, c2_h, c2_w, LOT_BLOCK_SIZE),
                );
            },
        );
    }

    // Anti-ring sigma filter — cleans mixed-block artifacts near edges.
    // Sky pixels in blocks straddling tree/sky boundary get pulled toward correct
    // sky values by excluding dark tree pixels from the local sigma average.
    // v29 texture law, keyed on the header (version + quality): sharpening
    // restores low-quality texture, and the anti-ring sigma filter — which
    // was measured to cost LPIPS, DISTS *and* PSNR once steps are fine —
    // switches off at high quality. Env knobs override for laboratory A/B.
    let v29_texture =
        header.version >= 29 && header.quality <= calibration::V29_TEXTURE_MAX_QUALITY;
    let v29_anti_ring_off =
        header.version >= 29 && header.quality >= calibration::V29_ANTI_RING_OFF_QUALITY;
    let post_deblock = post_t0.elapsed();
    let post_t1 = std::time::Instant::now();
    let anti_ring_scale = calibration::env_override("AUREA_ANTI_RING_SCALE")
        .map(|v| v.clamp(0.0, 4.0))
        .unwrap_or(if v29_anti_ring_off { 0.0 } else { 1.0 });
    if anti_ring_scale > 0.0 {
        dsp::anti_ring_sigma(&mut l_flat, height, width, detail_step * anti_ring_scale);
    }
    let post_ar = post_t1.elapsed();
    let post_t2 = std::time::Instant::now();

    // Texture-restoration chain on the luma plane (PTF space).
    // v32: the sharpening chain is muted on the diffuse-dark field (R·G,
    // the kryte-class signature). On flat-chroma dark images there is no
    // chromatic texture to reveal: sharpening there was measured to cost
    // every BD axis — SSIM-Y most of all — while the rate-matched pair wins
    // without it. The field is already reconstructed above, zero-bit.
    let sharpen_scale = if header.version >= 32 {
        1.0 - relief_field
    } else {
        1.0
    };
    let cas = *calibration::TUNABLE_CAS_SHARPEN;
    let xsharpen = calibration::env_override("AUREA_XSHARPEN")
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(if v29_texture {
            calibration::V29_XSHARPEN * sharpen_scale
        } else {
            0.0
        });
    let xsharpen_thresh = calibration::env_override("AUREA_XSHARPEN_THRESH")
        .map(|v| v.clamp(0.0, 0.5))
        .unwrap_or(calibration::V29_XSHARPEN_THRESH) as f32;
    let sharpen_plane = |plane: &mut Vec<f64>, w: usize, h: usize, strength: f64| {
        use rayon::prelude::*;
        let inv255 = 1.0 / 255.0;
        let mut plane_f32: Vec<f32> = plane.par_iter().map(|&v| (v * inv255) as f32).collect();
        if cas > 0.0 {
            plane_f32 = postprocess::apply_cas_sharpening(&plane_f32, w, h, cas as f32);
        }
        let radius = calibration::env_override("AUREA_XSHARPEN_RADIUS")
            .map(|v| (v as usize).clamp(1, 4))
            .unwrap_or(calibration::V29_XSHARPEN_RADIUS);
        let min_range = calibration::env_override("AUREA_XSHARPEN_MIN_RANGE")
            .unwrap_or(calibration::V29_XSHARPEN_MIN_RANGE) as f32;
        plane_f32 = postprocess::apply_xsharpen_gated(
            &plane_f32,
            w,
            h,
            strength as f32,
            xsharpen_thresh,
            radius,
            min_range,
        );
        let second_pass = *calibration::TUNABLE_XSHARPEN2;
        if second_pass > 0.0 {
            plane_f32 = postprocess::apply_xsharpen_radius(
                &plane_f32,
                w,
                h,
                second_pass as f32,
                xsharpen_thresh,
                1,
            );
        }
        plane
            .par_iter_mut()
            .zip(plane_f32.par_iter())
            .for_each(|(dst, &src)| {
                *dst = f64::from(src) * 255.0;
            });
    };
    if cas > 0.0 || xsharpen > 0.0 {
        sharpen_plane(&mut l_flat, width, height, xsharpen);
    }
    let post_xs = post_t2.elapsed();
    let post_t3 = std::time::Instant::now();
    // Laboratory linear unsharp mask (adds edge overshoot energy). The blur
    // is `usm_passes` rounds of a separable 1-2-1 kernel.
    let usm = calibration::env_override("AUREA_USM")
        .map(|v| v.clamp(0.0, 2.0))
        .unwrap_or(if v29_texture {
            calibration::V29_USM * sharpen_scale
        } else {
            0.0
        });
    let usm_chroma = *calibration::TUNABLE_USM_CHROMA;
    let usm_passes = calibration::env_override("AUREA_USM_RADIUS")
        .map(|v| (v as usize).clamp(1, 4))
        .unwrap_or(calibration::V29_USM_RADIUS);
    let apply_usm = |plane: &mut Vec<f64>, w: usize, h: usize, amount: f64, clamp255: bool| {
        use rayon::prelude::*;
        if amount <= 0.0 || h < 3 || w < 3 {
            return;
        }
        // Ping-pong blur: the horizontal pass reads `blur` and writes the
        // scratch, the vertical pass reads the scratch and writes `blur` —
        // the exact operations of the historical clone-per-pass loop with
        // zero per-pass allocation.
        let mut blur = plane.clone();
        let mut scratch = vec![0.0f64; blur.len()];
        for _ in 0..usm_passes {
            {
                let src = &blur;
                scratch
                    .par_chunks_mut(w)
                    .enumerate()
                    .for_each(|(r, row)| {
                        let base = r * w;
                        for c in 0..w {
                            let l = src[base + c.saturating_sub(1)];
                            let m = src[base + c];
                            let rgt = src[base + (c + 1).min(w - 1)];
                            row[c] = 0.25 * l + 0.5 * m + 0.25 * rgt;
                        }
                    });
            }
            let horizontal = &scratch;
            blur.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
                let up = r.saturating_sub(1) * w;
                let mid = r * w;
                let down = (r + 1).min(h - 1) * w;
                for c in 0..w {
                    row[c] = 0.25 * horizontal[up + c]
                        + 0.5 * horizontal[mid + c]
                        + 0.25 * horizontal[down + c];
                }
            });
        }
        // Coring: micro-deltas are gradient or banding steps, not texture —
        // amplifying them was the measured chilling failure. Soft-knee.
        let coring = calibration::env_override("AUREA_USM_CORING")
            .unwrap_or(calibration::V29_USM_CORING);
        plane
            .par_iter_mut()
            .zip(blur.par_iter())
            .for_each(|(v, &b)| {
                let delta = *v - b;
                let cored = if delta.abs() <= coring {
                    0.0
                } else {
                    delta - coring * delta.signum()
                };
                let sharpened = *v + amount * cored;
                *v = if clamp255 {
                    sharpened.clamp(0.0, 255.0)
                } else {
                    sharpened
                };
            });
    };
    apply_usm(&mut l_flat, width, height, usm, true);
    if usm_chroma > 0.0 {
        apply_usm(&mut c1_sub_flat, c1_w, c1_h, usm_chroma, false);
        apply_usm(&mut c2_sub_flat, c2_w, c2_h, usm_chroma, false);
    }
    // Laboratory deterministic grain: a coordinate-hash pseudo-noise on luma,
    // scaled by the detail step. Restores the texture energy that
    // quantization removed; deterministic, so identical on both sides of any
    // future format promotion.
    let grain = *calibration::TUNABLE_GRAIN;
    if grain > 0.0 {
        use rayon::prelude::*;
        let amplitude = grain * detail_step;
        // Darkness gate exponent: 0 = uniform grain (measured negative),
        // p > 0 concentrates the energy in dark texture where the Weber
        // budget transfer starves detail the most.
        let dark_exp = *calibration::TUNABLE_GRAIN_DARK_EXP;
        l_flat
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, line)| {
                for (col, v) in line.iter_mut().enumerate() {
                    let mut h = (row as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((col as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                    h ^= h >> 29;
                    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    h ^= h >> 32;
                    // Uniform in [-1, 1), 53-bit mantissa precision.
                    let noise = (h >> 11) as f64 * (2.0 / 9007199254740992.0) - 1.0;
                    let gate = if dark_exp > 0.0 {
                        (1.0 - (*v / 255.0).clamp(0.0, 1.0)).powf(dark_exp)
                    } else {
                        1.0
                    };
                    *v += amplitude * gate * noise;
                }
            });
    }

    if std::env::var("AUREA_TIMINGS").map(|v| v == "1").unwrap_or(false) {
        eprintln!(
            "[AUREA_TIMINGS]   dec:post_detail  deblock={:.3}s ar={:.3}s xs={:.3}s usm={:.3}s",
            post_deblock.as_secs_f64(),
            post_ar.as_secs_f64(),
            post_xs.as_secs_f64(),
            post_t3.elapsed().as_secs_f64(),
        );
    }
    let xsharpen_chroma = *calibration::TUNABLE_XSHARPEN_CHROMA;
    if xsharpen_chroma > 0.0 {
        // Chroma planes are signed around zero; shift into [0,1] for the filter.
        for (plane, (w, h)) in [
            (&mut c1_sub_flat, (c1_w, c1_h)),
            (&mut c2_sub_flat, (c2_w, c2_h)),
        ] {
            for v in plane.iter_mut() {
                *v += 128.0;
            }
            sharpen_plane(plane, w, h, xsharpen_chroma);
            for v in plane.iter_mut() {
                *v -= 128.0;
            }
        }
    }
    drop(t_deblock);

    // The PTF inverse is fused into the final GCT+packing pass below: same
    // per-pixel operations in the same order, without materializing the PTF
    // plane and the three RGB planes (five full-plane round-trips saved).

    // Upsample chroma (v13 requires full-resolution 4:4:4).
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
    if use_chroma_resid {
        let mask_size = reader.read_u32("chroma-residual mask length")? as usize;
        let block_rows = height.div_ceil(8);
        let block_cols = width.div_ceil(8);
        let block_count = block_rows
            .checked_mul(block_cols)
            .ok_or_else(|| invalid_format("chroma-residual block count overflow"))?;
        let expected_mask_size = block_count.div_ceil(8);
        if mask_size != expected_mask_size {
            return Err(invalid_format("invalid chroma-residual mask length"));
        }
        let c1_mask = reader.take(mask_size, "chroma-residual mask")?.to_vec();
        let resid_size = reader.read_u32("chroma-residual data length")? as usize;
        if resid_size > n_pixels {
            return Err(invalid_format("chroma-residual data exceeds image size"));
        }
        let c1_resid: Vec<i8> = reader
            .take(resid_size, "chroma-residual data")?
            .iter()
            .map(|&b| b as i8)
            .collect();
        let chroma_resid_step = detail_step * calibration::CHROMA_RESIDUAL_STEP_MULT;
        color::decode_chroma_residual(
            &mut c1_full,
            &c1_mask,
            &c1_resid,
            height,
            width,
            chroma_resid_step,
        );
    }

    if !partial_progressive && reader.remaining() != 0 {
        return Err(invalid_format(format!(
            "{} trailing bytes after final AUR2 section at body offset {}",
            reader.remaining(),
            reader.position(),
        )));
    }

    // PTF⁻¹ + GCT inverse + packing, one fused pass -> RGB.
    let t_gct = crate::timing::stage("dec:gct_inv");
    let n = n_pixels;
    let mut rgb = vec![0u8; n * 3];
    {
        use golden::{PHI, PHI_INV, PTF_GAMMA_INV};
        use rayon::prelude::*;
        // Identical constants to `rotate_inverse_with_luma_blend` at blend 0
        // (x·1.0 = x and x + 0.0 = x are exact).
        let golden_g = 0.5;
        let c1_to_g = (PHI_INV / (2.0 * PHI)) / golden_g;
        let c2_to_g = (1.0 / (2.0 * PHI)) / golden_g;
        let inv255 = 1.0 / 255.0;
        const PACK_CHUNK: usize = 4096;
        rgb.par_chunks_mut(3 * PACK_CHUNK)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci * PACK_CHUNK;
                for (j, px) in chunk.chunks_exact_mut(3).enumerate() {
                    let i = base + j;
                    let l = 255.0 * (l_flat[i] * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA_INV);
                    let c1 = c1_full[i];
                    let c2 = c2_full[i];
                    px[0] = (l + c2).round().clamp(0.0, 255.0) as u8;
                    px[1] = (l - c1_to_g * c1 - c2_to_g * c2).round().clamp(0.0, 255.0) as u8;
                    px[2] = (l + c1).round().clamp(0.0, 255.0) as u8;
                }
            });
    }
    drop(t_gct);
    crate::timing::report("decode");

    Ok(DecodedImage {
        rgb,
        width,
        height,
        progressive: progressive_info,
    })
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::aurea_encoder::{self, AureaEncoderParams};

    fn encoded_fixture_with_quality(quality: u8) -> Vec<u8> {
        let width = 32usize;
        let height = 24usize;
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                rgb.push(((x * 7 + y * 3) & 0xff) as u8);
                rgb.push(((x * 2 + y * 11) & 0xff) as u8);
                rgb.push(((x * 13 + y * 5) & 0xff) as u8);
            }
        }
        aurea_encoder::encode_aur2_v13(
            &rgb,
            width,
            height,
            &AureaEncoderParams {
                quality,
                n_representatives: 64,
                geometric: false,
                lot_topology: lot::LotTopology::AdaptiveCritical,
            },
        )
        .unwrap()
        .aurea_data
    }

    fn encoded_fixture() -> Vec<u8> {
        encoded_fixture_with_quality(75)
    }

    /// The version the default build emits. Both the spectral envelope and the
    /// logistic mix are promoted, so an unconfigured encoder writes v22; with
    /// mixing off it writes v21, and fully neutralized it writes v20.
    fn expected_version() -> u8 {
        let config = *calibration::TUNABLE_SPECTRAL_ENVELOPE;
        // v25 rides on the promoted v24 grammar: it keys the AC contexts on
        // intra-block activity instead of the Turing bucket. v26 additionally
        // opens each detail layer on the echo of its own skeleton.
        if *calibration::TUNABLE_ACTIVITY_CONTEXT && base_expected_version(&config) == 24 {
            return if *calibration::TUNABLE_ECHO_CONTEXT {
                if *calibration::TUNABLE_CHANNEL_MEMORY {
                    // v28 relieves the luma pivot and the chroma exchange
                    // slope on the diffuse-dark field; v29 adds the
                    // quality-keyed perceptual texture law on top.
                    if *calibration::TUNABLE_LUMA_PIVOT_RELIEF > 0.0
                        || *calibration::TUNABLE_SPECTRAL_CHROMA_RELIEF > 0.0
                    {
                        32
                    } else {
                        27
                    }
                } else {
                    26
                }
            } else {
                25
            };
        }
        base_expected_version(&config)
    }

    fn base_expected_version(config: &spectral_envelope::SpectralEnvelopeConfig) -> u8 {
        if config.is_neutral() {
            20
        } else if config.mix_rate == 0 {
            21
        } else if config.mix_inputs == spectral_envelope::MIX_INPUT_FOREIGN
            && !config.mix_pm1
            && config.apm_rate == 0
            && !config.mix_sign
        {
            22
        } else {
            let promoted = spectral_envelope::SpectralEnvelopeConfig::promoted();
            if config.mix_inputs == promoted.mix_inputs && config.mix_sign == promoted.mix_sign {
                24
            } else {
                23
            }
        }
    }

    #[test]
    fn v22_end_to_end_roundtrip_has_expected_shape() {
        let encoded = encoded_fixture();
        assert_eq!(&encoded[..4], b"AUR2");
        // v21 adds the causal spectral envelope, v22 the logistic mix.
        assert_eq!(encoded[4], expected_version());

        let decoded = decode_aurea(&encoded).unwrap();
        assert_eq!((decoded.width, decoded.height), (32, 24));
        assert_eq!(decoded.rgb.len(), 32 * 24 * 3);
    }

    /// A uniform image is the degenerate case of the zero-bit Turing field: its
    /// DoG is numerically zero, so the field must be zero on both sides.
    ///
    /// It used not to be. `normalize_01` guarded the flat case with an
    /// *absolute* `f64::EPSILON`, while the DC grid it derives from reaches
    /// ~4·10⁴ — so rounding residue of order 10⁻¹¹ slipped past the guard and
    /// was stretched to the full `[0, 1]` range. The encoder amplified the
    /// residue of an f64 grid and the decoder that of the same grid rounded
    /// through the f32 header, the entropy contexts disagreed, and the file
    /// became undecodable. Mid-greys failed at every quality; only pure black
    /// and pure white survived.
    #[test]
    fn uniform_images_survive_the_round_trip() {
        for level in [0u8, 16, 64, 128, 192, 240, 255] {
            for quality in [20u8, 40, 75] {
                let (width, height) = (64usize, 64usize);
                let rgb = vec![level; width * height * 3];
                let encoded = aurea_encoder::encode_aur2_v13(
                    &rgb,
                    width,
                    height,
                    &AureaEncoderParams {
                        quality,
                        n_representatives: 64,
                        geometric: false,
                        lot_topology: lot::LotTopology::AdaptiveCritical,
                    },
                )
                .unwrap()
                .aurea_data;
                let decoded = decode_aurea(&encoded)
                    .unwrap_or_else(|e| panic!("level={level} q={quality}: {e}"));
                assert_eq!((decoded.width, decoded.height), (width, height));
                // A flat source must also reconstruct flat: an amplified noise
                // field would show up as scatter across the plane.
                let spread = decoded.rgb.iter().copied().max().unwrap() as i32
                    - decoded.rgb.iter().copied().min().unwrap() as i32;
                assert!(
                    spread <= 2,
                    "level={level} q={quality}: flat source reconstructed with spread {spread}"
                );
            }
        }
    }

    #[test]
    fn v22_phase_codon_end_to_end_roundtrip() {
        for quality in [20u8, 40, 60, 80] {
            let encoded = encoded_fixture_with_quality(quality);
            assert_eq!(encoded[4], expected_version(), "quality={quality}");
            let decoded = decode_aurea(&encoded).unwrap();
            assert_eq!((decoded.width, decoded.height), (32, 24));
            assert_eq!(decoded.rgb.len(), 32 * 24 * 3);
        }
    }

    /// The envelope reads a block's earlier coefficients across the progressive
    /// boundary: a detail layer resolves neighbours either in the skeleton it
    /// inherits or in its own flat sequence. Decoding every prefix length
    /// exercises both branches and would diverge immediately on any asymmetry.
    #[test]
    fn spectral_envelope_survives_every_progressive_prefix() {
        let encoded = encoded_fixture_with_quality(40);
        let full = decode_aurea(&encoded).unwrap();
        let total = decode_aurea_with_progressive_layers(&encoded, None)
            .unwrap()
            .progressive
            .expect("fixture must be progressive")
            .total_layers;
        for layers in 0..=total {
            let partial = decode_aurea_with_progressive_layers(&encoded, Some(layers)).unwrap();
            assert_eq!((partial.width, partial.height), (32, 24));
            if layers == total {
                assert_eq!(partial.rgb, full.rgb);
            }
        }
    }

    #[test]
    fn progressive_prefix_is_physically_decodable_and_matches_full_source() {
        let encoded = encoded_fixture();
        let from_full = decode_aurea_with_progressive_layers(&encoded, Some(1)).unwrap();
        let info = from_full
            .progressive
            .as_ref()
            .expect("default v13 fixture must be progressive");
        assert_eq!(info.decoded_layers, 1);
        assert!(info.total_layers >= 1);
        assert!(info.prefix_bytes < encoded.len());

        let prefix = &encoded[..info.prefix_bytes];
        let from_prefix = decode_aurea_with_progressive_layers(prefix, Some(1)).unwrap();
        assert_eq!(from_prefix.rgb, from_full.rgb);
        assert_eq!(from_prefix.progressive.unwrap().prefix_bytes, prefix.len());
    }

    #[test]
    fn all_progressive_layers_match_the_default_full_decode() {
        let encoded = encoded_fixture();
        let full = decode_aurea(&encoded).unwrap();
        let layer_count = full.progressive.as_ref().unwrap().total_layers;
        let layered = decode_aurea_with_progressive_layers(&encoded, Some(layer_count)).unwrap();
        assert_eq!(layered.rgb, full.rgb);
    }

    #[test]
    fn v13_decoder_rejects_truncation_without_panicking() {
        let encoded = encoded_fixture();
        let cuts = [
            0,
            3,
            4,
            38,
            39,
            50,
            72,
            encoded.len() / 2,
            encoded.len() - 1,
        ];
        for cut in cuts {
            let result = std::panic::catch_unwind(|| decode_aurea(&encoded[..cut]));
            assert!(
                result.is_ok(),
                "decoder panicked for truncation at byte {cut}"
            );
            assert!(
                result.unwrap().is_err(),
                "decoder accepted truncation at byte {cut}"
            );
        }
    }

    #[test]
    fn v13_decoder_rejects_oversized_sections_and_trailing_bytes() {
        let encoded = encoded_fixture();

        // Base header (39) + chroma dims/flags (12) + Turing header (16)
        // + block-map dimensions (4) = map length at offsets 71..73.
        let mut oversized_map = encoded.clone();
        oversized_map[71..73].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(decode_aurea(&oversized_map).is_err());

        let mut trailing = encoded.clone();
        trailing.push(0x5a);
        assert!(decode_aurea(&trailing).is_err());
    }

    #[test]
    fn v13_decoder_rejects_resource_exhaustion_dimensions_early() {
        let mut encoded = encoded_fixture();
        encoded[6..8].copy_from_slice(&u16::MAX.to_le_bytes());
        encoded[8..10].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(decode_aurea(&encoded).is_err());
    }

    #[test]
    fn decoder_rejects_old_versions() {
        let mut encoded = encoded_fixture();
        encoded[4] = 12;
        assert!(matches!(
            decode_aurea(&encoded),
            Err(AureaError::UnsupportedVersion(12))
        ));
    }
}
