/// Native AUREA v10 encoder: RGB -> .aur file (AUR2 format).
/// Modifié pour l'architecture A.D.N. (Codons spatiaux)
///
/// v2 (LOT): Block-based Lapped Orthogonal Transform replaces CDF 9/7 wavelet.

use ndarray::Array2;

use crate::bitstream::{self, Aur2Header};
use crate::calibration;
use crate::color;
use crate::geometric;
use crate::lot;
use crate::rans;
use crate::scene_analysis;
use crate::wavelet;

// ======================================================================
// v3 feature flags (bitstream body)
// ======================================================================
pub const FLAG_CSF_MODULATION: u16   = 0x0001; // Point 5: CSF frequency-dependent QMAT
pub const FLAG_CHROMA_RESIDUAL: u16  = 0x0002; // Point 3: adaptive chroma residual
pub const FLAG_VARIABLE_BLOCKS: u16  = 0x0004; // Point 2: quadtree 8/16/32
pub const FLAG_SCENE_ANALYSIS: u16   = 0x0008; // Point 7: scene analysis in encoder
pub const FLAG_STRUCTURAL: u16       = 0x0010; // Point 4: structural coherence (4D codon)
pub const FLAG_DPCM_DC: u16          = 0x0020; // Stage 3: golden DPCM for DC

pub const DEFAULT_V3_FLAGS: u16 = FLAG_CSF_MODULATION | FLAG_SCENE_ANALYSIS | FLAG_DPCM_DC;

/// Golden DPCM prediction for DC grid in raster order.
/// Called by both encoder and decoder.
pub fn golden_dc_predict(dc_grid: &[i16], gy: usize, gx: usize, gw: usize) -> i16 {
    use crate::golden::{PHI_INV, PHI_INV2, PHI_INV3};

    if gy == 0 && gx == 0 { return 0; }
    if gy == 0 { return dc_grid[gx - 1]; }
    if gx == 0 { return dc_grid[(gy - 1) * gw]; }

    let left = dc_grid[gy * gw + gx - 1] as f64;
    let top = dc_grid[(gy - 1) * gw + gx] as f64;
    let diag = dc_grid[(gy - 1) * gw + gx - 1] as f64;
    let norm = PHI_INV + PHI_INV2 + PHI_INV3;
    let pred = (left * PHI_INV + top * PHI_INV2 + diag * PHI_INV3) / norm;
    pred.round() as i16
}

pub struct AureaEncoderParams {
    pub quality: u8,
    pub n_representatives: usize,
    pub geometric: bool,
}

pub struct AureaEncoderResult {
    pub aurea_data: Vec<u8>,
    pub compressed_size: usize,
}

pub fn auto_detail_step(l_channel: &[f64], height: usize, width: usize, quality: u8) -> f64 {
    let mut sum_dy2 = 0.0;
    let mut count_dy = 0usize;
    let mut sum_dx2 = 0.0;
    let mut count_dx = 0usize;

    for i in 0..height - 1 {
        for j in 0..width {
            let dy = l_channel[(i + 1) * width + j] - l_channel[i * width + j];
            sum_dy2 += dy * dy;
            count_dy += 1;
        }
    }
    for i in 0..height {
        for j in 0..width - 1 {
            let dx = l_channel[i * width + j + 1] - l_channel[i * width + j];
            sum_dx2 += dx * dx;
            count_dx += 1;
        }
    }
    let mean_energy = sum_dy2 / count_dy.max(1) as f64 + sum_dx2 / count_dx.max(1) as f64;
    let complexity = (mean_energy + 1.0).log2();
    let c_norm = ((complexity - 6.0) / 5.0).clamp(0.0, 1.0);
    let q_frac = (quality as f64 / 100.0).clamp(0.01, 1.0);
    let base_step = 2.0 + (1.0 - q_frac) * 28.0;
    let factor = 1.0 - 0.4 * c_norm;

    (base_step * factor).clamp(1.0, 48.0)
}

pub fn quality_to_n_repr(quality: u8) -> usize {
    match quality {
        1..=15 => 16, 16..=30 => 24, 31..=50 => 32,
        51..=70 => 48, 71..=85 => 64, 86..=95 => 96, _ => 128,
    }
}

pub fn encode_aurea_v2(
    rgb: &[u8], width: usize, height: usize, params: &AureaEncoderParams,
) -> Result<AureaEncoderResult, Box<dyn std::error::Error>> {
    encode_aur2(rgb, width, height, params)
}

pub fn encode_aur2(
    rgb: &[u8], width: usize, height: usize, params: &AureaEncoderParams,
) -> Result<AureaEncoderResult, Box<dyn std::error::Error>> {
    let n = width * height;

    // Point 7: Pre-encoding scene analysis on L channel
    let (l_ch, _, _) = color::golden_rotate_forward(rgb, n);
    let scene_profile = scene_analysis::quick_scene_classify(&l_ch, height, width);

    // Point 1: Adaptive transform choice
    let use_lot = scene_analysis::recommend_lot(&scene_profile);
    eprintln!("  Scene: {:?}, smooth={:.0}%, transform={}",
              scene_profile.scene_type, scene_profile.smooth_pct,
              if use_lot { "LOT" } else { "CDF97" });

    if use_lot {
        encode_aur2_lot(rgb, width, height, params)
    } else {
        // CDF97 path for very smooth images
        encode_aur2_v1(rgb, width, height, params)
    }
}

/// v1 encoder (CDF 9/7 wavelet + geometric primitives). Kept for reference/fallback.
#[allow(dead_code)]
pub fn encode_aur2_v1(
    rgb: &[u8], width: usize, height: usize, params: &AureaEncoderParams,
) -> Result<AureaEncoderResult, Box<dyn std::error::Error>> {
    let n = width * height;
    let (mut l_ch, c1_ch, c2_ch) = color::golden_rotate_forward(rgb, n);

    {
        use crate::golden::PTF_GAMMA;
        let inv255 = 1.0 / 255.0;
        for v in l_ch.iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA);
        }
    }

    let (c1_sub, c1_h, c1_w) = color::subsample_420_encode(&c1_ch, height, width);
    let (c2_sub, c2_h, c2_w) = color::subsample_422_encode(&c2_ch, height, width);

    let wv_levels = wavelet::auto_wv_levels_v10(height, width);

    let l_arr = Array2::from_shape_vec((height, width), l_ch.clone()).unwrap();
    let c1_arr = Array2::from_shape_vec((c1_h, c1_w), c1_sub.clone()).unwrap();
    let c2_arr = Array2::from_shape_vec((c2_h, c2_w), c2_sub.clone()).unwrap();

    let (l_ll, l_subs, l_sizes) = wavelet::wavelet_decompose(&l_arr, wv_levels);
    let (c1_ll, c1_subs, c1_sizes) = wavelet::wavelet_decompose(&c1_arr, wv_levels);
    let (c2_ll, c2_subs, c2_sizes) = wavelet::wavelet_decompose(&c2_arr, wv_levels);

    let l_ll_flat: Vec<f64> = l_ll.iter().copied().collect();
    let c1_ll_flat: Vec<f64> = c1_ll.iter().copied().collect();
    let c2_ll_flat: Vec<f64> = c2_ll.iter().copied().collect();

    let (l_ll_norm, l_ll_min, l_ll_max) = crate::normalize_ll(&l_ll_flat);
    let (c1_ll_norm, c1_ll_min, c1_ll_max) = crate::normalize_ll(&c1_ll_flat);
    let (c2_ll_norm, c2_ll_min, c2_ll_max) = crate::normalize_ll(&c2_ll_flat);

    let ll_ranges = [
        (l_ll_min as f32, l_ll_max as f32),
        (c1_ll_min as f32, c1_ll_max as f32),
        (c2_ll_min as f32, c2_ll_max as f32),
    ];

    let detail_step = auto_detail_step(&l_ch, height, width, params.quality);
    let max_passes = if params.quality >= 75 { 3 } else { 1 };
    let weights = wavelet::PERCEPTUAL_BAND_WEIGHTS_V9;

    eprintln!("  AUR2 encode geometric: {}x{}, wv={}, step={:.2}, passes={}", width, height, wv_levels, detail_step, max_passes);

    let mut body: Vec<u8> = Vec::new();

    body.extend_from_slice(&(c1_h as u16).to_le_bytes());
    body.extend_from_slice(&(c1_w as u16).to_le_bytes());
    body.extend_from_slice(&(c2_h as u16).to_le_bytes());
    body.extend_from_slice(&(c2_w as u16).to_le_bytes());

    for lv in 0..wv_levels {
        body.extend_from_slice(&(l_sizes[lv].0 as u16).to_le_bytes());
        body.extend_from_slice(&(l_sizes[lv].1 as u16).to_le_bytes());
        body.extend_from_slice(&(c1_sizes[lv].0 as u16).to_le_bytes());
        body.extend_from_slice(&(c1_sizes[lv].1 as u16).to_le_bytes());
        body.extend_from_slice(&(c2_sizes[lv].0 as u16).to_le_bytes());
        body.extend_from_slice(&(c2_sizes[lv].1 as u16).to_le_bytes());
    }

    body.extend_from_slice(&(l_ll.nrows() as u16).to_le_bytes());
    body.extend_from_slice(&(l_ll.ncols() as u16).to_le_bytes());
    body.extend_from_slice(&(c1_ll.nrows() as u16).to_le_bytes());
    body.extend_from_slice(&(c1_ll.ncols() as u16).to_le_bytes());
    body.extend_from_slice(&(c2_ll.nrows() as u16).to_le_bytes());
    body.extend_from_slice(&(c2_ll.ncols() as u16).to_le_bytes());

    let mut steps_l_all = Vec::with_capacity(wv_levels);
    let mut steps_c1_all = Vec::with_capacity(wv_levels);
    let mut steps_c2_all = Vec::with_capacity(wv_levels);

    for lv in 0..wv_levels {
        let level_scale = if lv < wavelet::LEVEL_SCALES.len() { wavelet::LEVEL_SCALES[lv] } else { 0.3 };
        let mut sl = [0.0f64; 3]; let mut sc1 = [0.0f64; 3]; let mut sc2 = [0.0f64; 3];
        for bi in 0..3 {
            sl[bi] = detail_step * level_scale * weights[bi];
            // Spectral sensitivity: blue (C1) less sensitive → PHI, red (C2) → 1.0
            sc1[bi] = sl[bi] * crate::golden::PHI;  // C1=B-L: blue, low sensitivity
            sc2[bi] = sl[bi] * 1.0;                  // C2=R-L: red, high sensitivity
        }
        steps_l_all.push(sl); steps_c1_all.push(sc1); steps_c2_all.push(sc2);
        for bi in 0..3 {
            body.extend_from_slice(&(sl[bi] as f32).to_le_bytes());
            body.extend_from_slice(&(sc1[bi] as f32).to_le_bytes());
            body.extend_from_slice(&(sc2[bi] as f32).to_le_bytes());
        }
    }
    let ll_step = (detail_step * 0.5).max(0.5);
    body.extend_from_slice(&(ll_step as f32).to_le_bytes());

    // ================================================================
    // 5. Encode LL subbands FIRST (Ribosome: LL serves as DNA template)
    // ================================================================
    struct LLChannelData<'a> { ll_norm: &'a [f64], ll_h: usize, ll_w: usize }

    let ll_channels = [
        LLChannelData { ll_norm: &l_ll_norm, ll_h: l_ll.nrows(), ll_w: l_ll.ncols() },
        LLChannelData { ll_norm: &c1_ll_norm, ll_h: c1_ll.nrows(), ll_w: c1_ll.ncols() },
        LLChannelData { ll_norm: &c2_ll_norm, ll_h: c2_ll.nrows(), ll_w: c2_ll.ncols() },
    ];

    // We need the L-channel quantized residual to simulate decoder LL reconstruction
    let mut l_ll_prediction: Option<Vec<f64>> = None;
    let mut l_ll_q_flat: Option<Vec<i16>> = None;

    for (ch_ll_idx, llch) in ll_channels.iter().enumerate() {
        let (patches, residual) = geometric::fit_ll_patches(llch.ll_norm, llch.ll_h, llch.ll_w, detail_step);
        let patch_bytes = geometric::serialize_poly_patches(&patches);
        body.extend_from_slice(&(patch_bytes.len() as u32).to_le_bytes());
        body.extend_from_slice(&patch_bytes);

        let ll_step_clamped = ll_step.max(0.1);
        let resid_arr = Array2::from_shape_vec((llch.ll_h, llch.ll_w), residual.clone()).unwrap();
        let q = wavelet::quantize_band(&resid_arr, ll_step_clamped);

        let flat: Vec<i16> = q.iter().map(|&v| v as i16).collect();
        let ordered = if llch.ll_h > 0 && llch.ll_w > 0 {
            let order = wavelet::morton_order(llch.ll_h, llch.ll_w);
            order.iter().map(|&idx| flat[idx]).collect::<Vec<i16>>()
        } else { flat.clone() };

        let encoded = rans::rans_encode_band(&ordered);
        body.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        body.extend_from_slice(&(llch.ll_h as u16).to_le_bytes());
        body.extend_from_slice(&(llch.ll_w as u16).to_le_bytes());
        body.extend_from_slice(&encoded);

        // Save L-channel LL data for codon simulation
        if ch_ll_idx == 0 {
            let prediction = geometric::render_ll_patches(&patches, llch.ll_h, llch.ll_w);
            l_ll_prediction = Some(prediction);
            l_ll_q_flat = Some(flat);
        }
    }

    // ================================================================
    // 5b. Simulate L-channel LL decode for Ribosome codon map
    // ================================================================
    let l_ll_h = l_ll.nrows();
    let l_ll_w = l_ll.ncols();
    let l_ll_decoded: Vec<f64> = {
        let prediction = l_ll_prediction.unwrap();
        let q_flat = l_ll_q_flat.unwrap();
        let ll_step_clamped = ll_step.max(0.1);
        let mut ll_norm_decoded = vec![0.0f64; l_ll_h * l_ll_w];
        for i in 0..ll_norm_decoded.len() {
            ll_norm_decoded[i] = prediction[i] + (q_flat[i] as f64 * ll_step_clamped);
        }
        // Denormalize to PTF luminance space
        let range = (ll_ranges[0].1 as f64 - ll_ranges[0].0 as f64).max(1e-6);
        ll_norm_decoded.iter().map(|&v| v * range / 255.0 + ll_ranges[0].0 as f64).collect()
    };

    // ================================================================
    // 6. Detail encode: geometric primitives + codon-adaptive quant
    // ================================================================
    struct ChannelData<'a> {
        subs: &'a [(Array2<f64>, Array2<f64>, Array2<f64>)],
        steps: &'a [[f64; 3]],
    }

    let channels = [
        ChannelData { subs: &l_subs, steps: &steps_l_all },
        ChannelData { subs: &c1_subs, steps: &steps_c1_all },
        ChannelData { subs: &c2_subs, steps: &steps_c2_all },
    ];

    for lv in (0..wv_levels).rev() {
        for (ch_idx, ch) in channels.iter().enumerate() {
            let (ref lh, ref hl, ref hh) = ch.subs[lv];

            // Geometric primitives extraction (supercordes phi)
            let (primitives, resid_lh, resid_hl, resid_hh) =
                geometric::encode_detail_subband(lh, hl, hh, max_passes, params.quality);

            let prim_bytes = geometric::serialize_primitives(&primitives);
            body.extend_from_slice(&prim_bytes);

            let resid_bands: [(&Array2<f64>, usize); 3] = [(&resid_lh, 0), (&resid_hl, 1), (&resid_hh, 2)];

            for (resid_band, bi) in resid_bands {
                let step = ch.steps[lv][bi].max(0.1);
                // Codon-adaptive step for ALL channels (Weber on chroma too)
                let step_map = wavelet::codon_step_map(
                    &l_ll_decoded, l_ll_h, l_ll_w,
                    resid_band.nrows(), resid_band.ncols(),
                    height, width, step,
                );
                let q = wavelet::quantize_band_map(resid_band, &step_map);

                let band_h = q.nrows();
                let band_w = q.ncols();
                let flat: Vec<i16> = q.iter().map(|&v| v as i16).collect();
                let ordered = if band_h > 0 && band_w > 0 {
                    let order = wavelet::morton_order(band_h, band_w);
                    order.iter().map(|&idx| flat[idx]).collect::<Vec<i16>>()
                } else { flat };

                let encoded = rans::rans_encode_band(&ordered);
                body.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
                body.extend_from_slice(&(band_h as u16).to_le_bytes());
                body.extend_from_slice(&(band_w as u16).to_le_bytes());
                body.extend_from_slice(&encoded);
            }
        }
    }

    let header = Aur2Header {
        version: 1, quality: params.quality, width, height, wv_levels, detail_step, ll_ranges,
    };

    let header_bytes = bitstream::write_aur2_header(&header);
    let total_size = header_bytes.len() + body.len();

    let mut aurea_data = Vec::with_capacity(total_size);
    aurea_data.extend_from_slice(&header_bytes);
    aurea_data.extend_from_slice(&body);

    Ok(AureaEncoderResult { compressed_size: aurea_data.len(), aurea_data })
}

// ======================================================================
// v2: LOT (Lapped Orthogonal Transform) encoder
// ======================================================================

const LOT_BLOCK_SIZE: usize = 16;
const LOT_AC_PER_BLOCK: usize = LOT_BLOCK_SIZE * LOT_BLOCK_SIZE - 1; // 255

/// Normalize DC grid values to [0, 255] range.
fn normalize_dc(dc: &[f64]) -> (Vec<f64>, f64, f64) {
    let min = dc.iter().copied().fold(f64::INFINITY, f64::min);
    let max = dc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-6);
    let norm: Vec<f64> = dc.iter().map(|&v| (v - min) * 255.0 / range).collect();
    (norm, min, max)
}

/// Zigzag scan order for a block_size x block_size block (AC coefficients only).
/// Returns indices in row-major order, skipping index 0 (DC).
fn ac_zigzag_order(block_size: usize) -> Vec<usize> {
    let n = block_size;
    let mut order = Vec::with_capacity(n * n - 1);

    for s in 0..(2 * n - 1) {
        if s % 2 == 0 {
            // Up-right
            let i_start = s.min(n - 1);
            let i_end = if s >= n { s - n + 1 } else { 0 };
            let mut i = i_start as i64;
            while i >= i_end as i64 {
                let j = s - i as usize;
                let idx = i as usize * n + j;
                if idx != 0 { // skip DC
                    order.push(idx);
                }
                i -= 1;
            }
        } else {
            // Down-left
            let i_start = if s >= n { s - n + 1 } else { 0 };
            let i_end = s.min(n - 1);
            for i in i_start..=i_end {
                let j = s - i;
                let idx = i * n + j;
                if idx != 0 { // skip DC
                    order.push(idx);
                }
            }
        }
    }

    order
}

/// LOT v3 encoder: all 8 points integrated.
/// GCT + PTF + scene analysis + CSF + structural coherence + chroma residual + rANS.
pub fn encode_aur2_lot(
    rgb: &[u8], width: usize, height: usize, params: &AureaEncoderParams,
) -> Result<AureaEncoderResult, Box<dyn std::error::Error>> {
    let n = width * height;

    // 1. GCT (Golden Color Transform)
    let (mut l_ch, c1_ch, c2_ch) = color::golden_rotate_forward(rgb, n);

    // 2. PTF (Perceptual Transfer Function) on luminance
    {
        use crate::golden::PTF_GAMMA;
        let inv255 = 1.0 / 255.0;
        for v in l_ch.iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA);
        }
    }

    // Point 7: Encoder-side scene analysis
    let scene_profile = scene_analysis::quick_scene_classify(&l_ch, height, width);
    let (step_factor, _deblock_hint) = scene_analysis::encoder_scene_adjust(&scene_profile);
    let scene_type_byte = match scene_profile.scene_type {
        scene_analysis::SceneType::Flat => 0u8,
        scene_analysis::SceneType::Architectural => 1,
        scene_analysis::SceneType::Perspective => 2,
        scene_analysis::SceneType::Organic => 3,
        scene_analysis::SceneType::Mixed => 4,
    };

    // 3. Chroma: full resolution (no subsampling — test mode)
    // 4:2:0/4:2:2 blurs chroma and creates upsampling artifacts on sharp color edges.
    // GCT decorrelation makes chroma sparse → full-res may compress efficiently.
    let (c1_sub, c1_h, c1_w) = (c1_ch.clone(), height, width);
    let (c2_sub, c2_h, c2_w) = (c2_ch.clone(), height, width);

    // 4. LOT analyze each channel (all at full resolution)
    let (l_dc, l_ac, l_gh, l_gw) = lot::lot_analyze_image(&l_ch, height, width, LOT_BLOCK_SIZE);
    let (c1_dc, c1_ac, c1_gh, c1_gw) = lot::lot_analyze_image(&c1_sub, c1_h, c1_w, LOT_BLOCK_SIZE);
    let (c2_dc, c2_ac, c2_gh, c2_gw) = lot::lot_analyze_image(&c2_sub, c2_h, c2_w, LOT_BLOCK_SIZE);

    // 5. Compute detail_step with scene-adaptive factor (Point 7)
    let detail_step = auto_detail_step(&l_ch, height, width, params.quality) * step_factor;
    // DC precision: fine step for smooth gradients. DC is ~1% of file with DPCM.
    // Old: 0.5 * detail_step → 73 levels for 0-255 range → visible banding
    // New: 0.1 * detail_step → 365 levels → smooth gradients
    let dc_step = (detail_step * 0.1).max(0.2);

    // Determine active flags
    let flags = DEFAULT_V3_FLAGS;

    eprintln!("  AUR2 v3 encode: {}x{}, block={}, step={:.2}, flags=0x{:04x}, scene={:?}",
              width, height, LOT_BLOCK_SIZE, detail_step, flags, scene_profile.scene_type);

    // 6. Normalize DC grids
    let (l_dc_norm, l_dc_min, l_dc_max) = normalize_dc(&l_dc);
    let (c1_dc_norm, c1_dc_min, c1_dc_max) = normalize_dc(&c1_dc);
    let (c2_dc_norm, c2_dc_min, c2_dc_max) = normalize_dc(&c2_dc);

    let ll_ranges = [
        (l_dc_min as f32, l_dc_max as f32),
        (c1_dc_min as f32, c1_dc_max as f32),
        (c2_dc_min as f32, c2_dc_max as f32),
    ];

    // 7. Build body
    let mut body: Vec<u8> = Vec::new();

    // Chroma dimensions
    body.extend_from_slice(&(c1_h as u16).to_le_bytes());
    body.extend_from_slice(&(c1_w as u16).to_le_bytes());
    body.extend_from_slice(&(c2_h as u16).to_le_bytes());
    body.extend_from_slice(&(c2_w as u16).to_le_bytes());

    // v3: flags + scene_type (replaces old duplicate detail_step)
    body.extend_from_slice(&flags.to_le_bytes());
    body.push(scene_type_byte);
    body.push(0u8); // reserved

    // AC zigzag order (precompute once)
    let zz_order = ac_zigzag_order(LOT_BLOCK_SIZE);
    assert_eq!(zz_order.len(), LOT_AC_PER_BLOCK);

    // === PASS 1: Quantize DC for all channels, reconstruct from quantized ===
    // This ensures the encoder uses the same DC values the decoder will see,
    // eliminating the DC mismatch that was costing -0.59 dB with CSF enabled.
    let dc_step_clamped = dc_step.max(0.1);
    struct DcChannelData {
        dc_encoded: Vec<u8>,
        dc_reconstructed: Vec<f64>,
    }
    let mut dc_data: Vec<DcChannelData> = Vec::with_capacity(3);

    let dc_norms: [&[f64]; 3] = [&l_dc_norm, &c1_dc_norm, &c2_dc_norm];
    let dc_ranges_raw = [
        (l_dc_min, l_dc_max), (c1_dc_min, c1_dc_max), (c2_dc_min, c2_dc_max),
    ];
    let grid_dims = [(l_gh, l_gw), (c1_gh, c1_gw), (c2_gh, c2_gw)];

    for i in 0..3 {
        let (gh, gw) = grid_dims[i];
        let dc_norm_ch = dc_norms[i];
        let (dc_min_ch, dc_max_ch) = dc_ranges_raw[i];
        let range = (dc_max_ch - dc_min_ch).max(1e-6);

        let dc_q: Vec<i16> = dc_norm_ch.iter().map(|&v| {
            let sign = if v >= 0.0 { 1.0 } else { -1.0 };
            let qv = (v.abs() / dc_step_clamped + 0.5 - wavelet::DEAD_ZONE).floor();
            if qv > 0.0 { (sign * qv) as i16 } else { 0i16 }
        }).collect();

        let dc_for_rans = if flags & FLAG_DPCM_DC != 0 {
            // DPCM: encode residuals in raster order (not Morton)
            let mut residuals = Vec::with_capacity(dc_q.len());
            for gy in 0..gh {
                for gx in 0..gw {
                    let pred = golden_dc_predict(&dc_q, gy, gx, gw);
                    residuals.push(dc_q[gy * gw + gx] - pred);
                }
            }
            residuals
        } else if gh > 0 && gw > 0 {
            // Legacy: Morton order
            let order = wavelet::morton_order(gh, gw);
            order.iter().map(|&idx| dc_q[idx]).collect::<Vec<i16>>()
        } else { dc_q.clone() };
        let dc_encoded = rans::rans_encode_band(&dc_for_rans);

        let dc_reconstructed: Vec<f64> = dc_q.iter().map(|&q| {
            q as f64 * dc_step_clamped * range / 255.0 + dc_min_ch
        }).collect();

        dc_data.push(DcChannelData { dc_encoded, dc_reconstructed });
    }

    // === PASS 2: Use reconstructed DC (matching decoder) for codon/CSF ===
    let l_dc_denorm = &dc_data[0].dc_reconstructed;
    let c1_dc_denorm = &dc_data[1].dc_reconstructed;
    let c2_dc_denorm = &dc_data[2].dc_reconstructed;

    // Encode each channel
    struct ChannelLOT<'a> {
        dc_norm: &'a [f64],
        ac_blocks: &'a [Vec<f64>],
        grid_h: usize,
        grid_w: usize,
        chroma_factor: f64,
    }

    let channels_lot = [
        ChannelLOT {
            dc_norm: &l_dc_norm, ac_blocks: &l_ac,
            grid_h: l_gh, grid_w: l_gw, chroma_factor: 1.0,
        },
        ChannelLOT {
            dc_norm: &c1_dc_norm, ac_blocks: &c1_ac,
            grid_h: c1_gh, grid_w: c1_gw,
            chroma_factor: crate::golden::PHI,
        },
        ChannelLOT {
            dc_norm: &c2_dc_norm, ac_blocks: &c2_ac,
            grid_h: c2_gh, grid_w: c2_gw,
            chroma_factor: 1.0,
        },
    ];

    let use_csf = flags & FLAG_CSF_MODULATION != 0;
    let use_structural = flags & FLAG_STRUCTURAL != 0;

    for (ch_idx, ch) in channels_lot.iter().enumerate() {
        body.extend_from_slice(&(ch.grid_h as u16).to_le_bytes());
        body.extend_from_slice(&(ch.grid_w as u16).to_le_bytes());

        // DC already encoded in pass 1
        body.extend_from_slice(&(dc_data[ch_idx].dc_encoded.len() as u32).to_le_bytes());
        body.extend_from_slice(&dc_data[ch_idx].dc_encoded);

        // AC stream with codon 4D + CSF modulation
        let n_blocks = ch.grid_h * ch.grid_w;
        let total_ac = n_blocks * LOT_AC_PER_BLOCK;
        let mut ac_flat: Vec<i16> = Vec::with_capacity(total_ac);

        // Global factor calibrated for 16x16 LOT (255 AC/block vs JPEG's 63)
        let lot_global_factor = 3.8;
        let ac_step = detail_step * ch.chroma_factor * lot_global_factor;

        for block_idx in 0..n_blocks {
            let ac = &ch.ac_blocks[block_idx];

            // Map block to DC grids (must match decoder's proportional mapping)
            let l_idx = (block_idx as f64 * l_dc_denorm.len() as f64
                        / n_blocks.max(1) as f64) as usize;
            let c1_idx = (block_idx as f64 * c1_dc_denorm.len() as f64
                         / n_blocks.max(1) as f64) as usize;
            let c2_idx = (block_idx as f64 * c2_dc_denorm.len() as f64
                         / n_blocks.max(1) as f64) as usize;
            let dc_l = l_dc_denorm[l_idx.min(l_dc_denorm.len().saturating_sub(1))];
            let dc_c1 = c1_dc_denorm[c1_idx.min(c1_dc_denorm.len().saturating_sub(1))];
            let dc_c2 = c2_dc_denorm[c2_idx.min(c2_dc_denorm.len().saturating_sub(1))];

            // Point 4+8: codon 4D (structural coherence) or 3D
            let codon_factor = if use_structural {
                lot::codon_4d_factor(dc_l, dc_c1, dc_c2, ac, LOT_BLOCK_SIZE)
            } else {
                let ac_energy: f64 = ac.iter().map(|v| v.abs()).sum();
                lot::codon_3d_factor(dc_l, dc_c1, dc_c2, ac_energy, ac.len())
            };
            let local_step = ac_step * codon_factor;
            let step_clamped = local_step.max(0.1);

            // Quantize AC with per-position QMAT + CSF modulation (Point 5)
            for &zz_idx in zz_order.iter() {
                let mut qfactor = lot::qmat_for_block_size(zz_idx / LOT_BLOCK_SIZE, zz_idx % LOT_BLOCK_SIZE, LOT_BLOCK_SIZE).max(0.1).powf(0.55);

                if use_csf {
                    let row = zz_idx / LOT_BLOCK_SIZE;
                    let col = zz_idx % LOT_BLOCK_SIZE;
                    let csf = lot::csf_qmat_factor(row, col, LOT_BLOCK_SIZE, dc_l);
                    qfactor *= csf;
                }

                let pos_step = step_clamped * qfactor;
                let coeff = ac[zz_idx - 1];
                let sign = if coeff >= 0.0 { 1.0 } else { -1.0 };
                let qv = (coeff.abs() / pos_step + 0.5 - wavelet::DEAD_ZONE).floor();
                ac_flat.push(if qv > 0.0 { (sign * qv) as i16 } else { 0i16 });
            }
        }

        let ac_encoded = rans::rans_encode_band(&ac_flat);
        body.extend_from_slice(&(ac_encoded.len() as u32).to_le_bytes());
        body.extend_from_slice(&ac_encoded);

        eprintln!("    ch dc: {} bytes, ac: {} bytes", dc_data[ch_idx].dc_encoded.len(), ac_encoded.len());
    }

    // Point 3: Chroma residual encoding (adaptive, saturation-gated)
    if flags & FLAG_CHROMA_RESIDUAL != 0 {
        // Compute saturation map at full resolution
        let sat_map = color::saturation_map(&c1_ch, &c2_ch, height, width);
        let chroma_resid_step = detail_step * calibration::CHROMA_RESIDUAL_STEP_MULT;

        // Encode C1 residual (4:2:0 loses most, blue channel)
        let (c1_mask, c1_resid) = color::encode_chroma_residual(
            &c1_ch, &c1_sub, &sat_map, height, width, c1_h, c1_w, chroma_resid_step,
        );
        body.extend_from_slice(&(c1_mask.len() as u32).to_le_bytes());
        body.extend_from_slice(&c1_mask);
        body.extend_from_slice(&(c1_resid.len() as u32).to_le_bytes());
        for &v in &c1_resid { body.push(v as u8); }

        eprintln!("    chroma residual: mask={} bytes, resid={} values ({} active blocks)",
                  c1_mask.len(), c1_resid.len(),
                  c1_mask.iter().map(|b| b.count_ones()).sum::<u32>());
    }

    // 8. Header: version=3, wv_levels=0 (LOT)
    let header = Aur2Header {
        version: 3,
        quality: params.quality,
        width,
        height,
        wv_levels: 0,
        detail_step,
        ll_ranges,
    };

    let header_bytes = bitstream::write_aur2_header(&header);
    let mut aurea_data = Vec::with_capacity(header_bytes.len() + body.len());
    aurea_data.extend_from_slice(&header_bytes);
    aurea_data.extend_from_slice(&body);

    Ok(AureaEncoderResult { compressed_size: aurea_data.len(), aurea_data })
}

// ======================================================================
// Point 6: Rate control (binary search on quality)
// ======================================================================

/// Encode with target bits-per-pixel (rate control, Point 6).
/// Binary-searches on quality parameter to converge within tolerance.
pub fn encode_aur2_rate_controlled(
    rgb: &[u8], width: usize, height: usize, target_bpp: f64,
) -> Result<AureaEncoderResult, Box<dyn std::error::Error>> {
    let n_pixels = width * height;
    let _target_bytes = (target_bpp * n_pixels as f64 / 8.0) as usize;

    let mut q_lo: u8 = 10;
    let mut q_hi: u8 = 98;
    let mut best_result: Option<AureaEncoderResult> = None;

    for _iter in 0..calibration::RATE_CONTROL_MAX_ITER {
        let q_mid = (q_lo as u16 + q_hi as u16) / 2;
        let q = q_mid as u8;

        let params = AureaEncoderParams {
            quality: q,
            n_representatives: quality_to_n_repr(q),
            geometric: false,
        };

        let result = encode_aur2(rgb, width, height, &params)?;
        let actual_bpp = result.compressed_size as f64 * 8.0 / n_pixels as f64;
        let ratio = actual_bpp / target_bpp;

        eprintln!("  Rate control iter {}: q={}, bpp={:.3} (target={:.3}, ratio={:.2})",
                  _iter, q, actual_bpp, target_bpp, ratio);

        if (ratio - 1.0).abs() < calibration::RATE_CONTROL_TOLERANCE {
            return Ok(result);
        }

        if actual_bpp > target_bpp {
            q_hi = q; // too many bits, reduce quality
        } else {
            q_lo = q; // too few bits, increase quality
        }
        best_result = Some(result);

        if q_hi - q_lo <= 1 { break; }
    }

    best_result.ok_or_else(|| "Rate control failed to converge".into())
}