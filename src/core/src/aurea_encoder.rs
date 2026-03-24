/// Native AUREA v10 encoder: RGB -> .aur file (AUR2 format).
/// Modified for the A.D.N. architecture (spatial codons).
///
/// v2 (LOT): Block-based Lapped Orthogonal Transform replaces CDF 9/7 wavelet.
/// v12: Turing morphogenesis + Psychovisual Pivot (ADN8)

use ndarray::Array2;

use crate::bitstream::{self, Aur2Header, TuringHeader};
use crate::calibration;
use crate::cfl;
use crate::color;
use crate::geometric::{self, MatchKind, PrimitiveMatch};
use crate::hierarchy::{self, HierarchyParams};
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
pub const FLAG_QUALITY_ADAPTIVE: u16 = 0x0040; // ADN4: quality-adaptive lot_factor, qmat_power, spin
pub const FLAG_LOT_OVERLAP: u16      = 0x0080; // ADN4: 50% LOT overlap at high quality
pub const FLAG_WEBER_TRNA: u16       = 0x0100; // ADN5: Weber-Fechner luminance tRNA on AC step
pub const FLAG_BAYESIAN_HIERARCHY: u16 = 0x0200; // v12: Bayesian predictive hierarchy
pub const FLAG_CFL: u16              = 0x0400; // v12: Chroma-from-Luma prediction

/// RLE escape token: signals a run of zeros. Format: [TOKEN_RUN, count].
pub const TOKEN_RUN: i16 = 32000;

pub const DEFAULT_V3_FLAGS: u16 = FLAG_CSF_MODULATION | FLAG_SCENE_ANALYSIS | FLAG_DPCM_DC | FLAG_QUALITY_ADAPTIVE | FLAG_WEBER_TRNA;

/// Golden DPCM prediction for DC grid in raster order.
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

// ======================================================================
// ADN8: Psychovisual Turing Pivot (high-rate vs low-rate)
// ======================================================================
/// Computes the Turing modulation factor with a psychovisual pivot.
/// At low bitrate (quality < 50): gamma > 0, preserves structure (sharp edges), sacrifices smooth areas.
/// At high bitrate (quality > 60): gamma < 0, spatial masking (sacrifices complex textures), protects smooth areas (anti-banding).
pub fn psychovisual_turing_pivot(raw_turing_mod: f64, quality: u8) -> f64 {
    // 1. Calibration parameters
    let gamma_low = 1.25;   // Edge preservation strength at low bitrate
    let gamma_high = -0.85; // Anti-banding protection strength at high bitrate

    // 2. Normalize quality to [0.0, 1.0]
    let q_norm = (quality as f64 / 100.0).clamp(0.0, 1.0);

    // 3. Cubic smoothstep (S-curve, flat tangents at endpoints)
    // Q=0 -> 0.0 | Q=50 -> 0.5 | Q=100 -> 1.0
    let s_curve = q_norm * q_norm * (3.0 - 2.0 * q_norm);

    // 4. Interpolate gamma via smoothstep
    let gamma = gamma_low * (1.0 - s_curve) + gamma_high * s_curve;

    // 5. Safety clamp for exponential stability
    let safe_mod = raw_turing_mod.clamp(0.1, 10.0);

    safe_mod.powf(gamma)
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
    // Exponential mapping calibrated on 6 HD images (adn5 benchmark).
    // Correction factor 1.3x applied to close the gap with JPEG.
    // Q=0 → ~40, Q=50 → ~7.2, Q=75 → ~3.3, Q=90 → ~2.0, Q=100 → ~1.0
    let base_step = 39.0 * (-3.5 * q_frac).exp() + 0.9;
    let factor = 1.0 - 0.4 * c_norm;

    (base_step * factor).clamp(0.1, 48.0)
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

    let (l_ch, _, _) = color::golden_rotate_forward(rgb, n);
    let scene_profile = scene_analysis::quick_scene_classify(&l_ch, height, width);

    let use_lot = scene_analysis::recommend_lot(&scene_profile);
    eprintln!("  Scene: {:?}, smooth={:.0}%, transform={}",
              scene_profile.scene_type, scene_profile.smooth_pct,
              if use_lot { "LOT" } else { "CDF97" });

    if use_lot {
        encode_aur2_lot(rgb, width, height, params)
    } else {
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
            sc1[bi] = sl[bi] * crate::golden::PHI;
            sc2[bi] = sl[bi] * 1.0;
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

    struct LLChannelData<'a> { ll_norm: &'a [f64], ll_h: usize, ll_w: usize }

    let ll_channels = [
        LLChannelData { ll_norm: &l_ll_norm, ll_h: l_ll.nrows(), ll_w: l_ll.ncols() },
        LLChannelData { ll_norm: &c1_ll_norm, ll_h: c1_ll.nrows(), ll_w: c1_ll.ncols() },
        LLChannelData { ll_norm: &c2_ll_norm, ll_h: c2_ll.nrows(), ll_w: c2_ll.ncols() },
    ];

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

        if ch_ll_idx == 0 {
            let prediction = geometric::render_ll_patches(&patches, llch.ll_h, llch.ll_w);
            l_ll_prediction = Some(prediction);
            l_ll_q_flat = Some(flat);
        }
    }

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
        let range = (ll_ranges[0].1 as f64 - ll_ranges[0].0 as f64).max(1e-6);
        ll_norm_decoded.iter().map(|&v| v * range / 255.0 + ll_ranges[0].0 as f64).collect()
    };

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

            let (primitives, resid_lh, resid_hl, resid_hh) =
                geometric::encode_detail_subband(lh, hl, hh, max_passes, params.quality);

            let prim_bytes = geometric::serialize_primitives(&primitives);
            body.extend_from_slice(&prim_bytes);

            let resid_bands: [(&Array2<f64>, usize); 3] = [(&resid_lh, 0), (&resid_hl, 1), (&resid_hh, 2)];

            for (resid_band, bi) in resid_bands {
                let step = ch.steps[lv][bi].max(0.1);
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
const LOT_AC_PER_BLOCK: usize = LOT_BLOCK_SIZE * LOT_BLOCK_SIZE - 1;

fn normalize_dc(dc: &[f64]) -> (Vec<f64>, f64, f64) {
    let min = dc.iter().copied().fold(f64::INFINITY, f64::min);
    let max = dc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-6);
    let norm: Vec<f64> = dc.iter().map(|&v| (v - min) * 255.0 / range).collect();
    (norm, min, max)
}

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

pub fn encode_aur2_lot(
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

    let scene_profile = scene_analysis::quick_scene_classify(&l_ch, height, width);
    let (step_factor, _deblock_hint) = scene_analysis::encoder_scene_adjust(&scene_profile);
    let scene_type_byte = match scene_profile.scene_type {
        scene_analysis::SceneType::Flat => 0u8,
        scene_analysis::SceneType::Architectural => 1,
        scene_analysis::SceneType::Perspective => 2,
        scene_analysis::SceneType::Organic => 3,
        scene_analysis::SceneType::Mixed => 4,
    };

    let (c1_sub, c1_h, c1_w) = (c1_ch.clone(), height, width);
    let (c2_sub, c2_h, c2_w) = (c2_ch.clone(), height, width);

    let mut flags = DEFAULT_V3_FLAGS;
    flags |= FLAG_VARIABLE_BLOCKS;
    let use_overlap = flags & FLAG_LOT_OVERLAP != 0;
    let use_variable = flags & FLAG_VARIABLE_BLOCKS != 0;

    let (l_dc, l_ac, l_gh, l_gw, l_block_map, l_blocks);
    let (c1_dc, c1_ac, c1_gh, c1_gw);
    let (c2_dc, c2_ac, c2_gh, c2_gw);

    if use_variable {
        let detail_step_pre = auto_detail_step(&l_ch, height, width, params.quality) * step_factor;
        let (size_grid, bgh, bgw) = lot::classify_blocks(&l_ch, height, width, detail_step_pre);

        let n8 = size_grid.iter().filter(|&&s| s == 8).count();
        let n16 = size_grid.iter().filter(|&&s| s == 16).count();
        let n32 = size_grid.iter().filter(|&&s| s == 32).count();
        let total = size_grid.len();
        eprintln!("  Adaptive mesh: 8×8={:.0}% 16×16={:.0}% 32×32={:.0}%",
                  n8 as f64 / total as f64 * 100.0,
                  n16 as f64 / total as f64 * 100.0,
                  n32 as f64 / total as f64 * 100.0);

        l_block_map = Some((size_grid.clone(), bgh, bgw));

        let (dc, ac, blocks) = lot::lot_analyze_variable(
            &l_ch, height, width, &size_grid, bgh, bgw, use_overlap);
        l_blocks = Some(blocks.clone());
        l_dc = dc; l_ac = ac;
        l_gh = blocks.len();
        l_gw = 1;

        let (dc1, ac1, _) = lot::lot_analyze_variable(
            &c1_sub, c1_h, c1_w, &size_grid, bgh, bgw, use_overlap);
        c1_dc = dc1; c1_ac = ac1; c1_gh = l_gh; c1_gw = 1;

        let (dc2, ac2, _) = lot::lot_analyze_variable(
            &c2_sub, c2_h, c2_w, &size_grid, bgh, bgw, use_overlap);
        c2_dc = dc2; c2_ac = ac2; c2_gh = l_gh; c2_gw = 1;
    } else {
        l_block_map = None;
        l_blocks = None;
        let r = lot::lot_analyze_image(&l_ch, height, width, LOT_BLOCK_SIZE, use_overlap);
        l_dc = r.0; l_ac = r.1; l_gh = r.2; l_gw = r.3;
        let r1 = lot::lot_analyze_image(&c1_sub, c1_h, c1_w, LOT_BLOCK_SIZE, use_overlap);
        c1_dc = r1.0; c1_ac = r1.1; c1_gh = r1.2; c1_gw = r1.3;
        let r2 = lot::lot_analyze_image(&c2_sub, c2_h, c2_w, LOT_BLOCK_SIZE, use_overlap);
        c2_dc = r2.0; c2_ac = r2.1; c2_gh = r2.2; c2_gw = r2.3;
    }

    let detail_step = auto_detail_step(&l_ch, height, width, params.quality) * step_factor;
    let dc_step = (detail_step * 0.1).max(0.2);

    let lot_global_factor = if flags & FLAG_QUALITY_ADAPTIVE != 0 {
        calibration::lot_factor_for_quality(params.quality)
    } else {
        3.8
    };
    let dead_zone = if flags & FLAG_QUALITY_ADAPTIVE != 0 {
        calibration::dead_zone_for_quality(params.quality)
    } else {
        wavelet::DEAD_ZONE
    };
    let qmat_power = if flags & FLAG_QUALITY_ADAPTIVE != 0 {
        calibration::qmat_power_for_quality(params.quality)
    } else {
        0.55
    };

    eprintln!("  AUR2 v3 encode: {}x{}, block={}, step={:.2}, lot_f={:.2}, dz={:.3}, qp={:.2}, flags=0x{:04x}",
              width, height, LOT_BLOCK_SIZE, detail_step,
              lot_global_factor, dead_zone, qmat_power, flags);

    let (l_dc_norm, l_dc_min, l_dc_max) = normalize_dc(&l_dc);
    let (c1_dc_norm, c1_dc_min, c1_dc_max) = normalize_dc(&c1_dc);
    let (c2_dc_norm, c2_dc_min, c2_dc_max) = normalize_dc(&c2_dc);

    let ll_ranges = [
        (l_dc_min as f32, l_dc_max as f32),
        (c1_dc_min as f32, c1_dc_max as f32),
        (c2_dc_min as f32, c2_dc_max as f32),
    ];

    let mut body: Vec<u8> = Vec::new();

    body.extend_from_slice(&(c1_h as u16).to_le_bytes());
    body.extend_from_slice(&(c1_w as u16).to_le_bytes());
    body.extend_from_slice(&(c2_h as u16).to_le_bytes());
    body.extend_from_slice(&(c2_w as u16).to_le_bytes());

    body.extend_from_slice(&flags.to_le_bytes());
    body.push(scene_type_byte);
    body.push(0u8);

    if let Some((ref size_grid, bgh, bgw)) = l_block_map {
        body.extend_from_slice(&(bgh as u16).to_le_bytes());
        body.extend_from_slice(&(bgw as u16).to_le_bytes());
        let encoded_map = lot::encode_block_map(size_grid);
        body.extend_from_slice(&(encoded_map.len() as u16).to_le_bytes());
        body.extend_from_slice(&encoded_map);
    }

    let zz_16 = ac_zigzag_order(16);
    let zz_8 = ac_zigzag_order(8);
    let zz_32 = ac_zigzag_order(32);

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
            let qv = (v.abs() / dc_step_clamped + 0.5 - dead_zone).floor();
            if qv > 0.0 { (sign * qv) as i16 } else { 0i16 }
        }).collect();

        let dc_for_rans = if flags & FLAG_DPCM_DC != 0 {
            let mut residuals = Vec::with_capacity(dc_q.len());
            for gy in 0..gh {
                for gx in 0..gw {
                    let pred = golden_dc_predict(&dc_q, gy, gx, gw);
                    residuals.push(dc_q[gy * gw + gx] - pred);
                }
            }
            residuals
        } else if gh > 0 && gw > 0 {
            let order = wavelet::morton_order(gh, gw);
            order.iter().map(|&idx| dc_q[idx]).collect::<Vec<i16>>()
        } else { dc_q.clone() };
        let dc_encoded = rans::rans_encode_band(&dc_for_rans);

        let dc_reconstructed: Vec<f64> = dc_q.iter().map(|&q| {
            q as f64 * dc_step_clamped * range / 255.0 + dc_min_ch
        }).collect();

        dc_data.push(DcChannelData { dc_encoded, dc_reconstructed });
    }

    let l_dc_denorm = &dc_data[0].dc_reconstructed;

    let bs2 = (LOT_BLOCK_SIZE * LOT_BLOCK_SIZE) as f64;
    let l_dc_pixel: Vec<f64> = l_dc_denorm.iter().map(|&v| v / bs2).collect();

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

    let l_foveal_map = lot::foveal_saliency_map(
        &dc_data[0].dc_reconstructed, channels_lot[0].grid_h, channels_lot[0].grid_w,
    );

    for (ch_idx, ch) in channels_lot.iter().enumerate() {
        body.extend_from_slice(&(ch.grid_h as u16).to_le_bytes());
        body.extend_from_slice(&(ch.grid_w as u16).to_le_bytes());

        body.extend_from_slice(&(dc_data[ch_idx].dc_encoded.len() as u32).to_le_bytes());
        body.extend_from_slice(&dc_data[ch_idx].dc_encoded);

        let n_blocks = ch.grid_h * (if use_variable { 1 } else { ch.grid_w });
        let mut ac_flat: Vec<i16> = Vec::new();

        let ac_step = detail_step * ch.chroma_factor * lot_global_factor;

        for block_idx in 0..n_blocks {
            let ac = &ch.ac_blocks[block_idx];

            let block_size = if use_variable {
                if let Some(ref blocks) = l_blocks {
                    if block_idx < blocks.len() { blocks[block_idx].2 } else { LOT_BLOCK_SIZE }
                } else { LOT_BLOCK_SIZE }
            } else { LOT_BLOCK_SIZE };

            let zz_order = match block_size {
                8 => &zz_8,
                32 => &zz_32,
                _ => &zz_16,
            };

            let l_fov_idx = (block_idx as f64 * l_foveal_map.len() as f64
                            / n_blocks.max(1) as f64) as usize;
            let foveal_factor = l_foveal_map[l_fov_idx.min(l_foveal_map.len().saturating_sub(1))];

            let l_idx = (block_idx as f64 * l_dc_pixel.len() as f64
                        / n_blocks.max(1) as f64) as usize;
            let dc_l = l_dc_pixel[l_idx.min(l_dc_pixel.len().saturating_sub(1))];

            let local_step = ac_step * foveal_factor;
            let step_clamped = local_step.max(0.1);

            let ac_per_block_v3 = block_size * block_size - 1;
            for (zz_pos, &zz_idx) in zz_order.iter().enumerate() {
                let row = zz_idx / block_size;
                let col = zz_idx % block_size;
                let mut qfactor = lot::qmat_for_block_size(row, col, block_size).max(0.1).powf(qmat_power);

                if use_csf {
                    let csf = lot::csf_qmat_factor(row, col, block_size, dc_l);
                    qfactor *= csf;
                }

                let pos_step = step_clamped * qfactor;
                let coeff = if (zz_idx - 1) < ac.len() { ac[zz_idx - 1] } else { 0.0 };
                let sign = if coeff >= 0.0 { 1.0 } else { -1.0 };
                let pos_dz = calibration::dead_zone_for_position(zz_pos, ac_per_block_v3, dead_zone);
                let qv = (coeff.abs() / pos_step + 0.5 - pos_dz).floor();
                ac_flat.push(if qv > 0.0 { (sign * qv) as i16 } else { 0i16 });
            }
        }

        let ac_encoded = rans::rans_encode_band(&ac_flat);
        body.extend_from_slice(&(ac_encoded.len() as u32).to_le_bytes());
        body.extend_from_slice(&ac_encoded);
    }

    if flags & FLAG_CHROMA_RESIDUAL != 0 {
        let sat_map = color::saturation_map(&c1_ch, &c2_ch, height, width);
        let chroma_resid_step = detail_step * calibration::CHROMA_RESIDUAL_STEP_MULT;

        let (c1_mask, c1_resid) = color::encode_chroma_residual(
            &c1_ch, &c1_sub, &sat_map, height, width, c1_h, c1_w, chroma_resid_step,
        );
        body.extend_from_slice(&(c1_mask.len() as u32).to_le_bytes());
        body.extend_from_slice(&c1_mask);
        body.extend_from_slice(&(c1_resid.len() as u32).to_le_bytes());
        for &v in &c1_resid { body.push(v as u8); }
    }

    let version = if use_overlap { 4 } else { 3 };
    let header = Aur2Header {
        version,
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
// v12: Bayesian predictive hierarchy encoder (LOT + Turing morphogenesis)
// ======================================================================

fn serialize_primitive_matches(matches: &[PrimitiveMatch]) -> Vec<u8> {
    let n = matches.len();
    let mut buf: Vec<u8> = Vec::new();

    buf.extend_from_slice(&(n as u16).to_le_bytes());

    if n == 0 {
        return buf;
    }

    let kind_bytes = (n + 3) / 4;
    let mut kinds = vec![0u8; kind_bytes];
    for (i, m) in matches.iter().enumerate() {
        let kind_val: u8 = match m.kind {
            MatchKind::Predicted => 0b00,
            MatchKind::Residual  => 0b01,
            MatchKind::Surprise  => 0b10,
        };
        let byte_idx = i / 4;
        let bit_shift = 6 - (i % 4) * 2;
        kinds[byte_idx] |= kind_val << bit_shift;
    }
    buf.extend_from_slice(&kinds);

    for m in matches {
        match m.kind {
            MatchKind::Predicted => {}
            MatchKind::Residual => {
                let cidx = m.contour_idx.unwrap_or(0) as u16;
                buf.extend_from_slice(&cidx.to_le_bytes());
                let (dy, dx) = m.delta_pos.unwrap_or((0, 0));
                buf.extend_from_slice(&dy.to_le_bytes());
                buf.extend_from_slice(&dx.to_le_bytes());
                buf.push(m.delta_angle.unwrap_or(0) as u8);
                buf.push(m.delta_amp.unwrap_or(0) as u8);
            }
            MatchKind::Surprise => {
                match &m.original {
                    geometric::Primitive::Segment { x1, y1, x2, y2, amplitude, phase } => {
                        buf.push(0x00);
                        buf.extend_from_slice(&x1.to_le_bytes());
                        buf.extend_from_slice(&y1.to_le_bytes());
                        buf.extend_from_slice(&x2.to_le_bytes());
                        buf.extend_from_slice(&y2.to_le_bytes());
                        buf.push(*amplitude as u8);
                        buf.push(*phase as u8);
                    }
                    geometric::Primitive::Arc { cx, cy, radius, theta_start, theta_end, amplitude, phase } => {
                        buf.push(0x01);
                        buf.extend_from_slice(&cx.to_le_bytes());
                        buf.extend_from_slice(&cy.to_le_bytes());
                        buf.extend_from_slice(&radius.to_le_bytes());
                        buf.push(*theta_start as u8);
                        buf.push(*theta_end as u8);
                        buf.push(*amplitude as u8);
                        buf.push(*phase as u8);
                    }
                }
            }
        }
    }

    buf
}

pub fn encode_aur2_v12(
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

    let scene_profile = scene_analysis::quick_scene_classify(&l_ch, height, width);
    let (step_factor, _deblock_hint) = scene_analysis::encoder_scene_adjust(&scene_profile);
    let scene_type_byte = match scene_profile.scene_type {
        scene_analysis::SceneType::Flat => 0u8,
        scene_analysis::SceneType::Architectural => 1,
        scene_analysis::SceneType::Perspective => 2,
        scene_analysis::SceneType::Organic => 3,
        scene_analysis::SceneType::Mixed => 4,
    };

    let (c1_sub, c1_h, c1_w) = (c1_ch.clone(), height, width);
    let (c2_sub, c2_h, c2_w) = (c2_ch.clone(), height, width);

    let mut flags = DEFAULT_V3_FLAGS;
    flags |= FLAG_VARIABLE_BLOCKS;
    flags |= FLAG_BAYESIAN_HIERARCHY;
    flags |= FLAG_CFL;
    let use_overlap = flags & FLAG_LOT_OVERLAP != 0;
    let use_variable = flags & FLAG_VARIABLE_BLOCKS != 0;
    let use_cfl = flags & FLAG_CFL != 0;

    let (l_dc, l_ac, l_gh, l_gw, l_block_map, l_blocks);
    let (c1_dc, c1_ac, c1_gh, c1_gw);
    let (c2_dc, c2_ac, c2_gh, c2_gw);

    if use_variable {
        let detail_step_pre = auto_detail_step(&l_ch, height, width, params.quality) * step_factor;
        let (size_grid, bgh, bgw) = lot::classify_blocks(&l_ch, height, width, detail_step_pre);

        l_block_map = Some((size_grid.clone(), bgh, bgw));

        let (dc, ac, blocks) = lot::lot_analyze_variable(
            &l_ch, height, width, &size_grid, bgh, bgw, use_overlap);
        l_blocks = Some(blocks.clone());
        l_dc = dc; l_ac = ac;
        l_gh = blocks.len();
        l_gw = 1;

        let (dc1, ac1, _) = lot::lot_analyze_variable(
            &c1_sub, c1_h, c1_w, &size_grid, bgh, bgw, use_overlap);
        c1_dc = dc1; c1_ac = ac1; c1_gh = l_gh; c1_gw = 1;

        let (dc2, ac2, _) = lot::lot_analyze_variable(
            &c2_sub, c2_h, c2_w, &size_grid, bgh, bgw, use_overlap);
        c2_dc = dc2; c2_ac = ac2; c2_gh = l_gh; c2_gw = 1;
    } else {
        l_block_map = None;
        l_blocks = None;
        let r = lot::lot_analyze_image(&l_ch, height, width, LOT_BLOCK_SIZE, use_overlap);
        l_dc = r.0; l_ac = r.1; l_gh = r.2; l_gw = r.3;
        let r1 = lot::lot_analyze_image(&c1_sub, c1_h, c1_w, LOT_BLOCK_SIZE, use_overlap);
        c1_dc = r1.0; c1_ac = r1.1; c1_gh = r1.2; c1_gw = r1.3;
        let r2 = lot::lot_analyze_image(&c2_sub, c2_h, c2_w, LOT_BLOCK_SIZE, use_overlap);
        c2_dc = r2.0; c2_ac = r2.1; c2_gh = r2.2; c2_gw = r2.3;
    }

    let detail_step = auto_detail_step(&l_ch, height, width, params.quality) * step_factor;
    let dc_step = (detail_step * 0.1).max(0.2);

    let lot_global_factor = if flags & FLAG_QUALITY_ADAPTIVE != 0 {
        calibration::lot_factor_for_quality(params.quality)
    } else {
        3.8
    };
    let dead_zone = if flags & FLAG_QUALITY_ADAPTIVE != 0 {
        calibration::dead_zone_for_quality(params.quality)
    } else {
        wavelet::DEAD_ZONE
    };
    let qmat_power = if flags & FLAG_QUALITY_ADAPTIVE != 0 {
        calibration::qmat_power_for_quality(params.quality)
    } else {
        0.55
    };

    let (l_dc_norm, l_dc_min, l_dc_max) = normalize_dc(&l_dc);
    let (c1_dc_norm, c1_dc_min, c1_dc_max) = normalize_dc(&c1_dc);
    let (c2_dc_norm, c2_dc_min, c2_dc_max) = normalize_dc(&c2_dc);

    let ll_ranges = [
        (l_dc_min as f32, l_dc_max as f32),
        (c1_dc_min as f32, c1_dc_max as f32),
        (c2_dc_min as f32, c2_dc_max as f32),
    ];

    let mut body: Vec<u8> = Vec::new();

    body.extend_from_slice(&(c1_h as u16).to_le_bytes());
    body.extend_from_slice(&(c1_w as u16).to_le_bytes());
    body.extend_from_slice(&(c2_h as u16).to_le_bytes());
    body.extend_from_slice(&(c2_w as u16).to_le_bytes());

    body.extend_from_slice(&flags.to_le_bytes());
    body.push(scene_type_byte);
    body.push(0u8);

    let turing_hdr = TuringHeader::default_params();
    let turing_hdr_bytes = turing_hdr.write();
    body.extend_from_slice(&turing_hdr_bytes);

    if let Some((ref size_grid, bgh, bgw)) = l_block_map {
        body.extend_from_slice(&(bgh as u16).to_le_bytes());
        body.extend_from_slice(&(bgw as u16).to_le_bytes());
        // rANS compress the block map (codes 0/1/2 — highly compressible when >90% are 16)
        let map_symbols: Vec<i16> = size_grid.iter().map(|&s| match s {
            8 => 0i16, 16 => 1, 32 => 2, _ => 1,
        }).collect();
        let map_ctx = vec![0u8; map_symbols.len()];
        let encoded_map = rans::rans_encode_band_v12(&map_symbols, &map_ctx);
        body.extend_from_slice(&(encoded_map.len() as u16).to_le_bytes());
        body.extend_from_slice(&encoded_map);
    }

    let zz_16 = ac_zigzag_order(16);
    let zz_8 = ac_zigzag_order(8);
    let zz_32 = ac_zigzag_order(32);

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
            let qv = (v.abs() / dc_step_clamped + 0.5 - dead_zone).floor();
            if qv > 0.0 { (sign * qv) as i16 } else { 0i16 }
        }).collect();

        let dc_for_rans = if flags & FLAG_DPCM_DC != 0 {
            let mut residuals = Vec::with_capacity(dc_q.len());
            for gy in 0..gh {
                for gx in 0..gw {
                    let pred = golden_dc_predict(&dc_q, gy, gx, gw);
                    residuals.push(dc_q[gy * gw + gx] - pred);
                }
            }
            residuals
        } else if gh > 0 && gw > 0 {
            let order = wavelet::morton_order(gh, gw);
            order.iter().map(|&idx| dc_q[idx]).collect::<Vec<i16>>()
        } else { dc_q.clone() };
        // v12: rANS v12 (Exp-Golomb) for DC — more efficient than v1 unary
        let dc_ctx = vec![0u8; dc_for_rans.len()];
        let dc_encoded = rans::rans_encode_band_v12(&dc_for_rans, &dc_ctx);

        let dc_reconstructed: Vec<f64> = dc_q.iter().map(|&q| {
            q as f64 * dc_step_clamped * range / 255.0 + dc_min_ch
        }).collect();

        dc_data.push(DcChannelData { dc_encoded, dc_reconstructed });
    }

    let l_dc_denorm = &dc_data[0].dc_reconstructed;
    let hierarchy_params = HierarchyParams::from(&turing_hdr);
    let turing_field = hierarchy::compute_level1(l_dc_denorm, l_gh, l_gw, &hierarchy_params);

    let turing_buckets = hierarchy::build_turing_buckets(
        &turing_field, l_gh, l_gw, LOT_BLOCK_SIZE,
    );

    let bs2 = (LOT_BLOCK_SIZE * LOT_BLOCK_SIZE) as f64;
    let l_dc_pixel: Vec<f64> = l_dc_denorm.iter().map(|&v| v / bs2).collect();

    #[allow(dead_code)]
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

    let l_foveal_map = lot::foveal_saliency_map(
        &dc_data[0].dc_reconstructed, channels_lot[0].grid_h, channels_lot[0].grid_w,
    );

    let primitive_matches: Vec<PrimitiveMatch> = Vec::new();
    let match_bytes = serialize_primitive_matches(&primitive_matches);

    // ================================================================
    // CfL two-pass encoding: luma first, then chroma with prediction
    // ================================================================

    // Helper closure: encode one channel's AC into the body, returning
    // the per-block dequantized AC vectors (in natural order, indexed
    // by zz_idx-1) if `collect_l_rec` is true.
    // `cfl_residual_blocks` — if Some, the chroma AC has already been
    // replaced by the CfL residual for active blocks (pre-quantization).
    let encode_channel_ac = |
        ch: &ChannelLOT,
        ch_idx: usize,
        body: &mut Vec<u8>,
        collect_l_rec: bool,
    | -> Vec<Vec<f64>> {
        let n_blocks = ch.grid_h * (if use_variable { 1 } else { ch.grid_w });
        let mut ac_flat: Vec<i16> = Vec::new();
        let mut ac_turing: Vec<u8> = Vec::new();
        let mut eob_positions: Vec<u16> = Vec::with_capacity(n_blocks);

        let ac_step = detail_step * ch.chroma_factor * lot_global_factor;

        let ch_block_buckets = if ch_idx == 0 {
            turing_buckets.clone()
        } else {
            hierarchy::build_turing_buckets(
                &turing_field, ch.grid_h, ch.grid_w, LOT_BLOCK_SIZE,
            )
        };

        // Optionally collect dequantized AC per block (for luma reconstruction)
        let mut l_rec_blocks: Vec<Vec<f64>> = if collect_l_rec {
            Vec::with_capacity(n_blocks)
        } else {
            Vec::new()
        };

        for block_idx in 0..n_blocks {
            let ac = &ch.ac_blocks[block_idx];
            let block_size = if use_variable {
                if let Some(ref blocks) = l_blocks {
                    if block_idx < blocks.len() { blocks[block_idx].2 } else { LOT_BLOCK_SIZE }
                } else { LOT_BLOCK_SIZE }
            } else { LOT_BLOCK_SIZE };
            let zz_order = match block_size { 8 => &zz_8, 32 => &zz_32, _ => &zz_16 };
            let l_fov_idx = (block_idx as f64 * l_foveal_map.len() as f64 / n_blocks.max(1) as f64) as usize;
            let foveal_factor = l_foveal_map[l_fov_idx.min(l_foveal_map.len().saturating_sub(1))];
            let l_idx = (block_idx as f64 * l_dc_pixel.len() as f64 / n_blocks.max(1) as f64) as usize;
            let dc_l = l_dc_pixel[l_idx.min(l_dc_pixel.len().saturating_sub(1))];
            let raw_turing_mod = if block_idx < turing_field.step_modulation.len() { turing_field.step_modulation[block_idx] } else { 1.0 };
            let turing_mod = psychovisual_turing_pivot(raw_turing_mod, params.quality);
            let local_step = ac_step * foveal_factor * turing_mod;
            let step_clamped = local_step.max(0.1);

            let block_start = ac_flat.len();
            let ac_per_block = block_size * block_size - 1;

            // If collecting L_rec, prepare a buffer in natural (zz_idx-1) order
            let mut l_rec_block = if collect_l_rec {
                vec![0.0f64; ac_per_block]
            } else {
                Vec::new()
            };

            for (zz_pos, &zz_idx) in zz_order.iter().enumerate() {
                let row = zz_idx / block_size;
                let col = zz_idx % block_size;
                let mut qfactor = lot::qmat_for_block_size(row, col, block_size).max(0.1).powf(qmat_power);
                if use_csf {
                    let csf = lot::csf_qmat_factor(row, col, block_size, dc_l);
                    qfactor *= csf;
                }
                let pos_step = step_clamped * qfactor;
                let coeff = if (zz_idx - 1) < ac.len() { ac[zz_idx - 1] } else { 0.0 };
                let local_dz = calibration::dead_zone_for_position(zz_pos, ac_per_block, dead_zone);
                let sign = if coeff >= 0.0 { 1.0 } else { -1.0 };
                let qv = (coeff.abs() / pos_step + 0.5 - local_dz).floor();
                let qi = if qv > 0.0 { (sign * qv) as i16 } else { 0i16 };
                ac_flat.push(qi);
                ac_turing.push(0);

                // Dequantize: L_rec = qi * pos_step (what the decoder will see)
                if collect_l_rec && (zz_idx - 1) < ac_per_block {
                    l_rec_block[zz_idx - 1] = qi as f64 * pos_step;
                }
            }

            if collect_l_rec {
                l_rec_blocks.push(l_rec_block);
            }

            // EOB: truncate trailing zeros
            let block_end = ac_flat.len();
            let n_in_block = block_end - block_start;
            let mut last_nz = 0usize;
            for i in (0..n_in_block).rev() {
                if ac_flat[block_start + i] != 0 { last_nz = i + 1; break; }
            }
            ac_flat.truncate(block_start + last_nz);
            ac_turing.truncate(block_start + last_nz);
            eob_positions.push(last_nz as u16);
        }

        // Write EOB positions: DPCM + rANS compression
        {
            let mut eob_deltas: Vec<i16> = Vec::with_capacity(n_blocks);
            let mut prev_eob: i16 = 0;
            for &eob in &eob_positions {
                eob_deltas.push(eob as i16 - prev_eob);
                prev_eob = eob as i16;
            }
            let eob_turing = vec![0u8; eob_deltas.len()]; // uniform context
            let eob_encoded = rans::rans_encode_band_v12(&eob_deltas, &eob_turing);
            body.extend_from_slice(&(n_blocks as u32).to_le_bytes());
            body.extend_from_slice(&(eob_encoded.len() as u32).to_le_bytes());
            body.extend_from_slice(&eob_encoded);
        }

        // Encode AC via rANS v12
        let ac_encoded = rans::rans_encode_band_v12(&ac_flat, &ac_turing);
        body.extend_from_slice(&(ac_encoded.len() as u32).to_le_bytes());
        body.extend_from_slice(&ac_encoded);

        l_rec_blocks
    };

    // ------------------------------------------------------------------
    // Pass 1: Encode luma channel (ch_idx = 0)
    // ------------------------------------------------------------------
    {
        let ch = &channels_lot[0];
        body.extend_from_slice(&(ch.grid_h as u16).to_le_bytes());
        body.extend_from_slice(&(ch.grid_w as u16).to_le_bytes());
        body.extend_from_slice(&(dc_data[0].dc_encoded.len() as u32).to_le_bytes());
        body.extend_from_slice(&dc_data[0].dc_encoded);
        // Match section (luma only)
        body.extend_from_slice(&(match_bytes.len() as u32).to_le_bytes());
        body.extend_from_slice(&match_bytes);
    }
    let l_rec_ac_blocks = encode_channel_ac(&channels_lot[0], 0, &mut body, use_cfl);

    // ------------------------------------------------------------------
    // Pass 2: Encode chroma channels (ch_idx = 1, 2) with CfL prediction
    // ------------------------------------------------------------------
    for ch_idx in 1..3usize {
        let ch = &channels_lot[ch_idx];
        let n_blocks = ch.grid_h * (if use_variable { 1 } else { ch.grid_w });

        // CfL analysis: per-block decision and alpha indices (AC frequency domain)
        let mut cfl_flags_vec: Vec<bool> = vec![false; n_blocks];
        let mut cfl_alpha_idx: Vec<usize> = vec![3; n_blocks]; // 3 = alpha 0.0 (neutral)

        if use_cfl && !l_rec_ac_blocks.is_empty() {
            for block_idx in 0..n_blocks {
                if block_idx >= l_rec_ac_blocks.len() { break; }
                let l_rec_block = &l_rec_ac_blocks[block_idx];
                let c_block = &ch.ac_blocks[block_idx];
                // Both are in LOT AC domain (natural order, zero-mean by definition)
                let len = l_rec_block.len().min(c_block.len());
                if len == 0 { continue; }
                let (alpha_idx, _alpha_val, use_block) = cfl::estimate_alpha_ac(&l_rec_block[..len], &c_block[..len]);
                if use_block {
                    cfl_flags_vec[block_idx] = true;
                    cfl_alpha_idx[block_idx] = alpha_idx;
                }
            }
        }

        // Write channel header (grid dims + DC)
        body.extend_from_slice(&(ch.grid_h as u16).to_le_bytes());
        body.extend_from_slice(&(ch.grid_w as u16).to_le_bytes());
        body.extend_from_slice(&(dc_data[ch_idx].dc_encoded.len() as u32).to_le_bytes());
        body.extend_from_slice(&dc_data[ch_idx].dc_encoded);

        // Write CfL metadata before AC data — rANS compressed
        if use_cfl {
            // Pack: [flag0, flag1, ..., flagN, alpha_active0, alpha_active1, ...]
            // Each flag is 0 or 1, each alpha is 0..7. All as i16 for rANS v12.
            let mut cfl_symbols: Vec<i16> = Vec::with_capacity(n_blocks * 2);
            for &active in &cfl_flags_vec {
                cfl_symbols.push(if active { 1 } else { 0 });
            }
            for block_idx in 0..n_blocks {
                if cfl_flags_vec[block_idx] {
                    cfl_symbols.push((cfl_alpha_idx[block_idx] & 0x7) as i16);
                }
            }
            let cfl_ctx = vec![0u8; cfl_symbols.len()];
            let cfl_encoded = rans::rans_encode_band_v12(&cfl_symbols, &cfl_ctx);
            // Write: n_symbols (u32) + encoded_len (u32) + data
            body.extend_from_slice(&(cfl_symbols.len() as u32).to_le_bytes());
            body.extend_from_slice(&(cfl_encoded.len() as u32).to_le_bytes());
            body.extend_from_slice(&cfl_encoded);
        }

        // Build modified AC blocks with CfL residual applied
        if use_cfl && !l_rec_ac_blocks.is_empty() {
            // Create a modified ChannelLOT with CfL-predicted AC
            let mut modified_ac: Vec<Vec<f64>> = ch.ac_blocks.to_vec();
            for block_idx in 0..n_blocks {
                if cfl_flags_vec[block_idx] && block_idx < l_rec_ac_blocks.len() {
                    let l_rec_block = &l_rec_ac_blocks[block_idx];
                    let c_block = &ch.ac_blocks[block_idx];
                    let len = l_rec_block.len().min(c_block.len());
                    let residual = cfl::apply_prediction(&l_rec_block[..len], &c_block[..len], cfl_alpha_idx[block_idx]);
                    // Replace the AC coefficients with the residual
                    modified_ac[block_idx] = residual;
                }
            }
            let modified_ch = ChannelLOT {
                dc_norm: ch.dc_norm,
                ac_blocks: &modified_ac,
                grid_h: ch.grid_h,
                grid_w: ch.grid_w,
                chroma_factor: ch.chroma_factor,
            };
            encode_channel_ac(&modified_ch, ch_idx, &mut body, false);
        } else {
            encode_channel_ac(ch, ch_idx, &mut body, false);
        }
    }

    if flags & FLAG_CHROMA_RESIDUAL != 0 {
        let sat_map = color::saturation_map(&c1_ch, &c2_ch, height, width);
        let chroma_resid_step = detail_step * calibration::CHROMA_RESIDUAL_STEP_MULT;

        let (c1_mask, c1_resid) = color::encode_chroma_residual(
            &c1_ch, &c1_sub, &sat_map, height, width, c1_h, c1_w, chroma_resid_step,
        );
        body.extend_from_slice(&(c1_mask.len() as u32).to_le_bytes());
        body.extend_from_slice(&c1_mask);
        body.extend_from_slice(&(c1_resid.len() as u32).to_le_bytes());
        for &v in &c1_resid { body.push(v as u8); }
    }

    let header = Aur2Header {
        version: 12,
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

pub fn encode_aur2_hex(
    rgb: &[u8],
    width: usize,
    height: usize,
    quality: u8,
) -> Result<AureaEncoderResult, Box<dyn std::error::Error>> {
    crate::hex_encoder::encode_aur2_hex(rgb, width, height, quality)
}

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

        if (ratio - 1.0).abs() < calibration::RATE_CONTROL_TOLERANCE {
            return Ok(result);
        }

        if actual_bpp > target_bpp {
            q_hi = q;
        } else {
            q_lo = q;
        }
        best_result = Some(result);

        if q_hi - q_lo <= 1 { break; }
    }

    best_result.ok_or_else(|| "Rate control failed to converge".into())
}

pub fn encode_unified(
    rgb: &[u8], width: usize, height: usize, params: &crate::codec_params::CodecParams,
) -> Result<AureaEncoderResult, Box<dyn std::error::Error>> {
    use crate::codec_params::Pipeline;
    match params.pipeline {
        Pipeline::LotAdn4 => {
            let p = AureaEncoderParams {
                quality: params.quality,
                n_representatives: quality_to_n_repr(params.quality),
                geometric: false,
            };
            encode_aur2(rgb, width, height, &p)
        }
        Pipeline::EdgeEnergy => {
            crate::hex_encoder::encode_edge_energy(rgb, width, height, params.quality)
        }
        Pipeline::HexPyramid => {
            crate::hex_encoder::encode_aur2_hex(rgb, width, height, params.quality)
        }
        Pipeline::EdgeEnergyDna => {
            crate::hex_encoder::encode_edge_energy_dna(rgb, width, height, params.quality)
        }
        Pipeline::Optica => {
            crate::hex_encoder::encode_optica(rgb, width, height, params)
        }
    }
}