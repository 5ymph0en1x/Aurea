/// Native AUREA v13 encoder: RGB -> .aur file (AUR2 format).
/// Modified for the D.N.A. architecture (spatial codons).
///
/// v2 (LOT): Block-based Lapped Orthogonal Transform replaces CDF 9/7 wavelet.
/// v13: 2-D Turing field + Psychovisual Pivot (DNA8)
use crate::bitstream::{self, Aur2Header, TuringHeader};
use crate::calibration;
use crate::cfl;
use crate::color;
use crate::geometric::{self, MatchKind, PrimitiveMatch};
use crate::hierarchy::{self, HierarchyParams};
use crate::lot;
use crate::rans;
use crate::scene_analysis;
use crate::spectral_envelope::{SpectralNeighborTables, SpectralSegment};
use crate::trellis;
use crate::wavelet;

// ======================================================================
// v3 feature flags (bitstream body)
// ======================================================================
pub const FLAG_CSF_MODULATION: u16 = 0x0001; // Point 5: CSF frequency-dependent QMAT
pub const FLAG_CHROMA_RESIDUAL: u16 = 0x0002; // Point 3: adaptive chroma residual
pub const FLAG_VARIABLE_BLOCKS: u16 = 0x0004; // Point 2: quadtree 8/16/32
pub const FLAG_SCENE_ANALYSIS: u16 = 0x0008; // Point 7: scene analysis in encoder
pub const FLAG_STRUCTURAL: u16 = 0x0010; // Point 4: structural coherence (4D codon)
pub const FLAG_DPCM_DC: u16 = 0x0020; // Stage 3: golden DPCM for DC
pub const FLAG_QUALITY_ADAPTIVE: u16 = 0x0040; // DNA4: quality-adaptive lot_factor, qmat_power, spin
pub const FLAG_LOT_OVERLAP: u16 = 0x0080; // DNA4: 50% LOT overlap at high quality
pub const FLAG_WEBER_TRNA: u16 = 0x0100; // DNA5: Weber-Fechner luminance tRNA on AC step
pub const FLAG_BAYESIAN_HIERARCHY: u16 = 0x0200; // Bayesian predictive hierarchy
pub const FLAG_CFL: u16 = 0x0400; // Chroma-from-Luma prediction
pub const FLAG_PHYLLOTACTIC_SCAN: u16 = 0x0800; // Golden-angle AC block order
pub const FLAG_SPIRAL_ARC_SCAN: u16 = 0x1000; // Continuous fixed-pitch AC spiral
pub const FLAG_FOLDED_SPECTRAL_SCAN: u16 = 0x2000; // Fixed-pitch DCT-quadrant spiral
pub const FLAG_PROGRESSIVE_PACKETS: u16 = 0x4000; // Independent Fibonacci-sized AC arcs
pub const FLAG_CHROMA_BREATHING: u16 = 0x8000; // DC-driven chroma budget redistribution

pub const DEFAULT_V3_FLAGS: u16 = FLAG_CSF_MODULATION
    | FLAG_SCENE_ANALYSIS
    | FLAG_DPCM_DC
    | FLAG_QUALITY_ADAPTIVE
    | FLAG_WEBER_TRNA;

/// Golden DPCM prediction for DC grid in raster order.
pub fn golden_dc_predict(dc_grid: &[i16], gy: usize, gx: usize, gw: usize) -> i16 {
    use crate::golden::{PHI_INV, PHI_INV2, PHI_INV3};

    if gy == 0 && gx == 0 {
        return 0;
    }
    if gy == 0 {
        return dc_grid[gx - 1];
    }
    if gx == 0 {
        return dc_grid[(gy - 1) * gw];
    }

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
    /// P0: which block lattice the transform runs on. The default must stay
    /// `AdaptiveCritical` so the standard v27 output remains byte-identical.
    pub lot_topology: lot::LotTopology,
}

pub struct AureaEncoderResult {
    pub aurea_data: Vec<u8>,
    pub compressed_size: usize,
}

// ======================================================================
// DNA8: Psychovisual Turing Pivot (high-rate vs low-rate)
// ======================================================================
/// Computes the Turing modulation factor with a psychovisual pivot.
/// At low bitrate (quality < 50): gamma > 0, preserves structure (sharp edges), sacrifices smooth areas.
/// At high bitrate (quality > 60): gamma < 0, spatial masking (sacrifices complex textures), protects smooth areas (anti-banding).
pub fn psychovisual_turing_pivot(raw_turing_mod: f64, quality: u8) -> f64 {
    // 1. Calibration parameters (tunable via AUREA_PIVOT_GAMMA_LOW / _HIGH)
    let [gamma_low, gamma_high] = *crate::calibration::TUNABLE_PIVOT;

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
    // Exponential mapping calibrated on 6 HD images (DNA5 benchmark).
    // Correction factor 1.3x applied to close the gap with JPEG.
    // Q=0 → ~40, Q=50 → ~7.2, Q=75 → ~3.3, Q=90 → ~2.0, Q=100 → ~1.0
    let base_step = 39.0 * (-3.5 * q_frac).exp() + 0.9;
    let factor = 1.0 - 0.4 * c_norm;

    (base_step * factor).clamp(0.1, 48.0)
}

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

// ======================================================================
// v13 LOT encoder utilities
// ======================================================================

const LOT_BLOCK_SIZE: usize = 16;

/// Number of raw Exp-Golomb payload bits carried by a non-zero magnitude in
/// the v20 grammar. The adaptive zero/sign decisions are unchanged by the
/// refinement because it never changes support or sign.
#[inline]
fn codon_magnitude_payload_bits(magnitude: u16) -> u32 {
    if magnitude <= 1 {
        0
    } else {
        let n1 = magnitude as u32 - 1;
        let significant = 32 - n1.leading_zeros();
        2 * significant - 1
    }
}

/// Quadratic error of one transform coefficient after inverse GCT.  Since
/// LOT is linear, comparing candidates here preserves the exact causal color
/// coupling before spatial synthesis.  The extra Y term targets the last
/// objective gap against JPEG 4:2:0 without hiding RGB regressions.
#[inline]
fn color_codon_distortion(e_l: f64, e_c1: f64, e_c2: f64, y_weight: f64) -> f64 {
    use crate::golden::{PHI, PHI_INV};

    let c1_to_g = (PHI_INV / (2.0 * PHI)) / 0.5;
    let c2_to_g = (1.0 / (2.0 * PHI)) / 0.5;
    let e_r = e_l + e_c2;
    let e_g = e_l - c1_to_g * e_c1 - c2_to_g * e_c2;
    let e_b = e_l + e_c1;
    let e_y = 0.299 * e_r + 0.587 * e_g + 0.114 * e_b;
    e_r * e_r + e_g * e_g + e_b * e_b + 3.0 * y_weight * e_y * e_y
}

/// Refine the last member of a causal color codon.  Candidate support and
/// sign stay fixed, hence EOBs, packet topology and zero/sign grammars remain
/// stable.  Only the neighbouring magnitude below/above the trellis choice
/// is considered.
#[inline]
fn refine_c2_codon(
    current_q: i16,
    step: f64,
    target_l: f64,
    rec_l: f64,
    target_c1: f64,
    rec_c1: f64,
    target_c2: f64,
    c2_prediction: f64,
    strength: f64,
    y_weight: f64,
    lambda: f64,
    rate_weight: f64,
) -> (i16, f64, f64) {
    if current_q == 0 || strength <= 0.0 || !step.is_finite() || step <= 0.0 {
        return (current_q, 0.0, 0.0);
    }

    let sign = if current_q < 0 { -1i32 } else { 1i32 };
    let current_magnitude = current_q.unsigned_abs() as i32;
    let e_l = rec_l - target_l;
    let e_c1 = rec_c1 - target_c1;
    let basis_norm = 1.0 + crate::golden::PHI_INV2;

    let objective = |q: i16| {
        let rec_c2 = q as f64 * step + c2_prediction;
        let e_c2 = rec_c2 - target_c2;
        let independent = basis_norm * e_c2 * e_c2;
        let joint = color_codon_distortion(e_l, e_c1, e_c2, y_weight);
        (1.0 - strength) * independent + strength * joint
    };

    let current_distortion = objective(current_q);
    let current_payload = codon_magnitude_payload_bits(current_q.unsigned_abs()) as f64;
    let mut best_q = current_q;
    let mut best_cost = current_distortion;
    let mut best_distortion = current_distortion;
    let mut best_rate_delta = 0.0;

    for magnitude in [current_magnitude.saturating_sub(1), current_magnitude + 1] {
        if magnitude <= 0 || magnitude > i16::MAX as i32 {
            continue;
        }
        let candidate = (sign * magnitude) as i16;
        let candidate_distortion = objective(candidate);
        let payload = codon_magnitude_payload_bits(candidate.unsigned_abs()) as f64;
        let rate_delta = payload - current_payload;
        // The witness redistributes an existing codon budget; it may not buy
        // a longer magnitude representation.  Exact rANS bytes can still
        // move slightly through adaptive energy contexts and are measured by
        // the outer benchmark.
        if rate_delta > 0.0 {
            continue;
        }
        let cost = candidate_distortion + lambda * rate_weight * rate_delta;
        if cost + 1e-12 < best_cost {
            best_q = candidate;
            best_cost = cost;
            best_distortion = candidate_distortion;
            best_rate_delta = rate_delta;
        }
    }

    (
        best_q,
        (current_distortion - best_distortion).max(0.0),
        best_rate_delta,
    )
}

/// First causal chroma coordinate.  C2 is provisionally assumed exact; the
/// following C2 pass then closes the triplet using the reconstructed C1.
#[inline]
fn refine_c1_codon(
    current_q: i16,
    step: f64,
    target_l: f64,
    rec_l: f64,
    target_c1: f64,
    c1_prediction: f64,
    strength: f64,
    y_weight: f64,
    lambda: f64,
    rate_weight: f64,
) -> (i16, f64, f64) {
    if current_q == 0 || strength <= 0.0 || !step.is_finite() || step <= 0.0 {
        return (current_q, 0.0, 0.0);
    }

    let sign = if current_q < 0 { -1i32 } else { 1i32 };
    let current_magnitude = current_q.unsigned_abs() as i32;
    let e_l = rec_l - target_l;
    let c1_to_g = crate::golden::PHI_INV2;
    let basis_norm = 1.0 + c1_to_g * c1_to_g;
    let objective = |q: i16| {
        let rec_c1 = q as f64 * step + c1_prediction;
        let e_c1 = rec_c1 - target_c1;
        let independent = basis_norm * e_c1 * e_c1;
        let joint = color_codon_distortion(e_l, e_c1, 0.0, y_weight);
        (1.0 - strength) * independent + strength * joint
    };

    let current_distortion = objective(current_q);
    let current_payload = codon_magnitude_payload_bits(current_q.unsigned_abs()) as f64;
    let mut best_q = current_q;
    let mut best_cost = current_distortion;
    let mut best_distortion = current_distortion;
    let mut best_rate_delta = 0.0;

    for magnitude in [current_magnitude.saturating_sub(1), current_magnitude + 1] {
        if magnitude <= 0 || magnitude > i16::MAX as i32 {
            continue;
        }
        let candidate = (sign * magnitude) as i16;
        let candidate_distortion = objective(candidate);
        let payload = codon_magnitude_payload_bits(candidate.unsigned_abs()) as f64;
        let rate_delta = payload - current_payload;
        if rate_delta > 0.0 {
            continue;
        }
        let cost = candidate_distortion + lambda * rate_weight * rate_delta;
        if cost + 1e-12 < best_cost {
            best_q = candidate;
            best_cost = cost;
            best_distortion = candidate_distortion;
            best_rate_delta = rate_delta;
        }
    }

    (
        best_q,
        (current_distortion - best_distortion).max(0.0),
        best_rate_delta,
    )
}

fn normalize_dc(dc: &[f64]) -> (Vec<f64>, f64, f64) {
    // The header carries this range as f32 (`Header::ll_ranges`), and the
    // decoder reconstructs the DC grid from the rounded value while the
    // encoder used to keep the f64 original. Every zero-bit field is built on
    // that grid, so the two sides were modelling slightly different images.
    // Rounding here makes both use bit-identical numbers.
    //
    // The Turing field itself survived the discrepancy — Sobel cancels the
    // offset and `normalize_01` cancels the scale — but the consumers that
    // quantize an *absolute* DC magnitude did not: the brightness clamp in
    // `hierarchy::bright_compensation_step_modulations` and the 8-level
    // `round()` in `hierarchy::spectral_exchange_local_levels` can both land
    // on the other side of a decision boundary for a knife-edge block.
    let min = dc.iter().copied().fold(f64::INFINITY, f64::min) as f32 as f64;
    let max = dc.iter().copied().fold(f64::NEG_INFINITY, f64::max) as f32 as f64;
    let range = (max - min).max(1e-6);
    // Rounding can move `min` above the true minimum by ~1e-7 relative, so a
    // sample may fall a hair outside [0, 255]. Clamp rather than propagate it.
    let norm: Vec<f64> = dc
        .iter()
        .map(|&v| ((v - min) * 255.0 / range).clamp(0.0, 255.0))
        .collect();
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

/// Quantized AC trace retained only while `AUREA_CODON_STATS=1`.
///
/// This is deliberately outside the bitstream contract: it lets the R&D lab
/// measure dependencies in the exact symbols seen by rANS without changing a
/// single encoded byte.
struct CodonChannelTrace {
    blocks: Vec<Vec<i16>>,
    eob: Vec<u16>,
}

fn entropy_bits(counts: &[u64]) -> f64 {
    let total = counts.iter().sum::<u64>() as f64;
    if total == 0.0 {
        return 0.0;
    }
    counts
        .iter()
        .filter(|&&count| count != 0)
        .map(|&count| {
            let p = count as f64 / total;
            -(count as f64) * p.log2()
        })
        .sum()
}

fn emit_phase_pair_stats(label: &str, joint: &[u64; 4]) {
    let samples = joint.iter().sum::<u64>();
    if samples == 0 {
        return;
    }
    let predictor = [joint[0] + joint[1], joint[2] + joint[3]];
    let sign = [joint[0] + joint[2], joint[1] + joint[3]];
    let conditional_bits = entropy_bits(joint) - entropy_bits(&predictor);
    let raw_gain = samples as f64 - conditional_bits;
    let same = joint[0] + joint[3];
    eprintln!(
        "[AUREA_PHASE_STATS] {label} samples={samples} same_phase={:.6} conditional_bps={:.6} raw_gain_bits={raw_gain:.1} raw_gain_bps={:.6} sign_bias_bps={:.6}",
        same as f64 / samples as f64,
        conditional_bits / samples as f64,
        raw_gain / samples as f64,
        entropy_bits(&sign) / samples as f64,
    );
}

#[inline]
fn codon_magnitude_class(value: i16) -> usize {
    match value.unsigned_abs() {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        _ => 3,
    }
}

#[inline]
fn codon_eob_class(eob: u16) -> usize {
    match eob {
        0 => 0,
        1..=8 => 1,
        9..=32 => 2,
        33..=128 => 3,
        _ => 4,
    }
}

/// Emit empirical entropy ceilings for two candidate dense codons:
///
/// 1. a joint L/C1/C2 support or magnitude symbol on the common AC prefix;
/// 2. a dyadic parent -> child support context inside each LOT spectrum.
///
/// Reported gains are mutual-information ceilings, not promised codec gains:
/// framing, model adaptation and progressive packet boundaries still have to
/// be paid by a real implementation.
fn emit_codon_dependency_stats(traces: &[CodonChannelTrace], block_sizes: &[usize]) {
    if traces.len() != 3 {
        return;
    }

    let n_blocks = traces
        .iter()
        .map(|trace| trace.blocks.len())
        .min()
        .unwrap_or(0);

    // Cross-channel codons are measured only where all three independent
    // streams currently emit a symbol. This avoids crediting the proposal for
    // trailing zeros that EOB already removes.
    let mut support_joint = [0u64; 8];
    let mut support_marginal = [[0u64; 2]; 3];
    let mut magnitude_joint = [0u64; 64];
    let mut magnitude_marginal = [[0u64; 4]; 3];
    let mut common_samples = 0u64;

    // EOB structure is evaluated separately with coarse, stable classes.
    let mut eob_joint = [0u64; 125];
    let mut eob_marginal = [[0u64; 5]; 3];

    for block_idx in 0..n_blocks {
        let common = traces
            .iter()
            .map(|trace| trace.eob[block_idx] as usize)
            .min()
            .unwrap_or(0);
        for coeff_idx in 0..common {
            let values = [
                traces[0].blocks[block_idx][coeff_idx],
                traces[1].blocks[block_idx][coeff_idx],
                traces[2].blocks[block_idx][coeff_idx],
            ];
            let mut support_code = 0usize;
            let mut magnitude_code = 0usize;
            for channel in 0..3 {
                let support = usize::from(values[channel] != 0);
                let magnitude = codon_magnitude_class(values[channel]);
                support_marginal[channel][support] += 1;
                magnitude_marginal[channel][magnitude] += 1;
                support_code |= support << channel;
                magnitude_code |= magnitude << (2 * channel);
            }
            support_joint[support_code] += 1;
            magnitude_joint[magnitude_code] += 1;
            common_samples += 1;
        }

        let mut joint_class = 0usize;
        let mut stride = 1usize;
        for channel in 0..3 {
            let class = codon_eob_class(traces[channel].eob[block_idx]);
            eob_marginal[channel][class] += 1;
            joint_class += class * stride;
            stride *= 5;
        }
        eob_joint[joint_class] += 1;
    }

    let support_independent = support_marginal
        .iter()
        .map(|counts| entropy_bits(counts))
        .sum::<f64>();
    let support_dense = entropy_bits(&support_joint);
    let magnitude_independent = magnitude_marginal
        .iter()
        .map(|counts| entropy_bits(counts))
        .sum::<f64>();
    let magnitude_dense = entropy_bits(&magnitude_joint);
    let eob_independent = eob_marginal
        .iter()
        .map(|counts| entropy_bits(counts))
        .sum::<f64>();
    let eob_dense = entropy_bits(&eob_joint);

    let support_gain = (support_independent - support_dense).max(0.0);
    let magnitude_gain = (magnitude_independent - magnitude_dense).max(0.0);
    let eob_gain = (eob_independent - eob_dense).max(0.0);
    let common_den = common_samples.max(1) as f64;
    let block_den = n_blocks.max(1) as f64;

    eprintln!(
        "[AUREA_CODON_STATS] cross samples={common_samples} support_gain_bits={support_gain:.1} support_gain_bps={:.6} magnitude_gain_bits={magnitude_gain:.1} magnitude_gain_bps={:.6}",
        support_gain / common_den,
        magnitude_gain / common_den,
    );
    eprintln!(
        "[AUREA_CODON_STATS] eob blocks={n_blocks} class_gain_bits={eob_gain:.1} class_gain_bpb={:.6}",
        eob_gain / block_den,
    );

    // A dyadic frequency tree is used as an analysis proxy for a wavelet
    // parenthood: (u, v) -> (floor(u/2), floor(v/2)). This does not pretend
    // that LOT coefficients are wavelets; it measures whether a zero-bit
    // parent context could densify their support decisions.
    let zigzags = [
        (8usize, ac_zigzag_order(8)),
        (16usize, ac_zigzag_order(16)),
        (32usize, ac_zigzag_order(32)),
    ];
    let names = ["L", "C1", "C2"];

    // A transform coefficient is a signed projection: its sign is the
    // quantized 0/pi phase of the corresponding basis wave. The production
    // grammar currently spends one equiprobable bit on every non-zero sign.
    // These counters measure whether causal phase relationships can replace
    // that particle-like representation with a conditional wave model.
    let mut parent_phase = [[0u64; 4]; 3];
    let mut previous_phase = [[0u64; 4]; 3];
    let mut cross_l_c1 = [0u64; 4];
    let mut cross_l_c2 = [0u64; 4];
    let mut cross_c1_c2 = [0u64; 4];
    // channel * 3^4 + ternary(parent, previous, L, C1). State zero means
    // unavailable/zero, one positive and two negative. Current-channel values
    // are never admitted into the context.
    let mut combined_phase = vec![[0u64; 2]; 3 * 81];
    let mut raw_phase = [0u64; 2];

    for channel in 0..3 {
        let mut joint = [0u64; 4];
        let mut parent_marginal = [0u64; 2];
        let mut child_marginal = [0u64; 2];

        for block_idx in 0..n_blocks {
            let block_size = block_sizes.get(block_idx).copied().unwrap_or(16);
            let zigzag = zigzags
                .iter()
                .find(|(size, _)| *size == block_size)
                .map(|(_, order)| order)
                .unwrap_or(&zigzags[1].1);
            let mut natural = vec![0i16; block_size * block_size];
            for (position, &value) in traces[channel].blocks[block_idx].iter().enumerate() {
                if let Some(&natural_idx) = zigzag.get(position) {
                    natural[natural_idx] = value;
                }
            }

            let mut previous_nonzero = None::<i16>;

            // Restrict children to coefficients that the current stream really
            // emits (up to EOB). Counting the implicit zero tail here would
            // falsely attribute EOB's existing gain to the wavelet context.
            for child_position in 0..traces[channel].blocks[block_idx].len() {
                let natural_idx = zigzag[child_position];
                let row = natural_idx / block_size;
                let col = natural_idx % block_size;
                let parent_row = row / 2;
                let parent_col = col / 2;
                let has_ac_parent = parent_row != 0 || parent_col != 0;
                if has_ac_parent {
                    let parent = usize::from(natural[parent_row * block_size + parent_col] != 0);
                    let child = usize::from(natural[natural_idx] != 0);
                    parent_marginal[parent] += 1;
                    child_marginal[child] += 1;
                    joint[parent * 2 + child] += 1;
                }

                let child_value = natural[natural_idx];
                if child_value != 0 {
                    let child_sign = usize::from(child_value < 0);
                    raw_phase[child_sign] += 1;

                    let parent_value = if has_ac_parent {
                        natural[parent_row * block_size + parent_col]
                    } else {
                        0
                    };
                    if parent_value != 0 {
                        let parent_sign = usize::from(parent_value < 0);
                        parent_phase[channel][parent_sign * 2 + child_sign] += 1;
                    }
                    if let Some(previous_value) = previous_nonzero {
                        let previous_sign = usize::from(previous_value < 0);
                        previous_phase[channel][previous_sign * 2 + child_sign] += 1;
                    }

                    let l_value = if channel > 0 {
                        traces[0].blocks[block_idx]
                            .get(child_position)
                            .copied()
                            .unwrap_or(0)
                    } else {
                        0
                    };
                    let c1_value = if channel > 1 {
                        traces[1].blocks[block_idx]
                            .get(child_position)
                            .copied()
                            .unwrap_or(0)
                    } else {
                        0
                    };
                    if channel == 1 && l_value != 0 {
                        cross_l_c1[usize::from(l_value < 0) * 2 + child_sign] += 1;
                    }
                    if channel == 2 {
                        if l_value != 0 {
                            cross_l_c2[usize::from(l_value < 0) * 2 + child_sign] += 1;
                        }
                        if c1_value != 0 {
                            cross_c1_c2[usize::from(c1_value < 0) * 2 + child_sign] += 1;
                        }
                    }

                    let phase_state = |value: i16| -> usize {
                        if value == 0 {
                            0
                        } else if value > 0 {
                            1
                        } else {
                            2
                        }
                    };
                    let states = [
                        phase_state(parent_value),
                        phase_state(previous_nonzero.unwrap_or(0)),
                        phase_state(l_value),
                        phase_state(c1_value),
                    ];
                    let mut context = channel * 81;
                    let mut stride = 1usize;
                    for state in states {
                        context += state * stride;
                        stride *= 3;
                    }
                    combined_phase[context][child_sign] += 1;
                    previous_nonzero = Some(child_value);
                }
            }
        }

        let pairs = joint.iter().sum::<u64>();
        let gain = (entropy_bits(&parent_marginal) + entropy_bits(&child_marginal)
            - entropy_bits(&joint))
        .max(0.0);
        let parent_zero = joint[0] + joint[1];
        let parent_nonzero = joint[2] + joint[3];
        let child_given_zero = joint[1] as f64 / parent_zero.max(1) as f64;
        let child_given_nonzero = joint[3] as f64 / parent_nonzero.max(1) as f64;
        eprintln!(
            "[AUREA_CODON_STATS] wavelet channel={} pairs={} p_child_nz_parent0={:.6} p_child_nz_parent1={:.6} gain_bits={:.1} gain_bppair={:.6}",
            names[channel],
            pairs,
            child_given_zero,
            child_given_nonzero,
            gain,
            gain / pairs.max(1) as f64,
        );
    }

    for channel in 0..3 {
        emit_phase_pair_stats(
            &format!("parent channel={}", names[channel]),
            &parent_phase[channel],
        );
        emit_phase_pair_stats(
            &format!("previous_nonzero channel={}", names[channel]),
            &previous_phase[channel],
        );
    }
    emit_phase_pair_stats("cross predictor=L child=C1", &cross_l_c1);
    emit_phase_pair_stats("cross predictor=L child=C2", &cross_l_c2);
    emit_phase_pair_stats("cross predictor=C1 child=C2", &cross_c1_c2);

    let phase_samples = raw_phase.iter().sum::<u64>();
    let combined_bits = combined_phase
        .iter()
        .map(|counts| entropy_bits(counts))
        .sum::<f64>();
    let combined_gain = phase_samples as f64 - combined_bits;
    eprintln!(
        "[AUREA_PHASE_STATS] combined_causal samples={phase_samples} contexts={} conditional_bps={:.6} raw_gain_bits={combined_gain:.1} raw_gain_bps={:.6} global_sign_bps={:.6}",
        combined_phase
            .iter()
            .filter(|counts| counts[0] + counts[1] != 0)
            .count(),
        combined_bits / phase_samples.max(1) as f64,
        combined_gain / phase_samples.max(1) as f64,
        entropy_bits(&raw_phase) / phase_samples.max(1) as f64,
    );
}

#[derive(Clone, Copy, Default)]
struct FourierPhaseStats {
    samples: u64,
    matches: u64,
}

impl FourierPhaseStats {
    #[inline]
    fn observe(&mut self, prediction: f64, current: i16) {
        if current == 0 || prediction.abs() < 1e-9 {
            return;
        }
        self.samples += 1;
        self.matches += u64::from((prediction < 0.0) == (current < 0));
    }

    fn emit(self, channel: &str, label: &str) {
        if self.samples == 0 {
            return;
        }
        let residual = [self.matches, self.samples - self.matches];
        let conditional_bits = entropy_bits(&residual);
        eprintln!(
            "[AUREA_FOURIER_STATS] channel={channel} predictor={label} samples={} phase_match={:.6} residual_bps={:.6} raw_gain_bits={:.1} raw_gain_bps={:.6}",
            self.samples,
            self.matches as f64 / self.samples as f64,
            conditional_bits / self.samples as f64,
            self.samples as f64 - conditional_bits,
            1.0 - conditional_bits / self.samples as f64,
        );
    }
}

/// Analyze a fixed LOT sub-band as a spatially sampled oscillation.  All
/// predictors are causal in the physical progressive order: skeleton
/// coefficients use earlier raster blocks, details use earlier spiral arcs.
/// The AR(2) recurrence is the real form of a conjugate Fourier pair,
/// `x[n] = 2 cos(omega) x[n-1] - x[n-2]`.
fn emit_fourier_cycle_stats(
    traces: &[CodonChannelTrace],
    blocks: &[(usize, usize, usize)],
    progressive_packets: Option<&Vec<Vec<usize>>>,
) {
    if traces.len() != 3 || blocks.is_empty() {
        return;
    }
    let n_blocks = traces
        .iter()
        .map(|trace| trace.blocks.len())
        .min()
        .unwrap_or(0)
        .min(blocks.len());
    let mut index_by_geometry = std::collections::HashMap::with_capacity(n_blocks);
    for (index, &(y, x, size)) in blocks.iter().take(n_blocks).enumerate() {
        index_by_geometry.insert((y, x, size), index);
    }

    let mut detail_rank = vec![usize::MAX; n_blocks];
    if let Some(packets) = progressive_packets {
        let mut rank = 0usize;
        for packet in packets {
            for &block_idx in packet {
                if block_idx < n_blocks {
                    detail_rank[block_idx] = rank;
                    rank += 1;
                }
            }
        }
    }
    let is_causal = |position: usize, predecessor: usize, current: usize| -> bool {
        if position < crate::scan::PROGRESSIVE_SKELETON_COEFFS {
            predecessor < current
        } else {
            detail_rank.get(predecessor).copied().unwrap_or(usize::MAX)
                < detail_rank.get(current).copied().unwrap_or(usize::MAX)
        }
    };

    let zigzags = [
        (8usize, ac_zigzag_order(8)),
        (16usize, ac_zigzag_order(16)),
        (32usize, ac_zigzag_order(32)),
    ];
    let channel_names = ["L", "C1", "C2"];
    let labels = [
        "left_hold",
        "left_dct_parity",
        "top_hold",
        "top_dct_parity",
        "left_linear",
        "top_linear",
        "left_fourier_w0",
        "left_fourier_w1",
        "left_fourier_w2",
        "left_fourier_w3",
        "left_fourier_w4",
        "left_fourier_adaptive",
        "top_fourier_adaptive",
        "adaptive_strong_axis",
    ];

    for (channel, trace) in traces.iter().enumerate() {
        let mut stats = [FourierPhaseStats::default(); 14];
        for block_idx in 0..n_blocks {
            let (y, x, size) = blocks[block_idx];
            let zigzag = zigzags
                .iter()
                .find(|(candidate, _)| *candidate == size)
                .map(|(_, order)| order)
                .unwrap_or(&zigzags[1].1);
            let current_block = &trace.blocks[block_idx];
            for (position, &current) in current_block.iter().enumerate() {
                if current == 0 || position >= zigzag.len() {
                    continue;
                }
                let natural = zigzag[position];
                let v = natural / size;
                let u = natural % size;
                let get_predecessor = |axis: usize, distance: usize| -> Option<(usize, i16)> {
                    let geometry = if axis == 0 {
                        x.checked_sub(distance * size).map(|px| (y, px, size))
                    } else {
                        y.checked_sub(distance * size).map(|py| (py, x, size))
                    }?;
                    let predecessor = *index_by_geometry.get(&geometry)?;
                    if !is_causal(position, predecessor, block_idx) {
                        return None;
                    }
                    let value = trace.blocks[predecessor]
                        .get(position)
                        .copied()
                        .unwrap_or(0);
                    (value != 0).then_some((predecessor, value))
                };

                let left1 = get_predecessor(0, 1).map(|(_, value)| value);
                let top1 = get_predecessor(1, 1).map(|(_, value)| value);
                if let Some(value) = left1 {
                    stats[0].observe(value as f64, current);
                    let parity = if u & 1 == 0 { 1.0 } else { -1.0 };
                    stats[1].observe(parity * value as f64, current);
                }
                if let Some(value) = top1 {
                    stats[2].observe(value as f64, current);
                    let parity = if v & 1 == 0 { 1.0 } else { -1.0 };
                    stats[3].observe(parity * value as f64, current);
                }

                let left2 = get_predecessor(0, 2).map(|(_, value)| value);
                let top2 = get_predecessor(1, 2).map(|(_, value)| value);
                if let (Some(x1), Some(x2)) = (left1, left2) {
                    stats[4].observe(2.0 * x1 as f64 - x2 as f64, current);
                    for (offset, a) in [
                        2.0,
                        std::f64::consts::SQRT_2,
                        0.0,
                        -std::f64::consts::SQRT_2,
                        -2.0,
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        stats[6 + offset].observe(a * x1 as f64 - x2 as f64, current);
                    }
                }
                if let (Some(x1), Some(x2)) = (top1, top2) {
                    stats[5].observe(2.0 * x1 as f64 - x2 as f64, current);
                }

                let left3 = get_predecessor(0, 3).map(|(_, value)| value);
                let top3 = get_predecessor(1, 3).map(|(_, value)| value);
                let adaptive = |x1: i16, x2: i16, x3: i16| -> f64 {
                    let a = ((x1 as f64 + x3 as f64) / x2 as f64).clamp(-2.0, 2.0);
                    a * x1 as f64 - x2 as f64
                };
                let left_prediction = match (left1, left2, left3) {
                    (Some(x1), Some(x2), Some(x3)) => Some(adaptive(x1, x2, x3)),
                    _ => None,
                };
                let top_prediction = match (top1, top2, top3) {
                    (Some(x1), Some(x2), Some(x3)) => Some(adaptive(x1, x2, x3)),
                    _ => None,
                };
                if let Some(prediction) = left_prediction {
                    stats[11].observe(prediction, current);
                }
                if let Some(prediction) = top_prediction {
                    stats[12].observe(prediction, current);
                }
                let strong_axis = match (left_prediction, top_prediction, left1, top1) {
                    (Some(lp), Some(tp), Some(l), Some(t)) => {
                        Some(if l.unsigned_abs() >= t.unsigned_abs() {
                            lp
                        } else {
                            tp
                        })
                    }
                    (Some(lp), _, _, _) => Some(lp),
                    (_, Some(tp), _, _) => Some(tp),
                    _ => None,
                };
                if let Some(prediction) = strong_axis {
                    stats[13].observe(prediction, current);
                }
            }
        }
        for (label, measured) in labels.iter().zip(stats) {
            measured.emit(channel_names[channel], label);
        }
    }
}

/// Encode an exact, lossless alternative representation of the progressive AC
/// payload for measurement only. Each spectral position becomes one 3-bit
/// support codon, followed by the non-zero values of the participating
/// channels. Existing EOB streams remain untouched, so the comparison neither
/// steals their gain nor needs an artificial full-block zero tail.
fn emit_dense_codon_witness(
    traces: &[CodonChannelTrace],
    progressive_packets: Option<&Vec<Vec<usize>>>,
    turing_buckets: &[u8],
    current_ac_bytes: usize,
) {
    if traces.len() != 3 {
        return;
    }
    let n_blocks = traces
        .iter()
        .map(|trace| trace.blocks.len())
        .min()
        .unwrap_or(0);
    let all_blocks: Vec<usize> = (0..n_blocks).collect();
    let mut layers: Vec<(&[usize], usize, Option<usize>)> = Vec::new();
    layers.push((
        &all_blocks,
        0,
        Some(crate::scan::PROGRESSIVE_SKELETON_COEFFS),
    ));
    if let Some(packets) = progressive_packets {
        for packet in packets {
            layers.push((packet, crate::scan::PROGRESSIVE_SKELETON_COEFFS, None));
        }
    }

    let mut mask_bytes = 0usize;
    let mut payload_bytes = 0usize;
    let mut mask_symbols = 0usize;
    let mut nonzero_symbols = 0usize;

    for (block_indices, start, end_cap) in layers {
        let mut masks = Vec::<u8>::new();
        let mut payloads: [Vec<i16>; 3] = std::array::from_fn(|_| Vec::new());
        let mut contexts: [Vec<u8>; 3] = std::array::from_fn(|_| Vec::new());

        for &block_idx in block_indices {
            if block_idx >= n_blocks {
                continue;
            }
            let union_end = traces
                .iter()
                .map(|trace| trace.blocks[block_idx].len())
                .max()
                .unwrap_or(0);
            let end = end_cap.map(|cap| union_end.min(cap)).unwrap_or(union_end);
            if start >= end {
                continue;
            }
            let context = turing_buckets.get(block_idx).copied().unwrap_or(0);
            for coeff_idx in start..end {
                let mut mask = 0u8;
                for channel in 0..3 {
                    let value = traces[channel].blocks[block_idx]
                        .get(coeff_idx)
                        .copied()
                        .unwrap_or(0);
                    if value != 0 {
                        mask |= 1 << channel;
                        payloads[channel].push(value);
                        contexts[channel].push(context);
                        nonzero_symbols += 1;
                    }
                }
                masks.push(mask);
            }
        }

        mask_symbols += masks.len();
        mask_bytes += rans::rans_compress_bytes(&masks).len();
        for channel in 0..3 {
            payload_bytes +=
                rans::rans_encode_band_v12(&payloads[channel], &contexts[channel]).len();
        }
    }

    let dense_bytes = mask_bytes + payload_bytes;
    let delta = dense_bytes as i64 - current_ac_bytes as i64;
    let delta_pct = if current_ac_bytes == 0 {
        0.0
    } else {
        100.0 * delta as f64 / current_ac_bytes as f64
    };
    eprintln!(
        "[AUREA_CODON_WITNESS] current_ac_bytes={current_ac_bytes} dense_ac_bytes={dense_bytes} delta_bytes={delta} delta_pct={delta_pct:.4} mask_bytes={mask_bytes} payload_bytes={payload_bytes} mask_symbols={mask_symbols} nonzero_symbols={nonzero_symbols}"
    );
}

/// Exact byte witness for a causal dyadic-parent context. The coefficient
/// grammar and progressive ordering stay identical; only the adaptive rANS
/// state selected for each symbol changes.
fn emit_wavelet_parent_witness(
    traces: &[CodonChannelTrace],
    block_sizes: &[usize],
    progressive_packets: Option<&Vec<Vec<usize>>>,
    turing_buckets: &[u8],
    current_ac_bytes: usize,
) {
    if traces.len() != 3 {
        return;
    }
    let n_blocks = traces
        .iter()
        .map(|trace| trace.blocks.len())
        .min()
        .unwrap_or(0);
    let all_blocks: Vec<usize> = (0..n_blocks).collect();
    let mut layers: Vec<(&[usize], usize, Option<usize>)> = vec![(
        &all_blocks,
        0,
        Some(crate::scan::PROGRESSIVE_SKELETON_COEFFS),
    )];
    if let Some(packets) = progressive_packets {
        for packet in packets {
            layers.push((packet, crate::scan::PROGRESSIVE_SKELETON_COEFFS, None));
        }
    }
    let zigzags = [
        (8usize, ac_zigzag_order(8)),
        (16usize, ac_zigzag_order(16)),
        (32usize, ac_zigzag_order(32)),
    ];

    let mut wavelet_bytes = 0usize;
    let mut symbols = 0usize;
    let mut parent_nonzero_symbols = 0usize;
    for (block_indices, start, end_cap) in layers {
        for channel in 0..3 {
            let mut values = Vec::<i16>::new();
            let mut contexts = Vec::<u8>::new();
            let mut parents = Vec::<bool>::new();
            for &block_idx in block_indices {
                if block_idx >= n_blocks {
                    continue;
                }
                let block = &traces[channel].blocks[block_idx];
                let end = end_cap
                    .map(|cap| block.len().min(cap))
                    .unwrap_or(block.len());
                if start >= end {
                    continue;
                }
                let block_size = block_sizes.get(block_idx).copied().unwrap_or(16);
                let zigzag = zigzags
                    .iter()
                    .find(|(size, _)| *size == block_size)
                    .map(|(_, order)| order)
                    .unwrap_or(&zigzags[1].1);
                let mut natural = vec![0i16; block_size * block_size];
                for (position, &value) in block.iter().enumerate() {
                    natural[zigzag[position]] = value;
                }
                let context = turing_buckets.get(block_idx).copied().unwrap_or(0);
                for position in start..end {
                    let natural_idx = zigzag[position];
                    let row = natural_idx / block_size;
                    let col = natural_idx % block_size;
                    let parent_row = row / 2;
                    let parent_col = col / 2;
                    let parent_nonzero = (parent_row != 0 || parent_col != 0)
                        && natural[parent_row * block_size + parent_col] != 0;
                    values.push(block[position]);
                    contexts.push(context);
                    parents.push(parent_nonzero);
                    parent_nonzero_symbols += usize::from(parent_nonzero);
                }
            }
            symbols += values.len();
            wavelet_bytes += rans::rans_encode_band_v12_parent(&values, &contexts, &parents).len();
        }
    }

    let delta = wavelet_bytes as i64 - current_ac_bytes as i64;
    let delta_pct = if current_ac_bytes == 0 {
        0.0
    } else {
        100.0 * delta as f64 / current_ac_bytes as f64
    };
    eprintln!(
        "[AUREA_WAVELET_WITNESS] current_ac_bytes={current_ac_bytes} parent_ac_bytes={wavelet_bytes} delta_bytes={delta} delta_pct={delta_pct:.4} symbols={symbols} parent_nonzero_symbols={parent_nonzero_symbols}"
    );
}

/// Test a true causal triplet factorization while preserving the three
/// progressive streams and their individual EOBs. C1 sees L at the same
/// spectral position; C2 sees both L and C1. Two codon alphabets are compared:
/// support state and magnitude class of the already available components.
fn emit_vector_codon_witness(
    traces: &[CodonChannelTrace],
    block_sizes: &[usize],
    progressive_packets: Option<&Vec<Vec<usize>>>,
    turing_buckets: &[u8],
    production_ac_bytes: usize,
) {
    if traces.len() != 3 {
        return;
    }
    let n_blocks = traces
        .iter()
        .map(|trace| trace.blocks.len())
        .min()
        .unwrap_or(0);
    let all_blocks: Vec<usize> = (0..n_blocks).collect();
    let mut layers: Vec<(&[usize], usize, Option<usize>)> = vec![(
        &all_blocks,
        0,
        Some(crate::scan::PROGRESSIVE_SKELETON_COEFFS),
    )];
    if let Some(packets) = progressive_packets {
        for packet in packets {
            layers.push((packet, crate::scan::PROGRESSIVE_SKELETON_COEFFS, None));
        }
    }
    let zigzags = [
        (8usize, ac_zigzag_order(8)),
        (16usize, ac_zigzag_order(16)),
        (32usize, ac_zigzag_order(32)),
    ];

    let mut support_zero_bytes = 0usize;
    let mut support_full_bytes = 0usize;
    let mut magnitude_zero_bytes = 0usize;
    let mut magnitude_full_bytes = 0usize;

    for (block_indices, start, end_cap) in layers {
        for channel in 0..3 {
            let mut values = Vec::<i16>::new();
            let mut contexts = Vec::<u8>::new();
            let mut parents = Vec::<bool>::new();
            let mut support_classes = Vec::<u8>::new();
            let mut magnitude_classes = Vec::<u8>::new();

            for &block_idx in block_indices {
                if block_idx >= n_blocks {
                    continue;
                }
                let block = &traces[channel].blocks[block_idx];
                let end = end_cap
                    .map(|cap| block.len().min(cap))
                    .unwrap_or(block.len());
                if start >= end {
                    continue;
                }
                let block_size = block_sizes.get(block_idx).copied().unwrap_or(16);
                let zigzag = zigzags
                    .iter()
                    .find(|(size, _)| *size == block_size)
                    .map(|(_, order)| order)
                    .unwrap_or(&zigzags[1].1);
                let mut natural = vec![0i16; block_size * block_size];
                for (position, &value) in block.iter().enumerate() {
                    natural[zigzag[position]] = value;
                }
                let turing = turing_buckets.get(block_idx).copied().unwrap_or(0);

                for position in start..end {
                    let natural_idx = zigzag[position];
                    let row = natural_idx / block_size;
                    let col = natural_idx % block_size;
                    let parent_row = row / 2;
                    let parent_col = col / 2;
                    let parent_nonzero = (parent_row != 0 || parent_col != 0)
                        && natural[parent_row * block_size + parent_col] != 0;
                    let l_value = traces[0].blocks[block_idx]
                        .get(position)
                        .copied()
                        .unwrap_or(0);
                    let c1_value = traces[1].blocks[block_idx]
                        .get(position)
                        .copied()
                        .unwrap_or(0);
                    let support_class = match channel {
                        0 => 0,
                        1 => u8::from(l_value != 0),
                        _ => u8::from(l_value != 0) | (u8::from(c1_value != 0) << 1),
                    };
                    let magnitude_class = match channel {
                        0 => 0,
                        1 => codon_magnitude_class(l_value) as u8,
                        _ => codon_magnitude_class(l_value).max(codon_magnitude_class(c1_value))
                            as u8,
                    };
                    values.push(block[position]);
                    contexts.push(turing);
                    parents.push(parent_nonzero);
                    support_classes.push(support_class);
                    magnitude_classes.push(magnitude_class);
                }
            }

            support_zero_bytes += rans::rans_encode_band_v17_codon(
                &values,
                &contexts,
                &parents,
                &support_classes,
                false,
            )
            .len();
            support_full_bytes += rans::rans_encode_band_v17_codon(
                &values,
                &contexts,
                &parents,
                &support_classes,
                true,
            )
            .len();
            magnitude_zero_bytes += rans::rans_encode_band_v17_codon(
                &values,
                &contexts,
                &parents,
                &magnitude_classes,
                false,
            )
            .len();
            magnitude_full_bytes += rans::rans_encode_band_v17_codon(
                &values,
                &contexts,
                &parents,
                &magnitude_classes,
                true,
            )
            .len();
        }
    }

    let report = |name: &str, bytes: usize| {
        let delta = bytes as i64 - production_ac_bytes as i64;
        let percent = if production_ac_bytes == 0 {
            0.0
        } else {
            100.0 * delta as f64 / production_ac_bytes as f64
        };
        eprintln!(
            "[AUREA_VECTOR_CODON] variant={name} production_ac_bytes={production_ac_bytes} codon_ac_bytes={bytes} delta_bytes={delta} delta_pct={percent:.4}"
        );
    };
    report("support_zero", support_zero_bytes);
    report("support_zero_pm1", support_full_bytes);
    report("magnitude_zero", magnitude_zero_bytes);
    report("magnitude_zero_pm1", magnitude_full_bytes);
}

/// Exact byte witness for the strongest measured wave dependency: the C2
/// coefficient sign is coded as agreement/opposition with the already decoded
/// C1 sign at the same block/frequency coordinate. Pixels, quantized values,
/// packet boundaries and all magnitude decisions remain strictly unchanged.
fn emit_phase_codon_witness(
    traces: &[CodonChannelTrace],
    block_sizes: &[usize],
    block_geometries: Option<&[(usize, usize, usize)]>,
    progressive_packets: Option<&Vec<Vec<usize>>>,
    turing_buckets: &[u8],
    quality: u8,
    production_ac_bytes: usize,
) {
    if traces.len() != 3 {
        return;
    }
    let n_blocks = traces
        .iter()
        .map(|trace| trace.blocks.len())
        .min()
        .unwrap_or(0);
    let all_blocks: Vec<usize> = (0..n_blocks).collect();
    let zigzags = [
        (8usize, ac_zigzag_order(8)),
        (16usize, ac_zigzag_order(16)),
        (32usize, ac_zigzag_order(32)),
    ];

    for &(initial_p_same, adapt_shift) in &[
        (2048u32, 4u32),
        (2048, 6),
        (2048, 8),
        (3072, 4),
        (3072, 6),
        (3072, 8),
        (4096, 4),
        (4096, 6),
        (4096, 8),
        (8192, 4),
        (8192, 6),
        (8192, 8),
    ] {
        let mut phase_bytes = 0usize;
        let mut predicted_signs = 0usize;
        let mut layers: Vec<(&[usize], usize, Option<usize>)> = vec![(
            &all_blocks,
            0,
            Some(crate::scan::PROGRESSIVE_SKELETON_COEFFS),
        )];
        if let Some(packets) = progressive_packets {
            for packet in packets {
                layers.push((packet, crate::scan::PROGRESSIVE_SKELETON_COEFFS, None));
            }
        }

        for (block_indices, start, end_cap) in layers {
            for channel in 0..3 {
                let mut values = Vec::<i16>::new();
                let mut contexts = Vec::<u8>::new();
                let mut parents = Vec::<bool>::new();
                let mut codon_classes = Vec::<u8>::new();
                let mut phase_predictors = Vec::<i8>::new();

                for &block_idx in block_indices {
                    if block_idx >= n_blocks {
                        continue;
                    }
                    let block = &traces[channel].blocks[block_idx];
                    let end = end_cap
                        .map(|cap| block.len().min(cap))
                        .unwrap_or(block.len());
                    if start >= end {
                        continue;
                    }
                    let block_size = block_sizes.get(block_idx).copied().unwrap_or(16);
                    let zigzag = zigzags
                        .iter()
                        .find(|(size, _)| *size == block_size)
                        .map(|(_, order)| order)
                        .unwrap_or(&zigzags[1].1);
                    let mut natural = vec![0i16; block_size * block_size];
                    for (position, &value) in block.iter().enumerate() {
                        natural[zigzag[position]] = value;
                    }
                    let turing = turing_buckets.get(block_idx).copied().unwrap_or(0);

                    for position in start..end {
                        let natural_idx = zigzag[position];
                        let row = natural_idx / block_size;
                        let col = natural_idx % block_size;
                        let parent_row = row / 2;
                        let parent_col = col / 2;
                        let parent_nonzero = (parent_row != 0 || parent_col != 0)
                            && natural[parent_row * block_size + parent_col] != 0;
                        let l_value = traces[0].blocks[block_idx]
                            .get(position)
                            .copied()
                            .unwrap_or(0);
                        let c1_value = traces[1].blocks[block_idx]
                            .get(position)
                            .copied()
                            .unwrap_or(0);
                        let predictor = if channel == 2 && c1_value != 0 {
                            i8::from(c1_value < 0)
                        } else {
                            -1
                        };
                        if predictor >= 0 && block[position] != 0 {
                            predicted_signs += 1;
                        }
                        values.push(block[position]);
                        contexts.push(turing);
                        parents.push(parent_nonzero);
                        codon_classes.push(rans::vector_codon_class(
                            quality, channel, l_value, c1_value,
                        ));
                        phase_predictors.push(predictor);
                    }
                }

                phase_bytes += rans::rans_encode_band_v19_phase(
                    &values,
                    &contexts,
                    &parents,
                    &codon_classes,
                    &phase_predictors,
                    initial_p_same,
                    adapt_shift,
                )
                .len();
            }
        }

        let delta = phase_bytes as i64 - production_ac_bytes as i64;
        let delta_pct = if production_ac_bytes == 0 {
            0.0
        } else {
            100.0 * delta as f64 / production_ac_bytes as f64
        };
        eprintln!(
            "[AUREA_PHASE_WITNESS] initial_p_same={initial_p_same} adapt_shift={adapt_shift} production_ac_bytes={production_ac_bytes} phase_ac_bytes={phase_bytes} delta_bytes={delta} delta_pct={delta_pct:.4} predicted_signs={predicted_signs}"
        );
    }

    let Some(blocks) = block_geometries else {
        return;
    };
    let mut index_by_geometry = std::collections::HashMap::with_capacity(n_blocks);
    for (index, &(y, x, size)) in blocks.iter().take(n_blocks).enumerate() {
        index_by_geometry.insert((y, x, size), index);
    }
    let mut detail_rank = vec![usize::MAX; n_blocks];
    if let Some(packets) = progressive_packets {
        let mut rank = 0usize;
        for packet in packets {
            for &block_idx in packet {
                if block_idx < n_blocks {
                    detail_rank[block_idx] = rank;
                    rank += 1;
                }
            }
        }
    }
    let is_causal = |position: usize, predecessor: usize, current: usize| -> bool {
        if position < crate::scan::PROGRESSIVE_SKELETON_COEFFS {
            predecessor < current
        } else {
            detail_rank.get(predecessor).copied().unwrap_or(usize::MAX)
                < detail_rank.get(current).copied().unwrap_or(usize::MAX)
        }
    };

    // Exact byte witness for Fourier phase transport. L and C1 borrow the
    // vertically preceding coefficient with the DCT continuation
    // `(-1)^v`; C2 keeps the stronger intra-codon C1 predictor.
    for &(variant, l_initial_p_same, c1_initial_p_same, spatial_adapt_shift, spatial_mask) in &[
        ("both_p8192_s6", 8192u32, 8192u32, 6u32, 0b11u8),
        ("both_p8192_s8", 8192, 8192, 8, 0b11),
        ("both_p9216_s6", 9216, 9216, 6, 0b11),
        ("both_p9216_s8", 9216, 9216, 8, 0b11),
        ("both_p10240_s6", 10240, 10240, 6, 0b11),
        ("both_p10240_s8", 10240, 10240, 8, 0b11),
        ("l_only", 8192, 9216, 8, 0b01),
        ("c1_only", 8192, 9216, 8, 0b10),
        ("split_priors", 8192, 9216, 8, 0b11),
    ] {
        let mut phase_bytes = 0usize;
        let mut spatial_signs = 0usize;
        let mut cross_signs = 0usize;
        let mut layers: Vec<(&[usize], usize, Option<usize>)> = vec![(
            &all_blocks,
            0,
            Some(crate::scan::PROGRESSIVE_SKELETON_COEFFS),
        )];
        if let Some(packets) = progressive_packets {
            for packet in packets {
                layers.push((packet, crate::scan::PROGRESSIVE_SKELETON_COEFFS, None));
            }
        }

        for (block_indices, start, end_cap) in layers {
            for channel in 0..3 {
                let mut values = Vec::<i16>::new();
                let mut contexts = Vec::<u8>::new();
                let mut parents = Vec::<bool>::new();
                let mut codon_classes = Vec::<u8>::new();
                let mut phase_predictors = Vec::<i8>::new();
                for &block_idx in block_indices {
                    if block_idx >= n_blocks || block_idx >= blocks.len() {
                        continue;
                    }
                    let block = &traces[channel].blocks[block_idx];
                    let end = end_cap
                        .map(|cap| block.len().min(cap))
                        .unwrap_or(block.len());
                    if start >= end {
                        continue;
                    }
                    let block_size = block_sizes.get(block_idx).copied().unwrap_or(16);
                    let zigzag = zigzags
                        .iter()
                        .find(|(size, _)| *size == block_size)
                        .map(|(_, order)| order)
                        .unwrap_or(&zigzags[1].1);
                    let mut natural = vec![0i16; block_size * block_size];
                    for (position, &value) in block.iter().enumerate() {
                        natural[zigzag[position]] = value;
                    }
                    let turing = turing_buckets.get(block_idx).copied().unwrap_or(0);
                    let (y, x, size) = blocks[block_idx];
                    let top_idx = y
                        .checked_sub(size)
                        .and_then(|top_y| index_by_geometry.get(&(top_y, x, size)).copied());

                    for position in start..end {
                        let natural_idx = zigzag[position];
                        let row = natural_idx / block_size;
                        let col = natural_idx % block_size;
                        let parent_row = row / 2;
                        let parent_col = col / 2;
                        let parent_nonzero = (parent_row != 0 || parent_col != 0)
                            && natural[parent_row * block_size + parent_col] != 0;
                        let l_value = traces[0].blocks[block_idx]
                            .get(position)
                            .copied()
                            .unwrap_or(0);
                        let c1_value = traces[1].blocks[block_idx]
                            .get(position)
                            .copied()
                            .unwrap_or(0);
                        let predictor = if channel == 2 && c1_value != 0 {
                            if block[position] != 0 {
                                cross_signs += 1;
                            }
                            i8::from(c1_value < 0)
                        } else if channel < 2 && spatial_mask & (1 << channel) != 0 {
                            top_idx
                                .filter(|&top| is_causal(position, top, block_idx))
                                .and_then(|top| traces[channel].blocks[top].get(position).copied())
                                .filter(|&value| value != 0)
                                .map(|value| {
                                    if block[position] != 0 {
                                        spatial_signs += 1;
                                    }
                                    i8::from((value < 0) ^ (row & 1 != 0))
                                })
                                .unwrap_or(-1)
                        } else {
                            -1
                        };
                        values.push(block[position]);
                        contexts.push(turing);
                        parents.push(parent_nonzero);
                        codon_classes.push(rans::vector_codon_class(
                            quality, channel, l_value, c1_value,
                        ));
                        phase_predictors.push(predictor);
                    }
                }
                let (initial, shift) = if channel == 2 {
                    (rans::PHASE_INITIAL_P_SAME, rans::PHASE_ADAPT_SHIFT)
                } else if channel == 0 {
                    (l_initial_p_same, spatial_adapt_shift)
                } else {
                    (c1_initial_p_same, spatial_adapt_shift)
                };
                phase_bytes += rans::rans_encode_band_v19_phase(
                    &values,
                    &contexts,
                    &parents,
                    &codon_classes,
                    &phase_predictors,
                    initial,
                    shift,
                )
                .len();
            }
        }
        let delta = phase_bytes as i64 - production_ac_bytes as i64;
        let delta_pct = if production_ac_bytes == 0 {
            0.0
        } else {
            100.0 * delta as f64 / production_ac_bytes as f64
        };
        eprintln!(
            "[AUREA_FOURIER_WITNESS] variant={variant} l_initial={l_initial_p_same} c1_initial={c1_initial_p_same} adapt_shift={spatial_adapt_shift} production_ac_bytes={production_ac_bytes} fourier_ac_bytes={phase_bytes} delta_bytes={delta} delta_pct={delta_pct:.4} spatial_signs={spatial_signs} cross_signs={cross_signs}"
        );
    }
}

// ======================================================================
// Cycle 20 — enveloppes amplitude/phase par voisinage spectral
// ======================================================================

/// Télémétrie neutre du voisinage spectral. Ne change pas un octet : elle
/// mesure, sur les symboles quantifiés exacts, ce que le voisinage causal
/// intra-bloc sait déjà d'un coefficient avant qu'on ne le lise.
///
/// Le voisinage est *toujours* causal quel que soit le découpage progressif :
/// dans le zigzag, `left`, `top`, `diag`, `left2`, `top2` et le parent dyadique
/// appartiennent tous à une anti-diagonale antérieure. La marche par bloc
/// ci-dessous est donc exactement celle que verra le décodeur.
fn emit_spectral_envelope_stats(
    traces: &[CodonChannelTrace],
    block_sizes: &[usize],
    tables: &SpectralNeighborTables,
) {
    use crate::spectral_envelope as se;

    if traces.len() != 3 {
        return;
    }
    let names = ["L", "C1", "C2"];
    let class_names = [
        "none",
        "separable",
        "row_keep",
        "row_flip",
        "col_keep",
        "col_flip",
        "pair",
        "parent",
        "left",
        "top",
    ];

    let mut config = se::SpectralEnvelopeConfig::neutral();
    config.phase_mask = se::PHASE_MASK_ALL;
    config.phase_channels = 0b111;

    for channel in 0..3 {
        let mut phase = [[0u64; 2]; se::PHASE_CLASS_COUNT];
        let mut nonzero_signs = 0u64;
        // Support conditionnel par mode d'enveloppe : [mode][bucket][actif]
        let mut amplitude = [[[0u64; 2]; se::AMPLITUDE_STATES]; 4];
        let mut symbols = 0u64;

        for (block_idx, block) in traces[channel].blocks.iter().enumerate() {
            let block_size = block_sizes
                .get(block_idx)
                .copied()
                .unwrap_or(LOT_BLOCK_SIZE);
            let table = tables.table(SpectralNeighborTables::table_class(block_size));
            for position in 0..block.len() {
                let active = usize::from(block[position] != 0);
                symbols += 1;
                for mode in 0..4u8 {
                    let mut mode_config = config;
                    mode_config.amplitude_mode = mode;
                    let descriptor =
                        se::evaluate(table, position, &mode_config, false, |q| block[q]);
                    amplitude[mode as usize][descriptor.amplitude as usize][active] += 1;
                }
                if block[position] == 0 {
                    continue;
                }
                nonzero_signs += 1;
                let descriptor = se::evaluate(table, position, &config, true, |q| block[q]);
                let observed_negative = block[position] < 0;
                let matched = usize::from(descriptor.phase_negative == observed_negative);
                phase[descriptor.phase_class as usize][matched] += 1;
            }
        }

        let mut conditional_bits = 0.0f64;
        let mut covered = 0u64;
        for (class, counts) in phase.iter().enumerate() {
            let total = counts[0] + counts[1];
            if total == 0 {
                continue;
            }
            let bits = entropy_bits(&[counts[0], counts[1]]);
            if class != se::PHASE_NONE as usize {
                conditional_bits += bits;
                covered += total;
            }
            eprintln!(
                "[AUREA_ENVELOPE_STATS] channel={} phase_class={} samples={total} agreement={:.6} residual_bps={:.6} gain_bits={:.1}",
                names[channel],
                class_names[class],
                counts[1] as f64 / total as f64,
                bits / total as f64,
                total as f64 - bits,
            );
        }
        eprintln!(
            "[AUREA_ENVELOPE_STATS] channel={} phase_total signs={nonzero_signs} predicted={covered} coverage={:.6} raw_gain_bits={:.1} raw_gain_bps={:.6}",
            names[channel],
            covered as f64 / nonzero_signs.max(1) as f64,
            covered as f64 - conditional_bits,
            (covered as f64 - conditional_bits) / nonzero_signs.max(1) as f64,
        );

        let mode_names = ["parent_v16", "parent_x_local", "magnitude", "support"];
        for (mode, buckets) in amplitude.iter().enumerate() {
            let marginal = [
                buckets.iter().map(|counts| counts[0]).sum::<u64>(),
                buckets.iter().map(|counts| counts[1]).sum::<u64>(),
            ];
            let joint = buckets
                .iter()
                .map(|counts| entropy_bits(&[counts[0], counts[1]]))
                .sum::<f64>();
            let gain = (entropy_bits(&marginal) - joint).max(0.0);
            let occupancy: Vec<String> = buckets
                .iter()
                .map(|counts| {
                    let total = counts[0] + counts[1];
                    format!("{:.4}", counts[1] as f64 / total.max(1) as f64)
                })
                .collect();
            eprintln!(
                "[AUREA_ENVELOPE_STATS] channel={} amplitude={} symbols={symbols} support_gain_bits={gain:.1} support_gain_bps={:.6} p_active_by_bucket=[{}]",
                names[channel],
                mode_names[mode],
                gain / symbols.max(1) as f64,
                occupancy.join(","),
            );
        }
    }
}

/// Cycle 21, piste 2 — l'enveloppe de Lphi aide-t-elle celle des chromas ?
///
/// Le codon v17 donne déjà à C1 la classe de magnitude de Lphi *à la même
/// position*, et à C2 le maximum de Lphi et C1. La question ouverte est de
/// savoir si l'enveloppe de **voisinage** de Lphi ajoute quoi que ce soit à ce
/// que le codec sait déjà. Le risque de doublon est réel : on le mesure au lieu
/// de le supposer, en comparant trois conditionnements emboîtés sur les mêmes
/// symboles quantifiés. Seul l'écart entre le deuxième et le troisième compte.
fn emit_cross_channel_envelope_stats(
    traces: &[CodonChannelTrace],
    block_sizes: &[usize],
    tables: &SpectralNeighborTables,
    quality: u8,
) {
    use crate::spectral_envelope as se;

    if traces.len() != 3 {
        return;
    }
    let names = ["L", "C1", "C2"];
    let mut config = se::SpectralEnvelopeConfig::neutral();
    config.amplitude_mode = se::AMPLITUDE_MAGNITUDE;

    for channel in 1..3usize {
        // [own envelope][codon class][Lphi envelope][active]
        let mut joint = [[[[0u64; 2]; se::AMPLITUDE_STATES]; 4]; se::AMPLITUDE_STATES];
        let mut symbols = 0u64;

        for (block_idx, block) in traces[channel].blocks.iter().enumerate() {
            let block_size = block_sizes
                .get(block_idx)
                .copied()
                .unwrap_or(LOT_BLOCK_SIZE);
            let table = tables.table(SpectralNeighborTables::table_class(block_size));
            let luma = &traces[0].blocks[block_idx];
            for position in 0..block.len() {
                let own = se::evaluate(table, position, &config, false, |q| block[q]).amplitude;
                let luma_envelope = se::evaluate(table, position, &config, false, |q| {
                    luma.get(q).copied().unwrap_or(0)
                })
                .amplitude;
                let l_value = luma.get(position).copied().unwrap_or(0);
                let c1_value = traces[1].blocks[block_idx]
                    .get(position)
                    .copied()
                    .unwrap_or(0);
                let codon = rans::vector_codon_class(quality, channel, l_value, c1_value) as usize;
                let active = usize::from(block[position] != 0);
                joint[own as usize][codon][luma_envelope as usize][active] += 1;
                symbols += 1;
            }
        }

        // Conditional entropies of three nested conditionings.
        let mut own_only = [[0u64; 2]; se::AMPLITUDE_STATES];
        let mut own_codon = [[[0u64; 2]; 4]; se::AMPLITUDE_STATES];
        for own in 0..se::AMPLITUDE_STATES {
            for codon in 0..4 {
                for luma_envelope in 0..se::AMPLITUDE_STATES {
                    for active in 0..2 {
                        let count = joint[own][codon][luma_envelope][active];
                        own_only[own][active] += count;
                        own_codon[own][codon][active] += count;
                    }
                }
            }
        }

        let bits_own = own_only
            .iter()
            .map(|counts| entropy_bits(&counts[..]))
            .sum::<f64>();
        let bits_own_codon = own_codon
            .iter()
            .flat_map(|row| row.iter())
            .map(|counts| entropy_bits(&counts[..]))
            .sum::<f64>();
        let bits_full = joint
            .iter()
            .flat_map(|row| row.iter())
            .flat_map(|cell| cell.iter())
            .map(|counts| entropy_bits(&counts[..]))
            .sum::<f64>();

        let scale = symbols.max(1) as f64;
        eprintln!(
            "[AUREA_CROSS_ENVELOPE] channel={} symbols={symbols} bps_own={:.6} bps_own_codon={:.6} bps_full={:.6} codon_gain_bps={:.6} luma_envelope_gain_bps={:.6} luma_envelope_gain_bits={:.1}",
            names[channel],
            bits_own / scale,
            bits_own_codon / scale,
            bits_full / scale,
            (bits_own - bits_own_codon) / scale,
            (bits_own_codon - bits_full) / scale,
            bits_own_codon - bits_full,
        );
    }
}

/// Une variante de laboratoire du cycle 20, nommée pour le journal.
struct EnvelopeVariant {
    label: &'static str,
    config: crate::spectral_envelope::SpectralEnvelopeConfig,
}

/// Témoin octet-exact : réencode l'AC entier sous plusieurs configurations et
/// compare aux octets réellement produits. La première variante est neutre et
/// doit reproduire `production_ac_bytes` à l'octet près — c'est l'autotest de
/// symétrie du harnais lui-même.
#[allow(clippy::too_many_arguments)]
fn emit_spectral_envelope_witness(
    traces: &[CodonChannelTrace],
    block_sizes: &[usize],
    block_geometries: Option<&[(usize, usize, usize)]>,
    progressive_packets: Option<&Vec<Vec<usize>>>,
    turing_buckets: &[u8],
    quality: u8,
    tables: &SpectralNeighborTables,
    production_ac_bytes: usize,
) {
    use crate::spectral_envelope as se;

    if traces.len() != 3 {
        return;
    }
    let n_blocks = traces
        .iter()
        .map(|trace| trace.blocks.len())
        .min()
        .unwrap_or(0);
    let all_blocks: Vec<usize> = (0..n_blocks).collect();
    let zigzags = [
        (8usize, ac_zigzag_order(8)),
        (16usize, ac_zigzag_order(16)),
        (32usize, ac_zigzag_order(32)),
    ];

    // Prédicteurs historiques v19/v20, reproduits à l'identique.
    let mut detail_rank = vec![usize::MAX; n_blocks];
    if let Some(packets) = progressive_packets {
        let mut rank = 0usize;
        for packet in packets {
            for &block_idx in packet {
                if block_idx < n_blocks {
                    detail_rank[block_idx] = rank;
                    rank += 1;
                }
            }
        }
    }
    let top_blocks: Vec<Option<usize>> = match block_geometries {
        Some(blocks) => {
            let mut by_geometry = std::collections::HashMap::with_capacity(n_blocks);
            for (index, &(y, x, size)) in blocks.iter().take(n_blocks).enumerate() {
                by_geometry.insert((y, x, size), index);
            }
            blocks
                .iter()
                .take(n_blocks)
                .map(|&(y, x, size)| {
                    y.checked_sub(size)
                        .and_then(|top_y| by_geometry.get(&(top_y, x, size)).copied())
                })
                .collect()
        }
        None => vec![None; n_blocks],
    };
    let is_causal = |position: usize, predecessor: usize, current: usize| -> bool {
        if position < crate::scan::PROGRESSIVE_SKELETON_COEFFS {
            predecessor < current
        } else {
            detail_rank.get(predecessor).copied().unwrap_or(usize::MAX)
                < detail_rank.get(current).copied().unwrap_or(usize::MAX)
        }
    };

    // (label, amplitude, split_pm1, phase mask, shift, prior, coupling)
    //
    // `promoted_selftest` re-derives the configuration the encoder actually
    // used and must land on a delta of exactly zero. It is the self-test of the
    // harness — every other row is only meaningful because that one is null.
    // `neutral_v20` is the A/B reference: its delta is the cycle's whole gain,
    // read with the sign reversed.
    const COHERENT: u16 = se::PHASE_MASK_COHERENT;
    const SEPARABLE: u16 = se::PHASE_MASK_SEPARABLE;
    const ALL: u16 = se::PHASE_MASK_ALL;
    const MAG: u8 = se::AMPLITUDE_MAGNITUDE;
    const SCALED: u8 = se::AMPLITUDE_MAGNITUDE_SCALED;
    const MEMORY: u8 = se::PHASE_CONTEXT_SEPARABLE_MEMORY;
    const TURING: u8 = se::PHASE_CONTEXT_SEPARABLE_TURING;
    const SEP_ENERGY: u8 = se::PHASE_CONTEXT_SEPARABLE_ENERGY;
    const X_MAX: u8 = se::CROSS_CODON_MAX;
    const X_ENV: u8 = se::CROSS_CODON_ENVELOPE;
    const X_SUM: u8 = se::CROSS_CODON_SUM;
    const FOREIGN: u8 = se::MIX_INPUT_FOREIGN;
    const SCALED_IN: u8 = se::MIX_INPUT_SCALED;
    const BAND: u8 = se::MIX_INPUT_BAND;
    const SIZE_IN: u8 = se::MIX_INPUT_SIZE;
    const SUPPORT_IN: u8 = se::MIX_INPUT_SUPPORT;
    const TRIO: u8 = FOREIGN | SCALED_IN | BAND;
    // (label, mix inputs, mix channels, mix sign)
    // Everything else is the promoted v23 profile.
    const SWEEP: &[(&str, u8, u8, bool)] = &[
        ("v23_promoted", TRIO, 0b111, false),
        // Lead — a fifth and sixth opinion, both refused as contexts before.
        ("in_size", TRIO | SIZE_IN, 0b111, false),
        ("in_support", TRIO | SUPPORT_IN, 0b111, false),
        ("in_size_support", TRIO | SIZE_IN | SUPPORT_IN, 0b111, false),
        // Lead — mix the sign, the last unmixed decision.
        ("sign", TRIO, 0b111, true),
        ("support_sign", TRIO | SUPPORT_IN, 0b111, true),
        (
            "sign_size_support",
            TRIO | SIZE_IN | SUPPORT_IN,
            0b111,
            true,
        ),
        // Lead — isolate what the mix is worth on Lphi.
        ("chroma_only", TRIO, 0b110, false),
        ("luma_only", TRIO, 0b001, false),
        ("chroma_only_full", TRIO | SIZE_IN | SUPPORT_IN, 0b110, true),
    ];

    let mut variants: Vec<EnvelopeVariant> = vec![EnvelopeVariant {
        label: "promoted_selftest",
        config: *crate::calibration::TUNABLE_SPECTRAL_ENVELOPE,
    }];
    variants.extend(
        SWEEP
            .iter()
            .map(|&(label, mix_inputs, mix_channels, mix_sign)| {
                let mut config = se::SpectralEnvelopeConfig::promoted();
                config.mix_inputs = mix_inputs;
                config.mix_channels = mix_channels;
                config.mix_sign = mix_sign;
                EnvelopeVariant { label, config }
            }),
    );

    for variant in &variants {
        let mut envelope_bytes = 0usize;
        let mut layers: Vec<(&[usize], usize, Option<usize>)> = vec![(
            &all_blocks,
            0,
            Some(crate::scan::PROGRESSIVE_SKELETON_COEFFS),
        )];
        if let Some(packets) = progressive_packets {
            for packet in packets {
                layers.push((packet, crate::scan::PROGRESSIVE_SKELETON_COEFFS, None));
            }
        }

        for (block_indices, layer_start, end_cap) in layers {
            for channel in 0..3usize {
                let mut values = Vec::<i16>::new();
                let mut contexts = Vec::<u8>::new();
                let mut codon_classes = Vec::<u8>::new();
                let mut legacy_phase = Vec::<i8>::new();
                let mut foreign_amplitudes = Vec::<u8>::new();
                let mut segments = Vec::<SpectralSegment>::new();

                for &block_idx in block_indices {
                    if block_idx >= n_blocks {
                        continue;
                    }
                    let block = &traces[channel].blocks[block_idx];
                    let end = end_cap
                        .map(|cap| block.len().min(cap))
                        .unwrap_or(block.len());
                    let start = layer_start.min(end);
                    if start >= end {
                        continue;
                    }
                    let block_size = block_sizes
                        .get(block_idx)
                        .copied()
                        .unwrap_or(LOT_BLOCK_SIZE);
                    let zigzag = zigzags
                        .iter()
                        .find(|(size, _)| *size == block_size)
                        .map(|(_, order)| order)
                        .unwrap_or(&zigzags[1].1);
                    let table_class = SpectralNeighborTables::table_class(block_size);
                    segments.push(if layer_start == 0 {
                        SpectralSegment::root(values.len(), end - start, table_class)
                    } else {
                        SpectralSegment::detail(
                            values.len(),
                            start,
                            end - start,
                            table_class,
                            &block[..start],
                        )
                    });
                    let turing = turing_buckets.get(block_idx).copied().unwrap_or(0);

                    for position in start..end {
                        let l_value = traces[0].blocks[block_idx]
                            .get(position)
                            .copied()
                            .unwrap_or(0);
                        let c1_value = traces[1].blocks[block_idx]
                            .get(position)
                            .copied()
                            .unwrap_or(0);
                        let predictor = if channel == 2 && c1_value != 0 {
                            i8::from(c1_value < 0)
                        } else if channel == 1 {
                            top_blocks
                                .get(block_idx)
                                .copied()
                                .flatten()
                                .filter(|&top| is_causal(position, top, block_idx))
                                .and_then(|top| traces[1].blocks[top].get(position).copied())
                                .filter(|&value| value != 0)
                                .map(|value| {
                                    let row = zigzag[position] / block_size;
                                    i8::from((value < 0) ^ (row & 1 != 0))
                                })
                                .unwrap_or(-1)
                        } else {
                            -1
                        };
                        values.push(block[position]);
                        contexts.push(turing);
                        let base = rans::vector_codon_class(quality, channel, l_value, c1_value);
                        let needs_foreign = channel > 0
                            && (variant.config.cross_codon_mode
                                != crate::spectral_envelope::CROSS_CODON_NONE
                                || variant.config.mix_rate > 0);
                        let foreign = if needs_foreign {
                            let luma = &traces[0].blocks[block_idx];
                            crate::spectral_envelope::foreign_amplitude(
                                tables.table(SpectralNeighborTables::table_class(block_size)),
                                position,
                                |q| luma.get(q).copied().unwrap_or(0),
                            )
                        } else {
                            0
                        };
                        if variant.config.mix_rate > 0 {
                            foreign_amplitudes.push(foreign);
                        }
                        codon_classes.push(crate::spectral_envelope::combine_codon_class(
                            variant.config.cross_codon_mode,
                            base,
                            foreign,
                        ));
                        legacy_phase.push(predictor);
                    }
                }

                let (phase_initial, phase_shift) = if channel == 1 {
                    (
                        rans::FOURIER_PHASE_INITIAL_P_SAME,
                        rans::FOURIER_PHASE_ADAPT_SHIFT,
                    )
                } else {
                    (rans::PHASE_INITIAL_P_SAME, rans::PHASE_ADAPT_SHIFT)
                };
                envelope_bytes += rans::rans_encode_band_v21_envelope(
                    &values,
                    &contexts,
                    &codon_classes,
                    &legacy_phase,
                    &foreign_amplitudes,
                    &segments,
                    tables,
                    &variant.config,
                    channel,
                    phase_initial,
                    phase_shift,
                    *calibration::TUNABLE_ACTIVITY_CONTEXT,
                    &[],
                    &mut rans::EnvelopeTables::default(),
                )
                .len();
            }
        }

        let delta = envelope_bytes as i64 - production_ac_bytes as i64;
        let delta_pct = if production_ac_bytes == 0 {
            0.0
        } else {
            100.0 * delta as f64 / production_ac_bytes as f64
        };
        eprintln!(
            "[AUREA_ENVELOPE_WITNESS] variant={} mix_rate={} mix_inputs={} mix_channels={} mix_sign={} pm1={} apm={} shift={} production_ac_bytes={production_ac_bytes} envelope_ac_bytes={envelope_bytes} delta_bytes={delta} delta_pct={delta_pct:.4}",
            variant.label,
            variant.config.mix_rate,
            variant.config.mix_inputs,
            variant.config.mix_channels,
            u8::from(variant.config.mix_sign),
            u8::from(variant.config.mix_pm1),
            variant.config.apm_rate,
            variant.config.phase_adapt_shift,
        );
    }
}

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
            MatchKind::Residual => 0b01,
            MatchKind::Surprise => 0b10,
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
            MatchKind::Surprise => match &m.original {
                geometric::Primitive::Segment {
                    x1,
                    y1,
                    x2,
                    y2,
                    amplitude,
                    phase,
                } => {
                    buf.push(0x00);
                    buf.extend_from_slice(&x1.to_le_bytes());
                    buf.extend_from_slice(&y1.to_le_bytes());
                    buf.extend_from_slice(&x2.to_le_bytes());
                    buf.extend_from_slice(&y2.to_le_bytes());
                    buf.push(*amplitude as u8);
                    buf.push(*phase as u8);
                }
                geometric::Primitive::Arc {
                    cx,
                    cy,
                    radius,
                    theta_start,
                    theta_end,
                    amplitude,
                    phase,
                } => {
                    buf.push(0x01);
                    buf.extend_from_slice(&cx.to_le_bytes());
                    buf.extend_from_slice(&cy.to_le_bytes());
                    buf.extend_from_slice(&radius.to_le_bytes());
                    buf.push(*theta_start as u8);
                    buf.push(*theta_end as u8);
                    buf.push(*amplitude as u8);
                    buf.push(*phase as u8);
                }
            },
        }
    }

    buf
}

pub fn encode_aur2_v13(
    rgb: &[u8],
    width: usize,
    height: usize,
    params: &AureaEncoderParams,
) -> Result<AureaEncoderResult, Box<dyn std::error::Error>> {
    if width == 0 || height == 0 {
        return Err("image dimensions must be non-zero".into());
    }
    if width > u16::MAX as usize || height > u16::MAX as usize {
        return Err("AUR2 dimensions are limited to 65535x65535".into());
    }
    if !(1..=100).contains(&params.quality) {
        return Err("quality must be in the range 1..=100".into());
    }
    let n = width
        .checked_mul(height)
        .ok_or("image dimensions overflow")?;
    let expected_rgb = n.checked_mul(3).ok_or("RGB buffer size overflow")?;
    if rgb.len() != expected_rgb {
        return Err(format!(
            "RGB buffer length mismatch: expected {expected_rgb}, got {}",
            rgb.len()
        )
        .into());
    }

    let t_gct = crate::timing::stage("enc:gct");
    let (mut l_ch, c1_ch, c2_ch) = color::golden_rotate_forward(rgb, n);
    drop(t_gct);

    {
        let _t = crate::timing::stage("enc:ptf");
        use crate::golden::PTF_GAMMA;
        use rayon::prelude::*;
        let inv255 = 1.0 / 255.0;
        l_ch.par_iter_mut().for_each(|v| {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA);
        });
    }

    let t_scene = crate::timing::stage("enc:scene");
    let scene_profile = scene_analysis::quick_scene_classify(&l_ch, height, width);
    let (step_factor, _deblock_hint) = scene_analysis::encoder_scene_adjust(&scene_profile);
    let scene_type_byte = match scene_profile.scene_type {
        scene_analysis::SceneType::Flat => 0u8,
        scene_analysis::SceneType::Architectural => 1,
        scene_analysis::SceneType::Perspective => 2,
        scene_analysis::SceneType::Organic => 3,
        scene_analysis::SceneType::Mixed => 4,
    };
    drop(t_scene);

    // v13: 4:4:4 — no subsampling, avoid cloning full-res chroma planes
    let (c1_sub, c1_h, c1_w) = (&c1_ch, height, width);
    let (c2_sub, c2_h, c2_w) = (&c2_ch, height, width);

    // Quantize the experimental strength before doing any codec work. Both
    // sides therefore use the exact same self-described value.
    let chroma_breathing_code = (*calibration::TUNABLE_CHROMA_BREATHING * 255.0).round() as u8;
    let chroma_breathing_strength = chroma_breathing_code as f64 / 255.0;

    // Bright compensation (« contre-don de Weber ») strength, quantized to
    // 5 bits carried in the scene-type byte's upper bits. Legacy v13 files
    // have zero there, which is exactly the neutral state.
    let bright_code = (*calibration::TUNABLE_BRIGHT_COMPENSATION * 31.0).round() as u8;
    let bright_strength = bright_code as f64 / 31.0;

    let mut flags = DEFAULT_V3_FLAGS;
    if params.lot_topology.uses_variable_blocks() {
        flags |= FLAG_VARIABLE_BLOCKS;
    }
    if params.lot_topology.uses_overlap() {
        flags |= FLAG_LOT_OVERLAP;
    }
    flags |= FLAG_BAYESIAN_HIERARCHY;
    flags |= FLAG_CFL;
    match *calibration::TUNABLE_SPIRAL_SCAN_MODE {
        1 => flags |= FLAG_PHYLLOTACTIC_SCAN,
        2 => flags |= FLAG_SPIRAL_ARC_SCAN,
        _ => {}
    }
    if *calibration::TUNABLE_SPECTRAL_SCAN_MODE == 1 {
        flags |= FLAG_FOLDED_SPECTRAL_SCAN;
    }
    if *calibration::TUNABLE_PROGRESSIVE_PACKETS {
        flags |= FLAG_PROGRESSIVE_PACKETS;
    }
    if chroma_breathing_code != 0 {
        flags |= FLAG_CHROMA_BREATHING;
    }
    let use_overlap = flags & FLAG_LOT_OVERLAP != 0;
    let use_variable = flags & FLAG_VARIABLE_BLOCKS != 0;
    let use_cfl = flags & FLAG_CFL != 0;
    let use_phyllotactic_scan = flags & FLAG_PHYLLOTACTIC_SCAN != 0;
    let use_spiral_arc_scan = flags & FLAG_SPIRAL_ARC_SCAN != 0;
    let use_progressive_packets = flags & FLAG_PROGRESSIVE_PACKETS != 0;
    if use_progressive_packets && (use_phyllotactic_scan || use_spiral_arc_scan) {
        return Err("progressive packets cannot be combined with a spatial AC scan".into());
    }

    let (l_dc, l_ac, l_gh, l_gw, l_block_map, l_blocks);
    let (c1_dc, c1_ac, c1_gh, c1_gw);
    let (c2_dc, c2_ac, c2_gh, c2_gw);

    if use_variable {
        let t_classify = crate::timing::stage("enc:block_classify");
        let detail_step_pre = auto_detail_step(&l_ch, height, width, params.quality)
            * step_factor
            * *calibration::TUNABLE_STEP_SCALE;
        let (mut size_grid, bgh, bgw) = lot::classify_blocks(&l_ch, height, width, detail_step_pre);
        if let Some(forced) = params.lot_topology.forced_cell_size() {
            // M1: keep the whole adaptive machinery, only remove the choice.
            // Same grammar, same map, one symbol — so any difference against
            // M0 is the size adaptation and nothing else.
            size_grid.iter_mut().for_each(|cell| *cell = forced);
        }
        drop(t_classify);
        let _t_lot = crate::timing::stage("enc:lot_forward");

        l_block_map = Some((size_grid.clone(), bgh, bgw));

        let (dc, ac, blocks) =
            lot::lot_analyze_variable(&l_ch, height, width, &size_grid, bgh, bgw, use_overlap);
        l_blocks = Some(blocks.clone());
        l_dc = dc;
        l_ac = ac;
        l_gh = blocks.len();
        l_gw = 1;

        let (dc1, ac1, _) =
            lot::lot_analyze_variable(&c1_sub, c1_h, c1_w, &size_grid, bgh, bgw, use_overlap);
        c1_dc = dc1;
        c1_ac = ac1;
        c1_gh = l_gh;
        c1_gw = 1;

        let (dc2, ac2, _) =
            lot::lot_analyze_variable(&c2_sub, c2_h, c2_w, &size_grid, bgh, bgw, use_overlap);
        c2_dc = dc2;
        c2_ac = ac2;
        c2_gh = l_gh;
        c2_gw = 1;
    } else {
        let _t_lot = crate::timing::stage("enc:lot_forward");
        l_block_map = None;
        l_blocks = None;
        let r = lot::lot_analyze_image(&l_ch, height, width, LOT_BLOCK_SIZE, use_overlap);
        l_dc = r.0;
        l_ac = r.1;
        l_gh = r.2;
        l_gw = r.3;
        let r1 = lot::lot_analyze_image(&c1_sub, c1_h, c1_w, LOT_BLOCK_SIZE, use_overlap);
        c1_dc = r1.0;
        c1_ac = r1.1;
        c1_gh = r1.2;
        c1_gw = r1.3;
        let r2 = lot::lot_analyze_image(&c2_sub, c2_h, c2_w, LOT_BLOCK_SIZE, use_overlap);
        c2_dc = r2.0;
        c2_ac = r2.1;
        c2_gh = r2.2;
        c2_gw = r2.3;
    }

    // The header carries detail_step as f32 and the decoder derives every
    // step (dc_step included) from the rounded value. Roundtrip here so both
    // sides quantize and rebuild the zero-bit fields from bit-identical
    // numbers — the f64 original drifts ~1e-8 relative, enough to flip
    // near-tied spiral-arc ordering on degenerate content (see
    // `normalize_dc` for the same doctrine on the DC range).
    let detail_step = (auto_detail_step(&l_ch, height, width, params.quality)
        * step_factor
        * *calibration::TUNABLE_STEP_SCALE) as f32 as f64;
    let dc_step = (detail_step * 0.1).max(0.2);

    let lot_global_factor = if flags & FLAG_QUALITY_ADAPTIVE != 0 {
        calibration::lot_factor_for_quality(params.quality)
    } else {
        3.8
    };
    let dead_zone = *calibration::TUNABLE_DEAD_ZONE_SCALE
        * if flags & FLAG_QUALITY_ADAPTIVE != 0 {
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
    // Lower 3 bits: scene type (0-4). Upper 5 bits: bright-compensation code.
    body.push((scene_type_byte & 0x07) | (bright_code << 3));
    body.push(chroma_breathing_code);

    let turing_hdr = TuringHeader::default_params();
    let turing_hdr_bytes = turing_hdr.write();
    body.extend_from_slice(&turing_hdr_bytes);

    if let Some((ref size_grid, bgh, bgw)) = l_block_map {
        body.extend_from_slice(&(bgh as u16).to_le_bytes());
        body.extend_from_slice(&(bgw as u16).to_le_bytes());
        // rANS compress the block map (codes 0/1/2 — highly compressible when >90% are 16)
        let map_symbols: Vec<i16> = size_grid
            .iter()
            .map(|&s| match s {
                8 => 0i16,
                16 => 1,
                32 => 2,
                _ => 1,
            })
            .collect();
        let map_ctx = vec![0u8; map_symbols.len()];
        let encoded_map = rans::rans_encode_band_v12(&map_symbols, &map_ctx);
        if encoded_map.len() > u16::MAX as usize {
            return Err("compressed block map exceeds the AUR2 u16 section limit".into());
        }
        body.extend_from_slice(&(encoded_map.len() as u16).to_le_bytes());
        body.extend_from_slice(&encoded_map);
    }

    let use_folded_spectral_scan = flags & FLAG_FOLDED_SPECTRAL_SCAN != 0;
    let zz_16 = if use_folded_spectral_scan {
        crate::scan::fixed_pitch_folded_spiral_order(16)
    } else {
        ac_zigzag_order(16)
    };
    let zz_8 = if use_folded_spectral_scan {
        crate::scan::fixed_pitch_folded_spiral_order(8)
    } else {
        ac_zigzag_order(8)
    };
    let zz_32 = if use_folded_spectral_scan {
        crate::scan::fixed_pitch_folded_spiral_order(32)
    } else {
        ac_zigzag_order(32)
    };
    let spectral_parents_8 = spectral_parent_positions(&zz_8, 8);
    let spectral_parents_16 = spectral_parent_positions(&zz_16, 16);
    let spectral_parents_32 = spectral_parent_positions(&zz_32, 32);

    // Cycle 20: causal amplitude/phase neighbourhood inside each transform.
    // Built once for the three sizes; neutral configurations never touch it.
    let spectral_config = *calibration::TUNABLE_SPECTRAL_ENVELOPE;
    let use_spectral_envelope = !spectral_config.is_neutral();

    // Each grammar marks its own streams. The spectral envelope changes the AC
    // contexts (v21) and logistic mixing changes how P(zero) is formed at all
    // (v22); neither may ever be handed to an earlier reader.
    let base_version: u8 = if spectral_config.mix_rate > 0 {
        // v22 mixes one secondary opinion into P(zero); v23 mixes several,
        // extends the mix to P(|x|=1) and adds the APM refinement.
        let se = crate::spectral_envelope::SpectralEnvelopeConfig::promoted();
        if spectral_config.mix_inputs == crate::spectral_envelope::MIX_INPUT_FOREIGN
            && !spectral_config.mix_pm1
            && spectral_config.apm_rate == 0
            && !spectral_config.mix_sign
        {
            22
        } else if spectral_config.mix_inputs == se.mix_inputs
            && spectral_config.mix_sign == se.mix_sign
        {
            24
        } else {
            23
        }
    } else if use_spectral_envelope {
        21
    } else {
        20
    };

    // Cycle 27 — v25 keys the AC contexts on intra-block activity (the EOB
    // position) instead of the Turing bucket. It rides on the promoted v24
    // grammar only, so the version ladder stays truthful: an experimental
    // spectral configuration keeps its own number and the older context.
    let use_activity_context = *calibration::TUNABLE_ACTIVITY_CONTEXT && base_version == 24;

    // Cycle 27 — v26 opens each detail layer on the echo of its own ping.
    // Layer 0 delivers every block's first coefficients and is fully decoded
    // before any detail layer, yet each layer starts a fresh rANS stream with
    // flat tables: a block's detail coefficients were coded knowing nothing
    // about the low frequencies already sent for that same block.
    let use_echo_context = use_activity_context && *calibration::TUNABLE_ECHO_CONTEXT;

    // Cycle 27 — v27 lets each detail layer re-enter the channel layer 0 has
    // already ionized, instead of breaking down flat air again. Layer 0 is
    // always present, so carrying its converged tables forward is deterministic
    // and safe under progressive truncation.
    let use_channel_memory = use_echo_context && *calibration::TUNABLE_CHANNEL_MEMORY;

    // Cycle 30 — v28 relieves the luma pivot and the chroma exchange slope on
    // the diffuse-dark field (kryte-class images). The reliefs change the
    // quantization steps, so a v28 stream may never be handed to an earlier
    // reader; conversely the witnesses `AUREA_LUMA_PIVOT_RELIEF=0` and
    // `AUREA_SPECTRAL_CHROMA_RELIEF=0` emit a v27 stream, bit-exact with the
    // cycle-27 profile.
    let use_diffuse_relief = use_channel_memory
        && (*calibration::TUNABLE_LUMA_PIVOT_RELIEF > 0.0
            || *calibration::TUNABLE_SPECTRAL_CHROMA_RELIEF > 0.0);
    // Cycle 31 — v29 adds the perceptual texture law on top of the v28
    // reliefs; cycle 32 — v30 adds the EOB-driven luma step field. The
    // rupture doctrine applies: a v30 stream is never handed to an earlier
    // reader, and no older emission path is maintained beyond the witnesses.
    // v33: identical bitstream to v32 — the bump carries the decoder's
    // strict gas-deblock semantics (periodic-content immunity, cycle 39).
    let version = if use_diffuse_relief {
        33
    } else if use_channel_memory {
        27
    } else if use_echo_context {
        26
    } else if use_activity_context {
        25
    } else {
        base_version
    };
    let spectral_tables = SpectralNeighborTables::new(&zz_8, &zz_16, &zz_32);

    let t_rans_dc = crate::timing::stage("enc:rans_dc");
    let dc_step_clamped = dc_step.max(0.1);
    struct DcChannelData {
        dc_encoded: Vec<u8>,
        dc_reconstructed: Vec<f64>,
    }
    let mut dc_data: Vec<DcChannelData> = Vec::with_capacity(3);

    let dc_norms: [&[f64]; 3] = [&l_dc_norm, &c1_dc_norm, &c2_dc_norm];
    let dc_ranges_raw = [
        (l_dc_min, l_dc_max),
        (c1_dc_min, c1_dc_max),
        (c2_dc_min, c2_dc_max),
    ];
    let grid_dims = [(l_gh, l_gw), (c1_gh, c1_gw), (c2_gh, c2_gw)];

    for i in 0..3 {
        let (gh, gw) = grid_dims[i];
        let dc_norm_ch = dc_norms[i];
        let (dc_min_ch, dc_max_ch) = dc_ranges_raw[i];
        let range = (dc_max_ch - dc_min_ch).max(1e-6);

        // The dead zone belongs to the AC quantizer: AC coefficients are
        // centred and sparse, so pulling small ones to zero is a win. The
        // normalized DC is neither — it is strictly non-negative and dense, so
        // the dead zone never zeroes anything and instead shaves every block
        // downward by `dead_zone * dc_step`, darkening the whole image.
        let dc_dead_zone = dead_zone * *calibration::TUNABLE_DC_DEAD_ZONE;

        // Cycle 25 — Jensen compensation, luminance only: the PTF is applied to
        // Lphi alone, so only its inverse is nonlinear and only its blocks are
        // displaced by a change of intra-block variance.
        let jensen = *calibration::TUNABLE_DC_JENSEN;
        let jensen_shift: Option<Vec<f64>> = (i == 0 && jensen > 0.0).then(|| {
            let curvature = 1.0 / crate::golden::PTF_GAMMA - 1.0;
            let nominal_step = (detail_step * lot_global_factor).max(1e-6);
            // A dead-zone quantizer zeroes a coefficient exactly when
            // |c| < (0.5 + dead_zone) * step.
            let threshold = (0.5 + dead_zone) * nominal_step;
            let noise = nominal_step * nominal_step / 12.0;
            l_ac.iter()
                .zip(&l_dc)
                .map(|(block, &dc_raw)| {
                    let mut lost = 0.0f64;
                    let mut added = 0.0f64;
                    for &c in block {
                        if c.abs() < threshold {
                            lost += c * c;
                        } else {
                            added += noise;
                        }
                    }
                    // `dc_raw` carries the PTF-domain block mean scaled by the
                    // LOT's DC gain; the calibration constant absorbs that gain
                    // along with the Parseval factor.
                    let mean_ptf = (dc_raw.abs() / 256.0).max(1.0);
                    jensen * 0.5 * curvature * (lost - added) / mean_ptf
                })
                .collect()
        });

        let dc_q: Vec<i16> = dc_norm_ch
            .iter()
            .enumerate()
            .map(|(block_idx, &v0)| {
                let v = match jensen_shift {
                    // The shift is derived in raw DC units; `dc_norm_ch` lives
                    // on the normalized 0..255 ladder, hence the rescaling.
                    Some(ref shift) => v0 + shift[block_idx] * 255.0 / range,
                    None => v0,
                };
                let sign = if v >= 0.0 { 1.0 } else { -1.0 };
                let qv = (v.abs() / dc_step_clamped + 0.5 - dc_dead_zone).floor();
                if qv > 0.0 { (sign * qv) as i16 } else { 0i16 }
            })
            .collect();

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
        } else {
            dc_q.clone()
        };
        // rANS v12 entropy model (Exp-Golomb) for DC.
        let dc_ctx = vec![0u8; dc_for_rans.len()];
        let dc_encoded = rans::rans_encode_band_v12(&dc_for_rans, &dc_ctx);

        let dc_reconstructed: Vec<f64> = dc_q
            .iter()
            .map(|&q| q as f64 * dc_step_clamped * range / 255.0 + dc_min_ch)
            .collect();

        dc_data.push(DcChannelData {
            dc_encoded,
            dc_reconstructed,
        });
    }

    drop(t_rans_dc);
    let t_hier = crate::timing::stage("enc:turing_hierarchy");
    let l_dc_denorm = &dc_data[0].dc_reconstructed;
    let hierarchy_params = HierarchyParams::from(&turing_hdr);
    // v13: project adaptive-block DC values back onto their real 8x8 cell
    // positions before computing any zero-bit spatial predictor.
    let (turing_dc_grid, turing_grid_h, turing_grid_w) =
        if let (Some((_, bgh, bgw)), Some(blocks)) = (&l_block_map, &l_blocks) {
            (
                hierarchy::expand_adaptive_dc_grid(l_dc_denorm, blocks, *bgh, *bgw, use_overlap)
                    .map_err(|e| format!("invalid adaptive layout: {e}"))?,
                *bgh,
                *bgw,
            )
        } else {
            (l_dc_denorm.clone(), l_gh, l_gw)
        };
    let chroma_dc_grids = if let Some(ref blocks) = l_blocks {
        [
            hierarchy::expand_adaptive_dc_grid(
                &dc_data[1].dc_reconstructed,
                blocks,
                turing_grid_h,
                turing_grid_w,
                use_overlap,
            )
            .map_err(|e| format!("invalid adaptive C1 DC layout: {e}"))?,
            hierarchy::expand_adaptive_dc_grid(
                &dc_data[2].dc_reconstructed,
                blocks,
                turing_grid_h,
                turing_grid_w,
                use_overlap,
            )
            .map_err(|e| format!("invalid adaptive C2 DC layout: {e}"))?,
        ]
    } else {
        [
            dc_data[1].dc_reconstructed.clone(),
            dc_data[2].dc_reconstructed.clone(),
        ]
    };
    if let Some(dir) = std::env::var_os("AUREA_DEBUG_EOB_DUMP") {
        use std::io::Write;
        let path = std::path::Path::new(&dir).join("enc_dcgrid.txt");
        if let Ok(mut f) = std::fs::File::create(path) {
            for &v in &turing_dc_grid {
                let _ = writeln!(f, "{:016x}", v.to_bits());
            }
        }
    }
    let turing_field = hierarchy::compute_level1(
        &turing_dc_grid,
        turing_grid_h,
        turing_grid_w,
        &hierarchy_params,
    );

    let turing_buckets = if let Some(ref blocks) = l_blocks {
        hierarchy::build_adaptive_turing_buckets(&turing_field, blocks, use_overlap)
    } else {
        hierarchy::build_turing_buckets(&turing_field, l_gh, l_gw, 1)
    };
    let block_scan_order: Vec<usize> = if use_phyllotactic_scan || use_spiral_arc_scan {
        if let Some(ref blocks) = l_blocks {
            if use_spiral_arc_scan {
                crate::scan::adaptive_spiral_arc_order(
                    blocks,
                    turing_grid_h,
                    turing_grid_w,
                    use_overlap,
                    &turing_field.inhibitor_gradient,
                )
            } else {
                crate::scan::adaptive_phyllotactic_order(
                    blocks,
                    turing_grid_h,
                    turing_grid_w,
                    use_overlap,
                    &turing_field.inhibitor_gradient,
                )
            }
        } else {
            (0..l_gh * l_gw).collect()
        }
    } else {
        (0..l_gh * l_gw).collect()
    };
    let progressive_packets: Option<Vec<Vec<usize>>> = if use_progressive_packets {
        let spiral_order = if let Some(ref blocks) = l_blocks {
            crate::scan::adaptive_spiral_arc_order(
                blocks,
                turing_grid_h,
                turing_grid_w,
                use_overlap,
                &turing_field.inhibitor_gradient,
            )
        } else {
            (0..l_gh * l_gw).collect()
        };
        if let Some(dir) = std::env::var_os("AUREA_DEBUG_EOB_DUMP") {
            use std::io::Write;
            let path = std::path::Path::new(&dir).join("enc_spiral.txt");
            if let Ok(mut f) = std::fs::File::create(path) {
                for &b in &spiral_order {
                    let _ = writeln!(f, "{b}");
                }
            }
        }
        Some(crate::scan::progressive_spiral_packets(&spiral_order))
    } else {
        None
    };
    let fourier_top_blocks: Vec<Option<usize>> = if let Some(ref blocks) = l_blocks {
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
        vec![None; l_gh * l_gw]
    };
    let mut progressive_block_rank = vec![usize::MAX; fourier_top_blocks.len()];
    if let Some(ref packets) = progressive_packets {
        let mut rank = 0usize;
        for packet in packets {
            for &block_idx in packet {
                if block_idx < progressive_block_rank.len() {
                    progressive_block_rank[block_idx] = rank;
                    rank += 1;
                }
            }
        }
    }
    let local_turing_step_mods = if let Some(ref blocks) = l_blocks {
        hierarchy::sample_scalar_for_blocks(
            &turing_field.step_modulation,
            turing_field.grid_h,
            turing_field.grid_w,
            blocks,
            use_overlap,
        )
    } else {
        turing_field.step_modulation.clone()
    };
    let multiscale_strength = *calibration::TUNABLE_TURING_MULTISCALE;
    let turing_step_mods = if multiscale_strength > 0.0 {
        let wider_params = HierarchyParams {
            sigma_a: hierarchy_params.sigma_a * crate::golden::PHI,
            sigma_ratio: hierarchy_params.sigma_ratio,
            k_step: hierarchy_params.k_step,
            mag_calibration: hierarchy_params.mag_calibration,
        };
        let wider_field =
            hierarchy::compute_level1(&turing_dc_grid, turing_grid_h, turing_grid_w, &wider_params);
        let wider_mods = if let Some(ref blocks) = l_blocks {
            hierarchy::sample_scalar_for_blocks(
                &wider_field.step_modulation,
                wider_field.grid_h,
                wider_field.grid_w,
                blocks,
                use_overlap,
            )
        } else {
            wider_field.step_modulation
        };
        hierarchy::phi_multiscale_step_modulations(
            &local_turing_step_mods,
            &wider_mods,
            multiscale_strength,
        )
    } else {
        local_turing_step_mods
    };
    let pivoted_turing_mods: Vec<f64> = turing_step_mods
        .iter()
        .map(|&raw_modulation| psychovisual_turing_pivot(raw_modulation, params.quality))
        .collect();
    let turing_block_sizes: Vec<usize> = if let Some(ref blocks) = l_blocks {
        blocks
            .iter()
            .map(|&(_, _, block_size)| block_size)
            .collect()
    } else {
        vec![LOT_BLOCK_SIZE; pivoted_turing_mods.len()]
    };
    let turing_mods = hierarchy::homeostatic_step_modulations(
        &pivoted_turing_mods,
        &turing_block_sizes,
        *calibration::TUNABLE_TURING_HOMEOSTASIS,
    );
    let spiral_field_strength = *calibration::TUNABLE_SPIRAL_FIELD_STRENGTH;
    let turing_mods = if spiral_field_strength > 0.0 {
        if let Some(ref blocks) = l_blocks {
            let spiral_order = crate::scan::adaptive_spiral_arc_order(
                blocks,
                turing_grid_h,
                turing_grid_w,
                use_overlap,
                &turing_field.inhibitor_gradient,
            );
            hierarchy::spiral_surprise_step_modulations(
                &turing_mods,
                &spiral_order,
                &turing_block_sizes,
                spiral_field_strength,
            )
        } else {
            turing_mods
        }
    } else {
        turing_mods
    };

    let subsonic_wave_strength = *calibration::TUNABLE_SUBSONIC_WAVE_STRENGTH;
    let subsonic_waves = if subsonic_wave_strength > 0.0 {
        if let Some(ref blocks) = l_blocks {
            hierarchy::subsonic_wave_descriptors(
                &turing_dc_grid,
                turing_grid_h,
                turing_grid_w,
                blocks,
                use_overlap,
            )
        } else {
            vec![hierarchy::SubsonicWaveDescriptor::default(); turing_mods.len()]
        }
    } else {
        Vec::new()
    };
    if subsonic_wave_strength > 0.0
        && std::env::var("AUREA_SUBSONIC_WAVE_STATS").as_deref() == Ok("1")
    {
        let n = subsonic_waves.len().max(1) as f64;
        let mean_phase = subsonic_waves.iter().map(|wave| wave.phase).sum::<f64>() / n;
        let mean_coherence = subsonic_waves
            .iter()
            .map(|wave| wave.coherence)
            .sum::<f64>()
            / n;
        let active = subsonic_waves
            .iter()
            .filter(|wave| wave.coherence >= 0.25)
            .count();
        eprintln!(
            "[AUREA_SUBSONIC_WAVE] strength={subsonic_wave_strength:.3} mean_phase={mean_phase:.4} mean_coherence={mean_coherence:.4} active={active}/{}",
            subsonic_waves.len()
        );
    }

    let bs2 = (LOT_BLOCK_SIZE * LOT_BLOCK_SIZE) as f64;
    let l_dc_pixel: Vec<f64> = l_dc_denorm.iter().map(|&v| v / bs2).collect();

    // Zero-bit bright compensation on luma steps (self-described strength).
    let bright_mods = hierarchy::bright_compensation_step_modulations(
        &l_dc_pixel,
        &turing_block_sizes,
        bright_strength * *calibration::TUNABLE_BRIGHT_COMPENSATION_GAIN,
    );

    // Zero-bit spectral exchange: gamma from the three
    // dense DC grids, delta balancing it exactly in AC-weighted log space.
    // AUR2 v18+ fixes eta=1, luma strength=1.75 and chroma strength=1.0;
    // process environment is not part of the decoding semantics.
    let spectral_exchange_on = *calibration::TUNABLE_SPECTRAL_EXCHANGE >= 0.5;
    let exchange_gamma = hierarchy::spectral_exchange_gamma(
        &turing_dc_grid,
        &chroma_dc_grids[0],
        &chroma_dc_grids[1],
        turing_grid_h,
        turing_grid_w,
        if spectral_exchange_on { 1.0 } else { 0.0 },
    );
    // Diffuse-chroma relief field: signature × dark gate, both zero-bit from
    // decoded DC grids. Shared by the luma-pivot relief and the chroma
    // exchange relief below.
    let chroma_diffusion = hierarchy::chroma_diffusion_signature(
        &chroma_dc_grids[0],
        &chroma_dc_grids[1],
        exchange_gamma,
    );
    let luma_relief_gate = hierarchy::luma_relief_dark_gate(&turing_dc_grid);
    let relief_field = chroma_diffusion.repair * luma_relief_gate;
    // The relief strengths are a v28 format property: a stream emitted under
    // any earlier version must quantize exactly as that version specifies.
    let (luma_pivot_relief, spectral_chroma_relief) = if version >= 28 {
        (
            *calibration::TUNABLE_LUMA_PIVOT_RELIEF,
            *calibration::TUNABLE_SPECTRAL_CHROMA_RELIEF,
        )
    } else {
        (0.0, 0.0)
    };
    let chroma_exchange_relief =
        (1.0 - spectral_chroma_relief * relief_field).clamp(0.0, 1.0);
    let exchange_chroma_gamma =
        exchange_gamma * *calibration::TUNABLE_SPECTRAL_CHROMA * chroma_exchange_relief;
    // Cycle 21: redistribute the very same chroma budget across blocks
    // according to the local chroma/luma activity. Neutral at strength 0.
    let (exchange_local_levels, exchange_local_ladder) = match l_blocks.as_ref() {
        Some(blocks) if exchange_gamma > 0.0 => hierarchy::spectral_exchange_local_levels(
            &turing_dc_grid,
            &chroma_dc_grids[0],
            &chroma_dc_grids[1],
            turing_grid_h,
            turing_grid_w,
            blocks,
            use_overlap,
            *calibration::TUNABLE_SPECTRAL_LOCAL,
        ),
        _ => (Vec::new(), {
            let mut table = [0.0f64; hierarchy::SPECTRAL_LOCAL_LEVELS];
            table[0] = 1.0;
            table
        }),
    };
    let exchange_local_on =
        !exchange_local_levels.is_empty() && exchange_local_levels.iter().any(|&level| level != 0);
    // Adaptive relief of the luma counter-gift on diffuse-chroma fields:
    // zero-bit, floored at the v17 baseline, and exactly neutral when the
    // relief knob is zero. The dark gate restricts the relief to scenes where
    // the Weber surplus funds it. The balancing delta follows the *relieved*
    // chroma slope so the exchange stays log-neutral at luma strength 1.
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
        crate::golden::PHI.powf(-exchange_delta)
    } else {
        1.0
    };

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
            dc_norm: &l_dc_norm,
            ac_blocks: &l_ac,
            grid_h: l_gh,
            grid_w: l_gw,
            chroma_factor: 1.0,
        },
        ChannelLOT {
            dc_norm: &c1_dc_norm,
            ac_blocks: &c1_ac,
            grid_h: c1_gh,
            grid_w: c1_gw,
            chroma_factor: c1_chroma_factor,
        },
        ChannelLOT {
            dc_norm: &c2_dc_norm,
            ac_blocks: &c2_ac,
            grid_h: c2_gh,
            grid_w: c2_gw,
            chroma_factor: 1.0,
        },
    ];

    let use_csf = flags & FLAG_CSF_MODULATION != 0;

    let foveal_cell_map = lot::foveal_saliency_map(&turing_dc_grid, turing_grid_h, turing_grid_w);
    let l_foveal_map = if let Some(ref blocks) = l_blocks {
        hierarchy::sample_scalar_for_blocks(
            &foveal_cell_map,
            turing_grid_h,
            turing_grid_w,
            blocks,
            use_overlap,
        )
    } else {
        foveal_cell_map
    };

    drop(t_hier);
    let primitive_matches: Vec<PrimitiveMatch> = Vec::new();
    let match_bytes = serialize_primitive_matches(&primitive_matches);

    // ================================================================
    // Trellis RDO: compute base lambda from quality and detail_step
    // ================================================================
    let use_trellis = params.quality <= 90; // disable at near-lossless quality
    let base_lambda = calibration::trellis_lambda(params.quality, detail_step, version >= 29);
    let use_wavelet_parent_context = params.quality <= rans::WAVELET_PARENT_MAX_QUALITY;
    let use_vector_codon = spectral_exchange_on && use_progressive_packets;
    let use_causal_parent_context = use_wavelet_parent_context || use_vector_codon;

    // ================================================================
    // CfL two-pass encoding: luma first, then chroma with prediction
    // ================================================================

    // Helper closure: encode one channel's AC into the body, returning
    // the per-block dequantized AC vectors (in natural order, indexed
    // by zz_idx-1) if `collect_l_rec` is true.
    // `cfl_residual_blocks` — if Some, the chroma AC has already been
    // replaced by the CfL residual for active blocks (pre-quantization).
    struct ProgressiveChannelAc {
        eob_count: usize,
        eob_encoded: Vec<u8>,
        packets: Vec<(usize, Vec<u8>)>,
        quantized_blocks: Vec<Vec<i16>>,
        codon_trace: Option<CodonChannelTrace>,
    }
    struct CodonRdoContext<'a> {
        l_target: &'a [Vec<f64>],
        c1_target: &'a [Vec<f64>],
        c2_target: &'a [Vec<f64>],
        l_rec: &'a [Vec<f64>],
        c1_rec: &'a [Vec<f64>],
        c2_cfl_flags: &'a [bool],
        c2_cfl_alpha: &'a [usize],
    }
    struct C1CodonRdoContext<'a> {
        l_target: &'a [Vec<f64>],
        c1_target: &'a [Vec<f64>],
        l_rec: &'a [Vec<f64>],
        c1_cfl_flags: &'a [bool],
        c1_cfl_alpha: &'a [usize],
    }
    let mut progressive_channels: Vec<ProgressiveChannelAc> = Vec::new();
    let codon_stats_enabled = std::env::var("AUREA_CODON_STATS")
        .map(|value| value == "1")
        .unwrap_or(false);

    let encode_channel_ac = |ch: &ChannelLOT,
                             ch_idx: usize,
                             body: &mut Vec<u8>,
                             collect_l_rec: bool,
                             chroma_step_mods: Option<&[f64]>,
                             prior_codon_blocks: &[&[Vec<i16>]],
                             c1_codon_rdo: Option<&C1CodonRdoContext<'_>>,
                             codon_rdo: Option<&CodonRdoContext<'_>>|
     -> (Vec<Vec<f64>>, Option<ProgressiveChannelAc>) {
        let t_setup = crate::timing::stage("enc:quant_setup");
        let n_blocks = ch.grid_h * (if use_variable { 1 } else { ch.grid_w });
        let mut ac_flat: Vec<i16> = Vec::new();
        let mut ac_turing: Vec<u8> = Vec::new();
        let mut ac_by_block: Vec<Vec<i16>> = if use_progressive_packets {
            vec![Vec::new(); n_blocks]
        } else {
            Vec::new()
        };
        let mut eob_positions: Vec<u16> = vec![0; n_blocks];

        let ac_step = detail_step * ch.chroma_factor * lot_global_factor;
        // Spectral exchange: finer luma steps, coarser chroma HF steps.
        let ac_step = if ch_idx == 0 {
            ac_step * luma_exchange_scale
        } else {
            ac_step
        };
        let texture_law = version >= 29 && params.quality <= calibration::V29_TEXTURE_MAX_QUALITY;
        let trellis_distortion_weight = match ch_idx {
            0 => *calibration::TUNABLE_TRELLIS_LUMA_DISTORTION,
            1 => *calibration::TUNABLE_TRELLIS_C1_DISTORTION,
            // v29 texture law: C2 (the red axis) carries most of the chroma
            // texture at low quality; its distortion weight rises to 1.75.
            _ => calibration::env_override("AUREA_TRELLIS_C2_DISTORTION").unwrap_or(
                if texture_law {
                    calibration::V29_C2_DISTORTION_TEXTURE
                } else {
                    1.0
                },
            ),
        };

        // Precompute QMAT^power LUTs — eliminates powf() from inner loop
        let qmat_pow_8: Vec<f64> = (0..64)
            .map(|i| lot::qmat_lookup(i / 8, i % 8, 8).powf(qmat_power))
            .collect();
        let qmat_pow_16: Vec<f64> = (0..256)
            .map(|i| lot::qmat_lookup(i / 16, i % 16, 16).powf(qmat_power))
            .collect();
        let qmat_pow_32: Vec<f64> = (0..1024)
            .map(|i| lot::qmat_lookup(i / 32, i % 32, 32).powf(qmat_power))
            .collect();

        let ch_block_buckets = turing_buckets.clone();

        // Lambda × activity (encoder-only): negative k lowers lambda on
        // high-energy blocks so the trellis keeps their texture, financed by
        // more zeroing in smooth blocks. (activity/geomean)^k in log space.
        let lambda_activity_k = calibration::env_override("AUREA_LAMBDA_ACTIVITY")
            .map(|v| v.clamp(-2.0, 2.0))
            .unwrap_or(if texture_law {
                calibration::V29_LAMBDA_ACTIVITY_TEXTURE
            } else {
                0.0
            });
        let activity_lambda_mods: Vec<f64> = if lambda_activity_k != 0.0 {
            let log_energies: Vec<f64> = ch
                .ac_blocks
                .iter()
                .map(|block| {
                    (block.iter().map(|c| c * c).sum::<f64>() / block.len().max(1) as f64)
                        .max(1e-9)
                        .ln()
                })
                .collect();
            let log_mean =
                log_energies.iter().sum::<f64>() / log_energies.len().max(1) as f64;
            log_energies
                .iter()
                .map(|&le| ((le - log_mean) * 0.5 * lambda_activity_k).exp())
                .collect()
        } else {
            Vec::new()
        };

        // CSF freq-norm LUTs per zigzag position (same sqrt formula as
        // lot::csf_qmat_factor; the per-block factor stays 1 + ((ka*f)*f)
        // with identical operation order, so the result is bit-exact).
        let csf_freq_for = |zz: &[usize], block_size: usize| -> Vec<f64> {
            let bs_max = (block_size - 1).max(1) as f64;
            zz.iter()
                .map(|&zz_idx| {
                    let row = zz_idx / block_size;
                    let col = zz_idx % block_size;
                    ((row * row + col * col) as f64).sqrt() / (bs_max * std::f64::consts::SQRT_2)
                })
                .collect()
        };
        let csf_freq_8: Vec<f64> = csf_freq_for(&zz_8, 8);
        let csf_freq_16: Vec<f64> = csf_freq_for(&zz_16, 16);
        let csf_freq_32: Vec<f64> = csf_freq_for(&zz_32, 32);

        // Spectral exchange chroma tables: phi^(gamma * freq_norm) per zigzag
        // position. None when the mechanism is neutral (bit-exact off state).
        // One table set per local level; a single set when the field is
        // uniform, which is the exact v21 state.
        let exch_for = |freqs: &[f64], gamma: f64| -> Vec<f64> {
            freqs
                .iter()
                .map(|&f| crate::golden::PHI.powf(gamma * f))
                .collect()
        };
        let exch_tables: Option<Vec<[Vec<f64>; 3]>> = if exchange_chroma_gamma > 0.0 && ch_idx > 0 {
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
                            exch_for(&csf_freq_8, gamma),
                            exch_for(&csf_freq_16, gamma),
                            exch_for(&csf_freq_32, gamma),
                        ]
                    })
                    .collect(),
            )
        } else {
            None
        };

        // Directional QMAT: per-block gradient angle + strength from Turing field
        let gradient_angles = if let Some(ref blocks) = l_blocks {
            hierarchy::build_adaptive_directional_structures(
                &turing_field,
                blocks,
                use_overlap,
                *calibration::TUNABLE_DIRECTIONAL_COHERENCE,
            )
        } else {
            hierarchy::build_gradient_angles(&turing_field, ch.grid_h, ch.grid_w, LOT_BLOCK_SIZE)
                .into_iter()
                .map(|(angle, strength)| (angle, strength, 1.0))
                .collect()
        };
        let grad_threshold = *calibration::TUNABLE_GRADIENT_THRESHOLD;

        // Precompute 8 rotated QMAT LUTs per block size (angles 0°, 22.5°, ..., 157.5°)
        let n_angles = 8usize;
        let angle_step = std::f64::consts::PI / n_angles as f64;
        let rotated_qmats_8: Vec<Vec<f64>> = (0..n_angles)
            .map(|a| lot::precompute_rotated_qmat(8, a as f64 * angle_step, qmat_power))
            .collect();
        let rotated_qmats_16: Vec<Vec<f64>> = (0..n_angles)
            .map(|a| lot::precompute_rotated_qmat(16, a as f64 * angle_step, qmat_power))
            .collect();
        let rotated_qmats_32: Vec<Vec<f64>> = (0..n_angles)
            .map(|a| lot::precompute_rotated_qmat(32, a as f64 * angle_step, qmat_power))
            .collect();

        // CfL must retain the native coefficient count of every adaptive
        // block.  A fixed 16x16-sized slab silently truncated 32x32 blocks to
        // 255 AC values, so the encoder and decoder did not form the chroma
        // residual in the same spectral space.  Keep native-sized blocks;
        // this path is only collected for channels used as causal predictors.
        let mut l_rec_blocks: Vec<Vec<f64>> = if collect_l_rec {
            vec![Vec::new(); n_blocks]
        } else {
            Vec::new()
        };
        drop(t_setup);
        let t_quant = crate::timing::stage("enc:quant_trellis");

        // Per-block quantization results, computed in parallel. The greedy
        // context model restarts from its INIT tables at every block, so
        // blocks are fully independent: the parallel map produces exactly
        // the same decisions as the previous sequential loop (bit-exact),
        // and the ordered collect preserves the scan order for assembly.
        struct BlockQuantResult {
            block_idx: usize,
            tb: u8,
            qvals: Vec<i16>,
            eob: usize,
            /// Dense L_rec slice (natural AC order, default_ac_per_block long),
            /// only filled when collect_l_rec is set.
            l_rec: Vec<f64>,
            vector_rdo_tested: usize,
            vector_rdo_changed: usize,
            vector_rdo_distortion_gain: f64,
            vector_rdo_rate_delta: f64,
        }

        use rayon::prelude::*;
        let block_capacity = |idx: usize| -> usize {
            let size = if use_variable {
                l_blocks
                    .as_ref()
                    .and_then(|blocks| blocks.get(idx))
                    .map(|&(_, _, size)| size)
                    .unwrap_or(LOT_BLOCK_SIZE)
            } else {
                LOT_BLOCK_SIZE
            };
            size * size - 1
        };

        // Per-block quantization, parameterized for the EOB-step second pass:
        // `eob_step_mod` multiplies the local step, `forced_eob` clamps the
        // block to the transmitted (first-pass) EOB so both sides derive the
        // same zero-bit field.
        // `greedy_probe`: pass-1 mode of the one-pass EOB scheme — skip the
        // trellis entirely and report the greedy EOB, which then becomes the
        // transmitted EOB the modulation field derives from. Encoder-only
        // freedom (any EOB choice is a valid stream), no format change; the
        // stream differs from the two-trellis-pass scheme, so promotion rode
        // through the full frontier + showcase re-validation.
        let quant_block = |block_idx: usize,
                           eob_step_mod: f64,
                           forced_eob: Option<usize>,
                           greedy_probe: bool,
                           scratch: &mut trellis::TrellisScratch|
         -> BlockQuantResult {
            {
                let ac = &ch.ac_blocks[block_idx];
                let block_size = if use_variable {
                    if let Some(ref blocks) = l_blocks {
                        if block_idx < blocks.len() {
                            blocks[block_idx].2
                        } else {
                            LOT_BLOCK_SIZE
                        }
                    } else {
                        LOT_BLOCK_SIZE
                    }
                } else {
                    LOT_BLOCK_SIZE
                };
                let zz_order = match block_size {
                    8 => &zz_8,
                    32 => &zz_32,
                    _ => &zz_16,
                };
                let l_fov_idx = (block_idx as f64 * l_foveal_map.len() as f64
                    / n_blocks.max(1) as f64) as usize;
                let foveal_factor =
                    l_foveal_map[l_fov_idx.min(l_foveal_map.len().saturating_sub(1))];
                let l_idx =
                    (block_idx as f64 * l_dc_pixel.len() as f64 / n_blocks.max(1) as f64) as usize;
                let dc_l = l_dc_pixel[l_idx.min(l_dc_pixel.len().saturating_sub(1))];
                let turing_mod = turing_mods.get(block_idx).copied().unwrap_or(1.0);
                // Per-channel extra step modulations: chroma breathing for C1/C2,
                // bright compensation for L. 1.0 when absent.
                let chroma_breathing_mod = chroma_step_mods
                    .and_then(|mods| mods.get(block_idx))
                    .copied()
                    .unwrap_or(1.0);
                let local_step =
                    ac_step * foveal_factor * turing_mod * chroma_breathing_mod * eob_step_mod;
                let step_clamped = local_step.max(0.1);
                let block_lambda = base_lambda
                    * foveal_factor
                    * turing_mod
                    * chroma_breathing_mod
                    * activity_lambda_mods.get(block_idx).copied().unwrap_or(1.0);

                let ac_per_block = block_size * block_size - 1;

                // Phase A: collect per-position quantization parameters
                let mut raw_coeffs = Vec::with_capacity(ac_per_block);
                let mut pos_steps_vec = Vec::with_capacity(ac_per_block);
                let mut dead_zones_vec = Vec::with_capacity(ac_per_block);
                let mut zz_indices = Vec::with_capacity(ac_per_block);

                // Select QMAT: rotated if gradient is strong enough, standard otherwise
                let (blk_angle, blk_strength, blk_qmat_blend) = if block_idx < gradient_angles.len()
                {
                    gradient_angles[block_idx]
                } else {
                    (0.0, 0.0, 0.0)
                };
                let use_rotated = blk_strength > grad_threshold && blk_qmat_blend > 0.0;
                let block_qmat: Option<&Vec<f64>> = if use_rotated {
                    // Quantize angle to nearest of 8 precomputed rotations
                    let mut a = blk_angle.rem_euclid(std::f64::consts::PI);
                    let idx = ((a / angle_step + 0.5) as usize) % n_angles;
                    let luts = match block_size {
                        8 => &rotated_qmats_8,
                        32 => &rotated_qmats_32,
                        _ => &rotated_qmats_16,
                    };
                    Some(&luts[idx])
                } else {
                    None
                };

                // Same operation order as lot::csf_qmat_factor:
                // csf = 1.0 + ((CSF_DARK_BOOST * (1 - lum_norm)) * f) * f
                let csf_ka = calibration::CSF_DARK_BOOST * (1.0 - (dc_l / 255.0).clamp(0.0, 1.0));
                let csf_freq = match block_size {
                    8 => &csf_freq_8,
                    32 => &csf_freq_32,
                    _ => &csf_freq_16,
                };
                let qmat_pow = match block_size {
                    8 => &qmat_pow_8,
                    32 => &qmat_pow_32,
                    _ => &qmat_pow_16,
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

                for (zz_pos, &zz_idx) in zz_order.iter().enumerate() {
                    let row = zz_idx / block_size;
                    let col = zz_idx % block_size;
                    let standard_qfactor = qmat_pow[row * block_size + col];
                    let mut qfactor = if let Some(rqmat) = block_qmat {
                        hierarchy::blend_qmat_factors(
                            standard_qfactor,
                            rqmat[row * block_size + col],
                            blk_qmat_blend,
                        )
                    } else {
                        standard_qfactor
                    };
                    if use_csf {
                        let f = csf_freq[zz_pos];
                        let csf = 1.0 + csf_ka * f * f;
                        qfactor *= csf;
                    }
                    if let Some(ref tables) = exch_tables {
                        let level =
                            exchange_local_levels.get(block_idx).copied().unwrap_or(0) as usize;
                        let set = &tables[level.min(tables.len() - 1)];
                        let exch = match block_size {
                            8 => &set[0],
                            32 => &set[2],
                            _ => &set[1],
                        };
                        qfactor *= exch[zz_pos];
                    }
                    if let Some(ref wave) = wave_qfactors {
                        qfactor *= wave[zz_idx];
                    }
                    let pos_step = step_clamped * qfactor;
                    let coeff = if (zz_idx - 1) < ac.len() {
                        ac[zz_idx - 1]
                    } else {
                        0.0
                    };
                    let local_dz =
                        calibration::dead_zone_for_position(zz_pos, ac_per_block, dead_zone)
                            * (1.0 + 0.3 * (1.0 - foveal_factor));
                    raw_coeffs.push(coeff);
                    pos_steps_vec.push(pos_step);
                    dead_zones_vec.push(local_dz);
                    zz_indices.push(zz_idx);
                }

                // Phase B: quantize (greedy + optional trellis)
                let tb = ch_block_buckets[block_idx];
                let (greedy_qvals, snapshots) = trellis::greedy_quantize_and_snapshot(
                    &raw_coeffs,
                    &pos_steps_vec,
                    &dead_zones_vec,
                    tb,
                    use_trellis && !greedy_probe,
                );
                if greedy_probe {
                    let eob = greedy_qvals
                        .iter()
                        .rposition(|&v| v != 0)
                        .map(|p| p + 1)
                        .unwrap_or(0);
                    return BlockQuantResult {
                        block_idx,
                        tb,
                        qvals: greedy_qvals,
                        eob,
                        l_rec: Vec::new(),
                        vector_rdo_tested: 0,
                        vector_rdo_changed: 0,
                        vector_rdo_distortion_gain: 0.0,
                        vector_rdo_rate_delta: 0.0,
                    };
                }

                let (mut qvals, eob) = if use_trellis {
                    trellis::trellis_quantize_block_scratch(
                        &raw_coeffs,
                        &pos_steps_vec,
                        &dead_zones_vec,
                        tb,
                        block_lambda,
                        trellis_distortion_weight,
                        &snapshots,
                        scratch,
                    )
                } else {
                    let eob = greedy_qvals
                        .iter()
                        .rposition(|&v| v != 0)
                        .map(|p| p + 1)
                        .unwrap_or(0);
                    (greedy_qvals, eob)
                };

                // Joint color-codon witness.  L and C1 are causal and already
                // reconstructed exactly as the decoder will see them.  Refine
                // C2 against their coupled inverse-GCT error before rANS.
                let mut vector_rdo_tested = 0usize;
                let mut vector_rdo_changed = 0usize;
                let mut vector_rdo_distortion_gain = 0.0f64;
                let mut vector_rdo_rate_delta = 0.0f64;
                let vector_rdo_strength = *calibration::TUNABLE_VECTOR_RDO_STRENGTH;
                if ch_idx == 1 && vector_rdo_strength > 0.0 {
                    if let Some(context) = c1_codon_rdo {
                        let target_l = context.l_target.get(block_idx);
                        let target_c1 = context.c1_target.get(block_idx);
                        let rec_l = context.l_rec.get(block_idx);
                        if let (Some(target_l), Some(target_c1), Some(rec_l)) =
                            (target_l, target_c1, rec_l)
                        {
                            let cfl_alpha = if context
                                .c1_cfl_flags
                                .get(block_idx)
                                .copied()
                                .unwrap_or(false)
                            {
                                cfl::ALPHA_PALETTE[context
                                    .c1_cfl_alpha
                                    .get(block_idx)
                                    .copied()
                                    .unwrap_or(3)
                                    .min(7)]
                            } else {
                                0.0
                            };
                            for position in 0..eob.min(qvals.len()) {
                                let current_q = qvals[position];
                                if current_q == 0 {
                                    continue;
                                }
                                let natural = zz_indices[position] - 1;
                                if natural >= target_l.len()
                                    || natural >= target_c1.len()
                                    || natural >= rec_l.len()
                                {
                                    continue;
                                }
                                vector_rdo_tested += 1;
                                let prediction = cfl_alpha * rec_l[natural];
                                let (refined, gain, rate_delta) = refine_c1_codon(
                                    current_q,
                                    pos_steps_vec[position],
                                    target_l[natural],
                                    rec_l[natural],
                                    target_c1[natural],
                                    prediction,
                                    vector_rdo_strength,
                                    *calibration::TUNABLE_VECTOR_RDO_Y_WEIGHT,
                                    block_lambda,
                                    *calibration::TUNABLE_VECTOR_RDO_RATE_WEIGHT,
                                );
                                if refined != current_q {
                                    qvals[position] = refined;
                                    vector_rdo_changed += 1;
                                    vector_rdo_distortion_gain += gain;
                                    vector_rdo_rate_delta += rate_delta;
                                }
                            }
                        }
                    }
                } else if ch_idx == 2 && vector_rdo_strength > 0.0 {
                    if let Some(context) = codon_rdo {
                        let target_l = context.l_target.get(block_idx);
                        let target_c1 = context.c1_target.get(block_idx);
                        let target_c2 = context.c2_target.get(block_idx);
                        let rec_l = context.l_rec.get(block_idx);
                        let rec_c1 = context.c1_rec.get(block_idx);
                        if let (
                            Some(target_l),
                            Some(target_c1),
                            Some(target_c2),
                            Some(rec_l),
                            Some(rec_c1),
                        ) = (target_l, target_c1, target_c2, rec_l, rec_c1)
                        {
                            let cfl_alpha = if context
                                .c2_cfl_flags
                                .get(block_idx)
                                .copied()
                                .unwrap_or(false)
                            {
                                cfl::ALPHA_PALETTE[context
                                    .c2_cfl_alpha
                                    .get(block_idx)
                                    .copied()
                                    .unwrap_or(3)
                                    .min(7)]
                            } else {
                                0.0
                            };
                            for position in 0..eob.min(qvals.len()) {
                                let current_q = qvals[position];
                                if current_q == 0 {
                                    continue;
                                }
                                let natural = zz_indices[position] - 1;
                                if natural >= target_l.len()
                                    || natural >= target_c1.len()
                                    || natural >= target_c2.len()
                                    || natural >= rec_l.len()
                                    || natural >= rec_c1.len()
                                {
                                    continue;
                                }
                                vector_rdo_tested += 1;
                                let prediction = cfl_alpha * rec_l[natural];
                                let (refined, gain, rate_delta) = refine_c2_codon(
                                    current_q,
                                    pos_steps_vec[position],
                                    target_l[natural],
                                    rec_l[natural],
                                    target_c1[natural],
                                    rec_c1[natural],
                                    target_c2[natural],
                                    prediction,
                                    vector_rdo_strength,
                                    *calibration::TUNABLE_VECTOR_RDO_Y_WEIGHT,
                                    block_lambda,
                                    *calibration::TUNABLE_VECTOR_RDO_RATE_WEIGHT,
                                );
                                if refined != current_q {
                                    qvals[position] = refined;
                                    vector_rdo_changed += 1;
                                    vector_rdo_distortion_gain += gain;
                                    vector_rdo_rate_delta += rate_delta;
                                }
                            }
                        }
                    }
                }

                // EOB-step second pass: the transmitted EOB is pass 1's, so
                // the block is clamped to it — coefficients beyond are
                // zeroed, a shorter natural EOB keeps its trailing zeros.
                let (qvals, eob) = if let Some(forced) = forced_eob {
                    let mut qvals = qvals;
                    for value in qvals.iter_mut().skip(forced) {
                        *value = 0;
                    }
                    (qvals, forced.min(ac_per_block))
                } else {
                    (qvals, eob)
                };

                // Dequantize for L_rec (CfL needs the quantized values the decoder will see)
                let mut l_rec = Vec::new();
                if collect_l_rec {
                    let dequant_offset = *calibration::TUNABLE_DEQUANT_OFFSET;
                    l_rec = vec![0.0f64; ac_per_block];
                    for (k, &zz_idx) in zz_indices.iter().enumerate() {
                        if (zz_idx - 1) < ac_per_block {
                            let qi = if k < qvals.len() { qvals[k] } else { 0 };
                            let q_off = qi as f64 + dequant_offset * (qi as f64).signum();
                            l_rec[zz_idx - 1] = q_off * pos_steps_vec[k];
                        }
                    }
                }

                BlockQuantResult {
                    block_idx,
                    tb,
                    qvals,
                    eob,
                    l_rec,
                    vector_rdo_tested,
                    vector_rdo_changed,
                    vector_rdo_distortion_gain,
                    vector_rdo_rate_delta,
                }
            }
        };

        // Cycle 32 — EOB-driven luma step field (AUR2 v30). Pass 1 fixes the
        // EOBs; if the field is active, pass 2 requantizes with the modulated
        // steps under the pass-1 EOBs, which are what the stream transmits.
        let eob_step_min_q = calibration::env_override("AUREA_EOB_STEP_MIN_Q")
            .map(|v| v as u8)
            .unwrap_or(calibration::V30_EOB_STEP_MIN_QUALITY);
        // v30: luma field; v31 extends it to the chroma channels, each driven
        // by its own transmitted EOBs.
        let eob_step_eta = if params.quality >= eob_step_min_q {
            if ch_idx == 0 {
                // v32: softened on the diffuse-dark field (SSIM protection).
                // Only below the anti-ring boundary: at fine steps the field
                // does not collapse variance, and its full strength is what
                // wins the high-quality texture metrics.
                let luma_relief = if version >= 32
                    && params.quality < calibration::V29_ANTI_RING_OFF_QUALITY
                {
                    1.0 - calibration::V32_EOB_LUMA_RELIEF * relief_field
                } else {
                    1.0
                };
                calibration::env_override("AUREA_EOB_STEP")
                    .map(|v| v.clamp(0.0, 2.0))
                    .unwrap_or(if version >= 30 {
                        calibration::V30_EOB_STEP_ETA * luma_relief
                    } else {
                        0.0
                    })
            } else {
                calibration::env_override("AUREA_EOB_STEP_CHROMA")
                    .map(|v| v.clamp(0.0, 2.0))
                    .unwrap_or(if version >= 31 {
                        calibration::V31_EOB_STEP_CHROMA_ETA
                            * calibration::eob_chroma_gate(exchange_gamma)
                    } else {
                        0.0
                    })
            }
        } else {
            0.0
        };
        // One-pass EOB scheme: when the modulation field is active, pass 1
        // only exists to fix the transmitted EOBs — the greedy quantizer's
        // EOBs serve directly and the trellis runs once, in pass 2.
        let one_pass_eob = calibration::env_override("AUREA_EOB_ONE_PASS")
            .map(|v| v >= 0.5)
            .unwrap_or(calibration::V33_EOB_ONE_PASS);
        let greedy_probe = one_pass_eob && eob_step_eta > 0.0 && use_progressive_packets;
        let mod_stats_enabled = std::env::var("AUREA_MOD_STATS").as_deref() == Ok("1");
        let mut stats_eob_mods: Option<Vec<f64>> = None;
        let pass1: Vec<BlockQuantResult> = block_scan_order
            .par_iter()
            .map_init(trellis::TrellisScratch::default, |scratch, &block_idx| {
                quant_block(block_idx, 1.0, None, greedy_probe, scratch)
            })
            .collect();
        let block_results: Vec<BlockQuantResult> =
            if eob_step_eta > 0.0 && use_progressive_packets {
                let mut eobs = vec![0usize; n_blocks];
                for result in &pass1 {
                    eobs[result.block_idx] = result.eob;
                }
                let capacities: Vec<f64> =
                    (0..n_blocks).map(|i| block_capacity(i) as f64).collect();
                let mods = hierarchy::eob_step_mods(&eobs, &capacities, eob_step_eta);
                if mod_stats_enabled {
                    stats_eob_mods = Some(mods.clone());
                }
                block_scan_order
                    .par_iter()
                    .map_init(trellis::TrellisScratch::default, |scratch, &block_idx| {
                        quant_block(
                            block_idx,
                            mods[block_idx],
                            Some(eobs[block_idx]),
                            false,
                            scratch,
                        )
                    })
                    .collect()
            } else {
                pass1
            };

        // Modulator-stacking observability (AUREA_MOD_STATS=1): the effective
        // per-block step product foveal × turing × per-channel extra × EOB
        // field, exactly as `quant_block` multiplies it. Read-only, printed
        // once per channel; the log-neutral regularizers bound each factor,
        // not their product — this line is where the product is watched.
        if mod_stats_enabled {
            let mut products = Vec::with_capacity(n_blocks);
            let mut comp_min = [f64::INFINITY; 4];
            let mut comp_max = [f64::NEG_INFINITY; 4];
            let mut floor_hits = 0usize;
            for block_idx in 0..n_blocks {
                let l_fov_idx = (block_idx as f64 * l_foveal_map.len() as f64
                    / n_blocks.max(1) as f64) as usize;
                let foveal =
                    l_foveal_map[l_fov_idx.min(l_foveal_map.len().saturating_sub(1))];
                let turing = turing_mods.get(block_idx).copied().unwrap_or(1.0);
                let extra = chroma_step_mods
                    .and_then(|mods| mods.get(block_idx))
                    .copied()
                    .unwrap_or(1.0);
                let eobm = stats_eob_mods
                    .as_ref()
                    .and_then(|mods| mods.get(block_idx))
                    .copied()
                    .unwrap_or(1.0);
                for (slot, value) in [foveal, turing, extra, eobm].into_iter().enumerate() {
                    comp_min[slot] = comp_min[slot].min(value);
                    comp_max[slot] = comp_max[slot].max(value);
                }
                let product = foveal * turing * extra * eobm;
                if ac_step * product < 0.1 {
                    floor_hits += 1;
                }
                products.push(product);
            }
            products.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pick = |q: f64| products[((products.len() - 1) as f64 * q) as usize];
            let gmean = (products.iter().map(|p| p.ln()).sum::<f64>()
                / products.len().max(1) as f64)
                .exp();
            let phi = crate::golden::PHI;
            let band = |lo: f64, hi: f64| {
                products.iter().filter(|&&p| p < lo || p > hi).count()
            };
            eprintln!(
                "[AUREA_MOD_STATS] ch={ch_idx} n={} ac_step={ac_step:.4} \
                 prod_min={:.4} p01={:.4} gmean={:.4} p99={:.4} prod_max={:.4} \
                 out_phi={} out_phi2={} floor_hits={floor_hits} \
                 fov=[{:.3},{:.3}] turing=[{:.3},{:.3}] extra=[{:.3},{:.3}] eob=[{:.3},{:.3}]",
                products.len(),
                pick(0.0),
                pick(0.01),
                gmean,
                pick(0.99),
                pick(1.0),
                band(1.0 / phi, phi),
                band(1.0 / (phi * phi), phi * phi),
                comp_min[0], comp_max[0],
                comp_min[1], comp_max[1],
                comp_min[2], comp_max[2],
                comp_min[3], comp_max[3],
            );
        }

        // Sequential assembly in the original scan order (identical writes).
        for result in &block_results {
            let block_idx = result.block_idx;
            if use_progressive_packets {
                ac_by_block[block_idx].extend_from_slice(&result.qvals[..result.eob]);
            } else {
                for &qi in &result.qvals[..result.eob] {
                    ac_flat.push(qi);
                    ac_turing.push(result.tb);
                }
            }
            eob_positions[block_idx] = result.eob as u16;
            if collect_l_rec {
                l_rec_blocks[block_idx] = result.l_rec.clone();
            }
        }

        // P0 counters. `coefficients_before_eob` is the transform's own output
        // count, before the EOB truncation; comparing it against `nonzero_ac`
        // separates what a topology costs at the transform from what it costs
        // at the coder — the distinction a lapped variant has to be judged on.
        if crate::timing::enabled() {
            let mut capacity = 0.0f64;
            let mut nonzero = 0.0f64;
            let mut eob_sum = 0.0f64;
            for result in &block_results {
                capacity += ch.ac_blocks[result.block_idx].len() as f64;
                nonzero += result.qvals[..result.eob]
                    .iter()
                    .filter(|&&q| q != 0)
                    .count() as f64;
                eob_sum += result.eob as f64;
            }
            crate::timing::count("analysis_blocks", block_results.len() as f64);
            crate::timing::count("coefficients_before_eob", capacity);
            crate::timing::count("nonzero_ac", nonzero);
            crate::timing::count("eob_sum", eob_sum);
        }

        // Cycle 27 — the AC context axis.
        //
        // `ch_block_buckets` comes from the Turing field, a Sobel of the DC
        // grid: it measures how a block differs from its neighbours and is
        // blind to the structure inside it. The EOB position is exactly that
        // missing quantity — it counts the coefficients the block carries —
        // and the decoder reads every EOB before any AC coefficient, so keying
        // the contexts on it costs no bits and stays symmetric.
        //
        // The EOB stream itself keeps the Turing bucket: a symbol cannot be its
        // own context.
        let ch_ac_buckets: Vec<u8> = if use_activity_context {
            rans::activity_buckets(&eob_positions, block_capacity)
        } else {
            ch_block_buckets.clone()
        };
        if use_activity_context && !use_progressive_packets {
            // The monolithic stream was filled with the Turing bucket during
            // assembly, before every block's EOB was known.
            ac_turing.clear();
            for result in &block_results {
                ac_turing.extend(std::iter::repeat_n(
                    ch_ac_buckets[result.block_idx],
                    result.eob,
                ));
            }
        }

        let c1_fourier_predictor = |block_idx: usize, position: usize| -> i8 {
            if ch_idx != 1 {
                return -1;
            }
            let Some(top_idx) = fourier_top_blocks.get(block_idx).copied().flatten() else {
                return -1;
            };
            let causal = if position < crate::scan::PROGRESSIVE_SKELETON_COEFFS {
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
            if !causal {
                return -1;
            }
            let value = ac_by_block
                .get(top_idx)
                .and_then(|block| block.get(position))
                .copied()
                .unwrap_or(0);
            if value == 0 {
                return -1;
            }
            let block_size = if use_variable {
                l_blocks
                    .as_ref()
                    .and_then(|blocks| blocks.get(block_idx))
                    .map(|&(_, _, size)| size)
                    .unwrap_or(LOT_BLOCK_SIZE)
            } else {
                LOT_BLOCK_SIZE
            };
            let zigzag = match block_size {
                8 => &zz_8,
                32 => &zz_32,
                _ => &zz_16,
            };
            let row = zigzag.get(position).copied().unwrap_or(0) / block_size;
            i8::from((value < 0) ^ (row & 1 != 0))
        };

        if ch_idx > 0
            && *calibration::TUNABLE_VECTOR_RDO_STRENGTH > 0.0
            && std::env::var("AUREA_VECTOR_RDO_STATS")
                .map(|value| value == "1")
                .unwrap_or(false)
        {
            let tested = block_results
                .iter()
                .map(|result| result.vector_rdo_tested)
                .sum::<usize>();
            let changed = block_results
                .iter()
                .map(|result| result.vector_rdo_changed)
                .sum::<usize>();
            let gain = block_results
                .iter()
                .map(|result| result.vector_rdo_distortion_gain)
                .sum::<f64>();
            let rate_delta = block_results
                .iter()
                .map(|result| result.vector_rdo_rate_delta)
                .sum::<f64>();
            eprintln!(
                "[AUREA_VECTOR_RDO] channel=C{ch_idx} tested={tested} changed={changed} ({:.3}%) transform_gain={gain:.3} payload_delta_bits={rate_delta:.1}",
                100.0 * changed as f64 / tested.max(1) as f64,
            );
        }

        drop(t_quant);
        let _t_rans = crate::timing::stage("enc:rans_ac");
        // Encode EOB positions: DPCM + rANS compression.
        if let Some(dir) = std::env::var_os("AUREA_DEBUG_EOB_DUMP") {
            use std::io::Write;
            let path = std::path::Path::new(&dir).join(format!("enc_ch{ch_idx}.txt"));
            if let Ok(mut f) = std::fs::File::create(path) {
                for (i, &e) in eob_positions.iter().enumerate() {
                    let b = ch_block_buckets.get(i).copied().unwrap_or(255);
                    let _ = writeln!(f, "{i} {e} {b}");
                }
            }
            if use_progressive_packets {
                let mut bad = 0usize;
                for (i, block) in ac_by_block.iter().enumerate() {
                    if block.len() != eob_positions[i] as usize {
                        if bad < 8 {
                            eprintln!(
                                "[EOB_INVARIANT] ch={ch_idx} block={i} ac_len={} eob={}",
                                block.len(),
                                eob_positions[i]
                            );
                        }
                        bad += 1;
                    }
                }
                eprintln!("[EOB_INVARIANT] ch={ch_idx} violations={bad}/{}", ac_by_block.len());
            }
        }
        let eob_encoded = {
            let mut eob_deltas: Vec<i16> = Vec::with_capacity(n_blocks);
            let mut prev_eob: i16 = 0;
            for &eob in &eob_positions {
                eob_deltas.push(eob as i16 - prev_eob);
                prev_eob = eob as i16;
            }
            // v13: use per-block Turing buckets for EOB context (was uniform zeros)
            let eob_turing: Vec<u8> = ch_block_buckets.clone();
            let eob_encoded = rans::rans_encode_band_v12(&eob_deltas, &eob_turing);
            if !use_progressive_packets {
                body.extend_from_slice(&(n_blocks as u32).to_le_bytes());
                body.extend_from_slice(&(eob_encoded.len() as u32).to_le_bytes());
                body.extend_from_slice(&eob_encoded);
            }
            eob_encoded
        };

        // Cycle 21, lead 2: the chroma codon may also read Lphi's neighbourhood
        // envelope. Lphi is fully decoded before the chromas in every layer, so
        // the witness is causal; folding it into the existing four-state class
        // keeps the context count identical.
        let cross_codon_mode = spectral_config.cross_codon_mode;
        let mix_rate = spectral_config.mix_rate;
        let foreign_of = |block_idx: usize, position: usize| -> u8 {
            if ch_idx == 0 {
                return 0;
            }
            let block_size = if use_variable {
                l_blocks
                    .as_ref()
                    .and_then(|blocks| blocks.get(block_idx))
                    .map(|&(_, _, size)| size)
                    .unwrap_or(LOT_BLOCK_SIZE)
            } else {
                LOT_BLOCK_SIZE
            };
            let table = spectral_tables.table(SpectralNeighborTables::table_class(block_size));
            match prior_codon_blocks
                .first()
                .and_then(|blocks| blocks.get(block_idx))
            {
                Some(values) => crate::spectral_envelope::foreign_amplitude(table, position, |q| {
                    values.get(q).copied().unwrap_or(0)
                }),
                None => 0,
            }
        };
        let cross_codon = |block_idx: usize, position: usize, base: u8| -> u8 {
            if ch_idx == 0 || cross_codon_mode == crate::spectral_envelope::CROSS_CODON_NONE {
                return base;
            }
            let block_size = if use_variable {
                l_blocks
                    .as_ref()
                    .and_then(|blocks| blocks.get(block_idx))
                    .map(|&(_, _, size)| size)
                    .unwrap_or(LOT_BLOCK_SIZE)
            } else {
                LOT_BLOCK_SIZE
            };
            let table = spectral_tables.table(SpectralNeighborTables::table_class(block_size));
            let foreign = match prior_codon_blocks
                .first()
                .and_then(|blocks| blocks.get(block_idx))
            {
                Some(values) => crate::spectral_envelope::foreign_amplitude(table, position, |q| {
                    values.get(q).copied().unwrap_or(0)
                }),
                None => 0,
            };
            crate::spectral_envelope::combine_codon_class(cross_codon_mode, base, foreign)
        };

        let progressive_data = if let Some(ref packets) = progressive_packets {
            let skeleton_coeffs = crate::scan::PROGRESSIVE_SKELETON_COEFFS;
            let mut encoded_packets = Vec::with_capacity(packets.len() + 1);

            // Layer 0: a low-frequency structural blanket over the whole image.
            let skeleton_count = ac_by_block
                .iter()
                .map(|block| block.len().min(skeleton_coeffs))
                .sum::<usize>();
            let mut skeleton_symbols = Vec::with_capacity(skeleton_count);
            let mut skeleton_contexts = Vec::with_capacity(skeleton_count);
            let mut skeleton_parents = if use_causal_parent_context {
                Vec::with_capacity(skeleton_count)
            } else {
                Vec::new()
            };
            let mut skeleton_codon_classes = if use_vector_codon {
                Vec::with_capacity(skeleton_count)
            } else {
                Vec::new()
            };
            let mut skeleton_phase_predictors = if use_vector_codon {
                Vec::with_capacity(skeleton_count)
            } else {
                Vec::new()
            };
            let mut skeleton_segments = if use_spectral_envelope {
                Vec::with_capacity(ac_by_block.len())
            } else {
                Vec::new()
            };
            // Layer 0 ionizes the channel; the detail layers re-enter it.
            let mut layer0_memory = rans::EnvelopeTables::default();
            let mut skeleton_foreign = if mix_rate > 0 {
                Vec::with_capacity(skeleton_count)
            } else {
                Vec::new()
            };
            for (block_idx, block) in ac_by_block.iter().enumerate() {
                let count = block.len().min(skeleton_coeffs);
                if use_spectral_envelope && count > 0 {
                    let block_size = if use_variable {
                        l_blocks
                            .as_ref()
                            .and_then(|blocks| blocks.get(block_idx))
                            .map(|&(_, _, size)| size)
                            .unwrap_or(LOT_BLOCK_SIZE)
                    } else {
                        LOT_BLOCK_SIZE
                    };
                    skeleton_segments.push(SpectralSegment::root(
                        skeleton_symbols.len(),
                        count,
                        SpectralNeighborTables::table_class(block_size),
                    ));
                }
                skeleton_symbols.extend_from_slice(&block[..count]);
                skeleton_contexts.extend(std::iter::repeat_n(ch_ac_buckets[block_idx], count));
                if use_causal_parent_context {
                    let block_size = if use_variable {
                        l_blocks
                            .as_ref()
                            .and_then(|blocks| blocks.get(block_idx))
                            .map(|&(_, _, size)| size)
                            .unwrap_or(LOT_BLOCK_SIZE)
                    } else {
                        LOT_BLOCK_SIZE
                    };
                    let parent_positions = match block_size {
                        8 => &spectral_parents_8,
                        32 => &spectral_parents_32,
                        _ => &spectral_parents_16,
                    };
                    for position in 0..count {
                        skeleton_parents.push(
                            parent_positions[position]
                                .and_then(|parent| block.get(parent))
                                .is_some_and(|&value| value != 0),
                        );
                    }
                }
                if use_vector_codon {
                    for position in 0..count {
                        let l_value = prior_codon_blocks
                            .first()
                            .and_then(|blocks| blocks.get(block_idx))
                            .and_then(|values| values.get(position))
                            .copied()
                            .unwrap_or(0);
                        let c1_value = prior_codon_blocks
                            .get(1)
                            .and_then(|blocks| blocks.get(block_idx))
                            .and_then(|values| values.get(position))
                            .copied()
                            .unwrap_or(0);
                        skeleton_codon_classes.push(cross_codon(
                            block_idx,
                            position,
                            rans::vector_codon_class(params.quality, ch_idx, l_value, c1_value),
                        ));
                        if mix_rate > 0 {
                            skeleton_foreign.push(foreign_of(block_idx, position));
                        }
                        skeleton_phase_predictors.push(if ch_idx == 2 && c1_value != 0 {
                            i8::from(c1_value < 0)
                        } else {
                            c1_fourier_predictor(block_idx, position)
                        });
                    }
                }
            }
            let skeleton_encoded = if use_vector_codon {
                let (phase_initial, phase_shift) = if ch_idx == 1 {
                    (
                        rans::FOURIER_PHASE_INITIAL_P_SAME,
                        rans::FOURIER_PHASE_ADAPT_SHIFT,
                    )
                } else {
                    (rans::PHASE_INITIAL_P_SAME, rans::PHASE_ADAPT_SHIFT)
                };
                if use_spectral_envelope {
                    rans::rans_encode_band_v21_envelope(
                        &skeleton_symbols,
                        &skeleton_contexts,
                        &skeleton_codon_classes,
                        &skeleton_phase_predictors,
                        &skeleton_foreign,
                        &skeleton_segments,
                        &spectral_tables,
                        &spectral_config,
                        ch_idx,
                        phase_initial,
                        phase_shift,
                        use_activity_context,
                        // Layer 0 *is* the ping: there is no echo yet.
                        &[],
                        &mut layer0_memory,
                    )
                } else {
                    rans::rans_encode_band_v19_phase(
                        &skeleton_symbols,
                        &skeleton_contexts,
                        &skeleton_parents,
                        &skeleton_codon_classes,
                        &skeleton_phase_predictors,
                        phase_initial,
                        phase_shift,
                    )
                }
            } else if use_wavelet_parent_context {
                rans::rans_encode_band_v12_parent(
                    &skeleton_symbols,
                    &skeleton_contexts,
                    &skeleton_parents,
                )
            } else {
                rans::rans_encode_band_v12(&skeleton_symbols, &skeleton_contexts)
            };
            encoded_packets.push((skeleton_count, skeleton_encoded));

            // Following layers: residual AC details along fixed spiral arcs.
            // Each packet re-enters the channel from a *clone* of layer 0's
            // converged state and reads only frozen quantized data, so the
            // packets are mutually independent: they encode in parallel and
            // land in their original order. Byte-identical to the sequential
            // loop — every packet is its own rANS stream.
            let detail_packets: Vec<(usize, Vec<u8>)> = packets
                .par_iter()
                .map(|packet| {
                let symbol_count = packet
                    .iter()
                    .map(|&idx| ac_by_block[idx].len().saturating_sub(skeleton_coeffs))
                    .sum::<usize>();
                let mut packet_symbols = Vec::with_capacity(symbol_count);
                let mut packet_contexts = Vec::with_capacity(symbol_count);
                let mut packet_parents = if use_causal_parent_context {
                    Vec::with_capacity(symbol_count)
                } else {
                    Vec::new()
                };
                let mut packet_codon_classes = if use_vector_codon {
                    Vec::with_capacity(symbol_count)
                } else {
                    Vec::new()
                };
                let mut packet_phase_predictors = if use_vector_codon {
                    Vec::with_capacity(symbol_count)
                } else {
                    Vec::new()
                };
                let mut packet_segments = if use_spectral_envelope {
                    Vec::with_capacity(packet.len())
                } else {
                    Vec::new()
                };
                let mut packet_foreign = if mix_rate > 0 {
                    Vec::with_capacity(symbol_count)
                } else {
                    Vec::new()
                };
                let mut packet_memory = if use_channel_memory {
                    layer0_memory.clone()
                } else {
                    rans::EnvelopeTables::default()
                };
                let mut packet_echo = if use_echo_context {
                    Vec::with_capacity(packet.len())
                } else {
                    Vec::new()
                };
                for &block_idx in packet {
                    let start = ac_by_block[block_idx].len().min(skeleton_coeffs);
                    if use_echo_context && ac_by_block[block_idx].len() > start {
                        // The ping: this block's already-transmitted skeleton.
                        packet_echo.push(rans::skeleton_echo(&ac_by_block[block_idx][..start]));
                    }
                    if use_spectral_envelope && ac_by_block[block_idx].len() > start {
                        let block_size = if use_variable {
                            l_blocks
                                .as_ref()
                                .and_then(|blocks| blocks.get(block_idx))
                                .map(|&(_, _, size)| size)
                                .unwrap_or(LOT_BLOCK_SIZE)
                        } else {
                            LOT_BLOCK_SIZE
                        };
                        packet_segments.push(SpectralSegment::detail(
                            packet_symbols.len(),
                            start,
                            ac_by_block[block_idx].len() - start,
                            SpectralNeighborTables::table_class(block_size),
                            &ac_by_block[block_idx][..start],
                        ));
                    }
                    packet_symbols.extend_from_slice(&ac_by_block[block_idx][start..]);
                    packet_contexts.extend(std::iter::repeat_n(
                        ch_ac_buckets[block_idx],
                        ac_by_block[block_idx].len() - start,
                    ));
                    if use_causal_parent_context {
                        let block = &ac_by_block[block_idx];
                        let block_size = if use_variable {
                            l_blocks
                                .as_ref()
                                .and_then(|blocks| blocks.get(block_idx))
                                .map(|&(_, _, size)| size)
                                .unwrap_or(LOT_BLOCK_SIZE)
                        } else {
                            LOT_BLOCK_SIZE
                        };
                        let parent_positions = match block_size {
                            8 => &spectral_parents_8,
                            32 => &spectral_parents_32,
                            _ => &spectral_parents_16,
                        };
                        for position in start..block.len() {
                            packet_parents.push(
                                parent_positions[position]
                                    .and_then(|parent| block.get(parent))
                                    .is_some_and(|&value| value != 0),
                            );
                        }
                    }
                    if use_vector_codon {
                        for position in start..ac_by_block[block_idx].len() {
                            let l_value = prior_codon_blocks
                                .first()
                                .and_then(|blocks| blocks.get(block_idx))
                                .and_then(|values| values.get(position))
                                .copied()
                                .unwrap_or(0);
                            let c1_value = prior_codon_blocks
                                .get(1)
                                .and_then(|blocks| blocks.get(block_idx))
                                .and_then(|values| values.get(position))
                                .copied()
                                .unwrap_or(0);
                            packet_codon_classes.push(cross_codon(
                                block_idx,
                                position,
                                rans::vector_codon_class(params.quality, ch_idx, l_value, c1_value),
                            ));
                            if mix_rate > 0 {
                                packet_foreign.push(foreign_of(block_idx, position));
                            }
                            packet_phase_predictors.push(if ch_idx == 2 && c1_value != 0 {
                                i8::from(c1_value < 0)
                            } else {
                                c1_fourier_predictor(block_idx, position)
                            });
                        }
                    }
                }
                let encoded = if use_vector_codon {
                    let (phase_initial, phase_shift) = if ch_idx == 1 {
                        (
                            rans::FOURIER_PHASE_INITIAL_P_SAME,
                            rans::FOURIER_PHASE_ADAPT_SHIFT,
                        )
                    } else {
                        (rans::PHASE_INITIAL_P_SAME, rans::PHASE_ADAPT_SHIFT)
                    };
                    if use_spectral_envelope {
                        rans::rans_encode_band_v21_envelope(
                            &packet_symbols,
                            &packet_contexts,
                            &packet_codon_classes,
                            &packet_phase_predictors,
                            &packet_foreign,
                            &packet_segments,
                            &spectral_tables,
                            &spectral_config,
                            ch_idx,
                            phase_initial,
                            phase_shift,
                            use_activity_context,
                            &packet_echo,
                            &mut packet_memory,
                        )
                    } else {
                        rans::rans_encode_band_v19_phase(
                            &packet_symbols,
                            &packet_contexts,
                            &packet_parents,
                            &packet_codon_classes,
                            &packet_phase_predictors,
                            phase_initial,
                            phase_shift,
                        )
                    }
                } else if use_wavelet_parent_context {
                    rans::rans_encode_band_v12_parent(
                        &packet_symbols,
                        &packet_contexts,
                        &packet_parents,
                    )
                } else {
                    rans::rans_encode_band_v12(&packet_symbols, &packet_contexts)
                };
                (symbol_count, encoded)
                })
                .collect();
            if std::env::var_os("AUREA_DEBUG_EOB_DUMP").is_some() {
                for (arc_idx, (packet, (count, _))) in
                    packets.iter().zip(detail_packets.iter()).enumerate()
                {
                    let from_eobs: usize = packet
                        .iter()
                        .map(|&idx| (eob_positions[idx] as usize).saturating_sub(skeleton_coeffs))
                        .sum();
                    eprintln!(
                        "[LAYER_COUNT] ch={ch_idx} arc={arc_idx} blocks={} written={count} from_eobs={from_eobs}{}",
                        packet.len(),
                        if *count != from_eobs { "  <== MISMATCH" } else { "" },
                    );
                }
            }
            encoded_packets.extend(detail_packets);
            Some(ProgressiveChannelAc {
                eob_count: n_blocks,
                eob_encoded,
                packets: encoded_packets,
                codon_trace: if codon_stats_enabled {
                    Some(CodonChannelTrace {
                        blocks: ac_by_block.clone(),
                        eob: eob_positions,
                    })
                } else {
                    None
                },
                quantized_blocks: ac_by_block,
            })
        } else {
            // Encode the monolithic AC stream via rANS v12.
            let ac_encoded = rans::rans_encode_band_v12(&ac_flat, &ac_turing);
            body.extend_from_slice(&(ac_encoded.len() as u32).to_le_bytes());
            body.extend_from_slice(&ac_encoded);
            None
        };

        (l_rec_blocks, progressive_data)
    };

    // ------------------------------------------------------------------
    // Pass 1: Encode luma channel (ch_idx = 0)
    // ------------------------------------------------------------------
    {
        let ch = &channels_lot[0];
        // v13 stores n_blocks as u32 to support large images with adaptive blocks.
        let n_blocks_ch0 = ch.grid_h * ch.grid_w;
        body.extend_from_slice(&(n_blocks_ch0 as u32).to_le_bytes());
        body.extend_from_slice(&(dc_data[0].dc_encoded.len() as u32).to_le_bytes());
        body.extend_from_slice(&dc_data[0].dc_encoded);
        // Match section (luma only)
        body.extend_from_slice(&(match_bytes.len() as u32).to_le_bytes());
        body.extend_from_slice(&match_bytes);
    }
    let (l_rec_ac_blocks, l_progressive) = encode_channel_ac(
        &channels_lot[0],
        0,
        &mut body,
        use_cfl,
        Some(&bright_mods),
        &[],
        None,
        None,
    );
    if let Some(channel) = l_progressive {
        progressive_channels.push(channel);
    }

    // ------------------------------------------------------------------
    // Pass 2: Encode chroma channels (ch_idx = 1, 2) with CfL prediction
    // ------------------------------------------------------------------
    let mut c1_rec_ac_blocks: Vec<Vec<f64>> = Vec::new();
    for ch_idx in 1..3usize {
        let ch = &channels_lot[ch_idx];
        let n_blocks = ch.grid_h * (if use_variable { 1 } else { ch.grid_w });
        let prior_codon_blocks: Vec<&[Vec<i16>]> = progressive_channels
            .iter()
            .map(|channel| channel.quantized_blocks.as_slice())
            .collect();

        // CfL analysis: per-block decision and alpha indices (AC frequency domain)
        let t_cfl = crate::timing::stage("enc:cfl");
        let mut cfl_flags_vec: Vec<bool> = vec![false; n_blocks];
        let mut cfl_alpha_idx: Vec<usize> = vec![3; n_blocks]; // 3 = alpha 0.0 (neutral)

        if use_cfl && !l_rec_ac_blocks.is_empty() {
            for block_idx in 0..n_blocks {
                if block_idx >= l_rec_ac_blocks.len() {
                    break;
                }
                let l_rec_block = &l_rec_ac_blocks[block_idx];
                let c_block = &ch.ac_blocks[block_idx];
                // Both are in LOT AC domain (natural order, zero-mean by definition)
                let len = l_rec_block.len().min(c_block.len());
                if len == 0 {
                    continue;
                }
                let (alpha_idx, _alpha_val, use_block) =
                    cfl::estimate_alpha_ac(&l_rec_block[..len], &c_block[..len]);
                if use_block {
                    cfl_flags_vec[block_idx] = true;
                    cfl_alpha_idx[block_idx] = alpha_idx;
                }
            }
        }

        let chroma_step_mods = if let Some(ref blocks) = l_blocks {
            hierarchy::chroma_breathing_step_modulations(
                &turing_dc_grid,
                &chroma_dc_grids[ch_idx - 1],
                turing_grid_h,
                turing_grid_w,
                blocks,
                use_overlap,
                chroma_breathing_strength,
            )
        } else {
            vec![1.0; n_blocks]
        };
        let codon_rdo_context =
            if ch_idx == 2 && !l_rec_ac_blocks.is_empty() && !c1_rec_ac_blocks.is_empty() {
                Some(CodonRdoContext {
                    l_target: channels_lot[0].ac_blocks,
                    c1_target: channels_lot[1].ac_blocks,
                    c2_target: channels_lot[2].ac_blocks,
                    l_rec: &l_rec_ac_blocks,
                    c1_rec: &c1_rec_ac_blocks,
                    c2_cfl_flags: &cfl_flags_vec,
                    c2_cfl_alpha: &cfl_alpha_idx,
                })
            } else {
                None
            };
        let c1_codon_rdo_context = if ch_idx == 1 && !l_rec_ac_blocks.is_empty() {
            Some(C1CodonRdoContext {
                l_target: channels_lot[0].ac_blocks,
                c1_target: channels_lot[1].ac_blocks,
                l_rec: &l_rec_ac_blocks,
                c1_cfl_flags: &cfl_flags_vec,
                c1_cfl_alpha: &cfl_alpha_idx,
            })
        } else {
            None
        };
        drop(t_cfl);

        // Write channel header (n_blocks as u32 + DC)
        let n_blocks_ch = ch.grid_h * ch.grid_w;
        body.extend_from_slice(&(n_blocks_ch as u32).to_le_bytes());
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
                    let residual = cfl::apply_prediction(
                        &l_rec_block[..len],
                        &c_block[..len],
                        cfl_alpha_idx[block_idx],
                    );
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
            let (residual_rec, progressive) = encode_channel_ac(
                &modified_ch,
                ch_idx,
                &mut body,
                ch_idx == 1,
                Some(&chroma_step_mods),
                &prior_codon_blocks,
                c1_codon_rdo_context.as_ref(),
                codon_rdo_context.as_ref(),
            );
            drop(prior_codon_blocks);
            if ch_idx == 1 {
                c1_rec_ac_blocks = residual_rec;
                for block_idx in 0..c1_rec_ac_blocks.len().min(n_blocks) {
                    if cfl_flags_vec[block_idx] && block_idx < l_rec_ac_blocks.len() {
                        c1_rec_ac_blocks[block_idx] = cfl::reconstruct(
                            &l_rec_ac_blocks[block_idx],
                            &c1_rec_ac_blocks[block_idx],
                            cfl_alpha_idx[block_idx],
                        );
                    }
                }
            }
            if let Some(channel) = progressive {
                progressive_channels.push(channel);
            }
        } else {
            let (rec, progressive) = encode_channel_ac(
                ch,
                ch_idx,
                &mut body,
                ch_idx == 1,
                Some(&chroma_step_mods),
                &prior_codon_blocks,
                c1_codon_rdo_context.as_ref(),
                codon_rdo_context.as_ref(),
            );
            drop(prior_codon_blocks);
            if ch_idx == 1 {
                c1_rec_ac_blocks = rec;
            }
            if let Some(channel) = progressive {
                progressive_channels.push(channel);
            }
        }
    }

    if codon_stats_enabled && progressive_channels.len() == 3 {
        let current_ac_bytes = progressive_channels
            .iter()
            .flat_map(|channel| channel.packets.iter())
            .map(|(_, encoded)| encoded.len())
            .sum::<usize>();
        let traces: Vec<CodonChannelTrace> = progressive_channels
            .iter_mut()
            .filter_map(|channel| channel.codon_trace.take())
            .collect();
        let block_sizes: Vec<usize> = if let Some(ref blocks) = l_blocks {
            blocks
                .iter()
                .map(|&(_, _, block_size)| block_size)
                .collect()
        } else {
            vec![LOT_BLOCK_SIZE; traces.first().map(|trace| trace.blocks.len()).unwrap_or(0)]
        };
        emit_codon_dependency_stats(&traces, &block_sizes);
        if let Some(ref blocks) = l_blocks {
            emit_fourier_cycle_stats(&traces, blocks, progressive_packets.as_ref());
        }
        emit_dense_codon_witness(
            &traces,
            progressive_packets.as_ref(),
            &turing_buckets,
            current_ac_bytes,
        );
        emit_wavelet_parent_witness(
            &traces,
            &block_sizes,
            progressive_packets.as_ref(),
            &turing_buckets,
            current_ac_bytes,
        );
        emit_vector_codon_witness(
            &traces,
            &block_sizes,
            progressive_packets.as_ref(),
            &turing_buckets,
            current_ac_bytes,
        );
        emit_phase_codon_witness(
            &traces,
            &block_sizes,
            l_blocks.as_deref(),
            progressive_packets.as_ref(),
            &turing_buckets,
            params.quality,
            current_ac_bytes,
        );
        emit_spectral_envelope_stats(&traces, &block_sizes, &spectral_tables);
        emit_cross_channel_envelope_stats(&traces, &block_sizes, &spectral_tables, params.quality);
        emit_spectral_envelope_witness(
            &traces,
            &block_sizes,
            l_blocks.as_deref(),
            progressive_packets.as_ref(),
            &turing_buckets,
            params.quality,
            &spectral_tables,
            current_ac_bytes,
        );
    }

    // Diagnostic only (AUREA_STREAM_STATS=1): per-channel byte composition on
    // stderr. No effect whatsoever on the bitstream.
    if std::env::var("AUREA_STREAM_STATS")
        .map(|v| v == "1")
        .unwrap_or(false)
        && progressive_channels.len() == 3
    {
        eprintln!(
            "[AUREA_STREAM_STATS] spectral_exchange gamma={exchange_gamma:.4} chroma_gamma={exchange_chroma_gamma:.4} delta={exchange_delta:.4}"
        );
        eprintln!(
            "[AUREA_STREAM_STATS] chroma_diffusion repair={:.4} rms={:.4} hue_coherence={:.4} c1_factor={c1_chroma_factor:.4}",
            chroma_diffusion.repair, chroma_diffusion.chroma_rms, chroma_diffusion.hue_coherence,
        );
        eprintln!(
            "[AUREA_STREAM_STATS] luma_relief dark_fraction={:.4} gate={luma_relief_gate:.4} strength={luma_strength:.4} chroma_relief={chroma_exchange_relief:.4}",
            hierarchy::luma_dark_fraction(&turing_dc_grid),
        );
        {
            let (survivors, total) = l_ac.iter().fold((0usize, 0usize), |(s, t), block| {
                (
                    s + block.iter().filter(|c| c.abs() >= detail_step).count(),
                    t + block.len(),
                )
            });
            eprintln!(
                "[AUREA_STREAM_STATS] texture_signal survivor_fraction={:.5} (|AC| >= detail_step {detail_step:.2})",
                survivors as f64 / total.max(1) as f64,
            );
        }
        if let Some(ref blocks) = l_blocks {
            let n8 = blocks.iter().filter(|&&(_, _, bs)| bs == 8).count();
            let n16 = blocks.iter().filter(|&&(_, _, bs)| bs == 16).count();
            let n32 = blocks.iter().filter(|&&(_, _, bs)| bs == 32).count();
            eprintln!(
                "[AUREA_STREAM_STATS] blocks: total={} 8px={} 16px={} 32px={}",
                blocks.len(),
                n8,
                n16,
                n32
            );
        }
        let names = ["L", "C1", "C2"];
        for (ch_idx, channel) in progressive_channels.iter().enumerate() {
            let dc_bytes = dc_data[ch_idx].dc_encoded.len();
            let eob_bytes = channel.eob_encoded.len();
            let ac_bytes: usize = channel.packets.iter().map(|(_, e)| e.len()).sum();
            let ac_symbols: usize = channel.packets.iter().map(|(s, _)| s).sum();
            eprintln!(
                "[AUREA_STREAM_STATS] {:<2} dc={:>8} eob={:>8} ac={:>9} ({} symbols)",
                names[ch_idx], dc_bytes, eob_bytes, ac_bytes, ac_symbols
            );
        }
    }

    let t_assemble = crate::timing::stage("enc:assemble");
    if use_progressive_packets {
        if progressive_channels.len() != 3 {
            return Err("progressive AC staging did not produce three channels".into());
        }
        for channel in &progressive_channels {
            body.extend_from_slice(&(channel.eob_count as u32).to_le_bytes());
            body.extend_from_slice(&(channel.eob_encoded.len() as u32).to_le_bytes());
            body.extend_from_slice(&channel.eob_encoded);
        }
        let layer_count = progressive_channels[0].packets.len();
        if progressive_channels
            .iter()
            .any(|channel| channel.packets.len() != layer_count)
        {
            return Err("progressive AC channel layer counts disagree".into());
        }
        body.extend_from_slice(&(layer_count as u32).to_le_bytes());
        for layer_idx in 0..layer_count {
            for channel in &progressive_channels {
                let (symbol_count, encoded) = &channel.packets[layer_idx];
                body.extend_from_slice(&(*symbol_count as u32).to_le_bytes());
                body.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
                body.extend_from_slice(encoded);
            }
        }
    }

    if flags & FLAG_CHROMA_RESIDUAL != 0 {
        let sat_map = color::saturation_map(&c1_ch, &c2_ch, height, width);
        let chroma_resid_step = detail_step * calibration::CHROMA_RESIDUAL_STEP_MULT;

        let (c1_mask, c1_resid) = color::encode_chroma_residual(
            &c1_ch,
            &c1_sub,
            &sat_map,
            height,
            width,
            c1_h,
            c1_w,
            chroma_resid_step,
        );
        body.extend_from_slice(&(c1_mask.len() as u32).to_le_bytes());
        body.extend_from_slice(&c1_mask);
        body.extend_from_slice(&(c1_resid.len() as u32).to_le_bytes());
        for &v in &c1_resid {
            body.push(v as u8);
        }
    }

    if !use_vector_codon {
        return Err(
            "AUR2 v20 requires spectral exchange and progressive phase-codon packets".into(),
        );
    }

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
    drop(t_assemble);
    crate::timing::report("encode");

    Ok(AureaEncoderResult {
        compressed_size: aurea_data.len(),
        aurea_data,
    })
}

pub fn encode_unified(
    rgb: &[u8],
    width: usize,
    height: usize,
    params: &crate::codec_params::CodecParams,
) -> Result<AureaEncoderResult, Box<dyn std::error::Error>> {
    let p = AureaEncoderParams {
        quality: params.quality,
        n_representatives: quality_to_n_repr(params.quality),
        geometric: false,
        lot_topology: params.lot_topology,
    };
    encode_aur2_v13(rgb, width, height, &p)
}
