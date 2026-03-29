//! Calibrated constants from 12 HD reference images (Point 8).
//! All empirical thresholds and factors centralized here for systematic tuning.
//!
//! In-the-loop optimization: all tunable constants can be overridden via env vars
//! (prefix AUREA_). When unset, compiled defaults are used.

use std::sync::LazyLock;

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(default)
}

// ======================================================================
// In-the-loop tunable parameters (LazyLock — read env once at startup)
// ======================================================================

/// tRNA step factors: [dark, mid-dark, mid-bright(=1.0 ref), bright]
pub static TUNABLE_TRNA: LazyLock<[f64; 4]> = LazyLock::new(|| [
    env_f64("AUREA_TRNA_DARK", 0.7),
    env_f64("AUREA_TRNA_MIDDARK", 0.9),
    1.0,
    env_f64("AUREA_TRNA_BRIGHT", 1.2),
]);

/// qmat_power_for_quality parameters: [low_plateau, high_floor, q_transition, q_range]
pub static TUNABLE_POWER: LazyLock<[f64; 4]> = LazyLock::new(|| [
    env_f64("AUREA_POWER_LOW", 0.55),
    env_f64("AUREA_POWER_HIGH", 0.05),
    env_f64("AUREA_POWER_QTRANS", 60.0),
    env_f64("AUREA_POWER_QRANGE", 40.0),
]);

/// Psychovisual Turing pivot: [gamma_low, gamma_high]
pub static TUNABLE_PIVOT: LazyLock<[f64; 2]> = LazyLock::new(|| [
    env_f64("AUREA_PIVOT_GAMMA_LOW", 1.25),
    env_f64("AUREA_PIVOT_GAMMA_HIGH", -0.85),
]);

/// Foveal saliency exponent
pub static TUNABLE_FOVEAL_K: LazyLock<f64> = LazyLock::new(||
    env_f64("AUREA_FOVEAL_K", 1.5)
);

/// Gradient strength threshold for QMAT rotation (0-1, normalized).
/// Below this, blocks use the standard (non-rotated) QMAT.
/// Combined with Trellis RDO (lambda=5): +0.35dB for ~+1% bpp net.
/// Trellis absorbs the bpp cost, rotation adds the quality.
pub static TUNABLE_GRADIENT_THRESHOLD: LazyLock<f64> = LazyLock::new(||
    env_f64("AUREA_GRADIENT_THRESHOLD", 0.15)
);

/// Trellis RDO lambda scale factor (higher = more aggressive zeroing).
/// Calibrated on phoenix q=50: 5.0 gives -0.73% bpp for -0.01 dB PSNR.
pub static TUNABLE_TRELLIS_LAMBDA: LazyLock<f64> = LazyLock::new(||
    env_f64("AUREA_TRELLIS_LAMBDA", 5.0)
);

/// Compute the Lagrange multiplier for trellis RDO.
/// Based on λ ~ step² / (12·ln2) with quality-adaptive scaling.
/// Low quality: aggressive RDO (λ_scale=1.5), high quality: gentle (λ_scale=0.3).
pub fn trellis_lambda(quality: u8, detail_step: f64) -> f64 {
    let base = detail_step * detail_step / 8.33; // 12 * ln(2) ≈ 8.33
    let q = quality as f64;
    let q_scale = if q <= 50.0 {
        1.5
    } else if q >= 90.0 {
        0.3
    } else {
        let t = ((q - 50.0) / 40.0).clamp(0.0, 1.0);
        let s = t * t * (3.0 - 2.0 * t); // smoothstep
        1.5 * (1.0 - s) + 0.3 * s
    };
    (base * q_scale * *TUNABLE_TRELLIS_LAMBDA).max(0.01)
}

// ======================================================================
// Codon 3D/4D calibrated thresholds
// ======================================================================

/// Luminance zone thresholds (PTF space, 0-255).
/// Calibrated for optimal Weber-Fechner zones on varied content.
pub const CODON_LUM_THRESHOLDS: [f64; 3] = [64.0, 128.0, 192.0];

/// tRNA step factors per luminance zone.
/// Zone 0 (dark): finest quantization (Weber sensitivity highest)
/// Zone 3 (bright): coarsest (less perceptible)
/// Ratio 3:1 matches human Weber fraction (~3% dark vs ~1% bright JND).
/// Recalibrated after DC-space fix: old [0.3, 0.65, 1.0, 1.35] had 4.5× ratio
/// but was inoperant (all blocks at 1.35). Now active, 3× ratio is perceptually correct.
/// Tightened from [0.5, 0.75, 1.0, 1.5] — the 3:1 ratio was too aggressive,
/// causing dark areas to consume disproportionate bits for marginal PSNR gain.
/// Tunable via AUREA_TRNA_DARK, AUREA_TRNA_MIDDARK, AUREA_TRNA_BRIGHT.
pub const CODON_TRNA: [f64; 4] = [0.7, 0.9, 1.0, 1.2];

/// Saturation energy threshold for chroma-adaptive codon.
pub const CODON_SAT_THRESHOLD: f64 = 20.0;

/// Saturation factor (15% finer for saturated blocks).
pub const CODON_SAT_FACTOR: f64 = 0.85;

/// Detail AC energy threshold per coefficient.
pub const CODON_DETAIL_THRESHOLD: f64 = 6.0;

/// Detail factor (15% finer for highly detailed blocks).
pub const CODON_DETAIL_FACTOR: f64 = 0.85;

/// Structural codon factor range (DC gradient-based).
/// MIN = structural (preserve, fine quantization). MAX = smooth (compress, coarse).
/// Ratio MAX/MIN = 1.625 — gentle adaptation that doesn't destroy bright textures.
pub const CODON_STRUCT_FACTOR_MIN: f64 = 0.80;
pub const CODON_STRUCT_FACTOR_MAX: f64 = 1.30;

// ======================================================================
// CSF (Contrast Sensitivity Function) parameters (Point 5)
// ======================================================================

/// Dark-frequency interaction boost.
/// In dark regions, HF is less visible -> coarser quantization.
/// csf_factor = 1 + CSF_DARK_BOOST * (1 - lum_norm) * freq_norm^2
pub const CSF_DARK_BOOST: f64 = 0.5;

// ======================================================================
// Deblocking thresholds
// ======================================================================

/// Gas detection: interior gradient below this = smooth region.
pub const DEBLOCK_GAS_THRESHOLD: f64 = 4.0;

/// Anti-ring sigma edge detection threshold.
pub const ANTI_RING_EDGE_THRESHOLD: f64 = 6.0;

// ======================================================================
// Rate control (Point 6)
// ======================================================================

/// Maximum binary search iterations for rate control.
pub const RATE_CONTROL_MAX_ITER: usize = 8;

/// Convergence tolerance (5% of target bpp).
pub const RATE_CONTROL_TOLERANCE: f64 = 0.05;

// ======================================================================
// Scene classification thresholds (Point 7)
// ======================================================================

/// Smooth percentage for Flat classification.
pub const SCENE_SMOOTH_FLAT: f64 = 50.0;

/// Smooth percentage below which = Organic.
pub const SCENE_SMOOTH_ORGANIC: f64 = 15.0;

/// DC gradient threshold for smooth pixel detection.
pub const SCENE_GRAD_SMOOTH: f64 = 3.0;

// ======================================================================
// Chroma residual (Point 3)
// ======================================================================

/// Block size for chroma residual mask (8x8).
pub const CHROMA_RESIDUAL_BLOCK: usize = 8;

/// Chroma residual quantization step multiplier (relative to detail_step).
pub const CHROMA_RESIDUAL_STEP_MULT: f64 = 0.5;

// ======================================================================
// Variable blocks (Point 2)
// ======================================================================

/// Variance merge threshold = step^2 * this factor.
pub const VARIABLE_BLOCK_MERGE_FACTOR: f64 = 0.25;

/// Minimum percentage of non-16 blocks to justify variable mode.
pub const VARIABLE_BLOCK_MIN_PCT: f64 = 10.0;

// ======================================================================
// Quality-adaptive parameters (DNA4)
// ======================================================================

/// LOT global AC step multiplier.
/// Recalibrated for structural codon (DC gradient, range [0.8, 1.3]).
/// Old: 3.8 * 1.35 (broken luminance codon, all blocks at 1.35) = effective 5.13.
/// New: 4.9 * avg_structural(~1.05) = effective ~5.15 ≈ same bpp budget.
/// The structural codon preserves edges/textures, compresses smooth regions.
pub fn lot_factor_for_quality(_quality: u8) -> f64 {
    4.9
}

/// Dead zone threshold — quality-adaptive, with frequency-dependent override.
/// Use `dead_zone_for_quality` for the base dead zone (low/mid frequencies).
/// Use `dead_zone_for_position` for per-coefficient dead zone (adds HF floor).
pub fn dead_zone_for_quality(quality: u8) -> f64 {
    let q = quality as f64;
    if q <= 70.0 {
        DEAD_ZONE_CONSTANT // 0.22 — full dead zone for compression
    } else {
        // Linear ramp from 0.22 at Q=70 to 0.02 at Q=100
        let t = ((q - 70.0) / 30.0).clamp(0.0, 1.0);
        DEAD_ZONE_CONSTANT * (1.0 - t) + 0.02 * t
    }
}

/// Frequency-dependent dead zone: keeps a floor on the last quarter of the
/// zigzag scan (very high frequencies = sensor noise) even at high quality.
/// `zigzag_pos`: 0-based position in zigzag order
/// `ac_per_block`: total AC coefficients in the block (e.g. 255 for 16x16)
/// `base_dz`: the quality-adaptive base dead zone from `dead_zone_for_quality`
#[inline]
pub fn dead_zone_for_position(zigzag_pos: usize, ac_per_block: usize, base_dz: f64) -> f64 {
    let frac = zigzag_pos as f64 / ac_per_block.max(1) as f64;
    if frac > 0.75 {
        base_dz.max(0.20)
    } else if frac > 0.5 {
        base_dz.max(0.10)
    } else {
        base_dz
    }
}

/// The calibrated dead zone constant (matches wavelet::DEAD_ZONE).
pub const DEAD_ZONE_CONSTANT: f64 = 0.22;

/// QMAT power exponent — quality-adaptive.
/// At low quality, 0.55 penalizes HF heavily (saves bits).
/// At high quality, flattens toward 0.0 (all frequencies equal = max PSNR).
pub fn qmat_power_for_quality(quality: u8) -> f64 {
    let [p_low, p_high, q_trans, q_range] = *TUNABLE_POWER;
    let q = quality as f64;
    if q <= q_trans {
        p_low
    } else {
        let t = ((q - q_trans) / q_range).clamp(0.0, 1.0);
        let s = t * t * (3.0 - 2.0 * t); // smoothstep
        p_low * (1.0 - s) + p_high * s
    }
}

/// Spectral spin median blend weight, decreasing with quality.
/// q=20 → 0.38 (strong correction), q=95 → 0.06 (minimal).
pub fn spin_weight_for_quality(quality: u8) -> f64 {
    let q = (quality as f64 / 100.0).clamp(0.01, 1.0);
    0.05 + 0.33 * (1.0 - q)
}

// --- Turing Morphogenesis (v12 Bayesian Hierarchy) ---

/// Activator Gaussian sigma in DC-grid units (blocks).
/// Small σ captures local edge structure.
pub const TURING_SIGMA_A: f64 = 1.5;

/// Ratio σ_inhibitor / σ_activator = φ².
/// Yields Turing wavelength λ_T ∝ φ² ≈ 2.6 blocks.
pub const TURING_SIGMA_RATIO: f64 = 2.618033988749895; // PHI * PHI

/// Step modulation exponent: step_factor = φ^(-k * T_norm).
/// k=0.5 → crests get step / φ^0.5 ≈ /1.27 (finer), valleys unchanged.
pub const TURING_K_STEP: f64 = 0.5;

/// Minimum T_norm to seed a ridge trace.
pub const TURING_RIDGE_THRESHOLD: f64 = 0.15;

/// Maximum Hausdorff distance (in DC-grid units) for primitive matching.
pub const TURING_MATCH_DISTANCE_MAX: f64 = 2.0;

/// Search radius for candidate contour matching.
pub const TURING_MATCH_RADIUS: f64 = 4.0;

/// Primitive classified as Predicted if all residuals below these thresholds.
pub const TURING_SURPRISE_POS: f64 = 0.5;         // blocks
pub const TURING_SURPRISE_ANGLE: f64 = 0.392699;  // π/8
pub const TURING_SURPRISE_AMP: f64 = 2.0;         // Fibonacci levels

/// Magnitude prediction calibration factor (initial, tuned on Kodak).
pub const TURING_MAG_CALIBRATION: f64 = 1.0;

/// QMAT parametric constants (Point 2: variable blocks)
/// Formula: Q = Q_A · d^Q_BETA + Q_GAMMA · r · c
/// avec d = sqrt(r² + Q_ALPHA·c²)   et (r, c) ∈ [0,1]²
///
/// Single source of truth pour lot::qmat_for_block_size.
/// LPIPS-calibrées sur le corpus Kodak + 12 images 2K personnelles.
pub const Q_A: f64     = 5.0 * std::f64::consts::PI + 2.0;
pub const Q_ALPHA: f64 = 11.0 / 13.0;
pub const Q_BETA: f64  = 5.0 * crate::golden::PHI / 9.0;
pub const Q_GAMMA: f64 = 176.0 * std::f64::consts::PI / 35.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lot_factor_constant() {
        // 5.5: recalibrated for working Weber-Fechner codon (avg tRNA ~0.94)
        // 5.5 * 0.94 ≈ 5.16 ≈ old 3.8 * 1.35 (when codon was broken at 1.35)
        for q in [1u8, 20, 50, 75, 95, 100] {
            assert!((lot_factor_for_quality(q) - 4.9).abs() < 1e-10,
                "lot_factor must be 4.9 at q={}", q);
        }
    }

    #[test]
    fn test_dead_zone_quality_adaptive() {
        // Low quality: full dead zone 0.22
        assert!((dead_zone_for_quality(50) - 0.22).abs() < 1e-10);
        assert!((dead_zone_for_quality(70) - 0.22).abs() < 1e-10);
        // High quality: collapses toward 0.02
        assert!(dead_zone_for_quality(85) < 0.15);
        assert!(dead_zone_for_quality(100) < 0.05);
        // Monotonically decreasing above Q=70
        let mut prev = dead_zone_for_quality(70);
        for q in (75..=100).step_by(5) {
            let dz = dead_zone_for_quality(q);
            assert!(dz <= prev, "dead_zone must decrease: q={} dz={} prev={}", q, dz, prev);
            prev = dz;
        }
    }

    #[test]
    fn test_qmat_power_quality_adaptive() {
        // Low quality: full HF penalization 0.55
        assert!((qmat_power_for_quality(50) - 0.55).abs() < 1e-10);
        assert!((qmat_power_for_quality(60) - 0.55).abs() < 1e-10);
        // High quality: flattens toward 0.05
        assert!(qmat_power_for_quality(80) < 0.40);
        assert!(qmat_power_for_quality(100) < 0.10);
        // Monotonically decreasing above Q=60
        let mut prev = qmat_power_for_quality(60);
        for q in (65..=100).step_by(5) {
            let qp = qmat_power_for_quality(q);
            assert!(qp <= prev, "qmat_power must decrease: q={} qp={} prev={}", q, qp, prev);
            prev = qp;
        }
    }

    #[test]
    fn test_spin_weight_monotonic_decreasing() {
        // Spin weight is the ONLY quality-adaptive encoder/decoder param
        // Strong correction at low q, minimal at high q
        let mut prev = spin_weight_for_quality(1);
        for q in (10..=100).step_by(10) {
            let w = spin_weight_for_quality(q);
            assert!(w <= prev, "spin_weight must decrease: q={} w={} prev={}", q, w, prev);
            prev = w;
        }
    }

    #[test]
    fn test_spin_weight_bounds() {
        assert!(spin_weight_for_quality(1) <= 0.40);
        assert!(spin_weight_for_quality(100) >= 0.04);
    }
}
