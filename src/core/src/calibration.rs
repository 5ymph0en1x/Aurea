//! Calibrated constants from 12 HD reference images (Point 8).
//! All empirical thresholds and factors centralized here for systematic tuning.

// ======================================================================
// Codon 3D/4D calibrated thresholds
// ======================================================================

/// Luminance zone thresholds (PTF space, 0-255).
/// Calibrated for optimal Weber-Fechner zones on varied content.
pub const CODON_LUM_THRESHOLDS: [f64; 3] = [64.0, 128.0, 192.0];

/// tRNA step factors per luminance zone.
/// Zone 0 (dark): finest quantization (Weber sensitivity highest)
/// Zone 3 (bright): coarsest (less perceptible)
pub const CODON_TRNA: [f64; 4] = [0.3, 0.65, 1.0, 1.35];

/// Saturation energy threshold for chroma-adaptive codon.
pub const CODON_SAT_THRESHOLD: f64 = 20.0;

/// Saturation factor (15% finer for saturated blocks).
pub const CODON_SAT_FACTOR: f64 = 0.85;

/// Detail AC energy threshold per coefficient.
pub const CODON_DETAIL_THRESHOLD: f64 = 6.0;

/// Detail factor (15% finer for highly detailed blocks).
pub const CODON_DETAIL_FACTOR: f64 = 0.85;

/// Structural coherence factor range (finer for coherent structures).
pub const CODON_STRUCT_FACTOR_MIN: f64 = 0.80;

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
