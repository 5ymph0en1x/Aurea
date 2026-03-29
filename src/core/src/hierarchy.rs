//! Orchestration module for the 4-level Bayesian predictive hierarchy (v12).
//!
//! Level 0: DC grid (existing, unchanged)
//! Level 1: Turing morphogenesis field (zero-bit, reconstructed from DC)
//! Level 2: Primitive matching (only surprises encoded)
//! Level 3: rANS with Bayesian priors (enriched context model)

use crate::bitstream::TuringHeader;
use crate::geometric::{self, Primitive, PrimitiveMatch};
use crate::turing::{self, TuringField, PredictedContour};

/// Parameters for v12 hierarchical encoding, derived from the TuringHeader.
pub struct HierarchyParams {
    pub sigma_a: f64,
    pub sigma_ratio: f64,
    pub k_step: f64,
    pub mag_calibration: f64,
}

impl From<&TuringHeader> for HierarchyParams {
    fn from(hdr: &TuringHeader) -> Self {
        Self {
            sigma_a: hdr.sigma_a as f64,
            sigma_ratio: hdr.sigma_ratio as f64,
            k_step: hdr.k_step as f64,
            mag_calibration: hdr.mag_calibration as f64,
        }
    }
}

impl Default for HierarchyParams {
    fn default() -> Self {
        Self::from(&TuringHeader::default_params())
    }
}

/// Level 1: Compute Turing field from decoded DC grid.
/// Called identically on encoder and decoder.
pub fn compute_level1(
    decoded_dc: &[f64],
    grid_h: usize,
    grid_w: usize,
    params: &HierarchyParams,
) -> TuringField {
    turing::compute_turing_field(
        decoded_dc, grid_h, grid_w,
        params.sigma_a, params.sigma_ratio, params.k_step,
    )
}

/// Level 1 contour extraction.
pub fn extract_predicted_contours(tf: &TuringField) -> Vec<PredictedContour> {
    turing::trace_ridges(&tf.field, tf.grid_h, tf.grid_w, Some(&tf.inhibitor_gradient))
}

/// Level 2: Match extracted primitives against Turing predictions.
pub fn match_level2(
    primitives: &[Primitive],
    contours: &[PredictedContour],
    block_size: usize,
) -> Vec<PrimitiveMatch> {
    geometric::match_primitives(primitives, contours, block_size)
}

/// Build turing_buckets array for Level 3 rANS encoding.
/// Maps each coefficient position (in raster order) to a turing bucket (0-3).
pub fn build_turing_buckets(
    tf: &TuringField,
    coeff_h: usize,
    coeff_w: usize,
    block_size: usize,
) -> Vec<u8> {
    let scale = (block_size / 2).max(1); // detail bands are half-resolution
    let mut buckets = Vec::with_capacity(coeff_h * coeff_w);
    for y in 0..coeff_h {
        for x in 0..coeff_w {
            let gy = (y / scale).min(tf.grid_h.saturating_sub(1));
            let gx = (x / scale).min(tf.grid_w.saturating_sub(1));
            let t = tf.field[gy * tf.grid_w + gx];
            buckets.push(turing::turing_bucket(t) as u8);
        }
    }
    buckets
}

/// Build per-block gradient angle and strength from the Turing inhibitor field.
/// Returns `(angle_radians, normalized_strength)` per block.
/// `strength` is `inhibitor_gradient / max_gradient`, in [0, 1].
pub fn build_gradient_angles(
    tf: &TuringField,
    n_blocks_h: usize,
    n_blocks_w: usize,
    block_size: usize,
) -> Vec<(f64, f64)> {
    let n_blocks = n_blocks_h * n_blocks_w;
    let max_grad = tf.inhibitor_gradient.iter().cloned()
        .fold(0.0f64, f64::max)
        .max(1e-6);

    let mut result = Vec::with_capacity(n_blocks);
    for by in 0..n_blocks_h {
        for bx in 0..n_blocks_w {
            let gy = by.min(tf.grid_h.saturating_sub(1));
            let gx = bx.min(tf.grid_w.saturating_sub(1));
            let idx = gy * tf.grid_w + gx;
            let angle = tf.inhibitor_gradient_angle[idx];
            let strength = tf.inhibitor_gradient[idx] / max_grad;
            result.push((angle, strength));
        }
    }
    result
}

/// Build predicted magnitudes array for Level 3 residual coding.
pub fn build_pred_magnitudes(
    tf: &TuringField,
    coeff_h: usize,
    coeff_w: usize,
    block_size: usize,
    mag_calibration: f64,
) -> Vec<f64> {
    let scale = (block_size / 2).max(1);
    let mut preds = Vec::with_capacity(coeff_h * coeff_w);
    for y in 0..coeff_h {
        for x in 0..coeff_w {
            let gy = (y / scale).min(tf.grid_h.saturating_sub(1));
            let gx = (x / scale).min(tf.grid_w.saturating_sub(1));
            let gi = tf.inhibitor_gradient[gy * tf.grid_w + gx];
            preds.push(turing::predicted_magnitude(gi, mag_calibration));
        }
    }
    preds
}
