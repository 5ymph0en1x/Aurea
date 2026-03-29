//! Chroma-from-Luma (CfL) prediction module — AC frequency domain.
//!
//! LOT is linear, so: LOT(C - alpha*L) = LOT(C) - alpha*LOT(L).
//! Prediction and subtraction happen directly on AC coefficients.
//! AC coefficients have zero mean by definition (DC is separate),
//! so no mean subtraction is needed in the regression.
//!
//! alpha is quantized on 3 bits (8 levels).

/// Quantized alpha palette: 8 values covering [-0.75, 1.0].
pub const ALPHA_PALETTE: [f64; 8] = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];

/// Find closest palette entry for a raw alpha value.
/// Returns (palette_index, quantized_alpha).
pub fn quantize_alpha(alpha: f64) -> (usize, f64) {
    let mut best_idx = 3usize; // 0.0 default
    let mut min_err = f64::MAX;
    for (i, &val) in ALPHA_PALETTE.iter().enumerate() {
        let err = (alpha - val).abs();
        if err < min_err {
            min_err = err;
            best_idx = i;
        }
    }
    (best_idx, ALPHA_PALETTE[best_idx])
}

/// Estimate CfL alpha directly in the LOT AC frequency domain.
/// Returns (alpha_idx, alpha_val, use_cfl).
///
/// `l_ac_rec`: reconstructed (dequantized) luma AC for this block
/// `c_ac_orig`: original chroma AC for this block
///
/// Both are in LOT frequency domain (natural order), zero-indexed (AC only, no DC).
/// Since AC coefficients are zero-mean by definition, we skip mean subtraction
/// and compute a direct least-squares regression: alpha = sum(L*C) / sum(L*L).
///
/// An R^2 > 0.25 (|R| > 0.5) correlation test gates the decision.
pub fn estimate_alpha_ac(l_ac_rec: &[f64], c_ac_orig: &[f64]) -> (usize, f64, bool) {
    let n = l_ac_rec.len().min(c_ac_orig.len());
    if n == 0 {
        return (3, 0.0, false);
    }

    let mut sum_l2 = 0.0;
    let mut sum_c2 = 0.0;
    let mut sum_lc = 0.0;
    for i in 0..n {
        let l = l_ac_rec[i];
        let c = c_ac_orig[i];
        sum_l2 += l * l;
        sum_c2 += c * c;
        sum_lc += l * c;
    }

    // Guard: skip if either channel has negligible energy
    if sum_l2 < 1e-4 || sum_c2 < 1e-4 {
        return (3, 0.0, false);
    }

    // R^2 test: only use CfL if |R| > 0.5 (R^2 > 0.25)
    let r_squared = (sum_lc * sum_lc) / (sum_l2 * sum_c2);
    if r_squared < 0.25 {
        return (3, 0.0, false);
    }

    let raw_alpha = sum_lc / sum_l2;
    let (idx, q_alpha) = quantize_alpha(raw_alpha);

    if q_alpha.abs() < 0.01 {
        return (3, 0.0, false);
    }

    (idx, q_alpha, true)
}

/// Apply CfL prediction: subtract alpha * L_rec from chroma block.
/// Returns the residual (in AC frequency domain).
pub fn apply_prediction(l_rec_block: &[f64], c_block: &[f64], alpha_idx: usize) -> Vec<f64> {
    let alpha = ALPHA_PALETTE[alpha_idx.min(7)];
    c_block.iter().zip(l_rec_block.iter())
        .map(|(&c, &l)| c - alpha * l)
        .collect()
}

/// Reconstruct chroma from residual + alpha * L_rec (decoder side).
/// Both residual and l_rec_block are in AC frequency domain.
pub fn reconstruct(l_rec_block: &[f64], residual: &[f64], alpha_idx: usize) -> Vec<f64> {
    let alpha = ALPHA_PALETTE[alpha_idx.min(7)];
    residual.iter().zip(l_rec_block.iter())
        .map(|(&r, &l)| r + alpha * l)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_alpha_ac_correlated() {
        // AC coefficients (zero-mean): perfectly correlated with alpha=0.5
        let l = vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0];
        let c = vec![0.5, -0.5, 1.0, -1.0, 1.5, -1.5];
        let (idx, alpha, use_cfl) = estimate_alpha_ac(&l, &c);
        assert_eq!(ALPHA_PALETTE[idx], 0.5);
        assert!((alpha - 0.5).abs() < 1e-10);
        assert!(use_cfl, "well-correlated blocks should use CfL");
    }

    #[test]
    fn test_estimate_alpha_ac_uncorrelated() {
        // Orthogonal signals: no correlation
        let l = vec![1.0, 0.0, -1.0, 0.0];
        let c = vec![0.0, 1.0, 0.0, -1.0];
        let (idx, _alpha, use_cfl) = estimate_alpha_ac(&l, &c);
        assert_eq!(ALPHA_PALETTE[idx], 0.0);
        assert!(!use_cfl, "uncorrelated blocks should not use CfL");
    }

    #[test]
    fn test_estimate_alpha_ac_negligible_energy() {
        let l = vec![0.0, 0.0, 0.0, 0.0];
        let c = vec![1.0, -1.0, 0.5, -0.5];
        let (_idx, _alpha, use_cfl) = estimate_alpha_ac(&l, &c);
        assert!(!use_cfl, "negligible luma energy should skip CfL");
    }

    #[test]
    fn test_quantize_alpha() {
        let (idx, val) = quantize_alpha(0.6);
        assert_eq!(idx, 5); // closest to 0.5
        assert_eq!(val, 0.5);

        let (idx, val) = quantize_alpha(-0.6);
        assert_eq!(idx, 1); // closest to -0.5
        assert_eq!(val, -0.5);

        let (idx, val) = quantize_alpha(0.0);
        assert_eq!(idx, 3); // exact 0.0
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_roundtrip_ac() {
        // AC-domain roundtrip
        let l = vec![10.0, -10.0, 5.0, -5.0, 3.0, -3.0];
        let c = vec![5.0, -5.0, 2.5, -2.5, 1.5, -1.5]; // alpha=0.5
        let (idx, _alpha, use_cfl) = estimate_alpha_ac(&l, &c);
        assert!(use_cfl);
        let residual = apply_prediction(&l, &c, idx);
        let recon = reconstruct(&l, &residual, idx);
        for i in 0..c.len() {
            assert!((recon[i] - c[i]).abs() < 1e-10, "CfL AC roundtrip mismatch at {i}");
        }
    }
}
