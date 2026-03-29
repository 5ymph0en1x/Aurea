//! Trellis quantization (Rate-Distortion Optimization) for AUREA LOT encoder.
//!
//! Uses the Viterbi algorithm to find the optimal sequence of quantization
//! decisions per block, minimizing J = D + λ·R where:
//!   - D = sum of squared quantization errors (transform domain)
//!   - R = estimated entropy cost from the rANS v12 context model
//!
//! The decoder is completely unchanged — trellis only affects the encoder's
//! quantization decisions within the existing bitstream format.

// ======================================================================
// Constants (mirrored from rans.rs to avoid coupling)
// ======================================================================

const PROB_BITS: u32 = 14;
const PROB_SCALE: u32 = 1 << PROB_BITS; // 16384
const ADAPT_SHIFT: u32 = 4;
const ENERGY_ALPHA: u32 = 218;
const ENERGY_SCALE: u32 = 256;
const P_ZERO_INIT: u32 = 14000;
const P_PM1_INIT: u32 = 12000;

/// Max active states per trellis stage (pruning bound).
const MAX_STATES: usize = 16;
/// Reduced max states for large blocks (32×32).
const MAX_STATES_LARGE: usize = 8;
/// Minimum non-zero count in greedy to bother with trellis.
const MIN_NZ_FOR_TRELLIS: usize = 3;

// ======================================================================
// Precomputed log₂ LUT: LOG2_LUT[p] = -log₂(p / PROB_SCALE) for p in [0, PROB_SCALE]
// ======================================================================

use std::sync::LazyLock;

static LOG2_LUT: LazyLock<Vec<f64>> = LazyLock::new(|| {
    let mut lut = vec![0.0f64; (PROB_SCALE + 1) as usize];
    lut[0] = 20.0; // sentinel for p=0 (should never be used)
    for p in 1..=PROB_SCALE {
        lut[p as usize] = -(p as f64 / PROB_SCALE as f64).log2();
    }
    lut
});

/// Cost in bits of coding a symbol with probability p/PROB_SCALE.
#[inline]
fn bits_from_prob(p: u32) -> f64 {
    LOG2_LUT[p.clamp(1, PROB_SCALE - 1) as usize]
}

/// Cost in bits of coding a symbol with probability (PROB_SCALE - p)/PROB_SCALE.
#[inline]
fn bits_from_complement(p: u32) -> f64 {
    bits_from_prob(PROB_SCALE - p.clamp(1, PROB_SCALE - 1))
}

// ======================================================================
// Context model functions (mirrored from rans.rs)
// ======================================================================

#[inline]
fn run_class(run_zeros: u32) -> u8 {
    match run_zeros {
        0 => 0,
        1 => 1,
        2 | 3 => 2,
        _ => 3,
    }
}

#[inline]
fn energy_bucket(energy: u32) -> u8 {
    if energy < 16 { 0 }
    else if energy < 64 { 1 }
    else if energy < 160 { 2 }
    else { 3 }
}

/// Representative energy value for each bucket (midpoint, used for IIR approximation).
#[inline]
fn bucket_representative(eb: u8) -> u32 {
    match eb {
        0 => 8,
        1 => 40,
        2 => 112,
        _ => 208,
    }
}

// ======================================================================
// Trellis state
// ======================================================================

/// Compact trellis state tracking the rANS v12 context model variables.
/// Packed into 16 bits for efficient indexing.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TState {
    run_zeros: u8,      // 0-4 (capped at 4 = "4+" bucket)
    prev_nz: bool,
    energy_bucket: u8,  // 0-3
    prev_was_pm1: bool,
}

impl TState {
    fn initial() -> Self {
        Self {
            run_zeros: 0,
            prev_nz: false,
            energy_bucket: 0,
            prev_was_pm1: false,
        }
    }

    /// Pack into a u16 index for array-based storage.
    /// Layout: run(3 bits) | prev_nz(1) | energy(2) | prev_pm1(1) = 7 bits → max 128
    #[inline]
    fn pack(&self) -> usize {
        let r = self.run_zeros.min(4) as usize;
        let pnz = if self.prev_nz { 1 } else { 0 };
        let eb = self.energy_bucket.min(3) as usize;
        let pp = if self.prev_was_pm1 { 1 } else { 0 };
        (r << 4) | (pnz << 3) | (eb << 1) | pp
    }

    /// Compute P(zero) context index matching ContextModelV12::zero_ctx_index().
    #[inline]
    fn zero_ctx(&self, turing_bucket: u8) -> usize {
        let rc = run_class(self.run_zeros as u32) as usize;
        let pnz = if self.prev_nz { 1 } else { 0 };
        let eb = self.energy_bucket as usize;
        let tb = turing_bucket.min(3) as usize;
        rc * 32 + pnz * 16 + eb * 4 + tb
    }

    /// Compute P(±1|nz) context index matching ContextModelV12::pm1_ctx_index().
    #[inline]
    fn pm1_ctx(&self, turing_bucket: u8) -> usize {
        let eb = self.energy_bucket as usize;
        let pp = if self.prev_was_pm1 { 1 } else { 0 };
        let tb = turing_bucket.min(3) as usize;
        eb * 8 + pp * 4 + tb
    }

    /// Transition to new state after observing quantized value q.
    #[inline]
    fn transition(&self, q: i16) -> Self {
        let is_zero = q == 0;
        let abs_val = q.unsigned_abs() as u32;

        // Energy IIR approximation using bucket representatives
        let sample_energy = (abs_val * 32).min(ENERGY_SCALE);
        let approx_energy = (ENERGY_ALPHA * bucket_representative(self.energy_bucket) as u32
            + (ENERGY_SCALE - ENERGY_ALPHA) * sample_energy)
            / ENERGY_SCALE;
        let new_eb = energy_bucket(approx_energy);

        if is_zero {
            TState {
                run_zeros: (self.run_zeros + 1).min(4),
                prev_nz: false,
                energy_bucket: new_eb,
                prev_was_pm1: self.prev_was_pm1,
            }
        } else {
            TState {
                run_zeros: 0,
                prev_nz: true,
                energy_bucket: new_eb,
                prev_was_pm1: abs_val == 1,
            }
        }
    }
}

// ======================================================================
// Probability snapshot
// ======================================================================

/// Snapshot of the rANS v12 context model at one zigzag position.
/// Captured during a greedy forward simulation for use by the trellis.
pub struct CtxSnapshot {
    /// P(zero) table (128 entries) at this position
    p_zero: [u32; 128],
    /// P(±1|nz) table (32 entries) at this position
    p_pm1: [u32; 32],
}

// ======================================================================
// Rate cost estimation
// ======================================================================

/// Estimate the bit cost of coding quantized value `q` given state `state`
/// and probability tables from the snapshot.
#[inline]
fn rate_cost(q: i16, state: &TState, snap: &CtxSnapshot, tb: u8) -> f64 {
    let p_zero = snap.p_zero[state.zero_ctx(tb)].clamp(1, PROB_SCALE - 1);

    if q == 0 {
        return bits_from_prob(p_zero);
    }

    let abs_v = q.unsigned_abs();
    // Cost of "not zero"
    let mut bits = bits_from_complement(p_zero);
    // Sign bit (equiprobable)
    bits += 1.0;

    let p_pm1 = snap.p_pm1[state.pm1_ctx(tb)].clamp(1, PROB_SCALE - 1);

    if abs_v == 1 {
        bits += bits_from_prob(p_pm1);
    } else {
        // Not ±1
        bits += bits_from_complement(p_pm1);
        // Exp-Golomb order 0 for (abs_v - 2)
        let n_eg = abs_v as u32 - 2;
        let n1 = n_eg + 1;
        let nbits = 32 - n1.leading_zeros();
        bits += (2 * nbits - 1) as f64;
    }

    bits
}

// ======================================================================
// Greedy quantization with snapshot capture
// ======================================================================

/// Perform greedy (standard) quantization while capturing context model snapshots
/// at each position. These snapshots are then used by the trellis for rate estimation.
///
/// Returns (greedy_quantized_values, snapshots).
pub fn greedy_quantize_and_snapshot(
    coeffs: &[f64],
    pos_steps: &[f64],
    dead_zones: &[f64],
    turing_bucket: u8,
) -> (Vec<i16>, Vec<CtxSnapshot>) {
    let n = coeffs.len();
    let mut qvals = Vec::with_capacity(n);
    let mut snapshots = Vec::with_capacity(n);

    // Simulate the rANS v12 context model forward pass
    let mut p_zero = [P_ZERO_INIT; 128];
    let mut p_pm1 = [P_PM1_INIT; 32];
    let mut energy: u32 = 0;
    let mut prev_nz = false;
    let mut run_zeros: u32 = 0;
    let mut prev_was_pm1 = false;
    let tb = turing_bucket.min(3);

    for i in 0..n {
        // Capture snapshot BEFORE processing this symbol
        snapshots.push(CtxSnapshot {
            p_zero,
            p_pm1,
        });

        // Greedy quantization
        let coeff = coeffs[i];
        let step = pos_steps[i];
        let dz = dead_zones[i];
        let inv_step = 1.0 / step;
        let sign = if coeff >= 0.0 { 1.0 } else { -1.0 };
        let qv = (coeff.abs() * inv_step + 0.5 - dz).floor();
        let qi = if qv > 0.0 { (sign * qv) as i16 } else { 0i16 };
        qvals.push(qi);

        // Update context model (mirror ContextModelV12::update)
        let is_zero = qi == 0;
        let is_pm1 = qi.abs() == 1;
        let abs_val = qi.unsigned_abs() as u32;

        // Adapt P(zero)
        let rc = run_class(run_zeros) as usize;
        let pnz = if prev_nz { 1 } else { 0 };
        let eb = energy_bucket(energy) as usize;
        let cz = rc * 32 + pnz * 16 + eb * 4 + tb as usize;
        if is_zero {
            p_zero[cz] += (PROB_SCALE - p_zero[cz]) >> ADAPT_SHIFT;
        } else {
            p_zero[cz] -= p_zero[cz] >> ADAPT_SHIFT;
        }

        if !is_zero {
            let pp = if prev_was_pm1 { 1 } else { 0 };
            let cp = eb * 8 + pp * 4 + tb as usize;
            if is_pm1 {
                p_pm1[cp] += (PROB_SCALE - p_pm1[cp]) >> ADAPT_SHIFT;
            } else {
                p_pm1[cp] -= p_pm1[cp] >> ADAPT_SHIFT;
            }
        }

        // Energy IIR
        let sample_energy = (abs_val * 32).min(ENERGY_SCALE);
        energy = (ENERGY_ALPHA * energy
            + (ENERGY_SCALE - ENERGY_ALPHA) * sample_energy)
            / ENERGY_SCALE;

        // Run/nz state
        if is_zero {
            run_zeros = run_zeros.saturating_add(1);
        } else {
            run_zeros = 0;
            prev_was_pm1 = is_pm1;
        }
        prev_nz = !is_zero;
    }

    (qvals, snapshots)
}

// ======================================================================
// Trellis node (Viterbi)
// ======================================================================

#[derive(Clone, Copy)]
struct TNode {
    cost: f64,
    prev_idx: u8,   // index into previous stage's active state list
    quant_val: i16,
}

impl TNode {
    fn penalty() -> Self {
        Self { cost: f64::MAX, prev_idx: 0, quant_val: 0 }
    }
}

// ======================================================================
// Core Viterbi trellis
// ======================================================================

/// Trellis-optimized quantization for one block of AC coefficients.
///
/// # Arguments
/// - `coeffs`: raw LOT AC coefficients in zigzag order
/// - `pos_steps`: per-position quantization step
/// - `dead_zones`: per-position dead zone
/// - `turing_bucket`: Turing complexity bucket for this block (0-3)
/// - `lambda`: Lagrange multiplier (higher = more aggressive zeroing)
/// - `snapshots`: probability snapshots from `greedy_quantize_and_snapshot`
///
/// # Returns
/// `(optimized_quantized_values, eob_position)`
pub fn trellis_quantize_block(
    coeffs: &[f64],
    pos_steps: &[f64],
    dead_zones: &[f64],
    turing_bucket: u8,
    lambda: f64,
    snapshots: &[CtxSnapshot],
) -> (Vec<i16>, usize) {
    let n = coeffs.len();
    if n == 0 {
        return (Vec::new(), 0);
    }

    let max_states = if n > 300 { MAX_STATES_LARGE } else { MAX_STATES };
    let tb = turing_bucket.min(3);

    // Active state list: (TState, TNode) pairs
    // We keep two lists and swap them each stage.
    let mut prev_states: Vec<(TState, TNode)> = Vec::with_capacity(max_states);
    prev_states.push((TState::initial(), TNode { cost: 0.0, prev_idx: 0, quant_val: 0 }));

    // Traceback storage: for each stage, store the list of (state, node) pairs
    let mut traceback: Vec<Vec<(TState, TNode)>> = Vec::with_capacity(n);

    for i in 0..n {
        let coeff = coeffs[i];
        let step = pos_steps[i];
        let dz = dead_zones[i];
        let snap = &snapshots[i];

        // Generate candidates
        let inv_step = 1.0 / step;
        let sign = if coeff >= 0.0 { 1.0 } else { -1.0 };
        let qv = (coeff.abs() * inv_step + 0.5 - dz).floor();
        let q0 = if qv > 0.0 { (sign * qv) as i16 } else { 0i16 };

        let candidates = generate_candidates(q0, coeff, step);

        // Build next state list
        let mut next_map: Vec<(TState, TNode)> = Vec::with_capacity(prev_states.len() * candidates.len());

        for (src_idx, (src_state, src_node)) in prev_states.iter().enumerate() {
            for &q_cand in &candidates {
                // Distortion
                let err = coeff - q_cand as f64 * step;
                let d = err * err;

                // Rate
                let r = rate_cost(q_cand, src_state, snap, tb);

                // RD cost
                let j = d + lambda * r;
                let total = src_node.cost + j;

                let dst_state = src_state.transition(q_cand);
                let dst_node = TNode {
                    cost: total,
                    prev_idx: src_idx as u8,
                    quant_val: q_cand,
                };

                // Insert or update: keep best cost per state
                if let Some(existing) = next_map.iter_mut().find(|(s, _)| *s == dst_state) {
                    if total < existing.1.cost {
                        existing.1 = dst_node;
                    }
                } else {
                    next_map.push((dst_state, dst_node));
                }
            }
        }

        // Prune: keep only top max_states by cost
        if next_map.len() > max_states {
            next_map.sort_by(|a, b| a.1.cost.partial_cmp(&b.1.cost).unwrap_or(std::cmp::Ordering::Equal));
            next_map.truncate(max_states);
        }

        traceback.push(std::mem::replace(&mut prev_states, Vec::new()));
        prev_states = next_map;
    }

    // Find best final state
    let best_final_idx = prev_states.iter()
        .enumerate()
        .min_by(|a, b| a.1 .1.cost.partial_cmp(&b.1 .1.cost).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    // Backward traceback
    let mut result = vec![0i16; n];
    let mut cur_idx = best_final_idx;

    // Last stage value
    result[n - 1] = prev_states[cur_idx].1.quant_val;
    let mut trace_idx = prev_states[cur_idx].1.prev_idx as usize;

    for i in (0..n - 1).rev() {
        let stage = &traceback[i + 1];
        if trace_idx < stage.len() {
            result[i] = stage[trace_idx].1.quant_val;
            trace_idx = stage[trace_idx].1.prev_idx as usize;
        }
    }

    // EOB: find last non-zero
    let eob = result.iter().rposition(|&v| v != 0).map(|p| p + 1).unwrap_or(0);

    (result, eob)
}

/// Generate quantization candidates for a coefficient.
/// Returns 1-3 candidates: greedy, toward-zero, and possibly zero.
#[inline]
fn generate_candidates(q0: i16, coeff: f64, step: f64) -> Vec<i16> {
    let abs_q0 = q0.unsigned_abs();
    let normalized = coeff.abs() / step;

    // Trivial: coefficient is negligible
    if normalized < 0.1 {
        return vec![0];
    }

    let mut cands = Vec::with_capacity(3);
    cands.push(q0);

    if q0 == 0 {
        // Greedy says zero. Also consider ±1 if coefficient is close to threshold.
        if normalized > 0.3 {
            let sign = if coeff >= 0.0 { 1i16 } else { -1 };
            cands.push(sign);
        }
    } else {
        // Round toward zero by 1
        let q_down = if abs_q0 == 1 {
            0i16
        } else {
            let sign = if q0 > 0 { 1i16 } else { -1 };
            sign * (abs_q0 as i16 - 1)
        };
        cands.push(q_down);

        // Also consider zero if the coefficient is small
        if abs_q0 <= 4 && q_down != 0 {
            cands.push(0);
        }
    }

    cands
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_pack_unique() {
        // Verify all reachable states produce unique pack values
        let mut seen = std::collections::HashSet::new();
        for run in 0..=4u8 {
            for pnz in [false, true] {
                for eb in 0..=3u8 {
                    for pp in [false, true] {
                        let s = TState { run_zeros: run, prev_nz: pnz, energy_bucket: eb, prev_was_pm1: pp };
                        assert!(seen.insert(s.pack()), "Duplicate pack for {:?}", (run, pnz, eb, pp));
                    }
                }
            }
        }
    }

    #[test]
    fn test_ctx_indices_match_rans() {
        // Verify our context index computation matches rans.rs layout
        let s = TState { run_zeros: 0, prev_nz: true, energy_bucket: 2, prev_was_pm1: false };
        let tb = 1u8;
        // zero_ctx: rc(0)*32 + pnz(1)*16 + eb(2)*4 + tb(1) = 0 + 16 + 8 + 1 = 25
        assert_eq!(s.zero_ctx(tb), 25);
        // pm1_ctx: eb(2)*8 + pp(0)*4 + tb(1) = 16 + 0 + 1 = 17
        assert_eq!(s.pm1_ctx(tb), 17);
    }

    #[test]
    fn test_transition_zero() {
        let s = TState::initial();
        let s2 = s.transition(0);
        assert_eq!(s2.run_zeros, 1);
        assert!(!s2.prev_nz);
        let s3 = s2.transition(0);
        assert_eq!(s3.run_zeros, 2);
        let s4 = s3.transition(0).transition(0).transition(0);
        assert_eq!(s4.run_zeros, 4); // capped at 4
    }

    #[test]
    fn test_transition_nonzero() {
        let s = TState { run_zeros: 3, prev_nz: false, energy_bucket: 0, prev_was_pm1: false };
        let s2 = s.transition(5);
        assert_eq!(s2.run_zeros, 0);
        assert!(s2.prev_nz);
        assert!(!s2.prev_was_pm1);

        let s3 = s.transition(1);
        assert!(s3.prev_was_pm1);
    }

    #[test]
    fn test_rate_cost_zero_is_cheap() {
        // With default probabilities, P(zero)=14000/16384 ≈ 85.4%, so coding zero is cheap
        let snap = CtxSnapshot { p_zero: [P_ZERO_INIT; 128], p_pm1: [P_PM1_INIT; 32] };
        let s = TState::initial();
        let cost_zero = rate_cost(0, &s, &snap, 0);
        let cost_one = rate_cost(1, &s, &snap, 0);
        assert!(cost_zero < cost_one, "zero should be cheaper: {} vs {}", cost_zero, cost_one);
        assert!(cost_zero < 0.5, "zero should be very cheap: {}", cost_zero);
    }

    #[test]
    fn test_greedy_all_zeros() {
        let coeffs = vec![0.0; 10];
        let steps = vec![1.0; 10];
        let dzs = vec![0.2; 10];
        let (qvals, snaps) = greedy_quantize_and_snapshot(&coeffs, &steps, &dzs, 0);
        assert!(qvals.iter().all(|&v| v == 0));
        assert_eq!(snaps.len(), 10);
    }

    #[test]
    fn test_lambda_zero_equals_greedy() {
        // With lambda=0, trellis should minimize distortion only → same as greedy
        let coeffs = vec![2.3, 0.4, -1.7, 0.0, 3.1, 0.2, -0.1, 0.0];
        let steps = vec![1.0; 8];
        let dzs = vec![0.15; 8];
        let (greedy, snapshots) = greedy_quantize_and_snapshot(&coeffs, &steps, &dzs, 0);
        let (trellis, _eob) = trellis_quantize_block(&coeffs, &steps, &dzs, 0, 0.0, &snapshots);
        assert_eq!(greedy, trellis, "lambda=0 should produce greedy result");
    }

    #[test]
    fn test_lambda_huge_all_zeros() {
        // With very large lambda, rate dominates → everything should be zeroed
        let coeffs = vec![1.5, 0.8, -0.6, 0.3, 2.1];
        let steps = vec![1.0; 5];
        let dzs = vec![0.15; 5];
        let (_, snapshots) = greedy_quantize_and_snapshot(&coeffs, &steps, &dzs, 0);
        let (trellis, eob) = trellis_quantize_block(&coeffs, &steps, &dzs, 0, 1e6, &snapshots);
        assert_eq!(eob, 0, "huge lambda should zero everything");
        assert!(trellis.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_trellis_preserves_large_coefficients() {
        // Large coefficients should not be zeroed even at moderate lambda
        let mut coeffs = vec![0.0; 20];
        coeffs[5] = 15.0; // very large
        let steps = vec![1.0; 20];
        let dzs = vec![0.15; 20];
        let (_, snapshots) = greedy_quantize_and_snapshot(&coeffs, &steps, &dzs, 0);
        let (trellis, _eob) = trellis_quantize_block(&coeffs, &steps, &dzs, 0, 5.0, &snapshots);
        assert!(trellis[5].abs() >= 10, "large coefficient should be preserved");
    }

    #[test]
    fn test_trellis_can_zero_small_trailing() {
        // A small trailing ±1 should be zeroed at moderate lambda
        // because it saves the entire zero-run coding cost
        let mut coeffs = vec![0.0; 20];
        coeffs[0] = 5.0;   // large, should survive
        coeffs[19] = 0.6;  // small trailing, should be zeroed
        let steps = vec![1.0; 20];
        let dzs = vec![0.15; 20];
        let (greedy, snapshots) = greedy_quantize_and_snapshot(&coeffs, &steps, &dzs, 0);
        let (trellis, trellis_eob) = trellis_quantize_block(&coeffs, &steps, &dzs, 0, 3.0, &snapshots);
        let greedy_eob = greedy.iter().rposition(|&v| v != 0).map(|p| p + 1).unwrap_or(0);
        // Trellis should produce shorter EOB by zeroing the trailing small coefficient
        assert!(trellis_eob <= greedy_eob,
            "trellis should zero trailing small coeff: trellis_eob={}, greedy_eob={}", trellis_eob, greedy_eob);
    }

    #[test]
    fn test_log2_lut_accuracy() {
        let lut = &*LOG2_LUT;
        // P(zero) = 14000/16384 → -log2(14000/16384) ≈ 0.227 bits
        let cost = lut[14000];
        assert!((cost - 0.227).abs() < 0.01, "LUT inaccurate: {}", cost);
        // P = 8192 → -log2(0.5) = 1.0
        assert!((lut[8192] - 1.0).abs() < 0.001);
    }
}
