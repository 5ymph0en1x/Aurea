//! Scan order tables for AC coefficient serialization.
//! Golden spiral (Stage 5) and legacy zigzag.

/// Golden angle: pi * (3 - sqrt(5)) rad = 137.508 deg
const GOLDEN_ANGLE: f64 = 2.3999632297286535;

/// Golden spiral scan order for block_size x block_size AC positions.
/// Returns block-position indices > 0 (DC at index 0 excluded).
pub fn golden_spiral_order(block_size: usize) -> Vec<usize> {
    let n_ac = block_size * block_size - 1;
    let half = block_size as f64 / 2.0;

    let mut positions: Vec<(f64, f64)> = Vec::with_capacity(n_ac);
    for n in 1..=n_ac {
        let r = (n as f64 / n_ac as f64).sqrt() * half;
        let theta = n as f64 * GOLDEN_ANGLE;
        let col = (half + r * theta.cos()).round().clamp(0.0, (block_size - 1) as f64);
        let row = (half + r * theta.sin()).round().clamp(0.0, (block_size - 1) as f64);
        positions.push((row, col));
    }

    let mut used = vec![false; block_size * block_size];
    used[0] = true; // DC reserved
    let mut order = Vec::with_capacity(n_ac);

    for &(row_f, col_f) in &positions {
        let row = row_f as usize;
        let col = col_f as usize;
        let target = row * block_size + col;

        if !used[target] {
            order.push(target);
            used[target] = true;
        } else {
            let mut best = None;
            let mut best_dist = usize::MAX;
            for r in 0..block_size {
                for c in 0..block_size {
                    let idx = r * block_size + c;
                    if used[idx] { continue; }
                    let dist = ((r as i32 - row as i32).unsigned_abs()
                        + (c as i32 - col as i32).unsigned_abs()) as usize;
                    if dist < best_dist {
                        best_dist = dist;
                        best = Some(idx);
                    }
                }
            }
            if let Some(idx) = best {
                order.push(idx);
                used[idx] = true;
            }
        }
    }

    order
}

/// Validate QMAT monotonicity along the spiral.
/// Returns percentage of adjacent pairs that violate monotonicity.
pub fn validate_spiral_monotonicity(spiral: &[usize], qmat: &[f64]) -> f64 {
    if spiral.len() < 2 { return 0.0; }
    let mut violations = 0;
    for i in 1..spiral.len() {
        if qmat[spiral[i]] < qmat[spiral[i - 1]] - 0.5 {
            violations += 1;
        }
    }
    violations as f64 / (spiral.len() - 1) as f64 * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_spiral_covers_all_ac() {
        let order = golden_spiral_order(16);
        assert_eq!(order.len(), 255);
        let mut seen = std::collections::HashSet::new();
        for &idx in &order {
            assert!(seen.insert(idx), "duplicate: {}", idx);
        }
        assert!(!order.contains(&0), "DC must not be in scan");
    }

    #[test]
    fn test_golden_spiral_monotonicity() {
        let order = golden_spiral_order(16);
        let viol = validate_spiral_monotonicity(&order, &crate::lot::QMAT_16);
        eprintln!("Golden spiral monotonicity violations: {:.1}%", viol);
        assert!(viol < 50.0, "too many violations: {:.1}%", viol);
    }

    #[test]
    fn test_golden_spiral_8x8() {
        let order = golden_spiral_order(8);
        assert_eq!(order.len(), 63);
        let mut seen = std::collections::HashSet::new();
        for &idx in &order { assert!(seen.insert(idx)); }
    }
}
