//! Constants and algorithms based on the golden ratio phi.
//!
//! - Golden spiral scan: phi^{-1} for low-discrepancy traversal order
//! - Constants phi, phi^{-1}, phi^{-2} for the golden color rotation

/// Golden ratio: phi = (1 + sqrt(5)) / 2
pub const PHI: f64 = 1.618033988749895;
/// Inverse of the golden ratio: phi^{-1} = phi - 1
pub const PHI_INV: f64 = 0.6180339887498949;
/// Square of the inverse: phi^{-2} = 2 - phi
pub const PHI_INV2: f64 = 0.3819660112501051;

/// GCD (Euclidean algorithm).
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Generate the golden spiral scan order for an h x w grid.
/// Returns order[k] = raster index of the k-th visited coefficient.
/// All h*w indices are visited exactly once.
pub fn golden_scan_order(h: usize, w: usize) -> Vec<usize> {
    let n = h * w;
    if n <= 1 {
        return (0..n).collect();
    }

    // Step based on phi^{-1} ~ 0.6180339887
    let phi_inv = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut step = (n as f64 * phi_inv).round() as usize;

    // Step must be coprime with n to visit all elements
    if step == 0 { step = 1; }
    while gcd(step, n) > 1 {
        step += 1;
    }

    let mut order = Vec::with_capacity(n);
    let mut pos = 0usize;
    for _ in 0..n {
        order.push(pos);
        pos = (pos + step) % n;
    }

    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visits_all() {
        for h in 1..=25 {
            for w in 1..=25 {
                let order = golden_scan_order(h, w);
                let n = h * w;
                assert_eq!(order.len(), n);
                let mut seen = vec![false; n];
                for &idx in &order {
                    assert!(idx < n, "index {} out of range for {}x{}", idx, h, w);
                    assert!(!seen[idx], "duplicate at h={}, w={}, idx={}", h, w, idx);
                    seen[idx] = true;
                }
            }
        }
    }

    #[test]
    fn test_deterministic() {
        let a = golden_scan_order(10, 15);
        let b = golden_scan_order(10, 15);
        assert_eq!(a, b);
    }
}
