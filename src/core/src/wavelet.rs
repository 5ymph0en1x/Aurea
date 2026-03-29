//! Wavelet utilities retained for v13.
//! Only morton_order (Z-order scan) and DEAD_ZONE are still used.

/// Default dead zone threshold for AC quantization (quality-adaptive override in calibration.rs).
pub const DEAD_ZONE: f64 = 0.22;

/// Compute Z-order for a pixel (x, y) via bit interleaving.
#[inline]
fn spread_bits(v: u32) -> u32 {
    let mut v = v;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    v
}

/// Build Morton Z-order for an h x w grid.
/// Returns a vector of raster indices sorted by their Z value.
pub fn morton_order(h: usize, w: usize) -> Vec<usize> {
    let n = h * w;
    let mut z_vals: Vec<(u32, usize)> = Vec::with_capacity(n);
    for y in 0..h {
        for x in 0..w {
            let z = spread_bits(x as u32) | (spread_bits(y as u32) << 1);
            z_vals.push((z, y * w + x));
        }
    }
    z_vals.sort_unstable_by_key(|&(z, _)| z);
    z_vals.iter().map(|&(_, idx)| idx).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morton_covers_all() {
        let order = morton_order(8, 8);
        assert_eq!(order.len(), 64);
        let mut seen = vec![false; 64];
        for &idx in &order {
            assert!(!seen[idx], "duplicate index {}", idx);
            seen[idx] = true;
        }
    }

    #[test]
    fn test_morton_deterministic() {
        let a = morton_order(4, 6);
        let b = morton_order(4, 6);
        assert_eq!(a, b);
    }
}
