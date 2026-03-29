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

/// Golden covering spiral for a grid of macroblocks (grid_h × grid_w).
/// Uses the GOLDEN ANGLE (137.508°) — the same phyllotaxis as sunflowers.
/// Each successive block is placed at angle n × golden_angle, radius sqrt(n/N).
/// This is the natural optimal covering pattern: no two blocks are angularly
/// aligned, maximizing spatial decorrelation along the spiral path.
///
/// Returns (gy, gx) pairs in golden spiral order, starting from grid center.
pub fn covering_spiral(grid_h: usize, grid_w: usize) -> Vec<(usize, usize)> {
    let n = grid_h * grid_w;
    if n == 0 { return Vec::new(); }

    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);

    let cy = grid_h as f64 / 2.0;
    let cx = grid_w as f64 / 2.0;
    let r_max = (cy * cy + cx * cx).sqrt();

    // Generate golden spiral positions and map to grid
    for k in 0..n {
        let r = (k as f64 / n as f64).sqrt() * r_max;
        let theta = k as f64 * GOLDEN_ANGLE;
        let gy = (cy + r * theta.sin()).round().clamp(0.0, (grid_h - 1) as f64) as usize;
        let gx = (cx + r * theta.cos()).round().clamp(0.0, (grid_w - 1) as f64) as usize;

        if !visited[gy * grid_w + gx] {
            order.push((gy, gx));
            visited[gy * grid_w + gx] = true;
        }
    }

    // Golden spiral may not hit every cell (collisions). Fill remaining cells
    // by nearest-unvisited search from each missed position.
    if order.len() < n {
        // Raster sweep to catch stragglers (rare, only at grid edges)
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                if !visited[gy * grid_w + gx] {
                    order.push((gy, gx));
                    visited[gy * grid_w + gx] = true;
                }
            }
        }
    }

    order
}

/// Convert spiral order to raster-index mapping.
/// Returns `spiral_to_raster[i]` = raster index of the i-th spiral block.
pub fn spiral_to_raster(grid_h: usize, grid_w: usize) -> Vec<usize> {
    covering_spiral(grid_h, grid_w)
        .iter()
        .map(|&(gy, gx)| gy * grid_w + gx)
        .collect()
}

/// Convert raster order to spiral position.
/// Returns `raster_to_spiral[raster_idx]` = position in spiral order.
pub fn raster_to_spiral(grid_h: usize, grid_w: usize) -> Vec<usize> {
    let s2r = spiral_to_raster(grid_h, grid_w);
    let n = grid_h * grid_w;
    let mut r2s = vec![0usize; n];
    for (spiral_pos, &raster_idx) in s2r.iter().enumerate() {
        if raster_idx < n {
            r2s[raster_idx] = spiral_pos;
        }
    }
    r2s
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
        let viol = validate_spiral_monotonicity(&order, &*crate::lot::QMAT_16);
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

    #[test]
    fn test_covering_spiral_covers_all() {
        for (gh, gw) in [(10, 15), (8, 8), (1, 1), (3, 7), (20, 30)] {
            let order = covering_spiral(gh, gw);
            assert_eq!(order.len(), gh * gw,
                "spiral must cover all blocks for {}x{}", gh, gw);
            let mut seen = std::collections::HashSet::new();
            for &(gy, gx) in &order {
                assert!(gy < gh && gx < gw, "out of bounds: ({},{}) in {}x{}", gy, gx, gh, gw);
                assert!(seen.insert((gy, gx)), "duplicate: ({},{}) in {}x{}", gy, gx, gh, gw);
            }
        }
    }

    #[test]
    fn test_covering_spiral_starts_near_center() {
        let order = covering_spiral(10, 15);
        let (gy, gx) = order[0];
        // Golden spiral k=0: r=0, maps to (cy, cx) = (grid_h/2, grid_w/2)
        // Rounding may shift by ±1 pixel
        assert!((gy as i32 - 5).abs() <= 1 && (gx as i32 - 7).abs() <= 1,
            "spiral must start near center, got ({}, {})", gy, gx);
    }

    #[test]
    fn test_spiral_raster_roundtrip() {
        let (gh, gw) = (10, 15);
        let s2r = spiral_to_raster(gh, gw);
        let r2s = raster_to_spiral(gh, gw);
        for (spiral_pos, &raster_idx) in s2r.iter().enumerate() {
            assert_eq!(r2s[raster_idx], spiral_pos,
                "roundtrip failed at spiral_pos={}, raster_idx={}", spiral_pos, raster_idx);
        }
    }
}
