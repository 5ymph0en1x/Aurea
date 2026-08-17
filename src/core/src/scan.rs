//! Scan order tables for AC coefficient serialization.
//! Golden spiral (Stage 5) and legacy zigzag.

/// Golden angle: pi * (3 - sqrt(5)) rad = 137.508 deg
const GOLDEN_ANGLE: f64 = 2.3999632297286535;

/// Fibonacci-sized fixed arc used by the progressive packet container.
pub const PROGRESSIVE_PACKET_BLOCKS: usize = 233;

/// Low-frequency AC coefficients from every block carried by the first layer.
pub const PROGRESSIVE_SKELETON_COEFFS: usize = 5;

/// Split a continuous spiral traversal into fixed arcs, then restore raster
/// order inside every arc for entropy coding. Concatenating the packets as sets
/// preserves the spiral priority, while each independent rANS stream sees a
/// familiar local order.
pub fn progressive_spiral_packets(spiral_order: &[usize]) -> Vec<Vec<usize>> {
    spiral_order
        .chunks(PROGRESSIVE_PACKET_BLOCKS)
        .map(|arc| {
            let mut raster_packet = arc.to_vec();
            raster_packet.sort_unstable();
            raster_packet
        })
        .collect()
}

/// A resolution-extensible phyllotactic order for adaptive transform blocks.
///
/// The spiral is expressed in 8x8 cell units with a fixed golden angle and a
/// fixed sample density. Increasing the canvas only lets the same arc develop
/// farther; it does not renormalize its inner trajectory. The fovea is the
/// nearest cell to the saliency-weighted centroid, falling back to the image
/// centre for a flat field.
pub fn adaptive_phyllotactic_order(
    blocks: &[(usize, usize, usize)],
    grid_h: usize,
    grid_w: usize,
    use_overlap: bool,
    saliency: &[f64],
) -> Vec<usize> {
    if blocks.is_empty() || grid_h == 0 || grid_w == 0 {
        return Vec::new();
    }

    let mut weighted_y = 0.0;
    let mut weighted_x = 0.0;
    let mut weight_sum = 0.0;
    if saliency.len() == grid_h * grid_w {
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let value = saliency[gy * grid_w + gx];
                let weight = if value.is_finite() {
                    value.max(0.0)
                } else {
                    0.0
                };
                weighted_y += gy as f64 * weight;
                weighted_x += gx as f64 * weight;
                weight_sum += weight;
            }
        }
    }
    let centre_y = (grid_h.saturating_sub(1)) as f64 * 0.5;
    let centre_x = (grid_w.saturating_sub(1)) as f64 * 0.5;
    let anchor_y = if weight_sum > 1e-20 {
        (weighted_y / weight_sum).round()
    } else {
        centre_y.round()
    }
    .clamp(0.0, grid_h.saturating_sub(1) as f64);
    let anchor_x = if weight_sum > 1e-20 {
        (weighted_x / weight_sum).round()
    } else {
        centre_x.round()
    }
    .clamp(0.0, grid_w.saturating_sub(1) as f64);

    let mut cell_owner = vec![usize::MAX; grid_h * grid_w];
    for (block_idx, &(by, bx, block_size)) in blocks.iter().enumerate() {
        let top_y = if use_overlap {
            by.saturating_sub(block_size / 2)
        } else {
            by
        };
        let top_x = if use_overlap {
            bx.saturating_sub(block_size / 2)
        } else {
            bx
        };
        let gy0 = (top_y / 8).min(grid_h - 1);
        let gx0 = (top_x / 8).min(grid_w - 1);
        let span = (block_size / 8).max(1);
        for gy in gy0..(gy0 + span).min(grid_h) {
            for gx in gx0..(gx0 + span).min(grid_w) {
                cell_owner[gy * grid_w + gx] = block_idx;
            }
        }
    }

    let corner_radius = [
        (0.0, 0.0),
        (0.0, grid_w.saturating_sub(1) as f64),
        (grid_h.saturating_sub(1) as f64, 0.0),
        (
            grid_h.saturating_sub(1) as f64,
            grid_w.saturating_sub(1) as f64,
        ),
    ]
    .into_iter()
    .map(|(y, x)| ((y - anchor_y).powi(2) + (x - anchor_x).powi(2)).sqrt())
    .fold(0.0f64, f64::max);

    // Four samples per cell area keep the fixed Fermat arc dense enough after
    // rounding while remaining independent of image dimensions.
    const SAMPLES_PER_CELL: f64 = 4.0;
    let max_k = (SAMPLES_PER_CELL
        * std::f64::consts::PI
        * (corner_radius + std::f64::consts::SQRT_2).powi(2))
    .ceil() as usize;
    let mut visited = vec![false; blocks.len()];
    let mut order = Vec::with_capacity(blocks.len());
    for k in 0..=max_k {
        let radius = (k as f64 / (SAMPLES_PER_CELL * std::f64::consts::PI)).sqrt();
        let theta = k as f64 * GOLDEN_ANGLE;
        let gy_f = anchor_y + radius * theta.sin();
        let gx_f = anchor_x + radius * theta.cos();
        if gy_f < -0.5 || gx_f < -0.5 || gy_f > grid_h as f64 - 0.5 || gx_f > grid_w as f64 - 0.5 {
            continue;
        }
        let gy = gy_f.round().clamp(0.0, (grid_h - 1) as f64) as usize;
        let gx = gx_f.round().clamp(0.0, (grid_w - 1) as f64) as usize;
        let block_idx = cell_owner[gy * grid_w + gx];
        if block_idx < blocks.len() && !visited[block_idx] {
            visited[block_idx] = true;
            order.push(block_idx);
        }
    }

    // Defensive completion for degenerate cropped layouts. Radius is the
    // primary key; raster index is a deterministic final tie-breaker.
    let mut remaining: Vec<usize> = (0..blocks.len()).filter(|&idx| !visited[idx]).collect();
    remaining.sort_by(|&a, &b| {
        let block_centre = |idx: usize| {
            let (by, bx, block_size) = blocks[idx];
            let top_y = if use_overlap {
                by.saturating_sub(block_size / 2)
            } else {
                by
            };
            let top_x = if use_overlap {
                bx.saturating_sub(block_size / 2)
            } else {
                bx
            };
            (
                top_y as f64 / 8.0 + block_size as f64 / 16.0 - 0.5,
                top_x as f64 / 8.0 + block_size as f64 / 16.0 - 0.5,
            )
        };
        let (ay, ax) = block_centre(a);
        let (by, bx) = block_centre(b);
        let ar = (ay - anchor_y).powi(2) + (ax - anchor_x).powi(2);
        let br = (by - anchor_y).powi(2) + (bx - anchor_x).powi(2);
        ar.total_cmp(&br).then_with(|| a.cmp(&b))
    });
    order.extend(remaining);
    order
}

/// Continuous centre-out spiral for adaptive blocks.
///
/// Each block centre is projected onto the closest turn of the Archimedean
/// spiral `r = theta / (2*pi)`, whose radial pitch is exactly one 8x8 cell.
/// Sorting by the unwrapped spiral angle gives a local, resolution-extensible
/// path: adding outer cells appends turns without rescaling the inner arc.
pub fn adaptive_spiral_arc_order(
    blocks: &[(usize, usize, usize)],
    grid_h: usize,
    grid_w: usize,
    use_overlap: bool,
    saliency: &[f64],
) -> Vec<usize> {
    if blocks.is_empty() || grid_h == 0 || grid_w == 0 {
        return Vec::new();
    }

    let mut weighted_y = 0.0;
    let mut weighted_x = 0.0;
    let mut weight_sum = 0.0;
    if saliency.len() == grid_h * grid_w {
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let value = saliency[gy * grid_w + gx];
                let weight = if value.is_finite() {
                    value.max(0.0)
                } else {
                    0.0
                };
                weighted_y += gy as f64 * weight;
                weighted_x += gx as f64 * weight;
                weight_sum += weight;
            }
        }
    }
    let geometric_y = (grid_h.saturating_sub(1)) as f64 * 0.5;
    let geometric_x = (grid_w.saturating_sub(1)) as f64 * 0.5;
    let anchor_y = if weight_sum > 1e-20 {
        (weighted_y / weight_sum).round()
    } else {
        geometric_y.round()
    }
    .clamp(0.0, grid_h.saturating_sub(1) as f64);
    let anchor_x = if weight_sum > 1e-20 {
        (weighted_x / weight_sum).round()
    } else {
        geometric_x.round()
    }
    .clamp(0.0, grid_w.saturating_sub(1) as f64);

    let block_centre = |idx: usize| {
        let (by, bx, block_size) = blocks[idx];
        let top_y = if use_overlap {
            by.saturating_sub(block_size / 2)
        } else {
            by
        };
        let top_x = if use_overlap {
            bx.saturating_sub(block_size / 2)
        } else {
            bx
        };
        (
            top_y as f64 / 8.0 + block_size as f64 / 16.0 - 0.5,
            top_x as f64 / 8.0 + block_size as f64 / 16.0 - 0.5,
        )
    };
    let spiral_key = |idx: usize| {
        let (y, x) = block_centre(idx);
        let dy = y - anchor_y;
        let dx = x - anchor_x;
        let radius = (dy * dy + dx * dx).sqrt();
        let angle = dy.atan2(dx).rem_euclid(std::f64::consts::TAU);
        let turn = (radius - angle / std::f64::consts::TAU).round().max(0.0);
        (turn * std::f64::consts::TAU + angle, radius)
    };

    let mut order: Vec<usize> = (0..blocks.len()).collect();
    order.sort_by(|&a, &b| {
        let (a_theta, a_radius) = spiral_key(a);
        let (b_theta, b_radius) = spiral_key(b);
        a_theta
            .total_cmp(&b_theta)
            .then_with(|| a_radius.total_cmp(&b_radius))
            .then_with(|| a.cmp(&b))
    });
    order
}

/// Golden spiral scan order for block_size x block_size AC positions.
/// Returns block-position indices > 0 (DC at index 0 excluded).
pub fn golden_spiral_order(block_size: usize) -> Vec<usize> {
    let n_ac = block_size * block_size - 1;
    let half = block_size as f64 / 2.0;

    let mut positions: Vec<(f64, f64)> = Vec::with_capacity(n_ac);
    for n in 1..=n_ac {
        let r = (n as f64 / n_ac as f64).sqrt() * half;
        let theta = n as f64 * GOLDEN_ANGLE;
        let col = (half + r * theta.cos())
            .round()
            .clamp(0.0, (block_size - 1) as f64);
        let row = (half + r * theta.sin())
            .round()
            .clamp(0.0, (block_size - 1) as f64);
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
                    if used[idx] {
                        continue;
                    }
                    let dist = ((r as i32 - row as i32).unsigned_abs()
                        + (c as i32 - col as i32).unsigned_abs())
                        as usize;
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

/// Fixed-pitch Archimedean scan for the non-negative DCT quadrant.
///
/// A full-plane spiral `r = theta / (2*pi)` is folded with absolute Cartesian
/// coordinates into the quadrant containing the DCT coefficients. Every
/// lattice point is assigned to the closest visit of that continuous arc.
/// The pitch is one coefficient per revolution and does not depend on block
/// size, so increasing transform resolution only extends the traversal.
pub fn fixed_pitch_folded_spiral_order(block_size: usize) -> Vec<usize> {
    if block_size == 0 {
        return Vec::new();
    }
    let tau = std::f64::consts::TAU;
    let mut keyed = Vec::with_capacity(block_size * block_size - 1);

    for row in 0..block_size {
        for col in 0..block_size {
            if row == 0 && col == 0 {
                continue;
            }
            let x = col as f64;
            let y = row as f64;
            let radius = x.hypot(y);
            let alpha = y.atan2(x);
            let phases = [
                alpha,
                std::f64::consts::PI - alpha,
                std::f64::consts::PI + alpha,
                tau - alpha,
            ];
            let centre_cycle = radius.floor() as i32;
            let mut best_theta = 0.0;
            let mut best_distance = f64::INFINITY;

            for cycle in (centre_cycle - 2).max(0)..=centre_cycle + 2 {
                for &phase in &phases {
                    let theta = cycle as f64 * tau + phase;
                    if theta <= 0.0 {
                        continue;
                    }
                    let arc_radius = theta / tau;
                    let arc_x = arc_radius * theta.cos().abs();
                    let arc_y = arc_radius * theta.sin().abs();
                    let distance = (arc_x - x).powi(2) + (arc_y - y).powi(2);
                    if distance < best_distance || (distance == best_distance && theta < best_theta)
                    {
                        best_distance = distance;
                        best_theta = theta;
                    }
                }
            }
            keyed.push((row * block_size + col, best_theta, radius));
        }
    }

    keyed.sort_by(|&(a_idx, a_theta, a_radius), &(b_idx, b_theta, b_radius)| {
        a_theta
            .total_cmp(&b_theta)
            .then_with(|| a_radius.total_cmp(&b_radius))
            .then_with(|| a_idx.cmp(&b_idx))
    });
    keyed.into_iter().map(|(idx, _, _)| idx).collect()
}

/// Validate QMAT monotonicity along the spiral.
/// Returns percentage of adjacent pairs that violate monotonicity.
pub fn validate_spiral_monotonicity(spiral: &[usize], qmat: &[f64]) -> f64 {
    if spiral.len() < 2 {
        return 0.0;
    }
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
    if n == 0 {
        return Vec::new();
    }

    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);

    let cy = grid_h as f64 / 2.0;
    let cx = grid_w as f64 / 2.0;
    let r_max = (cy * cy + cx * cx).sqrt();

    // Generate golden spiral positions and map to grid
    for k in 0..n {
        let r = (k as f64 / n as f64).sqrt() * r_max;
        let theta = k as f64 * GOLDEN_ANGLE;
        let gy = (cy + r * theta.sin())
            .round()
            .clamp(0.0, (grid_h - 1) as f64) as usize;
        let gx = (cx + r * theta.cos())
            .round()
            .clamp(0.0, (grid_w - 1) as f64) as usize;

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
        for &idx in &order {
            assert!(seen.insert(idx));
        }
    }

    #[test]
    fn folded_spectral_spiral_covers_every_ac_once() {
        let order = fixed_pitch_folded_spiral_order(16);
        assert_eq!(order.len(), 255);
        let mut seen = std::collections::HashSet::new();
        for &idx in &order {
            assert!(idx > 0 && idx < 256);
            assert!(seen.insert(idx), "duplicate: {idx}");
        }
    }

    #[test]
    fn folded_spectral_spiral_extends_with_transform_resolution() {
        let small = fixed_pitch_folded_spiral_order(8);
        let large = fixed_pitch_folded_spiral_order(16);
        let large_prefix_projection: Vec<usize> = large
            .into_iter()
            .filter_map(|idx| {
                let row = idx / 16;
                let col = idx % 16;
                (row < 8 && col < 8 && (row != 0 || col != 0)).then_some(row * 8 + col)
            })
            .collect();
        assert_eq!(large_prefix_projection, small);
    }

    #[test]
    fn progressive_packets_preserve_arc_membership_and_raster_interior() {
        let spiral: Vec<usize> = (0..PROGRESSIVE_PACKET_BLOCKS * 2 + 17)
            .map(|idx| (idx * 97) % (PROGRESSIVE_PACKET_BLOCKS * 2 + 17))
            .collect();
        let packets = progressive_spiral_packets(&spiral);
        assert_eq!(packets.len(), 3);
        assert_eq!(packets[0].len(), PROGRESSIVE_PACKET_BLOCKS);
        assert_eq!(packets[1].len(), PROGRESSIVE_PACKET_BLOCKS);
        assert_eq!(packets[2].len(), 17);
        assert!(packets.iter().all(|packet| packet.is_sorted()));

        for (packet_idx, packet) in packets.iter().enumerate() {
            let start = packet_idx * PROGRESSIVE_PACKET_BLOCKS;
            let end = (start + PROGRESSIVE_PACKET_BLOCKS).min(spiral.len());
            let mut expected = spiral[start..end].to_vec();
            expected.sort_unstable();
            assert_eq!(*packet, expected);
        }
    }

    #[test]
    fn test_covering_spiral_covers_all() {
        for (gh, gw) in [(10, 15), (8, 8), (1, 1), (3, 7), (20, 30)] {
            let order = covering_spiral(gh, gw);
            assert_eq!(
                order.len(),
                gh * gw,
                "spiral must cover all blocks for {}x{}",
                gh,
                gw
            );
            let mut seen = std::collections::HashSet::new();
            for &(gy, gx) in &order {
                assert!(
                    gy < gh && gx < gw,
                    "out of bounds: ({},{}) in {}x{}",
                    gy,
                    gx,
                    gh,
                    gw
                );
                assert!(
                    seen.insert((gy, gx)),
                    "duplicate: ({},{}) in {}x{}",
                    gy,
                    gx,
                    gh,
                    gw
                );
            }
        }
    }

    #[test]
    fn test_covering_spiral_starts_near_center() {
        let order = covering_spiral(10, 15);
        let (gy, gx) = order[0];
        // Golden spiral k=0: r=0, maps to (cy, cx) = (grid_h/2, grid_w/2)
        // Rounding may shift by ±1 pixel
        assert!(
            (gy as i32 - 5).abs() <= 1 && (gx as i32 - 7).abs() <= 1,
            "spiral must start near center, got ({}, {})",
            gy,
            gx
        );
    }

    #[test]
    fn test_spiral_raster_roundtrip() {
        let (gh, gw) = (10, 15);
        let s2r = spiral_to_raster(gh, gw);
        let r2s = raster_to_spiral(gh, gw);
        for (spiral_pos, &raster_idx) in s2r.iter().enumerate() {
            assert_eq!(
                r2s[raster_idx], spiral_pos,
                "roundtrip failed at spiral_pos={}, raster_idx={}",
                spiral_pos, raster_idx
            );
        }
    }

    #[test]
    fn adaptive_phyllotaxis_covers_every_block_once() {
        let size_grid = vec![8u8; 24];
        let blocks = crate::lot::iter_blocks(&size_grid, 4, 6, false);
        let mut saliency = vec![0.0; 24];
        saliency[2 * 6 + 4] = 1.0;
        let order = adaptive_phyllotactic_order(&blocks, 4, 6, false, &saliency);
        assert_eq!(order.len(), blocks.len());
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..blocks.len()).collect::<Vec<_>>());
        assert_eq!(order[0], 2 * 6 + 4);
    }

    #[test]
    fn adaptive_phyllotaxis_is_deterministic_for_mixed_blocks() {
        let size_grid = vec![16u8; 24];
        let blocks = crate::lot::iter_blocks(&size_grid, 4, 6, false);
        let saliency: Vec<f64> = (0..24).map(|idx| (idx % 7) as f64).collect();
        let first = adaptive_phyllotactic_order(&blocks, 4, 6, false, &saliency);
        let second = adaptive_phyllotactic_order(&blocks, 4, 6, false, &saliency);
        assert_eq!(first, second);
    }

    #[test]
    fn continuous_spiral_covers_every_block_and_keeps_a_local_path() {
        let size_grid = vec![8u8; 64];
        let blocks = crate::lot::iter_blocks(&size_grid, 8, 8, false);
        let saliency = vec![0.0; 64];
        let order = adaptive_spiral_arc_order(&blocks, 8, 8, false, &saliency);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());

        let mean_step = order
            .windows(2)
            .map(|pair| {
                let a = blocks[pair[0]];
                let b = blocks[pair[1]];
                let dy = a.0.abs_diff(b.0) as f64 / 8.0;
                let dx = a.1.abs_diff(b.1) as f64 / 8.0;
                (dy * dy + dx * dx).sqrt()
            })
            .sum::<f64>()
            / (order.len() - 1) as f64;
        assert!(mean_step < 2.0, "spiral path is not local: {mean_step}");
    }
}
