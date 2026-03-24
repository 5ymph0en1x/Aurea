//! Edge-energy hexagonal codec: pixels are atoms, edges carry vibrational energy.
//!
//! Instead of encoding pixel values, we encode gradients between adjacent pixels
//! on a 6-connected hex-like grid. The gradients are classified by frequency band:
//! - IR (large, slow) = major contours -> encode precisely
//! - Visible (medium) = transitions -> quantize + rANS
//! - UV (small, fast) = micro-texture -> dead zone zeros them -> free compression
//!
//! The 6-connectivity on a rectangular grid matches hexagonal topology:
//! each pixel has neighbors at right, left, down, up, down-right, up-left.
//! Each pixel "owns" 3 edges (right, down, down-right); the other 3 are owned
//! by the respective neighbors.

/// The 3 edge directions each pixel "owns" (shared with the neighbor at that offset).
/// Using hex-like 6-connectivity on a rectangular grid.
pub const EDGE_DIRS: [(i32, i32); 3] = [
    (1, 0),   // right
    (0, 1),   // down
    (1, 1),   // down-right (diagonal = hex connectivity)
];

/// Compute all edge energies for a plane.
///
/// Returns 3 planes of gradients (one per direction), same size as input.
/// `edge[dir][y * w + x] = plane[ny, nx] - plane[y, x]` where `(nx, ny)` is
/// the neighbor in direction `dir`. Boundary edges (where the neighbor is
/// out-of-bounds) are zero.
pub fn compute_edges(plane: &[f64], w: usize, h: usize) -> [Vec<f64>; 3] {
    let n = w * h;
    let mut edges = [vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]];

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let val = plane[idx];

            for (dir, &(dx, dy)) in EDGE_DIRS.iter().enumerate() {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                    let nidx = ny as usize * w + nx as usize;
                    edges[dir][idx] = plane[nidx] - val;
                }
                // else: boundary edge = 0 (no gradient)
            }
        }
    }
    edges
}

/// Reconstruct pixel values from edge energies + a single reference pixel value.
///
/// Two-phase approach:
/// 1. **Scan-line seeding**: direct integration along rows (using the "right"
///    edge, dir=0) and columns (using the "down" edge, dir=1), then average.
///    This gives an O(1)-per-pixel initial estimate that captures the global
///    structure immediately.
/// 2. **Poisson refinement**: iterative Gauss-Seidel relaxation using all 6
///    neighbors to smooth out inconsistencies from quantization and the
///    diagonal (dir=2) edges not used in the seed.
///
/// `reference_value`: the value of pixel (0,0).
/// `edges`: 3 gradient planes from `compute_edges`.
/// `n_iterations`: number of Gauss-Seidel iterations for phase 2 (50 typical).
/// `anchors`: optional sparse grid of pinned pixel values (pos, value).
///            Prevents drift on channels with sparse edges (chroma).
///
/// Returns the reconstructed pixel plane.
pub fn reconstruct_from_edges(
    edges: &[Vec<f64>; 3],
    w: usize,
    h: usize,
    reference_value: f64,
    n_iterations: usize,
) -> Vec<f64> {
    reconstruct_from_edges_anchored(edges, w, h, reference_value, n_iterations, &[])
}

/// Reconstruct with anchor grid — pinned values prevent drift accumulation.
pub fn reconstruct_from_edges_anchored(
    edges: &[Vec<f64>; 3],
    w: usize,
    h: usize,
    reference_value: f64,
    n_iterations: usize,
    anchors: &[(usize, f64)], // (pixel_index, pinned_value)
) -> Vec<f64> {
    let n = w * h;

    // Phase 1: scan-line seeding.
    // Integrate horizontally along each row using edge dir=0 (right).
    // First column: integrate vertically from (0,0) using edge dir=1 (down).
    let mut plane_h = vec![0.0f64; n];

    // Seed first column: integrate down from reference
    plane_h[0] = reference_value;
    for y in 1..h {
        // edge[1][(y-1)*w + 0] = plane[y,0] - plane[y-1,0]
        plane_h[y * w] = plane_h[(y - 1) * w] + edges[1][(y - 1) * w];
    }
    // Then integrate each row rightward
    for y in 0..h {
        for x in 1..w {
            let idx = y * w + x;
            // edge[0][y*w + x-1] = plane[y,x] - plane[y,x-1]
            plane_h[idx] = plane_h[idx - 1] + edges[0][idx - 1];
        }
    }

    // Also integrate vertically: first row rightward, then each column downward.
    let mut plane_v = vec![0.0f64; n];

    // Seed first row: integrate right from reference
    plane_v[0] = reference_value;
    for x in 1..w {
        plane_v[x] = plane_v[x - 1] + edges[0][x - 1];
    }
    // Then integrate each column downward
    for x in 0..w {
        for y in 1..h {
            let idx = y * w + x;
            plane_v[idx] = plane_v[(y - 1) * w + x] + edges[1][(y - 1) * w + x];
        }
    }

    // Average the two integration paths for a better initial estimate
    let mut plane = vec![0.0f64; n];
    for i in 0..n {
        plane[i] = (plane_h[i] + plane_v[i]) * 0.5;
    }
    // Pin reference pixel + anchors in initial estimate
    plane[0] = reference_value;
    for &(idx, val) in anchors {
        if idx < n { plane[idx] = val; }
    }

    // Phase 2: Poisson refinement using all 6 neighbors (including diagonal).
    for _iter in 0..n_iterations {
        let prev = plane.clone();

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let mut sum = 0.0f64;
                let mut count = 0u32;

                for (dir, &(dx, dy)) in EDGE_DIRS.iter().enumerate() {
                    // Forward neighbor (I own this edge)
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        let nidx = ny as usize * w + nx as usize;
                        // edge[dir][idx] = neighbor - me, so me = neighbor - edge
                        sum += prev[nidx] - edges[dir][idx];
                        count += 1;
                    }

                    // Backward neighbor (they own the edge to me)
                    let bx = x as i32 - dx;
                    let by = y as i32 - dy;
                    if bx >= 0 && bx < w as i32 && by >= 0 && by < h as i32 {
                        let bidx = by as usize * w + bx as usize;
                        // edge[dir][bidx] = me - backward, so me = backward + edge
                        sum += prev[bidx] + edges[dir][bidx];
                        count += 1;
                    }
                }

                if count > 0 {
                    plane[idx] = sum / count as f64;
                }
            }
        }

        // Pin reference pixel + all anchors
        plane[0] = reference_value;
        for &(idx, val) in anchors {
            if idx < n { plane[idx] = val; }
        }
    }

    plane
}

/// Reconstruct from edges using anchor-tile integration.
/// Each anchor defines a tile. Within each tile, integrate from the anchor
/// using the scan-line method (no Poisson iteration needed).
/// This eliminates drift because each pixel is at most `spacing` pixels
/// from its anchor — error cannot accumulate beyond one tile.
pub fn reconstruct_from_edges_tiled(
    edges: &[Vec<f64>; 3],
    w: usize,
    h: usize,
    anchors: &[(usize, f64)],
    spacing: usize,
) -> Vec<f64> {
    let n = w * h;
    let mut plane = vec![0.0f64; n];

    // Build anchor lookup: for each tile (ty, tx), find the anchor value
    let cols = (w + spacing - 1) / spacing;
    let rows = (h + spacing - 1) / spacing;
    let mut anchor_map = vec![0.0f64; cols * rows];
    for &(idx, val) in anchors {
        let ay = idx / w;
        let ax = idx % w;
        let ty = ay / spacing;
        let tx = ax / spacing;
        if ty < rows && tx < cols {
            anchor_map[ty * cols + tx] = val;
        }
    }

    // For each tile, integrate from its anchor
    for ty in 0..rows {
        for tx in 0..cols {
            let anchor_val = anchor_map[ty * cols + tx];
            let y0 = ty * spacing;
            let x0 = tx * spacing;
            let y1 = ((ty + 1) * spacing).min(h);
            let x1 = ((tx + 1) * spacing).min(w);

            // Seed anchor pixel
            if y0 < h && x0 < w {
                plane[y0 * w + x0] = anchor_val;
            }

            // Integrate first column of tile downward
            for y in (y0 + 1)..y1 {
                let idx = y * w + x0;
                let prev = (y - 1) * w + x0;
                // edge[1][prev] = plane[y,x] - plane[y-1,x]
                plane[idx] = plane[prev] + edges[1][prev];
            }

            // Integrate each row rightward
            for y in y0..y1 {
                for x in (x0 + 1)..x1 {
                    let idx = y * w + x;
                    // edge[0][idx-1] = plane[y,x] - plane[y,x-1]
                    plane[idx] = plane[idx - 1] + edges[0][idx - 1];
                }
            }
        }
    }

    plane
}

/// Store the 4 image borders (top, bottom, left, right rows/columns).
/// These are the TRUE Dirichlet boundary conditions for Poisson reconstruction.
/// Cost: 2*(W+H) pixels per channel ≈ 25 KB for a 2688×1536 image.
pub fn extract_image_borders(plane: &[f64], w: usize, h: usize) -> Vec<f64> {
    // Order: top row (w), bottom row (w), left column (h), right column (h)
    let mut borders = Vec::with_capacity(2 * w + 2 * h);
    // Top row
    borders.extend_from_slice(&plane[0..w]);
    // Bottom row
    borders.extend_from_slice(&plane[(h - 1) * w..h * w]);
    // Left column
    for y in 0..h { borders.push(plane[y * w]); }
    // Right column
    for y in 0..h { borders.push(plane[y * w + w - 1]); }
    borders
}

/// Reconstruct from edges using tiled integration + Poisson smoothing.
/// Phase 1: tiled integration from anchors (fast, creates tile seams).
/// Phase 2: pin image borders + run Poisson iterations to smooth seams.
/// The borders are TRUE pixel values → Dirichlet conditions → unique solution.
pub fn reconstruct_tiled_then_smooth(
    edges: &[Vec<f64>; 3],
    w: usize, h: usize,
    anchors: &[(usize, f64)],
    spacing: usize,
    borders: &[f64], // from extract_image_borders: top(w) + bottom(w) + left(h) + right(h)
    smooth_iterations: usize,
) -> Vec<f64> {
    // Phase 1: tiled integration (fast, seams at tile boundaries)
    let mut plane = reconstruct_from_edges_tiled(edges, w, h, anchors, spacing);

    // Phase 2: pin image borders and run Poisson smoothing to erase seams
    let n = w * h;

    // Build border mask: which pixels are pinned
    let mut pinned = vec![false; n];
    let mut border_vals = vec![0.0f64; n];

    if borders.len() >= 2 * w + 2 * h {
        // Top row
        for x in 0..w {
            pinned[x] = true;
            border_vals[x] = borders[x];
            plane[x] = borders[x];
        }
        // Bottom row
        for x in 0..w {
            let idx = (h - 1) * w + x;
            pinned[idx] = true;
            border_vals[idx] = borders[w + x];
            plane[idx] = borders[w + x];
        }
        // Left column
        for y in 0..h {
            let idx = y * w;
            pinned[idx] = true;
            border_vals[idx] = borders[2 * w + y];
            plane[idx] = borders[2 * w + y];
        }
        // Right column
        for y in 0..h {
            let idx = y * w + w - 1;
            pinned[idx] = true;
            border_vals[idx] = borders[2 * w + h + y];
            plane[idx] = borders[2 * w + h + y];
        }
    }

    // Poisson smoothing: each non-pinned pixel moves toward the average
    // implied by its neighbors + edge energies. Pinned pixels don't move.
    for _iter in 0..smooth_iterations {
        let prev = plane.clone();

        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let idx = y * w + x;
                if pinned[idx] { continue; }

                let mut sum = 0.0f64;
                let mut count = 0u32;

                for (dir, &(dx, dy)) in EDGE_DIRS.iter().enumerate() {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        let nidx = ny as usize * w + nx as usize;
                        sum += prev[nidx] - edges[dir][idx];
                        count += 1;
                    }
                    let bx = x as i32 - dx;
                    let by = y as i32 - dy;
                    if bx >= 0 && bx < w as i32 && by >= 0 && by < h as i32 {
                        let bidx = by as usize * w + bx as usize;
                        sum += prev[bidx] + edges[dir][bidx];
                        count += 1;
                    }
                }

                if count > 0 {
                    plane[idx] = sum / count as f64;
                }
            }
        }
    }

    plane
}

/// Cross-integration from 4 image borders: NO tiles, NO grid artifacts.
/// 4 integration passes (top→down, bottom→up, left→right, right→left),
/// each weighted by inverse distance to its source border.
/// Result: globally smooth reconstruction, unique solution from borders + edges.
pub fn reconstruct_cross_integration(
    edges: &[Vec<f64>; 3],
    w: usize, h: usize,
    borders: &[f64], // top(w) + bottom(w) + left(h) + right(h)
) -> Vec<f64> {
    let n = w * h;
    if borders.len() < 2 * w + 2 * h { return vec![0.0; n]; }

    let top = &borders[0..w];
    let bottom = &borders[w..2 * w];
    let left = &borders[2 * w..2 * w + h];
    let right = &borders[2 * w + h..2 * w + 2 * h];

    // Pass 1: Top → bottom (integrate using edge dir=1 = down)
    let mut pass_tb = vec![0.0f64; n];
    for x in 0..w { pass_tb[x] = top[x]; }
    for y in 1..h {
        for x in 0..w {
            pass_tb[y * w + x] = pass_tb[(y - 1) * w + x] + edges[1][(y - 1) * w + x];
        }
    }

    // Pass 2: Bottom → top (integrate upward = subtract down-edge in reverse)
    let mut pass_bt = vec![0.0f64; n];
    for x in 0..w { pass_bt[(h - 1) * w + x] = bottom[x]; }
    for y in (0..h - 1).rev() {
        for x in 0..w {
            // edge[1][y*w+x] = plane[y+1,x] - plane[y,x], so plane[y,x] = plane[y+1,x] - edge
            pass_bt[y * w + x] = pass_bt[(y + 1) * w + x] - edges[1][y * w + x];
        }
    }

    // Pass 3: Left → right (integrate using edge dir=0 = right)
    let mut pass_lr = vec![0.0f64; n];
    for y in 0..h { pass_lr[y * w] = left[y]; }
    for y in 0..h {
        for x in 1..w {
            pass_lr[y * w + x] = pass_lr[y * w + x - 1] + edges[0][y * w + x - 1];
        }
    }

    // Pass 4: Right → left (integrate leftward = subtract right-edge in reverse)
    let mut pass_rl = vec![0.0f64; n];
    for y in 0..h { pass_rl[y * w + w - 1] = right[y]; }
    for y in 0..h {
        for x in (0..w - 1).rev() {
            // edge[0][y*w+x] = plane[y,x+1] - plane[y,x], so plane[y,x] = plane[y,x+1] - edge
            pass_rl[y * w + x] = pass_rl[y * w + x + 1] - edges[0][y * w + x];
        }
    }

    // Weighted average: each pass weighted by inverse distance to its source border.
    // Closer to border = higher weight for that border's pass.
    let mut plane = vec![0.0f64; n];
    for y in 0..h {
        for x in 0..w {
            let dt = (y + 1) as f64;           // distance to top
            let db = (h - y) as f64;           // distance to bottom
            let dl = (x + 1) as f64;           // distance to left
            let dr = (w - x) as f64;           // distance to right

            // Inverse-distance weights
            let wt = 1.0 / dt;
            let wb = 1.0 / db;
            let wl = 1.0 / dl;
            let wr = 1.0 / dr;
            let w_sum = wt + wb + wl + wr;

            plane[y * w + x] = (
                pass_tb[y * w + x] * wt +
                pass_bt[y * w + x] * wb +
                pass_lr[y * w + x] * wl +
                pass_rl[y * w + x] * wr
            ) / w_sum;
        }
    }

    plane
}

/// Hilbert curve: space-filling path visiting every pixel once.
/// Optimal spatial locality → minimal edge energy along the path.
/// Returns pixel indices in Hilbert order.
pub fn hilbert_path(w: usize, h: usize) -> Vec<usize> {
    // Use a simple recursive Hilbert curve for power-of-2 sizes,
    // with raster fallback for non-power-of-2 regions.
    let n = w * h;
    let mut path = Vec::with_capacity(n);
    let mut visited = vec![false; n];

    // Start from border pixel (0,0)
    let mut x = 0usize;
    let mut y = 0usize;
    path.push(y * w + x);
    visited[y * w + x] = true;

    // Greedy nearest-unvisited-neighbor walk with spatial locality preference.
    // At each step, move to the unvisited neighbor that's closest in Hilbert sense.
    // For simplicity: use a snake scan (boustrophedon) which has good locality.
    let mut going_right = true;
    loop {
        if going_right {
            if x + 1 < w && !visited[y * w + x + 1] {
                x += 1;
            } else if y + 1 < h {
                y += 1;
                going_right = false;
            } else {
                break;
            }
        } else {
            if x > 0 && !visited[y * w + x - 1] {
                x -= 1;
            } else if y + 1 < h {
                y += 1;
                going_right = true;
            } else {
                break;
            }
        }
        let idx = y * w + x;
        if !visited[idx] {
            path.push(idx);
            visited[idx] = true;
        }
    }

    // Catch any missed pixels (shouldn't happen with boustrophedon)
    for i in 0..n {
        if !visited[i] { path.push(i); }
    }

    path
}

/// Golden 3-reference predictor for snake-scan DPCM.
///
/// Uses 3 causal neighbors weighted by golden ratio powers:
///   prev × φ⁻¹ + top × φ⁻² + diag × φ⁻³
///
/// Proven optimal among 4 alternatives tested (5-ref, adaptive, Paeth).
/// The CONSISTENCY of a fixed weighted average matters more than per-pixel
/// precision — switching between prediction modes increases entropy.
#[inline]
fn golden_predict(recon: &[f64], w: usize, h: usize, x: usize, y: usize, fallback: f64) -> f64 {
    use crate::golden::{PHI_INV, PHI_INV2, PHI_INV3};

    let mut sum = 0.0f64;
    let mut weight = 0.0f64;

    // 1. Previous-in-path (strongest horizontal correlation)
    sum += fallback * PHI_INV;
    weight += PHI_INV;

    // 2. Top (strongest vertical correlation — always fully reconstructed)
    if y > 0 {
        sum += recon[(y - 1) * w + x] * PHI_INV2;
        weight += PHI_INV2;
    }

    // 3. Top-left OR top-right diagonal
    if y > 0 && x > 0 {
        sum += recon[(y - 1) * w + x - 1] * PHI_INV3;
        weight += PHI_INV3;
    } else if y > 0 && x + 1 < w {
        sum += recon[(y - 1) * w + x + 1] * PHI_INV3;
        weight += PHI_INV3;
    }

    sum / weight
}

/// Compute local texture energy around a pixel in the reconstructed plane.
/// Used for adaptive DPCM step: smooth areas -> coarse step, textured -> fine step.
/// Uses a 5-pixel causal neighborhood (pixels already visited along the path).
#[inline]
fn local_texture_energy(recon: &[f64], w: usize, h: usize, x: usize, y: usize) -> f64 {
    let idx = y * w + x;
    let val = recon[idx];
    let mut sum_sq = 0.0f64;
    let mut count = 0;

    // Causal neighbors: left, up, up-left, up-right (already reconstructed in snake scan)
    if x > 0 {
        let d = val - recon[idx - 1];
        sum_sq += d * d;
        count += 1;
    }
    if y > 0 {
        let d = val - recon[(y - 1) * w + x];
        sum_sq += d * d;
        count += 1;
    }
    if y > 0 && x > 0 {
        let d = val - recon[(y - 1) * w + x - 1];
        sum_sq += d * d;
        count += 1;
    }
    if y > 0 && x + 1 < w {
        let d = val - recon[(y - 1) * w + x + 1];
        sum_sq += d * d;
        count += 1;
    }

    if count > 0 { (sum_sq / count as f64).sqrt() } else { 0.0 }
}

/// Unified adaptive DPCM step: hex oracle + local energy.
/// Two-level guidance:
/// 1. Hex supercordes (oracle from bitstream): overrides when available
/// 2. Local texture energy (causal from reconstruction): refinement
/// Oracle says solid -> preserve. Oracle says gas + local confirms -> compress.
#[inline]
fn adaptive_step(base_step: f64, energy: f64, hex_is_solid: bool) -> f64 {
    if hex_is_solid {
        base_step // oracle: structure present -> preserve
    } else {
        // Gas region: refine by local energy
        let ratio = if energy < 1.5 {
            3.0 // confirmed gas
        } else if energy < 4.0 {
            1.5 // mild
        } else {
            1.0 // local says solid despite hex gas -> conservative
        };
        (base_step * ratio).max(0.5)
    }
}

// ---------------------------------------------------------------------------
// Hex supercordes map: lightweight oracle for adaptive DPCM
// ---------------------------------------------------------------------------

/// Build a per-pixel boolean map: true = solid (structure), false = gas (smooth).
/// Uses supercordes_classify on the hex grid DC means, then maps each pixel
/// to its nearest hex vertex. Cost: ~64K classifications for a HD image.
pub fn build_hex_solid_map(
    plane: &[f64], w: usize, h: usize,
) -> Vec<bool> {
    use crate::hex::{HexGrid, HEX_R, compute_hex_shape};
    use crate::dsp::{supercordes_classify, Supercorde};

    let grid = HexGrid::new(w, h);
    let n = w * h;

    // Compute DC mean per hex cell
    let shape = compute_hex_shape();
    let mut dc_means = Vec::with_capacity(grid.n_hexes());
    for row in 0..grid.rows {
        for col in 0..grid.cols {
            let (cx, cy) = grid.center(col, row);
            let mut sum = 0.0f64;
            let mut cnt = 0;
            for &(dx, dy) in &shape.voronoi {
                let px = (cx + dx as f64).round() as isize;
                let py = (cy + dy as f64).round() as isize;
                if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
                    sum += plane[py as usize * w + px as usize];
                    cnt += 1;
                }
            }
            dc_means.push(if cnt > 0 { sum / cnt as f64 } else { 0.0 });
        }
    }

    // Supercordes classify the hex grid
    let classes = supercordes_classify(&dc_means, grid.rows, grid.cols);

    // Map each pixel to nearest hex -> solid/gas
    let mut solid_map = vec![false; n];
    let dx = crate::hex::HEX_DX;
    let dy = crate::hex::HEX_DY;

    for y in 0..h {
        for x in 0..w {
            // Approximate nearest hex
            let col = ((x as f64) / dx).round() as usize;
            let row_approx = if col % 2 == 1 {
                ((y as f64 - dy * 0.5) / dy).round() as isize
            } else {
                (y as f64 / dy).round() as isize
            };
            let row = row_approx.clamp(0, grid.rows as isize - 1) as usize;
            let col = col.min(grid.cols - 1);

            let class_idx = row * grid.cols + col;
            if class_idx < classes.len() {
                solid_map[y * w + x] = classes[class_idx] != Supercorde::Rien;
            }
        }
    }

    solid_map
}

/// Serialize solid map as packed bits (1 bit per hex, not per pixel).
/// Returns packed bytes + (grid_cols, grid_rows) for decoder.
pub fn pack_solid_map(
    plane: &[f64], w: usize, h: usize,
) -> (Vec<u8>, usize, usize) {
    use crate::hex::HexGrid;
    use crate::dsp::{supercordes_classify, Supercorde};

    let grid = HexGrid::new(w, h);
    let shape = crate::hex::compute_hex_shape();
    let mut dc_means = Vec::with_capacity(grid.n_hexes());
    for row in 0..grid.rows {
        for col in 0..grid.cols {
            let (cx, cy) = grid.center(col, row);
            let mut sum = 0.0f64;
            let mut cnt = 0;
            for &(dx, dy) in &shape.voronoi {
                let px = (cx + dx as f64).round() as isize;
                let py = (cy + dy as f64).round() as isize;
                if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
                    sum += plane[py as usize * w + px as usize];
                    cnt += 1;
                }
            }
            dc_means.push(if cnt > 0 { sum / cnt as f64 } else { 0.0 });
        }
    }
    let classes = supercordes_classify(&dc_means, grid.rows, grid.cols);
    let n_hexes = classes.len();

    // Pack as bits: 1 = solid, 0 = gas
    let n_bytes = (n_hexes + 7) / 8;
    let mut packed = vec![0u8; n_bytes];
    for (i, class) in classes.iter().enumerate() {
        if *class != Supercorde::Rien {
            packed[i / 8] |= 1 << (7 - i % 8);
        }
    }

    (packed, grid.cols, grid.rows)
}

/// Unpack solid map from bits and expand to per-pixel boolean.
pub fn unpack_solid_map(
    packed: &[u8], grid_cols: usize, grid_rows: usize, w: usize, h: usize,
) -> Vec<bool> {
    let n = w * h;
    let mut solid_map = vec![false; n];
    let dx = crate::hex::HEX_DX;
    let dy = crate::hex::HEX_DY;

    for y in 0..h {
        for x in 0..w {
            let col = ((x as f64) / dx).round().min((grid_cols - 1) as f64).max(0.0) as usize;
            let row_approx = if col % 2 == 1 {
                ((y as f64 - dy * 0.5) / dy).round() as isize
            } else {
                (y as f64 / dy).round() as isize
            };
            let row = row_approx.clamp(0, grid_rows as isize - 1) as usize;
            let hex_idx = row * grid_cols + col;
            let byte_idx = hex_idx / 8;
            let bit_idx = 7 - hex_idx % 8;
            if byte_idx < packed.len() {
                solid_map[y * w + x] = (packed[byte_idx] >> bit_idx) & 1 == 1;
            }
        }
    }
    solid_map
}

// ---------------------------------------------------------------------------
// DNA-guided adaptive step (polymerase codons)
// ---------------------------------------------------------------------------

/// Compute the DNA codon at a pixel position from the local gradient field.
/// Uses the 3 hex-edge directions (right, down, down-right) as LH, HL, HH analogs.
/// Returns a Codon that classifies the local structure.
///
/// IMPORTANT: computed from RECONSTRUCTED pixels so encoder and decoder agree
/// when used in causal mode (Option A). When codons are stored in the bitstream
/// (Option B), the encoder computes from recon and transmits explicitly.
pub fn pixel_codon(recon: &[f64], w: usize, h: usize, x: usize, y: usize, step: f64) -> crate::polymerase::Codon {
    use crate::polymerase::Nucleobase;

    let idx = y * w + x;
    let val = recon[idx];

    // 3 gradient directions (hex edges): right, down, down-right
    let grad_right = if x + 1 < w { recon[idx + 1] - val } else { 0.0 };
    let grad_down  = if y + 1 < h { recon[(y + 1) * w + x] - val } else { 0.0 };
    let grad_diag  = if x + 1 < w && y + 1 < h { recon[(y + 1) * w + x + 1] - val } else { 0.0 };

    crate::polymerase::Codon {
        lh: Nucleobase::from_f64(grad_right, step),
        hl: Nucleobase::from_f64(grad_down, step),
        hh: Nucleobase::from_f64(grad_diag, step),
    }
}

/// Relativistic chroma step: E_visual = C × L².
/// In dark areas (L→0), chroma perception collapses (Hunt effect, CIE CAM02).
/// The chroma DPCM step is amplified by 1/L², creating a "black hole" that
/// zeros out chroma in shadows — physically correct and perceptually invisible.
/// `luma_value`: the reconstructed L value at this pixel (PTF space, 0-255).
#[inline]
pub fn relativistic_chroma_step(base_step: f64, luma_value: f64) -> f64 {
    // E_visual = C × L : relativistic chroma perception.
    // Dark areas (L < 40 PTF): chroma is physically invisible → force delta to 0.
    //   Return f64::INFINITY as step → any delta rounds to 0 → perfect black hole.
    // Midtones (L=100): gentle 2× amplification.
    // Bright (L=200+): near base step.
    // PTF gamma 0.65 maps: pixel=50 → PTF≈88, pixel=30 → PTF≈68.
    // Threshold 80 PTF ≈ pixel 42 → captures the dark zone where cones deactivate.
    if luma_value < 80.0 {
        return f64::MAX; // TRUE black hole: delta is ALWAYS 0, no bias accumulation
    }
    let l_norm = (luma_value / 255.0).clamp(0.15, 1.0);
    (base_step / l_norm).min(base_step * 8.0)
}

/// DNA-guided adaptive step: the codon tells the photon how fast to move.
/// Intron (smooth/gas): 3x coarser step -> aggressive compression, invisible.
/// Solid (any non-intron): base step -> preserve structure.
/// Principle: "only compress the gas, leave the solid untouched."
#[inline]
pub fn dna_adaptive_step(base_step: f64, codon: &crate::polymerase::Codon) -> f64 {
    if codon.is_intron() {
        (base_step * 3.0).max(0.5)  // gas: 3x coarser (huge savings, invisible)
    } else {
        base_step  // solid: base step (preserve structure)
    }
}

/// Encode a plane as adaptive DPCM along a space-filling path.
/// The step varies per pixel based on local texture of the RECONSTRUCTED image.
/// Encoder and decoder compute identical steps (causal neighborhood, same recon).
pub fn encode_path_dpcm(
    plane: &[f64],
    path: &[usize],
    base_step: f64,
    dead_zone: f64,
    solid_map: &[bool],
    luma_plane: Option<&[f64]>, // if Some: relativistic chroma step (E=C×L²)
) -> Vec<i16> {
    let n = path.len();
    if n == 0 { return Vec::new(); }

    // Determine image dimensions from path (max index)
    let max_idx = *path.iter().max().unwrap_or(&0);
    // Estimate w from path pattern (snake scan: first row is sequential)
    let w = if path.len() > 1 && path[1] == path[0] + 1 {
        // Find where the first row ends (next y-coordinate change)
        let mut row_len = 1;
        while row_len < path.len() && path[row_len] == path[0] + row_len { row_len += 1; }
        row_len
    } else { (max_idx + 1).min(path.len()) };
    let h = (max_idx + 1 + w - 1) / w.max(1);

    let mut recon = vec![0.0f64; max_idx + 1];
    let mut deltas = Vec::with_capacity(n);

    for (i, &idx) in path.iter().enumerate() {
        let val = plane[idx];
        let y = idx / w;
        let x = idx % w;

        if i == 0 {
            let q = (val / base_step).round().clamp(-32000.0, 32000.0) as i16;
            deltas.push(q);
            recon[idx] = q as f64 * base_step;
        } else {
            // Unified adaptive step: hex oracle + local energy
            let energy = local_texture_energy(&recon, w, h, x, y);
            let is_solid = if idx < solid_map.len() { solid_map[idx] } else { false };
            let mut step = adaptive_step(base_step, energy, is_solid);

            // E=C×L² : relativistic chroma modulation (Hunt effect)
            // In dark areas, chroma perception collapses → step grows as 1/L²
            if let Some(luma) = luma_plane {
                if idx < luma.len() {
                    step = relativistic_chroma_step(step, luma[idx]);
                }
            }

            // Golden multi-reference prediction: left × φ⁻¹ + top × φ⁻² + diag × φ⁻³
            // All three are already reconstructed in the snake scan.
            // Falls back to previous-in-path when neighbors aren't available.
            let pred = golden_predict(&recon, w, h, x, y, recon[path[i - 1]]);

            let delta = val - pred;
            let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
            let qv = (delta.abs() / step + 0.5 - dead_zone).floor();
            let q = if qv > 0.0 { (sign * qv).clamp(-32000.0, 32000.0) as i16 } else { 0 };
            deltas.push(q);
            // If step is MAX (black hole), force recon to 0 to prevent bias accumulation
            if step >= f64::MAX * 0.5 {
                recon[idx] = 0.0;
            } else {
                recon[idx] = pred + q as f64 * step;
            }
        }
    }
    deltas
}

/// Decode adaptive DPCM along the path: reconstruct the plane.
/// Computes the same adaptive step as the encoder (causal, from reconstructed pixels).
pub fn decode_path_dpcm(
    deltas: &[i16],
    path: &[usize],
    base_step: f64,
    n_pixels: usize,
    solid_map: &[bool],
    luma_plane: Option<&[f64]>, // if Some: relativistic chroma step (E=C×L²)
) -> Vec<f64> {
    let mut plane = vec![0.0f64; n_pixels];
    if path.is_empty() || deltas.is_empty() { return plane; }

    let max_idx = *path.iter().max().unwrap_or(&0);
    let w = if path.len() > 1 && path[1] == path[0] + 1 {
        let mut row_len = 1;
        while row_len < path.len() && path[row_len] == path[0] + row_len { row_len += 1; }
        row_len
    } else { (max_idx + 1).min(path.len()) };
    let h = (max_idx + 1 + w - 1) / w.max(1);

    for (i, &idx) in path.iter().enumerate() {
        if i >= deltas.len() { break; }

        if i == 0 {
            plane[idx] = deltas[i] as f64 * base_step;
        } else {
            let y = idx / w;
            let x = idx % w;
            let energy = local_texture_energy(&plane, w, h, x, y);
            let is_solid = if idx < solid_map.len() { solid_map[idx] } else { false };
            let mut step = adaptive_step(base_step, energy, is_solid);

            // E=C×L² : relativistic chroma modulation
            if let Some(luma) = luma_plane {
                if idx < luma.len() {
                    step = relativistic_chroma_step(step, luma[idx]);
                }
            }

            let pred = golden_predict(&plane, w, h, x, y, plane[path[i - 1]]);
            if step >= f64::MAX * 0.5 {
                plane[idx] = 0.0; // black hole: chroma = 0 in dark areas
            } else {
                plane[idx] = pred + deltas[i] as f64 * step;
            }
        }
    }
    plane
}

// ---------------------------------------------------------------------------
// DNA-guided path DPCM (mode=2): codons stored in bitstream
// ---------------------------------------------------------------------------

/// Encode a plane as DNA-guided adaptive DPCM along a space-filling path.
///
/// The encoder maintains a running reconstruction and computes codons from it.
/// Codons are returned as amino acid bytes (0..124) for explicit storage in the
/// bitstream. The adaptive step uses `dna_adaptive_step` instead of
/// `local_texture_energy` + `adaptive_step`.
///
/// Returns (deltas, amino_acids) where amino_acids has one entry per pixel.
pub fn encode_path_dpcm_dna(
    plane: &[f64],
    path: &[usize],
    base_step: f64,
    dead_zone: f64,
) -> (Vec<i16>, Vec<u8>) {
    let n = path.len();
    if n == 0 { return (Vec::new(), Vec::new()); }

    let max_idx = *path.iter().max().unwrap_or(&0);
    let w = if path.len() > 1 && path[1] == path[0] + 1 {
        let mut row_len = 1;
        while row_len < path.len() && path[row_len] == path[0] + row_len { row_len += 1; }
        row_len
    } else { (max_idx + 1).min(path.len()) };
    let h = (max_idx + 1 + w - 1) / w.max(1);

    let mut recon = vec![0.0f64; max_idx + 1];
    let mut deltas = Vec::with_capacity(n);
    let mut amino_acids = Vec::with_capacity(n);

    for (i, &idx) in path.iter().enumerate() {
        let val = plane[idx];
        let y = idx / w;
        let x = idx % w;

        // Compute codon from current reconstruction state
        let codon = pixel_codon(&recon, w, h, x, y, base_step);
        amino_acids.push(codon.to_amino_acid());

        if i == 0 {
            let q = (val / base_step).round().clamp(-32000.0, 32000.0) as i16;
            deltas.push(q);
            recon[idx] = q as f64 * base_step;
        } else {
            // DNA-guided adaptive step
            let step = dna_adaptive_step(base_step, &codon);

            let delta = val - recon[path[i - 1]];
            let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
            let qv = (delta.abs() / step + 0.5 - dead_zone).floor();
            let q = if qv > 0.0 { (sign * qv).clamp(-32000.0, 32000.0) as i16 } else { 0 };
            deltas.push(q);
            recon[idx] = recon[path[i - 1]] + q as f64 * step;
        }
    }
    (deltas, amino_acids)
}

/// Decode DNA-guided DPCM along the path using stored amino acid codons.
///
/// The decoder reads the codon from the bitstream (oracle knowledge) rather than
/// recomputing it. This ensures perfect encoder/decoder agreement on the adaptive
/// step, and gives the decoder structural knowledge of the image.
pub fn decode_path_dpcm_dna(
    deltas: &[i16],
    path: &[usize],
    base_step: f64,
    n_pixels: usize,
    amino_acids: &[u8],
) -> Vec<f64> {
    use crate::polymerase::Codon;

    let mut plane = vec![0.0f64; n_pixels];
    if path.is_empty() || deltas.is_empty() { return plane; }

    for (i, &idx) in path.iter().enumerate() {
        if i >= deltas.len() { break; }

        if i == 0 {
            plane[idx] = deltas[i] as f64 * base_step;
        } else {
            // Read the stored codon for this pixel
            let codon = if i < amino_acids.len() {
                Codon::from_amino_acid(amino_acids[i])
            } else {
                // Fallback: intron (gas) if no codon available
                Codon {
                    lh: crate::polymerase::Nucleobase::Intron,
                    hl: crate::polymerase::Nucleobase::Intron,
                    hh: crate::polymerase::Nucleobase::Intron,
                }
            };
            let step = dna_adaptive_step(base_step, &codon);
            plane[idx] = plane[path[i - 1]] + deltas[i] as f64 * step;
        }
    }
    plane
}

/// Build anchor grid: sample the plane at regular intervals.
/// Returns (index, value) pairs for Poisson pinning.
/// Anchor spacing of 16 means one anchor per 16x16 block.
pub fn build_anchor_grid(plane: &[f64], w: usize, h: usize, spacing: usize) -> Vec<(usize, f64)> {
    let mut anchors = Vec::new();
    let mut y = 0;
    while y < h {
        let mut x = 0;
        while x < w {
            anchors.push((y * w + x, plane[y * w + x]));
            x += spacing;
        }
        y += spacing;
    }
    anchors
}

/// Serialize anchors as (u32 index, f32 value) pairs.
pub fn serialize_anchors(anchors: &[(usize, f64)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(anchors.len() * 8);
    for &(idx, val) in anchors {
        bytes.extend_from_slice(&(idx as u32).to_le_bytes());
        bytes.extend_from_slice(&(val as f32).to_le_bytes());
    }
    bytes
}

/// Deserialize anchors from bytes.
pub fn deserialize_anchors(data: &[u8]) -> Vec<(usize, f64)> {
    let mut anchors = Vec::new();
    let mut pos = 0;
    while pos + 8 <= data.len() {
        let idx = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        let val = f32::from_le_bytes([data[pos+4], data[pos+5], data[pos+6], data[pos+7]]) as f64;
        anchors.push((idx, val));
        pos += 8;
    }
    anchors
}

/// Classify edge energy into frequency bands.
///
/// Returns `(ir_count, visible_count, uv_count)` for diagnostics.
/// - IR: `|edge| > ir_threshold` (major contours)
/// - Visible: `uv_threshold < |edge| <= ir_threshold` (transitions)
/// - UV: `|edge| <= uv_threshold` (micro-texture, zeroed by dead zone)
pub fn classify_edges(
    edges: &[Vec<f64>; 3],
    ir_threshold: f64,
    uv_threshold: f64,
) -> (usize, usize, usize) {
    let mut ir = 0usize;
    let mut vis = 0usize;
    let mut uv = 0usize;
    for dir in 0..3 {
        for &e in &edges[dir] {
            let ae = e.abs();
            if ae > ir_threshold {
                ir += 1;
            } else if ae > uv_threshold {
                vis += 1;
            } else {
                uv += 1;
            }
        }
    }
    (ir, vis, uv)
}

/// Quantize edge energies with dead zone.
///
/// UV edges (below dead zone threshold) become zero, giving massive compression.
/// Uses the same quantizer formula as the rest of the codec:
/// `q = floor(|val| / step + 0.5 - dead_zone)`, zeroed if `q <= 0`.
pub fn quantize_edges(edges: &[Vec<f64>; 3], step: f64, dead_zone: f64) -> [Vec<i16>; 3] {
    let n = edges[0].len();
    let mut q_edges = [vec![0i16; n], vec![0i16; n], vec![0i16; n]];

    for dir in 0..3 {
        for i in 0..n {
            let val = edges[dir][i];
            let sign = if val >= 0.0 { 1.0 } else { -1.0 };
            let qv = (val.abs() / step + 0.5 - dead_zone).floor();
            q_edges[dir][i] = if qv > 0.0 {
                (sign * qv).clamp(-32000.0, 32000.0) as i16
            } else {
                0
            };
        }
    }
    q_edges
}

/// Dequantize edge energies.
pub fn dequantize_edges(q_edges: &[Vec<i16>; 3], step: f64) -> [Vec<f64>; 3] {
    let n = q_edges[0].len();
    let mut edges = [vec![0.0f64; n], vec![0.0f64; n], vec![0.0f64; n]];
    for dir in 0..3 {
        for i in 0..n {
            edges[dir][i] = q_edges[dir][i] as f64 * step;
        }
    }
    edges
}

/// Statistics about edge energy distribution for diagnostic purposes.
pub struct EdgeStats {
    pub total_edges: usize,
    pub zero_edges: usize,
    pub zero_fraction: f64,
    pub mean_abs: f64,
    pub max_abs: f64,
}

/// Compute statistics on quantized edge planes.
pub fn edge_stats(q_edges: &[Vec<i16>; 3]) -> EdgeStats {
    let mut total = 0usize;
    let mut zeros = 0usize;
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f64;
    for dir in 0..3 {
        for &v in &q_edges[dir] {
            total += 1;
            if v == 0 {
                zeros += 1;
            }
            let a = v.abs() as f64;
            sum_abs += a;
            if a > max_abs {
                max_abs = a;
            }
        }
    }
    EdgeStats {
        total_edges: total,
        zero_edges: zeros,
        zero_fraction: if total > 0 { zeros as f64 / total as f64 } else { 0.0 },
        mean_abs: if total > 0 { sum_abs / total as f64 } else { 0.0 },
        max_abs,
    }
}

// ---------------------------------------------------------------------------
// Hex-path polymerase codec: honeycomb labyrinth + rich DNA classification
// ---------------------------------------------------------------------------

/// Build the hex-center path in golden spiral order.
/// Returns (hex_index, cx, cy) for each hex in visit order.
/// Uses the covering_spiral from scan.rs to visit every hex exactly once,
/// starting from the center and spiraling outward at the golden angle.
pub fn hex_spiral_path(w: usize, h: usize) -> Vec<(usize, f64, f64)> {
    use crate::hex::HexGrid;
    use crate::scan::covering_spiral;

    let grid = HexGrid::new(w, h);
    let spiral = covering_spiral(grid.rows, grid.cols);
    spiral.iter().map(|&(row, col)| {
        let idx = row * grid.cols + col;
        let (cx, cy) = grid.center(col, row);
        (idx, cx, cy)
    }).collect()
}

/// Classify a hex cell using the polymerase: compute a codon from the 3
/// gradient directions sampled at the hex center.
/// Returns (amino_acid 0..124, is_gas).
pub fn classify_hex_polymerase(
    plane: &[f64], w: usize, h: usize,
    cx: f64, cy: f64, step: f64,
) -> (u8, bool) {
    let x = (cx.round() as usize).min(w.saturating_sub(1));
    let y = (cy.round() as usize).min(h.saturating_sub(1));
    let codon = pixel_codon(plane, w, h, x, y, step);
    (codon.to_amino_acid(), codon.is_intron())
}

/// Compute the DC (mean value) of a hex cell from the image plane,
/// sampling the voronoi pixels of the given shape.
pub fn hex_dc(
    plane: &[f64], w: usize, h: usize,
    cx: f64, cy: f64,
    shape: &crate::hex::HexShape,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    for &(dx, dy) in &shape.voronoi {
        let px = (cx + dx as f64).round() as isize;
        let py = (cy + dy as f64).round() as isize;
        if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
            sum += plane[py as usize * w + px as usize];
            count += 1;
        }
    }
    if count > 0 { sum / count as f64 } else { 0.0 }
}

/// For solid hexes: encode interior pixels as quantized deltas from DC.
/// Returns a vector of quantized deltas for the hex's voronoi pixels.
pub fn encode_hex_interior(
    plane: &[f64], w: usize, h: usize,
    cx: f64, cy: f64, dc: f64,
    shape: &crate::hex::HexShape, step: f64, dead_zone: f64,
) -> Vec<i16> {
    let mut deltas = Vec::with_capacity(shape.voronoi.len());
    for &(dx, dy) in &shape.voronoi {
        let px = (cx + dx as f64).round() as isize;
        let py = (cy + dy as f64).round() as isize;
        if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
            let val = plane[py as usize * w + px as usize];
            let delta = val - dc;
            let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
            let qv = (delta.abs() / step + 0.5 - dead_zone).floor();
            deltas.push(if qv > 0.0 { (sign * qv).clamp(-127.0, 127.0) as i16 } else { 0 });
        }
    }
    deltas
}

/// Polymerase-guided quantization step: the amino acid encodes how many
/// gradient directions are active, which determines the structural richness.
/// Gas (all intron) -> 3x coarser, mild -> 2x, solid -> base step.
/// Principle: "on ne compresse que le gaz."
pub fn polymerase_step(base_step: f64, amino_acid: u8) -> f64 {
    use crate::polymerase::Codon;

    let codon = Codon::from_amino_acid(amino_acid);
    // Count active (non-intron) bases
    let active = [codon.lh, codon.hl, codon.hh].iter()
        .filter(|b| **b != crate::polymerase::Nucleobase::Intron)
        .count();

    match active {
        0 => base_step * 3.0,  // pure intron (gas): 3x coarser
        1 => base_step * 2.0,  // one active direction: weak structure
        2 => base_step * 1.0,  // two directions: medium structure
        _ => base_step * 1.0,  // all three: solid, base step
    }
}

/// Encode one channel using hex-path polymerase DPCM.
///
/// 1. Visit hexes in golden spiral order
/// 2. Classify each hex with polymerase (codon -> amino acid)
/// 3. Compute hex DCs, DPCM-encode along spiral
/// 4. For solid hexes: encode all interior pixels as deltas from DC
/// 5. For gas hexes: only DC, interior will be interpolated at decode
///
/// Returns (dc_deltas, interior_deltas, amino_acids, gas_flags).
pub fn encode_hex_path_channel(
    plane: &[f64],
    hex_path: &[(usize, f64, f64)],
    shape: &crate::hex::HexShape,
    w: usize, h: usize,
    dc_step: f64,
    interior_step: f64,
    dead_zone: f64,
    base_step_classify: f64,
) -> (Vec<i16>, Vec<i16>, Vec<u8>, Vec<bool>) {
    let n_hexes = hex_path.len();

    // 1. Compute hex DCs and classify
    let mut hex_dcs = Vec::with_capacity(n_hexes);
    let mut amino_acids = Vec::with_capacity(n_hexes);
    let mut gas_flags = Vec::with_capacity(n_hexes);

    for &(_idx, cx, cy) in hex_path {
        let dc = hex_dc(plane, w, h, cx, cy, shape);
        let (aa, is_gas) = classify_hex_polymerase(plane, w, h, cx, cy, base_step_classify);
        hex_dcs.push(dc);
        amino_acids.push(aa);
        gas_flags.push(is_gas);
    }

    // 2. DPCM on DCs (golden prediction along spiral)
    let mut dc_deltas = Vec::with_capacity(n_hexes);
    let mut prev_dc_recon = 0.0f64;
    let mut recon_dcs = Vec::with_capacity(n_hexes);

    for &dc in &hex_dcs {
        let delta = dc - prev_dc_recon;
        let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
        let qv = (delta.abs() / dc_step + 0.5).floor();
        let q = if qv > 0.0 { (sign * qv).clamp(-32000.0, 32000.0) as i16 } else { 0 };
        dc_deltas.push(q);
        let recon_dc = prev_dc_recon + q as f64 * dc_step;
        recon_dcs.push(recon_dc);
        prev_dc_recon = recon_dc;
    }

    // 3. Interior pixels for solid hexes
    let mut interior_deltas = Vec::new();
    for (i, &(_idx, cx, cy)) in hex_path.iter().enumerate() {
        if !gas_flags[i] {
            // Use polymerase-guided step for this hex
            let step = polymerase_step(interior_step, amino_acids[i]);
            let deltas = encode_hex_interior(
                plane, w, h, cx, cy,
                recon_dcs[i], shape, step, dead_zone,
            );
            interior_deltas.extend_from_slice(&deltas);
        }
    }

    (dc_deltas, interior_deltas, amino_acids, gas_flags)
}

/// Encode one channel using hex polymerase with EXTERNAL gas flags.
/// Used for chroma channels that share L's structural classification.
pub fn encode_hex_path_channel_with_flags(
    plane: &[f64],
    hex_path: &[(usize, f64, f64)],
    shape: &crate::hex::HexShape,
    w: usize, h: usize,
    dc_step: f64,
    interior_step: f64,
    dead_zone: f64,
    base_step_classify: f64,
    external_gas_flags: &[bool],
) -> (Vec<i16>, Vec<i16>, Vec<u8>) {
    let n_hexes = hex_path.len();

    // 1. Compute hex DCs and classify (but use external gas flags)
    let mut hex_dcs = Vec::with_capacity(n_hexes);
    let mut amino_acids = Vec::with_capacity(n_hexes);

    for &(_idx, cx, cy) in hex_path {
        let dc = hex_dc(plane, w, h, cx, cy, shape);
        let (aa, _is_gas) = classify_hex_polymerase(plane, w, h, cx, cy, base_step_classify);
        hex_dcs.push(dc);
        amino_acids.push(aa);
    }

    // 2. DPCM on DCs
    let mut dc_deltas = Vec::with_capacity(n_hexes);
    let mut prev_dc_recon = 0.0f64;
    let mut recon_dcs = Vec::with_capacity(n_hexes);

    for &dc in &hex_dcs {
        let delta = dc - prev_dc_recon;
        let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
        let qv = (delta.abs() / dc_step + 0.5).floor();
        let q = if qv > 0.0 { (sign * qv).clamp(-32000.0, 32000.0) as i16 } else { 0 };
        dc_deltas.push(q);
        let recon_dc = prev_dc_recon + q as f64 * dc_step;
        recon_dcs.push(recon_dc);
        prev_dc_recon = recon_dc;
    }

    // 3. Interior pixels for solid hexes (using EXTERNAL gas flags)
    let mut interior_deltas = Vec::new();
    for (i, &(_idx, cx, cy)) in hex_path.iter().enumerate() {
        let is_gas = if i < external_gas_flags.len() { external_gas_flags[i] } else { true };
        if !is_gas {
            let step = polymerase_step(interior_step, amino_acids[i]);
            let deltas = encode_hex_interior(
                plane, w, h, cx, cy,
                recon_dcs[i], shape, step, dead_zone,
            );
            interior_deltas.extend_from_slice(&deltas);
        }
    }

    (dc_deltas, interior_deltas, amino_acids)
}

/// Decode one channel from hex-path polymerase DPCM.
///
/// 1. Decode DC deltas -> reconstruct hex DCs along spiral
/// 2. Scatter gas hex DCs: all voronoi pixels = DC value
/// 3. Scatter solid hex pixels: DC + delta * step
/// 4. Fill any remaining gaps with nearest DC
///
/// Returns the reconstructed pixel plane.
pub fn decode_hex_path_channel(
    dc_deltas: &[i16],
    interior_deltas: &[i16],
    amino_acids: &[u8],
    gas_flags: &[bool],
    hex_path: &[(usize, f64, f64)],
    shape: &crate::hex::HexShape,
    w: usize, h: usize,
    dc_step: f64,
    interior_step: f64,
) -> Vec<f64> {
    let n = w * h;
    let mut plane = vec![0.0f64; n];
    let mut written = vec![false; n];
    let n_hexes = hex_path.len().min(dc_deltas.len());

    // 1. Reconstruct hex DCs
    let mut recon_dcs = Vec::with_capacity(n_hexes);
    let mut prev_dc = 0.0f64;
    for i in 0..n_hexes {
        let dc = prev_dc + dc_deltas[i] as f64 * dc_step;
        recon_dcs.push(dc);
        prev_dc = dc;
    }

    // 2+3. Scatter pixels
    let mut interior_cursor = 0usize;
    for i in 0..n_hexes {
        let (_idx, cx, cy) = hex_path[i];
        let dc = recon_dcs[i];
        let is_gas = if i < gas_flags.len() { gas_flags[i] } else { true };
        let aa = if i < amino_acids.len() { amino_acids[i] } else { 124 }; // 124 = all intron

        if is_gas {
            // Gas hex: all voronoi pixels get the DC value
            for &(dx, dy) in &shape.voronoi {
                let px = (cx + dx as f64).round() as isize;
                let py = (cy + dy as f64).round() as isize;
                if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
                    let idx = py as usize * w + px as usize;
                    if !written[idx] {
                        plane[idx] = dc;
                        written[idx] = true;
                    }
                }
            }
        } else {
            // Solid hex: DC + interior deltas
            // The encoder produced one delta per in-bounds voronoi pixel,
            // so we must consume one delta per in-bounds pixel regardless
            // of whether the pixel was already written by a previous hex.
            let step = polymerase_step(interior_step, aa);

            for &(dx, dy) in &shape.voronoi {
                let px = (cx + dx as f64).round() as isize;
                let py = (cy + dy as f64).round() as isize;
                if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
                    let delta = if interior_cursor < interior_deltas.len() {
                        interior_deltas[interior_cursor] as f64 * step
                    } else {
                        0.0
                    };
                    interior_cursor += 1;

                    let idx = py as usize * w + px as usize;
                    if !written[idx] {
                        plane[idx] = dc + delta;
                        written[idx] = true;
                    }
                }
            }
        }
    }

    // 4. Fill unwritten pixels (border gaps not covered by any hex).
    //    Use a bounded iteration count to avoid infinite loops on edge cases.
    let max_fill_iters = ((w.max(h)) / 2).max(10);
    for _iter in 0..max_fill_iters {
        let mut changed = false;
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if written[idx] { continue; }
                // Check 4-neighbors
                let mut sum = 0.0;
                let mut cnt = 0;
                if x > 0 && written[idx - 1] { sum += plane[idx - 1]; cnt += 1; }
                if x + 1 < w && written[idx + 1] { sum += plane[idx + 1]; cnt += 1; }
                if y > 0 && written[idx - w] { sum += plane[idx - w]; cnt += 1; }
                if y + 1 < h && written[idx + w] { sum += plane[idx + w]; cnt += 1; }
                if cnt > 0 {
                    plane[idx] = sum / cnt as f64;
                    written[idx] = true;
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }

    plane
}

// ---------------------------------------------------------------------------
// Hex-Predictor: predict pixels from hex DC, encode residuals
// ---------------------------------------------------------------------------

/// Build a per-pixel hex DC prediction map.
/// Each pixel gets the DC value of the hex cell it belongs to.
/// This is a 2D spatial predictor (no accumulation, no drift).
pub fn build_hex_prediction_map(
    plane: &[f64], w: usize, h: usize,
    grid: &crate::hex::HexGrid,
    shape: &crate::hex::HexShape,
) -> Vec<f64> {
    let n = w * h;
    let mut prediction = vec![0.0f64; n];
    let mut assigned = vec![false; n];

    // For each hex cell, compute DC and assign to all voronoi pixels
    for row in 0..grid.rows {
        for col in 0..grid.cols {
            let (cx, cy) = grid.center(col, row);
            let dc = hex_dc(plane, w, h, cx, cy, shape);

            for &(dx, dy) in &shape.voronoi {
                let px = (cx + dx as f64).round() as isize;
                let py = (cy + dy as f64).round() as isize;
                if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
                    let idx = py as usize * w + px as usize;
                    if !assigned[idx] {
                        prediction[idx] = dc;
                        assigned[idx] = true;
                    }
                }
            }
        }
    }

    // Fill any unassigned border pixels with nearest assigned value
    for _iter in 0..10 {
        let mut changed = false;
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if assigned[idx] { continue; }
                let mut sum = 0.0;
                let mut cnt = 0;
                if x > 0 && assigned[idx - 1] { sum += prediction[idx - 1]; cnt += 1; }
                if x + 1 < w && assigned[idx + 1] { sum += prediction[idx + 1]; cnt += 1; }
                if y > 0 && assigned[idx - w] { sum += prediction[idx - w]; cnt += 1; }
                if y + 1 < h && assigned[idx + w] { sum += prediction[idx + w]; cnt += 1; }
                if cnt > 0 {
                    prediction[idx] = sum / cnt as f64;
                    assigned[idx] = true;
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }

    prediction
}

/// Build a per-pixel hex ownership map.
/// Each pixel gets the ID of the hex cell it belongs to (raster-order ID).
/// Returns (hex_id_map, n_hexes).
pub fn build_hex_ownership_map(
    w: usize, h: usize,
    grid: &crate::hex::HexGrid,
    shape: &crate::hex::HexShape,
) -> Vec<u32> {
    let n = w * h;
    let mut hex_id = vec![u32::MAX; n];

    let mut id: u32 = 0;
    for row in 0..grid.rows {
        for col in 0..grid.cols {
            let (cx, cy) = grid.center(col, row);
            for &(dx, dy) in &shape.voronoi {
                let px = (cx + dx as f64).round() as isize;
                let py = (cy + dy as f64).round() as isize;
                if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
                    let idx = py as usize * w + px as usize;
                    if hex_id[idx] == u32::MAX {
                        hex_id[idx] = id;
                    }
                }
            }
            id += 1;
        }
    }

    // Fill unassigned border pixels with nearest
    for _iter in 0..10 {
        let mut changed = false;
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if hex_id[idx] != u32::MAX { continue; }
                if x > 0 && hex_id[idx - 1] != u32::MAX { hex_id[idx] = hex_id[idx - 1]; changed = true; }
                else if x + 1 < w && hex_id[idx + 1] != u32::MAX { hex_id[idx] = hex_id[idx + 1]; changed = true; }
                else if y > 0 && hex_id[idx - w] != u32::MAX { hex_id[idx] = hex_id[idx - w]; changed = true; }
                else if y + 1 < h && hex_id[idx + w] != u32::MAX { hex_id[idx] = hex_id[idx + w]; changed = true; }
            }
        }
        if !changed { break; }
    }

    hex_id
}

/// Compute per-hex saliency from the DC gradient (Source 1 of the saliency field).
/// S(hex) = normalized gradient magnitude of the DC grid.
/// Crests = edges/transitions (high gradient), valleys = flats (low gradient).
pub fn compute_hex_saliency(
    recon_dcs: &[f64],
    grid: &crate::hex::HexGrid,
) -> Vec<f64> {
    let cols = grid.cols;
    let rows = grid.rows;
    let n = cols * rows;
    let mut saliency = vec![0.0f64; n];

    // Compute gradient magnitude at each hex via central differences
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let dc = recon_dcs[idx];

            // Horizontal gradient
            let gx = if col > 0 && col + 1 < cols {
                (recon_dcs[idx + 1] - recon_dcs[idx - 1]) * 0.5
            } else if col + 1 < cols {
                recon_dcs[idx + 1] - dc
            } else if col > 0 {
                dc - recon_dcs[idx - 1]
            } else { 0.0 };

            // Vertical gradient
            let gy = if row > 0 && row + 1 < rows {
                (recon_dcs[(row + 1) * cols + col] - recon_dcs[(row - 1) * cols + col]) * 0.5
            } else if row + 1 < rows {
                recon_dcs[(row + 1) * cols + col] - dc
            } else if row > 0 {
                dc - recon_dcs[(row - 1) * cols + col]
            } else { 0.0 };

            // Weber contrast: normalize by local luminance
            let weber = (gx * gx + gy * gy).sqrt() / (dc.abs() + 10.0);
            saliency[idx] = weber;
        }
    }

    // Normalize to [0, 1]
    let max_s = saliency.iter().cloned().fold(0.0f64, f64::max);
    if max_s > 0.0 {
        for s in saliency.iter_mut() {
            *s = (*s / max_s).min(1.0);
        }
    }

    saliency
}

/// Build a per-pixel saliency map by bilinear interpolation of hex saliency.
/// Same smooth interpolation as the DC prediction — no hex boundary artifacts.
pub fn build_saliency_map(
    hex_saliency: &[f64],
    w: usize, h: usize,
    grid: &crate::hex::HexGrid,
) -> Vec<f64> {
    use crate::hex::{HEX_DX, HEX_DY};

    let n = w * h;
    let mut smap = vec![0.0f64; n];

    for y in 0..h {
        for x in 0..w {
            let col_f = (x as f64 - HEX_DX * 0.5) / HEX_DX;
            let row_f = (y as f64 - HEX_DY * 0.5) / HEX_DY;

            let col0 = (col_f.floor() as isize).max(0) as usize;
            let row0 = (row_f.floor() as isize).max(0) as usize;
            let col1 = (col0 + 1).min(grid.cols.saturating_sub(1));
            let row1 = (row0 + 1).min(grid.rows.saturating_sub(1));

            let tx = (col_f - col0 as f64).clamp(0.0, 1.0);
            let ty = (row_f - row0 as f64).clamp(0.0, 1.0);
            let fx = tx * tx * (3.0 - 2.0 * tx);
            let fy = ty * ty * (3.0 - 2.0 * ty);

            let s00 = hex_saliency[row0 * grid.cols + col0];
            let s10 = hex_saliency[row0 * grid.cols + col1];
            let s01 = hex_saliency[row1 * grid.cols + col0];
            let s11 = hex_saliency[row1 * grid.cols + col1];

            let s_top = s00 * (1.0 - fx) + s10 * fx;
            let s_bot = s01 * (1.0 - fx) + s11 * fx;
            smap[y * w + x] = s_top * (1.0 - fy) + s_bot * fy;
        }
    }

    smap
}

/// Foveal step modulation: step(x,y) = step_base * φ^(-k * S(x,y))
/// Crests (S→1): step shrinks by φ^k → fine quantization → preserve edges
/// Valleys (S→0): step = step_base → coarse quantization → save bits
#[inline]
fn foveal_step(step_base: f64, saliency: f64, k: f64) -> f64 {
    use crate::golden::PHI;
    step_base * PHI.powf(-k * saliency)
}

/// Encode residuals with DPCM on hex-predicted residuals + foveal quantization.
/// DPCM exploits spatial correlation (adjacent residuals are similar) → low entropy.
/// The Gaussian-blurred hex prediction ensures no grid artifacts even with accumulation.
pub fn encode_hex_predicted_residuals(
    plane: &[f64],
    hex_pred: &[f64],
    path: &[usize],
    step: f64,
    dead_zone: f64,
    saliency_map: &[f64],
    _hex_id_map: &[u32],
) -> Vec<i16> {
    let n = path.len();
    let mut deltas = Vec::with_capacity(n);
    let k = 2.0;
    let mut prev_residual_recon = 0.0f64;

    for &idx in path {
        let residual = plane[idx] - hex_pred[idx];
        let delta = residual - prev_residual_recon;

        let s = saliency_map.get(idx).copied().unwrap_or(0.0);
        let local_step = foveal_step(step, s, k);
        let local_dz = dead_zone * s.min(1.0);

        let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
        let qv = (delta.abs() / local_step + 0.5 - local_dz).max(0.0).floor();
        let q = (sign * qv).clamp(-32000.0, 32000.0) as i16;

        deltas.push(q);
        prev_residual_recon += q as f64 * local_step;
    }

    deltas
}

/// Decode residuals with DPCM + hex prediction.
pub fn decode_hex_predicted_residuals(
    deltas: &[i16],
    hex_pred: &[f64],
    path: &[usize],
    step: f64,
    n_pixels: usize,
    saliency_map: &[f64],
    _hex_id_map: &[u32],
) -> Vec<f64> {
    let mut plane = vec![0.0f64; n_pixels];
    let k = 2.0;
    let mut prev_residual_recon = 0.0f64;

    for (i, &idx) in path.iter().enumerate() {
        let s = saliency_map.get(idx).copied().unwrap_or(0.0);
        let local_step = foveal_step(step, s, k);

        let delta = if i < deltas.len() { deltas[i] } else { 0 };
        prev_residual_recon += delta as f64 * local_step;
        plane[idx] = hex_pred[idx] + prev_residual_recon;
    }

    plane
}

/// Encode hex DCs with golden 2D predictor (left/top/diag weighted by φ).
/// Much better than 1D previous-only: captures 2D gradients → smaller residuals.
/// Fine DC step (~1-2) eliminates banding in flat zones.
pub fn encode_hex_dc_grid(
    plane: &[f64], w: usize, h: usize,
    grid: &crate::hex::HexGrid,
    shape: &crate::hex::HexShape,
    dc_step: f64,
) -> (Vec<i16>, Vec<f64>) {
    use crate::golden::{PHI_INV, PHI_INV2, PHI_INV3};

    let cols = grid.cols;
    let rows = grid.rows;
    let n_hexes = cols * rows;
    let mut dc_deltas = Vec::with_capacity(n_hexes);
    let mut recon_dcs = vec![0.0f64; n_hexes];

    // Ultra-fine DC step: in bokeh zones, the DC changes by ~0.3-1.0 per hex.
    // step=1.0 captures these micro-gradients → velvet-smooth interpolation.
    // The golden 2D predictor keeps deltas small, so step=1.0 doesn't hurt bpp much.
    let fine_step = 1.0_f64;

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let (cx, cy) = grid.center(col, row);
            let dc = hex_dc(plane, w, h, cx, cy, shape);

            // Golden 2D predictor: φ-weighted combination of left, top, diagonal
            let pred = if row == 0 && col == 0 {
                0.0
            } else if row == 0 {
                recon_dcs[idx - 1] // left only
            } else if col == 0 {
                recon_dcs[(row - 1) * cols + col] // top only
            } else {
                let left = recon_dcs[idx - 1];
                let top = recon_dcs[(row - 1) * cols + col];
                let diag = recon_dcs[(row - 1) * cols + (col - 1)];
                // Golden weights: φ⁻¹ left + φ⁻² top + φ⁻³ diag
                let w_sum = PHI_INV + PHI_INV2 + PHI_INV3;
                (left * PHI_INV + top * PHI_INV2 + diag * PHI_INV3) / w_sum
            };

            let delta = dc - pred;
            let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
            let qv = (delta.abs() / fine_step + 0.5).floor();
            let q = if qv > 0.0 { (sign * qv).clamp(-32000.0, 32000.0) as i16 } else { 0 };

            let recon_dc = pred + q as f64 * fine_step;
            dc_deltas.push(q);
            recon_dcs[idx] = recon_dc;
        }
    }

    (dc_deltas, recon_dcs)
}

/// Build a prediction map from quantized hex DCs + supercorde gradients.
///
/// Each pixel's prediction = DC + gradient_x * dx + gradient_y * dy
/// where the gradient comes from the hex's codon (LH=horizontal, HL=vertical).
///
/// This captures sharp-to-blur transitions (bokeh edges) as linear ramps
/// within each hex — residuals become near-zero even at strong gradients.
pub fn build_hex_prediction_from_recon_dcs(
    recon_dcs: &[f64], w: usize, h: usize,
    grid: &crate::hex::HexGrid,
    shape: &crate::hex::HexShape,
) -> Vec<f64> {
    // Delegate to gradient version with empty amino acids (DC-only fallback)
    let empty_aa = vec![124u8; recon_dcs.len()]; // 124 = all-intron = zero gradient
    build_hex_gradient_prediction(recon_dcs, &empty_aa, 1.0, w, h, grid, shape)
}

/// Build a smooth prediction map by bilinear interpolation of hex DCs.
///
/// Instead of piecewise-constant (one DC per hex → visible facets),
/// this interpolates between neighboring hex DCs → continuous smooth field.
/// The result is a low-frequency approximation of the image with NO visible
/// hex boundaries — perfect for capturing gradual transitions (bokeh, sky, shadows).
pub fn build_hex_gradient_prediction(
    recon_dcs: &[f64],
    _amino_acids: &[u8],
    _classify_step: f64,
    w: usize, h: usize,
    grid: &crate::hex::HexGrid,
    _shape: &crate::hex::HexShape,
) -> Vec<f64> {
    use crate::hex::{HEX_DX, HEX_DY};

    let n = w * h;
    let mut prediction = vec![0.0f64; n];

    // Build a 2D array of hex DCs for fast lookup
    // recon_dcs is row-major: recon_dcs[row * grid.cols + col]

    for y in 0..h {
        for x in 0..w {
            // Convert pixel (x, y) to continuous hex grid coordinates
            // Hex centers: cx = col * HEX_DX + HEX_DX/2, cy = row * HEX_DY + HEX_DY/2 + offset
            // Invert: col_f = (x - HEX_DX/2) / HEX_DX, row_f = (y - HEX_DY/2) / HEX_DY
            let col_f = (x as f64 - HEX_DX * 0.5) / HEX_DX;
            let row_f = (y as f64 - HEX_DY * 0.5) / HEX_DY;

            // Bilinear interpolation: find 4 surrounding hex centers
            let col0 = (col_f.floor() as isize).max(0) as usize;
            let row0 = (row_f.floor() as isize).max(0) as usize;
            let col1 = (col0 + 1).min(grid.cols.saturating_sub(1));
            let row1 = (row0 + 1).min(grid.rows.saturating_sub(1));

            let tx = (col_f - col0 as f64).clamp(0.0, 1.0);
            let ty = (row_f - row0 as f64).clamp(0.0, 1.0);

            // Smoothstep blend: 3t²-2t³ gives zero derivative at boundaries
            // → no visible grid lines at hex column/row transitions
            let fx = tx * tx * (3.0 - 2.0 * tx);
            let fy = ty * ty * (3.0 - 2.0 * ty);

            // 4 corner DCs
            let dc00 = recon_dcs[row0 * grid.cols + col0];
            let dc10 = recon_dcs[row0 * grid.cols + col1];
            let dc01 = recon_dcs[row1 * grid.cols + col0];
            let dc11 = recon_dcs[row1 * grid.cols + col1];

            // Smooth bicubic-like blend (C1 continuous)
            let dc_top = dc00 * (1.0 - fx) + dc10 * fx;
            let dc_bot = dc01 * (1.0 - fx) + dc11 * fx;
            let dc_interp = dc_top * (1.0 - fy) + dc_bot * fy;

            prediction[y * w + x] = dc_interp;
        }
    }

    // Kill hex grid periodicity: Gaussian blur at sigma ≈ HEX_DX/3.
    // This removes the 8.66px spectral spike from the prediction field
    // while preserving the large-scale structure.
    // Applied identically at encoder and decoder → zero bpp cost.
    let sigma = crate::hex::HEX_DX / 3.0;
    gaussian_blur_inplace(&mut prediction, w, h, sigma);

    prediction
}

/// Fast separable Gaussian blur (horizontal then vertical).
fn gaussian_blur_inplace(data: &mut [f64], w: usize, h: usize, sigma: f64) {
    if sigma < 0.5 { return; }

    let radius = (sigma * 2.5).ceil() as usize;
    let ksize = 2 * radius + 1;
    let mut kernel = vec![0.0f64; ksize];
    let mut ksum = 0.0;
    for i in 0..ksize {
        let x = i as f64 - radius as f64;
        kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        ksum += kernel[i];
    }
    for k in kernel.iter_mut() { *k /= ksum; }

    // Horizontal pass
    let mut row_buf = vec![0.0f64; w];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for ki in 0..ksize {
                let sx = (x as isize + ki as isize - radius as isize)
                    .max(0).min(w as isize - 1) as usize;
                sum += data[y * w + sx] * kernel[ki];
            }
            row_buf[x] = sum;
        }
        data[y * w..y * w + w].copy_from_slice(&row_buf);
    }

    // Vertical pass
    let mut col_buf = vec![0.0f64; h];
    for x in 0..w {
        for y in 0..h {
            let mut sum = 0.0;
            for ki in 0..ksize {
                let sy = (y as isize + ki as isize - radius as isize)
                    .max(0).min(h as isize - 1) as usize;
                sum += data[sy * w + x] * kernel[ki];
            }
            col_buf[y] = sum;
        }
        for y in 0..h {
            data[y * w + x] = col_buf[y];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_roundtrip_constant() {
        // Constant plane: all edges should be zero, reconstruction should be exact.
        let w = 16;
        let h = 16;
        let plane = vec![42.0f64; w * h];
        let edges = compute_edges(&plane, w, h);
        for dir in 0..3 {
            for &e in &edges[dir] {
                assert!((e).abs() < 1e-10, "constant plane should have zero edges");
            }
        }
        let recon = reconstruct_from_edges(&edges, w, h, 42.0, 20);
        for (i, &v) in recon.iter().enumerate() {
            assert!(
                (v - 42.0).abs() < 1e-6,
                "pixel {} should be 42.0, got {}",
                i, v
            );
        }
    }

    #[test]
    fn test_edge_roundtrip_gradient() {
        // Linear horizontal gradient: should reconstruct well.
        let w = 32;
        let h = 16;
        let mut plane = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                plane[y * w + x] = x as f64 * 8.0; // 0..248
            }
        }
        let edges = compute_edges(&plane, w, h);
        let recon = reconstruct_from_edges(&edges, w, h, plane[0], 100);
        let max_err: f64 = plane
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        eprintln!("gradient max error after 100 iters: {:.6}", max_err);
        assert!(max_err < 0.5, "max error {:.6} too high", max_err);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let w = 8;
        let h = 8;
        let mut plane = vec![0.0f64; w * h];
        for y in 0..h {
            for x in 0..w {
                plane[y * w + x] = (x as f64 * 10.0 + y as f64 * 3.0) % 256.0;
            }
        }
        let edges = compute_edges(&plane, w, h);
        let step = 2.0;
        let dz = 0.22;
        let q = quantize_edges(&edges, step, dz);
        let dq = dequantize_edges(&q, step);

        // Dequantized edges should be close to original (within quantization error)
        for dir in 0..3 {
            for i in 0..edges[dir].len() {
                let err = (edges[dir][i] - dq[dir][i]).abs();
                // Max quantization error is about step * (0.5 + dz)
                assert!(
                    err < step * 1.5,
                    "dir={} idx={}: edge={:.2}, dq={:.2}, err={:.2}",
                    dir, i, edges[dir][i], dq[dir][i], err
                );
            }
        }
    }

    #[test]
    fn test_classify_edges_basic() {
        let edges = [
            vec![0.5, 2.0, 5.0, 20.0],
            vec![0.0, 0.1, 3.0, 15.0],
            vec![0.0, 0.0, 0.0, 50.0],
        ];
        let (ir, vis, uv) = classify_edges(&edges, 10.0, 1.0);
        // IR (>10): 20, 15, 50 = 3
        // Visible (>1 and <=10): 2.0, 5.0, 3.0 = 3
        // UV (<=1): 0.5, 0.0, 0.1, 0.0, 0.0, 0.0 = 6
        assert_eq!(ir, 3, "expected 3 IR edges");
        assert_eq!(vis, 3, "expected 3 visible edges");
        assert_eq!(uv, 6, "expected 6 UV edges");
    }

    #[test]
    fn test_edge_stats_basic() {
        let q = [
            vec![0i16, 1, -2, 0],
            vec![0, 0, 0, 5],
            vec![0, 0, 0, 0],
        ];
        let stats = edge_stats(&q);
        assert_eq!(stats.total_edges, 12);
        assert_eq!(stats.zero_edges, 9);
        assert!((stats.zero_fraction - 0.75).abs() < 1e-10);
        assert!((stats.max_abs - 5.0).abs() < 1e-10);
    }
}
