//! Hexagonal grid geometry for DNA5 honeycomb tessellation.
//!
//! Flat-top hexagons with radius R=5 (Fibonacci number), providing
//! the coordinate system, pixel-to-hex mapping, and precomputed hex shape.

use std::f64::consts::PI;

/// Hex radius (Fibonacci number).
pub const HEX_R: usize = 5;

/// Horizontal spacing between hex centers: R * sqrt(3).
pub const HEX_DX: f64 = 8.660254037844387; // 5 * sqrt(3)

/// Vertical spacing between hex centers: R * 1.5.
pub const HEX_DY: f64 = 7.5;

/// Number of perimeter samples: 6 edges * R.
pub const HEX_PERIMETER: usize = 30;

// ---------------------------------------------------------------------------
// HexShape — precomputed pixel offsets relative to hex center
// ---------------------------------------------------------------------------

/// Precomputed pixel offsets for a single hexagon, relative to center (0,0).
pub struct HexShape {
    /// ~24-30 boundary pixels, clockwise from North vertex.
    pub perimeter: Vec<(i32, i32)>,
    /// ~50-80 pixels strictly inside the hex (excluding perimeter).
    pub interior: Vec<(i32, i32)>,
    /// All pixels closer to this center than to any of the 6 neighbor centers.
    pub voronoi: Vec<(i32, i32)>,
}

/// Compute the precomputed hex shape for a flat-top hexagon of radius `HEX_R`.
pub fn compute_hex_shape() -> HexShape {
    compute_hex_shape_r(HEX_R)
}

/// Compute the precomputed hex shape for a flat-top hexagon of arbitrary radius `r`.
///
/// Perimeter: 6 edges x r pixels.
/// Interior: all pixels inside hex of radius r.
/// Voronoi: all pixels closer to center than to any of the 6 neighbor centers.
pub fn compute_hex_shape_r(r_int: usize) -> HexShape {
    let r = r_int as f64;
    // dx = r * sqrt(3), dy = r * 1.5 (used by callers, not internally)

    // 1. Compute 6 vertices of a flat-top hexagon.
    //    Vertex i at angle = pi/3 * i  (flat-top: vertex 0 points right).
    //    We use the convention: angle = pi/3 * i  for i in 0..6.
    let mut vertices = Vec::with_capacity(6);
    for i in 0..6 {
        let angle = PI / 3.0 * i as f64;
        let vx = r * angle.cos();
        let vy = r * angle.sin();
        vertices.push((vx, vy));
    }

    // 2. Walk each edge sampling r_int points per edge -> ~6*r perimeter samples.
    //    For edge from vertex[i] to vertex[(i+1)%6], sample r_int points (excluding endpoint).
    let perimeter_cap = 6 * r_int + 6;
    let mut perimeter_set: Vec<(i32, i32)> = Vec::with_capacity(perimeter_cap);
    let mut seen_peri = std::collections::HashSet::new();

    for i in 0..6 {
        let (x0, y0) = vertices[i];
        let (x1, y1) = vertices[(i + 1) % 6];
        for s in 0..r_int {
            let t = s as f64 / r_int as f64;
            let px = x0 + t * (x1 - x0);
            let py = y0 + t * (y1 - y0);
            let ix = px.round() as i32;
            let iy = py.round() as i32;
            if seen_peri.insert((ix, iy)) {
                perimeter_set.push((ix, iy));
            }
        }
    }

    let perimeter = perimeter_set;

    // 3. Interior: scan bounding box, keep points inside hex AND not on perimeter
    //    AND closer to center than to any neighbor (Voronoi-consistent).
    let peri_lookup: std::collections::HashSet<(i32, i32)> =
        perimeter.iter().copied().collect();
    let neighbors = hex_neighbor_centers_r(r);

    let bound = (r + 1.0).ceil() as i32;
    let mut interior = Vec::with_capacity((r * r * 3.0) as usize);
    for iy in -bound..=bound {
        for ix in -bound..=bound {
            if peri_lookup.contains(&(ix, iy)) {
                continue;
            }
            if is_inside_hex(ix as f64, iy as f64, r) {
                // Also check Voronoi membership: must be closer to origin
                // than to any neighbor center (tie = include).
                let d0 = (ix * ix + iy * iy) as f64;
                let in_voronoi = neighbors.iter().all(|&(nx, ny)| {
                    let ddx = ix as f64 - nx;
                    let ddy = iy as f64 - ny;
                    d0 <= ddx * ddx + ddy * ddy
                });
                if in_voronoi {
                    interior.push((ix, iy));
                }
            }
        }
    }

    // 4. Voronoi: scan wider area, keep points closer to (0,0) than to any
    //    of the 6 neighbor hex centers.  (`neighbors` already computed above.)
    let scan_r = (r * 1.5 + 2.0).ceil() as i32;
    let mut voronoi = Vec::with_capacity((r * r * 4.0) as usize);
    for iy in -scan_r..=scan_r {
        for ix in -scan_r..=scan_r {
            let d0 = (ix as f64) * (ix as f64) + (iy as f64) * (iy as f64);
            let mut closest = true;
            for &(nx, ny) in &neighbors {
                let ddx = ix as f64 - nx;
                let ddy = iy as f64 - ny;
                if ddx * ddx + ddy * ddy < d0 {
                    closest = false;
                    break;
                }
            }
            if closest {
                voronoi.push((ix, iy));
            }
        }
    }

    HexShape { perimeter, interior, voronoi }
}

/// Horizontal spacing between hex centers for a given radius: R * sqrt(3).
#[inline]
pub fn hex_dx_r(r: f64) -> f64 {
    r * 3.0_f64.sqrt()
}

/// Vertical spacing between hex centers for a given radius: R * 1.5.
#[inline]
pub fn hex_dy_r(r: f64) -> f64 {
    r * 1.5
}

/// Returns the 6 neighbor hex center offsets for a flat-top hex grid (even-q)
/// with the default HEX_R radius.
#[allow(dead_code)]
fn hex_neighbor_centers() -> [(f64, f64); 6] {
    hex_neighbor_centers_r(HEX_R as f64)
}

/// Returns the 6 neighbor hex center offsets for a flat-top hex grid (even-q)
/// with arbitrary radius `r`.
fn hex_neighbor_centers_r(r: f64) -> [(f64, f64); 6] {
    let dx = hex_dx_r(r);
    let dy = hex_dy_r(r);
    [
        (dx, 0.0),           // East
        (dx / 2.0, dy),      // SE
        (-dx / 2.0, dy),     // SW
        (-dx, 0.0),          // West
        (-dx / 2.0, -dy),    // NW
        (dx / 2.0, -dy),     // NE
    ]
}

/// Test whether a point (x, y) lies inside a flat-top regular hexagon
/// centered at origin with circumradius `r`.
///
/// For a flat-top hex the three constraints are:
///   |y| <= r
///   |x| <= r * cos(30deg)     i.e.  r * sqrt(3)/2
///   |x| + |y| / tan(60deg) <= r * cos(30deg)
///       equivalently  |x| + |y| / sqrt(3) <= r * sqrt(3)/2
pub fn is_inside_hex(x: f64, y: f64, r: f64) -> bool {
    let ax = x.abs();
    let ay = y.abs();
    let half_w = r * (3.0_f64.sqrt() / 2.0); // r * cos(30)
    ay <= r && ax <= half_w && (ax + ay / 3.0_f64.sqrt()) <= half_w
}

// ---------------------------------------------------------------------------
// HexGrid — layout of hexagons covering an image
// ---------------------------------------------------------------------------

/// Hexagonal grid covering an image, using even-q offset coordinates (flat-top).
///
/// Odd columns are shifted down by `HEX_DY / 2`.
pub struct HexGrid {
    /// Number of hex columns.
    pub cols: usize,
    /// Number of hex rows.
    pub rows: usize,
    /// Source image width in pixels.
    pub img_w: usize,
    /// Source image height in pixels.
    pub img_h: usize,
}

impl HexGrid {
    /// Create a hex grid that covers the entire image.
    pub fn new(img_w: usize, img_h: usize) -> Self {
        // Columns: first center at x ~ HEX_DX/2, then every HEX_DX.
        // We need enough columns so that the last center + R covers img_w.
        let cols = ((img_w as f64) / HEX_DX).ceil() as usize + 1;
        // Rows: first center at y ~ HEX_DY/2, then every HEX_DY.
        // Odd columns shift by HEX_DY/2, so we need one extra row to cover.
        let rows = ((img_h as f64) / HEX_DY).ceil() as usize + 1;
        HexGrid { cols, rows, img_w, img_h }
    }

    /// Total number of hexes in the grid.
    #[inline]
    pub fn n_hexes(&self) -> usize {
        self.cols * self.rows
    }

    /// Pixel center (x, y) of hex at grid position (col, row).
    ///
    /// Even-q offset: odd columns are shifted down by `HEX_DY / 2`.
    #[inline]
    pub fn center(&self, col: usize, row: usize) -> (f64, f64) {
        let cx = col as f64 * HEX_DX + HEX_DX / 2.0;
        let cy = row as f64 * HEX_DY + HEX_DY / 2.0
            + if col % 2 == 1 { HEX_DY / 2.0 } else { 0.0 };
        (cx, cy)
    }

    /// Iterate all (col, row) in raster order (row-major).
    pub fn iter_raster(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.rows).flat_map(move |r| (0..self.cols).map(move |c| (c, r)))
    }
}

// ---------------------------------------------------------------------------
// Fractal granulometry — multi-scale hexagonal tessellation
// ---------------------------------------------------------------------------

/// Fibonacci hex radii for multi-scale tessellation.
pub const FIB_RADII: [usize; 4] = [3, 5, 8, 13];

/// Region size in pixels for the multi-scale radius map.
/// Each 16x16 region gets a single Fibonacci hex radius.
pub const GRANULOMETRY_REGION_SIZE: usize = 16;

/// Estimate local fractal dimension using box-counting at a pixel position.
/// Analyzes variance at multiple scales (2x2, 4x4, 8x8, 16x16) around (cx, cy).
/// Returns a value in [1.0, 2.0]: 1.0 = smooth, 2.0 = maximally complex.
pub fn local_fractal_dimension(
    plane: &[f64],
    w: usize,
    h: usize,
    cx: usize,
    cy: usize,
) -> f64 {
    let scales = [2usize, 4, 8, 16];
    let mut log_n = Vec::new();
    let mut log_inv_s = Vec::new();
    let threshold = 4.0; // variance threshold for "active" box

    for &s in &scales {
        let half = 8; // analyze 16x16 neighborhood
        let x0 = cx.saturating_sub(half);
        let y0 = cy.saturating_sub(half);
        let x1 = (cx + half).min(w);
        let y1 = (cy + half).min(h);

        let mut active = 0usize;
        let mut total = 0usize;

        let mut by = y0;
        while by + s <= y1 {
            let mut bx = x0;
            while bx + s <= x1 {
                // Compute variance of this s x s block
                let mut sum = 0.0f64;
                let mut sum2 = 0.0f64;
                let mut cnt = 0.0f64;
                for r in by..by + s {
                    for c in bx..bx + s {
                        if r < h && c < w {
                            let v = plane[r * w + c];
                            sum += v;
                            sum2 += v * v;
                            cnt += 1.0;
                        }
                    }
                }
                if cnt > 0.0 {
                    let mean = sum / cnt;
                    let var = sum2 / cnt - mean * mean;
                    if var > threshold {
                        active += 1;
                    }
                    total += 1;
                }
                bx += s;
            }
            by += s;
        }

        if total > 0 && active > 0 {
            log_n.push((active as f64).ln());
            log_inv_s.push((1.0 / s as f64).ln());
        }
    }

    // Linear regression: D = slope of log(N) vs log(1/s)
    if log_n.len() < 2 {
        return 1.0;
    }

    let n = log_n.len() as f64;
    let sx: f64 = log_inv_s.iter().sum();
    let sy: f64 = log_n.iter().sum();
    let sxx: f64 = log_inv_s.iter().map(|x| x * x).sum();
    let sxy: f64 = log_inv_s
        .iter()
        .zip(log_n.iter())
        .map(|(x, y)| x * y)
        .sum();

    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-10 {
        return 1.0;
    }

    let slope = (n * sxy - sx * sy) / denom;
    slope.clamp(1.0, 2.0)
}

/// Choose hex radius from fractal dimension.
/// D near 1.0 -> smooth -> large hex R=13 (massive compression).
/// D near 2.0 -> complex -> small hex R=3 (precision).
pub fn radius_from_fractal_dim(d: f64) -> usize {
    if d < 1.2 {
        13
    } else if d < 1.4 {
        8
    } else if d < 1.7 {
        5
    } else {
        3
    }
}

/// Encode radius as 2-bit index: 0=R3, 1=R5, 2=R8, 3=R13.
pub fn radius_to_2bit(r: usize) -> u8 {
    match r {
        3 => 0,
        5 => 1,
        8 => 2,
        13 => 3,
        _ => 1, // default to R=5
    }
}

/// Decode 2-bit index back to radius.
pub fn radius_from_2bit(code: u8) -> usize {
    match code & 0x03 {
        0 => 3,
        1 => 5,
        2 => 8,
        3 => 13,
        _ => unreachable!(),
    }
}

/// Build a radius map for the image: one radius per 16x16 region.
///
/// Returns (radius_map, n_regions_x, n_regions_y).
/// Each element is a Fibonacci radius (3, 5, 8, or 13).
pub fn build_radius_map(
    plane: &[f64],
    w: usize,
    h: usize,
) -> (Vec<usize>, usize, usize) {
    let n_rx = (w + GRANULOMETRY_REGION_SIZE - 1) / GRANULOMETRY_REGION_SIZE;
    let n_ry = (h + GRANULOMETRY_REGION_SIZE - 1) / GRANULOMETRY_REGION_SIZE;
    let mut map = Vec::with_capacity(n_rx * n_ry);

    for ry in 0..n_ry {
        for rx in 0..n_rx {
            let cx = rx * GRANULOMETRY_REGION_SIZE + GRANULOMETRY_REGION_SIZE / 2;
            let cy = ry * GRANULOMETRY_REGION_SIZE + GRANULOMETRY_REGION_SIZE / 2;
            let cx = cx.min(w.saturating_sub(1));
            let cy = cy.min(h.saturating_sub(1));
            let d = local_fractal_dimension(plane, w, h, cx, cy);
            map.push(radius_from_fractal_dim(d));
        }
    }

    (map, n_rx, n_ry)
}

/// Pack radius map into bytes (2 bits per region, 4 regions per byte, MSB first).
pub fn pack_radius_map(map: &[usize]) -> Vec<u8> {
    let n_bytes = (map.len() + 3) / 4;
    let mut bytes = vec![0u8; n_bytes];
    for (i, &r) in map.iter().enumerate() {
        let byte_idx = i / 4;
        let shift = 6 - (i % 4) * 2;
        bytes[byte_idx] |= radius_to_2bit(r) << shift;
    }
    bytes
}

/// Unpack radius map from bytes (2 bits per region, 4 regions per byte, MSB first).
pub fn unpack_radius_map(bytes: &[u8], n_regions: usize) -> Vec<usize> {
    let mut map = Vec::with_capacity(n_regions);
    for i in 0..n_regions {
        let byte_idx = i / 4;
        let shift = 6 - (i % 4) * 2;
        if byte_idx < bytes.len() {
            let code = (bytes[byte_idx] >> shift) & 0x03;
            map.push(radius_from_2bit(code));
        } else {
            map.push(HEX_R); // default
        }
    }
    map
}

/// Look up the assigned radius for a pixel position (px, py) from the radius map.
#[inline]
pub fn radius_at_pixel(
    px: usize,
    py: usize,
    radius_map: &[usize],
    n_rx: usize,
    _n_ry: usize,
) -> usize {
    let rx = px / GRANULOMETRY_REGION_SIZE;
    let ry = py / GRANULOMETRY_REGION_SIZE;
    let idx = ry * n_rx + rx;
    if idx < radius_map.len() {
        radius_map[idx]
    } else {
        HEX_R
    }
}

/// A multi-scale hexagonal grid where each region tile uses hexes of a given
/// Fibonacci radius.
///
/// The grid is a set of (center_x, center_y, radius, region_index) entries,
/// one per hex cell.
pub struct MultiScaleHexGrid {
    /// For each cell: (center_x, center_y, radius)
    pub cells: Vec<(f64, f64, usize)>,
    pub img_w: usize,
    pub img_h: usize,
}

impl MultiScaleHexGrid {
    /// Build the multi-scale grid from a radius map.
    ///
    /// For each region, tile it with hexes of the assigned radius.
    /// Hexes that have already been placed (by an adjacent region with a
    /// larger radius) are skipped via an occupancy grid.
    pub fn build(
        radius_map: &[usize],
        n_rx: usize,
        n_ry: usize,
        img_w: usize,
        img_h: usize,
    ) -> Self {
        // Process regions from LARGEST radius to smallest (greedy: big hexes claim first)
        let mut region_order: Vec<usize> = (0..n_rx * n_ry).collect();
        region_order.sort_by(|&a, &b| radius_map[b].cmp(&radius_map[a]));

        // Occupancy grid: tracks which pixel regions are already covered
        // (resolution = 1 pixel; use a bitmap for efficiency)
        let mut occupied = vec![false; img_w * img_h];
        let mut cells: Vec<(f64, f64, usize)> = Vec::new();

        for &region_idx in &region_order {
            let rx = region_idx % n_rx;
            let ry = region_idx / n_rx;
            let r = radius_map[region_idx];
            let rf = r as f64;
            let dx = hex_dx_r(rf);
            let dy = hex_dy_r(rf);

            // Region pixel bounds
            let region_x0 = rx * GRANULOMETRY_REGION_SIZE;
            let region_y0 = ry * GRANULOMETRY_REGION_SIZE;
            let region_x1 = ((rx + 1) * GRANULOMETRY_REGION_SIZE).min(img_w);
            let region_y1 = ((ry + 1) * GRANULOMETRY_REGION_SIZE).min(img_h);

            // Tile this region with hexes of radius r, using even-q layout
            let n_cols = ((region_x1 - region_x0) as f64 / dx).ceil() as usize + 1;
            let n_rows = ((region_y1 - region_y0) as f64 / dy).ceil() as usize + 1;

            for row in 0..n_rows {
                for col in 0..n_cols {
                    let cx = region_x0 as f64 + col as f64 * dx + dx / 2.0;
                    let cy = region_y0 as f64 + row as f64 * dy + dy / 2.0
                        + if col % 2 == 1 { dy / 2.0 } else { 0.0 };

                    // Check if this hex center is inside the image
                    let cxi = cx.round() as isize;
                    let cyi = cy.round() as isize;
                    if cxi < 0 || cxi >= img_w as isize || cyi < 0 || cyi >= img_h as isize {
                        continue;
                    }
                    let cxu = cxi as usize;
                    let cyu = cyi as usize;

                    // Check if center is within the region (with some tolerance for border hexes)
                    if cxu < region_x0.saturating_sub(r) || cxu >= region_x1 + r {
                        continue;
                    }
                    if cyu < region_y0.saturating_sub(r) || cyu >= region_y1 + r {
                        continue;
                    }

                    // Check occupancy: skip if center is already covered
                    if occupied[cyu * img_w + cxu] {
                        continue;
                    }

                    // Claim pixels for this hex (approximate: mark a square of r pixels)
                    let claim_r = r;
                    let y0 = cyu.saturating_sub(claim_r);
                    let y1 = (cyu + claim_r + 1).min(img_h);
                    let x0 = cxu.saturating_sub(claim_r);
                    let x1 = (cxu + claim_r + 1).min(img_w);
                    for py in y0..y1 {
                        for px in x0..x1 {
                            occupied[py * img_w + px] = true;
                        }
                    }

                    cells.push((cx, cy, r));
                }
            }
        }

        MultiScaleHexGrid {
            cells,
            img_w,
            img_h,
        }
    }

    /// Number of hex cells in the multi-scale grid.
    pub fn n_hexes(&self) -> usize {
        self.cells.len()
    }
}

// ---------------------------------------------------------------------------
// Hexagonal subdivision (fracturing) for DNA5
// ---------------------------------------------------------------------------

/// Subdivision code for a hexagon (2 bits).
///
/// Controls how the decoder reconstructs interior pixels:
/// - `Whole`:   no subdivision, normal harmonic decay
/// - `Bisect`:  diagonal V0-V3 splits hex into 2 halves, slower texture decay
/// - `Trisect`: diagonals V0-V2, V2-V4, V4-V0 create 3 regions
/// - `Full`:    6 triangles from center, vertex charges dominate interior
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum HexSubdiv {
    Whole = 0,    // No subdivision: 1 region
    Bisect = 1,   // Diagonal V0-V3: 2 regions
    Trisect = 2,  // Diagonals V0-V2, V2-V4, V4-V0: 3 regions
    Full = 3,     // All diagonals from center: 6 triangles
}

impl HexSubdiv {
    /// Convert from a raw 2-bit value.
    pub fn from_u8(v: u8) -> Self {
        match v & 0x03 {
            0 => HexSubdiv::Whole,
            1 => HexSubdiv::Bisect,
            2 => HexSubdiv::Trisect,
            3 => HexSubdiv::Full,
            _ => unreachable!(),
        }
    }

    /// Texture field decay power. Lower values = slower decay = vertex charges
    /// reach further into the interior, preserving more detail.
    pub fn decay_power(self) -> f64 {
        match self {
            HexSubdiv::Whole => 1.0,     // normal decay: phi^(-d)
            HexSubdiv::Bisect => 0.7,    // slower decay: phi^(-0.7d)
            HexSubdiv::Trisect => 0.5,   // even slower: phi^(-0.5d)
            HexSubdiv::Full => 0.3,      // very slow: phi^(-0.3d), charges dominate
        }
    }
}

impl Default for HexSubdiv {
    fn default() -> Self {
        HexSubdiv::Whole
    }
}

/// Signed-area test: returns positive if (px, py) is to the LEFT of the line
/// from (x0, y0) to (x1, y1), negative if to the right, zero if on the line.
#[inline]
fn signed_area(x0: f64, y0: f64, x1: f64, y1: f64, px: f64, py: f64) -> f64 {
    (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
}

/// For a given subdivision, return the sub-region index (0..max) for each
/// interior pixel. Uses signed-area tests against internal diagonal lines.
///
/// `vertices` are the 6 vertex positions of the hex (absolute coordinates).
/// `interior_offsets` are the interior pixel offsets from the shape.
/// `cx`, `cy` are the hex center coordinates.
///
/// Returns a Vec of `u8` with one sub-region index per interior pixel.
pub fn subdivide_interior(
    interior_offsets: &[(i32, i32)],
    subdiv: HexSubdiv,
    vertices: &[(f64, f64); 6],
    cx: f64,
    cy: f64,
) -> Vec<u8> {
    match subdiv {
        HexSubdiv::Whole => {
            // All pixels belong to region 0
            vec![0u8; interior_offsets.len()]
        }
        HexSubdiv::Bisect => {
            // Diagonal V0-V3 splits hex into 2 halves
            let (x0, y0) = vertices[0];
            let (x1, y1) = vertices[3];
            interior_offsets
                .iter()
                .map(|&(dx, dy)| {
                    let px = cx + dx as f64;
                    let py = cy + dy as f64;
                    if signed_area(x0, y0, x1, y1, px, py) >= 0.0 {
                        0
                    } else {
                        1
                    }
                })
                .collect()
        }
        HexSubdiv::Trisect => {
            // Diagonals V0-V2, V2-V4, V4-V0 create 3 quad-like regions
            let (x0, y0) = vertices[0];
            let (x2, y2) = vertices[2];
            let (x4, y4) = vertices[4];
            interior_offsets
                .iter()
                .map(|&(dx, dy)| {
                    let px = cx + dx as f64;
                    let py = cy + dy as f64;
                    let s02 = signed_area(x0, y0, x2, y2, px, py);
                    let s24 = signed_area(x2, y2, x4, y4, px, py);
                    let s40 = signed_area(x4, y4, x0, y0, px, py);
                    // Inside the triangle V0-V2-V4: all three positive (or zero)
                    if s02 >= 0.0 && s24 >= 0.0 && s40 >= 0.0 {
                        0
                    } else if s02 < 0.0 {
                        // Between V0-V2 and the hex boundary
                        1
                    } else {
                        // Between V2-V4 or V4-V0 and the hex boundary
                        2
                    }
                })
                .collect()
        }
        HexSubdiv::Full => {
            // 6 triangles: (center, V_i, V_{i+1}) for i=0..5
            interior_offsets
                .iter()
                .map(|&(dx, dy)| {
                    let px = cx + dx as f64;
                    let py = cy + dy as f64;
                    // Find which triangle the pixel belongs to by checking
                    // signed area against (center, V_i, V_{i+1})
                    for i in 0..6 {
                        let j = (i + 1) % 6;
                        let (vix, viy) = vertices[i];
                        let (vjx, vjy) = vertices[j];
                        // Check if point is inside triangle (cx, cy) -> V_i -> V_j
                        let s1 = signed_area(cx, cy, vix, viy, px, py);
                        let s2 = signed_area(vix, viy, vjx, vjy, px, py);
                        let s3 = signed_area(vjx, vjy, cx, cy, px, py);
                        if s1 >= 0.0 && s2 >= 0.0 && s3 >= 0.0 {
                            return i as u8;
                        }
                    }
                    // Fallback: find closest triangle by angle from center
                    let angle = (py - cy).atan2(px - cx);
                    let sector = ((angle / (PI / 3.0)).floor() as i32).rem_euclid(6);
                    sector as u8
                })
                .collect()
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
    fn test_hex_shape_perimeter_count() {
        let shape = compute_hex_shape();
        let n = shape.perimeter.len();
        println!("perimeter count = {}", n);
        assert!(
            (24..=30).contains(&n),
            "perimeter count {} not in [24, 30]",
            n
        );
    }

    #[test]
    fn test_hex_shape_interior_count() {
        let shape = compute_hex_shape();
        let n = shape.interior.len();
        println!("interior count = {}", n);
        assert!(
            (45..=80).contains(&n),
            "interior count {} not in [45, 80]",
            n
        );
    }

    #[test]
    fn test_hex_shape_voronoi_covers_interior() {
        let shape = compute_hex_shape();
        let voronoi_set: std::collections::HashSet<(i32, i32)> =
            shape.voronoi.iter().copied().collect();
        for &pt in &shape.interior {
            assert!(
                voronoi_set.contains(&pt),
                "interior pixel {:?} not in Voronoi set",
                pt
            );
        }
    }

    #[test]
    fn test_hex_grid_dimensions() {
        let grid = HexGrid::new(2688, 1536);
        println!("grid: {}cols x {}rows = {} hexes", grid.cols, grid.rows, grid.n_hexes());
        assert!(
            (300..=320).contains(&grid.cols),
            "cols {} not near 310",
            grid.cols
        );
        assert!(
            (200..=210).contains(&grid.rows),
            "rows {} not near 205",
            grid.rows
        );
    }

    #[test]
    fn test_hex_center_even_odd_offset() {
        let grid = HexGrid::new(100, 100);
        let (_, y_even) = grid.center(0, 0);
        let (_, y_odd) = grid.center(1, 0);
        let diff = (y_odd - y_even - HEX_DY / 2.0).abs();
        println!(
            "even y={:.2}, odd y={:.2}, diff from dy/2={:.4}",
            y_even, y_odd, diff
        );
        assert!(
            diff < 1e-9,
            "odd column not offset by dy/2: even={}, odd={}, expected diff={}",
            y_even,
            y_odd,
            HEX_DY / 2.0
        );
    }

    #[test]
    fn test_hex_subdiv_from_u8_roundtrip() {
        assert_eq!(HexSubdiv::from_u8(0), HexSubdiv::Whole);
        assert_eq!(HexSubdiv::from_u8(1), HexSubdiv::Bisect);
        assert_eq!(HexSubdiv::from_u8(2), HexSubdiv::Trisect);
        assert_eq!(HexSubdiv::from_u8(3), HexSubdiv::Full);
        // Masking: only lower 2 bits matter
        assert_eq!(HexSubdiv::from_u8(0xFF), HexSubdiv::Full);
        assert_eq!(HexSubdiv::from_u8(0x04), HexSubdiv::Whole);
    }

    #[test]
    fn test_hex_subdiv_decay_power_ordering() {
        // More subdivision = slower decay (lower power)
        assert!(HexSubdiv::Whole.decay_power() > HexSubdiv::Bisect.decay_power());
        assert!(HexSubdiv::Bisect.decay_power() > HexSubdiv::Trisect.decay_power());
        assert!(HexSubdiv::Trisect.decay_power() > HexSubdiv::Full.decay_power());
    }

    #[test]
    fn test_subdivide_interior_whole() {
        let shape = compute_hex_shape();
        let r = HEX_R as f64;
        let cx = 10.0;
        let cy = 10.0;
        let mut vertices = [(0.0, 0.0); 6];
        for i in 0..6 {
            let angle = PI / 3.0 * i as f64;
            vertices[i] = (cx + r * angle.cos(), cy + r * angle.sin());
        }
        let regions = subdivide_interior(&shape.interior, HexSubdiv::Whole, &vertices, cx, cy);
        assert_eq!(regions.len(), shape.interior.len());
        // All pixels should be region 0
        assert!(regions.iter().all(|&r| r == 0));
    }

    #[test]
    fn test_subdivide_interior_bisect_both_sides() {
        let shape = compute_hex_shape();
        let r = HEX_R as f64;
        let cx = 0.0;
        let cy = 0.0;
        let mut vertices = [(0.0, 0.0); 6];
        for i in 0..6 {
            let angle = PI / 3.0 * i as f64;
            vertices[i] = (cx + r * angle.cos(), cy + r * angle.sin());
        }
        let regions = subdivide_interior(&shape.interior, HexSubdiv::Bisect, &vertices, cx, cy);
        assert_eq!(regions.len(), shape.interior.len());
        let n_region0 = regions.iter().filter(|&&r| r == 0).count();
        let n_region1 = regions.iter().filter(|&&r| r == 1).count();
        println!("bisect: region0={}, region1={}", n_region0, n_region1);
        // Both sides should have some pixels
        assert!(n_region0 > 0, "bisect should have pixels in region 0");
        assert!(n_region1 > 0, "bisect should have pixels in region 1");
        // Only regions 0 and 1
        assert!(regions.iter().all(|&r| r <= 1));
    }

    #[test]
    fn test_subdivide_interior_full_six_triangles() {
        let shape = compute_hex_shape();
        let r = HEX_R as f64;
        let cx = 0.0;
        let cy = 0.0;
        let mut vertices = [(0.0, 0.0); 6];
        for i in 0..6 {
            let angle = PI / 3.0 * i as f64;
            vertices[i] = (cx + r * angle.cos(), cy + r * angle.sin());
        }
        let regions = subdivide_interior(&shape.interior, HexSubdiv::Full, &vertices, cx, cy);
        assert_eq!(regions.len(), shape.interior.len());
        // Should have up to 6 different regions
        let max_region = *regions.iter().max().unwrap_or(&0);
        println!("full: max_region={}, total interior={}", max_region, regions.len());
        assert!(max_region <= 5, "full subdivision should have at most 6 regions (0..5)");
        // Should use multiple regions
        let distinct: std::collections::HashSet<u8> = regions.iter().copied().collect();
        println!("full: distinct regions = {:?}", distinct);
        assert!(distinct.len() >= 3, "full subdivision should produce at least 3 distinct regions, got {}", distinct.len());
    }

    // --- Fractal granulometry tests ---

    #[test]
    fn test_compute_hex_shape_r_all_fib_radii() {
        for &r in &FIB_RADII {
            let shape = compute_hex_shape_r(r);
            let n_peri = shape.perimeter.len();
            let n_int = shape.interior.len();
            let n_vor = shape.voronoi.len();
            println!(
                "R={}: perimeter={}, interior={}, voronoi={}",
                r, n_peri, n_int, n_vor
            );
            // Perimeter: should be roughly 6*r but some dedup
            assert!(
                n_peri >= r * 3 && n_peri <= r * 7,
                "R={}: perimeter count {} unexpected",
                r, n_peri
            );
            // Interior should be nonempty for r >= 3
            assert!(
                n_int > 0,
                "R={}: interior should be nonempty, got {}",
                r, n_int
            );
            // Voronoi should cover interior
            let voronoi_set: std::collections::HashSet<(i32, i32)> =
                shape.voronoi.iter().copied().collect();
            for &pt in &shape.interior {
                assert!(
                    voronoi_set.contains(&pt),
                    "R={}: interior pixel {:?} not in Voronoi set",
                    r, pt
                );
            }
        }
    }

    #[test]
    fn test_fractal_dimension_smooth() {
        // Constant plane: fractal dimension should be ~1.0
        let w = 64;
        let h = 64;
        let plane = vec![128.0; w * h];
        let d = local_fractal_dimension(&plane, w, h, 32, 32);
        println!("constant plane: D = {:.3}", d);
        assert!(d < 1.3, "constant plane D={:.3} should be < 1.3", d);
    }

    #[test]
    fn test_fractal_dimension_noisy() {
        // Random noise: fractal dimension should be near 2.0
        let w = 64;
        let h = 64;
        let mut plane = vec![0.0f64; w * h];
        // Deterministic pseudo-random noise (LCG)
        let mut rng = 12345u64;
        for v in plane.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((rng >> 32) as f64 / u32::MAX as f64) * 255.0;
        }
        let d = local_fractal_dimension(&plane, w, h, 32, 32);
        println!("noisy plane: D = {:.3}", d);
        assert!(d > 1.5, "noisy plane D={:.3} should be > 1.5", d);
    }

    #[test]
    fn test_radius_from_fractal_dim_ordering() {
        // Smooth -> large radius, complex -> small radius
        let r_smooth = radius_from_fractal_dim(1.0);
        let r_mid = radius_from_fractal_dim(1.5);
        let r_complex = radius_from_fractal_dim(2.0);
        println!(
            "D=1.0 -> R={}, D=1.5 -> R={}, D=2.0 -> R={}",
            r_smooth, r_mid, r_complex
        );
        assert!(r_smooth >= r_mid);
        assert!(r_mid >= r_complex);
        assert_eq!(r_smooth, 13);
        assert_eq!(r_complex, 3);
    }

    #[test]
    fn test_radius_2bit_roundtrip() {
        for &r in &FIB_RADII {
            let code = radius_to_2bit(r);
            let r2 = radius_from_2bit(code);
            assert_eq!(r, r2, "R={} -> code={} -> R={}", r, code, r2);
        }
    }

    #[test]
    fn test_radius_map_pack_unpack() {
        let map = vec![3, 5, 8, 13, 3, 5, 8];
        let packed = pack_radius_map(&map);
        let unpacked = unpack_radius_map(&packed, map.len());
        assert_eq!(map, unpacked);
    }

    #[test]
    fn test_build_radius_map_basic() {
        let w = 128;
        let h = 128;
        let plane = vec![128.0; w * h]; // smooth plane
        let (map, n_rx, n_ry) = build_radius_map(&plane, w, h);
        println!(
            "radius map: {}x{} = {} regions",
            n_rx, n_ry,
            map.len()
        );
        assert_eq!(map.len(), n_rx * n_ry);
        // All smooth -> should get large radii (R=13 or R=8)
        for &r in &map {
            assert!(
                FIB_RADII.contains(&r),
                "radius {} not a Fibonacci radius",
                r
            );
        }
    }

    #[test]
    fn test_multi_scale_hex_grid_builds() {
        let w = 128;
        let h = 128;
        let plane = vec![128.0; w * h];
        let (map, n_rx, n_ry) = build_radius_map(&plane, w, h);
        let grid = MultiScaleHexGrid::build(&map, n_rx, n_ry, w, h);
        println!("multi-scale grid: {} hexes", grid.n_hexes());
        assert!(grid.n_hexes() > 0, "grid should have at least one hex");
        // All cells should have valid Fibonacci radii
        for &(cx, cy, r) in &grid.cells {
            assert!(FIB_RADII.contains(&r), "cell at ({:.1},{:.1}) has non-Fibonacci radius {}", cx, cy, r);
        }
    }
}
