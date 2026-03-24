//! Perimeter DCT-II encoder for ADN5 hexagonal codec.
//!
//! Samples the perimeter of each hexagon (~28 pixels), transforms it with
//! a DCT-II to get ~N/2 real coefficients, and classifies each vertex as
//! Segment / Arc / Rien using Weber-Fechner weighted gradients.

use std::f64::consts::PI;

use crate::hex::{HexShape, HexSubdiv};

// ---------------------------------------------------------------------------
// Radius-dependent encoding parameters (shared between encoder and decoder)
// ---------------------------------------------------------------------------

/// Max harmonics per cell, scaled by radius and quality.
/// Small hexes (R=3) need very few coefficients; large hexes (R=13) get more.
pub fn max_harmonics_for_radius(r: usize, quality: u8) -> usize {
    match r {
        3 => 3,
        5 => if quality >= 50 { 5 } else { 3 },
        8 => if quality >= 50 { 7 } else { 4 },
        13 => if quality >= 50 { 8 } else { 5 },
        _ => 5,
    }
}

/// Whether to encode/decode vertex codons for this radius.
/// R=3 hexes have perimeter ~= interior, harmonic reconstruction is already
/// excellent — codons add minimal value but cost 2-6 bytes each.
pub fn encode_codons_for_radius(r: usize) -> bool {
    r >= 5
}

/// Whether to encode/decode interior residuals for this radius.
/// ALL radii get residuals — even R=13 smooth areas need subtle gradient
/// corrections to avoid posterization (banding on sky/road).
pub fn encode_residual_for_radius(_r: usize) -> bool {
    true
}

/// Hex-step radius factor for AC quantization.
/// Larger hexes cover more area and benefit from coarser quantization.
pub fn hex_radius_factor(r: usize) -> f64 {
    match r {
        3 => 4.0,
        5 => 5.0,
        8 => 6.0,
        13 => 7.0,  // was 12.0 — less aggressive, preserve nuances on smooth areas
        _ => 6.0,
    }
}

/// Residual quantization step factor, scaled by radius.
pub fn resid_step_factor(r: usize) -> f64 {
    match r {
        3 => 0.5,
        5 => 0.8,
        8 => 1.2,
        13 => 0.6,  // fine residual for smooth areas — captures subtle gradients
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// Fibonacci lookup tables for amplitude/frequency snapping
// ---------------------------------------------------------------------------

/// Fibonacci frequency indices: [1, 2, 3, 5, 8]
const FIB_FREQS: [u8; 5] = [1, 2, 3, 5, 8];

/// Fibonacci amplitude levels: [0, 1, 2, 3, 5, 8, 13, 21]
const FIB_AMPS: [u8; 8] = [0, 1, 2, 3, 5, 8, 13, 21];

// ---------------------------------------------------------------------------
// VertexCodon
// ---------------------------------------------------------------------------

/// Classification of a hex vertex for ADN5 encoding.
///
/// - `Rien`    (0): flat region, no significant gradient
/// - `Segment` (1): dominant edge in one direction
/// - `Arc`     (2): curved edge / corner (gradients in both directions)
#[derive(Clone, Copy, Debug, Default)]
pub struct VertexCodon {
    /// 0 = Rien, 1 = Segment, 2 = Arc
    pub vtype: u8,
    /// Index into FIB_FREQS [1,2,3,5,8] (0..4)
    pub freq_idx: u8,
    /// Index into FIB_AMPS [0,1,2,3,5,8,13,21] (0..7)
    pub amp_idx: u8,
    /// Edge direction for Segment (radians), 0.0 for Rien/Arc
    pub angle: f64,
}

// ---------------------------------------------------------------------------
// Perimeter sampling
// ---------------------------------------------------------------------------

/// Sample the perimeter pixel values from a luminance plane at hex center
/// (cx, cy), using the precomputed shape offsets.
///
/// Coordinates that fall outside image bounds are clamped to the nearest edge.
/// Returns a Vec of length `shape.perimeter.len()`.
pub fn sample_perimeter(
    plane: &[f64],
    img_w: usize,
    img_h: usize,
    cx: f64,
    cy: f64,
    shape: &HexShape,
) -> Vec<f64> {
    let mut samples = Vec::with_capacity(shape.perimeter.len());
    for &(dx, dy) in &shape.perimeter {
        let px = (cx + dx as f64).round() as isize;
        let py = (cy + dy as f64).round() as isize;
        let px_c = px.clamp(0, (img_w as isize) - 1) as usize;
        let py_c = py.clamp(0, (img_h as isize) - 1) as usize;
        samples.push(plane[py_c * img_w + px_c]);
    }
    samples
}

// ---------------------------------------------------------------------------
// DCT-II forward and inverse
// ---------------------------------------------------------------------------

/// Forward DCT-II on a 1D signal of length N.
/// Returns N/2 real coefficients.
///
/// ```text
/// X[k] = sum_{n=0}^{N-1} x[n] * cos(pi/N * (n + 0.5) * k)
/// ```
///
/// for k = 0 .. N/2 - 1.
pub fn dct_ii_1d(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let k_max = n / 2;
    let mut coeffs = Vec::with_capacity(k_max);
    let pi_over_n = PI / n as f64;
    for k in 0..k_max {
        let mut sum = 0.0;
        for (i, &x) in input.iter().enumerate() {
            sum += x * (pi_over_n * (i as f64 + 0.5) * k as f64).cos();
        }
        coeffs.push(sum);
    }
    coeffs
}

/// Inverse DCT-II: reconstruct `n_output` samples from `coeffs.len()` coefficients.
///
/// ```text
/// x[n] = (1/N) * X[0] + (2/N) * sum_{k=1}^{K-1} X[k] * cos(pi/N * k * (n + 0.5))
/// ```
///
/// where N = `n_output` and K = `coeffs.len()`.
pub fn idct_ii_1d(coeffs: &[f64], n_output: usize) -> Vec<f64> {
    if coeffs.is_empty() || n_output == 0 {
        return vec![0.0; n_output];
    }
    let n = n_output as f64;
    let k_count = coeffs.len();
    let pi_over_n = PI / n;
    let inv_n = 1.0 / n;
    let two_inv_n = 2.0 * inv_n;

    let mut output = Vec::with_capacity(n_output);
    for i in 0..n_output {
        let n_plus_half = i as f64 + 0.5;
        let mut val = inv_n * coeffs[0]; // DC term (k=0)
        for k in 1..k_count {
            val += two_inv_n * coeffs[k] * (pi_over_n * k as f64 * n_plus_half).cos();
        }
        output.push(val);
    }
    output
}

// ---------------------------------------------------------------------------
// Vertex classification
// ---------------------------------------------------------------------------

/// Snap a positive magnitude to the nearest Fibonacci amplitude index (0..7).
fn snap_amp(mag: f64) -> u8 {
    let mut best_idx = 0u8;
    let mut best_dist = f64::MAX;
    for (i, &a) in FIB_AMPS.iter().enumerate() {
        let d = (mag - a as f64).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Snap a positive magnitude to the nearest Fibonacci frequency index (0..4).
fn snap_freq(mag: f64) -> u8 {
    let mut best_idx = 0u8;
    let mut best_dist = f64::MAX;
    for (i, &f) in FIB_FREQS.iter().enumerate() {
        let d = (mag - f as f64).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Classify a vertex position on the luminance plane.
///
/// Computes the gradient at (vx, vy) using central differences, applies
/// Weber-Fechner weighting (amplifies dark-area gradients), then classifies:
/// - `weber_mag < 2.0` -> Rien (flat)
/// - dominant direction > 2x minor -> Segment (with angle)
/// - else -> Arc (curved edge / corner)
///
/// Maps the magnitude to `freq_idx` and `amp_idx` using Fibonacci snap.
pub fn classify_vertex(
    plane: &[f64],
    img_w: usize,
    img_h: usize,
    vx: f64,
    vy: f64,
) -> VertexCodon {
    let ix = vx.round() as isize;
    let iy = vy.round() as isize;

    // Clamp to valid pixel coordinates, leaving room for central differences
    let x = ix.clamp(1, (img_w as isize) - 2) as usize;
    let y = iy.clamp(1, (img_h as isize) - 2) as usize;

    // Central differences for gradient
    let gx = plane[y * img_w + x + 1] - plane[y * img_w + x - 1];
    let gy = plane[(y + 1) * img_w + x] - plane[(y - 1) * img_w + x];

    // Weber-Fechner: amplify gradients in dark regions.
    // Perceptual sensitivity ~ 1 / (luminance + epsilon).
    let lum = plane[y * img_w + x];
    let weber = 1.0 / (lum.max(1.0) / 255.0 + 0.05);

    let weber_gx = gx.abs() * weber;
    let weber_gy = gy.abs() * weber;
    let weber_mag = (weber_gx * weber_gx + weber_gy * weber_gy).sqrt();

    if weber_mag < 2.0 {
        // Flat region
        return VertexCodon {
            vtype: 0,
            freq_idx: 0,
            amp_idx: 0,
            angle: 0.0,
        };
    }

    let (dominant, minor) = if weber_gx >= weber_gy {
        (weber_gx, weber_gy)
    } else {
        (weber_gy, weber_gx)
    };

    let vtype = if dominant > 2.0 * minor {
        1 // Segment: strongly directional edge
    } else {
        2 // Arc: gradients in both directions
    };

    let angle = if vtype == 1 {
        gy.atan2(gx)
    } else {
        0.0
    };

    let freq_idx = snap_freq(weber_mag.ln().max(0.0) + 1.0);
    let amp_idx = snap_amp(weber_mag.min(21.0));

    VertexCodon {
        vtype,
        freq_idx,
        amp_idx,
        angle,
    }
}

// ---------------------------------------------------------------------------
// Codon byte packing (vtype:2 + freq:3 + amp:3 = 8 bits)
// ---------------------------------------------------------------------------

/// Pack a VertexCodon into a single byte: vtype(2 bits) | freq_idx(3 bits) | amp_idx(3 bits).
pub fn pack_codon(codon: &VertexCodon) -> u8 {
    ((codon.vtype & 0x03) << 6) | ((codon.freq_idx & 0x07) << 3) | (codon.amp_idx & 0x07)
}

/// Unpack a single byte into a VertexCodon. Angle is always 0.0 (not stored).
pub fn unpack_codon(byte: u8) -> VertexCodon {
    VertexCodon {
        vtype: (byte >> 6) & 0x03,
        freq_idx: (byte >> 3) & 0x07,
        amp_idx: byte & 0x07,
        angle: 0.0,
    }
}

// ---------------------------------------------------------------------------
// Hexagonal subdivision determination
// ---------------------------------------------------------------------------

/// Determine the subdivision code for a hex based on its 6 vertex codons.
///
/// The number of active (non-Rien) vertices indicates how much structural
/// detail is present in the hex:
/// - 0 active:   Whole (smooth area, maximum compression)
/// - 1-2 active: Bisect (linear edge crosses the hex)
/// - 3-4 active: Trisect (moderate detail)
/// - 5-6 active: Full (complex texture, vertex charges dominate)
pub fn determine_subdivision(codons: &[VertexCodon; 6]) -> HexSubdiv {
    let n_active = codons.iter().filter(|c| c.vtype != 0).count();
    match n_active {
        0 => HexSubdiv::Whole,
        1..=2 => HexSubdiv::Bisect,
        3..=4 => HexSubdiv::Trisect,
        _ => HexSubdiv::Full,
    }
}

/// Pack subdivision codes (2 bits each) into a byte stream.
///
/// 4 codes per byte, MSB first: `[s0<<6 | s1<<4 | s2<<2 | s3]`.
/// The output length is `ceil(n_hexes / 4)`.
pub fn pack_subdivisions(subdivs: &[HexSubdiv]) -> Vec<u8> {
    let n_bytes = (subdivs.len() + 3) / 4;
    let mut bytes = vec![0u8; n_bytes];
    for (i, &s) in subdivs.iter().enumerate() {
        let byte_idx = i / 4;
        let shift = 6 - (i % 4) * 2;
        bytes[byte_idx] |= (s as u8) << shift;
    }
    bytes
}

/// Unpack subdivision codes from a byte stream (2 bits each, MSB first).
pub fn unpack_subdivisions(bytes: &[u8], n_hexes: usize) -> Vec<HexSubdiv> {
    let mut subdivs = Vec::with_capacity(n_hexes);
    for i in 0..n_hexes {
        let byte_idx = i / 4;
        let shift = 6 - (i % 4) * 2;
        if byte_idx < bytes.len() {
            let code = (bytes[byte_idx] >> shift) & 0x03;
            subdivs.push(HexSubdiv::from_u8(code));
        } else {
            subdivs.push(HexSubdiv::Whole);
        }
    }
    subdivs
}

// ---------------------------------------------------------------------------
// Hierarchical hex pyramid helpers
// ---------------------------------------------------------------------------

/// Fibonacci radii for pyramid levels, coarse to fine.
pub const PYRAMID_RADII: [usize; 4] = [13, 8, 5, 3];

/// Compute RMS energy of a region around (cx, cy) with radius r in the plane.
/// Used to determine whether a refinement hex is worth encoding.
pub fn sample_region_energy(
    plane: &[f64],
    w: usize,
    h: usize,
    cx: f64,
    cy: f64,
    r: usize,
) -> f64 {
    let mut sum2 = 0.0;
    let mut count = 0usize;
    let bound = r as i32;
    for dy in -bound..=bound {
        for dx in -bound..=bound {
            let px = (cx as i32 + dx).clamp(0, w as i32 - 1) as usize;
            let py = (cy as i32 + dy).clamp(0, h as i32 - 1) as usize;
            let v = plane[py * w + px];
            sum2 += v * v;
            count += 1;
        }
    }
    (sum2 / count.max(1) as f64).sqrt()
}

/// Build a regular hex grid for a given radius covering the entire image.
/// Returns Vec of (cx, cy) center coordinates, plus (cols, rows) dimensions.
pub fn build_regular_hex_grid(
    width: usize,
    height: usize,
    r: usize,
) -> (Vec<(f64, f64)>, usize, usize) {
    let rf = r as f64;
    let dx = crate::hex::hex_dx_r(rf);
    let dy = crate::hex::hex_dy_r(rf);
    let cols = (width as f64 / dx).ceil() as usize + 1;
    let rows = (height as f64 / dy).ceil() as usize + 1;

    let mut centers = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let cx = col as f64 * dx + dx / 2.0;
            let cy = row as f64 * dy + dy / 2.0
                + if col % 2 == 1 { dy / 2.0 } else { 0.0 };
            centers.push((cx, cy));
        }
    }
    (centers, cols, rows)
}

/// Scatter a single hex's reconstruction onto a pixel buffer.
/// Used during encoding to build the reconstruction for residual computation.
///
/// `first_write_wins`: if true, only writes to unwritten pixels (level 0 behavior).
/// `written`: tracks which pixels have been written (for first_write_wins mode).
/// When `first_write_wins` is false, values are added (refinement levels).
#[allow(dead_code)]
fn scatter_hex_to_buffer(
    buffer: &mut [f64],
    written: &mut [bool],
    width: usize,
    height: usize,
    cx: f64,
    cy: f64,
    shape: &HexShape,
    perimeter_values: &[f64],
    interior_vals: &[f64],
    mean_peri: f64,
    first_write_wins: bool,
) {
    let peri_lookup: std::collections::HashMap<(i32, i32), usize> = shape
        .perimeter
        .iter()
        .enumerate()
        .map(|(i, &pt)| (pt, i))
        .collect();
    let interior_lookup: std::collections::HashMap<(i32, i32), usize> = shape
        .interior
        .iter()
        .enumerate()
        .map(|(i, &pt)| (pt, i))
        .collect();

    for &(ddx, ddy) in &shape.voronoi {
        let px = (cx + ddx as f64).round() as isize;
        let py = (cy + ddy as f64).round() as isize;
        if px < 0 || px >= width as isize || py < 0 || py >= height as isize {
            continue;
        }
        let idx = py as usize * width + px as usize;

        if first_write_wins && written[idx] {
            continue;
        }

        let val = if let Some(&peri_idx) = peri_lookup.get(&(ddx, ddy)) {
            if peri_idx < perimeter_values.len() {
                perimeter_values[peri_idx]
            } else {
                mean_peri
            }
        } else if let Some(&int_idx) = interior_lookup.get(&(ddx, ddy)) {
            if int_idx < interior_vals.len() {
                interior_vals[int_idx]
            } else {
                mean_peri
            }
        } else {
            mean_peri
        };

        if first_write_wins {
            buffer[idx] = val;
            written[idx] = true;
        } else {
            buffer[idx] += val;
        }
    }
}

// ---------------------------------------------------------------------------
// Encode a single pyramid level
// ---------------------------------------------------------------------------

/// Encoded data for one pyramid level.
struct PyramidLevelData {
    /// Hex positions that were encoded (col, row indices into the regular grid)
    active_indices: Vec<u32>,
    /// DC residuals per channel [3][n_active]
    dc_residuals: Vec<Vec<i16>>,
    /// AC quantized per channel (flat, variable length per hex)
    ac_quantized: Vec<Vec<i16>>,
    /// Codon bytes (L channel only)
    codons: Vec<u8>,
    /// Subdivision codes per hex
    subdivisions: Vec<HexSubdiv>,
    /// Max harmonics per active hex (used for diagnostics)
    #[allow(dead_code)]
    max_harmonics: Vec<usize>,
    /// DC step value
    dc_step: f64,
    /// Grid dimensions
    grid_cols: usize,
    grid_rows: usize,
    /// Radius for this level
    radius: usize,
    /// Number of active hexes
    n_active: usize,
}

/// Encode one pyramid level from the given plane (original or residual).
/// Returns the level data and the reconstruction buffer for this level.
fn encode_pyramid_level(
    planes: [&[f64]; 3],
    width: usize,
    height: usize,
    r: usize,
    quality: u8,
    detail_step: f64,
    threshold: f64,
    is_level0: bool,
) -> (PyramidLevelData, Vec<Vec<f64>>) {
    use crate::golden::PHI;
    use crate::hex::compute_hex_shape_r;
    use crate::hex_decoder::{MultiScaleWeights, reconstruct_hex_interior_subdivided_r};
    let n = width * height;
    let shape = compute_hex_shape_r(r);
    let (all_centers, grid_cols, grid_rows) = build_regular_hex_grid(width, height, r);
    let _n_total = all_centers.len();

    // Determine which hexes to encode
    let mut active_indices: Vec<u32> = Vec::new();
    let mut active_centers: Vec<(f64, f64)> = Vec::new();

    for (idx, &(cx, cy)) in all_centers.iter().enumerate() {
        if is_level0 {
            // Level 0: encode ALL hexes
            active_indices.push(idx as u32);
            active_centers.push((cx, cy));
        } else {
            // Refinement levels: only encode where residual energy is significant
            // Check L channel energy (dominant)
            let energy = sample_region_energy(planes[0], width, height, cx, cy, r);
            if energy >= threshold {
                active_indices.push(idx as u32);
                active_centers.push((cx, cy));
            }
        }
    }

    let n_active = active_centers.len();
    let n_peri = shape.perimeter.len();
    let n_coeffs = n_peri / 2;
    let mh_base = max_harmonics_for_radius(r, quality).min(n_coeffs.saturating_sub(1));

    // Pyramid quantization: all levels use the same radius-based scaling.
    // The refinement levels naturally encode smaller residual values,
    // so the same quantization step gives effectively finer precision.
    let hex_step = detail_step * hex_radius_factor(r);

    // DC step
    let dc_step_val = (((detail_step * 0.3).max(0.3)) as f32) as f64;

    // Encode each channel
    let mut all_dc: Vec<Vec<f64>> = Vec::with_capacity(3);
    let mut all_ac: Vec<Vec<i16>> = Vec::with_capacity(3);
    let mut l_codons: Vec<u8> = Vec::new();
    let mut subdivisions: Vec<HexSubdiv> = Vec::with_capacity(n_active);
    let max_harmonics_vec: Vec<usize> = vec![mh_base; n_active];

    for (ch_idx, &channel) in planes.iter().enumerate() {
        let mut dc_values: Vec<f64> = Vec::with_capacity(n_active);
        let mut ac_quantized: Vec<i16> = Vec::new();

        for (_ai, &(cx, cy)) in active_centers.iter().enumerate() {
            let perimeter = sample_perimeter(channel, width, height, cx, cy, &shape);
            let coeffs = dct_ii_1d(&perimeter);

            dc_values.push(coeffs[0]);

            for k in 1..=mh_base {
                if k >= n_coeffs { break; }
                let golden_step = hex_step * (k as f64 * PHI).powf(0.55);
                let step = golden_step.max(0.1);
                let val = coeffs[k];
                let sign = if val >= 0.0 { 1.0 } else { -1.0 };
                let dead_zone = 0.22;
                let qv = (val.abs() / step + 0.5 - dead_zone).floor();
                let qi = if qv > 0.0 { (sign * qv) as i16 } else { 0i16 };
                ac_quantized.push(qi);
            }

            // Classify vertices (L channel only) — codons only for level 0
            if ch_idx == 0 {
                if is_level0 && encode_codons_for_radius(r) {
                    let rf = r as f64;
                    let mut all_codons_arr = [VertexCodon::default(); 6];
                    for vi in 0..6 {
                        let angle = std::f64::consts::PI / 3.0 * vi as f64;
                        let vx = cx + rf * angle.cos();
                        let vy = cy + rf * angle.sin();
                        all_codons_arr[vi] = classify_vertex(channel, width, height, vx, vy);
                    }
                    let subdiv = determine_subdivision(&all_codons_arr);
                    subdivisions.push(subdiv);
                    let n_codons = match subdiv {
                        HexSubdiv::Whole => 2,
                        _ => 6,
                    };
                    for vi in 0..n_codons {
                        l_codons.push(pack_codon(&all_codons_arr[vi]));
                    }
                } else {
                    // Refinement levels or R<5: no codons, always Whole
                    subdivisions.push(HexSubdiv::Whole);
                }
            }
        }

        all_dc.push(dc_values);
        all_ac.push(ac_quantized);
    }

    // DC DPCM
    let mut all_dc_residuals: Vec<Vec<i16>> = Vec::with_capacity(3);
    for dc_values in &all_dc {
        let mut residuals = Vec::with_capacity(dc_values.len());
        let mut prev_recon = 0.0f64;
        for &dc in dc_values.iter() {
            let n_peri_f = n_peri as f64;
            let dc_normalized = dc / n_peri_f;
            let delta = dc_normalized - prev_recon;
            let sign = if delta >= 0.0 { 1.0 } else { -1.0 };
            let qi = (delta.abs() / dc_step_val + 0.5).floor();
            let q = if qi > 0.0 { (sign * qi) as i16 } else { 0i16 };
            residuals.push(q);
            prev_recon += q as f64 * dc_step_val;
        }
        all_dc_residuals.push(residuals);
    }

    // Build reconstruction buffers for all 3 channels.
    // We need to reconstruct exactly as the decoder would to compute the residual
    // plane for the next level.
    let msw = MultiScaleWeights::precompute();
    let hw = msw.for_radius(r);
    let phi = PHI;
    let rf = r as f64;

    let mut recon_planes: Vec<Vec<f64>> = vec![vec![0.0f64; n]; 3];
    // Written buffer: tracks which pixels have been covered (for first-write-wins in level 0).
    // For refinement levels, this is not used (first_write_wins = false).
    let mut written_buf = vec![false; n];

    // Reconstruct DC values from DPCM residuals (exactly as decoder would)
    let mut dc_values_recon: Vec<Vec<f64>> = Vec::with_capacity(3);
    for ch_idx in 0..3 {
        let mut dc_ch: Vec<f64> = Vec::with_capacity(n_active);
        let mut prev_norm = 0.0f64;
        for &q in &all_dc_residuals[ch_idx] {
            let dc_norm = prev_norm + q as f64 * dc_step_val;
            dc_ch.push(dc_norm * n_peri as f64);
            prev_norm = dc_norm;
        }
        dc_values_recon.push(dc_ch);
    }

    // Build lookup maps for scatter
    let peri_lookup: std::collections::HashMap<(i32, i32), usize> = shape
        .perimeter
        .iter()
        .enumerate()
        .map(|(i, &pt)| (pt, i))
        .collect();
    let interior_lookup: std::collections::HashMap<(i32, i32), usize> = shape
        .interior
        .iter()
        .enumerate()
        .map(|(i, &pt)| (pt, i))
        .collect();

    let mut codon_cursor = 0usize;
    for (ai, &(cx, cy)) in active_centers.iter().enumerate() {
        let subdiv = subdivisions[ai];
        let n_codons_stored = if is_level0 && encode_codons_for_radius(r) {
            match subdiv {
                HexSubdiv::Whole => 2,
                _ => 6,
            }
        } else {
            0
        };

        // Build codons
        let codons: Vec<VertexCodon> = if is_level0 && encode_codons_for_radius(r) {
            (0..6)
                .map(|i| {
                    if i < n_codons_stored && (codon_cursor + i) < l_codons.len() {
                        unpack_codon(l_codons[codon_cursor + i])
                    } else {
                        VertexCodon::default()
                    }
                })
                .collect()
        } else {
            vec![VertexCodon::default(); 6]
        };

        let interior_positions: Vec<(f64, f64)> = shape
            .interior
            .iter()
            .map(|&(x, y)| (x as f64, y as f64))
            .collect();

        let vertex_positions: Vec<(f64, f64)> = (0..6)
            .map(|i| {
                let angle = std::f64::consts::PI / 3.0 * i as f64;
                (cx + rf * angle.cos(), cy + rf * angle.sin())
            })
            .collect();

        // Reconstruct all 3 channels for this hex
        let mut perimeter_values_ch: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut interior_vals_ch: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut mean_peri_ch = [0.0f64; 3];

        for ch_idx in 0..3 {
            let mut coeffs = vec![0.0f64; n_coeffs];
            coeffs[0] = dc_values_recon[ch_idx][ai];

            let ac_offset = ai * mh_base;
            for k in 1..=mh_base {
                if k >= n_coeffs { break; }
                let ac_idx = ac_offset + (k - 1);
                if ac_idx < all_ac[ch_idx].len() {
                    let golden_step = hex_step * (k as f64 * phi).powf(0.55);
                    let step = golden_step.max(0.1);
                    coeffs[k] = all_ac[ch_idx][ac_idx] as f64 * step;
                }
            }

            let perimeter_values = idct_ii_1d(&coeffs, n_peri);

            let interior_vals = reconstruct_hex_interior_subdivided_r(
                hw,
                &perimeter_values,
                &interior_positions,
                &vertex_positions,
                &codons,
                subdiv,
                r,
            );

            mean_peri_ch[ch_idx] = if perimeter_values.is_empty() {
                0.0
            } else {
                perimeter_values.iter().sum::<f64>() / perimeter_values.len() as f64
            };

            perimeter_values_ch[ch_idx] = perimeter_values;
            interior_vals_ch[ch_idx] = interior_vals;
        }

        // Scatter all channels for this hex (voronoi outer, channels inner)
        // Exactly matches decoder logic.
        for &(ddx, ddy) in &shape.voronoi {
            let px = (cx + ddx as f64).round() as isize;
            let py = (cy + ddy as f64).round() as isize;
            if px < 0 || px >= width as isize || py < 0 || py >= height as isize {
                continue;
            }
            let idx = py as usize * width + px as usize;

            if is_level0 && written_buf[idx] {
                continue; // first-write-wins for level 0
            }

            for ch_idx in 0..3 {
                let val = if let Some(&peri_idx) = peri_lookup.get(&(ddx, ddy)) {
                    if peri_idx < perimeter_values_ch[ch_idx].len() {
                        perimeter_values_ch[ch_idx][peri_idx]
                    } else { mean_peri_ch[ch_idx] }
                } else if let Some(&int_idx) = interior_lookup.get(&(ddx, ddy)) {
                    if int_idx < interior_vals_ch[ch_idx].len() {
                        interior_vals_ch[ch_idx][int_idx]
                    } else { mean_peri_ch[ch_idx] }
                } else {
                    mean_peri_ch[ch_idx]
                };

                if is_level0 {
                    recon_planes[ch_idx][idx] = val;
                } else {
                    recon_planes[ch_idx][idx] += val;
                }
            }

            if is_level0 {
                written_buf[idx] = true;
            }
        }

        codon_cursor += n_codons_stored;
    }

    let level_data = PyramidLevelData {
        active_indices,
        dc_residuals: all_dc_residuals,
        ac_quantized: all_ac,
        codons: l_codons,
        subdivisions,
        max_harmonics: max_harmonics_vec,
        dc_step: dc_step_val,
        grid_cols,
        grid_rows,
        radius: r,
        n_active,
    };

    (level_data, recon_planes)
}

// ---------------------------------------------------------------------------
// Top-level hex encoder — hierarchical hex pyramid
// ---------------------------------------------------------------------------

/// Encoded data for one pixel-residual refinement level.
struct PixelResidualLevelData {
    /// Quantization step for this level
    resid_step: f64,
    /// rANS-encoded streams per channel [3]
    encoded_streams: Vec<Vec<u8>>,
    /// Number of pixels per channel (diagnostic)
    #[allow(dead_code)]
    n_pixels: usize,
}

/// Encode an RGB image into AUR2 v7 hierarchical hex pyramid format.
///
/// Pipeline: GCT -> PTF -> pyramid where level 0 (R=13) uses hex perimeter +
/// harmonic reconstruction, and levels 1-3 encode per-pixel residuals directly
/// (quantized + rANS). This avoids the Laplace interpolation problem on
/// oscillating residual patterns.
///
/// Level 0 (R=13): hex perimeter + DCT-II + Laplace interior (coarse structure).
/// Levels 1-3: direct per-pixel residual (quantized + rANS), progressively finer.
///
/// Decoding: level 0 reconstruction + sum of per-pixel residuals.
pub fn encode_aur2_hex(
    rgb: &[u8],
    width: usize,
    height: usize,
    quality: u8,
) -> Result<crate::aurea_encoder::AureaEncoderResult, Box<dyn std::error::Error>> {
    use crate::aurea_encoder;
    use crate::bitstream::{self, Aur2Header, HexHeader};
    use crate::color;
    use crate::golden::PTF_GAMMA;
    use crate::hex::HEX_R;
    use crate::rans;

    let n = width * height;

    // 1. GCT: RGB -> (L, C1, C2)
    let (mut l_ch, c1_ch, c2_ch) = color::golden_rotate_forward(rgb, n);

    // 2. PTF on L channel: v = 255 * (v/255)^gamma
    {
        let inv255 = 1.0 / 255.0;
        for v in l_ch.iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA);
        }
    }

    // 3. Auto detail step. Roundtrip through f32 so encoder and decoder agree.
    let detail_step = (aurea_encoder::auto_detail_step(&l_ch, height, width, quality) as f32) as f64;

    // 4. Build the hierarchical pyramid
    let n_levels = PYRAMID_RADII.len();

    // Current planes: start as original channels, become residuals after each level
    let mut current_l = l_ch.clone();
    let mut current_c1 = c1_ch.clone();
    let mut current_c2 = c2_ch.clone();

    // Level 0: hex perimeter encoding (coarse structure)
    let level0_r = PYRAMID_RADII[0];
    eprintln!("  Pyramid level 0 (R={}): hex perimeter, threshold=0.00", level0_r);

    let (level0_data, recon_planes) = encode_pyramid_level(
        [&current_l, &current_c1, &current_c2],
        width,
        height,
        level0_r,
        quality,
        detail_step,
        0.0,  // threshold = 0 for level 0 (encode all)
        true, // is_level0
    );

    eprintln!("    {} active hexes / {} grid ({}x{})",
        level0_data.n_active,
        level0_data.grid_cols * level0_data.grid_rows,
        level0_data.grid_cols, level0_data.grid_rows);

    // Compute residual after level 0
    for i in 0..n {
        current_l[i] -= recon_planes[0][i];
        current_c1[i] -= recon_planes[1][i];
        current_c2[i] -= recon_planes[2][i];
    }

    {
        let rms_l: f64 = (current_l.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        let rms_c1: f64 = (current_c1.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        let rms_c2: f64 = (current_c2.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        eprintln!("    Residual RMS: L={:.2}, C1={:.2}, C2={:.2}", rms_l, rms_c1, rms_c2);
    }

    // Levels 1-3: per-pixel residual encoding
    let mut pixel_level_data_vec: Vec<PixelResidualLevelData> = Vec::with_capacity(n_levels - 1);

    for level in 1..n_levels {
        // Coarser quantization at deeper levels (diminishing returns)
        // Residual step: hex R=13 already captured the coarse signal, so the
        // residual is smaller → we can use a coarser step. Level 1 = 2.5×, 2 = 3.5×, 3 = 4.5×.
        let resid_step = detail_step * (1.2 + 0.8 * level as f64);
        // Roundtrip through f32 so encoder and decoder agree exactly
        let resid_step = (resid_step as f32) as f64;

        eprintln!("  Pyramid level {} (pixel residual): step={:.3}", level, resid_step);

        let mut encoded_streams: Vec<Vec<u8>> = Vec::with_capacity(3);

        // Helper closure to quantize+subtract one channel plane in-place
        fn quantize_plane(plane: &mut [f64], resid_step: f64) -> Vec<i16> {
            let n = plane.len();
            let mut quantized: Vec<i16> = Vec::with_capacity(n);
            for i in 0..n {
                let val = plane[i];
                let sign = if val >= 0.0 { 1.0 } else { -1.0 };
                let dead_zone = 0.22;
                let q = (val.abs() / resid_step + 0.5 - dead_zone).floor();
                let qi = if q > 0.0 {
                    (sign * q).clamp(-127.0, 127.0) as i16
                } else {
                    0
                };
                quantized.push(qi);
            }
            // Subtract reconstruction so next level sees the remaining residual
            for i in 0..n {
                plane[i] -= quantized[i] as f64 * resid_step;
            }
            quantized
        }

        // Process each channel: quantize, subtract, encode
        let q_l = quantize_plane(&mut current_l, resid_step);
        encoded_streams.push(rans::rans_encode_band(&q_l));
        let q_c1 = quantize_plane(&mut current_c1, resid_step);
        encoded_streams.push(rans::rans_encode_band(&q_c1));
        let q_c2 = quantize_plane(&mut current_c2, resid_step);
        encoded_streams.push(rans::rans_encode_band(&q_c2));

        {
            let rms_l: f64 = (current_l.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
            let rms_c1: f64 = (current_c1.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
            let rms_c2: f64 = (current_c2.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
            eprintln!("    Residual RMS: L={:.2}, C1={:.2}, C2={:.2}", rms_l, rms_c1, rms_c2);
        }

        pixel_level_data_vec.push(PixelResidualLevelData {
            resid_step,
            encoded_streams,
            n_pixels: n,
        });
    }

    // 5. Serialize into the body
    let mut body: Vec<u8> = Vec::new();

    // Hex header (backward compat structure, reserved=3 signals pyramid mode)
    let hex_header = HexHeader {
        hex_radius: HEX_R as u8,
        hex_cols: 0,
        hex_rows: 0,
        reserved: 3, // reserved=3 signals hierarchical pyramid
    };
    body.extend_from_slice(&bitstream::write_hex_header(&hex_header));

    // Pyramid header
    body.push(n_levels as u8);     // number of levels
    body.push(1u8);                 // mode: 1 = hex level 0 + pixel residual levels 1+
    body.extend_from_slice(&(detail_step as f32).to_le_bytes()); // detail_step (f32)

    // Level 0: hex perimeter data (same format as before)
    {
        let r = level0_data.radius;
        let n_active = level0_data.n_active;
        let dc_step_val = level0_data.dc_step;

        // Level header
        body.push(r as u8);
        body.extend_from_slice(&(n_active as u32).to_le_bytes());
        body.extend_from_slice(&(level0_data.grid_cols as u16).to_le_bytes());
        body.extend_from_slice(&(level0_data.grid_rows as u16).to_le_bytes());
        body.extend_from_slice(&(dc_step_val as f32).to_le_bytes());

        // Active hex indices
        {
            let mut idx_bytes: Vec<u8> = Vec::with_capacity(n_active * 4);
            for &idx in &level0_data.active_indices {
                idx_bytes.extend_from_slice(&idx.to_le_bytes());
            }
            let idx_compressed = rans::rans_compress_bytes(&idx_bytes);
            body.extend_from_slice(&(idx_compressed.len() as u32).to_le_bytes());
            body.extend_from_slice(&idx_compressed);
        }

        // DC streams (3 channels)
        for ch_idx in 0..3 {
            let dc_encoded = rans::rans_encode_band(&level0_data.dc_residuals[ch_idx]);
            body.extend_from_slice(&(dc_encoded.len() as u32).to_le_bytes());
            body.extend_from_slice(&dc_encoded);
        }

        // AC streams (3 channels)
        for ch_idx in 0..3 {
            let ac_encoded = rans::rans_encode_band(&level0_data.ac_quantized[ch_idx]);
            body.extend_from_slice(&(ac_encoded.len() as u32).to_le_bytes());
            body.extend_from_slice(&ac_encoded);
        }

        // Codon stream
        {
            let codon_compressed = rans::rans_compress_bytes(&level0_data.codons);
            body.extend_from_slice(&(codon_compressed.len() as u32).to_le_bytes());
            body.extend_from_slice(&codon_compressed);
        }

        // Subdivision stream
        {
            let subdiv_packed = pack_subdivisions(&level0_data.subdivisions);
            let subdiv_compressed = rans::rans_compress_bytes(&subdiv_packed);
            body.extend_from_slice(&(subdiv_compressed.len() as u32).to_le_bytes());
            body.extend_from_slice(&subdiv_compressed);
        }

        eprintln!("    Level 0 R={}: {} active, body so far {} bytes", r, n_active, body.len());
    }

    // Levels 1-3: per-pixel residual data
    for (li, pld) in pixel_level_data_vec.iter().enumerate() {
        let level = li + 1;
        // Level header: resid_step (f32)
        body.extend_from_slice(&(pld.resid_step as f32).to_le_bytes());

        // 3 channels: stream_size (u32) + rANS data
        for ch in 0..3 {
            body.extend_from_slice(&(pld.encoded_streams[ch].len() as u32).to_le_bytes());
            body.extend_from_slice(&pld.encoded_streams[ch]);
        }

        eprintln!("    Level {} (pixel): body so far {} bytes", level, body.len());
    }

    // 6. AUR2 header with version=7 (hierarchical hex pyramid)
    let header = Aur2Header {
        version: 7,
        quality,
        width,
        height,
        wv_levels: 0,
        detail_step,
        ll_ranges: [(0.0, 0.0); 3],
    };

    let header_bytes = bitstream::write_aur2_header(&header);
    let total_size = header_bytes.len() + body.len();
    let mut aurea_data = Vec::with_capacity(total_size);
    aurea_data.extend_from_slice(&header_bytes);
    aurea_data.extend_from_slice(&body);

    eprintln!("  Pyramid total: {} bytes ({:.2} bpp)",
        total_size, total_size as f64 * 8.0 / (width * height) as f64);

    Ok(aurea_encoder::AureaEncoderResult {
        aurea_data,
        compressed_size: total_size,
    })
}

// ---------------------------------------------------------------------------
// Edge-energy encoder (AUR2 version 8)
// ---------------------------------------------------------------------------

/// Encode using edge-energy model (AUR2 version 8).
///
/// Pipeline: GCT -> PTF on L -> compute edge energies for each channel
/// (3 directions x 3 channels = 9 planes) -> store reference pixel values ->
/// quantize edges with golden step + dead zone -> rANS encode each plane ->
/// assemble bitstream.
pub fn encode_edge_energy(
    rgb: &[u8],
    width: usize,
    height: usize,
    quality: u8,
) -> Result<crate::aurea_encoder::AureaEncoderResult, Box<dyn std::error::Error>> {
    use crate::aurea_encoder;
    use crate::bitstream::{self, Aur2Header};
    use crate::color;
    use crate::golden::PTF_GAMMA;
    use crate::hex_edge;
    use crate::rans;

    let n = width * height;

    // 1. GCT: RGB -> (L, C1, C2)
    let (mut l_ch, c1_ch, c2_ch) = color::golden_rotate_forward(rgb, n);

    // 2. PTF on L channel: v = 255 * (v/255)^gamma
    {
        let inv255 = 1.0 / 255.0;
        for v in l_ch.iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA);
        }
    }

    // 3. Auto detail step (roundtrip through f32 for encoder/decoder agreement)
    let detail_step = (aurea_encoder::auto_detail_step(&l_ch, height, width, quality) as f32) as f64;

    // 4. Store reference pixel values + anchor grids for drift prevention
    let ref_l = l_ch[0] as f32;
    let ref_c1 = c1_ch[0] as f32;
    let ref_c2 = c2_ch[0] as f32;

    // Path-DPCM mode: no anchors or borders needed.

    // 5. Hamiltonian path: each photon visits every pixel once.
    // Boustrophedon (snake scan) for optimal spatial locality.
    let path = hex_edge::hilbert_path(width, height);
    eprintln!("  Path: {} pixels ({}×{})", path.len(), width, height);

    // 6. DPCM along the path: ONE delta per pixel (not 3 edges).
    // The path ensures each pixel is adjacent to the previous → small deltas.
    let q_frac = (quality as f64 / 100.0).clamp(0.01, 1.0);
    // Aggressive DPCM step: target ~1 bpp at q=75.
    // Path DPCM deltas are pixel-to-pixel differences along snake scan.
    // For natural images, deltas follow a Laplacian distribution peaked at 0.
    // Coarser step → more zeros → better rANS compression → lower bpp.
    // Luma step: fine enough to preserve building textures (gradients 3-5).
    // q=75 → 5.75, q=50 → 10.0, q=95 → 3.25
    // Sweet spot: q=75 → 7.75 (between 5.75/too fine and 10.0/too coarse)
    // q=75 → 8.5: preserves building texture while keeping bpp reasonable
    let edge_step_l = ((4.0 + (1.0 - q_frac) * 18.0) as f32) as f64;
    // Chroma step: FINER than luma. GCT chroma (C1=B-L, C2=R-L) has small
    // values (±20 range). Same step as luma (10.0) zeros most chroma deltas.
    // 0.3× gives step≈3.0 → preserves chroma differences of 2+ (visible colors).
    let edge_step_c = ((edge_step_l * 0.3) as f32) as f64;
    let dead_zone = 0.22; // standard dead zone

    // Roundtrip through f32
    let edge_step_l = (edge_step_l as f32) as f64;
    let edge_step_c = (edge_step_c as f32) as f64;

    // 6b. Hex supercordes oracle: structural map from L channel.
    // Cost: ~64K bits for HD image (~0.012 bpp). Guides DPCM step.
    let (solid_packed, solid_cols, solid_rows) = hex_edge::pack_solid_map(&l_ch, width, height);
    let solid_map = hex_edge::unpack_solid_map(&solid_packed, solid_cols, solid_rows, width, height);
    let n_solid = solid_map.iter().filter(|&&s| s).count();
    eprintln!("  Hex oracle: {}x{} grid, {:.1}% solid",
        solid_cols, solid_rows, n_solid as f64 / solid_map.len() as f64 * 100.0);

    // L channel: oracle-guided adaptive DPCM (no chroma modulation)
    let deltas_l = hex_edge::encode_path_dpcm(&l_ch, &path, edge_step_l, dead_zone, &solid_map, None);

    // C1, C2: finer step (0.3× luma), NO relativistic modulation for now.
    // The GCT chroma has small values — a fine step preserves visible color.
    // Relativistic E=CL² will be reintroduced as dead-zone modulation, not step modulation.
    let deltas_c1 = hex_edge::encode_path_dpcm(&c1_ch, &path, edge_step_c, dead_zone, &solid_map, None);
    let deltas_c2 = hex_edge::encode_path_dpcm(&c2_ch, &path, edge_step_c, dead_zone, &solid_map, None);

    // Stats
    let nz_l = deltas_l.iter().filter(|&&d| d != 0).count();
    let nz_c1 = deltas_c1.iter().filter(|&&d| d != 0).count();
    let nz_c2 = deltas_c2.iter().filter(|&&d| d != 0).count();
    eprintln!("  Path DPCM: L={:.1}% nonzero, C1={:.1}%, C2={:.1}%",
        nz_l as f64 / deltas_l.len() as f64 * 100.0,
        nz_c1 as f64 / deltas_c1.len() as f64 * 100.0,
        nz_c2 as f64 / deltas_c2.len() as f64 * 100.0);

    // 7. rANS encode: 3 streams (one per channel), not 9.
    let encoded_streams: Vec<Vec<u8>> = vec![
        rans::rans_encode_band(&deltas_l),
        rans::rans_encode_band(&deltas_c1),
        rans::rans_encode_band(&deltas_c2),
    ];

    // 8. Assemble the body
    let mut body: Vec<u8> = Vec::new();

    // Edge-energy header: reference values + step sizes
    body.extend_from_slice(&ref_l.to_le_bytes());      // 4 bytes
    body.extend_from_slice(&ref_c1.to_le_bytes());     // 4 bytes
    body.extend_from_slice(&ref_c2.to_le_bytes());     // 4 bytes
    body.extend_from_slice(&(edge_step_l as f32).to_le_bytes());  // 4 bytes
    body.extend_from_slice(&(edge_step_c as f32).to_le_bytes());  // 4 bytes

    // Mode byte = 1 signals unified path-DPCM with hex oracle.
    body.push(1u8);

    // Hex supercordes solid map (packed bits, rANS compressed)
    let solid_compressed = rans::rans_compress_bytes(&solid_packed);
    body.extend_from_slice(&(solid_cols as u16).to_le_bytes());
    body.extend_from_slice(&(solid_rows as u16).to_le_bytes());
    body.extend_from_slice(&(solid_compressed.len() as u32).to_le_bytes());
    body.extend_from_slice(&solid_compressed);
    eprintln!("  Solid map: {} bytes ({:.3} bpp)",
        solid_compressed.len(), solid_compressed.len() as f64 * 8.0 / (width * height) as f64);

    // 3 rANS streams: one per channel (L, C1, C2)
    for stream in &encoded_streams {
        body.extend_from_slice(&(stream.len() as u32).to_le_bytes());
        body.extend_from_slice(stream);
    }

    // 9. AUR2 header with version=8 (edge-energy)
    let header = Aur2Header {
        version: 8,
        quality,
        width,
        height,
        wv_levels: 0,
        detail_step,
        ll_ranges: [(0.0, 0.0); 3],
    };

    let header_bytes = bitstream::write_aur2_header(&header);
    let total_size = header_bytes.len() + body.len();
    let mut aurea_data = Vec::with_capacity(total_size);
    aurea_data.extend_from_slice(&header_bytes);
    aurea_data.extend_from_slice(&body);

    let bpp = total_size as f64 * 8.0 / (width * height) as f64;
    eprintln!("  Edge-energy total: {} bytes ({:.2} bpp)", total_size, bpp);

    Ok(aurea_encoder::AureaEncoderResult {
        aurea_data,
        compressed_size: total_size,
    })
}

// ---------------------------------------------------------------------------
// DNA-guided edge-energy encoder (AUR2 version 8, mode=2)
// ---------------------------------------------------------------------------

/// Encode using DNA-guided path DPCM (AUR2 version 8, mode=2).
///
/// Pipeline: GCT -> PTF on L -> snake-scan Hamiltonian path -> DNA-guided
/// adaptive DPCM (codons from reconstructed gradient field) -> rANS encode
/// delta streams + codon streams -> assemble bitstream.
///
/// Body format (mode=2):
/// ```text
/// ref_l(f32) + ref_c1(f32) + ref_c2(f32) + step_l(f32) + step_c(f32)
/// mode: u8 = 2 (DNA-guided path DPCM)
/// Codon streams: 3 x (size(u32) + rANS-compressed amino acids)
/// DPCM streams:  3 x (size(u32) + rANS data)
/// ```
pub fn encode_edge_energy_dna(
    rgb: &[u8],
    width: usize,
    height: usize,
    quality: u8,
) -> Result<crate::aurea_encoder::AureaEncoderResult, Box<dyn std::error::Error>> {
    use crate::aurea_encoder;
    use crate::bitstream::{self, Aur2Header};
    use crate::color;
    use crate::golden::PTF_GAMMA;
    use crate::hex_edge;
    use crate::rans;

    let n = width * height;

    // 1. GCT: RGB -> (L, C1, C2)
    let (mut l_ch, c1_ch, c2_ch) = color::golden_rotate_forward(rgb, n);

    // 2. PTF on L channel: v = 255 * (v/255)^gamma
    {
        let inv255 = 1.0 / 255.0;
        for v in l_ch.iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA);
        }
    }

    // 3. Auto detail step (roundtrip through f32 for encoder/decoder agreement)
    let detail_step = (aurea_encoder::auto_detail_step(&l_ch, height, width, quality) as f32) as f64;

    // 4. Store reference pixel values
    let ref_l = l_ch[0] as f32;
    let ref_c1 = c1_ch[0] as f32;
    let ref_c2 = c2_ch[0] as f32;

    // 5. Hamiltonian path: snake scan
    let path = hex_edge::hilbert_path(width, height);
    eprintln!("  DNA path: {} pixels ({}x{})", path.len(), width, height);

    // 6. Compute DPCM step sizes (same formula as mode=1)
    let q_frac = (quality as f64 / 100.0).clamp(0.01, 1.0);
    let edge_step_l = ((5.0 + (1.0 - q_frac) * 20.0) as f32) as f64;
    let edge_step_c = ((edge_step_l * 1.0) as f32) as f64;
    let dead_zone = 0.22;

    // Roundtrip through f32
    let edge_step_l = (edge_step_l as f32) as f64;
    let edge_step_c = (edge_step_c as f32) as f64;

    // 7. DNA-guided DPCM: encode each channel, collect codons
    let (deltas_l, codons_l) = hex_edge::encode_path_dpcm_dna(&l_ch, &path, edge_step_l, dead_zone);
    let (deltas_c1, codons_c1) = hex_edge::encode_path_dpcm_dna(&c1_ch, &path, edge_step_c, dead_zone);
    let (deltas_c2, codons_c2) = hex_edge::encode_path_dpcm_dna(&c2_ch, &path, edge_step_c, dead_zone);

    // Stats
    let nz_l = deltas_l.iter().filter(|&&d| d != 0).count();
    let nz_c1 = deltas_c1.iter().filter(|&&d| d != 0).count();
    let nz_c2 = deltas_c2.iter().filter(|&&d| d != 0).count();
    let intron_l = codons_l.iter().filter(|&&aa| {
        crate::polymerase::Codon::from_amino_acid(aa).is_intron()
    }).count();
    eprintln!("  DNA DPCM: L={:.1}% nz, C1={:.1}%, C2={:.1}%",
        nz_l as f64 / deltas_l.len().max(1) as f64 * 100.0,
        nz_c1 as f64 / deltas_c1.len().max(1) as f64 * 100.0,
        nz_c2 as f64 / deltas_c2.len().max(1) as f64 * 100.0);
    eprintln!("  DNA codons: L={:.1}% intron (gas)",
        intron_l as f64 / codons_l.len().max(1) as f64 * 100.0);

    // 8. rANS encode: 3 delta streams + 3 codon streams
    let encoded_deltas: Vec<Vec<u8>> = vec![
        rans::rans_encode_band(&deltas_l),
        rans::rans_encode_band(&deltas_c1),
        rans::rans_encode_band(&deltas_c2),
    ];
    let encoded_codons: Vec<Vec<u8>> = vec![
        rans::rans_compress_bytes(&codons_l),
        rans::rans_compress_bytes(&codons_c1),
        rans::rans_compress_bytes(&codons_c2),
    ];

    // 9. Assemble the body
    let mut body: Vec<u8> = Vec::new();

    // Edge-energy header: reference values + step sizes (20 bytes)
    body.extend_from_slice(&ref_l.to_le_bytes());
    body.extend_from_slice(&ref_c1.to_le_bytes());
    body.extend_from_slice(&ref_c2.to_le_bytes());
    body.extend_from_slice(&(edge_step_l as f32).to_le_bytes());
    body.extend_from_slice(&(edge_step_c as f32).to_le_bytes());

    // Mode byte = 2 signals DNA-guided path DPCM
    body.push(2u8);

    // Codon streams first (decoder needs them before DPCM reconstruction)
    for stream in &encoded_codons {
        body.extend_from_slice(&(stream.len() as u32).to_le_bytes());
        body.extend_from_slice(stream);
    }

    // DPCM delta streams
    for stream in &encoded_deltas {
        body.extend_from_slice(&(stream.len() as u32).to_le_bytes());
        body.extend_from_slice(stream);
    }

    // 10. AUR2 header with version=8 (edge-energy, mode=2 DNA-guided)
    let header = Aur2Header {
        version: 8,
        quality,
        width,
        height,
        wv_levels: 0,
        detail_step,
        ll_ranges: [(0.0, 0.0); 3],
    };

    let header_bytes = bitstream::write_aur2_header(&header);
    let total_size = header_bytes.len() + body.len();
    let mut aurea_data = Vec::with_capacity(total_size);
    aurea_data.extend_from_slice(&header_bytes);
    aurea_data.extend_from_slice(&body);

    let bpp = total_size as f64 * 8.0 / (width * height) as f64;
    let codon_bpp = encoded_codons.iter().map(|s| s.len()).sum::<usize>() as f64 * 8.0 / n as f64;
    let delta_bpp = encoded_deltas.iter().map(|s| s.len()).sum::<usize>() as f64 * 8.0 / n as f64;
    eprintln!("  DNA edge-energy: {} bytes ({:.2} bpp = {:.2} codons + {:.2} deltas)",
        total_size, bpp, codon_bpp, delta_bpp);

    Ok(aurea_encoder::AureaEncoderResult {
        aurea_data,
        compressed_size: total_size,
    })
}

// ---------------------------------------------------------------------------
// Hex-path polymerase encoder (AUR2 version 8, mode=3)
// ---------------------------------------------------------------------------

/// Encode using hex-path polymerase DPCM (AUR2 version 8, mode=3).
///
/// Pipeline: GCT -> PTF on L -> hex grid R=5 -> golden spiral visit order ->
/// polymerase classification per hex (codon -> gas/solid) ->
/// DC DPCM along spiral -> interior DPCM for solid hexes ->
/// rANS encode all streams -> assemble bitstream.
///
/// Body format (mode=3):
/// ```text
/// ref_l(f32) + ref_c1(f32) + ref_c2(f32) + step_l(f32) + step_c(f32)
/// mode: u8 = 3 (hex-path polymerase)
/// n_hexes: u32
/// gas_map: (size(u32) + rANS-compressed packed bits)
/// codon_streams: 3 x (size(u32) + rANS-compressed amino acids)
/// dc_streams: 3 x (size(u32) + rANS-encoded i16 deltas)
/// interior_streams: 3 x (size(u32) + rANS-encoded i16 deltas)
/// ```
pub fn encode_edge_energy_hex(
    rgb: &[u8],
    width: usize,
    height: usize,
    quality: u8,
) -> Result<crate::aurea_encoder::AureaEncoderResult, Box<dyn std::error::Error>> {
    use crate::aurea_encoder;
    use crate::bitstream::{self, Aur2Header};
    use crate::color;
    use crate::golden::PTF_GAMMA;
    use crate::hex;
    use crate::hex_edge;
    use crate::rans;

    let n = width * height;

    // 1. GCT: RGB -> (L, C1, C2)
    let (mut l_ch, c1_ch, c2_ch) = color::golden_rotate_forward(rgb, n);

    // 2. PTF on L channel: v = 255 * (v/255)^gamma
    {
        let inv255 = 1.0 / 255.0;
        for v in l_ch.iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA);
        }
    }

    // 3. Auto detail step (roundtrip through f32 for encoder/decoder agreement)
    let detail_step = (aurea_encoder::auto_detail_step(&l_ch, height, width, quality) as f32) as f64;

    // 4. Reference pixel values
    let ref_l = l_ch[0] as f32;
    let ref_c1 = c1_ch[0] as f32;
    let ref_c2 = c2_ch[0] as f32;

    // 5. Build hex grid and spiral path
    let shape = hex::compute_hex_shape();
    let hex_path = hex_edge::hex_spiral_path(width, height);
    let n_hexes = hex_path.len();
    eprintln!("  Hex path: {} hexes ({}x{})", n_hexes, width, height);

    // 6. Compute DPCM steps
    let q_frac = (quality as f64 / 100.0).clamp(0.01, 1.0);
    // DC step: coarser for smooth regions (hex DCs vary slowly along spiral)
    let dc_step_l = ((1.0 + (1.0 - q_frac) * 4.0) as f32) as f64;
    let dc_step_c = ((dc_step_l * 1.5) as f32) as f64;
    // Interior step: finer to preserve pixel detail within solid hexes
    let interior_step_l = ((3.0 + (1.0 - q_frac) * 12.0) as f32) as f64;
    let interior_step_c = ((interior_step_l * 1.2) as f32) as f64;
    let dead_zone = 0.22;
    // Classification step: use the L interior step as reference for gradient thresholding
    let classify_step = interior_step_l;

    // Roundtrip through f32
    let dc_step_l = (dc_step_l as f32) as f64;
    let dc_step_c = (dc_step_c as f32) as f64;
    let interior_step_l = (interior_step_l as f32) as f64;
    let interior_step_c = (interior_step_c as f32) as f64;

    // 7. Encode each channel
    let channels: [&[f64]; 3] = [&l_ch, &c1_ch, &c2_ch];
    let dc_steps = [dc_step_l, dc_step_c, dc_step_c];
    let int_steps = [interior_step_l, interior_step_c, interior_step_c];

    let mut all_dc_deltas: Vec<Vec<i16>> = Vec::with_capacity(3);
    let mut all_interior_deltas: Vec<Vec<i16>> = Vec::with_capacity(3);
    let mut all_amino_acids: Vec<Vec<u8>> = Vec::with_capacity(3);
    let mut all_gas_flags: Vec<Vec<bool>> = Vec::with_capacity(3);

    for ch in 0..3 {
        let (dc_deltas, interior_deltas, amino_acids, gas_flags) =
            hex_edge::encode_hex_path_channel(
                channels[ch],
                &hex_path,
                &shape,
                width, height,
                dc_steps[ch],
                int_steps[ch],
                dead_zone,
                classify_step,
            );

        if ch == 0 {
            let n_gas = gas_flags.iter().filter(|&&g| g).count();
            let n_solid = gas_flags.iter().filter(|&&g| !g).count();
            eprintln!("  Polymerase: {:.1}% gas ({} hexes), {:.1}% solid ({} hexes)",
                n_gas as f64 / n_hexes as f64 * 100.0, n_gas,
                n_solid as f64 / n_hexes as f64 * 100.0, n_solid);
            eprintln!("  DC deltas: {}, interior deltas: {}",
                dc_deltas.len(), interior_deltas.len());
        }

        all_dc_deltas.push(dc_deltas);
        all_interior_deltas.push(interior_deltas);
        all_amino_acids.push(amino_acids);
        all_gas_flags.push(gas_flags);
    }

    // 8. Pack gas map (from L channel classification, shared across channels)
    //    1 bit per hex: 0 = solid, 1 = gas
    let gas_flags = &all_gas_flags[0]; // L channel drives classification
    let n_gas_bytes = (n_hexes + 7) / 8;
    let mut gas_packed = vec![0u8; n_gas_bytes];
    for (i, &is_gas) in gas_flags.iter().enumerate() {
        if is_gas {
            gas_packed[i / 8] |= 1 << (7 - i % 8);
        }
    }

    // 9. rANS encode all streams
    let gas_compressed = rans::rans_compress_bytes(&gas_packed);

    let mut encoded_codons: Vec<Vec<u8>> = Vec::with_capacity(3);
    let mut encoded_dcs: Vec<Vec<u8>> = Vec::with_capacity(3);
    let mut encoded_interiors: Vec<Vec<u8>> = Vec::with_capacity(3);

    for ch in 0..3 {
        encoded_codons.push(rans::rans_compress_bytes(&all_amino_acids[ch]));
        encoded_dcs.push(rans::rans_encode_band(&all_dc_deltas[ch]));
        encoded_interiors.push(rans::rans_encode_band(&all_interior_deltas[ch]));
    }

    // 10. Assemble body
    let mut body: Vec<u8> = Vec::new();

    // Edge-energy header: reference values + step sizes (20 bytes)
    body.extend_from_slice(&ref_l.to_le_bytes());
    body.extend_from_slice(&ref_c1.to_le_bytes());
    body.extend_from_slice(&ref_c2.to_le_bytes());
    body.extend_from_slice(&(dc_step_l as f32).to_le_bytes());
    body.extend_from_slice(&(dc_step_c as f32).to_le_bytes());

    // Mode byte = 3 signals hex-path polymerase
    body.push(3u8);

    // Interior step sizes (needed by decoder for dequantization)
    body.extend_from_slice(&(interior_step_l as f32).to_le_bytes());
    body.extend_from_slice(&(interior_step_c as f32).to_le_bytes());

    // Number of hexes
    body.extend_from_slice(&(n_hexes as u32).to_le_bytes());

    // Gas map
    body.extend_from_slice(&(gas_compressed.len() as u32).to_le_bytes());
    body.extend_from_slice(&gas_compressed);

    // Codon streams (3 channels)
    for ch in 0..3 {
        body.extend_from_slice(&(encoded_codons[ch].len() as u32).to_le_bytes());
        body.extend_from_slice(&encoded_codons[ch]);
    }

    // DC streams (3 channels)
    for ch in 0..3 {
        body.extend_from_slice(&(encoded_dcs[ch].len() as u32).to_le_bytes());
        body.extend_from_slice(&encoded_dcs[ch]);
    }

    // Interior streams (3 channels): count + rANS data
    for ch in 0..3 {
        // Write the exact number of interior deltas (needed by decoder for rANS)
        let n_interior = all_interior_deltas[ch].len();
        body.extend_from_slice(&(n_interior as u32).to_le_bytes());
        body.extend_from_slice(&(encoded_interiors[ch].len() as u32).to_le_bytes());
        body.extend_from_slice(&encoded_interiors[ch]);
    }

    // 11. AUR2 header with version=8
    let header = Aur2Header {
        version: 8,
        quality,
        width,
        height,
        wv_levels: 0,
        detail_step,
        ll_ranges: [(0.0, 0.0); 3],
    };

    let header_bytes = bitstream::write_aur2_header(&header);
    let total_size = header_bytes.len() + body.len();
    let mut aurea_data = Vec::with_capacity(total_size);
    aurea_data.extend_from_slice(&header_bytes);
    aurea_data.extend_from_slice(&body);

    let bpp = total_size as f64 * 8.0 / (width * height) as f64;
    let gas_bpp = gas_compressed.len() as f64 * 8.0 / n as f64;
    let codon_bpp = encoded_codons.iter().map(|s| s.len()).sum::<usize>() as f64 * 8.0 / n as f64;
    let dc_bpp = encoded_dcs.iter().map(|s| s.len()).sum::<usize>() as f64 * 8.0 / n as f64;
    let int_bpp = encoded_interiors.iter().map(|s| s.len()).sum::<usize>() as f64 * 8.0 / n as f64;
    eprintln!("  Hex polymerase: {} bytes ({:.2} bpp = {:.3} gas + {:.3} codons + {:.3} dc + {:.3} interior)",
        total_size, bpp, gas_bpp, codon_bpp, dc_bpp, int_bpp);

    Ok(aurea_encoder::AureaEncoderResult {
        aurea_data,
        compressed_size: total_size,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_roundtrip() {
        // Linear gradient 0..29
        let n = 30;
        let input: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let coeffs = dct_ii_1d(&input);
        assert_eq!(coeffs.len(), n / 2);

        let reconstructed = idct_ii_1d(&coeffs, n);
        assert_eq!(reconstructed.len(), n);

        let max_err = input
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        println!(
            "DCT roundtrip: {} coeffs, max error = {:.6}",
            coeffs.len(),
            max_err
        );
        assert!(
            max_err < 0.5,
            "DCT roundtrip max error {:.6} >= 0.5",
            max_err
        );
    }

    #[test]
    fn test_classify_vertex_flat() {
        // Flat plane of constant 128.0
        let w = 32;
        let h = 32;
        let plane = vec![128.0; w * h];

        let codon = classify_vertex(&plane, w, h, 16.0, 16.0);
        println!("flat codon: {:?}", codon);
        assert_eq!(codon.vtype, 0, "flat region should be Rien (vtype=0)");
    }

    #[test]
    fn test_subdivision_determination() {
        let rien = VertexCodon {
            vtype: 0,
            freq_idx: 0,
            amp_idx: 0,
            angle: 0.0,
        };
        let seg = VertexCodon {
            vtype: 1,
            freq_idx: 2,
            amp_idx: 3,
            angle: 0.0,
        };

        // All Rien -> Whole
        assert_eq!(
            determine_subdivision(&[rien; 6]),
            HexSubdiv::Whole
        );

        // 1 active -> Bisect
        let mut mixed1 = [rien; 6];
        mixed1[0] = seg;
        assert_eq!(
            determine_subdivision(&mixed1),
            HexSubdiv::Bisect
        );

        // 2 active -> Bisect
        let mut mixed2 = [rien; 6];
        mixed2[0] = seg;
        mixed2[3] = seg;
        assert_eq!(
            determine_subdivision(&mixed2),
            HexSubdiv::Bisect
        );

        // 3 active -> Trisect
        let mut mixed3 = [rien; 6];
        mixed3[0] = seg;
        mixed3[2] = seg;
        mixed3[4] = seg;
        assert_eq!(
            determine_subdivision(&mixed3),
            HexSubdiv::Trisect
        );

        // 5 active -> Full
        let mut mixed5 = [seg; 6];
        mixed5[3] = rien;
        assert_eq!(
            determine_subdivision(&mixed5),
            HexSubdiv::Full
        );

        // All active -> Full
        assert_eq!(
            determine_subdivision(&[seg; 6]),
            HexSubdiv::Full
        );
    }

    #[test]
    fn test_subdivision_pack_unpack_roundtrip() {
        use crate::hex::HexSubdiv;

        let subdivs = vec![
            HexSubdiv::Whole,
            HexSubdiv::Bisect,
            HexSubdiv::Trisect,
            HexSubdiv::Full,
            HexSubdiv::Whole,
            HexSubdiv::Full,
            HexSubdiv::Bisect,
        ];

        let packed = pack_subdivisions(&subdivs);
        let unpacked = unpack_subdivisions(&packed, subdivs.len());

        assert_eq!(unpacked.len(), subdivs.len());
        for (i, (&orig, &decoded)) in subdivs.iter().zip(unpacked.iter()).enumerate() {
            assert_eq!(orig, decoded, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_classify_vertex_edge() {
        // Horizontal step edge: top half = 50, bottom half = 200
        let w = 32;
        let h = 32;
        let mut plane = vec![50.0; w * h];
        for y in (h / 2)..h {
            for x in 0..w {
                plane[y * w + x] = 200.0;
            }
        }

        // Place vertex right at the edge boundary
        let codon = classify_vertex(&plane, w, h, 16.0, 16.0);
        println!("edge codon: {:?}", codon);
        assert_eq!(
            codon.vtype, 1,
            "horizontal step edge should be Segment (vtype=1)"
        );
    }
}

// ---------------------------------------------------------------------------
// Optica v11 encoder (photon synthesis + capillary chroma + hyper-sparse rANS)
// ---------------------------------------------------------------------------

/// Encode using the Optica v11 pipeline.
///
/// Pipeline: GCT -> PTF -> photon noise stripping -> path-DPCM with rANS v11 ->
/// capillary chroma (Voronoi sparse sampling) -> serialize v11 bitstream.
pub fn encode_optica(
    rgb: &[u8],
    width: usize,
    height: usize,
    params: &crate::codec_params::CodecParams,
) -> Result<crate::aurea_encoder::AureaEncoderResult, Box<dyn std::error::Error>> {
    use crate::aurea_encoder;
    use crate::bitstream::{self, Aur2Header};
    use crate::color;
    use crate::golden::PTF_GAMMA;
    use crate::hex_edge;
    use crate::rans;
    use crate::photon;
    use crate::capillary;
    use crate::hex;

    let quality = params.quality;
    let n = width * height;

    // 1. GCT: RGB -> (L, C1, C2)
    let (mut l_ch, c1_ch, c2_ch) = color::golden_rotate_forward(rgb, n);

    // 2. PTF on L channel
    {
        let inv255 = 1.0 / 255.0;
        for v in l_ch.iter_mut() {
            *v = 255.0 * (*v * inv255).clamp(0.0, 1.0).powf(PTF_GAMMA);
        }
    }

    // 3. Photon profiling (estimation only — stored for future activation)
    let photon_map = photon::estimate_tile_sigmas(&l_ch, height, width);
    let n_noisy = photon_map.sigmas.iter().filter(|&&s| s > 0).count();
    eprintln!("  Photon: {}/{} tiles profiled", n_noisy, photon_map.sigmas.len());

    // 4. Step computation (same as v8)
    let detail_step = (aurea_encoder::auto_detail_step(&l_ch, height, width, quality) as f32) as f64;
    let q_frac = (quality as f64 / 100.0).clamp(0.01, 1.0);

    // DC step: used for hex DC grid DPCM (coarse, ~8-13 at q=50)
    let edge_step_l = ((4.0 + (1.0 - q_frac) * 18.0) as f32) as f64;
    let edge_step_c = ((edge_step_l * params.edge_step_c_ratio) as f32) as f64;

    // Residual step with DPCM: deltas between adjacent residuals are small,
    // so step can be the same scale as v8's edge_step.
    // Foveal saliency modulates: edges get step*φ^(-2)≈0.38x, flats keep step.
    let res_step_l = ((4.0 + (1.0 - q_frac) * 18.0) as f32) as f64;
    let res_step_c = ((res_step_l * params.edge_step_c_ratio) as f32) as f64;

    let dead_zone = params.dead_zone;

    let edge_step_l = (edge_step_l as f32) as f64;
    let edge_step_c = (edge_step_c as f32) as f64;
    let res_step_l = (res_step_l as f32) as f64;
    let res_step_c = (res_step_c as f32) as f64;

    // 5. Reference pixel values
    let ref_l = l_ch[0] as f32;
    let ref_c1 = c1_ch[0] as f32;
    let ref_c2 = c2_ch[0] as f32;

    // 6. Hex prediction: compute DC per hex → build per-pixel prediction map
    let grid = hex::HexGrid::new(width, height);
    let shape = hex::compute_hex_shape();
    let n_hexes = grid.cols * grid.rows;

    // Encode hex DCs as DPCM (transmitted in bitstream for decoder)
    let (dc_deltas_l, recon_dcs_l) = hex_edge::encode_hex_dc_grid(&l_ch, width, height, &grid, &shape, edge_step_l);
    let (dc_deltas_c1, recon_dcs_c1) = hex_edge::encode_hex_dc_grid(&c1_ch, width, height, &grid, &shape, edge_step_c);
    let (dc_deltas_c2, recon_dcs_c2) = hex_edge::encode_hex_dc_grid(&c2_ch, width, height, &grid, &shape, edge_step_c);

    // Build smooth prediction maps: bilinear interpolation of DC grids
    // No codons needed — the saliency field captures structure at zero bpp cost.
    let pred_l = hex_edge::build_hex_prediction_from_recon_dcs(&recon_dcs_l, width, height, &grid, &shape);
    let pred_c1 = hex_edge::build_hex_prediction_from_recon_dcs(&recon_dcs_c1, width, height, &grid, &shape);
    let pred_c2 = hex_edge::build_hex_prediction_from_recon_dcs(&recon_dcs_c2, width, height, &grid, &shape);

    let hex_id_map = hex_edge::build_hex_ownership_map(width, height, &grid, &shape);
    eprintln!("  Hex predictor: {}x{} = {} hexes", grid.cols, grid.rows, n_hexes);

    // 7. Saliency field S(x,y): replaces binary gas/solid with continuous relief
    // Computed from DC gradient (Weber contrast) — reconstructible at decoder, zero bpp.
    let saliency_l = hex_edge::compute_hex_saliency(&recon_dcs_l, &grid);
    let saliency_map_l = hex_edge::build_saliency_map(&saliency_l, width, height, &grid);
    let saliency_map_c1 = hex_edge::build_saliency_map(
        &hex_edge::compute_hex_saliency(&recon_dcs_c1, &grid), width, height, &grid);
    let saliency_map_c2 = hex_edge::build_saliency_map(
        &hex_edge::compute_hex_saliency(&recon_dcs_c2, &grid), width, height, &grid);

    let avg_s = saliency_l.iter().sum::<f64>() / saliency_l.len() as f64;
    eprintln!("  Saliency: avg={:.3}, range φ^(-1.5·S) = [{:.2}x, {:.2}x] step",
        avg_s, 1.0, crate::golden::PHI.powf(-1.5));

    // 8. Hilbert path + encode residuals with foveal quantization
    let path = hex_edge::hilbert_path(width, height);

    let deltas_l = hex_edge::encode_hex_predicted_residuals(&l_ch, &pred_l, &path, res_step_l, dead_zone, &saliency_map_l, &hex_id_map);
    let deltas_c1 = hex_edge::encode_hex_predicted_residuals(&c1_ch, &pred_c1, &path, res_step_c, dead_zone, &saliency_map_c1, &hex_id_map);
    let deltas_c2 = hex_edge::encode_hex_predicted_residuals(&c2_ch, &pred_c2, &path, res_step_c, dead_zone, &saliency_map_c2, &hex_id_map);

    let nz_l = deltas_l.iter().filter(|&&d| d != 0).count();
    let nz_c1 = deltas_c1.iter().filter(|&&d| d != 0).count();
    let nz_c2 = deltas_c2.iter().filter(|&&d| d != 0).count();
    eprintln!("  Residuals: L {:.1}% nz, C1 {:.1}%, C2 {:.1}%",
        nz_l as f64 / n as f64 * 100.0,
        nz_c1 as f64 / n as f64 * 100.0,
        nz_c2 as f64 / n as f64 * 100.0);

    // 9. rANS v11 encode all streams
    let l_encoded = rans::rans_encode_band_v11(&deltas_l);
    let c1_encoded = rans::rans_encode_band_v11(&deltas_c1);
    let c2_encoded = rans::rans_encode_band_v11(&deltas_c2);

    // DC grid streams (hex-level, much smaller than pixel streams)
    let dc_l_encoded = rans::rans_encode_band_v11(&dc_deltas_l);
    let dc_c1_encoded = rans::rans_encode_band_v11(&dc_deltas_c1);
    let dc_c2_encoded = rans::rans_encode_band_v11(&dc_deltas_c2);

    // 9. Serialize photon map
    let photon_encoded = photon::encode_photon_map(&photon_map);

    // 10. Assemble v11 body
    let mut body: Vec<u8> = Vec::new();

    // Edge-energy header (21 bytes)
    body.extend_from_slice(&ref_l.to_le_bytes());
    body.extend_from_slice(&ref_c1.to_le_bytes());
    body.extend_from_slice(&ref_c2.to_le_bytes());
    body.extend_from_slice(&(edge_step_l as f32).to_le_bytes());
    body.extend_from_slice(&(edge_step_c as f32).to_le_bytes());
    body.push(11u8); // mode=11 (Optica)

    // Residual step sizes (finer than DC step for independent quantization)
    body.extend_from_slice(&(res_step_l as f32).to_le_bytes());
    body.extend_from_slice(&(res_step_c as f32).to_le_bytes());

    // Hex grid dimensions
    body.extend_from_slice(&(grid.cols as u16).to_le_bytes());
    body.extend_from_slice(&(grid.rows as u16).to_le_bytes());
    body.extend_from_slice(&(n_hexes as u32).to_le_bytes());

    // Hex DC grids (3 channels, rANS v11)
    // The decoder reconstructs saliency S(x,y) from these DCs — zero extra bpp.
    for dc_stream in &[&dc_l_encoded, &dc_c1_encoded, &dc_c2_encoded] {
        body.extend_from_slice(&(dc_stream.len() as u32).to_le_bytes());
        body.extend_from_slice(dc_stream);
    }

    // Residual streams (3 channels, rANS v11)
    body.extend_from_slice(&(l_encoded.len() as u32).to_le_bytes());
    body.extend_from_slice(&l_encoded);
    body.extend_from_slice(&(c1_encoded.len() as u32).to_le_bytes());
    body.extend_from_slice(&c1_encoded);
    body.extend_from_slice(&(c2_encoded.len() as u32).to_le_bytes());
    body.extend_from_slice(&c2_encoded);

    // Photon map
    body.extend_from_slice(&(photon_encoded.len() as u32).to_le_bytes());
    body.extend_from_slice(&photon_encoded);

    // 11. AUR2 header (version=11)
    let header = Aur2Header {
        version: 11,
        quality,
        width,
        height,
        wv_levels: 0,
        detail_step,
        ll_ranges: [(0.0, 0.0); 3],
    };

    let header_bytes = bitstream::write_aur2_header(&header);
    let total_size = header_bytes.len() + body.len();
    let mut aurea_data = Vec::with_capacity(total_size);
    aurea_data.extend_from_slice(&header_bytes);
    aurea_data.extend_from_slice(&body);

    let bpp = total_size as f64 * 8.0 / n as f64;
    eprintln!("  Optica v11 total: {} bytes ({:.2} bpp)", total_size, bpp);

    Ok(aurea_encoder::AureaEncoderResult {
        aurea_data,
        compressed_size: total_size,
    })
}
