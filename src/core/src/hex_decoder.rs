//! Harmonic weight matrix for hexagonal cell interior reconstruction.
//!
//! The decoder reconstructs interior pixels from boundary (perimeter) values
//! by solving the discrete Laplace equation via SOR (Successive Over-Relaxation). The result
//! is a precomputed weight matrix W[n_interior × n_perimeter] where each row
//! tells "how much does each perimeter pixel contribute to this interior pixel?"

use crate::golden::PHI_INV;
use crate::hex::{compute_hex_shape_r, HexSubdiv, HEX_R};
use crate::hex_encoder::VertexCodon;
use std::collections::HashMap;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Fibonacci lookup tables for texture field (f64 versions)
// ---------------------------------------------------------------------------

const FIB_FREQS: [f64; 5] = [1.0, 2.0, 3.0, 5.0, 8.0];
const FIB_AMPS: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0];

/// Precomputed harmonic interpolation weights for a hex cell.
///
/// `weights` is row-major: `weights[i * n_perimeter + k]` = contribution of
/// perimeter pixel k to interior pixel i.
pub struct HarmonicWeights {
    pub weights: Vec<f64>,
    pub n_interior: usize,
    pub n_perimeter: usize,
}

impl HarmonicWeights {
    /// Precompute the harmonic weight matrix for the default hex radius (HEX_R=5).
    pub fn precompute() -> Self {
        Self::precompute_r(HEX_R)
    }

    /// Precompute the harmonic weight matrix by solving Laplace's equation
    /// on the hex interior for each perimeter source, for an arbitrary radius.
    ///
    /// For each perimeter pixel k, we set boundary condition: pixel k = 1.0,
    /// all other perimeter pixels = 0.0, then relax interior pixels via SOR
    /// (Successive Over-Relaxation) which converges 10-25x faster than Jacobi
    /// for the discrete Laplace equation.
    pub fn precompute_r(r: usize) -> Self {
        let shape = compute_hex_shape_r(r);
        let n_perimeter = shape.perimeter.len();
        let n_interior = shape.interior.len();

        // Build a lookup from (dx, dy) -> index-in-field for all hex pixels.
        // Field = perimeter pixels (indices 0..n_perimeter) + interior pixels (n_perimeter..).
        let n_total = n_perimeter + n_interior;
        let mut coord_to_idx: HashMap<(i32, i32), usize> = HashMap::with_capacity(n_total);

        for (i, &pt) in shape.perimeter.iter().enumerate() {
            coord_to_idx.insert(pt, i);
        }
        for (i, &pt) in shape.interior.iter().enumerate() {
            coord_to_idx.insert(pt, n_perimeter + i);
        }

        // 6-connectivity offsets (cardinal + diagonal for better hex geometry)
        let deltas: [(i32, i32); 6] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)];

        // Precompute neighbor lists for each interior pixel (only neighbors within the hex).
        let interior_neighbors: Vec<Vec<usize>> = shape
            .interior
            .iter()
            .map(|&(ix, iy)| {
                deltas
                    .iter()
                    .filter_map(|&(dx, dy)| coord_to_idx.get(&(ix + dx, iy + dy)).copied())
                    .collect()
            })
            .collect();

        // SOR parameters: omega ~ 1.7 is near-optimal for Laplace on compact domains.
        // Scale iterations with radius for convergence on larger hexes.
        let omega: f64 = 1.7;
        let n_sor = (50.0 * (r as f64 / 5.0).powi(2)).ceil() as usize;
        let n_sor = n_sor.max(20).min(500);
        let mut weights = vec![0.0f64; n_interior * n_perimeter];

        for k in 0..n_perimeter {
            // Field values: perimeter pixels are fixed boundary conditions,
            // interior pixels are relaxed.
            let mut field = vec![0.0f64; n_total];
            field[k] = 1.0; // source perimeter pixel

            for _iter in 0..n_sor {
                for (i, neighbors) in interior_neighbors.iter().enumerate() {
                    if neighbors.is_empty() {
                        continue;
                    }
                    let int_idx = n_perimeter + i;
                    let old = field[int_idx];
                    // Gauss-Seidel update: reads from current field (already-updated values)
                    let sum: f64 = neighbors.iter().map(|&ni| field[ni]).sum();
                    let gs = sum / neighbors.len() as f64;
                    // SOR: over-relax the Gauss-Seidel update
                    field[int_idx] = old + omega * (gs - old);
                }
            }

            // Extract interior values as weights for source k
            for i in 0..n_interior {
                weights[i * n_perimeter + k] = field[n_perimeter + i];
            }
        }

        HarmonicWeights {
            weights,
            n_interior,
            n_perimeter,
        }
    }
}

/// Precomputed harmonic weights for all four Fibonacci hex radii.
///
/// This avoids recomputing the expensive SOR relaxation on every hex;
/// instead, the decoder precomputes weights for R=3,5,8,13 once and
/// selects the appropriate matrix for each hex cell.
pub struct MultiScaleWeights {
    pub w3: HarmonicWeights,
    pub w5: HarmonicWeights,
    pub w8: HarmonicWeights,
    pub w13: HarmonicWeights,
}

impl MultiScaleWeights {
    /// Precompute harmonic weight matrices for all Fibonacci radii.
    pub fn precompute() -> Self {
        MultiScaleWeights {
            w3: HarmonicWeights::precompute_r(3),
            w5: HarmonicWeights::precompute_r(5),
            w8: HarmonicWeights::precompute_r(8),
            w13: HarmonicWeights::precompute_r(13),
        }
    }

    /// Get the weight matrix for a given radius.
    pub fn for_radius(&self, r: usize) -> &HarmonicWeights {
        match r {
            3 => &self.w3,
            8 => &self.w8,
            13 => &self.w13,
            _ => &self.w5,
        }
    }
}

/// Reconstruct interior pixel values from perimeter values using the
/// precomputed harmonic weight matrix.
///
/// `perimeter_values` must have length `n_perimeter`.
/// Returns a vector of length `n_interior`.
pub fn reconstruct_background(hw: &HarmonicWeights, perimeter_values: &[f64]) -> Vec<f64> {
    assert_eq!(
        perimeter_values.len(),
        hw.n_perimeter,
        "perimeter_values length {} != n_perimeter {}",
        perimeter_values.len(),
        hw.n_perimeter
    );

    let mut interior = vec![0.0f64; hw.n_interior];
    for i in 0..hw.n_interior {
        let row_offset = i * hw.n_perimeter;
        let mut sum = 0.0f64;
        for k in 0..hw.n_perimeter {
            sum += hw.weights[row_offset + k] * perimeter_values[k];
        }
        interior[i] = sum;
    }
    interior
}

// ---------------------------------------------------------------------------
// Phi-potential texture field
// ---------------------------------------------------------------------------

/// Compute the texture field contribution at pixel (px, py) from all active
/// vertex codons, using default decay (no subdivision).
///
/// Each active vertex (vtype != 0, amp_idx != 0) emits a wave that decays
/// as phi^(-distance):
/// - **Segment**: wave = amp * sin(2pi * freq * projection_along_angle / HEX_R)
/// - **Arc**: wave = amp * sin(2pi * freq * distance / HEX_R)
///
/// Returns the summed field value at (px, py).
pub fn texture_field(
    px: f64,
    py: f64,
    vertices: &[(f64, f64)],
    codons: &[VertexCodon],
) -> f64 {
    texture_field_subdivided(px, py, vertices, codons, HexSubdiv::Whole)
}

/// Compute the texture field contribution at pixel (px, py) with
/// subdivision-aware decay power.
///
/// When `subdiv != Whole`, the vertex charges use a slower decay
/// (`phi^(-d * decay_power)`) so they reach further into the interior,
/// preserving more detail in textured regions.
pub fn texture_field_subdivided(
    px: f64,
    py: f64,
    vertices: &[(f64, f64)],
    codons: &[VertexCodon],
    subdiv: HexSubdiv,
) -> f64 {
    texture_field_subdivided_r(px, py, vertices, codons, subdiv, HEX_R)
}

/// Compute the texture field with a specific hex radius.
pub fn texture_field_subdivided_r(
    px: f64,
    py: f64,
    vertices: &[(f64, f64)],
    codons: &[VertexCodon],
    subdiv: HexSubdiv,
    hex_radius: usize,
) -> f64 {
    let hex_r = hex_radius as f64;
    let decay_power = subdiv.decay_power();
    let mut field = 0.0f64;

    for (i, codon) in codons.iter().enumerate() {
        if codon.vtype == 0 || codon.amp_idx == 0 {
            continue;
        }
        if i >= vertices.len() {
            break;
        }

        let (vx, vy) = vertices[i];
        let dx = px - vx;
        let dy = py - vy;
        let d = (dx * dx + dy * dy).sqrt();

        let attenuation = PHI_INV.powf(d * decay_power);
        let freq = FIB_FREQS[codon.freq_idx as usize % FIB_FREQS.len()];
        let amp = FIB_AMPS[codon.amp_idx as usize % FIB_AMPS.len()];

        let wave = match codon.vtype {
            1 => {
                // Segment: project displacement onto edge direction
                let projection = dx * codon.angle.cos() + dy * codon.angle.sin();
                amp * (2.0 * PI * freq * projection / hex_r).sin()
            }
            _ => {
                // Arc (vtype=2 or any other): radial wave
                amp * (2.0 * PI * freq * d / hex_r).sin()
            }
        };

        field += wave * attenuation;
    }

    field
}

/// Reconstruct hex interior pixels by combining harmonic background
/// (from perimeter values) with the phi-potential texture field
/// (from vertex codons). Uses default subdivision (Whole).
///
/// Returns a Vec of length `hw.n_interior`, one value per interior pixel.
pub fn reconstruct_hex_interior(
    hw: &HarmonicWeights,
    perimeter_values: &[f64],
    interior_positions: &[(f64, f64)],
    vertex_positions: &[(f64, f64)],
    codons: &[VertexCodon],
) -> Vec<f64> {
    reconstruct_hex_interior_subdivided(
        hw,
        perimeter_values,
        interior_positions,
        vertex_positions,
        codons,
        HexSubdiv::Whole,
    )
}

/// Reconstruct hex interior pixels with subdivision-aware texture decay.
///
/// When `subdiv != Whole`, the texture field uses a slower decay power,
/// making vertex charges reach further into the interior and preserving
/// more structural detail in textured hexes.
///
/// Returns a Vec of length `hw.n_interior`, one value per interior pixel.
pub fn reconstruct_hex_interior_subdivided(
    hw: &HarmonicWeights,
    perimeter_values: &[f64],
    interior_positions: &[(f64, f64)],
    vertex_positions: &[(f64, f64)],
    codons: &[VertexCodon],
    subdiv: HexSubdiv,
) -> Vec<f64> {
    reconstruct_hex_interior_subdivided_r(
        hw, perimeter_values, interior_positions,
        vertex_positions, codons, subdiv, HEX_R,
    )
}

/// Reconstruct hex interior pixels with subdivision-aware texture decay
/// for an arbitrary hex radius.
///
/// Returns a Vec of length `hw.n_interior`, one value per interior pixel.
pub fn reconstruct_hex_interior_subdivided_r(
    hw: &HarmonicWeights,
    perimeter_values: &[f64],
    interior_positions: &[(f64, f64)],
    vertex_positions: &[(f64, f64)],
    codons: &[VertexCodon],
    subdiv: HexSubdiv,
    hex_radius: usize,
) -> Vec<f64> {
    let background = reconstruct_background(hw, perimeter_values);

    let mut result = Vec::with_capacity(hw.n_interior);
    for (i, &(px, py)) in interior_positions.iter().enumerate() {
        let bg = if i < background.len() { background[i] } else { 0.0 };
        let tex = texture_field_subdivided_r(px, py, vertex_positions, codons, subdiv, hex_radius);
        result.push(bg + tex);
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex::compute_hex_shape;

    #[test]
    fn test_harmonic_weights_dimensions() {
        let shape = compute_hex_shape();
        let hw = HarmonicWeights::precompute();

        println!(
            "n_interior={}, n_perimeter={}, weights.len()={}",
            hw.n_interior,
            hw.n_perimeter,
            hw.weights.len()
        );

        // Dimensions should match the hex shape
        assert_eq!(hw.n_interior, shape.interior.len());
        assert_eq!(hw.n_perimeter, shape.perimeter.len());
        assert_eq!(hw.weights.len(), hw.n_interior * hw.n_perimeter);

        // Sanity: expect ~49 interior, ~28 perimeter (approximate)
        assert!(
            hw.n_interior > 30,
            "n_interior {} too small",
            hw.n_interior
        );
        assert!(
            hw.n_perimeter > 20,
            "n_perimeter {} too small",
            hw.n_perimeter
        );
    }

    #[test]
    fn test_harmonic_weights_sum_to_one() {
        let hw = HarmonicWeights::precompute();

        for i in 0..hw.n_interior {
            let row_offset = i * hw.n_perimeter;
            let sum: f64 = hw.weights[row_offset..row_offset + hw.n_perimeter]
                .iter()
                .sum();
            println!("interior pixel {}: weight sum = {:.6}", i, sum);
            assert!(
                (sum - 1.0).abs() < 0.15,
                "interior pixel {} weight sum {:.6} deviates from 1.0 by more than 0.15",
                i,
                sum
            );
        }
    }

    #[test]
    fn test_reconstruct_constant_perimeter() {
        let hw = HarmonicWeights::precompute();
        let perimeter_values = vec![128.0; hw.n_perimeter];

        let interior = reconstruct_background(&hw, &perimeter_values);

        println!("Constant perimeter=128.0:");
        for (i, &val) in interior.iter().enumerate() {
            println!("  interior[{}] = {:.4}", i, val);
            assert!(
                (val - 128.0).abs() < 5.0,
                "interior[{}] = {:.4}, expected ~128.0 (within 5.0)",
                i,
                val
            );
        }
    }

    #[test]
    fn test_reconstruct_gradient_perimeter() {
        let hw = HarmonicWeights::precompute();

        // Linear gradient: perimeter values go from 50 to 200
        let perimeter_values: Vec<f64> = (0..hw.n_perimeter)
            .map(|k| 50.0 + 150.0 * k as f64 / (hw.n_perimeter - 1).max(1) as f64)
            .collect();

        let pmin = perimeter_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let pmax = perimeter_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let interior = reconstruct_background(&hw, &perimeter_values);

        println!(
            "Gradient perimeter: min={:.1}, max={:.1}",
            pmin, pmax
        );
        for (i, &val) in interior.iter().enumerate() {
            println!("  interior[{}] = {:.4}", i, val);
            assert!(
                val >= pmin - 10.0 && val <= pmax + 10.0,
                "interior[{}] = {:.4} outside [{:.1}, {:.1}]",
                i,
                val,
                pmin - 10.0,
                pmax + 10.0
            );
        }
    }

    // --- Phi-potential texture field tests ---

    /// Helper: build the 6 hex vertex positions for R=5, flat-top.
    fn hex_vertices_r5() -> Vec<(f64, f64)> {
        let r = 5.0f64;
        (0..6)
            .map(|i| {
                let angle = std::f64::consts::PI / 3.0 * i as f64 - std::f64::consts::PI / 6.0;
                (r * angle.cos(), r * angle.sin())
            })
            .collect()
    }

    #[test]
    fn test_texture_field_rien_is_zero() {
        let vertices = hex_vertices_r5();
        let codons = vec![
            VertexCodon { vtype: 0, freq_idx: 0, amp_idx: 0, angle: 0.0 };
            6
        ];

        let field = texture_field(0.0, 0.0, &vertices, &codons);
        println!("Rien codons at center: field = {:.6}", field);
        assert!(
            field.abs() < 1e-12,
            "All-Rien codons should produce zero field, got {}",
            field
        );
    }

    #[test]
    fn test_texture_field_segment_nonzero() {
        let vertices = hex_vertices_r5();
        let mut codons = vec![
            VertexCodon { vtype: 0, freq_idx: 0, amp_idx: 0, angle: 0.0 };
            6
        ];
        // Activate vertex 0 as Segment with freq_idx=1 (freq=2.0), amp_idx=3 (amp=3.0)
        codons[0] = VertexCodon {
            vtype: 1,
            freq_idx: 1,
            amp_idx: 3,
            angle: 0.0, // along x-axis
        };

        let field = texture_field(0.0, 0.0, &vertices, &codons);
        println!("Segment codon at center: field = {:.6}", field);
        assert!(
            field.abs() > 1e-6,
            "Active Segment codon should produce non-zero field at center, got {}",
            field
        );
    }

    #[test]
    fn test_texture_field_subdivided_stronger_with_more_subdivision() {
        let vertices = hex_vertices_r5();
        let mut codons = vec![
            VertexCodon { vtype: 0, freq_idx: 0, amp_idx: 0, angle: 0.0 };
            6
        ];
        // Activate vertex 0 as Segment
        codons[0] = VertexCodon {
            vtype: 1,
            freq_idx: 1,
            amp_idx: 3,
            angle: 0.0,
        };

        // Sample at the center: texture field should be stronger with more subdivision
        // because the decay is slower (charges reach further)
        let whole = texture_field_subdivided(0.0, 0.0, &vertices, &codons, HexSubdiv::Whole).abs();
        let bisect = texture_field_subdivided(0.0, 0.0, &vertices, &codons, HexSubdiv::Bisect).abs();
        let full = texture_field_subdivided(0.0, 0.0, &vertices, &codons, HexSubdiv::Full).abs();

        println!("texture at center: whole={:.4}, bisect={:.4}, full={:.4}", whole, bisect, full);
        // More subdivision = slower decay = stronger texture at center
        assert!(
            bisect >= whole - 1e-6,
            "bisect ({:.4}) should be >= whole ({:.4})",
            bisect, whole
        );
        assert!(
            full >= bisect - 1e-6,
            "full ({:.4}) should be >= bisect ({:.4})",
            full, bisect
        );
    }

    #[test]
    fn test_full_reconstruction_constant() {
        let hw = HarmonicWeights::precompute();
        let shape = compute_hex_shape();

        let perimeter_values = vec![100.0; hw.n_perimeter];
        let codons = vec![
            VertexCodon { vtype: 0, freq_idx: 0, amp_idx: 0, angle: 0.0 };
            6
        ];

        // Interior positions from the hex shape (as f64)
        let interior_positions: Vec<(f64, f64)> = shape
            .interior
            .iter()
            .map(|&(x, y)| (x as f64, y as f64))
            .collect();

        let vertices = hex_vertices_r5();
        let result = reconstruct_hex_interior(
            &hw,
            &perimeter_values,
            &interior_positions,
            &vertices,
            &codons,
        );

        println!("Constant perimeter + Rien codons:");
        for (i, &val) in result.iter().enumerate() {
            println!("  result[{}] = {:.4}", i, val);
            assert!(
                (val - 100.0).abs() < 5.0,
                "result[{}] = {:.4}, expected ~100.0 (within 5.0)",
                i,
                val
            );
        }
    }

    // --- Multi-scale weight tests ---

    #[test]
    fn test_harmonic_weights_r3_dimensions() {
        let shape = compute_hex_shape_r(3);
        let hw = HarmonicWeights::precompute_r(3);
        println!(
            "R=3: n_interior={}, n_perimeter={}",
            hw.n_interior, hw.n_perimeter
        );
        assert_eq!(hw.n_interior, shape.interior.len());
        assert_eq!(hw.n_perimeter, shape.perimeter.len());
        assert!(hw.n_interior > 0, "R=3 should have interior pixels");
    }

    #[test]
    fn test_harmonic_weights_r13_dimensions() {
        let shape = compute_hex_shape_r(13);
        let hw = HarmonicWeights::precompute_r(13);
        println!(
            "R=13: n_interior={}, n_perimeter={}",
            hw.n_interior, hw.n_perimeter
        );
        assert_eq!(hw.n_interior, shape.interior.len());
        assert_eq!(hw.n_perimeter, shape.perimeter.len());
        // R=13 should have many more interior pixels than R=5
        let hw5 = HarmonicWeights::precompute_r(5);
        assert!(
            hw.n_interior > hw5.n_interior * 2,
            "R=13 interior ({}) should be much larger than R=5 interior ({})",
            hw.n_interior, hw5.n_interior
        );
    }

    #[test]
    fn test_multi_scale_weights_for_radius() {
        let msw = MultiScaleWeights::precompute();
        assert_eq!(msw.for_radius(3).n_perimeter, msw.w3.n_perimeter);
        assert_eq!(msw.for_radius(5).n_perimeter, msw.w5.n_perimeter);
        assert_eq!(msw.for_radius(8).n_perimeter, msw.w8.n_perimeter);
        assert_eq!(msw.for_radius(13).n_perimeter, msw.w13.n_perimeter);
        // Unknown radius falls back to R=5
        assert_eq!(msw.for_radius(7).n_perimeter, msw.w5.n_perimeter);
    }

    #[test]
    fn test_reconstruct_constant_r3() {
        let hw = HarmonicWeights::precompute_r(3);
        let perimeter_values = vec![128.0; hw.n_perimeter];
        let interior = reconstruct_background(&hw, &perimeter_values);
        println!("R=3 constant perimeter=128.0:");
        for (i, &val) in interior.iter().enumerate() {
            assert!(
                (val - 128.0).abs() < 10.0,
                "R=3 interior[{}] = {:.4}, expected ~128.0",
                i, val
            );
        }
    }
}
