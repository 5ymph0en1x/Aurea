//! Capillary Chroma — sparse hex Voronoi chroma sampling + luma-guided SOR reconstruction.
//!
//! Encoder: sample C1/C2 only at hex Voronoi centers (much sparser than 4:2:0).
//! Decoder: reconstruct full chroma via SOR diffusion guided by luma edges.

use crate::hex::{HexGrid, HexShape};

/// Sample chroma values at hex Voronoi centers.
/// Returns one mean value per hex cell (grid.cols * grid.rows values).
pub fn sample_chroma_at_voronoi(
    chroma: &[f64], w: usize, h: usize,
    grid: &HexGrid, shape: &HexShape,
) -> Vec<f64> {
    let n_hexes = grid.cols * grid.rows;
    let mut samples = Vec::with_capacity(n_hexes);

    for row in 0..grid.rows {
        for col in 0..grid.cols {
            let (cx, cy) = grid.center(col, row);
            let cx_i = cx.round() as i32;
            let cy_i = cy.round() as i32;

            let mut sum = 0.0;
            let mut count = 0;

            for &(dx, dy) in &shape.voronoi {
                let px = cx_i + dx;
                let py = cy_i + dy;
                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                    sum += chroma[py as usize * w + px as usize];
                    count += 1;
                }
            }

            samples.push(if count > 0 { sum / count as f64 } else { 0.0 });
        }
    }

    samples
}

/// Encode sampled chroma as DPCM deltas (raster order through hex grid).
pub fn encode_capillary_dpcm(
    samples: &[f64], step: f64, dead_zone: f64,
) -> Vec<i16> {
    let mut deltas = Vec::with_capacity(samples.len());
    let mut prev = 0.0f64;

    for &sample in samples {
        let delta = sample - prev;
        // Dead-zone quantization
        let q = if delta.abs() < (0.5 + dead_zone) * step {
            0i16
        } else {
            (delta / step).round() as i16
        };
        deltas.push(q);
        prev += q as f64 * step;
    }

    deltas
}

/// Decode capillary DPCM: inverse accumulation.
pub fn decode_capillary_dpcm(deltas: &[i16], step: f64) -> Vec<f64> {
    let mut samples = Vec::with_capacity(deltas.len());
    let mut prev = 0.0f64;

    for &d in deltas {
        prev += d as f64 * step;
        samples.push(prev);
    }

    samples
}

/// Reconstruct full-resolution chroma from sparse hex Voronoi samples using
/// luma-guided SOR (Successive Over-Relaxation) diffusion.
///
/// The luma plane guides the diffusion: chroma doesn't bleed across luma edges.
pub fn reconstruct_capillary_chroma(
    luma: &[f64],
    voronoi_values: &[f64],
    grid: &HexGrid,
    shape: &HexShape,
    w: usize, h: usize,
    sigma_guide: f64,
    n_iter: usize,
    omega: f64,
) -> Vec<f64> {
    let n_pixels = w * h;
    let mut output = vec![0.0f64; n_pixels];
    let mut pinned = vec![false; n_pixels];

    // 1. Scatter Voronoi center values
    let mut sample_idx = 0;
    for row in 0..grid.rows {
        for col in 0..grid.cols {
            let (cx, cy) = grid.center(col, row);
            let px = cx.round() as i32;
            let py = cy.round() as i32;
            if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                let idx = py as usize * w + px as usize;
                output[idx] = voronoi_values[sample_idx];
                pinned[idx] = true;
            }
            sample_idx += 1;
        }
    }

    // Also fill Voronoi regions with initial center value for faster convergence
    sample_idx = 0;
    for row in 0..grid.rows {
        for col in 0..grid.cols {
            let (cx, cy) = grid.center(col, row);
            let cx_i = cx.round() as i32;
            let cy_i = cy.round() as i32;
            let val = voronoi_values[sample_idx];

            for &(dx, dy) in &shape.voronoi {
                let px = cx_i + dx;
                let py = cy_i + dy;
                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                    let idx = py as usize * w + px as usize;
                    if !pinned[idx] {
                        output[idx] = val;
                    }
                }
            }
            sample_idx += 1;
        }
    }

    // 2. Precompute bilateral weight LUT: exp(-d^2 / (2*sigma^2))
    let inv_2sigma2 = 1.0 / (2.0 * sigma_guide * sigma_guide);
    let mut exp_lut = [0.0f64; 256];
    for d in 0..256 {
        exp_lut[d] = (-(d as f64 * d as f64) * inv_2sigma2).exp();
    }

    // 3. SOR iterations with luma-guided bilateral weights
    for _iter in 0..n_iter {
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if pinned[idx] {
                    continue;
                }

                let l_center = luma[idx];
                let mut wsum = 0.0;
                let mut vsum = 0.0;

                // 4-connected neighbors
                let neighbors: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for &(dx, dy) in &neighbors {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        let nidx = ny as usize * w + nx as usize;
                        let l_diff = (luma[nidx] - l_center).abs();
                        let d_int = (l_diff as usize).min(255);
                        let weight = exp_lut[d_int];
                        wsum += weight;
                        vsum += weight * output[nidx];
                    }
                }

                if wsum > 0.0 {
                    let new_val = vsum / wsum;
                    output[idx] = (1.0 - omega) * output[idx] + omega * new_val;
                }
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capillary_dpcm_roundtrip() {
        let samples = vec![10.0, 12.0, 11.5, 15.0, 10.0];
        let step = 1.0;
        let deltas = encode_capillary_dpcm(&samples, step, 0.0);
        let decoded = decode_capillary_dpcm(&deltas, step);
        for (orig, dec) in samples.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < step, "DPCM roundtrip error: {} vs {}", orig, dec);
        }
    }

    #[test]
    fn test_sor_flat_chroma() {
        // Constant chroma + constant luma → SOR should return same value everywhere
        let w = 32;
        let h = 32;
        let luma = vec![128.0; w * h];
        let grid = HexGrid::new(w, h);
        let shape = crate::hex::compute_hex_shape();
        let n_hexes = grid.cols * grid.rows;

        let voronoi_values = vec![50.0; n_hexes];
        let result = reconstruct_capillary_chroma(
            &luma, &voronoi_values, &grid, &shape, w, h, 10.0, 30, 1.7,
        );

        // All pixels should converge to ~50.0
        let mae: f64 = result.iter().map(|&v| (v - 50.0).abs()).sum::<f64>() / result.len() as f64;
        assert!(mae < 1.0, "flat chroma SOR MAE = {} (expected < 1.0)", mae);
    }
}
