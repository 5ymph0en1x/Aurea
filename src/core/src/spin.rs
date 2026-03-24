//! Fibonacci spectral spin + dither.
//! Decoder-only enhancements for PSNR recovery and perceived sharpness.

use rayon::prelude::*;

/// Blend ratio: conservative start (v6 spectral spin precedent).
/// Target: phi_inv² = 0.382 (golden partition). Increase empirically.
const SPIN_MEDIAN_WEIGHT: f64 = 0.382; // phi_inv² — golden partition

/// Median regularization of AC coefficients across neighboring blocks.
/// For each AC frequency position, blend toward the median of the same
/// position in neighboring blocks. Corrects quantization error.
pub fn fibonacci_spectral_spin(
    ac_blocks: &mut [Vec<f64>],
    grid_h: usize,
    grid_w: usize,
    ac_per_block: usize,
    local_steps: &[f64],
    spin_weight: f64,
) {
    if grid_h * grid_w < 2 { return; }

    let med_weight = spin_weight.clamp(0.0, 0.5);

    let snapshot: Vec<Vec<f64>> = ac_blocks.iter().map(|b| b.clone()).collect();

    // Gas/Solid classification in frequency domain.
    // Gas (smooth): low AC energy → variations between blocks are quantization noise → spin.
    // Solid (textured): high AC energy → variations are SIGNAL → preserve intact.
    // Same philosophy as deblock_gas_only in spatial domain.
    let block_energies: Vec<f64> = snapshot.iter().enumerate().map(|(i, block)| {
        let step = local_steps[i].max(0.1);
        // Normalize by local step: energy/step = number of "significant" quanta.
        // Gas: few significant coefficients. Solid: many.
        block.iter().map(|v| v.abs()).sum::<f64>() / (block.len().max(1) as f64 * step)
    }).collect();

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let idx = gy * grid_w + gx;
            let step = local_steps[idx].max(0.1);

            // Gas/Solid gate: normalized energy below 1.0 = gas, above = solid.
            // Gas gets full spin, solid gets none. Linear ramp in [0.5, 1.5].
            let e = block_energies[idx];
            let gas_gate = (1.5 - e).clamp(0.0, 1.0);
            let block_med_weight = med_weight * gas_gate;
            if block_med_weight < 1e-6 { continue; } // solid block: skip entirely
            let block_orig_weight = 1.0 - block_med_weight;

            for ac_pos in 0..ac_per_block {
                let current = snapshot[idx][ac_pos];

                let mut neighbors = Vec::with_capacity(5);
                neighbors.push(current);
                if gy > 0 { neighbors.push(snapshot[(gy - 1) * grid_w + gx][ac_pos]); }
                if gy + 1 < grid_h { neighbors.push(snapshot[(gy + 1) * grid_w + gx][ac_pos]); }
                if gx > 0 { neighbors.push(snapshot[gy * grid_w + gx - 1][ac_pos]); }
                if gx + 1 < grid_w { neighbors.push(snapshot[gy * grid_w + gx + 1][ac_pos]); }

                neighbors.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let med = neighbors[neighbors.len() / 2];

                if (med - current).abs() < step {
                    ac_blocks[idx][ac_pos] = block_orig_weight * current + block_med_weight * med;
                }
            }
        }
    }
}

/// Deterministic Fibonacci dither to break quantization plateaus.
/// Applied in spatial domain (PTF space) after LOT synthesis.
pub fn fibonacci_dither(
    plane: &mut [f64],
    h: usize, w: usize,
    detail_step: f64,
    dc_grid: &[f64],
    grid_h: usize, grid_w: usize,
    block_size: usize,
) {
    use crate::golden::PHI_INV2;
    use crate::lot;

    if h < 2 || w < 2 { return; }

    let src = plane.to_vec();
    let lut = fibonacci_lut();
    const PHI_INT: u64 = 0x9E3779B97F4A7C15;

    // Parallel per-row dithering
    plane.par_chunks_mut(w).enumerate().for_each(|(y, row_out)| {
        for x in 0..w {
            let fib_phase = ((y as u64).wrapping_mul(2654435761)
                .wrapping_add((x as u64).wrapping_mul(40503))
                .wrapping_mul(PHI_INT)) >> 56;
            let fib_val = lut[(fib_phase & 0xFF) as usize];

            let gy = (y / block_size).min(grid_h.saturating_sub(1));
            let gx = (x / block_size).min(grid_w.saturating_sub(1));
            let dc_l = dc_grid[gy * grid_w + gx];
            let local_step = detail_step * lot::codon_dc_factor(dc_l, 0.0, 0.0);

            let amplitude = local_step * PHI_INV2 * 0.5;

            let grad_y = if y + 1 < h { (src[(y + 1) * w + x] - src[y * w + x]).abs() } else { 0.0 };
            let grad_x = if x + 1 < w { (src[y * w + x + 1] - src[y * w + x]).abs() } else { 0.0 };
            let grad = grad_y.max(grad_x);
            let texture_mask = (1.0 - grad / local_step.max(1.0)).clamp(0.0, 1.0);

            row_out[x] = src[y * w + x] + fib_val * amplitude * texture_mask;
        }
    });
}

/// Bilinear DC interpolation: smooths block-constant DC into continuous gradients.
/// Instead of each 16x16 block having a flat DC, interpolate between block centers.
/// This eliminates the "Minecraft staircase" effect in smooth gradients.
/// Decoder-only, zero bitstream cost.
pub fn interpolate_dc_grid(
    plane: &mut [f64],
    h: usize, w: usize,
    dc_grid: &[f64],
    grid_h: usize, grid_w: usize,
    block_size: usize,
) {
    if grid_h < 2 || grid_w < 2 { return; }

    let half = block_size as f64 / 2.0; // block center offset

    // Build smooth DC plane by bilinear interpolation of DC grid
    let mut smooth_dc = vec![0.0f64; h * w];
    for y in 0..h {
        // Map pixel to fractional grid position (centered on block)
        let gy_f = (y as f64 - half + 0.5) / block_size as f64;
        let gy0 = (gy_f.floor() as i64).clamp(0, grid_h as i64 - 2) as usize;
        let gy1 = gy0 + 1;
        let dy = (gy_f - gy0 as f64).clamp(0.0, 1.0);

        for x in 0..w {
            let gx_f = (x as f64 - half + 0.5) / block_size as f64;
            let gx0 = (gx_f.floor() as i64).clamp(0, grid_w as i64 - 2) as usize;
            let gx1 = gx0 + 1;
            let dx = (gx_f - gx0 as f64).clamp(0.0, 1.0);

            let v00 = dc_grid[gy0 * grid_w + gx0];
            let v01 = dc_grid[gy0 * grid_w + gx1];
            let v10 = dc_grid[gy1 * grid_w + gx0];
            let v11 = dc_grid[gy1 * grid_w + gx1];

            smooth_dc[y * w + x] = v00 * (1.0 - dy) * (1.0 - dx)
                + v01 * (1.0 - dy) * dx
                + v10 * dy * (1.0 - dx)
                + v11 * dy * dx;
        }
    }

    // The LOT synthesis already produces: pixel = IDCT(DC + AC)
    // For a block with DC value D, the flat contribution is D/block_size (after IDCT normalization)
    // We want to replace the block-constant DC with the interpolated DC.
    // Strategy: compute the block-constant DC for each pixel, then replace with smooth DC.
    // pixel_new = pixel_old - dc_block_constant + dc_interpolated

    let stride = block_size;
    for y in 0..h {
        let gy = (y / stride).min(grid_h.saturating_sub(1));
        for x in 0..w {
            let gx = (x / stride).min(grid_w.saturating_sub(1));
            let dc_block = dc_grid[gy * grid_w + gx];
            let dc_smooth = smooth_dc[y * w + x];
            plane[y * w + x] += dc_smooth - dc_block;
        }
    }
}

/// Echolot-style spectral spin: spatial median filter on reconstructed plane.
/// Applied AFTER LOT synthesis (spatial domain), not on AC blocks.
/// 5x5 median, 92/8 blend — same as echolot's proven +0.35 dB formula.
pub fn spatial_spectral_spin(plane: &mut [f64], h: usize, w: usize) {
    if h < 5 || w < 5 { return; }

    let src = plane.to_vec();
    const PAD: usize = 2;

    // Parallel per-row: each row's median filter is independent
    let interior_rows = h - 2 * PAD;
    let mut output_rows = vec![0.0f64; interior_rows * w];

    output_rows.par_chunks_mut(w).enumerate().for_each(|(ri, row_out)| {
        let y = ri + PAD;
        let mut buf = [0.0f64; 25];
        for x in PAD..w - PAD {
            let mut count = 0;
            for dy in 0..5 {
                for dx in 0..5 {
                    buf[count] = src[(y + dy - PAD) * w + (x + dx - PAD)];
                    count += 1;
                }
            }
            buf[..25].select_nth_unstable_by(12, |a, b| a.partial_cmp(b).unwrap());
            row_out[x] = src[y * w + x] * 0.92 + buf[12] * 0.08;
        }
        // Copy border pixels unchanged
        for x in 0..PAD { row_out[x] = src[y * w + x]; }
        for x in w - PAD..w { row_out[x] = src[y * w + x]; }
    });

    // Write back
    for ri in 0..interior_rows {
        let y = ri + PAD;
        plane[y * w..(y + 1) * w].copy_from_slice(&output_rows[ri * w..(ri + 1) * w]);
    }
}

fn fibonacci_lut() -> [f64; 256] {
    let mut lut = [0.0f64; 256];
    let mut a: u64 = 1;
    let mut b: u64 = 1;
    for i in 0..256 {
        lut[i] = ((a % 256) as f64 / 128.0) - 1.0;
        let c = a.wrapping_add(b);
        a = b;
        b = c;
    }
    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_lut_range() {
        let lut = fibonacci_lut();
        for &v in &lut {
            assert!(v >= -1.0 && v <= 1.0, "out of range: {}", v);
        }
    }

    #[test]
    fn test_spectral_spin_flat_blocks() {
        let mut ac = vec![vec![10.0, 5.0, -3.0, 1.0]; 4];
        let steps = vec![20.0; 4];
        fibonacci_spectral_spin(&mut ac, 2, 2, 4, &steps, 0.382);
        for block in &ac {
            assert!((block[0] - 10.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_spectral_spin_corrects_gas_outlier() {
        // Gas blocks (low energy relative to step): spin should correct
        let mut ac = vec![vec![1.0], vec![3.0], vec![1.0]];
        let steps = vec![15.0; 3]; // energy/step << 1 → gas
        fibonacci_spectral_spin(&mut ac, 1, 3, 1, &steps, 0.382);
        // Gas gate ≈ 1.0, so correction ≈ full weight
        assert!(ac[1][0] < 3.0, "gas outlier should be corrected toward neighbors, got {}", ac[1][0]);
        assert!(ac[1][0] > 1.0, "but not overcorrected, got {}", ac[1][0]);
    }

    #[test]
    fn test_spectral_spin_preserves_solid() {
        // Solid blocks (high energy): spin should NOT blur
        let mut ac = vec![vec![50.0], vec![60.0], vec![50.0]];
        let steps = vec![15.0; 3]; // energy/step >> 1 → solid
        let orig = 60.0;
        fibonacci_spectral_spin(&mut ac, 1, 3, 1, &steps, 0.382);
        // Solid gate ≈ 0.0, value should be nearly unchanged
        assert!((ac[1][0] - orig).abs() < 1.0,
            "solid block should be preserved, got {} expected ~{}", ac[1][0], orig);
    }

    #[test]
    fn test_spectral_spin_guard_clause() {
        let mut ac = vec![vec![10.0], vec![100.0], vec![10.0]];
        let steps = vec![15.0; 3];
        fibonacci_spectral_spin(&mut ac, 1, 3, 1, &steps, 0.382);
        assert!((ac[1][0] - 100.0).abs() < 0.01, "should be unchanged, got {}", ac[1][0]);
    }

    #[test]
    fn test_spectral_spin_zero_weight() {
        let mut ac = vec![vec![10.0], vec![20.0], vec![10.0]];
        let steps = vec![15.0; 3];
        fibonacci_spectral_spin(&mut ac, 1, 3, 1, &steps, 0.0);
        assert!((ac[1][0] - 20.0).abs() < 0.01, "zero weight should not change values, got {}", ac[1][0]);
    }

    #[test]
    fn test_dither_deterministic() {
        let mut p1 = vec![128.0; 64];
        let mut p2 = vec![128.0; 64];
        let dc = vec![128.0; 4];
        fibonacci_dither(&mut p1, 8, 8, 20.0, &dc, 2, 2, 4);
        fibonacci_dither(&mut p2, 8, 8, 20.0, &dc, 2, 2, 4);
        for i in 0..64 { assert!((p1[i] - p2[i]).abs() < 1e-10); }
    }

    #[test]
    fn test_dither_amplitude_bounded() {
        let mut plane = vec![128.0; 64];
        let dc = vec![128.0; 4];
        fibonacci_dither(&mut plane, 8, 8, 20.0, &dc, 2, 2, 4);
        for &v in &plane { assert!((v - 128.0).abs() < 5.0, "too large: {}", v - 128.0); }
    }
}
