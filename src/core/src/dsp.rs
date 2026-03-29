/// Signal processing: noise estimation, FFT Wiener denoising, directional sharpening.

use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

// ======================================================================
// Utilities
// ======================================================================

/// Index with boundary reflection (scipy 'reflect' mode).
#[inline]
fn reflect(i: i32, n: i32) -> usize {
    if i < 0 {
        (-i).min(n - 1) as usize
    } else if i >= n {
        (2 * (n - 1) - i).max(0) as usize
    } else {
        i as usize
    }
}

/// Approximate median via select_nth_unstable (O(n) average).
fn approx_median(data: &mut [f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    data.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    data[mid]
}

// ======================================================================
// 3x3 Convolution
// ======================================================================

/// 3x3 convolution with boundary reflection (parallel per row).
fn conv_3x3(plane: &[f64], h: usize, w: usize, kernel: &[[f64; 3]; 3]) -> Vec<f64> {
    let hi = h as i32;
    let wi = w as i32;
    let mut result = vec![0.0f64; h * w];

    result.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = 0.0;
            for ky in 0..3i32 {
                let iy = reflect(y as i32 + ky - 1, hi);
                for kx in 0..3i32 {
                    let ix = reflect(x as i32 + kx - 1, wi);
                    sum += plane[iy * w + ix] * kernel[ky as usize][kx as usize];
                }
            }
            row[x] = sum;
        }
    });

    result
}

// ======================================================================
// Separable Gaussian filter (sigma=1.0, radius=4)
// ======================================================================

fn gaussian_weights() -> [f64; 9] {
    let mut w = [0.0f64; 9];
    let mut sum = 0.0;
    for i in 0..9 {
        let x = i as f64 - 4.0;
        w[i] = (-x * x / 2.0).exp();
        sum += w[i];
    }
    for v in w.iter_mut() {
        *v /= sum;
    }
    w
}

fn gaussian_blur(plane: &[f64], h: usize, w: usize) -> Vec<f64> {
    let gw = gaussian_weights();
    let radius = 4i32;
    let hi = h as i32;
    let wi = w as i32;

    // Horizontal pass
    let mut tmp = vec![0.0f64; h * w];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = 0.0;
            for k in 0..9 {
                let ix = reflect(x as i32 + k as i32 - radius, wi);
                sum += plane[y * w + ix] * gw[k];
            }
            row[x] = sum;
        }
    });

    // Vertical pass
    let mut result = vec![0.0f64; h * w];
    result.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = 0.0;
            for k in 0..9 {
                let iy = reflect(y as i32 + k as i32 - radius, hi);
                sum += tmp[iy * w + x] * gw[k];
            }
            row[x] = sum;
        }
    });

    result
}

// ======================================================================
// Noise estimation (Donoho-Johnstone, MAD on Laplacian)
// ======================================================================

/// 5-point Laplacian, returns only interior pixels.
fn laplacian_interior(plane: &[f64], h: usize, w: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity((h - 2) * (w - 2));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let idx = y * w + x;
            let lap = plane[idx - w] + plane[idx + w] + plane[idx - 1] + plane[idx + 1]
                - 4.0 * plane[idx];
            result.push(lap);
        }
    }
    result
}

/// Estimate sensor noise sigma (MAD on Laplacian, 3 RGB channels).
pub fn estimate_noise_sigma(r: &[f64], g: &[f64], b: &[f64], h: usize, w: usize) -> f64 {
    if h <= 2 || w <= 2 {
        return 0.0;
    }
    let norm = 0.6745 * 20.0f64.sqrt();
    let channels = [r, g, b];

    let sigma_sum: f64 = channels
        .iter()
        .map(|ch| {
            let mut lap = laplacian_interior(ch, h, w);
            let med = approx_median(&mut lap);
            let mut abs_dev: Vec<f64> = lap.iter().map(|&v| (v - med).abs()).collect();
            let mad = approx_median(&mut abs_dev);
            mad / norm
        })
        .sum();

    sigma_sum / 3.0
}

// ======================================================================
// FFT Wiener Denoising
// ======================================================================

/// Cache-friendly block transpose.
fn transpose_complex(
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    rows: usize,
    cols: usize,
) {
    const BLK: usize = 64;
    for yb in (0..rows).step_by(BLK) {
        let ye = rows.min(yb + BLK);
        for xb in (0..cols).step_by(BLK) {
            let xe = cols.min(xb + BLK);
            for y in yb..ye {
                for x in xb..xe {
                    dst[x * rows + y] = src[y * cols + x];
                }
            }
        }
    }
}

fn fft2(data: &mut [Complex<f64>], h: usize, w: usize) {
    let mut planner = FftPlanner::new();

    // FFT each row (parallel)
    let fft_row = planner.plan_fft_forward(w);
    data.par_chunks_mut(w)
        .for_each(|row| fft_row.process(row));

    // Transpose -> FFT columns -> Transpose back
    let mut tr = vec![Complex::new(0.0, 0.0); h * w];
    transpose_complex(data, &mut tr, h, w);

    let fft_col = planner.plan_fft_forward(h);
    tr.par_chunks_mut(h)
        .for_each(|col| fft_col.process(col));

    transpose_complex(&tr, data, w, h);
}

fn ifft2(data: &mut [Complex<f64>], h: usize, w: usize) {
    let mut planner = FftPlanner::new();

    let ifft_row = planner.plan_fft_inverse(w);
    data.par_chunks_mut(w)
        .for_each(|row| ifft_row.process(row));

    let mut tr = vec![Complex::new(0.0, 0.0); h * w];
    transpose_complex(data, &mut tr, h, w);

    let ifft_col = planner.plan_fft_inverse(h);
    tr.par_chunks_mut(h)
        .for_each(|col| ifft_col.process(col));

    transpose_complex(&tr, data, w, h);
}

/// Denoise an f64 plane with FFT Wiener filter (in-place).
/// Does nothing if sigma_noise < 5.0.
pub fn denoise_fft_plane(plane: &mut [f64], h: usize, w: usize, sigma_noise: f64) {
    if sigma_noise < 5.0 {
        return;
    }

    let n = h * w;
    let noise_psd = sigma_noise * sigma_noise;
    let n_f64 = n as f64;

    // Real -> Complex
    let mut freq: Vec<Complex<f64>> = plane.iter().map(|&v| Complex::new(v, 0.0)).collect();

    fft2(&mut freq, h, w);

    // Wiener filter
    for c in freq.iter_mut() {
        let power = c.norm_sqr() / n_f64;
        let wiener = (power - noise_psd).max(0.0) / (power + 1e-10);
        *c *= wiener;
    }

    ifft2(&mut freq, h, w);

    // Normalize (rustfft does not normalize IFFT) + clip
    let inv_n = 1.0 / n_f64;
    for i in 0..n {
        plane[i] = (freq[i].re * inv_n).clamp(0.0, 255.0);
    }
}

// ======================================================================
// Directional H/V Sharpening
// ======================================================================

/// Enhance H/V edges on the Y plane (oblique effect, directional unsharp mask).
pub fn directional_sharpen(y: &mut [f64], h: usize, w: usize, strength: f64) {
    if strength <= 0.0 || h < 3 || w < 3 {
        return;
    }

    // Directional Sobel kernels (/8)
    let kern_h: [[f64; 3]; 3] = [
        [-0.125, -0.25, -0.125],
        [0.0, 0.0, 0.0],
        [0.125, 0.25, 0.125],
    ];
    let kern_v: [[f64; 3]; 3] = [
        [-0.125, 0.0, 0.125],
        [-0.25, 0.0, 0.25],
        [-0.125, 0.0, 0.125],
    ];
    let kern_d1: [[f64; 3]; 3] = [
        [0.0, -0.125, -0.25],
        [0.125, 0.0, -0.125],
        [0.25, 0.125, 0.0],
    ];
    let kern_d2: [[f64; 3]; 3] = [
        [-0.25, -0.125, 0.0],
        [-0.125, 0.0, 0.125],
        [0.0, 0.125, 0.25],
    ];

    // 4 convolutions in parallel
    let ((grad_h, grad_v), (grad_d1, grad_d2)) = rayon::join(
        || rayon::join(
            || conv_3x3(y, h, w, &kern_h),
            || conv_3x3(y, h, w, &kern_v),
        ),
        || rayon::join(
            || conv_3x3(y, h, w, &kern_d1),
            || conv_3x3(y, h, w, &kern_d2),
        ),
    );

    let n = h * w;
    let blurred = gaussian_blur(y, h, w);

    // HV mask + directional unsharp mask
    for i in 0..n {
        let hv = grad_h[i].abs() + grad_v[i].abs();
        let oblique = grad_d1[i].abs() + grad_d2[i].abs();
        let ratio = hv / (oblique + 1e-6);
        let mask = ((ratio - 0.8) / 0.4).clamp(0.0, 1.0);
        let detail = y[i] - blurred[i];
        y[i] = (y[i] + strength * detail * mask).clamp(0.0, 255.0);
    }
}

// ======================================================================
// Ternary supercordes sharpening (DNA4 decoder-side)
// ======================================================================

/// Ternary supercordes classification per block.
#[derive(Clone, Copy, PartialEq)]
pub enum Supercorde {
    /// No structure — flat/noise region. Don't touch.
    Rien,
    /// Linear structure — edge with dominant direction. Sharpen perpendicular.
    Segment { angle_rad: f64 },
    /// Curved/mixed structure — no single direction. Sharpen isotropically.
    Arc,
}

/// Classify each block in the DC grid as Segment, Arc, or Rien.
/// Uses DC gradient magnitude and directionality.
/// Identical in encoder and decoder (pure DC grid computation).
pub fn supercordes_classify(
    dc_grid: &[f64], grid_h: usize, grid_w: usize,
) -> Vec<Supercorde> {
    let n = grid_h * grid_w;
    if n == 0 { return Vec::new(); }

    // Compute per-block gradient: dx, dy from DC neighbors
    let mut classes = vec![Supercorde::Rien; n];
    let mut magnitudes = vec![0.0f64; n];

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let idx = gy * grid_w + gx;
            let dc = dc_grid[idx];

            // Central differences (mirror at boundaries)
            let left  = if gx > 0 { dc_grid[gy * grid_w + gx - 1] } else { dc };
            let right = if gx + 1 < grid_w { dc_grid[gy * grid_w + gx + 1] } else { dc };
            let top   = if gy > 0 { dc_grid[(gy - 1) * grid_w + gx] } else { dc };
            let bot   = if gy + 1 < grid_h { dc_grid[(gy + 1) * grid_w + gx] } else { dc };

            let dx = (right - left) * 0.5;
            let dy = (bot - top) * 0.5;
            let raw_mag = (dx * dx + dy * dy).sqrt();
            // Weber-Fechner: scale gradient by inverse luminance.
            // Dark areas (dc=20): gradient×10 amplification → subtle texture detected.
            // Bright areas (dc=200): gradient×1.2 → only strong edges matter.
            let weber = 255.0 / (dc.abs() + 25.0); // +25 avoids div/0, caps at ~10×
            magnitudes[idx] = raw_mag * weber;
        }
    }

    // Median magnitude = threshold between structure and rien
    let mut sorted_mag = magnitudes.clone();
    sorted_mag.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median_mag = sorted_mag[sorted_mag.len() / 2].max(0.5);

    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let idx = gy * grid_w + gx;
            let mag = magnitudes[idx];

            if mag < median_mag * 0.5 {
                // Below half the median: RIEN (no structure)
                classes[idx] = Supercorde::Rien;
            } else {
                // Structural block: check directionality
                let dc = dc_grid[idx];
                let left  = if gx > 0 { dc_grid[gy * grid_w + gx - 1] } else { dc };
                let right = if gx + 1 < grid_w { dc_grid[gy * grid_w + gx + 1] } else { dc };
                let top   = if gy > 0 { dc_grid[(gy - 1) * grid_w + gx] } else { dc };
                let bot   = if gy + 1 < grid_h { dc_grid[(gy + 1) * grid_w + gx] } else { dc };

                let dx = (right - left) * 0.5;
                let dy = (bot - top) * 0.5;

                // Anisotropy: how dominant is one direction?
                let abs_dx = dx.abs();
                let abs_dy = dy.abs();
                let dominant = abs_dx.max(abs_dy);
                let minor = abs_dx.min(abs_dy);

                if dominant > minor * 2.0 {
                    // Strong anisotropy: SEGMENT (edge with direction)
                    let angle = dy.atan2(dx); // gradient direction
                    classes[idx] = Supercorde::Segment { angle_rad: angle };
                } else {
                    // Isotropic structure: ARC (curve, corner, texture)
                    classes[idx] = Supercorde::Arc;
                }
            }
        }
    }
    classes
}

/// Ternary supercordes sharpening: decoder-side, zero bpp cost.
/// - Rien: no sharpening (preserve smoothness, don't amplify noise)
/// - Segment: directional unsharp mask perpendicular to edge
/// - Arc: isotropic unsharp mask
///
/// Strength is gentle (phi^-2 ≈ 0.38) to avoid ringing.
pub fn supercordes_sharpen(
    plane: &mut [f64], h: usize, w: usize,
    dc_grid: &[f64], grid_h: usize, grid_w: usize,
    block_size: usize,
    strength: f64,
) {
    if h < 3 || w < 3 || grid_h < 2 || grid_w < 2 { return; }

    let classes = supercordes_classify(dc_grid, grid_h, grid_w);

    // Compute blurred version for unsharp mask
    let blurred = gaussian_blur(plane, h, w);

    let src = plane.to_vec();
    let hi = h as i32;
    let wi = w as i32;

    for y in 1..h-1 {
        let gy = (y / block_size).min(grid_h - 1);
        for x in 1..w-1 {
            let gx = (x / block_size).min(grid_w - 1);
            let class = classes[gy * grid_w + gx];

            match class {
                Supercorde::Rien => {
                    // No sharpening. Crystal clear = don't touch what's already clean.
                }
                Supercorde::Segment { angle_rad } => {
                    // Sharpen perpendicular to the edge direction.
                    // Edge direction = angle_rad. Sharpen direction = angle_rad + pi/2.
                    let perp = angle_rad + std::f64::consts::FRAC_PI_2;
                    let cos_p = perp.cos();
                    let sin_p = perp.sin();

                    // Sample 2 pixels along perpendicular direction
                    let x1 = reflect((x as f64 + cos_p).round() as i32, wi) as usize;
                    let y1 = reflect((y as f64 + sin_p).round() as i32, hi) as usize;
                    let x2 = reflect((x as f64 - cos_p).round() as i32, wi) as usize;
                    let y2 = reflect((y as f64 - sin_p).round() as i32, hi) as usize;

                    // Directional Laplacian: 2*center - neighbor1 - neighbor2
                    let center = src[y * w + x];
                    let laplacian = 2.0 * center - src[y1 * w + x1] - src[y2 * w + x2];

                    // Unsharp: add scaled Laplacian (sharpens perpendicular to edge)
                    plane[y * w + x] = (center + strength * 0.5 * laplacian).clamp(0.0, 255.0);
                }
                Supercorde::Arc => {
                    // Isotropic unsharp mask: enhance all directions equally
                    let detail = src[y * w + x] - blurred[y * w + x];
                    plane[y * w + x] = (src[y * w + x] + strength * detail).clamp(0.0, 255.0);
                }
            }
        }
    }
}

// ======================================================================
// Quantum collapse: Schrodinger rendering (float -> u8 with error diffusion)
// ======================================================================

/// Schrödinger collapse: converts a floating-point plane to u8 using
/// Floyd-Steinberg error diffusion. Each pixel's rounding error is
/// propagated to its neighbors, preserving perceptual gradients.
///
/// This is the "measurement" step: the continuous superposition of
/// possible values collapses into a single certain value, informed
/// by the quantum state of the neighborhood.
///
/// Applied as the LAST step before RGB output — after all reconstruction,
/// inverse PTF, and inverse GCT.
pub fn schrodinger_collapse(plane: &[f64], h: usize, w: usize) -> Vec<u8> {
    let mut buf: Vec<f64> = plane.to_vec();
    let mut out = vec![0u8; h * w];

    for y in 0..h {
        for x in 0..w {
            let old = buf[y * w + x].clamp(0.0, 255.0);
            let new = old.round().clamp(0.0, 255.0);
            let error = old - new;
            out[y * w + x] = new as u8;

            // Floyd-Steinberg diffusion: propagate error to 4 neighbors
            //   [·] [7]
            // [3] [5] [1]  (divided by 16)
            if x + 1 < w {
                buf[y * w + x + 1] += error * 7.0 / 16.0;
            }
            if y + 1 < h {
                if x > 0 {
                    buf[(y + 1) * w + x - 1] += error * 3.0 / 16.0;
                }
                buf[(y + 1) * w + x] += error * 5.0 / 16.0;
                if x + 1 < w {
                    buf[(y + 1) * w + x + 1] += error * 1.0 / 16.0;
                }
            }
        }
    }
    out
}

// ======================================================================
// LL-Guided Anti-Ringing Filter (spatial domain, post-recomposition)
// ======================================================================

/// Uses the LL subband (upsampled to full resolution) as a "ground truth"
/// low-frequency reference. Where the reconstruction oscillates around the
/// LL value, it's ringing → pull toward LL. Where it deviates consistently,
/// it's real detail → keep.
///
/// This directly targets CDF 9/7 sidelobes that form DURING reconstruction
/// as interference between quantized coefficients at different levels.
pub fn ll_guided_deringing(
    plane: &mut [f64],
    ll_decoded: &[f64], ll_h: usize, ll_w: usize,
    h: usize, w: usize,
    detail_step: f64,
) {
    if h < 4 || w < 4 || ll_h == 0 || ll_w == 0 { return; }

    // Upsample LL to full resolution (bilinear)
    let mut ll_full = vec![0.0f64; h * w];
    let sy = if h > 1 { (ll_h as f64 - 1.0) / (h as f64 - 1.0) } else { 0.0 };
    let sx = if w > 1 { (ll_w as f64 - 1.0) / (w as f64 - 1.0) } else { 0.0 };
    for i in 0..h {
        let fi = (i as f64 * sy).min((ll_h - 1) as f64);
        let i0 = fi as usize;
        let i1 = (i0 + 1).min(ll_h - 1);
        let di = fi - i0 as f64;
        for j in 0..w {
            let fj = (j as f64 * sx).min((ll_w - 1) as f64);
            let j0 = fj as usize;
            let j1 = (j0 + 1).min(ll_w - 1);
            let dj = fj - j0 as f64;
            ll_full[i * w + j] =
                ll_decoded[i0 * ll_w + j0] * (1.0 - di) * (1.0 - dj)
              + ll_decoded[i0 * ll_w + j1] * (1.0 - di) * dj
              + ll_decoded[i1 * ll_w + j0] * di * (1.0 - dj)
              + ll_decoded[i1 * ll_w + j1] * di * dj;
        }
    }

    // For each pixel: compare reconstruction with LL reference.
    // The "detail" is recon - LL. Where this detail oscillates locally
    // (sign changes with neighbors), it's ringing → attenuate.
    // Where it's consistent, it's real structure → keep.

    // Self-guided deringing: use Gaussian-blurred reconstruction as reference
    // (not the coarse LL). The blur removes ringing oscillations while
    // preserving the local structure that the LL misses.
    //
    // Where the original reconstruction deviates from the blurred version
    // AND the LL is smooth → it's ringing → blend toward blurred.
    // Where the LL has edges → keep reconstruction (real detail).
    let src = plane.to_vec();
    let blurred = gaussian_blur(&src, h, w); // sigma=1, radius=4

    // LL gradient: edge detector at the LL scale
    let mut ll_edge = vec![0.0f64; h * w];
    for i in 1..h-1 {
        for j in 1..w-1 {
            let gy = (ll_full[(i+1)*w+j] - ll_full[(i-1)*w+j]).abs();
            let gx = (ll_full[i*w+j+1] - ll_full[i*w+j-1]).abs();
            ll_edge[i*w+j] = gy.max(gx);
        }
    }

    for i in 0..h {
        for j in 0..w {
            let idx = i * w + j;

            // Near LL edges: don't touch (real structure)
            if ll_edge[idx] > detail_step * 0.4 { continue; }

            // Oscillation = difference between reconstruction and blur
            let oscillation = (src[idx] - blurred[idx]).abs();
            if oscillation < 0.5 { continue; } // no oscillation

            // Blend strength: proportional to oscillation, bounded by step
            let max_osc = detail_step * 0.5;
            let blend = (oscillation / max_osc).clamp(0.0, 0.5); // max 50% blend
            plane[idx] = blurred[idx] * blend + src[idx] * (1.0 - blend);
        }
    }
}

// ======================================================================
// LOT Deblocking: Gas/Solid boundary smoothing (spatial domain)
// ======================================================================

/// Biomimetic LOT deblocking filter (inspired by H.264/HEVC loop filter).
/// Targets ONLY the block grid (every block_size pixels).
/// At each grid line: checks if the step across the boundary is an artifact
/// (interior of both blocks is smooth) vs a real edge (interior is NOT smooth).
/// Uses [1,2,1]/4 kernel to sew boundary pixels when artifact is detected.
/// Works on any plane (L, C1, C2).
pub fn deblock_lot_grid(plane: &mut [f64], h: usize, w: usize, block_size: usize) {
    if h < block_size * 2 || w < block_size * 2 { return; }

    let src = plane.to_vec();

    // Vertical grid lines (x = k * block_size)
    for gx in 1..(w / block_size) {
        let bx = gx * block_size;
        for y in 0..h {
            if bx < 2 || bx + 1 >= w { continue; }

            // Step at boundary
            let p1 = src[y * w + bx - 1]; // left of boundary
            let p2 = src[y * w + bx];     // right of boundary
            let boundary_step = (p2 - p1).abs();

            // Interior smoothness: compare with steps INSIDE each block
            let inner_left = if bx >= 2 {
                (src[y * w + bx - 1] - src[y * w + bx - 2]).abs()
            } else { boundary_step };
            let inner_right = if bx + 1 < w {
                (src[y * w + bx + 1] - src[y * w + bx]).abs()
            } else { boundary_step };

            // Artifact = boundary step is LARGER than interior steps
            // (real edge: interior steps are also large)
            let interior_max = inner_left.max(inner_right);
            let is_gas = interior_max < 4.0; // Increased to catch noisy sky
            let is_gas_artifact = is_gas && boundary_step > interior_max;
            let is_solid_artifact = !is_gas && boundary_step > interior_max * 1.5 + 1.0;

            if is_gas_artifact || is_solid_artifact {
                if is_gas {
                    // Smooth biomimetic deblocking for Gas (smooth regions like sky)
                    // We use a cosine window to distribute the step, avoiding linear Mach bands
                    let radius = block_size / 2;
                    let p0 = src[y * w + bx - 1];
                    let q0 = src[y * w + bx];
                    let diff = q0 - p0;

                    for r in 1..=radius {
                        if bx >= r && bx + r - 1 < w {
                            let t = (r as f64 - 0.5) / (radius as f64);
                            let weight = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
                            let offset = (diff / 2.0) * weight;

                            plane[y * w + bx - r] = src[y * w + bx - r] + offset;
                            plane[y * w + bx + r - 1] = src[y * w + bx + r - 1] - offset;
                        }
                    }
                } else {
                    // Sew with [1, 2, 1] / 4 kernel
                    if bx >= 1 && bx + 1 < w {
                        let a = src[y * w + bx - 1];
                        let b = src[y * w + bx];
                        // Weighted blend: pull both sides toward the midpoint
                        plane[y * w + bx - 1] = (a * 3.0 + b) / 4.0;
                        plane[y * w + bx]     = (a + b * 3.0) / 4.0;
                    }
                    // Extended sewing: 2 pixels on each side
                    if bx >= 2 && bx + 2 < w {
                        let a2 = src[y * w + bx - 2];
                        let b2 = src[y * w + bx + 1];
                        let mid = (src[y * w + bx - 1] + src[y * w + bx]) / 2.0;
                        plane[y * w + bx - 2] = (a2 * 3.0 + mid) / 4.0;
                        plane[y * w + bx + 1] = (b2 * 3.0 + mid) / 4.0;
                    }
                }
            }
        }
    }

    // Horizontal grid lines (y = k * block_size)
    let src2 = plane.to_vec();
    for gy in 1..(h / block_size) {
        let by = gy * block_size;
        for x in 0..w {
            if by < 2 || by + 1 >= h { continue; }

            let p1 = src2[(by - 1) * w + x];
            let p2 = src2[by * w + x];
            let boundary_step = (p2 - p1).abs();

            let inner_top = if by >= 2 {
                (src2[(by - 1) * w + x] - src2[(by - 2) * w + x]).abs()
            } else { boundary_step };
            let inner_bot = if by + 1 < h {
                (src2[(by + 1) * w + x] - src2[by * w + x]).abs()
            } else { boundary_step };

            let interior_max = inner_top.max(inner_bot);
            let is_gas = interior_max < 4.0;
            let is_gas_artifact = is_gas && boundary_step > interior_max;
            let is_solid_artifact = !is_gas && boundary_step > interior_max * 1.5 + 1.0;

            if is_gas_artifact || is_solid_artifact {
                if is_gas {
                    let radius = block_size / 2;
                    let p0 = src2[(by - 1) * w + x];
                    let q0 = src2[by * w + x];
                    let diff = q0 - p0;

                    for r in 1..=radius {
                        if by >= r && by + r - 1 < h {
                            let t = (r as f64 - 0.5) / (radius as f64);
                            let weight = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
                            let offset = (diff / 2.0) * weight;

                            plane[(by - r) * w + x] = src2[(by - r) * w + x] + offset;
                            plane[(by + r - 1) * w + x] = src2[(by + r - 1) * w + x] - offset;
                        }
                    }
                } else {
                    if by >= 1 && by + 1 < h {
                        let a = src2[(by - 1) * w + x];
                        let b = src2[by * w + x];
                        plane[(by - 1) * w + x] = (a * 3.0 + b) / 4.0;
                        plane[by * w + x]       = (a + b * 3.0) / 4.0;
                    }
                    if by >= 2 && by + 2 < h {
                        let a2 = src2[(by - 2) * w + x];
                        let b2 = src2[(by + 1) * w + x];
                        let mid = (src2[(by - 1) * w + x] + src2[by * w + x]) / 2.0;
                        plane[(by - 2) * w + x] = (a2 * 3.0 + mid) / 4.0;
                        plane[(by + 1) * w + x] = (b2 * 3.0 + mid) / 4.0;
                    }
                }
            }
        }
    }
}

// ======================================================================
// Velvety Gas Filter (Spatial Domain 2D Smoothing)
// ======================================================================

/// Applies an intense edge-preserving 2D smooth to completely melt 
/// LOT macroblocks in the "gas" (smooth/sky) regions.
pub fn velvet_gas_filter(plane: &mut [f64], h: usize, w: usize, block_size: usize, strength: f64) {
    if h < 2 || w < 2 || strength < 0.01 { return; }

    // Scene-adaptive: strength scales radius and permissiveness
    let radius = ((block_size as f64 / 2.0) * strength).round().clamp(2.0, block_size as f64) as usize;
    let threshold = (3.0 / strength.max(0.1)).clamp(2.0, 10.0);

    let mut temp = vec![0.0; h * w];

    // Single pass — no multi-pass blur (destroys texture)
    for _pass in 0..1 {
        // Horizontal pass
        for y in 0..h {
            for x in 0..w {
                let mut sum = plane[y * w + x];
                let mut weight = 1.0;
                let center_val = plane[y * w + x];

                // Right
                for r in 1..=radius {
                    if x + r >= w { break; }
                    let val = plane[y * w + (x + r)];
                    if (val - center_val).abs() > threshold { break; }
                    sum += val;
                    weight += 1.0;
                }
                // Left
                for r in 1..=radius {
                    if x < r { break; }
                    let val = plane[y * w + (x - r)];
                    if (val - center_val).abs() > threshold { break; }
                    sum += val;
                    weight += 1.0;
                }
                temp[y * w + x] = sum / weight;
            }
        }

        // Vertical pass
        for x in 0..w {
            for y in 0..h {
                let mut sum = temp[y * w + x];
                let mut weight = 1.0;
                let center_val = temp[y * w + x];

                // Down
                for r in 1..=radius {
                    if y + r >= h { break; }
                    let val = temp[(y + r) * w + x];
                    if (val - center_val).abs() > threshold { break; }
                    sum += val;
                    weight += 1.0;
                }
                // Up
                for r in 1..=radius {
                    if y < r { break; }
                    let val = temp[(y - r) * w + x];
                    if (val - center_val).abs() > threshold { break; }
                    sum += val;
                    weight += 1.0;
                }
                plane[y * w + x] = sum / weight;
            }
        }
    }
}

// ======================================================================
// Anti-ringing Sigma Filter (Lee 1983)
// ======================================================================

/// Anti-ringing sigma filter: smooths wavelet halos near edges.
/// Only averages neighbors whose value is close (within +-sigma_t).
/// Ringing oscillations are excluded because they exceed the threshold.
pub fn anti_ring_sigma(plane: &mut [f64], h: usize, w: usize, step: f64) {
    const EDGE_THRESH: f64 = 6.0;  // more sensitive edge detection
    const PAD: usize = 3;          // wider window: CDF 9/7 sidelobes extend ~8px

    if step < 2.0 || h < 7 || w < 7 {
        return;
    }

    // Base sigma_t, will be modulated by local luminance
    let sigma_t = (step * 0.5).clamp(2.0, 12.0);
    let n = h * w;

    // --- L-inf gradient ---
    let mut grad = vec![0.0f64; n];
    for r in 0..h {
        for c in 0..w {
            let idx = r * w + c;
            let gy = if r + 1 < h {
                (plane[(r + 1) * w + c] - plane[idx]).abs()
            } else {
                0.0
            };
            let gx = if c + 1 < w {
                (plane[r * w + c + 1] - plane[idx]).abs()
            } else {
                0.0
            };
            grad[idx] = gy.max(gx);
        }
    }

    // --- 5x5 box-max dilation + thresholding -> edge_zone mask ---
    let mut edge_zone = vec![false; n];
    let mut any_edge = false;
    for r in 0..h {
        for c in 0..w {
            let mut mx = 0.0f64;
            let r_lo = if r >= PAD { r - PAD } else { 0 };
            let r_hi = (r + PAD).min(h - 1);
            let c_lo = if c >= PAD { c - PAD } else { 0 };
            let c_hi = (c + PAD).min(w - 1);
            for rr in r_lo..=r_hi {
                for cc in c_lo..=c_hi {
                    let g = grad[rr * w + cc];
                    if g > mx { mx = g; }
                }
            }
            if mx > EDGE_THRESH {
                edge_zone[r * w + c] = true;
                any_edge = true;
            }
        }
    }

    if !any_edge {
        return;
    }

    // --- 5x5 sigma filter, parallelized per row ---
    let src = plane.to_vec(); // snapshot before filtering
    let ih = h as i32;
    let iw = w as i32;
    let ipad = PAD as i32;

    plane
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(r, row)| {
            let ir = r as i32;
            for c in 0..w {
                if !edge_zone[r * w + c] {
                    continue;
                }
                let center = src[r * w + c];

                // Luminance-adaptive sigma: tighter in dark zones (Weber)
                let local_sigma = if center < 60.0 {
                    sigma_t * 0.4  // dark: very tight
                } else if center < 120.0 {
                    sigma_t * 0.7  // mid-dark
                } else {
                    sigma_t        // normal
                };

                let mut total = 0.0f64;
                let mut count = 0u32;

                for dy in -ipad..=ipad {
                    let rr = reflect(ir + dy, ih);
                    for dx in -ipad..=ipad {
                        let cc = reflect(c as i32 + dx, iw);
                        let val = src[rr * w + cc];
                        if (val - center).abs() <= local_sigma {
                            total += val;
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    row[c] = total / count as f64;
                }
            }
        });
}

// ======================================================================
// Gas-only deblocking: chirurgical, PSNR-preserving
// ======================================================================

/// Lightweight gas-only deblocking. Only touches the 2 pixels at each
/// block boundary where BOTH sides are smooth (gas phase).
/// Uses [1,2,1]/4 kernel — the minimum intervention to erase the grid.
///
/// Gas detection: average gradient in a 4-pixel corridor on each side
/// of the boundary. If both sides have gradient < threshold, it's gas.
///
/// Unlike deblock_lot_grid (which uses cosine window over 8px, -0.59 dB),
/// this touches only the immediate boundary pixels (≤ -0.1 dB target).
pub fn deblock_gas_only(plane: &mut [f64], h: usize, w: usize, block_size: usize) {
    if h < block_size * 2 || w < block_size * 2 { return; }

    // Gas threshold: pixels with gradient below this are "gas" (smooth)
    const GAS_THRESH: f64 = 2.5;
    // Blend strength: phi_inv² ≈ 0.382 (golden attenuation)
    const BLEND: f64 = 0.382;

    let src = plane.to_vec();

    // Vertical grid lines (x = k * block_size)
    for gx in 1..(w / block_size) {
        let bx = gx * block_size;
        if bx < 2 || bx + 1 >= w { continue; }

        for y in 0..h {
            let p_left  = src[y * w + bx - 1];
            let p_right = src[y * w + bx];
            let boundary_step = (p_right - p_left).abs();

            // Check interior smoothness (2 pixels on each side)
            let grad_left = if bx >= 2 {
                (src[y * w + bx - 1] - src[y * w + bx - 2]).abs()
            } else { boundary_step };
            let grad_right = if bx + 2 < w {
                (src[y * w + bx + 1] - src[y * w + bx]).abs()
            } else { boundary_step };

            // Gas: both sides smooth AND boundary step is an artifact
            let is_gas = grad_left < GAS_THRESH && grad_right < GAS_THRESH
                         && boundary_step > grad_left.max(grad_right);

            if is_gas {
                // Weber-adaptive blend: stronger in dark areas where blocking is more visible.
                // Dark (pixel~20): blend ≈ 0.85 (aggressive). Bright (pixel~200): blend ≈ 0.38.
                let lum = (p_left.abs() + p_right.abs()) * 0.5;
                let weber_blend = (BLEND + (1.0 - BLEND) * (1.0 - lum / 255.0)).min(0.95);
                let mid = (p_left + p_right) * 0.5;
                plane[y * w + bx - 1] = p_left  + weber_blend * (mid - p_left);
                plane[y * w + bx]     = p_right + weber_blend * (mid - p_right);
            }
        }
    }

    // Horizontal grid lines (y = k * block_size)
    let src2 = plane.to_vec(); // refresh after vertical pass
    for gy in 1..(h / block_size) {
        let by = gy * block_size;
        if by < 2 || by + 1 >= h { continue; }

        for x in 0..w {
            let p_top    = src2[(by - 1) * w + x];
            let p_bottom = src2[by * w + x];
            let boundary_step = (p_bottom - p_top).abs();

            let grad_top = if by >= 2 {
                (src2[(by - 1) * w + x] - src2[(by - 2) * w + x]).abs()
            } else { boundary_step };
            let grad_bot = if by + 2 < h {
                (src2[(by + 1) * w + x] - src2[by * w + x]).abs()
            } else { boundary_step };

            let is_gas = grad_top < GAS_THRESH && grad_bot < GAS_THRESH
                         && boundary_step > grad_top.max(grad_bot);

            if is_gas {
                let lum = (p_top.abs() + p_bottom.abs()) * 0.5;
                let weber_blend = (BLEND + (1.0 - BLEND) * (1.0 - lum / 255.0)).min(0.95);
                let mid = (p_top + p_bottom) * 0.5;
                plane[(by - 1) * w + x] = p_top    + weber_blend * (mid - p_top);
                plane[by * w + x]       = p_bottom + weber_blend * (mid - p_bottom);
            }
        }
    }
}

// ==========================================================================
// Post-decode sharpening (CASP + Adaptive)
// ==========================================================================

/// CASP (Contrast Adaptive Sharpening) — unsharp mask with Gaussian blur,
/// edge-adaptive strength to prevent halos.
///
/// sigma: blur radius in pixels (1.5-3.0 for HD images).
/// strength: sharpening amount (0.5 = subtle, 1.0 = visible, 2.0 = aggressive).
pub fn casp_sharpen(plane: &mut [f64], h: usize, w: usize, strength: f64) {
    if strength <= 0.0 || h < 3 || w < 3 {
        return;
    }
    let src = plane.to_vec();

    // Gaussian blur radius scales with image size for visible effect
    let sigma = if h.max(w) > 2000 { 2.0 } else if h.max(w) > 1000 { 1.5 } else { 1.0 };
    let blurred = gaussian_blur_2d(&src, h, w, sigma);

    // Sobel edge magnitude for adaptive control
    let mut edge = vec![0.0f64; h * w];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = (src[(y - 1) * w + x + 1] + 2.0 * src[y * w + x + 1] + src[(y + 1) * w + x + 1])
                   - (src[(y - 1) * w + x - 1] + 2.0 * src[y * w + x - 1] + src[(y + 1) * w + x - 1]);
            let gy = (src[(y + 1) * w + x - 1] + 2.0 * src[(y + 1) * w + x] + src[(y + 1) * w + x + 1])
                   - (src[(y - 1) * w + x - 1] + 2.0 * src[(y - 1) * w + x] + src[(y - 1) * w + x + 1]);
            edge[y * w + x] = (gx * gx + gy * gy).sqrt();
        }
    }
    let max_edge = edge.iter().cloned().fold(1.0f64, f64::max);
    let inv_max = 1.0 / max_edge;

    // Unsharp mask: output = src + strength * (src - blurred) * edge_attenuation
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let detail = src[idx] - blurred[idx];
            // Attenuate near strong edges (prevent halos), boost in smooth areas
            let edge_norm = (edge[idx] * inv_max).min(1.0);
            let atten = 1.0 - edge_norm * 0.7; // retain 30% sharpening even at edges
            plane[idx] = (src[idx] + strength * atten * detail).clamp(0.0, 255.0);
        }
    }
}

/// Separable Gaussian blur for dsp module (self-contained).
fn gaussian_blur_2d(input: &[f64], h: usize, w: usize, sigma: f64) -> Vec<f64> {
    if sigma < 0.01 { return input.to_vec(); }
    let radius = (3.0 * sigma).ceil() as usize;
    let ks = 2 * radius + 1;

    // Build normalized 1D kernel
    let inv_2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut kernel = Vec::with_capacity(ks);
    let mut sum = 0.0;
    for i in 0..ks {
        let d = i as f64 - radius as f64;
        let v = (-d * d * inv_2s2).exp();
        kernel.push(v);
        sum += v;
    }
    for k in &mut kernel { *k /= sum; }

    // Horizontal pass
    let mut temp = vec![0.0; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for k in 0..ks {
                let xx = (x as i64 + k as i64 - radius as i64).clamp(0, w as i64 - 1) as usize;
                acc += kernel[k] * input[y * w + xx];
            }
            temp[y * w + x] = acc;
        }
    }

    // Vertical pass
    let mut output = vec![0.0; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for k in 0..ks {
                let yy = (y as i64 + k as i64 - radius as i64).clamp(0, h as i64 - 1) as usize;
                acc += kernel[k] * temp[yy * w + x];
            }
            output[y * w + x] = acc;
        }
    }
    output
}

/// Adaptive sharpening: edge-aware unsharp mask.
///
/// Computes a Sobel edge mask, then applies an unsharp mask whose strength
/// is *reduced* near strong edges (avoids halo/ringing) and *increased*
/// in smooth areas (recovers lost detail from quantization).
pub fn adaptive_sharpen(plane: &mut [f64], h: usize, w: usize, strength: f64, _edge_threshold: f64) {
    if strength <= 0.0 || h < 3 || w < 3 {
        return;
    }
    let src = plane.to_vec();

    // 1. Sobel edge magnitude
    let mut edge = vec![0.0f64; h * w];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = (src[(y - 1) * w + x + 1] + 2.0 * src[y * w + x + 1] + src[(y + 1) * w + x + 1])
                   - (src[(y - 1) * w + x - 1] + 2.0 * src[y * w + x - 1] + src[(y + 1) * w + x - 1]);
            let gy = (src[(y + 1) * w + x - 1] + 2.0 * src[(y + 1) * w + x] + src[(y + 1) * w + x + 1])
                   - (src[(y - 1) * w + x - 1] + 2.0 * src[(y - 1) * w + x] + src[(y - 1) * w + x + 1]);
            edge[y * w + x] = (gx * gx + gy * gy).sqrt();
        }
    }

    // Normalize edge map to [0, 1]
    let max_edge = edge.iter().cloned().fold(0.0f64, f64::max);
    if max_edge > 0.0 {
        let inv = 1.0 / max_edge;
        for e in edge.iter_mut() {
            *e *= inv;
        }
    }

    // 2. Blur the edge mask (simple 3×3 box blur for speed)
    let mut edge_blur = vec![0.0f64; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            let mut cnt = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let ny = y as i32 + dy;
                    let nx = x as i32 + dx;
                    if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                        sum += edge[ny as usize * w + nx as usize];
                        cnt += 1.0;
                    }
                }
            }
            edge_blur[y * w + x] = sum / cnt;
        }
    }

    // 3. Compute low-pass (3×3 box blur of source)
    let mut blurred = vec![0.0f64; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            let mut cnt = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let ny = y as i32 + dy;
                    let nx = x as i32 + dx;
                    if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                        sum += src[ny as usize * w + nx as usize];
                        cnt += 1.0;
                    }
                }
            }
            blurred[y * w + x] = sum / cnt;
        }
    }

    // 4. Unsharp mask modulated by inverse edge strength
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let detail = src[idx] - blurred[idx];
            // More sharpening in smooth areas, less at edges
            let factor = strength * (1.0 - edge_blur[idx]);
            plane[idx] = (src[idx] + factor * detail).clamp(0.0, 255.0);
        }
    }
}

/// Apply post-decode sharpening based on mode.
/// mode: 0=off, 1=unsharp (directional_sharpen), 2=CASP, 3=adaptive
pub fn post_sharpen(plane: &mut [f64], h: usize, w: usize, mode: u8, strength: f64) {
    match mode {
        1 => directional_sharpen(plane, h, w, strength),
        2 => casp_sharpen(plane, h, w, strength),
        3 => adaptive_sharpen(plane, h, w, strength, 0.2),
        _ => {} // 0 or unknown = off
    }
}

