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
// Anti-ringing Sigma Filter (Lee 1983)
// ======================================================================

/// Anti-ringing sigma filter: smooths wavelet halos near edges.
/// Only averages neighbors whose value is close (within +-sigma_t).
/// Ringing oscillations are excluded because they exceed the threshold.
pub fn anti_ring_sigma(plane: &mut [f64], h: usize, w: usize, step: f64) {
    const EDGE_THRESH: f64 = 8.0;
    const PAD: usize = 2;

    if step < 2.0 || h < 5 || w < 5 {
        return;
    }

    let sigma_t = (step * 0.5).clamp(3.0, 12.0);
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
                let mut total = 0.0f64;
                let mut count = 0u32;

                for dy in -ipad..=ipad {
                    let rr = reflect(ir + dy, ih);
                    for dx in -ipad..=ipad {
                        let cc = reflect(c as i32 + dx, iw);
                        let val = src[rr * w + cc];
                        if (val - center).abs() <= sigma_t {
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
