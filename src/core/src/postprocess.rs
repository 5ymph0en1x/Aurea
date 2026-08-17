//! Post-processing and pre-processing filters inspired by Avisynth.
//! Contains filters to remove the "blurry haze" from post-compression
//! and optimize entropy for pre-compression.

/// Applies an "XSharpen" filter based on the XSharpenPlus Avisynth script.
/// Ideal for removing compression haze by pushing pixels close to a local
/// extremum toward the local minimum or maximum (non-linear sharpening).
///
/// * `input`: Image plane (ideally luminance/luma) normalized between 0.0 and 1.0
/// * `strength`: Filter strength (e.g. 0.5 for 50%, corresponding to `str / 256.` in Avisynth)
/// * `threshold`: Edge detection threshold (e.g. 0.03)
pub fn apply_xsharpen(
    input: &[f32],
    width: usize,
    height: usize,
    strength: f32,
    threshold: f32,
) -> Vec<f32> {
    apply_xsharpen_radius(input, width, height, strength, threshold, 1)
}

/// XSharpen with a configurable square analysis radius. Radius 1 is the
/// historical 3x3 filter; radius 2 pushes toward 5x5 extrema, which restores
/// coarser texture punch at low qualities.
pub fn apply_xsharpen_radius(
    input: &[f32],
    width: usize,
    height: usize,
    strength: f32,
    threshold: f32,
    radius: usize,
) -> Vec<f32> {
    apply_xsharpen_gated(input, width, height, strength, threshold, radius, 0.0)
}

/// XSharpen with a local-contrast floor: pixels whose neighbourhood range
/// `max - min` is below `min_range` are left untouched. On gentle gradients
/// every pixel sits mid-way between very close extrema, so the unfloored
/// filter quantizes the ramp into visible banding — the exact failure
/// measured on `chilling`. Real texture and edges carry range far above the
/// floor and keep the full restoration.
pub fn apply_xsharpen_gated(
    input: &[f32],
    width: usize,
    height: usize,
    strength: f32,
    threshold: f32,
    radius: usize,
    min_range: f32,
) -> Vec<f32> {
    use rayon::prelude::*;

    let mut output = input.to_vec();
    let strength = strength.clamp(0.0, 1.0);
    let radius = radius.clamp(1, 4);
    if height <= 2 * radius || width <= 2 * radius {
        return output;
    }

    // Separable neighbourhood extrema: horizontal min/max per row, then a
    // vertical combine. min/max over the same value set is exact regardless
    // of visiting order (and ±0.0 representatives are arithmetically
    // indistinguishable downstream), so this is bit-identical to the direct
    // (2r+1)² window while doing O(2r+1) work per pixel — and every row is
    // independent, so both passes parallelize.
    let n = input.len();
    let mut row_min = vec![f32::MAX; n];
    let mut row_max = vec![f32::MIN; n];
    row_min
        .par_chunks_mut(width)
        .zip(row_max.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (rmin, rmax))| {
            let base = y * width;
            for x in radius..(width - radius) {
                let mut mn = f32::MAX;
                let mut mx = f32::MIN;
                for v in &input[base + x - radius..=base + x + radius] {
                    if *v < mn {
                        mn = *v;
                    }
                    if *v > mx {
                        mx = *v;
                    }
                }
                rmin[x] = mn;
                rmax[x] = mx;
            }
        });

    output.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        if y < radius || y >= height - radius {
            return;
        }
        for x in radius..(width - radius) {
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            for dy in (y - radius)..=(y + radius) {
                let v1 = row_min[dy * width + x];
                if v1 < min_val {
                    min_val = v1;
                }
                let v2 = row_max[dy * width + x];
                if v2 > max_val {
                    max_val = v2;
                }
            }
            if max_val - min_val < min_range {
                continue;
            }

            let center = input[y * width + x];

            // Distances to local extrema (X and Y in the Avisynth script)
            let dist_to_min = center - min_val;
            let dist_to_max = max_val - center;
            let min_dist = dist_to_min.min(dist_to_max);

            // Avisynth ternary operator: {thr} < X Y < N M ? x - {str} * x + x ?
            // If the pixel is close enough to an extremum (below threshold)
            if min_dist < threshold {
                let target = if dist_to_min < dist_to_max {
                    min_val
                } else {
                    max_val
                };
                row[x] = center + strength * (target - center);
            }
        }
    });
    output
}

/// Applies CAS (Contrast Adaptive Sharpening).
/// Often the best choice for removing a global blurry haze without
/// destroying the image with noise (overshoot/ringing).
///
/// * `sharpness`: Between 0.0 (no effect) and 1.0 (maximum effect)
pub fn apply_cas_sharpening(
    input: &[f32],
    width: usize,
    height: usize,
    sharpness: f32,
) -> Vec<f32> {
    // Initialize output as a copy of input — border pixels pass through unchanged.
    let mut output = input.to_vec();

    // Standard CAS weight typically varies around -0.125
    let sharp_weight = -0.125 * sharpness.clamp(0.0, 1.0);

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = y * width + x;

            // Cross pattern
            let top = input[idx - width];
            let bottom = input[idx + width];
            let left = input[idx - 1];
            let right = input[idx + 1];
            let center = input[idx];

            let min_val = top.min(bottom).min(left).min(right).min(center);
            let max_val = top.max(bottom).max(left).max(right).max(center);

            // Adjust weight based on local contrast
            let d_min = min_val;
            let d_max = 1.0 - max_val; // Assuming 0.0-1.0 range
            let amp_limit = d_min.min(d_max);

            let mut w = 0.0;
            if max_val > 0.0 {
                w = (amp_limit / max_val).sqrt() * sharp_weight;
            }

            let filtered = (top * w + bottom * w + left * w + right * w + center) / (4.0 * w + 1.0);

            output[idx] = filtered.clamp(0.0, 1.0);
        }
    }
    output
}

/// Inspired by "SPresso" (Spatial Pressdown).
/// Best used BEFORE encoding (in aurea_encoder.rs).
/// Applies local smoothing that suppresses chaotic micro-details
/// to greatly facilitate compression without blurring edges (edge-preserving).
pub fn apply_spresso_prefilter(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut output = input.to_vec();
    let mut window = [0.0; 9];

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut k = 0;
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let idx = ((y as isize + dy) as usize) * width + ((x as isize + dx) as usize);
                    window[k] = input[idx];
                    k += 1;
                }
            }

            // Sort to find median (pseudo-minblur used by SPresso)
            window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = window[4];
            let center = input[y * width + x];

            // Conservative blend: gently push the pixel toward the median
            // only if local variance is low (noise).
            let diff = (center - median).abs();
            if diff < 0.1 {
                // Tolerance threshold to preserve strong edges
                output[y * width + x] = median * 0.5 + center * 0.5;
            }
        }
    }
    output
}
