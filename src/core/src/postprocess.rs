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
    let mut output = input.to_vec();
    let strength = strength.clamp(0.0, 1.0);

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut min_val: f32 = f32::MAX;
            let mut max_val: f32 = f32::MIN;

            // Analyze 3x3 neighborhood (equivalent to max/min x[-1,-1] to x[1,1] in Avisynth)
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let idx = ((y as isize + dy) as usize) * width + ((x as isize + dx) as usize);
                    let val = input[idx];
                    if val < min_val { min_val = val; }
                    if val > max_val { max_val = val; }
                }
            }

            let center_idx = y * width + x;
            let center = input[center_idx];

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
                // Apply strength
                output[center_idx] = center + strength * (target - center);
            }
        }
    }
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

            let filtered = (top * w + bottom * w + left * w + right * w + center)
                         / (4.0 * w + 1.0);

            output[idx] = filtered.clamp(0.0, 1.0);
        }
    }
    output
}

/// Inspired by "SPresso" (Spatial Pressdown).
/// Best used BEFORE encoding (in aurea_encoder.rs).
/// Applies local smoothing that suppresses chaotic micro-details
/// to greatly facilitate compression without blurring edges (edge-preserving).
pub fn apply_spresso_prefilter(
    input: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
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
            if diff < 0.1 { // Tolerance threshold to preserve strong edges
                output[y * width + x] = median * 0.5 + center * 0.5;
            }
        }
    }
    output
}
