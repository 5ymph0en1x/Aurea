/// Scene analysis from geometric primitives.
/// The supercordes (segments + arcs) extracted by the encoder are a structural
/// radiograph of the image. Their statistics reveal the geometric nature of the scene.

use crate::geometric::Primitive;
use std::f64::consts::PI;

/// Quick scene profile derived from the DC grid (low-res thumbnail).
/// Used by the decoder to adapt filter strength — no bitstream overhead.
#[derive(Debug)]
pub struct DcSceneProfile {
    pub scene_type: SceneType,
    pub smooth_pct: f64,    // % of DC grid that is "smooth" (low gradient)
    pub anisotropy: f64,    // orientation anisotropy of DC gradients
    pub velvet_strength: f64, // recommended velvet filter strength [0..2]
}

/// Analyze the decoded DC grid to derive scene characteristics.
/// The DC grid is a low-res version of the L channel (~1/16th resolution).
pub fn analyze_dc_grid(dc: &[f64], gh: usize, gw: usize) -> DcSceneProfile {
    let n = gh * gw;
    if n < 4 {
        return DcSceneProfile {
            scene_type: SceneType::Flat, smooth_pct: 100.0,
            anisotropy: 0.0, velvet_strength: 1.0,
        };
    }

    // Gradient analysis on DC grid
    let mut grad_h = vec![0.0f64; n]; // horizontal gradient
    let mut grad_v = vec![0.0f64; n]; // vertical gradient
    let mut mag = vec![0.0f64; n];

    for y in 0..gh {
        for x in 0..gw {
            let gx = if x + 1 < gw { dc[y * gw + x + 1] - dc[y * gw + x] } else { 0.0 };
            let gy = if y + 1 < gh { dc[(y + 1) * gw + x] - dc[y * gw + x] } else { 0.0 };
            grad_h[y * gw + x] = gx;
            grad_v[y * gw + x] = gy;
            mag[y * gw + x] = (gx * gx + gy * gy).sqrt();
        }
    }

    // Smooth percentage (low gradient = uniform region = gas)
    let smooth_count = mag.iter().filter(|&&m| m < 3.0).count();
    let smooth_pct = 100.0 * smooth_count as f64 / n as f64;

    // Orientation histogram (4 bins: H, diag1, V, diag2)
    let mut hist = [0.0f64; 4];
    for i in 0..n {
        if mag[i] < 1.0 { continue; } // skip flat areas
        let angle = grad_v[i].atan2(grad_h[i]).rem_euclid(PI);
        let bin = ((angle / PI * 4.0) as usize).min(3);
        hist[bin] += mag[i];
    }
    let hist_sum: f64 = hist.iter().sum();
    if hist_sum > 0.0 { for v in hist.iter_mut() { *v /= hist_sum; } }

    // Anisotropy
    let max_entropy = (4.0f64).ln();
    let entropy: f64 = hist.iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| -v * v.ln())
        .sum();
    let anisotropy = 1.0 - entropy / max_entropy;

    // H/V dominance
    let hv_ratio = hist[0] + hist[2]; // horizontal + vertical

    // Scene classification
    let scene_type = if smooth_pct > 50.0 {
        SceneType::Flat
    } else if hv_ratio > 0.6 && anisotropy > 0.1 {
        SceneType::Architectural
    } else if anisotropy > 0.15 {
        SceneType::Perspective
    } else if smooth_pct < 15.0 {
        SceneType::Organic
    } else {
        SceneType::Mixed
    };

    // Velvet strength: adapt to scene type
    // Flat/smooth → strong velvet (lots of gas to smooth)
    // Organic/textured → weak velvet (texture masks blocking)
    // Architectural → medium velvet (straight lines need care)
    let velvet_strength = match scene_type {
        SceneType::Flat => 1.5,
        SceneType::Architectural => 1.0,
        SceneType::Perspective => 0.9,
        SceneType::Mixed => 0.8,
        SceneType::Organic => 0.5,
    };

    DcSceneProfile {
        scene_type, smooth_pct, anisotropy, velvet_strength,
    }
}

/// Geometric profile of an image, derived from its primitives.
#[derive(Debug, Clone)]
pub struct SceneProfile {
    /// Total number of primitives
    pub n_primitives: usize,
    /// Number of segments vs arcs
    pub n_segments: usize,
    pub n_arcs: usize,
    /// Segment/arc ratio: >1 = architectural/geometric, <1 = organic/natural
    pub segment_arc_ratio: f64,

    /// Dominant orientations (histogram of segment angles, 8 bins of 22.5°)
    pub orientation_histogram: [f64; 8],
    /// Anisotropy: 0 = isotropic (no dominant direction), 1 = strongly directional
    pub anisotropy: f64,
    /// Dominant angle (radians, 0 = horizontal)
    pub dominant_angle: f64,

    /// Perspective score: 0 = no perspective, 1 = strong perspective
    /// Detected by convergence of segment directions toward vanishing points
    pub perspective_score: f64,

    /// Average primitive length (in pixels)
    pub avg_length: f64,
    /// Primitive density (primitives per 1000 pixels²)
    pub density: f64,

    /// Scene type classification
    pub scene_type: SceneType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SceneType {
    Architectural, // dominated by straight lines, right angles
    Organic,       // dominated by curves, irregular shapes
    Perspective,   // strong vanishing point(s)
    Flat,          // few primitives, mostly smooth
    Mixed,         // balanced mix
}

/// Analyze geometric primitives to derive scene profile.
pub fn analyze_primitives(primitives: &[Primitive], img_h: usize, img_w: usize) -> SceneProfile {
    let n = primitives.len();
    if n == 0 {
        return SceneProfile {
            n_primitives: 0, n_segments: 0, n_arcs: 0,
            segment_arc_ratio: 1.0,
            orientation_histogram: [0.0; 8],
            anisotropy: 0.0, dominant_angle: 0.0,
            perspective_score: 0.0,
            avg_length: 0.0, density: 0.0,
            scene_type: SceneType::Flat,
        };
    }

    let mut n_segments = 0usize;
    let mut n_arcs = 0usize;
    let mut orientation_hist = [0.0f64; 8];
    let mut total_length = 0.0f64;
    let mut angles: Vec<f64> = Vec::with_capacity(n);

    for prim in primitives {
        match prim {
            Primitive::Segment { x1, y1, x2, y2, .. } => {
                n_segments += 1;
                let dx = *x2 as f64 - *x1 as f64;
                let dy = *y2 as f64 - *y1 as f64;
                let len = (dx * dx + dy * dy).sqrt();
                total_length += len;

                // Angle in [0, π)
                let angle = dy.atan2(dx).rem_euclid(PI);
                angles.push(angle);

                // Bin into 8 orientations (0°, 22.5°, 45°, ...)
                let bin = ((angle / PI * 8.0) as usize).min(7);
                orientation_hist[bin] += len; // weight by length
            }
            Primitive::Arc { radius, .. } => {
                n_arcs += 1;
                let len = *radius as f64 * PI * 0.5; // approximate quarter-arc length
                total_length += len;
            }
        }
    }

    let segment_arc_ratio = if n_arcs > 0 {
        n_segments as f64 / n_arcs as f64
    } else if n_segments > 0 {
        100.0 // all segments, no arcs
    } else {
        1.0
    };

    // Normalize orientation histogram
    let hist_sum: f64 = orientation_hist.iter().sum();
    if hist_sum > 0.0 {
        for v in orientation_hist.iter_mut() { *v /= hist_sum; }
    }

    // Anisotropy: entropy-based. Uniform distribution = 0, single bin = 1.
    let max_entropy = (8.0f64).ln();
    let entropy: f64 = orientation_hist.iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| -v * v.ln())
        .sum();
    let anisotropy = 1.0 - entropy / max_entropy;

    // Dominant angle
    let dominant_bin = orientation_hist.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let dominant_angle = (dominant_bin as f64 + 0.5) * PI / 8.0;

    // Perspective score: detect if segments converge toward common vanishing points
    // Simple heuristic: if many non-parallel segments exist AND they aren't H/V,
    // there's likely perspective.
    let h_v_ratio = orientation_hist[0] + orientation_hist[4]; // horizontal + vertical
    let diagonal_ratio = 1.0 - h_v_ratio;
    let perspective_score = if n_segments > 10 {
        (diagonal_ratio * anisotropy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let avg_length = if n > 0 { total_length / n as f64 } else { 0.0 };
    let area = (img_h * img_w) as f64;
    let density = n as f64 / (area / 1000.0);

    // Classify scene type
    let scene_type = if density < 0.5 {
        SceneType::Flat
    } else if perspective_score > 0.4 {
        SceneType::Perspective
    } else if segment_arc_ratio > 3.0 && anisotropy > 0.3 {
        SceneType::Architectural
    } else if segment_arc_ratio < 0.5 {
        SceneType::Organic
    } else {
        SceneType::Mixed
    };

    SceneProfile {
        n_primitives: n, n_segments, n_arcs,
        segment_arc_ratio,
        orientation_histogram: orientation_hist,
        anisotropy, dominant_angle,
        perspective_score,
        avg_length, density,
        scene_type,
    }
}

/// Derive deblocking parameters from scene profile.
/// Returns (velvet_radius_factor, velvet_threshold_factor, deblock_strength).
/// Architectural scenes need more aggressive deblocking (grid lines clash with structure).
/// Organic scenes can tolerate more blocking (hidden by texture).
pub fn scene_adaptive_params(profile: &SceneProfile) -> (f64, f64, f64) {
    match profile.scene_type {
        SceneType::Architectural => {
            // Strong deblocking: block grid aligned with architectural lines is very visible
            (1.5, 0.8, 1.5)
        }
        SceneType::Perspective => {
            // Medium-strong: perspective lines must stay straight
            (1.3, 0.9, 1.3)
        }
        SceneType::Flat => {
            // Gentle: few features, blocking barely visible
            (0.5, 1.2, 0.7)
        }
        SceneType::Organic => {
            // Gentle: curves mask blocking naturally
            (0.7, 1.0, 0.8)
        }
        SceneType::Mixed => {
            // Default balanced
            (1.0, 1.0, 1.0)
        }
    }
}

// ======================================================================
// Point 7: Encoder-side scene analysis
// ======================================================================

/// Quick scene classification from the L channel (encoder-side, Point 7).
/// Downsamples the L channel to ~1/16 resolution, classifies scene type.
/// Used BEFORE encoding to guide transform choice and parameters.
pub fn quick_scene_classify(l_channel: &[f64], h: usize, w: usize) -> DcSceneProfile {
    let step = 16usize;
    let gh = (h + step - 1) / step;
    let gw = (w + step - 1) / step;

    let mut dc_grid = Vec::with_capacity(gh * gw);
    for gy in 0..gh {
        for gx in 0..gw {
            let y0 = gy * step;
            let x0 = gx * step;
            let y1 = (y0 + step).min(h);
            let x1 = (x0 + step).min(w);
            let mut sum = 0.0;
            let mut count = 0;
            for y in y0..y1 {
                for x in x0..x1 {
                    sum += l_channel[y * w + x];
                    count += 1;
                }
            }
            dc_grid.push(sum / count.max(1) as f64);
        }
    }

    analyze_dc_grid(&dc_grid, gh, gw)
}

/// Recommend LOT vs CDF97 based on scene analysis (Point 1).
/// Returns true if LOT is recommended, false for CDF97.
///
/// LOT is better for most content (no inter-level interference).
/// CDF97 may be better for very smooth images with gradients (better energy compaction).
pub fn recommend_lot(profile: &DcSceneProfile) -> bool {
    // LOT for almost everything. CDF97 only for very flat smooth gradients.
    // Even then, LOT with deblocking is usually competitive.
    match profile.scene_type {
        SceneType::Flat => profile.smooth_pct < 85.0, // very flat → CDF97 may be better
        _ => true, // LOT for everything else
    }
}

/// Compute encoder-side parameter adjustments based on scene analysis (Point 7).
/// Returns (detail_step_factor, deblock_strength_hint).
pub fn encoder_scene_adjust(profile: &DcSceneProfile) -> (f64, f64) {
    match profile.scene_type {
        SceneType::Flat => (0.9, 1.5),          // flat: can quantize slightly finer, strong deblock
        SceneType::Architectural => (1.0, 1.0),  // architectural: standard
        SceneType::Perspective => (0.95, 0.9),    // perspective: slightly finer for lines
        SceneType::Organic => (1.1, 0.5),         // organic: texture masks artifacts, save bits
        SceneType::Mixed => (1.0, 0.8),           // mixed: balanced
    }
}

/// Perceptual quality comparison between original and decoded images.
/// Returns metrics that understand geometric integrity, not just pixel error.
pub struct PerceptualQuality {
    pub psnr: f64,
    pub mae: f64,
    /// Edge preservation: ratio of edge energy in decoded vs original
    pub edge_preservation: f64,
    /// Gradient smoothness: how smooth are the gradients in decoded (vs original)
    pub gradient_smoothness_ratio: f64,
    /// Blocking visibility: average step at block grid lines vs interior
    pub blocking_visibility: f64,
}

pub fn compare_quality(
    original: &[f64], decoded: &[f64],
    h: usize, w: usize, block_size: usize,
) -> PerceptualQuality {
    let n = h * w;
    let mut mse = 0.0f64;
    let mut mae = 0.0f64;

    for i in 0..n {
        let d = original[i] - decoded[i];
        mse += d * d;
        mae += d.abs();
    }
    mse /= n as f64;
    mae /= n as f64;
    let psnr = 10.0 * (255.0 * 255.0 / mse).log10();

    // Edge preservation: compare gradient energy
    let grad_energy = |img: &[f64]| -> f64 {
        let mut e = 0.0;
        for y in 0..h - 1 {
            for x in 0..w - 1 {
                let gy = img[(y + 1) * w + x] - img[y * w + x];
                let gx = img[y * w + x + 1] - img[y * w + x];
                e += gy * gy + gx * gx;
            }
        }
        e / n as f64
    };
    let ge_orig = grad_energy(original);
    let ge_dec = grad_energy(decoded);
    let edge_preservation = ge_dec / ge_orig.max(1e-10);

    // Gradient smoothness: ratio of second-order to first-order gradient
    let smooth_ratio = |img: &[f64]| -> f64 {
        let mut g1 = 0.0f64;
        let mut g2 = 0.0f64;
        let mut count = 0u64;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let c = img[y * w + x];
                let lap = img[(y - 1) * w + x] + img[(y + 1) * w + x]
                        + img[y * w + x - 1] + img[y * w + x + 1] - 4.0 * c;
                let gx = (img[y * w + x + 1] - img[y * w + x - 1]).abs();
                let gy = (img[(y + 1) * w + x] - img[(y - 1) * w + x]).abs();
                g1 += gx + gy;
                g2 += lap.abs();
                count += 1;
            }
        }
        if g1 > 0.0 { g2 / g1 } else { 0.0 }
    };
    let sr_orig = smooth_ratio(original);
    let sr_dec = smooth_ratio(decoded);
    let gradient_smoothness_ratio = sr_dec / sr_orig.max(1e-10);

    // Blocking visibility: average |step| at grid lines vs interior
    let mut grid_step = 0.0f64;
    let mut grid_count = 0u64;
    let mut interior_step = 0.0f64;
    let mut interior_count = 0u64;

    for y in 0..h {
        for x in 1..w {
            let step = (decoded[y * w + x] - decoded[y * w + x - 1]).abs();
            if x % block_size == 0 {
                grid_step += step; grid_count += 1;
            } else {
                interior_step += step; interior_count += 1;
            }
        }
    }
    let blocking_visibility = if interior_count > 0 && grid_count > 0 {
        (grid_step / grid_count as f64) / (interior_step / interior_count as f64).max(1e-10)
    } else { 1.0 };

    PerceptualQuality {
        psnr, mae, edge_preservation,
        gradient_smoothness_ratio, blocking_visibility,
    }
}
