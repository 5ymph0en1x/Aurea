//! Orchestration module for the AUR2 v13 4-level Bayesian predictive hierarchy.
//!
//! Level 0: DC grid (existing, unchanged)
//! Level 1: Turing morphogenesis field (zero-bit, reconstructed from DC)
//! Level 2: Primitive matching (only surprises encoded)
//! Level 3: rANS with Bayesian priors (enriched context model)

use crate::bitstream::TuringHeader;
use crate::geometric::{self, Primitive, PrimitiveMatch};
use crate::turing::{self, PredictedContour, TuringField};

/// Parameters for v13 hierarchical encoding, derived from the TuringHeader.
pub struct HierarchyParams {
    pub sigma_a: f64,
    pub sigma_ratio: f64,
    pub k_step: f64,
    pub mag_calibration: f64,
}

impl From<&TuringHeader> for HierarchyParams {
    fn from(hdr: &TuringHeader) -> Self {
        Self {
            sigma_a: hdr.sigma_a as f64,
            sigma_ratio: hdr.sigma_ratio as f64,
            k_step: hdr.k_step as f64,
            mag_calibration: hdr.mag_calibration as f64,
        }
    }
}

impl Default for HierarchyParams {
    fn default() -> Self {
        Self::from(&TuringHeader::default_params())
    }
}

/// Level 1: Compute Turing field from decoded DC grid.
/// Called identically on encoder and decoder.
pub fn compute_level1(
    decoded_dc: &[f64],
    grid_h: usize,
    grid_w: usize,
    params: &HierarchyParams,
) -> TuringField {
    turing::compute_turing_field(
        decoded_dc,
        grid_h,
        grid_w,
        params.sigma_a,
        params.sigma_ratio,
        params.k_step,
    )
}

/// Expand one decoded DC value per adaptive block onto the underlying 8x8
/// cell grid.  The resulting dense grid preserves the real 2-D neighbourhood
/// of the image and can therefore be used by the zero-bit predictors.
pub fn expand_adaptive_dc_grid(
    decoded_dc: &[f64],
    blocks: &[(usize, usize, usize)],
    grid_h: usize,
    grid_w: usize,
    use_overlap: bool,
) -> Result<Vec<f64>, &'static str> {
    if decoded_dc.len() != blocks.len() || grid_h == 0 || grid_w == 0 {
        return Err("adaptive DC layout size mismatch");
    }

    let mut dense = vec![0.0; grid_h * grid_w];
    let mut covered = vec![false; grid_h * grid_w];
    for (block_idx, &(by, bx, block_size)) in blocks.iter().enumerate() {
        if !matches!(block_size, 8 | 16 | 32) {
            return Err("unsupported adaptive block size");
        }
        let top_y = if use_overlap {
            by.saturating_sub(block_size / 2)
        } else {
            by
        };
        let top_x = if use_overlap {
            bx.saturating_sub(block_size / 2)
        } else {
            bx
        };
        let gy0 = top_y / 8;
        let gx0 = top_x / 8;
        let span = block_size / 8;

        if gy0 >= grid_h || gx0 >= grid_w {
            return Err("adaptive block lies outside the cell grid");
        }
        for gy in gy0..(gy0 + span).min(grid_h) {
            for gx in gx0..(gx0 + span).min(grid_w) {
                let idx = gy * grid_w + gx;
                if covered[idx] {
                    return Err("overlapping adaptive blocks in cell grid");
                }
                dense[idx] = decoded_dc[block_idx];
                covered[idx] = true;
            }
        }
    }

    if covered.iter().any(|&v| !v) {
        return Err("adaptive blocks do not cover the cell grid");
    }
    Ok(dense)
}

fn block_cell_bounds(
    block: (usize, usize, usize),
    grid_h: usize,
    grid_w: usize,
    use_overlap: bool,
) -> (usize, usize, usize, usize) {
    let (by, bx, block_size) = block;
    let top_y = if use_overlap {
        by.saturating_sub(block_size / 2)
    } else {
        by
    };
    let top_x = if use_overlap {
        bx.saturating_sub(block_size / 2)
    } else {
        bx
    };
    let gy0 = (top_y / 8).min(grid_h.saturating_sub(1));
    let gx0 = (top_x / 8).min(grid_w.saturating_sub(1));
    let span = (block_size / 8).max(1);
    let gy1 = (gy0 + span).min(grid_h);
    let gx1 = (gx0 + span).min(grid_w);
    (gy0, gy1, gx0, gx1)
}

/// Average a dense cell-grid field over each adaptive block.
pub fn sample_scalar_for_blocks(
    field: &[f64],
    grid_h: usize,
    grid_w: usize,
    blocks: &[(usize, usize, usize)],
    use_overlap: bool,
) -> Vec<f64> {
    debug_assert_eq!(field.len(), grid_h * grid_w);
    blocks
        .iter()
        .map(|&block| {
            let (gy0, gy1, gx0, gx1) = block_cell_bounds(block, grid_h, grid_w, use_overlap);
            let mut sum = 0.0;
            let mut count = 0usize;
            for gy in gy0..gy1 {
                for gx in gx0..gx1 {
                    sum += field[gy * grid_w + gx];
                    count += 1;
                }
            }
            if count == 0 { 0.0 } else { sum / count as f64 }
        })
        .collect()
}

/// Redistribute a chroma channel's AC budget from decoder-reconstructible
/// structure only. DC gradients identify chroma contours that are not
/// explained by luminance.
/// The psychovisual law protects luma-anchored chroma contours and spends less
/// on isolated chromatic detail, whose spatial acuity is lower.
///
/// The returned quantization-step factors have an AC-coefficient-weighted
/// geometric mean of exactly one (apart from floating-point round-off). Thus
/// this is a redistribution of the chroma budget, not a global quality shift.
pub fn chroma_breathing_step_modulations(
    luma_dc: &[f64],
    chroma_dc: &[f64],
    grid_h: usize,
    grid_w: usize,
    blocks: &[(usize, usize, usize)],
    use_overlap: bool,
    strength: f64,
) -> Vec<f64> {
    let strength = strength.clamp(0.0, 1.0);
    if strength == 0.0
        || grid_h == 0
        || grid_w == 0
        || luma_dc.len() != grid_h * grid_w
        || chroma_dc.len() != grid_h * grid_w
        || blocks.is_empty()
    {
        return vec![1.0; blocks.len()];
    }

    let gradient = |field: &[f64], y: usize, x: usize| -> (f64, f64) {
        let left = field[y * grid_w + x.saturating_sub(1)];
        let right = field[y * grid_w + (x + 1).min(grid_w - 1)];
        let top = field[y.saturating_sub(1) * grid_w + x];
        let bottom = field[(y + 1).min(grid_h - 1) * grid_w + x];
        ((right - left) * 0.5, (bottom - top) * 0.5)
    };

    let mut luma_gradients = Vec::with_capacity(grid_h * grid_w);
    let mut chroma_gradients = Vec::with_capacity(grid_h * grid_w);
    let mut luma_energy = 0.0;
    let mut chroma_energy = 0.0;
    for y in 0..grid_h {
        for x in 0..grid_w {
            let lg = gradient(luma_dc, y, x);
            let cg = gradient(chroma_dc, y, x);
            luma_energy += lg.0 * lg.0 + lg.1 * lg.1;
            chroma_energy += cg.0 * cg.0 + cg.1 * cg.1;
            luma_gradients.push(lg);
            chroma_gradients.push(cg);
        }
    }

    let cell_count = (grid_h * grid_w) as f64;
    let luma_rms = (luma_energy / cell_count).sqrt();
    let chroma_rms = (chroma_energy / cell_count).sqrt();
    if chroma_rms <= 1e-12 {
        return vec![1.0; blocks.len()];
    }

    let mut dc_divergence = Vec::with_capacity(grid_h * grid_w);
    for (&(lgx, lgy), &(cgx, cgy)) in luma_gradients.iter().zip(&chroma_gradients) {
        let luma_mag = lgx.hypot(lgy);
        let chroma_mag = cgx.hypot(cgy);
        let luma_norm = luma_mag / luma_rms.max(1e-12);
        let chroma_norm = chroma_mag / chroma_rms;
        let activity = chroma_norm / (chroma_norm + 0.5);

        // Axial coherence deliberately treats opposite gradients as the same
        // contour: only the presence and orientation of the edge matter here.
        let alignment = if luma_mag > 1e-12 && chroma_mag > 1e-12 {
            ((lgx * cgx + lgy * cgy).abs() / (luma_mag * chroma_mag)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let support = (luma_norm / chroma_norm.max(1e-12)).clamp(0.0, 1.0);
        let explained = alignment * alignment * support;
        dc_divergence.push(activity * (1.0 - explained));
    }

    let divergence = sample_scalar_for_blocks(&dc_divergence, grid_h, grid_w, blocks, use_overlap);

    let (weighted_sum, weight_sum) = divergence.iter().zip(blocks).fold(
        (0.0, 0.0),
        |(sum, weights), (&value, &(_, _, block_size))| {
            let weight = (block_size * block_size - 1) as f64;
            (sum + weight * value, weights + weight)
        },
    );
    let mean = weighted_sum / weight_sum.max(1.0);
    let log_span = crate::golden::PHI.ln();
    divergence
        .iter()
        .map(|&value| (strength * log_span * (value - mean)).exp())
        .collect()
}

/// One Turing context bucket per adaptive transform block.
pub fn build_adaptive_turing_buckets(
    tf: &TuringField,
    blocks: &[(usize, usize, usize)],
    use_overlap: bool,
) -> Vec<u8> {
    sample_scalar_for_blocks(&tf.field, tf.grid_h, tf.grid_w, blocks, use_overlap)
        .into_iter()
        .map(|t| turing::turing_bucket(t) as u8)
        .collect()
}

/// One direction and normalized gradient strength per adaptive block.  Vector
/// averaging avoids the wrap-around error of directly averaging angles.
pub fn build_adaptive_gradient_angles(
    tf: &TuringField,
    blocks: &[(usize, usize, usize)],
    use_overlap: bool,
) -> Vec<(f64, f64)> {
    let max_grad = tf
        .inhibitor_gradient
        .iter()
        .copied()
        .fold(0.0f64, f64::max)
        .max(1e-6);
    blocks
        .iter()
        .map(|&block| {
            let (gy0, gy1, gx0, gx1) = block_cell_bounds(block, tf.grid_h, tf.grid_w, use_overlap);
            let mut vx = 0.0;
            let mut vy = 0.0;
            let mut strength = 0.0;
            let mut count = 0usize;
            for gy in gy0..gy1 {
                for gx in gx0..gx1 {
                    let idx = gy * tf.grid_w + gx;
                    let mag = tf.inhibitor_gradient[idx];
                    let angle = tf.inhibitor_gradient_angle[idx];
                    vx += mag * angle.cos();
                    vy += mag * angle.sin();
                    strength += mag;
                    count += 1;
                }
            }
            if count == 0 {
                (0.0, 0.0)
            } else {
                (
                    vy.atan2(vx),
                    (strength / count as f64 / max_grad).clamp(0.0, 1.0),
                )
            }
        })
        .collect()
}

/// One dominant gradient orientation and tensor coherence per adaptive block.
/// Unlike vector averaging, the second-moment tensor treats opposite gradient
/// signs as the same edge orientation instead of cancelling them out.
pub fn build_adaptive_structure_tensors(
    tf: &TuringField,
    blocks: &[(usize, usize, usize)],
    use_overlap: bool,
) -> Vec<(f64, f64)> {
    blocks
        .iter()
        .map(|&block| {
            let (gy0, gy1, gx0, gx1) = block_cell_bounds(block, tf.grid_h, tf.grid_w, use_overlap);
            let mut jxx = 0.0;
            let mut jyy = 0.0;
            let mut jxy = 0.0;
            for gy in gy0..gy1 {
                for gx in gx0..gx1 {
                    let idx = gy * tf.grid_w + gx;
                    let magnitude = tf.inhibitor_gradient[idx];
                    let angle = tf.inhibitor_gradient_angle[idx];
                    let grad_x = magnitude * angle.cos();
                    let grad_y = magnitude * angle.sin();
                    jxx += grad_x * grad_x;
                    jyy += grad_y * grad_y;
                    jxy += grad_x * grad_y;
                }
            }
            let trace = jxx + jyy;
            if trace <= 1e-20 {
                (0.0, 0.0)
            } else {
                let anisotropy = ((jxx - jyy).powi(2) + 4.0 * jxy * jxy).sqrt();
                (
                    0.5 * (2.0 * jxy).atan2(jxx - jyy),
                    (anisotropy / trace).clamp(0.0, 1.0),
                )
            }
        })
        .collect()
}

/// Blend the calibrated vector direction with a structure-tensor direction.
/// The angle is blended axially (period pi), while QMAT influence fades toward
/// tensor coherence. Strength zero reproduces the legacy `(angle, strength, 1)`
/// tuple exactly for every block.
pub fn build_adaptive_directional_structures(
    tf: &TuringField,
    blocks: &[(usize, usize, usize)],
    use_overlap: bool,
    tensor_strength: f64,
) -> Vec<(f64, f64, f64)> {
    let legacy = build_adaptive_gradient_angles(tf, blocks, use_overlap);
    let tensor_strength = tensor_strength.clamp(0.0, 1.0);
    if tensor_strength == 0.0 {
        return legacy
            .into_iter()
            .map(|(angle, strength)| (angle, strength, 1.0))
            .collect();
    }
    let tensors = build_adaptive_structure_tensors(tf, blocks, use_overlap);
    legacy
        .into_iter()
        .zip(tensors)
        .map(
            |((legacy_angle, gradient_strength), (tensor_angle, coherence))| {
                let legacy_weight = 1.0 - tensor_strength;
                let axial_x = legacy_weight * (2.0 * legacy_angle).cos()
                    + tensor_strength * (2.0 * tensor_angle).cos();
                let axial_y = legacy_weight * (2.0 * legacy_angle).sin()
                    + tensor_strength * (2.0 * tensor_angle).sin();
                let angle = if axial_x.abs() + axial_y.abs() <= 1e-20 {
                    legacy_angle
                } else {
                    0.5 * axial_y.atan2(axial_x)
                };
                let qmat_blend = 1.0 - tensor_strength * (1.0 - coherence);
                (angle, gradient_strength, qmat_blend.clamp(0.0, 1.0))
            },
        )
        .collect()
}

/// Geometric interpolation preserves the multiplicative meaning of QMAT
/// factors. Endpoints are explicit so the neutral experiment is bit-exact.
#[inline]
pub fn blend_qmat_factors(standard: f64, rotated: f64, blend: f64) -> f64 {
    let blend = blend.clamp(0.0, 1.0);
    if blend == 0.0 {
        standard
    } else if blend == 1.0 {
        rotated
    } else {
        (standard.ln() * (1.0 - blend) + rotated.ln() * blend).exp()
    }
}

/// Zero-bit description of a slowly varying luminance wave reconstructed
/// from the transmitted DC lattice. `phase=0` is gradient-dominated (a local
/// zero crossing, fundamental mode); `phase=1` is curvature-dominated (a
/// crest/trough, second mode). Angle is axial and coherence measures whether
/// neighbouring quadratures agree on one propagation direction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SubsonicWaveDescriptor {
    pub angle: f64,
    pub phase: f64,
    pub coherence: f64,
}

/// Recover local wave quadratures from a dense DC grid, then average them on
/// the real adaptive LOT support. Gradient and Hessian RMS are normalized
/// separately so slope and curvature act as phase quadratures rather than as
/// quantities with incompatible units.
pub fn subsonic_wave_descriptors(
    dense_dc: &[f64],
    grid_h: usize,
    grid_w: usize,
    blocks: &[(usize, usize, usize)],
    use_overlap: bool,
) -> Vec<SubsonicWaveDescriptor> {
    if grid_h < 2 || grid_w < 2 || dense_dc.len() < grid_h.saturating_mul(grid_w) {
        return vec![SubsonicWaveDescriptor::default(); blocks.len()];
    }

    #[derive(Clone, Copy, Default)]
    struct CellWave {
        gradient: f64,
        curvature: f64,
        gradient_angle: f64,
        curvature_angle: f64,
    }

    let sample = |y: isize, x: isize| -> f64 {
        let yy = y.clamp(0, grid_h as isize - 1) as usize;
        let xx = x.clamp(0, grid_w as isize - 1) as usize;
        dense_dc[yy * grid_w + xx]
    };
    let mut cells = vec![CellWave::default(); grid_h * grid_w];
    let mut gradient_energy = 0.0;
    let mut curvature_energy = 0.0;
    for y in 0..grid_h {
        for x in 0..grid_w {
            let y = y as isize;
            let x = x as isize;
            let centre = sample(y, x);
            let gx = 0.5 * (sample(y, x + 1) - sample(y, x - 1));
            let gy = 0.5 * (sample(y + 1, x) - sample(y - 1, x));
            let hxx = sample(y, x + 1) - 2.0 * centre + sample(y, x - 1);
            let hyy = sample(y + 1, x) - 2.0 * centre + sample(y - 1, x);
            let hxy = 0.25
                * (sample(y + 1, x + 1) - sample(y + 1, x - 1) - sample(y - 1, x + 1)
                    + sample(y - 1, x - 1));
            let gradient = gx.hypot(gy);
            let curvature = (hxx * hxx + 2.0 * hxy * hxy + hyy * hyy).sqrt();
            let idx = y as usize * grid_w + x as usize;
            cells[idx] = CellWave {
                gradient,
                curvature,
                gradient_angle: gy.atan2(gx),
                curvature_angle: 0.5 * (2.0 * hxy).atan2(hxx - hyy),
            };
            gradient_energy += gradient * gradient;
            curvature_energy += curvature * curvature;
        }
    }
    let n = (grid_h * grid_w).max(1) as f64;
    let gradient_rms = (gradient_energy / n).sqrt().max(1e-12);
    let curvature_rms = (curvature_energy / n).sqrt().max(1e-12);

    blocks
        .iter()
        .map(|&block| {
            let (gy0, gy1, gx0, gx1) = block_cell_bounds(block, grid_h, grid_w, use_overlap);
            let mut axial_x = 0.0;
            let mut axial_y = 0.0;
            let mut phase_sum = 0.0;
            let mut weight_sum = 0.0;
            let mut cell_count = 0usize;
            for gy in gy0..gy1 {
                for gx in gx0..gx1 {
                    let wave = cells[gy * grid_w + gx];
                    let gradient = wave.gradient / gradient_rms;
                    let curvature = wave.curvature / curvature_rms;
                    let weight = gradient.hypot(curvature);
                    if weight <= 1e-12 {
                        continue;
                    }
                    let phase = curvature / (gradient + curvature).max(1e-12);
                    let angle = if gradient >= curvature {
                        wave.gradient_angle
                    } else {
                        wave.curvature_angle
                    };
                    axial_x += weight * (2.0 * angle).cos();
                    axial_y += weight * (2.0 * angle).sin();
                    phase_sum += weight * phase;
                    weight_sum += weight;
                    cell_count += 1;
                }
            }
            if weight_sum <= 1e-12 || cell_count == 0 {
                return SubsonicWaveDescriptor::default();
            }
            let axial_coherence = axial_x.hypot(axial_y) / weight_sum;
            let mean_envelope = weight_sum / cell_count as f64;
            let activity = (mean_envelope / 1.25).clamp(0.0, 1.0);
            SubsonicWaveDescriptor {
                angle: 0.5 * axial_y.atan2(axial_x),
                phase: (phase_sum / weight_sum).clamp(0.0, 1.0),
                coherence: (axial_coherence * activity).clamp(0.0, 1.0),
            }
        })
        .collect()
}

/// Natural-order AC quantization factors for one phase-coherent wave packet.
/// This is a local exchange between the first and second directional
/// harmonics, not a tax on the full AC tail. A gradient-dominated phase gives
/// budget to the fundamental; a curvature-dominated phase gives it to the
/// second mode; exact quadrature is neutral. The two Gaussian packets are
/// normalized against each other so their total log-budget is exactly zero.
pub fn subsonic_wave_qfactors(
    descriptor: SubsonicWaveDescriptor,
    block_size: usize,
    strength: f64,
) -> Vec<f64> {
    let mut factors = vec![1.0; block_size.saturating_mul(block_size)];
    let strength = strength.clamp(0.0, 1.5);
    if strength == 0.0 || descriptor.coherence <= 1e-12 || block_size < 2 {
        return factors;
    }
    let phase_bias = 1.0 - 2.0 * descriptor.phase.clamp(0.0, 1.0);
    if phase_bias.abs() <= 1e-12 {
        return factors;
    }
    let (sin_a, cos_a) = descriptor.angle.sin_cos();
    let parallel_sigma = 0.58;
    let perpendicular_sigma = 0.72;
    let mut fundamental = vec![0.0; factors.len()];
    let mut second = vec![0.0; factors.len()];
    let mut fundamental_sum = 0.0;
    let mut second_sum = 0.0;
    for row in 0..block_size {
        for col in 0..block_size {
            if row == 0 && col == 0 {
                continue;
            }
            // Spatial x maps to the DCT column index and spatial y to the
            // row index. Keeping this dual coordinate system explicit avoids
            // rotating a horizontal wave into the vertical spectrum.
            let freq_x = col as f64;
            let freq_y = row as f64;
            let parallel = (freq_x * cos_a + freq_y * sin_a).abs();
            let perpendicular = (-freq_x * sin_a + freq_y * cos_a).abs();
            let dn = perpendicular / perpendicular_sigma;
            let d1 = (parallel - 1.0) / parallel_sigma;
            let d2 = (parallel - 2.0) / parallel_sigma;
            let kernel1 = (-0.5 * (d1 * d1 + dn * dn)).exp();
            let kernel2 = (-0.5 * (d2 * d2 + dn * dn)).exp();
            let idx = row * block_size + col;
            fundamental[idx] = kernel1;
            second[idx] = kernel2;
            fundamental_sum += kernel1;
            second_sum += kernel2;
        }
    }
    let donor_scale = fundamental_sum / second_sum.max(1e-12);
    let exponent = strength * descriptor.coherence * phase_bias;
    for idx in 1..factors.len() {
        let exchange = fundamental[idx] - donor_scale * second[idx];
        factors[idx] = crate::golden::PHI.powf(-exponent * exchange);
    }
    factors
}

/// Rebalance already-pivoted Turing step modulations without adding side data.
///
/// The centre is a geometric mean because quantization steps combine
/// multiplicatively. Each block is weighted by its number of AC coefficients,
/// so the conserved quantity follows the actual transform budget. `strength=0`
/// is an exact pass-through; `strength=1` recentres the weighted geometric mean
/// on one (apart from the final safety clamp).
/// Zero-bit spectral exchange (« sous-échantillonnage doux », lab cycle 12).
///
/// Measures how much chroma structure the image really has, from the three
/// already-decoded dense DC grids. Returns gamma ∈ [0, eta]: 0 when chroma
/// activity rivals luma (protect the color advantage), up to eta when the
/// chroma field is flat (JPEG 4:2:0 territory — trade invisible chroma HF
/// for luminance budget).
///
/// Both sides call this with identical inputs, so the field costs zero bits.
pub fn spectral_exchange_gamma(
    dense_l_dc: &[f64],
    dense_c1_dc: &[f64],
    dense_c2_dc: &[f64],
    grid_h: usize,
    grid_w: usize,
    eta: f64,
) -> f64 {
    let eta = eta.clamp(0.0, 1.0);
    if eta == 0.0 || grid_h < 2 || grid_w < 2 {
        return 0.0;
    }
    // RMS of forward-difference gradients on a dense cell grid.
    let rms_grad = |grid: &[f64]| -> f64 {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let idx = gy * grid_w + gx;
                if gx + 1 < grid_w {
                    let d = grid[idx + 1] - grid[idx];
                    sum += d * d;
                    count += 1;
                }
                if gy + 1 < grid_h {
                    let d = grid[idx + grid_w] - grid[idx];
                    sum += d * d;
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum / count as f64).sqrt()
        }
    };
    let rms_l = rms_grad(dense_l_dc);
    let rms_c = 0.5 * (rms_grad(dense_c1_dc) + rms_grad(dense_c2_dc));
    if rms_l <= 1e-9 {
        return 0.0;
    }
    let activity_ratio = (rms_c / rms_l).min(1.0);
    eta * (1.0 - activity_ratio)
}

/// Number of quantized levels of the local spectral exchange. Small on
/// purpose: the chroma coarsening tables are precomputed once per level, and a
/// coarse ladder is also what keeps the field reproducible across the two
/// sides without a single transmitted bit.
pub const SPECTRAL_LOCAL_LEVELS: usize = 8;

/// Cycle 21 — local redistribution of the spectral exchange.
///
/// The global gamma answers *how much chroma this image can afford to give up*.
/// Inside one frame the honest answer is rarely uniform: a flat sky and a
/// coloured face share the same picture and currently share the same trade.
///
/// For every block this recomputes the exact same chroma/luma gradient ratio
/// the global field uses, but over a local window of DC cells. The result is
/// turned into a multiplicative ratio around the global gamma, quantized onto
/// `SPECTRAL_LOCAL_LEVELS`, then **renormalized so the AC-weighted mean of the
/// per-block gammas is exactly the global gamma**. The total trade is therefore
/// unchanged — only its distribution moves — which is what makes the experiment
/// readable: any metric difference comes from *where* the bits went, never from
/// how many were spent.
///
/// `strength = 0` returns a uniform field and reproduces v21 bit-exactly.
///
/// Returns one level index per block plus the per-level gamma multipliers.
pub fn spectral_exchange_local_levels(
    dense_l_dc: &[f64],
    dense_c1_dc: &[f64],
    dense_c2_dc: &[f64],
    grid_h: usize,
    grid_w: usize,
    blocks: &[(usize, usize, usize)],
    use_overlap: bool,
    strength: f64,
) -> (Vec<u8>, [f64; SPECTRAL_LOCAL_LEVELS]) {
    let uniform = (vec![0u8; blocks.len()], {
        let mut table = [0.0f64; SPECTRAL_LOCAL_LEVELS];
        table[0] = 1.0;
        table
    });
    let strength = strength.clamp(0.0, 1.0);
    if strength == 0.0 || blocks.is_empty() || grid_h < 2 || grid_w < 2 {
        return uniform;
    }

    // Local RMS of forward-difference gradients over a cell window, using the
    // very same difference stencil as the global field.
    let window_rms = |grid: &[f64], y0: usize, y1: usize, x0: usize, x1: usize| -> f64 {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for gy in y0..y1 {
            for gx in x0..x1 {
                let idx = gy * grid_w + gx;
                if gx + 1 < grid_w {
                    let d = grid[idx + 1] - grid[idx];
                    sum += d * d;
                    count += 1;
                }
                if gy + 1 < grid_h {
                    let d = grid[idx + grid_w] - grid[idx];
                    sum += d * d;
                    count += 1;
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            (sum / count as f64).sqrt()
        }
    };

    // Raw local gamma per block, on the same 0..1 scale as the global one.
    let mut raw = Vec::with_capacity(blocks.len());
    for &(by, bx, block_size) in blocks {
        let top_y = if use_overlap {
            by.saturating_sub(block_size / 2)
        } else {
            by
        };
        let top_x = if use_overlap {
            bx.saturating_sub(block_size / 2)
        } else {
            bx
        };
        let span = (block_size / 8).max(1);
        // One cell of margin: a block alone is too small a sample for a ratio.
        let y0 = (top_y / 8).saturating_sub(1);
        let x0 = (top_x / 8).saturating_sub(1);
        let y1 = (top_y / 8 + span + 1).min(grid_h);
        let x1 = (top_x / 8 + span + 1).min(grid_w);
        if y0 >= y1 || x0 >= x1 {
            raw.push(0.0);
            continue;
        }
        let rms_l = window_rms(dense_l_dc, y0, y1, x0, x1);
        let rms_c = 0.5
            * (window_rms(dense_c1_dc, y0, y1, x0, x1) + window_rms(dense_c2_dc, y0, y1, x0, x1));
        if rms_l <= 1e-9 {
            raw.push(0.0);
        } else {
            raw.push(1.0 - (rms_c / rms_l).min(1.0));
        }
    }

    // The invariant to preserve is the AC-weighted log-mean of the step
    // multipliers, which `spectral_exchange_delta` balances as
    // `sum_b w_b * fbar(size_b) * gamma_b`. Weighting by AC count alone would
    // let the field leak quality whenever the local level correlates with
    // block size — so the mean frequency belongs in the weight.
    let fbar_8 = mean_freq_norm(8);
    let fbar_16 = mean_freq_norm(16);
    let fbar_32 = mean_freq_norm(32);
    let weight = |block_size: usize| -> f64 {
        let fbar = match block_size {
            8 => fbar_8,
            32 => fbar_32,
            _ => fbar_16,
        };
        block_size
            .saturating_mul(block_size)
            .saturating_sub(1)
            .max(1) as f64
            * fbar
    };
    let mut weighted = 0.0f64;
    let mut weight_sum = 0.0f64;
    for (&value, &(_, _, block_size)) in raw.iter().zip(blocks) {
        let w = weight(block_size);
        weighted += w * value;
        weight_sum += w;
    }
    let mean = weighted / weight_sum.max(1e-12);
    if mean <= 1e-9 {
        return uniform;
    }

    // Ratio around the mean, damped by `strength` and clamped so no block ever
    // gives up more than twice the average trade.
    let quantize = |value: f64| -> u8 {
        let ratio = (1.0 - strength) + strength * (value / mean);
        let clamped = ratio.clamp(0.0, 2.0);
        let level = (clamped / 2.0 * (SPECTRAL_LOCAL_LEVELS - 1) as f64).round();
        (level as usize).min(SPECTRAL_LOCAL_LEVELS - 1) as u8
    };
    let levels: Vec<u8> = raw.iter().map(|&value| quantize(value)).collect();

    // Renormalize the quantized ladder so the AC-weighted mean multiplier is
    // exactly 1: the image spends the same chroma budget as before.
    let mut ladder = [0.0f64; SPECTRAL_LOCAL_LEVELS];
    for (index, slot) in ladder.iter_mut().enumerate() {
        *slot = 2.0 * index as f64 / (SPECTRAL_LOCAL_LEVELS - 1) as f64;
    }
    let mut ladder_weighted = 0.0f64;
    for (&level, &(_, _, block_size)) in levels.iter().zip(blocks) {
        ladder_weighted += weight(block_size) * ladder[level as usize];
    }
    let ladder_mean = ladder_weighted / weight_sum.max(1e-12);
    if ladder_mean <= 1e-9 {
        return uniform;
    }
    for slot in ladder.iter_mut() {
        *slot /= ladder_mean;
    }
    (levels, ladder)
}

/// Global zero-bit signature of a diffuse chromatic field.
///
/// The adaptive LOT stores DC as `mean * 16^2` for every transform size, so
/// chroma RMS is divided by 256 to return to sample-domain units.  A repair is
/// useful only when four conditions meet: chroma is smoother than luminance,
/// its amplitude is neither negligible nor dominant, and its mean hue vector
/// is weak compared with its RMS amplitude.  Smooth gates avoid image-class
/// discontinuities and are reconstructed identically by the decoder.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ChromaDiffusionSignature {
    pub repair: f64,
    pub chroma_rms: f64,
    pub hue_coherence: f64,
}

pub fn chroma_diffusion_signature(
    dense_c1_dc: &[f64],
    dense_c2_dc: &[f64],
    spectral_gamma: f64,
) -> ChromaDiffusionSignature {
    let n = dense_c1_dc.len().min(dense_c2_dc.len());
    if n == 0 {
        return ChromaDiffusionSignature::default();
    }

    let (sum_c1, sum_c2, sum_norm2) = dense_c1_dc
        .iter()
        .zip(dense_c2_dc)
        .take(n)
        .fold((0.0, 0.0, 0.0), |(s1, s2, sn), (&c1, &c2)| {
            (s1 + c1, s2 + c2, sn + c1 * c1 + c2 * c2)
        });
    let inv_n = 1.0 / n as f64;
    let rms_dc = (sum_norm2 * inv_n).sqrt();
    let chroma_rms = rms_dc / 256.0;
    let hue_coherence =
        ((sum_c1 * inv_n).hypot(sum_c2 * inv_n) / rms_dc.max(1e-12)).clamp(0.0, 1.0);

    let smoothstep = |lo: f64, hi: f64, value: f64| -> f64 {
        let t = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let smooth_chroma = smoothstep(0.58, 0.70, spectral_gamma);
    let enough_chroma = smoothstep(14.0, 20.0, chroma_rms);
    let moderate_chroma = 1.0 - smoothstep(30.0, 42.0, chroma_rms);
    let diffuse_hue = 1.0 - smoothstep(0.15, 0.35, hue_coherence);
    let repair = (smooth_chroma * enough_chroma * moderate_chroma * diffuse_hue).clamp(0.0, 1.0);

    ChromaDiffusionSignature {
        repair,
        chroma_rms,
        hue_coherence,
    }
}

/// Fraction of dense luma DC cells inside the Weber over-delivery zone:
/// PTF-space block means below the PTF image of linear luminance 40, the
/// darkness band where the kryte diagnosis measured AUREA's systematic MSE
/// advantage over JPEG (tile ratio 0.83). The adaptive DC grid stores
/// `mean * 256`, hence the scaling. Zero-bit: both sides reconstruct it from
/// the decoded DC grid.
pub fn luma_dark_fraction(dense_l_dc: &[f64]) -> f64 {
    if dense_l_dc.is_empty() {
        return 0.0;
    }
    let threshold = 256.0 * 255.0 * (40.0f64 / 255.0).powf(crate::golden::PTF_GAMMA);
    let dark = dense_l_dc.iter().filter(|&&dc| dc < threshold).count();
    dark as f64 / dense_l_dc.len() as f64
}

/// Dark gate of the luma-pivot relief. The relief hands back luma refinement
/// on diffuse-chroma fields, but that surplus only exists where the Weber
/// chain over-delivers — dark scenes. On bright scenes (kodim21: lighthouse,
/// daylight) the finer luma steps earn their keep and relieving them loses
/// RGB. Smooth gate, reconstructed identically on both sides.
pub fn luma_relief_dark_gate(dense_l_dc: &[f64]) -> f64 {
    let fraction = luma_dark_fraction(dense_l_dc);
    let t = ((fraction - 0.30) / (0.45 - 0.30)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Zero-bit step-modulation field from the transmitted EOB positions
/// (cycle 32, AUR2 v30). The normalized activity `a_b = eob_b / capacity_b`
/// has rank correlation 0.887 with a block's squared error — against 0.39
/// for the DC-Sobel field that used to be the only step driver. Blocks that
/// carry the error receive a finer step, financed log-neutrally (AC-count
/// weights) by the quiet ones. Every channel's EOB is decoded before any AC
/// coefficient, so both sides derive the exact same field; the encoder pays
/// a second quantization pass whose EOBs are clamped to the transmitted
/// (first-pass) values.
pub fn eob_step_mods(eobs: &[usize], capacities: &[f64], eta: f64) -> Vec<f64> {
    if eta == 0.0 || eobs.is_empty() {
        return vec![1.0; eobs.len()];
    }
    let activities: Vec<f64> = eobs
        .iter()
        .zip(capacities)
        .map(|(&e, &c)| (e as f64 / c.max(1.0)).clamp(0.0, 1.0))
        .collect();
    let mut weighted = 0.0f64;
    let mut weight_sum = 0.0f64;
    for (&a, &c) in activities.iter().zip(capacities) {
        weighted += c * a;
        weight_sum += c;
    }
    let mean = weighted / weight_sum.max(1e-12);
    activities
        .iter()
        .map(|&a| crate::golden::PHI.powf(-eta * (a - mean)))
        .collect()
}

/// C1 step factor for the diffuse-chroma repair. C2 deliberately stays at one:
/// counter-gift sweeps moved the error from Cb to Cr instead of repairing it.
pub fn chroma_diffusion_step_factor(base_c1_factor: f64, repair: f64) -> f64 {
    let base = base_c1_factor.max(1e-9);
    let repair = repair.clamp(0.0, 1.0);
    if repair == 0.0 {
        base
    } else {
        base.powf(1.0 - repair)
    }
}

/// Mean normalized frequency of the AC positions of one block size, using
/// the same freq_norm formula as the CSF: sqrt(r² + c²) / (bs_max · √2).
/// Deterministic — identical on both sides.
pub fn mean_freq_norm(block_size: usize) -> f64 {
    let bs_max = (block_size - 1).max(1) as f64;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for r in 0..block_size {
        for c in 0..block_size {
            if r == 0 && c == 0 {
                continue;
            }
            sum += ((r * r + c * c) as f64).sqrt() / (bs_max * std::f64::consts::SQRT_2);
            count += 1;
        }
    }
    sum / count.max(1) as f64
}

/// Luma refinement exponent delta balancing the chroma HF coarsening in
/// AC-weighted log space: 2 · gamma · fbar (two chroma channels, each with
/// the same block layout and AC weights as luma). The global AC-weighted
/// log-mean of all step multipliers is then exactly zero.
pub fn spectral_exchange_delta(gamma: f64, block_sizes: &[usize]) -> f64 {
    if gamma <= 0.0 || block_sizes.is_empty() {
        return 0.0;
    }
    let fbar_8 = mean_freq_norm(8);
    let fbar_16 = mean_freq_norm(16);
    let fbar_32 = mean_freq_norm(32);
    let mut weighted = 0.0f64;
    let mut weight_sum = 0.0f64;
    for &bs in block_sizes {
        let fbar = match bs {
            8 => fbar_8,
            32 => fbar_32,
            _ => fbar_16,
        };
        let w = bs.saturating_mul(bs).saturating_sub(1).max(1) as f64;
        weighted += w * fbar;
        weight_sum += w;
    }
    let fbar = weighted / weight_sum.max(1.0);
    2.0 * gamma * fbar
}

/// Zero-bit bright-compensation on the luma steps (« contre-don de Weber »).
///
/// Bright blocks receive a finer quantization step, funded by dark blocks
/// where the Weber design (PTF) already over-delivers. `block_dc_pixel` is
/// the per-block mean luminance in PTF space (~0-255), reconstructed from
/// the transmitted DC on both sides — the field costs zero bits.
///
/// The AC-weighted (`size² - 1`) log-space recentering keeps the weighted
/// geometric mean of the factors exactly 1: no hidden global quality shift.
/// Each factor stays within [phi^-eta, phi^eta]. eta = 0 returns pure 1.0
/// factors and reproduces the current profile bit-exactly. Values above one
/// are reserved for laboratory exploration of the promoted boundary.
pub fn bright_compensation_step_modulations(
    block_dc_pixel: &[f64],
    block_sizes: &[usize],
    eta: f64,
) -> Vec<f64> {
    let eta = eta.clamp(0.0, 2.5);
    let n = block_dc_pixel.len();
    if eta == 0.0 || n == 0 {
        return vec![1.0; n];
    }

    let brightness: Vec<f64> = block_dc_pixel
        .iter()
        .map(|&dc| {
            let safe = if dc.is_finite() { dc } else { 0.0 };
            (safe / 255.0).clamp(0.0, 1.0)
        })
        .collect();

    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for (idx, &b) in brightness.iter().enumerate() {
        let block_size = block_sizes.get(idx).copied().unwrap_or(16);
        let weight = block_size
            .saturating_mul(block_size)
            .saturating_sub(1)
            .max(1) as f64;
        weighted_sum += weight * b;
        weight_sum += weight;
    }
    let mean_b = weighted_sum / weight_sum.max(1.0);

    let ln_phi = crate::golden::PHI.ln();
    brightness
        .iter()
        .map(|&b| (-ln_phi * eta * (b - mean_b)).exp())
        .collect()
}

pub fn homeostatic_step_modulations(
    pivoted_modulations: &[f64],
    block_sizes: &[usize],
    strength: f64,
) -> Vec<f64> {
    let strength = strength.clamp(0.0, 1.0);
    if strength == 0.0 || pivoted_modulations.is_empty() {
        return pivoted_modulations.to_vec();
    }

    let mut weighted_log_sum = 0.0;
    let mut weight_sum = 0.0;
    for (idx, &modulation) in pivoted_modulations.iter().enumerate() {
        let block_size = block_sizes.get(idx).copied().unwrap_or(16);
        let weight = block_size
            .saturating_mul(block_size)
            .saturating_sub(1)
            .max(1) as f64;
        let safe_modulation = if modulation.is_finite() {
            modulation.clamp(0.1, 10.0)
        } else {
            1.0
        };
        weighted_log_sum += weight * safe_modulation.ln();
        weight_sum += weight;
    }

    let log_centre = weighted_log_sum / weight_sum.max(1.0);
    let correction = (-strength * log_centre).exp();
    pivoted_modulations
        .iter()
        .map(|&modulation| {
            let safe_modulation = if modulation.is_finite() {
                modulation.clamp(0.1, 10.0)
            } else {
                1.0
            };
            (safe_modulation * correction).clamp(0.1, 10.0)
        })
        .collect()
}

/// Couple the calibrated local Turing scale with a second scale spaced by phi.
/// At full strength the log-space weights are phi^-1 for the local field and
/// phi^-2 for the wider field; these weights sum to one. Strength zero is an
/// exact pass-through and therefore keeps the frozen v13 bitstream unchanged.
pub fn phi_multiscale_step_modulations(local: &[f64], wider: &[f64], strength: f64) -> Vec<f64> {
    let strength = strength.clamp(0.0, 1.0);
    if strength == 0.0 || local.len() != wider.len() {
        return local.to_vec();
    }
    let wider_weight = strength * crate::golden::PHI_INV2;
    let local_weight = 1.0 - wider_weight;
    local
        .iter()
        .zip(wider)
        .map(|(&local_modulation, &wider_modulation)| {
            let local_safe = if local_modulation.is_finite() {
                local_modulation.clamp(0.1, 10.0)
            } else {
                1.0
            };
            let wider_safe = if wider_modulation.is_finite() {
                wider_modulation.clamp(0.1, 10.0)
            } else {
                1.0
            };
            (local_weight * local_safe.ln() + wider_weight * wider_safe.ln()).exp()
        })
        .collect()
}

/// Redistribute the local quantization budget according to surprise along a
/// continuous centre-to-periphery spiral. A golden-memory predictor follows
/// the log-modulation signal. Unexpected blocks receive a finer step while
/// predictable blocks pay the corresponding cost.
///
/// Surprise is centred by each block's AC population, so the weighted
/// geometric mean of all modulation steps is conserved. This makes the field
/// a redistribution rather than a hidden global quality change.
pub fn spiral_surprise_step_modulations(
    modulations: &[f64],
    spiral_order: &[usize],
    block_sizes: &[usize],
    strength: f64,
) -> Vec<f64> {
    let strength = strength.clamp(0.0, 1.0);
    if strength == 0.0
        || modulations.is_empty()
        || spiral_order.len() != modulations.len()
        || block_sizes.len() != modulations.len()
    {
        return modulations.to_vec();
    }

    let safe_logs: Vec<f64> = modulations
        .iter()
        .map(|&value| {
            if value.is_finite() {
                value.clamp(0.1, 10.0).ln()
            } else {
                0.0
            }
        })
        .collect();
    let first_idx = spiral_order[0];
    if first_idx >= modulations.len() {
        return modulations.to_vec();
    }

    let mut prediction = safe_logs[first_idx];
    let mut surprise = vec![0.0; modulations.len()];
    for &block_idx in spiral_order {
        if block_idx >= modulations.len() {
            return modulations.to_vec();
        }
        let observation = safe_logs[block_idx];
        surprise[block_idx] = (observation - prediction).abs();
        prediction = crate::golden::PHI_INV * prediction + crate::golden::PHI_INV2 * observation;
    }

    let (weighted_surprise, weight_sum) = surprise.iter().zip(block_sizes).fold(
        (0.0, 0.0),
        |(sum, weights), (&value, &block_size)| {
            let weight = block_size.saturating_mul(block_size).saturating_sub(1) as f64;
            (sum + weight * value, weights + weight)
        },
    );
    if weight_sum <= 0.0 {
        return modulations.to_vec();
    }
    let centre = weighted_surprise / weight_sum;

    safe_logs
        .into_iter()
        .zip(surprise)
        .map(|(log_modulation, value)| {
            (log_modulation - strength * (value - centre))
                .exp()
                .clamp(0.1, 10.0)
        })
        .collect()
}

/// Level 1 contour extraction.
pub fn extract_predicted_contours(tf: &TuringField) -> Vec<PredictedContour> {
    turing::trace_ridges(
        &tf.field,
        tf.grid_h,
        tf.grid_w,
        Some(&tf.inhibitor_gradient),
    )
}

/// Level 2: Match extracted primitives against Turing predictions.
pub fn match_level2(
    primitives: &[Primitive],
    contours: &[PredictedContour],
    block_size: usize,
) -> Vec<PrimitiveMatch> {
    geometric::match_primitives(primitives, contours, block_size)
}

/// Build turing_buckets array for Level 3 rANS encoding.
/// Maps each coefficient position (in raster order) to a turing bucket (0-3).
pub fn build_turing_buckets(
    tf: &TuringField,
    coeff_h: usize,
    coeff_w: usize,
    block_size: usize,
) -> Vec<u8> {
    let scale = (block_size / 2).max(1); // detail bands are half-resolution
    let mut buckets = Vec::with_capacity(coeff_h * coeff_w);
    for y in 0..coeff_h {
        for x in 0..coeff_w {
            let gy = (y / scale).min(tf.grid_h.saturating_sub(1));
            let gx = (x / scale).min(tf.grid_w.saturating_sub(1));
            let t = tf.field[gy * tf.grid_w + gx];
            buckets.push(turing::turing_bucket(t) as u8);
        }
    }
    buckets
}

/// Build per-block gradient angle and strength from the Turing inhibitor field.
/// Returns `(angle_radians, normalized_strength)` per block.
/// `strength` is `inhibitor_gradient / max_gradient`, in [0, 1].
pub fn build_gradient_angles(
    tf: &TuringField,
    n_blocks_h: usize,
    n_blocks_w: usize,
    block_size: usize,
) -> Vec<(f64, f64)> {
    let n_blocks = n_blocks_h * n_blocks_w;
    let max_grad = tf
        .inhibitor_gradient
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        .max(1e-6);

    let mut result = Vec::with_capacity(n_blocks);
    for by in 0..n_blocks_h {
        for bx in 0..n_blocks_w {
            let gy = by.min(tf.grid_h.saturating_sub(1));
            let gx = bx.min(tf.grid_w.saturating_sub(1));
            let idx = gy * tf.grid_w + gx;
            let angle = tf.inhibitor_gradient_angle[idx];
            let strength = tf.inhibitor_gradient[idx] / max_grad;
            result.push((angle, strength));
        }
    }
    result
}

/// Build predicted magnitudes array for Level 3 residual coding.
pub fn build_pred_magnitudes(
    tf: &TuringField,
    coeff_h: usize,
    coeff_w: usize,
    block_size: usize,
    mag_calibration: f64,
) -> Vec<f64> {
    let scale = (block_size / 2).max(1);
    let mut preds = Vec::with_capacity(coeff_h * coeff_w);
    for y in 0..coeff_h {
        for x in 0..coeff_w {
            let gy = (y / scale).min(tf.grid_h.saturating_sub(1));
            let gx = (x / scale).min(tf.grid_w.saturating_sub(1));
            let gi = tf.inhibitor_gradient[gy * tf.grid_w + gx];
            preds.push(turing::predicted_magnitude(gi, mag_calibration));
        }
    }
    preds
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a grid whose chroma activity is concentrated in the left half:
    /// exactly the situation a single global gamma cannot describe.
    fn split_activity_grids(grid: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut luma = vec![0.0f64; grid * grid];
        let mut c1 = vec![0.0f64; grid * grid];
        let mut c2 = vec![0.0f64; grid * grid];
        for y in 0..grid {
            for x in 0..grid {
                let idx = y * grid + x;
                luma[idx] = ((x * 37 + y * 11) % 64) as f64;
                if x < grid / 2 {
                    c1[idx] = ((x * 53 + y * 29) % 64) as f64;
                    c2[idx] = ((x * 17 + y * 41) % 64) as f64;
                }
            }
        }
        (luma, c1, c2)
    }

    #[test]
    fn local_spectral_exchange_is_neutral_at_zero_strength() {
        let grid = 8usize;
        let (luma, c1, c2) = split_activity_grids(grid);
        let size_grid = vec![8u8; grid * grid];
        let blocks = crate::lot::iter_blocks(&size_grid, grid, grid, false);
        let (levels, ladder) =
            spectral_exchange_local_levels(&luma, &c1, &c2, grid, grid, &blocks, false, 0.0);
        assert!(levels.iter().all(|&level| level == 0));
        assert_eq!(ladder[0], 1.0);
    }

    /// The field may move the trade around, never change its total. The
    /// AC-and-frequency-weighted mean multiplier must stay exactly 1, which is
    /// what keeps the luma counter-gift balanced.
    #[test]
    fn local_spectral_exchange_preserves_the_global_budget() {
        let grid = 16usize;
        let (luma, c1, c2) = split_activity_grids(grid);
        let size_grid = vec![8u8; grid * grid];
        let blocks = crate::lot::iter_blocks(&size_grid, grid, grid, false);
        for &strength in &[0.25f64, 0.5, 0.75, 1.0] {
            let (levels, ladder) = spectral_exchange_local_levels(
                &luma, &c1, &c2, grid, grid, &blocks, false, strength,
            );
            assert_eq!(levels.len(), blocks.len());
            let fbar = mean_freq_norm(8);
            let mut weighted = 0.0f64;
            let mut weight_sum = 0.0f64;
            for (&level, &(_, _, size)) in levels.iter().zip(&blocks) {
                let w = (size * size - 1) as f64 * fbar;
                weighted += w * ladder[level as usize];
                weight_sum += w;
            }
            let mean = weighted / weight_sum;
            assert!(
                (mean - 1.0).abs() < 1e-9,
                "strength={strength} weighted mean multiplier = {mean}"
            );
            // The split image must actually produce a non-uniform field.
            let distinct: std::collections::BTreeSet<u8> = levels.iter().copied().collect();
            assert!(
                distinct.len() > 1,
                "strength={strength} produced a uniform field on a split image"
            );
        }
    }

    #[test]
    fn adaptive_dc_expansion_preserves_two_dimensional_layout() {
        let size_grid = vec![16u8; 16];
        let blocks = crate::lot::iter_blocks(&size_grid, 4, 4, false);
        assert_eq!(blocks.len(), 4);

        let dense =
            expand_adaptive_dc_grid(&[10.0, 20.0, 30.0, 40.0], &blocks, 4, 4, false).unwrap();
        assert_eq!(
            dense,
            vec![
                10.0, 10.0, 20.0, 20.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 30.0, 30.0,
                40.0, 40.0,
            ]
        );

        let sampled = sample_scalar_for_blocks(&dense, 4, 4, &blocks, false);
        assert_eq!(sampled, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn adaptive_turing_field_keeps_real_grid_dimensions() {
        let size_grid = vec![16u8; 24];
        let blocks = crate::lot::iter_blocks(&size_grid, 4, 6, false);
        let dc: Vec<f64> = (0..blocks.len()).map(|i| i as f64 * 10.0).collect();
        let dense = expand_adaptive_dc_grid(&dc, &blocks, 4, 6, false).unwrap();
        let tf = compute_level1(&dense, 4, 6, &HierarchyParams::default());
        assert_eq!((tf.grid_h, tf.grid_w), (4, 6));
        assert_eq!(
            build_adaptive_turing_buckets(&tf, &blocks, false).len(),
            blocks.len()
        );
        assert_eq!(
            build_adaptive_gradient_angles(&tf, &blocks, false).len(),
            blocks.len()
        );
    }

    #[test]
    fn zero_homeostasis_is_an_exact_passthrough() {
        let input = vec![0.72, 1.0, 1.41];
        assert_eq!(
            homeostatic_step_modulations(&input, &[8, 16, 32], 0.0),
            input
        );
    }

    #[test]
    fn full_homeostasis_centres_the_ac_weighted_geometric_mean() {
        let sizes = [8usize, 16, 32];
        let balanced = homeostatic_step_modulations(&[0.72, 1.0, 1.41], &sizes, 1.0);
        let (weighted_log_sum, weight_sum) = balanced.iter().zip(sizes).fold(
            (0.0, 0.0),
            |(log_sum, weights), (&modulation, block_size)| {
                let weight = (block_size * block_size - 1) as f64;
                (log_sum + weight * modulation.ln(), weights + weight)
            },
        );
        assert!((weighted_log_sum / weight_sum).abs() < 1e-12);
    }

    #[test]
    fn full_homeostasis_neutralizes_a_uniform_field() {
        let balanced = homeostatic_step_modulations(&[1.25; 4], &[16; 4], 1.0);
        assert!(balanced.iter().all(|&value| (value - 1.0).abs() < 1e-12));
    }

    #[test]
    fn zero_subsonic_wave_strength_is_bit_exactly_neutral() {
        let descriptor = SubsonicWaveDescriptor {
            angle: 0.37,
            phase: 0.61,
            coherence: 0.92,
        };
        assert_eq!(subsonic_wave_qfactors(descriptor, 16, 0.0), vec![1.0; 256]);
    }

    #[test]
    fn subsonic_wave_factors_conserve_log_budget() {
        let descriptor = SubsonicWaveDescriptor {
            angle: 0.41,
            phase: 0.73,
            coherence: 0.88,
        };
        let factors = subsonic_wave_qfactors(descriptor, 16, 0.9);
        let mean_log = factors
            .iter()
            .skip(1)
            .map(|factor| factor.ln())
            .sum::<f64>()
            / (factors.len() - 1) as f64;
        assert!(mean_log.abs() < 1e-12, "mean log factor = {mean_log}");
    }

    #[test]
    fn subsonic_phase_moves_protection_from_fundamental_to_second_mode() {
        let fundamental = subsonic_wave_qfactors(
            SubsonicWaveDescriptor {
                angle: 0.0,
                phase: 0.0,
                coherence: 1.0,
            },
            8,
            1.0,
        );
        let crest = subsonic_wave_qfactors(
            SubsonicWaveDescriptor {
                angle: 0.0,
                phase: 1.0,
                coherence: 1.0,
            },
            8,
            1.0,
        );
        assert!(fundamental[1] < fundamental[2]);
        assert!(crest[2] < crest[1]);
    }

    #[test]
    fn subsonic_descriptors_are_finite_and_bounded() {
        let grid_h = 8;
        let grid_w = 8;
        let dense: Vec<f64> = (0..grid_h)
            .flat_map(|y| {
                (0..grid_w).map(move |x| {
                    80.0 * (std::f64::consts::TAU * x as f64 / grid_w as f64).sin() + 2.0 * y as f64
                })
            })
            .collect();
        let blocks = vec![
            (0usize, 0usize, 32usize),
            (0, 32, 32),
            (32, 0, 32),
            (32, 32, 32),
        ];
        let descriptors = subsonic_wave_descriptors(&dense, grid_h, grid_w, &blocks, false);
        assert_eq!(descriptors.len(), blocks.len());
        assert!(descriptors.iter().all(|wave| {
            wave.angle.is_finite()
                && wave.phase.is_finite()
                && wave.coherence.is_finite()
                && (0.0..=1.0).contains(&wave.phase)
                && (0.0..=1.0).contains(&wave.coherence)
        }));
    }

    #[test]
    fn zero_chroma_breathing_is_an_exact_passthrough() {
        let blocks = crate::lot::iter_blocks(&vec![8u8; 4], 2, 2, false);
        let values = chroma_breathing_step_modulations(
            &[0.0, 1.0, 0.0, 1.0],
            &[0.0, 0.0, 1.0, 1.0],
            2,
            2,
            &blocks,
            false,
            0.0,
        );
        assert_eq!(values, vec![1.0; 4]);
    }

    #[test]
    fn diffusion_signature_selects_moderate_incoherent_chroma() {
        let amplitude = 24.0 * 256.0;
        let c1 = vec![amplitude, -amplitude, amplitude, -amplitude];
        let c2 = vec![0.0; 4];
        let diffuse = chroma_diffusion_signature(&c1, &c2, 0.6823);
        assert!(diffuse.repair > 0.9);
        assert!((diffuse.chroma_rms - 24.0).abs() < 1e-12);

        let coherent = chroma_diffusion_signature(&vec![amplitude; 4], &c2, 0.6823);
        assert_eq!(coherent.repair, 0.0);
    }

    #[test]
    fn diffusion_step_factor_preserves_endpoints() {
        let base = crate::golden::PHI;
        assert_eq!(chroma_diffusion_step_factor(base, 0.0), base);
        assert!((chroma_diffusion_step_factor(base, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn chroma_breathing_preserves_the_weighted_geometric_mean() {
        let size_grid = vec![8u8, 16, 16, 8, 16, 16];
        let blocks = crate::lot::iter_blocks(&size_grid, 2, 3, false);
        let luma = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let chroma = vec![0.0, 0.0, 2.0, 1.0, 1.0, 3.0];
        let mods = chroma_breathing_step_modulations(&luma, &chroma, 2, 3, &blocks, false, 1.0);
        let (log_sum, weight_sum) = mods.iter().zip(&blocks).fold(
            (0.0, 0.0),
            |(sum, weights), (&value, &(_, _, block_size))| {
                let weight = (block_size * block_size - 1) as f64;
                (sum + weight * value.ln(), weights + weight)
            },
        );
        assert!((log_sum / weight_sum).abs() < 1e-12);
    }

    #[test]
    fn independent_chroma_edges_yield_budget_to_luma_anchored_structure() {
        let blocks = crate::lot::iter_blocks(&vec![8u8; 9], 3, 3, false);
        let luma = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let chroma = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
        let mods = chroma_breathing_step_modulations(&luma, &chroma, 3, 3, &blocks, false, 1.0);
        let middle_row = (mods[3] + mods[4] + mods[5]) / 3.0;
        let shared_rows = (mods[0] + mods[1] + mods[2] + mods[6] + mods[7] + mods[8]) / 6.0;
        assert!(middle_row > shared_rows);
    }

    #[test]
    fn tensor_keeps_opposite_gradients_coherent() {
        let tf = TuringField {
            field: vec![0.0; 4],
            step_modulation: vec![1.0; 4],
            inhibitor_gradient: vec![1.0; 4],
            inhibitor_gradient_angle: vec![0.0, std::f64::consts::PI, 0.0, std::f64::consts::PI],
            grid_h: 2,
            grid_w: 2,
        };
        let tensor = build_adaptive_structure_tensors(&tf, &[(0, 0, 16)], false)[0];
        assert!(tensor.0.abs() < 1e-12);
        assert!((tensor.1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn isotropic_tensor_has_zero_coherence() {
        let tf = TuringField {
            field: vec![0.0; 4],
            step_modulation: vec![1.0; 4],
            inhibitor_gradient: vec![1.0; 4],
            inhibitor_gradient_angle: vec![
                0.0,
                std::f64::consts::FRAC_PI_2,
                std::f64::consts::PI,
                -std::f64::consts::FRAC_PI_2,
            ],
            grid_h: 2,
            grid_w: 2,
        };
        let tensor = build_adaptive_structure_tensors(&tf, &[(0, 0, 16)], false)[0];
        assert!(tensor.1.abs() < 1e-12);
    }

    #[test]
    fn qmat_blend_preserves_endpoints_and_geometric_midpoint() {
        assert_eq!(blend_qmat_factors(2.0, 8.0, 0.0), 2.0);
        assert_eq!(blend_qmat_factors(2.0, 8.0, 1.0), 8.0);
        assert!((blend_qmat_factors(2.0, 8.0, 0.5) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn zero_multiscale_is_an_exact_passthrough() {
        let local = vec![0.8, 1.2];
        assert_eq!(
            phi_multiscale_step_modulations(&local, &[1.4, 0.7], 0.0),
            local
        );
    }

    #[test]
    fn full_multiscale_uses_golden_log_weights() {
        let blended = phi_multiscale_step_modulations(&[2.0], &[8.0], 1.0)[0];
        let expected =
            (crate::golden::PHI_INV * 2.0f64.ln() + crate::golden::PHI_INV2 * 8.0f64.ln()).exp();
        assert!((blended - expected).abs() < 1e-12);
    }

    #[test]
    fn zero_spiral_surprise_is_an_exact_passthrough() {
        let input = vec![0.72, 1.0, 1.41, 0.91];
        assert_eq!(
            spiral_surprise_step_modulations(&input, &[1, 3, 0, 2], &[8, 16, 32, 16], 0.0),
            input
        );
    }

    #[test]
    fn spiral_surprise_conserves_the_ac_weighted_log_budget() {
        let input = vec![0.72, 1.0, 1.41, 0.91];
        let sizes = [8usize, 16, 32, 16];
        let redistributed = spiral_surprise_step_modulations(&input, &[1, 3, 0, 2], &sizes, 0.7);
        let weighted_log_sum = |values: &[f64]| {
            values
                .iter()
                .zip(sizes)
                .map(|(&value, block_size)| {
                    let weight = (block_size * block_size - 1) as f64;
                    weight * value.ln()
                })
                .sum::<f64>()
        };
        assert!((weighted_log_sum(&input) - weighted_log_sum(&redistributed)).abs() < 1e-10);
    }
}
