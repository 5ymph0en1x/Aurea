//! Parametric Photon Synthesis — tiled noise profiling, stripping, and re-injection.
//!
//! Encoder: estimate per-tile noise sigma, strip noise before encoding.
//! Decoder: re-inject calibrated synthetic noise to restore natural texture.

use std::io;

/// Tiled noise sigma map. Each tile is 16x16 pixels.
pub struct PhotonMap {
    /// Quantized sigma per tile (row-major). real_sigma = sigmas[i] as f64 * scale.
    pub sigmas: Vec<u8>,
    pub tile_cols: usize,
    pub tile_rows: usize,
    /// Scale factor for sigma dequantization.
    pub scale: f32,
    /// Noise re-injection ratio (0.0-1.0, typical 0.3).
    pub injection_ratio: f32,
}

const TILE_SIZE: usize = 16;
const SIGMA_THRESHOLD: f64 = 5.0;

/// Estimate per-tile noise sigma from a single channel plane (after GCT+PTF).
/// Uses Donoho-Johnstone MAD estimator on the Laplacian per tile.
pub fn estimate_tile_sigmas(plane: &[f64], h: usize, w: usize) -> PhotonMap {
    let tile_rows = (h + TILE_SIZE - 1) / TILE_SIZE;
    let tile_cols = (w + TILE_SIZE - 1) / TILE_SIZE;
    let n_tiles = tile_rows * tile_cols;

    let mut raw_sigmas = vec![0.0f64; n_tiles];

    for tr in 0..tile_rows {
        for tc in 0..tile_cols {
            let y0 = tr * TILE_SIZE;
            let x0 = tc * TILE_SIZE;
            let y1 = (y0 + TILE_SIZE).min(h);
            let x1 = (x0 + TILE_SIZE).min(w);

            // Compute Laplacian absolute values within tile (skip border pixels)
            let mut laplacian_abs = Vec::new();
            for y in y0.max(1)..y1.min(h - 1) {
                for x in x0.max(1)..x1.min(w - 1) {
                    let c = plane[y * w + x];
                    let lap = plane[(y - 1) * w + x] + plane[(y + 1) * w + x]
                            + plane[y * w + (x - 1)] + plane[y * w + (x + 1)]
                            - 4.0 * c;
                    laplacian_abs.push(lap.abs());
                }
            }

            if laplacian_abs.is_empty() {
                raw_sigmas[tr * tile_cols + tc] = 0.0;
                continue;
            }

            // MAD (Median Absolute Deviation)
            laplacian_abs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let median = laplacian_abs[laplacian_abs.len() / 2];
            // Donoho-Johnstone: sigma = MAD / (0.6745 * sqrt(20))
            // sqrt(20) = factor for 2D Laplacian kernel normalization
            let sigma = median / (0.6745 * 20.0_f64.sqrt());

            raw_sigmas[tr * tile_cols + tc] = if sigma < SIGMA_THRESHOLD { 0.0 } else { sigma };
        }
    }

    // Quantize to u8 with scale
    let max_sigma = raw_sigmas.iter().cloned().fold(0.0f64, f64::max);
    let scale = if max_sigma > 0.0 { (max_sigma / 255.0) as f32 } else { 1.0 };

    let sigmas: Vec<u8> = raw_sigmas.iter().map(|&s| {
        if s <= 0.0 { 0 } else { (s / scale as f64).round().clamp(1.0, 255.0) as u8 }
    }).collect();

    PhotonMap {
        sigmas,
        tile_cols,
        tile_rows,
        scale,
        injection_ratio: 0.3,
    }
}

/// Strip noise from a plane using per-tile sigma (5x5 sigma filter).
pub fn strip_photon_noise(plane: &mut [f64], h: usize, w: usize, map: &PhotonMap) {
    // Work on a copy for reading
    let src = plane.to_vec();

    for tr in 0..map.tile_rows {
        for tc in 0..map.tile_cols {
            let sigma_q = map.sigmas[tr * map.tile_cols + tc];
            if sigma_q == 0 {
                continue; // clean tile, skip
            }
            let sigma = sigma_q as f64 * map.scale as f64;
            let threshold = 2.0 * sigma;

            let y0 = tr * TILE_SIZE;
            let x0 = tc * TILE_SIZE;
            let y1 = (y0 + TILE_SIZE).min(h);
            let x1 = (x0 + TILE_SIZE).min(w);

            for y in y0..y1 {
                for x in x0..x1 {
                    let center = src[y * w + x];

                    // 5x5 sigma filter: mean of neighbors within threshold
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    let ky_lo = if y >= 2 { y - 2 } else { 0 };
                    let ky_hi = (y + 3).min(h);
                    let kx_lo = if x >= 2 { x - 2 } else { 0 };
                    let kx_hi = (x + 3).min(w);

                    for ky in ky_lo..ky_hi {
                        for kx in kx_lo..kx_hi {
                            let v = src[ky * w + kx];
                            if (v - center).abs() <= threshold {
                                sum += v;
                                count += 1.0;
                            }
                        }
                    }

                    let filtered = if count > 0.0 { sum / count } else { center };

                    // Wiener-style blending: gain from local variance vs noise variance
                    // Compute local variance in 5x5 window
                    let mut var_sum = 0.0;
                    let mut var_count = 0.0;
                    for ky in ky_lo..ky_hi {
                        for kx in kx_lo..kx_hi {
                            let d = src[ky * w + kx] - center;
                            var_sum += d * d;
                            var_count += 1.0;
                        }
                    }
                    let local_var = if var_count > 1.0 { var_sum / var_count } else { 0.0 };
                    let sigma2 = sigma * sigma;
                    let gain = ((local_var - sigma2).max(0.0)) / (local_var + 1e-6);

                    plane[y * w + x] = center * gain + filtered * (1.0 - gain);
                }
            }
        }
    }
}

/// Serialize a PhotonMap to bytes for bitstream embedding.
/// Format: tile_cols(u16) | tile_rows(u16) | scale(f32) | injection_ratio(f32) | rANS_compressed_sigmas
pub fn encode_photon_map(map: &PhotonMap) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(map.tile_cols as u16).to_le_bytes());
    buf.extend_from_slice(&(map.tile_rows as u16).to_le_bytes());
    buf.extend_from_slice(&map.scale.to_le_bytes());
    buf.extend_from_slice(&map.injection_ratio.to_le_bytes());
    let compressed_sigmas = crate::rans::rans_compress_bytes(&map.sigmas);
    buf.extend_from_slice(&compressed_sigmas);
    buf
}

/// Parse a PhotonMap from bytes. Returns (map, bytes_consumed).
pub fn decode_photon_map(data: &[u8]) -> io::Result<(PhotonMap, usize)> {
    if data.len() < 12 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "photon map too short"));
    }
    let tile_cols = u16::from_le_bytes([data[0], data[1]]) as usize;
    let tile_rows = u16::from_le_bytes([data[2], data[3]]) as usize;
    let scale = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let injection_ratio = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);

    let compressed = &data[12..];
    let sigmas = crate::rans::rans_decompress_bytes(compressed);
    let consumed = 12 + compressed.len();

    Ok((PhotonMap {
        sigmas,
        tile_cols,
        tile_rows,
        scale,
        injection_ratio,
    }, consumed))
}

/// Re-inject synthetic noise into a decoded plane.
/// Uses deterministic PCG for reproducibility. Channel: 0=L, 1=C1, 2=C2.
pub fn inject_photon_noise(
    plane: &mut [f64], h: usize, w: usize,
    map: &PhotonMap, channel: usize,
) {
    let injection = map.injection_ratio as f64;
    if injection <= 0.0 {
        return;
    }
    // Channel scaling: chroma gets 0.5x noise
    let channel_scale = if channel == 0 { 1.0 } else { 0.5 };

    for tr in 0..map.tile_rows {
        for tc in 0..map.tile_cols {
            let sigma_q = map.sigmas[tr * map.tile_cols + tc];
            if sigma_q == 0 {
                continue;
            }
            let sigma = sigma_q as f64 * map.scale as f64 * injection * channel_scale;

            let y0 = tr * TILE_SIZE;
            let x0 = tc * TILE_SIZE;
            let y1 = (y0 + TILE_SIZE).min(h);
            let x1 = (x0 + TILE_SIZE).min(w);

            // Deterministic PCG64 seed per tile+channel
            let seed = ((tr as u64) * 65536 + tc as u64) * 3 + channel as u64;
            let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);

            // Compute local variance for texture mask (5x5 window at tile center)
            let cy = (y0 + y1) / 2;
            let cx = (x0 + x1) / 2;
            let mut tile_var = 0.0;
            let mut tile_mean = 0.0;
            let mut tc_count: f64 = 0.0;
            let vy0 = if cy >= 2 { cy - 2 } else { 0 };
            let vy1 = (cy + 3).min(h);
            let vx0 = if cx >= 2 { cx - 2 } else { 0 };
            let vx1 = (cx + 3).min(w);
            for vy in vy0..vy1 {
                for vx in vx0..vx1 {
                    tile_mean += plane[vy * w + vx];
                    tc_count += 1.0;
                }
            }
            tile_mean /= tc_count.max(1.0);
            for vy in vy0..vy1 {
                for vx in vx0..vx1 {
                    let d = plane[vy * w + vx] - tile_mean;
                    tile_var += d * d;
                }
            }
            tile_var /= tc_count.max(1.0);

            // Texture mask: smooth regions get more noise, textured less
            let mask = 1.0 - (tile_var / (tile_var + 16.0)).min(1.0);

            for y in y0..y1 {
                for x in x0..x1 {
                    // PCG step: simple LCG for speed
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let u1 = (state >> 11) as f64 / (1u64 << 53) as f64;
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;

                    // Box-Muller for Gaussian sample
                    let r = (-2.0 * u1.max(1e-20).ln()).sqrt();
                    let theta = std::f64::consts::TAU * u2;
                    let gaussian = r * theta.cos();

                    plane[y * w + x] += gaussian * sigma * mask;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_photon_map_roundtrip() {
        let map = PhotonMap {
            sigmas: vec![0, 10, 20, 0, 5, 30, 0, 0, 15],
            tile_cols: 3,
            tile_rows: 3,
            scale: 0.12,
            injection_ratio: 0.3,
        };
        let encoded = encode_photon_map(&map);
        let (decoded, _) = decode_photon_map(&encoded).unwrap();
        assert_eq!(decoded.tile_cols, 3);
        assert_eq!(decoded.tile_rows, 3);
        assert_eq!(decoded.scale, 0.12f32);
        assert_eq!(decoded.injection_ratio, 0.3f32);
        assert_eq!(decoded.sigmas, vec![0, 10, 20, 0, 5, 30, 0, 0, 15]);
    }

    #[test]
    fn test_estimate_tile_sigmas_flat() {
        // Uniform plane → all sigmas should be 0 (below threshold)
        let plane = vec![128.0; 64 * 64];
        let map = estimate_tile_sigmas(&plane, 64, 64);
        assert_eq!(map.tile_rows, 4);
        assert_eq!(map.tile_cols, 4);
        for &s in &map.sigmas {
            assert_eq!(s, 0, "flat plane should have zero sigma");
        }
    }

    #[test]
    fn test_strip_noise_idempotent_on_clean() {
        // Clean plane (sigma=0 everywhere) should not be modified
        let mut plane = vec![128.0; 32 * 32];
        let map = estimate_tile_sigmas(&plane, 32, 32);
        let before = plane.clone();
        strip_photon_noise(&mut plane, 32, 32, &map);
        assert_eq!(plane, before);
    }
}
