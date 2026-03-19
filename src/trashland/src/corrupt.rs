/// Satellite signal loss corruption effects.
///
/// Five layers activated progressively as signal degrades:
///   signal < 80 : macroblocking (zero wavelet detail blocks)
///   signal < 60 : chroma glitch (corrupt VQ centroids + band swap)
///   signal < 45 : line desync (horizontal row shifts)
///   signal < 30 : partial freeze (blocks replaced by mean color)
///   signal < 15 : pixel explosion (random RGB patches + label corruption)

/// Deterministic PRNG (splitmix64).
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Random u32 in [0, max).
    pub fn next_u32(&mut self, max: u32) -> u32 {
        (self.next_u64() >> 32) as u32 % max
    }

    /// Random f64 in [0.0, 1.0).
    #[allow(dead_code)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Random i32 in [-range, +range].
    pub fn next_i32_range(&mut self, range: i32) -> i32 {
        if range == 0 { return 0; }
        let r = self.next_u32((range * 2 + 1) as u32) as i32;
        r - range
    }
}

// ======================================================================
// Layer 1: Macroblocking (signal < 80)
// Zero out wavelet detail coefficients in random block-aligned regions.
// ======================================================================

/// Corrupt wavelet detail bands by zeroing 32x32 blocks.
/// `intensity` in [0.0, 1.0] controls what fraction of blocks are killed.
pub fn macroblocking(
    detail_flat: &mut [f64],
    h: usize,
    w: usize,
    intensity: f64,
    rng: &mut Rng,
) {
    const BLOCK: usize = 32;
    let bh = (h + BLOCK - 1) / BLOCK;
    let bw = (w + BLOCK - 1) / BLOCK;
    let n_blocks = bh * bw;

    let n_kill = ((n_blocks as f64 * intensity * 0.4).round() as usize).min(n_blocks);

    for _ in 0..n_kill {
        let by = rng.next_u32(bh as u32) as usize;
        let bx = rng.next_u32(bw as u32) as usize;
        let y0 = by * BLOCK;
        let x0 = bx * BLOCK;
        let y1 = (y0 + BLOCK).min(h);
        let x1 = (x0 + BLOCK).min(w);

        for y in y0..y1 {
            for x in x0..x1 {
                detail_flat[y * w + x] = 0.0;
            }
        }
    }
}

/// Apply macroblocking to all three subbands (LH, HL, HH) of a wavelet level.
pub fn macroblocking_subbands(
    lh: &mut [f64], lh_h: usize, lh_w: usize,
    hl: &mut [f64], hl_h: usize, hl_w: usize,
    hh: &mut [f64], hh_h: usize, hh_w: usize,
    intensity: f64,
    rng: &mut Rng,
) {
    macroblocking(lh, lh_h, lh_w, intensity, rng);
    macroblocking(hl, hl_h, hl_w, intensity, rng);
    macroblocking(hh, hh_h, hh_w, intensity, rng);
}

// ======================================================================
// Layer 2: Chroma glitch (signal < 60)
// Corrupt VQ centroids and swap horizontal chroma bands.
// ======================================================================

/// Shift random chroma centroids by aberrant values.
/// Produces the green/pink/cyan bands characteristic of satellite dropout.
pub fn corrupt_chroma_centroids(
    centroids: &mut [f64],
    intensity: f64,
    rng: &mut Rng,
) {
    let n = centroids.len();
    let n_corrupt = ((n as f64 * intensity * 0.5).round() as usize).max(1).min(n);

    for _ in 0..n_corrupt {
        let idx = rng.next_u32(n as u32) as usize;
        let shift = rng.next_i32_range((60.0 * intensity) as i32) as f64;
        centroids[idx] = (centroids[idx] + shift).clamp(0.0, 255.0);
    }
    // Re-sort: fibonacci_correction requires sorted centroids.
    // The color glitch effect is preserved because the VALUES are wrong.
    centroids.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
}

/// Swap horizontal bands between C1 and C2 chroma planes.
/// Creates the characteristic color-banding of satellite signal loss.
pub fn chroma_band_swap(
    c1_flat: &mut [f64],
    c2_flat: &mut [f64],
    h: usize,
    w: usize,
    intensity: f64,
    rng: &mut Rng,
) {
    let n_bands = ((h as f64 * intensity * 0.15).round() as usize).max(1);

    for _ in 0..n_bands {
        let y_start = rng.next_u32(h as u32) as usize;
        let band_h = (rng.next_u32(16) as usize + 4).min(h - y_start);
        let y_end = (y_start + band_h).min(h);

        for y in y_start..y_end {
            for x in 0..w {
                let idx = y * w + x;
                std::mem::swap(&mut c1_flat[idx], &mut c2_flat[idx]);
            }
        }
    }
}

/// Shift entire horizontal bands of a chroma plane by a random offset.
#[allow(dead_code)]
pub fn chroma_horizontal_shift(
    plane: &mut [f64],
    h: usize,
    w: usize,
    intensity: f64,
    rng: &mut Rng,
) {
    let n_bands = ((h as f64 * intensity * 0.1).round() as usize).max(1);

    for _ in 0..n_bands {
        let y_start = rng.next_u32(h as u32) as usize;
        let band_h = (rng.next_u32(8) as usize + 2).min(h - y_start);
        let y_end = (y_start + band_h).min(h);
        let shift = rng.next_i32_range((w as f64 * 0.1 * intensity) as i32);

        for y in y_start..y_end {
            let row_start = y * w;
            let row: Vec<f64> = plane[row_start..row_start + w].to_vec();
            for x in 0..w {
                let src_x = ((x as i32 - shift).rem_euclid(w as i32)) as usize;
                plane[row_start + x] = row[src_x];
            }
        }
    }
}

// ======================================================================
// Layer 3: Line desync (signal < 45)
// Horizontal row shifts on the final RGB image.
// ======================================================================

/// Shift random horizontal bands of the RGB image.
/// Simulates decoder losing horizontal sync.
pub fn line_desync(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    intensity: f64,
    rng: &mut Rng,
) {
    let n_bands = ((height as f64 * intensity * 0.12).round() as usize).max(1);

    for _ in 0..n_bands {
        let y_start = rng.next_u32(height as u32) as usize;
        let band_h = (rng.next_u32(12) as usize + 1).min(height - y_start);
        let y_end = (y_start + band_h).min(height);
        let shift = rng.next_i32_range((width as f64 * 0.15 * intensity) as i32);

        if shift == 0 { continue; }

        for y in y_start..y_end {
            let row_start = y * width * 3;
            let row: Vec<u8> = rgb[row_start..row_start + width * 3].to_vec();
            for x in 0..width {
                let src_x = ((x as i32 - shift).rem_euclid(width as i32)) as usize;
                rgb[row_start + x * 3]     = row[src_x * 3];
                rgb[row_start + x * 3 + 1] = row[src_x * 3 + 1];
                rgb[row_start + x * 3 + 2] = row[src_x * 3 + 2];
            }
        }
    }
}

// ======================================================================
// Layer 4: Partial freeze (signal < 30)
// Replace rectangular blocks with their mean color.
// ======================================================================

/// Replace random blocks with their mean color (solid rectangle).
/// Simulates decoder freezing on stale/missing data.
pub fn freeze_blocks(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    intensity: f64,
    rng: &mut Rng,
) {
    let block_min = 16usize;
    let block_max = 64usize;
    let n_blocks = ((width as f64 * height as f64 / 2048.0 * intensity).round() as usize).max(1);

    for _ in 0..n_blocks {
        let bw = rng.next_u32((block_max - block_min) as u32) as usize + block_min;
        let bh = rng.next_u32((block_max - block_min) as u32) as usize + block_min;
        let bx = rng.next_u32(width as u32) as usize;
        let by = rng.next_u32(height as u32) as usize;
        let x1 = (bx + bw).min(width);
        let y1 = (by + bh).min(height);

        // Compute mean color of the block
        let mut sum_r = 0u64;
        let mut sum_g = 0u64;
        let mut sum_b = 0u64;
        let mut count = 0u64;

        for y in by..y1 {
            for x in bx..x1 {
                let idx = (y * width + x) * 3;
                sum_r += rgb[idx] as u64;
                sum_g += rgb[idx + 1] as u64;
                sum_b += rgb[idx + 2] as u64;
                count += 1;
            }
        }

        if count == 0 { continue; }
        let mean_r = (sum_r / count) as u8;
        let mean_g = (sum_g / count) as u8;
        let mean_b = (sum_b / count) as u8;

        // Add slight color aberration (satellite decoder artifact)
        let aberr_r = (mean_r as i16 + rng.next_i32_range((20.0 * intensity) as i32) as i16)
            .clamp(0, 255) as u8;
        let aberr_g = (mean_g as i16 + rng.next_i32_range((20.0 * intensity) as i32) as i16)
            .clamp(0, 255) as u8;
        let aberr_b = (mean_b as i16 + rng.next_i32_range((20.0 * intensity) as i32) as i16)
            .clamp(0, 255) as u8;

        // Fill block with solid color
        for y in by..y1 {
            for x in bx..x1 {
                let idx = (y * width + x) * 3;
                rgb[idx]     = aberr_r;
                rgb[idx + 1] = aberr_g;
                rgb[idx + 2] = aberr_b;
            }
        }
    }
}

// ======================================================================
// Layer 5: Pixel explosion (signal < 15)
// Random RGB patches + massive corruption.
// ======================================================================

/// Inject patches of random RGB pixels.
/// The satellite signal is nearly dead.
pub fn pixel_explosion(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    intensity: f64,
    rng: &mut Rng,
) {
    let n_patches = ((width as f64 * height as f64 / 1024.0 * intensity).round() as usize).max(1);

    for _ in 0..n_patches {
        let pw = rng.next_u32(32) as usize + 4;
        let ph = rng.next_u32(8) as usize + 1;
        let px = rng.next_u32(width as u32) as usize;
        let py = rng.next_u32(height as u32) as usize;
        let x1 = (px + pw).min(width);
        let y1 = (py + ph).min(height);

        // Pick an aberrant base color (bright green, magenta, cyan -- satellite classics)
        let base_color: (u8, u8, u8) = match rng.next_u32(5) {
            0 => (0, 255, 0),     // bright green
            1 => (255, 0, 255),   // magenta
            2 => (0, 255, 255),   // cyan
            3 => (255, 255, 0),   // yellow
            _ => (255, 0, 0),     // red
        };

        for y in py..y1 {
            for x in px..x1 {
                let idx = (y * width + x) * 3;
                // Mix base color with random noise
                let noise_r = rng.next_u32(80) as i16 - 40;
                let noise_g = rng.next_u32(80) as i16 - 40;
                let noise_b = rng.next_u32(80) as i16 - 40;
                rgb[idx]     = (base_color.0 as i16 + noise_r).clamp(0, 255) as u8;
                rgb[idx + 1] = (base_color.1 as i16 + noise_g).clamp(0, 255) as u8;
                rgb[idx + 2] = (base_color.2 as i16 + noise_b).clamp(0, 255) as u8;
            }
        }
    }
}

/// Corrupt Paeth label residuals (adds garbage to VQ reconstruction).
pub fn corrupt_paeth_labels(
    pred: &mut [i16],
    intensity: f64,
    rng: &mut Rng,
) {
    let n = pred.len();
    let n_corrupt = ((n as f64 * intensity * 0.08).round() as usize).max(1).min(n);

    for _ in 0..n_corrupt {
        let idx = rng.next_u32(n as u32) as usize;
        pred[idx] = pred[idx].wrapping_add(rng.next_i32_range(5) as i16);
    }
}

// ======================================================================
// Master corruption dispatcher
// ======================================================================

/// Corruption parameters derived from signal strength.
pub struct CorruptionParams {
    /// Macroblocking intensity (0 = none, 1 = maximum)
    pub macroblock: f64,
    /// Chroma glitch intensity
    pub chroma: f64,
    /// Line desync intensity
    pub desync: f64,
    /// Freeze block intensity
    pub freeze: f64,
    /// Pixel explosion intensity
    pub explosion: f64,
}

impl CorruptionParams {
    /// Derive corruption parameters from signal strength (0-100).
    /// 100 = perfect signal, 0 = total loss.
    pub fn from_signal(signal: u8) -> Self {
        let s = signal as f64;
        Self {
            macroblock:  if s < 80.0 { (80.0 - s) / 80.0 } else { 0.0 },
            chroma:      if s < 60.0 { (60.0 - s) / 60.0 } else { 0.0 },
            desync:      if s < 45.0 { (45.0 - s) / 45.0 } else { 0.0 },
            freeze:      if s < 30.0 { (30.0 - s) / 30.0 } else { 0.0 },
            explosion:   if s < 15.0 { (15.0 - s) / 15.0 } else { 0.0 },
        }
    }
}
