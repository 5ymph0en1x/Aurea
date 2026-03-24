/// Trashland pipeline: encode through AUREA v12, decode, corrupt.
///
/// The corruption layers simulate satellite transmission failure:
///   1. Encode the image with AUREA v12 at medium quality (lossy compression)
///   2. Decode it back (this gives the "clean compressed" baseline)
///   3. Apply 5 progressive corruption layers based on signal strength

use crate::corrupt::{self, Rng};

/// Run the full trashland pipeline.
/// Returns the corrupted RGB image as Vec<u8>.
pub fn trashland_pipeline(
    rgb: &[u8],
    width: usize,
    height: usize,
    signal: u8,
    seed: u64,
) -> Vec<u8> {
    let mut rng = Rng::new(seed);

    // Step 1: Encode with AUREA v12 at quality proportional to signal
    // Higher signal = higher quality base (less codec artifacts to start with)
    let encode_quality = (signal as u16 * 80 / 100 + 15).min(90) as u8;
    let params = aurea_core::aurea_encoder::AureaEncoderParams {
        quality: encode_quality,
        n_representatives: aurea_core::aurea_encoder::quality_to_n_repr(encode_quality),
        geometric: false,
    };

    let encoded = aurea_core::aurea_encoder::encode_aur2_v12(rgb, width, height, &params)
        .expect("AUREA v12 encode failed");

    // Step 2: Decode back to RGB
    let decoded = aurea_core::decode_aurea(&encoded.aurea_data)
        .expect("AUREA decode failed");
    let mut result = decoded.rgb;

    // Step 3: Compute corruption intensity (0 = pristine, 1 = destroyed)
    let intensity = 1.0 - (signal as f64 / 100.0);

    // Layer 1: Macroblocking (signal < 80)
    // Applied on the decoded image as block-mean replacement
    if signal < 80 {
        let macro_intensity = ((80 - signal) as f64 / 80.0).min(1.0);
        // Zero out random 32x32 blocks (replace with block mean)
        corrupt::freeze_blocks(&mut result, width, height, macro_intensity * 0.3, &mut rng);
    }

    // Layer 2: Chroma glitch (signal < 60)
    // Swap color channels in horizontal bands
    if signal < 60 {
        let chroma_intensity = ((60 - signal) as f64 / 60.0).min(1.0);
        // Work on separate planes for chroma swap
        let mut r_plane: Vec<f64> = (0..width * height).map(|i| result[i * 3] as f64).collect();
        let mut g_plane: Vec<f64> = (0..width * height).map(|i| result[i * 3 + 1] as f64).collect();
        let mut b_plane: Vec<f64> = (0..width * height).map(|i| result[i * 3 + 2] as f64).collect();

        corrupt::chroma_band_swap(&mut r_plane, &mut b_plane, height, width, chroma_intensity, &mut rng);
        corrupt::chroma_horizontal_shift(&mut g_plane, height, width, chroma_intensity * 0.5, &mut rng);

        for i in 0..width * height {
            result[i * 3]     = r_plane[i].clamp(0.0, 255.0) as u8;
            result[i * 3 + 1] = g_plane[i].clamp(0.0, 255.0) as u8;
            result[i * 3 + 2] = b_plane[i].clamp(0.0, 255.0) as u8;
        }
    }

    // Layer 3: Line desync (signal < 45)
    if signal < 45 {
        let desync_intensity = ((45 - signal) as f64 / 45.0).min(1.0);
        corrupt::line_desync(&mut result, width, height, desync_intensity, &mut rng);
    }

    // Layer 4: Partial freeze (signal < 30)
    if signal < 30 {
        let freeze_intensity = ((30 - signal) as f64 / 30.0).min(1.0);
        corrupt::freeze_blocks(&mut result, width, height, freeze_intensity, &mut rng);
    }

    // Layer 5: Pixel explosion (signal < 15)
    if signal < 15 {
        let explosion_intensity = ((15 - signal) as f64 / 15.0).min(1.0);
        corrupt::pixel_explosion(&mut result, width, height, explosion_intensity, &mut rng);
    }

    result
}
