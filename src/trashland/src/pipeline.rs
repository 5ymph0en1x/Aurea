/// Trashland pipeline: encode through Aurea, corrupt, decode.
///
/// Legacy pipeline stubbed out for v10 rewrite.
/// The corruption happens at two levels:
/// 1. Wavelet domain (before inverse transform) -- organic codec artifacts
/// 2. Pixel domain (after reconstruction) -- brutal spatial glitches

/// Run the full trashland pipeline.
/// Returns the corrupted RGB image as Vec<u8>.
pub fn trashland_pipeline(
    _rgb: &[u8],
    _width: usize,
    _height: usize,
    _signal: u8,
    _seed: u64,
) -> Vec<u8> {
    todo!("v10 rewrite — trashland pipeline will be rebuilt on AUR2 primitives")
}
