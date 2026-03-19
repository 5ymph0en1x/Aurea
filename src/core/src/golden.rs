//! Golden ratio constants used across the codec.

/// Golden ratio: phi = (1 + sqrt(5)) / 2
pub const PHI: f64 = 1.618033988749895;
/// Inverse of the golden ratio: phi^{-1} = phi - 1
pub const PHI_INV: f64 = 0.6180339887498949;
/// Square of the inverse: phi^{-2} = 2 - phi
pub const PHI_INV2: f64 = 0.3819660112501051;
/// Cube of the inverse: phi^{-3} = phi^{-2} * phi^{-1}
pub const PHI_INV3: f64 = 0.2360679774997897;

/// Perceptual Transfer Function (PTF) gamma for luminance.
/// Applied before wavelet transform to expand dark levels,
/// giving perceptually uniform quantization (Weber-Fechner).
pub const PTF_GAMMA: f64 = 0.65;
pub const PTF_GAMMA_INV: f64 = 1.0 / PTF_GAMMA;

