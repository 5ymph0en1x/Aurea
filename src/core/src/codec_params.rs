//! Centralized codec parameters for AUREA.
//! All tunable constants in one place for systematic calibration.

/// Pipeline selector: which encoding algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pipeline {
    /// LOT 16x16 + structural codon (AUR2 version 3/4)
    LotAdn4,
    /// Hexagonal pyramid (AUR2 version 7)
    HexPyramid,
    /// Edge-energy snake-scan DPCM + hex oracle (AUR2 version 8)
    EdgeEnergy,
    /// DNA-guided edge-energy (AUR2 version 8, mode 2)
    EdgeEnergyDna,
    /// Optica v11: photon synthesis + capillary chroma + hyper-sparse rANS
    Optica,
}

impl Default for Pipeline {
    fn default() -> Self { Pipeline::EdgeEnergy }
}

/// Centralized codec parameters.
#[derive(Debug, Clone)]
pub struct CodecParams {
    pub pipeline: Pipeline,
    pub quality: u8,

    // --- Golden constants ---
    pub ptf_gamma: f64,
    pub dead_zone: f64,

    // --- Edge-energy specific ---
    pub edge_step_c_ratio: f64,  // chroma step = luma step * this
    pub gas_coarsen_factor: f64, // gas areas get step * this

    // --- Hex oracle ---
    pub hex_radius: usize,

    // --- Optica v11 ---
    pub photon_injection_ratio: f64,  // decoder noise re-injection level (0.0-1.0)
    pub capillary_sigma_guide: f64,   // edge sensitivity for chroma SOR diffusion
    pub capillary_sor_iter: usize,    // SOR iterations for chroma reconstruction

    // --- Post-decode sharpening ---
    pub post_sharpen: u8,        // 0 = off, 1 = unsharp, 2 = casp, 3 = adaptive
    pub sharpen_strength: f64,   // force (0.0..1.0 for CASP, larger for others)
}

impl Default for CodecParams {
    fn default() -> Self {
        CodecParams {
            pipeline: Pipeline::EdgeEnergy,
            quality: 75,
            ptf_gamma: 0.65,
            dead_zone: 0.22,
            edge_step_c_ratio: 0.3,
            gas_coarsen_factor: 3.0,
            hex_radius: 5,
            photon_injection_ratio: 0.3,
            capillary_sigma_guide: 10.0,
            capillary_sor_iter: 30,
            post_sharpen: 2,       // CASP by default
            sharpen_strength: 0.5,
        }
    }
}

impl CodecParams {
    pub fn with_quality(quality: u8) -> Self {
        CodecParams { quality, ..Default::default() }
    }

    pub fn with_pipeline(pipeline: Pipeline, quality: u8) -> Self {
        CodecParams { pipeline, quality, ..Default::default() }
    }
}
