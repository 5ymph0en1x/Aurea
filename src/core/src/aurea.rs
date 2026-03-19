/// Decoder for the AUREA format (.aur).
///
/// AUR2 v10 pipeline: AUR2 header -> metadata preamble -> detail subbands
/// (primitives + rANS residuals) -> LL subbands (patches + rANS residuals)
/// -> wavelet recompose -> anti-ring -> GCT inverse -> RGB.

pub const AUREA_MAGIC: &[u8; 4] = b"AURA";

/// Parse the decompressed payload of a legacy .aur file.
pub fn parse_aurea_payload(_data: &[u8]) -> std::io::Result<crate::bitstream::X267V6Stream> {
    todo!("v10 rewrite -- legacy AURA format not supported in AUR2 decoder")
}
