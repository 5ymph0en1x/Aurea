//! rANS (range Asymmetric Numeral Systems) entropy coder.
//!
//! Byte-aligned rANS with 32-bit state, zero external dependencies.
//! Replaces LZMA for the AUREA codec detail bands and Paeth residuals.
//!
//! Two main coding modes:
//! - Band coding: context-adaptive (run-class x prev_nz x energy_bucket)
//! - Paeth coding: Laplacian model (sign x magnitude bucket)

use crate::spectral_envelope::{
    AMPLITUDE_STATES, CODON_STATES, PHASE_CLASS_COUNT, PHASE_CONTEXT_ENERGY_ALL,
    PHASE_CONTEXT_SEPARABLE_ENERGY, PHASE_CONTEXT_SEPARABLE_MEMORY, PHASE_CONTEXT_SEPARABLE_SIZE,
    PHASE_CONTEXT_SEPARABLE_TURING, PHASE_NONE, PHASE_SEPARABLE, SpectralCursor,
    SpectralEnvelopeConfig, SpectralNeighborTables, SpectralSegment,
};

// ======================================================================
// Constants
// ======================================================================

const RANS_BYTE_L: u32 = 1 << 23;
const PROB_BITS: u32 = 14;
const PROB_SCALE: u32 = 1 << PROB_BITS; // 16384

// Context model
const N_CTX_ZERO: usize = 32; // run_class(4) x prev_nz(2) x energy_bucket(4)
const N_CTX_PM1: usize = 8; // energy_bucket(4) x prev_was_pm1(2)
const ADAPT_SHIFT: u32 = 4; // adaptation rate: 1/16 per symbol
const ENERGY_ALPHA: u32 = 218; // IIR smoothing: alpha/256
const ENERGY_SCALE: u32 = 256;

// v11 expanded context model (hyper-sparse: 6 run buckets instead of 4)
const N_CTX_ZERO_V11: usize = 48; // run_class_v11(6) x prev_nz(2) x energy_bucket(4)

// v12 Bayesian predictive hierarchy context model
const N_CTX_ZERO_V12: usize = 128; // run_class(4) x prev_nz(2) x energy_bucket(4) x turing_bucket(4)
const N_CTX_PM1_V12: usize = 32; // energy_bucket(4) x prev_was_pm1(2) x turing_bucket(4)

/// Above this quality, dense AC streams already provide enough sequential
/// evidence to v12 and the extra parent split becomes neutral or unstable.
pub const WAVELET_PARENT_MAX_QUALITY: u8 = 70;

/// In sparse streams the mere support of the earlier color components is the
/// strongest witness. Once the quantizer retains more amplitudes, their coarse
/// magnitude becomes more predictive than a binary support mask.
pub const VECTOR_CODON_SUPPORT_MAX_QUALITY: u8 = 40;

/// Fixed v19 prior for the C1->C2 phase agreement residual. The GCT chroma
/// axes are predominantly anti-phase, so agreement starts at 1/8 and then
/// adapts within each progressive packet.
pub const PHASE_INITIAL_P_SAME: u32 = PROB_SCALE / 8;
pub const PHASE_ADAPT_SHIFT: u32 = 6;
/// v20 C1 spatial phase continuation: empirical `9/16` prior with slow
/// adaptation, derived from the Fourier/DCT identity across vertical blocks.
pub const FOURIER_PHASE_INITIAL_P_SAME: u32 = 9 * PROB_SCALE / 16;
pub const FOURIER_PHASE_ADAPT_SHIFT: u32 = 8;

#[inline]
fn vector_codon_magnitude_class(value: i16) -> u8 {
    match value.unsigned_abs() {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        _ => 3,
    }
}

/// Zero-bit causal class for the Lφ/C1/C2 color codon. Components are coded
/// in that order, so every input used here is already known by the decoder.
#[inline]
pub fn vector_codon_class(quality: u8, channel: usize, l_value: i16, c1_value: i16) -> u8 {
    if channel == 0 {
        return 0;
    }
    if quality <= VECTOR_CODON_SUPPORT_MAX_QUALITY {
        if channel == 1 {
            u8::from(l_value != 0)
        } else {
            u8::from(l_value != 0) | (u8::from(c1_value != 0) << 1)
        }
    } else if channel == 1 {
        vector_codon_magnitude_class(l_value)
    } else {
        vector_codon_magnitude_class(l_value).max(vector_codon_magnitude_class(c1_value))
    }
}

// Paeth context model
const N_PAETH_CTX: usize = 8; // sign(2) x mag_bucket(4)
const PAETH_ADAPT_SHIFT: u32 = 5; // slower adaptation for Paeth (1/32)

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RansDecodeError {
    Truncated,
    InvalidState,
    InvalidContextLength { expected: usize, got: usize },
    InvalidExpGolombPrefix(u32),
    TrailingData { consumed: usize, available: usize },
}

impl std::fmt::Display for RansDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "truncated rANS stream"),
            Self::InvalidState => write!(f, "invalid rANS state"),
            Self::InvalidContextLength { expected, got } => {
                write!(
                    f,
                    "rANS context length mismatch: expected {expected}, got {got}"
                )
            }
            Self::InvalidExpGolombPrefix(n) => write!(f, "invalid Exp-Golomb prefix length {n}"),
            Self::TrailingData {
                consumed,
                available,
            } => {
                write!(
                    f,
                    "rANS stream has trailing data: consumed {consumed} of {available} bytes"
                )
            }
        }
    }
}

impl std::error::Error for RansDecodeError {}

// ======================================================================
// rANS Encoder
// ======================================================================

pub struct RansEncoder {
    state: u32,
    buf: Vec<u8>,
}

impl RansEncoder {
    pub fn new() -> Self {
        Self {
            state: RANS_BYTE_L,
            buf: Vec::new(),
        }
    }

    /// Encode one symbol with cumulative frequency `cum_freq` and frequency `freq`.
    /// Both are in [0, PROB_SCALE). freq must be > 0.
    #[inline]
    pub fn put(&mut self, cum_freq: u32, freq: u32) {
        debug_assert!(freq > 0, "freq must be > 0");
        debug_assert!(
            cum_freq + freq <= PROB_SCALE,
            "cum_freq + freq must be <= PROB_SCALE"
        );

        // Renormalize: output bytes while state is too large
        let max_state = ((RANS_BYTE_L >> PROB_BITS) << 8) * freq;
        while self.state >= max_state {
            self.buf.push((self.state & 0xFF) as u8);
            self.state >>= 8;
        }
        // Encode
        self.state = (self.state / freq) * PROB_SCALE + (self.state % freq) + cum_freq;
    }

    /// Encode a raw bit (probability 1/2).
    #[inline]
    pub fn put_bit(&mut self, bit: bool) {
        let half = PROB_SCALE / 2;
        self.put(if bit { half } else { 0 }, half);
    }

    /// Finish encoding. Flushes state and reverses the buffer.
    /// Returns the compressed bytes.
    pub fn finish(mut self) -> Vec<u8> {
        // Flush final state as 4 bytes (little-endian into the reversed buffer)
        self.buf.push((self.state >> 0) as u8);
        self.buf.push((self.state >> 8) as u8);
        self.buf.push((self.state >> 16) as u8);
        self.buf.push((self.state >> 24) as u8);
        // rANS encodes in reverse order, so reverse the output
        self.buf.reverse();
        self.buf
    }
}

// ======================================================================
// rANS Decoder
// ======================================================================

pub struct RansDecoder<'a> {
    state: u32,
    data: &'a [u8],
    pos: usize,
}

impl<'a> RansDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        // Read the initial state from the first 4 bytes (big-endian after reversal).
        // The encoder writes state bytes LE then reverses, so the first 4 bytes
        // of the output are the state in big-endian order.
        let b0 = data.get(0).copied().unwrap_or(0) as u32;
        let b1 = data.get(1).copied().unwrap_or(0) as u32;
        let b2 = data.get(2).copied().unwrap_or(0) as u32;
        let b3 = data.get(3).copied().unwrap_or(0) as u32;
        let state = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
        Self {
            state,
            data,
            pos: 4,
        }
    }

    /// Strict constructor used by file decoders.  A non-empty rANS stream
    /// always starts with a complete normalized 32-bit state.
    pub fn new_checked(data: &'a [u8]) -> Result<Self, RansDecodeError> {
        if data.len() < 4 {
            return Err(RansDecodeError::Truncated);
        }
        let dec = Self::new(data);
        if dec.state < RANS_BYTE_L {
            return Err(RansDecodeError::InvalidState);
        }
        Ok(dec)
    }

    /// Get the cumulative frequency from the current state (for symbol lookup).
    #[inline]
    pub fn get(&self) -> u32 {
        self.state & (PROB_SCALE - 1)
    }

    /// Advance the decoder state after identifying the symbol.
    #[inline]
    pub fn advance(&mut self, cum_freq: u32, freq: u32) {
        // Decode step: invert the encoder's state transformation
        self.state = freq * (self.state >> PROB_BITS) + (self.state & (PROB_SCALE - 1)) - cum_freq;
        // Renormalize
        while self.state < RANS_BYTE_L {
            let byte = self.data.get(self.pos).copied().unwrap_or(0) as u32;
            self.pos += 1;
            self.state = (self.state << 8) | byte;
        }
    }

    /// Checked counterpart of `advance`: never synthesizes bytes after EOF
    /// and rejects arithmetic/state corruption.
    #[inline]
    pub fn advance_checked(&mut self, cum_freq: u32, freq: u32) -> Result<(), RansDecodeError> {
        if freq == 0 || cum_freq.checked_add(freq).is_none_or(|v| v > PROB_SCALE) {
            return Err(RansDecodeError::InvalidState);
        }
        let low = self.state & (PROB_SCALE - 1);
        let remainder = low
            .checked_sub(cum_freq)
            .ok_or(RansDecodeError::InvalidState)?;
        if remainder >= freq {
            return Err(RansDecodeError::InvalidState);
        }
        self.state = freq
            .checked_mul(self.state >> PROB_BITS)
            .and_then(|v| v.checked_add(remainder))
            .ok_or(RansDecodeError::InvalidState)?;
        while self.state < RANS_BYTE_L {
            let byte = self
                .data
                .get(self.pos)
                .copied()
                .ok_or(RansDecodeError::Truncated)? as u32;
            self.pos += 1;
            self.state = (self.state << 8) | byte;
        }
        Ok(())
    }

    /// Decode a raw bit (probability 1/2).
    #[inline]
    pub fn get_bit(&mut self) -> bool {
        let half = PROB_SCALE / 2;
        let cum = self.get();
        let bit = cum >= half;
        self.advance(if bit { half } else { 0 }, half);
        bit
    }

    #[inline]
    pub fn get_bit_checked(&mut self) -> Result<bool, RansDecodeError> {
        let half = PROB_SCALE / 2;
        let cum = self.get();
        let bit = cum >= half;
        self.advance_checked(if bit { half } else { 0 }, half)?;
        Ok(bit)
    }

    /// Returns the number of bytes consumed from the input so far.
    pub fn pos(&self) -> usize {
        self.pos
    }
}

// ======================================================================
// Adaptive Context Model (detail bands)
// ======================================================================

/// Classify run length into 4 buckets: 0, 1, 2-3, 4+
#[inline]
fn run_class(run_zeros: u32) -> usize {
    match run_zeros {
        0 => 0,
        1 => 1,
        2 | 3 => 2,
        _ => 3,
    }
}

/// Classify run length into 6 buckets for v11 hyper-sparse contexts.
/// Extra granularity for long zero-runs (denoised gas regions).
#[inline]
fn run_class_v11(run_zeros: u32) -> usize {
    match run_zeros {
        0 => 0,
        1 => 1,
        2 | 3 => 2,
        4..=15 => 3,
        16..=63 => 4,
        _ => 5,
    }
}

/// Classify energy into 4 buckets.
#[inline]
fn energy_bucket(energy: u32) -> usize {
    // energy is in [0, 256] scale
    if energy < 16 {
        0
    } else if energy < 64 {
        1
    } else if energy < 160 {
        2
    } else {
        3
    }
}

// ============================================================================
// Cycle 27 — activity context
// ============================================================================
//
// The `turing_bucket` context axis is derived from `turing::compute_turing_field`,
// which runs Sobel on the *DC grid*. That measures how a block differs from its
// neighbours, never the structure inside it: a block of fine texture has a block
// mean identical to its neighbours and reads as flat. Measured at q40 over the
// corpus (248 880 blocks), rank correlation with per-block squared error is
// +0.389 for the DC gradient against +0.887 for intra-block activity.
//
// The EOB position is exactly that activity — it is the count of coded
// coefficients in the block — and the decoder reads every EOB of every channel
// before it decodes any AC coefficient. So this context costs no bits, needs no
// new side information, and is causally available on both sides.

/// Energy to restart the IIR from when entering a block.
///
/// `energy_bucket` classifies >=160 as its loudest bucket; 176 sits just above
/// that with margin, and the plateau measured flat across 160-240. Grading the
/// value by the block's activity was tried and lost to a flat constant: the
/// effect is not "how loud is this block" but "the opening coefficients of any
/// block are its loudest", which the carried-over IIR state contradicted.
const BLOCK_ENERGY_PRIME: u32 = 176;

/// Energy left behind by a block's *skeleton* — the radar echo of the ping.
///
/// AUREA is progressive: layer 0 carries the first `PROGRESSIVE_SKELETON_COEFFS`
/// coefficients of every block and is fully decoded before any detail layer.
/// Yet each layer opens its own rANS stream with flat tables, so the detail
/// coefficients of a block are coded knowing nothing about the low frequencies
/// already delivered for that same block — information both sides hold.
///
/// Running the coder's own IIR over the decoded skeleton answers "how loud did
/// this block come back", and that is what the detail layer should open on.
/// Zero bits: the ping was already paid for.
pub fn skeleton_echo(skeleton: &[i16]) -> u32 {
    let mut energy = BLOCK_ENERGY_PRIME;
    for &q in skeleton {
        let sample = (q.unsigned_abs() as u32 * 32).min(ENERGY_SCALE);
        energy = (ENERGY_ALPHA * energy + (ENERGY_SCALE - ENERGY_ALPHA) * sample) / ENERGY_SCALE;
    }
    energy
}

/// Activity score of a block: its EOB as a fraction of the block's capacity,
/// on a 0..4096 fixed-point scale.
///
/// Normalizing by capacity is what makes 8x8, 16x16 and 32x32 blocks
/// comparable — a raw EOB of 50 is a dense 8x8 and a nearly empty 32x32.
#[inline]
pub fn activity_score(eob: usize, capacity: usize) -> u32 {
    if capacity == 0 {
        return 0;
    }
    ((eob.min(capacity) as u64 * 4096) / capacity as u64) as u32
}

/// Quartile cuts of a channel's activity scores.
///
/// Deriving the cuts from the image's own distribution avoids magic thresholds
/// that would only suit one quality or one kind of content, and stays exactly
/// symmetric: both sides hold the same EOB array before any AC is coded.
pub fn activity_cuts(scores: &[u32]) -> [u32; 3] {
    if scores.is_empty() {
        return [0, 0, 0];
    }
    let mut sorted = scores.to_vec();
    sorted.sort_unstable();
    let pick = |quarter: usize| sorted[(sorted.len() * quarter / 4).min(sorted.len() - 1)];
    [pick(1), pick(2), pick(3)]
}

/// Bucket a block's activity score against the channel's cuts.
///
/// Comparison against cut *values* rather than ranks keeps ties consistent on
/// both sides regardless of sort stability.
#[inline]
pub fn activity_bucket(score: u32, cuts: &[u32; 3]) -> u8 {
    if score < cuts[0] {
        0
    } else if score < cuts[1] {
        1
    } else if score < cuts[2] {
        2
    } else {
        3
    }
}

/// Per-block activity buckets for a channel, from its EOB positions and the
/// per-block AC capacity.
pub fn activity_buckets(eob: &[u16], capacity: impl Fn(usize) -> usize) -> Vec<u8> {
    let scores: Vec<u32> = eob
        .iter()
        .enumerate()
        .map(|(idx, &e)| activity_score(e as usize, capacity(idx)))
        .collect();
    let cuts = activity_cuts(&scores);
    scores
        .iter()
        .map(|&score| activity_bucket(score, &cuts))
        .collect()
}

struct ContextModel {
    /// P(zero) for each of the 32 contexts: run_class(4) x prev_nz(2) x energy_bucket(4)
    p_zero: [u32; N_CTX_ZERO],
    /// P(pm1 | nonzero) for each of the 8 contexts: energy_bucket(4) x prev_was_pm1(2)
    p_pm1: [u32; N_CTX_PM1],
    /// Exponentially smoothed energy (scale 256)
    energy: u32,
    /// Previous symbol was nonzero
    prev_nz: bool,
    /// Consecutive zero count
    run_zeros: u32,
    /// Previous nonzero symbol was +-1
    prev_was_pm1: bool,
}

impl ContextModel {
    fn new() -> Self {
        // Initialize P(zero) = 14000/16384 ~ 85.4% (matches typical wavelet detail stats)
        let mut p_zero = [0u32; N_CTX_ZERO];
        for p in p_zero.iter_mut() {
            *p = 14000;
        }
        // Initialize P(pm1|nz) = 12000/16384 ~ 73.2%
        let mut p_pm1 = [0u32; N_CTX_PM1];
        for p in p_pm1.iter_mut() {
            *p = 12000;
        }
        Self {
            p_zero,
            p_pm1,
            energy: 0,
            prev_nz: false,
            run_zeros: 0,
            prev_was_pm1: true,
        }
    }

    /// Compute the context index for P(zero).
    #[inline]
    fn ctx_zero(&self) -> usize {
        let rc = run_class(self.run_zeros);
        let pnz = if self.prev_nz { 1 } else { 0 };
        let eb = energy_bucket(self.energy);
        rc * 8 + pnz * 4 + eb
    }

    /// Compute the context index for P(pm1|nonzero).
    #[inline]
    fn ctx_pm1(&self) -> usize {
        let eb = energy_bucket(self.energy);
        let pp = if self.prev_was_pm1 { 1 } else { 0 };
        eb * 2 + pp
    }

    /// Get P(zero) for current context, clamped to valid range [1, PROB_SCALE-1].
    #[inline]
    fn get_p_zero(&self) -> u32 {
        self.p_zero[self.ctx_zero()].clamp(1, PROB_SCALE - 1)
    }

    /// Get P(pm1|nonzero) for current context, clamped to valid range [1, PROB_SCALE-1].
    #[inline]
    fn get_p_pm1(&self) -> u32 {
        self.p_pm1[self.ctx_pm1()].clamp(1, PROB_SCALE - 1)
    }

    /// Update the model after observing a symbol.
    fn update(&mut self, value: i16) {
        let is_zero = value == 0;
        let is_pm1 = value.unsigned_abs() == 1;
        let abs_val = value.unsigned_abs() as u32;

        // Adapt P(zero)
        let cz = self.ctx_zero();
        if is_zero {
            // Increase P(zero): p += (PROB_SCALE - p) >> ADAPT_SHIFT
            self.p_zero[cz] += (PROB_SCALE - self.p_zero[cz]) >> ADAPT_SHIFT;
        } else {
            // Decrease P(zero): p -= p >> ADAPT_SHIFT
            self.p_zero[cz] -= self.p_zero[cz] >> ADAPT_SHIFT;
        }

        if !is_zero {
            // Adapt P(pm1|nonzero)
            let cp = self.ctx_pm1();
            if is_pm1 {
                self.p_pm1[cp] += (PROB_SCALE - self.p_pm1[cp]) >> ADAPT_SHIFT;
            } else {
                self.p_pm1[cp] -= self.p_pm1[cp] >> ADAPT_SHIFT;
            }
        }

        // Update energy (IIR filter)
        // energy = alpha * energy + (1-alpha) * abs_val * ENERGY_SCALE / amplitude
        // We keep it simple: energy = alpha * energy + (256-alpha) * min(abs_val * 32, 256)
        let sample_energy = (abs_val * 32).min(ENERGY_SCALE);
        self.energy = (ENERGY_ALPHA * self.energy + (ENERGY_SCALE - ENERGY_ALPHA) * sample_energy)
            / ENERGY_SCALE;

        // Update run state
        if is_zero {
            self.run_zeros = self.run_zeros.saturating_add(1);
        } else {
            self.run_zeros = 0;
            self.prev_was_pm1 = is_pm1;
        }
        self.prev_nz = !is_zero;
    }
}

// ======================================================================
// v11 Adaptive Context Model (hyper-sparse: 6 run-class buckets)
// ======================================================================

struct ContextModelV11 {
    /// P(zero) for each of the 48 contexts: run_class_v11(6) x prev_nz(2) x energy_bucket(4)
    p_zero: [u32; N_CTX_ZERO_V11],
    /// P(pm1 | nonzero) for each of the 8 contexts: energy_bucket(4) x prev_was_pm1(2)
    p_pm1: [u32; N_CTX_PM1],
    /// Exponentially smoothed energy (scale 256)
    energy: u32,
    /// Previous symbol was nonzero
    prev_nz: bool,
    /// Consecutive zero count
    run_zeros: u32,
    /// Previous nonzero symbol was +-1
    prev_was_pm1: bool,
}

impl ContextModelV11 {
    fn new() -> Self {
        let mut p_zero = [0u32; N_CTX_ZERO_V11];
        for p in p_zero.iter_mut() {
            *p = 14000;
        }
        let mut p_pm1 = [0u32; N_CTX_PM1];
        for p in p_pm1.iter_mut() {
            *p = 12000;
        }
        Self {
            p_zero,
            p_pm1,
            energy: 0,
            prev_nz: false,
            run_zeros: 0,
            prev_was_pm1: true,
        }
    }

    #[inline]
    fn ctx_zero(&self) -> usize {
        let rc = run_class_v11(self.run_zeros);
        let pnz = if self.prev_nz { 1 } else { 0 };
        let eb = energy_bucket(self.energy);
        rc * 8 + pnz * 4 + eb
    }

    #[inline]
    fn ctx_pm1(&self) -> usize {
        let eb = energy_bucket(self.energy);
        let pp = if self.prev_was_pm1 { 1 } else { 0 };
        eb * 2 + pp
    }

    #[inline]
    fn get_p_zero(&self) -> u32 {
        self.p_zero[self.ctx_zero()].clamp(1, PROB_SCALE - 1)
    }

    #[inline]
    fn get_p_pm1(&self) -> u32 {
        self.p_pm1[self.ctx_pm1()].clamp(1, PROB_SCALE - 1)
    }

    fn update(&mut self, value: i16) {
        let is_zero = value == 0;
        let is_pm1 = value.unsigned_abs() == 1;
        let abs_val = value.unsigned_abs() as u32;

        let cz = self.ctx_zero();
        if is_zero {
            self.p_zero[cz] += (PROB_SCALE - self.p_zero[cz]) >> ADAPT_SHIFT;
        } else {
            self.p_zero[cz] -= self.p_zero[cz] >> ADAPT_SHIFT;
        }

        if !is_zero {
            let cp = self.ctx_pm1();
            if is_pm1 {
                self.p_pm1[cp] += (PROB_SCALE - self.p_pm1[cp]) >> ADAPT_SHIFT;
            } else {
                self.p_pm1[cp] -= self.p_pm1[cp] >> ADAPT_SHIFT;
            }
        }

        let sample_energy = (abs_val * 32).min(ENERGY_SCALE);
        self.energy = (ENERGY_ALPHA * self.energy + (ENERGY_SCALE - ENERGY_ALPHA) * sample_energy)
            / ENERGY_SCALE;

        if is_zero {
            self.run_zeros = self.run_zeros.saturating_add(1);
        } else {
            self.run_zeros = 0;
            self.prev_was_pm1 = is_pm1;
        }
        self.prev_nz = !is_zero;
    }
}

// ======================================================================
// v12 Adaptive Context Model (Bayesian predictive hierarchy)
// ======================================================================

struct ContextModelV12 {
    /// P(zero) for each of the 128 contexts: run_class(4) x prev_nz(2) x energy_bucket(4) x turing_bucket(4)
    p_zero: [u32; N_CTX_ZERO_V12],
    /// P(pm1 | nonzero) for each of the 32 contexts: energy_bucket(4) x prev_was_pm1(2) x turing_bucket(4)
    p_pm1: [u32; N_CTX_PM1_V12],
    /// Exponentially smoothed energy (scale 256)
    energy: u32,
    /// Previous symbol was nonzero
    prev_nz: bool,
    /// Consecutive zero count
    run_zeros: u32,
    /// Previous nonzero symbol was +-1
    prev_was_pm1: bool,
    /// Turing complexity level bucket (0-3), set externally per band
    turing_level: u8,
}

impl ContextModelV12 {
    fn new() -> Self {
        let mut p_zero = [0u32; N_CTX_ZERO_V12];
        for p in p_zero.iter_mut() {
            *p = 14000;
        }
        let mut p_pm1 = [0u32; N_CTX_PM1_V12];
        for p in p_pm1.iter_mut() {
            *p = 12000;
        }
        Self {
            p_zero,
            p_pm1,
            energy: 0,
            prev_nz: false,
            run_zeros: 0,
            prev_was_pm1: false,
            turing_level: 0,
        }
    }

    /// Set the Turing complexity bucket for the current band (clamped to 0-3).
    #[inline]
    fn set_turing_level(&mut self, bucket: u8) {
        self.turing_level = bucket.min(3);
    }

    /// Compute the context index for P(zero).
    /// Layout: run_class * 32 + prev_nz * 16 + energy_bucket * 4 + turing_bucket
    #[inline]
    fn zero_ctx_index(&self) -> usize {
        let rc = run_class(self.run_zeros);
        let pnz = if self.prev_nz { 1 } else { 0 };
        let eb = energy_bucket(self.energy);
        let tb = self.turing_level as usize;
        rc * 32 + pnz * 16 + eb * 4 + tb
    }

    /// Compute the context index for P(pm1|nonzero).
    /// Layout: energy_bucket * 8 + prev_was_pm1 * 4 + turing_bucket
    #[inline]
    fn pm1_ctx_index(&self) -> usize {
        let eb = energy_bucket(self.energy);
        let pp = if self.prev_was_pm1 { 1 } else { 0 };
        let tb = self.turing_level as usize;
        eb * 8 + pp * 4 + tb
    }

    /// Get P(zero) for current context, clamped to valid range [1, PROB_SCALE-1].
    #[inline]
    fn get_p_zero(&self) -> u32 {
        self.p_zero[self.zero_ctx_index()].clamp(1, PROB_SCALE - 1)
    }

    /// Get P(pm1|nonzero) for current context, clamped to valid range [1, PROB_SCALE-1].
    #[inline]
    fn get_p_pm1(&self) -> u32 {
        self.p_pm1[self.pm1_ctx_index()].clamp(1, PROB_SCALE - 1)
    }

    /// Update the model after observing a symbol.
    fn update(&mut self, symbol: i16) {
        let is_zero = symbol == 0;
        let is_pm1 = symbol.unsigned_abs() == 1;
        let abs_val = symbol.unsigned_abs() as u32;

        // Adapt P(zero) for current context
        let cz = self.zero_ctx_index();
        if is_zero {
            self.p_zero[cz] += (PROB_SCALE - self.p_zero[cz]) >> ADAPT_SHIFT;
        } else {
            self.p_zero[cz] -= self.p_zero[cz] >> ADAPT_SHIFT;
        }

        if !is_zero {
            // Adapt P(pm1|nonzero) for current context
            let cp = self.pm1_ctx_index();
            if is_pm1 {
                self.p_pm1[cp] += (PROB_SCALE - self.p_pm1[cp]) >> ADAPT_SHIFT;
            } else {
                self.p_pm1[cp] -= self.p_pm1[cp] >> ADAPT_SHIFT;
            }
        }

        // Update energy (IIR filter): energy = alpha * energy + (1-alpha) * sample
        let sample_energy = (abs_val * 32).min(ENERGY_SCALE);
        self.energy = (ENERGY_ALPHA * self.energy + (ENERGY_SCALE - ENERGY_ALPHA) * sample_energy)
            / ENERGY_SCALE;

        // Update run/nz state
        if is_zero {
            self.run_zeros = self.run_zeros.saturating_add(1);
        } else {
            self.run_zeros = 0;
            self.prev_was_pm1 = is_pm1;
        }
        self.prev_nz = !is_zero;
    }
}

/// v12 sequential state with one additional zero-probability dimension for a
/// causal dyadic-parent support bit. Run length, energy and +/-1 adaptation
/// remain shared, avoiding the discontinuities caused by two independent
/// models.
struct ContextModelWavelet {
    p_zero: [u32; N_CTX_ZERO_V12 * 2],
    p_pm1: [u32; N_CTX_PM1_V12],
    energy: u32,
    prev_nz: bool,
    run_zeros: u32,
    prev_was_pm1: bool,
    turing_level: u8,
    parent_nonzero: bool,
}

impl ContextModelWavelet {
    fn new() -> Self {
        Self {
            p_zero: [14000; N_CTX_ZERO_V12 * 2],
            p_pm1: [12000; N_CTX_PM1_V12],
            energy: 0,
            prev_nz: false,
            run_zeros: 0,
            prev_was_pm1: false,
            turing_level: 0,
            parent_nonzero: false,
        }
    }

    #[inline]
    fn set_external_context(&mut self, turing_bucket: u8, parent_nonzero: bool) {
        self.turing_level = turing_bucket.min(3);
        self.parent_nonzero = parent_nonzero;
    }

    #[inline]
    fn zero_ctx_index(&self) -> usize {
        let run = run_class(self.run_zeros);
        let previous_nonzero = usize::from(self.prev_nz);
        let energy = energy_bucket(self.energy);
        let base = run * 32 + previous_nonzero * 16 + energy * 4 + self.turing_level as usize;
        base + usize::from(self.parent_nonzero) * N_CTX_ZERO_V12
    }

    #[inline]
    fn pm1_ctx_index(&self) -> usize {
        energy_bucket(self.energy) * 8
            + usize::from(self.prev_was_pm1) * 4
            + self.turing_level as usize
    }

    #[inline]
    fn get_p_zero(&self) -> u32 {
        self.p_zero[self.zero_ctx_index()].clamp(1, PROB_SCALE - 1)
    }

    #[inline]
    fn get_p_pm1(&self) -> u32 {
        self.p_pm1[self.pm1_ctx_index()].clamp(1, PROB_SCALE - 1)
    }

    fn update(&mut self, symbol: i16) {
        let is_zero = symbol == 0;
        let is_pm1 = symbol.unsigned_abs() == 1;
        let absolute = symbol.unsigned_abs() as u32;
        let zero_context = self.zero_ctx_index();
        if is_zero {
            self.p_zero[zero_context] += (PROB_SCALE - self.p_zero[zero_context]) >> ADAPT_SHIFT;
        } else {
            self.p_zero[zero_context] -= self.p_zero[zero_context] >> ADAPT_SHIFT;
            let pm1_context = self.pm1_ctx_index();
            if is_pm1 {
                self.p_pm1[pm1_context] += (PROB_SCALE - self.p_pm1[pm1_context]) >> ADAPT_SHIFT;
            } else {
                self.p_pm1[pm1_context] -= self.p_pm1[pm1_context] >> ADAPT_SHIFT;
            }
        }
        let sample_energy = (absolute * 32).min(ENERGY_SCALE);
        self.energy = (ENERGY_ALPHA * self.energy + (ENERGY_SCALE - ENERGY_ALPHA) * sample_energy)
            / ENERGY_SCALE;
        if is_zero {
            self.run_zeros = self.run_zeros.saturating_add(1);
        } else {
            self.run_zeros = 0;
            self.prev_was_pm1 = is_pm1;
        }
        self.prev_nz = !is_zero;
    }
}

/// Laboratory model for a causal color codon. The triplet is factorized as
/// P(L) P(C1|L) P(C2|L,C1): `codon_class` is derived from channels already
/// decoded at the same block/frequency position and therefore costs zero bits.
struct ContextModelCodon {
    p_zero: [u32; N_CTX_ZERO_V12 * 2 * 4],
    p_pm1: [u32; N_CTX_PM1_V12 * 4],
    energy: u32,
    prev_nz: bool,
    run_zeros: u32,
    prev_was_pm1: bool,
    turing_level: u8,
    parent_nonzero: bool,
    codon_class: u8,
    split_pm1: bool,
}

impl ContextModelCodon {
    fn new(split_pm1: bool) -> Self {
        Self {
            p_zero: [14000; N_CTX_ZERO_V12 * 2 * 4],
            p_pm1: [12000; N_CTX_PM1_V12 * 4],
            energy: 0,
            prev_nz: false,
            run_zeros: 0,
            prev_was_pm1: false,
            turing_level: 0,
            parent_nonzero: false,
            codon_class: 0,
            split_pm1,
        }
    }

    #[inline]
    fn set_external_context(&mut self, turing_bucket: u8, parent_nonzero: bool, codon_class: u8) {
        self.turing_level = turing_bucket.min(3);
        self.parent_nonzero = parent_nonzero;
        self.codon_class = codon_class.min(3);
    }

    #[inline]
    fn zero_ctx_index(&self) -> usize {
        let run = run_class(self.run_zeros);
        let previous_nonzero = usize::from(self.prev_nz);
        let energy = energy_bucket(self.energy);
        let base = run * 32 + previous_nonzero * 16 + energy * 4 + self.turing_level as usize;
        base + usize::from(self.parent_nonzero) * N_CTX_ZERO_V12
            + self.codon_class as usize * N_CTX_ZERO_V12 * 2
    }

    #[inline]
    fn pm1_ctx_index(&self) -> usize {
        let base = energy_bucket(self.energy) * 8
            + usize::from(self.prev_was_pm1) * 4
            + self.turing_level as usize;
        if self.split_pm1 {
            base + self.codon_class as usize * N_CTX_PM1_V12
        } else {
            base
        }
    }

    #[inline]
    fn get_p_zero(&self) -> u32 {
        self.p_zero[self.zero_ctx_index()].clamp(1, PROB_SCALE - 1)
    }

    #[inline]
    fn get_p_pm1(&self) -> u32 {
        self.p_pm1[self.pm1_ctx_index()].clamp(1, PROB_SCALE - 1)
    }

    fn update(&mut self, symbol: i16) {
        let is_zero = symbol == 0;
        let is_pm1 = symbol.unsigned_abs() == 1;
        let absolute = symbol.unsigned_abs() as u32;
        let zero_context = self.zero_ctx_index();
        if is_zero {
            self.p_zero[zero_context] += (PROB_SCALE - self.p_zero[zero_context]) >> ADAPT_SHIFT;
        } else {
            self.p_zero[zero_context] -= self.p_zero[zero_context] >> ADAPT_SHIFT;
            let pm1_context = self.pm1_ctx_index();
            if is_pm1 {
                self.p_pm1[pm1_context] += (PROB_SCALE - self.p_pm1[pm1_context]) >> ADAPT_SHIFT;
            } else {
                self.p_pm1[pm1_context] -= self.p_pm1[pm1_context] >> ADAPT_SHIFT;
            }
        }
        let sample_energy = (absolute * 32).min(ENERGY_SCALE);
        self.energy = (ENERGY_ALPHA * self.energy + (ENERGY_SCALE - ENERGY_ALPHA) * sample_energy)
            / ENERGY_SCALE;
        if is_zero {
            self.run_zeros = self.run_zeros.saturating_add(1);
        } else {
            self.run_zeros = 0;
            self.prev_was_pm1 = is_pm1;
        }
        self.prev_nz = !is_zero;
    }
}

// ======================================================================
// Cycle 22 — logistic context mixing
// ======================================================================
//
// Cycle 21 measured 0.03 to 0.10 bit/symbol of causal, free information in
// Lphi's neighbourhood that the chroma codon cannot reach: folding it into the
// existing four states destroys what the codon already carried, and giving it
// its own axis pays a dilution that cancels the gain. Both walls come from the
// same assumption — that one context index must carry every witness.
//
// Mixing drops that assumption. Two models observe the same bit through
// different contexts and each states a probability; a third, tiny model learns
// how much to trust each of them, in the logit domain where probabilities add.
// Neither is diluted, neither is destroyed.
//
// Everything below is integer. A codec cannot afford floating point in a path
// the decoder must reproduce exactly, on any machine, forever.

/// Probability domain of the mixer: 12 bits, `0..MIX_ONE`.
const MIX_BITS: u32 = 12;
const MIX_ONE: i32 = 1 << MIX_BITS;
/// Logit domain bound. `stretch` saturates here, `squash` accepts it.
const MIX_STRETCH_LIMIT: i32 = 2047;
/// Fixed-point shift of the mixer weights.
const MIX_WEIGHT_SHIFT: u32 = 16;
/// A weight of exactly 1.0.
const MIX_WEIGHT_ONE: i32 = 1 << MIX_WEIGHT_SHIFT;

/// `squash(x) = MIX_ONE / (1 + exp(-x/256))`, tabulated on 33 anchors and
/// linearly interpolated — the classic integer logistic of the context-mixing
/// literature. Deterministic on every platform.
#[inline]
fn mix_squash(x: i32) -> i32 {
    const TABLE: [i32; 33] = [
        1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994,
        3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094,
    ];
    let x = x.clamp(-MIX_STRETCH_LIMIT, MIX_STRETCH_LIMIT);
    let w = x & 127;
    let index = ((x >> 7) + 16) as usize;
    (TABLE[index + 1] * w + TABLE[index] * (128 - w)) >> 7
}

/// Inverse of [`mix_squash`], built once by monotone inversion of the table so
/// the two are exact mutual inverses on the integer lattice.
struct MixStretch {
    table: [i16; MIX_ONE as usize],
}

impl MixStretch {
    fn new() -> Self {
        let mut table = [0i16; MIX_ONE as usize];
        let mut position = 0i32;
        for x in -MIX_STRETCH_LIMIT..=MIX_STRETCH_LIMIT {
            let value = mix_squash(x);
            while position <= value {
                if (position as usize) < table.len() {
                    table[position as usize] = x as i16;
                }
                position += 1;
            }
        }
        while (position as usize) < table.len() {
            table[position as usize] = MIX_STRETCH_LIMIT as i16;
            position += 1;
        }
        Self { table }
    }

    #[inline]
    fn get(&self, p: i32) -> i32 {
        self.table[p.clamp(0, MIX_ONE - 1) as usize] as i32
    }
}

static MIX_STRETCH: std::sync::LazyLock<MixStretch> = std::sync::LazyLock::new(MixStretch::new);

/// Maximum number of opinions the mixer arbitrates.
const MIX_MAX_INPUTS: usize = 6;

/// An N-input logistic mixer with one weight set per selector.
///
/// The selector is deliberately coarse — the mixer must learn *how much to
/// trust each model*, which is a far simpler question than the models' own, and
/// splitting it finely would only reintroduce the dilution mixing exists to
/// avoid.
struct Mixer {
    weights: Vec<[i32; MIX_MAX_INPUTS]>,
    learning_rate: i32,
    inputs: usize,
    selector: usize,
    stretched: [i32; MIX_MAX_INPUTS],
}

impl Mixer {
    fn new(selectors: usize, learning_rate: i32, inputs: usize) -> Self {
        // Start at (1, 0, ...): the mixer begins as the primary model alone and
        // has to earn any departure from it.
        let mut initial = [0i32; MIX_MAX_INPUTS];
        initial[0] = MIX_WEIGHT_ONE;
        Self {
            weights: vec![initial; selectors.max(1)],
            learning_rate: learning_rate.max(1),
            inputs: inputs.clamp(1, MIX_MAX_INPUTS),
            selector: 0,
            stretched: [0; MIX_MAX_INPUTS],
        }
    }

    /// Mix probabilities on `0..MIX_ONE`, returning the same domain.
    #[inline]
    fn mix(&mut self, selector: usize, probabilities: &[i32]) -> i32 {
        self.selector = selector.min(self.weights.len() - 1);
        let stretch = &*MIX_STRETCH;
        let mut dot = 0i64;
        let w = self.weights[self.selector];
        for slot in 0..self.inputs {
            let value = stretch.get(probabilities[slot]);
            self.stretched[slot] = value;
            dot += w[slot] as i64 * value as i64;
        }
        mix_squash((dot >> MIX_WEIGHT_SHIFT) as i32).clamp(1, MIX_ONE - 1)
    }

    /// Gradient step on the coding loss. `bit` is the event the mixed
    /// probability was describing.
    #[inline]
    fn update(&mut self, mixed: i32, bit: bool) {
        let error = ((i32::from(bit) << MIX_BITS) - mixed) * self.learning_rate;
        let w = &mut self.weights[self.selector];
        for slot in 0..self.inputs {
            w[slot] += (self.stretched[slot] * error) >> 10;
            w[slot] = w[slot].clamp(-(MIX_WEIGHT_ONE * 4), MIX_WEIGHT_ONE * 4);
        }
    }
}

/// Cycle 23 — adaptive probability map applied *after* the mix.
///
/// The mixer produces a probability; the APM asks whether that probability has
/// historically been well calibrated in this coarse context, and corrects it.
/// It is indexed by the mixed probability itself, quantized on the logit
/// ladder, so it learns the shape of the mixer's own bias rather than anything
/// about the image.
struct Apm {
    table: Vec<u16>,
    rate: u32,
    pending: usize,
    weight: i32,
}

impl Apm {
    const BUCKETS: usize = 33;

    fn new(contexts: usize, rate: u32) -> Self {
        let mut table = vec![0u16; contexts.max(1) * Self::BUCKETS];
        for context in 0..contexts.max(1) {
            for bucket in 0..Self::BUCKETS {
                let x = (bucket as i32 - 16) * 128;
                table[context * Self::BUCKETS + bucket] = mix_squash(x) as u16;
            }
        }
        Self {
            table,
            rate: rate.clamp(1, 15),
            pending: usize::MAX,
            weight: 0,
        }
    }

    /// Refine `p`, remembering which pair of anchors to correct afterwards.
    #[inline]
    fn refine(&mut self, context: usize, p: i32) -> i32 {
        let stretched = MIX_STRETCH.get(p).clamp(-2047, 2047) + 2048;
        let bucket = (stretched >> 7) as usize;
        self.weight = stretched & 127;
        let base = context * Self::BUCKETS + bucket.min(Self::BUCKETS - 2);
        self.pending = base;
        let low = self.table[base] as i32;
        let high = self.table[base + 1] as i32;
        ((low * (128 - self.weight) + high * self.weight) >> 7).clamp(1, MIX_ONE - 1)
    }

    #[inline]
    fn update(&mut self, bit: bool) {
        if self.pending == usize::MAX {
            return;
        }
        let target = if bit { MIX_ONE - 1 } else { 0 };
        // Correct the anchor the interpolation leaned on most.
        let index = if self.weight >= 64 {
            self.pending + 1
        } else {
            self.pending
        };
        let slot = &mut self.table[index];
        let current = *slot as i32;
        *slot = (current + ((target - current) >> self.rate)) as u16;
        self.pending = usize::MAX;
    }
}

/// A secondary opinion in the mix: the same coefficient seen through a witness
/// the primary model does not carry.
///
/// Each is deliberately *smaller* than the primary. A second opinion only has
/// to be roughly right and roughly independent; making it as detailed as the
/// primary would just reproduce the primary's dilution — which is exactly what
/// cycle 21 measured when it tried to give these witnesses their own context
/// axis instead of their own model.
struct SecondaryModel {
    p: Vec<u32>,
    stride: usize,
}

impl SecondaryModel {
    /// run_class(4) x energy_bucket(4) x amplitude(4) x witness(4)
    const CONTEXTS: usize = 4 * 4 * AMPLITUDE_STATES * 4;

    fn new(planes: usize) -> Self {
        Self {
            p: vec![MIX_ONE as u32 * 7 / 8; Self::CONTEXTS * planes.max(1)],
            stride: Self::CONTEXTS,
        }
    }

    #[inline]
    fn index(plane: usize, run: usize, energy: usize, amplitude: u8, witness: u8) -> usize {
        let local = ((run * 4 + energy) * AMPLITUDE_STATES
            + (amplitude as usize).min(AMPLITUDE_STATES - 1))
            * 4
            + (witness as usize).min(3);
        plane * Self::CONTEXTS + local
    }

    #[inline]
    fn get(&self, index: usize) -> i32 {
        (self.p[index] as i32).clamp(1, MIX_ONE - 1)
    }

    #[inline]
    fn update(&mut self, index: usize, event: bool) {
        let slot = &mut self.p[index];
        if event {
            *slot += ((MIX_ONE as u32 - *slot) >> 5).max(1);
        } else {
            *slot -= (*slot >> 5).max(1);
        }
        *slot = (*slot).clamp(1, MIX_ONE as u32 - 1);
    }

    #[inline]
    fn planes(&self) -> usize {
        self.p.len() / self.stride
    }
}

/// Coarse selector count for the mixer: `witness(4) x amplitude(4)`.
// Cycle 28 — multiple importance sampling was tried here and rejected.
//
// A path tracer combines estimators by weighting each with its own confidence
// at the sample it drew, so whichever strategy knows something dominates there.
// The obvious transfer is to split this selector by how committed the primary
// model is, letting the mixer learn "lean on the secondaries when the primary
// is unsure". Measured on the sentinels: +0.041 % rate with a 2-way split,
// +0.114 % with a 4-way one — monotone in the split, so any conditioning loses.
//
// The reason is the same one cycle 27 §10 established from the other side. The
// mixer is *data*-starved, not information-starved: it holds few weights and
// converges fast, so quartering the data each weight vector sees costs more
// than the conditioning returns. The big context tables are the opposite
// regime — 4096 slots, slow convergence — which is why carrying *them* between
// layers paid and splitting the mixer does not. Confidence conditioning would
// only earn its keep on a model that already sees enough of each case.
const MIX_SELECTORS: usize = 4 * AMPLITUDE_STATES;

/// Frequency band of a coefficient, on four coarse steps. The primary model
/// only ever knew frequency through its IIR energy memory, which lags it.
#[inline]
fn mix_band(position: u16) -> u8 {
    match position {
        0..=3 => 0,
        4..=15 => 1,
        16..=63 => 2,
        _ => 3,
    }
}

/// Routes `P(zero)` — and optionally `P(|x| = 1)` — either through the v21
/// single model or through the mix. When disabled it returns the primary
/// model's value from the *same* call the v21 path uses, so bit-exactness of
/// the neutral state is structural rather than a coincidence to be re-verified.
struct MixingState {
    enabled: bool,
    mix_pm1: bool,
    witnesses: [u8; MIX_MAX_INPUTS],
    used: usize,
    zero_mixer: Mixer,
    zero_models: SecondaryModel,
    zero_apm: Option<Apm>,
    pm1_mixer: Mixer,
    pm1_models: SecondaryModel,
    mix_sign: bool,
    sign_mixer: Mixer,
    /// Second opinion on phase agreement: `class x separable memory x
    /// neighbourhood energy`. Both witnesses were measured at cycles 20 and 21
    /// and rejected as *contexts* of the single counter; here they are an
    /// input, which is a different question.
    sign_model: Vec<u32>,
    pending_zero: Option<([usize; MIX_MAX_INPUTS], i32, i32)>,
    pending_pm1: Option<([usize; MIX_MAX_INPUTS], i32)>,
    pending_sign: Option<(usize, i32)>,
}

impl MixingState {
    fn new(config: &SpectralEnvelopeConfig, channel: usize) -> Self {
        // A witness that needs an earlier channel is unavailable on Lphi.
        let mut witnesses = [0u8; MIX_MAX_INPUTS];
        let mut used = 0usize;
        if config.mix_rate > 0 && (config.mix_channels >> channel) & 1 != 0 {
            for &bit in &[
                crate::spectral_envelope::MIX_INPUT_FOREIGN,
                crate::spectral_envelope::MIX_INPUT_SCALED,
                crate::spectral_envelope::MIX_INPUT_BAND,
                crate::spectral_envelope::MIX_INPUT_SIZE,
                crate::spectral_envelope::MIX_INPUT_SUPPORT,
            ] {
                if config.mix_inputs & bit == 0 {
                    continue;
                }
                if bit == crate::spectral_envelope::MIX_INPUT_FOREIGN && channel == 0 {
                    continue;
                }
                if used + 1 < MIX_MAX_INPUTS {
                    witnesses[used] = bit;
                    used += 1;
                }
            }
        }
        let enabled = used > 0;
        let inputs = used + 1;
        Self {
            enabled,
            mix_pm1: config.mix_pm1,
            witnesses,
            used,
            zero_mixer: Mixer::new(MIX_SELECTORS, config.mix_rate as i32, inputs),
            zero_models: SecondaryModel::new(used.max(1)),
            zero_apm: (config.apm_rate > 0)
                .then(|| Apm::new(AMPLITUDE_STATES, config.apm_rate as u32)),
            pm1_mixer: Mixer::new(MIX_SELECTORS, config.mix_rate as i32, inputs),
            pm1_models: SecondaryModel::new(used.max(1)),
            mix_sign: config.mix_sign && enabled,
            sign_mixer: Mixer::new(PHASE_CLASS_COUNT, config.mix_rate as i32, 2),
            sign_model: vec![MIX_ONE as u32 / 2; PHASE_CLASS_COUNT * 3 * AMPLITUDE_STATES],
            pending_zero: None,
            pending_pm1: None,
            pending_sign: None,
        }
    }

    /// Mix the phase-agreement probability. `primary` is the class counter the
    /// cycle-20 grammar would have used on its own.
    #[inline]
    fn p_same(&mut self, primary: u32, class: u8, memory: u8, energy: u8) -> u32 {
        if !self.mix_sign || class == PHASE_NONE {
            self.pending_sign = None;
            return primary;
        }
        let index = (class as usize * 3 + (memory as usize).min(2)) * AMPLITUDE_STATES
            + (energy as usize).min(AMPLITUDE_STATES - 1);
        let secondary = (self.sign_model[index] as i32).clamp(1, MIX_ONE - 1);
        let scaled = ((primary >> (PROB_BITS - MIX_BITS)) as i32).clamp(1, MIX_ONE - 1);
        let mixed = self.sign_mixer.mix(class as usize, &[scaled, secondary]);
        self.pending_sign = Some((index, mixed));
        ((mixed as u32) << (PROB_BITS - MIX_BITS)).clamp(1, PROB_SCALE - 1)
    }

    #[inline]
    fn update_sign(&mut self, same: bool) {
        if let Some((index, mixed)) = self.pending_sign.take() {
            self.sign_mixer.update(mixed, same);
            let slot = &mut self.sign_model[index];
            if same {
                *slot += ((MIX_ONE as u32 - *slot) >> 5).max(1);
            } else {
                *slot -= (*slot >> 5).max(1);
            }
            *slot = (*slot).clamp(1, MIX_ONE as u32 - 1);
        }
    }

    /// Value of witness `slot` for the current coefficient.
    #[inline]
    fn witness(
        &self,
        slot: usize,
        probe: &crate::spectral_envelope::SpectralProbe,
        foreign: u8,
    ) -> u8 {
        match self.witnesses[slot] {
            crate::spectral_envelope::MIX_INPUT_FOREIGN => foreign,
            crate::spectral_envelope::MIX_INPUT_SCALED => probe.scaled_amplitude,
            crate::spectral_envelope::MIX_INPUT_BAND => mix_band(probe.position),
            crate::spectral_envelope::MIX_INPUT_SIZE => probe.size_class,
            _ => probe.support_count,
        }
    }

    #[inline]
    fn selector(&self, primary_witness: u8, amplitude: u8) -> usize {
        (primary_witness as usize).min(3) * AMPLITUDE_STATES
            + (amplitude as usize).min(AMPLITUDE_STATES - 1)
    }

    #[inline]
    fn p_zero(
        &mut self,
        context: &ContextModelEnvelope,
        probe: &crate::spectral_envelope::SpectralProbe,
        foreign: u8,
    ) -> u32 {
        let primary = context.get_p_zero();
        if !self.enabled {
            self.pending_zero = None;
            return primary;
        }
        let run = context.run_bucket();
        let energy = context.energy_index();
        let mut probabilities = [0i32; MIX_MAX_INPUTS];
        let mut indices = [0usize; MIX_MAX_INPUTS];
        probabilities[0] = ((primary >> (PROB_BITS - MIX_BITS)) as i32).clamp(1, MIX_ONE - 1);
        let mut first_witness = 0u8;
        for slot in 0..self.used {
            let witness = self.witness(slot, probe, foreign);
            if slot == 0 {
                first_witness = witness;
            }
            let index = SecondaryModel::index(slot, run, energy, probe.amplitude, witness);
            indices[slot] = index;
            probabilities[slot + 1] = self.zero_models.get(index);
        }
        let selector = self.selector(first_witness, probe.amplitude);
        let mixed = self.zero_mixer.mix(selector, &probabilities);
        let refined = match self.zero_apm.as_mut() {
            Some(apm) => {
                let raw = apm.refine((probe.amplitude as usize).min(AMPLITUDE_STATES - 1), mixed);
                // Blend the raw mix with its correction, the usual guard
                // against an APM that has not yet seen enough of a context.
                ((mixed + 3 * raw) >> 2).clamp(1, MIX_ONE - 1)
            }
            None => mixed,
        };
        self.pending_zero = Some((indices, mixed, refined));
        ((refined as u32) << (PROB_BITS - MIX_BITS)).clamp(1, PROB_SCALE - 1)
    }

    #[inline]
    fn update_zero(&mut self, is_zero: bool) {
        if let Some((indices, mixed, _)) = self.pending_zero.take() {
            self.zero_mixer.update(mixed, is_zero);
            for slot in 0..self.used {
                self.zero_models.update(indices[slot], is_zero);
            }
            if let Some(apm) = self.zero_apm.as_mut() {
                apm.update(is_zero);
            }
        }
    }

    #[inline]
    fn p_pm1(
        &mut self,
        context: &ContextModelEnvelope,
        probe: &crate::spectral_envelope::SpectralProbe,
        foreign: u8,
    ) -> u32 {
        let primary = context.get_p_pm1();
        if !self.enabled || !self.mix_pm1 {
            self.pending_pm1 = None;
            return primary;
        }
        let run = context.run_bucket();
        let energy = context.energy_index();
        let mut probabilities = [0i32; MIX_MAX_INPUTS];
        let mut indices = [0usize; MIX_MAX_INPUTS];
        probabilities[0] = ((primary >> (PROB_BITS - MIX_BITS)) as i32).clamp(1, MIX_ONE - 1);
        let mut first_witness = 0u8;
        for slot in 0..self.used {
            let witness = self.witness(slot, probe, foreign);
            if slot == 0 {
                first_witness = witness;
            }
            let index = SecondaryModel::index(slot, run, energy, probe.amplitude, witness);
            indices[slot] = index;
            probabilities[slot + 1] = self.pm1_models.get(index);
        }
        let selector = self.selector(first_witness, probe.amplitude);
        let mixed = self.pm1_mixer.mix(selector, &probabilities);
        self.pending_pm1 = Some((indices, mixed));
        ((mixed as u32) << (PROB_BITS - MIX_BITS)).clamp(1, PROB_SCALE - 1)
    }

    #[inline]
    fn update_pm1(&mut self, is_pm1: bool) {
        if let Some((indices, mixed)) = self.pending_pm1.take() {
            self.pm1_mixer.update(mixed, is_pm1);
            for slot in 0..self.used {
                self.pm1_models.update(indices[slot], is_pm1);
            }
        }
    }
}

/// Cycle 20 model. Identical to [`ContextModelCodon`] except that the binary
/// dyadic-parent axis is replaced by a four-state *amplitude envelope* read
/// from the causal spectral neighbourhood. With
/// [`SpectralEnvelopeConfig::neutral`] the envelope collapses back onto the
/// parent support and the partition of symbols into contexts is exactly the
/// v20 one, so the emitted bytes are unchanged.
/// The coder's adapted probability tables, carried between progressive layers.
///
/// Layer 0 delivers every block's first coefficients; then every detail layer
/// opens a *fresh* stream and breaks down flat air all over again. Lightning
/// does not work that way — a return stroke reuses the channel the first stroke
/// already ionized, which is exactly why it is faster and carries more. The
/// adapted tables are that ionized channel.
///
/// Carrying layer 0's converged tables into each detail layer is deterministic
/// and safe under progressive truncation, because layer 0 is always present.
/// Unlike the trellis priming of cycle 27 §8 it is not self-referential: the
/// skeleton is genuinely transmitted data, not a previous pass's own guesses.
#[derive(Clone)]
pub struct EnvelopeTables {
    p_zero: Vec<u32>,
    p_pm1: Vec<u32>,
    /// Everything the logistic mixer, its secondary models, the APM and the
    /// phase counters learned. `None` until a layer has actually run: the very
    /// first stream still has to break down virgin air.
    mixing: Option<MixingSnapshot>,
    phase: Option<Vec<u32>>,
}

/// The adaptive state of the mixing stack, snapshotted between layers.
#[derive(Clone)]
struct MixingSnapshot {
    zero_weights: Vec<[i32; MIX_MAX_INPUTS]>,
    zero_models: Vec<u32>,
    zero_apm: Option<Vec<u16>>,
    pm1_weights: Vec<[i32; MIX_MAX_INPUTS]>,
    pm1_models: Vec<u32>,
    sign_weights: Vec<[i32; MIX_MAX_INPUTS]>,
    sign_model: Vec<u32>,
}

impl Default for EnvelopeTables {
    fn default() -> Self {
        Self {
            p_zero: vec![14000; N_CTX_ZERO_V12 * AMPLITUDE_STATES * CODON_STATES],
            p_pm1: vec![12000; N_CTX_PM1_V12 * AMPLITUDE_STATES * CODON_STATES],
            mixing: None,
            phase: None,
        }
    }
}

impl MixingState {
    fn snapshot(&self) -> MixingSnapshot {
        MixingSnapshot {
            zero_weights: self.zero_mixer.weights.clone(),
            zero_models: self.zero_models.p.clone(),
            zero_apm: self.zero_apm.as_ref().map(|apm| apm.table.clone()),
            pm1_weights: self.pm1_mixer.weights.clone(),
            pm1_models: self.pm1_models.p.clone(),
            sign_weights: self.sign_mixer.weights.clone(),
            sign_model: self.sign_model.clone(),
        }
    }

    /// Restore only where the shapes agree: a snapshot taken under a different
    /// configuration must never be forced onto this one.
    fn restore(&mut self, snapshot: &MixingSnapshot) {
        let copy = |dst: &mut Vec<[i32; MIX_MAX_INPUTS]>, src: &Vec<[i32; MIX_MAX_INPUTS]>| {
            if dst.len() == src.len() {
                dst.clone_from(src);
            }
        };
        copy(&mut self.zero_mixer.weights, &snapshot.zero_weights);
        copy(&mut self.pm1_mixer.weights, &snapshot.pm1_weights);
        copy(&mut self.sign_mixer.weights, &snapshot.sign_weights);
        if self.zero_models.p.len() == snapshot.zero_models.len() {
            self.zero_models.p.clone_from(&snapshot.zero_models);
        }
        if self.pm1_models.p.len() == snapshot.pm1_models.len() {
            self.pm1_models.p.clone_from(&snapshot.pm1_models);
        }
        if self.sign_model.len() == snapshot.sign_model.len() {
            self.sign_model.clone_from(&snapshot.sign_model);
        }
        if let (Some(apm), Some(table)) = (self.zero_apm.as_mut(), snapshot.zero_apm.as_ref()) {
            if apm.table.len() == table.len() {
                apm.table.clone_from(table);
            }
        }
    }
}

impl PhaseCounters {
    fn snapshot(&self) -> Vec<u32> {
        self.p_same.to_vec()
    }

    fn restore(&mut self, saved: &[u32]) {
        if saved.len() == self.p_same.len() {
            self.p_same.copy_from_slice(saved);
        }
    }
}

struct ContextModelEnvelope {
    p_zero: [u32; N_CTX_ZERO_V12 * AMPLITUDE_STATES * CODON_STATES],
    p_pm1: [u32; N_CTX_PM1_V12 * AMPLITUDE_STATES * CODON_STATES],
    energy: u32,
    prev_nz: bool,
    run_zeros: u32,
    prev_was_pm1: bool,
    turing_level: u8,
    amplitude: u8,
    codon_class: u8,
    amplitude_splits_pm1: bool,
}

impl ContextModelEnvelope {
    fn new(amplitude_splits_pm1: bool) -> Self {
        Self {
            p_zero: [14000; N_CTX_ZERO_V12 * AMPLITUDE_STATES * CODON_STATES],
            p_pm1: [12000; N_CTX_PM1_V12 * AMPLITUDE_STATES * CODON_STATES],
            energy: 0,
            prev_nz: false,
            run_zeros: 0,
            prev_was_pm1: false,
            turing_level: 0,
            amplitude: 0,
            codon_class: 0,
            amplitude_splits_pm1,
        }
    }

    #[inline]
    fn set_external_context(&mut self, turing_bucket: u8, amplitude: u8, codon_class: u8) {
        self.turing_level = turing_bucket.min(3);
        self.amplitude = amplitude.min(AMPLITUDE_STATES as u8 - 1);
        self.codon_class = codon_class.min(CODON_STATES as u8 - 1);
    }

    #[inline]
    fn zero_ctx_index(&self) -> usize {
        let run = run_class(self.run_zeros);
        let previous_nonzero = usize::from(self.prev_nz);
        let energy = energy_bucket(self.energy);
        let base = run * 32 + previous_nonzero * 16 + energy * 4 + self.turing_level as usize;
        base + self.amplitude as usize * N_CTX_ZERO_V12
            + self.codon_class as usize * N_CTX_ZERO_V12 * AMPLITUDE_STATES
    }

    #[inline]
    fn pm1_ctx_index(&self) -> usize {
        let base = energy_bucket(self.energy) * 8
            + usize::from(self.prev_was_pm1) * 4
            + self.turing_level as usize;
        let with_codon = base + self.codon_class as usize * N_CTX_PM1_V12;
        if self.amplitude_splits_pm1 {
            with_codon + self.amplitude as usize * N_CTX_PM1_V12 * CODON_STATES
        } else {
            with_codon
        }
    }

    #[inline]
    fn get_p_zero(&self) -> u32 {
        self.p_zero[self.zero_ctx_index()].clamp(1, PROB_SCALE - 1)
    }

    #[inline]
    fn get_p_pm1(&self) -> u32 {
        self.p_pm1[self.pm1_ctx_index()].clamp(1, PROB_SCALE - 1)
    }

    /// Running IIR magnitude scale on the `0..=ENERGY_SCALE` ladder. Both sides
    /// hold the identical value, so it can index a scale-aware envelope for
    /// free.
    #[inline]
    fn energy(&self) -> u32 {
        self.energy
    }

    /// Cycle 27 — re-enter the energy IIR at a block boundary.
    ///
    /// The IIR runs over the flat coefficient sequence and therefore crosses
    /// block boundaries. Blocks are truncated at their EOB, so the last symbol
    /// of a block is its last *nonzero* coefficient — typically a magnitude-1
    /// tail value. The first coefficient of the next block is a low-frequency
    /// one, typically the largest in the block. So the model predicts the
    /// loudest sample of a block from the quietest sample of its predecessor.
    ///
    /// Only v25 streams do this; earlier grammars carry the IIR straight
    /// across the boundary and must keep doing so to stay decodable.
    #[inline]
    fn enter_block(&mut self, primed: u32) {
        self.energy = primed;
    }

    /// Re-enter an already-ionized channel: adopt tables a previous layer left.
    fn load(&mut self, tables: &EnvelopeTables) {
        self.p_zero.copy_from_slice(&tables.p_zero);
        self.p_pm1.copy_from_slice(&tables.p_pm1);
    }

    /// Leave the channel ionized for whatever layer comes next.
    fn store(&self, tables: &mut EnvelopeTables) {
        tables.p_zero.copy_from_slice(&self.p_zero);
        tables.p_pm1.copy_from_slice(&self.p_pm1);
    }

    #[inline]
    fn run_bucket(&self) -> usize {
        run_class(self.run_zeros)
    }

    #[inline]
    fn energy_index(&self) -> usize {
        energy_bucket(self.energy)
    }

    fn update(&mut self, symbol: i16) {
        let is_zero = symbol == 0;
        let is_pm1 = symbol.unsigned_abs() == 1;
        let absolute = symbol.unsigned_abs() as u32;
        let zero_context = self.zero_ctx_index();
        if is_zero {
            self.p_zero[zero_context] += (PROB_SCALE - self.p_zero[zero_context]) >> ADAPT_SHIFT;
        } else {
            self.p_zero[zero_context] -= self.p_zero[zero_context] >> ADAPT_SHIFT;
            let pm1_context = self.pm1_ctx_index();
            if is_pm1 {
                self.p_pm1[pm1_context] += (PROB_SCALE - self.p_pm1[pm1_context]) >> ADAPT_SHIFT;
            } else {
                self.p_pm1[pm1_context] -= self.p_pm1[pm1_context] >> ADAPT_SHIFT;
            }
        }
        let sample_energy = (absolute * 32).min(ENERGY_SCALE);
        self.energy = (ENERGY_ALPHA * self.energy + (ENERGY_SCALE - ENERGY_ALPHA) * sample_energy)
            / ENERGY_SCALE;
        if is_zero {
            self.run_zeros = self.run_zeros.saturating_add(1);
        } else {
            self.run_zeros = 0;
            self.prev_was_pm1 = is_pm1;
        }
        self.prev_nz = !is_zero;
    }
}

/// Adaptive agreement counters: slot 0 tracks the historical predictor
/// (C1 -> C2 residual, C1 vertical continuation), slots 1.. track one spectral
/// phase class each.
struct PhaseCounters {
    p_same: [u32; PHASE_CLASS_COUNT * AMPLITUDE_STATES],
    legacy_shift: u32,
    spectral_shift: u32,
    context_mode: u8,
    /// Cycle 21: did the separable law hold the previous time it fired inside
    /// the current block? `0` unknown, `1` held, `2` failed. Reset at every
    /// block boundary, so the witness is causal and costs nothing.
    separable_memory: u8,
    block: u32,
}

impl PhaseCounters {
    fn new(legacy_initial: u32, legacy_shift: u32, config: &SpectralEnvelopeConfig) -> Self {
        let mut p_same = [config.phase_initial_p_same.clamp(1, PROB_SCALE - 1);
            PHASE_CLASS_COUNT * AMPLITUDE_STATES];
        // The historical predictor keeps its own tuned prior and never splits,
        // so only its first slot is ever read.
        for slot in 0..AMPLITUDE_STATES {
            p_same[PHASE_NONE as usize * AMPLITUDE_STATES + slot] =
                legacy_initial.clamp(1, PROB_SCALE - 1);
        }
        Self {
            p_same,
            legacy_shift: legacy_shift.clamp(1, 15),
            spectral_shift: config.phase_adapt_shift.clamp(1, 15),
            context_mode: config.phase_context,
            separable_memory: 0,
            block: u32::MAX,
        }
    }

    /// A block boundary clears every intra-block memory.
    #[inline]
    fn enter_block(&mut self, block: u32) {
        if block != self.block {
            self.block = block;
            self.separable_memory = 0;
        }
    }

    /// Which counter of the class carries this coefficient.
    #[inline]
    fn sub(&self, class: u8, energy: u8, turing: u8, size_class: u8) -> usize {
        if class == PHASE_NONE {
            return 0;
        }
        let separable_only = class == PHASE_SEPARABLE;
        let raw = match self.context_mode {
            PHASE_CONTEXT_ENERGY_ALL => energy as usize,
            PHASE_CONTEXT_SEPARABLE_MEMORY if separable_only => self.separable_memory as usize,
            PHASE_CONTEXT_SEPARABLE_TURING if separable_only => turing as usize,
            PHASE_CONTEXT_SEPARABLE_ENERGY if separable_only => energy as usize,
            PHASE_CONTEXT_SEPARABLE_SIZE if separable_only => size_class as usize,
            _ => 0,
        };
        raw.min(AMPLITUDE_STATES - 1)
    }

    #[inline]
    fn memory(&self) -> u8 {
        self.separable_memory
    }

    #[inline]
    fn get(&self, class: u8, energy: u8, turing: u8, size_class: u8) -> u32 {
        let slot = class as usize * AMPLITUDE_STATES + self.sub(class, energy, turing, size_class);
        self.p_same[slot].clamp(1, PROB_SCALE - 1)
    }

    #[inline]
    fn update(&mut self, class: u8, energy: u8, turing: u8, size_class: u8, same: bool) {
        let shift = if class == PHASE_NONE {
            self.legacy_shift
        } else {
            self.spectral_shift
        };
        let index = class as usize * AMPLITUDE_STATES + self.sub(class, energy, turing, size_class);
        let slot = &mut self.p_same[index];
        if same {
            *slot += (PROB_SCALE - *slot) >> shift;
        } else {
            *slot -= *slot >> shift;
        }
        if class == PHASE_SEPARABLE {
            self.separable_memory = if same { 1 } else { 2 };
        }
    }
}

/// Resolve which predictor governs the sign of the current coefficient.
/// The spectral law only takes over when the historical predictor is silent,
/// unless the configuration explicitly gives it precedence — measured and
/// rejected in cycle 20, kept as a laboratory primitive.
///
/// `spectral` is a closure so the neighbourhood's second-order slots are only
/// read when the coefficient is non-zero *and* the legacy predictor is silent.
#[inline]
fn resolve_phase<F>(
    legacy: Option<bool>,
    config: &SpectralEnvelopeConfig,
    spectral: F,
) -> Option<(u8, bool)>
where
    F: FnOnce() -> Option<(u8, bool)>,
{
    if legacy.is_none() || config.phase_takes_precedence {
        if let Some(resolved) = spectral() {
            return Some(resolved);
        }
    }
    legacy.map(|negative| (PHASE_NONE, negative))
}

/// Encode an AC band with the cycle-20 spectral envelope grammar.
///
/// `legacy_phase` carries the v19/v20 predictors (`-1` = none, `0` = predicted
/// positive, `1` = predicted negative). `segments` describes where each block
/// lives inside `coeffs` so that the neighbourhood can be rebuilt identically
/// on both sides.
#[allow(clippy::too_many_arguments)]
pub fn rans_encode_band_v21_envelope(
    coeffs: &[i16],
    turing_buckets: &[u8],
    codon_classes: &[u8],
    legacy_phase: &[i8],
    foreign_amplitudes: &[u8],
    segments: &[SpectralSegment],
    tables: &SpectralNeighborTables,
    config: &SpectralEnvelopeConfig,
    channel: usize,
    legacy_initial_p_same: u32,
    legacy_adapt_shift: u32,
    prime_block_energy: bool,
    block_echo: &[u32],
    channel_memory: &mut EnvelopeTables,
) -> Vec<u8> {
    if coeffs.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(coeffs.len(), codon_classes.len());
    debug_assert_eq!(coeffs.len(), legacy_phase.len());
    debug_assert_eq!(
        coeffs.len(),
        segments
            .iter()
            .map(|segment| segment.len as usize)
            .sum::<usize>(),
        "spectral segments must tile the flat sequence exactly"
    );

    struct DecisionEnvelope {
        is_zero_cum: u32,
        is_zero_freq: u32,
        is_large_cum: Option<u32>,
        is_large_freq: Option<u32>,
        sign_cum: Option<u32>,
        sign_freq: Option<u32>,
        eg_bits_payload: u32,
        eg_bits_len: u8,
    }

    let mut context = ContextModelEnvelope::new(config.amplitude_splits_pm1);
    context.load(channel_memory);
    let mut phase = PhaseCounters::new(legacy_initial_p_same, legacy_adapt_shift, config);
    let mut cursor = SpectralCursor::new(segments, tables, *config, channel);
    let mut decisions = Vec::<DecisionEnvelope>::with_capacity(coeffs.len());
    let mut mixing = MixingState::new(config, channel);
    let mut entered_block = u32::MAX;
    if let Some(saved) = channel_memory.mixing.as_ref() {
        mixing.restore(saved);
    }
    if let Some(saved) = channel_memory.phase.as_ref() {
        phase.restore(saved);
    }

    for (index, &value) in coeffs.iter().enumerate() {
        let turing = turing_buckets.get(index).copied().unwrap_or(0);
        let probed = cursor.probe(index, coeffs, context.energy());
        phase.enter_block(probed.block);
        if prime_block_energy && probed.block != entered_block {
            entered_block = probed.block;
            context.enter_block(
                block_echo
                    .get(probed.block as usize)
                    .copied()
                    .unwrap_or(BLOCK_ENERGY_PRIME),
            );
        }
        context.set_external_context(
            turing,
            probed.amplitude,
            codon_classes.get(index).copied().unwrap_or(0),
        );

        let is_zero = value == 0;
        let foreign = foreign_amplitudes.get(index).copied().unwrap_or(0);
        let p_zero = mixing.p_zero(&context, &probed, foreign);
        let (is_zero_cum, is_zero_freq) = if is_zero {
            (0, p_zero)
        } else {
            (p_zero, PROB_SCALE - p_zero)
        };
        let mut decision = DecisionEnvelope {
            is_zero_cum,
            is_zero_freq,
            is_large_cum: None,
            is_large_freq: None,
            sign_cum: None,
            sign_freq: None,
            eg_bits_payload: 0,
            eg_bits_len: 0,
        };

        if !is_zero {
            let absolute = value.unsigned_abs();
            let is_pm1 = absolute == 1;
            let p_pm1 = mixing.p_pm1(&context, &probed, foreign);
            let (large_cum, large_freq) = if is_pm1 {
                (0, p_pm1)
            } else {
                (p_pm1, PROB_SCALE - p_pm1)
            };
            decision.is_large_cum = Some(large_cum);
            decision.is_large_freq = Some(large_freq);

            let negative = value < 0;
            let legacy = legacy_phase
                .get(index)
                .copied()
                .filter(|&predictor| predictor >= 0)
                .map(|predictor| predictor != 0);
            let resolved = resolve_phase(legacy, config, || cursor.phase(&probed, index, coeffs));
            if let Some((class, predicted_negative)) = resolved {
                let memory = phase.memory();
                let p_same = mixing.p_same(
                    phase.get(class, probed.energy, turing, probed.size_class),
                    class,
                    memory,
                    probed.energy,
                );
                let same = negative == predicted_negative;
                let (cum, freq) = if same {
                    (0, p_same)
                } else {
                    (p_same, PROB_SCALE - p_same)
                };
                decision.sign_cum = Some(cum);
                decision.sign_freq = Some(freq);
                mixing.update_sign(same);
                phase.update(class, probed.energy, turing, probed.size_class, same);
            } else {
                let half = PROB_SCALE / 2;
                decision.sign_cum = Some(if negative { half } else { 0 });
                decision.sign_freq = Some(half);
            }

            if !is_pm1 {
                let n1 = (absolute as u32).saturating_sub(2) + 1;
                let nbits = 32 - n1.leading_zeros();
                let prefix_zeros = nbits - 1;
                let mut payload = 0u32;
                let mut length = prefix_zeros as u8;
                payload |= 1 << length;
                length += 1;
                for bit in (0..prefix_zeros).rev() {
                    if (n1 >> bit) & 1 != 0 {
                        payload |= 1 << length;
                    }
                    length += 1;
                }
                decision.eg_bits_payload = payload;
                decision.eg_bits_len = length;
            }
        }
        decisions.push(decision);
        mixing.update_zero(is_zero);
        if !is_zero {
            mixing.update_pm1(value.unsigned_abs() == 1);
        }
        context.update(value);
    }

    let mut encoder = RansEncoder::new();
    for decision in decisions.iter().rev() {
        if let Some(large_cum) = decision.is_large_cum {
            if large_cum > 0 {
                for bit in (0..decision.eg_bits_len).rev() {
                    encoder.put_bit((decision.eg_bits_payload >> bit) & 1 != 0);
                }
            }
        }
        if let (Some(cum), Some(freq)) = (decision.sign_cum, decision.sign_freq) {
            encoder.put(cum, freq);
        }
        if let (Some(cum), Some(freq)) = (decision.is_large_cum, decision.is_large_freq) {
            encoder.put(cum, freq);
        }
        encoder.put(decision.is_zero_cum, decision.is_zero_freq);
    }
    context.store(channel_memory);
    channel_memory.mixing = Some(mixing.snapshot());
    channel_memory.phase = Some(phase.snapshot());
    encoder.finish()
}

/// Decode the cycle-20 spectral envelope grammar. Every context is rebuilt
/// from symbols already produced in this very call or delivered by a previous
/// progressive layer, so nothing here reads side information.
#[allow(clippy::too_many_arguments)]
pub fn rans_decode_band_v21_envelope(
    data: &[u8],
    n_coeffs: usize,
    turing_buckets: &[u8],
    codon_classes: &[u8],
    legacy_phase: &[PhasePredictorRef],
    foreign_amplitudes: &[u8],
    segments: &[SpectralSegment],
    tables: &SpectralNeighborTables,
    config: &SpectralEnvelopeConfig,
    channel: usize,
    legacy_initial_p_same: u32,
    legacy_adapt_shift: u32,
    prime_block_energy: bool,
    block_echo: &[u32],
    channel_memory: &mut EnvelopeTables,
) -> Result<(Vec<i16>, usize), RansDecodeError> {
    if n_coeffs == 0 {
        if data.is_empty() {
            return Ok((Vec::new(), 0));
        }
        return Err(RansDecodeError::TrailingData {
            consumed: 0,
            available: data.len(),
        });
    }
    if turing_buckets.len() != n_coeffs
        || codon_classes.len() != n_coeffs
        || legacy_phase.len() != n_coeffs
    {
        return Err(RansDecodeError::InvalidContextLength {
            expected: n_coeffs,
            got: turing_buckets
                .len()
                .min(codon_classes.len())
                .min(legacy_phase.len()),
        });
    }
    let covered = segments
        .iter()
        .map(|segment| segment.len as usize)
        .sum::<usize>();
    if covered != n_coeffs {
        return Err(RansDecodeError::InvalidContextLength {
            expected: n_coeffs,
            got: covered,
        });
    }

    let mut decoder = RansDecoder::new_checked(data)?;
    let mut context = ContextModelEnvelope::new(config.amplitude_splits_pm1);
    context.load(channel_memory);
    let mut phase = PhaseCounters::new(legacy_initial_p_same, legacy_adapt_shift, config);
    let mut cursor = SpectralCursor::new(segments, tables, *config, channel);
    let mut result = Vec::<i16>::with_capacity(n_coeffs);
    let mut mixing = MixingState::new(config, channel);
    let mut entered_block = u32::MAX;
    if let Some(saved) = channel_memory.mixing.as_ref() {
        mixing.restore(saved);
    }
    if let Some(saved) = channel_memory.phase.as_ref() {
        phase.restore(saved);
    }

    for index in 0..n_coeffs {
        let turing = turing_buckets[index];
        let probed = cursor.probe(index, &result, context.energy());
        phase.enter_block(probed.block);
        if prime_block_energy && probed.block != entered_block {
            entered_block = probed.block;
            context.enter_block(
                block_echo
                    .get(probed.block as usize)
                    .copied()
                    .unwrap_or(BLOCK_ENERGY_PRIME),
            );
        }
        context.set_external_context(turing, probed.amplitude, codon_classes[index]);

        let foreign = foreign_amplitudes.get(index).copied().unwrap_or(0);
        let p_zero = mixing.p_zero(&context, &probed, foreign);
        let is_zero = decoder.get() < p_zero;
        if is_zero {
            decoder.advance_checked(0, p_zero)?;
            result.push(0);
            mixing.update_zero(true);
            context.update(0);
            continue;
        }
        decoder.advance_checked(p_zero, PROB_SCALE - p_zero)?;
        mixing.update_zero(false);

        let p_pm1 = mixing.p_pm1(&context, &probed, foreign);
        let is_pm1 = decoder.get() < p_pm1;
        if is_pm1 {
            decoder.advance_checked(0, p_pm1)?;
        } else {
            decoder.advance_checked(p_pm1, PROB_SCALE - p_pm1)?;
        }
        mixing.update_pm1(is_pm1);

        let legacy = match legacy_phase[index] {
            PhasePredictorRef::None => None,
            PhasePredictorRef::Known(negative) => Some(negative),
            PhasePredictorRef::Decoded {
                index: predictor_index,
                invert,
            } => {
                let Some(&value) = result.get(predictor_index) else {
                    return Err(RansDecodeError::InvalidState);
                };
                (value != 0).then_some((value < 0) ^ invert)
            }
        };

        let resolved = resolve_phase(legacy, config, || cursor.phase(&probed, index, &result));
        let negative = if let Some((class, predicted_negative)) = resolved {
            let memory = phase.memory();
            let p_same = mixing.p_same(
                phase.get(class, probed.energy, turing, probed.size_class),
                class,
                memory,
                probed.energy,
            );
            let same = decoder.get() < p_same;
            if same {
                decoder.advance_checked(0, p_same)?;
            } else {
                decoder.advance_checked(p_same, PROB_SCALE - p_same)?;
            }
            mixing.update_sign(same);
            phase.update(class, probed.energy, turing, probed.size_class, same);
            if same {
                predicted_negative
            } else {
                !predicted_negative
            }
        } else {
            decoder.get_bit_checked()?
        };

        let magnitude = if is_pm1 {
            1u32
        } else {
            let mut prefix_zeros = 0u32;
            loop {
                if decoder.get_bit_checked()? {
                    break;
                }
                prefix_zeros += 1;
                if prefix_zeros > 14 {
                    return Err(RansDecodeError::InvalidExpGolombPrefix(prefix_zeros));
                }
            }
            let mut n1 = 1u32 << prefix_zeros;
            for bit in (0..prefix_zeros).rev() {
                if decoder.get_bit_checked()? {
                    n1 |= 1 << bit;
                }
            }
            n1 - 1 + 2
        };
        if magnitude > i16::MAX as u32 + u32::from(negative) {
            return Err(RansDecodeError::InvalidState);
        }
        let value = if negative {
            -(magnitude as i32) as i16
        } else {
            magnitude as i16
        };
        result.push(value);
        context.update(value);
    }

    if decoder.state != RANS_BYTE_L {
        return Err(RansDecodeError::InvalidState);
    }
    if decoder.pos() != data.len() {
        return Err(RansDecodeError::TrailingData {
            consumed: decoder.pos(),
            available: data.len(),
        });
    }
    context.store(channel_memory);
    channel_memory.mixing = Some(mixing.snapshot());
    channel_memory.phase = Some(phase.snapshot());
    Ok((result, decoder.pos()))
}

// ======================================================================
// Paeth Context Model
// ======================================================================

struct PaethContextModel {
    /// P(zero) for each context: prev_sign(2) x prev_mag_bucket(4)
    p_zero: [u32; N_PAETH_CTX],
    prev_sign: u8,       // 0 = non-negative, 1 = negative
    prev_mag_bucket: u8, // 0..3
}

impl PaethContextModel {
    fn new() -> Self {
        // Paeth residuals are mostly near zero. Init P(zero) ~ 40%.
        let mut p_zero = [0u32; N_PAETH_CTX];
        for p in p_zero.iter_mut() {
            *p = 6554; // ~40% of 16384
        }
        Self {
            p_zero,
            prev_sign: 0,
            prev_mag_bucket: 0,
        }
    }

    /// Classify magnitude into 4 buckets: 0, 1, 2-3, 4+
    #[inline]
    fn mag_bucket(abs_val: u16) -> u8 {
        match abs_val {
            0 => 0,
            1 => 1,
            2 | 3 => 2,
            _ => 3,
        }
    }

    #[inline]
    fn ctx(&self) -> usize {
        self.prev_sign as usize * 4 + self.prev_mag_bucket as usize
    }

    #[inline]
    fn get_p_zero(&self) -> u32 {
        self.p_zero[self.ctx()].clamp(1, PROB_SCALE - 1)
    }

    fn update(&mut self, value: i16) {
        let is_zero = value == 0;
        let c = self.ctx();
        if is_zero {
            self.p_zero[c] += (PROB_SCALE - self.p_zero[c]) >> PAETH_ADAPT_SHIFT;
        } else {
            self.p_zero[c] -= self.p_zero[c] >> PAETH_ADAPT_SHIFT;
        }
        let abs_val = value.unsigned_abs();
        self.prev_sign = if value < 0 { 1 } else { 0 };
        self.prev_mag_bucket = Self::mag_bucket(abs_val);
    }
}

// ======================================================================
// Band Encoding / Decoding (detail bands)
// ======================================================================

/// Encode a quantized band (Morton-ordered i16 values) using rANS + context model.
/// Returns compressed bytes.
pub fn rans_encode_band(coeffs: &[i16]) -> Vec<u8> {
    if coeffs.is_empty() {
        return Vec::new();
    }

    let n = coeffs.len();

    // rANS encodes in reverse order: we process symbols from last to first,
    // and the encoder outputs bytes in reverse.
    let mut enc = RansEncoder::new();
    let mut ctx = ContextModel::new();

    // We need to encode backwards, but the context model depends on forward state.
    // Solution: run forward pass to collect all decisions, then encode backwards.
    struct Decision {
        is_zero_cum: u32,
        is_zero_freq: u32,
        is_large_cum: Option<u32>,
        is_large_freq: Option<u32>,
        sign: Option<bool>,
        // For large values: unary bits for |v|-2
        unary_len: u32,
    }

    let mut decisions = Vec::with_capacity(n);

    for &val in coeffs.iter() {
        let is_zero = val == 0;
        let p_zero = ctx.get_p_zero();

        let (iz_cum, iz_freq) = if is_zero {
            // Symbol "zero": cum=0, freq=p_zero
            (0, p_zero)
        } else {
            // Symbol "nonzero": cum=p_zero, freq=PROB_SCALE-p_zero
            (p_zero, PROB_SCALE - p_zero)
        };

        let mut dec = Decision {
            is_zero_cum: iz_cum,
            is_zero_freq: iz_freq,
            is_large_cum: None,
            is_large_freq: None,
            sign: None,
            unary_len: 0,
        };

        if !is_zero {
            let abs_val = val.unsigned_abs();
            let is_pm1 = abs_val == 1;
            let p_pm1 = ctx.get_p_pm1();

            let (il_cum, il_freq) = if is_pm1 {
                // "pm1": cum=0, freq=p_pm1
                (0, p_pm1)
            } else {
                // "large": cum=p_pm1, freq=PROB_SCALE-p_pm1
                (p_pm1, PROB_SCALE - p_pm1)
            };
            dec.is_large_cum = Some(il_cum);
            dec.is_large_freq = Some(il_freq);
            dec.sign = Some(val < 0);

            if !is_pm1 {
                // Unary coding for magnitude: |v| - 2 ones followed by a zero
                dec.unary_len = (abs_val as u32).saturating_sub(2);
            }
        }

        decisions.push(dec);
        ctx.update(val);
    }

    // Encode backwards
    for dec in decisions.iter().rev() {
        // Encode in reverse order of the decision tree, innermost first:
        // Unary bits (innermost) -> sign -> is_large -> is_zero (outermost)

        // 1. Unary magnitude bits: only for "large" values (|v| >= 2).
        //    For large values, is_large_cum > 0 (equals p_pm1), while for pm1 it's 0.
        if let Some(il_cum) = dec.is_large_cum {
            if il_cum > 0 {
                // This is a large value. Encode (|v|-2) ones + one terminating zero.
                // Terminator first (innermost = last decoded)
                enc.put_bit(false);
                for _ in 0..dec.unary_len {
                    enc.put_bit(true);
                }
            }
        }

        // 2. Sign bit
        if let Some(sign) = dec.sign {
            enc.put_bit(sign);
        }

        // 3. Is large (|v|>=2) or pm1
        if let (Some(cum), Some(freq)) = (dec.is_large_cum, dec.is_large_freq) {
            enc.put(cum, freq);
        }

        // 4. Is zero
        enc.put(dec.is_zero_cum, dec.is_zero_freq);
    }

    enc.finish()
}

/// Decode a band from rANS-compressed bytes.
/// Returns (quantized coefficients, bytes consumed).
pub fn rans_decode_band(data: &[u8], n_coeffs: usize) -> (Vec<i16>, usize) {
    if n_coeffs == 0 {
        return (Vec::new(), 0);
    }

    let mut dec = RansDecoder::new(data);
    let mut ctx = ContextModel::new();
    let mut result = Vec::with_capacity(n_coeffs);

    for _ in 0..n_coeffs {
        let p_zero = ctx.get_p_zero();

        // Decode is_zero
        let cum = dec.get();
        let is_zero = cum < p_zero;
        if is_zero {
            dec.advance(0, p_zero);
            result.push(0);
            ctx.update(0);
        } else {
            dec.advance(p_zero, PROB_SCALE - p_zero);

            // Decode is_pm1 vs large
            let p_pm1 = ctx.get_p_pm1();
            let cum2 = dec.get();
            let is_pm1 = cum2 < p_pm1;

            if is_pm1 {
                dec.advance(0, p_pm1);
                // Decode sign
                let negative = dec.get_bit();
                let val = if negative { -1i16 } else { 1i16 };
                result.push(val);
                ctx.update(val);
            } else {
                dec.advance(p_pm1, PROB_SCALE - p_pm1);
                // Decode sign
                let negative = dec.get_bit();
                // Decode magnitude via unary: count ones until we see a zero
                let mut mag: u32 = 2; // minimum magnitude for "large"
                loop {
                    let bit = dec.get_bit();
                    if !bit {
                        break;
                    }
                    mag += 1;
                }
                let val = if negative { -(mag as i16) } else { mag as i16 };
                result.push(val);
                ctx.update(val);
            }
        }
    }

    (result, dec.pos())
}

// ======================================================================
// v11 Band Encoding / Decoding (hyper-sparse contexts)
// ======================================================================

/// Encode a quantized band using v11 hyper-sparse rANS contexts.
/// 6 run-class buckets (vs 4 in v8) for better compression of denoised regions.
pub fn rans_encode_band_v11(coeffs: &[i16]) -> Vec<u8> {
    if coeffs.is_empty() {
        return Vec::new();
    }

    let n = coeffs.len();
    let mut enc = RansEncoder::new();
    let mut ctx = ContextModelV11::new();

    struct Decision {
        is_zero_cum: u32,
        is_zero_freq: u32,
        is_large_cum: Option<u32>,
        is_large_freq: Option<u32>,
        sign: Option<bool>,
        unary_len: u32,
    }

    let mut decisions = Vec::with_capacity(n);

    for &val in coeffs.iter() {
        let is_zero = val == 0;
        let p_zero = ctx.get_p_zero();

        let (iz_cum, iz_freq) = if is_zero {
            (0, p_zero)
        } else {
            (p_zero, PROB_SCALE - p_zero)
        };

        let mut dec = Decision {
            is_zero_cum: iz_cum,
            is_zero_freq: iz_freq,
            is_large_cum: None,
            is_large_freq: None,
            sign: None,
            unary_len: 0,
        };

        if !is_zero {
            let abs_val = val.unsigned_abs();
            let is_pm1 = abs_val == 1;
            let p_pm1 = ctx.get_p_pm1();

            let (il_cum, il_freq) = if is_pm1 {
                (0, p_pm1)
            } else {
                (p_pm1, PROB_SCALE - p_pm1)
            };
            dec.is_large_cum = Some(il_cum);
            dec.is_large_freq = Some(il_freq);
            dec.sign = Some(val < 0);

            if !is_pm1 {
                dec.unary_len = (abs_val as u32).saturating_sub(2);
            }
        }

        decisions.push(dec);
        ctx.update(val);
    }

    // Encode backwards
    for dec in decisions.iter().rev() {
        if let Some(il_cum) = dec.is_large_cum {
            if il_cum > 0 {
                enc.put_bit(false);
                for _ in 0..dec.unary_len {
                    enc.put_bit(true);
                }
            }
        }
        if let Some(sign) = dec.sign {
            enc.put_bit(sign);
        }
        if let (Some(cum), Some(freq)) = (dec.is_large_cum, dec.is_large_freq) {
            enc.put(cum, freq);
        }
        enc.put(dec.is_zero_cum, dec.is_zero_freq);
    }

    enc.finish()
}

/// Decode a band from v11 hyper-sparse rANS-compressed bytes.
/// Returns (quantized coefficients, bytes consumed).
pub fn rans_decode_band_v11(data: &[u8], n_coeffs: usize) -> (Vec<i16>, usize) {
    if n_coeffs == 0 {
        return (Vec::new(), 0);
    }

    let mut dec = RansDecoder::new(data);
    let mut ctx = ContextModelV11::new();
    let mut result = Vec::with_capacity(n_coeffs);

    for _ in 0..n_coeffs {
        let p_zero = ctx.get_p_zero();

        let cum = dec.get();
        let is_zero = cum < p_zero;
        if is_zero {
            dec.advance(0, p_zero);
            result.push(0);
            ctx.update(0);
        } else {
            dec.advance(p_zero, PROB_SCALE - p_zero);

            let p_pm1 = ctx.get_p_pm1();
            let cum2 = dec.get();
            let is_pm1 = cum2 < p_pm1;

            if is_pm1 {
                dec.advance(0, p_pm1);
                let negative = dec.get_bit();
                let val = if negative { -1i16 } else { 1i16 };
                result.push(val);
                ctx.update(val);
            } else {
                dec.advance(p_pm1, PROB_SCALE - p_pm1);
                let negative = dec.get_bit();
                let mut mag: u32 = 2;
                loop {
                    let bit = dec.get_bit();
                    if !bit {
                        break;
                    }
                    mag += 1;
                }
                let val = if negative { -(mag as i16) } else { mag as i16 };
                result.push(val);
                ctx.update(val);
            }
        }
    }

    (result, dec.pos())
}

// ======================================================================
// v12 Bayesian rANS Encode / Decode (with per-coefficient turing_bucket)
//
// Uses Rice-Golomb coding for magnitudes >= 2, with adaptive order k
// derived from the context's smoothed energy (zero-cost signaling).

/// Derive Rice parameter k from the context's smoothed energy.
/// energy is scaled by 32 (sample_energy = abs_val * 32).
/// k = floor(log2(energy / 32 + 1)), clamped to [0, 6].
#[inline]
fn rice_k(energy: u32) -> u32 {
    let avg_mag = (energy / 32).max(1);
    (32 - avg_mag.leading_zeros()).saturating_sub(1).clamp(0, 6)
}
// ======================================================================

/// Encode a detail band with v12 Bayesian context model.
/// `turing_buckets` maps each coefficient position to a turing_bucket (0-3).
/// Length must match `coeffs.len()`.
pub fn rans_encode_band_v12(coeffs: &[i16], turing_buckets: &[u8]) -> Vec<u8> {
    if coeffs.is_empty() {
        return Vec::new();
    }

    let n = coeffs.len();
    let mut enc = RansEncoder::new();
    let mut ctx = ContextModelV12::new();

    struct DecisionV12 {
        is_zero_cum: u32,
        is_zero_freq: u32,
        is_large_cum: Option<u32>,
        is_large_freq: Option<u32>,
        sign: Option<bool>,
        eg_bits_payload: u32, // Exp-Golomb bits packed (LSB first)
        eg_bits_len: u8,      // number of valid bits (0 = none)
    }

    let mut decisions = Vec::<DecisionV12>::with_capacity(n);

    for (i, &val) in coeffs.iter().enumerate() {
        if i < turing_buckets.len() {
            ctx.set_turing_level(turing_buckets[i]);
        }

        let is_zero = val == 0;
        let p_zero = ctx.get_p_zero();

        let (iz_cum, iz_freq) = if is_zero {
            (0, p_zero)
        } else {
            (p_zero, PROB_SCALE - p_zero)
        };

        let mut dec = DecisionV12 {
            is_zero_cum: iz_cum,
            is_zero_freq: iz_freq,
            is_large_cum: None,
            is_large_freq: None,
            sign: None,
            eg_bits_payload: 0,
            eg_bits_len: 0,
        };

        if !is_zero {
            let abs_val = val.unsigned_abs();
            let is_pm1 = abs_val == 1;
            let p_pm1 = ctx.get_p_pm1();

            let (il_cum, il_freq) = if is_pm1 {
                (0, p_pm1)
            } else {
                (p_pm1, PROB_SCALE - p_pm1)
            };
            dec.is_large_cum = Some(il_cum);
            dec.is_large_freq = Some(il_freq);
            dec.sign = Some(val < 0);

            if !is_pm1 {
                // Exp-Golomb order 0 of (abs_val - 2), packed into u32
                let n_eg = (abs_val as u32).saturating_sub(2);
                let n1 = n_eg + 1;
                let nbits = 32 - n1.leading_zeros(); // total significant bits
                let prefix_zeros = nbits - 1;
                let total_bits = 2 * nbits - 1; // prefix zeros + 1 + suffix
                // Pack: prefix zeros, then 1, then suffix (LSB first order)
                let mut payload: u32 = 0;
                let mut len: u8 = 0;
                for _ in 0..prefix_zeros {
                    len += 1;
                } // zeros are already 0 in payload
                payload |= 1 << len;
                len += 1; // the 1 bit
                for b in (0..prefix_zeros).rev() {
                    if (n1 >> b) & 1 != 0 {
                        payload |= 1 << len;
                    }
                    len += 1;
                }
                dec.eg_bits_payload = payload;
                dec.eg_bits_len = len;
            }
        }

        decisions.push(dec);
        ctx.update(val);
    }

    // Encode backwards
    for dec in decisions.iter().rev() {
        if let Some(il_cum) = dec.is_large_cum {
            if il_cum > 0 {
                // Unpack EG bits from u32 (reversed: MSB first)
                for i in (0..dec.eg_bits_len).rev() {
                    enc.put_bit((dec.eg_bits_payload >> i) & 1 != 0);
                }
            }
        }
        if let Some(sign) = dec.sign {
            enc.put_bit(sign);
        }
        if let (Some(cum), Some(freq)) = (dec.is_large_cum, dec.is_large_freq) {
            enc.put(cum, freq);
        }
        enc.put(dec.is_zero_cum, dec.is_zero_freq);
    }

    enc.finish()
}

/// Experimental v16 witness: the v12 grammar is unchanged, but its Bayesian
/// state is split by a decoded dyadic-parent support bit. No side information
/// is encoded; a decoder can derive the bit from the already decoded lower
/// frequency coefficient. Kept separate from the production path until the
/// corpus benchmark proves that the extra context pays for itself.
pub fn rans_encode_band_v12_parent(
    coeffs: &[i16],
    turing_buckets: &[u8],
    parent_nonzero: &[bool],
) -> Vec<u8> {
    if coeffs.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(coeffs.len(), parent_nonzero.len());

    struct DecisionV12Parent {
        is_zero_cum: u32,
        is_zero_freq: u32,
        is_large_cum: Option<u32>,
        is_large_freq: Option<u32>,
        sign: Option<bool>,
        eg_bits_payload: u32,
        eg_bits_len: u8,
    }

    let mut context = ContextModelWavelet::new();
    let mut decisions = Vec::<DecisionV12Parent>::with_capacity(coeffs.len());

    for (i, &value) in coeffs.iter().enumerate() {
        context.set_external_context(
            turing_buckets.get(i).copied().unwrap_or(0),
            parent_nonzero.get(i).copied().unwrap_or(false),
        );

        let is_zero = value == 0;
        let p_zero = context.get_p_zero();
        let (is_zero_cum, is_zero_freq) = if is_zero {
            (0, p_zero)
        } else {
            (p_zero, PROB_SCALE - p_zero)
        };
        let mut decision = DecisionV12Parent {
            is_zero_cum,
            is_zero_freq,
            is_large_cum: None,
            is_large_freq: None,
            sign: None,
            eg_bits_payload: 0,
            eg_bits_len: 0,
        };

        if !is_zero {
            let absolute = value.unsigned_abs();
            let is_pm1 = absolute == 1;
            let p_pm1 = context.get_p_pm1();
            let (is_large_cum, is_large_freq) = if is_pm1 {
                (0, p_pm1)
            } else {
                (p_pm1, PROB_SCALE - p_pm1)
            };
            decision.is_large_cum = Some(is_large_cum);
            decision.is_large_freq = Some(is_large_freq);
            decision.sign = Some(value < 0);

            if !is_pm1 {
                let n1 = (absolute as u32).saturating_sub(2) + 1;
                let nbits = 32 - n1.leading_zeros();
                let prefix_zeros = nbits - 1;
                let mut payload = 0u32;
                let mut len = prefix_zeros as u8;
                payload |= 1 << len;
                len += 1;
                for bit in (0..prefix_zeros).rev() {
                    if (n1 >> bit) & 1 != 0 {
                        payload |= 1 << len;
                    }
                    len += 1;
                }
                decision.eg_bits_payload = payload;
                decision.eg_bits_len = len;
            }
        }

        decisions.push(decision);
        context.update(value);
    }

    let mut encoder = RansEncoder::new();
    for decision in decisions.iter().rev() {
        if let Some(is_large_cum) = decision.is_large_cum {
            if is_large_cum > 0 {
                for bit in (0..decision.eg_bits_len).rev() {
                    encoder.put_bit((decision.eg_bits_payload >> bit) & 1 != 0);
                }
            }
        }
        if let Some(sign) = decision.sign {
            encoder.put_bit(sign);
        }
        if let (Some(cum), Some(freq)) = (decision.is_large_cum, decision.is_large_freq) {
            encoder.put(cum, freq);
        }
        encoder.put(decision.is_zero_cum, decision.is_zero_freq);
    }
    encoder.finish()
}

/// Exact color-codon witness. `codon_classes` are zero-bit classes reconstructed
/// from already coded components of the same L/C1/C2 triplet. When
/// `split_pm1` is false only P(zero) is conditioned; when true, the +/-1
/// magnitude decision is conditioned as well.
pub fn rans_encode_band_v17_codon(
    coeffs: &[i16],
    turing_buckets: &[u8],
    parent_nonzero: &[bool],
    codon_classes: &[u8],
    split_pm1: bool,
) -> Vec<u8> {
    if coeffs.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(coeffs.len(), parent_nonzero.len());
    debug_assert_eq!(coeffs.len(), codon_classes.len());

    struct DecisionCodon {
        is_zero_cum: u32,
        is_zero_freq: u32,
        is_large_cum: Option<u32>,
        is_large_freq: Option<u32>,
        sign: Option<bool>,
        eg_bits_payload: u32,
        eg_bits_len: u8,
    }

    let mut context = ContextModelCodon::new(split_pm1);
    let mut decisions = Vec::<DecisionCodon>::with_capacity(coeffs.len());
    for (index, &value) in coeffs.iter().enumerate() {
        context.set_external_context(
            turing_buckets.get(index).copied().unwrap_or(0),
            parent_nonzero.get(index).copied().unwrap_or(false),
            codon_classes.get(index).copied().unwrap_or(0),
        );

        let is_zero = value == 0;
        let p_zero = context.get_p_zero();
        let (is_zero_cum, is_zero_freq) = if is_zero {
            (0, p_zero)
        } else {
            (p_zero, PROB_SCALE - p_zero)
        };
        let mut decision = DecisionCodon {
            is_zero_cum,
            is_zero_freq,
            is_large_cum: None,
            is_large_freq: None,
            sign: None,
            eg_bits_payload: 0,
            eg_bits_len: 0,
        };

        if !is_zero {
            let absolute = value.unsigned_abs();
            let is_pm1 = absolute == 1;
            let p_pm1 = context.get_p_pm1();
            let (large_cum, large_freq) = if is_pm1 {
                (0, p_pm1)
            } else {
                (p_pm1, PROB_SCALE - p_pm1)
            };
            decision.is_large_cum = Some(large_cum);
            decision.is_large_freq = Some(large_freq);
            decision.sign = Some(value < 0);

            if !is_pm1 {
                let n1 = (absolute as u32).saturating_sub(2) + 1;
                let nbits = 32 - n1.leading_zeros();
                let prefix_zeros = nbits - 1;
                let mut payload = 0u32;
                let mut length = prefix_zeros as u8;
                payload |= 1 << length;
                length += 1;
                for bit in (0..prefix_zeros).rev() {
                    if (n1 >> bit) & 1 != 0 {
                        payload |= 1 << length;
                    }
                    length += 1;
                }
                decision.eg_bits_payload = payload;
                decision.eg_bits_len = length;
            }
        }
        decisions.push(decision);
        context.update(value);
    }

    let mut encoder = RansEncoder::new();
    for decision in decisions.iter().rev() {
        if let Some(large_cum) = decision.is_large_cum {
            if large_cum > 0 {
                for bit in (0..decision.eg_bits_len).rev() {
                    encoder.put_bit((decision.eg_bits_payload >> bit) & 1 != 0);
                }
            }
        }
        if let Some(sign) = decision.sign {
            encoder.put_bit(sign);
        }
        if let (Some(cum), Some(freq)) = (decision.is_large_cum, decision.is_large_freq) {
            encoder.put(cum, freq);
        }
        encoder.put(decision.is_zero_cum, decision.is_zero_freq);
    }
    encoder.finish()
}

/// Laboratory/production v19 phase coder. The magnitude/support grammar is identical
/// to v17, but a non-zero coefficient may borrow a causal phase predictor
/// from an earlier color component. `phase_predictors` uses -1 for no
/// predictor, 0 for positive and 1 for negative. In the predicted case rANS
/// codes phase agreement instead of an equiprobable raw sign bit.
pub fn rans_encode_band_v19_phase(
    coeffs: &[i16],
    turing_buckets: &[u8],
    parent_nonzero: &[bool],
    codon_classes: &[u8],
    phase_predictors: &[i8],
    initial_p_same: u32,
    phase_adapt_shift: u32,
) -> Vec<u8> {
    if coeffs.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(coeffs.len(), parent_nonzero.len());
    debug_assert_eq!(coeffs.len(), codon_classes.len());
    debug_assert_eq!(coeffs.len(), phase_predictors.len());

    struct DecisionPhase {
        is_zero_cum: u32,
        is_zero_freq: u32,
        is_large_cum: Option<u32>,
        is_large_freq: Option<u32>,
        sign_cum: Option<u32>,
        sign_freq: Option<u32>,
        eg_bits_payload: u32,
        eg_bits_len: u8,
    }

    let mut context = ContextModelCodon::new(true);
    let mut p_same = initial_p_same.clamp(1, PROB_SCALE - 1);
    let adapt_shift = phase_adapt_shift.clamp(1, 15);
    let mut decisions = Vec::<DecisionPhase>::with_capacity(coeffs.len());
    for (index, &value) in coeffs.iter().enumerate() {
        context.set_external_context(
            turing_buckets.get(index).copied().unwrap_or(0),
            parent_nonzero.get(index).copied().unwrap_or(false),
            codon_classes.get(index).copied().unwrap_or(0),
        );

        let is_zero = value == 0;
        let p_zero = context.get_p_zero();
        let (is_zero_cum, is_zero_freq) = if is_zero {
            (0, p_zero)
        } else {
            (p_zero, PROB_SCALE - p_zero)
        };
        let mut decision = DecisionPhase {
            is_zero_cum,
            is_zero_freq,
            is_large_cum: None,
            is_large_freq: None,
            sign_cum: None,
            sign_freq: None,
            eg_bits_payload: 0,
            eg_bits_len: 0,
        };

        if !is_zero {
            let absolute = value.unsigned_abs();
            let is_pm1 = absolute == 1;
            let p_pm1 = context.get_p_pm1();
            let (large_cum, large_freq) = if is_pm1 {
                (0, p_pm1)
            } else {
                (p_pm1, PROB_SCALE - p_pm1)
            };
            decision.is_large_cum = Some(large_cum);
            decision.is_large_freq = Some(large_freq);

            let negative = value < 0;
            let predictor = phase_predictors.get(index).copied().unwrap_or(-1);
            if predictor >= 0 {
                let same = negative == (predictor != 0);
                let (cum, freq) = if same {
                    (0, p_same)
                } else {
                    (p_same, PROB_SCALE - p_same)
                };
                decision.sign_cum = Some(cum);
                decision.sign_freq = Some(freq);
                if same {
                    p_same += (PROB_SCALE - p_same) >> adapt_shift;
                } else {
                    p_same -= p_same >> adapt_shift;
                }
            } else {
                let half = PROB_SCALE / 2;
                decision.sign_cum = Some(if negative { half } else { 0 });
                decision.sign_freq = Some(half);
            }

            if !is_pm1 {
                let n1 = (absolute as u32).saturating_sub(2) + 1;
                let nbits = 32 - n1.leading_zeros();
                let prefix_zeros = nbits - 1;
                let mut payload = 0u32;
                let mut length = prefix_zeros as u8;
                payload |= 1 << length;
                length += 1;
                for bit in (0..prefix_zeros).rev() {
                    if (n1 >> bit) & 1 != 0 {
                        payload |= 1 << length;
                    }
                    length += 1;
                }
                decision.eg_bits_payload = payload;
                decision.eg_bits_len = length;
            }
        }
        decisions.push(decision);
        context.update(value);
    }

    let mut encoder = RansEncoder::new();
    for decision in decisions.iter().rev() {
        if let Some(large_cum) = decision.is_large_cum {
            if large_cum > 0 {
                for bit in (0..decision.eg_bits_len).rev() {
                    encoder.put_bit((decision.eg_bits_payload >> bit) & 1 != 0);
                }
            }
        }
        if let (Some(cum), Some(freq)) = (decision.sign_cum, decision.sign_freq) {
            encoder.put(cum, freq);
        }
        if let (Some(cum), Some(freq)) = (decision.is_large_cum, decision.is_large_freq) {
            encoder.put(cum, freq);
        }
        encoder.put(decision.is_zero_cum, decision.is_zero_freq);
    }
    encoder.finish()
}

/// Decode a band from v12 Bayesian rANS-compressed bytes.
/// `turing_buckets` maps each coefficient position to a turing_bucket (0-3).
/// Length must match `n_coeffs`.
pub fn rans_decode_band_v12(
    data: &[u8],
    n_coeffs: usize,
    turing_buckets: &[u8],
) -> Result<(Vec<i16>, usize), RansDecodeError> {
    if n_coeffs == 0 {
        if data.is_empty() {
            return Ok((Vec::new(), 0));
        }
        return Err(RansDecodeError::TrailingData {
            consumed: 0,
            available: data.len(),
        });
    }
    if turing_buckets.len() != n_coeffs {
        return Err(RansDecodeError::InvalidContextLength {
            expected: n_coeffs,
            got: turing_buckets.len(),
        });
    }

    let mut dec = RansDecoder::new_checked(data)?;
    let mut ctx = ContextModelV12::new();
    let mut result = Vec::with_capacity(n_coeffs);

    for i in 0..n_coeffs {
        // v12: set turing_level from external bucket array before each coefficient
        ctx.set_turing_level(turing_buckets[i]);

        let p_zero = ctx.get_p_zero();

        let cum = dec.get();
        let is_zero = cum < p_zero;
        if is_zero {
            dec.advance_checked(0, p_zero)?;
            result.push(0);
            ctx.update(0);
        } else {
            dec.advance_checked(p_zero, PROB_SCALE - p_zero)?;

            let p_pm1 = ctx.get_p_pm1();
            let cum2 = dec.get();
            let is_pm1 = cum2 < p_pm1;

            if is_pm1 {
                dec.advance_checked(0, p_pm1)?;
                let negative = dec.get_bit_checked()?;
                let val = if negative { -1i16 } else { 1i16 };
                result.push(val);
                ctx.update(val);
            } else {
                dec.advance_checked(p_pm1, PROB_SCALE - p_pm1)?;
                let negative = dec.get_bit_checked()?;
                // Exp-Golomb order 0 decoding
                let mut prefix_zeros: u32 = 0;
                loop {
                    if dec.get_bit_checked()? {
                        break;
                    }
                    prefix_zeros += 1;
                    // i16 magnitudes require at most 14 leading zeroes.
                    if prefix_zeros > 14 {
                        return Err(RansDecodeError::InvalidExpGolombPrefix(prefix_zeros));
                    }
                }
                let mut n1: u32 = 1 << prefix_zeros;
                for b in (0..prefix_zeros).rev() {
                    if dec.get_bit_checked()? {
                        n1 |= 1 << b;
                    }
                }
                let mag = n1 - 1 + 2;
                if mag > i16::MAX as u32 + u32::from(negative) {
                    return Err(RansDecodeError::InvalidState);
                }
                let val = if negative {
                    -(mag as i32) as i16
                } else {
                    mag as i16
                };
                result.push(val);
                ctx.update(val);
            }
        }
    }

    if dec.state != RANS_BYTE_L {
        return Err(RansDecodeError::InvalidState);
    }
    if dec.pos() != data.len() {
        return Err(RansDecodeError::TrailingData {
            consumed: dec.pos(),
            available: data.len(),
        });
    }
    Ok((result, dec.pos()))
}

/// Source of the dyadic-parent support bit while decoding a v16 AC layer.
/// A parent either belongs to a prior progressive layer (`Known`) or precedes
/// the child in the current flat rANS sequence (`Decoded`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParentSupportRef {
    Known(bool),
    Decoded(usize),
}

/// Causal source of a v20 phase prediction. A spatial predecessor may be in
/// an earlier progressive packet (`Known`) or earlier in the current flat
/// rANS sequence (`Decoded`). `invert` carries the DCT continuation phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhasePredictorRef {
    None,
    Known(bool),
    Decoded { index: usize, invert: bool },
}

/// Decode the experimental/production v16 parent-context AC grammar.
pub fn rans_decode_band_v12_parent(
    data: &[u8],
    n_coeffs: usize,
    turing_buckets: &[u8],
    parent_refs: &[ParentSupportRef],
) -> Result<(Vec<i16>, usize), RansDecodeError> {
    if n_coeffs == 0 {
        if data.is_empty() {
            return Ok((Vec::new(), 0));
        }
        return Err(RansDecodeError::TrailingData {
            consumed: 0,
            available: data.len(),
        });
    }
    if turing_buckets.len() != n_coeffs || parent_refs.len() != n_coeffs {
        return Err(RansDecodeError::InvalidContextLength {
            expected: n_coeffs,
            got: turing_buckets.len().min(parent_refs.len()),
        });
    }

    let mut decoder = RansDecoder::new_checked(data)?;
    let mut context = ContextModelWavelet::new();
    let mut result = Vec::<i16>::with_capacity(n_coeffs);

    for index in 0..n_coeffs {
        let parent_nonzero = match parent_refs[index] {
            ParentSupportRef::Known(value) => value,
            ParentSupportRef::Decoded(parent_index) => {
                let Some(&parent) = result.get(parent_index) else {
                    return Err(RansDecodeError::InvalidState);
                };
                parent != 0
            }
        };
        context.set_external_context(turing_buckets[index], parent_nonzero);

        let p_zero = context.get_p_zero();
        let is_zero = decoder.get() < p_zero;
        if is_zero {
            decoder.advance_checked(0, p_zero)?;
            result.push(0);
            context.update(0);
            continue;
        }
        decoder.advance_checked(p_zero, PROB_SCALE - p_zero)?;

        let p_pm1 = context.get_p_pm1();
        let is_pm1 = decoder.get() < p_pm1;
        let value = if is_pm1 {
            decoder.advance_checked(0, p_pm1)?;
            if decoder.get_bit_checked()? {
                -1i16
            } else {
                1i16
            }
        } else {
            decoder.advance_checked(p_pm1, PROB_SCALE - p_pm1)?;
            let negative = decoder.get_bit_checked()?;
            let mut prefix_zeros = 0u32;
            loop {
                if decoder.get_bit_checked()? {
                    break;
                }
                prefix_zeros += 1;
                if prefix_zeros > 14 {
                    return Err(RansDecodeError::InvalidExpGolombPrefix(prefix_zeros));
                }
            }
            let mut n1 = 1u32 << prefix_zeros;
            for bit in (0..prefix_zeros).rev() {
                if decoder.get_bit_checked()? {
                    n1 |= 1 << bit;
                }
            }
            let magnitude = n1 - 1 + 2;
            if magnitude > i16::MAX as u32 + u32::from(negative) {
                return Err(RansDecodeError::InvalidState);
            }
            if negative {
                -(magnitude as i32) as i16
            } else {
                magnitude as i16
            }
        };
        result.push(value);
        context.update(value);
    }

    if decoder.state != RANS_BYTE_L {
        return Err(RansDecodeError::InvalidState);
    }
    if decoder.pos() != data.len() {
        return Err(RansDecodeError::TrailingData {
            consumed: decoder.pos(),
            available: data.len(),
        });
    }
    Ok((result, decoder.pos()))
}

/// Decode the v17 causal color codon. The coefficient grammar is unchanged;
/// only P(zero) and P(|x|=1) are split by the already decoded color class.
pub fn rans_decode_band_v17_codon(
    data: &[u8],
    n_coeffs: usize,
    turing_buckets: &[u8],
    parent_refs: &[ParentSupportRef],
    codon_classes: &[u8],
    split_pm1: bool,
) -> Result<(Vec<i16>, usize), RansDecodeError> {
    if n_coeffs == 0 {
        if data.is_empty() {
            return Ok((Vec::new(), 0));
        }
        return Err(RansDecodeError::TrailingData {
            consumed: 0,
            available: data.len(),
        });
    }
    if turing_buckets.len() != n_coeffs
        || parent_refs.len() != n_coeffs
        || codon_classes.len() != n_coeffs
    {
        return Err(RansDecodeError::InvalidContextLength {
            expected: n_coeffs,
            got: turing_buckets
                .len()
                .min(parent_refs.len())
                .min(codon_classes.len()),
        });
    }

    let mut decoder = RansDecoder::new_checked(data)?;
    let mut context = ContextModelCodon::new(split_pm1);
    let mut result = Vec::<i16>::with_capacity(n_coeffs);

    for index in 0..n_coeffs {
        let parent_nonzero = match parent_refs[index] {
            ParentSupportRef::Known(value) => value,
            ParentSupportRef::Decoded(parent_index) => {
                let Some(&parent) = result.get(parent_index) else {
                    return Err(RansDecodeError::InvalidState);
                };
                parent != 0
            }
        };
        context.set_external_context(turing_buckets[index], parent_nonzero, codon_classes[index]);

        let p_zero = context.get_p_zero();
        let is_zero = decoder.get() < p_zero;
        if is_zero {
            decoder.advance_checked(0, p_zero)?;
            result.push(0);
            context.update(0);
            continue;
        }
        decoder.advance_checked(p_zero, PROB_SCALE - p_zero)?;

        let p_pm1 = context.get_p_pm1();
        let is_pm1 = decoder.get() < p_pm1;
        let value = if is_pm1 {
            decoder.advance_checked(0, p_pm1)?;
            if decoder.get_bit_checked()? {
                -1i16
            } else {
                1i16
            }
        } else {
            decoder.advance_checked(p_pm1, PROB_SCALE - p_pm1)?;
            let negative = decoder.get_bit_checked()?;
            let mut prefix_zeros = 0u32;
            loop {
                if decoder.get_bit_checked()? {
                    break;
                }
                prefix_zeros += 1;
                if prefix_zeros > 14 {
                    return Err(RansDecodeError::InvalidExpGolombPrefix(prefix_zeros));
                }
            }
            let mut n1 = 1u32 << prefix_zeros;
            for bit in (0..prefix_zeros).rev() {
                if decoder.get_bit_checked()? {
                    n1 |= 1 << bit;
                }
            }
            let magnitude = n1 - 1 + 2;
            if magnitude > i16::MAX as u32 + u32::from(negative) {
                return Err(RansDecodeError::InvalidState);
            }
            if negative {
                -(magnitude as i32) as i16
            } else {
                magnitude as i16
            }
        };
        result.push(value);
        context.update(value);
    }

    if decoder.state != RANS_BYTE_L {
        return Err(RansDecodeError::InvalidState);
    }
    if decoder.pos() != data.len() {
        return Err(RansDecodeError::TrailingData {
            consumed: decoder.pos(),
            available: data.len(),
        });
    }
    Ok((result, decoder.pos()))
}

/// Decode the v19 chroma phase grammar. C2 phase is represented as an
/// adaptive agreement/opposition residual against the already decoded C1
/// coefficient at the same spectral coordinate.
pub fn rans_decode_band_v19_phase(
    data: &[u8],
    n_coeffs: usize,
    turing_buckets: &[u8],
    parent_refs: &[ParentSupportRef],
    codon_classes: &[u8],
    phase_predictors: &[i8],
) -> Result<(Vec<i16>, usize), RansDecodeError> {
    if n_coeffs == 0 {
        if data.is_empty() {
            return Ok((Vec::new(), 0));
        }
        return Err(RansDecodeError::TrailingData {
            consumed: 0,
            available: data.len(),
        });
    }
    if turing_buckets.len() != n_coeffs
        || parent_refs.len() != n_coeffs
        || codon_classes.len() != n_coeffs
        || phase_predictors.len() != n_coeffs
    {
        return Err(RansDecodeError::InvalidContextLength {
            expected: n_coeffs,
            got: turing_buckets
                .len()
                .min(parent_refs.len())
                .min(codon_classes.len())
                .min(phase_predictors.len()),
        });
    }

    let mut decoder = RansDecoder::new_checked(data)?;
    let mut context = ContextModelCodon::new(true);
    let mut p_same = PHASE_INITIAL_P_SAME;
    let mut result = Vec::<i16>::with_capacity(n_coeffs);

    for index in 0..n_coeffs {
        let parent_nonzero = match parent_refs[index] {
            ParentSupportRef::Known(value) => value,
            ParentSupportRef::Decoded(parent_index) => {
                let Some(&parent) = result.get(parent_index) else {
                    return Err(RansDecodeError::InvalidState);
                };
                parent != 0
            }
        };
        context.set_external_context(turing_buckets[index], parent_nonzero, codon_classes[index]);

        let p_zero = context.get_p_zero();
        let is_zero = decoder.get() < p_zero;
        if is_zero {
            decoder.advance_checked(0, p_zero)?;
            result.push(0);
            context.update(0);
            continue;
        }
        decoder.advance_checked(p_zero, PROB_SCALE - p_zero)?;

        let p_pm1 = context.get_p_pm1();
        let is_pm1 = decoder.get() < p_pm1;
        if is_pm1 {
            decoder.advance_checked(0, p_pm1)?;
        } else {
            decoder.advance_checked(p_pm1, PROB_SCALE - p_pm1)?;
        }

        let predictor = phase_predictors[index];
        let negative = if predictor >= 0 {
            let same = decoder.get() < p_same;
            if same {
                decoder.advance_checked(0, p_same)?;
                p_same += (PROB_SCALE - p_same) >> PHASE_ADAPT_SHIFT;
            } else {
                decoder.advance_checked(p_same, PROB_SCALE - p_same)?;
                p_same -= p_same >> PHASE_ADAPT_SHIFT;
            }
            if same { predictor != 0 } else { predictor == 0 }
        } else {
            decoder.get_bit_checked()?
        };

        let magnitude = if is_pm1 {
            1u32
        } else {
            let mut prefix_zeros = 0u32;
            loop {
                if decoder.get_bit_checked()? {
                    break;
                }
                prefix_zeros += 1;
                if prefix_zeros > 14 {
                    return Err(RansDecodeError::InvalidExpGolombPrefix(prefix_zeros));
                }
            }
            let mut n1 = 1u32 << prefix_zeros;
            for bit in (0..prefix_zeros).rev() {
                if decoder.get_bit_checked()? {
                    n1 |= 1 << bit;
                }
            }
            n1 - 1 + 2
        };
        if magnitude > i16::MAX as u32 + u32::from(negative) {
            return Err(RansDecodeError::InvalidState);
        }
        let value = if negative {
            -(magnitude as i32) as i16
        } else {
            magnitude as i16
        };
        result.push(value);
        context.update(value);
    }

    if decoder.state != RANS_BYTE_L {
        return Err(RansDecodeError::InvalidState);
    }
    if decoder.pos() != data.len() {
        return Err(RansDecodeError::TrailingData {
            consumed: decoder.pos(),
            available: data.len(),
        });
    }
    Ok((result, decoder.pos()))
}

/// Decode the v20 phase grammar with a causal reference that may point into
/// the current rANS sequence. The coefficient grammar is unchanged; only the
/// source and prior of the sign prediction differ by channel.
pub fn rans_decode_band_v20_phase(
    data: &[u8],
    n_coeffs: usize,
    turing_buckets: &[u8],
    parent_refs: &[ParentSupportRef],
    codon_classes: &[u8],
    phase_refs: &[PhasePredictorRef],
    initial_p_same: u32,
    phase_adapt_shift: u32,
) -> Result<(Vec<i16>, usize), RansDecodeError> {
    if n_coeffs == 0 {
        if data.is_empty() {
            return Ok((Vec::new(), 0));
        }
        return Err(RansDecodeError::TrailingData {
            consumed: 0,
            available: data.len(),
        });
    }
    if turing_buckets.len() != n_coeffs
        || parent_refs.len() != n_coeffs
        || codon_classes.len() != n_coeffs
        || phase_refs.len() != n_coeffs
    {
        return Err(RansDecodeError::InvalidContextLength {
            expected: n_coeffs,
            got: turing_buckets
                .len()
                .min(parent_refs.len())
                .min(codon_classes.len())
                .min(phase_refs.len()),
        });
    }

    let mut decoder = RansDecoder::new_checked(data)?;
    let mut context = ContextModelCodon::new(true);
    let mut p_same = initial_p_same.clamp(1, PROB_SCALE - 1);
    let adapt_shift = phase_adapt_shift.clamp(1, 15);
    let mut result = Vec::<i16>::with_capacity(n_coeffs);

    for index in 0..n_coeffs {
        let parent_nonzero = match parent_refs[index] {
            ParentSupportRef::Known(value) => value,
            ParentSupportRef::Decoded(parent_index) => {
                let Some(&parent) = result.get(parent_index) else {
                    return Err(RansDecodeError::InvalidState);
                };
                parent != 0
            }
        };
        context.set_external_context(turing_buckets[index], parent_nonzero, codon_classes[index]);

        let p_zero = context.get_p_zero();
        let is_zero = decoder.get() < p_zero;
        if is_zero {
            decoder.advance_checked(0, p_zero)?;
            result.push(0);
            context.update(0);
            continue;
        }
        decoder.advance_checked(p_zero, PROB_SCALE - p_zero)?;

        let p_pm1 = context.get_p_pm1();
        let is_pm1 = decoder.get() < p_pm1;
        if is_pm1 {
            decoder.advance_checked(0, p_pm1)?;
        } else {
            decoder.advance_checked(p_pm1, PROB_SCALE - p_pm1)?;
        }

        let predictor = match phase_refs[index] {
            PhasePredictorRef::None => None,
            PhasePredictorRef::Known(negative) => Some(negative),
            PhasePredictorRef::Decoded {
                index: predictor_index,
                invert,
            } => {
                let Some(&value) = result.get(predictor_index) else {
                    return Err(RansDecodeError::InvalidState);
                };
                (value != 0).then_some((value < 0) ^ invert)
            }
        };
        let negative = if let Some(predicted_negative) = predictor {
            let same = decoder.get() < p_same;
            if same {
                decoder.advance_checked(0, p_same)?;
                p_same += (PROB_SCALE - p_same) >> adapt_shift;
            } else {
                decoder.advance_checked(p_same, PROB_SCALE - p_same)?;
                p_same -= p_same >> adapt_shift;
            }
            if same {
                predicted_negative
            } else {
                !predicted_negative
            }
        } else {
            decoder.get_bit_checked()?
        };

        let magnitude = if is_pm1 {
            1u32
        } else {
            let mut prefix_zeros = 0u32;
            loop {
                if decoder.get_bit_checked()? {
                    break;
                }
                prefix_zeros += 1;
                if prefix_zeros > 14 {
                    return Err(RansDecodeError::InvalidExpGolombPrefix(prefix_zeros));
                }
            }
            let mut n1 = 1u32 << prefix_zeros;
            for bit in (0..prefix_zeros).rev() {
                if decoder.get_bit_checked()? {
                    n1 |= 1 << bit;
                }
            }
            n1 - 1 + 2
        };
        if magnitude > i16::MAX as u32 + u32::from(negative) {
            return Err(RansDecodeError::InvalidState);
        }
        let value = if negative {
            -(magnitude as i32) as i16
        } else {
            magnitude as i16
        };
        result.push(value);
        context.update(value);
    }

    if decoder.state != RANS_BYTE_L {
        return Err(RansDecodeError::InvalidState);
    }
    if decoder.pos() != data.len() {
        return Err(RansDecodeError::TrailingData {
            consumed: decoder.pos(),
            available: data.len(),
        });
    }
    Ok((result, decoder.pos()))
}

// ======================================================================
// RLE Pre-Transform (separate layer before rANS)
// ======================================================================

/// Convert AC coefficients into RLE pairs: [run0, level0, run1, level1, ...]
/// run = number of preceding zeros (as i16, always >= 0)
/// level = non-zero coefficient value
/// Returns: (rle_symbols, n_pairs)
pub fn ac_to_rle(coeffs: &[i16]) -> (Vec<i16>, usize) {
    let mut symbols = Vec::new();
    let mut run: i16 = 0;
    let mut n_pairs = 0usize;
    for &val in coeffs {
        if val == 0 {
            run += 1;
        } else {
            symbols.push(run); // run length
            symbols.push(val); // non-zero value
            run = 0;
            n_pairs += 1;
        }
    }
    (symbols, n_pairs)
}

/// Expand RLE symbols back into AC coefficients.
/// `rle_symbols`: [run0, level0, run1, level1, ...]
/// `n_coeffs`: total number of output coefficients (with trailing zeros)
pub fn rle_to_ac(rle_symbols: &[i16], n_coeffs: usize) -> Vec<i16> {
    let mut result = Vec::with_capacity(n_coeffs);
    let mut i = 0;
    while i + 1 < rle_symbols.len() {
        let run = rle_symbols[i].max(0) as usize;
        let level = rle_symbols[i + 1];
        for _ in 0..run {
            result.push(0);
        }
        result.push(level);
        i += 2;
    }
    // Pad with trailing zeros
    while result.len() < n_coeffs {
        result.push(0);
    }
    result.truncate(n_coeffs);
    result
}

// ======================================================================
// Paeth Residual Encoding / Decoding
// ======================================================================

/// Encode Paeth residuals using rANS with a Laplacian-style adaptive model.
///
/// Encoding scheme per residual:
/// 1. is_zero (adaptive P(zero))
/// 2. If nonzero: sign bit (raw 1/2)
/// 3. Magnitude - 1 in Exp-Golomb-like coding:
///    - magnitude_class in 2 bits (0: mag=1, 1: mag 2-3, 2: mag 4-7, 3: mag 8+)
///    - extra bits within the class
///    - For class 3 (mag>=8): encode (mag-8) in unary (ones + zero terminator)
pub fn rans_encode_paeth(residuals: &[i16]) -> Vec<u8> {
    if residuals.is_empty() {
        return Vec::new();
    }

    let n = residuals.len();
    let mut enc = RansEncoder::new();
    let mut ctx = PaethContextModel::new();

    // Forward pass to collect decisions
    struct PaethDecision {
        is_zero_cum: u32,
        is_zero_freq: u32,
        sign: Option<bool>,
        mag_bits: Vec<bool>, // magnitude bits to encode
    }

    let mut decisions = Vec::with_capacity(n);

    for &val in residuals.iter() {
        let is_zero = val == 0;
        let p_zero = ctx.get_p_zero();

        let (iz_cum, iz_freq) = if is_zero {
            (0, p_zero)
        } else {
            (p_zero, PROB_SCALE - p_zero)
        };

        let mut dec = PaethDecision {
            is_zero_cum: iz_cum,
            is_zero_freq: iz_freq,
            sign: None,
            mag_bits: Vec::new(),
        };

        if !is_zero {
            dec.sign = Some(val < 0);
            let abs_val = val.unsigned_abs();

            // Encode magnitude using Exp-Golomb unary:
            // mag=1: just one zero bit (terminator)
            // mag=2: one 1-bit, then zero bit
            // mag=k: (k-1) 1-bits, then zero bit
            // This is unary coding of (mag-1)
            for _ in 0..(abs_val - 1) {
                dec.mag_bits.push(true);
            }
            dec.mag_bits.push(false); // terminator
        }

        decisions.push(dec);
        ctx.update(val);
    }

    // Encode backwards
    for dec in decisions.iter().rev() {
        // Innermost first: magnitude bits, then sign, then is_zero

        // 1. Magnitude bits (reverse order for unary)
        for &bit in dec.mag_bits.iter().rev() {
            enc.put_bit(bit);
        }

        // 2. Sign
        if let Some(sign) = dec.sign {
            enc.put_bit(sign);
        }

        // 3. Is zero
        enc.put(dec.is_zero_cum, dec.is_zero_freq);
    }

    enc.finish()
}

/// Decode Paeth residuals from rANS-compressed bytes.
/// Returns (residuals, bytes consumed).
pub fn rans_decode_paeth(data: &[u8], n: usize) -> (Vec<i16>, usize) {
    if n == 0 {
        return (Vec::new(), 0);
    }

    let mut dec = RansDecoder::new(data);
    let mut ctx = PaethContextModel::new();
    let mut result = Vec::with_capacity(n);

    for _ in 0..n {
        let p_zero = ctx.get_p_zero();

        // Decode is_zero
        let cum = dec.get();
        let is_zero = cum < p_zero;

        if is_zero {
            dec.advance(0, p_zero);
            result.push(0);
            ctx.update(0);
        } else {
            dec.advance(p_zero, PROB_SCALE - p_zero);

            // Decode sign
            let negative = dec.get_bit();

            // Decode magnitude via unary: count ones until zero
            let mut mag: u16 = 1;
            loop {
                let bit = dec.get_bit();
                if !bit {
                    break;
                }
                mag += 1;
                // Safety: cap magnitude to avoid infinite loop on corrupted data
                if mag >= 32000 {
                    break;
                }
            }

            let val = if negative { -(mag as i16) } else { mag as i16 };
            result.push(val);
            ctx.update(val);
        }
    }

    (result, dec.pos())
}

// ======================================================================
// Byte-level adaptive compression (drop-in LZMA replacement)
// ======================================================================

/// Order-0 adaptive byte model for general-purpose compression.
/// Each byte is coded with an adaptive frequency table (256 symbols).
///
/// Key invariant: the scaled CDF computed by `prob()` must be exactly
/// reversible by `decode_byte()`. We achieve this by having `decode_byte()`
/// call `prob()` to verify the exact boundaries, ensuring identical rounding.
struct AdaptiveByteModel {
    freqs: [u16; 256],
    total: u16,
    /// Cached cumulative frequencies (cum[i] = sum of freqs[0..i])
    cum: [u32; 257],
}

impl AdaptiveByteModel {
    fn new() -> Self {
        let mut m = Self {
            freqs: [1u16; 256],
            total: 256,
            cum: [0u32; 257],
        };
        m.rebuild_cum();
        m
    }

    /// Rebuild the cumulative table from freqs.
    fn rebuild_cum(&mut self) {
        self.cum[0] = 0;
        for i in 0..256 {
            self.cum[i + 1] = self.cum[i] + self.freqs[i] as u32;
        }
    }

    /// Get (cum_freq, freq) for a given byte, scaled to PROB_SCALE.
    /// The scaling uses floor division for cum, and ensures freq >= 1.
    #[inline]
    fn prob(&self, byte: u8) -> (u32, u32) {
        let total = self.total as u32;
        let cum_start = self.cum[byte as usize];
        let cum_end = self.cum[byte as usize + 1];

        let cum_scaled = cum_start * PROB_SCALE / total;
        let end_scaled = cum_end * PROB_SCALE / total;
        let freq_scaled = if byte == 255 {
            // Last symbol: freq = PROB_SCALE - cum_scaled to fill the range exactly
            PROB_SCALE - cum_scaled
        } else {
            (end_scaled - cum_scaled).max(1)
        };
        (cum_scaled, freq_scaled)
    }

    /// Update the model after observing a byte.
    fn update(&mut self, byte: u8) {
        self.freqs[byte as usize] += 1;
        self.total += 1;
        // Update cumulative table incrementally: only entries after `byte` change
        for i in (byte as usize + 1)..=256 {
            self.cum[i] += 1;
        }
        // Rescale if total gets too large (prevents overflow and keeps adaptation recent)
        if self.total > 8192 {
            self.total = 0;
            for f in &mut self.freqs {
                *f = (*f >> 1).max(1);
                self.total += *f;
            }
            self.rebuild_cum();
        }
    }

    /// Decode: find which byte corresponds to a cumulative frequency value.
    /// Uses the same scaling as prob() to guarantee exact match.
    fn decode_byte(&self, cum_val: u32) -> u8 {
        let total = self.total as u32;
        // Binary search: find the byte whose scaled range contains cum_val
        let mut lo: usize = 0;
        let mut hi: usize = 256;
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            let mid_cum_scaled = self.cum[mid] * PROB_SCALE / total;
            if mid_cum_scaled <= cum_val {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo as u8
    }
}

/// Compress arbitrary bytes using rANS with adaptive order-0 model.
/// Returns compressed data with a 4-byte LE length prefix.
pub fn rans_compress_bytes(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    // Forward pass: record probabilities at each position (model state varies)
    let mut model = AdaptiveByteModel::new();
    let mut probs: Vec<(u32, u32)> = Vec::with_capacity(input.len());
    for &byte in input {
        probs.push(model.prob(byte));
        model.update(byte);
    }

    // Encode backwards (rANS requirement)
    let mut enc = RansEncoder::new();
    for i in (0..input.len()).rev() {
        let (cum, freq) = probs[i];
        enc.put(cum, freq);
    }

    let out = enc.finish();
    // Prepend the original length as u32 LE
    let len_bytes = (input.len() as u32).to_le_bytes();
    let mut result = Vec::with_capacity(4 + out.len());
    result.extend_from_slice(&len_bytes);
    result.extend(out);
    result
}

/// Decompress bytes from rANS-compressed data (produced by rans_compress_bytes).
/// The input starts with a 4-byte LE original length, followed by rANS data.
pub fn rans_decompress_bytes(compressed: &[u8]) -> Vec<u8> {
    if compressed.len() < 4 {
        return Vec::new();
    }

    let original_len =
        u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]) as usize;

    if original_len == 0 {
        return Vec::new();
    }

    let mut dec = RansDecoder::new(&compressed[4..]);
    let mut model = AdaptiveByteModel::new();
    let mut output = Vec::with_capacity(original_len);

    for _ in 0..original_len {
        let cum_val = dec.get();
        let byte = model.decode_byte(cum_val);
        let (cum, freq) = model.prob(byte);
        dec.advance(cum, freq);
        model.update(byte);
        output.push(byte);
    }

    output
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rans_roundtrip_basic() {
        // Simple sequence: mix of small values
        let input: Vec<i16> = vec![0, 1, -1, 0, 0, 2, -3, 0, 0, 0, 1, 0, -1, 5, 0];
        let encoded = rans_encode_band(&input);
        let (decoded, _bytes) = rans_decode_band(&encoded, input.len());
        assert_eq!(input, decoded, "basic roundtrip failed");
    }

    #[test]
    fn test_rans_v16_parent_context_roundtrip() {
        let input = vec![1i16, 0, -1, 2, 0, 0, 3, -2, 0, 1];
        let buckets = vec![0u8, 0, 1, 1, 2, 2, 3, 3, 0, 1];
        let parent_refs = vec![
            ParentSupportRef::Known(false),
            ParentSupportRef::Decoded(0),
            ParentSupportRef::Decoded(0),
            ParentSupportRef::Decoded(2),
            ParentSupportRef::Decoded(1),
            ParentSupportRef::Known(true),
            ParentSupportRef::Decoded(3),
            ParentSupportRef::Decoded(6),
            ParentSupportRef::Decoded(5),
            ParentSupportRef::Decoded(7),
        ];
        let parents: Vec<bool> = parent_refs
            .iter()
            .map(|reference| match *reference {
                ParentSupportRef::Known(value) => value,
                ParentSupportRef::Decoded(index) => input[index] != 0,
            })
            .collect();
        let encoded = rans_encode_band_v12_parent(&input, &buckets, &parents);
        let (decoded, consumed) =
            rans_decode_band_v12_parent(&encoded, input.len(), &buckets, &parent_refs).unwrap();
        assert_eq!(decoded, input);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn test_rans_v17_vector_codon_roundtrip() {
        let input = vec![1i16, 0, -1, 2, 0, 0, 3, -2, 0, 1, 7, -12];
        let buckets = vec![0u8, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3];
        let codon_classes = vec![0u8, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1, 3];
        let parent_refs = vec![
            ParentSupportRef::Known(false),
            ParentSupportRef::Decoded(0),
            ParentSupportRef::Decoded(0),
            ParentSupportRef::Decoded(2),
            ParentSupportRef::Decoded(1),
            ParentSupportRef::Known(true),
            ParentSupportRef::Decoded(3),
            ParentSupportRef::Decoded(6),
            ParentSupportRef::Decoded(5),
            ParentSupportRef::Decoded(7),
            ParentSupportRef::Decoded(8),
            ParentSupportRef::Decoded(10),
        ];
        let parents: Vec<bool> = parent_refs
            .iter()
            .map(|reference| match *reference {
                ParentSupportRef::Known(value) => value,
                ParentSupportRef::Decoded(index) => input[index] != 0,
            })
            .collect();

        for split_pm1 in [false, true] {
            let encoded =
                rans_encode_band_v17_codon(&input, &buckets, &parents, &codon_classes, split_pm1);
            let (decoded, consumed) = rans_decode_band_v17_codon(
                &encoded,
                input.len(),
                &buckets,
                &parent_refs,
                &codon_classes,
                split_pm1,
            )
            .unwrap();
            assert_eq!(decoded, input);
            assert_eq!(consumed, encoded.len());
        }
    }

    #[test]
    fn test_rans_v19_phase_codon_roundtrip() {
        let input = vec![1i16, 0, -1, 2, 0, -1, 3, -2, 0, 1, 7, -12, -1, 4];
        let buckets = vec![0u8, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 1, 0];
        let codon_classes = vec![0u8, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1, 3, 2, 1];
        let phase_predictors = vec![-1i8, -1, 0, 1, -1, 0, 0, 1, -1, 1, 1, 0, 0, 1];
        let parent_refs = vec![
            ParentSupportRef::Known(false),
            ParentSupportRef::Decoded(0),
            ParentSupportRef::Decoded(0),
            ParentSupportRef::Decoded(2),
            ParentSupportRef::Decoded(1),
            ParentSupportRef::Known(true),
            ParentSupportRef::Decoded(3),
            ParentSupportRef::Decoded(6),
            ParentSupportRef::Decoded(5),
            ParentSupportRef::Decoded(7),
            ParentSupportRef::Decoded(8),
            ParentSupportRef::Decoded(10),
            ParentSupportRef::Decoded(11),
            ParentSupportRef::Decoded(12),
        ];
        let parents: Vec<bool> = parent_refs
            .iter()
            .map(|reference| match *reference {
                ParentSupportRef::Known(value) => value,
                ParentSupportRef::Decoded(index) => input[index] != 0,
            })
            .collect();
        let encoded = rans_encode_band_v19_phase(
            &input,
            &buckets,
            &parents,
            &codon_classes,
            &phase_predictors,
            PHASE_INITIAL_P_SAME,
            PHASE_ADAPT_SHIFT,
        );
        let (decoded, consumed) = rans_decode_band_v19_phase(
            &encoded,
            input.len(),
            &buckets,
            &parent_refs,
            &codon_classes,
            &phase_predictors,
        )
        .unwrap();
        assert_eq!(decoded, input);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn test_rans_v20_fourier_phase_roundtrip() {
        let input = vec![1i16, 0, -1, 2, 0, -1, 3, -2, 0, 1, 7, -12, -1, 4];
        let buckets = vec![0u8, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 1, 0];
        let codon_classes = vec![0u8, 1, 2, 3, 0, 2, 1, 3, 2, 0, 1, 3, 2, 1];
        let phase_refs = vec![
            PhasePredictorRef::None,
            PhasePredictorRef::None,
            PhasePredictorRef::Decoded {
                index: 0,
                invert: false,
            },
            PhasePredictorRef::Decoded {
                index: 2,
                invert: true,
            },
            PhasePredictorRef::Decoded {
                index: 3,
                invert: false,
            },
            PhasePredictorRef::Known(true),
            PhasePredictorRef::Decoded {
                index: 3,
                invert: false,
            },
            PhasePredictorRef::Decoded {
                index: 6,
                invert: true,
            },
            PhasePredictorRef::Decoded {
                index: 5,
                invert: false,
            },
            PhasePredictorRef::Decoded {
                index: 7,
                invert: true,
            },
            PhasePredictorRef::Decoded {
                index: 9,
                invert: false,
            },
            PhasePredictorRef::Decoded {
                index: 10,
                invert: true,
            },
            PhasePredictorRef::Known(true),
            PhasePredictorRef::Decoded {
                index: 12,
                invert: true,
            },
        ];
        let phase_predictors: Vec<i8> = phase_refs
            .iter()
            .map(|reference| match *reference {
                PhasePredictorRef::None => -1,
                PhasePredictorRef::Known(negative) => i8::from(negative),
                PhasePredictorRef::Decoded { index, invert } => {
                    let value = input[index];
                    if value == 0 {
                        -1
                    } else {
                        i8::from((value < 0) ^ invert)
                    }
                }
            })
            .collect();
        let parent_refs = vec![
            ParentSupportRef::Known(false),
            ParentSupportRef::Decoded(0),
            ParentSupportRef::Decoded(0),
            ParentSupportRef::Decoded(2),
            ParentSupportRef::Decoded(1),
            ParentSupportRef::Known(true),
            ParentSupportRef::Decoded(3),
            ParentSupportRef::Decoded(6),
            ParentSupportRef::Decoded(5),
            ParentSupportRef::Decoded(7),
            ParentSupportRef::Decoded(8),
            ParentSupportRef::Decoded(10),
            ParentSupportRef::Decoded(11),
            ParentSupportRef::Decoded(12),
        ];
        let parents: Vec<bool> = parent_refs
            .iter()
            .map(|reference| match *reference {
                ParentSupportRef::Known(value) => value,
                ParentSupportRef::Decoded(index) => input[index] != 0,
            })
            .collect();
        let encoded = rans_encode_band_v19_phase(
            &input,
            &buckets,
            &parents,
            &codon_classes,
            &phase_predictors,
            FOURIER_PHASE_INITIAL_P_SAME,
            FOURIER_PHASE_ADAPT_SHIFT,
        );
        let (decoded, consumed) = rans_decode_band_v20_phase(
            &encoded,
            input.len(),
            &buckets,
            &parent_refs,
            &codon_classes,
            &phase_refs,
            FOURIER_PHASE_INITIAL_P_SAME,
            FOURIER_PHASE_ADAPT_SHIFT,
        )
        .unwrap();
        assert_eq!(decoded, input);
        assert_eq!(consumed, encoded.len());
    }

    // ==================================================================
    // Cycle 20 — spectral envelope
    // ==================================================================

    fn envelope_fixture() -> (
        Vec<i16>,
        Vec<u8>,
        Vec<u8>,
        Vec<i8>,
        Vec<SpectralSegment>,
        SpectralNeighborTables,
    ) {
        use crate::spectral_envelope::{MAX_PREFIX, SpectralNeighborTables as Tables};
        let zigzag = |n: usize| -> Vec<usize> {
            let mut order = Vec::with_capacity(n * n - 1);
            for s in 0..(2 * n - 1) {
                if s % 2 == 0 {
                    let i_start = s.min(n - 1);
                    let i_end = if s >= n { s - n + 1 } else { 0 };
                    let mut i = i_start as i64;
                    while i >= i_end as i64 {
                        let j = s - i as usize;
                        let idx = i as usize * n + j;
                        if idx != 0 {
                            order.push(idx);
                        }
                        i -= 1;
                    }
                } else {
                    let i_start = if s >= n { s - n + 1 } else { 0 };
                    let i_end = s.min(n - 1);
                    for i in i_start..=i_end {
                        let j = s - i;
                        let idx = i * n + j;
                        if idx != 0 {
                            order.push(idx);
                        }
                    }
                }
            }
            order
        };
        let tables = Tables::new(&zigzag(8), &zigzag(16), &zigzag(32));

        // Four blocks of a detail layer: sparse tails with realistic runs.
        let mut state = 0xC0FF_EE01u32;
        let mut next = |modulus: i32| -> i32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 13) % modulus as u32) as i32
        };
        let mut coeffs = Vec::new();
        let mut segments = Vec::new();
        for block in 0..4usize {
            let length = 40 + block * 17;
            let prefix: Vec<i16> = (0..MAX_PREFIX).map(|_| (next(9) - 4) as i16).collect();
            segments.push(SpectralSegment::detail(
                coeffs.len(),
                MAX_PREFIX,
                length,
                1,
                &prefix,
            ));
            for _ in 0..length {
                let value = if next(10) < 6 { 0 } else { next(11) - 5 };
                coeffs.push(value as i16);
            }
        }
        let turing: Vec<u8> = (0..coeffs.len()).map(|i| (i % 4) as u8).collect();
        let codon: Vec<u8> = (0..coeffs.len()).map(|i| ((i * 7) % 4) as u8).collect();
        let legacy: Vec<i8> = (0..coeffs.len())
            .map(|i| match i % 5 {
                0 => -1,
                1 => 0,
                2 => 1,
                3 => -1,
                _ => 0,
            })
            .collect();
        (coeffs, turing, codon, legacy, segments, tables)
    }

    /// The neutral configuration must reproduce the promoted v20 grammar byte
    /// for byte: the envelope collapses onto the dyadic parent support and no
    /// spectral phase class is ever selected.
    #[test]
    fn neutral_spectral_envelope_matches_v20_bytes() {
        use crate::spectral_envelope::{SpectralCursor, SpectralEnvelopeConfig};
        let (coeffs, turing, codon, legacy, segments, tables) = envelope_fixture();
        let config = SpectralEnvelopeConfig::neutral();

        // The v20 witness needs the same parent support bits, read through the
        // very same neighbourhood table.
        let mut cursor = SpectralCursor::new(&segments, &tables, config, 0);
        let parents: Vec<bool> = (0..coeffs.len())
            .map(|index| cursor.descriptor(index, &coeffs).parent_nonzero)
            .collect();
        let reference = rans_encode_band_v19_phase(
            &coeffs,
            &turing,
            &parents,
            &codon,
            &legacy,
            PHASE_INITIAL_P_SAME,
            PHASE_ADAPT_SHIFT,
        );

        for channel in 0..3usize {
            let envelope = rans_encode_band_v21_envelope(
                &coeffs,
                &turing,
                &codon,
                &legacy,
                &[],
                &segments,
                &tables,
                &config,
                channel,
                PHASE_INITIAL_P_SAME,
                PHASE_ADAPT_SHIFT,
                false, // v20 parity: no block-entry priming
                &[],
                &mut EnvelopeTables::default(),
            );
            assert_eq!(reference, envelope, "channel={channel}");
        }
    }

    #[test]
    fn spectral_envelope_roundtrips_for_every_variant() {
        use crate::spectral_envelope::{
            AMPLITUDE_MAGNITUDE, AMPLITUDE_MAGNITUDE_SCALED, AMPLITUDE_PARENT_LOCAL,
            AMPLITUDE_SUPPORT, PHASE_MASK_ALL, PHASE_MASK_COHERENT, PHASE_MASK_SEPARABLE,
            SpectralEnvelopeConfig,
        };
        let (coeffs, turing, codon, legacy, segments, tables) = envelope_fixture();
        // Cycle 22: a plausible foreign witness so the mixer is exercised.
        let foreign: Vec<u8> = (0..coeffs.len()).map(|i| ((i * 5) % 4) as u8).collect();
        let legacy_refs: Vec<PhasePredictorRef> = legacy
            .iter()
            .map(|&predictor| {
                if predictor < 0 {
                    PhasePredictorRef::None
                } else {
                    PhasePredictorRef::Known(predictor != 0)
                }
            })
            .collect();

        for amplitude in [
            0,
            AMPLITUDE_PARENT_LOCAL,
            AMPLITUDE_MAGNITUDE,
            AMPLITUDE_SUPPORT,
            AMPLITUDE_MAGNITUDE_SCALED,
        ] {
            for mask in [
                0u16,
                PHASE_MASK_SEPARABLE,
                PHASE_MASK_COHERENT,
                PHASE_MASK_ALL,
            ] {
                for context in [0u8, 1, 2, 3, 4, 5] {
                    for mix_rate in [0u8, 3] {
                        let splits_pm1 = context % 2 == 0;
                        let mut config = SpectralEnvelopeConfig::neutral();
                        config.amplitude_mode = amplitude;
                        config.amplitude_splits_pm1 = splits_pm1;
                        config.phase_mask = mask;
                        config.phase_channels = 0b111;
                        config.phase_context = context;
                        config.mix_rate = mix_rate;
                        for channel in 0..3usize {
                            let encoded = rans_encode_band_v21_envelope(
                                &coeffs,
                                &turing,
                                &codon,
                                &legacy,
                                &foreign,
                                &segments,
                                &tables,
                                &config,
                                channel,
                                PHASE_INITIAL_P_SAME,
                                PHASE_ADAPT_SHIFT,
                                true,
                                &[],
                                &mut EnvelopeTables::default(),
                            );
                            let (decoded, consumed) = rans_decode_band_v21_envelope(
                                &encoded,
                                coeffs.len(),
                                &turing,
                                &codon,
                                &legacy_refs,
                                &foreign,
                                &segments,
                                &tables,
                                &config,
                                channel,
                                PHASE_INITIAL_P_SAME,
                                PHASE_ADAPT_SHIFT,
                                true,
                                &[],
                                &mut EnvelopeTables::default(),
                            )
                            .unwrap();
                            assert_eq!(
                                decoded, coeffs,
                                "amp={amplitude} mask={mask} ctx={context} mix={mix_rate} ch={channel}"
                            );
                            assert_eq!(consumed, encoded.len());
                        }
                    }
                }
            }
        }
    }

    /// Cycle 23 — every mixer shape must round-trip. The mixer, the secondary
    /// models and the APM all carry adaptive state that the decoder rebuilds
    /// step by step; any asymmetry desynchronizes on the first symbol.
    #[test]
    fn mixed_grammar_roundtrips_for_every_input_set() {
        use crate::spectral_envelope::{
            MIX_INPUT_BAND, MIX_INPUT_FOREIGN, MIX_INPUT_SCALED, SpectralEnvelopeConfig,
        };
        let (coeffs, turing, codon, legacy, segments, tables) = envelope_fixture();
        let foreign: Vec<u8> = (0..coeffs.len()).map(|i| ((i * 5) % 4) as u8).collect();
        let legacy_refs: Vec<PhasePredictorRef> = legacy
            .iter()
            .map(|&predictor| {
                if predictor < 0 {
                    PhasePredictorRef::None
                } else {
                    PhasePredictorRef::Known(predictor != 0)
                }
            })
            .collect();

        for inputs in [
            MIX_INPUT_FOREIGN,
            MIX_INPUT_SCALED,
            MIX_INPUT_BAND,
            MIX_INPUT_FOREIGN | MIX_INPUT_SCALED,
            MIX_INPUT_FOREIGN | MIX_INPUT_BAND,
            MIX_INPUT_FOREIGN | MIX_INPUT_SCALED | MIX_INPUT_BAND,
        ] {
            for mix_pm1 in [false, true] {
                for apm_rate in [0u8, 6] {
                    for rate in [1u8, 3] {
                        let mut config = SpectralEnvelopeConfig::promoted();
                        config.mix_rate = rate;
                        config.mix_inputs = inputs;
                        config.mix_pm1 = mix_pm1;
                        config.apm_rate = apm_rate;
                        for channel in 0..3usize {
                            let encoded = rans_encode_band_v21_envelope(
                                &coeffs,
                                &turing,
                                &codon,
                                &legacy,
                                &foreign,
                                &segments,
                                &tables,
                                &config,
                                channel,
                                PHASE_INITIAL_P_SAME,
                                PHASE_ADAPT_SHIFT,
                                true,
                                &[],
                                &mut EnvelopeTables::default(),
                            );
                            let (decoded, consumed) = rans_decode_band_v21_envelope(
                                &encoded,
                                coeffs.len(),
                                &turing,
                                &codon,
                                &legacy_refs,
                                &foreign,
                                &segments,
                                &tables,
                                &config,
                                channel,
                                PHASE_INITIAL_P_SAME,
                                PHASE_ADAPT_SHIFT,
                                true,
                                &[],
                                &mut EnvelopeTables::default(),
                            )
                            .unwrap();
                            assert_eq!(
                                decoded, coeffs,
                                "inputs={inputs} pm1={mix_pm1} apm={apm_rate} r={rate} ch={channel}"
                            );
                            assert_eq!(consumed, encoded.len());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn spectral_envelope_rejects_inconsistent_segments() {
        use crate::spectral_envelope::{PHASE_MASK_ALL, SpectralEnvelopeConfig};
        let (coeffs, turing, codon, legacy, mut segments, tables) = envelope_fixture();
        let mut config = SpectralEnvelopeConfig::neutral();
        config.amplitude_mode = 2;
        config.phase_mask = PHASE_MASK_ALL;
        config.phase_channels = 0b111;
        let encoded = rans_encode_band_v21_envelope(
            &coeffs,
            &turing,
            &codon,
            &legacy,
            &[],
            &segments,
            &tables,
            &config,
            0,
            PHASE_INITIAL_P_SAME,
            PHASE_ADAPT_SHIFT,
            true,
            &[],
            &mut EnvelopeTables::default(),
        );
        let legacy_refs = vec![PhasePredictorRef::None; coeffs.len()];
        segments.pop();
        let error = rans_decode_band_v21_envelope(
            &encoded,
            coeffs.len(),
            &turing,
            &codon,
            &legacy_refs,
            &[],
            &segments,
            &tables,
            &config,
            0,
            PHASE_INITIAL_P_SAME,
            PHASE_ADAPT_SHIFT,
            true,
            &[],
            &mut EnvelopeTables::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            RansDecodeError::InvalidContextLength { .. }
        ));
    }

    #[test]
    fn test_rans_roundtrip_empty() {
        let input: Vec<i16> = vec![];
        let encoded = rans_encode_band(&input);
        assert!(encoded.is_empty());
        let (decoded, _) = rans_decode_band(&encoded, 0);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_rans_roundtrip_all_zeros() {
        let input = vec![0i16; 1000];
        let encoded = rans_encode_band(&input);
        let (decoded, _) = rans_decode_band(&encoded, input.len());
        assert_eq!(input, decoded);
        // All zeros should compress very well
        assert!(
            encoded.len() < 100,
            "all-zeros compressed to {} bytes (expected < 100)",
            encoded.len()
        );
    }

    #[test]
    fn test_rans_roundtrip_all_ones() {
        let input = vec![1i16; 500];
        let encoded = rans_encode_band(&input);
        let (decoded, _) = rans_decode_band(&encoded, input.len());
        assert_eq!(input, decoded);
    }

    #[test]
    fn test_rans_roundtrip_large_values() {
        let input: Vec<i16> = vec![0, 0, 15, 0, -20, 0, 0, 0, 100, -50, 0, 0, 7, 0, 0, -8];
        let encoded = rans_encode_band(&input);
        let (decoded, _) = rans_decode_band(&encoded, input.len());
        assert_eq!(input, decoded, "large values roundtrip failed");
    }

    #[test]
    fn test_rans_skewed() {
        // 90% zeros, 7% pm1, 3% larger — simulates wavelet data
        let n = 10000;
        let mut input = Vec::with_capacity(n);
        let mut rng_state: u64 = 42;

        for _ in 0..n {
            // Simple LCG PRNG
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((rng_state >> 33) as u32) % 100;
            if r < 90 {
                input.push(0);
            } else if r < 97 {
                if rng_state & 0x10000 != 0 {
                    input.push(1);
                } else {
                    input.push(-1);
                }
            } else {
                let mag = (((rng_state >> 16) as u32 % 10) + 2) as i16;
                if rng_state & 0x20000 != 0 {
                    input.push(mag);
                } else {
                    input.push(-mag);
                }
            }
        }

        let encoded = rans_encode_band(&input);
        let (decoded, _) = rans_decode_band(&encoded, input.len());
        assert_eq!(input, decoded, "skewed roundtrip failed");

        // Check compression ratio: 90% zeros should compress well
        let bpp = (encoded.len() as f64 * 8.0) / n as f64;
        assert!(bpp < 2.0, "skewed data: {} bpp (expected < 2.0)", bpp);
    }

    #[test]
    fn test_rans_band_roundtrip() {
        // Simulate a realistic detail band: mostly zeros, some pm1, few larger
        let n = 4096; // 64x64 band
        let mut coeffs = vec![0i16; n];

        // Sprinkle some nonzero values in Morton-like patterns
        let mut rng: u64 = 12345;
        for c in coeffs.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((rng >> 33) as u32) % 100;
            if r < 86 {
                *c = 0;
            } else if r < 97 {
                *c = if rng & 0x8000 != 0 { 1 } else { -1 };
            } else {
                let mag = (((rng >> 16) as u32 % 8) + 2) as i16;
                *c = if rng & 0x4000 != 0 { mag } else { -mag };
            }
        }

        let encoded = rans_encode_band(&coeffs);
        let (decoded, bytes_consumed) = rans_decode_band(&encoded, n);
        assert_eq!(coeffs, decoded, "band roundtrip failed");
        assert!(
            bytes_consumed <= encoded.len(),
            "consumed more bytes than available"
        );
    }

    #[test]
    fn test_rans_paeth_roundtrip() {
        // Paeth residuals: Laplacian distribution centered at 0
        let n = 5000;
        let mut residuals = Vec::with_capacity(n);
        let mut rng: u64 = 9999;

        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Approximate Laplacian: geometric distribution for magnitude
            let u = ((rng >> 16) as u32) % 256;
            let mag = if u < 100 {
                0i16
            } else if u < 170 {
                1
            } else if u < 210 {
                2
            } else if u < 235 {
                3
            } else if u < 248 {
                ((rng >> 8) as i16 % 5).abs() + 4
            } else {
                ((rng >> 8) as i16 % 20).abs() + 9
            };
            let sign = if rng & 0x80000000 != 0 { -1i16 } else { 1i16 };
            let val = if mag == 0 { 0 } else { mag * sign };
            residuals.push(val);
        }

        let encoded = rans_encode_paeth(&residuals);
        let (decoded, _bytes) = rans_decode_paeth(&encoded, n);
        assert_eq!(residuals, decoded, "paeth roundtrip failed");
    }

    #[test]
    fn test_rans_paeth_roundtrip_small() {
        // Small values typical of Paeth residuals for well-predicted images
        let input: Vec<i16> = vec![0, 0, 1, 0, -1, 0, 0, 2, 0, -1, 0, 0, 0, -2, 3, 0, 0, 1];
        let encoded = rans_encode_paeth(&input);
        let (decoded, _) = rans_decode_paeth(&encoded, input.len());
        assert_eq!(input, decoded);
    }

    #[test]
    fn test_rans_paeth_empty() {
        let encoded = rans_encode_paeth(&[]);
        assert!(encoded.is_empty());
        let (decoded, _) = rans_decode_paeth(&encoded, 0);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_rans_vs_entropy() {
        // Generate data with known entropy and verify compressed size is close
        let n = 10000;
        let mut data = Vec::with_capacity(n);
        let mut rng: u64 = 7777;

        // 90% zeros, 10% ones => H = -0.9*log2(0.9) - 0.1*log2(0.1) ~ 0.469 bits/symbol
        // Total entropy ~ 4690 bits ~ 587 bytes
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((rng >> 33) as u32) % 100;
            if r < 90 {
                data.push(0i16);
            } else {
                data.push(1);
            }
        }

        let encoded = rans_encode_band(&data);
        let (decoded, _) = rans_decode_band(&encoded, n);
        assert_eq!(data, decoded);

        // Compute actual entropy
        let n_zero = data.iter().filter(|&&v| v == 0).count();
        let n_one = n - n_zero;
        let p0 = n_zero as f64 / n as f64;
        let p1 = n_one as f64 / n as f64;
        let entropy_bits = -(p0 * p0.log2() + p1 * p1.log2()) * n as f64;
        let entropy_bytes = (entropy_bits / 8.0).ceil() as usize;

        // rANS with adaptive context model has overhead from:
        // - 32 context slots adapting from scratch on 10K symbols
        // - 4-byte state flush
        // - context model mispredictions during warmup
        // Allow 30% overhead for these factors.
        let overhead_factor = 1.30;
        let max_expected = (entropy_bytes as f64 * overhead_factor) as usize + 20; // +20 for state bytes
        assert!(
            encoded.len() <= max_expected,
            "encoded {} bytes, expected <= {} (entropy = {} bytes)",
            encoded.len(),
            max_expected,
            entropy_bytes
        );

        // Also verify it's not too small (sanity check — can't beat entropy)
        // Allow 50% of entropy as floor (context model might adapt better than static)
        let min_expected = entropy_bytes / 2;
        assert!(
            encoded.len() >= min_expected,
            "encoded {} bytes, suspiciously small (entropy = {} bytes)",
            encoded.len(),
            entropy_bytes
        );
    }

    #[test]
    fn test_rans_encoder_decoder_raw_bits() {
        // Test raw bit encoding/decoding directly
        let mut enc = RansEncoder::new();
        let bits = [
            true, false, true, true, false, false, true, false, true, true, true, false, false,
            false, true, true,
        ];

        // Encode backwards (rANS requirement)
        for &b in bits.iter().rev() {
            enc.put_bit(b);
        }
        let data = enc.finish();

        let mut dec = RansDecoder::new(&data);
        let mut decoded_bits = Vec::new();
        for _ in 0..bits.len() {
            decoded_bits.push(dec.get_bit());
        }
        assert_eq!(&bits[..], &decoded_bits[..]);
    }

    #[test]
    fn test_rans_encoder_decoder_adaptive() {
        // Test encoding/decoding with varying probabilities
        let mut enc = RansEncoder::new();

        // Encode a sequence where symbol 0 has varying probability
        let symbols = [0u32, 0, 0, 1, 0, 1, 1, 0, 0, 0];
        let probs: Vec<u32> = vec![
            8192, 12000, 14000, 4000, 10000, 6000, 3000, 9000, 11000, 15000,
        ];

        // Encode backwards
        for i in (0..symbols.len()).rev() {
            let p = probs[i];
            let (cum, freq) = if symbols[i] == 0 {
                (0, p)
            } else {
                (p, PROB_SCALE - p)
            };
            enc.put(cum, freq);
        }

        let data = enc.finish();

        let mut dec = RansDecoder::new(&data);
        for i in 0..symbols.len() {
            let p = probs[i];
            let cum = dec.get();
            let sym = if cum < p { 0 } else { 1 };
            assert_eq!(sym, symbols[i], "mismatch at position {}", i);
            let (c, f) = if sym == 0 {
                (0, p)
            } else {
                (p, PROB_SCALE - p)
            };
            dec.advance(c, f);
        }
    }

    #[test]
    fn test_rans_deterministic() {
        // Same input must always produce the same output
        let input: Vec<i16> = vec![0, 1, -1, 3, 0, 0, -5, 0, 2, 0];
        let enc1 = rans_encode_band(&input);
        let enc2 = rans_encode_band(&input);
        assert_eq!(enc1, enc2, "encoding is not deterministic");
    }

    #[test]
    fn test_rans_band_stress() {
        // Stress test with many different patterns
        let patterns: Vec<Vec<i16>> = vec![
            vec![0; 100],
            vec![1; 100],
            vec![-1; 100],
            (0..100).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect(),
            (0..100).map(|i| (i % 7 - 3) as i16).collect(),
            (0..200)
                .map(|i| if i < 180 { 0 } else { (i - 180) as i16 + 2 })
                .collect(),
            vec![0, 0, 0, 30, 0, 0, 0, -30, 0, 0],
        ];

        for (idx, pattern) in patterns.iter().enumerate() {
            let encoded = rans_encode_band(pattern);
            let (decoded, _) = rans_decode_band(&encoded, pattern.len());
            assert_eq!(pattern, &decoded, "stress pattern {} failed", idx);
        }
    }

    #[test]
    fn test_rans_paeth_large_residuals() {
        // Test with larger Paeth residuals (e.g., at image edges)
        let input: Vec<i16> = vec![0, 50, -30, 0, 0, 100, -80, 0, 0, 15, -200, 0, 127, -128];
        let encoded = rans_encode_paeth(&input);
        let (decoded, _) = rans_decode_paeth(&encoded, input.len());
        assert_eq!(input, decoded, "large paeth residuals roundtrip failed");
    }

    #[test]
    fn test_rans_compress_bytes_roundtrip() {
        let input = b"Hello, world! This is a test of rANS byte-level compression.";
        let compressed = rans_compress_bytes(input);
        let decompressed = rans_decompress_bytes(&compressed);
        assert_eq!(
            &input[..],
            &decompressed[..],
            "byte compression roundtrip failed"
        );
    }

    #[test]
    fn test_rans_compress_bytes_empty() {
        let compressed = rans_compress_bytes(&[]);
        assert!(compressed.is_empty());
        let decompressed = rans_decompress_bytes(&compressed);
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_rans_compress_bytes_single() {
        let input = [42u8];
        let compressed = rans_compress_bytes(&input);
        let decompressed = rans_decompress_bytes(&compressed);
        assert_eq!(&input[..], &decompressed[..]);
    }

    #[test]
    fn test_rans_compress_bytes_repeated() {
        // Highly compressible: all same byte
        let input = vec![0xAAu8; 10000];
        let compressed = rans_compress_bytes(&input);
        let decompressed = rans_decompress_bytes(&compressed);
        assert_eq!(input, decompressed);
        // Should compress very well
        assert!(
            compressed.len() < 500,
            "repeated bytes compressed to {} (expected < 500)",
            compressed.len()
        );
    }

    #[test]
    fn test_rans_compress_bytes_all_values() {
        // All 256 byte values, then repeat
        let mut input = Vec::with_capacity(512);
        for _ in 0..2 {
            for b in 0..=255u8 {
                input.push(b);
            }
        }
        let compressed = rans_compress_bytes(&input);
        let decompressed = rans_decompress_bytes(&compressed);
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_rans_compress_bytes_structured() {
        // Simulate structured wavelet-like data: mostly small values
        let n = 5000;
        let mut input = Vec::with_capacity(n);
        let mut rng: u64 = 31337;
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((rng >> 33) as u32) % 100;
            if r < 60 {
                input.push(0u8);
            } else if r < 85 {
                input.push(1u8);
            } else if r < 95 {
                input.push(((rng >> 16) as u8) % 10 + 2);
            } else {
                input.push((rng >> 8) as u8);
            }
        }
        let compressed = rans_compress_bytes(&input);
        let decompressed = rans_decompress_bytes(&compressed);
        assert_eq!(
            input, decompressed,
            "structured byte compression roundtrip failed"
        );
        // Structured data should compress
        assert!(
            compressed.len() < input.len(),
            "structured data not compressed: {} >= {}",
            compressed.len(),
            input.len()
        );
    }

    // ==================================================================
    // v11 hyper-sparse rANS tests
    // ==================================================================

    #[test]
    fn test_rans_v11_roundtrip_basic() {
        let input: Vec<i16> = vec![0, 1, -1, 0, 0, 2, -3, 0, 0, 0, 1, 0, -1, 5, 0];
        let encoded = rans_encode_band_v11(&input);
        let (decoded, _bytes) = rans_decode_band_v11(&encoded, input.len());
        assert_eq!(input, decoded, "v11 basic roundtrip failed");
    }

    #[test]
    fn test_rans_v11_roundtrip_empty() {
        let input: Vec<i16> = vec![];
        let encoded = rans_encode_band_v11(&input);
        assert!(encoded.is_empty());
        let (decoded, _) = rans_decode_band_v11(&encoded, 0);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_rans_v11_roundtrip_all_zeros() {
        let input = vec![0i16; 1000];
        let encoded = rans_encode_band_v11(&input);
        let (decoded, _) = rans_decode_band_v11(&encoded, input.len());
        assert_eq!(input, decoded);
        assert!(
            encoded.len() < 100,
            "v11 all-zeros compressed to {} bytes",
            encoded.len()
        );
    }

    #[test]
    fn test_rans_v11_roundtrip_all_ones() {
        let input = vec![1i16; 500];
        let encoded = rans_encode_band_v11(&input);
        let (decoded, _) = rans_decode_band_v11(&encoded, input.len());
        assert_eq!(input, decoded);
    }

    #[test]
    fn test_rans_v11_roundtrip_large_values() {
        let input: Vec<i16> = vec![0, 0, 10, -20, 0, 0, 0, 100, 0, -50, 0, 0, 200, -200];
        let encoded = rans_encode_band_v11(&input);
        let (decoded, _) = rans_decode_band_v11(&encoded, input.len());
        assert_eq!(input, decoded, "v11 large values roundtrip failed");
    }

    #[test]
    fn test_rans_v11_hyper_sparse() {
        // 99% zeros with scattered nonzero values — long zero runs.
        // v11 with 6 run-class buckets should beat v8's 4 buckets.
        let n = 10000;
        let mut input = Vec::with_capacity(n);
        let mut rng: u64 = 42;
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((rng >> 33) as u32) % 100;
            if r < 99 {
                input.push(0i16);
            } else {
                input.push(((rng >> 16) as i16) % 5 + 1);
            }
        }

        let encoded_v8 = rans_encode_band(&input);
        let encoded_v11 = rans_encode_band_v11(&input);

        // Verify v11 roundtrip
        let (decoded, _) = rans_decode_band_v11(&encoded_v11, input.len());
        assert_eq!(input, decoded, "v11 hyper-sparse roundtrip failed");

        // v11 should compress at least as well as v8 on hyper-sparse data
        assert!(
            encoded_v11.len() <= encoded_v8.len(),
            "v11 ({} bytes) should be <= v8 ({} bytes) on hyper-sparse data",
            encoded_v11.len(),
            encoded_v8.len()
        );
    }

    #[test]
    fn test_rans_v11_stress() {
        let patterns: Vec<Vec<i16>> = vec![
            vec![0; 100],
            vec![1; 100],
            vec![-1; 100],
            (0..100).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect(),
            (0..100).map(|i| (i % 7 - 3) as i16).collect(),
            (0..200)
                .map(|i| if i < 180 { 0 } else { (i - 180) as i16 + 2 })
                .collect(),
            vec![0, 0, 0, 30, 0, 0, 0, -30, 0, 0],
        ];

        for (idx, pattern) in patterns.iter().enumerate() {
            let encoded = rans_encode_band_v11(pattern);
            let (decoded, _) = rans_decode_band_v11(&encoded, pattern.len());
            assert_eq!(pattern, &decoded, "v11 stress pattern {} failed", idx);
        }
    }

    #[test]
    fn test_rans_v8_backward_compat() {
        // Verify old v8 functions produce deterministic output (regression guard)
        let input: Vec<i16> = vec![0, 1, -1, 0, 0, 2, -3, 0, 0, 0, 1, 0, -1, 5, 0];
        let enc1 = rans_encode_band(&input);
        let enc2 = rans_encode_band(&input);
        assert_eq!(enc1, enc2, "v8 encoding changed (backward compat broken)");
        let (dec, _) = rans_decode_band(&enc1, input.len());
        assert_eq!(input, dec, "v8 decode changed (backward compat broken)");
    }

    // ==================================================================
    // v12 context model tests
    // ==================================================================

    #[test]
    fn test_context_model_v12_bucket_indexing() {
        let mut cm = ContextModelV12::new();
        cm.set_turing_level(0);
        let idx0 = cm.zero_ctx_index();
        cm.set_turing_level(3);
        let idx3 = cm.zero_ctx_index();
        assert_ne!(idx0, idx3);
        assert!(idx0 < N_CTX_ZERO_V12);
        assert!(idx3 < N_CTX_ZERO_V12);
    }

    #[test]
    fn test_context_model_v12_symmetry_with_v11() {
        let cm = ContextModelV12::new();
        let p_zero_calm = cm.p_zero[0];
        assert!(
            p_zero_calm > 12000,
            "default P(zero) should be ~85%: {p_zero_calm}"
        );
    }

    #[test]
    fn test_context_model_v12_update() {
        let mut cm = ContextModelV12::new();
        cm.set_turing_level(2);
        let before = cm.p_zero[cm.zero_ctx_index()];
        cm.update(0); // zero symbol → should increase p_zero
        let after = cm.p_zero[cm.zero_ctx_index()];
        assert!(after >= before, "zero symbol should increase P(zero)");
    }

    #[test]
    fn test_rans_v12_checked_roundtrip() {
        let input = vec![0, 0, 1, -1, 2, -7, 0, 31, -32768, 32767, 0];
        let buckets: Vec<u8> = (0..input.len()).map(|i| (i % 4) as u8).collect();
        let encoded = rans_encode_band_v12(&input, &buckets);
        let (decoded, consumed) = rans_decode_band_v12(&encoded, input.len(), &buckets).unwrap();
        assert_eq!(decoded, input);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn test_rans_v12_rejects_every_truncation() {
        let input = vec![0, 1, -2, 0, 17, -31, 0, 0, 3];
        let buckets = vec![2; input.len()];
        let encoded = rans_encode_band_v12(&input, &buckets);
        for cut in 0..encoded.len() {
            assert!(
                rans_decode_band_v12(&encoded[..cut], input.len(), &buckets).is_err(),
                "truncation at byte {cut} was accepted",
            );
        }
    }

    #[test]
    fn test_rans_v12_rejects_context_mismatch_and_trailing_data() {
        let input = vec![0, 1, 0, -3];
        let buckets = vec![0; input.len()];
        let encoded = rans_encode_band_v12(&input, &buckets);
        assert!(matches!(
            rans_decode_band_v12(&encoded, input.len(), &buckets[..3]),
            Err(RansDecodeError::InvalidContextLength { .. }),
        ));

        let mut with_trailing = encoded;
        with_trailing.push(0);
        assert!(matches!(
            rans_decode_band_v12(&with_trailing, input.len(), &buckets),
            Err(RansDecodeError::TrailingData { .. }),
        ));
    }
}
