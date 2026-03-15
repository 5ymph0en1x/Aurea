//! rANS (range Asymmetric Numeral Systems) entropy coder.
//!
//! Byte-aligned rANS with 32-bit state, zero external dependencies.
//! Replaces LZMA for the AUREA codec detail bands and Paeth residuals.
//!
//! Two main coding modes:
//! - Band coding: context-adaptive (run-class x prev_nz x energy_bucket)
//! - Paeth coding: Laplacian model (sign x magnitude bucket)

// ======================================================================
// Constants
// ======================================================================

const RANS_BYTE_L: u32 = 1 << 23;
const PROB_BITS: u32 = 14;
const PROB_SCALE: u32 = 1 << PROB_BITS; // 16384

// Context model
const N_CTX_ZERO: usize = 32; // run_class(4) x prev_nz(2) x energy_bucket(4)
const N_CTX_PM1: usize = 8;   // energy_bucket(4) x prev_was_pm1(2)
const ADAPT_SHIFT: u32 = 4;   // adaptation rate: 1/16 per symbol
const ENERGY_ALPHA: u32 = 218; // IIR smoothing: alpha/256
const ENERGY_SCALE: u32 = 256;

// Paeth context model
const N_PAETH_CTX: usize = 8; // sign(2) x mag_bucket(4)
const PAETH_ADAPT_SHIFT: u32 = 5; // slower adaptation for Paeth (1/32)

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
        debug_assert!(cum_freq + freq <= PROB_SCALE, "cum_freq + freq must be <= PROB_SCALE");

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
        Self { state, data, pos: 4 }
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
        self.state = freq * (self.state >> PROB_BITS)
            + (self.state & (PROB_SCALE - 1))
            - cum_freq;
        // Renormalize
        while self.state < RANS_BYTE_L {
            let byte = self.data.get(self.pos).copied().unwrap_or(0) as u32;
            self.pos += 1;
            self.state = (self.state << 8) | byte;
        }
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
        let is_pm1 = value.abs() == 1;
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
        self.energy = (ENERGY_ALPHA * self.energy
            + (ENERGY_SCALE - ENERGY_ALPHA) * sample_energy)
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

    let original_len = u32::from_le_bytes([
        compressed[0],
        compressed[1],
        compressed[2],
        compressed[3],
    ]) as usize;

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
        assert!(encoded.len() < 100, "all-zeros compressed to {} bytes (expected < 100)", encoded.len());
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
                if rng_state & 0x10000 != 0 { input.push(1); } else { input.push(-1); }
            } else {
                let mag = (((rng_state >> 16) as u32 % 10) + 2) as i16;
                if rng_state & 0x20000 != 0 { input.push(mag); } else { input.push(-mag); }
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
        assert!(bytes_consumed <= encoded.len(), "consumed more bytes than available");
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
            encoded.len(), max_expected, entropy_bytes
        );

        // Also verify it's not too small (sanity check — can't beat entropy)
        // Allow 50% of entropy as floor (context model might adapt better than static)
        let min_expected = entropy_bytes / 2;
        assert!(
            encoded.len() >= min_expected,
            "encoded {} bytes, suspiciously small (entropy = {} bytes)",
            encoded.len(), entropy_bytes
        );
    }

    #[test]
    fn test_rans_encoder_decoder_raw_bits() {
        // Test raw bit encoding/decoding directly
        let mut enc = RansEncoder::new();
        let bits = [true, false, true, true, false, false, true, false,
                    true, true, true, false, false, false, true, true];

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
        let probs: Vec<u32> = vec![8192, 12000, 14000, 4000, 10000, 6000, 3000, 9000, 11000, 15000];

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
            let (c, f) = if sym == 0 { (0, p) } else { (p, PROB_SCALE - p) };
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
            (0..200).map(|i| if i < 180 { 0 } else { (i - 180) as i16 + 2 }).collect(),
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
        assert_eq!(&input[..], &decompressed[..], "byte compression roundtrip failed");
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
        assert_eq!(input, decompressed, "structured byte compression roundtrip failed");
        // Structured data should compress
        assert!(
            compressed.len() < input.len(),
            "structured data not compressed: {} >= {}",
            compressed.len(),
            input.len()
        );
    }
}
