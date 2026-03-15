//! Zeckendorf coding (Fibonacci coding).
//!
//! Each positive integer n >= 1 is encoded as its Zeckendorf representation
//! (sum of non-consecutive Fibonacci numbers), terminated by "11".
//! Self-delimiting universal code, optimal for geometric distributions.

/// Fibonacci numbers: FIB[i] = F(i+2).
/// FIB[0]=1, FIB[1]=2, FIB[2]=3, FIB[3]=5, FIB[4]=8, ...
const FIB: [u64; 45] = [
    1, 2, 3, 5, 8, 13, 21, 34, 55, 89,
    144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946,
    17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040, 1346269,
    2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155, 165580141,
    267914296, 433494437, 701408733, 1134903170, 1836311903,
];

/// Bit-by-bit writer into a byte buffer.
pub struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    bit_count: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter { bytes: Vec::new(), current: 0, bit_count: 0 }
    }

    pub fn with_capacity(bits: usize) -> Self {
        BitWriter {
            bytes: Vec::with_capacity((bits + 7) / 8),
            current: 0,
            bit_count: 0,
        }
    }

    #[inline]
    pub fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current |= 1 << self.bit_count;
        }
        self.bit_count += 1;
        if self.bit_count == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.bit_count = 0;
        }
    }

    /// Write `nbits` bits of `val` in LSB-first (consistent with write_bit).
    pub fn write_bits(&mut self, val: u64, nbits: usize) {
        for i in 0..nbits {
            self.write_bit((val >> i) & 1 != 0);
        }
    }

    /// Current position in bits.
    pub fn bit_position(&self) -> usize {
        self.bytes.len() * 8 + self.bit_count as usize
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.bit_count > 0 {
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

/// Bit-by-bit reader from a byte buffer.
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, byte_pos: 0, bit_pos: 0 }
    }

    #[inline]
    pub fn read_bit(&mut self) -> Option<bool> {
        if self.byte_pos >= self.data.len() {
            return None;
        }
        let bit = (self.data[self.byte_pos] >> self.bit_pos) & 1 != 0;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
        Some(bit)
    }

    /// Read `nbits` bits and return the value (LSB-first).
    pub fn read_bits(&mut self, nbits: usize) -> Option<u64> {
        let mut val: u64 = 0;
        for i in 0..nbits {
            val |= (self.read_bit()? as u64) << i;
        }
        Some(val)
    }

    /// Current position in bits.
    pub fn bit_position(&self) -> usize {
        self.byte_pos * 8 + self.bit_pos as usize
    }
}

/// Encode integer n >= 1 in Zeckendorf representation.
/// Emits bits into the BitWriter, terminating with a "11" doublet.
pub fn zeckendorf_encode(writer: &mut BitWriter, n: u64) {
    debug_assert!(n >= 1, "Zeckendorf requires n >= 1");

    // Find maximum index k such that FIB[k] <= n
    let mut k = 0usize;
    while k + 1 < FIB.len() && FIB[k + 1] <= n {
        k += 1;
    }

    // Greedy decomposition (unique representation without consecutive Fibonacci)
    let mut used = 0u64;
    let mut remaining = n;
    for i in (0..=k).rev() {
        if FIB[i] <= remaining {
            used |= 1 << i;
            remaining -= FIB[i];
            if remaining == 0 { break; }
        }
    }

    // Emit bits from LSB (FIB[0]) to MSB (FIB[k])
    for i in 0..=k {
        writer.write_bit(used & (1 << i) != 0);
    }
    // Terminator bit -> creates "11" at the end
    writer.write_bit(true);
}

/// Decode a Zeckendorf-encoded integer from the BitReader.
/// Reads until the "11" doublet marking the end of the code.
pub fn zeckendorf_decode(reader: &mut BitReader) -> Option<u64> {
    let mut n = 0u64;
    let mut prev = false;
    let mut idx = 0usize;

    loop {
        let bit = reader.read_bit()?;
        if bit && prev {
            break;
        }
        if bit && idx < FIB.len() {
            n += FIB[idx];
        }
        prev = bit;
        idx += 1;
    }

    Some(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        for n in 1..=200u64 {
            let mut w = BitWriter::new();
            zeckendorf_encode(&mut w, n);
            let bytes = w.finish();
            let mut r = BitReader::new(&bytes);
            let decoded = zeckendorf_decode(&mut r).unwrap();
            assert_eq!(n, decoded, "roundtrip failed for n={}", n);
        }
    }

    #[test]
    fn test_multiple_values() {
        let values = vec![1, 5, 3, 8, 2, 13, 1, 1, 12, 7, 21, 100];
        let mut w = BitWriter::new();
        for &v in &values {
            zeckendorf_encode(&mut w, v);
        }
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        for &expected in &values {
            let decoded = zeckendorf_decode(&mut r).unwrap();
            assert_eq!(expected, decoded);
        }
    }
}
