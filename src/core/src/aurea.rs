/// Decoder for the AUREA format (.aur).
///
/// v1: CDF 9/7 wavelet + VQ/fibonacci on LL,
/// but detail bands use three Fibonacci-inspired mechanisms:
/// 1. Fibonacci quantization: levels at Fibonacci positions (Weber-Fechner)
/// 2. Golden spiral scan: traversal by golden ratio phi (spatial anti-aliasing)
/// 3. Zeckendorf coding: self-delimiting universal code based on Fibonacci
///
/// v2 "Golden Fusion": Golden Color Transform (GCT) -- phi-based golden rotation.
/// L_phi luminance, C1/C2 chromas, chroma quantization at step * phi.

use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::Array2;
use std::io::{self, Cursor};

use crate::bitstream::{X267V6Header, X267V6Stream};
use crate::golden;
use crate::zeckendorf::{BitReader, zeckendorf_decode};

pub const AUREA_MAGIC: &[u8; 4] = b"AURA";

/// Fibonacci quantization levels.
/// Index 0 = zero. Index i corresponds to the i-th Fibonacci number >= 1.
pub const FIB_QUANT_LEVELS: [i32; 13] = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233];

/// AUREA codec constants
const LEVEL_SCALES: [f64; 4] = [1.0, 0.9, 0.5, 0.3];
const PERCEPTUAL_WEIGHTS: [f64; 3] = [1.3, 0.65, 1.3];
const CHROMA_FACTOR: f64 = 1.5;

/// Parse the LZMA-decompressed payload of an .aur file (without the AURA magic).
/// Supports v1 (YCbCr 4:2:0) and v2 (independent RGB + cross-channel).
pub fn parse_aurea_payload(data: &[u8]) -> io::Result<X267V6Stream> {
    let mut c = Cursor::new(data);

    let version = c.read_u8()?;
    if ![1, 2, 3, 4, 5, 6, 7].contains(&version) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported AUREA version {} (expected 1-7)", version),
        ));
    }

    let quality = c.read_u8()?;

    // Dimensions (W first, then H)
    let w = c.read_u16::<LittleEndian>()? as usize;
    let h = c.read_u16::<LittleEndian>()? as usize;
    let wv_levels = c.read_u8()? as usize;

    // v2+: detail_step stored in header (auto-adaptive)
    let stored_detail_step = if version >= 2 {
        Some(c.read_f32::<LittleEndian>()? as f64)
    } else {
        None
    };

    // Chromas in 4:2:0 (v1 Cb/Cr, v2 GCT C1/C2)
    let hc = (h + 1) / 2;
    let wc = (w + 1) / 2;

    // LL ranges: 3 x (f32_min, f32_max) -- G/R/B in v2, Y/Cb/Cr in v1
    let y_min = c.read_f32::<LittleEndian>()?;
    let y_max = c.read_f32::<LittleEndian>()?;
    let cb_min = c.read_f32::<LittleEndian>()?;
    let cb_max = c.read_f32::<LittleEndian>()?;
    let cr_min = c.read_f32::<LittleEndian>()?;
    let cr_max = c.read_f32::<LittleEndian>()?;
    let ll_ranges = [(y_min, y_max), (cb_min, cb_max), (cr_min, cr_max)];

    // Centroid counts
    let n_y = c.read_u8()? as usize;
    let n_c = c.read_u8()? as usize;
    let n_cb = n_c;
    let n_cr = n_c;

    // Centroids
    let mut centroids_y = Vec::with_capacity(n_y);
    for _ in 0..n_y { centroids_y.push(c.read_u8()? as f64); }
    let mut centroids_cb = Vec::with_capacity(n_cb);
    for _ in 0..n_cb { centroids_cb.push(c.read_u8()? as f64); }
    let mut centroids_cr = Vec::with_capacity(n_cr);
    for _ in 0..n_cr { centroids_cr.push(c.read_u8()? as f64); }

    // Compute wavelet sizes
    let mut yh = h; let mut yw = w;
    let mut ch = hc; let mut cw = wc;
    let mut y_sizes = Vec::with_capacity(wv_levels);
    let mut cb_sizes = Vec::with_capacity(wv_levels);

    for _ in 0..wv_levels {
        y_sizes.push((yh, yw));
        cb_sizes.push((ch, cw));
        yh = (yh + 1) / 2;
        yw = (yw + 1) / 2;
        ch = (ch + 1) / 2;
        cw = (cw + 1) / 2;
    }
    let ll_y_size = (yh, yw);
    let ll_c_size = (ch, cw);

    // Paeth residuals — format depends on version
    let pos = c.position() as usize;
    let pred_y;
    let pred_cb;
    let pred_cr;
    let det_start;

    if version >= 2 {
        // v2+: per-stream rANS encoding — each Paeth stream is [u32 size][rANS data]
        let mut p = pos;
        let sz_l = u32::from_le_bytes([data[p], data[p+1], data[p+2], data[p+3]]) as usize; p += 4;
        let (dec_l, _) = crate::rans::rans_decode_paeth(&data[p..p+sz_l], yh * yw);
        pred_y = dec_l; p += sz_l;

        let sz_c1 = u32::from_le_bytes([data[p], data[p+1], data[p+2], data[p+3]]) as usize; p += 4;
        let (dec_c1, _) = crate::rans::rans_decode_paeth(&data[p..p+sz_c1], ch * cw);
        pred_cb = dec_c1; p += sz_c1;

        let sz_c2 = u32::from_le_bytes([data[p], data[p+1], data[p+2], data[p+3]]) as usize; p += 4;
        let (dec_c2, _) = crate::rans::rans_decode_paeth(&data[p..p+sz_c2], ch * cw);
        pred_cr = dec_c2; p += sz_c2;

        det_start = p;
    } else {
        // v1: raw i8/i16 Paeth residuals
        let remaining = &data[pos..];
        let mut c2 = Cursor::new(remaining);
        pred_y = unpack_pred_residuals(&mut c2, yh * yw)?;
        pred_cb = unpack_pred_residuals(&mut c2, ch * cw)?;
        pred_cr = unpack_pred_residuals(&mut c2, ch * cw)?;
        det_start = pos + c2.position() as usize;
    }

    // No pred_coeffs (GCT handles decorrelation)
    let pred_coeffs = None;

    // Detail data
    let detail_data = data[det_start..].to_vec();

    // Compute steps from quality (v1: fixed formula, v2+: step stored in header)
    let detail_step = if let Some(s) = stored_detail_step {
        s
    } else {
        ((101.0 - quality as f64) / 8.0).max(1.0)
    };
    let chroma_factor = if version >= 2 { 1.5 } else { CHROMA_FACTOR };
    let mut steps_y = Vec::with_capacity(wv_levels);
    let mut steps_c = Vec::with_capacity(wv_levels);
    for lv in 0..wv_levels {
        let ls = if lv < LEVEL_SCALES.len() { LEVEL_SCALES[lv] } else { 0.3 };
        let mut sy = [0.0f64; 3];
        let mut sc = [0.0f64; 3];
        for bi in 0..3 {
            sy[bi] = detail_step * ls * PERCEPTUAL_WEIGHTS[bi];
            sc[bi] = sy[bi] * chroma_factor;
        }
        steps_y.push(sy);
        steps_c.push(sc);
    }

    let ctx_flags = if version == 7 || version == 6 {
        // v6/v7 geometric: residuals use Morton+2bit (v6 sigmap, v7 rANS-band)
        crate::wavelet::FLAG_MORTON_2BIT
    } else if version == 5 {
        crate::wavelet::FLAG_MORTON_2BIT | crate::wavelet::FLAG_ADAPTIVE_QUANT | crate::wavelet::FLAG_ADN
    } else if version == 4 {
        crate::wavelet::FLAG_MORTON_2BIT | crate::wavelet::FLAG_ADAPTIVE_QUANT | crate::wavelet::FLAG_CONTEXT
    } else {
        crate::wavelet::FLAG_MORTON_2BIT | crate::wavelet::FLAG_ADAPTIVE_QUANT
    };

    let header = X267V6Header {
        h, w, hc, wc,
        n_y, n_cb, n_cr,
        noise_sigma: 0.0,
        wv_levels,
        ll_ranges,
        cr_sizes: cb_sizes.clone(),
        y_sizes,
        cb_sizes,
        ll_y_size,
        ll_c_size,
        ll_cr_size: ll_c_size,
        steps_cr: steps_c.clone(),
        steps_y,
        steps_c,
        flags: ctx_flags,
    };

    Ok(X267V6Stream {
        header,
        centroids_y,
        centroids_cb,
        centroids_cr,
        pred_y,
        pred_cb,
        pred_cr,
        detail_data,
        detail_offset: 0,
        band_size_prefixed: true,
        mera_angles: None,
        aurea_bands: false,
        aurea_v2: version >= 2,
        pred_coeffs,
        geometric: version == 6 || version == 7,
        rans_bands: version == 7,
    })
}

/// Decode an AUREA detail band with size prefix (u32 LE).
pub fn decode_aurea_band_with_prefix(
    det: &[u8], pos: usize, h: usize, w: usize, step: f64,
) -> (Array2<f64>, usize) {
    let band_size = u32::from_le_bytes([
        det[pos], det[pos + 1], det[pos + 2], det[pos + 3],
    ]) as usize;
    let data_start = pos + 4;
    let band_data = &det[data_start..data_start + band_size];
    let band = decode_aurea_band(band_data, h, w, step);
    (band, data_start + band_size)
}

/// Decode a band from a Zeckendorf + golden scan stream.
fn decode_aurea_band(data: &[u8], h: usize, w: usize, step: f64) -> Array2<f64> {
    let n = h * w;
    if n == 0 {
        return Array2::zeros((h, w));
    }

    let scan = golden::golden_scan_order(h, w);
    let mut reader = BitReader::new(data);
    let mut flat = vec![0.0f64; n];

    for &pos in &scan {
        let sig = reader.read_bit().unwrap_or(false);
        if sig {
            let negative = reader.read_bit().unwrap_or(false);
            let fib_idx = zeckendorf_decode(&mut reader).unwrap_or(1) as usize;
            let fib_idx = fib_idx.min(FIB_QUANT_LEVELS.len() - 1);
            let fib_val = FIB_QUANT_LEVELS[fib_idx] as f64;
            flat[pos] = if negative { -fib_val * step } else { fib_val * step };
        }
    }

    Array2::from_shape_vec((h, w), flat).unwrap()
}

/// Decompress Paeth residuals (int8 or int16 depending on marker).
fn unpack_pred_residuals(cursor: &mut Cursor<&[u8]>, size: usize) -> io::Result<Vec<i16>> {
    let marker = cursor.read_u8()?;
    let mut res = Vec::with_capacity(size);

    if marker == 0x08 {
        for _ in 0..size {
            let v = cursor.read_i8()?;
            res.push(v as i16);
        }
    } else {
        for _ in 0..size {
            let v = cursor.read_i16::<LittleEndian>()?;
            res.push(v);
        }
    }

    Ok(res)
}
