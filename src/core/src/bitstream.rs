/// Reading and writing of the XTS file format and internal x267 bitstream.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Cursor, Read, Write};

pub const XTS_MAGIC: &[u8; 4] = b"XTS1";
pub const X267_MAGIC: &[u8; 4] = b"x267";

/// Header of the x267 stream after decompression.
#[derive(Debug, Clone)]
pub struct X267Header {
    pub version: u8,
    pub h: usize,
    pub w: usize,
    pub hc: usize,
    pub wc: usize,
    pub n_y: usize,
    pub n_cb: usize,
    pub n_cr: usize,
    pub alpha_bits: u8,  // v3 only
    pub residu_bits: u8,
    pub res_max_abs: f32,
    pub noise_sigma: f32,  // v5+
}

/// Decoded bitstream data.
#[derive(Debug)]
pub struct X267Stream {
    pub header: X267Header,
    pub centroids_y: Vec<f64>,
    pub centroids_cb: Vec<f64>,
    pub centroids_cr: Vec<f64>,
    pub pred_y: Vec<i16>,
    pub pred_cb: Vec<i16>,
    pub pred_cr: Vec<i16>,
    pub y_pixel_res: Option<Vec<i8>>,
}

/// Decompress raw data (after the XTS1 magic).
/// Detects the format by magic bytes: XZ (0xFD 0x37 0x7A 0x58) or zlib (0x78).
pub fn decompress_xts_payload(data: &[u8]) -> io::Result<Vec<u8>> {
    if data.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Empty payload"));
    }

    // XZ/LZMA: magic 0xFD 0x37 0x7A 0x58 (v4/v5/v6)
    if data.len() >= 6 && data[0] == 0xFD && data[1] == 0x37 && data[2] == 0x7A && data[3] == 0x58 {
        let mut decoder = xz2::read::XzDecoder::new(data);
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf)?;
        return Ok(buf);
    }

    // zlib: CMF byte typically 0x78 (v2/v3)
    let mut decoder = flate2::read::ZlibDecoder::new(data);
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Decompress Paeth residuals (int8 or int16 depending on marker).
fn unpack_pred_residuals(cursor: &mut Cursor<&[u8]>, size: usize) -> io::Result<Vec<i16>> {
    let marker = cursor.read_u8()?;
    let mut res = Vec::with_capacity(size);

    if marker == 0x08 {
        // int8
        for _ in 0..size {
            let v = cursor.read_i8()?;
            res.push(v as i16);
        }
    } else {
        // int16
        for _ in 0..size {
            let v = cursor.read_i16::<LittleEndian>()?;
            res.push(v);
        }
    }

    Ok(res)
}

/// Parse a complete XTS file (magic + compressed payload).
pub fn parse_xts(file_data: &[u8]) -> io::Result<X267Stream> {
    if file_data.len() < 4 || &file_data[0..4] != XTS_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not an XTS file"));
    }

    let stream = decompress_xts_payload(&file_data[4..])?;
    parse_x267_stream(&stream)
}

/// Parse the decompressed x267 stream.
pub fn parse_x267_stream(stream: &[u8]) -> io::Result<X267Stream> {
    let mut cursor = Cursor::new(stream);

    // Magic
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if &magic != X267_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Not an x267 stream (got {:?})", magic),
        ));
    }

    // Version
    let version = cursor.read_u8()?;
    if ![2, 3, 4, 5, 6].contains(&version) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported version {}", version),
        ));
    }

    // V6: completely different pipeline
    if version == 6 {
        return parse_x267_v6(stream);
    }

    // Dimensions
    let h = cursor.read_u16::<LittleEndian>()? as usize;
    let w = cursor.read_u16::<LittleEndian>()? as usize;
    let hc = cursor.read_u16::<LittleEndian>()? as usize;
    let wc = cursor.read_u16::<LittleEndian>()? as usize;

    // Number of centroids
    let n_y = cursor.read_u16::<LittleEndian>()? as usize;
    let n_cb = cursor.read_u8()? as usize;
    let n_cr = cursor.read_u8()? as usize;

    // Alpha bits (v3 only)
    let alpha_bits = if version == 3 {
        cursor.read_u8()?
    } else {
        0
    };

    // Residual bits + max_abs
    let residu_bits = cursor.read_u8()?;
    let res_max_abs = cursor.read_f32::<LittleEndian>()?;

    // Noise sigma (v5+)
    let noise_sigma = if version >= 5 {
        cursor.read_f32::<LittleEndian>()?
    } else {
        0.0
    };

    let header = X267Header {
        version, h, w, hc, wc,
        n_y, n_cb, n_cr,
        alpha_bits, residu_bits, res_max_abs, noise_sigma,
    };

    // Centroids (uint8 -> f64)
    let mut centroids_y = Vec::with_capacity(n_y);
    for _ in 0..n_y {
        centroids_y.push(cursor.read_u8()? as f64);
    }
    let mut centroids_cb = Vec::with_capacity(n_cb);
    for _ in 0..n_cb {
        centroids_cb.push(cursor.read_u8()? as f64);
    }
    let mut centroids_cr = Vec::with_capacity(n_cr);
    for _ in 0..n_cr {
        centroids_cr.push(cursor.read_u8()? as f64);
    }

    // Paeth residuals
    let n = h * w;
    let nc = hc * wc;

    // Re-borrow the underlying slice for unpacking
    let pos = cursor.position() as usize;
    let remaining = &stream[pos..];
    let mut cursor2 = Cursor::new(remaining);

    let pred_y = unpack_pred_residuals(&mut cursor2, n)?;
    let pred_cb = unpack_pred_residuals(&mut cursor2, nc)?;
    let pred_cr = unpack_pred_residuals(&mut cursor2, nc)?;

    // Pixel residuals Y
    let y_pixel_res = if residu_bits > 0 {
        let pos2 = cursor2.position() as usize;
        let rem2 = &remaining[pos2..];
        if rem2.len() >= n {
            let mut res = Vec::with_capacity(n);
            for i in 0..n {
                res.push(rem2[i] as i8);
            }
            Some(res)
        } else {
            None
        }
    } else {
        None
    };

    Ok(X267Stream {
        header,
        centroids_y,
        centroids_cb,
        centroids_cr,
        pred_y,
        pred_cb,
        pred_cr,
        y_pixel_res,
    })
}

// ======================================================================
// Writing (encoder)
// ======================================================================

/// Pack Paeth residuals as i8 (marker 0x08) or i16 (marker 0x10).
pub fn pack_pred_residuals(residuals: &[i16]) -> Vec<u8> {
    let min_val = residuals.iter().copied().min().unwrap_or(0);
    let max_val = residuals.iter().copied().max().unwrap_or(0);

    if min_val >= -128 && max_val <= 127 {
        let mut buf = Vec::with_capacity(1 + residuals.len());
        buf.push(0x08);
        for &r in residuals {
            buf.push(r as i8 as u8);
        }
        buf
    } else {
        let mut buf = Vec::with_capacity(1 + residuals.len() * 2);
        buf.push(0x10);
        for &r in residuals {
            buf.extend_from_slice(&r.to_le_bytes());
        }
        buf
    }
}

/// Write the x267 v5 stream (uncompressed).
#[allow(clippy::too_many_arguments)]
pub fn write_x267_stream(
    h: usize, w: usize, hc: usize, wc: usize,
    n_y: usize, n_cb: usize, n_cr: usize,
    residu_bits: u8, res_max_abs: f32,
    noise_sigma: f32,
    centroids_y: &[f64], centroids_cb: &[f64], centroids_cr: &[f64],
    pred_y: &[i16], pred_cb: &[i16], pred_cr: &[i16],
    y_pixel_res: Option<&[i8]>,
) -> Vec<u8> {
    let mut stream: Vec<u8> = Vec::new();

    // Magic + version
    stream.extend_from_slice(b"x267");
    stream.push(5);

    // Dimensions
    stream.write_u16::<LittleEndian>(h as u16).unwrap();
    stream.write_u16::<LittleEndian>(w as u16).unwrap();
    stream.write_u16::<LittleEndian>(hc as u16).unwrap();
    stream.write_u16::<LittleEndian>(wc as u16).unwrap();

    // Centroid counts
    stream.write_u16::<LittleEndian>(n_y as u16).unwrap();
    stream.push(n_cb as u8);
    stream.push(n_cr as u8);

    // Residual info
    stream.push(residu_bits);
    stream.write_f32::<LittleEndian>(res_max_abs).unwrap();

    // Noise sigma (v5)
    stream.write_f32::<LittleEndian>(noise_sigma).unwrap();

    // Centroids (uint8, sorted)
    for &c in centroids_y {
        stream.push(c.round().clamp(0.0, 255.0) as u8);
    }
    for &c in centroids_cb {
        stream.push(c.round().clamp(0.0, 255.0) as u8);
    }
    for &c in centroids_cr {
        stream.push(c.round().clamp(0.0, 255.0) as u8);
    }

    // Paeth residuals
    stream.extend(pack_pred_residuals(pred_y));
    stream.extend(pack_pred_residuals(pred_cb));
    stream.extend(pack_pred_residuals(pred_cr));

    // Pixel residuals Y
    if let Some(res) = y_pixel_res {
        for &r in res {
            stream.push(r as u8);
        }
    }

    stream
}

/// Compress with LZMA and add the XTS header.
pub fn write_xts(stream_data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut encoder = xz2::write::XzEncoder::new(Vec::new(), 9);
    encoder.write_all(stream_data)?;
    let compressed = encoder.finish()?;

    let mut xts = Vec::with_capacity(4 + compressed.len());
    xts.extend_from_slice(XTS_MAGIC);
    xts.extend(compressed);

    Ok(xts)
}

// ======================================================================
// V6: Hybrid wavelet CDF 9/7 + VQ/Fibonacci pipeline
// ======================================================================

/// V6-specific header.
#[derive(Debug, Clone)]
pub struct X267V6Header {
    pub h: usize,
    pub w: usize,
    pub hc: usize,
    pub wc: usize,
    pub n_y: usize,
    pub n_cb: usize,
    pub n_cr: usize,
    pub noise_sigma: f32,
    pub wv_levels: usize,
    pub ll_ranges: [(f32, f32); 3], // (min, max) for Y, Cb, Cr
    pub y_sizes: Vec<(usize, usize)>,
    pub cb_sizes: Vec<(usize, usize)>,
    pub cr_sizes: Vec<(usize, usize)>, // if different from cb_sizes (AUREA v2 GCT)
    pub ll_y_size: (usize, usize),
    pub ll_c_size: (usize, usize),
    pub ll_cr_size: (usize, usize), // if different from ll_c_size (AUREA v2 GCT)
    pub steps_y: Vec<[f64; 3]>,  // per level: [LH, HL, HH]
    pub steps_c: Vec<[f64; 3]>,
    pub steps_cr: Vec<[f64; 3]>, // if different from steps_c (AUREA v2 GCT)
    pub flags: u8,
}

/// Decoded v6 data.
#[derive(Debug)]
pub struct X267V6Stream {
    pub header: X267V6Header,
    pub centroids_y: Vec<f64>,
    pub centroids_cb: Vec<f64>,
    pub centroids_cr: Vec<f64>,
    pub pred_y: Vec<i16>,
    pub pred_cb: Vec<i16>,
    pub pred_cr: Vec<i16>,
    pub detail_data: Vec<u8>,
    pub detail_offset: usize,
    /// If true, each detail band is preceded by a u32 LE (size in bytes).
    /// Used by echolot (.echo), not by x267 v6.
    pub band_size_prefixed: bool,
    /// MERA post-disentangler angles (echolot v3+).
    /// Per channel [Y, Cb, Cr]: Vec<[f64; 6]> of length wv_levels.
    /// 6 angles per level = [LL-LH, LL-HL, LL-HH, LH-HL, LH-HH, HL-HH]
    pub mera_angles: Option<[Vec<[f64; 6]>; 3]>,
    /// If true, detail bands use the AUREA encoding
    /// (Fibonacci quantization + golden spiral scan + Zeckendorf coding).
    pub aurea_bands: bool,
    /// AUREA v2: RGB mode (no YCbCr) with cross-channel prediction.
    /// G is the primary channel, R and B are encoded as residuals of G.
    pub aurea_v2: bool,
    /// Cross-channel prediction coefficients (AUREA v2).
    /// pred_coeffs[lv * 3 + bi] = [alpha_R, alpha_B] for level lv, band bi.
    pub pred_coeffs: Option<Vec<[f32; 2]>>,
    /// AUREA v6/v7: geometric encoding.
    pub geometric: bool,
    /// AUREA v7: residual bands use rANS per-stream.
    pub rans_bands: bool,
}

/// Parse a v6 x267 stream.
fn parse_x267_v6(_stream: &[u8]) -> io::Result<X267Stream> {
    // Return an empty X267Stream with version=6
    // The real v6 decoding is done in lib.rs via parse_xts_v6()
    // This is just for interface compatibility

    let header = X267Header {
        version: 6,
        h: 0, w: 0, hc: 0, wc: 0,
        n_y: 0, n_cb: 0, n_cr: 0,
        alpha_bits: 0, residu_bits: 0, res_max_abs: 0.0,
        noise_sigma: 0.0,
    };

    Ok(X267Stream {
        header,
        centroids_y: vec![],
        centroids_cb: vec![],
        centroids_cr: vec![],
        pred_y: vec![],
        pred_cb: vec![],
        pred_cr: vec![],
        y_pixel_res: None,
    })
}

/// Full parse of the v6 stream with all data.
pub fn parse_x267_v6_full(stream: &[u8]) -> io::Result<X267V6Stream> {
    let mut c = Cursor::new(stream);

    // Magic + version
    let mut magic = [0u8; 4];
    c.read_exact(&mut magic)?;
    let _version = c.read_u8()?; // == 6

    // Dimensions
    let h = c.read_u16::<LittleEndian>()? as usize;
    let w = c.read_u16::<LittleEndian>()? as usize;
    let hc = c.read_u16::<LittleEndian>()? as usize;
    let wc = c.read_u16::<LittleEndian>()? as usize;

    let n_y = c.read_u16::<LittleEndian>()? as usize;
    let n_cb = c.read_u8()? as usize;
    let n_cr = c.read_u8()? as usize;

    let _residu_bits = c.read_u8()?;
    let _res_max_abs = c.read_f32::<LittleEndian>()?;
    let noise_sigma = c.read_f32::<LittleEndian>()?;
    let wv_levels = c.read_u8()? as usize;
    let flags = c.read_u8()?;

    // LL ranges
    let y_min = c.read_f32::<LittleEndian>()?;
    let y_max = c.read_f32::<LittleEndian>()?;
    let cb_min = c.read_f32::<LittleEndian>()?;
    let cb_max = c.read_f32::<LittleEndian>()?;
    let cr_min = c.read_f32::<LittleEndian>()?;
    let cr_max = c.read_f32::<LittleEndian>()?;
    let ll_ranges = [(y_min, y_max), (cb_min, cb_max), (cr_min, cr_max)];

    // Y subband sizes
    let mut y_sizes = Vec::with_capacity(wv_levels);
    for _ in 0..wv_levels {
        let sh = c.read_u16::<LittleEndian>()? as usize;
        let sw = c.read_u16::<LittleEndian>()? as usize;
        y_sizes.push((sh, sw));
    }
    let ll_yh = c.read_u16::<LittleEndian>()? as usize;
    let ll_yw = c.read_u16::<LittleEndian>()? as usize;

    // Cb subband sizes
    let mut cb_sizes = Vec::with_capacity(wv_levels);
    for _ in 0..wv_levels {
        let sh = c.read_u16::<LittleEndian>()? as usize;
        let sw = c.read_u16::<LittleEndian>()? as usize;
        cb_sizes.push((sh, sw));
    }
    let ll_ch = c.read_u16::<LittleEndian>()? as usize;
    let ll_cw = c.read_u16::<LittleEndian>()? as usize;

    // Steps per level per band
    let mut steps_y = Vec::with_capacity(wv_levels);
    let mut steps_c = Vec::with_capacity(wv_levels);
    for _ in 0..wv_levels {
        let mut sy = [0.0f64; 3];
        let mut sc = [0.0f64; 3];
        for bi in 0..3 {
            sy[bi] = c.read_f32::<LittleEndian>()? as f64;
            sc[bi] = c.read_f32::<LittleEndian>()? as f64;
        }
        steps_y.push(sy);
        steps_c.push(sc);
    }

    // ll_size, det_size
    let ll_size = c.read_u32::<LittleEndian>()? as usize;
    let _det_size = c.read_u32::<LittleEndian>()? as usize;

    let data_start = c.position() as usize;

    // Parse LL data
    let ll_data = &stream[data_start..data_start + ll_size];
    let mut lc = Cursor::new(ll_data);

    // Centroids
    let mut centroids_y = Vec::with_capacity(n_y);
    for _ in 0..n_y { centroids_y.push(lc.read_u8()? as f64); }
    let mut centroids_cb = Vec::with_capacity(n_cb);
    for _ in 0..n_cb { centroids_cb.push(lc.read_u8()? as f64); }
    let mut centroids_cr = Vec::with_capacity(n_cr);
    for _ in 0..n_cr { centroids_cr.push(lc.read_u8()? as f64); }

    // Paeth residuals
    let ll_pos = lc.position() as usize;
    let ll_rem = &ll_data[ll_pos..];
    let mut lc2 = Cursor::new(ll_rem);
    let pred_y = unpack_pred_residuals(&mut lc2, ll_yh * ll_yw)?;
    let pred_cb = unpack_pred_residuals(&mut lc2, ll_ch * ll_cw)?;
    let pred_cr = unpack_pred_residuals(&mut lc2, ll_ch * ll_cw)?;

    // Detail data starts after LL
    let det_start = data_start + ll_size;
    let detail_data = stream[det_start..].to_vec();

    let header = X267V6Header {
        h, w, hc, wc, n_y, n_cb, n_cr,
        noise_sigma, wv_levels, ll_ranges,
        cr_sizes: cb_sizes.clone(),
        y_sizes, cb_sizes,
        ll_y_size: (ll_yh, ll_yw),
        ll_c_size: (ll_ch, ll_cw),
        ll_cr_size: (ll_ch, ll_cw),
        steps_cr: steps_c.clone(),
        steps_y, steps_c,
        flags,
    };

    Ok(X267V6Stream {
        header,
        centroids_y, centroids_cb, centroids_cr,
        pred_y, pred_cb, pred_cr,
        detail_data,
        detail_offset: 0,
        band_size_prefixed: false,
        mera_angles: None,
        aurea_bands: false,
        aurea_v2: false,
        pred_coeffs: None,
        geometric: false,
        rans_bands: false,
    })
}

/// Write the v6 stream (uncompressed).
#[allow(clippy::too_many_arguments)]
pub fn write_x267_v6_stream(
    h: usize, w: usize, hc: usize, wc: usize,
    n_y: usize, n_cb: usize, n_cr: usize,
    noise_sigma: f32, wv_levels: usize, flags: u8,
    ll_ranges: &[(f32, f32); 3],
    y_sizes: &[(usize, usize)],
    cb_sizes: &[(usize, usize)],
    ll_y_size: (usize, usize), ll_c_size: (usize, usize),
    steps_y: &[[f64; 3]], steps_c: &[[f64; 3]],
    ll_stream: &[u8], det_stream: &[u8],
) -> Vec<u8> {
    let mut s: Vec<u8> = Vec::new();

    // Magic + version
    s.extend_from_slice(b"x267");
    s.push(6);

    // Dimensions
    s.write_u16::<LittleEndian>(h as u16).unwrap();
    s.write_u16::<LittleEndian>(w as u16).unwrap();
    s.write_u16::<LittleEndian>(hc as u16).unwrap();
    s.write_u16::<LittleEndian>(wc as u16).unwrap();

    // Centroid counts
    s.write_u16::<LittleEndian>(n_y as u16).unwrap();
    s.push(n_cb as u8);
    s.push(n_cr as u8);

    // Unused v6 fields (residu_bits=0, res_max_abs=0.0)
    s.push(0);
    s.write_f32::<LittleEndian>(0.0).unwrap();

    // Noise sigma
    s.write_f32::<LittleEndian>(noise_sigma).unwrap();

    // wv_levels + flags
    s.push(wv_levels as u8);
    s.push(flags);

    // LL ranges
    for &(mn, mx) in ll_ranges {
        s.write_f32::<LittleEndian>(mn).unwrap();
        s.write_f32::<LittleEndian>(mx).unwrap();
    }

    // Y sizes + LL Y size
    for &(sh, sw) in y_sizes {
        s.write_u16::<LittleEndian>(sh as u16).unwrap();
        s.write_u16::<LittleEndian>(sw as u16).unwrap();
    }
    s.write_u16::<LittleEndian>(ll_y_size.0 as u16).unwrap();
    s.write_u16::<LittleEndian>(ll_y_size.1 as u16).unwrap();

    // Cb sizes + LL C size
    for &(sh, sw) in cb_sizes {
        s.write_u16::<LittleEndian>(sh as u16).unwrap();
        s.write_u16::<LittleEndian>(sw as u16).unwrap();
    }
    s.write_u16::<LittleEndian>(ll_c_size.0 as u16).unwrap();
    s.write_u16::<LittleEndian>(ll_c_size.1 as u16).unwrap();

    // Steps
    for lv in 0..wv_levels {
        for bi in 0..3 {
            s.write_f32::<LittleEndian>(steps_y[lv][bi] as f32).unwrap();
            s.write_f32::<LittleEndian>(steps_c[lv][bi] as f32).unwrap();
        }
    }

    // LL + det sizes
    s.write_u32::<LittleEndian>(ll_stream.len() as u32).unwrap();
    s.write_u32::<LittleEndian>(det_stream.len() as u32).unwrap();

    // Data
    s.extend_from_slice(ll_stream);
    s.extend_from_slice(det_stream);

    s
}
