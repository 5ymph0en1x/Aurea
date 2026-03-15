/// YCbCr <-> RGB conversion (ITU-R BT.601) and 4:2:0 subsampling.

/// Convert a YCbCr pixel to RGB, returns (R, G, B) clipped to [0, 255].
#[inline]
pub fn ycbcr_to_rgb_pixel(y: f64, cb: f64, cr: f64) -> (u8, u8, u8) {
    let cb = cb - 128.0;
    let cr = cr - 128.0;
    let r = (y + 1.402 * cr).clamp(0.0, 255.0) as u8;
    let g = (y - 0.344136 * cb - 0.714136 * cr).clamp(0.0, 255.0) as u8;
    let b = (y + 1.772 * cb).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

/// Convert a YCbCr image (H, W, 3) to RGB (H, W, 3).
/// The Y, Cb, Cr planes are passed separately as f64.
pub fn ycbcr_to_rgb(y_plane: &[f64], cb_plane: &[f64], cr_plane: &[f64], len: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(len * 3);
    for i in 0..len {
        let (r, g, b) = ycbcr_to_rgb_pixel(y_plane[i], cb_plane[i], cr_plane[i]);
        rgb.push(r);
        rgb.push(g);
        rgb.push(b);
    }
    rgb
}

/// Bilinear upsampling of a chroma plane (Hc, Wc) to (H, W).
/// Reproduces exactly scipy.ndimage.zoom(channel, (fy, fx), order=1).
/// scipy mapping: in_coord = out_coord * (in_size - 1) / (out_size - 1)
pub fn upsample_420(channel: &[f64], hc: usize, wc: usize, target_h: usize, target_w: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; target_h * target_w];

    let scale_y = if target_h > 1 { (hc as f64 - 1.0) / (target_h as f64 - 1.0) } else { 0.0 };
    let scale_x = if target_w > 1 { (wc as f64 - 1.0) / (target_w as f64 - 1.0) } else { 0.0 };

    for y in 0..target_h {
        let sy = y as f64 * scale_y;
        let iy0 = (sy as usize).min(hc - 1);
        let iy1 = (iy0 + 1).min(hc - 1);
        let dy = sy - iy0 as f64;

        for x in 0..target_w {
            let sx = x as f64 * scale_x;
            let ix0 = (sx as usize).min(wc - 1);
            let ix1 = (ix0 + 1).min(wc - 1);
            let dx = sx - ix0 as f64;

            let v00 = channel[iy0 * wc + ix0];
            let v01 = channel[iy0 * wc + ix1];
            let v10 = channel[iy1 * wc + ix0];
            let v11 = channel[iy1 * wc + ix1];

            let v = v00 * (1.0 - dy) * (1.0 - dx)
                  + v01 * (1.0 - dy) * dx
                  + v10 * dy * (1.0 - dx)
                  + v11 * dy * dx;

            result[y * target_w + x] = v;
        }
    }

    result
}

/// RGB -> YCbCr BT.601 conversion. Returns 3 planes (Y, Cb, Cr) as f64.
pub fn rgb_to_ycbcr_planes(rgb: &[u8], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut y = Vec::with_capacity(n);
    let mut cb = Vec::with_capacity(n);
    let mut cr = Vec::with_capacity(n);

    for i in 0..n {
        let r = rgb[i * 3] as f64;
        let g = rgb[i * 3 + 1] as f64;
        let b = rgb[i * 3 + 2] as f64;

        y.push((0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 255.0));
        cb.push((-0.169 * r - 0.331 * g + 0.500 * b + 128.0).clamp(0.0, 255.0));
        cr.push((0.500 * r - 0.419 * g - 0.081 * b + 128.0).clamp(0.0, 255.0));
    }

    (y, cb, cr)
}

/// RGB (separate f64 planes) -> YCbCr BT.601 conversion. Returns 3 planes (Y, Cb, Cr).
pub fn rgb_to_ycbcr_from_f64(r: &[f64], g: &[f64], b: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut y = Vec::with_capacity(n);
    let mut cb = Vec::with_capacity(n);
    let mut cr = Vec::with_capacity(n);

    for i in 0..n {
        y.push((0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]).clamp(0.0, 255.0));
        cb.push((-0.169 * r[i] - 0.331 * g[i] + 0.500 * b[i] + 128.0).clamp(0.0, 255.0));
        cr.push((0.500 * r[i] - 0.419 * g[i] - 0.081 * b[i] + 128.0).clamp(0.0, 255.0));
    }

    (y, cb, cr)
}

/// 4:2:0 subsampling by averaging 2x2 blocks.
/// Returns (channel_sub, hc, wc).
pub fn subsample_420_encode(channel: &[f64], h: usize, w: usize) -> (Vec<f64>, usize, usize) {
    let hc = (h + 1) / 2;
    let wc = (w + 1) / 2;
    let mut result = vec![0.0f64; hc * wc];

    for cy in 0..hc {
        for cx in 0..wc {
            let y0 = cy * 2;
            let y1 = (y0 + 1).min(h - 1);
            let x0 = cx * 2;
            let x1 = (x0 + 1).min(w - 1);

            let v00 = channel[y0 * w + x0];
            let v01 = channel[y0 * w + x1];
            let v10 = channel[y1 * w + x0];
            let v11 = channel[y1 * w + x1];

            result[cy * wc + cx] = (v00 + v01 + v10 + v11) / 4.0;
        }
    }

    (result, hc, wc)
}

/// Golden Color Transform (GCT) -- golden rotation of the color space.
///
/// Golden luminance: L_phi = (R + phi*G + phi^-1*B) / (2*phi)
///   Weights: R=0.309, G=0.500, B=0.191 (close to BT.601, naturally!)
/// Chromas: C1 = B - L_phi, C2 = R - L_phi (centered on zero for gray)
///
/// Inverse: R = L + C2, G = L - phi^-2*C1 - phi^-1*C2, B = L + C1
///
/// Properties: luma/chroma decorrelation, gray -> (L, 0, 0),
/// the inverse uses only phi^-1 and phi^-2 (Fibonacci sequence).
pub fn golden_rotate_forward(rgb: &[u8], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use crate::golden::{PHI, PHI_INV};
    let norm = 1.0 / (2.0 * PHI); // 1/(2phi) so the weights sum to 1

    let mut l = Vec::with_capacity(n);
    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);

    for i in 0..n {
        let r = rgb[i * 3] as f64;
        let g = rgb[i * 3 + 1] as f64;
        let b = rgb[i * 3 + 2] as f64;

        let lum = (r + PHI * g + PHI_INV * b) * norm;
        l.push(lum);
        c1.push(b - lum);
        c2.push(r - lum);
    }

    (l, c1, c2)
}

/// Inverse of GCT: (L_phi, C1, C2) -> (R, G, B) as f64.
pub fn golden_rotate_inverse(l: &[f64], c1: &[f64], c2: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    use crate::golden::{PHI_INV, PHI_INV2};

    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);

    for i in 0..n {
        r.push(l[i] + c2[i]);
        g.push(l[i] - PHI_INV2 * c1[i] - PHI_INV * c2[i]);
        b.push(l[i] + c1[i]);
    }

    (r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ycbcr_white() {
        // Pure white: Y=255, Cb=128, Cr=128 -> R=255, G=255, B=255
        let (r, g, b) = ycbcr_to_rgb_pixel(255.0, 128.0, 128.0);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn test_ycbcr_black() {
        let (r, g, b) = ycbcr_to_rgb_pixel(0.0, 128.0, 128.0);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn test_gct_roundtrip() {
        // GCT forward + inverse should give original values
        let rgb: Vec<u8> = vec![200, 100, 50, 0, 0, 0, 255, 255, 255, 128, 128, 128];
        let n = 4;
        let (l, c1, c2) = golden_rotate_forward(&rgb, n);

        // Gray (128,128,128) -> C1=0, C2=0
        assert!((c1[3]).abs() < 1e-10, "gray C1 should be 0, got {}", c1[3]);
        assert!((c2[3]).abs() < 1e-10, "gray C2 should be 0, got {}", c2[3]);

        // Inverse
        let (r, g, b) = golden_rotate_inverse(&l, &c1, &c2, n);
        for i in 0..n {
            let ro = rgb[i * 3] as f64;
            let go = rgb[i * 3 + 1] as f64;
            let bo = rgb[i * 3 + 2] as f64;
            assert!((r[i] - ro).abs() < 1e-10, "R mismatch at {}: {} vs {}", i, r[i], ro);
            assert!((g[i] - go).abs() < 1e-10, "G mismatch at {}: {} vs {}", i, g[i], go);
            assert!((b[i] - bo).abs() < 1e-10, "B mismatch at {}: {} vs {}", i, b[i], bo);
        }
    }

    #[test]
    fn test_upsample_identity() {
        // 2x2 -> 4x4
        let ch = vec![10.0, 20.0, 30.0, 40.0];
        let up = upsample_420(&ch, 2, 2, 4, 4);
        assert_eq!(up.len(), 16);
        // Corners should be close to source values
        assert!((up[0] - 10.0).abs() < 1.0);
    }
}
