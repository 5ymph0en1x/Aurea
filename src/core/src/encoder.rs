/// x267 v6+ encoder: RGB -> XTS.
/// Hybrid wavelet CDF 9/7 + VQ/Fibonacci pipeline.
/// v6+ improvements: zigzag, adaptive quant, inter-scale prediction, chroma factor 2.0.

use ndarray::Array2;

use crate::bitstream;
use crate::color;
use crate::dsp;
use crate::paeth;
use crate::vq;
use crate::wavelet;

/// v6+ encoding parameters.
pub struct EncoderParams {
    pub n_representatives: usize,
    pub quality: u8,
    pub wv_levels: usize, // 0 = auto
    pub sharpen_strength: f64,
}

/// Encoding result.
pub struct EncoderResult {
    pub xts_data: Vec<u8>,
    pub compressed_size: usize,
    pub noise_sigma: f64,
}

/// Encode an RGB image as an XTS file (v6+).
pub fn encode_xts(
    rgb: &[u8],
    width: usize,
    height: usize,
    params: &EncoderParams,
) -> Result<EncoderResult, Box<dyn std::error::Error>> {
    let n = height * width;

    // 0. Decompose RGB into f64 planes
    let mut r_ch: Vec<f64> = (0..n).map(|i| rgb[i * 3] as f64).collect();
    let mut g_ch: Vec<f64> = (0..n).map(|i| rgb[i * 3 + 1] as f64).collect();
    let mut b_ch: Vec<f64> = (0..n).map(|i| rgb[i * 3 + 2] as f64).collect();

    // 1. Sensor noise estimation
    let noise_sigma = dsp::estimate_noise_sigma(&r_ch, &g_ch, &b_ch, height, width);

    // 2. FFT Wiener denoising (if sigma >= 5.0)
    dsp::denoise_fft_plane(&mut r_ch, height, width, noise_sigma);
    dsp::denoise_fft_plane(&mut g_ch, height, width, noise_sigma);
    dsp::denoise_fft_plane(&mut b_ch, height, width, noise_sigma);

    // 3. RGB -> YCbCr
    let (mut y_ch, cb_ch, cr_ch) = color::rgb_to_ycbcr_from_f64(&r_ch, &g_ch, &b_ch, n);

    // 4. Auto detail_step (BEFORE sharpen, as in Python)
    let y_pre_sharpen = Array2::from_shape_vec((height, width), y_ch.clone())?;
    let detail_step = wavelet::auto_detail_step(&y_pre_sharpen, params.quality);

    // 5. Directional H/V sharpening on Y
    dsp::directional_sharpen(&mut y_ch, height, width, params.sharpen_strength);

    // 6. 4:2:0 subsampling chroma
    let (cb_sub, hc, wc) = color::subsample_420_encode(&cb_ch, height, width);
    let (cr_sub, _, _) = color::subsample_420_encode(&cr_ch, height, width);

    // 7. Auto wv_levels if not specified
    let wv_levels = if params.wv_levels == 0 {
        wavelet::auto_wv_levels(height, width)
    } else {
        params.wv_levels
    };

    // 8. CDF 9/7 wavelet decomposition
    let y_arr = Array2::from_shape_vec((height, width), y_ch.clone())?;
    let cb_arr = Array2::from_shape_vec((hc, wc), cb_sub.clone())?;
    let cr_arr = Array2::from_shape_vec((hc, wc), cr_sub.clone())?;

    let (y_ll, y_subs, y_sizes) = wavelet::wavelet_decompose(&y_arr, wv_levels);
    let (cb_ll, cb_subs, cb_sizes) = wavelet::wavelet_decompose(&cb_arr, wv_levels);
    let (cr_ll, cr_subs, _cr_sizes) = wavelet::wavelet_decompose(&cr_arr, wv_levels);

    let ll_yh = y_ll.nrows();
    let ll_yw = y_ll.ncols();
    let ll_ch = cb_ll.nrows();
    let ll_cw = cb_ll.ncols();

    // 9. Normalize LL to [0, 255] for VQ + fibonacci
    let y_ll_flat: Vec<f64> = y_ll.iter().copied().collect();
    let cb_ll_flat: Vec<f64> = cb_ll.iter().copied().collect();
    let cr_ll_flat: Vec<f64> = cr_ll.iter().copied().collect();

    let (y_ll_norm, y_ll_min, y_ll_max) = crate::normalize_ll(&y_ll_flat);
    let (cb_ll_norm, cb_ll_min, cb_ll_max) = crate::normalize_ll(&cb_ll_flat);
    let (cr_ll_norm, cr_ll_min, cr_ll_max) = crate::normalize_ll(&cr_ll_flat);

    // 10. VQ quantization + refinement
    let n_y = params.n_representatives;
    let n_c = 16usize.max(n_y * 3 / 4);

    let centroids_y = vq::kmeans_1d(&y_ll_norm, n_y, 10);
    let centroids_cb = vq::kmeans_1d(&cb_ll_norm, n_c, 10);
    let centroids_cr = vq::kmeans_1d(&cr_ll_norm, n_c, 10);

    let centroids_y = vq::refine_centroids(&y_ll_norm, &centroids_y, ll_yh, ll_yw, 5);
    let centroids_cb = vq::refine_centroids(&cb_ll_norm, &centroids_cb, ll_ch, ll_cw, 5);
    let centroids_cr = vq::refine_centroids(&cr_ll_norm, &centroids_cr, ll_ch, ll_cw, 5);

    // Quantize centroids to uint8
    let cy_u8: Vec<f64> = centroids_y.iter().map(|&c| c.round().clamp(0.0, 255.0)).collect();
    let ccb_u8: Vec<f64> = centroids_cb.iter().map(|&c| c.round().clamp(0.0, 255.0)).collect();
    let ccr_u8: Vec<f64> = centroids_cr.iter().map(|&c| c.round().clamp(0.0, 255.0)).collect();

    // 11. Label assignment + Paeth prediction
    let labels_y = vq::assign_nearest(&y_ll_norm, &cy_u8);
    let labels_cb = vq::assign_nearest(&cb_ll_norm, &ccb_u8);
    let labels_cr = vq::assign_nearest(&cr_ll_norm, &ccr_u8);

    let pred_y = paeth::paeth_predict_2d(&labels_y, ll_yh, ll_yw);
    let pred_cb = paeth::paeth_predict_2d(&labels_cb, ll_ch, ll_cw);
    let pred_cr = paeth::paeth_predict_2d(&labels_cr, ll_ch, ll_cw);

    // 12. LL stream (centroids + Paeth residuals)
    let mut ll_stream: Vec<u8> = Vec::new();
    for &c in &cy_u8 { ll_stream.push(c as u8); }
    for &c in &ccb_u8 { ll_stream.push(c as u8); }
    for &c in &ccr_u8 { ll_stream.push(c as u8); }
    ll_stream.extend(bitstream::pack_pred_residuals(&pred_y));
    ll_stream.extend(bitstream::pack_pred_residuals(&pred_cb));
    ll_stream.extend(bitstream::pack_pred_residuals(&pred_cr));

    // 13. Encode detail bands (v2: zigzag + adaptive quant + inter-scale)
    let flags = wavelet::DEFAULT_FLAGS;
    let use_interscale = flags & wavelet::FLAG_INTERSCALE != 0;

    let mut det_stream: Vec<u8> = Vec::new();
    let mut steps_y_all = vec![[0.0f64; 3]; wv_levels];
    let mut steps_c_all = vec![[0.0f64; 3]; wv_levels];

    // Reconstructed bands (decoder view) for inter-scale prediction
    let mut recon_y: Vec<Option<Vec<Array2<f64>>>> = vec![None; wv_levels];
    let mut recon_cb: Vec<Option<Vec<Array2<f64>>>> = vec![None; wv_levels];
    let mut recon_cr: Vec<Option<Vec<Array2<f64>>>> = vec![None; wv_levels];

    // Encode deepest level first for inter-scale prediction
    for lv_idx in 0..wv_levels {
        let lv = wv_levels - 1 - lv_idx;

        let level_scale = if lv < wavelet::LEVEL_SCALES.len() {
            wavelet::LEVEL_SCALES[lv]
        } else {
            0.3
        };

        let (y_h, y_w) = y_sizes[lv];
        let (c_h, c_w) = cb_sizes[lv];
        let y_bsizes = wavelet::detail_band_sizes(y_h, y_w);
        let c_bsizes = wavelet::detail_band_sizes(c_h, c_w);

        let mut sy = [0.0f64; 3];
        let mut sc = [0.0f64; 3];
        let mut ry: Vec<Array2<f64>> = Vec::with_capacity(3);
        let mut rcb: Vec<Array2<f64>> = Vec::with_capacity(3);
        let mut rcr: Vec<Array2<f64>> = Vec::with_capacity(3);

        for bi in 0..3 {
            let step_y = detail_step * level_scale * wavelet::PERCEPTUAL_BAND_WEIGHTS[bi];
            let step_c = step_y * wavelet::CHROMA_DETAIL_FACTOR;
            sy[bi] = step_y;
            sc[bi] = step_c;

            let orig_y = y_subs[lv].band(bi);
            let orig_cb = cb_subs[lv].band(bi);
            let orig_cr = cr_subs[lv].band(bi);

            // Inter-scale prediction: subtract prediction from deeper level
            let (to_enc_y, pred_y_opt) = interscale_predict(
                orig_y, &recon_y[..], lv, bi, y_bsizes[bi], use_interscale, wv_levels,
            );
            let (to_enc_cb, pred_cb_opt) = interscale_predict(
                orig_cb, &recon_cb[..], lv, bi, c_bsizes[bi], use_interscale, wv_levels,
            );
            let (to_enc_cr, pred_cr_opt) = interscale_predict(
                orig_cr, &recon_cr[..], lv, bi, c_bsizes[bi], use_interscale, wv_levels,
            );

            // Encode with v2 (zigzag + adaptive quant)
            let (data_y, deq_y) = wavelet::encode_detail_band_v2(&to_enc_y, step_y, flags);
            det_stream.extend_from_slice(&data_y);

            let (data_cb, deq_cb) = wavelet::encode_detail_band_v2(&to_enc_cb, step_c, flags);
            det_stream.extend_from_slice(&data_cb);

            let (data_cr, deq_cr) = wavelet::encode_detail_band_v2(&to_enc_cr, step_c, flags);
            det_stream.extend_from_slice(&data_cr);

            // Reconstruct (decoder view) for prediction of shallower levels
            ry.push(interscale_reconstruct(deq_y, pred_y_opt));
            rcb.push(interscale_reconstruct(deq_cb, pred_cb_opt));
            rcr.push(interscale_reconstruct(deq_cr, pred_cr_opt));
        }

        steps_y_all[lv] = sy;
        steps_c_all[lv] = sc;
        recon_y[lv] = Some(ry);
        recon_cb[lv] = Some(rcb);
        recon_cr[lv] = Some(rcr);
    }

    // 14. v6 bitstream
    let ll_ranges = [
        (y_ll_min as f32, y_ll_max as f32),
        (cb_ll_min as f32, cb_ll_max as f32),
        (cr_ll_min as f32, cr_ll_max as f32),
    ];

    let stream_data = bitstream::write_x267_v6_stream(
        height, width, hc, wc,
        n_y, n_c, n_c,
        noise_sigma as f32, wv_levels, flags,
        &ll_ranges,
        &y_sizes, &cb_sizes,
        (ll_yh, ll_yw), (ll_ch, ll_cw),
        &steps_y_all, &steps_c_all,
        &ll_stream, &det_stream,
    );

    // 15. LZMA compression + XTS wrapper
    let xts_data = bitstream::write_xts(&stream_data)?;
    let compressed_size = xts_data.len();

    Ok(EncoderResult {
        xts_data,
        compressed_size,
        noise_sigma,
    })
}

/// Compute inter-scale prediction and return (band_to_encode, prediction_opt).
fn interscale_predict(
    orig: &Array2<f64>,
    recon: &[Option<Vec<Array2<f64>>>],
    lv: usize, bi: usize,
    target_size: (usize, usize),
    use_interscale: bool,
    wv_levels: usize,
) -> (Array2<f64>, Option<Array2<f64>>) {
    if use_interscale && lv + 1 < wv_levels {
        if let Some(ref deeper) = recon[lv + 1] {
            let (bh, bw) = target_size;
            let pred = wavelet::upsample_band(&deeper[bi], bh, bw);
            return (orig - &pred, Some(pred));
        }
    }
    (orig.clone(), None)
}

/// Reconstruct the final band (dequantized + prediction if applicable).
fn interscale_reconstruct(deq: Array2<f64>, pred_opt: Option<Array2<f64>>) -> Array2<f64> {
    match pred_opt {
        Some(pred) => &deq + &pred,
        None => deq,
    }
}

/// Helper trait for accessing bands from a tuple.
trait SubbandAccess {
    fn band(&self, idx: usize) -> &Array2<f64>;
}

impl SubbandAccess for (Array2<f64>, Array2<f64>, Array2<f64>) {
    fn band(&self, idx: usize) -> &Array2<f64> {
        match idx {
            0 => &self.0,
            1 => &self.1,
            2 => &self.2,
            _ => panic!("band index out of range"),
        }
    }
}
