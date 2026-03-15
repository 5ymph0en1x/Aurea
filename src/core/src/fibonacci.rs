/// Fibonacci correction -- de-banding + de-ringing (zero bit cost).
///
/// Reconstruction by quantum spin diffusion:
/// - Scan-line interpolation V and H
/// - Heisenberg smoothing (boundary jitter averaging)
/// - Quantum superposition (oscillation zones)
/// - 2D Fibonacci dithering (quantum vacuum noise)
/// - Quantum confinement (overshoot capping)

use rayon::prelude::*;

const PHI: f64 = 1.618033988749895; // (1 + sqrt(5)) / 2
const PHI_SQ: f64 = PHI * PHI;

// ======================================================================
// Distance transform EDT (Meijster, separable 2D)
// ======================================================================

/// Exact EDT distance transform, row-major friendly.
/// mask: true = object (distance=0). Returns euclidean distance of non-objects.
pub(crate) fn distance_transform_edt(mask: &[bool], h: usize, w: usize) -> Vec<f64> {
    let inf = (h + w) as f64 + 1.0;

    // Phase 1: vertical distance to nearest object, per ROW (row-major)
    // We store g[y][x] = minimum vertical distance
    let mut g = vec![inf; h * w];

    // Forward (top -> bottom)
    for x in 0..w {
        g[x] = if mask[x] { 0.0 } else { inf };
    }
    for y in 1..h {
        let row = y * w;
        let prev_row = (y - 1) * w;
        for x in 0..w {
            g[row + x] = if mask[row + x] { 0.0 } else { g[prev_row + x] + 1.0 };
        }
    }
    // Backward (bottom -> top)
    for y in (0..h.saturating_sub(1)).rev() {
        let row = y * w;
        let next_row = (y + 1) * w;
        for x in 0..w {
            let below = g[next_row + x] + 1.0;
            if below < g[row + x] {
                g[row + x] = below;
            }
        }
    }

    // Phase 2: 2D EDT per row via lower envelope of parabolas (parallel)
    let result: Vec<f64> = (0..h).into_par_iter().flat_map(|y| {
        let row = y * w;
        let mut s_buf = vec![0usize; w];
        let mut t_buf = vec![0.0f64; w + 1];
        let mut row_result = vec![0.0f64; w];

        let mut q: isize = -1;
        for u in 0..w {
            let gu = g[row + u];
            let fu = gu * gu;
            let u_f = u as f64;
            while q >= 0 {
                let su = s_buf[q as usize];
                let su_f = su as f64;
                let gsu = g[row + su];
                let fsu = gsu * gsu;
                let sep = (u_f * u_f - su_f * su_f + fu - fsu) / (2.0 * (u_f - su_f));
                if sep <= t_buf[q as usize] {
                    q -= 1;
                } else {
                    break;
                }
            }
            q += 1;
            s_buf[q as usize] = u;
            t_buf[q as usize] = if q == 0 {
                f64::NEG_INFINITY
            } else {
                let su = s_buf[(q - 1) as usize];
                let su_f = su as f64;
                let gsu = g[row + su];
                let fsu = gsu * gsu;
                (u_f * u_f - su_f * su_f + fu - fsu) / (2.0 * (u_f - su_f))
            };
        }
        t_buf[(q + 1) as usize] = f64::INFINITY;

        let mut j = 0usize;
        for u in 0..w {
            while t_buf[j + 1] < u as f64 {
                j += 1;
            }
            let su = s_buf[j];
            let d = u as f64 - su as f64;
            let gsu = g[row + su];
            row_result[u] = (d * d + gsu * gsu).sqrt();
        }
        row_result
    }).collect();

    result
}

// ======================================================================
// Optimized filters
// ======================================================================

/// 1D horizontal box filter by prefix sum, parallelized per row.
fn uniform_filter1d_h(data: &[f64], h: usize, w: usize, size: usize) -> Vec<f64> {
    let half = (size / 2) as isize;

    let out: Vec<f64> = (0..h).into_par_iter().flat_map(|y| {
        let row = y * w;
        let mut prefix = Vec::with_capacity(w + 1);
        prefix.push(0.0);
        let mut s = 0.0f64;
        for x in 0..w {
            s += data[row + x];
            prefix.push(s);
        }
        let mut row_out = vec![0.0f64; w];
        for x in 0..w {
            let left = (x as isize - half).max(0) as usize;
            let right = ((x as isize + half) as usize).min(w - 1);
            let n = right - left + 1;
            row_out[x] = (prefix[right + 1] - prefix[left]) / n as f64;
        }
        row_out
    }).collect();
    out
}

/// 1D vertical box filter by prefix sum, parallelized per column block.
fn uniform_filter1d_v(data: &[f64], h: usize, w: usize, size: usize) -> Vec<f64> {
    let half = (size / 2) as isize;
    const BLOCK: usize = 64;

    let n_blocks = (w + BLOCK - 1) / BLOCK;

    // Each block produces its results in a temporary buffer
    let block_results: Vec<(usize, usize, Vec<f64>)> = (0..n_blocks).into_par_iter().map(|b| {
        let x_start = b * BLOCK;
        let x_end = (x_start + BLOCK).min(w);
        let bw = x_end - x_start;

        let mut col_buf = vec![0.0f64; bw * h];
        let mut prefix = vec![0.0f64; h + 1];

        // Copy the block transposed
        for y in 0..h {
            let src_row = y * w;
            for dx in 0..bw {
                col_buf[dx * h + y] = data[src_row + x_start + dx];
            }
        }

        // Prefix sum + filter
        for dx in 0..bw {
            let col_offset = dx * h;
            prefix[0] = 0.0;
            for y in 0..h {
                prefix[y + 1] = prefix[y] + col_buf[col_offset + y];
            }
            for y in 0..h {
                let top = (y as isize - half).max(0) as usize;
                let bot = ((y as isize + half) as usize).min(h - 1);
                let n = bot - top + 1;
                col_buf[col_offset + y] = (prefix[bot + 1] - prefix[top]) / n as f64;
            }
        }

        (x_start, bw, col_buf)
    }).collect();

    // Assemble into output
    let mut out = vec![0.0f64; h * w];
    for (x_start, bw, col_buf) in &block_results {
        for y in 0..h {
            let dst_row = y * w;
            for dx in 0..*bw {
                out[dst_row + x_start + dx] = col_buf[dx * h + y];
            }
        }
    }
    out
}

/// Separable 2D uniform filter.
fn uniform_filter_2d(data: &[f64], h: usize, w: usize, size: usize) -> Vec<f64> {
    let tmp = uniform_filter1d_h(data, h, w, size);
    uniform_filter1d_v(&tmp, h, w, size, )
}

/// Separable min/max filter (van Herk / Gil-Werman, O(N) per axis).
fn local_min_max(data: &[i32], h: usize, w: usize, size: usize) -> (Vec<i32>, Vec<i32>) {
    // Pass 1: horizontal
    let (h_mins, h_maxs) = minmax_1d_h(data, h, w, size);
    // Pass 2: vertical on horizontal result
    let (mins, maxs) = minmax_1d_v(&h_mins, &h_maxs, h, w, size);
    (mins, maxs)
}

fn minmax_1d_h(data: &[i32], h: usize, w: usize, size: usize) -> (Vec<i32>, Vec<i32>) {
    let half = size / 2;

    let rows: Vec<(Vec<i32>, Vec<i32>)> = (0..h).into_par_iter().map(|y| {
        let row = y * w;
        let mut row_mins = vec![0i32; w];
        let mut row_maxs = vec![0i32; w];
        for x in 0..w {
            let x0 = x.saturating_sub(half);
            let x1 = (x + half).min(w - 1);
            let mut lo = data[row + x0];
            let mut hi = lo;
            for xx in (x0 + 1)..=x1 {
                let v = data[row + xx];
                if v < lo { lo = v; }
                if v > hi { hi = v; }
            }
            row_mins[x] = lo;
            row_maxs[x] = hi;
        }
        (row_mins, row_maxs)
    }).collect();

    let mut mins = vec![0i32; h * w];
    let mut maxs = vec![0i32; h * w];
    for (y, (row_mins, row_maxs)) in rows.into_iter().enumerate() {
        let row = y * w;
        mins[row..row + w].copy_from_slice(&row_mins);
        maxs[row..row + w].copy_from_slice(&row_maxs);
    }
    (mins, maxs)
}

fn minmax_1d_v(h_mins: &[i32], h_maxs: &[i32], h: usize, w: usize, size: usize) -> (Vec<i32>, Vec<i32>) {
    let half = size / 2;
    const BLOCK: usize = 64;
    let n_blocks = (w + BLOCK - 1) / BLOCK;

    let block_results: Vec<(usize, usize, Vec<i32>, Vec<i32>)> = (0..n_blocks).into_par_iter().map(|b| {
        let x_start = b * BLOCK;
        let x_end = (x_start + BLOCK).min(w);
        let bw = x_end - x_start;

        let mut col_mins = vec![i32::MAX; (h + 1) * bw];
        let mut col_maxs = vec![i32::MIN; (h + 1) * bw];

        for y in 0..h {
            let src_row = y * w;
            let buf_row = (y + 1) * bw;
            for dx in 0..bw {
                col_mins[buf_row + dx] = h_mins[src_row + x_start + dx];
                col_maxs[buf_row + dx] = h_maxs[src_row + x_start + dx];
            }
        }

        let mut block_mins = vec![0i32; h * bw];
        let mut block_maxs = vec![0i32; h * bw];

        for y in 0..h {
            let y0 = y.saturating_sub(half);
            let y1 = (y + half).min(h - 1);
            for dx in 0..bw {
                let mut lo = i32::MAX;
                let mut hi = i32::MIN;
                for yy in y0..=y1 {
                    let buf_idx = (yy + 1) * bw + dx;
                    if col_mins[buf_idx] < lo { lo = col_mins[buf_idx]; }
                    if col_maxs[buf_idx] > hi { hi = col_maxs[buf_idx]; }
                }
                block_mins[y * bw + dx] = lo;
                block_maxs[y * bw + dx] = hi;
            }
        }

        (x_start, bw, block_mins, block_maxs)
    }).collect();

    let mut mins = vec![0i32; h * w];
    let mut maxs = vec![0i32; h * w];
    for (x_start, bw, block_mins, block_maxs) in &block_results {
        for y in 0..h {
            let dst = y * w + x_start;
            let src = y * bw;
            mins[dst..dst + bw].copy_from_slice(&block_mins[src..src + bw]);
            maxs[dst..dst + bw].copy_from_slice(&block_maxs[src..src + bw]);
        }
    }
    (mins, maxs)
}

// ======================================================================
// Fibonacci correction -- fused pipeline
// ======================================================================

/// Complete Fibonacci correction, optimized.
pub fn fibonacci_correction(
    recon_2d: &[f64],
    labels_2d: &[i32],
    centroids: &[f64],
    h: usize,
    w: usize,
) -> Vec<f64> {
    let n = h * w;
    let n_c = centroids.len();
    let lab = labels_2d;
    // --- Phase 1: Transition detection (edges, has_higher, has_lower) ---
    // Compact has_higher/has_lower/edges as u8 flags to save memory
    const F_HIGHER: u8 = 1;
    const F_LOWER: u8 = 2;
    const F_EDGE: u8 = 4;
    let mut flags = vec![0u8; n];

    // Vertical transitions
    for y in 0..h - 1 {
        let row = y * w;
        let next_row = (y + 1) * w;
        for x in 0..w {
            let dv = lab[next_row + x] - lab[row + x];
            let adv = dv.unsigned_abs();
            if adv >= 1 && adv <= 2 {
                if dv > 0 {
                    flags[row + x] |= F_HIGHER;
                    flags[next_row + x] |= F_LOWER;
                } else {
                    flags[row + x] |= F_LOWER;
                    flags[next_row + x] |= F_HIGHER;
                }
            } else if adv > 2 {
                flags[row + x] |= F_EDGE;
                flags[next_row + x] |= F_EDGE;
            }
        }
    }

    // Horizontal transitions
    for y in 0..h {
        let row = y * w;
        for x in 0..w - 1 {
            let dh = lab[row + x + 1] - lab[row + x];
            let adh = dh.unsigned_abs();
            if adh >= 1 && adh <= 2 {
                if dh > 0 {
                    flags[row + x] |= F_HIGHER;
                    flags[row + x + 1] |= F_LOWER;
                } else {
                    flags[row + x] |= F_LOWER;
                    flags[row + x + 1] |= F_HIGHER;
                }
            } else if adh > 2 {
                flags[row + x] |= F_EDGE;
                flags[row + x + 1] |= F_EDGE;
            }
        }
    }


    // --- Phase 2: Vertical scan (parallelized per column block) ---
    // Each column is independent: parallelized per block of 64 columns.
    const CBLK: usize = 64;
    let n_cblk = (w + CBLK - 1) / CBLK;

    let mut bw_v = vec![0.0f64; n];
    let mut frac_v = vec![0.5f64; n];
    let mut s_top_v = vec![0.0f64; n];
    let mut s_bot_v = vec![0.0f64; n];

    {
        let vert_blocks: Vec<_> = (0..n_cblk).into_par_iter().map(|b| {
            let x_start = b * CBLK;
            let x_end = (x_start + CBLK).min(w);
            let bw = x_end - x_start;

            // Forward (d_up, s_top)
            let mut d_up = vec![0.0f64; bw * h];
            let mut st = vec![0.0f64; bw * h];
            for y in 1..h {
                for dx in 0..bw {
                    let x = x_start + dx;
                    let bi = y * bw + dx;
                    let bp = (y - 1) * bw + dx;
                    let lab_cur = lab[y * w + x];
                    let lab_prev = lab[(y - 1) * w + x];
                    if lab_cur == lab_prev {
                        d_up[bi] = d_up[bp] + 1.0;
                        st[bi] = st[bp];
                    } else if lab_prev < lab_cur {
                        st[bi] = -0.5;
                    } else if lab_prev > lab_cur {
                        st[bi] = 0.5;
                    }
                }
            }

            // Backward (d_down, s_bot)
            let mut d_down = vec![0.0f64; bw * h];
            let mut sb = vec![0.0f64; bw * h];
            for y in (0..h - 1).rev() {
                for dx in 0..bw {
                    let x = x_start + dx;
                    let bi = y * bw + dx;
                    let bn = (y + 1) * bw + dx;
                    let lab_cur = lab[y * w + x];
                    let lab_next = lab[(y + 1) * w + x];
                    if lab_cur == lab_next {
                        d_down[bi] = d_down[bn] + 1.0;
                        sb[bi] = sb[bn];
                    } else if lab_next > lab_cur {
                        sb[bi] = 0.5;
                    } else if lab_next < lab_cur {
                        sb[bi] = -0.5;
                    }
                }
            }

            // Fuse bw_v, frac_v in the block
            let mut bw_blk = vec![0.0f64; bw * h];
            let mut frac_blk = vec![0.5f64; bw * h];
            for i in 0..bw * h {
                let total = d_up[i] + d_down[i];
                bw_blk[i] = total;
                if total > 0.0 {
                    frac_blk[i] = d_up[i] / total;
                }
            }

            (x_start, bw, bw_blk, frac_blk, st, sb)
        }).collect();

        // Scatter to row-major
        for (x_start, bw, bw_blk, frac_blk, st, sb) in &vert_blocks {
            for y in 0..h {
                for dx in 0..*bw {
                    let idx = y * w + x_start + dx;
                    let bi = y * bw + dx;
                    bw_v[idx] = bw_blk[bi];
                    frac_v[idx] = frac_blk[bi];
                    s_top_v[idx] = st[bi];
                    s_bot_v[idx] = sb[bi];
                }
            }
        }
    }

    // Heisenberg horizontal smoothing of frac_v, then compute spin_v
    let spin_v = {
        let frac_v_smooth = uniform_filter1d_h(&frac_v, h, w, 5);
        // Mask-apply (parallelized per row)
        frac_v.par_chunks_mut(w).enumerate().for_each(|(y, fv_row)| {
            let row = y * w;
            for x in 0..w {
                let idx = row + x;
                let mut safe = true;
                if x >= 1 && lab[idx] != lab[idx - 1] { safe = false; }
                if safe && x >= 2 && lab[idx] != lab[idx - 2] { safe = false; }
                if safe && x + 1 < w && lab[idx] != lab[idx + 1] { safe = false; }
                if safe && x + 2 < w && lab[idx] != lab[idx + 2] { safe = false; }
                if safe {
                    fv_row[x] = frac_v_smooth[idx];
                }
            }
        });
        let mut spin_v = s_top_v;
        for i in 0..n {
            spin_v[i] = spin_v[i] * (1.0 - frac_v[i]) + s_bot_v[i] * frac_v[i];
        }
        spin_v
    };

    // --- Phase 3: Horizontal scan (parallelized per row) ---
    // Each row is independent: forward + backward + fusion in one pass.
    let mut bw_h = vec![0.0f64; n];
    let mut frac_h = vec![0.5f64; n];
    let mut s_left_h = vec![0.0f64; n];
    let mut s_right_h = vec![0.0f64; n];

    bw_h.par_chunks_mut(w)
        .zip(frac_h.par_chunks_mut(w))
        .zip(s_left_h.par_chunks_mut(w))
        .zip(s_right_h.par_chunks_mut(w))
        .zip(lab.par_chunks(w))
        .for_each(|((((bw_row, fh_row), sl_row), sr_row), lab_row)| {
            // Forward: d_left, s_left
            let mut d_left = vec![0.0f64; w];
            for x in 1..w {
                if lab_row[x] == lab_row[x - 1] {
                    d_left[x] = d_left[x - 1] + 1.0;
                    sl_row[x] = sl_row[x - 1];
                } else if lab_row[x - 1] < lab_row[x] {
                    sl_row[x] = -0.5;
                } else if lab_row[x - 1] > lab_row[x] {
                    sl_row[x] = 0.5;
                }
            }
            // Backward: d_right, s_right
            let mut d_right = vec![0.0f64; w];
            for x in (0..w - 1).rev() {
                if lab_row[x] == lab_row[x + 1] {
                    d_right[x] = d_right[x + 1] + 1.0;
                    sr_row[x] = sr_row[x + 1];
                } else if lab_row[x + 1] > lab_row[x] {
                    sr_row[x] = 0.5;
                } else if lab_row[x + 1] < lab_row[x] {
                    sr_row[x] = -0.5;
                }
            }
            // Fuse bw_h, frac_h
            for x in 0..w {
                let total = d_left[x] + d_right[x];
                bw_row[x] = total;
                if total > 0.0 {
                    fh_row[x] = d_left[x] / total;
                }
            }
        });

    // Heisenberg vertical smoothing of frac_h, then spin_h
    let spin_h = {
        let frac_h_smooth = uniform_filter1d_v(&frac_h, h, w, 5);
        // Mask-apply (parallelized per row)
        frac_h.par_chunks_mut(w).enumerate().for_each(|(y, fh_row)| {
            let row = y * w;
            for x in 0..w {
                let idx = row + x;
                let mut safe = true;
                if y >= 1 && lab[idx] != lab[idx - w] { safe = false; }
                if safe && y >= 2 && lab[idx] != lab[idx - 2 * w] { safe = false; }
                if safe && y + 1 < h && lab[idx] != lab[idx + w] { safe = false; }
                if safe && y + 2 < h && lab[idx] != lab[idx + 2 * w] { safe = false; }
                if safe {
                    fh_row[x] = frac_h_smooth[idx];
                }
            }
        });
        let mut spin_h = s_left_h;
        for i in 0..n {
            spin_h[i] = spin_h[i] * (1.0 - frac_h[i]) + s_right_h[i] * frac_h[i];
        }
        spin_h
    };

    // --- Phase 4: EDT for coherence ---
    let edges: Vec<bool> = flags.iter().map(|&f| f & F_EDGE != 0).collect();
    let d_edge = distance_transform_edt(&edges, h, w);

    // --- Phase 5: Quantum superposition ---
    let lab_f64: Vec<f64> = lab.iter().map(|&v| v as f64).collect();
    let smooth_lab = uniform_filter_2d(&lab_f64, h, w, 3);
    let (lab_min5, lab_max5) = local_min_max(lab, h, w, 5);
    // --- Phase 6: Final SIMD-friendly pass (parallelized per row) ---
    // Strategy: separate gathers (centroid lookups) from arithmetic
    // to enable AVX2/FMA vectorization of the inner loop.
    let inv3: f64 = 1.0 / 3.0;

    let result: Vec<f64> = (0..h).into_par_iter().flat_map(|y| {
        let row = y * w;
        let y_phi = y as f64 * PHI;

        // Row slices (compiler knows len=w, eliminates bounds checks)
        let lab_row = &lab[row..row + w];
        let bwv = &bw_v[row..row + w];
        let bwh = &bw_h[row..row + w];
        let sv = &spin_v[row..row + w];
        let sh = &spin_h[row..row + w];
        let de = &d_edge[row..row + w];
        let sl = &smooth_lab[row..row + w];
        let lmin = &lab_min5[row..row + w];
        let lmax = &lab_max5[row..row + w];
        let rec = &recon_2d[row..row + w];

        // Pass 1: pre-compute centroid lookups (scalar, eliminates gathers)
        let mut cv = vec![0.0f64; w];
        let mut ga = vec![0.0f64; w];
        let mut gb = vec![0.0f64; w];
        let mut sr = vec![0.0f64; w];
        let mut fl = vec![0.0f64; w];

        for x in 0..w {
            let k = (lab_row[x] as usize).min(n_c - 1);
            cv[x] = centroids[k];
            ga[x] = centroids[(k + 1).min(n_c - 1)] - cv[x];
            gb[x] = cv[x] - centroids[k.saturating_sub(1)];

            let s = sl[x];
            let lf = s.floor().max(0.0).min((n_c - 1) as f64) as usize;
            let alpha = s - s.floor();
            sr[x] = centroids[lf] + alpha * (centroids[(lf + 1).min(n_c - 1)] - centroids[lf]);
            fl[x] = alpha.min(1.0 - alpha);
        }

        // Pass 2: pure arithmetic (vectorizable, no gathers)
        let mut out = vec![0.0f64; w];
        for x in 0..w {
            let g_above = ga[x];
            let g_below = gb[x];

            // Spin weights (branchless: mask * division)
            let bv = bwv[x];
            let bh = bwh[x];
            let mv = if bv > 0.0 { 1.0f64 } else { 0.0 };
            let mh = if bh > 0.0 { 1.0f64 } else { 0.0 };
            let wv = mv / (bv + 1.0);
            let wh = mh / (bh + 1.0);
            let wt = (wv + wh).max(1e-30);

            // Spin + neutralization (branchless fused clamp)
            let spin_raw = (sv[x] * wv + sh[x] * wh) / wt;
            let lo = if g_above > 0.0 { -0.5 } else { 0.0 };
            let hi = if g_below > 0.0 { 0.5 } else { 0.0 };
            let spin = spin_raw.clamp(lo, hi);

            // Asymmetric spacing + coherence
            let spacing = if spin >= 0.0 { g_above } else { g_below };
            let coherence = (de[x] * inv3).clamp(0.0, 1.0);
            let coherent_corr = spin * spacing * coherence;

            // Quantum superposition (branchless)
            let super_corr = sr[x] - rec[x];
            let lr_mask = if (lmax[x] - lmin[x]) <= 1 { 1.0f64 } else { 0.0 };
            let osc_weight = lr_mask * ((fl[x] - 0.15) * 5.0).clamp(0.0, 1.0);
            let combined_corr = coherent_corr + osc_weight * (super_corr * coherence - coherent_corr);

            // 2D Fibonacci dithering (fract + FMA-friendly)
            let fib_phase = (y_phi + x as f64 * PHI_SQ).fract();
            let min_bw = (bv + 1.0).min(bh + 1.0);
            let vacuum_amp = (1.0 / min_bw).clamp(0.01, 0.15);
            let fib_amp = (vacuum_amp - 0.25) * coherence + 0.25;
            let fib_corr = (fib_phase - 0.5) * g_above * fib_amp;

            // Quantum confinement
            let total_dev = (rec[x] - cv[x]) + combined_corr + fib_corr;
            out[x] = cv[x] + total_dev.clamp(-g_below * 0.5, g_above * 0.5);
        }
        out
    }).collect();

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_transform() {
        let mask = vec![
            false, false, false,
            false, true,  false,
            false, false, false,
        ];
        let d = distance_transform_edt(&mask, 3, 3);
        assert_eq!(d[4], 0.0);
        assert!((d[0] - std::f64::consts::SQRT_2).abs() < 0.01);
        assert!((d[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_uniform_filter1d_h() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let out = uniform_filter1d_h(&data, 1, 5, 3);
        assert!((out[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_fibonacci_uniform_labels() {
        let centroids = vec![50.0, 100.0, 150.0];
        let labels = vec![1i32; 9];
        let recon: Vec<f64> = labels.iter().map(|&l| centroids[l as usize]).collect();
        let result = fibonacci_correction(&recon, &labels, &centroids, 3, 3);
        for i in 0..9 {
            assert!((result[i] - 100.0).abs() < 30.0, "pixel {} = {}", i, result[i]);
        }
    }
}
