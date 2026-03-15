/// 1D vector quantization: k-means++, assignment, refinement.

use rayon::prelude::*;

/// Assign each value to the nearest sorted centroid (binary search).
pub fn assign_nearest(data: &[f64], centroids: &[f64]) -> Vec<i32> {
    let n_c = centroids.len();
    data.par_iter().map(|&x| {
        let idx = centroids.partition_point(|&c| c < x);
        if idx == 0 {
            0i32
        } else if idx >= n_c {
            (n_c - 1) as i32
        } else if (x - centroids[idx - 1]).abs() <= (x - centroids[idx]).abs() {
            (idx - 1) as i32
        } else {
            idx as i32
        }
    }).collect()
}

/// Simple RNG (splitmix64) for deterministic initialization.
struct Rng64 {
    state: u64,
}

impl Rng64 {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// k-means++ initialization: places centroids where data is dense.
fn kmeans_pp_init(data: &[f64], k: usize, seed: u64) -> Vec<f64> {
    let n = data.len();
    let mut rng = Rng64::new(seed);

    // First centroid: random point
    let first_idx = rng.next_u64() as usize % n;
    let mut centroids = vec![data[first_idx]];

    // Buffer for D(x)^2
    let mut dist_sq = vec![f64::INFINITY; n];

    for _ in 1..k {
        // Update D(x)^2 with the last added centroid
        let last_c = *centroids.last().unwrap();
        let mut total = 0.0f64;
        for i in 0..n {
            let d = (data[i] - last_c) * (data[i] - last_c);
            if d < dist_sq[i] {
                dist_sq[i] = d;
            }
            total += dist_sq[i];
        }

        // Choose next centroid proportional to D(x)^2
        let threshold = total * rng.next_f64();
        let mut cumsum = 0.0;
        let mut chosen = 0;
        for i in 0..n {
            cumsum += dist_sq[i];
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen]);
    }

    centroids.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    centroids
}

/// 1D k-means with k-means++ initialization (deterministic, seed=42).
pub fn kmeans_1d(data: &[f64], k: usize, max_iter: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 || k == 0 { return vec![]; }

    // k-means++ initialization
    let mut centroids = kmeans_pp_init(data, k, 42);

    // k-means iterations
    for _ in 0..max_iter {
        let mut sums = vec![0.0f64; k];
        let mut counts = vec![0usize; k];

        for &x in data {
            let label = assign_one(x, &centroids);
            sums[label] += x;
            counts[label] += 1;
        }

        for j in 0..k {
            if counts[j] > 0 {
                centroids[j] = sums[j] / counts[j] as f64;
            }
        }
        centroids.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    }

    centroids
}

#[inline]
fn assign_one(x: f64, centroids: &[f64]) -> usize {
    let n_c = centroids.len();
    let idx = centroids.partition_point(|&c| c < x);
    if idx == 0 { return 0; }
    if idx >= n_c { return n_c - 1; }
    if (x - centroids[idx - 1]).abs() <= (x - centroids[idx]).abs() {
        idx - 1
    } else {
        idx
    }
}

/// Refine centroids: encode-reconstruct-measure-adjust loop.
/// Without fibonacci_correction (the spin correction is symmetric).
pub fn refine_centroids(
    data_flat: &[f64], centroids_in: &[f64], _h: usize, _w: usize,
    max_rounds: usize,
) -> Vec<f64> {
    let n = data_flat.len();
    let k = centroids_in.len();

    let mut centroids = centroids_in.to_vec();
    let mut best_centroids = centroids.clone();
    let mut best_mae = f64::INFINITY;

    for round_idx in 0..max_rounds {
        // 1. Quantize as the bitstream does (uint8)
        let c_u8: Vec<f64> = centroids.iter()
            .map(|&c| c.round().clamp(0.0, 255.0))
            .collect();

        // 2. Assign and reconstruct (raw VQ)
        let labels = assign_nearest(data_flat, &c_u8);

        // 3. Mean absolute error
        let mut total_abs_err = 0.0f64;
        for i in 0..n {
            total_abs_err += (data_flat[i] - c_u8[labels[i] as usize]).abs();
        }
        let mae = total_abs_err / n as f64;

        if mae < best_mae {
            best_mae = mae;
            best_centroids = centroids.clone();
        }

        // 4. Adjust by cluster mean error
        let damping = 0.8 / (1.0 + round_idx as f64);
        let mut err_sums = vec![0.0f64; k];
        let mut err_counts = vec![0usize; k];

        for i in 0..n {
            let l = labels[i] as usize;
            err_sums[l] += data_flat[i] - c_u8[l];
            err_counts[l] += 1;
        }

        let mut new_centroids = centroids.clone();
        for j in 0..k {
            if err_counts[j] >= 10 {
                new_centroids[j] += err_sums[j] / err_counts[j] as f64 * damping;
            }
        }

        new_centroids.iter_mut().for_each(|c| *c = c.clamp(0.0, 255.0));
        new_centroids.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        centroids = new_centroids;
    }

    // Regularization: minimum spacing between centroids
    let mut result = best_centroids;
    let n_c = result.len();
    let min_sp = (256.0 / n_c as f64 * 0.5).max(2.5);
    for i in 0..n_c - 1 {
        if result[i + 1] - result[i] < min_sp {
            let mid = (result[i] + result[i + 1]) / 2.0;
            result[i] = mid - min_sp / 2.0;
            result[i + 1] = mid + min_sp / 2.0;
        }
    }
    result.iter_mut().for_each(|c| *c = c.clamp(0.0, 255.0));
    result.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assign_nearest() {
        let centroids = vec![10.0, 50.0, 100.0];
        let data = vec![0.0, 29.0, 76.0, 200.0];
        let labels = assign_nearest(&data, &centroids);
        assert_eq!(labels, vec![0, 0, 2, 2]);
    }

    #[test]
    fn test_kmeans_1d_basic() {
        let data = vec![1.0, 2.0, 3.0, 100.0, 101.0, 102.0];
        let centroids = kmeans_1d(&data, 2, 10);
        assert_eq!(centroids.len(), 2);
        assert!((centroids[0] - 2.0).abs() < 1.0);
        assert!((centroids[1] - 101.0).abs() < 1.0);
    }
}
