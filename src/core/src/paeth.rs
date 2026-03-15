/// PNG-style Paeth prediction on 2D label maps.

/// Paeth predictor: chooses among a (left), b (above), c (upper-left)
/// the one closest to p = a + b - c.
#[inline]
fn paeth_predictor(a: i32, b: i32, c: i32) -> i32 {
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Reconstruct 2D labels from Paeth residuals (sequential).
/// Input: residuals (H * W) in row-major.
/// Output: labels (H * W) in row-major.
pub fn paeth_unpredict_2d(residuals: &[i16], h: usize, w: usize) -> Vec<i32> {
    let mut lab = vec![0i32; h * w];

    // (0,0)
    lab[0] = residuals[0] as i32;

    // First row: prediction = left
    for x in 1..w {
        lab[x] = residuals[x] as i32 + lab[x - 1];
    }

    // First column: prediction = above
    for y in 1..h {
        lab[y * w] = residuals[y * w] as i32 + lab[(y - 1) * w];
    }

    // Rest: full Paeth
    for y in 1..h {
        for x in 1..w {
            let a = lab[y * w + (x - 1)];         // left
            let b = lab[(y - 1) * w + x];          // above
            let c = lab[(y - 1) * w + (x - 1)];   // upper-left
            let pred = paeth_predictor(a, b, c);
            lab[y * w + x] = residuals[y * w + x] as i32 + pred;
        }
    }

    lab
}

/// Paeth prediction on 2D labels (encoder).
/// Returns the residuals (label - prediction) as i16.
pub fn paeth_predict_2d(labels: &[i32], h: usize, w: usize) -> Vec<i16> {
    let mut residuals = vec![0i16; h * w];

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let val = labels[idx];
            let pred = if y == 0 && x == 0 {
                0
            } else if y == 0 {
                labels[idx - 1]
            } else if x == 0 {
                labels[(y - 1) * w + x]
            } else {
                let a = labels[idx - 1];             // left
                let b = labels[(y - 1) * w + x];     // above
                let c = labels[(y - 1) * w + x - 1]; // upper-left
                paeth_predictor(a, b, c)
            };
            residuals[idx] = (val - pred) as i16;
        }
    }

    residuals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paeth_predictor() {
        // If a=b=c, pred=a
        assert_eq!(paeth_predictor(5, 5, 5), 5);
        // Horizontal gradient: a=10, b=5, c=5, p=10, pa=0 -> a
        assert_eq!(paeth_predictor(10, 5, 5), 10);
        // Vertical gradient: a=5, b=10, c=5, p=10, pa=5, pb=0 -> b
        assert_eq!(paeth_predictor(5, 10, 5), 10);
    }

    #[test]
    fn test_unpredict_uniform() {
        // Uniform image: all residuals are 0 except the first
        let mut res = vec![0i16; 9];
        res[0] = 42;
        let lab = paeth_unpredict_2d(&res, 3, 3);
        assert!(lab.iter().all(|&v| v == 42));
    }
}
