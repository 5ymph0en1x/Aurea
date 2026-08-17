//! Per-stage wall-clock instrumentation and counters.
//!
//! Enabled by `AUREA_TIMINGS=1`, or implicitly by `AUREA_STATS_JSON=<path>`,
//! which additionally appends one JSON object per encode/decode to that file.
//! Zero cost when disabled: one LazyLock<bool> test, no allocation.

use std::sync::{LazyLock, Mutex};
use std::time::Instant;

static JSON_PATH: LazyLock<Option<String>> = LazyLock::new(|| {
    std::env::var("AUREA_STATS_JSON")
        .ok()
        .filter(|p| !p.is_empty())
});

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("AUREA_TIMINGS")
        .map(|v| v == "1")
        .unwrap_or(false)
        || JSON_PATH.is_some()
});

static STAGES: Mutex<Vec<(&'static str, f64)>> = Mutex::new(Vec::new());

/// Named scalar counters, accumulated by summation under the same gate as the
/// stages. P0 needs these to tell a transform cost from an entropy cost: a
/// lapped topology analyses far more blocks, and only a per-phase split shows
/// whether the extra coefficients are paid at the transform, the trellis or the
/// coder.
static COUNTERS: Mutex<Vec<(&'static str, f64)>> = Mutex::new(Vec::new());

/// Add `value` to the counter `label`.
#[inline]
pub fn count(label: &'static str, value: f64) {
    if *ENABLED {
        COUNTERS.lock().unwrap().push((label, value));
    }
}

#[inline]
pub fn enabled() -> bool {
    *ENABLED
}

pub struct StageGuard {
    label: &'static str,
    start: Instant,
}

/// Begin timing a stage. Returns None (and does nothing) unless AUREA_TIMINGS=1.
#[inline]
pub fn stage(label: &'static str) -> Option<StageGuard> {
    if *ENABLED {
        Some(StageGuard {
            label,
            start: Instant::now(),
        })
    } else {
        None
    }
}

impl Drop for StageGuard {
    fn drop(&mut self) {
        let dt = self.start.elapsed().as_secs_f64();
        STAGES.lock().unwrap().push((self.label, dt));
    }
}

/// Aggregate repeated labels by summation, keeping first-seen order.
fn aggregate(entries: &[(&'static str, f64)]) -> (Vec<&'static str>, Vec<f64>) {
    let mut order: Vec<&'static str> = Vec::new();
    let mut sums: Vec<f64> = Vec::new();
    for &(label, value) in entries {
        match order.iter().position(|&l| l == label) {
            Some(i) => sums[i] += value,
            None => {
                order.push(label);
                sums.push(value);
            }
        }
    }
    (order, sums)
}

/// Append one JSON object describing this run, then clear the counters.
///
/// JSON Lines rather than a single document: a sweep writes hundreds of runs
/// and each must survive an interrupted process.
fn write_json(context: &str, stage_order: &[&'static str], stage_sums: &[f64]) {
    let Some(path) = JSON_PATH.as_ref() else {
        COUNTERS.lock().unwrap().clear();
        return;
    };
    let mut counters = COUNTERS.lock().unwrap();
    let (order, sums) = aggregate(&counters);
    let mut line = String::from("{");
    line.push_str(&format!("\"context\":\"{context}\""));
    for (label, value) in stage_order.iter().zip(stage_sums) {
        // Milliseconds: the roadmap's counters are named *_ms.
        line.push_str(&format!(
            ",\"{}_ms\":{:.4}",
            label.replace(':', "_"),
            value * 1000.0
        ));
    }
    for (label, value) in order.iter().zip(&sums) {
        line.push_str(&format!(",\"{label}\":{value}"));
    }
    line.push_str(
        "}
",
    );
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path.as_str())
    {
        let _ = file.write_all(line.as_bytes());
    }
    counters.clear();
}

/// Print accumulated stages to stderr and clear the accumulator.
/// Repeated labels (e.g. one per channel) are aggregated, first-seen order kept.
pub fn report(context: &str) {
    if !*ENABLED {
        return;
    }
    let mut stages = STAGES.lock().unwrap();
    let mut order: Vec<&'static str> = Vec::new();
    let mut sums: Vec<f64> = Vec::new();
    for &(label, t) in stages.iter() {
        match order.iter().position(|&l| l == label) {
            Some(i) => sums[i] += t,
            None => {
                order.push(label);
                sums.push(t);
            }
        }
    }
    write_json(context, &order, &sums);
    let total: f64 = sums.iter().sum();
    if std::env::var("AUREA_TIMINGS")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        eprintln!("[AUREA_TIMINGS] {context} — total instrumente {total:.3}s");
    } else {
        stages.clear();
        return;
    }
    for (label, t) in order.iter().zip(&sums) {
        eprintln!(
            "[AUREA_TIMINGS]   {label:<32} {t:>8.3}s ({:>5.1}%)",
            100.0 * t / total.max(1e-9)
        );
    }
    stages.clear();
}
