/// AUREA Trashland -- satellite signal loss simulator.
///
/// Takes an image, runs it through the Aurea codec pipeline,
/// corrupts it at various levels, and outputs the degraded result.
///
/// Usage:
///   aurea-trashland photo.png --signal 30
///   aurea-trashland photo.png --signal 10 --seed 42 -o wrecked.png

mod corrupt;
mod pipeline;

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "aurea-trashland",
    about = "Satellite signal loss effect via the Aurea codec pipeline",
    long_about = "Encodes an image through the Aurea wavelet/VQ pipeline, then corrupts it\n\
                  at multiple levels to simulate satellite transmission failure.\n\n\
                  Signal 100 = pristine, Signal 0 = total destruction."
)]
struct Cli {
    /// Input image (PNG, JPEG, BMP)
    input: PathBuf,

    /// Output image (default: <input>_trashland.png)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Signal strength: 0 (destroyed) to 100 (pristine)
    #[arg(short, long, default_value_t = 50)]
    signal: u8,

    /// Random seed for reproducible corruption
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() {
    let cli = Cli::parse();

    let signal = cli.signal.min(100);
    let seed = if cli.seed == 0 {
        // Derive seed from filename + signal for variety
        let name = cli.input.to_string_lossy();
        let mut h: u64 = 0xcbf29ce484222325;
        for b in name.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^ (signal as u64 * 0x9e3779b97f4a7c15)
    } else {
        cli.seed
    };

    let output = cli.output.unwrap_or_else(|| {
        let stem = cli.input.file_stem().unwrap_or_default().to_string_lossy();
        cli.input.with_file_name(format!("{}_trashland.png", stem))
    });

    // Load image
    let t0 = Instant::now();
    let img = image::open(&cli.input).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot open {}: {}", cli.input.display(), e);
        std::process::exit(1);
    }).to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);
    let rgb = img.into_raw();

    println!("AUREA Trashland -- satellite signal loss simulator");
    println!("  Input   : {} ({}x{})", cli.input.display(), width, height);
    println!("  Signal  : {}%{}", signal, if signal < 15 { " [CRITICAL]" } else if signal < 30 { " [SEVERE]" } else if signal < 50 { " [DEGRADED]" } else if signal < 80 { " [UNSTABLE]" } else { " [OK]" });
    println!("  Seed    : {}", seed);

    // Run the trashland pipeline
    let result = pipeline::trashland_pipeline(&rgb, width, height, signal, seed);

    // Save output
    let out_img = image::RgbImage::from_raw(width as u32, height as u32, result)
        .expect("Failed to create output image");
    out_img.save(&output).unwrap_or_else(|e| {
        eprintln!("ERROR: Cannot save {}: {}", output.display(), e);
        std::process::exit(1);
    });

    let elapsed = t0.elapsed();
    println!("  Output  : {}", output.display());
    println!("  Time    : {:.0} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  Status  : Signal degradation applied successfully");
}
