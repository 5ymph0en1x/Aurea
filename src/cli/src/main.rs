use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use aurea_core::aurea_encoder::{self, AureaEncoderParams};

// v2 only (v1 removed)

#[derive(Parser)]
#[command(name = "aurea", about = "AUREA codec (.aur) — Fibonacci/phi/Zeckendorf wavelet codec")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Decode a .aur file to an image (PNG, BMP, JPEG)
    Decode {
        /// Source .aur file
        input: PathBuf,
        /// Output image
        output: PathBuf,
    },
    /// Encode an image to a .aur file
    Encode {
        /// Source image (PNG, BMP, JPEG...)
        input: PathBuf,
        /// Output .aur file (default: same name with .aur)
        output: Option<PathBuf>,
        /// Quality (1-100, default 75)
        #[arg(short, long, default_value_t = 75)]
        quality: u8,
    },
    /// Display information about a .aur file
    Info {
        /// .aur file
        input: PathBuf,
    },
}

fn cmd_decode(input: &PathBuf, output: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let file_data = fs::read(input)?;
    let decoded = aurea_core::decode_aurea(&file_data)?;
    let elapsed = t0.elapsed();

    let img = image::RgbImage::from_raw(
        decoded.width as u32,
        decoded.height as u32,
        decoded.rgb,
    )
    .ok_or("Unable to create image buffer")?;

    img.save(output)?;

    println!("Decode: {} -> {}", input.display(), output.display());
    println!("  Dimensions : {}x{}", decoded.width, decoded.height);
    println!("  Time       : {:.0} ms", elapsed.as_secs_f64() * 1000.0);

    Ok(())
}

fn cmd_info(input: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file_data = fs::read(input)?;
    let file_size = file_data.len();

    if file_data.len() < 4 {
        return Err("File too short".into());
    }

    if &file_data[0..4] == b"AUR2" {
        // AUR2 v10 format
        let (header, _) = aurea_core::bitstream::parse_aur2_header(&file_data)
            .map_err(|e| format!("Parse error: {}", e))?;
        let n = header.width * header.height;
        let bpp = file_size as f64 * 8.0 / n as f64;
        let ratio = (n * 3) as f64 / file_size as f64;

        println!("File      : {}", input.display());
        if header.version >= 2 {
            println!("Format    : AUREA v2-LOT (.aur) [AUR2]");
        } else {
            println!("Format    : AUREA v10 (.aur) [AUR2]");
        }
        println!("Size      : {} bytes ({:.2} bpp)", file_size, bpp);
        println!("Image     : {}x{} ({} pixels)", header.width, header.height, n);
        println!("Quality   : {}", header.quality);
        if header.version >= 2 {
            println!("Transform : LOT (Lapped Orthogonal Transform, 16x16 blocks)");
        } else {
            println!("Wavelets  : {} levels", header.wv_levels);
            println!("Pipeline  : Primitives-First (phi superstrings + polynomial patches)");
        }
        println!("Coding    : rANS");
        println!("Color     : Golden Color Transform (GCT)");
        println!("Ratio     : {:.1}x vs RAW", ratio);
    } else {
        // Unknown/legacy format
        println!("File      : {}", input.display());
        println!("Format    : Unknown (not AUR2)");
        println!("Size      : {} bytes", file_size);
        return Err("Unsupported format. Only AUR2 (v10) is supported.".into());
    }

    Ok(())
}

fn cmd_encode(
    input: &PathBuf,
    output: &PathBuf,
    quality: u8,
) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let img = image::open(input)?.to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);
    let rgb = img.into_raw();

    let n_repr = aurea_encoder::quality_to_n_repr(quality);

    println!("Encode AUREA v10 : {} (q={}, n_repr={})", input.display(), quality, n_repr);
    println!("  Image     : {}x{}", width, height);

    let params = AureaEncoderParams {
        quality,
        n_representatives: n_repr,
        geometric: true,
    };

    let result = aurea_encoder::encode_aurea_v2(&rgb, width, height, &params)?;

    fs::write(output, &result.aurea_data)?;

    let elapsed = t0.elapsed();
    let n = width * height;
    let bpp = result.compressed_size as f64 * 8.0 / n as f64;
    let ratio = (n * 3) as f64 / result.compressed_size as f64;

    println!("  Output    : {} ({} bytes)", output.display(), result.compressed_size);
    println!("  Ratio     : {:.1}x vs RAW ({:.2} bpp)", ratio, bpp);
    println!("  Time      : {:.0} ms", elapsed.as_secs_f64() * 1000.0);

    Ok(())
}

fn main() {
    let cli = Cli::parse();

    let result = match &cli.command {
        Commands::Decode { input, output } => cmd_decode(input, output),
        Commands::Encode { input, output, quality } => {
            let out = output.clone().unwrap_or_else(|| input.with_extension("aur"));
            cmd_encode(input, &out, *quality)
        }
        Commands::Info { input } => cmd_info(input),
    };

    if let Err(e) = result {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    }
}
