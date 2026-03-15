use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use aurea_core::aurea::AUREA_MAGIC;
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
        /// Geometric coding (AUREA v6)
        #[arg(short = 'g', long)]
        geometric: bool,
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

    if file_data.len() < 4 || &file_data[0..4] != AUREA_MAGIC {
        return Err("Not a .aur file (invalid magic)".into());
    }

    let after_magic = &file_data[4..];
    let payload = if after_magic.len() >= 2 && after_magic[0] == 0xFD && after_magic[1] == 0x37 {
        aurea_core::bitstream::decompress_xts_payload(after_magic)?
    } else if !after_magic.is_empty() && after_magic[0] >= 1 && after_magic[0] <= 7 {
        after_magic.to_vec()
    } else {
        aurea_core::rans::rans_decompress_bytes(after_magic)
    };

    if payload.len() < 7 {
        return Err("Payload too short".into());
    }

    let version = payload[0];
    let quality = payload[1];
    let w = u16::from_le_bytes([payload[2], payload[3]]) as usize;
    let h = u16::from_le_bytes([payload[4], payload[5]]) as usize;
    let wv_levels = payload[6];

    let n = w * h;
    let bpp = file_size as f64 * 8.0 / n as f64;
    let ratio = (n * 3) as f64 / file_size as f64;

    println!("File      : {}", input.display());
    println!("Format    : AUREA v{} (.aur)", version);
    println!("Size      : {} bytes ({:.2} bpp)", file_size, bpp);
    println!("Image     : {}x{} ({} pixels)", w, h, n);
    println!("Quality   : {}", quality);
    println!("Wavelets  : {} levels", wv_levels);
    println!("Quant.    : Fibonacci (phi)");
    println!("Scan      : Golden spiral (phi)");
    println!("Coding    : Zeckendorf");
    if version >= 2 {
        println!("Mode      : Golden Color Transform (GCT, phi-based)");
    } else {
        println!("Mode      : YCbCr 4:2:0");
    }
    let is_lzma = after_magic.len() >= 2 && after_magic[0] == 0xFD && after_magic[1] == 0x37;
    let comp_name = if is_lzma { "LZMA (legacy)" } else { "rANS" };
    if version == 6 {
        println!("Coding    : Geometric (segments + arcs) + {}", comp_name);
    } else {
        println!("Compression: {}", comp_name);
    }
    println!("Ratio     : {:.1}x vs RAW", ratio);

    Ok(())
}

fn cmd_encode(
    input: &PathBuf,
    output: &PathBuf,
    quality: u8,
    geometric: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let img = image::open(input)?.to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);
    let rgb = img.into_raw();

    let n_repr = aurea_encoder::quality_to_n_repr(quality);

    let ver = if geometric { "v6 (geometric)" } else { "v2" };
    println!("Encode AUREA {} : {} (q={}, n_repr={})", ver, input.display(), quality, n_repr);
    println!("  Image     : {}x{}", width, height);

    let params = AureaEncoderParams {
        quality,
        n_representatives: n_repr,
        geometric,
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
        Commands::Encode { input, output, quality, geometric } => {
            let out = output.clone().unwrap_or_else(|| input.with_extension("aur"));
            cmd_encode(input, &out, *quality, *geometric)
        }
        Commands::Info { input } => cmd_info(input),
    };

    if let Err(e) = result {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    }
}
