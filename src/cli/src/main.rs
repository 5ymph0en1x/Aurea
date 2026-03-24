use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use aurea_core::aurea_encoder;
use aurea_core::codec_params::{CodecParams, Pipeline};

#[derive(Parser)]
#[command(name = "aurea", about = "AUREA codec (.aur) — Fibonacci/phi/Zeckendorf image codec")]
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
        /// Pipeline: lot, edge-energy, optica, hex, dna (default: lot)
        #[arg(long, default_value = "lot")]
        pipeline: String,
    },
    /// Display information about a .aur file
    Info {
        /// .aur file
        input: PathBuf,
    },
}

fn parse_pipeline(s: &str) -> Result<Pipeline, String> {
    match s {
        "edge-energy" | "ee" => Ok(Pipeline::EdgeEnergy),
        "optica" | "v11" => Ok(Pipeline::Optica),
        "lot" => Ok(Pipeline::LotAdn4),
        "hex" => Ok(Pipeline::HexPyramid),
        "dna" => Ok(Pipeline::EdgeEnergyDna),
        _ => Err(format!("Unknown pipeline '{}'. Choose: edge-energy, optica, lot, hex, dna", s)),
    }
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
        let (header, _) = aurea_core::bitstream::parse_aur2_header(&file_data)
            .map_err(|e| format!("Parse error: {}", e))?;
        let n = header.width * header.height;
        let bpp = file_size as f64 * 8.0 / n as f64;
        let ratio = (n * 3) as f64 / file_size as f64;

        println!("File      : {}", input.display());
        if header.version == 11 {
            println!("Format    : AUREA Optica v11 (.aur) [AUR2]");
        } else if header.version == 8 {
            println!("Format    : AUREA Edge-Energy v8 (.aur) [AUR2]");
        } else if header.version >= 2 {
            println!("Format    : AUREA v{} (.aur) [AUR2]", header.version);
        } else {
            println!("Format    : AUREA v1 (.aur) [AUR2]");
        }
        println!("Size      : {} bytes ({:.2} bpp)", file_size, bpp);
        println!("Image     : {}x{} ({} pixels)", header.width, header.height, n);
        println!("Quality   : {}", header.quality);
        if header.version == 11 {
            println!("Pipeline  : Optica (Photon Synthesis + Capillary Chroma + Hyper-Sparse rANS)");
        } else if header.version == 8 {
            println!("Pipeline  : Edge-Energy DPCM + Hex Oracle");
        } else if header.version >= 2 {
            println!("Transform : LOT (Lapped Orthogonal Transform)");
        } else {
            println!("Wavelets  : {} levels", header.wv_levels);
        }
        println!("Coding    : rANS");
        println!("Color     : Golden Color Transform (GCT)");
        println!("Ratio     : {:.1}x vs RAW", ratio);
    } else {
        println!("File      : {}", input.display());
        println!("Format    : Unknown (not AUR2)");
        println!("Size      : {} bytes", file_size);
        return Err("Unsupported format. Only AUR2 is supported.".into());
    }

    Ok(())
}

fn cmd_encode(
    input: &PathBuf,
    output: &PathBuf,
    quality: u8,
    pipeline: Pipeline,
) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let img = image::open(input)?.to_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);
    let rgb = img.into_raw();

    let pipeline_name = match pipeline {
        Pipeline::Optica => "Optica v11",
        Pipeline::EdgeEnergy => "Edge-Energy v8",
        Pipeline::EdgeEnergyDna => "Edge-Energy DNA v8",
        Pipeline::HexPyramid => "Hex Pyramid v7",
        Pipeline::LotAdn4 => "LOT v3/4",
    };

    println!("Encode AUREA {} : {} (q={})", pipeline_name, input.display(), quality);
    println!("  Image     : {}x{}", width, height);

    let params = CodecParams::with_pipeline(pipeline, quality);
    let result = aurea_encoder::encode_unified(&rgb, width, height, &params)?;

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
        Commands::Encode { input, output, quality, pipeline } => {
            let out = output.clone().unwrap_or_else(|| input.with_extension("aur"));
            match parse_pipeline(pipeline) {
                Ok(p) => cmd_encode(input, &out, *quality, p),
                Err(e) => Err(e.into()),
            }
        }
        Commands::Info { input } => cmd_info(input),
    };

    if let Err(e) = result {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    }
}
