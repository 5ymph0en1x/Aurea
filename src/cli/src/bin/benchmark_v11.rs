use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;
use std::time::Instant;
use image::{DynamicImage, GenericImageView, ImageFormat};

const QUALITIES: [u32; 8] = [20, 30, 40, 50, 60, 70, 80, 90];
const RAW_DIR: &str = "benchmark/adn3";
const OUT_DIR: &str = "benchmark/final_comparison";

struct ResultPoint {
    name: String,
    q: u32,
    aur_bpp: f64,
    aur_psnr: f64,
    jpg_bpp: f64,
    jpg_psnr: f64,
}

fn calculate_psnr(orig: &DynamicImage, dec: &DynamicImage) -> f64 {
    let (w, h) = orig.dimensions();
    let mut mse = 0.0;
    let o_rgb = orig.to_rgb8();
    let d_rgb = dec.to_rgb8();

    for (p1, p2) in o_rgb.pixels().zip(d_rgb.pixels()) {
        for i in 0..3 {
            let diff = p1[i] as f64 - p2[i] as f64;
            mse += diff * diff;
        }
    }
    mse /= (w * h * 3) as f64;
    if mse < 1e-10 { return 99.0; }
    10.0 * (255.0 * 255.0 / mse).log10()
}

fn main() {
    let raw_path = Path::new(RAW_DIR);
    let out_path = Path::new(OUT_DIR);
    
    if !out_path.exists() {
        fs::create_dir_all(out_path).unwrap();
    }

    let entries = fs::read_dir(raw_path).unwrap();
    let mut images = Vec::new();
    for entry in entries {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "png") {
            images.push(path);
        }
    }
    images.sort();

    println!("Starting Benchmark v11 (Scale-Invariant Mesh LOT)");
    println!("Comparing 12 images at 8 quality levels...");

    let mut results = Vec::new();

    for img_path in &images {
        let stem = img_path.file_stem().unwrap().to_str().unwrap();
        println!("Processing: {}", stem);
        
        let orig = image::open(img_path).expect("Failed to open original image");
        let (w, h) = orig.dimensions();
        let n_pixels = (w * h) as f64;

        for &q in &QUALITIES {
            // --- AUREA ---
            let aur_file = out_path.join(format!("{}_q{}.aur", stem, q));
            let aur_dec = out_path.join(format!("{}_q{}_aur.png", stem, q));
            
            // Encode
            let start = Instant::now();
            let status = Command::new("target/release/aurea.exe")
                .args(["encode", img_path.to_str().unwrap(), aur_file.to_str().unwrap(), "-q", &q.to_string()])
                .status()
                .expect("Failed to run aurea encode");
            
            let aur_size = fs::metadata(&aur_file).unwrap().len();
            let aur_bpp = (aur_size * 8) as f64 / n_pixels;

            // Decode
            Command::new("target/release/aurea.exe")
                .args(["decode", aur_file.to_str().unwrap(), aur_dec.to_str().unwrap()])
                .status()
                .expect("Failed to run aurea decode");
            
            let dec_aur = image::open(&aur_dec).expect("Failed to open decoded Aurea image");
            let aur_psnr = calculate_psnr(&orig, &dec_aur);

            // --- JPEG ---
            let jpg_file = out_path.join(format!("{}_q{}.jpg", stem, q));
            orig.save_with_format(&jpg_file, ImageFormat::Jpeg).unwrap(); // Default is usually not enough, we need to control Q
            
            // For precise Q in JPEG, we'll use a small python trick or a dedicated crate if available.
            // Actually, the 'image' crate doesn't expose quality easily in a single call without specialized encoders.
            // Let's use 'magick' if available or just shell out to a quick python snippet for the JPEG part to be fair.
            
            let jpg_q_cmd = format!("from PIL import Image; Image.open(r'{}').convert('RGB').save(r'{}', 'JPEG', quality={})", 
                img_path.to_str().unwrap(), jpg_file.to_str().unwrap(), q);
            
            Command::new("python")
                .args(["-c", &jpg_q_cmd])
                .status()
                .expect("Failed to run python for JPEG encoding");

            let jpg_size = fs::metadata(&jpg_file).unwrap().len();
            let jpg_bpp = (jpg_size * 8) as f64 / n_pixels;
            
            let dec_jpg = image::open(&jpg_file).expect("Failed to open decoded JPEG");
            let jpg_psnr = calculate_psnr(&orig, &dec_jpg);

            results.push(ResultPoint {
                name: stem.to_string(),
                q,
                aur_bpp,
                aur_psnr,
                jpg_bpp,
                jpg_psnr,
            });

            println!("  Q={}: AUREA={:.3}bpp, {:.2}dB | JPEG={:.3}bpp, {:.2}dB", q, aur_bpp, aur_psnr, jpg_bpp, jpg_psnr);
        }
    }

    // Save results to JSON
    let mut json_str = String::from("[
");
    for (i, r) in results.iter().enumerate() {
        json_str.push_str(&format!(
            "  {{\"name\": \"{}\", \"q\": {}, \"aur_bpp\": {:.4}, \"aur_psnr\": {:.4}, \"jpg_bpp\": {:.4}, \"jpg_psnr\": {:.4}}}",
            r.name, r.q, r.aur_bpp, r.aur_psnr, r.jpg_bpp, r.jpg_psnr
        ));
        if i < results.len() - 1 { json_str.push_str(",\n"); }
    }
    json_str.push_str("\n]");
    fs::write(out_path.join("benchmark_v11.json"), json_str).unwrap();
    
    println!("Benchmark complete. Results saved to benchmark/final_comparison/benchmark_v11.json");
}
