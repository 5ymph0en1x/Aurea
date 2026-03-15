# Aurea

**An image codec built on the golden ratio.**

[![License: MIT](https://img.shields.io/badge/License-MIT-gold.svg)](LICENSE)

Aurea is a lossy image codec that weaves the golden ratio into every stage of compression -- from color decorrelation to quantization to entropy coding. It produces `.aur` files that are competitive with JPEG at medium-to-high quality settings while encoding structural information that block-based codecs discard.

Written in Rust. Ships as a CLI encoder/decoder, a native GUI viewer, and a Windows shell extension with Explorer thumbnails and WIC integration.

---

## How It Works

### Encode

```
RGB --> Golden Color Transform --> 4:2:0 Chroma Subsampling --> CDF 9/7 Wavelets
  --> LL subband: VQ + Paeth prediction + Fibonacci correction
  --> Detail subbands: quantize + Morton Z-order + 2-bit significance map
  --> rANS entropy coding --> .aur file
```

### Decode

```
.aur --> rANS decode --> LL reconstruction (VQ + Fibonacci spin correction)
  --> Detail bands (spectral spin refinement)
  --> CDF 9/7 wavelet recompose --> anti-ring sigma filter
  --> Inverse Golden Color Transform --> RGB
```

---

## Technical Overview

### Golden Color Transform

Standard codecs use YCbCr. Aurea uses a color space derived from the golden ratio:

```
L  = (R + phi * G + phi^-1 * B) / (2 * phi)
C1, C2 = chrominance channels
```

where phi = (1 + sqrt(5)) / 2. This transform exploits the fact that green carries the most perceptual information in natural images, weighting it by phi while blue receives the conjugate weight phi^-1. The result is tighter energy compaction than the traditional ITU-R BT.601 matrix, which means fewer bits spent on residual chrominance.

Chrominance is subsampled 4:2:0, with chroma centroids set to 3/4 of the luminance count.

### Wavelet Decomposition

Aurea uses the Cohen-Daubechies-Feauveau 9/7 biorthogonal wavelet, the same transform at the heart of JPEG 2000. The number of decomposition levels is chosen automatically based on image dimensions: 1 level for images under 256 pixels on a side, 2 for under 4096, and 3 for larger images.

The LL (low-low) subband -- the blurred thumbnail that carries most of the image energy -- is encoded with vector quantization and Paeth spatial prediction. Detail subbands (LH, HL, HH) are scalar-quantized with perceptual weights tuned to human contrast sensitivity: horizontal details (HL) are quantized more aggressively than vertical (LH) or diagonal (HH), reflecting the eye's anisotropic sensitivity to directional artifacts.

### Fibonacci Quantization and Spin Correction

The LL subband undergoes vector quantization where representative centroids are matched using Weber-Fechner psychovisual distance rather than simple Euclidean distance. After quantization, a scan-line interpolation pass called "spin correction" recovers sub-centroid detail:

- **Heisenberg smoothing**: Averages out jagged boundary positions by smoothing fractional interpolation weights across same-label segments.
- **Quantum superposition**: Detects label-oscillation zones near centroid boundaries and reconstructs from interpolation between adjacent centroids instead of forcing a hard assignment.
- **Quantum confinement**: Caps total deviation (residual + spin + dither) at half the centroid spacing to prevent overshoot.
- **Fibonacci dithering**: Adds structured sub-pixel noise using 2D Fibonacci sequences with adaptive amplitude, breaking up any remaining banding.

### Geometric Primitives (v6)

An optional encoding mode extracts segments and circular arcs from the image that oscillate at the natural CDF 9/7 frequency of 1/(2*phi), approximately 0.309 cycles per pixel. Forces along these primitives are quantized to Fibonacci levels. This captures long-range structural correlations that pure wavelet coding misses, saving an additional 0.8% on average across natural images.

Enable with `--geometric` on the command line.

### Detail Band Encoding

Detail coefficients are encoded with a pipeline designed for sparsity:

1. **Dead zone quantization** (dz = 0.15): Coefficients within 0.65 * step of zero are rounded to zero instead of the usual 0.5 * step threshold. This eliminates low-amplitude quantization noise that hurts both PSNR and compression, delivering better quality and smaller files simultaneously.
2. **Morton Z-order scan**: Coefficients are traversed in Z-order (Morton curve) instead of raster order, preserving 2D spatial locality for the entropy coder.
3. **2-bit significance map**: Each coefficient position is encoded as one of four symbols (zero, +1, -1, other), with "other" followed by the actual value. This exploits the fact that over 50% of non-zero detail coefficients are exactly +/-1.
4. **rANS entropy coding**: Asymmetric numeral systems with a Laplacian context model compress the resulting symbol stream.

### Decoder Refinements

Two decoder-side filters improve quality at zero bitrate cost:

- **Spectral spin**: A gentle median filter correction in the wavelet domain (`0.92 * band + 0.08 * median_filter(band, 5)`), averaging +0.35 dB across the Kodak test suite.
- **Anti-ring sigma filter**: A 5x5 edge-aware filter that suppresses ringing near sharp transitions, adding another +0.08 dB on average.

---

## Installation

### Download

Grab the latest release from the [GitHub Releases](../../releases) page:

**aurea-windows-x64.zip** containing:
- `aurea.exe` -- command-line encoder/decoder
- `aurea-viewer.exe` -- GUI image viewer
- `aurea_shell.dll` -- Windows Explorer shell extension
- `install.ps1` / `uninstall.ps1` -- integration scripts

### Build from Source

Requires Rust (edition 2024).

```
cargo build --release
```

Produces `target/release/aurea.exe`, `aurea-viewer.exe`, and `aurea_shell.dll`.

### Windows Integration

Run as Administrator:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

This enables:
- Native thumbnails for `.aur` files in Explorer
- Double-click to open in AUREA Viewer
- Right-click context menu: "Convert to AUREA" on any image
- Windows Photo Viewer and WIC-compatible applications can open `.aur` files

To remove:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\uninstall.ps1
```

---

## Usage

```
aurea encode photo.png -q 75                    # encode at quality 75
aurea encode photo.png output.aur --quality 50  # explicit output and quality
aurea encode photo.png --geometric              # enable v6 geometric primitives
aurea decode image.aur output.png               # decode to PNG
aurea info image.aur                            # show file metadata
```

Quality ranges from 1 (smallest file) to 100 (highest fidelity). The default is 75, which targets a balance comparable to JPEG quality 85.

---

## File Format

| Field | Value |
|---|---|
| Extension | `.aur` |
| Magic bytes | `AURA` (0x41 0x55 0x52 0x41) |
| MIME type | `image/x-aurea` |
| Byte order | Little-endian |

---

## Samples

The `samples/` directory contains example `.aur` files at various quality levels alongside their PNG originals, so you can inspect compression artifacts and file sizes without needing to encode anything yourself.

---

## Project Structure

```
Aurea/
├── src/
│   ├── core/       # Codec library (aurea-core)
│   ├── cli/        # Command-line interface
│   ├── viewer/     # GUI viewer (minifb)
│   └── shell/      # Windows Explorer extension (COM/WIC)
├── scripts/        # Windows install/uninstall
├── samples/        # Example images
└── .github/        # CI/CD workflows
```

---

## Why "Aurea"

From *aurea ratio* -- the golden ratio. The codec is named for the mathematical constant that governs its color transform, its quantization geometry, and its dithering patterns. Whether this makes it a better codec or simply a more beautiful one is left as an exercise for the viewer.

---

## License

MIT. See [LICENSE](LICENSE) for details.
