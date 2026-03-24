# AUREA

**A lossy image codec built on the Golden Ratio, Turing morphogenesis, and rANS entropy coding.**

[![License: MIT](https://img.shields.io/badge/License-MIT-gold.svg)](LICENSE)

AUREA is an experimental image codec that replaces JPEG's 1992-era Huffman tables and fixed 8x8 blocks with a modern pipeline: variable-size Lapped Orthogonal Transform (8/16/32), psychovisual Turing saliency fields, Chroma-from-Luma prediction, and rANS entropy coding with Exp-Golomb magnitudes.

On the standard **Kodak 24** benchmark, AUREA v12 achieves **-5.9% BD-Rate vs JPEG** (22/24 images won), while retaining full **4:4:4 chroma** resolution — no color subsampling, no chroma bleeding.

Written entirely in Rust. Ships as a CLI encoder/decoder, a native GUI viewer, and a Windows shell extension with Explorer thumbnails.

---

<p align="center">
  <img src="samples/alchemist_JPEG-q85.jpg" width="45%" alt="JPEG 4:2:0"/>
  <img src="samples/alchemist_AUREA-q85.png" width="45%" alt="AUREA 4:4:4"/>
</p>

*JPEG (left) vs AUREA (right) at similar bitrate. Notice how AUREA preserves sharp color transitions at full 4:4:4 resolution, while JPEG introduces chroma bleeding from 4:2:0 subsampling.*


<p align="center">
  <img src="samples/rate_distortion.png" />
</p>

---

## Architecture

### 1. Golden Color Transform (GCT)

AUREA uses a color space derived from the golden ratio:

```
L  = (R + phi * G + phi^-1 * B) / (2 * phi)
C1 = B - L    (blue-yellow chroma)
C2 = R - L    (red-cyan chroma)
```

Green receives the phi weight (~0.500), red ~0.309, blue ~0.191 — close to BT.601 but derived from a single constant. The inverse uses only phi^-1 and phi^-2. This decorrelation makes chroma naturally sparse, enabling **4:4:4 encoding** (no subsampling) at competitive bitrates.

A **Perceptual Transfer Function** (PTF, gamma 0.65) is applied to luminance before transform, expanding dark levels to match the Weber-Fechner law of human perception.

### 2. Lapped Orthogonal Transform (LOT)

A variable-size LOT with sine-window lapping replaces the fixed 8x8 DCT:

- **8x8** for dense, high-frequency textures
- **16x16** (default) for general content
- **32x32** for smooth gradients and skies

Block sizes are chosen adaptively: smooth 8x8 cells merge into larger blocks. Each block undergoes a separable 2D DCT-II with precomputed cosine LUT (no runtime trigonometry). The sine window provides overlap-add reconstruction without blocking artifacts.

### 3. Turing Morphogenesis (zero-bit saliency)

A Difference-of-Gaussians saliency field is computed from the DC grid:

```
Activator  = GaussianBlur(Sobel(DC), sigma_a = 1.5)
Inhibitor  = GaussianBlur(Sobel(DC), sigma_i = sigma_a * phi^2)
Turing     = normalize(ReLU(Activator - Inhibitor))
step_mod   = phi^(-0.5 * T_norm)
```

This produces a per-block quantization modulation: edges get finer quantization (preserve structure), smooth regions get coarser (save bits). **Cost: zero bits** — both encoder and decoder compute identical fields from the already-transmitted DC grid.

A **psychovisual pivot** adapts behavior to bitrate: at low quality, gamma > 0 preserves edges; at high quality, gamma < 0 protects smooth areas (anti-banding). The transition uses a cubic smoothstep.

### 4. Quantization

Each AC coefficient receives a custom quantization step:

```
step = detail_step * lot_factor * QMAT[freq] * CSF(freq, luminance)
     * foveal(block) * turing_mod(block) * chroma_factor
```

- **QMAT**: 16x16 frequency weighting matrix, with quality-adaptive power (0.55 at low Q, 0.05 at high Q via smoothstep)
- **CSF**: Contrast sensitivity — dark regions tolerate coarser HF quantization
- **Dead zone**: Quality-adaptive (0.22 at Q<=70, ramps to 0.02 at Q=100), with a frequency-dependent floor for the last 25% of zigzag order (sensor noise suppression)

### 5. Chroma-from-Luma Prediction (CfL)

For each block, a least-squares regression estimates alpha = sum(L*C) / sum(L*L) in the **AC frequency domain** (not spatial):

- Gated by R^2 > 0.25 correlation test
- Alpha quantized to 3 bits (8-value palette from -0.75 to 1.0)
- Chroma residual = C_ac - alpha * L_rec_ac (lower energy, fewer bits)

This exploits the LOT's linearity: LOT(C - alpha*L) = LOT(C) - alpha*LOT(L).

### 6. Entropy Coding (rANS v12)

All streams use **range Asymmetric Numeral Systems** with Exp-Golomb magnitude coding:

| Stream | Encoding |
|--------|----------|
| DC grid | Golden DPCM prediction + rANS v12 |
| AC coefficients | Zigzag scan + EOB truncation + rANS v12 |
| EOB positions | DPCM delta + rANS v12 |
| CfL metadata | Flags + alpha indices packed + rANS v12 |
| Block map | Size codes (0/1/2) + rANS v12 |

The **Golden DPCM** prediction for DC: `pred = (phi^-1 * left + phi^-2 * top + phi^-3 * diag) / sum`.

**Exp-Golomb order 0** for AC magnitudes >= 2: encodes value n as floor(log2(n+1)) zero bits + binary suffix. Far more efficient than unary for the Laplacian-tailed coefficient distribution.

---

## Benchmark Results

### Kodak 24 (768x512) — BD-Rate vs JPEG

| Image | BD-Rate | | Image | BD-Rate |
|-------|--------:|-|-------|--------:|
| kodim01 | -1.6% | | kodim13 | -7.5% |
| kodim02 | +1.6% | | kodim14 | -2.5% |
| kodim03 | -10.4% | | kodim15 | -5.7% |
| kodim04 | -7.5% | | kodim16 | -12.6% |
| kodim05 | -2.2% | | kodim17 | -6.7% |
| kodim06 | -7.0% | | kodim18 | -3.5% |
| kodim07 | -4.7% | | kodim19 | -11.3% |
| kodim08 | +1.7% | | kodim20 | -6.5% |
| kodim09 | -10.4% | | kodim21 | -6.8% |
| kodim10 | -5.3% | | kodim22 | -5.8% |
| kodim11 | -1.1% | | kodim23 | -14.3% |
| kodim12 | -9.1% | | kodim24 | -2.9% |

**Average BD-Rate: -5.9%** (negative = AUREA saves bits at equal PSNR).
**Wins: 22/24 images.**

AUREA qualities tested: 20, 30, 40, 50, 60, 70, 80, 90. JPEG qualities: 10-95.
BD-Rate computed via cubic polynomial fit on log(rate) vs PSNR curves.

---

## Installation

### Download
Grab the latest release from the [Releases](../../releases) page:

**`aurea-windows-x64.zip`** contains:
- `aurea.exe` — Command-line encoder/decoder
- `aurea-viewer.exe` — GUI image viewer
- `aurea_shell.dll` — Windows Explorer shell extension
- `install.ps1` / `uninstall.ps1` — One-click integration

### Build from Source

Requires Rust (edition 2024, MSRV 1.85+).

```bash
cargo build --release --workspace
```

### Windows Shell Integration

Run as Administrator for native `.aur` thumbnails in Explorer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

---

## CLI Usage

```bash
# Encode at default quality (75)
aurea encode photo.png output.aur

# Encode at maximum quality
aurea encode photo.png output.aur -q 95

# Decode back to PNG
aurea decode compressed.aur restored.png

# View file metadata
aurea info compressed.aur
```

Quality ranges from 1 to 100. Default is 75.

---

## Project Structure

```
Aurea/
  src/
    core/             # Core codec library (aurea-core)
      aurea_encoder.rs  # v12 encoder pipeline
      lib.rs            # Decoder routing (v3, v10, v12)
      lot.rs            # Lapped Orthogonal Transform (8/16/32, cosine LUT, rayon)
      rans.rs           # rANS entropy coder (v1 + v12 Exp-Golomb)
      turing.rs         # Turing morphogenesis field (DoG saliency)
      cfl.rs            # Chroma-from-Luma AC-domain prediction
      hierarchy.rs      # Bayesian predictive hierarchy orchestration
      calibration.rs    # Quality-adaptive parameters and calibrated constants
      color.rs          # Golden Color Transform (4:4:4, rayon)
      spin.rs           # Fibonacci spectral spin (decoder refinement)
      dsp.rs            # Signal processing (Gaussian blur, anti-ring)
      golden.rs         # Phi constants and PTF
      scan.rs           # Zigzag and golden spiral scan orders
      scene_analysis.rs # DC-based scene classification
      geometric.rs      # Geometric primitives (phi-frequency superstrings)
      polymerase.rs     # DNA-inspired structural sequencing
    cli/              # Command-line interface
    viewer/           # GUI viewer (native Windows)
    shell/            # Windows Explorer extension (COM/WIC)
  benchmark/          # Test images and benchmark scripts
  docs/               # Architecture specs and design documents
  scripts/            # Windows install/uninstall
```

```mermaid
flowchart TD
    %% --- STYLES ---
    classDef encoder fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef decoder fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef bitstream fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef zeroCost fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,stroke-dasharray: 5 5;
    classDef highlight fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;

    %% ================= ENCODER PIPELINE =================
    subgraph ENCODER ["ENCODER PIPELINE (Left)"]
        direction TB
        E_RGB["RGB Image (W × H × 3)"]:::encoder
        
        %% Stage 1
        E_GCT["S1: Golden Color Transform (GCT)\nL_φ, C1, C2 using φ"]:::highlight
        E_RGB --> E_GCT
        E_GCT --> E_C1["C1 Plane (4:4:4)"]:::encoder
        E_GCT --> E_C2["C2 Plane (4:4:4)"]:::encoder
        E_GCT --> E_PTF["S1: Perceptual Transfer Function\nExpands dark levels"]:::encoder
        E_PTF --> E_Scene["Scene Analysis\n(step_factor)"]:::encoder
        
        %% Stage 2
        E_PTF --> E_Block["S2: Variable Block Classification\n(8x8 to 32x32)"]:::encoder
        E_C1 --> E_Block
        E_C2 --> E_Block
        E_Block --> E_LOT["S2: Lapped Orthogonal Transform\nForward 2D DCT-II"]:::encoder
        
        E_LOT --> E_DC["DC Grid (1 per block)"]:::encoder
        E_LOT --> E_AC["AC Blocks (N²-1 per block)"]:::encoder
        
        %% Stage 3 & 4
        E_DC --> E_DC_Q["S3: Dead-zone Quantization"]:::encoder
        E_DC_Q --> E_DPCM["S3: Golden DPCM Prediction"]:::highlight
        E_DPCM --> E_DC_Entropy["S3: rANS v12 Entropy Coding"]:::encoder
        
        E_DC --> E_Turing["S4: Turing Morphogenesis Field\n(Zero-bit Saliency, DoG)"]:::zeroCost
        E_Turing --> E_Pivot["Psychovisual Turing Pivot\n(Quality adaptive mod)"]:::zeroCost
        
        %% Stage 5
        E_AC --> E_AC_Q["S5: AC Quantization\nFreq, CSF, Foveal, Turing mods"]:::encoder
        E_Pivot --> E_AC_Q
        E_AC_Q --> E_Zigzag["Zigzag Scan & EOB Truncation"]:::encoder
        
        %% Stage 6
        E_Zigzag --> E_CfL["S6: Chroma-from-Luma (CfL)\nPrediction in AC domain"]:::encoder
        
        %% Stage 7
        E_CfL --> E_Entropy["S7: rANS v12 Assembly\n(EOB, AC, CfL, Map)"]:::encoder
    end

    %% ================= BITSTREAM =================
    subgraph BITSTREAM ["AUR2 BITSTREAM (Center)"]
        direction TB
        B_Head["AUR2 Header (39 bytes)\nMagic, Version, Q, Dims"]:::bitstream
        B_Pre["Body Preamble\nChroma dims, Turing Params, Map"]:::bitstream
        B_Ch0["Channel 0: Luma (L_φ)\nDC, EOB, AC"]:::bitstream
        B_Ch1["Channel 1: C1 (Blue-Yellow)\nDC, CfL, EOB, AC"]:::bitstream
        B_Ch2["Channel 2: C2 (Red-Cyan)\nDC, CfL, EOB, AC"]:::bitstream
        
        B_Head --> B_Pre --> B_Ch0 --> B_Ch1 --> B_Ch2
    end

    %% ================= DECODER PIPELINE =================
    subgraph DECODER ["DECODER PIPELINE (Right)"]
        direction TB
        
        %% Stage D1
        D_Parse["D1: Bitstream Parsing"]:::decoder
        
        %% Stage D2
        D_Parse --> D_DC_Dec["D2: rANS v12 Decode (DC)"]:::decoder
        D_DC_Dec --> D_DPCM_Inv["D2: Golden DPCM Inverse"]:::highlight
        D_DPCM_Inv --> D_DC_Recon["D2: Dequantize DC"]:::decoder
        
        %% Stage D3
        D_DC_Recon --> D_Turing["D3: Turing Field Recon\n(Identical to Encoder)"]:::zeroCost
        
        %% Stage D4
        D_Parse --> D_AC_Dec["D4: rANS v12 Decode (AC/EOB)"]:::decoder
        D_AC_Dec --> D_AC_Scatter["D4: Scatter & Zero-fill"]:::decoder
        D_Turing --> D_AC_Dequant["D4: Dequantize AC\n(Exact mirror)"]:::decoder
        D_AC_Scatter --> D_AC_Dequant
        
        %% Stage D5
        D_Parse --> D_CfL_Dec["D5: rANS v12 Decode (CfL)"]:::decoder
        D_AC_Dequant --> D_CfL_Apply["D5: CfL Reconstruction\nC_ac = res + α·L_rec"]:::decoder
        D_CfL_Dec --> D_CfL_Apply
        
        %% Stage D6
        D_DC_Recon --> D_LOT_Inv["D6: LOT Synthesis\nInverse 2D DCT-II"]:::decoder
        D_CfL_Apply --> D_LOT_Inv
        D_LOT_Inv --> D_Overlap["D6: Sine Window &\nOverlap-Add Accumulation"]:::decoder
        
        %% Stage D7
        D_Overlap --> D_PTF_Inv["D7: Inverse PTF"]:::decoder
        D_PTF_Inv --> D_GCT_Inv["D7: Inverse Golden Color Transform"]:::highlight
        D_GCT_Inv --> D_RGB["RGB Image Output"]:::decoder
    end

    %% ================= CONNECTIONS ACROSS COLUMNS =================
    E_DC_Entropy --> B_Ch0
    E_Entropy --> B_Ch0
    
    B_Ch2 --> D_Parse
```

---

## License

MIT. See [LICENSE](LICENSE) for details.
