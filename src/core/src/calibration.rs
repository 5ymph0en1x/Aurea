//! Calibrated constants from 12 HD reference images (Point 8).
//! All empirical thresholds and factors centralized here for systematic tuning.
//!
//! In-the-loop optimization: all tunable constants can be overridden via env vars
//! (prefix AUREA_). When unset, compiled defaults are used.

use std::sync::LazyLock;

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(default)
}

// ======================================================================
// In-the-loop tunable parameters (LazyLock — read env once at startup)
// ======================================================================

/// tRNA step factors: [dark, mid-dark, mid-bright(=1.0 ref), bright]
///
/// **Ce levier est mort.** Son unique consommateur, `lot::luminance_trna`,
/// n'est plus appelé nulle part : il a été remplacé par une loi continue (voir
/// le commentaire de `lot.rs`, « Replaces the stepped codon_structural_map +
/// luminance_trna »), et le décodeur lie `FLAG_WEBER_TRNA` à une variable
/// inutilisée. Le balayage de recalibration du cycle 26 le confirme
/// expérimentalement : `AUREA_TRNA_DARK` et `AUREA_TRNA_BRIGHT` donnent
/// `0,000 %` de BD-rate à *toutes* leurs valeurs.
///
/// Conservé le temps de trancher entre le retrait du code mort et le
/// rebranchement de la loi étagée ; documenté pour qu'une future calibration
/// ne perde pas son temps dessus.
pub static TUNABLE_TRNA: LazyLock<[f64; 4]> = LazyLock::new(|| {
    [
        env_f64("AUREA_TRNA_DARK", 0.7),
        env_f64("AUREA_TRNA_MIDDARK", 0.9),
        1.0,
        env_f64("AUREA_TRNA_BRIGHT", 1.2),
    ]
});

/// qmat_power_for_quality parameters: [low_plateau, high_floor, q_transition, q_range]
pub static TUNABLE_POWER: LazyLock<[f64; 4]> = LazyLock::new(|| {
    [
        env_f64("AUREA_POWER_LOW", 0.55),
        env_f64("AUREA_POWER_HIGH", 0.05),
        env_f64("AUREA_POWER_QTRANS", 60.0),
        env_f64("AUREA_POWER_QRANGE", 40.0),
    ]
});

/// Psychovisual Turing pivot: [gamma_low, gamma_high]
pub static TUNABLE_PIVOT: LazyLock<[f64; 2]> = LazyLock::new(|| {
    [
        env_f64("AUREA_PIVOT_GAMMA_LOW", 0.80),
        env_f64("AUREA_PIVOT_GAMMA_HIGH", -0.13),
    ]
});

/// Foveal saliency exponent. The dense 2-D Turing field already captures the
/// useful spatial allocation; Kodak 24 recalibration selected a neutral
/// additional foveal exponent.
pub static TUNABLE_FOVEAL_K: LazyLock<f64> = LazyLock::new(|| env_f64("AUREA_FOVEAL_K", 0.0));

/// Turing step-modulation exponent written into the v13 bitstream header.
/// The decoder reads the serialized value, so this override is encoder-side
/// only and remains self-contained in every generated file.
pub static TUNABLE_TURING_K_STEP: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_TURING_K_STEP", TURING_K_STEP));

/// Cycle 27 — key the AC entropy contexts on intra-block activity instead of
/// the Turing bucket.
///
/// The Turing field is a Sobel of the DC grid, so it measures how a block
/// differs from its neighbours and is blind to the structure inside it.
/// Measured at q40 over the corpus (248 880 blocks), rank correlation with
/// per-block squared error: +0.389 for the DC gradient, +0.887 for intra-block
/// activity. The EOB position *is* that activity, is already transmitted, and
/// is decoded before any AC coefficient — so the better signal costs no bits.
///
/// Streams that use it are marked AUR2 v25, so a reader never has to guess.
/// Set `AUREA_ACTIVITY_CONTEXT=0` to emit the v24 grammar as an A/B witness.
pub static TUNABLE_ACTIVITY_CONTEXT: LazyLock<bool> =
    LazyLock::new(|| env_f64("AUREA_ACTIVITY_CONTEXT", 1.0) != 0.0);

/// Cycle 27 — open each progressive detail layer on the echo of its own ping.
///
/// Layer 0 carries the first `PROGRESSIVE_SKELETON_COEFFS` coefficients of every
/// block and is fully decoded before any detail layer, but every layer opens a
/// fresh rANS stream with flat tables. So a block's detail coefficients were
/// coded knowing nothing about the low frequencies already delivered for that
/// same block — information both sides already hold, for free.
///
/// `rans::skeleton_echo` replays the coder's own energy IIR over the decoded
/// skeleton, and the detail layer restarts from that value instead of a flat
/// constant. Streams that use it are marked AUR2 v26. Set
/// `AUREA_ECHO_CONTEXT=0` to emit v25 as an A/B witness.
pub static TUNABLE_ECHO_CONTEXT: LazyLock<bool> =
    LazyLock::new(|| env_f64("AUREA_ECHO_CONTEXT", 1.0) != 0.0);

/// Cycle 27 — let each progressive detail layer re-enter the channel that
/// layer 0 has already ionized.
///
/// Every layer used to open a fresh rANS stream with flat tables, so the coder
/// broke down virgin air at the start of each one. Lightning does not: a return
/// stroke reuses the ionized channel, which is why it is faster and carries
/// more. Carrying layer 0's converged tables into each detail layer is the same
/// move — deterministic, and safe under progressive truncation because layer 0
/// is always present.
///
/// Streams that use it are marked AUR2 v27. Set `AUREA_CHANNEL_MEMORY=0` to
/// emit v26 as an A/B witness.
pub static TUNABLE_CHANNEL_MEMORY: LazyLock<bool> =
    LazyLock::new(|| env_f64("AUREA_CHANNEL_MEMORY", 1.0) != 0.0);

/// Experimental zero-bit homeostasis applied after the psychovisual pivot.
/// 0.0 preserves the calibrated v13 behaviour exactly; 1.0 recentres the
/// coefficient-weighted geometric mean of the per-block modulation on 1.0.
/// The same value must be used by the encoder and decoder during experiments.
pub static TUNABLE_TURING_HOMEOSTASIS: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_TURING_HOMEOSTASIS", 0.0).clamp(0.0, 1.0));

/// Gradient strength threshold for QMAT rotation (0-1, normalized).
/// The real 2-D field produces stable directions even at low normalized
/// strength, so calibration enables rotation for every non-flat block.
pub static TUNABLE_GRADIENT_THRESHOLD: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_GRADIENT_THRESHOLD", 0.0));

/// Experimental interpolation from the legacy vector-averaged direction
/// (0.0) to a structure-tensor direction whose QMAT influence is attenuated
/// by local coherence (1.0). Encoder and decoder must use the same value.
pub static TUNABLE_DIRECTIONAL_COHERENCE: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_DIRECTIONAL_COHERENCE", 0.0).clamp(0.0, 1.0));

/// Experimental contribution of a second Turing scale whose activator sigma
/// is multiplied by phi. Zero is the exact single-scale v13 path.
pub static TUNABLE_TURING_MULTISCALE: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_TURING_MULTISCALE", 0.0).clamp(0.0, 1.0));

/// Self-described spatial AC scan: 0=raster, 1=golden-angle phyllotaxis,
/// 2=continuous fixed-pitch spiral arc. The encoder writes the selected mode
/// into existing v13 feature flags, so decoding needs no environment override.
pub static TUNABLE_SPIRAL_SCAN_MODE: LazyLock<u8> = LazyLock::new(|| {
    env_f64("AUREA_SPIRAL_SCAN_MODE", 0.0)
        .round()
        .clamp(0.0, 2.0) as u8
});

/// Experimental saccadic-surprise strength along the continuous spiral arc.
/// Zero is an exact pass-through. During laboratory sweeps the same value must
/// be supplied to encoder and decoder; promotion would serialize the winner.
pub static TUNABLE_SPIRAL_FIELD_STRENGTH: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_SPIRAL_FIELD_STRENGTH", 0.0).clamp(0.0, 1.0));

/// Self-described spectral coefficient scan: 0=zigzag, 1=fixed-pitch folded
/// spiral rooted at the DC coefficient.
pub static TUNABLE_SPECTRAL_SCAN_MODE: LazyLock<u8> = LazyLock::new(|| {
    env_f64("AUREA_SPECTRAL_SCAN_MODE", 0.0)
        .round()
        .clamp(0.0, 1.0) as u8
});

/// Self-described progressive AC packetization. Each packet follows one fixed
/// spiral arc, while its blocks are entropy-coded in raster order. Promoted by
/// the Kodak 24 validation; zero keeps the frozen monolithic stream available
/// as a laboratory control.
pub static TUNABLE_PROGRESSIVE_PACKETS: LazyLock<bool> =
    LazyLock::new(|| env_f64("AUREA_PROGRESSIVE_PACKETS", 1.0) >= 0.5);

/// Promoted zero-bit chromatic breathing. The encoder quantizes this
/// strength to one byte in the bitstream, so the decoder never depends on an
/// environment override. Zero is an exact pass-through.
pub static TUNABLE_CHROMA_BREATHING: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_CHROMA_BREATHING", 1.0).clamp(0.0, 1.0));

/// Trellis RDO lambda scale factor (higher = more aggressive zeroing).
///
/// Recalibrated at cycle 26 from `5.0` to `16.0`. The original value was fitted
/// against a v20-era entropy grammar; four cycles of context mixing later the
/// coder is ~2.6 % leaner, so the rate a coefficient really costs — and with it
/// the rate-distortion optimum — has moved.
///
/// `16.0` is the last value that is a **strict Pareto improvement**: smaller
/// files *and* higher PSNR at every operating point. Beyond it the codec merely
/// slides along its own RD curve (at `48.0`, `-14.6 %` bpp for `-0.87` dB at
/// q20), which a BD-rate computed over shifted rate ranges reports as a gain it
/// is not. Raw rate/PSNR points, not BD-rate alone, decided this value.
pub static TUNABLE_TRELLIS_LAMBDA: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_TRELLIS_LAMBDA", 16.0));

/// Laboratory knob: C1 (blue-yellow) chroma step factor. Default is the
/// historical PHI. NOT self-described — the same value must be provided to
/// encoder and decoder during experiments.
pub static TUNABLE_CHROMA_PHI: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_CHROMA_PHI", crate::golden::PHI));

/// Strength of the zero-bit C1 repair for diffuse chroma (v15). The 0.85
/// Pareto point balances Cb and Cr on the detected field; zero is the v14
/// witness.
pub static TUNABLE_CHROMA_DIFFUSION_REPAIR: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_CHROMA_DIFFUSION_REPAIR", 0.85).clamp(0.0, 1.0));

/// Zero-bit bright-compensation strength on the luma steps (« contre-don de
/// Weber », promoted 2026-08-14). Self-described: the encoder quantizes the
/// strength to 5 bits stored in the scene-type byte's upper bits, so the
/// decoder always uses the file's value. The witness
/// `AUREA_BRIGHT_COMPENSATION=0` reproduces the previous profile bit-exactly.
pub static TUNABLE_BRIGHT_COMPENSATION: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_BRIGHT_COMPENSATION", 1.0).clamp(0.0, 1.0));

/// v15 multiplier applied after decoding the self-described bright strength.
/// The 2.5 calibration is the tested upper Weber point; 1 is the v14 witness.
pub static TUNABLE_BRIGHT_COMPENSATION_GAIN: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_BRIGHT_COMPENSATION_GAIN", 2.5).clamp(0.0, 2.5));

/// Zero-bit spectral exchange (« sous-échantillonnage doux », promoted
/// 2026-08-14, cycle 12): coarser chroma HF, finer luma, gated per image by
/// the chroma DC activity. Binary at encode time: >= 0.5 emits an AUR2 v16
/// stream with the exchange on (eta = 1); the witness
/// `AUREA_SPECTRAL_EXCHANGE=0` emits a v13 stream, bit-exact with the
/// cycle-11 profile. Decoding follows the file version, never this knob.
pub static TUNABLE_SPECTRAL_EXCHANGE: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_SPECTRAL_EXCHANGE", 1.0).clamp(0.0, 1.0));

/// AUR2 v18 spectral-exchange semantics. The luma counter-gift is stronger
/// than the chroma gift: the native-corpus frontier moved from +3.331% to
/// -0.410% BD-rate Y against optimized JPEG 4:2:0 while RGB stayed on its
/// Pareto frontier. These values are part of the format, not process state.
pub const SPECTRAL_LUMA_STRENGTH: f64 = 1.75;

/// Cycle 26 — recalibration knobs for the psychovisual levers.
///
/// Every one of these was fitted against a v20-era entropy grammar and a DC
/// that carried a systematic downward bias. Both are gone: the coder is ~2.6 %
/// leaner and the DC is unbiased. A point that was Pareto-optimal then is not
/// necessarily optimal now, so each is made sweepable.
///
/// `AUREA_SPECTRAL_LUMA` in particular was set to `1.75` at cycle 17 to buy
/// back a luminance deficit of `+3.33 %` against JPEG 4:2:0. Luminance now
/// stands at `-3.92 %`; the chroma it was paying with may be worth reclaiming.
/// It is symmetric — encoder and decoder must agree — so a promotion would be
/// a format change.
pub static TUNABLE_SPECTRAL_LUMA: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_SPECTRAL_LUMA", SPECTRAL_LUMA_STRENGTH).clamp(0.0, 4.0));
pub static TUNABLE_SPECTRAL_CHROMA: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_SPECTRAL_CHROMA", SPECTRAL_CHROMA_STRENGTH).clamp(0.0, 4.0));

/// Laboratory knob: adaptive relief of the luma counter-gift on diffuse-chroma
/// images. The v18 luma strength (1.75) is uniform, calibrated to win the
/// *mean* Y frontier; on the one field where the diffuse-chroma signature
/// fires (kryte: repair = 0.94, zero elsewhere on `originals`), luminance is
/// already won by a wide margin while Cb/Cr/Delta-E pay the bill. The
/// effective strength becomes `(TUNABLE_SPECTRAL_LUMA - relief * repair)`
/// floored at 1.0 (the v17 baseline). The signature is reconstructed from the
/// decoded chroma DC grids on both sides, so the field stays zero-bit.
/// Symmetric — encoder and decoder must agree; `0` reproduces v27 bit-exactly.
/// Promoted AUR2 v28 defaults (cycle 30): full luma relief to the v17 floor,
/// chroma slope relief at the minimax point of the measured window
/// [0.13, 0.20] where every kryte metric beats both JPEG profiles.
pub const LUMA_PIVOT_RELIEF: f64 = 1.0;
pub const SPECTRAL_CHROMA_RELIEF: f64 = 0.15;

pub static TUNABLE_LUMA_PIVOT_RELIEF: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_LUMA_PIVOT_RELIEF", LUMA_PIVOT_RELIEF).clamp(0.0, 2.0));

/// Companion of the luma-pivot relief on the chroma side: on the same
/// `R * G` field (diffuse chroma × dark Weber surplus) the spectral-exchange
/// chroma coarsening slope is scaled by `1 - relief * R * G`, handing the HF
/// gift back where it was taken. The balancing luma delta shrinks with it, so
/// the AC-weighted log-mean of all step multipliers stays exactly zero.
/// Symmetric; `0` reproduces the luma-relief-only behaviour bit-exactly.
pub static TUNABLE_SPECTRAL_CHROMA_RELIEF: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_SPECTRAL_CHROMA_RELIEF", SPECTRAL_CHROMA_RELIEF).clamp(0.0, 1.0));

/// Laboratory gates on the decoder post chain (deblock + anti-ring). The
/// encoder never simulates the post chain, so these only shape the decoded
/// output — but the same file must decode identically everywhere, so any
/// non-default value is measurement-only and a default change is a format
/// (version) change. `AUREA_ANTI_RING_SCALE` multiplies the step handed to
/// the sigma filter (0 disables it); `AUREA_DEBLOCK=0` skips deblocking.
pub static TUNABLE_ANTI_RING_SCALE: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_ANTI_RING_SCALE", 1.0).clamp(0.0, 4.0));
pub static TUNABLE_DEBLOCK: LazyLock<bool> =
    LazyLock::new(|| env_f64("AUREA_DEBLOCK", 1.0) >= 0.5);
pub static TUNABLE_CAS_SHARPEN: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_CAS_SHARPEN", 0.0).clamp(0.0, 1.0));
pub static TUNABLE_XSHARPEN: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_XSHARPEN", 0.0).clamp(0.0, 1.0));
pub static TUNABLE_XSHARPEN_THRESH: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_XSHARPEN_THRESH", 0.03).clamp(0.0, 0.5));
pub static TUNABLE_XSHARPEN_CHROMA: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_XSHARPEN_CHROMA", 0.0).clamp(0.0, 1.0));
/// Second XSharpen pass at radius 1 after the main pass — a two-scale
/// sharpening: broad punch first, fine punch on top.
pub static TUNABLE_XSHARPEN2: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_XSHARPEN2", 0.0).clamp(0.0, 1.0));
/// Deterministic luma grain synthesis, in units of the detail step. A hash of
/// the pixel coordinates drives the sign, so the field is identical on every
/// platform and needs no transmitted state. Measured 2026-08-16: white noise
/// reads as noise, not texture — LPIPS degrades at every useful amplitude.
/// Kept as a documented negative.
pub static TUNABLE_GRAIN: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_GRAIN", 0.0).clamp(0.0, 1.0));
pub static TUNABLE_GRAIN_DARK_EXP: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_GRAIN_DARK_EXP", 0.0).clamp(0.0, 8.0));
/// Encoder-only trellis-lambda modulation by block AC activity:
/// `lambda *= (activity / geomean)^k`. Removed at cycle 27 as MSE-neutral;
/// reinstated as a *perceptual* lever — negative `k` keeps texture in busy
/// blocks (lower lambda) financed by more zeroing in smooth ones. The lambda
/// is never mirrored by the decoder, so this is free of any format concern.
pub static TUNABLE_LAMBDA_ACTIVITY: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_LAMBDA_ACTIVITY", 0.0).clamp(-2.0, 2.0));

/// Encoder-only global scale on the detail step. The step is carried in the
/// header (f32), so the decoder follows the file automatically — this knob
/// renormalizes the rate after a redistribution lever (e.g. reduced bright
/// compensation) so rate-matched comparisons keep their anchor.
pub static TUNABLE_STEP_SCALE: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_STEP_SCALE", 1.0).clamp(0.25, 4.0));

/// AC dequantization offset in bin units: nonzero coefficients reconstruct at
/// `(|q| + delta) * step` instead of `q * step`. Positive values restore the
/// texture energy quantization removed, exactly where it was (zeros stay
/// zero). Symmetric — the encoder mirrors it in its own reconstruction (CfL,
/// codon RDO) — so any non-zero default is a format change.
pub static TUNABLE_DEQUANT_OFFSET: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_DEQUANT_OFFSET", 0.0).clamp(-0.5, 0.5));

/// Linear unsharp-mask amount on luma (1-2-1 separable blur as the low-pass).
/// Unlike XSharpen it adds overshoot around edges — the ringing energy that
/// block-DCT codecs naturally carry and that texture metrics read as detail.
pub static TUNABLE_USM: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_USM", 0.0).clamp(0.0, 2.0));
/// Number of 1-2-1 blur passes in the USM low-pass (wider blur → broader
/// overshoot that survives the metric grid's downscale).
pub static TUNABLE_USM_RADIUS: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("AUREA_USM_RADIUS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 4)
});
pub static TUNABLE_USM_CHROMA: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_USM_CHROMA", 0.0).clamp(0.0, 2.0));
pub static TUNABLE_XSHARPEN_RADIUS: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("AUREA_XSHARPEN_RADIUS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, 4)
});
/// Multiplier on the AC dead zone. Encoder-only, recalibrated to `0.0` at
/// cycle 26 — the AC dead zone is now off.
///
/// A dead zone and a trellis both decide which coefficients become zero: the
/// first blindly, on a fixed magnitude threshold, the second by rate-distortion
/// using the entropy model's real cost. Four cycles of context mixing made the
/// second steadily more accurate and left the first frozen at a threshold
/// calibrated for a coder that no longer exists.
///
/// Replacing blind zeroing with informed zeroing turns out not to be a
/// trade-off at all: it yields fewer bits *and* higher PSNR at every quality.
/// `1.0` restores the historical behaviour as an A/B witness.
pub static TUNABLE_DEAD_ZONE_SCALE: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_DEAD_ZONE_SCALE", 0.0).clamp(0.0, 3.0));
pub const SPECTRAL_CHROMA_STRENGTH: f64 = 1.0;

/// Encoder-only laboratory weights for the trellis distortion term in the
/// causal Lphi/C1/C2 order. The promoted v18 witness is exactly (1, 1, 1).
/// Non-neutral values do not affect decoding or require side information:
/// they only change which already-representable quantized coefficients the
/// encoder selects. These knobs are intentionally separate while the true
/// RGB/Y quadratic codon objective is being calibrated.
pub static TUNABLE_TRELLIS_LUMA_DISTORTION: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_TRELLIS_LUMA_DISTORTION", 1.0).clamp(0.05, 8.0));
pub static TUNABLE_TRELLIS_C1_DISTORTION: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_TRELLIS_C1_DISTORTION", 1.0).clamp(0.05, 8.0));
pub static TUNABLE_TRELLIS_C2_DISTORTION: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_TRELLIS_C2_DISTORTION", 1.0).clamp(0.05, 8.0));

/// Encoder-only joint color-codon RDO. C1 is refined after Lphi, then C2 is
/// refined against the exactly reconstructed `(Lphi, C1)` pair. This adds no
/// side data and does not change the decoder. Zero is an exact pass-through.
pub static TUNABLE_VECTOR_RDO_STRENGTH: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_VECTOR_RDO_STRENGTH", 1.0).clamp(0.0, 1.0));

/// Additional BT.601-luminance weight inside the joint RGB quadratic.
pub static TUNABLE_VECTOR_RDO_Y_WEIGHT: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_VECTOR_RDO_Y_WEIGHT", 4.0).clamp(0.0, 8.0));

/// Exp-Golomb magnitude-payload rate penalty used by the local refinement.
/// A hard guard separately forbids candidates with a longer payload.
pub static TUNABLE_VECTOR_RDO_RATE_WEIGHT: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_VECTOR_RDO_RATE_WEIGHT", 1.0).clamp(0.0, 4.0));

/// Symmetric laboratory witness for phase-aware low-frequency luma
/// allocation. Zero is the promoted v18 profile bit-exactly. Non-zero values
/// require the same environment at encode and decode time and therefore may
/// only be promoted with new fixed bitstream semantics.
pub static TUNABLE_SUBSONIC_WAVE_STRENGTH: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_SUBSONIC_WAVE_STRENGTH", 0.0).clamp(0.0, 1.5));

/// Cycle 20 — enveloppes amplitude/phase par voisinage spectral (AUR2 v21).
///
/// Par défaut le profil promu est actif et le fichier est entièrement
/// auto-décrit : la version 21 de l'en-tête suffit au décodeur.
///
/// `AUREA_SPECTRAL_ENVELOPE=0` ramène exactement à la grammaire v20 et produit
/// un fichier v20 : c'est le commutateur A/B du laboratoire, pas un réglage.
///
/// Les leviers fins ci-dessous restent disponibles pour prolonger le cycle.
/// Toute valeur qui s'écarte du profil promu doit être fournie identiquement à
/// l'encodeur et au décodeur, comme pour toute expérience non promue.
///
/// - `AUREA_SPECTRAL_ENVELOPE_AMPLITUDE` : 0 parent v16, 1 parent x support
///   local, 2 somme des magnitudes, 3 nombre de voisins actifs, 4 somme des
///   magnitudes rapportée à l'énergie courante du modèle.
/// - `AUREA_SPECTRAL_ENVELOPE_PHASE` : masque des classes de phase autorisées.
/// - `AUREA_SPECTRAL_ENVELOPE_CHANNELS` : masque des canaux (bit 0 = Lphi).
/// - `AUREA_SPECTRAL_ENVELOPE_PRIOR` : `P(accord)` initial, sur 14 bits.
/// - `AUREA_SPECTRAL_ENVELOPE_SHIFT` : vitesse d'adaptation `1 / 2^shift`.
/// - `AUREA_SPECTRAL_ENVELOPE_PRECEDENCE` : la phase spectrale prime sur le
///   prédicteur historique au lieu de le compléter.
/// - `AUREA_SPECTRAL_ENVELOPE_PM1` : l'axe d'amplitude sépare aussi `P(|x|=1)`.
/// - `AUREA_SPECTRAL_ENVELOPE_MIX` : taux d'apprentissage du mélangeur, `0`
///   désactive le mélange et produit un flux v21.
/// - `AUREA_SPECTRAL_ENVELOPE_MIX_INPUTS` : masque des avis secondaires —
///   bit 0 voisinage de Lphi, bit 1 enveloppe rapportée à l'échelle, bit 2
///   bande fréquentielle explicite, bit 3 taille de transformée (mesurée
///   perdante), bit 4 comptage de voisins actifs.
/// - `AUREA_SPECTRAL_ENVELOPE_MIX_CHANNELS` : canaux où le mélange est actif.
/// - `AUREA_SPECTRAL_ENVELOPE_MIX_SIGN` : mélanger aussi l'accord de phase.
/// - `AUREA_SPECTRAL_ENVELOPE_MIX_PM1` : mélanger aussi `P(|x|=1)`.
/// - `AUREA_SPECTRAL_ENVELOPE_APM` : vitesse d'affinage APM après le mélange,
///   `0` désactive.
/// - `AUREA_SPECTRAL_ENVELOPE_COUPLING` : contexte des compteurs d'accord de
///   phase — 0 aucun, 1 énergie du voisinage sur toutes les classes, 2 mémoire
///   intra-bloc de la loi séparable, 3 bucket de Turing du bloc, 4 énergie du
///   voisinage sur la seule classe séparable.
pub static TUNABLE_SPECTRAL_ENVELOPE: LazyLock<crate::spectral_envelope::SpectralEnvelopeConfig> =
    LazyLock::new(|| {
        use crate::spectral_envelope::SpectralEnvelopeConfig;
        let promoted = SpectralEnvelopeConfig::promoted();
        if env_f64("AUREA_SPECTRAL_ENVELOPE", 1.0) < 0.5 {
            return SpectralEnvelopeConfig::neutral();
        }
        let mut config = promoted;
        config.amplitude_mode = env_f64(
            "AUREA_SPECTRAL_ENVELOPE_AMPLITUDE",
            promoted.amplitude_mode as f64,
        )
        .clamp(0.0, 3.0)
        .round() as u8;
        config.phase_mask = env_f64("AUREA_SPECTRAL_ENVELOPE_PHASE", promoted.phase_mask as f64)
            .clamp(0.0, 1023.0)
            .round() as u16;
        config.phase_channels = env_f64(
            "AUREA_SPECTRAL_ENVELOPE_CHANNELS",
            promoted.phase_channels as f64,
        )
        .clamp(0.0, 7.0)
        .round() as u8;
        config.phase_initial_p_same = env_f64(
            "AUREA_SPECTRAL_ENVELOPE_PRIOR",
            promoted.phase_initial_p_same as f64,
        )
        .clamp(1.0, 16383.0)
        .round() as u32;
        config.phase_adapt_shift = env_f64(
            "AUREA_SPECTRAL_ENVELOPE_SHIFT",
            promoted.phase_adapt_shift as f64,
        )
        .clamp(1.0, 15.0)
        .round() as u32;
        config.phase_takes_precedence = env_f64(
            "AUREA_SPECTRAL_ENVELOPE_PRECEDENCE",
            f64::from(u8::from(promoted.phase_takes_precedence)),
        ) >= 0.5;
        config.amplitude_splits_pm1 = env_f64(
            "AUREA_SPECTRAL_ENVELOPE_PM1",
            f64::from(u8::from(promoted.amplitude_splits_pm1)),
        ) >= 0.5;
        config.phase_context = env_f64(
            "AUREA_SPECTRAL_ENVELOPE_COUPLING",
            promoted.phase_context as f64,
        )
        .clamp(0.0, 5.0)
        .round() as u8;
        config.mix_rate = env_f64("AUREA_SPECTRAL_ENVELOPE_MIX", promoted.mix_rate as f64)
            .clamp(0.0, 32.0)
            .round() as u8;
        config.mix_inputs = env_f64(
            "AUREA_SPECTRAL_ENVELOPE_MIX_INPUTS",
            promoted.mix_inputs as f64,
        )
        .clamp(0.0, 31.0)
        .round() as u8;
        config.mix_channels = env_f64(
            "AUREA_SPECTRAL_ENVELOPE_MIX_CHANNELS",
            promoted.mix_channels as f64,
        )
        .clamp(0.0, 7.0)
        .round() as u8;
        config.mix_sign = env_f64(
            "AUREA_SPECTRAL_ENVELOPE_MIX_SIGN",
            f64::from(u8::from(promoted.mix_sign)),
        ) >= 0.5;
        config.mix_pm1 = env_f64(
            "AUREA_SPECTRAL_ENVELOPE_MIX_PM1",
            f64::from(u8::from(promoted.mix_pm1)),
        ) >= 0.5;
        config.apm_rate = env_f64("AUREA_SPECTRAL_ENVELOPE_APM", promoted.apm_rate as f64)
            .clamp(0.0, 15.0)
            .round() as u8;
        config
    });

/// Cycle 21 — redistribution locale de l'échange spectral (`AUREA_SPECTRAL_LOCAL`).
///
/// Le gamma global répond à « combien de chroma cette *image* peut céder ».
/// À l'intérieur d'une même image la réponse est rarement uniforme. Ce levier
/// recalcule le même rapport d'activité chroma/luma sur une fenêtre locale de
/// cellules DC, puis renormalise pour que la moyenne pondérée AC des gammas
/// par bloc soit exactement le gamma global : le budget total ne bouge pas,
/// seule sa répartition change.
///
/// `0` reproduit la v21 bit-exactement. Toute valeur non nulle doit être
/// fournie identiquement à l'encodeur et au décodeur tant que l'expérience
/// n'est pas promue.
pub static TUNABLE_SPECTRAL_LOCAL: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_SPECTRAL_LOCAL", 0.0).clamp(0.0, 1.0));

/// Cycle 25 — part de la zone morte AC réellement appliquée au DC.
///
/// `DEAD_ZONE` est documentée dans `wavelet.rs` comme le seuil de la
/// quantification **AC** : les coefficients AC sont centrés et épars, les
/// annuler est souhaitable. Le DC normalisé est au contraire strictement
/// positif et dense — `sign` y vaut toujours `+1` — de sorte que la zone morte
/// n'y annule rien mais **rabote chaque bloc vers le bas** de
/// `dead_zone × dc_step`. Le résultat est un assombrissement systématique
/// proportionnel au pas, mesuré à `-0,18` à `-0,43` en espace PTF selon
/// l'image et la qualité.
///
/// La correction est purement côté encodeur — la déquantification ne change
/// pas — donc elle n'entraîne aucune rupture de format : les décodeurs
/// existants lisent les nouveaux fichiers sans modification.
///
/// `0.0` (défaut promu) quantifie le DC au plus proche ; `1.0` restaure le
/// comportement historique bit-exactement et sert de témoin A/B.
pub static TUNABLE_DC_DEAD_ZONE: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_DC_DEAD_ZONE", 0.0).clamp(0.0, 1.0));

/// Cycle 25 — compensation de Jensen sur le DC de luminance.
///
/// La PTF inverse `g(P) = 255·(P/255)^(1/0.65)` est **convexe**, donc la
/// moyenne linéaire d'un bloc ne vaut pas `g(moyenne)` mais
/// `g(D) + ½·g''(D)·V`, où `V` est la variance intra-bloc en espace PTF. Le
/// codec modifie `V` de deux façons opposées : il **retire** l'énergie des
/// coefficients annulés et **ajoute** le bruit d'arrondi des survivants. La
/// reconstruction se retrouve donc décalée de `½·g''(D)·(V_rec − V_src)`, vers
/// le sombre là où la texture domine, vers le clair là où le bruit domine.
///
/// La distribution mesurée le confirme : `+0,179` dans les ombres profondes,
/// `-0,367` dans les hautes lumières, quasi nul au milieu.
///
/// La correction relève le DC de `δ = ½·(g''/g')(D)·(V_src − V_rec)`, et
/// puisque `g''/g' = (1/γ − 1)/P`, elle s'écrit `δ = (1/γ − 1)·ΔV/(2D)`. Elle
/// est appliquée avant la quantification du DC, donc en une seule passe et
/// sans rupture de format. La constante absorbe la normalisation de la LOT.
///
/// `0.0` désactive la compensation bit-exactement.
pub static TUNABLE_DC_JENSEN: LazyLock<f64> =
    LazyLock::new(|| env_f64("AUREA_DC_JENSEN", 0.0).clamp(0.0, 4.0));

/// Compute the Lagrange multiplier for trellis RDO.
/// Based on λ ~ step² / (12·ln2) with quality-adaptive scaling.
/// Low quality: aggressive RDO (λ_scale=1.5), high quality: gentle (λ_scale=0.3).
pub fn trellis_lambda(quality: u8, detail_step: f64, texture_law: bool) -> f64 {
    let base = detail_step * detail_step / 8.33; // 12 * ln(2) ≈ 8.33
    let q = quality as f64;
    let q_scale = if q <= 50.0 {
        1.5
    } else if q >= 90.0 {
        0.3
    } else {
        let t = ((q - 50.0) / 40.0).clamp(0.0, 1.0);
        let s = t * t * (3.0 - 2.0 * t); // smoothstep
        1.5 * (1.0 - s) + 0.3 * s
    };
    // v29 texture law: at low qualities the MSE-Pareto lambda (16) zeroes the
    // texture DISTS/LPIPS live on; 8 keeps it. Env override wins (laboratory).
    let lambda_mult = env_override("AUREA_TRELLIS_LAMBDA").unwrap_or(
        if texture_law && quality <= V29_TEXTURE_MAX_QUALITY {
            V29_TRELLIS_LAMBDA_TEXTURE
        } else {
            16.0
        },
    );
    (base * q_scale * lambda_mult).max(0.01)
}

// ======================================================================
// AUR2 v29 — perceptual texture law (cycle 31)
// ======================================================================
// Calibrated on the rate-matched perceptual harness (LPIPS/DISTS vs
// size-matched JPEG, `scripts/perceptual_ab.py`): at low qualities the
// codec's MSE-optimal machinery starves fine texture (dark pebbles of
// `phoenix`), which sharpening alone cannot re-invent; at high qualities
// the anti-ring sigma filter destroys legitimate detail on every axis.
// The law is keyed on the header quality; env knobs override it for A/B.
pub const V29_TEXTURE_MAX_QUALITY: u8 = 30;
pub const V29_ANTI_RING_OFF_QUALITY: u8 = 70;
pub const V29_XSHARPEN: f64 = 0.4;
pub const V29_XSHARPEN_THRESH: f64 = 0.025;
pub const V29_XSHARPEN_RADIUS: usize = 2;
/// Local-contrast floor of the sharpening (0-1 units): neighbourhoods whose
/// range is below this are gentle gradients — sharpening them manufactures
/// banding (measured on `chilling`) instead of restoring texture.
pub const V29_XSHARPEN_MIN_RANGE: f64 = 0.04;
pub const V29_USM: f64 = 0.15;
pub const V29_USM_RADIUS: usize = 2;
/// USM coring (plane units, 0-255): |v - blur| below this is gradient or
/// banding step, not texture — it passes through un-amplified.
pub const V29_USM_CORING: f64 = 0.75;
pub const V29_TRELLIS_LAMBDA_TEXTURE: f64 = 8.0;
pub const V29_LAMBDA_ACTIVITY_TEXTURE: f64 = 1.0;
pub const V29_C2_DISTORTION_TEXTURE: f64 = 1.75;

/// AUR2 v30 — EOB-driven luma step field (cycle 32). Strength of the
/// zero-bit modulation `phi^(-eta * (activity - mean))` derived from the
/// transmitted EOB positions. Applied only above `V30_EOB_STEP_MIN_QUALITY`:
/// below it the texture law owns the allocation and the showcase anchor
/// must not move. `0` while the sweep decides the promotion.
pub const V30_EOB_STEP_ETA: f64 = 1.0;
pub const V30_EOB_STEP_MIN_QUALITY: u8 = 31;

/// Cycle 36 — one-pass EOB scheme: pass 1 of the modulation field uses the
/// greedy quantizer's EOBs instead of a full trellis run. Encoder-only
/// stream freedom (no format change), but the stream differs, so the
/// default only flips after the full frontier + showcase re-validation.
pub const V33_EOB_ONE_PASS: bool = true;

/// AUR2 v31 — the EOB step field extended to the chroma channels (cycle 33).
/// Each channel's own transmitted EOBs drive its field; the same quality
/// gate applies. `0` while the sweep decides the promotion.
pub const V31_EOB_STEP_CHROMA_ETA: f64 = 1.0;
/// The chroma field only fires where the spectral exchange actually took the
/// chroma budget (high gamma: flat-chroma images like kryte 0.68, honey
/// 0.81). Chroma-structured images (dispair 0.52, phoenix 0.50, agree 0.27)
/// keep their luminance intact — the field's rate cost was measured to push
/// dispair's thin Y margin back over zero. Gamma is stable across qualities
/// to ±0.0006, and both sides already compute it from the decoded DC grids.
pub const V31_EOB_CHROMA_GATE_LO: f64 = 0.55;
pub const V31_EOB_CHROMA_GATE_HI: f64 = 0.65;

/// AUR2 v32 — on the diffuse-dark field (R·G) the sharpening chain is muted
/// (no chromatic texture to reveal; SSIM pays) and the luma EOB field is
/// softened by this fraction: the field's variance collapse is what SSIM
/// measures, and the kryte-class images have luminance margin to spare.
pub const V32_EOB_LUMA_RELIEF: f64 = 0.3;

/// Smoothstep gate of the chroma EOB field on the spectral-exchange gamma.
pub fn eob_chroma_gate(exchange_gamma: f64) -> f64 {
    let t = ((exchange_gamma - V31_EOB_CHROMA_GATE_LO)
        / (V31_EOB_CHROMA_GATE_HI - V31_EOB_CHROMA_GATE_LO))
        .clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Explicit environment override, `None` when the variable is unset or
/// unparsable — the caller then applies the format's own law.
pub fn env_override(name: &str) -> Option<f64> {
    std::env::var(name).ok().and_then(|v| v.parse::<f64>().ok())
}

// ======================================================================
// Codon 3D/4D calibrated thresholds
// ======================================================================

/// Luminance zone thresholds (PTF space, 0-255).
/// Calibrated for optimal Weber-Fechner zones on varied content.
pub const CODON_LUM_THRESHOLDS: [f64; 3] = [64.0, 128.0, 192.0];

/// tRNA step factors per luminance zone.
/// Zone 0 (dark): finest quantization (Weber sensitivity highest)
/// Zone 3 (bright): coarsest (less perceptible)
/// Ratio 3:1 matches human Weber fraction (~3% dark vs ~1% bright JND).
/// Recalibrated after DC-space fix: old [0.3, 0.65, 1.0, 1.35] had 4.5× ratio
/// but was inoperant (all blocks at 1.35). Now active, 3× ratio is perceptually correct.
/// Tightened from [0.5, 0.75, 1.0, 1.5] — the 3:1 ratio was too aggressive,
/// causing dark areas to consume disproportionate bits for marginal PSNR gain.
/// Tunable via AUREA_TRNA_DARK, AUREA_TRNA_MIDDARK, AUREA_TRNA_BRIGHT.
pub const CODON_TRNA: [f64; 4] = [0.7, 0.9, 1.0, 1.2];

/// Saturation energy threshold for chroma-adaptive codon.
pub const CODON_SAT_THRESHOLD: f64 = 20.0;

/// Saturation factor (15% finer for saturated blocks).
pub const CODON_SAT_FACTOR: f64 = 0.85;

/// Detail AC energy threshold per coefficient.
pub const CODON_DETAIL_THRESHOLD: f64 = 6.0;

/// Detail factor (15% finer for highly detailed blocks).
pub const CODON_DETAIL_FACTOR: f64 = 0.85;

/// Structural codon factor range (DC gradient-based).
/// MIN = structural (preserve, fine quantization). MAX = smooth (compress, coarse).
/// Ratio MAX/MIN = 1.625 — gentle adaptation that doesn't destroy bright textures.
pub const CODON_STRUCT_FACTOR_MIN: f64 = 0.80;
pub const CODON_STRUCT_FACTOR_MAX: f64 = 1.30;

// ======================================================================
// CSF (Contrast Sensitivity Function) parameters (Point 5)
// ======================================================================

/// Dark-frequency interaction boost.
/// In dark regions, HF is less visible -> coarser quantization.
/// csf_factor = 1 + CSF_DARK_BOOST * (1 - lum_norm) * freq_norm^2
pub const CSF_DARK_BOOST: f64 = 0.5;

// ======================================================================
// Deblocking thresholds
// ======================================================================

/// Gas detection: interior gradient below this = smooth region.
pub const DEBLOCK_GAS_THRESHOLD: f64 = 4.0;

/// Anti-ring sigma edge detection threshold.
pub const ANTI_RING_EDGE_THRESHOLD: f64 = 6.0;

// ======================================================================
// Rate control (Point 6)
// ======================================================================

/// Maximum binary search iterations for rate control.
pub const RATE_CONTROL_MAX_ITER: usize = 8;

/// Convergence tolerance (5% of target bpp).
pub const RATE_CONTROL_TOLERANCE: f64 = 0.05;

// ======================================================================
// Scene classification thresholds (Point 7)
// ======================================================================

/// Smooth percentage for Flat classification.
pub const SCENE_SMOOTH_FLAT: f64 = 50.0;

/// Smooth percentage below which = Organic.
pub const SCENE_SMOOTH_ORGANIC: f64 = 15.0;

/// DC gradient threshold for smooth pixel detection.
pub const SCENE_GRAD_SMOOTH: f64 = 3.0;

// ======================================================================
// Chroma residual (Point 3)
// ======================================================================

/// Block size for chroma residual mask (8x8).
pub const CHROMA_RESIDUAL_BLOCK: usize = 8;

/// Chroma residual quantization step multiplier (relative to detail_step).
pub const CHROMA_RESIDUAL_STEP_MULT: f64 = 0.5;

// ======================================================================
// Variable blocks (Point 2)
// ======================================================================

/// Variance merge threshold = step^2 * this factor.
pub const VARIABLE_BLOCK_MERGE_FACTOR: f64 = 0.25;

/// Minimum percentage of non-16 blocks to justify variable mode.
pub const VARIABLE_BLOCK_MIN_PCT: f64 = 10.0;

// ======================================================================
// Quality-adaptive parameters (DNA4)
// ======================================================================

/// LOT global AC step multiplier.
/// Recalibrated for structural codon (DC gradient, range [0.8, 1.3]).
/// Old: 3.8 * 1.35 (broken luminance codon, all blocks at 1.35) = effective 5.13.
/// New: 4.9 * avg_structural(~1.05) = effective ~5.15 ≈ same bpp budget.
/// The structural codon preserves edges/textures, compresses smooth regions.
pub fn lot_factor_for_quality(_quality: u8) -> f64 {
    4.9
}

/// Dead zone threshold — quality-adaptive, with frequency-dependent override.
/// Use `dead_zone_for_quality` for the base dead zone (low/mid frequencies).
/// Use `dead_zone_for_position` for per-coefficient dead zone (adds HF floor).
pub fn dead_zone_for_quality(quality: u8) -> f64 {
    let q = quality as f64;
    if q <= 70.0 {
        DEAD_ZONE_CONSTANT // 0.22 — full dead zone for compression
    } else {
        // Linear ramp from 0.22 at Q=70 to 0.02 at Q=100
        let t = ((q - 70.0) / 30.0).clamp(0.0, 1.0);
        DEAD_ZONE_CONSTANT * (1.0 - t) + 0.02 * t
    }
}

/// Frequency-dependent dead zone: keeps a floor on the last quarter of the
/// zigzag scan (very high frequencies = sensor noise) even at high quality.
/// `zigzag_pos`: 0-based position in zigzag order
/// `ac_per_block`: total AC coefficients in the block (e.g. 255 for 16x16)
/// `base_dz`: the quality-adaptive base dead zone from `dead_zone_for_quality`
#[inline]
pub fn dead_zone_for_position(zigzag_pos: usize, ac_per_block: usize, base_dz: f64) -> f64 {
    let frac = zigzag_pos as f64 / ac_per_block.max(1) as f64;
    if frac > 0.75 {
        base_dz.max(0.20)
    } else if frac > 0.5 {
        base_dz.max(0.10)
    } else {
        base_dz
    }
}

/// The calibrated dead zone constant (matches wavelet::DEAD_ZONE).
pub const DEAD_ZONE_CONSTANT: f64 = 0.22;

/// QMAT power exponent — quality-adaptive.
/// At low quality, 0.55 penalizes HF heavily (saves bits).
/// At high quality, flattens toward 0.0 (all frequencies equal = max PSNR).
fn qmat_power_curve(quality: u8, p_low: f64, p_high: f64, q_trans: f64, q_range: f64) -> f64 {
    let q = quality as f64;
    if q <= q_trans {
        p_low
    } else {
        let t = ((q - q_trans) / q_range).clamp(0.0, 1.0);
        let s = t * t * (3.0 - 2.0 * t); // smoothstep
        p_low * (1.0 - s) + p_high * s
    }
}

pub fn qmat_power_for_quality(quality: u8) -> f64 {
    let [p_low, p_high, q_trans, q_range] = *TUNABLE_POWER;
    qmat_power_curve(quality, p_low, p_high, q_trans, q_range)
}

/// The low-quality QMAT plateau streams up to AUR2 v24 were written with.
/// Kept only as the documented provenance of the recalibrated value; the
/// decoder no longer accepts those streams, following the lab's standing
/// practice of assuming the compatibility break rather than carrying a legacy
/// dequantization path.
pub const LEGACY_QMAT_POWER_LOW: f64 = 0.55;

/// Spectral spin median blend weight, decreasing with quality.
/// q=20 → 0.38 (strong correction), q=95 → 0.06 (minimal).
pub fn spin_weight_for_quality(quality: u8) -> f64 {
    let q = (quality as f64 / 100.0).clamp(0.01, 1.0);
    0.05 + 0.33 * (1.0 - q)
}

// --- Turing Morphogenesis (v12 Bayesian Hierarchy) ---

/// Activator Gaussian sigma in DC-grid units (blocks).
/// Small σ captures local edge structure.
pub const TURING_SIGMA_A: f64 = 1.5;

/// Ratio σ_inhibitor / σ_activator = φ².
/// Yields Turing wavelength λ_T ∝ φ² ≈ 2.6 blocks.
pub const TURING_SIGMA_RATIO: f64 = 2.618033988749895; // PHI * PHI

/// Step modulation exponent: step_factor = φ^(-k * T_norm).
/// k=1 is the canonical scale; the psychovisual pivot exponents now directly
/// represent the effective low/high-quality modulation strengths.
pub const TURING_K_STEP: f64 = 1.0;

/// Minimum T_norm to seed a ridge trace.
pub const TURING_RIDGE_THRESHOLD: f64 = 0.15;

/// Maximum Hausdorff distance (in DC-grid units) for primitive matching.
pub const TURING_MATCH_DISTANCE_MAX: f64 = 2.0;

/// Search radius for candidate contour matching.
pub const TURING_MATCH_RADIUS: f64 = 4.0;

/// Primitive classified as Predicted if all residuals below these thresholds.
pub const TURING_SURPRISE_POS: f64 = 0.5; // blocks
pub const TURING_SURPRISE_ANGLE: f64 = 0.392699; // π/8
pub const TURING_SURPRISE_AMP: f64 = 2.0; // Fibonacci levels

/// Magnitude prediction calibration factor (initial, tuned on Kodak).
pub const TURING_MAG_CALIBRATION: f64 = 1.0;

/// QMAT parametric constants (Point 2: variable blocks)
/// Formula: Q = Q_A · d^Q_BETA + Q_GAMMA · r · c
/// avec d = sqrt(r² + Q_ALPHA·c²)   et (r, c) ∈ [0,1]²
///
/// Single source of truth pour lot::qmat_for_block_size.
/// LPIPS-calibrées sur le corpus Kodak + 12 images 2K personnelles.
pub const Q_A: f64 = 5.0 * std::f64::consts::PI + 2.0;
pub const Q_ALPHA: f64 = 11.0 / 13.0;
pub const Q_BETA: f64 = 5.0 * crate::golden::PHI / 9.0;
pub const Q_GAMMA: f64 = 176.0 * std::f64::consts::PI / 35.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lot_factor_constant() {
        // 5.5: recalibrated for working Weber-Fechner codon (avg tRNA ~0.94)
        // 5.5 * 0.94 ≈ 5.16 ≈ old 3.8 * 1.35 (when codon was broken at 1.35)
        for q in [1u8, 20, 50, 75, 95, 100] {
            assert!(
                (lot_factor_for_quality(q) - 4.9).abs() < 1e-10,
                "lot_factor must be 4.9 at q={}",
                q
            );
        }
    }

    #[test]
    fn test_dead_zone_quality_adaptive() {
        // Low quality: full dead zone 0.22
        assert!((dead_zone_for_quality(50) - 0.22).abs() < 1e-10);
        assert!((dead_zone_for_quality(70) - 0.22).abs() < 1e-10);
        // High quality: collapses toward 0.02
        assert!(dead_zone_for_quality(85) < 0.15);
        assert!(dead_zone_for_quality(100) < 0.05);
        // Monotonically decreasing above Q=70
        let mut prev = dead_zone_for_quality(70);
        for q in (75..=100).step_by(5) {
            let dz = dead_zone_for_quality(q);
            assert!(
                dz <= prev,
                "dead_zone must decrease: q={} dz={} prev={}",
                q,
                dz,
                prev
            );
            prev = dz;
        }
    }

    #[test]
    fn test_qmat_power_quality_adaptive() {
        // Low quality: full HF penalization 0.55
        assert!((qmat_power_for_quality(50) - 0.55).abs() < 1e-10);
        assert!((qmat_power_for_quality(60) - 0.55).abs() < 1e-10);
        // High quality: flattens toward 0.05
        assert!(qmat_power_for_quality(80) < 0.40);
        assert!(qmat_power_for_quality(100) < 0.10);
        // Monotonically decreasing above Q=60
        let mut prev = qmat_power_for_quality(60);
        for q in (65..=100).step_by(5) {
            let qp = qmat_power_for_quality(q);
            assert!(
                qp <= prev,
                "qmat_power must decrease: q={} qp={} prev={}",
                q,
                qp,
                prev
            );
            prev = qp;
        }
    }

    #[test]
    fn test_spin_weight_monotonic_decreasing() {
        // Spin weight is the ONLY quality-adaptive encoder/decoder param
        // Strong correction at low q, minimal at high q
        let mut prev = spin_weight_for_quality(1);
        for q in (10..=100).step_by(10) {
            let w = spin_weight_for_quality(q);
            assert!(
                w <= prev,
                "spin_weight must decrease: q={} w={} prev={}",
                q,
                w,
                prev
            );
            prev = w;
        }
    }

    #[test]
    fn test_spin_weight_bounds() {
        assert!(spin_weight_for_quality(1) <= 0.40);
        assert!(spin_weight_for_quality(100) >= 0.04);
    }
}
