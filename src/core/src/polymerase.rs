/// Module A.D.N. - Séquençage Géométrique par Polymérase
/// Remplace la détection de primitives PCA/Taubin par un parcours guidé par les codons.

use ndarray::Array2;
use crate::golden::PHI;
use crate::rans;

/// Énergies discrètes des bases nucléotidiques
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nucleobase {
    A = 0, // Forte énergie positive (> PHI)
    C = 1, // Faible énergie positive (0.5 à PHI)
    G = 2, // Faible énergie négative (-PHI à -0.5)
    T = 3, // Forte énergie négative (< -PHI)
    Intron = 4, // Dead-zone (zéro)
}

impl Nucleobase {
    /// Convertit un coefficient continu en base nucléotidique
    pub fn from_f64(val: f64, step: f64) -> Self {
        let n = val / step;
        if n.abs() < 0.5 { return Nucleobase::Intron; }
        if n > PHI { Nucleobase::A }
        else if n > 0.0 { Nucleobase::C }
        else if n > -PHI { Nucleobase::G }
        else { Nucleobase::T }
    }

    /// Traduit la base nucléotidique en énergie reconstituée (f64)
    /// Utilise les centroïdes optimaux d'une distribution Laplacienne
    pub fn decode_f64(&self, step: f64) -> f64 {
        let n = match self {
            Nucleobase::A => PHI + 1.0,  // PHI^2 = 2.618 (centroide Laplacien optimal [PHI, +inf])
            Nucleobase::C => 0.96,      // Centroide Laplacien optimal [0.5, PHI]
            Nucleobase::G => -0.96,
            Nucleobase::T => -(PHI + 1.0),
            Nucleobase::Intron => 0.0,
        };
        n * step
    }

    /// Magnitude absolue pour calculer l'orientation géométrique
    pub fn vector_weight(&self) -> f64 {
        match self {
            Nucleobase::A | Nucleobase::T => 2.0,
            Nucleobase::C | Nucleobase::G => 1.0,
            Nucleobase::Intron => 0.0,
        }
    }
}

/// Un Codon est un triplet inter-bandes (LH, HL, HH).
/// Il y a 5^3 = 125 combinaisons possibles d'états d'énergie spatiale.
#[derive(Debug, Clone, Copy)]
pub struct Codon {
    pub lh: Nucleobase,
    pub hl: Nucleobase,
    pub hh: Nucleobase,
}

impl Codon {
    /// Convertit le Codon en un "Acide Aminé" (un octet entre 0 et 124)
    pub fn to_amino_acid(&self) -> u8 {
        (self.lh as u8) * 25 + (self.hl as u8) * 5 + (self.hh as u8)
    }

    /// Reconstruit un Codon depuis un Acide Aminé (0 - 124)
    pub fn from_amino_acid(aa: u8) -> Self {
        let lh_val = aa / 25;
        let hl_val = (aa % 25) / 5;
        let hh_val = aa % 5;

        let to_base = |v| match v {
            0 => Nucleobase::A, 1 => Nucleobase::C, 2 => Nucleobase::G,
            3 => Nucleobase::T, _ => Nucleobase::Intron,
        };

        Codon { lh: to_base(lh_val), hl: to_base(hl_val), hh: to_base(hh_val) }
    }

    /// La "boussole" de la polymérase. Indique l'orientation de la crête.
    pub fn propagation_vector(&self) -> (f64, f64) {
        let w_lh = self.lh.vector_weight();
        let w_hl = self.hl.vector_weight();
        let w_hh = self.hh.vector_weight();

        // LH (horizontal contour) -> propagation X
        // HL (vertical contour) -> propagation Y
        let mut dx = w_lh + w_hh * 0.707;
        let mut dy = w_hl + w_hh * 0.707;

        let norm = (dx * dx + dy * dy).sqrt();
        if norm > 0.0 {
            dx /= norm; dy /= norm;
        }

        (dx, dy)
    }

    pub fn is_intron(&self) -> bool {
        self.lh == Nucleobase::Intron && self.hl == Nucleobase::Intron && self.hh == Nucleobase::Intron
    }
}

/// Gène : Séquence d'information codant un contour géométrique.
pub struct Gene {
    pub start_x: u16,
    pub start_y: u16,
    pub sequence: Vec<Codon>, // La séquence détermine le tracé exact
}

impl Gene {
    /// Re-simule le parcours de la polymérase pour "traduire" le gène en pixels.
    pub fn translate_path(&self) -> Vec<(usize, usize)> {
        let mut cx = self.start_x as f64;
        let mut cy = self.start_y as f64;
        let mut path = Vec::with_capacity(self.sequence.len());

        let mut prev_dx = 0.0;
        let mut prev_dy = 0.0;

        for codon in &self.sequence {
            let ix = cx.round() as usize;
            let iy = cy.round() as usize;
            path.push((ix, iy));

            let (mut dx, mut dy) = codon.propagation_vector();

            // Momentum : Garantir que la polymérase avance toujours vers l'avant
            // et ne fait pas d'allers-retours sur place.
            if prev_dx != 0.0 || prev_dy != 0.0 {
                if dx * prev_dx + dy * prev_dy < 0.0 {
                    dx = -dx;
                    dy = -dy;
                }
            } else if dx == 0.0 && dy == 0.0 {
                dx = 1.0; // Fallback initial
            }

            prev_dx = dx;
            prev_dy = dy;
            cx += dx;
            cy += dy;
        }
        path
    }
}

// =====================================================================
// Transcription & Élongation
// =====================================================================

fn transcribe_mrna(lh: &Array2<f64>, hl: &Array2<f64>, hh: &Array2<f64>, step: f64) -> Array2<Codon> {
    let (h, w) = (lh.nrows(), lh.ncols());
    let mut mrna = Array2::from_elem((h, w), Codon { lh: Nucleobase::Intron, hl: Nucleobase::Intron, hh: Nucleobase::Intron });

    for y in 0..h {
        for x in 0..w {
            mrna[[y, x]] = Codon {
                lh: Nucleobase::from_f64(lh[[y, x]], step),
                hl: Nucleobase::from_f64(hl[[y, x]], step),
                hh: Nucleobase::from_f64(hh[[y, x]], step),
            };
        }
    }
    mrna
}

/// Tolérance d'Introns traversés avant arrêt (pont ADN).
const INTRON_BRIDGE: usize = 2;

fn sequence_genes(mrna: &mut Array2<Codon>) -> Vec<Gene> {
    let (h, w) = (mrna.nrows(), mrna.ncols());
    let mut genes = Vec::new();
    let intron = Codon { lh: Nucleobase::Intron, hl: Nucleobase::Intron, hh: Nucleobase::Intron };

    for y in 0..h {
        for x in 0..w {
            if mrna[[y, x]].is_intron() { continue; }

            let mut current_gene = Gene {
                start_x: x as u16,
                start_y: y as u16,
                sequence: Vec::new(),
            };

            let mut cx = x as f64;
            let mut cy = y as f64;
            let mut prev_dx = 0.0;
            let mut prev_dy = 0.0;
            let mut intron_run = 0usize;

            for _ in 0..4096 {
                let ix = cx.round() as usize;
                let iy = cy.round() as usize;

                if ix >= w || iy >= h { break; }

                let codon = mrna[[iy, ix]];

                if codon.is_intron() {
                    intron_run += 1;
                    if intron_run > INTRON_BRIDGE { break; }
                    // Pont Intron : on stocke l'Intron dans la séquence
                    // (énergie zéro, rANS le comprime quasi-gratuitement)
                    current_gene.sequence.push(intron);
                } else {
                    intron_run = 0;
                    current_gene.sequence.push(codon);
                    mrna[[iy, ix]] = intron;
                }

                let (mut dx, mut dy) = codon.propagation_vector();

                if prev_dx != 0.0 || prev_dy != 0.0 {
                    if dx * prev_dx + dy * prev_dy < 0.0 {
                        dx = -dx; dy = -dy;
                    }
                } else if dx == 0.0 && dy == 0.0 {
                    dx = 1.0;
                }

                prev_dx = dx; prev_dy = dy;
                cx += dx; cy += dy;
            }

            // Élaguer les Introns terminaux
            while current_gene.sequence.last().map_or(false, |c| c.is_intron()) {
                current_gene.sequence.pop();
            }

            // Filtre de viabilité : longueur + énergie
            if current_gene.sequence.len() >= 16 {
                // Au moins 2 codons à forte énergie (A ou T) pour justifier le coût
                let strong_count = current_gene.sequence.iter().filter(|c|
                    c.lh == Nucleobase::A || c.lh == Nucleobase::T ||
                    c.hl == Nucleobase::A || c.hl == Nucleobase::T ||
                    c.hh == Nucleobase::A || c.hh == Nucleobase::T
                ).count();
                if strong_count >= 2 {
                    genes.push(current_gene);
                }
            }
        }
    }
    genes
}

// =====================================================================
// API Codec (Remplaçant direct de geometric.rs pour les détails)
// =====================================================================

/// Encode les détails via séquençage génétique.
pub fn encode_detail_subband_dna(
    lh: &Array2<f64>, hl: &Array2<f64>, hh: &Array2<f64>,
    max_passes: usize, step: f64
) -> (Vec<Gene>, Array2<f64>, Array2<f64>, Array2<f64>) {

    let mut all_genes = Vec::new();
    let mut resid_lh = lh.clone();
    let mut resid_hl = hl.clone();
    let mut resid_hh = hh.clone();

    for _ in 0..max_passes {
        let mut mrna = transcribe_mrna(&resid_lh, &resid_hl, &resid_hh, step);
        let genes = sequence_genes(&mut mrna);

        if genes.is_empty() { break; }

        for gene in &genes {
            let path = gene.translate_path();
            for (i, &(ix, iy)) in path.iter().enumerate() {
                if ix < lh.ncols() && iy < lh.nrows() {
                    let codon = &gene.sequence[i];
                    resid_lh[[iy, ix]] -= codon.lh.decode_f64(step);
                    resid_hl[[iy, ix]] -= codon.hl.decode_f64(step);
                    resid_hh[[iy, ix]] -= codon.hh.decode_f64(step);
                }
            }
        }
        all_genes.extend(genes);
    }

    (all_genes, resid_lh, resid_hl, resid_hh)
}

/// Traduit et rend les gènes sur l'image (Décodage)
pub fn render_genes(genes: &[Gene], h: usize, w: usize, step: f64) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let mut lh = Array2::<f64>::zeros((h, w));
    let mut hl = Array2::<f64>::zeros((h, w));
    let mut hh = Array2::<f64>::zeros((h, w));

    for gene in genes {
        let path = gene.translate_path();
        for (i, &(ix, iy)) in path.iter().enumerate() {
            if ix < w && iy < h {
                let codon = &gene.sequence[i];
                lh[[iy, ix]] += codon.lh.decode_f64(step);
                hl[[iy, ix]] += codon.hl.decode_f64(step);
                hh[[iy, ix]] += codon.hh.decode_f64(step);
            }
        }
    }
    (lh, hl, hh)
}

/// Le facteur de raffinement résiduel aux positions gène.
/// Le résidu ADN est toujours < bin_width/2 ≈ 0.56*step, donc le quantificateur
/// standard (step) ne capture rien. On utilise step/REFINE pour y accéder.
pub const GENE_REFINE_FACTOR: f64 = 3.0;

/// Construit un step_map pour une sous-bande: step/REFINE aux positions gène, step ailleurs.
pub fn build_refined_step_map(
    genes: &[Gene], band_h: usize, band_w: usize, base_step: f64,
    band_index: usize, // 0=LH, 1=HL, 2=HH
) -> Vec<f64> {
    let n = band_h * band_w;
    let mut step_map = vec![base_step; n];
    let refined_step = base_step / GENE_REFINE_FACTOR;

    for gene in genes {
        let path = gene.translate_path();
        for &(ix, iy) in &path {
            if ix < band_w && iy < band_h {
                step_map[iy * band_w + ix] = refined_step;
            }
        }
    }
    step_map
}

// =====================================================================
// Sérialisation Biomimétique (Empaquetage rANS)
// =====================================================================

// Zigzag encoding: maps small signed values to small unsigned values.
// 0→0, -1→1, 1→2, -2→3, 2→4, ...
#[inline]
fn zigzag_encode(val: i16) -> u16 {
    ((val << 1) ^ (val >> 15)) as u16
}

#[inline]
fn zigzag_decode(val: u16) -> i16 {
    ((val >> 1) as i16) ^ (-((val & 1) as i16))
}

pub fn pack_genome(genes: &[Gene]) -> Vec<u8> {
    let total_codons: usize = genes.iter().map(|g| g.sequence.len()).sum();
    eprintln!("    genome: {} genes, {} codons total (avg {:.1}/gene)",
              genes.len(), total_codons,
              if genes.is_empty() { 0.0 } else { total_codons as f64 / genes.len() as f64 });

    let mut raw_bytes = Vec::with_capacity(2 + genes.len() * 8 + total_codons);
    raw_bytes.extend_from_slice(&(genes.len() as u16).to_le_bytes());

    let mut prev_x: i32 = 0;
    let mut prev_y: i32 = 0;

    for gene in genes {
        // Delta-zigzag: positions relatives au gène précédent
        let dx = gene.start_x as i32 - prev_x;
        let dy = gene.start_y as i32 - prev_y;
        let zx = zigzag_encode(dx as i16);
        let zy = zigzag_encode(dy as i16);
        raw_bytes.extend_from_slice(&zx.to_le_bytes());
        raw_bytes.extend_from_slice(&zy.to_le_bytes());
        raw_bytes.extend_from_slice(&(gene.sequence.len() as u16).to_le_bytes());

        for codon in &gene.sequence {
            raw_bytes.push(codon.to_amino_acid());
        }

        prev_x = gene.start_x as i32;
        prev_y = gene.start_y as i32;
    }

    let raw_size = raw_bytes.len();
    let compressed = rans::rans_compress_bytes(&raw_bytes);
    eprintln!("    genome raw: {} bytes -> compressed: {} bytes ({:.0}%)",
              raw_size, compressed.len(), compressed.len() as f64 / raw_size.max(1) as f64 * 100.0);
    compressed
}

pub fn unpack_genome(data: &[u8]) -> (Vec<Gene>, usize) {
    if data.is_empty() { return (Vec::new(), 0); }

    let raw_bytes = rans::rans_decompress_bytes(data);
    if raw_bytes.len() < 2 { return (Vec::new(), data.len()); }

    let n_genes = u16::from_le_bytes([raw_bytes[0], raw_bytes[1]]) as usize;
    let mut p = 2;
    let mut genes = Vec::with_capacity(n_genes);
    let mut prev_x: i32 = 0;
    let mut prev_y: i32 = 0;

    for _ in 0..n_genes {
        if p + 6 > raw_bytes.len() { break; }
        let zx = u16::from_le_bytes([raw_bytes[p], raw_bytes[p+1]]); p += 2;
        let zy = u16::from_le_bytes([raw_bytes[p], raw_bytes[p+1]]); p += 2;
        let seq_len = u16::from_le_bytes([raw_bytes[p], raw_bytes[p+1]]) as usize; p += 2;

        let start_x = (prev_x + zigzag_decode(zx) as i32) as u16;
        let start_y = (prev_y + zigzag_decode(zy) as i32) as u16;

        if p + seq_len > raw_bytes.len() { break; }
        let mut sequence = Vec::with_capacity(seq_len);
        for _ in 0..seq_len {
            sequence.push(Codon::from_amino_acid(raw_bytes[p]));
            p += 1;
        }

        prev_x = start_x as i32;
        prev_y = start_y as i32;
        genes.push(Gene { start_x, start_y, sequence });
    }

    (genes, data.len())
}