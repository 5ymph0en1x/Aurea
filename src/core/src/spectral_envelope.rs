//! Cycle 20 — enveloppes amplitude/phase cohérentes par voisinage spectral.
//!
//! Jusqu'à la v20 le codec ne regardait qu'un seul témoin intra-bloc : le
//! support binaire du parent dyadique `(v/2, u/2)`. Le spectre d'une LOT n'est
//! pourtant pas une collection de coefficients indépendants : un contour ou une
//! texture localement bande-étroite y dessine une *crête* dont l'amplitude
//! décroît continûment et dont les signes suivent une loi séparable.
//!
//! Pour un contenu séparable — l'impulsion, l'arête, le produit de deux
//! sinusoïdes — la DCT-II vaut
//!
//! ```text
//! C[v, u] ≈ A(v, u) · cos(v · alpha) · cos(u · beta)
//! ```
//!
//! avec `alpha = pi (y0 + 1/2) / N` et `beta = pi (x0 + 1/2) / N`. En posant
//! `s = signe`, le produit des quatre coins d'un carré unité du plan
//! fréquentiel vaut exactement `+1` :
//!
//! ```text
//! s[v, u] · s[v, u-1] · s[v-1, u] · s[v-1, u-1] = g(v)^2 g(v-1)^2 f(u)^2 f(u-1)^2 = 1
//! ```
//!
//! d'où la **loi de phase séparable**, gratuite et causale dans le zigzag :
//!
//! ```text
//! s[v, u] = s[v, u-1] XOR s[v-1, u] XOR s[v-1, u-1]
//! ```
//!
//! Sur les bords du spectre (`v = 0` ou `u = 0`) la relation dégénère en une
//! suite unidimensionnelle `signe(cos(u beta))`, dont les *transitions* sont
//! quasi-périodiques. Le prédicteur y devient `s[u] = s[u-2]`, et la classe
//! retient en plus la transition précédente : la mémoire adaptative sépare
//! alors d'elle-même les blocs à densité de flip faible et forte.
//!
//! L'enveloppe d'amplitude est construite sur le même voisinage : elle gradue
//! ce que le bit de parent v16 ne faisait que trancher.
//!
//! Toutes les grandeurs de ce module sont des fonctions de coefficients déjà
//! reconstruits dans le *même* bloc et le *même* canal. Encodeur et décodeur
//! appellent littéralement le même évaluateur : la symétrie est une propriété
//! du code, pas une convention à maintenir.

/// Voisins causaux retenus autour d'un coefficient.
pub const SPECTRAL_NEIGHBORS: usize = 6;
/// `(v, u-1)`
pub const NEIGHBOR_LEFT: usize = 0;
/// `(v-1, u)`
pub const NEIGHBOR_TOP: usize = 1;
/// `(v-1, u-1)`
pub const NEIGHBOR_DIAG: usize = 2;
/// `(v, u-2)`
pub const NEIGHBOR_LEFT2: usize = 3;
/// `(v-2, u)`
pub const NEIGHBOR_TOP2: usize = 4;
/// `(v/2, u/2)` — le parent dyadique de la v16.
pub const NEIGHBOR_PARENT: usize = 5;

/// Marqueur d'un voisin absent, hors bloc, égal au DC ou non causal.
pub const ABSENT: u32 = u32::MAX;

/// Nombre maximal de coefficients qu'une couche antérieure peut avoir déjà
/// livrés en tête d'un bloc. Vaut la taille du squelette progressif.
pub const MAX_PREFIX: usize = crate::scan::PROGRESSIVE_SKELETON_COEFFS;

// ======================================================================
// Classes de phase
// ======================================================================

/// Aucune prédiction spectrale : le signe reste au prédicteur historique
/// (C1 -> C2, continuation spatiale de C1) ou à un bit brut.
pub const PHASE_NONE: u8 = 0;
/// Loi séparable exacte : `left`, `top` et `diag` sont tous non nuls.
pub const PHASE_SEPARABLE: u8 = 1;
/// Continuation 1D horizontale, transition précédente nulle.
pub const PHASE_ROW_KEEP: u8 = 2;
/// Continuation 1D horizontale, transition précédente non nulle.
pub const PHASE_ROW_FLIP: u8 = 3;
/// Continuation 1D verticale, transition précédente nulle.
pub const PHASE_COL_KEEP: u8 = 4;
/// Continuation 1D verticale, transition précédente non nulle.
pub const PHASE_COL_FLIP: u8 = 5;
/// Paire `left`/`top` sans diagonale : loi séparable amputée.
pub const PHASE_PAIR: u8 = 6;
/// Seul le parent dyadique porte une phase.
pub const PHASE_PARENT: u8 = 7;
/// Seul le voisin horizontal porte une phase.
pub const PHASE_LEFT: u8 = 8;
/// Seul le voisin vertical porte une phase.
pub const PHASE_TOP: u8 = 9;

/// Nombre de compteurs adaptatifs de phase spectrale.
pub const PHASE_CLASS_COUNT: usize = 10;

/// Masque activant toutes les classes sauf `PHASE_NONE`.
pub const PHASE_MASK_ALL: u16 = 0b11_1111_1110;
/// Masque ne retenant que la loi séparable exacte.
pub const PHASE_MASK_SEPARABLE: u16 = 1 << PHASE_SEPARABLE;
/// Loi séparable plus les deux continuations unidimensionnelles.
pub const PHASE_MASK_COHERENT: u16 = PHASE_MASK_SEPARABLE
    | (1 << PHASE_ROW_KEEP)
    | (1 << PHASE_ROW_FLIP)
    | (1 << PHASE_COL_KEEP)
    | (1 << PHASE_COL_FLIP);

// ======================================================================
// Modes d'enveloppe d'amplitude
// ======================================================================

/// Support binaire du parent dyadique : reproduit exactement l'axe v16/v20.
pub const AMPLITUDE_LEGACY: u8 = 0;
/// Support du parent croisé au support local `left`/`top` — sur-ensemble strict
/// de `AMPLITUDE_LEGACY`.
pub const AMPLITUDE_PARENT_LOCAL: u8 = 1;
/// Somme des magnitudes du voisinage, graduée en quatre classes.
pub const AMPLITUDE_MAGNITUDE: u8 = 2;
/// Nombre de voisins non nuls, gradué en quatre classes.
pub const AMPLITUDE_SUPPORT: u8 = 3;
/// Cycle 21 — somme des magnitudes rapportée à l'énergie courante du modèle.
///
/// Les seuils absolus `{0} {1} {2..3} {4+}` saturent quand la qualité monte :
/// à Q80 presque tout voisinage peuplé tombe dans la dernière classe et l'axe
/// dégénère en « silencieux / bruyant ». La mémoire IIR d'énergie que le modèle
/// entropique entretient déjà donne l'échelle locale gratuitement et à
/// l'identique des deux côtés ; les seuils la suivent au lieu de la subir.
pub const AMPLITUDE_MAGNITUDE_SCALED: u8 = 4;

/// Nombre d'états de l'axe d'amplitude dans les tables de contexte.
pub const AMPLITUDE_STATES: usize = 4;

// ======================================================================
// Contextes de la loi séparable
// ======================================================================

/// Un seul compteur par classe de phase — le profil promu du cycle 20.
pub const PHASE_CONTEXT_NONE: u8 = 0;
/// Énergie du voisinage, sur toutes les classes. Mesuré et rejeté au cycle 20.
pub const PHASE_CONTEXT_ENERGY_ALL: u8 = 1;
/// Mémoire intra-bloc : la loi séparable a-t-elle tenu la fois précédente dans
/// ce même bloc ? `0` inconnu, `1` elle a tenu, `2` elle a échoué. Le témoin
/// est auto-calibrant, causal et strictement gratuit.
pub const PHASE_CONTEXT_SEPARABLE_MEMORY: u8 = 2;
/// Bucket de Turing du bloc, sur la seule classe séparable.
pub const PHASE_CONTEXT_SEPARABLE_TURING: u8 = 3;
/// Énergie du voisinage, sur la seule classe séparable.
pub const PHASE_CONTEXT_SEPARABLE_ENERGY: u8 = 4;
// ======================================================================
// Cycle 23 — entrées du mélange
// ======================================================================

/// Enveloppe de voisinage de Lphi — l'avis promu au cycle 22.
pub const MIX_INPUT_FOREIGN: u8 = 1 << 0;
/// Enveloppe rapportée à l'échelle locale. Rejetée comme *contexte* au cycle
/// 21 parce que normaliser efface l'information de bande fréquentielle ; comme
/// *entrée* elle n'efface rien, elle propose un second point de vue.
pub const MIX_INPUT_SCALED: u8 = 1 << 1;
/// Bande fréquentielle explicite. Le modèle primaire n'a jamais connu la
/// fréquence que par sa mémoire IIR d'énergie, qui n'en est qu'un indice
/// indirect et retardé.
pub const MIX_INPUT_BAND: u8 = 1 << 2;
/// Cycle 24 — taille de la transformée. Refusée comme contexte de la loi
/// séparable au cycle 22 ; le corollaire du cycle 23 veut qu'on la réessaie
/// comme entrée.
pub const MIX_INPUT_SIZE: u8 = 1 << 3;
/// Cycle 24 — nombre de voisins actifs, une lecture du *même* voisinage que
/// l'enveloppe promue mais en comptage plutôt qu'en somme. Perdante comme axe
/// d'amplitude au cycle 20 (`-0,755 %` contre `-1,193 %`).
pub const MIX_INPUT_SUPPORT: u8 = 1 << 4;

/// Cycle 22 — taille de la transformée, sur la seule classe séparable.
///
/// Le quadtree a déjà tranché : un bloc 32x32 est un bloc que l'encodeur a jugé
/// lisse, et c'est précisément là qu'un contenu séparable est le plus probable.
/// Le témoin est dans la syntaxe, il ne coûte rien, et le cycle 21 ne l'avait
/// pas essayé.
pub const PHASE_CONTEXT_SEPARABLE_SIZE: u8 = 5;

// ======================================================================
// Configuration
// ======================================================================

/// Réglage complet de l'expérience. L'état neutre
/// (`SpectralEnvelopeConfig::neutral()`) reproduit octet pour octet la
/// grammaire v20.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpectralEnvelopeConfig {
    /// `AMPLITUDE_*`.
    pub amplitude_mode: u8,
    /// L'axe d'amplitude sépare aussi `P(|x| = 1)`.
    pub amplitude_splits_pm1: bool,
    /// Bit `c` autorise la classe de phase `c`.
    pub phase_mask: u16,
    /// Canaux (bit `ch`) où la phase spectrale est évaluée.
    pub phase_channels: u8,
    /// `P(accord)` initial des compteurs de phase spectrale.
    pub phase_initial_p_same: u32,
    /// Vitesse d'adaptation des compteurs de phase spectrale.
    pub phase_adapt_shift: u32,
    /// La phase spectrale prime sur le prédicteur historique lorsqu'elle est
    /// disponible. Sinon elle ne sert que de repli.
    pub phase_takes_precedence: bool,
    /// `CROSS_CODON_*` — le codon chromatique lit aussi le voisinage de Lphi.
    pub cross_codon_mode: u8,
    /// Cycle 22 — mélange logistique de plusieurs modèles de `P(zéro)`.
    /// `0` désactive et conserve le chemin v21 exact ; sinon la valeur est le
    /// taux d'apprentissage du mélangeur.
    pub mix_rate: u8,
    /// Cycle 23 — `MIX_INPUT_*`, quels avis secondaires entrent dans le
    /// mélange. Ce qui a échoué comme *contexte* mérite d'être réessayé comme
    /// *entrée* : un index doit tout porter d'un coup, une entrée n'a qu'à
    /// être approximativement juste et approximativement indépendante.
    pub mix_inputs: u8,
    /// Cycle 23 — mélanger aussi `P(|x| = 1)`, qui suit la même grammaire et
    /// que le cycle 22 avait laissé au modèle primaire seul.
    pub mix_pm1: bool,
    /// Cycle 23 — affinage APM après le mélange, indexé par la probabilité
    /// mélangée elle-même. `0` désactive.
    pub apm_rate: u8,
    /// Cycle 24 — canaux (bit `ch`) où le mélange est actif.
    pub mix_channels: u8,
    /// Cycle 24 — mélanger aussi l'accord de phase. Les compteurs de signe sont
    /// restés des modèles uniques indexés par classe depuis le cycle 20, alors
    /// que la loi séparable dispose de témoins mesurés — mémoire intra-bloc,
    /// énergie du voisinage — qui n'avaient pas survécu à la dilution d'un
    /// index.
    pub mix_sign: bool,
    /// `PHASE_CONTEXT_*` — ce qui dédouble les compteurs d'accord de phase.
    ///
    /// La loi séparable est exacte sur le signal continu ; ce qui la dégrade
    /// est le contenu *non séparable*, pas le bruit. Le cycle 20 a rejeté
    /// l'énergie du voisinage comme témoin de fiabilité ; le cycle 21 essaie
    /// des témoins qui décrivent le bloc plutôt que le coefficient.
    pub phase_context: u8,
}

impl SpectralEnvelopeConfig {
    /// État neutre : la v20 exacte, sans aucun calcul de voisinage.
    pub const fn neutral() -> Self {
        Self {
            amplitude_mode: AMPLITUDE_LEGACY,
            amplitude_splits_pm1: false,
            phase_mask: 0,
            phase_channels: 0,
            phase_initial_p_same: 1 << 13,
            phase_adapt_shift: 6,
            phase_takes_precedence: false,
            cross_codon_mode: CROSS_CODON_NONE,
            mix_rate: 0,
            mix_inputs: MIX_INPUT_FOREIGN,
            mix_pm1: false,
            apm_rate: 0,
            mix_channels: 0b111,
            mix_sign: false,
            phase_context: PHASE_CONTEXT_NONE,
        }
    }

    /// Profil promu du cycle 20.
    ///
    /// L'enveloppe d'amplitude est la somme des magnitudes du voisinage,
    /// graduée en quatre classes, et elle sépare aussi `P(|x| = 1)`. La phase
    /// retient les seules lois qui décrivent une géométrie : la loi séparable
    /// exacte et les deux continuations unidimensionnelles. Les classes
    /// résiduelles — paire tronquée, parent seul, voisin seul — restent
    /// désactivées : elles n'apportent rien et paient le bruit d'adaptation.
    ///
    /// Le couplage énergie/phase a été mesuré puis rejeté : la loi séparable
    /// échoue sur le contenu non séparable, pas sur le bruit de quantification,
    /// et l'amplitude du voisinage n'en est donc pas un bon témoin.
    ///
    /// Le cycle 24 ajoute un cinquième avis — le comptage de voisins actifs,
    /// perdant comme axe d'amplitude au cycle 20 — et mélange enfin l'accord de
    /// phase, dernière décision restée sur un modèle unique. La taille de
    /// transformée, elle, reste refusée : elle avait échoué comme contexte au
    /// cycle 22 et échoue aussi comme entrée, ce qui montre que le corollaire
    /// « réessayer comme entrée » n'est pas automatique — encore faut-il que le
    /// témoin porte une information.
    ///
    /// Le cycle 23 porte le mélange à quatre avis — le voisinage de Lphi, une
    /// enveloppe rapportée à l'échelle locale et une bande fréquentielle
    /// explicite — l'étend à `P(|x| = 1)` et le fait suivre d'un affinage APM.
    /// Les deux premiers témoins avaient été *refusés comme contextes* aux
    /// cycles 21 et 22 : un index doit tout porter d'un coup, une entrée n'a
    /// qu'à être approximativement juste et approximativement indépendante.
    ///
    /// Le cycle 22 y ajoute le mélange logistique (`mix_rate = 1`). Le cycle 21
    /// avait mesuré `0,03` à `0,10` bit/symbole d'information causale et
    /// gratuite dans le voisinage de Lphi, inatteignable par un index de
    /// contexte : la fondre détruit la valeur ponctuelle du codon, lui donner
    /// son propre axe paie une dilution qui annule le gain. Le mélange ne fait
    /// ni l'un ni l'autre — deux modèles, deux avis, un arbitre qui apprend.
    /// Le taux `1` est le plus lent testé et le meilleur : la question posée au
    /// mélangeur est simple, elle n'a pas besoin d'être apprise vite.
    pub const fn promoted() -> Self {
        Self {
            amplitude_mode: AMPLITUDE_MAGNITUDE,
            amplitude_splits_pm1: true,
            phase_mask: PHASE_MASK_COHERENT,
            phase_channels: 0b111,
            phase_initial_p_same: 1 << 13,
            phase_adapt_shift: 5,
            phase_takes_precedence: false,
            cross_codon_mode: CROSS_CODON_NONE,
            mix_rate: 1,
            mix_inputs: MIX_INPUT_FOREIGN | MIX_INPUT_SCALED | MIX_INPUT_BAND | MIX_INPUT_SUPPORT,
            mix_pm1: true,
            apm_rate: 6,
            mix_channels: 0b111,
            mix_sign: true,
            phase_context: PHASE_CONTEXT_NONE,
        }
    }

    /// Vrai quand la configuration n'ajoute strictement rien à la v20.
    #[inline]
    pub const fn is_neutral(&self) -> bool {
        self.amplitude_mode == AMPLITUDE_LEGACY
            && !self.amplitude_splits_pm1
            && self.mix_rate == 0
            && (self.phase_mask == 0 || self.phase_channels == 0)
    }

    /// Vrai quand l'enveloppe d'amplitude doit être calculée.
    #[inline]
    pub const fn needs_amplitude(&self) -> bool {
        self.amplitude_mode != AMPLITUDE_LEGACY
    }

    /// Vrai quand la phase spectrale doit être évaluée pour ce canal.
    #[inline]
    pub const fn phase_enabled(&self, channel: usize) -> bool {
        self.phase_mask != 0 && (self.phase_channels >> channel) & 1 != 0
    }
}

impl Default for SpectralEnvelopeConfig {
    fn default() -> Self {
        Self::neutral()
    }
}

// ======================================================================
// Table de voisinage
// ======================================================================

/// Voisinage causal, exprimé en positions zigzag, pour une taille de
/// transformée donnée. Chaque entrée vaut `ABSENT` si le voisin sort du bloc,
/// tombe sur le DC (absent du parcours AC) ou n'est pas strictement antérieur.
#[derive(Clone, Debug)]
pub struct SpectralNeighborTable {
    positions: Vec<[u32; SPECTRAL_NEIGHBORS]>,
}

impl SpectralNeighborTable {
    /// Construit la table depuis le parcours AC réellement utilisé. La
    /// causalité n'est jamais supposée : elle est vérifiée position par
    /// position, ce qui laisse le mécanisme valide si le parcours change.
    pub fn new(zigzag: &[usize], block_size: usize) -> Self {
        let mut rank = vec![ABSENT; block_size * block_size];
        for (position, &natural_idx) in zigzag.iter().enumerate() {
            if natural_idx < rank.len() {
                rank[natural_idx] = position as u32;
            }
        }

        let positions = zigzag
            .iter()
            .enumerate()
            .map(|(position, &natural_idx)| {
                let row = natural_idx / block_size;
                let col = natural_idx % block_size;
                let mut entry = [ABSENT; SPECTRAL_NEIGHBORS];
                let mut resolve = |slot: usize, target_row: usize, target_col: usize| {
                    let target = target_row * block_size + target_col;
                    let candidate = rank.get(target).copied().unwrap_or(ABSENT);
                    if candidate != ABSENT && (candidate as usize) < position {
                        entry[slot] = candidate;
                    }
                };
                if col >= 1 {
                    resolve(NEIGHBOR_LEFT, row, col - 1);
                }
                if row >= 1 {
                    resolve(NEIGHBOR_TOP, row - 1, col);
                }
                if row >= 1 && col >= 1 {
                    resolve(NEIGHBOR_DIAG, row - 1, col - 1);
                }
                if col >= 2 {
                    resolve(NEIGHBOR_LEFT2, row, col - 2);
                }
                if row >= 2 {
                    resolve(NEIGHBOR_TOP2, row - 2, col);
                }
                if row / 2 != 0 || col / 2 != 0 {
                    resolve(NEIGHBOR_PARENT, row / 2, col / 2);
                }
                entry
            })
            .collect();

        Self { positions }
    }

    /// Nombre de positions AC couvertes.
    #[inline]
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    #[inline]
    fn entry(&self, position: usize) -> &[u32; SPECTRAL_NEIGHBORS] {
        const EMPTY: [u32; SPECTRAL_NEIGHBORS] = [ABSENT; SPECTRAL_NEIGHBORS];
        self.positions.get(position).unwrap_or(&EMPTY)
    }
}

/// Les trois tailles de transformée d'AUREA, indexées par classe 0/1/2.
#[derive(Clone, Debug)]
pub struct SpectralNeighborTables {
    tables: [SpectralNeighborTable; 3],
}

impl SpectralNeighborTables {
    /// `zigzags` fournit les parcours 8x8, 16x16 puis 32x32.
    pub fn new(zigzag_8: &[usize], zigzag_16: &[usize], zigzag_32: &[usize]) -> Self {
        Self {
            tables: [
                SpectralNeighborTable::new(zigzag_8, 8),
                SpectralNeighborTable::new(zigzag_16, 16),
                SpectralNeighborTable::new(zigzag_32, 32),
            ],
        }
    }

    /// Classe de table pour une taille de bloc.
    #[inline]
    pub const fn table_class(block_size: usize) -> u8 {
        match block_size {
            8 => 0,
            32 => 2,
            _ => 1,
        }
    }

    #[inline]
    pub fn table(&self, class: u8) -> &SpectralNeighborTable {
        &self.tables[(class as usize).min(2)]
    }
}

// ======================================================================
// Segments : la géométrie d'un bloc dans une séquence rANS plate
// ======================================================================

/// Plage contiguë occupée par un bloc dans la séquence plate d'une couche
/// progressive, augmentée des coefficients qu'une couche antérieure a déjà
/// livrés. C'est tout ce dont le décodeur a besoin pour reconstruire le
/// voisinage spectral de chaque symbole sans un seul bit supplémentaire.
#[derive(Clone, Copy, Debug)]
pub struct SpectralSegment {
    /// Index plat de la position zigzag `start`.
    pub offset: u32,
    /// Première position zigzag portée par cette couche.
    pub start: u16,
    /// Nombre de positions portées par cette couche.
    pub len: u32,
    /// Classe de table de voisinage (0 => 8x8, 1 => 16x16, 2 => 32x32).
    pub table: u8,
    /// Coefficients des positions `< start`, déjà décodés. Seules les `start`
    /// premières entrées ont un sens.
    pub prefix: [i16; MAX_PREFIX],
}

impl SpectralSegment {
    /// Segment d'une couche qui démarre au début du bloc.
    pub fn root(offset: usize, len: usize, table: u8) -> Self {
        Self {
            offset: offset as u32,
            start: 0,
            len: len as u32,
            table,
            prefix: [0; MAX_PREFIX],
        }
    }

    /// Segment d'une couche de détail, précédé par `prefix`.
    pub fn detail(offset: usize, start: usize, len: usize, table: u8, prefix: &[i16]) -> Self {
        let mut stored = [0i16; MAX_PREFIX];
        let count = start.min(MAX_PREFIX).min(prefix.len());
        stored[..count].copy_from_slice(&prefix[..count]);
        Self {
            offset: offset as u32,
            start: start.min(MAX_PREFIX) as u16,
            len: len as u32,
            table,
            prefix: stored,
        }
    }
}

// ======================================================================
// Descripteur
// ======================================================================

/// Ce que le voisinage spectral apprend sur un coefficient, avant de le lire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SpectralDescriptor {
    /// Classe d'enveloppe d'amplitude, `0..AMPLITUDE_STATES`.
    pub amplitude: u8,
    /// `PHASE_*`, ou `PHASE_NONE` si aucune loi ne s'applique.
    pub phase_class: u8,
    /// Signe prédit lorsque `phase_class != PHASE_NONE`.
    pub phase_negative: bool,
    /// Énergie du voisinage, `0..AMPLITUDE_STATES` — mesure la fiabilité de la
    /// phase indépendamment du mode d'amplitude retenu.
    pub energy: u8,
    /// Support du parent dyadique — l'axe historique, conservé pour l'état
    /// neutre et pour la télémétrie.
    pub parent_nonzero: bool,
}

#[inline]
fn magnitude_bucket(sum: u32) -> u8 {
    match sum {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        _ => 3,
    }
}

/// Même découpage, mais rapporté à l'échelle locale. `scale` est la mémoire
/// IIR d'énergie du modèle entropique, sur `0..=256`, où `32` correspond à une
/// magnitude typique de `1`. La magnitude de référence vaut donc
/// `max(1, scale / 32)` et les seuils la suivent : à bas débit le découpage est
/// exactement celui du cycle 20, à haut débit il se dilate au lieu de saturer.
#[inline]
fn magnitude_bucket_scaled(sum: u32, scale: u32) -> u8 {
    let reference = (scale / 32).max(1);
    if sum == 0 {
        0
    } else if sum <= reference {
        1
    } else if sum <= 3 * reference {
        2
    } else {
        3
    }
}

// ======================================================================
// Cycle 21, piste 2 — témoin inter-canaux
// ======================================================================

/// Le codon chromatique v17 reste inchangé.
pub const CROSS_CODON_NONE: u8 = 0;
/// `max(classe ponctuelle de Lphi, enveloppe de voisinage de Lphi)`.
pub const CROSS_CODON_MAX: u8 = 1;
/// L'enveloppe de voisinage de Lphi remplace la classe ponctuelle.
pub const CROSS_CODON_ENVELOPE: u8 = 2;
/// Somme saturée des deux classes.
pub const CROSS_CODON_SUM: u8 = 3;
/// Le témoin de voisinage occupe son *propre* bit au lieu d'être fondu dans la
/// classe ponctuelle, portant l'axe du codon de 4 à 8 états. Les trois modes
/// fusionnels ci-dessus ne diluent rien mais détruisent une information que le
/// codon portait déjà ; celui-ci paie une dilution réelle pour n'en détruire
/// aucune. C'est l'arbitrage que le cycle 21 mesure.
pub const CROSS_CODON_SPLIT: u8 = 4;

/// États de l'axe du codon dans les tables de contexte. Quatre sont utilisés
/// par la v17 ; les quatre autres n'existent que pour `CROSS_CODON_SPLIT`, et
/// rester inutilisés ne change rien à la partition des symboles.
pub const CODON_STATES: usize = 8;

/// Classe d'amplitude du voisinage d'un canal *étranger* à la même position.
///
/// Pour les chromas ce canal est Lphi, entièrement décodé avant elles dans
/// chaque couche progressive : le témoin reste causal et strictement gratuit.
/// La télémétrie du cycle 21 mesure qu'il apporte `0,026` à `0,103` bit par
/// symbole *au-delà* du codon v17, qui ne connaît que la valeur ponctuelle.
pub fn foreign_amplitude<F>(table: &SpectralNeighborTable, position: usize, mut fetch: F) -> u8
where
    F: FnMut(usize) -> i16,
{
    let entry = table.entry(position);
    let mut sum = 0u32;
    for slot in [NEIGHBOR_LEFT, NEIGHBOR_TOP, NEIGHBOR_DIAG, NEIGHBOR_PARENT] {
        if entry[slot] != ABSENT {
            sum += u32::from(fetch(entry[slot] as usize).unsigned_abs());
        }
    }
    magnitude_bucket(sum)
}

/// Fond le témoin de voisinage dans la classe du codon **sans ajouter un seul
/// contexte** : l'axe reste à quatre états. C'est délibéré — le cycle 15 a
/// montré qu'une dépendance mesurable n'est un codon utile que si son contexte
/// est causal *et* gratuit, et l'échec du masque conjoint venait de la
/// dilution, pas de l'absence de signal.
#[inline]
pub fn combine_codon_class(mode: u8, codon: u8, foreign: u8) -> u8 {
    match mode {
        CROSS_CODON_MAX => codon.max(foreign),
        CROSS_CODON_ENVELOPE => foreign,
        CROSS_CODON_SUM => (codon as u32 + foreign as u32).min(3) as u8,
        CROSS_CODON_SPLIT => codon.min(3) | (u8::from(foreign >= 2) << 2),
        _ => codon,
    }
}

/// Voisinage à demi évalué : l'axe d'amplitude est prêt, la loi de phase n'est
/// résolue que pour les coefficients qui se révèlent non nuls — la minorité —
/// et les deux voisins de second ordre ne sont alors même pas lus.
#[derive(Clone, Copy, Debug)]
pub struct SpectralProbe {
    /// Classe d'enveloppe d'amplitude, `0..AMPLITUDE_STATES`.
    pub amplitude: u8,
    /// Énergie du voisinage, `0..AMPLITUDE_STATES`.
    pub energy: u8,
    /// Enveloppe rapportée à l'échelle locale du modèle. Perdante comme axe de
    /// contexte au cycle 21, candidate comme entrée de mélange au cycle 23.
    pub scaled_amplitude: u8,
    /// Nombre de voisins actifs — une lecture par comptage du même voisinage
    /// que l'enveloppe promue lit par sommation.
    pub support_count: u8,
    /// Support du parent dyadique — l'axe historique.
    pub parent_nonzero: bool,
    /// Index du segment, c'est-à-dire du bloc, auquel appartient le symbole.
    /// Permet aux mémoires intra-bloc de savoir quand se réinitialiser.
    pub block: u32,
    /// Classe de taille de la transformée : `0` pour 8x8, `1` pour 16x16,
    /// `2` pour 32x32.
    pub size_class: u8,
    /// Position zigzag dans le bloc. Le modèle primaire n'a jamais connu la
    /// fréquence autrement que par sa mémoire IIR d'énergie, qui n'en est
    /// qu'un indice indirect ; le cycle 23 en fait une entrée à part entière.
    pub position: u16,
    entry: [u32; SPECTRAL_NEIGHBORS],
    values: [i16; SPECTRAL_NEIGHBORS],
    phase_enabled: bool,
}

impl SpectralProbe {
    /// Descripteur d'un voisinage entièrement absent.
    pub const EMPTY: Self = Self {
        amplitude: 0,
        energy: 0,
        scaled_amplitude: 0,
        support_count: 0,
        parent_nonzero: false,
        block: u32::MAX,
        size_class: 1,
        position: 0,
        entry: [ABSENT; SPECTRAL_NEIGHBORS],
        values: [0; SPECTRAL_NEIGHBORS],
        phase_enabled: false,
    };
}

/// Première moitié : l'enveloppe d'amplitude, requise pour `P(zéro)` et donc
/// pour chaque symbole. `fetch` ne reçoit que des positions zigzag strictement
/// antérieures ; le contrat est garanti par la table.
///
/// `scale` porte l'échelle locale de magnitude — la mémoire IIR d'énergie du
/// modèle entropique — et n'est lue que par `AMPLITUDE_MAGNITUDE_SCALED`.
pub fn probe<F>(
    table: &SpectralNeighborTable,
    position: usize,
    config: &SpectralEnvelopeConfig,
    phase_enabled: bool,
    scale: u32,
    mut fetch: F,
) -> SpectralProbe
where
    F: FnMut(usize) -> i16,
{
    let entry = *table.entry(position);
    let mut values = [0i16; SPECTRAL_NEIGHBORS];
    for slot in [NEIGHBOR_LEFT, NEIGHBOR_TOP, NEIGHBOR_DIAG, NEIGHBOR_PARENT] {
        if entry[slot] != ABSENT {
            values[slot] = fetch(entry[slot] as usize);
        }
    }

    let parent_nonzero = values[NEIGHBOR_PARENT] != 0;
    let energy_sum = u32::from(values[NEIGHBOR_LEFT].unsigned_abs())
        + u32::from(values[NEIGHBOR_TOP].unsigned_abs())
        + u32::from(values[NEIGHBOR_DIAG].unsigned_abs())
        + u32::from(values[NEIGHBOR_PARENT].unsigned_abs());
    let energy = magnitude_bucket(energy_sum);
    let amplitude = match config.amplitude_mode {
        AMPLITUDE_PARENT_LOCAL => {
            let local = values[NEIGHBOR_LEFT] != 0 || values[NEIGHBOR_TOP] != 0;
            u8::from(parent_nonzero) | (u8::from(local) << 1)
        }
        AMPLITUDE_MAGNITUDE => energy,
        AMPLITUDE_MAGNITUDE_SCALED => magnitude_bucket_scaled(energy_sum, scale),
        AMPLITUDE_SUPPORT => {
            let count = u32::from(values[NEIGHBOR_LEFT] != 0)
                + u32::from(values[NEIGHBOR_TOP] != 0)
                + u32::from(values[NEIGHBOR_DIAG] != 0)
                + u32::from(parent_nonzero);
            count.min(3) as u8
        }
        _ => u8::from(parent_nonzero),
    };

    let support_count = (u32::from(values[NEIGHBOR_LEFT] != 0)
        + u32::from(values[NEIGHBOR_TOP] != 0)
        + u32::from(values[NEIGHBOR_DIAG] != 0)
        + u32::from(parent_nonzero))
    .min(3) as u8;

    SpectralProbe {
        amplitude,
        energy,
        scaled_amplitude: magnitude_bucket_scaled(energy_sum, scale),
        support_count,
        parent_nonzero,
        block: u32::MAX,
        size_class: 1,
        position: position.min(u16::MAX as usize) as u16,
        entry,
        values,
        phase_enabled,
    }
}

/// Seconde moitié : la loi de phase, appelée seulement quand le coefficient est
/// non nul. Les deux voisins de second ordre sont lus ici, jamais avant.
///
/// Les classes sont essayées par fiabilité décroissante ; la première à la fois
/// autorisée *et* applicable gagne, de sorte que masquer une classe laisse la
/// suivante prendre le relais au lieu de désactiver la position.
pub fn resolve_phase_class<F>(
    probe: &SpectralProbe,
    config: &SpectralEnvelopeConfig,
    mut fetch: F,
) -> Option<(u8, bool)>
where
    F: FnMut(usize) -> i16,
{
    if !probe.phase_enabled {
        return None;
    }
    let values = probe.values;
    let negative = |slot: usize| values[slot] < 0;
    let present = |slot: usize| values[slot] != 0;
    let allowed = |class: u8| config.phase_mask & (1 << class) != 0;

    if present(NEIGHBOR_LEFT) && present(NEIGHBOR_TOP) && present(NEIGHBOR_DIAG) {
        if allowed(PHASE_SEPARABLE) {
            return Some((
                PHASE_SEPARABLE,
                negative(NEIGHBOR_LEFT) ^ negative(NEIGHBOR_TOP) ^ negative(NEIGHBOR_DIAG),
            ));
        }
    }

    let mut second_order = |slot: usize| -> i16 {
        if probe.entry[slot] == ABSENT {
            0
        } else {
            fetch(probe.entry[slot] as usize)
        }
    };

    if present(NEIGHBOR_LEFT) {
        let left2 = second_order(NEIGHBOR_LEFT2);
        if left2 != 0 {
            let flip = negative(NEIGHBOR_LEFT) != (left2 < 0);
            let class = if flip { PHASE_ROW_FLIP } else { PHASE_ROW_KEEP };
            if allowed(class) {
                return Some((class, left2 < 0));
            }
        }
    }

    if present(NEIGHBOR_TOP) {
        let top2 = second_order(NEIGHBOR_TOP2);
        if top2 != 0 {
            let flip = negative(NEIGHBOR_TOP) != (top2 < 0);
            let class = if flip { PHASE_COL_FLIP } else { PHASE_COL_KEEP };
            if allowed(class) {
                return Some((class, top2 < 0));
            }
        }
    }

    if present(NEIGHBOR_LEFT) && present(NEIGHBOR_TOP) && allowed(PHASE_PAIR) {
        return Some((PHASE_PAIR, negative(NEIGHBOR_LEFT) ^ negative(NEIGHBOR_TOP)));
    }
    if present(NEIGHBOR_LEFT) && allowed(PHASE_LEFT) {
        return Some((PHASE_LEFT, negative(NEIGHBOR_LEFT)));
    }
    if present(NEIGHBOR_TOP) && allowed(PHASE_TOP) {
        return Some((PHASE_TOP, negative(NEIGHBOR_TOP)));
    }
    if probe.parent_nonzero && allowed(PHASE_PARENT) {
        return Some((PHASE_PARENT, negative(NEIGHBOR_PARENT)));
    }
    None
}

/// Évaluation complète en un appel. Utilisée par la télémétrie et les tests,
/// où la lisibilité prime sur le coût.
pub fn evaluate<F>(
    table: &SpectralNeighborTable,
    position: usize,
    config: &SpectralEnvelopeConfig,
    phase_enabled: bool,
    mut fetch: F,
) -> SpectralDescriptor
where
    F: FnMut(usize) -> i16,
{
    let probed = probe(table, position, config, phase_enabled, 0, &mut fetch);
    let (phase_class, phase_negative) =
        resolve_phase_class(&probed, config, &mut fetch).unwrap_or((PHASE_NONE, false));
    SpectralDescriptor {
        amplitude: probed.amplitude,
        phase_class,
        phase_negative,
        energy: probed.energy,
        parent_nonzero: probed.parent_nonzero,
    }
}

// ======================================================================
// Curseur : parcours d'une séquence plate
// ======================================================================

/// Parcourt une séquence rANS plate segment par segment et produit le
/// descripteur de chaque symbole. Le même curseur sert à l'encodeur et au
/// décodeur ; seul diffère le tableau de valeurs qu'on lui présente.
pub struct SpectralCursor<'a> {
    segments: &'a [SpectralSegment],
    tables: &'a SpectralNeighborTables,
    config: SpectralEnvelopeConfig,
    phase_enabled: bool,
    cursor: usize,
}

impl<'a> SpectralCursor<'a> {
    pub fn new(
        segments: &'a [SpectralSegment],
        tables: &'a SpectralNeighborTables,
        config: SpectralEnvelopeConfig,
        channel: usize,
    ) -> Self {
        let phase_enabled = config.phase_enabled(channel);
        Self {
            segments,
            tables,
            config,
            phase_enabled,
            cursor: 0,
        }
    }

    /// Place le curseur sur le segment qui contient `flat`, et rend sa
    /// géométrie `(base, start)`. `None` si le symbole n'appartient à aucun
    /// segment, ce qui ne peut arriver que sur une description incohérente.
    #[inline]
    fn locate(&mut self, flat: usize) -> Option<(&'a SpectralSegment, usize, usize)> {
        while self.cursor < self.segments.len() {
            let segment = &self.segments[self.cursor];
            let base = segment.offset as usize;
            if flat < base {
                return None;
            }
            if flat < base + segment.len as usize {
                return Some((segment, base, segment.start as usize));
            }
            self.cursor += 1;
        }
        None
    }

    /// Première moitié du descripteur pour le symbole plat `flat`. `values`
    /// doit contenir au moins les symboles d'index `< flat` ; les appels
    /// doivent être strictement croissants en `flat`, ce qui est le cas des
    /// deux boucles rANS.
    pub fn probe(&mut self, flat: usize, values: &[i16], scale: u32) -> SpectralProbe {
        let Some((segment, base, start)) = self.locate(flat) else {
            return SpectralProbe::EMPTY;
        };
        let position = start + (flat - base);
        let table = self.tables.table(segment.table);
        let block = self.cursor as u32;
        let size_class = segment.table;
        let mut probed = probe(
            table,
            position,
            &self.config,
            self.phase_enabled,
            scale,
            |neighbor| fetch_neighbor(segment, base, start, values, neighbor),
        );
        probed.block = block;
        probed.size_class = size_class;
        probed
    }

    /// Seconde moitié, appelée seulement pour un coefficient non nul.
    pub fn phase(&self, probed: &SpectralProbe, flat: usize, values: &[i16]) -> Option<(u8, bool)> {
        let segment = self.segments.get(self.cursor)?;
        let base = segment.offset as usize;
        let start = segment.start as usize;
        debug_assert!(flat >= base && flat < base + segment.len as usize);
        resolve_phase_class(probed, &self.config, |neighbor| {
            fetch_neighbor(segment, base, start, values, neighbor)
        })
    }

    /// Descripteur complet — télémétrie et tests.
    pub fn descriptor(&mut self, flat: usize, values: &[i16]) -> SpectralDescriptor {
        let probed = self.probe(flat, values, 0);
        let (phase_class, phase_negative) = self
            .phase(&probed, flat, values)
            .unwrap_or((PHASE_NONE, false));
        SpectralDescriptor {
            amplitude: probed.amplitude,
            phase_class,
            phase_negative,
            energy: probed.energy,
            parent_nonzero: probed.parent_nonzero,
        }
    }
}

/// Un voisin vit soit dans le préfixe hérité d'une couche antérieure, soit dans
/// la séquence plate courante — jamais ailleurs, puisque la table ne rend que
/// des positions strictement antérieures.
#[inline]
fn fetch_neighbor(
    segment: &SpectralSegment,
    base: usize,
    start: usize,
    values: &[i16],
    neighbor: usize,
) -> i16 {
    if neighbor < start {
        segment.prefix[neighbor.min(MAX_PREFIX - 1)]
    } else {
        values.get(base + neighbor - start).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zigzag(block_size: usize) -> Vec<usize> {
        let n = block_size;
        let mut order = Vec::with_capacity(n * n - 1);
        for s in 0..(2 * n - 1) {
            if s % 2 == 0 {
                let i_start = s.min(n - 1);
                let i_end = if s >= n { s - n + 1 } else { 0 };
                let mut i = i_start as i64;
                while i >= i_end as i64 {
                    let j = s - i as usize;
                    let idx = i as usize * n + j;
                    if idx != 0 {
                        order.push(idx);
                    }
                    i -= 1;
                }
            } else {
                let i_start = if s >= n { s - n + 1 } else { 0 };
                let i_end = s.min(n - 1);
                for i in i_start..=i_end {
                    let j = s - i;
                    let idx = i * n + j;
                    if idx != 0 {
                        order.push(idx);
                    }
                }
            }
        }
        order
    }

    fn tables() -> SpectralNeighborTables {
        SpectralNeighborTables::new(&zigzag(8), &zigzag(16), &zigzag(32))
    }

    #[test]
    fn every_neighbor_is_strictly_causal() {
        for &size in &[8usize, 16, 32] {
            let order = zigzag(size);
            let table = SpectralNeighborTable::new(&order, size);
            assert_eq!(table.len(), size * size - 1);
            for position in 0..table.len() {
                for slot in 0..SPECTRAL_NEIGHBORS {
                    let neighbor = table.entry(position)[slot];
                    if neighbor != ABSENT {
                        assert!(
                            (neighbor as usize) < position,
                            "size={size} position={position} slot={slot} neighbor={neighbor}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn parent_slot_matches_the_v16_definition() {
        let size = 16usize;
        let order = zigzag(size);
        let table = SpectralNeighborTable::new(&order, size);
        let mut rank = vec![ABSENT; size * size];
        for (position, &natural) in order.iter().enumerate() {
            rank[natural] = position as u32;
        }
        for (position, &natural) in order.iter().enumerate() {
            let row = natural / size;
            let col = natural % size;
            let expected = if row / 2 == 0 && col / 2 == 0 {
                ABSENT
            } else {
                rank[(row / 2) * size + col / 2]
            };
            assert_eq!(table.entry(position)[NEIGHBOR_PARENT], expected);
        }
    }

    /// La loi séparable doit être exacte sur un contenu séparable synthétique.
    #[test]
    fn separable_phase_law_is_exact_on_separable_content() {
        let size = 16usize;
        let order = zigzag(size);
        let table = SpectralNeighborTable::new(&order, size);
        let mut config = SpectralEnvelopeConfig::neutral();
        config.phase_mask = PHASE_MASK_SEPARABLE;
        config.phase_channels = 0b111;

        for (y0, x0) in [(3usize, 5usize), (0, 0), (7, 11), (15, 2)] {
            let alpha = std::f64::consts::PI * (y0 as f64 + 0.5) / size as f64;
            let beta = std::f64::consts::PI * (x0 as f64 + 0.5) / size as f64;
            let mut natural = vec![0f64; size * size];
            for v in 0..size {
                for u in 0..size {
                    natural[v * size + u] = (v as f64 * alpha).cos() * (u as f64 * beta).cos();
                }
            }
            let coeffs: Vec<i16> = order
                .iter()
                .map(|&idx| {
                    let value = natural[idx];
                    // Quantification grossière : seuls les signes importent.
                    if value.abs() < 1e-9 {
                        0
                    } else if value > 0.0 {
                        1
                    } else {
                        -1
                    }
                })
                .collect();

            let mut checked = 0usize;
            for position in 0..coeffs.len() {
                if coeffs[position] == 0 {
                    continue;
                }
                let descriptor = evaluate(&table, position, &config, true, |q| coeffs[q]);
                if descriptor.phase_class != PHASE_SEPARABLE {
                    continue;
                }
                checked += 1;
                assert_eq!(
                    descriptor.phase_negative,
                    coeffs[position] < 0,
                    "impulse=({y0},{x0}) position={position}"
                );
            }
            assert!(
                checked > 100,
                "expected a dense separable coverage, got {checked}"
            );
        }
    }

    #[test]
    fn cursor_reproduces_direct_evaluation_across_layers() {
        let size = 16usize;
        let order = zigzag(size);
        let tables = tables();
        let table_class = SpectralNeighborTables::table_class(size);
        let mut config = SpectralEnvelopeConfig::neutral();
        config.amplitude_mode = AMPLITUDE_MAGNITUDE;
        config.phase_mask = PHASE_MASK_ALL;
        config.phase_channels = 0b111;

        // Deux blocs, couche de détail : les cinq premières positions viennent
        // du squelette et le reste de la séquence plate courante.
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 16) as i32 % 7 - 3) as i16
        };
        let blocks: Vec<Vec<i16>> = (0..2)
            .map(|_| (0..order.len()).map(|_| next()).collect())
            .collect();

        let start = MAX_PREFIX;
        let mut segments = Vec::new();
        let mut flat = Vec::new();
        for block in &blocks {
            segments.push(SpectralSegment::detail(
                flat.len(),
                start,
                block.len() - start,
                table_class,
                &block[..start],
            ));
            flat.extend_from_slice(&block[start..]);
        }

        let mut cursor = SpectralCursor::new(&segments, &tables, config, 0);
        let table = tables.table(table_class);
        let mut index = 0usize;
        for block in &blocks {
            for position in start..block.len() {
                let expected = evaluate(table, position, &config, true, |q| block[q]);
                let observed = cursor.descriptor(index, &flat);
                assert_eq!(observed, expected, "position={position}");
                index += 1;
            }
        }
    }
}
