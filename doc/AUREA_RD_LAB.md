# AUREA R&D Lab

## Intention

AUREA reste un codec expérimental où les mécanismes naturels ne servent pas
de décoration : ils doivent produire une règle de codage mesurable, symétrique
et intelligible. Une idée n'entre dans le profil par défaut qu'après avoir
montré son apport sur des images absentes de sa calibration.

## Règles du laboratoire

1. Une force dérivée d'un signal déjà décodé doit rester **zéro bit** et être
   reconstruite par exactement le même calcul à l'encodage et au décodage.
2. Chaque expérience possède un état neutre qui reproduit la base courante.
3. Le verdict principal est la courbe débit-distorsion, pas un seul point de
   qualité. RGB, luminance, SSIM, taille et temps restent tous visibles.
4. La calibration et la validation utilisent des ensembles distincts. Un gain
   doit survivre au Kodak complet et à une vérification de robustesse.
5. Un mécanisme qui ne gagne pas est documenté puis retiré du chemin actif :
   le codec célèbre aussi les équilibres en refusant l'accumulation gratuite.

## Cycle 1 — Homéostasie du budget perceptuel

Le champ de Turing produit une modulation locale, puis le pivot psychovisuel
l'oriente selon la qualité. L'expérience ajoute une conservation globale en
log-espace :

`m'_i = m_i / exp(eta * moyenne_ponderee(log(m)))`

Le poids d'un bloc est son nombre de coefficients AC (`taille² - 1`). À
`eta = 0`, le profil v13 recalibré est reproduit exactement. À `eta = 1`, la
moyenne géométrique pondérée des pas locaux vaut 1 : raffinement et
coarsissement se compensent à l'échelle de l'image.

Paramètre de laboratoire : `AUREA_TURING_HOMEOSTASIS`, borné à `[0, 1]`.
Pendant l'exploration, la même valeur doit être fournie à l'encodeur et au
décodeur. Le défaut reste `0` tant que le benchmark n'a pas prononcé son
verdict.

**Verdict 2026-08-14 : non promu.** `eta=0.75` perd `+0,0125 %` sur le split
d'apprentissage puis gagne `-0,0065 %` sur Kodak 24/4 qualités, avec IC95
`[-0,0227 ; +0,0101]`. Le signe et l'intervalle ne permettent pas de conclure.

## Pistes suivantes

- **Cohérence tensorielle** : remplacer le seuil directionnel binaire par un
  mélange continu gouverné par la cohérence du tenseur de structure.
- **Morphogenèse multi-échelle** : faire dialoguer des couples
  activateur/inhibiteur séparés selon des échelles liées par phi, sans side
  information.
- **Respiration chromatique** : explorer une conservation couplée luma/chroma
  où la chroma ne consomme du budget que lorsque sa structure diverge vraiment
  de la luminance.

## Cycle 2 — Cohérence tensorielle directionnelle

La moyenne vectorielle historique peut annuler deux gradients opposés alors
qu'ils décrivent la même arête. Le tenseur de structure additionne leurs
seconds moments, reconstruit une orientation axiale (période pi) et mesure sa
cohérence entre 0 (isotrope) et 1 (unidirectionnelle).

Le paramètre `AUREA_DIRECTIONAL_COHERENCE` interpole depuis le comportement
v13 exact (`0`) vers l'orientation tensorielle avec une influence QMAT égale à
la cohérence (`1`). Le mélange des QMAT se fait en log-espace afin de respecter
leur nature multiplicative. Le défaut reste neutre jusqu'au verdict Kodak.

**Verdict 2026-08-14 : non promu.** `eta=0.75` gagne `-0,0008 %` sur le split
mais perd `+0,0188 %` sur Kodak 24/4 qualités, avec seulement 7 images gagnantes
sur 24. La direction tensorielle reste disponible pour d'autres expériences,
mais sa force compilée demeure nulle.

## Cycle 3 — Morphogenèse emboîtée selon phi

Le champ historique observe une seule paire activateur/inhibiteur. Le prototype
ajoute un second champ zéro-bit dont le sigma activateur est multiplié par phi.
À pleine force, les modulations locale et large sont mêlées en log-espace avec
les poids `phi^-1` et `phi^-2`, dont la somme vaut exactement 1. Le champ local
reste responsable des contextes entropiques et de la direction ; seule
l'allocation des pas reçoit la respiration de l'échelle supérieure.

Paramètre : `AUREA_TURING_MULTISCALE`, de `0` (v13 exacte, sans calcul
supplémentaire) à `1` (mélange d'or complet).

**Verdict 2026-08-14 : non promu.** Le signal initial à 4 qualités était
cohérent (`-0,0031 %` apprentissage, `-0,0054 %` Kodak 24), mais le test complet
à 8 qualités inverse l'apprentissage (`+0,0030 %`). Sur Kodak 24, le BD-Rate
cubique RGB/luma combiné vaut `-0,0019 %`, IC95
`[-0,0066 ; +0,0027]`, pour 13 images gagnantes sur 24. Le second champ ajoute
en outre `+31,3 %` au temps de décodage sur 8 passages alternés à `kodim01/q60`.
Le rapport gain/coût est nettement défavorable ; le défaut reste `0` et évite
entièrement le calcul de la seconde échelle.

## Bilan du cycle

Les trois expériences sont reconstructibles, symétriques et bit-exactes dans
leur état neutre. Aucune ne franchit le seuil de promotion. La base v13
recalibrée reste donc le profil actif, sans changement de bitstream. Les
branches de laboratoire restent opt-in afin de permettre des croisements
futurs, mais elles n'ajoutent aucune force au chemin par défaut.

## Cycle 4 — Rétinotopie phyllotaxique

Cette expérience ne change aucun pas de quantification. Elle remplace seulement
l'ordre raster des blocs AC par une spirale de Fermat à angle d'or. Les deltas
d'EOB restent raster afin de conserver leur continuité prédictive. L'arc possède
une densité fixe en cellules 8×8 : une résolution plus grande prolonge la
trajectoire au lieu de renormaliser son cœur.

La fovéa est le barycentre quantifié du gradient inhibiteur Turing reconstruit
depuis le DC. L'ordre est donc zéro-bit. Son activation occupe un bit déjà
réservé dans les drapeaux v13 et n'ajoute aucun octet ; le fichier indique
lui-même au décodeur le parcours à employer. Paramètre encodeur de laboratoire :
`AUREA_SPIRAL_SCAN_MODE=1`.

Le premier mode phyllotaxique distribue les points par sauts successifs de
137,5°. Il préserve parfaitement l'image mais augmente le BD-Rate de `+0,270 %`
sur le split et `+0,214 %` sur Kodak 24 : bon échantillonnage, mauvaise
trajectoire pour un modèle adaptatif.

Le mode `AUREA_SPIRAL_SCAN_MODE=2` projette au contraire les centres de blocs
sur une spirale d'Archimède continue `r = theta/(2*pi)`, de pas radial fixe égal
à une cellule. Il teste l'hypothèse complémentaire : conserver la localité tout
en offrant un préfixe centre-vers-périphérie extensible.

**Verdict 2026-08-14 : ordre physique non promu.** Le mode continu réduit bien
la perte du premier parcours, mais reste à `+0,194 %` sur le split et
`+0,142 %` sur Kodak 24. La permutation perturbe davantage l'adaptation rANS
qu'elle ne rapproche les blocs utiles. Les deux parcours restent disponibles
comme primitives de laboratoire, tandis que le profil actif conserve l'ordre
raster.

## Cycle 5 — Mémoire saccadique et surprise spiralée

La spirale devient ici un champ psycho-visuel et ne déplace plus les symboles
dans le flux entropique. Le long de l'arc continu, un prédicteur à mémoire d'or
suit le logarithme de la modulation Turing :

`p_i = phi^-1 * p_(i-1) + phi^-2 * log(m_i)`

La surprise locale `s_i = abs(log(m_i) - p_(i-1))` mesure une rupture par
rapport à cette mémoire. Après centrage par le nombre de coefficients AC de
chaque bloc, la loi expérimentale affine les ruptures et fait payer leur coût
aux zones prévisibles :

`log(m'_i) = log(m_i) - eta * (s_i - moyenne_AC(s))`

La somme AC pondérée des logarithmes reste donc constante : le mécanisme
redistribue le budget sans changer secrètement la qualité globale. À `eta=0`,
le bitstream v13 recalibré est reproduit octet pour octet. Paramètre de
laboratoire : `AUREA_SPIRAL_FIELD_STRENGTH`, fourni symétriquement à l'encodeur
et au décodeur.

**Verdict 2026-08-14 : non promu.** Toutes les forces non nulles perdent sur le
split (`+0,010 %` à `+0,057 %`). `eta=0,25`, le meilleur résultat de validation,
donne `-0,0030 %` sur Kodak 24 mais seulement 11 images gagnantes sur 24, avec
un IC95 `[-0,0166 ; +0,0106]`. Cette inversion de signe et l'intervalle centré
sur zéro indiquent du bruit, pas un gain reproductible. La trajectoire spiralée
est validée comme support mathématique ; la surprise issue de la seule
modulation Turing n'est pas encore le bon signal d'allocation.

## Cycle 6 — Spirale spectrale repliée

Le prototype suivant agit à l'intérieur de chaque transformée plutôt qu'entre
les blocs. Une spirale d'Archimède `r = theta/(2*pi)`, de pas radial égal à un
coefficient, est repliée par valeur absolue dans le quadrant DCT positif. Chaque
coefficient rejoint la visite continue la plus proche de l'arc. La clé de
parcours ne dépend pas de la taille : après projection, l'ordre 8×8 est
strictement préservé dans les ordres 16×16 et 32×32.

Le mode est auto-décrit par le drapeau `0x2000` et s'active avec
`AUREA_SPECTRAL_SCAN_MODE=1`. Un fichier expérimental se décode donc sans
reproduire l'environnement de l'encodeur.

**Verdict 2026-08-14 : arrêt au point sentinelle.** Sur `kodim01/q60`, le flux
passe de 150 754 à 155 332 octets (`+3,037 %`). Le parcours extensible disperse
trop tôt des fréquences coûteuses et affaiblit fortement l'EOB ; un balayage
Kodak complet ne peut raisonnablement renverser un tel écart. Le résultat fixe
la frontière conceptuelle : la spirale doit ordonner des *couches déjà
hiérarchisées par importance*, pas remplacer directement l'ordre spectral
basse-fréquence-vers-haute-fréquence.

## Direction retenue — paquets perceptuels concentriques

Les essais séparent désormais trois rôles. Le raster reste le meilleur support
des contextes rANS, le zigzag reste le meilleur support spectral, et la spirale
est pertinente comme ordre de présentation progressif. La suite logique est un
conteneur en couches : DC et contours structurants d'abord, puis paquets AC
classés par importance perceptuelle, chaque couche conservant en interne son
ordre entropique optimal. L'arc fixe ordonnerait les paquets et pourrait se
prolonger avec la résolution sans imposer sa géométrie aux symboles eux-mêmes.

## Cycle 7 — Paquets perceptuels concentriques

Le conteneur retenu sépare enfin les trois ordres. Le DC, les métadonnées CfL et
les EOB forment une base commune. La première couche AC apporte les cinq
premiers coefficients zigzag de chaque bloc : cette trame structurelle globale
évite qu'une périphérie sans détails domine les premiers aperçus. Les couches
suivantes suivent la spirale continue par arcs fixes de 233 blocs. Dans chaque
arc, les blocs retrouvent l'ordre raster avant leur propre flux rANS.

Le fichier est physiquement organisé par couche et par canal :

`base -> structure(L,C1,C2) -> arc1(L,C1,C2) -> arc2(...) -> ...`

Chaque flux est indépendant. `aurea decode image.aur aperçu.png --layers N`
reconstruit les `N` premières couches et indique l'octet exact où ce préfixe se
termine. Un fichier réellement tronqué à cette frontière donne la même image
que le préfixe correspondant du fichier complet. Le mode occupe le drapeau
auto-décrit `0x4000`.

Sur `kodim01/q60`, la structure progressive complète passe de 150 754 à
150 244 octets (`-0,338 %`) et produit exactement le même PNG. La base DC pèse
8,2 % du flux ; la couche structurelle atteint 21,08 dB RGB à 18,7 %. Les arcs
suivants font monter la qualité jusqu'à 36,65 dB sans régression.

**Verdict 2026-08-14 : promu dans le profil par défaut.** Le BD-Rate combiné
RGB/luma gagne `-0,2065 %` sur le split puis `-0,3105 %` sur Kodak 24/4
qualités. 23 images sur 24 gagnent et l'IC95 `[-0,3790 ; -0,2419]` exclut zéro.
La division des cinq premiers AC améliore suffisamment la spécialisation des
modèles rANS pour payer les en-têtes de couches et produire un gain net. La
variable `AUREA_PROGRESSIVE_PACKETS=0` conserve seulement le flux monolithique
comme témoin de laboratoire ; la valeur par défaut est désormais active.

## Cycle 8 — Respiration chromatique structurelle

La chroma ne reçoit plus un facteur spatial uniforme. Pour chaque canal C1/C2,
l'encodeur et le décodeur développent les DC déjà quantifiés sur la même grille
8×8 que le champ Turing, puis calculent les gradients normalisés de luminance
`gL` et de chroma `gC`. L'activité chromatique et la part non expliquée par la
luminance donnent, cellule par cellule :

`aC = c / (c + 0,5)`

`d = aC * (1 - rho² * min(1, l/c))`

où `l = |gL|/RMS(gL)`, `c = |gC|/RMS(gC)` et
`rho = |gL.gC|/(|gL||gC|)`. La valeur absolue rend la cohérence axiale : deux
gradients opposés décrivent encore le même contour. Un canal chroma plat reste
strictement neutre.

La divergence moyenne de chaque bloc adaptatif est recentrée par son nombre de
coefficients AC `w_b = taille_b² - 1` :

`m_b = exp(ln(phi) * (d_b - moyenne_AC(d)))`

Ainsi `somme(w_b * ln(m_b)) = 0`. La moyenne géométrique pondérée des pas chroma
reste exactement 1 : aucun gain ne vient d'une baisse cachée de qualité. Les
contours chroma ancrés par la luminance reçoivent un pas plus fin ; les détails
chromatiques isolés, auxquels l'œil accorde une acuité spatiale moindre, cèdent
du budget. Chaque facteur individuel reste dans `[phi^-1, phi]`.

Deux hypothèses ont été éliminées avant promotion. Protéger en priorité la
divergence chromatique perd `+0,0148 %` sur Kodak 24 dès `eta=0,125`. Inverser
la loi mais la conditionner par les décisions CfL gagne seulement `-0,0383 %`
au meilleur point et reste instable. La divergence DC pure est à la fois plus
simple et beaucoup plus forte ; CfL n'intervient donc pas dans le mécanisme
final.

La force `eta=1` est auto-décrite : le drapeau `0x8000` active le mode et
l'ancien octet réservé contient l'amplitude quantifiée sur 8 bits (`255` pour
le profil promu). La carte de divergence elle-même ne coûte aucun bit. Le
témoin `AUREA_CHROMA_BREATHING=0` reproduit octet pour octet la base progressive
du cycle 7.

**Verdict 2026-08-14 : promu dans le profil par défaut.** Sur le split,
`eta=1` gagne `-0,4204 %` de BD-Rate combiné contre la base progressive. Sur
Kodak 24/4 qualités, le gain vaut `-0,3605 %` (`-0,1356 %` RGB et `-0,5855 %`
luminance), avec 23 images gagnantes sur 24, une médiane de `-0,3546 %` et un
IC95 bootstrap `[-0,4382 ; -0,2827]`. Les débits moyens baissent aux quatre
qualités : `-0,764 %` à q20, `-0,736 %` à q40, `-0,641 %` à q60 et `-0,375 %`
à q80. Sur `kodim01/q60`, le flux passe de 150 244 à 149 722 octets. Le profil
promu reproduit exactement le SHA-256 du candidat de laboratoire correspondant.

## Cycle 9 — Frontière JPEG native et fidélité chromatique

La version promue du cycle 8 a été confrontée directement à libjpeg-turbo via
Pillow 12.3 sur les douze PNG `originals`, à leur résolution native. AUREA est
mesuré aux qualités 20/40/60/80. JPEG parcourt seize qualités de 5 à 100. Deux
adversaires sont séparés : 4:2:0 optimisé et 4:4:4 optimisé. Pour chaque point
JPEG, le plus petit flux entre séquentiel et progressif est retenu. Les courbes
sont évaluées en PSNR RGB, Y, Cb et Cr, SSIM-Y, ainsi qu'en CIEDE2000 moyen et
p95. Le BD-Rate cubique historique est contrôlé par une intégration linéaire.

Face au JPEG 4:4:4, AUREA gagne `-18,119 %` de BD-Rate RGB sur 12 images sur 12
(IC95 `[-22,248 ; -14,075]`) ; le contrôle linéaire donne `-18,785 %`. Les
autres axes confirment le déplacement de frontière : `-4,891 %` en Y,
`-29,103 %` en Cb, `-40,725 %` en Cr, `-28,389 %` selon le Delta-E moyen et
`-20,259 %` selon son p95. `kryte` est le cas limite, mais reste gagnant contre
le 4:4:4 : `-1,731 %` RGB et `-1,269 %` Delta-E moyen.

Face au JPEG 4:2:0, AUREA gagne `-20,976 %` RGB sur 11 images sur 12 et
`-42,864 %` selon le Delta-E moyen sur 12 sur 12. Les gains Cb/Cr atteignent
respectivement `-67,175 %` et `-72,428 %`. Sur `phoenix`, le Cr AUREA le plus
dégradé (`43,314 dB`) dépasse même le meilleur JPEG 4:2:0 à q100
(`42,884 dB`) : les courbes ne se recouvrent pas et le BD-Rate Cr n'est donc
pas défini. En contrepartie, le JPEG 4:2:0 gagne `+19,955 %` contre AUREA en Y
seul et `+11,749 %` en SSIM-Y, car il abandonne les trois quarts des échantillons
chroma pour concentrer son budget sur la luminance. `kryte` est l'unique perte
RGB d'AUREA contre ce profil (`+4,327 %`), tout en restant gagnant en Delta-E
(`-12,623 %`).

La victoire débit-distorsion ne s'étend pas encore à la vitesse. AUREA demande
en moyenne 19,523 s à l'encodage et 11,344 s au décodage, contre 0,409 s et
0,316 s pour JPEG 4:4:4 : environ 48× et 36× plus lent. Cette dette de calcul
est désormais le principal écart objectif avec JPEG.

**Verdict 2026-08-14 : objectif JPEG atteint sur la frontière couleur.** AUREA
bat de façon robuste le JPEG optimisé à chroma équivalente et domine largement
la fidélité couleur du JPEG 4:2:0. La revendication doit rester précise : le
4:2:0 conserve une meilleure efficacité luminance et JPEG reste très supérieur
en vitesse. Les prochains travaux doivent viser `kryte`, la frontière Y du
4:2:0 et l'optimisation du temps de calcul sans sacrifier le gain couleur.

## Cycle 10 — Dette de calcul

Ce cycle ne touche ni la quantification ni le bitstream : il rembourse la
dette de vitesse mesurée au cycle 9 sous une contrainte plus forte que la
règle habituelle du laboratoire. Chaque levier est **bit-exact** : le flux
`.aur` et le PNG décodé restent identiques octet pour octet à la base v13
(SHA-256 vérifiés, sentinelle `kodim01/q60` = 149 722 octets — la valeur du
cycle 8 — plus round-trips 1×1, 61×97, 257×511 à q60 et q95). Le BD-Rate des
cycles 8-9 est donc conservé par construction, sans nouveau benchmark.

L'instrumentation `AUREA_TIMINGS=1` a d'abord établi le profil réel : le
trellis occupait 91 % de l'encodage à résolution native, et la
déquantification 59 % du décodage — le décodeur recalculait `powf(QMAT)` et
le CSF pour chaque coefficient, quand l'encodeur lisait des LUT.

Leviers principaux, tous symétriques quand le calcul est partagé :
les snapshots du trellis (640 octets copiés par position, soit ~7 Go de
memcpy sur une image 4 MP) deviennent des deltas de 16 octets rejoués
incrémentalement, le modèle v12 ne modifiant qu'une entrée P(zero) et au
plus une P(±1) par symbole ; la quantification est parallélisée par bloc,
le modèle greedy repartant de son état initial à chaque bloc ; le décodeur
reçoit le miroir exact des LUT QMAT/CSF de l'encodeur ; l'overlap-add de la
synthèse LOT passe en bandes de lignes disjointes qui préservent l'ordre
des sommes flottantes ; les passes colonnes des DCT/IDCT deviennent
ligne-major à chaînes d'accumulation identiques. La récursion rapide de Lee
a été écartée volontairement : elle changerait les arrondis et briserait la
bit-exactitude, pour un gain devenu marginal une fois l'overlap-add corrigé.

**Verdict 2026-08-14 : promu — vitesse 6,3× / 4,3×, fidélité intacte.** Sur
les douze `originals` natifs aux quatre qualités du protocole cycle 9,
l'encodage moyen passe de 19,523 s à 3,101 s et le décodage de 11,344 s à
2,636 s. L'écart avec le JPEG 4:4:4 optimisé tombe de 48× à **7,6×** à
l'encodage — l'objectif ≤ 10× est atteint — et de 36× à **8,3×** au
décodage, encore au-dessus de la cible ≤ 5×. Les chiffres sont mesurés sur
un i7-4700MQ à 4 cœurs ; les pistes restantes (rANS parallèle par couche,
plans en `f32`) sont documentées dans `doc/perf/phase_final.md` et devront,
elles, repasser par le verdict débit-distorsion complet.

## Cycle 11 — Contre-don de Weber (compensation des hautes lumières)

Le diagnostic de la sentinelle `kryte` (`doc/perf/kryte_diagnosis.md`) a
d'abord éliminé les suspects réglables : respiration chromatique, champ
Turing, pivot, lambda trellis et facteur φ du canal C1 sont tous neutres ou
RD-positifs sur kryte. L'obscurité globale non plus n'explique rien —
fine_ally et agree, plus sombres que kryte, sont les deux plus grandes
victoires du corpus. L'analyse par tuiles à débit apparié a livré la vraie
signature, parfaitement monotone en luminance : ratio MSE AUREA/JPEG de
0,83 dans les tuiles noires à 1,42 dans les hautes lumières. La conception
Weber (PTF gamma 0,65) concède les zones claires par construction ; sur les
autres images sombres, la domination chroma rembourse largement cette
concession, mais la chroma de kryte est quasi plate (47-53 dB pour tous les
codecs) : il ne reste que la guerre luminance, et AUREA y cédait les lampes,
fenêtres et surfaces claires.

Le mécanisme promu rend une partie du don : un champ zéro-bit sur le canal
L uniquement, reconstruit du DC déjà transmis. La luminance moyenne de
chaque bloc en espace PTF, `B_b` ∈ [0,1], est recentrée par les poids AC
`w_b = taille_b² - 1` :

`m_b = exp(-ln(phi) * eta * (B_b - moyenne_AC(B)))`

Ainsi `somme(w_b * ln(m_b)) = 0` : la moyenne géométrique pondérée des pas
luma reste exactement 1, aucun gain caché. Les blocs clairs reçoivent un pas
plus fin, financé par les blocs sombres où la PTF sur-livre déjà. Chaque
facteur reste dans `[phi^-eta, phi^eta]`.

La force est auto-décrite : quantifiée sur 5 bits dans les bits hauts de
l'octet scene_type (les fichiers v13 antérieurs y portent zéro, l'état
neutre exact ; le décodeur n'a jamais lu ces bits). Le témoin
`AUREA_BRIGHT_COMPENSATION=0` reproduit octet pour octet le profil du
cycle 10.

**Verdict 2026-08-14 : promu dans le profil par défaut à `eta=1`.** La
calibration (sweep eta sur kryte seule) et la validation sont séparées.
Sur Kodak 24/4 qualités, le BD-Rate combiné RGB/luma gagne `-0,428 %`
(RGB `-0,442 %`, luma `-0,414 %`), 20 images sur 24, médiane `-0,442 %`,
IC95 `[-0,751 ; -0,074]` excluant zéro. Sur les douze `originals`, le gain
est `-2,027 %` combiné avec **12 victoires sur 12**, IC95
`[-2,520 ; -1,533]` — y compris les championnes sombres fine_ally
(`-2,32 %`) et agree (`-1,73 %`) que le recentrage par image devait
protéger, et kryte elle-même (`-2,08 %`, `-2,43 %` en Y). La sentinelle
kodim01/q60 passe de 149 722 octets / 36,6425 dB à 150 119 octets /
36,7297 dB. Face au JPEG 4:2:0, la perte RGB résiduelle de kryte est
environ divisée par deux ; la frontière Y du 4:2:0 reste le prochain
adversaire.

## Cycle 12 — Échange spectral (« sous-échantillonnage doux », AUR2 v14)

La force du JPEG 4:2:0 est un troc : jeter trois quarts des échantillons
chroma pour concentrer le budget sur la luminance. Ce cycle offre à AUREA le
même troc, mais gradué, adaptatif et sans casse structurelle : au lieu de
réduire la résolution chroma, l'encodeur coarcit les pas des hautes
fréquences chroma — précisément ce que le sous-échantillonnage détruit — et
affine symétriquement les pas luma.

Le mécanisme reste fidèle à la charte. La force locale est un champ zéro-bit
dérivé des trois grilles DC denses déjà décodées :

`gamma = eta * (1 - min(1, RMS(gradC) / RMS(gradL)))`

Une image à chroma structurée (phoenix) s'auto-protège (gamma faible) ; une
chroma plate (kryte, gamma = 0,68) échange à pleine force. Les pas chroma
reçoivent `phi^(gamma * fn)` où `fn` est la fréquence normalisée du
coefficient, et les pas luma `phi^(-delta)` avec
`delta = 2 * gamma * moyenne_AC(fn)` : la moyenne log-espace pondérée AC de
tous les multiplicateurs des trois canaux est exactement nulle.

Les seize drapeaux et les octets réservés étant épuisés, la promotion passe
par la version du conteneur : un flux avec échange est marqué **AUR2 v14**
et le décodeur obéit à la version seule (jamais à l'environnement). Le
témoin `AUREA_SPECTRAL_EXCHANGE=0` émet un flux v13 identique octet pour
octet au profil du cycle 11, et les fichiers v13/v14 se décodent chacun
exactement — vérifié par SHA-256.

**Verdict 2026-08-14 : promu (défaut v14).** Calibration sur un trio
contrasté (kryte, phoenix, kodim03), validation séparée. Kodak 24/4
qualités : BD-Rate combiné RGB/luma **-6,60 %** (RGB -4,69 %, Y -8,52 %),
**24/24**, IC95 [-7,27 ; -5,86]. Originals : **-7,33 %** combiné
(RGB -3,99 %, **Y -10,68 %**), **12/12**, IC95 [-8,33 ; -6,31]. Le prix est
payé exclusivement en PSNR chroma (Cb +15,5 %, Cr +18,8 % contre sa propre
base) — une métrique sans témoin perceptuel : à débit apparié contre JPEG,
le **CIEDE2000 s'améliore aussi** (kryte 1,193→1,164 ; phoenix
1,100→1,079), car la luminance affinée domine le Delta-E. Sur les quatre
images sondes, AUREA v14 bat désormais les deux profils JPEG en RGB et en
Delta-E à octets inférieurs, et atteint la quasi-parité Y avec le 4:2:0 —
sur kryte même, le duel contre le 4:2:0 devient un match nul (RGB 42,19 vs
42,26, Delta-E 1,164 vs 1,159) avec 3,7 % d'octets de moins. La frontière Y
du 4:2:0, +19,96 % au cycle 9, est environ divisée par deux.

## Cycle 13 — Validation JPEG native complète de la v14

Le protocole du cycle 9 a été rejoué intégralement, sans réutiliser les ancres
précédentes : 12 PNG `originals` à résolution native, AUREA v14 aux qualités
20/40/60/80, JPEG aux seize qualités 5..100, profils 4:2:0 et 4:4:4, plus petit
flux entre séquentiel et progressif optimisés. Les 432 points portent PSNR
RGB/Y/Cb/Cr, SSIM-Y et CIEDE2000 moyen/p95. Les 384 points JPEG reproduisent
exactement le cycle 9 : mêmes octets, modes entropiques et métriques (écart
maximal nul). Le binaire v14 mesuré porte le SHA-256
`59ba9146c6408bb2a8a26116b8355cf6b3bc5025184801c201f55e92c29679a5`.

Face au JPEG 4:4:4, la v14 gagne **-23,299 % RGB** et **-17,330 % Y**, 12
images sur 12 dans les deux cas. Les autres verdicts restent favorables :
Cb -18,449 % (11/12), Cr -29,180 % (11/12), SSIM-Y -24,644 % (12/12),
Delta-E moyen -23,954 % (11/12) et p95 -15,760 % (10/12). Le contrôle
linéaire confirme RGB (-23,627 %) et Y (-18,037 %). La réserve chromatique
est localisée : `kryte` gagne encore RGB (-6,115 %) et Y (-9,864 %), mais
perd Cb (+27,297 %), Cr (+5,711 %) et Delta-E moyen (+10,847 %) ; `chilling`
ne perd que le Delta-E p95 (+0,971 %).

Face au JPEG 4:2:0, la v14 gagne désormais **-28,614 % RGB sur 12/12** :
`kryte`, seule défaite du cycle 9, bascule juste devant à -0,323 %. Le
Delta-E moyen gagne -41,000 % et son p95 -34,728 %, tous deux 12/12 ; Cb et
Cr gagnent respectivement -60,471 % et -66,836 % sur les onze courbes dont
le domaine se recouvre. La frontière Y résiduelle tombe de +19,955 % à
**+6,032 %** : elle reste à l'avantage du 4:2:0 en moyenne, mais l'écart est
divisé par plus de trois. Le SSIM-Y est indécis : cubique -2,408 % avec IC95
[-7,241 ; +2,192], contre +8,262 % au contrôle linéaire ; aucune revendication
de victoire ne doit reposer sur cette interpolation.

Contre la frontière v13 archivée du cycle 9, le gain cumulé de la v14 est
-5,716 % RGB, -12,697 % Y et -11,490 % SSIM-Y, chaque fois 12/12. Le troc
spectral apparaît sans ambiguïté : Cb +15,298 %, Cr +18,853 %, Delta-E moyen
+4,626 % et p95 +5,193 % contre cette ancienne base. À qualité nominale fixe,
la v14 dépense 6,7 à 14,5 % de bits supplémentaires mais gagne 0,46 à 1,08 dB
RGB et 1,41 à 1,69 dB Y ; le BD-Rate mesure le déplacement favorable de la
frontière, pas un flux v14 plus petit à numéro de qualité identique.

La vitesse progresse fortement contre le binaire v13 archivé : 4,740 s contre
19,523 s à l'encodage (-75,7 %) et 4,557 s contre 11,344 s au décodage
(-59,8 %). Elle reste néanmoins la dette principale face au JPEG 4:4:4 de ce
banc : 10,2× plus lente à l'encodage et 13,0× au décodage.

**Verdict 2026-08-14 : promotion v14 confirmée.** AUREA bat le JPEG optimisé
sur la frontière RGB des 12 images contre les deux profils, renverse nettement
la luminance du 4:4:4 et réduit fortement son retard Y contre le 4:2:0. La
revendication honnête est désormais « meilleur compromis RGB/structure que
JPEG sur ce corpus », avec deux réserves explicites : fidélité chromatique de
`kryte` contre le 4:4:4 et coût de décodage.

## Cycle 14 — Weber renforcé et réparation chromatique diffuse (AUR2 v15)

Deux objectifs ont été traités conjointement : les +6,032 % de BD-Rate Y
restants face au JPEG 4:2:0 et la blessure chromatique concentrée sur `kryte`.
Les premiers témoins ont éliminé trois faux leviers. Couper l'échange spectral
chroma ne rendait que ~3 points de Cb à `kryte` et perdait l'efficacité RGB/Y ;
augmenter uniformément le raffinement luma payait trop de bits ; enfin une
supposée erreur de moyenne DC par taille de bloc a été rejetée après remontée
jusqu'à la LOT : le DC variable est déjà normalisé par `256 / taille²`, donc la
division fixe par 256 est bien la moyenne lumineuse exacte.

Le levier Y propre reste le contre-don de Weber. Le gain post-décodage de sa
force passe de 1 à **2,5** : sur les sondes `kryte` + `chilling`, la progression
reste monotone jusqu'à cette borne, sans déplacement chromatique matériel. Le
gain 2 seul avait déjà déplacé la frontière corpus de Y +6,032 % à +4,735 % ;
2,5 finance ensuite le coût de la réparation ciblée sans reperdre ce progrès.

La réparation chromatique est un garde zéro-bit reconstruit des deux grilles
DC chroma. Avec `G` l'échange spectral, `A` le RMS chroma en unités échantillon,
`H = ||moyenne(C)|| / RMS(C)` la cohérence de teinte globale et `s(a,b,x)` un
smoothstep, son score vaut :

`R = s(0.58, 0.70, G) * s(14, 20, A) * (1 - s(30, 42, A)) * (1 - s(0.15, 0.35, H))`

Cette conjonction détecte une chroma modérée, lisse face à L et sans dominante
stable. À Q60, elle donne **R=0,9414 pour kryte et exactement zéro pour les onze
autres originals**. Le pas C1 devient `phi^(1 - 0,85*R)` ; C2 reste strictement
inchangé. La force 0,85 est le point minimax mesuré : sur `kryte` contre le
4:4:4, Cb/Cr valent +7,90/+8,96 % dans le sweep de calibration. Un contre-don
C2 a été essayé puis rejeté : il rembourse la taille mais déplace la faute vers
Cr (à transfert complet : Cb +1,52 %, Cr +36,45 %), sans réparation perceptuelle.

La validation finale reprend exactement le protocole natif à **432 points** :
48 points AUREA nouveaux et les 384 JPEG du cycle 13, réutilisés avec le SHA-256
`ce89335915c235935bbb2ba2338fa629f6c8ba9acf83ff727d89f24f7b3925c6`.
Une première agrégation à huit qualités JPEG a été invalidée avant conclusion ;
le rapport retenu restaure bien les seize qualités 5..100. Les livrables finaux,
réétiquetés v15 avec le SHA du binaire promu, sont dans
`target/benchmark_v15/jpeg_frontier_v15`.

Face au JPEG 4:2:0, v15 gagne **-29,580 % RGB (12/12)**, contre -28,614 % en
v14. La frontière Y descend à **+4,721 %** (4 victoires sur 12), IC95
[+0,168 ; +9,036] : le retard n'est pas statistiquement effacé, mais 1,311 point
supplémentaire est repris. Le Delta-E moyen gagne -41,635 % et le p95 -36,079 %,
12/12. Le SSIM-Y reste interdit de revendication : cubique -1,973 %, contrôle
linéaire +8,798 %.

Face au JPEG 4:4:4, v15 gagne **-24,063 % RGB**, **-18,396 % Y**, -19,951 % Cb,
-28,857 % Cr, -24,590 % Delta-E moyen et -16,995 % p95. Par rapport à v14,
RGB, Y, Cb et les deux Delta-E progressent ; Cr rend seulement 0,323 point.
Sur `kryte`, la réparation est nette : Cb/Cr passent de +27,297/+5,711 % à
**+8,161/+9,307 %**, le Delta-E moyen de +10,847 % à **+5,129 %**, et son p95
de +0,405 % à **-12,944 %**. La victoire RGB reste acquise (-7,118 % contre
4:4:4, -1,302 % contre 4:2:0).

**Verdict 2026-08-14 : promu en AUR2 v15.** Les défauts deviennent Weber 2,5
et réparation diffuse 0,85 ; le contre-don C2 rejeté est supprimé. Les 144 tests
passent. Le binaire final archivé est
`target/benchmark_v15/aurea_v15_weber25_diffusion085.exe`, SHA-256
`f765a36ea97221d7bb4ec47e2a2dbada80c6436d55ef7ea2240b0b6233e8f95d`.
Le binaire exact du benchmark portait le même algorithme sous variables
explicites ; sur `agree` et `kryte` à Q60, ses flux v14 et le défaut v15 ont la
même taille et produisent des PNG identiques par SHA-256 — seul l'octet de
version change.

## Cycle 15 — Parenté spectrale et codons densifiés (AUR2 v16)

La relance « vaguelettes » part d'un constat structurel : `wavelet.rs` ne
contient plus de transformée en ondelettes, seulement l'ordre Morton et la zone
morte. La multirésolution réelle d'Aurea est déjà portée par la LOT adaptative
8/16/32 et les paquets spiraux progressifs. Ajouter une seconde transformée
aurait doublonné ce socle ; la piste retenue fait donc émerger une hiérarchie
de type ondelette **dans le spectre LOT quantifié lui-même**.

Une télémétrie neutre `AUREA_CODON_STATS=1` a mesuré les symboles exacts remis
à rANS à Q60 sur les 12 `originals`. Pour chaque fréquence `(u,v)`, le parent
dyadique est `(floor(u/2), floor(v/2))`. Les scans repliés 8/16/32 placent
toujours ce parent avant l'enfant : zéro violation causale. Après exclusion de
la queue implicite déjà supprimée par EOB, l'information mutuelle support
parent/enfant atteint **2,423 Mbits** : 1,758 Mbits L, 0,203 Mbit C1 et
0,461 Mbit C2. Les probabilités conditionnelles sont matériellement séparées ;
sur `agree`, par exemple, l'enfant L est actif avec probabilité 0,1323 si son
parent est nul, contre 0,4348 si le parent est actif.

L'intrication inter-canaux existe aussi, mais son plafond ne suffit pas à
justifier n'importe quelle grammaire. Sur le préfixe AC commun aux trois
canaux, les masques de support contiennent 1,365 Mbit de dépendance et les
classes de magnitude `{0, 1, 2..3, 4+}` 2,914 Mbits ; les classes EOB en
ajoutent 0,243 Mbit. Un témoin lossless naïf a pourtant réuni chaque triplet
L/C1/C2 en masque 3 bits puis séparé les valeurs non nulles : il **grossit
l'AC de 13,400 %**. La réunion jusqu'à l'EOB maximal réintroduit trop de
positions déjà absentes des canaux courts et paie l'apprentissage du nouvel
alphabet. C'est le résultat négatif central de ce cycle : une dépendance
mesurable n'est un codon utile que si son contexte est causal et gratuit.

Le témoin retenu ne change donc ni l'ordre, ni les valeurs, ni la grammaire des
magnitudes. Il conserve une seule mémoire rANS pour les runs, l'énergie et
`+/-1`, mais dédouble uniquement `P(zéro)` selon le support du parent. Le bit de
parent n'est jamais écrit : le décodeur le lit dans un coefficient du squelette
précédent ou déjà décodé dans la couche courante. Deux modèles complets séparés
avaient perdu +0,118 % d'AC sur `agree` en cassant la mémoire des runs ; le
partage d'état gagne -0,789 % sur la même image.

Le régime dépend de la densité. Face au binaire v15 figé, les sommes corpus
same-quality valent :

| Qualité | v15 | v16 parent | Delta | Images gagnées |
|---:|---:|---:|---:|---:|
| 20 | 4 670 326 | 4 636 052 | **-0,734 %** | 12/12 |
| 40 | 7 517 246 | 7 476 274 | **-0,545 %** | 12/12 |
| 60 | 11 774 172 | 11 733 719 | **-0,344 %** | 12/12 |
| 70 | 16 473 115 | 16 439 123 | **-0,206 %** | 12/12 |
| 75 | 20 952 377 | 20 927 168 | -0,120 % | 11/12 |
| 80 | 27 084 979 | 27 084 979 | **0,000 %** | 12 égalités |

La règle finale active donc le parent jusqu'à **Q70 inclus** et revient au v12
au-delà, déterministement depuis la qualité déjà présente dans l'en-tête. Le
premier parent forcé à Q80 ne gagnait plus que -0,033 % global et reculait sur
quatre images ; cette micro-instabilité est supprimée. À Q60, les douze PNG
v16 décodés ont exactement le même SHA-256 que leurs reconstructions v15 : les
40 453 octets économisés ne déplacent aucun pixel.

Le protocole JPEG final conserve 432 points : 48 Aurea recalculés et les 384
JPEG validés du cycle 14. Face au JPEG 4:2:0, v16 atteint **-29,896 % RGB**,
**+4,367 % Y**, -61,888 % Cb, -66,611 % Cr, -41,907 % Delta-E moyen et
-36,379 % p95. Les déplacements v15→v16 sont respectivement -0,316, -0,354,
-0,194, -0,174, -0,272 et -0,300 point. Face au 4:4:4 : **-24,320 % RGB**,
**-18,669 % Y**, -20,199 % Cb, -29,062 % Cr, -24,870 % Delta-E moyen et
-17,294 % p95 ; chaque métrique progresse encore de 0,206 à 0,299 point.

**Verdict 2026-08-14 : promu en AUR2 v16.** Les 146 tests passent. Le binaire
final est `target/benchmark_v16/aurea_v16_wavelet_parent_q70.exe`, SHA-256
`42bfc806acbd28122bd31e6b322e8d0b7d965dea07295dad7c8d76711587ef92`.
Le rapport et ses courbes sont dans
`target/benchmark_v16/jpeg_frontier_v16_final`; l'analyse brute, le CSV, le
JSON et les témoins rejetés sont dans `target/wavelet_codon_lab`.

## Cycle 16 — Codon chromatique causal (AUR2 v17)

L'intuition du triplet est prise au sens strict : RGB est un codon de couleur,
et la GCT d'Aurea le réexprime sans le détruire sous la forme `(Lφ, C1, C2)`.
La compression jointe correcte n'est toutefois pas un nouvel alphabet global :

`P(Lφ, C1, C2) = P(Lφ) × P(C1 | Lφ) × P(C2 | Lφ, C1)`.

Pour chaque bloc et chaque position fréquentielle, les canaux restent écrits
et décodés dans cet ordre. La classe du codon est donc reconstruite depuis des
coefficients déjà connus et coûte **zéro bit**. Les trois EOB restent séparés,
les couches progressives et leurs arcs restent inchangés, et la mémoire rANS
séquentielle des runs et de l'énergie reste partagée. Seules `P(zéro)` et
`P(|x|=1)` sont séparées par la classe chromatique ; cela évite précisément
l'échec du masque conjoint naïf du cycle 15 (+13,400 % d'AC).

Quatre témoins rANS exacts ont été mesurés sur les 12 `originals` : support ou
classe de magnitude `{0, 1, 2..3, 4+}`, avec ou sans séparation de `P(|x|=1)`.
Le signal possède deux régimes naturels. Jusqu'à Q40, le flux est assez
parcimonieux pour que le support soit le meilleur témoin : C1 connaît
`[Lφ != 0]`, C2 connaît le masque `[Lφ != 0] + 2[C1 != 0]`. Au-dessus de Q40,
la magnitude devient informative : C1 connaît `mu(Lφ)` et C2
`max(mu(Lφ), mu(C1))`. Le seuil dépend uniquement de la qualité déjà dans
l'en-tête ; aucun sélecteur n'est transmis. La séparation de `P(|x|=1)` gagne
dans les deux régimes.

| Qualité | Régime retenu | Delta AC pondéré | Delta fichier | Images gagnées |
|---:|:---|---:|---:|---:|
| 20 | support + `±1` | **-0,809 %** | **-0,598 %** | 12/12 |
| 40 | support + `±1` | **-1,091 %** | **-0,883 %** | 12/12 |
| 60 | magnitude + `±1` | **-1,379 %** | **-1,185 %** | 12/12 |
| 70 | magnitude + `±1` | **-1,513 %** | **-1,347 %** | 12/12 |
| 80 | magnitude + `±1` | **-1,528 %** | **-1,423 %** | 12/12 |

Le codon subsume aussi le parent dyadique : en v17, son support est fourni au
modèle à toutes les qualités, y compris Q80 où le parent isolé de v16 était
devenu neutre. L'association parent + couleur reste profitable sur chaque
image. À Q20 et Q60, les reconstructions v16/v17 de `agree` ont des SHA-256
PNG identiques ; le corpus officiel confirme des deltas strictement nuls pour
PSNR RGB/Y/Cb/Cr, SSIM-Y et Delta-E à chaque qualité. Le gain est donc purement
entropique et ne modifie aucune décision psycho-visuelle.

La comparaison directe v16→v17 donne **-0,966 % de BD-rate RGB**, 12/12 images,
IC95 [-1,239 ; -0,722]. Face au JPEG 4:2:0, v17 atteint **-30,484 % RGB**,
**+3,331 % Y**, -62,202 % Cb, -66,857 % Cr, -42,422 % Delta-E moyen et
-36,929 % p95. Le retard luminance recule donc de 1,036 point sans déplacer un
pixel. Face au JPEG 4:4:4 : **-25,060 % RGB**, **-19,467 % Y**, -21,046 % Cb,
-29,809 % Cr, -25,622 % Delta-E moyen et -18,115 % p95.

**Verdict 2026-08-14 : promu en AUR2 v17.** Les 147 tests passent. Le binaire
final est `target/benchmark_v17/aurea_v17_vector_codon.exe`, SHA-256
`80c405deb4d46eef24524d6796e4c46c053e7bfb26b3c395ddb225e3fe6482d2`.
Le rapport JPEG et ses courbes sont dans
`target/benchmark_v17/jpeg_frontier_v17`; la comparaison v16/v17 dans
`target/benchmark_v17/v16_vs_v17`; les traces, le CSV, le JSON et la courbe
des quatre codons dans `target/wavelet_codon_lab/vector_codon_multiq`.

## Cycle 17 — Pivot luminance spectral (AUR2 v18)

Le `+3,331 % Y` de la v17 face au JPEG 4:2:0 a d'abord été traité comme une
possible erreur de représentation. La métrique officielle est bien le Y
BT.601 pleine plage. La première composante de la GCT est le `Lφ` doré, mais
les deux espaces sont reliés exactement par les chromas déjà présents :

`Y601 = Lφ - 0,110214049 C1 - 0,063785951 C2`.

Le résidu numérique de cette identité est inférieur à `2e-13`. Une GCT
interpolée jusqu'à un premier axe exactement BT.601 a ensuite été testée à
corps de bitstream constant. Elle dégrade la frontière : sur quatre sentinelles,
le mélange total donne `+4,39 % Y` contre le JPEG 4:2:0. Le défaut apparent
n'est donc ni une confusion de métrique, ni une mauvaise inversion de Y :
c'est une allocation insuffisante vers la luminance.

Deux leviers ont été comparés. Une pondération RDO du treillis par canal
n'apporte que quelques dixièmes de point sur une sentinelle et a été rejetée.
La force du contre-don luminance de l'échange spectral, elle, produit une
frontière régulière :

| Force L | RGB vs JPEG 4:2:0 | Y vs JPEG 4:2:0 | Cb vs JPEG 4:4:4 | Cr vs JPEG 4:4:4 |
|---:|---:|---:|---:|---:|
| 1,00 (v17) | -30,48 % | +3,33 % | -21,05 % | -29,81 % |
| 1,50 | -31,98 % | +0,68 % | -15,97 % | -25,12 % |
| **1,75** | **-32,52 %** | **-0,41 %** | **-13,21 %** | **-22,61 %** |
| 2,00 | -32,96 % | -1,39 % | -10,35 % | -19,96 % |

La force 1,75 est retenue plutôt que 2,00 : face à v17, elle gagne `-4,471 %`
de BD-rate Y et `-4,495 %` de SSIM-Y sur 12/12 images, avec un RGB neutre
(`-0,099 %`, IC95 traversant zéro). Le prix mesuré est `+9,629 % Cb`,
`+10,067 % Cr`, `+5,253 %` Delta-E moyen et `+5,740 %` p95. Malgré ce transfert,
v18 reste devant JPEG 4:2:0 de `-39,511 %` en Delta-E moyen et devant JPEG
4:4:4 de `-20,576 %`.

**Verdict 2026-08-15 : promu en AUR2 v18.** La force luma `1,75`, la force
chroma `1,0` et la base GCT dorée sont désormais des constantes du format,
indépendantes de l'environnement. Le décodeur rejette toutes les versions
antérieures, conformément à la rupture de compatibilité assumée. Les 149 tests
passent. Le binaire final est
`target/benchmark_v18/aurea_v18_luma_pivot.exe`, SHA-256
`b83672344e55f45735065d170dbed920aef575e0440b326ac280baf1749f2a78`.
Le benchmark officiel est dans `target/benchmark_v18/jpeg_frontier_v18`, et
la comparaison directe v17/v18 dans `target/benchmark_v18/v17_vs_v18`.

## Pistes suivantes

- **Frontière Y, sixième acte** : transformer le faible `-0,410 %` moyen face
  au 4:2:0 en victoire robuste par image, sans pousser davantage le scalaire
  global ; le prochain levier doit être local/fréquentiel et conservatif en
  Delta-E.
- **Codon chromatique signé** : mesurer si la cohérence de signe Lφ/C1/C2
  apporte un contexte gratuit aux magnitudes `>= 2`, sans fragmenter la
  mémoire adaptative ni réintroduire un alphabet conjoint.
- **Décodage ≤ 5×** : rANS parallèle par (couche × canal) et plans `f32`,
  sous verdict BD-Rate complet puisque la bit-exactitude serait rompue.
- **Généralisation du garde chromatique** : valider la porte diffuse sur un
  corpus externe avant d'élargir sa zone ; sur `originals`, elle est exactement
  ciblée, mais ce résultat ne suffit pas à prouver sa calibration universelle.

## Cycle 18 — Résidu de phase chromatique (AUR2 v19)

L'hypothèse « l'information se comporte comme une onde » a d'abord été testée
dans l'allocation psycho-visuelle. Depuis la grille DC déjà transmise, un
descripteur subsonique reconstruit le gradient lent, la courbure de Hessien et
une phase continue. Deux échanges log-neutres ont été essayés : une modulation
directionnelle large du spectre puis un transfert strict entre fondamentale
et second harmonique. Après correction de la convention `(x,y)` vers
`(colonne,ligne)` de la DCT, le premier modèle reste perdant sur les six
sentinelles : à force 0,25, le RGB face au JPEG 4:2:0 passe de -33,31 % à
-32,87 % et Y de -0,70 % à +0,39 %. Le second ne récupère pas le débit. Ces
champs restent neutres par défaut (`AUREA_SUBSONIC_WAVE_STRENGTH=0`).

La limite est informationnelle : le DC inter-blocs décrit l'enveloppe lente,
mais ne connaît pas la phase intra-bloc d'une projection LOT. La phase
quantifiée réellement disponible est déjà dans le **signe** de chaque
coefficient : `+` correspond à 0 et `-` à pi. La grammaire v18 payait encore
ce signe comme un bit équiprobable indépendant.

La télémétrie `AUREA_CODON_STATS=1` a donc mesuré quatre relations causales
sur les symboles quantifiés exacts : parent dyadique, dernier non-zéro du scan,
Lφ vers chroma et C1 vers C2. Parent et voisin restent pratiquement aléatoires.
La relation C1/C2 est au contraire massive : lorsque les deux coefficients
sont non nuls au même bloc et à la même fréquence, leur phase ne coïncide que
dans **5,696 % à 28,227 %** des cas selon l'image à Q40. Le plafond conditionnel
vaut 0,142 à 0,685 bit économisable par paire. Cette anti-phase est cohérente
avec les axes opposants de la GCT, `C1 = B-Lφ` et `C2 = R-Lφ`, mais le codec
n'en suppose pas la force : il l'apprend dans chaque paquet.

La factorisation v19 conserve tous les codons v17 :

`P(signe C2 | C1) = P(accord/opposition | signe C1)`.

C1 est entièrement décodé avant C2 ; le prédicteur coûte donc zéro bit. Si C1
est nul, le signe C2 reste un bit brut. Sinon rANS code l'accord avec un a
priori `P(accord)=1/8`, puis l'adapte à vitesse `1/64` dans le paquet. Aucun
support, EOB, magnitude, coefficient quantifié, arc progressif ou pas de
quantification ne change.

Le témoin rANS exact à Q40 économise **41 421 octets**, soit **-0,6105 %** de
l'AC agrégé, et gagne sur 12/12 images. Le protocole complet (12 `originals`,
Q20/40/60/80, JPEG 5..100) confirme une baisse moyenne du fichier qui croît
avec la densité :

| Qualité | Delta bpp v18→v19 | Delta de toutes les métriques |
|---:|---:|---:|
| 20 | **-0,364 %** | 0 |
| 40 | **-0,470 %** | 0 |
| 60 | **-0,566 %** | 0 |
| 80 | **-0,681 %** | 0 |

La comparaison directe donne **-0,548 % de BD-rate RGB**, 12/12, IC95
[-0,668 ; -0,427]. Les contrôles linéaire et cubique concordent, et les
BD-rates Y/Cb/Cr/SSIM/Delta-E gagnent tous environ un demi-point puisque les
courbes de distorsion sont strictement inchangées. Sur `agree/q40`, v18 passe
de 421 648 octets à 419 735 ; les PNG décodés partagent le SHA-256
`8794fb05433d2306e5e0972396622eaf094b4eadb237ff7a9d52347d727c4d44`.

Face au JPEG optimisé 4:2:0, v19 atteint **-32,842 % RGB**, **-0,961 % Y**,
-58,565 % Cb, -63,338 % Cr, -39,822 % Delta-E moyen et -35,118 % p95. Face au
4:4:4 : **-25,730 % RGB**, **-23,861 % Y**, -13,711 % Cb, -23,063 % Cr,
-21,025 % Delta-E moyen et -13,371 % p95. RGB gagne 12/12 dans les deux cas ;
le faible avantage Y contre 4:2:0 reste à 6/12 et son IC95 traverse zéro.

**Verdict 2026-08-15 : promu en AUR2 v19.** Le décodeur accepte v19 seul,
sans chemin de rétrocompatibilité. Les 154 tests passent. Le binaire final est
`target/benchmark_v19/aurea_v19_phase.exe`, SHA-256
`fc1c06a1543c662ac5e57b2c04ffd9e05a5f1365206ad6b5baa086f7ece9a20b`.
Le rapport JPEG et les courbes analytiques sont dans
`target/benchmark_v19/jpeg_frontier_v19`; la comparaison v18/v19 est dans
`target/benchmark_v19/direct_v18_v19`.

## Cycle 19 — RDO des codons et continuation de phase Fourier (AUR2 v20)

Ce cycle part d'un fil directeur devenu net : les gains robustes apparaissent
quand le codec exploite une dépendance **déjà causale et déjà observable**,
plutôt que lorsqu'il invente un champ perceptuel supplémentaire. Deux
corrections structurelles et une loi ondulatoire ont été soumises au même
protocole complet.

La première correction concerne CfL avec les blocs adaptatifs 32x32. La
reconstruction Lphi de l'encodeur réservait encore 255 coefficients AC, soit
la taille d'un bloc 16x16. Les blocs 32x32 en possèdent 1023 : la tranche était
tronquée avant le calcul du résidu chromatique. Chaque bloc conserve désormais
sa taille native. Face à v19, cette correction seule donne -0,008 % RGB,
-0,046 % Cr et -0,021 % Delta-E moyen ; Y recule de +0,012 %. Elle est gardée
car elle rétablit l'identité mathématique encodeur/décodeur, notamment sur
`kryte`, sans modifier la syntaxe.

Ensuite, le treillis scalaire a été prolongé par une descente coordonnée du
codon couleur. Pour une erreur transformée `(eL,eC1,eC2)`, l'encodeur évalue
l'erreur RGB exacte après GCT inverse, augmentée de la luminance BT.601 :

`D = eR^2 + eG^2 + eB^2 + 12 eY^2`.

C1 est raffiné après Lphi, puis C2 contre le couple `(Lphi,C1)` exactement
reconstruit. Seuls les niveaux de magnitude adjacents sont testés. Le support,
le signe et l'EOB sont invariants ; un garde interdit tout candidat dont le
payload Exp-Golomb serait plus long. La variante sans ce garde gagnait en
distorsion mais achetait environ 0,3 % d'octets : elle a été rejetée. Le point
promu est `strength=1`, `Y weight=4`, `rate weight=1`.

Face à la base CfL corrigée, le RDO conjoint gagne :

| Métrique | BD-rate cubique | Victoires | IC95 bootstrap |
|---|---:|---:|---:|
| RGB PSNR | **-0,041 %** | 12/12 | [-0,055 ; -0,029] |
| Y PSNR | **-0,020 %** | 12/12 | [-0,027 ; -0,014] |
| Cb PSNR | **-0,156 %** | 12/12 | [-0,189 ; -0,127] |
| Cr PSNR | **-0,126 %** | 12/12 | [-0,167 ; -0,090] |
| Delta-E moyen | **-0,151 %** | 12/12 | [-0,193 ; -0,117] |
| Delta-E p95 | **-0,158 %** | 10/12 | [-0,225 ; -0,081] |

La piste Fourier a ensuite été analysée sur les symboles quantifiés exacts,
par sous-bande et selon l'ordre causal des paquets progressifs. La conservation
de phase gauche/haut, l'extrapolation linéaire, un AR(2) fixe
`x[n]=2 cos(omega)x[n-1]-x[n-2]` et un AR(2) adaptatif ont été comparés. Les
modèles prédictifs généraux sont fragiles ; Lphi ne dépasse qu'environ 52 % de
phase correcte. La loi robuste est plus simple : la continuité verticale de
C1 entre deux blocs de même taille, à la même coordonnée DCT.

Pour la fréquence verticale `v`, une translation d'un bloc prolonge la phase
de la base cosinus par `pi*v`. Le prédicteur binaire devient donc :

`phase_hat(C1[b,v,u]) = phase(C1[top(b),v,u]) XOR (v mod 2)`.

Sur `kryte` Q40, cette parité atteint 58,27 % d'accord. Un bloc supérieur nul
désactive le prédicteur ; une référence dans le même paquet pointe vers un
symbole déjà décodé, et une référence d'une vague antérieure est connue. Aucun
index n'est transmis. Le prior `P(accord)=9/16` s'adapte lentement à vitesse
`1/256`. C2 conserve le prédicteur C1->C2 de v19, plus fort que sa propre
continuité spatiale.

Le témoin byte-exact C1 seul gagne à Q40 sur 12/12 images et économise 852
octets AC agrégés ; les variantes qui incluent Lphi perdent encore sur
`phoenix`. Le protocole complet confirme un gain Fourier modeste mais strict :

| Qualité | Delta bpp RDO→Fourier | Delta de toutes les métriques |
|---:|---:|---:|
| 20 | **-0,0123 %** | 0 |
| 40 | **-0,0102 %** | 0 |
| 60 | **-0,0080 %** | 0 |
| 80 | **-0,0040 %** | 0 |

La comparaison directe donne **-0,00817 % BD-rate RGB**, 12/12, IC95
[-0,01116 ; -0,00568]. Y, Cb, Cr, SSIM-Y et les deux Delta-E gagnent eux
aussi 12/12 parce que les pixels décodés sont identiques. Sur `kryte/q40`, le
fichier passe de 664 130 à 664 007 octets ; les deux PNG ont le SHA-256
`46b22984f301214fd5e4f14786c6d9df6e7e2e5f597e7c2f7fe02b484294e4df`.

Le cycle v19→v20 complet atteint **-0,057 % RGB**, **-0,016 % Y**, -0,175 %
Cb, -0,180 % Cr, -0,180 % Delta-E moyen et -0,165 % p95. Face au JPEG optimisé
4:2:0, v20 atteint **-32,877 % RGB**, **-0,975 % Y**, -58,654 % Cb,
-63,406 % Cr, -39,912 % Delta-E moyen et -35,226 % p95. Face au 4:4:4 :
**-25,770 % RGB**, **-23,875 % Y**, -13,845 % Cb, -23,205 % Cr,
-21,128 % Delta-E moyen et -13,502 % p95.

**Verdict 2026-08-15 : promu en AUR2 v20.** Le décodeur accepte v20 seul,
sans rétrocompatibilité. Les 155 tests du coeur passent. Le binaire final est
`target/benchmark_v20/aurea_v20_fourier.exe`, SHA-256
`507ad93f40ec12a5b17e50b2d55d92f165ba091da04863e3bbe42de263d428d1`.
Le rapport JPEG et les courbes analytiques recalculées sont dans
`target/benchmark_v20/jpeg_frontier_v20_final`; la comparaison Fourier isolée
est dans `target/benchmark_v20/direct_joint_v20`, et la comparaison complète
v19/v20 dans `target/benchmark_v20/direct_v19_v20`.

## Cycle 20 — Enveloppe amplitude/phase par voisinage spectral (AUR2 v21)

Depuis la v16, le seul témoin qu'un coefficient possédait à l'intérieur de sa
propre transformée était le **support binaire** du parent dyadique
`(v/2, u/2)` : un bit, un voisin. Le cycle 19 avait conclu que ce parent était
« pratiquement aléatoire » — mais il l'avait mesuré pour la *phase*, et pour un
voisin isolé. Le spectre d'une LOT n'est pas une collection de coefficients
indépendants : un contour ou une texture localement bande-étroite y dessine une
crête dont l'amplitude décroît continûment et dont les signes obéissent à une
loi exacte.

### La loi de phase séparable

Pour un contenu séparable — l'impulsion, l'arête, le produit de deux
sinusoïdes — la DCT-II vaut `C[v,u] ~ A(v,u) cos(v alpha) cos(u beta)` avec
`alpha = pi (y0 + 1/2) / N` et `beta = pi (x0 + 1/2) / N`. En posant
`s = signe`, `g(v) = signe(cos(v alpha))` et `f(u) = signe(cos(u beta))`, le
produit des quatre coins d'un carré unité du plan fréquentiel se simplifie
entièrement :

`s[v,u] . s[v,u-1] . s[v-1,u] . s[v-1,u-1] = g(v)^2 g(v-1)^2 f(u)^2 f(u-1)^2 = 1`

d'où un prédicteur exact, gratuit et causal dans le zigzag :

`s[v,u] = s[v,u-1] XOR s[v-1,u] XOR s[v-1,u-1]`

Sur les bords du spectre (`v = 0` ou `u = 0`) la relation dégénère en une suite
unidimensionnelle `signe(cos(u beta))`, un mot équilibré dont les *transitions*
sont quasi-périodiques. Le prédicteur y devient `s[u] = s[u-2]`, et la classe
retient en plus la transition précédente : la mémoire adaptative sépare alors
d'elle-même les blocs à densité de flip faible et forte, sans qu'aucun
estimateur de densité soit transmis.

Deux propriétés rendent ce voisinage plus solide que tous les prédicteurs des
cycles 18 et 19. Il est **toujours causal** : `left`, `top`, `diag`, `left2`,
`top2` et le parent appartiennent tous à une anti-diagonale antérieure du
zigzag, quel que soit le découpage progressif — aucun rang à vérifier, aucune
référence à désactiver, contrairement à la continuation spatiale de la v20. Et
il est **intra-bloc** : il ne dépend ni de la géométrie du quadtree, ni de
l'ordre spiralé des paquets.

### L'enveloppe d'amplitude

Le même voisinage gradue ce que le bit de parent ne faisait que trancher. La
somme `|left| + |top| + |diag| + |parent|`, classée en `{0}`, `{1}`, `{2..3}`,
`{4+}`, remplace l'axe binaire dans `P(zéro)` et sépare aussi `P(|x| = 1)`.
Le nombre de contextes double seulement, et la partition reste un sur-ensemble
strict de celle de la v16.

### Ce que la télémétrie a mesuré

`AUREA_CODON_STATS=1` mesure les symboles quantifiés exacts. Sur `kodim01` à
Q40, l'information mutuelle avec le support du coefficient courant passe de
`0,0168` bit/symbole pour le parent v16 à **`0,1160` bit/symbole** pour
l'enveloppe graduée sur Lphi — presque sept fois plus. La loi séparable couvre
`96,2 %` des signes Lphi et les prédit à `59,01 %` sur 63 616 échantillons ;
sur les chromas, plus lisses et davantage dominées par les contours, l'accord
monte à `74,07 %` (C2) et `79,05 %` (C1). Les classes faibles — paire tronquée,
parent seul, voisin seul — restent entre `49,3 %` et `50,8 %` : du hasard.

### Le témoin octet-exact

Le harnais réencode l'AC entier de chaque couche progressive sous chaque
configuration candidate. Sa première variante est neutre et **doit** rendre
exactement les octets de production : sur les 48 points du protocole, la dérive
neutre vaut `0` octet. C'est l'autotest du harnais, pas un résultat.

Sur les 12 `originals` à Q20/Q40/Q60/Q80, face au binaire v20 :

| Mécanisme | Delta AC agrégé | Images gagnées |
|---|---:|---:|
| Loi séparable seule | -0,205 % | 48/48 |
| Phase cohérente (séparable + continuations 1D) | -0,300 % | 48/48 |
| Enveloppe d'amplitude seule | -0,755 % | 48/48 |
| Enveloppe + séparation de `P(|x|=1)` | -1,193 % | 48/48 |
| **Enveloppe complète + phase cohérente** | **-1,493 %** | **48/48** |

Les deux moitiés sont additives : elles ne partagent aucun budget de bits, l'une
gouvernant le support et la magnitude, l'autre le seul signe.

Le régime dépend de la densité, mais ne s'effondre pas comme le parent isolé de
la v16 le faisait à Q80 :

| Qualité | Delta AC | Images gagnées |
|---:|---:|---:|
| 20 | **-2,751 %** | 12/12 |
| 40 | **-2,292 %** | 12/12 |
| 60 | **-1,815 %** | 12/12 |
| 80 | **-0,977 %** | 12/12 |

Par image, le gain va de `-1,135 %` (`kryte`) à `-2,095 %` (`phoenix`), 4/4
qualités sur chacune des douze.

### Trois résultats négatifs

**Le couplage énergie/phase est rejeté.** L'hypothèse était séduisante : la loi
séparable est exacte sur le signal continu, donc ce qui la dégrade serait le
bruit de quantification, dont le poids relatif décroît quand le voisinage est
fort — l'enveloppe serait alors l'estimateur naturel de la fiabilité de la
phase. Un compteur par classe *et* par niveau d'énergie donne `-1,472 %` contre
`-1,493 %` sans couplage. La loi échoue en réalité sur le contenu *non
séparable*, pas sur le bruit : l'amplitude n'en est pas le témoin.

**La priorité spectrale est rejetée.** Laisser la loi séparable primer sur les
prédicteurs v19/v20 quand les deux sont disponibles fait chuter le gain de phase
de `-0,36 %` à `-0,12 %` sur le corpus Q40, et perd sur 2 images. Le résidu
C1 -> C2 de la v19 reste le témoin le plus fort là où il existe ; la loi
spectrale doit le compléter, pas le remplacer.

**Les classes de phase faibles sont exclues.** Ajouter la paire tronquée, le
parent seul et le voisin seul au masque coûte `-1,437 %` contre `-1,493 %` :
mesurées au hasard, elles ne paient que le bruit d'adaptation de leurs propres
compteurs. Le masque promu ne retient que la loi séparable et les deux
continuations unidimensionnelles.

Deux enveloppes d'amplitude alternatives ont aussi perdu : le croisement
`parent x support local` (`-0,867 %`) et le simple décompte de voisins actifs
(`-0,764 %`). La somme des magnitudes gagne parce qu'elle est la seule des trois
à distinguer une crête forte d'un voisinage juste peuplé.

### Symétrie

Encodeur et décodeur appellent littéralement le même évaluateur
(`spectral_envelope::probe` puis `resolve_phase_class`), pilotés par la même
description de segments : la symétrie est une propriété du code, pas une
convention à maintenir. Le décodeur reconstruit chaque voisin soit dans le
squelette qu'il a hérité, soit dans la séquence plate qu'il est en train de
produire ; aucun index, aucun bit n'est transmis.

### Comparaison directe v20 -> v21

Le mécanisme ne touche ni le pas de quantification, ni le parcours, ni une
décision psycho-visuelle. La vérification est donc binaire : sur les 48 points
du protocole, les PNG décodés partagent leur SHA-256 avec la reconstruction v20.
**Zéro pixel n'a bougé**, et toutes les courbes de distorsion sont identiques.

| Qualité | Fichiers v20 | Fichiers v21 | Delta | Images gagnées |
|---:|---:|---:|---:|---:|
| 20 | 5 131 406 | 5 023 601 | **-2,101 %** | 12/12 |
| 40 | 8 170 976 | 8 016 412 | **-1,891 %** | 12/12 |
| 60 | 12 693 781 | 12 493 521 | **-1,578 %** | 12/12 |
| 80 | 28 705 865 | 28 443 335 | **-0,915 %** | 12/12 |
| **total** | **54 702 028** | **53 976 869** | **-1,326 %** | **48/48** |

Les `725 159` octets économisés sur les fichiers complets sont exactement les
`725 159` octets que le témoin annonçait sur l'AC seul : rien d'autre dans le
conteneur n'a changé, ce qui referme la boucle entre le harnais et le codec.

### Dette de calcul

Le cycle 10 a fait de la vitesse un verdict à part entière ; l'enveloppe doit
donc payer sa facture. Une première version évaluait la loi de phase pour tous
les symboles, y compris les zéros — la majorité — et coûtait `+12,4 %` à
l'encodage et `+16,4 %` au décodage. La résolution est désormais paresseuse :
`probe` ne lit que les quatre voisins de l'enveloppe, et les deux voisins de
second ordre ne sont touchés que si le coefficient se révèle non nul *et* que le
prédicteur historique se tait. À octets rigoureusement identiques, la dette
tombe à `+3,8 %` / `+5,1 %` sur `kryte` q60, `+7,2 %` / `+10,8 %` sur `phoenix`
q40 et `+12,8 %` / `+13,7 %` sur `kodim01` q60 — les petites images payant
proportionnellement le plus. Face aux `-1,33 %` de débit, le rapport est sans
commune mesure avec le `+31,3 %` de décodage pour `-0,0019 %` qui avait fait
rejeter la morphogenèse emboîtée du cycle 3.

### Verdict débit-distorsion

Le protocole complet — 12 `originals`, Q20/40/60/80, JPEG 5..100 — a recalculé
les 48 points Aurea contre les 384 points JPEG déjà validés du cycle 19. La
comparaison directe v20 -> v21 :

| Métrique | BD-rate cubique | Victoires | IC95 bootstrap | Contrôle linéaire |
|---|---:|---:|---:|---:|
| RGB PSNR | **-1,626 %** | 12/12 | [-1,818 ; -1,449] | -1,625 % |
| Y PSNR | **-1,621 %** | 12/12 | [-1,812 ; -1,447] | -1,620 % |
| Cb PSNR | -1,554 % | 12/12 | [-1,712 ; -1,402] | -1,574 % |
| Cr PSNR | -1,527 % | 12/12 | [-1,688 ; -1,381] | -1,561 % |
| SSIM-Y | **-1,802 %** | 12/12 | [-2,032 ; -1,592] | -1,778 % |
| Delta-E moyen | -1,644 % | 12/12 | [-1,833 ; -1,472] | -1,642 % |
| Delta-E p95 | -1,636 % | 12/12 | [-1,819 ; -1,469] | -1,637 % |

Toutes les métriques gagnent la même chose parce qu'elles mesurent toutes la
même image : à qualité fixe, la chaîne de mesure indépendante confirme
`+0,000` dB sur RGB, Y, Cb, Cr et `+0,000` sur Delta-E aux quatre qualités,
pour `-2,131 %`, `-1,955 %`, `-1,645 %` et `-0,954 %` de bpp. Le gain est
intégralement de débit.

Face au JPEG optimisé, la frontière avance sur chaque métrique :

| Référence | Métrique | v20 | v21 | Gain |
|---|---|---:|---:|---:|
| 4:2:0 | RGB PSNR | -32,877 % | **-34,106 %** | 1,229 pt |
| 4:2:0 | Y PSNR | -0,975 % (6/12) | **-2,579 %** (7/12) | 1,604 pt |
| 4:2:0 | SSIM-Y | -8,382 % | **-10,039 %** | 1,657 pt |
| 4:2:0 | Delta-E moyen | -39,912 % | **-41,008 %** | 1,096 pt |
| 4:4:4 | RGB PSNR | -25,770 % | **-26,971 %** | 1,201 pt |
| 4:4:4 | Y PSNR | -23,875 % | **-25,109 %** | 1,234 pt |
| 4:4:4 | SSIM-Y | -30,339 % | **-31,602 %** | 1,263 pt |

Le RGB gagne 12/12 contre les deux profils JPEG. L'avance Y face au 4:2:0
triple, mais reste 7/12 avec un IC95 `[-6,600 ; +1,129]` qui traverse zéro :
ce n'est toujours pas une victoire universelle en luminance, seulement une
avance moyenne mieux assise. La sixième piste de la frontière Y reste ouverte.

**Verdict 2026-08-15 : promu en AUR2 v21.** Le mécanisme est purement
entropique — aucun pas de quantification, aucun parcours, aucune décision
psycho-visuelle ne change — donc les pixels décodés sont identiques à ceux de la
v20. `AUREA_SPECTRAL_ENVELOPE=0` ramène exactement à la grammaire v20 et produit
un fichier v20 : c'est le commutateur A/B, pas un réglage. Les 163 tests du
coeur passent. Le binaire final est
`target/benchmark_v21/aurea_v21_spectral_envelope.exe`, SHA-256
`2c5cf5b42fc8c10bcccf171edae42f4aaf43f1e1e4af77a509584391a3c947903`.
Le rapport JPEG et ses courbes sont dans
`target/benchmark_v21/jpeg_frontier_v21`; la comparaison directe v20/v21 dans
`target/benchmark_v21/direct_v20_v21`; les balayages du témoin, la comparaison
octet à octet et les variantes rejetées dans `target/spectral_envelope_lab`.

## Pistes suivantes

- **Frontière Y, sixième acte** : `-2,579 %` face au 4:2:0 sur 7/12 images avec
  un IC95 encore ouvert. L'enveloppe a fait gagner 1,6 point sans toucher à
  l'allocation ; le levier suivant doit rester local/fréquentiel et conservatif
  en Delta-E.
- **Enveloppe inter-canaux** : le voisinage spectral est aujourd'hui strictement
  intra-canal. Mesurer si l'enveloppe d'amplitude de Lphi à la même fréquence
  affine celle des chromas, sachant que le codon v17 exploite déjà leur support
  et leur magnitude — le risque de doublon est réel et doit être mesuré, pas
  supposé.
- **Bucketing scale-aware** : la somme de magnitudes sature à `{4+}` quand la
  qualité monte, ce qui explique la décroissance du gain de `-2,10 %` à Q20 vers
  `-0,92 %` à Q80. Un seuillage indexé sur l'énergie courante du modèle pourrait
  rendre l'axe utile plus haut sans ajouter d'états.
- **Loi séparable d'ordre 2** : la loi exacte ne couvre que le cas où les trois
  voisins sont actifs. Sur Lphi elle n'atteint que `59 %` d'accord là où les
  chromas montent à `74-79 %`, ce qui situe précisément la limite : le contenu
  luma est bien moins séparable. Une classe conditionnée à la cohérence
  directionnelle du bloc — déjà calculée par le tenseur de structure du cycle 2 —
  séparerait peut-être les blocs où la loi doit tenir de ceux où elle ne peut pas.

## Cycle 21 — Les quatre pistes du cycle 20, mesurées

Les quatre pistes ouvertes par l'enveloppe spectrale ont été construites en
entier, chacune avec son état neutre bit-exact, et soumises au protocole. Aucune
n'est promue. Le profil actif reste la v21 telle quelle. Ce cycle est donc un
cycle d'équilibre : quatre mécanismes plausibles, quatre refus, et surtout
quatre raisons intelligibles qui resserrent la carte du territoire.

### Piste 1 — Échange spectral local (rejetée)

Le gamma de l'échange spectral est global depuis la v14 : un scalaire pour toute
l'image, tiré du rapport d'activité `RMS(gradC)/RMS(gradL)` sur les grilles DC
denses. À l'intérieur d'une image la réponse est pourtant rarement uniforme — un
ciel plat et un visage coloré partagent le même cadre et donc le même troc.

`hierarchy::spectral_exchange_local_levels` recalcule exactement le même rapport
sur une fenêtre locale de cellules DC, le quantifie sur huit niveaux, puis
**renormalise pour que la moyenne pondérée AC-et-fréquence des gammas par bloc
soit exactement le gamma global**. Le budget chroma total ne bouge donc pas ;
seule sa répartition change. C'est cette invariance qui rend l'expérience
lisible : tout écart de métrique vient de *où* sont allés les bits, jamais de
combien ont été dépensés. Un test unitaire vérifie l'invariant, et
`AUREA_SPECTRAL_LOCAL=0` reproduit la v21 octet pour octet.

Sur quatre sentinelles à Q40/Q60, la force 1 achète `+0,0126` dB de RGB pour
`+0,134 %` de débit — un taux de change à peine meilleur que celui du codec
lui-même, et **Y ne bouge pas** (`-0,0001` dB), alors que Y était toute la
raison d'être de la piste. Le protocole complet tranche :

| Métrique | BD-rate cubique | Victoires | IC95 |
|---|---:|---:|---:|
| RGB PSNR | -0,043 % | 8/12 | [-0,124 ; +0,051] |
| Y PSNR | **+0,012 %** | 7/12 | [-0,075 ; +0,102] |
| Cb PSNR | -0,199 % | 8/12 | [-0,534 ; +0,133] |
| Cr PSNR | -0,195 % | 6/12 | [-0,638 ; +0,236] |
| SSIM-Y | +0,027 % | 6/12 | [-0,041 ; +0,103] |
| Delta-E moyen | **+0,202 %** | 3/12 | **[+0,011 ; +0,398]** |
| Delta-E p95 | +0,332 % | 4/12 | [-0,060 ; +0,784] |

Tous les intervalles traversent zéro sauf un : **le Delta-E moyen, qui perd**.
La seule conclusion statistiquement soutenue est donc une dégradation
perceptuelle de la couleur, sur 3 images gagnantes seulement — et la piste
exigeait explicitement d'être conservative en Delta-E.

Le mécanisme est instructif dans son échec. Concentrer le grossissement chroma
là où la chroma est *localement plate* place l'erreur exactement là où l'œil la
voit le mieux : une zone chroma lisse révèle le moindre banding, quand une zone
chroma agitée le masque. Le PSNR Cb/Cr, qui ne pondère rien, enregistre même une
légère amélioration ; le Delta-E, qui pondère perceptuellement, enregistre la
perte. La leçon n'est pas que la localisation est mauvaise, c'est que **le
rapport d'activité chroma/luma n'est pas un masque perceptuel** : il mesure où
la chroma est prévisible, pas où son erreur serait invisible.

### Piste 2 — Enveloppe inter-canaux (rejetée)

Le codon v17 donne déjà à C1 la classe de magnitude de Lphi *à la même
position*, et à C2 le maximum de Lphi et C1. La question était de savoir si
l'enveloppe de **voisinage** de Lphi ajoute quoi que ce soit. Le doublon était
l'hypothèse prudente ; la télémétrie dit le contraire.

`[AUREA_CROSS_ENVELOPE]` compare trois conditionnements emboîtés sur les symboles
quantifiés exacts. L'enveloppe de Lphi apporte, *après* le codon, de
`0,026` à `0,103` bit par symbole — sur `kodim01` à Q20, C1 y gagne `0,1026`
bit/symbole quand le codon lui-même n'en apportait que `0,0729`. L'information
est donc massivement réelle.

Elle n'est pourtant pas extractible. Sur les 48 points du corpus :

| Combinaison | Delta AC | Images gagnées |
|---|---:|---:|
| `max(ponctuel, voisinage)` | +0,666 % | 0/48 |
| voisinage seul | +0,708 % | 1/48 |
| somme saturée | +0,583 % | 2/48 |
| axe propre, codon 4 -> 8 états | **+0,0355 %** | 16/48 |
| axe propre + mémoire séparable | +0,0321 % | 27/48 |

Les trois fusions à budget de contexte constant **détruisent** la valeur
ponctuelle que le codon portait déjà, et le paient très cher. Un axe propre ne
détruit rien mais paie la dilution : il ramène la perte à `+0,0355 %` sans
jamais passer sous zéro.

C'est la leçon du cycle 15 reformulée. Là-bas une dépendance mesurable échouait
faute de contexte causal et gratuit ; ici le contexte *est* causal et gratuit, et
l'obstacle est l'arbitrage entre destruction et dilution. Aucune des deux faces
n'est franchissable avec un alphabet de cette taille.

### Piste 3 — Bucketing indexé sur l'échelle (rejetée)

Les seuils absolus `{0} {1} {2..3} {4+}` semblaient devoir saturer quand la
qualité monte, ce qui expliquerait la décroissance du gain de `-2,10 %` à Q20
vers `-0,92 %` à Q80. `AMPLITUDE_MAGNITUDE_SCALED` rapporte donc la somme des
magnitudes à la mémoire IIR d'énergie que le modèle entretient déjà.

Le résultat est franc : `+0,785 %` d'AC pour la version pure, `0/48` victoires,
et `+0,005 %` une fois combinée au reste. **Le diagnostic était faux.** La
mémoire d'énergie du modèle est une quantité *séquentielle* : elle suit la
position dans le zigzag, où les basses fréquences sont fortes et les hautes
faibles, bien plus qu'elle ne suit la qualité. Diviser par elle ne dénormalise
pas la qualité — cela **efface l'information de bande fréquentielle** que le
seuil absolu portait implicitement. Le seuil absolu n'était pas naïf : il était
un détecteur de bande déguisé.

### Piste 4 — Loi séparable d'ordre 2 (marginale, non promue)

La loi séparable n'atteint que `59 %` d'accord sur Lphi contre `74-79 %` sur les
chromas : le contenu luma est simplement moins séparable. Trois témoins ont été
essayés pour distinguer les blocs où la loi doit tenir de ceux où elle ne le peut
pas — le bucket de Turing du bloc, l'énergie du voisinage, et une **mémoire
intra-bloc** notant si la loi a tenu la fois précédente dans ce même bloc.

La mémoire est la seule à gagner, et de peu : `-0,0033 %` d'AC sur les 48 points,
35/48 victoires. Le bucket de Turing est neutre (`-0,0002 %`) et l'énergie perd
(`+0,0042 %`, 3/48) — ce qui reconfirme le refus du couplage énergie/phase du
cycle 20. Le régime de la mémoire décroît avec la densité et **s'inverse** :

| Qualité | Delta AC | Images gagnées |
|---:|---:|---:|
| 20 | -0,0208 % | 12/12 |
| 40 | -0,0123 % | 11/12 |
| 60 | -0,0062 % | 10/12 |
| 80 | **+0,0027 %** | 2/12 |

Une porte de qualité `q <= 60`, sur le modèle de celle du parent dyadique au
cycle 15, donne exactement `-0,0048 %` d'AC, 33/36 sur les points ouverts et 11
images sur 12 en net. Le signe est stable, mais l'amplitude est trois cents fois
plus faible que le gain du cycle 20, `dispair` perd, et le mécanisme coûte une
mémoire par bloc des deux côtés. Sous la barre de promotion : la piste est
consignée avec ses chiffres, l'implémentation reste disponible sous
`AUREA_SPECTRAL_ENVELOPE_COUPLING=2`, et le profil par défaut ne bouge pas.

### Bilan

Les quatre expériences sont reconstructibles, symétriques et bit-exactes dans
leur état neutre ; les 165 tests passent. Aucune ne franchit le seuil, et le
profil v21 reste le profil actif sans changement de bitstream. Le cycle laisse
trois frontières mieux dessinées : le rapport d'activité chroma/luma n'est pas un
masque perceptuel, une dépendance gratuite et causale peut rester inexploitable
par simple arbitrage destruction/dilution, et le seuil absolu de l'enveloppe est
un détecteur de bande fréquentielle qu'il ne faut pas normaliser.

## Pistes suivantes

- **Masque chroma perceptuel** : la piste 1 a échoué parce qu'elle localisait le
  troc sur la *prévisibilité* de la chroma, pas sur la *visibilité* de son
  erreur. Un champ zéro-bit dérivé des grilles DC mais construit comme un
  masque — variance chroma locale en espace perceptuel, ou sensibilité
  CIEDE2000 estimée depuis L\* — inverserait le signe du raisonnement. Le
  verdict doit rester le Delta-E, seule métrique qui ait tranché ce cycle.
- **Mélange de contextes plutôt que fusion** : la piste 2 dispose de `0,03` à
  `0,10` bit/symbole causal et gratuit, bloqué par un arbitrage binaire entre
  détruire et diluer. Un mélange de deux modèles — l'un indexé par le codon
  v17, l'autre par l'enveloppe de Lphi, pondérés adaptativement — échappe par
  construction aux deux écueils. C'est un changement de nature du moteur
  entropique, pas un contexte de plus, et il mérite d'être évalué comme tel.
- **Taille de bloc comme témoin de séparabilité** : la piste 4 a essayé Turing,
  l'énergie et la mémoire intra-bloc, mais pas la taille de transformée. Les
  blocs 32x32 sont ceux que le quadtree a jugés lisses ; ce sont aussi ceux où
  un contenu séparable est le plus probable. Le témoin est déjà dans la
  syntaxe, il ne coûte rien, et il n'a pas été mesuré.
- **Frontière Y, toujours ouverte** : `-2,579 %` face au 4:2:0 sur 7/12 images.
  Le cycle 21 a écarté le levier d'allocation locale ; le gain de 1,6 point du
  cycle 20 est venu du seul débit. La piste la plus probable est donc désormais
  entropique, pas psycho-visuelle.

## Cycle 22 — Mélange logistique de contextes (AUR2 v22)

Le cycle 21 s'est arrêté sur un constat frustrant et précis. Le voisinage
spectral de Lphi porte, *après* le codon v17, `0,026` à `0,103` bit par symbole
d'information causale et gratuite. Aucune manière de l'écrire dans un index de
contexte ne la récupère : la fondre dans les quatre états existants détruit la
valeur ponctuelle que le codon portait déjà (`+0,58 %` à `+0,71 %` d'AC), lui
donner son propre axe ne détruit rien mais paie une dilution qui annule le gain
(`+0,036 %`). Les deux murs viennent de la même hypothèse tacite — **qu'un seul
index de contexte doit porter tous les témoins**.

### Deux avis valent mieux qu'un index

Le mélange abandonne cette hypothèse. Deux modèles observent le même bit à
travers deux contextes différents et énoncent chacun une probabilité ; un
troisième modèle, minuscule, apprend combien croire chacun. L'arbitrage se fait
dans le domaine logit, où les probabilités s'additionnent :

```text
logit(p) = w1 · logit(p_codon) + w2 · logit(p_voisinage)
```

Le modèle primaire reste exactement celui de la v21, indexé par le codon v17.
Le secondaire est délibérément *plus petit* — `run x énergie x enveloppe x
voisinage_Lphi` — parce qu'un second avis n'a besoin d'être qu'approximativement
juste et approximativement indépendant ; le détailler autant que le primaire ne
ferait que reproduire sa dilution. Le mélangeur, lui, ne dispose que de seize
jeux de poids : la question « à qui se fier » est bien plus simple que celle que
les modèles eux-mêmes traitent, et la découper finement réintroduirait
exactement la dilution que le mélange existe pour éviter.

Les poids partent de `(1, 0)` : le mélangeur commence comme le modèle primaire
seul et doit *gagner* le droit de s'en écarter.

**Tout est entier.** Un codec ne peut pas se permettre du flottant dans un
chemin que le décodeur doit reproduire exactement, sur n'importe quelle machine.
`squash` est la logistique tabulée sur 33 ancres avec interpolation linéaire,
`stretch` son inverse construit par inversion monotone de cette même table, et
les poids sont en virgule fixe sur 16 bits.

### Le régime

Le témoin octet-exact sur les 12 `originals` à Q20/40/60/80, face au binaire
v21, retient le taux d'apprentissage le plus lent :

| Taux | Delta AC | Images gagnées |
|---:|---:|---:|
| **1** | **-0,2825 %** | **48/48** |
| 2 | -0,2472 % | 48/48 |
| 3 | -0,1897 % | 48/48 |
| 5 | -0,0542 % | 40/48 |
| 8 | +0,1765 % | 20/48 |
| 12 | +0,5282 % | 10/48 |

Le gain est remarquablement **plat en qualité**, ce qu'aucun mécanisme des
cycles précédents n'avait fait :

| Qualité | Delta AC | Images gagnées |
|---:|---:|---:|
| 20 | -0,2491 % | 12/12 |
| 40 | -0,2616 % | 12/12 |
| 60 | **-0,2929 %** | 12/12 |
| 80 | -0,2882 % | 12/12 |

L'enveloppe du cycle 20 décroissait de `-2,75 %` à `-0,98 %` parce que son axe
d'amplitude saturait ; le mélange ne sature pas, parce qu'il n'ajoute aucun état
à saturer. Par image le gain va de `-0,126 %` (`burn_out`) à `-0,417 %`
(`chilling`), 4/4 qualités sur chacune des douze.

### Un négatif au passage

La **taille de transformée** comme témoin de séparabilité, la piste que le cycle
21 avait laissée ouverte faute de l'avoir essayée, ne donne rien :
`-0,0002 %` d'AC, 26/48 — du bruit. Le quadtree a jugé qu'un bloc 32x32 est
lisse, ce qui n'implique pas qu'il soit *séparable* : une texture douce mais
isotrope est lisse et parfaitement non séparable. Le témoin est gratuit et
disponible, il n'est simplement pas informatif.

### Un piège de compilation, et pourquoi le contrôle de non-régression existe

En ajoutant `PHASE_CONTEXT_SEPARABLE_SIZE` au `match` des contextes de phase,
la constante n'avait pas été importée dans `rans.rs`. Rust a alors traité
l'identifiant comme un **motif de liaison** — un bras attrape-tout — qui
détournait silencieusement le profil promu vers un sous-contexte erroné. Le
compilateur ne signalait qu'un `unused variable`, et l'autotest du harnais
restait à zéro puisque production et témoin empruntaient tous deux le chemin
fautif. Seul le contrôle « 16 encodages sentinelles identiques octet pour
octet » l'a rendu visible. C'est précisément le rôle de ce contrôle : un
autotest interne ne peut pas attraper un bug qui déplace la référence en même
temps que la mesure.

### Verdict débit-distorsion

Le protocole complet a recalculé les 48 points Aurea contre les 384 points JPEG
déjà validés. La comparaison directe v21 -> v22 :

| Métrique | BD-rate cubique | Victoires | IC95 bootstrap | Contrôle linéaire |
|---|---:|---:|---:|---:|
| RGB PSNR | **-0,240 %** | 12/12 | [-0,289 ; -0,194] | -0,235 % |
| Y PSNR | **-0,240 %** | 12/12 | [-0,288 ; -0,193] | -0,235 % |
| Cb PSNR | -0,250 % | 12/12 | [-0,299 ; -0,203] | -0,240 % |
| Cr PSNR | -0,253 % | 12/12 | [-0,301 ; -0,206] | -0,241 % |
| SSIM-Y | -0,221 % | 12/12 | [-0,263 ; -0,181] | -0,223 % |
| Delta-E moyen | -0,237 % | 12/12 | [-0,281 ; -0,195] | -0,234 % |
| Delta-E p95 | -0,240 % | 12/12 | [-0,285 ; -0,197] | -0,236 % |

Aucun intervalle ne traverse zéro. Comme au cycle 20, toutes les métriques
gagnent la même chose parce qu'elles mesurent la même image : à qualité fixe la
chaîne de mesure indépendante confirme `+0,000` dB sur RGB, Y, Cb, Cr et
`+0,000` sur Delta-E aux quatre qualités, pour `-0,183 %`, `-0,215 %`,
`-0,253 %` et `-0,257 %` de bpp.

Face au JPEG optimisé, la frontière avance encore :

| Référence | Métrique | v21 | v22 |
|---|---|---:|---:|
| 4:2:0 | RGB PSNR | -34,106 % | **-34,246 %** |
| 4:2:0 | Y PSNR | -2,579 % (7/12) | **-2,813 %** (7/12) |
| 4:2:0 | SSIM-Y | -10,039 % (11/12) | **-10,236 %** (12/12) |
| 4:2:0 | Delta-E moyen | -41,008 % | **-41,137 %** |
| 4:4:4 | RGB PSNR | -26,971 % | **-27,144 %** |
| 4:4:4 | Y PSNR | -25,109 % | **-25,287 %** |
| 4:4:4 | SSIM-Y | -31,602 % | **-31,750 %** |

Le SSIM-Y face au 4:2:0 passe de 11/12 à **12/12**. L'avance Y reste 7/12 avec
un IC95 `[-6,822 ; +0,877]` qui traverse toujours zéro : la frontière Y n'est
pas encore emportée, mais elle a encore gagné 0,23 point sans qu'aucune
décision psycho-visuelle ne soit touchée — ce qui confirme le pronostic du
cycle 21, à savoir que ce qui reste à prendre sur Y est entropique.

### Comparaison directe et dette de calcul

Sur les 48 points, les PNG décodés partagent leur SHA-256 avec la
reconstruction v21 : **zéro pixel n'a bougé**.

| Qualité | Fichiers v21 | Fichiers v22 | Delta | Images gagnées |
|---:|---:|---:|---:|---:|
| 20 | 5 023 601 | 5 014 107 | -0,189 % | 12/12 |
| 40 | 8 016 412 | 7 999 179 | -0,215 % | 12/12 |
| 60 | 12 493 521 | 12 461 793 | -0,254 % | 12/12 |
| 80 | 28 443 335 | 28 366 642 | -0,270 % | 12/12 |
| **total** | **53 976 869** | **53 841 721** | **-0,250 %** | **48/48** |

Comme au cycle 20, les `135 148` octets économisés sur les fichiers complets
sont exactement ceux que le témoin annonçait sur l'AC seul.

La dette de calcul est marginale : `+1,24 %` à l'encodage et `+1,94 %` au
décodage sur `kryte` q60, `-1,25 %` et `+4,73 %` sur `kodim01` q60 — l'écart
négatif étant du bruit de mesure. Le mélangeur ne touche que les canaux
chromatiques, n'ajoute qu'un produit scalaire de deux termes et deux
consultations de table par symbole, et n'alloue que seize jeux de poids. Face
aux `+4 %` à `+13 %` qu'avait coûtés l'enveloppe du cycle 20, c'est
négligeable.

**Verdict 2026-08-15 : promu en AUR2 v22.** Le mécanisme est purement
entropique : aucun pas de quantification, aucun parcours, aucune décision
psycho-visuelle ne change. `AUREA_SPECTRAL_ENVELOPE_MIX=0` produit un flux v21
bit-exact et `AUREA_SPECTRAL_ENVELOPE=0` un flux v20 ; le décodeur refuse un
flux v22 sans le moteur correspondant plutôt que de le décoder à moitié. Les
165 tests passent. Le binaire final est
`target/benchmark_v22/aurea_v22_mixed.exe`, SHA-256
`e1f5dab9a386ae2ae43b279727e3f03a67f6bb93f138d4316dc3f5823bcc8679`.
Le rapport JPEG est dans `target/benchmark_v22/jpeg_frontier_v22`, la
comparaison directe v21/v22 dans `target/benchmark_v22/direct_v21_v22`, et les
balayages du témoin dans `target/spectral_envelope_lab/witness_cycle22.csv`.

## Pistes suivantes

- **Mélanger davantage** : le mélange ne porte aujourd'hui que sur `P(zéro)` et
  seulement sur les chromas. `P(|x|=1)` suit la même grammaire et n'a pas été
  mélangé ; Lphi n'a pas de canal antérieur mais pourrait opposer deux vues de
  son propre voisinage. Le moteur existe désormais, l'extension est mécanique.
- **Un troisième avis** : le mélangeur accepte n'importe quel nombre d'entrées.
  Les témoins déjà mesurés et refusés faute de pouvoir entrer dans un index —
  la mémoire intra-bloc du cycle 21, l'enveloppe indexée sur l'échelle — sont
  exactement les candidats qu'un mélange peut accueillir sans dilution. Ce qui
  a échoué comme *contexte* mérite d'être réessayé comme *entrée*.
- **SSE / APM** : après le mélange, une seconde étape d'affinage de probabilité
  indexée par la probabilité mélangée elle-même est le complément classique du
  mélange logistique, et coûte peu.
- **Frontière Y** : `-2,813 %` face au 4:2:0, 7/12, IC95 encore ouvert. Deux
  cycles purement entropiques ont presque triplé l'avance sans toucher à
  l'allocation. La question est maintenant de savoir si Y peut être emporté par
  le seul débit, ou s'il faudra finir par un levier perceptuel — et le cycle 21
  a montré que ce levier devra être un masque de *visibilité*, pas un rapport
  d'activité.

## Cycle 23 — Le mélange élargi (AUR2 v23)

Le cycle 22 avait ouvert une porte sans la franchir. Il avait montré qu'un
témoin refusé comme *contexte* peut réussir comme *entrée* de mélange, mais ne
l'avait démontré que sur un seul témoin, un seul modèle secondaire, et une
seule décision — `P(zéro)` des chromas. Trois extensions évidentes restaient
ouvertes. Toutes les trois paient.

### Ce qui a échoué comme contexte réussit comme entrée

Le résultat central du cycle tient en une ligne. L'enveloppe rapportée à
l'échelle locale, mesurée au cycle 21, avait perdu `+0,785 %` d'AC en `0/48`
comme axe de contexte ; la *même* grandeur, sur les *mêmes* images, gagne
`-0,233 %` en `48/48` comme entrée de mélange.

| Témoin | Comme contexte | Comme entrée |
|---|---:|---:|
| Enveloppe rapportée à l'échelle | **+0,785 %** (0/48) | **-0,233 %** (48/48) |

Rien n'a changé dans le signal. Ce qui a changé, c'est qu'un index de contexte
doit *tout porter d'un coup* — chaque état ajouté divise les statistiques de
tous les autres — alors qu'une entrée de mélange n'a qu'à être approximativement
juste et approximativement indépendante. Le cycle 21 avait diagnostiqué que
normaliser par l'énergie « efface l'information de bande fréquentielle » ; c'est
exact, et c'est sans importance dès lors que le modèle primaire, lui, ne
l'efface pas.

### Une bande fréquentielle explicite

Le second témoin est plus surprenant encore par sa simplicité. Le modèle
primaire n'a **jamais** connu la fréquence autrement que par sa mémoire IIR
d'énergie, qui n'en est qu'un indice indirect et retardé. Une bande explicite —
la position zigzag en quatre paliers — apporte `-0,329 %` à elle seule, en
`48/48`. C'est le meilleur témoin unique du cycle, et il était disponible depuis
la v12.

### Mélanger la seconde décision, puis se relire

`P(|x| = 1)` suit la même grammaire que `P(zéro)` et le cycle 22 l'avait laissée
au modèle primaire seul. La mélanger apporte `-0,076 %` avec le seul témoin de
la v22 — mais `-0,894 %` une fois les trois témoins présents. L'effet est
nettement **sur-additif** : une seconde décision ne vaut d'être mélangée que si
le mélangeur a de quoi arbitrer.

L'APM, enfin, relit la probabilité mélangée à travers une table indexée par
cette probabilité elle-même, quantifiée sur l'échelle logit. Il n'apprend rien
sur l'image : il apprend la forme du biais résiduel du mélangeur. `-0,106 %` à
la vitesse 6, `48/48`.

### Le régime

| Variante | Delta AC | Images gagnées |
|---|---:|---:|
| Bande seule | -0,331 % | 48/48 |
| Bande + Lphi | -0,329 % | 48/48 |
| Échelle + Lphi | -0,233 % | 48/48 |
| Trois témoins | -0,359 % | 48/48 |
| APM seul (vitesse 6) | -0,106 % | 48/48 |
| `P(\|x\|=1)` seul | -0,076 % | 48/48 |
| Trois témoins + `P(\|x\|=1)` | -0,894 % | 48/48 |
| **Tout, taux 1** | **-1,161 %** | **48/48** |

Par qualité `-1,444 %`, `-1,301 %`, `-1,255 %` et `-1,047 %` ; par image de
`-1,011 %` (`dispair`) à `-1,546 %` (`burn_out`), 4/4 sur chacune des douze. Le
taux d'apprentissage `1` reste le meilleur, comme au cycle 22 : la question
posée au mélangeur est simple et n'a pas besoin d'être apprise vite.

### Verdict débit-distorsion

| Métrique | BD-rate cubique | Victoires | IC95 bootstrap | Contrôle linéaire |
|---|---:|---:|---:|---:|
| RGB PSNR | **-1,072 %** | 12/12 | [-1,169 ; -0,993] | -1,060 % |
| Y PSNR | **-1,076 %** | 12/12 | [-1,175 ; -0,995] | -1,061 % |
| Cb PSNR | -1,073 % | 12/12 | [-1,169 ; -0,996] | -1,054 % |
| Cr PSNR | -1,073 % | 12/12 | [-1,169 ; -0,997] | -1,052 % |
| SSIM-Y | -1,065 % | 12/12 | [-1,152 ; -0,989] | -1,070 % |
| Delta-E moyen | -1,061 % | 12/12 | [-1,154 ; -0,983] | -1,055 % |
| Delta-E p95 | -1,060 % | 12/12 | [-1,149 ; -0,987] | -1,054 % |

Aucun intervalle ne traverse zéro, et la chaîne de mesure indépendante confirme
`+0,000` dB sur toutes les métriques aux quatre qualités pour `-1,081 %`,
`-1,064 %`, `-1,094 %` et `-0,998 %` de bpp : gain de débit pur, une fois de
plus.

### La frontière Y est franchie

Face au JPEG optimisé 4:2:0, la luminance atteint **`-3,860 %`, IC95
`[-7,816 ; -0,204]`**. C'est la première fois depuis le cycle 17 — où elle
valait `+3,331 %` — que **l'intervalle ne traverse plus zéro**. L'avance
moyenne en luminance est désormais statistiquement soutenue.

La nuance mérite d'être dite : le compte par image reste `7/12`, et la borne
haute de l'intervalle est `-0,204 %`, à peine négative. Ce n'est donc pas encore
une victoire sur chaque image, c'est une avance moyenne qui a cessé d'être
attribuable au hasard. Le pronostic du cycle 21 — « ce qui reste à prendre sur Y
est entropique » — est confirmé de la manière la plus directe : trois cycles
consécutifs sans qu'une seule décision psycho-visuelle ne soit touchée.

| Référence | Métrique | v22 | v23 |
|---|---|---:|---:|
| 4:2:0 | RGB PSNR | -34,246 % | **-34,975 %** |
| 4:2:0 | Y PSNR | -2,813 % (IC ouvert) | **-3,860 %** (IC fermé) |
| 4:2:0 | SSIM-Y | -10,236 % | **-11,197 %** |
| 4:2:0 | Delta-E moyen | -41,137 % | **-41,779 %** |
| 4:2:0 | Delta-E p95 | -36,557 % | **-37,242 %** |
| 4:4:4 | RGB PSNR | -27,144 % | **-27,932 %** |
| 4:4:4 | Y PSNR | -25,287 % | **-26,097 %** |
| 4:4:4 | SSIM-Y | -31,750 % | **-32,486 %** |
| 4:4:4 | Delta-E p95 | -15,114 % (10/12) | **-16,012 %** (11/12) |

### Comparaison directe et dette de calcul

Sur les 48 points, les PNG décodés partagent leur SHA-256 avec la
reconstruction v22 : **zéro pixel n'a bougé**.

| Qualité | Fichiers v22 | Fichiers v23 | Delta | Images gagnées |
|---:|---:|---:|---:|---:|
| 20 | 5 014 107 | 4 959 210 | -1,095 % | 12/12 |
| 40 | 7 999 179 | 7 913 692 | -1,069 % | 12/12 |
| 60 | 12 461 793 | 12 326 273 | -1,087 % | 12/12 |
| 80 | 28 366 642 | 28 088 929 | -0,979 % | 12/12 |
| **total** | **53 841 721** | **53 288 104** | **-1,028 %** | **48/48** |

Les `553 617` octets économisés sur les fichiers complets sont, une troisième
fois de suite, exactement ceux que le témoin annonçait sur l'AC seul.

La dette de calcul est cette fois réelle : `+14,6 %` à l'encodage et `+12,7 %`
au décodage sur `kryte` q60, `+10,8 %` et `-0,3 %` sur `kodim01` q60. Quatre
modèles au lieu de deux, une seconde décision mélangée et un APM se paient.
Rapportée au `-1,03 %` de débit, la facture reste très favorable — et bien
au-dessous du `+31,3 %` de décodage pour `-0,0019 %` qui avait fait rejeter la
morphogenèse emboîtée du cycle 3.

**Verdict 2026-08-15 : promu en AUR2 v23.** Le mécanisme est purement
entropique : aucun pas de quantification, aucun parcours, aucune décision
psycho-visuelle ne change. Les quatre grammaires coexistent et sont
auto-décrites — `AUREA_SPECTRAL_ENVELOPE_MIX_INPUTS=1` avec `MIX_PM1=0` et
`APM=0` produit un v22, `AUREA_SPECTRAL_ENVELOPE_MIX=0` un v21,
`AUREA_SPECTRAL_ENVELOPE=0` un v20, et le décodeur refuse un flux dont il n'a
pas le moteur plutôt que de le décoder à moitié. Les 166 tests passent, dont
un qui vérifie le passage aller-retour des 48 formes de mélangeur. Le binaire
final est `target/benchmark_v23/aurea_v23_multimix.exe`, SHA-256
`ddafa4ea82f893dadce7095b98ba00d43e51c51690b4c8a53fba1274fb3ef841`.
Le rapport JPEG est dans `target/benchmark_v23/jpeg_frontier_v23`, la
comparaison directe dans `target/benchmark_v23/direct_v22_v23_bd`.

### Bilan des quatre cycles

| Version | Mécanisme | BD-rate RGB vs précédent |
|---|---|---:|
| v21 | Enveloppe amplitude/phase par voisinage spectral | -1,626 % |
| v22 | Mélange logistique, un avis secondaire | -0,240 % |
| v23 | Mélange élargi : quatre avis, `P(\|x\|=1)`, APM | -1,072 % |

Aucun de ces trois cycles n'a déplacé un seul pixel. Ensemble ils ont fait
passer la luminance face au JPEG 4:2:0 de `-0,975 %` avec un intervalle ouvert à
`-3,860 %` avec un intervalle fermé, et le RGB de `-32,877 %` à `-34,975 %`.

## Pistes suivantes

- **Un cinquième avis** : le mélangeur est plafonné à quatre entrées par
  `MIX_MAX_INPUTS`. Les candidats non essayés ne manquent pas — le bucket de
  Turing du bloc, la taille de transformée, la mémoire intra-bloc du cycle 21,
  tous refusés comme contextes et jamais présentés comme entrées.
- **Mélanger le signe** : les compteurs d'accord de phase restent des modèles
  uniques indexés par classe. Le mélange s'y appliquerait tel quel, et la loi
  séparable dispose de témoins mesurés — mémoire intra-bloc, énergie du
  voisinage — qui n'avaient pas survécu à la dilution d'un index.
- **Mélanger Lphi** : le mélange ne sert aujourd'hui que les chromas pour ce qui
  est du témoin étranger, mais la bande et l'échelle valent pour les trois
  canaux ; l'implémentation les active déjà partout, ce que le cycle n'a pas
  isolé faute de variante dédiée.
- **Victoire Y par image** : l'avance moyenne est acquise, le compte par image
  reste `7/12`. Comprendre *quelles* images perdent, et pourquoi, est désormais
  une question plus utile que de gratter un point de plus sur la moyenne.

## Cycle 24 — Consolidation du mélange (AUR2 v24)

Un cycle de consolidation : quatre pistes, deux gains modestes, un refus net et
une hypothèse falsifiée. Les ordres de grandeur ne sont plus ceux des cycles 22
et 23, ce qui est le signe attendu d'un mécanisme qui approche son rendement.

### Où le mélange paie réellement

Le cycle 23 avait activé la bande et l'échelle sur les trois canaux sans jamais
isoler leur contribution par canal. Deux variantes le tranchent :

| Mélange désactivé sur | Coût |
|---|---:|
| Lphi | **+0,942 %** |
| Chromas | +0,519 % |

Le gain du cycle 23 est donc majoritairement — mais pas exclusivement — luma,
alors même que le témoin fondateur du cycle 22, le voisinage de Lphi, est
chroma-seul par construction. Ce sont la bande et l'échelle, applicables
partout, qui portent l'essentiel.

### Le corollaire n'est pas automatique

Le cycle 23 avait établi qu'un témoin refusé comme *contexte* mérite d'être
réessayé comme *entrée*. Le cycle 24 en teste deux de plus et sépare les cas :

| Témoin | Comme contexte | Comme entrée |
|---|---:|---:|
| Comptage de voisins actifs | `-0,755 %` (cycle 20, perdant face à la somme) | **-0,023 %** (38/48) |
| Taille de transformée | `-0,0002 %` (cycle 22, bruit) | **+0,034 %** (0/48) |

Le comptage passe, la taille échoue des deux côtés. Le corollaire tient donc
sous condition : réessayer comme entrée ne sauve que les témoins qui *portent*
une information et que la dilution étouffait. La taille de transformée n'en
porte simplement pas — un bloc lisse n'est ni plus ni moins prévisible.

### Le signe, dernière décision non mélangée

Les compteurs d'accord de phase étaient restés des modèles uniques indexés par
classe depuis le cycle 20. Un second avis indexé par `classe x mémoire
intra-bloc x énergie du voisinage` — deux témoins mesurés et refusés comme
contextes aux cycles 20 et 21 — apporte `-0,031 %` en **48/48**. Petit, mais
propre sur chaque point.

Combiné au comptage de voisins, le profil retenu donne `-0,054 %`, net négatif
sur **les douze images et les quatre qualités**, 46/48 points individuels.

| Variante | Delta AC | Points gagnés |
|---|---:|---:|
| **Comptage + signe** | **-0,054 %** | 46/48 |
| Signe seul | -0,031 % | 48/48 |
| Comptage seul | -0,023 % | 38/48 |
| Comptage + taille + signe | -0,017 % | 34/48 |
| Taille seule | +0,034 % | 0/48 |

### Une hypothèse falsifiée sur la frontière Y

Le cycle 23 laissait ouverte la question de savoir *quelles* images perdent
encore en luminance. Les cinq perdantes sont `kryte` (`+0,50 %`),
`mind_your_steps` (`+1,02 %`), `peace` (`+1,22 %`), `dispair` (`+4,59 %`) et
`chilling` (`+4,93 %`).

L'hypothèse naturelle — ce sont les images pauvres en chroma, où l'échange
spectral n'a rien à troquer — est **fausse**. `chilling` a la plus forte
amplitude chroma du corpus (RMS 53,9) et perd ; `kryte` a le gamma d'échange le
plus élevé (0,873) et perd, quand `burn_out` a le deuxième plus élevé (0,847)
et gagne largement. Ni l'amplitude chroma, ni le gamma, ni le rapport
d'activité ne prédisent le résultat.

Ce que les données montrent en revanche, c'est une anomalie isolée : `kryte`
atteint `-5,13 %` de BD-rate RGB là où toutes les autres images sont entre
`-27 %` et `-46 %`, et `-12,43 %` en Cb contre `-47 %` à `-75 %`. Ce n'est pas
un continuum, c'est un cas particulier — et c'est déjà l'image que le cycle 11
avait dû diagnostiquer séparément. La question à instruire n'est donc pas
« quelle propriété prédit la perte Y » mais « qu'est-ce qui, dans `kryte`,
échappe au codec ».

### Verdict débit-distorsion

| Métrique | BD-rate cubique | Victoires | IC95 bootstrap |
|---|---:|---:|---:|
| RGB PSNR | **-0,060 %** | 12/12 | [-0,078 ; -0,043] |
| Y PSNR | -0,060 % | 12/12 | [-0,078 ; -0,044] |
| Cb PSNR | -0,055 % | 12/12 | [-0,070 ; -0,040] |
| Cr PSNR | -0,053 % | 12/12 | [-0,067 ; -0,039] |
| SSIM-Y | -0,069 % | 12/12 | [-0,094 ; -0,047] |
| Delta-E moyen | -0,060 % | 12/12 | [-0,078 ; -0,043] |
| Delta-E p95 | -0,059 % | 12/12 | [-0,076 ; -0,043] |

Aucun intervalle ne traverse zéro, et la chaîne de mesure confirme une fois de
plus `+0,000` dB sur toutes les métriques aux quatre qualités. Face au JPEG
optimisé 4:2:0 : `-35,024 %` RGB, `-3,919 %` Y (IC95 `[-7,863 ; -0,269]`,
toujours fermé), `-11,259 %` SSIM-Y, `-41,822 %` Delta-E moyen.

Sur les 48 points de la comparaison directe, les PNG décodés partagent leur
SHA-256 avec la reconstruction v23 : **zéro pixel n'a bougé**, pour `-0,048 %`
de fichiers agrégés (`-0,09 %` à Q20, `-0,02 %` à Q80).

**Verdict 2026-08-15 : promu en AUR2 v24.** Le gain est dix fois plus petit que
celui du cycle 23, ce qui est le signe attendu d'un mécanisme qui approche son
rendement ; il reste propre sur les douze images, les quatre qualités et les
sept métriques. Les cinq grammaires coexistent et sont auto-décrites. Les 166
tests passent. Binaire final `target/benchmark_v24/aurea_v24_signmix.exe`,
SHA-256 `3a8cd299e114ae8be3eb1d395496a8328b5383071935f92b11546c660ad4a665`.

### Bilan des cinq cycles

| Version | Mécanisme | BD-rate RGB vs précédent |
|---|---|---:|
| v21 | Enveloppe amplitude/phase par voisinage spectral | -1,626 % |
| v22 | Mélange logistique, un avis secondaire | -0,240 % |
| v23 | Mélange élargi : quatre avis, `P(\|x\|=1)`, APM | -1,072 % |
| v24 | Cinquième avis et mélange du signe | -0,060 % |

Aucun de ces quatre cycles n'a déplacé un seul pixel. Ensemble ils portent la
luminance face au JPEG 4:2:0 de `-0,975 %` (intervalle ouvert) à `-3,919 %`
(intervalle fermé), et le RGB de `-32,877 %` à `-35,024 %`.

## Pistes suivantes

- **L'anomalie `kryte`** : `-5,13 %` de BD-rate RGB contre `-27 %` à `-46 %`
  partout ailleurs, et `-12,43 %` en Cb contre `-47 %` à `-75 %`. C'est le seul
  point du corpus qui ne ressemble à aucun autre, et cinq cycles d'entropie n'y
  ont presque rien changé. Comprendre cette image vaut mieux que gratter une
  décimale sur les onze autres.
- **Le rendement du mélange** : `-1,072 %` puis `-0,060 %`. La courbe est
  claire. Les entrées restantes — bucket de Turing, mémoire intra-bloc pour
  `P(zéro)`, codon comme second avis — valent d'être mesurées, mais l'attente
  raisonnable est désormais la décimale, pas le point.
## Cycle 25 — La distribution de la luminance (AUR2 v24, correctifs encodeur)

Le PSNR agrégé dit *combien* d'erreur il y a, jamais *où*. Ce cycle part d'une
intuition simple : découper l'erreur de luminance par tranche tonale. Elle a
livré deux bugs et un négatif instructif.

### Bug 1 — les images uniformes ne se décodaient pas

Une image d'un gris uniforme s'encode puis **échoue au décodage**, à toutes les
qualités. Le noir et le blanc purs passaient, les gris moyens non.

`normalize_01` gardait le cas plat avec un seuil **absolu**, `f64::EPSILON`
(`2,2e-16`), alors que la grille DC dont dérive le DoG atteint `4·10⁴` — la LOT
adaptative stocke le DC comme `moyenne · 16²`. Le Sobel et les deux gaussiennes
y laissent un résidu d'arrondi de l'ordre de `10⁻¹¹`, quatre ordres de grandeur
au-dessus du garde. Sur une image plate, du **bruit de calcul pur était donc
étiré à pleine échelle `[0, 1]`**.

Ce n'est pas qu'inélégant. Le champ de Turing est zéro-bit : la règle 1 exige
qu'il soit reconstruit par exactement le même calcul des deux côtés. L'encodeur
part d'une grille `f64`, le décodeur de la même grille arrondie par l'en-tête
`f32` — chacun amplifiait donc un bruit *différent*. Mesure directe :
`bucket_sum = 52` à l'encodage contre `0` au décodage. Les contextes entropiques
divergeaient et le flux devenait illisible.

Le garde est désormais **relatif à l'échelle des données**. Les 48 encodages du
corpus restent identiques à l'octet — le cas ne se déclenche que sur une image
globalement plate — et un test verrouille sept niveaux de gris à trois qualités.

### Bug 2 — le DC héritait d'une zone morte prévue pour l'AC

La distribution montrait un biais d'assombrissement **monotone en luminance**,
de `-0,11` dans les ombres à `-0,62` dans le dernier bin, où il pesait `12 %`
du MSE local. Rapporté au niveau, un gain multiplicatif d'environ `-0,3 %`.

Le biais survivait au passage en espace PTF, donc n'était pas dû à la
non-linéarité ; il décroissait avec la qualité, donc venait de la
quantification ; et une reconstruction DC-seule le portait en entier. La cause
est explicite :

```rust
let qv = (v.abs() / dc_step_clamped + 0.5 - dead_zone).floor();
```

`DEAD_ZONE` est documentée dans `wavelet.rs` comme le seuil de la quantification
**AC** — les coefficients AC sont centrés et épars, les annuler est un gain. Le
DC normalisé est l'exact contraire : strictement non négatif et dense, `sign` y
vaut toujours `+1`. La zone morte n'y annulait donc jamais rien et se contentait
de **raboter chaque bloc vers le bas** de `dead_zone × dc_step`.

La correction est purement côté encodeur — la déquantification ne change pas —
donc **sans rupture de format** : les décodeurs existants lisent les nouveaux
fichiers.

| Métrique | BD-rate | Victoires | IC95 |
|---|---:|---:|---:|
| RGB PSNR | **-0,673 %** | 12/12 | [-0,871 ; -0,502] |
| Y PSNR | **-0,484 %** | 12/12 | [-0,553 ; -0,411] |
| Cb PSNR | -0,543 % | 12/12 | [-0,791 ; -0,347] |
| Cr PSNR | -0,660 % | 11/12 | [-0,969 ; -0,379] |
| **Delta-E moyen** | **-2,222 %** | 12/12 | [-3,366 ; -1,287] |

À débit constant (`+0,015 %` à `-0,002 %`), la distorsion gagne `+0,104` dB RGB
et `+0,078` dB Y à Q20. Le biais global tombe de `-0,180` à `-0,028`, et le
Y PSNR du corpus de `41,824` à `41,866` dB. C'est plus que les cycles 22, 23 et
24 réunis sur plusieurs métriques.

### Négatif — la compensation de Jensen n'est pas rentable

Il reste après correction un biais en **forme**, non plus en décalage :
`+0,179` dans les ombres profondes, `-0,367` dans les hautes lumières, quasi nul
au milieu. Le mécanisme est identifié : la PTF inverse étant convexe, la moyenne
linéaire d'un bloc vaut `g(D) + ½·g''(D)·V`, et le codec modifie `V` dans les
deux sens — il retire l'énergie des coefficients annulés et ajoute le bruit
d'arrondi des survivants. D'où le signe opposé aux deux extrémités.

La correction se dérive proprement : `δ = (1/γ − 1)·ΔV/(2D)`, applicable en une
seule passe puisque les coefficients AC existent avant la quantification du DC.
Elle a été implémentée et balayée. **Elle ne paie pas.**

| kappa | Y PSNR | Delta | Biais |
|---:|---:|---:|---:|
| 0,00 | 41,4549 | — | -0,0470 |
| 0,02 | 41,4555 | +0,5 mdB | -0,0364 |
| 0,08 | 41,4531 | **-1,8 mdB** | **-0,0021** |
| 0,25 | 41,4117 | -43 mdB | +0,0964 |

La valeur qui annule exactement le biais global fait *perdre* du PSNR. La raison
est instructive : l'estimateur de `ΔV` utilise un pas nominal qui ignore la
pondération fréquentielle du QMAT et toutes les modulations par bloc. Il corrige
donc la bonne quantité **en moyenne** et la mauvaise **bloc par bloc**, ajoutant
de la variance là où il retire du biais.

Le plafond théorique était de toute façon faible : le biais conditionnel ne pèse
que `0,45 %` du MSE, soit `0,0197` dB s'il était parfaitement annulé. Un
estimateur exact demanderait les vrais pas par coefficient, donc un encodage en
deux passes — `2×` le temps d'encodage pour au mieux `0,02` dB. Le rapport est
mauvais et le mécanisme reste neutre par défaut (`AUREA_DC_JENSEN=0`),
disponible pour un cycle qui voudrait payer la seconde passe.

## Cycle 26 — Recalibration psycho-visuelle (AUR2 v24, encodeur)

Quatre cycles consécutifs n'avaient touché que le débit. Les leviers
psycho-visuels, eux, dataient tous d'une grammaire entropique disparue et d'un
DC biaisé désormais corrigé. Un point Pareto d'alors n'est pas nécessairement un
point Pareto aujourd'hui : ce cycle les remesure tous.

### Un levier mort

`TUNABLE_TRNA` expose trois variables d'environnement qui ne font **rien** :
`lot::luminance_trna` n'est appelé nulle part — remplacé par une loi continue —
et le décodeur lie `FLAG_WEBER_TRNA` à une variable inutilisée. Le balayage le
confirme sans ambiguïté : `0,000 %` de BD-rate à *toutes* les valeurs testées.
Documenté sur place pour qu'une prochaine calibration n'y perde pas son temps.

### Le résultat : l'aveugle contre l'informé

Une zone morte et un treillis décident tous deux quels coefficients deviennent
nuls. Le premier le fait **aveuglément**, sur un seuil de magnitude figé ; le
second par **rate-distortion**, en consultant le coût réel du modèle entropique.
Quatre cycles de mélange de contextes ont rendu le second de plus en plus juste
et laissé le premier calibré pour un codeur qui n'existe plus.

Remplacer l'aveugle par l'informé n'est pas un arbitrage. À qualité fixe, sur
les douze images :

| Qualité | Débit | RGB | Y |
|---:|---:|---:|---:|
| 20 | **-2,032 %** | +0,086 dB | +0,138 dB |
| 40 | **-0,921 %** | +0,139 dB | +0,237 dB |
| 60 | -0,168 % | +0,161 dB | +0,305 dB |
| 80 | -0,261 % | +0,088 dB | +0,157 dB |

Moins de bits **et** plus de qualité partout. Zone morte AC à `0,0`, lambda du
treillis de `5,0` à `16,0`. Les deux vivent entièrement dans l'encodeur : la
déquantification ne change pas, donc **aucune rupture de format**.

| Métrique | BD-rate | Victoires | IC95 |
|---|---:|---:|---:|
| RGB PSNR | **-3,023 %** | 12/12 | [-3,280 ; -2,735] |
| Y PSNR | **-3,888 %** | 12/12 | [-4,121 ; -3,658] |
| Cb PSNR | -0,906 % | 9/12 | [-1,303 ; -0,494] |
| Cr PSNR | -1,149 % | 11/12 | [-1,547 ; -0,737] |
| SSIM-Y | -2,463 % | 12/12 | [-2,772 ; -2,209] |
| Delta-E moyen | -2,229 % | 12/12 | [-2,742 ; -1,791] |
| Delta-E p95 | -2,711 % | 12/12 | [-3,297 ; -2,136] |

Face au JPEG optimisé 4:2:0 : **-38,120 %** RGB, **-7,872 %** Y — et surtout
**10/12 images** en luminance contre 7/12, IC95 `[-11,863 ; -4,198]`. La
frontière Y cesse d'être une avance moyenne fragile pour devenir une victoire
majoritaire. Delta-E moyen atteint `-44,671 %`.

### Le piège de méthode, et deux candidats rejetés

Le balayage BD-rate désignait deux gagnants bien plus gros : pousser lambda
jusqu'à `48` (`-5,2 %` RGB) et abaisser le plateau QMAT bas débit à `0,40`
(`-5,6 %` RGB, gagnant même sur SSIM-Y et Delta-E). Les deux sont **faux**.

Le contrôle sur les points RD bruts le montre. À `lambda = 48`, q20 donne
`-14,6 %` de débit pour **`-0,87` dB** : le codec glisse le long de sa propre
courbe au lieu de l'améliorer. Le plateau QMAT à `0,40` fait l'inverse, il
*dépense* `+17,4 %` de débit pour `+1,29` dB — alors que la pente locale
(`~0,087` dB/%) en promettait `+1,51`. C'est une perte déguisée en gain.

Dans les deux cas le BD-rate se trompe parce que la plage de débit se décale
trop pour que l'ajustement cubique reste une interpolation. **Un BD-rate n'est
lisible que si les courbes se recouvrent** ; `lambda = 16` a été retenu comme la
dernière valeur vérifiée Pareto point par point, pas comme l'optimum d'une
intégrale. Le contrôle brut est désormais partie du protocole.

## Pistes suivantes

- **Le côté distorsion** : quatre cycles consécutifs n'ont touché que le débit.
  Les leviers psycho-visuels du codec — pivot de Turing, tRNA, échange spectral,
  compensation de Weber — n'ont pas été recalibrés depuis que la grammaire
  entropique a autant changé. Un point de fonctionnement optimal pour la v20 ne
  l'est plus forcément pour la v24.

## Cycle 27 — Détection de contours : la variabilité de Y est-elle compressible ?

**Question posée.** La luminance se concentre sur les arêtes plutôt qu'uniformément.
Cette variabilité est-elle exploitable avec des techniques modernes de détection
de contours ?

### 1. L'erreur est bien concentrée sur les arêtes (mesuré)

`scripts/analyze_edge_distribution.py 40 16`, corpus complet, 63,8 M pixels.
Densité d'erreur = (part de l'erreur quadratique) / (part des pixels).

| Sobel source (luma/px) | part pixels | RMS | part SSE | densité |
|---|---:|---:|---:|---:|
| 0-1 | 28,6 % | 0,916 | 6,0 % | **0,21** |
| 4-8 | 13,0 % | 2,151 | 15,0 % | 1,16 |
| 16-32 | 6,8 % | 3,246 | 17,8 % | 2,63 |
| >64 | 2,4 % | 4,207 | 10,5 % | **4,42** |

Monotone sur toute la plage, facteur 21 entre les extrêmes. 13,4 % des pixels
(gradient > 16) portent 42,7 % de l'erreur quadratique.

### 2. AUREA détecte déjà les contours — mais sur la mauvaise grandeur

`turing::compute_turing_field` applique Sobel **à la grille DC**, une valeur par
cellule 8×8. Ce n'est pas le contenu de contour d'un bloc : c'est l'écart d'un
bloc à ses *voisins*. Un bloc de texture fine a une moyenne identique à ses
voisins et se lit comme plat.

`scripts/analyze_edge_blindspot.py 40 16`, 248 880 blocs. Corrélation de rang
avec l'erreur quadratique par bloc :

| signal | ρ |
|---|---:|
| Sobel sur grille DC (ce que le codec voit) | **+0,389** |
| activité intra-bloc (ce qu'il manque) | **+0,887** |
| corrélation entre les deux | +0,579 |

Densité d'erreur croisée (lire *en ligne* : même gradient DC, contenu réel différent) :

| grad DC \ activité | 0-2 | 2-4 | 4-8 | 8-16 | 16-32 | >32 | amplitude ligne |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-1 | 0,15 | 0,87 | 1,26 | 1,95 | · | · | **13,0x** |
| 2-4 | 0,19 | 0,75 | 1,26 | 1,86 | 2,94 | 3,98 | **20,6x** |
| 8-16 | 0,18 | 0,49 | 1,08 | 1,90 | 2,78 | 3,53 | **19,2x** |
| >16 | 0,19 | 0,36 | 0,76 | 1,56 | 2,44 | 3,13 | **16,5x** |
| **amplitude colonne** | 1,3x | 2,5x | 1,7x | 1,3x | 1,2x | 1,3x | |

À activité fixée, le gradient DC n'explique plus qu'un facteur 1,2-2,5 — et le
signe s'inverse dans les colonnes basses. Le signal utilisé par le codec est
presque redondant une fois l'activité connue ; l'inverse est faux.

Ce signal pilote **deux** choses : le pas de quantification
(`local_step = ac_step * foveal_factor * turing_mod * ...`) et, via
`turing_bucket`, un axe de contexte entropique
(`N_CTX_ZERO_V12 = run_class(4) × prev_nz(2) × energy_bucket(4) × turing_bucket(4)`).

### 3. Levier testé et rejeté : lambda du treillis modulé par l'activité

Implémenté `AUREA_LAMBDA_ACTIVITY` : `lambda *= (activité / moyenne_géom)^k`,
côté encodeur seul (le lambda n'est pas miroité par le décodeur, donc aucune
contrainte de symétrie zéro-bit et aucun changement de format).

| k | RGB BD% | Y BD% | Pareto |
|---:|---:|---:|---|
| -0,30 | +0,055 | +0,007 | 1/16 dominants, 3/16 dominés |
| -0,10 | +0,016 | -0,013 | 3/16 dominants, 1/16 dominés |
| +0,10 | -0,020 | +0,005 | 2/16 dominants, 1/16 dominés |
| +0,30 | -0,047 | +0,065 | **0/16 dominants**, 2/16 dominés |

Nul. Les points bruts montrent un simple glissement le long de la courbe RD
existante (-1,5 % de débit pour -0,12 dB à q20). **Levier retiré du code.**

**Pourquoi c'est nul, et c'est le résultat utile** : le treillis RDO mesure déjà
le débit et la distorsion *réels* de chaque coefficient du bloc. L'activité est
dérivée de ces mêmes coefficients — elle n'apporte aucune information que le RDO
n'ait déjà. Moduler lambda ne fait que déplacer un point d'opération déjà
optimal au sens MSE.

Corollaire : la surface exploitable n'est pas là où elle en a l'air. Elle est là
où le signal serait *nouveau* — le pas de quantification et le contexte
entropique, tous deux pilotés par un signal à ρ=0,39.

### 4. Correction : la plage DC n'était pas bit-identique des deux côtés

L'en-tête transporte `ll_ranges` en **f32** ; l'encodeur déquantifiait le DC avec
les **f64** d'origine (`dc_ranges_raw`), le décodeur avec les f32 arrondis. Les
deux côtés construisaient donc des grilles DC divergeant de ~1e-7 relatif — et
tous les champs zéro-bit reposent sur cette grille.

Le champ de Turing y survivait : Sobel annule le décalage, `normalize_01` annule
l'échelle, la transformation affine se simplifie exactement. Les consommateurs
d'une magnitude DC *absolue*, non : le clamp de luminosité dans
`bright_compensation_step_modulations` et le `round()` sur 8 niveaux de
`spectral_exchange_local_levels` peuvent basculer de l'autre côté d'une frontière
de décision pour un bloc à la limite.

Corrigé dans `normalize_dc` : arrondi de min/max en f32 à la source, plus un
clamp [0,255] sur la sortie normalisée. Les deux côtés partent maintenant de
nombres bit-identiques. Coût mesuré : débit max 0,0001 %, qualité +0,00003 dB —
un no-op, ce qu'une correction de correction doit être. 167 tests verts.

### 5. AUR2 v25 — les contextes AC changent de clé (implémenté)

Le `turing_bucket` est un axe de contexte entropique
(`N_CTX_ZERO_V12 = run_class(4) × prev_nz(2) × energy_bucket(4) × turing_bucket(4)`).
Il pilotait donc aussi le codeur, avec ρ=0,39.

**Le signal de remplacement était déjà dans le flux.** La position EOB *est*
l'activité intra-bloc — c'est le nombre de coefficients que le bloc porte — et
le décodeur lit tous les EOB de tous les canaux **avant** le moindre
coefficient AC (`lib.rs`, boucle `eob_by_channel`). Aucun bit supplémentaire,
aucune circularité, symétrie parfaite.

Les seuils sont les **quartiles de la distribution EOB de l'image elle-même**,
normalisés par la capacité du bloc (`eob·4096/(taille²−1)`) pour que les blocs
8/16/32 soient comparables. Les deux côtés disposent du même tableau EOB, donc
les mêmes coupures. Pas de constante magique à recalibrer par qualité.

Le flux EOB, lui, garde le bucket de Turing : un symbole ne peut pas être son
propre contexte.

### 6. L'IIR d'énergie franchissait les frontières de blocs

En posant le point 5, un second défaut est apparu. `energy` est un filtre IIR
qui court sur la séquence plate de coefficients, frontières de blocs comprises.
Or les blocs sont tronqués à leur EOB : le dernier symbole d'un bloc est son
dernier coefficient **non nul**, typiquement de magnitude 1. Le premier
coefficient du bloc suivant est une basse fréquence, typiquement le plus grand
du bloc. Le modèle prédisait donc l'échantillon le plus fort d'un bloc à partir
du plus faible de son prédécesseur.

Trois variantes mesurées (sentinelles, total octets sur 4×4 points) :

| entrée de bloc | octets | vs report |
|---|---:|---:|
| report de l'IIR (v24) | 11 864 959 | — |
| remise à zéro | 11 865 931 | +0,008 % |
| amorçage gradué par l'activité | 11 841 790 | −0,195 % |
| **amorçage constant à 176** | **11 829 583** | **−0,297 %** |

La remise à zéro perd : ce n'est pas que le report soit trop fort, c'est qu'il
n'est **pas informatif**. Et le gradué perd contre le constant : l'effet n'est
pas « quelle est la densité de ce bloc » mais « les premiers coefficients de
*tout* bloc sont ses plus forts ». Le plateau est large (160-240 à ±0,02 %) ;
176 est juste au-dessus du seuil 160 du bucket 3, avec marge.

### 7. Résultat v25 — corpus complet, 48 points

Seul le *codage* des coefficients change, jamais leurs valeurs. La
reconstruction revient donc **bit-identique**, ce qui est vérifié à chaque
point par `scripts/measure_activity_context.py --full` :

| | |
|---|---:|
| débit moyen | **−0,430 %** |
| points de débit améliorés | **48/48** |
| images améliorées | **12/12** |
| reconstruction identique | **48/48** |
| meilleure image (`peace`) | −0,759 % |
| moins bonne (`mind_your_steps`) | −0,142 % |

Aucun compromis : le débit baisse à *tous* les points d'opération et la
distorsion ne bouge d'aucun bit. Le BD-rate vaut donc exactement −0,430 %, sans
interpolation — c'est le seul cas où le piège du cycle 26 ne peut pas se poser.

Un fichier v24 écrit par le binaire du cycle 26 se décode toujours à
l'identique avec le nouveau binaire (vérifié) ; v25 est en revanche la première
version qui n'est pas l'état neutre de la précédente, donc elle a sa propre
branche de lecture.

Binaire : `target/benchmark_v27/aurea_v27_activity_context.exe`,
SHA-256 `FC0A562C754F19499057B4DFFBA9CCB5D60B08AA6FC864270248E1072C1C43AF`.

### 8. Le treillis à deux passes — négatif net, et il explique pourquoi

Le point ouvert du §7 était que le treillis estime le débit avec le bucket de
Turing alors que le codeur utilise désormais l'activité. Implémenté, mesuré,
retiré. Deux découvertes, dans cet ordre.

**a) Le bucket est inerte dans le treillis, par construction.** Première
tentative : deuxième passe amorcée par les EOB de la première, pour donner au
treillis les vrais buckets. Résultat **strictement identique au bit près**, pour
+36 % de temps d'encodage. La raison est structurelle : les tables du treillis
repartent **plates** à chaque bloc (`p_zero = [P_ZERO_INIT; 128]`), le
`turing_bucket` est **constant sur tout le bloc**, et seule la tranche qu'il
sélectionne est jamais touchée. Les 128 emplacements démarrant tous à la même
valeur, changer le bucket ne peut pas modifier une seule décision. Le paramètre
était mort depuis toujours.

**b) Amorcer les tables est pire, monotonement.** Le vrai écart n'était donc pas
le bucket mais le prior : le codeur réel porte ses tables adaptées sur tout le
canal, le treillis les remet à plat à chaque bloc. Deuxième tentative :
apprendre les tables convergées sur la sortie de la première passe et les
réinjecter, avec un poids de mélange vers le plat.

| poids vers les tables apprises | RGB BD% | Y BD% | Pareto |
|---:|---:|---:|---|
| 0,25 | **+0,270** | +0,167 | 0/16 dominants, 6/16 dominés |
| 0,50 | **+0,558** | +0,342 | 0/16 dominants, 6/16 dominés |
| 1,00 | **+1,147** | +0,614 | 0/16 dominants, 6/16 dominés |

Monotone dans le poids, aucun point dominant à aucun réglage : le prior plat est
**meilleur** que les tables convergées du codeur. Deux mécanismes, tous deux
propres au fait que le treillis optimise bloc par bloc :

1. Les tables apprises viennent des décisions de la passe précédente. Les
   réinjecter fait optimiser le treillis contre sa propre ombre — les contextes
   que la passe 1 a remplis de zéros voient `P(zéro)→0,99` et se figent.
2. Un optimiseur par bloc ne peut pas tenir compte du fait que ses choix
   déplacent les tables partagées de *tous* les blocs suivants. Le prior plat
   est une régularisation contre cet effet.

Le prior plat n'est donc pas une approximation qu'on tolère pour le
parallélisme : c'est le bon choix, et le parallélisme est un bonus. Constaté
dans `trellis.rs` pour qu'on ne le redérive pas.

### 9. Perspective RADAR — l'écho du ping (AUR2 v26)

Idée posée : distribuer les bits comme un radar, l'écho étant le retour d'un
ping. Elle se réalise littéralement, parce qu'AUREA est progressif.

**Le ping existe déjà.** La couche 0 porte les `PROGRESSIVE_SKELETON_COEFFS = 5`
premiers coefficients de **tous** les blocs et est entièrement décodée avant la
moindre couche de détail. Or chaque couche ouvre son **propre flux rANS avec des
tables plates** : les coefficients de détail d'un bloc étaient codés sans rien
savoir des basses fréquences déjà livrées *pour ce même bloc*. Une information
que les deux côtés possèdent, gratuite, et que personne n'écoutait.

`rans::skeleton_echo` rejoue l'IIR d'énergie du codeur sur le squelette décodé :
« à quel volume ce bloc est-il revenu ». La couche de détail ouvre sur cette
valeur au lieu de la constante plate du §6. Zéro bit — le ping est déjà payé.

| corpus complet, 48 points | |
|---|---:|
| débit moyen | **−0,023 %** |
| images améliorées | **12/12** |
| points améliorés | 44/48 |
| reconstruction identique | 48/48 |

Faible mais strictement du même signe sur les douze images — un effet de bruit
changerait de signe. Purement entropique, donc débit pur.

**Deux variantes plus ambitieuses, mesurées et rejetées :**

- *L'écho comme axe de contexte.* Replier l'écho dans l'axe à 4 niveaux aux
  côtés de l'EOB (deux niveaux chacun au lieu de quatre pour l'EOB) donne
  **+0,007 %** : la résolution perdue sur l'EOB coûte plus que l'écho ne
  rapporte. L'axe vaut mieux entièrement dépensé sur le comptage.
- *L'écho comme moyenne directe* plutôt que continuation de l'IIR : 11 827 660
  contre 11 827 431 octets. Légèrement moins bon — la continuation de l'IIR est
  la mesure honnête de l'état que le codeur aurait atteint.

### 10. Le canal ionisé (AUR2 v27)

Idée posée : le « chemin de moindre résistance » des éclairs. La lecture qui
porte n'est pas la trajectoire, c'est le **coup de retour**. Un éclair ne
ré-ionise pas l'air : les arcs suivants réempruntent le canal que le premier a
ouvert, et c'est précisément pourquoi ils sont plus rapides et plus intenses.

Le codeur d'AUREA faisait l'inverse. Chaque couche progressive ouvrait un flux
rANS neuf avec des tables plates — elle claquait dans de l'air vierge — alors
que la couche 0 venait d'adapter ses tables **sur la même image**. Les couches
de détail repartent maintenant des tables convergées de la couche 0.

Déterministe et sûr à la troncature : la couche 0 est toujours présente. Et
contrairement à l'amorçage du treillis du §8, ce n'est **pas** auto-référentiel
— le squelette est une donnée réellement transmise, pas la sortie d'une passe
précédente du même optimiseur. C'est exactement la distinction qui avait fait
échouer le §8, et elle joue ici en sens inverse.

| corpus complet, 48 points | |
|---|---:|
| débit moyen | **−0,210 %** |
| images améliorées | **12/12** |
| points améliorés | 47/48 |
| reconstruction identique | 48/48 |

Meilleure image `peace` −0,284 %, moins bonne `fine_ally` −0,153 %. Purement
entropique, et aucun surcoût d'encodage mesurable.

**Ce qui est porté, et ce qui compte dedans.** Toute la pile adaptative migre :
les deux grandes tables de probabilité, les poids du mélangeur logistique, ses
modèles secondaires, l'APM et les compteurs de phase. Mais les tables portent
presque tout — s'en tenir à elles donne déjà **−0,193 %**, donc la pile de
mélange ne vaut que les 0,017 pp restants. C'est cohérent : le mélangeur
converge vite et a peu de poids, alors que `p_zero` compte 4096 contextes qui
demandent un long flux pour se stabiliser. Le gain est là où la convergence est
lente.

Chaque paquet repart d'un **clone** de l'état de la couche 0, jamais de l'état
du paquet précédent : les paquets restent indépendants, donc la troncature
progressive garde exactement sa sémantique.

Binaire : `target/benchmark_v29/aurea_v27_channel_memory.exe`,
SHA-256 `06D89A13F78C46459D43F0C3D5BA667EEF8656C3380E15BE9EE40DA2E1B597D0`.

### Bilan du cycle 27

| étape | débit | portée |
|---|---:|---|
| v25 — contextes AC sur l'activité intra-bloc | **−0,430 %** | 48/48 |
| v26 — amorçage par l'écho du squelette | **−0,023 %** | 44/48 |
| v27 — canal ionisé porté entre couches | **−0,210 %** | 47/48 |
| **cumul** | **−0,663 %** | distorsion strictement inchangée |

Plus la correction de la plage DC f32/f64 (no-op mesuré, aléa de symétrie
retiré), et cinq négatifs dont quatre expliqués structurellement : modulation de
lambda, treillis à deux passes (deux variantes), écho comme axe de contexte,
écho en moyenne directe.

## Cycle 28 — Trames carrées, phase de grille, et le MIS du path tracing

### 1. `kryte` est bien une image de trames — mais ce n'est pas ce qui le distingue

Observation à l'œil : les verrières de `kryte` forment un réseau de meneaux dense
sur tout le fond, plus des caisses et un établi. Structure périodique, axée sur
les axes.

Quantifié (part du gradient alignée sur les axes, 1,0 = purement H/V) :

| image | alignement | | image | alignement |
|---|---:|---|---|---:|
| honey | **0,3644** | | kryte | **0,3509** |
| mind_your_steps | 0,3616 | | fine_ally | 0,3253 |
| legacy | 0,3542 | | agree | 0,2711 |

`kryte` est dans le tiers haut mais **n'est pas une valeur aberrante** : `honey`
et `mind_your_steps` sont au-dessus. L'intuition visuelle est juste, la
statistique ne fait pas de `kryte` un cas à part.

### 2. La phase de la grille coûte — mais autant sur une photo

Le LOT d'AUREA a des frontières **dures** : `FLAG_LOT_OVERLAP` n'est jamais posé,
rien ne se recouvre. La transformée est donc sensible à la translation.

Protocole : contenu **identique**, placé à l'offset (dx, dy) dans un canevas
fixe, marges remplies par réplication du bord — aucune arête nouvelle, seule la
phase bouge. Un premier essai par recadrage successif était faussé : il faisait
entrer et sortir du contenu aux bords, et produisait une dérive monotone qui
n'avait rien d'un effet de grille.

| image | axe | amplitude best↔worst | variance à période 8 |
|---|---|---:|---:|
| kryte | dx | 0,713 % | 5,1 % |
| kryte | dy | 0,602 % | 10,1 % |
| peace | dx | **0,850 %** | 0,5 % |
| peace | dy | 0,651 % | 14,6 % |

Deux conclusions, dont une négative :

- **L'hypothèse d'alignement ne tient pas pour `kryte`.** `peace`, une photo sans
  trame, est *plus* sensible à la phase. Ce n'est pas ce qui explique l'anomalie
  `kryte`.
- **Mais la sensibilité à la phase est réelle et grosse** : 0,6 à 0,85 % entre le
  meilleur et le pire décalage d'un pixel — plus que tout le gain de la v27
  (0,21 %). Et la part à période 8 (l'harmonique réelle de la grille DC) est
  faible : ce n'est pas une résonance avec un motif, c'est la sensibilité
  générique d'une transformée à blocs durs.

**C'est le prix du recouvrement désactivé.** Une transformée *lappée* est
tolérante à la translation par construction — ses fonctions de base débordent les
frontières. AUREA s'appelle « Lapped Orthogonal Transform » et ne lappe pas :
`sine_window`, `apply_window_2d`, `lot_synthesize_block` sont du code mort. Cette
mesure chiffre ce que coûte cet abandon.

### 3. Le MIS du path tracing — testé, rejeté, et il explique la v27

Un path tracer combine plusieurs estimateurs en pondérant chacun par **sa propre
confiance sur l'échantillon tiré** : la stratégie qui sait quelque chose domine
là où elle sait. Le mélangeur logistique d'AUREA ne fait pas ça — ses poids sont
par sélecteur (témoin × amplitude), identiques que le modèle primaire dise
« 0,50, aucune idée » ou « 0,99, certain ».

Transfert testé : découper le sélecteur par l'engagement du modèle primaire.

| découpage | débit |
|---|---:|
| aucun (v27) | référence |
| 2 niveaux | **+0,041 %** |
| 4 niveaux | **+0,114 %** |

Monotone : tout conditionnement perd. Et la raison boucle avec le §10 du cycle
27, vue de l'autre côté. **Le mélangeur est affamé de données, pas
d'information** : peu de poids, convergence rapide, donc diviser par quatre ce
que voit chaque vecteur coûte plus que le conditionnement ne rapporte. Les
grandes tables de contexte sont le régime inverse — 4096 emplacements,
convergence lente — et c'est exactement pourquoi *les porter* entre couches a
payé alors que *découper* le mélangeur ne paye pas.

Règle qui sort des deux mesures : **conditionner davantage ne vaut que si le
modèle voit déjà assez de chaque cas ; sinon il faut au contraire lui porter du
contexte.** Consigné dans `rans.rs` à l'endroit du sélecteur.

### Pistes ouvertes

- **Ordre de parcours par moindre résistance.** L'autre lecture de l'éclair :
  trier les blocs *à l'intérieur* d'un paquet par résistance croissante plutôt
  qu'en ordre raster. La carte est gratuite (EOB et écho sont connus avant tout
  détail) et le garde-fou de causalité du prédicteur de phase est déjà fondé sur
  le rang, donc il suivrait automatiquement. Non mesuré ; l'axe d'activité étant
  déjà un contexte, le gain attendu est modeste.
- **Le pas de quantification** reste piloté par ρ=0,39 (cf. §2).
- **Redondance `energy_bucket` / `turing_bucket`** : les deux mesurent désormais
  l'activité. Un axe libéré vaut cher.
- **Le treillis reste aveugle aux contextes** (§8a), et le §8b suggère que le
  rendre sensible ne rapporterait pas.

## Cycle 29 — P0 clos : le recouvrement ne répare pas la phase

Le chantier P0 posait la question en trois lots. M0 a chiffré le défaut à
distorsion appariée : **1,964 % médian** (max 3,125 %) de débit entre le
meilleur et le pire décalage d'un pixel. M1 (toutes cellules forcées à 16,
même grammaire) a innocenté l'adaptation de taille : médiane 1,948 %, écart
-0,8 %. M2 ne pouvait pas être évalué comme changement de codec : le chemin
`use_overlap` fenêtre une DCT 16×16, avance de 8, et **garde les 256
coefficients** — une trame suréchantillonnée 4×, pas une transformée lappée
orthogonale (172 800 blocs contre 43 200 sur peace, débit 5,45×).

La question restante — le lapping supprime-t-il la dépendance à la phase ? —
a été tranchée sur la transformée seule (`scripts/p0_lapping_study.py`,
2026-08-16) : entropie d'ordre 0 des coefficients quantifiés à Y-PSNR apparié
38 dB, 16 offsets d'un pixel, luma 1024².

| transformée | kryte : débit / étalement | peace : débit / étalement |
|---|---:|---:|
| A critique 16×16 (M1) | réf. / 3,818 % | réf. / 2,482 % |
| B trame redondante (= `use_overlap`) | +223,4 % / 3,231 % | +472,7 % / 2,060 % |
| C préfiltre papillon critique (esprit JPEG-XR) | +15,8 % / 3,800 % | +21,3 % / 2,582 % |

**Verdict : négatif, chantier clos.** Aucune variante lappée ne réduit
matériellement l'étalement de phase — le préfiltre l'*augmente* même sur
peace — et les deux coûtent du débit là où l'échantillonnage critique est
gratuit. La sensibilité à la translation n'est donc pas une propriété des
frontières dures : c'est la quantification elle-même qui projette
différemment un contour translaté, fenêtré ou non. Une vraie LOT
orthogonale (post-filtre inversible plus soigné que le papillon) reste
théoriquement possible, mais le plafond mesurable est ~2 % et les deux
proxys disponibles partent de +16 à +473 % : le rapport risque/gain condamne
la piste. Les topologies `Fixed16Lapped50` / `Adaptive16x32Lapped50`
restent des primitives de laboratoire ; le profil actif demeure critique.

Découverte annexe du même examen : `classify_blocks` initialise la grille à
16 et ne sait que fusionner vers 32 — **le bloc 8×8 est inatteignable**
alors que la grammaire entière le supporte (carte 2 bits, LUT, treillis).
Vérifié en vif sur kryte q60 : 16 732 blocs de 16, 41 de 32, zéro de 8.
À trancher : implémenter le split descendant et le mesurer, ou retirer la
branche morte.

L'effort est redirigé vers les défaites restantes du corpus face au JPEG
optimisé (v27) : kryte vs 4:4:4 (Cb +15,77 %, Cr +16,98 %, ΔE moyen
+7,51 %), dispair vs 4:2:0 (Y +0,80 %) et mind_your_steps vs 4:4:4
(ΔE p95 +1,67 %). La plaie kryte se décompose avec les mesures des cycles
14 et 17 : ~3 points d'échange spectral, ~10 points de pivot luma uniforme
(1,75) qui gonfle le fichier contre un PSNR chroma stagnant. Le cycle
suivant introduit `AUREA_LUMA_PIVOT_RELIEF` : force luma effective
`(1,75 - relief × R)` plafonnée à 1,0, où `R` est la signature diffuse déjà
zéro-bit (0,9418 sur kryte, exactement 0 sur les onze autres).

## Cycle 30 — Relief du pivot luma sous porte sombre (laboratoire)

Le pivot luma 1,75 du cycle 17 est uniforme : calibré pour gagner la
frontière Y *moyenne*, il sur-livre sur kryte une luminance déjà largement
gagnée (Y -19,8 % contre le 4:4:4) pendant que Cb/Cr/ΔE paient la facture.
Le mécanisme du cycle rend ce pivot adaptatif, avec deux signaux zéro-bit
reconstruits des grilles DC déjà décodées :

`force_luma = max(1,75 - relief × R × G, 1,0)`

- `R` est la signature chroma diffuse existante (`chroma_diffusion_signature`),
- `G` est une **porte sombre** nouvelle : smoothstep(0,30, 0,45) de la
  fraction de cellules DC luma sous l'image PTF de Y=40 linéaire — la bande
  d'obscurité où le diagnostic kryte a mesuré la sur-livraison Weber
  systématique (ratio MSE 0,83). Le relief ne s'applique que là où le
  surplus sombre le finance.

La porte est née d'un contre-exemple de validation : kodim21 (phare, plein
jour) déclenche la signature diffuse (R=0,781) et perdait +2,29 % RGB /
+5,76 % Y sous relief nu. Sa fraction sombre vaut 0,0098 contre 0,4934 pour
kryte : la porte les sépare par un facteur 50, loin du genou. Les
calibrations : sweep du relief sur kryte seule (0,25/0,5/0,75/1,0, gain
chroma/ΔE monotone, RGB stable, plateau au plancher v17 dès 0,75) ; genou
de la porte posé d'après le diagnostic d'obscurité du cycle 11, vérifié
après coup sur kodim21/kryte.

Validation (48 points, ancres JPEG du rapport v27 réutilisées à
l'identique) : **44/48 points octet-identiques à v27** — les onze images à
R×G=0 ne bougent pas d'un bit ; Kodak 24 entière est à force 1,75 inchangée
(23 à R=0, kodim21 à G=0) ; sur 40 images div2k_valid, la conjonction R>0
et G>0 ne se déclenche **jamais** (R seul 6 fois, G seul 4 fois). Le
mécanisme est exactement aussi étroit que sa cible.

Sur kryte, à relief 1,0 (force effective au plancher 1,0) :

| BD-rate kryte | vs JPEG 4:2:0 | vs JPEG 4:4:4 | vs v27 |
|---|---:|---:|---:|
| RGB | -10,33 % | -15,64 % | +0,11 % |
| Y | -1,95 % | -17,31 % | +3,11 % |
| Cb | -24,77 % | **+1,52 %** | -12,18 % |
| Cr | -24,30 % | **+2,58 %** | -12,19 % |
| SSIM-Y | -1,35 % | -17,06 % | +2,24 % |
| ΔE moyen | -13,98 % | **-2,05 %** | -6,98 % |
| ΔE p95 | -30,56 % | -19,62 % | -8,10 % |

Le ΔE moyen contre le 4:4:4 passe de +7,51 % à **-2,05 %** : kryte gagne
désormais toutes les métriques perceptuelles (RGB, Y, SSIM-Y, ΔE moyen et
p95) contre les deux profils JPEG. Les seuls résidus positifs du corpus
entier sont Cb +1,52 % / Cr +2,58 % sur kryte contre le 4:4:4 — des PSNR
chroma à ~50 dB sur un canal quasi plat — plus les deux défaites marginales
préexistantes (dispair Y +0,80 % vs 4:2:0, mind_your_steps ΔE p95 +1,67 %
vs 4:4:4), non touchées par ce cycle. Le prix est payé dans le surplus :
Y kryte +3,11 % contre v27, tout en restant gagnant -1,95 % contre le
4:2:0.

### Second acte — relief de la pente chroma, et promotion AUR2 v28

Le relief luma seul laissait Cb +1,52 % / Cr +2,58 % contre le 4:4:4. Le
second levier rend le don là où il a été pris : sur le même champ `R × G`,
la pente d'échange chroma est multipliée par `1 - k × R × G`, et le delta
compensateur suit la pente *relevée* — l'échange reste log-neutre à force
luma 1. Le sweep de k sur kryte a d'abord montré la vraie tension : k=0,25
ferme Cb/Cr (-3,4/-2,3) mais fait basculer SSIM-Y contre le 4:2:0 (+0,38 %).
La sonde fine a localisé la fenêtre où *tout* est négatif en cubique :
k ∈ [0,13 ; 0,20], point minimax **k = 0,15** (pire marge : Cr -0,32 %).

**Promotion AUR2 v28.** Les défauts deviennent relief luma 1,0 et relief
chroma 0,15 ; l'encodeur marque v28 dès qu'un relief est actif et le
décodeur applique les reliefs d'après la version seule — un fichier v27 ou
antérieur décode toujours avec les deux à zéro (vérifié : PNG identiques au
binaire v27 gelé). Les témoins `AUREA_LUMA_PIVOT_RELIEF=0` +
`AUREA_SPECTRAL_CHROMA_RELIEF=0` émettent un flux v27 bit-exact (SHA
vérifié). Les 167 tests passent. Validation corpus : **11 images sur 12
strictement inchangées** (corps identique hors octet de version, PNG
identiques), Kodak 24 inchangée (kodim21 : corps identique), div2k 40
images : le champ ne se déclenche jamais.

kryte final (v28, cubique / contrôle linéaire) :

| BD-rate kryte | vs JPEG 4:2:0 | vs JPEG 4:4:4 |
|---|---:|---:|
| RGB | -10,23 / -10,91 % | -15,51 / -17,27 % |
| Y | -1,13 / -4,10 % | -16,51 / -20,19 % |
| Cb | -26,99 / -30,27 % | -1,45 / **+1,97** % |
| Cr | -26,47 / -29,36 % | -0,32 / **+3,65** % |
| SSIM-Y | -0,35 / **+6,51** % | -16,12 / -9,81 % |
| ΔE moyen | -15,56 / -18,56 % | -3,87 / -2,46 % |
| ΔE p95 | -31,74 / -30,00 % | -21,01 / -14,97 % |

Lecture honnête, conforme au piège de méthode du cycle 26 : **douze axes
sur quatorze sont gagnés sous les deux interpolations** — dont RGB, Y et
les deux ΔE partout. Cb/Cr contre le 4:4:4 et SSIM-Y contre le 4:2:0 sont
*indécis* (signes opposés selon l'interpolation) : avant ce cycle, Cb/Cr y
étaient perdus sous les **deux** (+15,8/+17,0 cubique, +19,2/+21,2
linéaire) ; ils sont désormais à zéro à la résolution de la mesure près.
L'analyse des gradients montre que la fenêtre « tout robuste » est vide —
SSIM-420 cubique bascule à k≈0,20, Cr-444 linéaire à k≈0,33 — donc aucune
revendication ne doit reposer sur ces trois axes, dans un sens comme dans
l'autre. Les défaites restantes du corpus sont inchangées : dispair Y
+0,80 % vs 4:2:0 et mind_your_steps ΔE p95 +1,67 % vs 4:4:4.

Binaire archivé : `target/benchmark_v30/aurea_v28_diffuse_relief.exe`,
SHA-256 `14C311DF77A1AC9BA3D2CF7055E3D4DB86AC621B315F8589DC962A39E283ABBD`.
Scripts : `sweep_luma_pivot_relief.py` (`--var` pour balayer l'un ou
l'autre relief), `validate_luma_pivot_relief.py`, `ab_kodim21_relief.py` ;
détail par image dans
`target/luma_pivot_relief_lab/validation_relief_1.0.csv`.

## Cycle 31 — Loi de texture perceptuelle (AUR2 v29) : la vitrine à 24/24

Objectif fixé : gagner LPIPS **et** DISTS sur les 24 paires à débit apparié
de la vitrine publique (état v28 : LPIPS 23/24, DISTS 21/24 ; défaites
réelles kryte q20 DISTS −8 %, phoenix q20 DISTS −22 %, kryte q75 LPIPS
−3 % ; plus deux « égalités » qui étaient un artefact du CSV à 4 décimales,
porté à 6). Outil : `scripts/perceptual_ab.py`, le protocole exact de la
vitrine (appariement JPEG par taille) en pleine précision.

### Le diagnostic qui a tout ouvert

Le déficit DISTS de phoenix est **localisé et luma** : quadrants bas (plage
de galets noirs) perdus, quadrants hauts gagnés, gris seul 0,0252 vs 0,0216.
La texture fine sombre est doublement affamée par construction : le
contre-don de Weber prélève sur les blocs sombres, et le champ de Turing
(ρ=0,39, aveugle à l'activité intra-bloc) lit les galets comme plats.

### Le piège de l'ancre, mesuré

À débit apparié, l'ancre JPEG suit la taille du fichier AUREA. Couper le
contre-don améliore le DISTS d'AUREA de 6 % mais gonfle le fichier de 10 % :
l'ancre saute de q56 à q61 et gagne plus vite (0,01836 → 0,01470). La pente
DISTS/débit de JPEG est plus raide que la nôtre dans cette zone : **tout
levier qui grossit le fichier est perdant ; seuls comptent les gains à
taille constante.** La renormalisation du pas (`AUREA_STEP_SCALE`,
encodeur-seul, le pas étant porté par l'en-tête) le confirme : le gain de
redistribution ne survit pas au retour à taille égale.

### Les leviers retenus (calibrés sur le harnais, tous à taille ~constante)

- **λ treillis 8** à basse qualité (au lieu de 16, Pareto-MSE du cycle 26) :
  le zérotage MSE-optimal tue la texture que LPIPS/DISTS mesurent.
- **λ × activité, k = +1,0** (`(activité/moy_géom)^k`, encodeur-seul) — le
  levier *retiré au cycle 27 comme MSE-neutre* : à k positif le treillis
  cesse de payer pour le chaos irreproductible, le fichier maigrit, l'ancre
  recule — phoenix passe de −6 % à **+8 % de marge DISTS**. Le signe négatif
  (garder plus de texture) est monotone perdant. Poids C2 1,75.
- **Post-chaîne de restauration** (décodeur, q ≤ 30) : XSharpen 0,4 /
  seuil 0,025 / **rayon 2** (le rayon 1 est effacé par le Lanczos de la
  grille métrique 1024 px) + unsharp mask 0,15 à blur double.
- **Anti-ring coupé à q ≥ 70** : à haute qualité le filtre sigma coûte
  LPIPS, DISTS **et** PSNR à la fois (kryte q75 : +0,11 dB en le coupant).
- **Plancher de contraste local + coring USM** : la loi uniforme a d'abord
  fait basculer `chilling` (image de dégradés lisses, LPIPS 0,0012 → 0,0051 :
  l'affûtage fabrique du banding quand les extrema locaux sont proches). Le
  plancher (portée 5×5 < 0,04 → intact) et le coring (|δ| ≤ 0,75 → intact)
  réparent chilling sans gate par image, en épargnant les dégradés partout.

### Négatifs documentés (tous mesurés sur le harnais)

CAS (−2 dB, tout empire) ; grain synthétique blanc, même gaté par
l'obscurité (le bruit n'est pas de la texture : LPIPS s'effondre) ; offset
de déquantification ±0,15 bin (catastrophique aux pas HF de q20 : −8 dB) ;
XSharpen bi-échelle ; USM à 3 passes ; λ×activité négatif ; affûtage chroma
(neutre à légèrement négatif).

### Résultat, et son prix

La vitrine (`D:/x267/docs`) régénérée intégralement au binaire v29 :
**24/24 LPIPS, 24/24 DISTS, 24/24 PSNR** (moyenne +1,86 dB), y compris
chilling (revenu à +0,13 dB au-dessus de son état v28). Le prix est payé sur
la frontière BD-MSE, et il est affiché honnêtement : RGB vs 4:2:0 passe de
−38,63 % à **−36,47 %** (toujours 12/12), Y vs 4:2:0 de −8,22 % (11/12) à
**−7,36 % (9/12)**, SSIM-Y 4:2:0 11/12 ; tout le reste demeure 12/12. La loi
n'agit qu'aux extrêmes de qualité (q ≤ 30 et q ≥ 70) ; v27/v28 se décodent
bit-identiques sous leur propre version (vérifié), le témoin double-zéro
émet toujours un flux v27 bit-exact, 167 tests verts.

Binaire archivé : `target/benchmark_v30/aurea_v29_texture_law.exe`,
SHA-256 `D7ED68666C86EBA3F9A50408E2C0CBE2A58A2F3DAA32307101ACE6E93A0CF1FE`.
Scripts : `perceptual_ab.py` (harnais), `prepare_showcase_v29.py`
(régénération complète). Constantes : `calibration.rs`, bloc « AUR2 v29 ».

### Pistes rouvertes par ce cycle

- Le **pas piloté par l'EOB** (ρ=0,887) reste le levier structurel pour
  réconcilier la frontière MSE et la texture — il rendrait une partie des
  points Y perdus sans sacrifier le 24/24.
- La loi q ≤ 30 est binaire ; un fondu continu en qualité éviterait la
  discontinuité à q31.
- Le 8×8 inatteignable demeure.

## Cycle 32 — Le pas piloté par l'EOB (AUR2 v30) : Y 12/12

La piste ouverte au cycle 27 est enfin réalisée : le pas de quantification
luma cesse d'être piloté par le seul champ DC-Sobel (ρ=0,39 avec l'erreur
quadratique) et reçoit un champ zéro-bit dérivé des **positions EOB
transmises** (ρ=0,887). L'activité normalisée `a_b = eob_b / capacité_b` est
recentrée aux poids AC et chaque bloc reçoit `phi^(-eta·(a_b - moyenne))` :
les blocs qui portent l'erreur reçoivent un pas plus fin, financé
log-neutralement par les blocs calmes.

**La circularité est résolue par une double passe à EOB forcé.** La passe 1
quantifie aux pas courants et fixe les EOB ; ce sont eux que le flux
transmet. La passe 2 requantifie avec les pas modulés, chaque bloc étant
clampé à son EOB de passe 1 (coefficients au-delà remis à zéro, zéros de
queue conservés en deçà). Le décodeur lit tous les EOB avant le moindre
coefficient AC (l'acquis de la v25) et dérive exactement le même champ :
symétrie parfaite, zéro bit, zéro en-tête. C'est la ligne du cycle 27
appliquée au pas lui-même : « amorcer depuis une donnée transmise marche ».

**Calibration.** Le sweep pleine-qualité (η ∈ 0,25..0,75) a montré un
déplacement de frontière authentique — RGB **et** Y gagnent ensemble,
monotones en η — mais deux tensions : dispair exigeait η ≥ 0,75, et à cette
force les fichiers q20 gonflaient (+5 % sur phoenix), menaçant l'ancre de
la vitrine 24/24, pendant que kryte rendait ses Cb/Cr. La résolution est le
**gate de qualité** : η = 0 pour q ≤ 30 (la loi de texture y possède
l'allocation et l'ancre ne bouge pas), **η = 1,0 au-dessus**. À cette loi,
les trois perdants Y basculent (dispair −1,32 %, kryte −1,80 %, chilling
−3,87 % contre leur base v29) et les points q20 sont octet-identiques.

**Verdict — régénération complète (48 points, 24 paires, binaire unique) :**

| | v27 | v29 | **v30** |
|---|---:|---:|---:|
| Y vs JPEG 4:2:0 | −8,47 % · 11/12 | −7,36 % · 9/12 | **−10,89 % · 12/12** |
| RGB vs 4:2:0 | −38,57 % · 12/12 | −36,47 % · 12/12 | −37,53 % · 12/12 |
| Y vs 4:4:4 | −29,85 % · 12/12 | −28,93 % · 12/12 | **−31,69 % · 12/12** |
| ΔE moyen vs 4:2:0 | −45,07 % · 12/12 | −45,59 % · 12/12 | −44,77 % · 12/12 |
| Vitrine LPIPS/DISTS/PSNR | — | 24/24 ×3 | **24/24 ×3** |

La frontière Y du 4:2:0 — l'adversaire désigné du journal depuis le cycle 9,
où elle valait **+19,955 %** — est morte : douze victoires sur douze, y
compris dispair, perdant sur cet axe depuis la première mesure. La vitrine
conserve son balayage parfait. Le coût : Cb/Cr rendent environ un point
(kryte vs 4:4:4 revient dans sa zone indécise), le débit croît de 1 à 3 %
aux qualités hautes contre plus de dB encore, et l'encodage luma paie une
seconde passe de treillis au-dessus de q30.

167 tests verts ; η = 0 reproduit la v29 au corps près (seul l'octet de
version change) ; conformément à la doctrine de rupture, aucun chemin
d'émission antérieur n'est entretenu au-delà des témoins de laboratoire.
Binaire archivé : `target/benchmark_v30/aurea_v30_eob_step.exe`, SHA-256
`E1FA9BBF13FDD8D4DBB6E615219291DDAF827DE26246BC6E69C90638E74ABAAD`.
Champ : `hierarchy::eob_step_mods` ; constantes `V30_EOB_STEP_ETA = 1,0`,
`V30_EOB_STEP_MIN_QUALITY = 31`.

### Pistes

- **Fondu de η en qualité** (0 → 1 entre q30 et q40) au lieu du seuil dur.
- **Le champ EOB pour la chroma** : C1/C2 ont leurs propres EOB transmis ;
  le même mécanisme pourrait rendre les points Cb/Cr cédés.
- **Une seule passe** : prédire l'EOB de passe 1 depuis l'énergie AC brute
  éviterait la seconde passe de treillis (~+40 % d'encodage luma).
- Le 8×8 inatteignable demeure.

## Cycle 33 — Le champ EOB chroma sous porte gamma (AUR2 v31)

Le champ du cycle 32 est étendu aux deux canaux chroma, chacun piloté par
ses propres EOB transmis (même double passe à EOB forcé, même gate q ≥ 31).
Le sweep a immédiatement montré la fermeture cherchée — kryte Cb/Cr contre
le 4:4:4 passent à −2,09/−0,80 dès η_c = 1,0, avec RGB et ΔE en prime — et
le prix attendu : le débit chroma ajouté rend ~1 point de Y partout, ce qui
suffisait à rebasculer dispair (marge −1,32 → +0,49).

**La porte gamma résout le conflit.** Le champ chroma n'est utile que là où
l'échange spectral a réellement pris le budget chroma : les images à chroma
plate. Le gamma d'échange — déjà calculé des deux côtés depuis les grilles
DC, stable en qualité à ±0,0006 — sépare exactement les deux populations :
kryte 0,682 et honey 0,813 au-dessus, dispair 0,523, phoenix 0,497,
chilling 0,497, agree 0,267 en dessous. `eob_chroma_gate` =
smoothstep(0,55, 0,65, gamma) : les deux camps saturent, aucune image du
corpus dans le genou. dispair (porte 0) est revenu octet pour octet à ses
flux v30 — Y −1,32 restauré — pendant que kryte (porte 1) garde sa
fermeture.

**Verdict — régénération complète (48 points, 24 paires) :**

| axe (cubique) | vs JPEG 4:2:0 | vs JPEG 4:4:4 |
|---|---:|---:|
| RGB | −37,68 % · 12/12 | −31,64 % · 12/12 |
| Y | **−10,59 % · 12/12** | −31,47 % · 12/12 |
| Cb | −61,58 % · 11/11 | **−19,23 % · 12/12** |
| Cr | −67,18 % · 11/11 | **−29,17 % · 12/12** |
| ΔE moyen | −45,16 % · 12/12 | −26,71 % · 12/12 |
| SSIM-Y | −12,03 % · 11/12 | −32,54 % · 12/12 |

La vitrine conserve son balayage **24/24 en LPIPS, DISTS et PSNR**. Sur
l'ensemble du corpus et des quatorze axes de frontière, il ne reste **une
seule cellule positive** : kryte SSIM-Y contre le 4:2:0 (+2,72 %), l'axe
dont les cycles 13-15 avaient déjà interdit toute revendication dans les
deux sens (interpolations cubique et linéaire en désaccord systématique) —
la perte y datait du champ luma v30 et n'est pas un fait du chroma.

167 tests verts. Binaire archivé :
`target/benchmark_v30/aurea_v31_eob_chroma.exe`, SHA-256
`49A88AD80F567C139191B3A904925A214CD21736FB39A3514116FD257A1E7669`.
Constantes : `V31_EOB_STEP_CHROMA_ETA = 1,0`, porte gamma 0,55-0,65
(`calibration::eob_chroma_gate`).

### Pistes

- kryte SSIM-Y vs 4:2:0, l'ultime cellule — probablement le prix structurel
  du troc spectral sur l'unique image sans guerre chroma ; à n'aborder
  qu'avec un témoin SSIM propre.
- Fondu continu des gates (q31, gamma) ; prédiction d'EOB une-passe ;
  le 8×8 inatteignable.

## Cycle 34 — L'ultime cellule (AUR2 v32) : le tableau tout vert

La décomposition A/B de kryte SSIM-Y vs 4:2:0 (+2,72 %) a innocenté les
deux premiers suspects et confondu le troisième. Réduire le champ EOB luma
ne rendait que 0,5 point pour 1,4 de Y ; rallumer l'anti-ring à q80
*aggravait* (+0,29). Le coupable est **la chaîne d'affûtage q20** : sans
XSharpen ni USM, le SSIM de kryte passe de +2,72 à +0,008 — et tous ses
autres axes BD s'améliorent. Sur une image à chroma plate et sombre,
l'affûtage n'a pas de texture chromatique à révéler ; il ne fait que
déformer les statistiques locales que SSIM mesure. Et kryte n'en a pas
besoin pour sa paire vitrine : λ8 seul y gagnait déjà LPIPS et DISTS.

Les deux gestes de la v32, tous deux portés par le champ diffus `R × G`
déjà objet du format depuis la v28 (0,9418 sur kryte, zéro sur les onze
autres, reconstruit au décodeur avant la post-chaîne) :

1. **Affûtage muté sur le champ diffus** : forces XSharpen/USM × (1 − R·G).
   kryte : +2,72 → +0,045.
2. **Champ EOB luma adouci de 30 % sur le même champ**, mais seulement sous
   la frontière anti-ring (q < 70) : aux pas fins, le champ n'effondre pas
   la variance et sa pleine force est ce qui gagne les métriques de texture
   à haute qualité. kryte : +0,045 → **−0,19**, pendant que la paire
   vitrine q75 garde son DISTS (0,000322 contre 0,000385) — la première
   version sans ce garde-fou de qualité l'avait rendue perdante de trois
   millionièmes, mesurée et corrigée.

**Verdict — régénération complète :** les **168 cellules** BD cubiques de
la frontière (12 images × 7 métriques × 2 profils JPEG optimisés) sont
négatives, et la vitrine conserve **LPIPS 24/24, DISTS 24/24, PSNR 24/24**.
Moyennes : RGB −37,74/−31,70, Y −10,67/−31,55, Cb −61,60/−19,26,
Cr −67,21/−29,21, SSIM-Y −12,28/−32,77, ΔE moyen −45,16/−26,70,
ΔE p95 −40,33/−19,63.

La revendication reste bornée par la méthode : cubique, sur ce corpus,
contre libjpeg-turbo optimisé ; les axes historiquement indécis au contrôle
linéaire le restent. Mais il n'existe plus une seule cellule où le
protocole donne l'avantage à JPEG. 167 tests verts. Binaire archivé :
`target/benchmark_v30/aurea_v32_all_green.exe`, SHA-256
`0A64CB5D314F696D…` (le SHA complet est dans `summary.json`).
Constantes : `V32_EOB_LUMA_RELIEF = 0,3`, portée q < 70 ; affûtage
× (1 − R·G).

## Cycle 35 — Second chantier vitesse (bit-exact)

Même contrainte qu'au cycle 10, plus dure encore que la charte : chaque
levier laisse le flux `.aur` **et** le PNG décodé octet pour octet
identiques (SHA vérifiés sur quatre sentinelles kryte/alchemist aux trois
qualités, 167 tests verts). Le BD du tableau vert est donc conservé par
construction.

Le profil de départ (alchemist 2560×1440, q60) : encodage dominé par le
treillis (50 %, double passe v30 comprise) et le rANS (36 %) ; décodage par
le rANS (47 %) et la post-chaîne (16-38 % selon la qualité — le XSharpen
v29 était mono-thread).

Leviers appliqués, tous bit-exacts :

1. **Recompilation agressive** (LTO fat, codegen-units=1, target-cpu
   natif) : gain nul mesuré — les boucles chaudes étaient déjà saturées.
   Conservée, documentée pour éviter d'y revenir.
2. **XSharpen séparable et parallèle** : les extrema de la fenêtre (2r+1)²
   deviennent deux passes 1-D (min/max sont exacts quel que soit l'ordre de
   visite, les représentants ±0,0 sont arithmétiquement indistinguables) et
   chaque passe se parallélise par lignes. Conversions f64↔f32 de la chaîne
   passées en rayon.
3. **rANS encodeur : paquets parallèles.** Depuis la v27, chaque paquet de
   détail repart d'un *clone* de l'état convergé de la couche 0 et ne lit
   que des données quantifiées figées : les paquets s'encodent en parallèle
   et se rangent dans l'ordre — `enc:rans_ac` passe de 0,77 à **0,25 s**
   (÷3,1).
4. **rANS décodeur : ordonnancement en trois phases.** Les octets de chaque
   paquet sont à des offsets connus (pré-lecture), puis : les paquets L en
   parallèle intégral (ils ne lisent que le squelette gelé de la couche 0
   et leurs propres zéros futurs — valeurs identiques au balayage
   historique) ; les paquets C1 séquentiels en ordre d'arcs (leur
   prédicteur de phase Fourier peut référencer le voisin du dessus décodé
   dans un arc antérieur) ; les paquets C2 en parallèle (ils ne lisent que
   le L et le C1 de leur propre arc). `dec:rans_ac` passe de 0,90 à
   **0,35 s** (÷2,6) ; le C1 séquentiel est le nouveau plancher.

**Verdict — protocole du cycle 9 (12 originals natifs × q20/40/60/80) :**

| | journal (cycle 13) | premier train | **train final** |
|---|---:|---:|---:|
| Encodage moyen | 4,740 s | 2,880 s | **2,398 s** (−49 %) |
| Décodage moyen | 4,557 s | 2,312 s | **2,012 s** (−56 %) |
| vs JPEG 4:4:4 encode (0,409 s) | 10,2× | 7,0× | **5,9×** |
| vs JPEG 4:4:4 decode (0,316 s) | 13,0× | 7,3× | **6,4×** |

Le second train de leviers, toujours bit-exact (SHA revérifiés, 167 tests) :

5. **Scratch de treillis par worker.** Le traceback plat allouait et
   initialisait `vec![penalty(); n · max_states]` à chaque bloc — ~65 Ko de
   memset par bloc 16×16, des gigaoctets par image, invisibles depuis le
   cycle 10 parce que le *memcpy* des snapshots avait été tué mais pas le
   *memset* du traceback. Le traceback ne lit jamais au-delà de `tb_lens`,
   donc les tampons persistent entre blocs (`map_init` rayon, un scratch
   par worker) sans ré-initialisation : le treillis passe de 1,14 à
   **0,90 s** (−21 %) sur la sonde.
6. **Deblock tri-canal concurrent** : les trois plans sont indépendants,
   `rayon::join` imbriqué.

Et ce, en portant tout ce que les cycles 20-34 ont ajouté depuis ces
chiffres : mélange logistique, APM, double passe EOB, post-chaîne de
texture. Binaire archivé : `target/benchmark_v30/aurea_v32_speed.exe`,
SHA-256 `400E8C741E9395AC…` — flux et reconstructions identiques au binaire
du tableau vert.

### Pistes restantes vers ≤ 5×

- Le **C1 séquentiel** borne la phase rANS décodeur : un ordonnancement en
  vagues sur le DAG réel des dépendances top-voisin le briserait.
- Le **treillis** (65 % de l'encodage restant) : prédiction d'EOB une-passe
  (cycle 32), et la passe 2 pourrait réutiliser les décisions de la passe 1
  comme point chaud d'amorçage.
- La post-chaîne q20 (0,45 s) : fusion des passes deblock/AR/USM en un
  parcours.

### Addendum — le plancher C1, mesuré puis requalifié

Avant d'attaquer la chaîne C1, la mesure. `AUREA_DAG_STATS=1` a d'abord
tranché la question des vagues : le graphe des dépendances trans-arcs est
une **chaîne parfaite** (52/53 arcs sur alchemist, 71/72 sur kryte
dépendent de l'arc immédiatement précédent, portée maximale 1 — la spirale
place le voisin du dessus dans l'anneau adjacent). Aucun ordonnancement en
vagues n'existe.

Puis la décomposition par phase (`dec:rans_phases`) a requalifié la cible
elle-même : sur alchemist/q60, le stage rANS (0,31 s) se répartit en
couche 0 ≈ 0,12 s (chaîne codon L₀→C1₀→C2₀ sur tous les blocs), détails L
0,10 s (déjà parallèles), **chaîne C1 0,055 s** et C2 0,033 s. Le
« dernier plancher séquentiel » pèse 5 % du décodage — briser sa chaîne au
prix d'un pipeline à snapshots ou d'accès non sûrs rendrait au mieux
~0,06 s. Un recouvrement propre a été pris (le décodage du squelette C1
court en parallèle de la phase L, `rayon::join`, bit-exact, 167 tests) ;
le reste est déclaré non rentable.

## Cycle 36 — Le triptyque final : deux mesures qui tuent un grand saut

Trois chantiers commandés : fusion de la post-chaîne, prédiction d'EOB
une-passe, et les plans f32. Les deux premiers ont rendu le troisième
non rentable — par la mesure, pas par renoncement.

**Post-chaîne (bit-exact).** L'USM clonait trois tampons de 72 Mo par
appel ; le blur passe en ping-pong a→b→a, zéro allocation par passe
(0,099 → 0,071 s). Et la lecture de la synthèse a révélé le vrai
gisement : **en échantillonnage critique, l'overlap-add entier était du
gaspillage** — chaque pixel appartient à un seul bloc, le tampon de poids
vaut 1,0 partout, la normalisation divise par un. Le chemin rapide écrit
chaque bloc directement (`0+x=x`, `x/1=x` : bit-exact) : plus de tampon de
poids, plus de passe de normalisation, l'étage passe de 0,21 à 0,15 s.

**EOB une-passe (encodeur-seul, revalidé).** La passe 1 du champ EOB ne
servait qu'à fixer les EOB transmis ; le quantifieur greedy — déjà calculé
— les fournit directement et le treillis ne tourne plus qu'une fois
(`AUREA_EOB_ONE_PASS`, promu par défaut). Le treillis passe de 0,98 à
0,64 s sur la sonde, et les fichiers *maigrissent* même (~−1,8 %, les EOB
greedy sont plus courts). Le flux change, donc revalidation intégrale :
sondes sur les trois marges fines (kryte SSIM-420 −0,12, dispair Y-420
−0,43, chilling −2,42 — toutes vertes), puis régénération complète :
**168/168 cellules négatives et vitrine 24/24 ×3 conservées**, moyennes
rendues d'un cheveu (Y-420 −9,69 contre −10,67).

**Le saut f32 : non rentable, et c'est mesuré.** Le f32 visait la bande
passante mémoire de la synthèse et de la post-chaîne. Le chemin critique
l'a *supprimée* au lieu de la réduire ; le périmètre restant (~0,05-0,08 s
espérés) ne paie pas une revalidation totale plus le risque de dérive
flottante sur 168 cellules. Verdict consigné, chemin f64 conservé.

**Protocole final (cycle 9, 12 originals × 4 qualités) :**

| | journal | cycle 35 | **cycle 36** |
|---|---:|---:|---:|
| Encodage moyen | 4,740 s | 2,398 s | **2,031 s** |
| Décodage moyen | 4,557 s | 2,012 s | **1,847 s** |
| vs JPEG 4:4:4 encode | 10,2× | 5,9× | **5,0× — objectif ≤5× atteint** |
| vs JPEG 4:4:4 decode | 13,0× | 6,4× | **5,8×** |

167 tests verts à chaque étape. Binaire final :
`target/benchmark_v30/aurea_v32_speed.exe`, SHA-256
`DA7360AA1406BE97…` — c'est aussi le binaire de la vitrine revalidée.

### Pistes

- Décode ≤5× : il manque 0,27 s sur un profil plat — anti-ring (0,16 s,
  le clone de plan est intrinsèque mais le filtre sigma pourrait se
  restreindre aux bandes touchées), couche 0 rANS (0,12 s), GCT inverse.
- L'encodage à 5,0× ouvre une question nouvelle : viser <1× un jour
  passera par le treillis lui-même (0,64 s), plus par son harnais.

## Cycle 37 — La queue fusionnée, et le 8×8 jugé

**Fusion de la queue du décodeur (bit-exact).** La fin du pipeline
matérialisait cinq plans de 72 Mo : la passe PTF inverse sur L, puis les
trois plans R/G/B de la rotation dorée, relus par l'empaquetage. Une seule
passe par pixel fait désormais PTF⁻¹ → GCT⁻¹ → arrondi → RGB (opérations
et ordre identiques ; l'aller-retour mémoire f64 est exact, et les
constantes à blend 0 sont les mêmes par `x·1=x`, `x+0=x`). Étages
`ptf_inv` + `gct_inv` : 0,13 → 0,06 s ; la sonde alchemist décode
**sous la seconde** (0,90-0,95 s contre 1,78 au départ de la campagne).
Au protocole complet, le gain se noie dans la variance machine (décode
1,84 s mesuré, fourchette réelle 1,79-1,85) : l'objectif ≤ 5× (1,58 s)
résiste encore, à ~0,2-0,25 s.

**Le 8×8 réveillé, mesuré, et rendormi pour cause.** Le split descendant
attendu depuis le cycle 29 est implémenté (`AUREA_SPLIT8_K` /
`AUREA_SPLIT8_HET`, groupes 2×2 alignés de cellules 16, critère de
variance et d'hétérogénéité, défaut 0 bit-exact). Sur kryte — l'image de
meneaux qui motivait la question — il se déclenche massivement (38 600
blocs 8px à K=2) et le verdict est **catastrophique et monotone** :

| BD vs base (kryte) | RGB | Y | SSIM-420 |
|---|---:|---:|---:|
| K=2 | +8,27 % | +10,12 % | +10,34 % |
| K=4 | +6,56 % | +8,13 % | +7,87 % |
| K=8 | +4,92 % | +6,26 % | +5,38 % |
| K=4 + hétérogène | +4,71 % | +5,99 % | +5,96 % |

L'overhead fixe par bloc (DC, EOB, contexte, carte) dévore le bénéfice
d'une transformée plus courte, et la 16 armée du champ EOB et du treillis
capture déjà la localité. Moins on splitte, moins on perd : l'optimum est
zéro. La question ouverte depuis trois cycles est close par la mesure —
**le 8×8 était inatteignable parce qu'il ne fallait pas l'atteindre.** Le
knob reste en témoin de laboratoire.

### Pistes vers les ~0,2 s de décode restants

Tout est parallèle, tout est plat : anti-ring en streaming (~0,04),
recouvrement C2₀/chaîne C1 (~0,03, au prix d'accès non sûrs), micro-SIMD
de l'IDCT. Le prochain vrai palier est ailleurs : machine plus large, ou
changement de représentation (f32) dont le devis reste défavorable.

La vraie carte du décodage restant (1,06-1,14 s sur la sonde) est **plate**
: rANS 30 %, post-chaîne 15-37 % selon la qualité, IDCT+overlap 18 %, GCT
8 %, champ de Turing 6 %, dequant 6 %. La route vers ≤ 5× ne passe plus
par un monstre unique mais par dix étages à −20 % chacun — ou par les
plans `f32` du cycle 10, qui eux devraient repasser par le verdict
débit-distorsion complet.

## Cycle 38 — le format devient natif : décodeur WIC et MSI 32.0.2

Demande : « je veux un support par la visionneuse windows ! » — plus le
menu contextuel de conversion et l'habillage de l'installeur restés en
attente du cycle précédent.

### Le décodeur WIC (`src/wic`, aurea_wic.dll)

Un serveur COM in-process de ~350 lignes (`windows-rs` 0.62, macro
`#[implement]` de `windows-core`) expose `IWICBitmapDecoder` +
`IWICBitmapFrameDecode` et délègue à `aurea_core::decode_aurea`, sortie
24bpp BGR, `CopyPixels` avec sous-rectangles. Trois pièges réels, tous
diagnostiqués par bissection d'erreurs HRESULT :

1. **Signature de trait** : `GetColorContexts` prend
   `*mut Option<IWICColorContext>`, pas `OutRef` (E0053, deux impls).
2. **Les codecs intégrés de Windows 10 sont invisibles au registre** —
   impossible de copier l'enregistrement du PNG ; la voie tierce est
   CLSID complet + `CLSID\{CATID décodeurs}\Instance\{clsid}`, et elle
   fonctionne depuis HKCU sans élévation (banc d'essai idéal : WPF
   `BitmapDecoder` passe par WIC).
3. **WIC laisse le stream après le contrôle de motif** : `Initialize`
   doit faire `Seek(0)` avant de lire, sinon BADIMAGE.

Fausse piste mémorable : le premier fichier de test (`x267/analysis`)
était un `.aur` d'une ère antérieure (magie `AURA` + xz) que le CLI
actuel refuse aussi — la magie réelle du conteneur est `AUR2`.

Vérifié de bout en bout : WIC pur (`CreateDecoderFromFilename`), WPF, et
la Visionneuse de photos Windows affichent kryte 2816×1536 plein cadre.

### MSI 32.0.2

Le paquet enregistre le décodeur en HKLM (motif `#x41555232`, formats,
catégorie), installe le ProgId **aurea.PhotoViewer** — le nom exact que
le UserChoice de la machine mémorisait déjà, donc le double-clic
fonctionne sans toucher au choix utilisateur — avec la commande
rundll32→PhotoViewer.dll (`[System64Folder]` ; `[SystemFolder]` en
paquet x64 est SysWOW64, sauvé uniquement par la relance x64 automatique
de rundll32), les miniatures Explorer via le PhotoThumbnailProvider
standard (qui rend au travers de WIC), le menu cascade « Convertir en
AUR » sur `SystemFileAssociations\image` (ExtendedSubCommandsKey → 4
presets), et des verbes Convertir en PNG / AUREA Viewer sur les .aur.
L'assistant est habillé : panneau latéral et bannière or-sur-noir
générés en PIL (Binary + contrôles Bitmap), titres dorés (TextStyle
COLORREF). Installé, vérifié, clés de test HKCU purgées — le décodage
passe par la DLL de Program Files.

Le format n'est plus un fichier étranger : Windows l'ouvre, le
miniaturise et le convertit d'un clic droit.

### Addendum — les flèches de la Visionneuse (navigation dossier)

Symptôme : impossible d'avancer/reculer aux flèches dans la Visionneuse.
Deux verrous distincts, isolés par bissection :

1. **Le DropTarget, pas la ligne de commande.** Lancée par sa commande
   rundll32 nue, la Visionneuse n'a aucun contexte de dossier — les
   flèches sont mortes même pour des PNG. Explorer ne donne ce contexte
   qu'au travers du DropTarget `{FFE2A43C-56B9-4BF5-9A79-CC6D4285608A}`
   déclaré sous `shell\open\DropTarget` du ProgId.
2. **Le filtre de la liste = le gestionnaire de propriétés photo.**
   KindMap, PerceivedType, OpenWithList, ShellImagePreview : identiques
   entre .aur et .png, aucun effet. Le discriminant mesuré est
   `PropertySystem\PropertyHandlers\.aur = {A38B883C-1682-497E-97B0-0A3A9E801682}`
   (+ `PhotoPropertyHandler\ContainerAssociations` conteneur AUREA →
   TIFF, comme le font WebP et HEIC) : la Visionneuse n'inclut dans sa
   liste de parcours que les extensions dotées de ce handler.

Après ajout : le cycle traverse agree.aur → agree.png → alchemist.aur →
kryte.aur → kryte.png. L'ensemble (plus KindMap et l'intégration Photo
Gallery documentée) est reversé dans le MSI 32.0.2, réinstallé, clés de
test HKCU purgées — l'état de la machine vient intégralement du paquet.
