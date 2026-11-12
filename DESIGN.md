# Project Typographic: Open-Source Contrastive Font Matching Engine (GlyphCLIP)

### System Architecture & Technical Specifications Manual (v3.0.0)

## Introduction

Project Typographic (**GlyphCLIP**) is an open-source, contrastive dual-encoder font retrieval system designed to democratize automated typeface discovery. While impressive tools such as Lens from Mixfonts have demonstrated the power of visual font search, existing systems often operate on fixed font catalogs where incorporating new typefaces requires extensive fine-tuning or full model retraining.

GlyphCLIP decouples font representation from a closed categorical label set. By pre-training a metric embedding backbone across a vast spectrum of typographic anatomies and structural variations, the engine learns a generalized visual font geometry rather than an enumerated class lookup. New fonts are indexed via zero-shot vector projection and stored directly in an external vector index, enabling seamless horizontal scalability and instant typeface ingestion without GPU retraining cycles.

To bridge the visual gap between analytical vector splines inside `.ttf`/`.otf` files and noisy, thresholded camera/screen crops at inference time—while remaining invariant to typeface weights, polarity flips, and pixel-grid aesthetics—**GlyphCLIP v3.0** unifies a **2-Channel Signed Distance Field (SDF) Geometric Bridge** (`[Ink Occupancy, Smooth TSDF]`), a **decoupled 52-class auxiliary character classification head** for Blind Auto-Detection, **from-scratch (`--scratch`) family-disjoint contrastive training across wide weight gaps (`|w1 - w2| <= 400`)**, **multi-scale stroke morphology**, **edge-orthogonality trait scoring**, **`IVF-Cosine` ANN indexing**, and **persistent batch font ingestion**.

---

## 1. Architectural Evolution (`v1.0` Raster Era $\rightarrow$ `v3.0` Vector-SDF)

GlyphCLIP evolved through five documented milestones validated against progressively stricter zero-shot and real-world failure modes:

| Generation | Canonical Input & Charset | Backbone & Head | Training Regime | Key Outcomes & Diagnosed Bottlenecks |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0 Initial Prototype**<br>(`train_local_fonts.py`) | **1-ch FreeType raster** (`DeterministicCanvasRenderer`), `10`-glyph `core` charset (`a,g,e,f,Q,R,S,O,M,t`), **no Bézier SDF** | `MobileNetV4Glyph` (`4.40M` at `512`-d; `256`-d in v1.0, **`freeze_canonical=True`**) | Random multi-character batches (`B=32`, `lr=2e-4`, `8` epochs) | Frozen random Gaussian canonical encoder acted as an unlearned hash: loss stagnated near $\ln(32) \approx 3.4655$ (`3.6%` Top-1); `funnel.png` failed (`0.0035`) as `5/6` letters (`F,u,n,l`) were missing from the `10`-glyph charset. |
| **v1.1 Scaled 1-ch Raster**<br>(`train_scaled.py`) | **1-ch FreeType raster** (`2x` oversampled, `168 px` ink height), `52`-glyph `alpha` charset (`A-Z, a-z`), discrete variable `wght` sampling, **no Bézier SDF** | `GlyphConvNeXt` (`20.55M` params, `512`-d), `shared_backbone=True`, unfrozen, no `char_head` | Same-character hard-negative batching (`B=32`, `120` steps/ep, `8` epochs, `lr=3e-4`) | Converged cleanly (`0.4889` loss, `84.7%` in-batch Top-1; `43.5%` Top-1 / `65.7%` Top-3 / `86.1%` Top-5 / `0.5787` MRR across `108` zero-shot queries; `#1` on `funnel.png` at `99.31%`). Remaining limits: discrete rasterization lost sub-pixel Bézier contours, weak Blind OCR (`10.7–13.1%` Top-1 on 96/82-font splits), and no family-disjoint split. |
| **v2.1 Analytical Vector-SDF** | **2-ch Vector-SDF** (`VectorGlyphPen` + `4x` winding + `DIST_MASK_PRECISE` TSDF), `52` glyphs | `GlyphConvNeXt` (`20.58M`) + `52`-class `char_head` | Warm-started (`0.7 / 0.3` stem split) from `v1.1` `glyphclip_scaled.pt`; mixed `50/50` (`1`-ch / `4`-ch) batches | Closed the vector-to-raster boundary gap and unlocked Blind Auto-Detection without text hints. |
| **v2.2 Trait & Pipeline Hardening** | 2-ch Vector-SDF + `ortho_ratio` (`compute_edge_orthogonality`), `adaptive_binarize`, `min_aspect=0.06` | `20.58M` + `char_head`; `IVF-Cosine` ANN + `_char_buckets` | Family-disjoint split (`724/88/82`), pixel-preserving degradation, `k=3..9` morph, `gap <= 200` (`p=0.40`), `--resume` warm-start | Hit `94.5% / 97.0%` Blind on `82` unseen fonts and solved white-on-dark (`Ram-Altman`) and bitmap (`ENSIA`) queries, but `Regular 400 -> Bold 700` scored `43.3%` Top-1 due to `gap <= 200` and random `max_per_family=3` weight selection. |
| **v3.0 Clean From-Scratch**<br>(Current Production) | Identical 2-ch Vector-SDF + `ortho_ratio` + `IVF-Cosine` (`2,912` curated prod / `5,876` held-out test vectors) | `GlyphConvNeXt` (`20,583,924` params) trained **from random init (`--scratch`)** | Diverse upright weight sampling (`max_per_family=4`), `2`-ep linear warmup + cosine, **`gap <= 400` (`p=0.45`)**, `805/108/96` zero-leakage split | Zero warm-start leakage; **doubled `Regular 400 -> Bold 700` Top-1 (`43.3% -> 86.7%`, `96.7%` Top-3)**, `91.9% / 97.1%` Blind (`0.9444` MRR) on `96` unseen fonts, and **`11/11 (100.0%)`** on the production suite. |

### 1.1 Legacy v1 Pre-Bézier Raster Pipeline Details

Before `v2.1` introduced `fontTools.pens.basePen.BasePen` vector spline extraction (`glyphclip/canvas/sdf.py`), GlyphCLIP `v1.0` and `v1.1` operated strictly on **1-channel grayscale pixel bitmaps** (`[B, 1, 224, 224]`):

- **Rendering (`glyphclip/canvas/renderer.py`, `DeterministicCanvasRenderer`):** Rather than parsing TrueType quadratic (`_qCurveToOne`) or OpenType/CFF cubic (`_curveToOne`) Bézier splines into exact contour windings and Euclidean distance fields, `v1` invoked Pillow's FreeType rasterizer (`ImageFont.truetype`) at `size=160` (`2x` oversampling), cropped the non-zero alpha bounding box, and isotropically rescaled it with `LANCZOS` to `168 px` (`75%` of `224`) on a black `uint8` canvas before dividing by `255.0`.
- **Training Harness Evolution (`v1.0` $\rightarrow$ `v1.1`):**
  - `v1.0` (`scripts/train_local_fonts.py`) trained only `visual_encoder` (`MobileNetV4Glyph`, `embed_dim=256`, `freeze_canonical=True`) against random multi-glyph batches over the `10`-glyph `core` charset, causing the frozen canonical branch to behave like an unlearned hash (`loss = 3.4655`, `3.6%` in-batch Top-1, `0.0035` score on `funnel.png`).
  - `v1.1` (`scripts/train_scaled.py`, saved to `checkpoints/glyphclip_scaled.pt`) switched to an unfrozen shared-backbone `GlyphConvNeXt` (`~20.55M` params, `embed_dim=512`), expanded the charset to `52` glyphs (`alpha`), added variable-font `wght` slicing and `120` same-character hard-negative steps per epoch (`8` epochs, `CosineAnnealingLR`), lifting `v1` to `43.5%` Top-1 / `0.5787` MRR on zero-shot test words and `#1` (`99.31%`) on `funnel.png` with hints.
- **Consequences of Pure Rasterization:** Discrete FreeType anti-aliasing discarded sub-pixel spline curvature before tensor construction, while binary/blurry query crops (`VisualSegmentationPipeline(include_sdf=False)`) exhibited discontinuous step edges. Without a smooth signed distance surface (`TSDF`) encoding contour signed distance ($\pm 8\text{ px}$) or an auxiliary OCR head (`char_head`), `v1.1` struggled with Blind Auto-Detection (`10.7%–13.1%` Top-1 on 96/82-font zero-shot splits) and weight/contrast drift—motivating the analytical 2-channel Vector-SDF geometric bridge introduced in `v2+`.

---

## 2. System Topology Overview

The `v3.0` system separates typeface matching into two symmetrical 2-channel pipelines converging into a shared `GlyphConvNeXt` metric space ($\mathbb{S}^{511}$):

```
+=======================================================================================================+
|                        CANONICAL FONT DOMAIN (OFFLINE CATALOG / LIVE INGESTION)                       |
|                                                                                                       |
|  .ttf / .otf Font File --> FontParser (OS/2, fvar axes) --> VectorSDFRenderer (fontTools BasePen)     |
|                                                                - Quadratic/Cubic Bezier subdivision   |
|                                                                - 4x subpixel winding + DIST_MASK_PRECISE
|                                                                - Output: [2, 224, 224] + ortho_ratio  |
|                                                                      |                                |
|                                                                      v                                |
|                                                             Shared GlyphConvNeXt (~20.58M params)     |
|                                                             z_c = encode_canonical(g) in S^511        |
+====================================================================== | ==============================+
                                                                        v
                                              +--------------------------------------------------+
                                              | GlyphVectorStore (embeddings.npy + ann_index.npz)|
                                              | O(1) char buckets + Spherical K-Means IVF-Cosine |
                                              +--------------------------------------------------+
                                                                        ^
+====================================================================== | ==============================+
|                               VISUAL INFERENCE DOMAIN (/v1/recognize)                                 |
|                                                                                                       |
|  Query Image --> VisualSegmentationPipeline                   Shared GlyphConvNeXt + char_head        |
|                   - minAreaRect deskew (<30 deg)       --->    - z_v = encode_visual(x) in S^511      |
|                   - Otsu polarity pre-inversion +              - char_logits in R^52 (Top-3 cands)    |
|                     scale-aware Sauvola (W<=151, k=0.2)               |                               |
|                   - CCA (min_aspect=0.06) + dot merging               v                               |
|                   - 168px height norm + TSDF + ortho_ratio    BayesianConsensusEngine                 |
|                   - Output: [N, 2, 224, 224]                  (Entropy + Missing + Ortho Trait)       |
+=======================================================================================================+
```

---

## 3. Mathematical Formulation & Multi-Task Objective

### 3.1 Dual-Encoder Mapping & Decoupled Auxiliary OCR Head

For a font $f \in \mathcal{F}$ and character $c \in \mathcal{C}$, let:

- $\mathbf{g}_c^{(f)} \in [0, 1]^{2 \times 224 \times 224}$ be the analytical vector-rendered 2-channel canonical glyph.
- $\mathbf{x}_c^{(f)} \in [0, 1]^{2 \times 224 \times 224}$ be a segmented or synthetically degraded visual crop.

In the production architecture (`DualEncoder` with `backbone_type="convnext"`, `shared_backbone=True`, `in_channels=2`, `embed_dim=512`, `num_chars=52`), both branches share parameters $\theta$ (`GlyphConvNeXt`):

$$\mathbf{h} = \text{LayerNorm}(\text{GAP}(f_{\text{trunk}}(\mathbf{u}; \theta))) \in \mathbb{R}^{640}$$

$$\mathbf{z} = \frac{\text{Proj}(\mathbf{h})}{\Vert \text{Proj}(\mathbf{h}) \Vert_2} \in \mathbb{S}^{511}, \qquad \mathbf{\ell} = W_{\text{char}} \mathbf{h} + \mathbf{b}_{\text{char}} \in \mathbb{R}^{52}$$

Branching `char_head` ($640 \to 52$) directly from the pooled trunk representation $\mathbf{h}$—before the metric projection MLP `Proj` ($640 \to 1024 \to 512$)—lets the contrastive hypersphere $\mathbb{S}^{511}$ focus purely on intra-character typographic style while giving the model fast character-hypothesis predictions for blind queries.

### 3.2 Contrastive & Joint Loss Functions

1. **Symmetric Temperature-Scaled InfoNCE (`InfoNCELoss`):** Given batch pairs $\{(\mathbf{z}_{v, i}, \mathbf{z}_{c, i})\}_{i=1}^B$ and learnable temperature $\tau = \operatorname{clip}(\exp(\log \tau), 10^{-4}, 1.0)$ (initialized at $\tau = 0.05$ in `scripts/train_sdf.py`, class default $0.07$):

   $$S_{i,j} = \frac{\langle \mathbf{z}_{v, i}, \mathbf{z}_{c, j} \rangle}{\tau}$$

   $$\mathcal{L}_{v \to c} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(S_{i,i})}{\sum_{j=1}^B \exp(S_{i,j})}, \qquad \mathcal{L}_{c \to v} = -\frac{1}{B} \sum_{j=1}^B \log \frac{\exp(S_{j,j})}{\sum_{i=1}^B \exp(S_{i,j})}$$

   $$\mathcal{L}_{\text{InfoNCE}} = \frac{\mathcal{L}_{v \to c} + \mathcal{L}_{c \to v}}{2}$$

2. **Joint Vector-SDF + Auxiliary Character Loss (`scripts/train_sdf.py`):**

   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{InfoNCE}} + 0.4\,\mathcal{L}_{\text{CE}}(\mathbf{\ell}_v, y) + 0.15\,\mathcal{L}_{\text{CE}}(\mathbf{\ell}_c, y)$$

   where $y \in \{0, \dots, 51\}$ is the true Latin character index (`get_charset("alpha")`).

3. **Mixed Hard-Negative Batch Sampling:** Training alternates evenly (50% / 50% per step) between:
   - **Same-character hard-negative batches (`num_chars=1`):** Up to `batch_size=32` distinct fonts all rendering the exact same character, forcing attention onto serifs, stroke contrast, terminals, and counters rather than glyph identity.
   - **Multi-character cross-font batches (`num_chars=4`):** 4 distinct characters sampled across multiple fonts per batch, maintaining global negative diversity and calibrating all 52 logits of `char_head`.

### 3.3 Character Sets & Typographic Entropy Weights

Defined in `glyphclip/canvas/charset.py`:

| Subset Key | Size | Glyphs Covered | Primary Use |
| :--- | :--- | :--- | :--- |
| `"full"` (`CANONICAL_CHARSET`) | 62 | `a-z`, `A-Z`, `0-9` | Live `/v1/index/font` ingestion |
| `"alpha"` | 52 | `a-z`, `A-Z` | Vector-SDF training, `char_head`, production catalog |
| `"core"` | 10 | `a, g, e, f, Q, R, S, O, M, t` | Compact/ablation baseline |

Each matched character is weighted in multi-glyph consensus via `DISCRIMINATIVE_WEIGHTS`:

- **High entropy:** `g`, `Q` ($1.4$); `a`, `R` ($1.3$); `e`, `f`, `S`, `G` ($1.2$); `t`, `s`, `r`, `k`, `x`, `B`, `M`, `W`, `A` ($1.1$)
- **Default:** `1.0`
- **Slightly reduced / neutral:** `O`, `o`, `c`, `C` ($0.9$)
- **Low entropy:** `l`, `i`, `I`, `1` ($0.6$)

---

## 4. Geometric Representation & Data Pipelines

### 4.1 Analytical 2-Channel Vector-SDF Pipeline (`glyphclip/canvas/sdf.py`)

#### Canonical Indexing Path (`VectorSDFRenderer`)

1. **Font Metadata & Axis Instancing (`FontParser`):** Parses `.ttf`/`.otf` tables (`name` IDs 1/2/16/17/6/13/14, `OS/2.usWeightClass`, `OS/2.fsSelection`, and `fvar`).
   - Variable fonts (`fvar` with `wght` axis): instantiated via `fontTools.varLib.instancer.instantiateVariableFont`. `FontParser.get_supported_weights` returns matching standard CSS weights in `{100..900}`; batch indexers (`scripts/build_production_catalog.py`, `IndependentTestHarness`) sample `[300, 400, 600, 700]` to bound index growth.
   - Static fonts: indexed at their nominal `OS/2` weight (`metadata.weight`).
2. **Adaptive Bézier Contour Extraction (`VectorGlyphPen(BasePen)`):** Reads glyph outlines directly from `tt.getGlyphSet()`, subdividing quadratic (`_qCurveToOne`, 2–14 steps) and cubic (`_curveToOne`, 2–16 steps) curves adaptively according to `chord_step = max(1.0, 0.008 * upm)`.
3. **Isotropic Geometry Normalization:** All contour points are bounded `[xmin, ymin, xmax, ymax]` and mapped isotropically into a $224 \times 224$ canvas:
   - Target height = $0.75 \times 224 = 168\text{ px}$ (unless constrained by max width $224 - 8 = 216\text{ px}$, in which case width scales proportionally), centered symmetrically on the canvas.
4. **Non-Zero Winding Subpixel Mask:** Contours are rasterized at $4\times$ supersampling (`hr_dim = 896`) with $16\times$ fixed-point coordinates (`cv2.fillPoly(..., shift=4)`). Each contour's signed shoelace area in Y-up font coordinates increments or decrements `winding`, yielding `hr_mask = (winding != 0)` so interior counters (`o`, `e`, `g`, `B`) carve out cleanly regardless of TrueType vs. PostScript/CFF winding direction.
5. **Dual Channel Synthesis:**
   - **Channel 0 (`Occupancy`):** `cv2.resize(hr_mask, (224, 224), interpolation=cv2.INTER_AREA) / 255.0`.
   - **Channel 1 (`Vector-TSDF`):** Exact Euclidean distance transform (`cv2.DIST_L2, cv2.DIST_MASK_PRECISE`) on `hr_mask` and its complement, shifted by $0.5\text{ px}$ at the contour interface, downsampled to $224 \times 224$, and mapped via smooth hyperbolic tangent truncation ($\sigma = 8.0$):

     $$\text{TSDF}(d) = 0.5 + 0.5 \cdot \tanh\left(\frac{d}{\sigma}\right) \in [0.0, 1.0]$$

     Background regions approach $0.0$, stroke cores approach $1.0$, and the exact glyph boundary sits at $0.5$ with steepest gradient.
   - Empty or bitmap-only glyphs fall back cleanly to `DeterministicCanvasRenderer` (`FreeType` via Pillow `2x` upscale + MD5-cached font handles bounded at 64 entries) followed by `binary_mask_to_tsdf`.

### 4.2 Visual Query Segmentation & Euclidean Bridge (`glyphclip/pipeline/`)

Real-world word images and screenshots are processed by `VisualSegmentationPipeline` (`glyphclip/pipeline/segmentation.py`):

1. **Grayscale Conversion:** Converts BGR input to 8-bit grayscale.
2. **Contour-Based Deskew (`deskew_image`):** Runs Canny edges `(50, 150)` and `cv2.minAreaRect` to estimate skew angle in $[-45^\circ, 45^\circ]$; warps with `INTER_CUBIC` and border replication when $0.5^\circ \le |\theta| \le 30.0^\circ$.
3. **Polarity-Aware Scale-Adaptive Binarization (`adaptive_binarize` in `binarization.py`):**
   - Runs Otsu thresholding + `normalize_polarity(otsu)` to determine whether background border pixels are bright or dark, pre-inverting light-on-dark images (`255 - gray`) before Sauvola so dark backgrounds never generate connected halo moats around adjacent glyphs.
   - Scales Sauvola window adaptively via `eff_window = max(sauvola_window, min(151, (min(H, W) // 3) | 1))` ($k = 0.2, R = 128.0$) so high-resolution thick display strokes do not hollow out.
4. **Morphological Closing & Contour Filtering:** Applies `2x2` rectangular `MORPH_CLOSE` and extracts external contours (`cv2.RETR_EXTERNAL`), keeping main glyph boxes (`area >= 64` and $0.06 \le w/h \le 2.5$, retaining narrow `I`/`l` stems) while storing small boxes (`area >= 12`) as candidate diacritics.
5. **Stacked Component Merging (`_merge_stacked_components`) & Sorting:** Attaches dots/diacritics (e.g., on `i`, `j`) to their horizontal stem box when x-overlap $\ge 0.4$ and vertical gap $\le \max(12, 0.8 \cdot h_{\text{stem}})$, merges overlapping main boxes ($\ge 0.45$ x-overlap), and sorts horizontally left-to-right.
   - *(Fallback in `/v1/recognize` and `IndependentTestHarness`)*: if segmentation finds zero bounding boxes, the full grayscale crop is normalized directly.
6. **Matching 168px Normalization, TSDF & Edge Trait (`normalize_glyph_crop` + `compute_edge_orthogonality`):**
   - When `include_sdf=True` (production): crops are scaled isotropically to `target_glyph_height=168` (`max_allowed_w=216`), centered on `224x224`, and paired with `binary_mask_to_tsdf(canvas, sigma=8.0)` to produce `[2, 224, 224]`.
   - `compute_edge_orthogonality()` measures `0°/90°` Sobel gradient alignment across contour pixels (`~0.95–0.98` for pixel/bitmap fonts vs. `~0.37–0.69` for curved/diagonal Bezier fonts).
   - When `include_sdf=False` (legacy 1-channel compatibility): crops scale to `max_glyph_dim=180` outputting `[1, 224, 224]`.

---

## 5. Neural Architectures & Serialization

### 5.1 Production Backbone: `GlyphConvNeXt` (`glyphclip/models/convnext.py`)

`DualEncoder` uses a weight-shared `GlyphConvNeXt` (`shared_backbone=True`) across both `visual_encoder` and `canonical_encoder`:

| Stage / Layer | Configuration | Output Shape |
| :--- | :--- | :--- |
| Input Tensor | 2-channel `[Occupancy, TSDF]` in `[0, 1]` | `[B, 2, 224, 224]` |
| Patch Stem | `Conv2d(2 -> 80, k=4, stride=4)` + `GroupNorm(1, 80)` | `[B, 80, 56, 56]` |
| Stage 1 | 3× `ConvNeXtBlock(80)` (`7x7` DWConv, LN, `4x` GELU MLP) | `[B, 80, 56, 56]` |
| Downsample + Stage 2 | `GN(80) -> Conv2d(80->160, k=2, s=2)` + 3× `ConvNeXtBlock(160)` | `[B, 160, 28, 28]` |
| Downsample + Stage 3 | `GN(160) -> Conv2d(160->320, k=2, s=2)` + 9× `ConvNeXtBlock(320)` | `[B, 320, 14, 14]` |
| Downsample + Stage 4 | `GN(320) -> Conv2d(320->640, k=2, s=2)` + 3× `ConvNeXtBlock(640)` | `[B, 640, 7, 7]` |
| Global Pooling + Norm | `mean([-2, -1])` + `LayerNorm(640, eps=1e-6)` | `[B, 640]` |
| Metric Projection Head | `Linear(640->1024) -> LN -> GELU -> Linear(1024->512)` + L2 Norm | `[B, 512]` on $\mathbb{S}^{511}$ |
| Auxiliary OCR Head (`char_head`) | Optional `Linear(640 -> 52)` (logits for `a-z, A-Z`) | `[B, 52]` |

- **Observed Parameter Count (shared backbone):** **20,583,924 parameters** with `in_channels=2, embed_dim=512, num_chars=52` (`20,550,592` when `num_chars=0`).
- **1-Channel Warm-Start (`DualEncoder.load_warmstart_from_1ch`):** Splits a legacy `[80, 1, 4, 4]` patch-stem kernel across channels at `0.7` (Occupancy) and `0.3` (TSDF) and loads remaining matching weights with `strict=False`.

### 5.2 Legacy Edge Option, 1-ch Warm-Start & Exporters (`mobilenet_v4.py`, `onnx_exporter.py`)

- `MobileNetV4Glyph` (`backbone_type="mobilenet"`): Lightweight `UIBBlock`-based encoder (`3x3` stride-2 stem + 4 stages outputting `[960, 7, 7]`, **4,402,016 parameters** at `embed_dim=512, in_channels=1`).
- `DualEncoder.load_warmstart_from_1ch`: Retained for legacy v2.1/v2.2 compatibility (splits `[80, 1, 4, 4]` stem at `0.7` / `0.3`), bypassed in **v3.0** via `--scratch`.
- `TorchScriptExporter` & `ONNXExporter` (`glyphclip/export/onnx_exporter.py`): Automatically infer `in_channels = getattr(model, "in_channels", 1)` and trace `(1, in_channels, 224, 224)` inputs (verified on both 1-channel `MobileNetV4Glyph` and 2-channel `GlyphConvNeXt` TorchScript export).

---

## 6. Vector Storage & Multi-Glyph Consensus Engine

### 6.1 IVF-Cosine ANN & Character-Bucketed Store (`glyphclip/index/vector_store.py`)

`GlyphVectorStore` stores L2-normalized `float32` embeddings in a contiguous matrix (`[N, 512]`) persisted on disk as:

- `font_index/embeddings.npy`
- `font_index/metadata.json` (`embed_dim`, ordered `ids`, ordered `metadata` records including `ortho_ratio`)
- `font_index/ann_index.npz` (Spherical K-Means centroids + cell assignments when `len(store) >= ann_threshold=512`)

Key retrieval mechanics:
- **$O(1)$ Character-Partitioned Buckets (`_char_buckets`):** Maps `glyph_char` directly to row indices so character-filtered queries avoid full-index metadata scans (`0.350 ms` measured for char-filtered search across `50,000 × 256` vectors).
- **Spherical K-Means `IVF-Cosine` ANN:** Automatically builds Voronoi cells (`K ≈ clamp(sqrt(N), 16, 256)`, probing `nprobe = ceil(n_cells * nprobe_ratio)` with `nprobe_ratio=0.25`) once `len >= ann_threshold`, with incremental assignment (`_assign_new_to_ann`) on live uploads.
- Record IDs follow `{postscript_name}_w{weight}_{char}`, so multiple static weights under the same font `family` coexist cleanly without overwriting one another.

### 6.2 Bayesian Multi-Glyph Consensus (`glyphclip/index/consensus.py`)

Given $M$ segmented glyph embeddings $\mathbf{z}_{v, 0}, \dots, \mathbf{z}_{v, M-1}$ and optional `query_ortho_ratios`:

1. **Character Constraint Resolution (per glyph index $i$):**
   - If explicit `query_chars[i]` was parsed from `query_text` (smart alignment strips unsegmented spaces/punctuation when alpha count matches crop count): restrict search strictly to `filter_char = query_chars[i]`.
   - Else if `DualEncoder.predict_char_candidates` is available (`num_chars > 0`, `top_k=3`, `min_prob=0.08`): filter index search to top predicted candidates `cands[i]`.
   - Else fallback: run unfiltered `top_k=20` search and majority-vote the highest similarity-weighted `glyph_char`.
2. **Candidate Retrieval & Edge-Orthogonality Consistency:** Searches with `top_k = max(self.top_k, 250 * num_filter_chars)`. For each candidate hit with `ortho_ratio` $o_c$ matching a query glyph with `ortho_ratio` $o_q$, applies a bounded mismatch penalty beyond `ortho_deadband = 0.18` (`ortho_penalty_weight = 1.4`) before taking the best instance per family $f$:

   $$S(f, i) = \max_{j} \max\left(0.0,\ \langle \mathbf{z}_{v, i}, \mathbf{z}_{c, j}^{(f)} \rangle - 1.4 \cdot \max(0,\ |o_{q, i} - o_{c, j}^{(f)}| - 0.18)\right)$$

3. **Entropy-Weighted Aggregation & Missing Penalty:** Using weights $w_i = w(c_i)$ from Section 3.3 and `missing_penalty_lambda = 0.5`:

   $$C(f) = \frac{\sum_{i=1}^M w_i \cdot S(f, i)}{\sum_{i=1}^M w_i} \cdot \left(1.0 - \lambda \cdot \frac{\text{MissingCount}(f)}{M}\right)$$

   Scores are clamped to $[0.0, 1.0]$ and sorted descending.

---

## 7. API Surface, Checkpoint Discovery & Web UI

Implemented in `glyphclip/server/app.py`, `glyphclip/server/schemas.py`, and `glyphclip/server/static/index.html`:

- **Automatic Model & Channel Discovery (`create_app`):**
  - Checkpoint priority: `checkpoints/glyphclip_sdf.pt` > `checkpoints/glyphclip_scaled.pt` > `checkpoints/glyphclip_trained.pt`.
  - In `v3.0`, `checkpoints/glyphclip_v3.pt` is promoted to `checkpoints/glyphclip_sdf.pt` (with legacy `v2.2` preserved at `checkpoints/glyphclip_v2_2.pt`).
  - Infers `in_channels` and `num_chars` automatically, configuring `VectorSDFRenderer` and `VisualSegmentationPipeline(include_sdf=(in_channels == 2))`.

| Endpoint | Method | Inputs | Response Schema / Behavior |
| :--- | :--- | :--- | :--- |
| `/`, `/ui` | `GET` | — | Serves single-page Tailwind Web UI (`glyphclip/server/static/index.html`) |
| `/health` | `GET` | — | `HealthResponse`: `status`, `version`, `device`, `total_vectors_indexed` |
| `/v1/catalog` | `GET` | `limit=150`, `query` | Distinct families, postscript names, sorted weights, glyph counts, license |
| `/v1/recognize` | `POST` | `file`, `query_text`, `top_k=5`, `include_metrics=false` | `RecognizeResponse`: segments word crop (`min_aspect=0.06`), computes `ortho_ratio`, uses aligned hints or `predict_char_candidates`, runs Bayesian consensus |
| `/v1/index/font` | `POST` | `file` (`.ttf`/`.otf`), `metadata` (JSON) | `IndexFontResponse`: parses metadata, renders 62-char vectors + `ortho_ratio`, adds to `vector_store`, persists to disk (`persisted=True`) |
| `/v1/index/fonts` | `POST` | `files` (multiple `.ttf`/`.otf`), `metadata` (JSON) | `BatchIndexFontResponse`: ingests multiple weights/files sequentially in one batch and persists `vector_store` once at the end |

---

## 8. Training Pipeline & Observed Benchmarks (`v3.0` vs. `v2.2` & `v1`)

### 8.1 Synthetic Visual Degradation (`glyphclip/pipeline/degradation.py`)

`TypographicDegradationPipeline` transforms canonical occupancy `canonical[:, 0:1]` online before `batch_occupancy_to_dual(..., sigma=8.0, thr_jitter=0.06, fast_mask=True)` converts the degraded crop back into a 2-channel `[Occ, TSDF]` visual sample (applying hard Sauvola-style thresholding with `p=0.6` and threshold jitter $\pm 0.06$):

- **Geometric:** `RandomAffine(degrees=(-6, 6), translate=(0.04, 0.04), scale=(0.92, 1.08), shear=(-5, 5))` + `RandomPerspective(distortion_scale=0.15, p=0.4)`
- **Multi-Scale Stroke Morphology:** `RandomStrokeMorphology(p=0.65, kernel_sizes=(3, 5, 7, 9))` (blended `max_pool2d` dilation/erosion at `alpha ∈ [0.35, 1.0]`)
- **Optical Resolution Drops (Pixel-Preserving):** `RandomChoice([Identity, Resize(64)->Resize(224, BILINEAR), Resize(96, BICUBIC)->Resize(224, BILINEAR)])` — `INTER_NEAREST` blocky upscaling was removed so the backbone preserves genuine bitmap/pixel-font corners
- **Blur & Noise:** `RandomApply([GaussianBlur(k=5, sigma=(0.6, 1.8))], p=0.5)` + `GaussianNoise(mean=0.0, sigma=0.08, clip=True)`

### 8.2 Clean From-Scratch Training & Weight-Diverse Corpus (`v3.0`)

- **Diverse Upright Weight Family-Disjoint Corpus (`glyphclip/pipeline/dataset.py`):** `FontDatasetSplitter.get_font_splits` partitions strictly by normalized font family (`normalize_family_key`) with `--max-fonts 0 --max-per-family 4` (`seed=42`). Unlike `v2.2`'s first-N slice, `v3.0` sorts upright weights (`~0..900`) and spaces selection evenly across the family spectrum (`Thin/Light`, `Regular`, `Bold`, `Black`) before filling remaining slots with italics, producing **`805 Train` (`440` unique train families, `477` cross-weight peers with `|w1 - w2| <= 400`) | `108 Val` | `96 Test` (`5,876` rendered vectors)** with **zero family overlap**.
- **From-Scratch Optimization & Wide-Gap Pairing (`scripts/train_sdf.py`):**
  - Run command: `--scratch --max-fonts 0 --max-per-family 4 --max-weight-gap 400 --p-cross-weight 0.45 --epochs 12 --lr 3e-4 --ckpt-name glyphclip_v3.pt`.
  - Pure random initialization (no `--resume` or `1-ch` warm-start), `2`-epoch `LinearLR` warmup (`0.1 -> 1.0`) into `CosineAnnealingLR` (`eta_min=1e-5`), distinct-family batching (`_sample_distinct_families`), and wide same-family cross-weight positives (`|w1 - w2| <= 400`, pairing `Regular 400 <-> Bold 700` from Epoch 0).
  - **Epoch 12 Validation (`108` unseen fonts):** Loss **`0.4305`** | Pairwise Top-1 **`97.9%`** | Aux OCR Top-1 / Top-3 **`79.2% / 96.5%`** | Learned Temperature $\tau = 0.0390$.

### 8.3 Measured Zero-Shot, Cross-Weight & Live API Benchmarks

- **Held-Out 96-Font Zero-Shot Test Split (`384` queries across `5,876` glyph vectors):**

| Model Checkpoint | Mode | Top-1 | Top-3 | MRR |
| :--- | :--- | :---: | :---: | :---: |
| **1-ch Raster Baseline v1.1** (`glyphclip_scaled.pt`) | Text-Aligned | 51.6% | — | 0.6074 |
| **1-ch Raster Baseline v1.1** (`glyphclip_scaled.pt`) | Blind Auto | 10.7% | — | 0.2183 |
| **2-ch Vector-SDF v3.0** (`glyphclip_v3.pt`, `--scratch`) | Text-Aligned | **86.2%** | **91.1%** | **0.8890** |
| **2-ch Vector-SDF v3.0** (`glyphclip_v3.pt`, `--scratch`) | Blind Auto | **91.9%** | **97.1%** | **0.9444** |

*(Historical reference on the earlier `82`-font v2.2 warm-started split, `328` queries / `4,316` vectors: v2.2 scored `90.9% / 95.1% / 0.9301` Text-Aligned and `94.5% / 97.0% / 0.9578` Blind vs. `71.0% / 13.1%` Top-1 for `glyphclip_scaled.pt`.)*

- **30-Family `Regular 400 -> Bold 700` Cross-Weight Stress Test (querying `'Sample'` in `Bold 700` against only `Regular 400` indexed):**

| Checkpoint | Mode | Top-1 | Top-3 | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **v2.2 (`glyphclip_v2_2.pt`)** | Hinted / Blind | `43.3%` (`13/30`) / `53.3%` (`16/30`) | `76.7%` (`23/30`) / — | `--max-weight-gap 200`, `--resume` warm-start |
| **v3.0 (`glyphclip_v3.pt`)** | Hinted | **86.7% (`26/30`)** | **96.7% (`29/30`)** | **+43.4 pts Top-1** (2× improvement) |
| **v3.0 (`glyphclip_v3.pt`)** | Blind Auto | **86.7% (`26/30`)** | **96.7% (`29/30`)** | Exact parity between Blind and Hinted |

  - **`modelica.png` (`BwModelica-Bold 700`) with ONLY `BwModelica-Regular (400)` indexed:** Climbed from **#13 (`77.1%`)** under `v2.2` to **#5 (`94.0%`)** under `v3.0` (`+16.9%` confidence gain).

- **Production Catalog (`2,912` vectors across `28` curated families) & Live `/v1/recognize` API:**
  - [`scripts/test_sdf_production.py`](file:///c:/Users/ASUS/Desktop/prog/GlyphCLIP/scripts/test_sdf_production.py) (`11` queries): **11/11 (100.0%) Top-1** in both Hint and Blind modes (vs. `10/11` in v2.2; `Times New Roman 'Typo'` resolved cleanly at `#1`).
  - `funnel.png` (Blind): **#1 `Funnel Display` (`99.5%`)** in **84.3 ms** (`6` glyphs).
  - `Ram-Altman@Funnel.png` (white text on dark green banner, Blind): **#1 `Funnel Display` (`99.5%`)** in **106.7 ms** (`9` glyphs).
  - `ENSIA@PixelOperator8.png` (unindexed pixel font tested against `Silkscreen`): **#1 `Silkscreen` (`48.2%`)** with Hint `"ENSIA"` (**51.2 ms**), **#2 `Silkscreen` (`55.9%`)** Blind behind `Signate Grotesk` (`58.1%`), `5` glyphs detected.
- **Unit Verification:** `28/28` tests passing (`python -m unittest discover tests -v`, `13.603s`).

---

## 9. Repository Structure & Technical Report Roadmap

### Current Package Layout

- `glyphclip/canvas/`: `charset.py`, `font_parser.py`, `renderer.py`, `sdf.py`
- `glyphclip/models/`: `convnext.py`, `dual_encoder.py`, `loss.py`, `mobilenet_v4.py`
- `glyphclip/pipeline/`: `binarization.py`, `segmentation.py`, `degradation.py`, `dataset.py`, `evaluator.py`
- `glyphclip/index/`: `vector_store.py`, `consensus.py`
- `glyphclip/server/`: `app.py`, `schemas.py`, `static/index.html`
- `glyphclip/export/`: `onnx_exporter.py`
- `scripts/`: `train_sdf.py`, `train_scaled.py`, `train_local_fonts.py`, `build_production_catalog.py`, `test_sdf_production.py`, `test_client.py`
- `checkpoints/`: `glyphclip_sdf.pt` (`v3.0` active), `glyphclip_v3.pt`, `glyphclip_v2_2.pt`, `glyphclip_scaled.pt` (`v1.1` baseline)
- `tests/`: `test_sdf.py`, `test_models.py`, `test_pipeline.py`, `test_consensus.py`, `test_renderer.py`, `test_api.py`

### Next Milestones (Technical Report & Publication Ablations)

1. **Technical Report Drafting:** Document the complete `v1.0 -> v3.0` design, mathematical formulation, Vector-SDF vs. 1-ch raster transition, cross-weight pairing results, and production systems architecture.
2. **Controlled Ablation Matrix & External Baselines:** Isolate contributions of (a) 2-ch Vector-TSDF vs. 1-ch FreeType raster, (b) decoupled `char_head` Blind consensus vs. Hint-locked search, (c) `--max-weight-gap 400` cross-weight pairing + multi-scale morphology, and (d) `ortho_ratio` trait scoring, benchmarked alongside frozen vision backbones (CLIP / DINOv2) and real-world photographed signage.
