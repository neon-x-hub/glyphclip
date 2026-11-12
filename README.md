# Project Typographic: GlyphCLIP (v3.0)

> **Open-Source Contrastive Dual-Encoder Font Matching Engine with 2-Channel Analytical Vector-SDF & Zero-Shot Ingestion**

GlyphCLIP decouples font identification from closed-set classification catalogs. Instead of retraining a classifier when new typefaces appear, GlyphCLIP maps glyph geometry onto a 512-dimensional unit hypersphere ($\mathbb{S}^{511}$) using a shared `GlyphConvNeXt` backbone (`~20.58M` parameters) trained contrastive-wise from scratch across wide weight gaps (`|w1 - w2| <= 400`). New `.ttf` and `.otf` fonts are ingested zero-shot into a local `IVF-Cosine` ANN vector store with zero GPU retraining.

📄 **Read the Full Research Paper:** [`report/main.pdf`](report/main.pdf)  
📝 **LaTeX Source & Bibliography:** [`report/main.tex`](report/main.tex) & [`report/references.bib`](report/references.bib)

---

## Benchmark Highlights (Held-Out Test Split)

Evaluated across **106 strictly held-out font families** (6,396 canonical vectors, 424 synthetic degraded word queries with blur, perspective skew, and noise) and **30 multi-weight stress families**:

| Model Generation | Input Representation | Text-Aligned Top-1 | Blind Auto-Detection Top-1 | Blind MRR | $\Delta w = 300$ Jump (Reg $\to$ Bold) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **v1.1 Scaled Baseline** | 1-ch FreeType Raster | 51.4% | 9.0% | 0.1856 | 3.3% |
| **v2.2 Vector-SDF (Warm)** | 2-ch TSDF ($\sigma=8.0$) | 87.5% | 94.1% | 0.9614 | 70.0% |
| **v3.0 Production (Ours)** | **2-ch TSDF + Ortho Trait** | **89.4%** | **95.0%** | **0.9667** | **93.3%** |
| *v3.0 (Ablation: No Char Head)* | 2-ch TSDF (No Aux Head) | — | 91.5% | 0.9395 | — |

* **Zero-Leakage From-Scratch Training:** Eliminating warmstart weight leakages and widening cross-weight pairing to $|w_1 - w_2| \le 400$ boosts $\Delta w = 300$ (Regular $\to$ Bold) retrieval to **93.3%** and extreme $\Delta w = 400$ jumps to **76.7%** (+26.7 percentage points over v2.2).
* **Sobel Edge-Orthogonality Trait:** Penalizes curved false positives when querying bitmap typefaces (`Pixel Operator 8`, `Minecraft`), promoting pixelated fonts like `Silkscreen` to Rank #1 while applying zero penalty to smooth display curves (`Funnel Display`).
* **Sub-Millisecond ANN Scaling:** $O(1)$ inverted character bucketing paired with spherical IVF-Cosine ANN executes in **0.034 ms** at 2,912 vectors and **0.950 ms** at 60,000 vectors with **100.0% Recall@1** ($n_{\text{probe}} = 25\%$).

---

## Key Capabilities

* **2-Channel Analytical Vector-SDF Geometric Bridge:** Parses `.ttf`/`.otf` spline contours directly (`fontTools` `BasePen`, `4x` subpixel non-zero winding, exact Euclidean TSDF) and pairs them symmetrically with polarity-aware, scale-adaptive query binarization (`[Occupancy, Smooth TSDF]`).
* **Blind Auto-Detection & Hint-Locked Modes:** A decoupled 52-class auxiliary character head (`char_head`) attached to pooled trunk features drives autonomous multi-glyph consensus on un-transcribed word crops without distorting the $\mathbb{S}^{511}$ typographic style manifold.
* **Cross-Weight & Polarity Invariance:** Multi-scale stroke morphology ($k \in \{3, 5, 7, 9\}$) and wide cross-weight pairing ($p=0.45, \Delta w \le 400$) combined with Otsu background polarity pre-inversion resolve hollow halos on dark banners and inverted graphics.
* **Fast Retrieval & Zero-Shot Persistence:** $O(1)$ character-bucketed lookup backed by spherical K-Means `IVF-Cosine` ANN (`ann_index.npz`), plus persistent single (`/v1/index/font`) and batch (`/v1/index/fonts`) ingestion endpoints and interactive Web UI.
* **Production Deployment Ready:** CPU/GPU accelerated FastAPI microservice, interactive Tailwind CSS UI, and TorchScript/ONNX serialization.

---

## Directory Structure

```
GlyphCLIP/
├── report/                            # Publication manuscript (November 2026)
│   ├── main.pdf                       # Compiled 9-page research paper
│   ├── main.tex                       # Complete LaTeX source (IEEE/ACM format)
│   ├── references.bib                 # Verified bibliography (11 citations)
│   └── figures/                       # Publication vector figures (PDF & PNG)
├── docs/                              # Experimental logs & visual assets
│   ├── ablation_results.json          # Complete raw ablation data (106 test fonts)
│   └── figures/                       # Benchmark plots & system architecture diagrams
├── DESIGN.md                          # Technical specification manual (v1.0 -> v3.0.0)
├── pyproject.toml                     # Package definition & dependencies
├── glyphclip/
│   ├── canvas/
│   │   ├── charset.py                 # Glyph subsets (62 full / 52 alpha / 10 core) & entropy weights
│   │   ├── font_parser.py             # fontTools wrapper, OS/2 weight & fvar axis parsing
│   │   ├── renderer.py                # Legacy 1-ch FreeType raster fallback renderer
│   │   └── sdf.py                     # 2-ch Bézier Vector-SDF renderer & edge orthogonality trait
│   ├── models/
│   │   ├── convnext.py                # Shared GlyphConvNeXt (~20.58M params) + 52-class char_head
│   │   ├── mobilenet_v4.py            # Legacy lightweight MobileNetV4 backbone (4.4M params)
│   │   ├── dual_encoder.py            # Shared/dual encoder wrapper & 1-ch warmstart loader
│   │   └── loss.py                    # Symmetric temperature-scaled InfoNCE loss
│   ├── pipeline/
│   │   ├── binarization.py            # Otsu polarity pre-inversion & scale-adaptive Sauvola
│   │   ├── segmentation.py            # Deskew, CCA, dot merging & isotropic TSDF normalization
│   │   ├── degradation.py             # Multi-scale stroke morphology, pixel-preserving blur & noise
│   │   ├── dataset.py                 # Family-disjoint splitter with spaced upright weight selection
│   │   └── evaluator.py               # Held-out zero-shot evaluation harness
│   ├── index/
│   │   ├── vector_store.py            # O(1) char-bucketed embeddings store & IVF-Cosine ANN index
│   │   └── consensus.py               # Bayesian multi-glyph consensus (entropy, missing, ortho trait)
│   ├── server/
│   │   ├── app.py                     # FastAPI REST microservice (/ui, /v1/recognize, /v1/index/*)
│   │   ├── schemas.py                 # Pydantic request/response schemas
│   │   └── static/index.html          # Single-page Tailwind Web UI
│   └── export/
│       └── onnx_exporter.py           # 1-ch / 2-ch TorchScript & ONNX exporter
├── tests/
│   ├── test_renderer.py               # Canvas rendering & normalization assertions
│   ├── test_sdf.py                    # Vector-SDF winding, TSDF & orthogonality tests
│   ├── test_models.py                 # Backbone, InfoNCE loss & degradation tests
│   ├── test_pipeline.py               # Binarization, dataset split & visual segmentation tests
│   ├── test_consensus.py              # Vector store ANN & Bayesian consensus tests
│   └── test_api.py                    # API endpoints & TorchScript export tests
└── scripts/
    ├── run_ablations.py               # End-to-end ablation harness (106 test fonts)
    ├── train_sdf.py                   # v2.2 / v3.0 2-ch Vector-SDF training script
    ├── train_scaled.py                # v1.1 scaled 1-ch raster ConvNeXt training script
    ├── train_local_fonts.py           # v1.0 initial local MobileNetV4 prototype trainer
    ├── build_production_catalog.py    # Curated 28-family production index builder
    ├── index_catalog.py               # Batch directory indexer
    ├── test_sdf_production.py         # Curated production suite verification (11/11 tests)
    ├── test_client.py                 # Live /v1/recognize HTTP test client
    └── debug/                         # Diagnostic and visualization tools
```

---

## Quickstart

### 1. Run Unit Tests

```bash
python -m unittest discover tests -v
```

### 2. Launch Web Interface & FastAPI Microservice

```bash
python -m uvicorn glyphclip.server.app:app --host 127.0.0.1 --port 8080
```

* **Web Interface:** [http://127.0.0.1:8080/](http://127.0.0.1:8080/)
* **Interactive API Docs:** [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)
* **Health Endpoint:** [http://127.0.0.1:8080/health](http://127.0.0.1:8080/health)

### 3. Build Catalog & Query via API

* **Build curated production index:**
  ```bash
  python scripts/build_production_catalog.py
  ```
* **Zero-Shot Font Ingestion (`POST /v1/index/font` or `/v1/index/fonts`):**
  ```bash
  curl -X POST "http://127.0.0.1:8080/v1/index/font" \
       -F "file=@/path/to/Inter-Bold.ttf" \
       -F 'metadata={"family": "Inter", "license": "OFL-1.1"}'
  ```
* **Recognize a Word Crop (`POST /v1/recognize`):**
  ```bash
  curl -X POST "http://127.0.0.1:8080/v1/recognize" \
       -F "file=@/path/to/query_word.png" \
       -F "top_k=5"
  ```

### 4. Reproduce Ablation Suite

```bash
python scripts/run_ablations.py --save-dir docs
```

### 5. Train From Scratch (`v3.0`)

```bash
python scripts/train_sdf.py --scratch --max-fonts 0 --max-per-family 4 --max-weight-gap 400 --p-cross-weight 0.45 --epochs 12 --lr 3e-4 --ckpt-name glyphclip_v3.pt
```

See [`DESIGN.md`](DESIGN.md) for the complete `v1.0 -> v3.0.0` technical architecture specification.

---

## Citation

If you use GlyphCLIP in your research or applications, please cite:

```bibtex
@article{memmou2026glyphclip,
  title={{GlyphCLIP: Contrastive Blind Zero-Shot Font Retrieval with Analytical B{\'e}zier-SDF Representations}},
  author={Memmou, Abderrahmane},
  institution={National Higher School of Artificial Intelligence, Algiers},
  year={2026},
  month={November}
}
```

---

## License

Apache License 2.0. Open-source font matching engine developed for Project Typographic.
