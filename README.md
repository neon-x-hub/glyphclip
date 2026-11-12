# Project Typographic: GlyphCLIP (v3.0)

> **Open-Source Contrastive Dual-Encoder Font Matching Engine with 2-Channel Analytical Vector-SDF & Zero-Shot Ingestion**

GlyphCLIP decouples font identification from closed-set classification catalogs. Instead of retraining a classifier when new typefaces appear, GlyphCLIP maps glyph geometry onto a 512-dimensional unit hypersphere ($\mathbb{S}^{511}$) using a shared `GlyphConvNeXt` backbone (`~20.58M` parameters) trained contrastive-wise from scratch across wide weight gaps (`|w1 - w2| <= 400`). New `.ttf` and `.otf` fonts are ingested zero-shot into a local `IVF-Cosine` ANN vector store with zero GPU retraining.

---

## Key Capabilities

* **2-Channel Analytical Vector-SDF Geometric Bridge:** Parses `.ttf`/`.otf` spline contours directly (`fontTools` `BasePen`, `4x` subpixel winding, precise Euclidean TSDF) and pairs them symmetrically with polarity-aware, scale-adaptive query binarization (`[Occupancy, Smooth TSDF]`).
* **Blind Auto-Detection & Hint-Locked Modes:** A decoupled 52-class auxiliary character head (`char_head`) supports zero-hint word recognition (`91.9%` Top-1 / `97.1%` Top-3 / `0.9444` MRR on 96 unseen test families), while explicit text hints lock per-glyph character matching cleanly.
* **Cross-Weight & Style Invariance:** From-scratch family-disjoint training across evenly spaced upright weights (`max_per_family=4`, `--max-weight-gap 400`, `--p-cross-weight 0.45`) with multi-scale stroke morphology (`k=3..9`) achieves `86.7%` Top-1 / `96.7%` Top-3 on `Regular 400 -> Bold 700` cross-weight retrieval, alongside `ortho_ratio` edge-orthogonality scoring for bitmap/pixel fonts.
* **Fast Retrieval & Zero-Shot Persistence:** $O(1)$ character-bucketed lookup backed by spherical K-Means `IVF-Cosine` ANN (`ann_index.npz`), plus persistent single (`/v1/index/font`) and batch (`/v1/index/fonts`) ingestion endpoints and interactive Web UI.
* **Deployment Ready:** CPU/GPU accelerated FastAPI microservice and TorchScript/ONNX serialization.

---

## Directory Structure

```
GlyphCLIP/
├── DESIGN.md                          # Technical specification manual (v3.0.0)
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
    ├── train_sdf.py                   # v2.2 / v3.0 2-ch Vector-SDF training & benchmark suite
    ├── train_scaled.py                # v1.1 scaled 1-ch raster ConvNeXt training script
    ├── train_local_fonts.py           # v1.0 initial local MobileNetV4 prototype trainer
    ├── train.py                       # Minimal contrastive training harness
    ├── build_production_catalog.py    # Curated 28-family production index builder
    ├── index_catalog.py               # Batch directory indexer
    ├── test_sdf_production.py         # Curated production suite verification
    ├── test_client.py                 # Live /v1/recognize HTTP test client
    └── debug/                         # One-off diagnostic scripts
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

### 4. Train From Scratch (`v3.0`)

```bash
python scripts/train_sdf.py --scratch --max-fonts 0 --max-per-family 4 --max-weight-gap 400 --p-cross-weight 0.45 --epochs 12 --lr 3e-4 --ckpt-name glyphclip_v3.pt
```

See [`DESIGN.md`](DESIGN.md) for the complete `v1.0 -> v3.0.0` technical architecture specification and benchmark suite.

---

## License

Apache License 2.0. Open-source font matching engine developed for Project Typographic.
