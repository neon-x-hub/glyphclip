"""FastAPI microservice implementation for font recognition and zero-shot indexing."""

import io
import json
import time
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import glyphclip
from glyphclip.canvas.charset import get_charset
from glyphclip.canvas.font_parser import FontParser
from glyphclip.canvas.renderer import DeterministicCanvasRenderer
from glyphclip.canvas.sdf import VectorSDFRenderer, compute_edge_orthogonality
from glyphclip.index.consensus import BayesianConsensusEngine
from glyphclip.index.vector_store import GlyphVectorStore
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.pipeline.segmentation import VisualSegmentationPipeline
from glyphclip.server.schemas import (
    BatchIndexFontResponse,
    GlyphScore,
    HealthResponse,
    IndexFontResponse,
    PredictionItem,
    RecognizeResponse,
)


def create_app(
    embed_dim: int = 512,
    backbone_type: str = "convnext",
    weights_path: Optional[str] = None,
    index_path: Optional[str] = None,
    device: Optional[str] = None,
) -> FastAPI:
    """Application factory for GlyphCLIP inference service."""
    app = FastAPI(
        title="GlyphCLIP Typographic Matching Engine",
        version=glyphclip.__version__,
        description="Open-Source Contrastive Dual-Encoder Font Recognition & Zero-Shot Indexing Service",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device

    torch_device = torch.device(device_str)

    # Auto-discover default checkpoint and index if available
    from pathlib import Path
    default_sdf = Path("checkpoints/glyphclip_sdf.pt")
    default_scaled = Path("checkpoints/glyphclip_scaled.pt")
    if default_sdf.exists():
        default_ckpt = default_sdf
    elif default_scaled.exists():
        default_ckpt = default_scaled
    else:
        default_ckpt = Path("checkpoints/glyphclip_trained.pt")
    default_idx = Path("font_index")

    if weights_path is None and default_ckpt.exists():
        weights_path = str(default_ckpt)
    if index_path is None and (default_idx / "metadata.json").exists():
        index_path = str(default_idx)

    if index_path and (Path(index_path) / "metadata.json").exists():
        vector_store = GlyphVectorStore.load(index_path)
        embed_dim = vector_store.embed_dim
    else:
        vector_store = GlyphVectorStore(embed_dim=embed_dim)

    in_channels = 1
    num_chars = 0
    state_dict = None
    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch_device, weights_only=True)
        stem_w = state_dict.get("visual_encoder.downsample_layers.0.0.weight")
        if stem_w is not None and stem_w.ndim == 4:
            in_channels = int(stem_w.shape[1])
        char_head_w = state_dict.get("visual_encoder.char_head.weight")
        if char_head_w is not None and char_head_w.ndim == 2:
            num_chars = int(char_head_w.shape[0])

    # Initialize shared components
    dual_encoder = DualEncoder(
        embed_dim=embed_dim,
        backbone_type=backbone_type,
        shared_backbone=True,
        in_channels=in_channels,
        num_chars=num_chars,
    ).to(torch_device)

    if state_dict is not None:
        dual_encoder.load_state_dict(state_dict)
    dual_encoder.eval()

    use_sdf = (in_channels == 2)
    consensus_engine = BayesianConsensusEngine(vector_store, top_k=50, missing_penalty_lambda=0.5)
    segmentation_pipeline = VisualSegmentationPipeline(include_sdf=use_sdf)
    canvas_renderer = DeterministicCanvasRenderer()
    sdf_renderer = VectorSDFRenderer() if use_sdf else None

    # Store state on app.state
    app.state.device = torch_device
    app.state.dual_encoder = dual_encoder
    app.state.vector_store = vector_store
    app.state.index_path = index_path
    app.state.consensus_engine = consensus_engine
    app.state.segmentation_pipeline = segmentation_pipeline
    app.state.canvas_renderer = canvas_renderer
    app.state.sdf_renderer = sdf_renderer
    app.state.use_sdf = use_sdf

    @app.get("/health", response_model=HealthResponse)
    def health_check():
        return HealthResponse(
            status="healthy",
            version=glyphclip.__version__,
            device=str(app.state.device),
            total_vectors_indexed=len(app.state.vector_store),
        )

    @app.get("/", response_class=HTMLResponse)
    @app.get("/ui", response_class=HTMLResponse)
    def serve_ui():
        """Serve the GlyphCLIP web interface."""
        static_html = Path(__file__).parent / "static" / "index.html"
        if static_html.exists():
            with open(static_html, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>GlyphCLIP UI</h1><p>Static index.html not found.</p>")

    @app.get("/v1/catalog")
    def get_catalog(limit: int = 150, query: Optional[str] = None):
        """Return distinct indexed font families and their metadata."""
        families = {}
        for meta in app.state.vector_store.metadata:
            fam = meta.get("family", "Unknown")
            if query and query.lower() not in fam.lower():
                continue
            if fam not in families:
                families[fam] = {
                    "family": fam,
                    "postscript_name": meta.get("postscript_name", fam),
                    "weights": set(),
                    "glyphs": set(),
                    "license": meta.get("license"),
                }
            if meta.get("weight"):
                families[fam]["weights"].add(meta.get("weight"))
            if meta.get("glyph_char"):
                families[fam]["glyphs"].add(meta.get("glyph_char"))

        items = []
        for fam, d in sorted(families.items())[:limit]:
            items.append({
                "family": d["family"],
                "postscript_name": d["postscript_name"],
                "weights": sorted(list(d["weights"])) if d["weights"] else [400],
                "glyph_count": len(d["glyphs"]),
                "license": d["license"],
            })

        return {
            "total_families": len(families),
            "items": items,
        }

    @app.post("/v1/recognize", response_model=RecognizeResponse)
    async def recognize(
        file: UploadFile = File(...),
        query_text: Optional[str] = Form(default=None),
        top_k: int = Form(default=5),
        include_metrics: bool = Form(default=False),
    ):
        """Recognize font family from a raster word image or photographic crop."""
        t0 = time.perf_counter()
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Decode image using OpenCV
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image.")

        # 1. Visual segmentation
        segmented = app.state.segmentation_pipeline.process(img)
        if not segmented:
            # Fallback: if no multi-glyph contours found, treat entire crop as single glyph
            canvas_np, canvas_tensor = app.state.segmentation_pipeline.normalize_glyph_crop(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            )
            segmented_tensors = [canvas_tensor]
        else:
            segmented_tensors = [g.canvas_tensor for g in segmented]

        # 2. Encode visual query crops via E_v
        batch_tensors = torch.stack(segmented_tensors, dim=0).to(app.state.device)
        with torch.no_grad():
            embeddings = app.state.dual_encoder.encode_visual(batch_tensors)
            embeddings_np = embeddings.cpu().numpy()

        query_vectors = [embeddings_np[i] for i in range(len(embeddings_np))]

        # Parse query characters if provided, ignoring skipped spaces/punctuation
        query_chars = None
        if isinstance(query_text, str) and query_text.strip():
            nonspace = [c for c in query_text.strip() if not c.isspace()]
            alpha_only = [c for c in nonspace if c.isalpha()]
            if len(alpha_only) == len(query_vectors):
                query_chars = alpha_only
            elif len(nonspace) == len(query_vectors):
                query_chars = [c if c.isalpha() else None for c in nonspace]
            else:
                sanitized = [c if c.isalpha() else None for c in nonspace]
                if len(sanitized) < len(query_vectors):
                    query_chars = sanitized + [None] * (len(query_vectors) - len(sanitized))
                else:
                    query_chars = sanitized[:len(query_vectors)]

        # 3. Consensus scoring (with auxiliary OCR candidate predictions when no hint provided)
        query_char_candidates = None
        if query_chars is None:
            query_char_candidates = app.state.dual_encoder.predict_char_candidates(
                batch_tensors, get_charset("alpha"), top_k=3, min_prob=0.08
            )

        query_ortho_ratios = [compute_edge_orthogonality(t[0]) for t in segmented_tensors]
        raw_predictions = app.state.consensus_engine.aggregate(
            query_vectors,
            query_chars=query_chars,
            max_results=top_k,
            query_char_candidates=query_char_candidates,
            query_ortho_ratios=query_ortho_ratios,
        )

        formatted_predictions = []
        for p in raw_predictions:
            per_glyph = {
                k: GlyphScore(char=v["char"], similarity=v["similarity"])
                for k, v in p.per_glyph_scores.items()
            }
            formatted_predictions.append(
                PredictionItem(
                    rank=p.rank,
                    family=p.family,
                    postscript_name=p.postscript_name,
                    weight=p.weight,
                    style=p.style,
                    confidence=p.confidence,
                    per_glyph_scores=per_glyph,
                    license=p.license,
                    download_url=p.download_url,
                )
            )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return RecognizeResponse(
            status="success",
            latency_ms=latency_ms,
            characters_detected=len(segmented_tensors),
            predictions=formatted_predictions,
        )

    def _ingest_font_bytes(contents: bytes, metadata: Optional[str], persist: bool = True) -> IndexFontResponse:
        user_meta = {}
        if isinstance(metadata, str) and metadata.strip():
            try:
                user_meta = json.loads(metadata)
            except Exception:
                pass

        try:
            parsed_meta = FontParser.parse(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse font: {str(e)}")

        family = user_meta.get("family") or parsed_meta.family
        license_info = user_meta.get("license") or parsed_meta.license
        download_url = user_meta.get("download_url")

        weights = FontParser.get_supported_weights(parsed_meta)
        charset = get_charset()

        indexed_count = 0
        canonical_tensors = []
        record_ids = []
        metadatas = []

        for w in weights:
            for char in charset:
                rec_id = f"{parsed_meta.postscript_name}_w{w}_{char}"
                try:
                    if app.state.use_sdf and app.state.sdf_renderer is not None:
                        tensor = app.state.sdf_renderer.render_dual_tensor(
                            contents, char=char, weight=w
                        )
                    else:
                        tensor = app.state.canvas_renderer.render_glyph_tensor(
                            contents, char=char, weight=w
                        )
                    canonical_tensors.append(tensor)
                    record_ids.append(rec_id)
                    metadatas.append({
                        "family": family,
                        "postscript_name": parsed_meta.postscript_name,
                        "weight": w,
                        "italic": parsed_meta.is_italic,
                        "glyph_char": char,
                        "license": license_info,
                        "download_url": download_url,
                        "ortho_ratio": round(compute_edge_orthogonality(tensor[0]), 4),
                    })
                except Exception:
                    continue

        persisted = False
        if canonical_tensors:
            batch = torch.stack(canonical_tensors, dim=0).to(app.state.device)
            with torch.no_grad():
                embs = app.state.dual_encoder.encode_canonical(batch).cpu().numpy()
            app.state.vector_store.add_batch(record_ids, embs, metadatas)
            indexed_count = len(record_ids)
            if persist and app.state.index_path:
                app.state.vector_store.save(app.state.index_path)
                persisted = True

        return IndexFontResponse(
            status="success",
            family=family,
            postscript_name=parsed_meta.postscript_name,
            indexed_glyphs=indexed_count,
            weights_indexed=weights,
            total_vectors_in_store=len(app.state.vector_store),
            persisted=persisted,
        )

    @app.post("/v1/index/font", response_model=IndexFontResponse)
    async def index_font(
        file: UploadFile = File(...),
        metadata: Optional[str] = Form(default=None),
    ):
        """Index a .ttf / .otf typeface directly into canonical vector space and persist to disk."""
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded font file is empty.")
        return _ingest_font_bytes(contents, metadata, persist=True)

    @app.post("/v1/index/fonts", response_model=BatchIndexFontResponse)
    async def index_fonts_batch(
        files: List[UploadFile] = File(...),
        metadata: Optional[str] = Form(default=None),
    ):
        """Index multiple .ttf / .otf files (e.g. multiple static weights of a family) and save once."""
        if not files:
            raise HTTPException(status_code=400, detail="No font files uploaded.")

        results: List[IndexFontResponse] = []
        for f in files:
            contents = await f.read()
            if not contents:
                raise HTTPException(status_code=400, detail=f"Uploaded font file '{f.filename}' is empty.")
            results.append(_ingest_font_bytes(contents, metadata, persist=False))

        persisted = False
        if results and app.state.index_path:
            app.state.vector_store.save(app.state.index_path)
            persisted = True
            for r in results:
                r.persisted = True

        return BatchIndexFontResponse(
            status="success",
            files_indexed=len(results),
            results=results,
            total_vectors_in_store=len(app.state.vector_store),
            persisted=persisted,
        )

    return app


app = create_app()
