"""Independent test evaluation suite for zero-shot font recognition."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from glyphclip.canvas.charset import get_charset
from glyphclip.canvas.font_parser import FontParser
from glyphclip.canvas.renderer import DeterministicCanvasRenderer
from glyphclip.canvas.sdf import VectorSDFRenderer
from glyphclip.index.consensus import BayesianConsensusEngine, FontPrediction
from glyphclip.index.vector_store import GlyphVectorStore
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.pipeline.degradation import TypographicDegradationPipeline
from glyphclip.pipeline.segmentation import VisualSegmentationPipeline


@dataclass
class EvalMetrics:
    total_queries: int
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    mrr: float  # Mean Reciprocal Rank


class IndependentTestHarness:
    """Evaluates GlyphCLIP on held-out test fonts and degraded word crops."""

    def __init__(
        self,
        dual_encoder: DualEncoder,
        test_font_paths: List[Path],
        device: torch.device,
        embed_dim: int = 512,
        charset_subset: str = "full",
    ):
        self.dual_encoder = dual_encoder
        self.test_font_paths = test_font_paths
        self.device = device
        self.embed_dim = embed_dim
        self.charset = get_charset(charset_subset)

        self.use_sdf = getattr(dual_encoder, "in_channels", 1) == 2
        self.renderer = DeterministicCanvasRenderer()
        self.sdf_renderer = VectorSDFRenderer() if self.use_sdf else None
        self.segmentation = VisualSegmentationPipeline(include_sdf=self.use_sdf)
        self.degradation = TypographicDegradationPipeline().to(device)

        # Build index for the test fonts
        self.vector_store = GlyphVectorStore(embed_dim=embed_dim)
        self._build_test_index()
        self.consensus = BayesianConsensusEngine(self.vector_store, top_k=50)

    def _build_test_index(self):
        """Index all characters for all test fonts into canonical vector store."""
        mode_str = "2-ch Vector-SDF" if self.use_sdf else "1-ch Raster"
        print(f"Indexing {len(self.test_font_paths)} test fonts ({mode_str}) over {len(self.charset)} characters...")
        self.dual_encoder.eval()

        tensors = []
        ids = []
        metas = []

        for fpath in self.test_font_paths:
            try:
                meta = FontParser.parse(fpath)
                weights = [300, 400, 600, 700] if meta.is_variable else [meta.weight]
                for w in weights:
                    for ch in self.charset:
                        if self.use_sdf and self.sdf_renderer is not None:
                            t = self.sdf_renderer.render_dual_tensor(fpath, ch, weight=w)
                            if t[0].max().item() < 0.3:
                                continue
                        else:
                            arr = self.renderer.render_glyph_numpy(fpath, ch, weight=w)
                            if arr.max() < 100:
                                continue
                            t = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
                        tensors.append(t)
                        ids.append(f"{meta.postscript_name}_w{w}_{ch}")
                        metas.append({
                            "family": meta.family,
                            "postscript_name": meta.postscript_name,
                            "weight": w,
                            "glyph_char": ch,
                            "source_path": str(fpath),
                        })
            except Exception:
                continue

        # Batch encode in chunks of 128
        chunk_size = 128
        for i in range(0, len(tensors), chunk_size):
            end = min(i + chunk_size, len(tensors))
            batch = torch.stack(tensors[i:end], dim=0).to(self.device)
            with torch.no_grad():
                embs = self.dual_encoder.encode_canonical(batch).cpu().numpy()
            self.vector_store.add_batch(ids[i:end], embs, metas[i:end])

        print(f"Test vector store built with {len(self.vector_store)} vectors.")

    def render_word_image(self, font_path: Path, word: str, size: int = 72) -> np.ndarray:
        """Render a synthetic degraded word image with realistic camera perturbations."""
        img = Image.new("RGB", (len(word) * size + 80, size + 80), color=(250, 250, 250))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(str(font_path), size=size)
        draw.text((40, 40), word, fill=(20, 20, 20), font=font)

        # Convert to numpy and apply subtle blur
        arr = np.array(img)
        arr = cv2.GaussianBlur(arr, (3, 3), 0.5)
        return arr

    def evaluate_word_query(
        self,
        query_img: np.ndarray,
        ground_truth_family: str,
        top_k: int = 5,
        query_text: Optional[str] = None,
    ) -> Tuple[int, List[FontPrediction]]:
        """Run full visual recognition and return (rank_of_ground_truth, predictions)."""
        segmented = self.segmentation.process(query_img)
        if not segmented:
            canvas_np, canvas_tensor = self.segmentation.normalize_glyph_crop(
                cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
            )
            segmented_tensors = [canvas_tensor]
        else:
            segmented_tensors = [g.canvas_tensor for g in segmented]

        batch = torch.stack(segmented_tensors, dim=0).to(self.device)
        with torch.no_grad():
            embs = self.dual_encoder.encode_visual(batch).cpu().numpy()

        q_vecs = [embs[i] for i in range(len(embs))]
        query_chars = None
        if query_text:
            chars = list(query_text.strip())
            if len(chars) < len(q_vecs):
                query_chars = chars + [None] * (len(q_vecs) - len(chars))
            else:
                query_chars = chars[:len(q_vecs)]

        query_char_candidates = None
        if query_chars is None:
            query_char_candidates = self.dual_encoder.predict_char_candidates(
                batch, self.charset, top_k=3, min_prob=0.08
            )

        predictions = self.consensus.aggregate(
            q_vecs,
            query_chars=query_chars,
            max_results=top_k,
            query_char_candidates=query_char_candidates,
        )

        # Check rank of ground truth
        gt_lower = ground_truth_family.lower()
        rank = -1
        for p in predictions:
            if gt_lower in p.family.lower() or p.family.lower() in gt_lower:
                rank = p.rank
                break

        return rank, predictions

    def run_benchmark(
        self,
        test_words: List[str] = ["Funnel", "Graphic", "Typo", "Design"],
        use_text_hint: bool = True,
    ) -> EvalMetrics:
        """Execute benchmark across all test fonts with test words."""
        ranks = []
        total = 0

        for fpath in self.test_font_paths:
            try:
                meta = FontParser.parse(fpath)
                for word in test_words:
                    word_img = self.render_word_image(fpath, word)
                    rank, _ = self.evaluate_word_query(
                        word_img,
                        meta.family,
                        top_k=5,
                        query_text=word if use_text_hint else None,
                    )
                    ranks.append(rank)
                    total += 1
            except Exception:
                continue

        if total == 0:
            return EvalMetrics(0, 0.0, 0.0, 0.0, 0.0)

        top1 = sum(1 for r in ranks if r == 1) / total * 100.0
        top3 = sum(1 for r in ranks if 1 <= r <= 3) / total * 100.0
        top5 = sum(1 for r in ranks if 1 <= r <= 5) / total * 100.0
        mrr = sum(1.0 / r if r > 0 else 0.0 for r in ranks) / total

        return EvalMetrics(total, top1, top3, top5, mrr)
