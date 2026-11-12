"""Unit tests for analytical vector SDF rendering, dual-channel segmentation, and 2-ch ConvNeXt."""

import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from glyphclip.canvas.charset import get_charset
from glyphclip.canvas.sdf import (
    VectorSDFRenderer,
    batch_occupancy_to_dual,
    binary_mask_to_tsdf,
    sdf_to_tsdf,
)
from glyphclip.index.consensus import BayesianConsensusEngine
from glyphclip.index.vector_store import GlyphVectorStore
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.pipeline.segmentation import VisualSegmentationPipeline


class TestVectorSDFPipeline(unittest.TestCase):
    """Test suite for 2-channel Vector-SDF geometric bridge."""

    def setUp(self):
        self.font_path = None
        for p in [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\times.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
        ]:
            if Path(p).exists():
                self.font_path = p
                break

    def test_tsdf_normalization_and_binary_edt(self):
        """Verify TSDF values range in [0, 1] and sit near 0.5 along contour edges."""
        sdf_zero = np.array([[0.0]], dtype=np.float32)
        self.assertAlmostEqual(float(sdf_to_tsdf(sdf_zero)[0, 0]), 0.5, places=5)

        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[60:160, 80:140] = 255
        tsdf = binary_mask_to_tsdf(mask, sigma=8.0)
        self.assertEqual(tsdf.shape, (224, 224))
        self.assertGreater(tsdf[110, 110], 0.9)  # Deep inside foreground
        self.assertLess(tsdf[10, 10], 0.05)      # Deep outside in background
        self.assertGreater(tsdf[60, 110], 0.45)  # Boundary pixel slightly inside (>0.5)
        self.assertLess(tsdf[59, 110], 0.55)     # Boundary adjacent outside (<0.5)

        # Edge cases: all black and all white
        empty_tsdf = binary_mask_to_tsdf(np.zeros((64, 64), dtype=np.uint8))
        self.assertEqual(float(empty_tsdf.max()), 0.0)
        full_tsdf = binary_mask_to_tsdf(np.full((64, 64), 255, dtype=np.uint8))
        self.assertEqual(float(full_tsdf.min()), 1.0)

    def test_vector_sdf_renderer_from_font_file(self):
        """Verify VectorSDFRenderer extracts splines from TTF and renders [2, 224, 224]."""
        if not self.font_path:
            self.skipTest("No local Windows font found.")

        renderer = VectorSDFRenderer(supersample=3)
        tensor = renderer.render_dual_tensor(self.font_path, "g")
        self.assertEqual(tensor.shape, (2, 224, 224))
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)
        # Ink occupancy channel should have strong ink presence
        self.assertGreater(tensor[0].max().item(), 0.9)
        # TSDF channel should vary smoothly across interior and exterior
        self.assertLess(tensor[1, 0, 0].item(), 0.1)
        self.assertGreater(tensor[1].max().item(), 0.65)

        # Space character returns empty zero channels cleanly
        space = renderer.render_dual_tensor(self.font_path, " ")
        self.assertEqual(space.max().item(), 0.0)

    def test_segmentation_sdf_and_dot_merging(self):
        """Verify include_sdf=True outputs [2, 224, 224] and merges stacked dots (e.g. 'i')."""
        seg = VisualSegmentationPipeline(include_sdf=True, min_area=50)

        # Draw a synthetic dotted 'i' next to an 'I' on white canvas
        canvas = np.full((120, 200), 255, dtype=np.uint8)
        # 'i': dot + stem
        cv2.circle(canvas, (45, 30), 5, 0, -1)
        cv2.rectangle(canvas, (41, 45), (49, 95), 0, -1)
        # 'I'
        cv2.rectangle(canvas, (110, 25), (125, 95), 0, -1)

        glyphs = seg.process(canvas)
        self.assertEqual(len(glyphs), 2)
        self.assertEqual(glyphs[0].canvas_tensor.shape, (2, 224, 224))
        self.assertEqual(glyphs[1].canvas_tensor.shape, (2, 224, 224))
        # First glyph bbox should span both dot (y<=30) and stem (y=95)
        self.assertLessEqual(glyphs[0].bbox[1], 30)
        self.assertGreaterEqual(glyphs[0].bbox[1] + glyphs[0].bbox[3], 92)

    def test_dual_encoder_2ch_and_warmstart(self):
        """Verify DualEncoder(in_channels=2) forward pass and 1-ch weight warmstart."""
        dual_1ch = DualEncoder(embed_dim=128, backbone_type="convnext", in_channels=1)
        dual_2ch = DualEncoder(embed_dim=128, backbone_type="convnext", in_channels=2)

        dual_2ch.load_warmstart_from_1ch(dual_1ch.state_dict())

        # Check batch_occupancy_to_dual helper
        occ = torch.rand(2, 1, 224, 224)
        dual_in = batch_occupancy_to_dual(occ)
        self.assertEqual(dual_in.shape, (2, 2, 224, 224))

        z_v, z_c = dual_2ch(dual_in, dual_in)
        self.assertEqual(z_v.shape, (2, 128))
        self.assertEqual(z_c.shape, (2, 128))
        norms = torch.norm(z_v, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_aux_char_head_and_multichar_consensus(self):
        """Verify 52-char auxiliary OCR head, candidate generation, and multi-char consensus filtering."""
        charset = get_charset("alpha")
        dual = DualEncoder(
            embed_dim=64, backbone_type="convnext", in_channels=2, num_chars=len(charset)
        )
        dual.eval()

        x = torch.rand(2, 2, 224, 224)
        z_v, logits = dual.encode_visual_with_chars(x)
        self.assertEqual(z_v.shape, (2, 64))
        self.assertIsNotNone(logits)
        self.assertEqual(logits.shape, (2, len(charset)))

        cands = dual.predict_char_candidates(x, charset, top_k=3, min_prob=0.0)
        self.assertIsNotNone(cands)
        self.assertEqual(len(cands), 2)
        for row in cands:
            self.assertEqual(len(row), 3)
            for ch in row:
                self.assertIn(ch, charset)

        # Verify multi-char filter_char list + soft consensus ranking
        vs = GlyphVectorStore(embed_dim=4)
        vecs = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],    # FontA 'I' (true match)
                [0.6, 0.8, 0.0, 0.0],    # FontB 'I'
                [0.0, 1.0, 0.0, 0.0],    # FontA 'l'
                [0.99, 0.141, 0.0, 0.0], # FontC 'Q' (spurious cross-char hit)
            ],
            dtype=np.float32,
        )
        metas = [
            {"family": "FontA", "postscript_name": "FontA", "glyph_char": "I"},
            {"family": "FontB", "postscript_name": "FontB", "glyph_char": "I"},
            {"family": "FontA", "postscript_name": "FontA", "glyph_char": "l"},
            {"family": "FontC", "postscript_name": "FontC", "glyph_char": "Q"},
        ]
        vs.add_batch(["a_I", "b_I", "a_l", "c_Q"], vecs, metas)

        # Multi-char filter allows 'I' and 'l' while excluding 'Q'
        hits = vs.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=5, filter_char=["I", "l"])
        hit_chars = {r.metadata["glyph_char"] for r in hits}
        self.assertNotIn("Q", hit_chars)
        self.assertIn("I", hit_chars)

        engine = BayesianConsensusEngine(vs)
        preds = engine.aggregate(
            [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)],
            query_chars=None,
            max_results=3,
            query_char_candidates=[["l", "I"]],
        )
        self.assertGreater(len(preds), 0)
        self.assertEqual(preds[0].family, "FontA")
        self.assertNotIn("FontC", [p.family for p in preds])


if __name__ == "__main__":
    unittest.main()
