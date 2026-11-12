"""Unit tests for GlyphVectorStore and BayesianConsensusEngine."""

import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from glyphclip.index.consensus import BayesianConsensusEngine
from glyphclip.index.vector_store import GlyphVectorStore


class TestConsensus(unittest.TestCase):
    """Test suite for vector storage and multi-glyph consensus scoring."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store = GlyphVectorStore(embed_dim=256)

        # Populate store with synthetic font families
        # Font A: Inter
        vec_a_g = np.zeros(256, dtype=np.float32)
        vec_a_g[0] = 1.0
        self.store.add("inter_g", vec_a_g, {"family": "Inter", "postscript_name": "Inter-Regular", "weight": 400, "glyph_char": "g"})

        vec_a_e = np.zeros(256, dtype=np.float32)
        vec_a_e[1] = 1.0
        self.store.add("inter_e", vec_a_e, {"family": "Inter", "postscript_name": "Inter-Regular", "weight": 400, "glyph_char": "e"})

        # Font B: Roboto
        vec_b_g = np.zeros(256, dtype=np.float32)
        vec_b_g[2] = 1.0
        self.store.add("roboto_g", vec_b_g, {"family": "Roboto", "postscript_name": "Roboto-Regular", "weight": 400, "glyph_char": "g"})

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_vector_store_search_and_persistence(self):
        """Verify vector search and disk persistence/loading."""
        query = np.zeros(256, dtype=np.float32)
        query[0] = 1.0  # Matches Inter 'g'

        results = self.store.search(query, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "inter_g")
        self.assertAlmostEqual(results[0].similarity, 1.0, places=5)

        # Test Save & Load
        save_path = Path(self.tmp_dir) / "test_store"
        self.store.save(save_path)
        loaded = GlyphVectorStore.load(save_path)
        self.assertEqual(len(loaded), len(self.store))

        loaded_results = loaded.search(query, top_k=1)
        self.assertEqual(loaded_results[0].id, "inter_g")

    def test_bayesian_consensus_multi_glyph(self):
        """Verify consensus engine aggregates hits and selects dominant font family."""
        engine = BayesianConsensusEngine(self.store, top_k=10, missing_penalty_lambda=0.5)

        # Query 1: matches Inter 'g'
        q1 = np.zeros(256, dtype=np.float32)
        q1[0] = 1.0
        # Query 2: matches Inter 'e'
        q2 = np.zeros(256, dtype=np.float32)
        q2[1] = 1.0

        predictions = engine.aggregate([q1, q2], query_chars=["g", "e"], max_results=3)
        self.assertGreater(len(predictions), 0)

        # Top prediction should be Inter
        top = predictions[0]
        self.assertEqual(top.family, "Inter")
        self.assertEqual(top.rank, 1)
        self.assertGreater(top.confidence, 0.9)
        self.assertIn("c_0", top.per_glyph_scores)
        self.assertIn("c_1", top.per_glyph_scores)

    def test_ivf_ann_index_recall_and_persistence(self):
        """Verify Spherical K-Means IVF ANN builds, searches char buckets accurately, and persists."""
        rng = np.random.default_rng(7)
        dim = 64
        n_vecs = 1200
        store = GlyphVectorStore(embed_dim=dim, ann_threshold=200, nprobe_ratio=0.35)

        centers = rng.normal(size=(24, dim)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        chars = ["a", "g", "e", "Q", "R"]

        vecs = []
        ids = []
        metas = []
        for i in range(n_vecs):
            c = centers[i % len(centers)] + 0.15 * rng.normal(size=dim).astype(np.float32)
            c /= np.linalg.norm(c)
            ch = chars[i % len(chars)]
            vecs.append(c)
            ids.append(f"vec_{i}")
            metas.append({"family": f"Fam_{i % 60}", "glyph_char": ch, "weight": 400})

        store.add_batch(ids, np.stack(vecs, axis=0), metas)
        self.assertTrue(store.has_ann)

        # Verify global ANN recall against exact ground-truth top-1 across queries
        top1_hits = 0
        for q_idx in range(20):
            q = vecs[q_idx * 17] + 0.05 * rng.normal(size=dim).astype(np.float32)
            ann_res = store.search(q, top_k=10, use_ann=True)
            exact_res = store.search(q, top_k=10, use_ann=False)
            self.assertEqual(len(ann_res), 10)
            if exact_res[0].id in [r.id for r in ann_res[:3]]:
                top1_hits += 1
        self.assertGreaterEqual(top1_hits, 18)

        # Verify character-bucketed ANN filtering
        char_hits = store.search(vecs[1], top_k=15, filter_char="g", use_ann=True)
        self.assertTrue(all(r.metadata["glyph_char"] == "g" for r in char_hits))

        # Incremental add assigns to existing IVF cell cleanly
        store.add("vec_extra", vecs[0], {"family": "ExtraFam", "glyph_char": "a", "weight": 700})
        self.assertTrue(store.has_ann)
        self.assertEqual(len(store), n_vecs + 1)

        # Save & load preserves ann_index.npz
        save_path = Path(self.tmp_dir) / "ann_store"
        store.save(save_path)
        self.assertTrue((save_path / "ann_index.npz").exists())
        reloaded = GlyphVectorStore.load(save_path)
        self.assertTrue(reloaded.has_ann)
        self.assertEqual(len(reloaded), n_vecs + 1)
        rel_hits = reloaded.search(vecs[1], top_k=5, filter_char="g")
        self.assertTrue(all(r.metadata["glyph_char"] == "g" for r in rel_hits))

    def test_edge_orthogonality_trait_ranking(self):
        """Verify orthogonal pixel-font trait favors pixelated candidates over smooth ones."""
        from glyphclip.canvas.sdf import compute_edge_orthogonality

        # Check square pixel-step block vs diagonal rotated diamond
        pixel_mask = np.zeros((100, 100), dtype=np.float32)
        pixel_mask[20:80, 20:80] = 1.0
        diamond = np.zeros((100, 100), dtype=np.float32)
        cv_pts = np.array([[50, 15], [85, 50], [50, 85], [15, 50]], dtype=np.int32)
        import cv2
        cv2.fillPoly(diamond, [cv_pts], 1.0)
        self.assertGreater(compute_edge_orthogonality(pixel_mask), 0.9)
        self.assertLess(compute_edge_orthogonality(diamond), 0.3)

        # Consensus tie-break / style separation
        store = GlyphVectorStore(embed_dim=8)
        q = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        c_smooth = np.array([0.92, 0.39, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        c_pixel = np.array([0.82, 0.57, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        store.add("smooth_S", c_smooth, {"family": "SmoothSans", "glyph_char": "S", "weight": 400, "ortho_ratio": 0.40})
        store.add("pixel_S", c_pixel, {"family": "PixelFont", "glyph_char": "S", "weight": 400, "ortho_ratio": 0.96})

        engine = BayesianConsensusEngine(store, ortho_penalty_weight=0.6, ortho_deadband=0.18)
        preds = engine.aggregate([q], query_chars=["S"], query_ortho_ratios=[0.95], max_results=2)
        self.assertEqual(preds[0].family, "PixelFont")


if __name__ == "__main__":
    unittest.main()
