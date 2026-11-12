"""Unit tests for FastAPI endpoints and model serialization."""

import asyncio
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi import UploadFile

from glyphclip.export.onnx_exporter import TorchScriptExporter
from glyphclip.index.vector_store import GlyphVectorStore
from glyphclip.models.convnext import GlyphConvNeXt
from glyphclip.models.mobilenet_v4 import MobileNetV4Glyph
from glyphclip.server.app import create_app


class TestAPIAndSerialization(unittest.TestCase):
    """Test suite for microservice API and TorchScript export."""

    @classmethod
    def setUpClass(cls):
        cls.font_path = None
        candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]
        for p in candidates:
            if Path(p).exists():
                cls.font_path = p
                break

        cls.tmp_dir = tempfile.mkdtemp()
        cls.app = create_app(device="cpu", index_path=str(Path(cls.tmp_dir) / "test_index"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_health_endpoint(self):
        """Verify /health endpoint returns healthy status and system metrics."""
        health_func = None
        for route in self.app.routes:
            if getattr(route, "path", None) == "/health":
                health_func = route.endpoint
                break
        self.assertIsNotNone(health_func)
        resp = health_func()
        self.assertEqual(resp.status, "healthy")
        self.assertEqual(resp.version, "2.1.0")

    def test_index_and_recognize_flow(self):
        """Verify indexing a font and recognizing a query image."""
        if not self.font_path:
            raise unittest.SkipTest("No font found.")

        index_func = None
        recognize_func = None
        for route in self.app.routes:
            if getattr(route, "path", None) == "/v1/index/font":
                index_func = route.endpoint
            elif getattr(route, "path", None) == "/v1/recognize":
                recognize_func = route.endpoint

        self.assertIsNotNone(index_func)
        self.assertIsNotNone(recognize_func)

        # 1. Index font
        with open(self.font_path, "rb") as f:
            font_bytes = f.read()

        font_upload = UploadFile(
            file=io.BytesIO(font_bytes),
            filename="test_font.ttf",
            headers={"content-type": "font/ttf"},
        )

        index_resp = asyncio.run(index_func(file=font_upload, metadata=None))
        self.assertEqual(index_resp.status, "success")
        self.assertGreater(index_resp.indexed_glyphs, 0)
        self.assertGreater(index_resp.total_vectors_in_store, 0)
        self.assertTrue(index_resp.persisted)

        # Verify uploaded font persisted to disk on index_path
        reloaded_store = GlyphVectorStore.load(Path(self.tmp_dir) / "test_index")
        self.assertEqual(len(reloaded_store), index_resp.total_vectors_in_store)

        # Verify batch multi-file font ingestion (/v1/index/fonts)
        batch_func = None
        for route in self.app.routes:
            if getattr(route, "path", None) == "/v1/index/fonts":
                batch_func = route.endpoint
                break
        self.assertIsNotNone(batch_func)
        u1 = UploadFile(file=io.BytesIO(font_bytes), filename="f1.ttf")
        batch_resp = asyncio.run(batch_func(files=[u1], metadata=None))
        self.assertEqual(batch_resp.status, "success")
        self.assertEqual(batch_resp.files_indexed, 1)
        self.assertTrue(batch_resp.persisted)

        # 2. Render a query image to test recognition
        query_canvas = np.zeros((224, 224, 3), dtype=np.uint8)
        # Draw stroke simulating letter
        cv2.putText(
            query_canvas, "g", (60, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 255, 255), 8
        )
        _, img_encoded = cv2.imencode(".png", query_canvas)

        img_upload = UploadFile(
            file=io.BytesIO(img_encoded.tobytes()),
            filename="query.png",
            headers={"content-type": "image/png"},
        )

        rec_resp = asyncio.run(recognize_func(file=img_upload, top_k=3, include_metrics=True))
        self.assertEqual(rec_resp.status, "success")
        self.assertGreaterEqual(rec_resp.characters_detected, 1)
        self.assertGreater(len(rec_resp.predictions), 0)
        self.assertEqual(rec_resp.predictions[0].rank, 1)
        self.assertIn(rec_resp.predictions[0].family, [index_resp.family, "Arial", "Calibri", "Segoe UI"])

    def test_torchscript_export_and_benchmark(self):
        """Verify model exports to TorchScript, matches numerically, and runs efficiently on CPU."""
        model = MobileNetV4Glyph().eval()
        export_path = Path(self.tmp_dir) / "mobilenet_v4.pt"

        # Export
        TorchScriptExporter.export(model, export_path, device="cpu")
        self.assertTrue(export_path.exists())

        # Verify numerical equivalence
        is_valid = TorchScriptExporter.verify(model, export_path, device="cpu")
        self.assertTrue(is_valid)

        # Benchmark CPU latency
        latency_ms = TorchScriptExporter.benchmark_cpu(export_path, num_warmup=2, num_runs=5)
        self.assertGreater(latency_ms, 0.0)
        # Should easily be within responsive desktop execution
        self.assertLess(latency_ms, 200.0)

        # Verify 2-channel Vector-SDF ConvNeXt TorchScript export
        sdf_model = GlyphConvNeXt(in_channels=2, depths=[1, 1, 1, 1], dims=[16, 32, 64, 128], embed_dim=64).eval()
        sdf_path = Path(self.tmp_dir) / "convnext_2ch.pt"
        TorchScriptExporter.export(sdf_model, sdf_path, device="cpu")
        self.assertTrue(TorchScriptExporter.verify(sdf_model, sdf_path, device="cpu"))


if __name__ == "__main__":
    unittest.main()
