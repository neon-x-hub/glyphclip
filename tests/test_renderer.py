"""Unit tests for canvas renderer, character sets, and font parser."""

import unittest
from pathlib import Path
import numpy as np
import torch

from glyphclip.canvas.charset import (
    CANONICAL_CHARSET,
    DISCRIMINATIVE_WEIGHTS,
    get_charset,
    get_entropy_weight,
)
from glyphclip.canvas.font_parser import FontMetadata, FontParser
from glyphclip.canvas.renderer import DeterministicCanvasRenderer


class TestCanvasRenderer(unittest.TestCase):
    """Test suite for deterministic rendering and font metadata extraction."""

    @classmethod
    def setUpClass(cls):
        # Look for a standard system font for testing
        test_font_candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        cls.font_path = None
        for p in test_font_candidates:
            if Path(p).exists():
                cls.font_path = p
                break
        
        if cls.font_path is None:
            raise unittest.SkipTest("No system font found for testing.")

    def setUp(self):
        self.renderer = DeterministicCanvasRenderer(canvas_size=224, target_height_ratio=0.75)

    def test_canonical_charset(self):
        """Verify canonical character set contains the 10 discriminative glyphs and full alphabet."""
        expected = ["a", "g", "e", "f", "Q", "R", "S", "O", "M", "t"]
        self.assertEqual(get_charset("core"), expected)
        full_set = get_charset("full")
        for char in expected:
            self.assertIn(char, full_set)
            weight = get_entropy_weight(char)
            self.assertGreater(weight, 0.5)

    def test_font_parser(self):
        """Verify font parser extracts family name and weight correctly."""
        meta = FontParser.parse(self.font_path)
        self.assertIsInstance(meta, FontMetadata)
        self.assertNotEqual(meta.family, "Unknown")
        self.assertGreater(meta.weight, 0)
        weights = FontParser.get_supported_weights(meta)
        self.assertIsInstance(weights, list)
        self.assertGreater(len(weights), 0)

    def test_render_glyph_image_dimensions(self):
        """Verify rendered image has exact 224x224 dimensions and mode L."""
        img = self.renderer.render_glyph_image(self.font_path, "g")
        self.assertEqual(img.size, (224, 224))
        self.assertEqual(img.mode, "L")

    def test_render_glyph_numpy(self):
        """Verify rendered numpy array shape, dtype, and intensity ranges."""
        arr = self.renderer.render_glyph_numpy(self.font_path, "Q")
        self.assertEqual(arr.shape, (224, 224))
        self.assertEqual(arr.dtype, np.uint8)
        self.assertEqual(arr[0, 0], 0)  # Corner is background
        self.assertGreater(arr.max(), 128)  # Stroke pixels exist

    def test_render_glyph_tensor(self):
        """Verify rendered tensor shape and value range [0.0, 1.0]."""
        tensor = self.renderer.render_glyph_tensor(self.font_path, "a")
        self.assertEqual(tensor.shape, (1, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_glyph_height_normalization(self):
        """Verify glyph bounding box height is close to target 168 pixels (75%)."""
        for char in ["g", "R", "S"]:
            arr = self.renderer.render_glyph_numpy(self.font_path, char)
            rows = np.any(arr > 30, axis=1)
            y_indices = np.where(rows)[0]
            if len(y_indices) > 0:
                measured_height = y_indices[-1] - y_indices[0] + 1
                # Should be within +/- 3 pixels of target 168
                self.assertAlmostEqual(measured_height, 168, delta=4)


if __name__ == "__main__":
    unittest.main()
