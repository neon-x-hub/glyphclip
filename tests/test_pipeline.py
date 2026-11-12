"""Unit tests for Sauvola binarization, polarity normalization, and visual segmentation."""

from pathlib import Path
import unittest
import numpy as np
import cv2
import torch

from glyphclip.canvas.font_parser import FontParser
from glyphclip.pipeline.binarization import normalize_polarity, sauvola_threshold
from glyphclip.pipeline.dataset import FontDatasetSplitter
from glyphclip.pipeline.degradation import RandomStrokeMorphology
from glyphclip.pipeline.segmentation import VisualSegmentationPipeline


class TestPipeline(unittest.TestCase):
    """Test suite for image preprocessing and visual glyph extraction."""

    def test_sauvola_threshold(self):
        """Verify Sauvola outputs strictly binary values {0, 255}."""
        # Create image with uneven gradient background and dark circle
        img = np.linspace(50, 200, 100, dtype=np.uint8)
        img = np.tile(img, (100, 1))
        # Draw stroke
        img[30:70, 30:70] = 20

        binary = sauvola_threshold(img, window_size=25, k=0.2)
        unique_vals = np.unique(binary)
        for val in unique_vals:
            self.assertIn(val, [0, 255])
        self.assertEqual(binary.shape, (100, 100))

    def test_polarity_normalization(self):
        """Verify polarity normalization inverts dark-on-light but preserves light-on-dark."""
        # Dark on light (border is white 255, center stroke is black 0)
        dark_on_light = np.full((100, 100), 255, dtype=np.uint8)
        dark_on_light[40:60, 40:60] = 0

        normalized, inverted = normalize_polarity(dark_on_light)
        self.assertTrue(inverted)
        self.assertEqual(normalized[0, 0], 0)     # Background became black
        self.assertEqual(normalized[50, 50], 255) # Stroke became white

        # Light on dark (border is black 0, stroke is white 255)
        light_on_dark = np.zeros((100, 100), dtype=np.uint8)
        light_on_dark[40:60, 40:60] = 255

        normalized2, inverted2 = normalize_polarity(light_on_dark)
        self.assertFalse(inverted2)
        self.assertEqual(normalized2[0, 0], 0)
        self.assertEqual(normalized2[50, 50], 255)

    def test_segmentation_pipeline_synthetic_word(self):
        """Verify segmentation extracts horizontally ordered characters."""
        pipeline = VisualSegmentationPipeline(min_area=50)

        # Create synthetic word canvas (white background, black letters "I I I")
        canvas = np.full((120, 300), 255, dtype=np.uint8)
        # Letter 1
        cv2.rectangle(canvas, (30, 20), (50, 90), 0, -1)
        # Letter 2
        cv2.rectangle(canvas, (110, 20), (130, 90), 0, -1)
        # Letter 3
        cv2.rectangle(canvas, (190, 20), (210, 90), 0, -1)

        glyphs = pipeline.process(canvas)
        self.assertEqual(len(glyphs), 3)

        # Check horizontal ordering
        for i in range(len(glyphs) - 1):
            self.assertLess(glyphs[i].bbox[0], glyphs[i + 1].bbox[0])

        # Check canvas normalization dimensions
        for g in glyphs:
            self.assertEqual(g.canvas_numpy.shape, (224, 224))
            self.assertEqual(g.canvas_tensor.shape, (1, 224, 224))

    def test_morphology_jitter_and_family_disjoint_split(self):
        """Verify RandomStrokeMorphology bounds and strict zero-overlap family splitting."""
        morph = RandomStrokeMorphology(p=1.0, kernel_size=3)
        x = torch.rand(4, 1, 64, 64)
        out = morph(x)
        self.assertEqual(out.shape, (4, 1, 64, 64))
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

        win_fonts = Path(r"C:\Windows\Fonts")
        if win_fonts.exists():
            tr, va, te = FontDatasetSplitter.get_font_splits([win_fonts], max_total_fonts=60, max_per_family=2)
            tr_fams = {FontDatasetSplitter.normalize_family_key(p, FontParser.parse(p).family) for p in tr}
            va_fams = {FontDatasetSplitter.normalize_family_key(p, FontParser.parse(p).family) for p in va}
            te_fams = {FontDatasetSplitter.normalize_family_key(p, FontParser.parse(p).family) for p in te}
            self.assertEqual(FontDatasetSplitter.normalize_family_key(Path("BwModelica-BoldItalic.otf"), "Bw Modelica Bold"), "bw modelica")
            self.assertGreater(len(tr), 0)
            self.assertGreater(len(te), 0)
            self.assertEqual(len(tr_fams & va_fams), 0)
            self.assertEqual(len(tr_fams & te_fams), 0)

    def test_light_on_dark_thick_stroke_segmentation(self):
        """Verify white thick-stroke text on a dark colored background segments cleanly."""
        # Dark green background (BGR=[20, 48, 30]) with light thick strokes
        canvas = np.full((300, 700, 3), (30, 48, 20), dtype=np.uint8)
        cv2.putText(canvas, "Ram-Altman", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (235, 240, 235), 14, cv2.LINE_AA)

        pipeline = VisualSegmentationPipeline(include_sdf=True)
        glyphs = pipeline.process(canvas)
        # Should segment multiple distinct letters (hyphen has aspect > 2.5 and is skipped)
        self.assertGreaterEqual(len(glyphs), 8)
        for g in glyphs:
            self.assertEqual(g.canvas_tensor.shape, (2, 224, 224))
            self.assertGreater((g.canvas_numpy > 127).mean(), 0.05)


if __name__ == "__main__":
    unittest.main()
