"""Unit tests for MobileNetV4, DualEncoder, InfoNCELoss, and DegradationPipeline."""

import unittest
import torch

from glyphclip.models.mobilenet_v4 import MobileNetV4Glyph
from glyphclip.models.dual_encoder import DualEncoder
from glyphclip.models.loss import InfoNCELoss
from glyphclip.pipeline.degradation import TypographicDegradationPipeline


class TestNeuralArchitecture(unittest.TestCase):
    """Test suite for neural components and contrastive loss."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_mobilenet_v4_forward_and_norm(self):
        """Verify model output shape is [B, 256] and unit normalized."""
        model = MobileNetV4Glyph(in_channels=1, embed_dim=256)
        model.eval()
        x = torch.randn(4, 1, 224, 224)
        with torch.no_grad():
            emb = model(x)
        self.assertEqual(emb.shape, (4, 256))
        # Check L2 normalization: norms should equal 1.0
        norms = torch.norm(emb, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_parameter_count(self):
        """Verify parameter count is within target lightweight envelope (~3.5M - 4.5M)."""
        model = MobileNetV4Glyph()
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 2_000_000)
        self.assertLess(total_params, 5_000_000)

    def test_dual_encoder(self):
        """Verify dual encoder separates canonical (frozen) and visual paths."""
        dual = DualEncoder(embed_dim=256, shared_backbone=False, freeze_canonical=True)
        # Check canonical encoder params require_grad is False
        for param in dual.canonical_encoder.parameters():
            self.assertFalse(param.requires_grad)

        x_vis = torch.randn(2, 1, 224, 224)
        x_can = torch.randn(2, 1, 224, 224)
        z_v, z_c = dual(x_vis, x_can)
        self.assertEqual(z_v.shape, (2, 256))
        self.assertEqual(z_c.shape, (2, 256))

    def test_infonce_loss(self):
        """Verify symmetric InfoNCE loss and metrics on synthetic embeddings."""
        criterion = InfoNCELoss(init_temperature=0.07, learnable=True)
        # Perfect alignment test: z_v == z_c
        z_c = torch.randn(8, 256)
        z_c = torch.nn.functional.normalize(z_c, p=2, dim=-1)
        z_v = z_c.clone()  # Identical

        loss, metrics = criterion(z_v, z_c)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0.0)
        self.assertAlmostEqual(metrics["in_batch_top1_acc"], 1.0)
        self.assertAlmostEqual(metrics["pos_sim"], 1.0, places=4)

    def test_degradation_pipeline(self):
        """Verify degradation pipeline transforms keep shape and clamped range."""
        pipeline = TypographicDegradationPipeline()
        x = torch.rand(2, 1, 224, 224)
        out = pipeline(x)
        self.assertEqual(out.shape, (2, 1, 224, 224))
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
