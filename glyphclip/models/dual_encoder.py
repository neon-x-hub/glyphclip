"""Dual-encoder framework for canonical and visual font embeddings.

Supports shared-backbone joint optimization to guarantee a cohesive
typographic metric manifold on S^(d-1).
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from glyphclip.models.convnext import GlyphConvNeXt
from glyphclip.models.mobilenet_v4 import MobileNetV4Glyph


class DualEncoder(nn.Module):
    """Contrastive Dual-Encoder: E_c (canonical) and E_v (visual query).

    By default, uses a shared backbone (shared_backbone=True) so that pristine
    vector outlines and real-world noisy visual crops are embedded directly into
    the same continuous metric space.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        backbone_type: str = "convnext",  # "convnext" (~20.5M) or "mobilenet" (~4.2M)
        shared_backbone: bool = True,
        freeze_canonical: bool = False,
        in_channels: int = 1,
        num_chars: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone_type = backbone_type
        self.shared_backbone = shared_backbone
        self.in_channels = in_channels
        self.num_chars = num_chars

        if backbone_type == "convnext":
            self.visual_encoder = GlyphConvNeXt(
                in_channels=in_channels, embed_dim=embed_dim, num_chars=num_chars
            )
            if shared_backbone:
                self.canonical_encoder = self.visual_encoder
            else:
                self.canonical_encoder = GlyphConvNeXt(
                    in_channels=in_channels, embed_dim=embed_dim, num_chars=num_chars
                )
        else:
            self.visual_encoder = MobileNetV4Glyph(in_channels=in_channels, embed_dim=embed_dim)
            if shared_backbone:
                self.canonical_encoder = self.visual_encoder
            else:
                self.canonical_encoder = MobileNetV4Glyph(in_channels=in_channels, embed_dim=embed_dim)

        if freeze_canonical and not shared_backbone:
            self.freeze_canonical_encoder()

    def load_warmstart_from_1ch(self, state_dict: dict) -> None:
        """Load 1-channel checkpoint weights into a multi-channel (e.g. 2-ch SDF) backbone."""
        own_state = self.state_dict()
        adapted = {}
        for k, v in state_dict.items():
            if k not in own_state:
                continue
            target_shape = own_state[k].shape
            if v.shape == target_shape:
                adapted[k] = v
            elif v.ndim == 4 and target_shape[1] == 2 and v.shape[1] == 1 and v.shape[0] == target_shape[0]:
                # Split 1-channel stem weight cleanly across Occupancy (Ch 0) and TSDF (Ch 1)
                w = torch.zeros(target_shape, dtype=v.dtype, device=v.device)
                w[:, 0:1, :, :] = v * 0.7
                w[:, 1:2, :, :] = v * 0.3
                adapted[k] = w
        self.load_state_dict(adapted, strict=False)

    def freeze_canonical_encoder(self):
        """Freeze parameters in canonical encoder E_c."""
        for param in self.canonical_encoder.parameters():
            param.requires_grad = False
        self.canonical_encoder.eval()

    def encode_canonical(self, canonical_glyphs: torch.Tensor) -> torch.Tensor:
        """Encode pristine canonical glyphs E_c(g) -> z_c in S^(d-1)."""
        return self.canonical_encoder(canonical_glyphs)

    def encode_visual(self, visual_crops: torch.Tensor) -> torch.Tensor:
        """Encode real-world/degraded query crops E_v(x) -> z_v in S^(d-1)."""
        return self.visual_encoder(visual_crops)

    def encode_canonical_with_chars(
        self, canonical_glyphs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode canonical glyphs and return auxiliary character logits if configured."""
        if hasattr(self.canonical_encoder, "forward_with_chars"):
            return self.canonical_encoder.forward_with_chars(canonical_glyphs)
        return self.canonical_encoder(canonical_glyphs), None

    def encode_visual_with_chars(
        self, visual_crops: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode visual crops and return auxiliary character logits if configured."""
        if hasattr(self.visual_encoder, "forward_with_chars"):
            return self.visual_encoder.forward_with_chars(visual_crops)
        return self.visual_encoder(visual_crops), None

    @torch.no_grad()
    def predict_char_candidates(
        self,
        visual_crops: torch.Tensor,
        charset: list[str],
        top_k: int = 3,
        min_prob: float = 0.08,
    ) -> Optional[list[list[str]]]:
        """Predict top-k candidate glyph characters for each visual crop via auxiliary head."""
        if self.num_chars <= 0 or getattr(self.visual_encoder, "char_head", None) is None:
            return None
        _, logits = self.encode_visual_with_chars(visual_crops)
        if logits is None:
            return None
        probs = torch.softmax(logits, dim=-1)
        k = min(top_k, probs.shape[-1], len(charset))
        if k <= 0:
            return None
        top_probs, top_idx = torch.topk(probs, k=k, dim=-1)
        candidates: list[list[str]] = []
        for p_row, i_row in zip(top_probs.cpu().tolist(), top_idx.cpu().tolist()):
            row_chars = [charset[i_row[0]]]
            for prob, idx in zip(p_row[1:], i_row[1:]):
                if prob >= min_prob and idx < len(charset):
                    ch = charset[idx]
                    if ch not in row_chars:
                        row_chars.append(ch)
            candidates.append(row_chars)
        return candidates

    def forward(
        self,
        visual_crops: torch.Tensor,
        canonical_glyphs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for training or inference.

        If canonical_glyphs is provided, encodes both (z_v, z_c).
        Otherwise returns (z_v, None) for online inference.
        """
        z_v = self.encode_visual(visual_crops)
        z_c = self.encode_canonical(canonical_glyphs) if canonical_glyphs is not None else None
        return z_v, z_c
