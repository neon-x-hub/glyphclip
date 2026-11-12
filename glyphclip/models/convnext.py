"""High-capacity ConvNeXt typographic backbone (~20.5M parameters).

Optimized for fine-grained typographic anatomical feature extraction:
hairline serifs, stem modulation, counter apertures, and terminal tension.
Outputs 512-dimensional normalized unit embeddings on S^511.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block with 7x7 depthwise conv, LayerNorm, and 4x inverted bottleneck."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        # Permute for channel-last LayerNorm: [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # Permute back: [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return residual + x


class GlyphConvNeXt(nn.Module):
    """Scaled ~20.5M parameter typographic vision encoder.

    Architecture:
        - 1-channel grayscale patch stem: 4x4 stride 4
        - 4 stages of ConvNeXt blocks with dims [80, 160, 320, 640] and depths [3, 3, 9, 3]
        - Global Average Pooling -> LayerNorm
        - 2-layer MLP projection head (640 -> 1024 -> embed_dim)
        - L2 Normalization onto unit hypersphere S^(embed_dim-1)
    """

    def __init__(
        self,
        in_channels: int = 1,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [80, 160, 320, 640],
        embed_dim: int = 512,
        num_chars: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_chars = num_chars

        # Downsample / stem layers
        self.downsample_layers = nn.ModuleList()
        # Stem: [1, 224, 224] -> [80, 56, 56]
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            nn.GroupNorm(1, dims[0]),
        )
        self.downsample_layers.append(stem)

        # Stage downsamplers: 56->28, 28->14, 14->7
        for i in range(3):
            down = nn.Sequential(
                nn.GroupNorm(1, dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(down)

        # Stage feature processing
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        # 512-dimensional metric projection head
        self.projection = nn.Sequential(
            nn.Linear(dims[-1], 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, embed_dim),
        )

        # Optional auxiliary character classification head for Blind Auto-Detection
        self.char_head: Optional[nn.Module] = (
            nn.Linear(dims[-1], num_chars) if num_chars > 0 else None
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features across all stages."""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward_with_chars(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (unit_embeddings [B, embed_dim], char_logits [B, num_chars] | None)."""
        feat = self.extract_features(x)
        pooled = feat.mean([-2, -1])  # Global average pool [B, 640]
        normed = self.norm(pooled)
        emb = F.normalize(self.projection(normed), p=2, dim=-1)
        char_logits = self.char_head(normed) if self.char_head is not None else None
        return emb, char_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting unit-normalized embedding vectors [B, embed_dim]."""
        emb, _ = self.forward_with_chars(x)
        return emb
