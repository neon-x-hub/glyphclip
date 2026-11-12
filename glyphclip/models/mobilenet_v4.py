"""MobileNetV4 backbone with Universal Inverted Bottleneck (UIB) blocks.

Optimized for typographic single-channel grayscale input [1, 224, 224],
output stage feature map [960, 7, 7], and unit-sphere normalized projection [256].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UIBBlock(nn.Module):
    """Universal Inverted Bottleneck (UIB) Block for MobileNetV4."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        expand_ratio: float,
        stride: int = 1,
        dw_kernel: int = 3,
    ):
        super().__init__()
        self.use_res = (stride == 1 and in_c == out_c)
        hidden_dim = int(round(in_c * expand_ratio))
        layers = []

        # Pointwise expansion
        if expand_ratio != 1.0:
            layers.extend([
                nn.Conv2d(in_c, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        else:
            hidden_dim = in_c

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=dw_kernel,
                stride=stride,
                padding=dw_kernel // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise projection
            nn.Conv2d(hidden_dim, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV4Glyph(nn.Module):
    """MobileNetV4 feature extractor and 256-d normalized projection head."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Stem: Conv2d (3x3, stride 2, in_c=1, out_c=32) -> [32, 112, 112]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # Stage 1: 112x112 -> 56x56
        self.stage1 = nn.Sequential(
            UIBBlock(32, 48, expand_ratio=2.0, stride=2, dw_kernel=3),
            UIBBlock(48, 48, expand_ratio=2.0, stride=1, dw_kernel=3),
        )

        # Stage 2: 56x56 -> 28x28
        self.stage2 = nn.Sequential(
            UIBBlock(48, 96, expand_ratio=3.0, stride=2, dw_kernel=3),
            UIBBlock(96, 96, expand_ratio=3.0, stride=1, dw_kernel=3),
            UIBBlock(96, 96, expand_ratio=3.0, stride=1, dw_kernel=3),
        )

        # Stage 3: 28x28 -> 14x14
        self.stage3 = nn.Sequential(
            UIBBlock(96, 192, expand_ratio=3.5, stride=2, dw_kernel=5),
            UIBBlock(192, 192, expand_ratio=3.5, stride=1, dw_kernel=3),
            UIBBlock(192, 192, expand_ratio=3.5, stride=1, dw_kernel=3),
        )

        # Stage 4: 14x14 -> 7x7, C=960
        self.stage4 = nn.Sequential(
            UIBBlock(192, 512, expand_ratio=4.0, stride=2, dw_kernel=5),
            UIBBlock(512, 960, expand_ratio=3.0, stride=1, dw_kernel=3),
        )

        # Adaptive Average Pooling: [1, 960, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection Head: Linear(960 -> 512) -> LayerNorm -> GELU -> Linear(512 -> 256)
        self.projection = nn.Sequential(
            nn.Linear(960, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial stage 4 feature map [B, 960, 7, 7]."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.stage4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting L2-normalized unit embeddings [B, embed_dim]."""
        feat_map = self.extract_features(x)
        pooled = self.pool(feat_map).flatten(1)
        projected = self.projection(pooled)
        # L2 Normalization onto unit hypersphere S^(d-1)
        return F.normalize(projected, p=2, dim=-1)
