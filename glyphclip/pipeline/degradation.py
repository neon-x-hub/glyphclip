"""Synthetic typographic degradation pipeline for contrastive visual training.

Simulates camera capture distortions, resolution drops, optical blur,
sensor noise, and perspective slant onto pristine vector glyph renders.
"""

import random
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T


class RandomStrokeMorphology(nn.Module):
    """Applies multi-scale random dilation or erosion to simulate font-weight/bleed shifts."""

    def __init__(
        self,
        p: float = 0.65,
        kernel_size: Optional[int] = None,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7, 9),
    ):
        super().__init__()
        self.p = p
        self.kernel_sizes = (kernel_size,) if kernel_size is not None else tuple(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return x
        k = random.choice(self.kernel_sizes)
        pad = k // 2
        alpha = random.uniform(0.35, 1.0)
        if random.random() < 0.5:
            # Dilation (thicker strokes -> Medium/Bold simulation)
            morphed = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
        else:
            # Erosion (thinner strokes -> Light/Thin simulation)
            morphed = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=pad)
        return alpha * morphed + (1.0 - alpha) * x


class TypographicDegradationPipeline(nn.Module):
    """Synthesizes real-world capture noise and stroke-weight jitter onto canonical renders."""

    def __init__(
        self,
        p_blur: float = 0.5,
        p_perspective: float = 0.4,
        p_morph: float = 0.65,
        noise_sigma: float = 0.08,
    ):
        super().__init__()
        self.transforms = T.Compose([
            # 1. Geometric perturbation
            T.RandomAffine(
                degrees=(-6, 6),
                translate=(0.04, 0.04),
                scale=(0.92, 1.08),
                shear=(-5, 5),
            ),
            T.RandomPerspective(distortion_scale=0.15, p=p_perspective),

            # 2. Multi-scale stroke-thickness / bleed variation (promotes weight invariance)
            RandomStrokeMorphology(p=p_morph, kernel_sizes=(3, 5, 7, 9)),

            # 3. Optical resolution drops (smooth interpolation only -- preserves pixel-font corners)
            T.RandomChoice([
                T.Identity(),
                T.Compose([
                    T.Resize((64, 64), interpolation=T.InterpolationMode.BILINEAR),
                    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
                ]),
                T.Compose([
                    T.Resize((96, 96), interpolation=T.InterpolationMode.BICUBIC),
                    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
                ]),
            ]),

            # 4. Optical blur and sensor noise
            T.RandomApply([T.GaussianBlur(kernel_size=(5, 5), sigma=(0.6, 1.8))], p=p_blur),
            T.GaussianNoise(mean=0.0, sigma=noise_sigma, clip=True),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply noise transforms and clamp values to valid intensity interval [0.0, 1.0]."""
        out = self.transforms(x)
        return torch.clamp(out, 0.0, 1.0)
