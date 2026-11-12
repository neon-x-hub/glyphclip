"""Synthetic degradation, binarization, segmentation, and dataset pipelines."""

from glyphclip.pipeline.binarization import (
    adaptive_binarize,
    normalize_polarity,
    sauvola_threshold,
)
from glyphclip.pipeline.degradation import TypographicDegradationPipeline
from glyphclip.pipeline.segmentation import SegmentedGlyph, VisualSegmentationPipeline

__all__ = [
    "adaptive_binarize",
    "normalize_polarity",
    "sauvola_threshold",
    "TypographicDegradationPipeline",
    "SegmentedGlyph",
    "VisualSegmentationPipeline",
]
