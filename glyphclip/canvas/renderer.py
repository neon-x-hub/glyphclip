"""Deterministic canvas renderer for typographic glyphs.

Ensures mathematically consistent 224x224 single-channel canonical rendering
with exact 75% height bounding box scaling and centered alignment.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import hashlib
import io

import numpy as np
import torch
from fontTools.ttLib import TTFont
from fontTools.varLib import mutator
from PIL import Image, ImageFont


class DeterministicCanvasRenderer:
    """Renders canonical glyph canvases matching the GlyphCLIP specification."""

    def __init__(
        self,
        canvas_size: int = 224,
        target_height_ratio: float = 0.75,
        render_upscale: int = 2,
    ):
        """Initialize the canvas renderer.

        Args:
            canvas_size: Square canvas dimension in pixels (default: 224).
            target_height_ratio: Target ratio of glyph bounding box to canvas height
                (default: 0.75 -> 168 px on a 224 px canvas).
            render_upscale: Internal oversampling factor for smooth anti-aliased scaling.
        """
        self.canvas_size = canvas_size
        self.target_glyph_height = int(round(canvas_size * target_height_ratio))  # 168 px
        self.render_upscale = render_upscale
        self._font_cache: Dict[Tuple[Any, Optional[int], int], ImageFont.FreeTypeFont] = {}

    def _cache_id(self, font_source: Union[str, Path, bytes, io.BytesIO]) -> Any:
        if isinstance(font_source, (str, Path)):
            return str(font_source)
        if isinstance(font_source, bytes):
            return hashlib.md5(font_source).hexdigest()
        pos = font_source.tell()
        font_source.seek(0)
        digest = hashlib.md5(font_source.read()).hexdigest()
        font_source.seek(pos)
        return digest

    def _prepare_font(
        self,
        font_source: Union[str, Path, bytes, io.BytesIO],
        weight: Optional[int] = None,
        size: int = 160,
    ) -> ImageFont.FreeTypeFont:
        """Load font and optionally instantiate variable font weight with caching."""
        cache_key = (self._cache_id(font_source), weight, size)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]
        if len(self._font_cache) >= 64:
            self._font_cache.clear()

        font_obj: Optional[ImageFont.FreeTypeFont] = None

        # Check if variable font instantiation is requested
        if weight is not None:
            try:
                if isinstance(font_source, (str, Path)):
                    tt = TTFont(str(font_source))
                elif isinstance(font_source, bytes):
                    tt = TTFont(io.BytesIO(font_source))
                else:
                    font_source.seek(0)
                    tt = TTFont(font_source)

                if "fvar" in tt:
                    # Instantiate at requested weight
                    inst = mutator.instantiateVariableFont(tt, {"wght": float(weight)})
                    buf = io.BytesIO()
                    inst.save(buf)
                    buf.seek(0)
                    font_obj = ImageFont.truetype(buf, size=size)
            except Exception:
                # Fallback to direct loading if variable instantiation fails
                pass

        if font_obj is None:
            if isinstance(font_source, (str, Path)):
                font_obj = ImageFont.truetype(str(font_source), size=size)
            elif isinstance(font_source, bytes):
                font_obj = ImageFont.truetype(io.BytesIO(font_source), size=size)
            else:
                font_source.seek(0)
                font_obj = ImageFont.truetype(font_source, size=size)

        self._font_cache[cache_key] = font_obj
        return font_obj

    def render_glyph_image(
        self,
        font_source: Union[str, Path, bytes, io.BytesIO],
        char: str,
        weight: Optional[int] = None,
    ) -> Image.Image:
        """Render a single character glyph to a PIL Image (mode 'L').

        Scales the glyph bounding box to exactly target_glyph_height (168px)
        and centers it horizontally and vertically on a 224x224 black canvas.

        Args:
            font_source: Path to font file or byte stream.
            char: Single glyph character to render (e.g. 'g', 'Q').
            weight: Optional discrete CSS weight for variable fonts (e.g. 700).

        Returns:
            PIL.Image of mode 'L' (8-bit grayscale), shape 224x224.
        """
        if not char:
            raise ValueError("Character cannot be empty.")

        # 1. Render at high resolution to obtain clean antialiased stroke outlines
        render_size = int(self.target_glyph_height * self.render_upscale)
        font = self._prepare_font(font_source, weight=weight, size=render_size)

        mask = font.getmask(char)
        bbox = mask.getbbox()

        # If glyph produces no ink (e.g. space or missing glyph)
        if not bbox or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            # Return empty black canvas
            return Image.new("L", (self.canvas_size, self.canvas_size), color=0)

        # Convert mask to image and crop tight ink bounding box
        high_res_img = Image.frombytes("L", mask.size, bytes(mask))
        cropped = high_res_img.crop(bbox)

        orig_w, orig_h = cropped.size
        if orig_h == 0 or orig_w == 0:
            return Image.new("L", (self.canvas_size, self.canvas_size), color=0)

        # 2. Rescale isotropically so height is exactly target_glyph_height (168px)
        scale = self.target_glyph_height / float(orig_h)
        new_h = self.target_glyph_height
        new_w = max(1, int(round(orig_w * scale)))

        # Ensure width does not exceed canvas margins
        max_allowed_w = self.canvas_size - 8
        if new_w > max_allowed_w:
            width_scale = max_allowed_w / float(new_w)
            new_w = max_allowed_w
            new_h = max(1, int(round(new_h * width_scale)))

        resampled = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # 3. Center on black canvas (224x224, 0 background, 255 strokes)
        canvas = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        paste_x = (self.canvas_size - new_w) // 2
        paste_y = (self.canvas_size - new_h) // 2
        canvas.paste(resampled, (paste_x, paste_y))

        return canvas

    def render_glyph_numpy(
        self,
        font_source: Union[str, Path, bytes, io.BytesIO],
        char: str,
        weight: Optional[int] = None,
    ) -> np.ndarray:
        """Render glyph as a 2D numpy uint8 array [224, 224]."""
        img = self.render_glyph_image(font_source, char, weight=weight)
        return np.array(img, dtype=np.uint8)

    def render_glyph_tensor(
        self,
        font_source: Union[str, Path, bytes, io.BytesIO],
        char: str,
        weight: Optional[int] = None,
    ) -> torch.Tensor:
        """Render glyph as a normalized PyTorch tensor [1, 224, 224] in [0.0, 1.0]."""
        arr = self.render_glyph_numpy(font_source, char, weight=weight)
        tensor = torch.from_numpy(arr).float() / 255.0
        return tensor.unsqueeze(0)  # Shape: [1, 224, 224]
