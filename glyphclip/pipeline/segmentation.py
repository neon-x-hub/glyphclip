"""Connected Component Analysis (CCA), contour filtering, and glyph normalization."""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
import torch

from glyphclip.canvas.sdf import binary_mask_to_tsdf
from glyphclip.pipeline.binarization import adaptive_binarize


@dataclass
class SegmentedGlyph:
    """Represents a segmented and normalized character crop from an input image."""
    index: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in original query image
    canvas_numpy: np.ndarray          # [224, 224] uint8, white stroke on black
    canvas_tensor: torch.Tensor       # [1 or 2, 224, 224] float32 in [0.0, 1.0]


class VisualSegmentationPipeline:
    """Segments real-world images into standardized 224x224 character canvases."""

    def __init__(
        self,
        target_size: int = 224,
        max_glyph_dim: int = 180,
        min_area: int = 64,
        min_aspect: float = 0.06,
        max_aspect: float = 2.5,
        sauvola_window: int = 25,
        sauvola_k: float = 0.2,
        include_sdf: bool = False,
        tsdf_sigma: float = 8.0,
    ):
        self.target_size = target_size
        self.max_glyph_dim = max_glyph_dim
        self.target_glyph_height = int(round(target_size * 0.75))  # 168 px canonical height
        self.min_area = min_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.sauvola_window = sauvola_window
        self.sauvola_k = sauvola_k
        self.include_sdf = include_sdf
        self.tsdf_sigma = tsdf_sigma

    def deskew_image(self, gray: np.ndarray) -> np.ndarray:
        """Estimate and correct dominant text skew using minAreaRect on contours."""
        # Find points of high gradient / edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) < 50:
            return gray

        # Compute minimum area bounding box
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Adjust angle to [-45, 45] range
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle

        # Ignore tiny insignificant tilts (< 0.5 degrees) or large false triggers (> 30 degrees)
        if abs(angle) < 0.5 or abs(angle) > 30.0:
            return gray

        h, w = gray.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(gray, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed

    def normalize_glyph_crop(self, crop: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Scale glyph crop isotropically and center on 224x224 canvas."""
        n_channels = 2 if self.include_sdf else 1
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            canvas = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            tensor = torch.zeros(n_channels, self.target_size, self.target_size, dtype=torch.float32)
            return canvas, tensor

        if self.include_sdf:
            # Match canonical VectorSDFRenderer normalization: 168px height, max 216px width
            scale = float(self.target_glyph_height) / float(h)
            new_h = self.target_glyph_height
            new_w = max(1, int(round(w * scale)))
            max_allowed_w = self.target_size - 8
            if new_w > max_allowed_w:
                width_scale = float(max_allowed_w) / float(new_w)
                new_w = max_allowed_w
                new_h = max(1, int(round(new_h * width_scale)))
        else:
            # Legacy max_glyph_dim scaling for 1-channel compatibility
            scale = float(self.max_glyph_dim) / max(h, w)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(
            crop, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        )

        # Place onto 224x224 black canvas
        canvas = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized

        occ_tensor = torch.from_numpy(canvas).float() / 255.0
        if not self.include_sdf:
            return canvas, occ_tensor.unsqueeze(0)

        tsdf = binary_mask_to_tsdf(canvas, sigma=self.tsdf_sigma)
        tsdf_tensor = torch.from_numpy(tsdf).float()
        return canvas, torch.stack([occ_tensor, tsdf_tensor], dim=0)

    @staticmethod
    def _merge_stacked_components(
        main_boxes: List[Tuple[int, int, int, int]],
        small_boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        """Merge diacritics/dots (e.g. on 'i', 'j') that overlap horizontally with main stems."""
        if not main_boxes:
            return main_boxes

        # Sort by x coordinate
        boxes = [list(b) for b in sorted(main_boxes, key=lambda b: b[0])]

        # 1. Merge any vertically stacked small dots into their overlapping main column
        for sx, sy, sw, sh in small_boxes:
            best_idx = -1
            best_overlap = 0.0
            for i, (mx, my, mw, mh) in enumerate(boxes):
                overlap_x = min(sx + sw, mx + mw) - max(sx, mx)
                if overlap_x <= 0:
                    continue
                ratio = overlap_x / float(max(1, min(sw, mw)))
                # Check vertical proximity (dot typically above/below within 0.8 * glyph height)
                v_dist = max(0, my - (sy + sh), sy - (my + mh))
                if ratio >= 0.4 and v_dist <= max(12.0, 0.8 * mh) and ratio > best_overlap:
                    best_overlap = ratio
                    best_idx = i
            if best_idx >= 0:
                mx, my, mw, mh = boxes[best_idx]
                nx1 = min(mx, sx)
                ny1 = min(my, sy)
                nx2 = max(mx + mw, sx + sw)
                ny2 = max(my + mh, sy + sh)
                boxes[best_idx] = [nx1, ny1, nx2 - nx1, ny2 - ny1]

        # 2. Merge main boxes that share >= 45% horizontal overlap (stacked components)
        merged: List[List[int]] = []
        for b in sorted(boxes, key=lambda b: b[0]):
            if not merged:
                merged.append(b)
                continue
            px, py, pw, ph = merged[-1]
            cx, cy, cw, ch = b
            overlap_x = min(px + pw, cx + cw) - max(px, cx)
            if overlap_x > 0 and (overlap_x / float(max(1, min(pw, cw)))) >= 0.45:
                nx1 = min(px, cx)
                ny1 = min(py, cy)
                nx2 = max(px + pw, cx + cw)
                ny2 = max(py + ph, cy + ch)
                merged[-1] = [nx1, ny1, nx2 - nx1, ny2 - ny1]
            else:
                merged.append(b)

        return [tuple(b) for b in merged]

    def process(self, image: np.ndarray) -> List[SegmentedGlyph]:
        """Execute full visual segmentation pipeline on an input query image.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            List of SegmentedGlyph objects sorted horizontally in reading order.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Deskew
        gray = self.deskew_image(gray)

        # 2 & 3. Polarity-aware adaptive Sauvola binarization (scale window to prevent thick-stroke hollowing)
        h_img, w_img = gray.shape[:2]
        eff_window = max(self.sauvola_window, min(151, (min(h_img, w_img) // 3) | 1))
        binary, _ = adaptive_binarize(gray, window_size=eff_window, k=self.sauvola_k)

        # 4. Morphological cleaning (subtle closing to connect split strokes)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 5. Connected Component Analysis / Contour extraction
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        small_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = float(w) / float(h) if h > 0 else 0.0

            if area >= self.min_area and (self.min_aspect <= aspect_ratio <= self.max_aspect):
                candidates.append((x, y, w, h))
            elif area >= 12:
                small_candidates.append((x, y, w, h))

        # 6. Merge vertically stacked diacritics/dots and sort horizontally
        candidates = self._merge_stacked_components(candidates, small_candidates)
        candidates.sort(key=lambda b: b[0])

        segmented_glyphs: List[SegmentedGlyph] = []
        for idx, (x, y, w, h) in enumerate(candidates):
            crop = binary[y : y + h, x : x + w]
            canvas_np, canvas_tensor = self.normalize_glyph_crop(crop)
            segmented_glyphs.append(
                SegmentedGlyph(
                    index=idx,
                    bbox=(x, y, w, h),
                    canvas_numpy=canvas_np,
                    canvas_tensor=canvas_tensor,
                )
            )

        return segmented_glyphs
