"""Analytical vector-contour Signed Distance Field (SDF) renderer for font glyphs.

Extracts exact mathematical quadratic/cubic Bézier outlines directly from
TTF/OTF font files via fontTools, normalizes contour geometry isotropically,
and produces a dual-channel geometric representation:
    Channel 0: Anti-aliased sub-pixel ink occupancy mask in [0, 1]
    Channel 1: Smooth Truncated Signed Distance Field (TSDF) in [0, 1]
"""

import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

from glyphclip.canvas.renderer import DeterministicCanvasRenderer


def sdf_to_tsdf(sdf: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Map raw pixel signed distance values smoothly into [0.0, 1.0].

    Zero-crossing (exact contour boundary) sits at 0.5 with steepest gradient,
    highlighting fine hairline serifs and stroke boundaries.
    """
    return (0.5 + 0.5 * np.tanh(sdf / sigma)).astype(np.float32)


def binary_mask_to_tsdf(
    mask: np.ndarray,
    sigma: float = 8.0,
    mask_size: int = cv2.DIST_MASK_PRECISE,
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute continuous TSDF channel from a 2D occupancy mask (uint8 or float).

    Used on visual query crops and degraded training samples to bridge into the
    exact same distance-field space as analytical vector SDFs.
    """
    if mask.dtype != np.uint8:
        bin_m = (mask >= threshold).astype(np.uint8) * 255
    else:
        thr_u8 = int(round(threshold * 255.0))
        bin_m = (mask >= thr_u8).astype(np.uint8) * 255

    fg_pixels = int(np.count_nonzero(bin_m))
    if fg_pixels == 0:
        return np.zeros(mask.shape[:2], dtype=np.float32)
    if fg_pixels == bin_m.size:
        return np.ones(mask.shape[:2], dtype=np.float32)

    d_in = cv2.distanceTransform(bin_m, cv2.DIST_L2, mask_size)
    d_out = cv2.distanceTransform(255 - bin_m, cv2.DIST_L2, mask_size)

    # Shift by 0.5px so the 0-isosurface lies halfway between inside/outside pixels
    sdf = np.where(
        bin_m > 0,
        np.maximum(0.0, d_in - 0.5),
        -np.maximum(0.0, d_out - 0.5),
    )
    return sdf_to_tsdf(sdf, sigma=sigma)


def batch_occupancy_to_dual(
    occ_batch: torch.Tensor,
    sigma: float = 8.0,
    thr_jitter: float = 0.06,
    fast_mask: bool = True,
) -> torch.Tensor:
    """Augment a [B, 1, H, W] visual occupancy batch into [B, 2, H, W] ([Occ, TSDF])."""
    device = occ_batch.device
    occ_np = occ_batch[:, 0].detach().cpu().numpy()
    b, h, w = occ_np.shape
    mask_size = cv2.DIST_MASK_5 if fast_mask else cv2.DIST_MASK_PRECISE

    dual_np = np.empty((b, 2, h, w), dtype=np.float32)
    for i in range(b):
        thr = 0.5 + (float(np.random.uniform(-thr_jitter, thr_jitter)) if thr_jitter > 0 else 0.0)
        m = occ_np[i]
        # Random 50% chance to simulate binary Sauvola thresholded occupancy on Channel 0
        if np.random.rand() < 0.6:
            bin_u8 = (m >= thr).astype(np.uint8) * 255
            dual_np[i, 0] = bin_u8.astype(np.float32) / 255.0
            dual_np[i, 1] = binary_mask_to_tsdf(bin_u8, sigma=sigma, mask_size=mask_size)
        else:
            dual_np[i, 0] = m
            dual_np[i, 1] = binary_mask_to_tsdf(m, sigma=sigma, mask_size=mask_size, threshold=thr)

    return torch.from_numpy(dual_np).to(device, non_blocking=True)


class VectorGlyphPen(BasePen):
    """FontTools BasePen that adaptively subdivides Bézier splines into polylines."""

    def __init__(self, glyph_set: Any, chord_step: float = 10.0):
        super().__init__(glyph_set)
        self.chord_step = max(1e-3, float(chord_step))
        self.contours: List[List[Tuple[float, float]]] = []
        self._pts: List[Tuple[float, float]] = []
        self._start: Optional[Tuple[float, float]] = None
        self._pt: Optional[Tuple[float, float]] = None

    def _moveTo(self, pt: Tuple[float, float]) -> None:
        if len(self._pts) >= 3:
            self.contours.append(self._pts)
        self._pts = [(float(pt[0]), float(pt[1]))]
        self._start = (float(pt[0]), float(pt[1]))
        self._pt = self._start

    def _lineTo(self, pt: Tuple[float, float]) -> None:
        p = (float(pt[0]), float(pt[1]))
        if self._pt is not None and p != self._pt:
            self._pts.append(p)
        self._pt = p

    def _qCurveToOne(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> None:
        if self._pt is None:
            return
        p0 = np.array(self._pt, dtype=np.float64)
        p1a = np.array(p1, dtype=np.float64)
        p2a = np.array(p2, dtype=np.float64)
        chord = float(np.linalg.norm(p1a - p0) + np.linalg.norm(p2a - p1a))
        n_steps = max(2, min(14, int(np.ceil(chord / self.chord_step))))
        for t in np.linspace(0.0, 1.0, n_steps + 1)[1:]:
            q = (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * p1a + (t ** 2) * p2a
            self._pts.append((float(q[0]), float(q[1])))
        self._pt = (float(p2[0]), float(p2[1]))

    def _curveToOne(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
    ) -> None:
        if self._pt is None:
            return
        p0 = np.array(self._pt, dtype=np.float64)
        p1a = np.array(p1, dtype=np.float64)
        p2a = np.array(p2, dtype=np.float64)
        p3a = np.array(p3, dtype=np.float64)
        chord = float(
            np.linalg.norm(p1a - p0) + np.linalg.norm(p2a - p1a) + np.linalg.norm(p3a - p2a)
        )
        n_steps = max(2, min(16, int(np.ceil(chord / self.chord_step))))
        for t in np.linspace(0.0, 1.0, n_steps + 1)[1:]:
            c = (
                ((1.0 - t) ** 3) * p0
                + 3.0 * ((1.0 - t) ** 2) * t * p1a
                + 3.0 * (1.0 - t) * (t ** 2) * p2a
                + (t ** 3) * p3a
            )
            self._pts.append((float(c[0]), float(c[1])))
        self._pt = (float(p3[0]), float(p3[1]))

    def _closePath(self) -> None:
        if len(self._pts) >= 2 and self._start is not None:
            if self._pts[-1] != self._start:
                self._pts.append(self._start)
            if len(self._pts) >= 3:
                self.contours.append(self._pts)
        self._pts = []
        self._pt = self._start

    def _endPath(self) -> None:
        self._closePath()


class VectorSDFRenderer:
    """Computes [Occupancy, Vector-TSDF] tensors directly from TTF/OTF Bézier geometry."""

    def __init__(
        self,
        canvas_size: int = 224,
        target_height_ratio: float = 0.75,
        supersample: int = 4,
        tsdf_sigma: float = 8.0,
    ):
        self.canvas_size = canvas_size
        self.target_glyph_height = int(round(canvas_size * target_height_ratio))  # 168 px
        self.supersample = supersample
        self.tsdf_sigma = tsdf_sigma
        self._tt_cache: Dict[Tuple[Any, Optional[int]], TTFont] = {}
        self._raster_fallback = DeterministicCanvasRenderer(
            canvas_size=canvas_size, target_height_ratio=target_height_ratio
        )

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

    def _get_ttfont(
        self,
        font_source: Union[str, Path, bytes, io.BytesIO],
        weight: Optional[int] = None,
    ) -> TTFont:
        cache_key = (self._cache_id(font_source), weight)
        if cache_key in self._tt_cache:
            return self._tt_cache[cache_key]
        if len(self._tt_cache) >= 64:
            self._tt_cache.clear()

        if isinstance(font_source, (str, Path)):
            tt = TTFont(str(font_source))
        elif isinstance(font_source, bytes):
            tt = TTFont(io.BytesIO(font_source))
        else:
            font_source.seek(0)
            tt = TTFont(font_source)

        if weight is not None and "fvar" in tt:
            try:
                tt = instancer.instantiateVariableFont(tt, {"wght": float(weight)})
            except Exception:
                pass

        self._tt_cache[cache_key] = tt
        return tt

    def render_dual_numpy(
        self,
        font_source: Union[str, Path, bytes, io.BytesIO],
        char: str,
        weight: Optional[int] = None,
    ) -> np.ndarray:
        """Render a 2-channel array [2, 224, 224] float32 in [0.0, 1.0].

        Channel 0: Anti-aliased vector occupancy mask
        Channel 1: Continuous vector-derived TSDF
        """
        if not char:
            raise ValueError("Character cannot be empty.")

        try:
            tt = self._get_ttfont(font_source, weight=weight)
            cmap = tt.getBestCmap()
            if not cmap or ord(char) not in cmap:
                return np.zeros((2, self.canvas_size, self.canvas_size), dtype=np.float32)

            glyph_name = cmap[ord(char)]
            glyph_set = tt.getGlyphSet()
            if glyph_name not in glyph_set:
                return np.zeros((2, self.canvas_size, self.canvas_size), dtype=np.float32)

            upm = float(tt["head"].unitsPerEm) if "head" in tt else 1000.0
            pen = VectorGlyphPen(glyph_set, chord_step=max(1.0, upm * 0.008))
            glyph_set[glyph_name].draw(pen)

            if not pen.contours:
                # Fallback if bitmap-only / empty outline glyph
                occ_u8 = self._raster_fallback.render_glyph_numpy(font_source, char, weight=weight)
                if occ_u8.max() < 20:
                    return np.zeros((2, self.canvas_size, self.canvas_size), dtype=np.float32)
                occ = occ_u8.astype(np.float32) / 255.0
                tsdf = binary_mask_to_tsdf(occ_u8, sigma=self.tsdf_sigma)
                return np.stack([occ, tsdf], axis=0)

            all_pts = np.vstack([np.array(c, dtype=np.float64) for c in pen.contours])
            xmin, ymin = all_pts.min(axis=0)
            xmax, ymax = all_pts.max(axis=0)
            orig_w = float(xmax - xmin)
            orig_h = float(ymax - ymin)
            if orig_h <= 1e-4 or orig_w <= 1e-4:
                return np.zeros((2, self.canvas_size, self.canvas_size), dtype=np.float32)

            # Isotropic scaling matching DeterministicCanvasRenderer (168px height, max 216px width)
            scale = float(self.target_glyph_height) / orig_h
            new_h = float(self.target_glyph_height)
            new_w = orig_w * scale
            max_allowed_w = float(self.canvas_size - 8)
            if new_w > max_allowed_w:
                width_scale = max_allowed_w / new_w
                new_w = max_allowed_w
                new_h *= width_scale
                scale *= width_scale

            offset_x = (float(self.canvas_size) - new_w) * 0.5
            offset_y = (float(self.canvas_size) - new_h) * 0.5

            ss = self.supersample
            hr_dim = self.canvas_size * ss
            winding = np.zeros((hr_dim, hr_dim), dtype=np.int16)

            # Non-zero winding fill across all contours at ss*16 subpixel precision (shift=4)
            tmp_buf = np.zeros((hr_dim, hr_dim), dtype=np.uint8)
            for contour in pen.contours:
                arr = np.array(contour, dtype=np.float64)
                # Signed shoelace area in font space (Y-up) determines outer vs hole winding
                area = 0.5 * float(np.sum(arr[:-1, 0] * arr[1:, 1] - arr[1:, 0] * arr[:-1, 1]))
                if abs(area) < 1e-6:
                    continue

                tx = (arr[:, 0] - xmin) * scale + offset_x
                ty = (ymax - arr[:, 1]) * scale + offset_y  # Flip Y for image coordinates
                pts_fixed = np.round(np.stack([tx, ty], axis=1) * float(ss * 16)).astype(np.int32)

                tmp_buf.fill(0)
                cv2.fillPoly(tmp_buf, [pts_fixed], 1, shift=4)
                if area > 0:
                    winding += tmp_buf
                else:
                    winding -= tmp_buf

            hr_mask = (winding != 0).astype(np.uint8) * 255
            if hr_mask.max() == 0:
                return np.zeros((2, self.canvas_size, self.canvas_size), dtype=np.float32)

            # Channel 0: High-quality area-downsampled subpixel occupancy in [0, 1]
            occ = cv2.resize(
                hr_mask, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_AREA
            ).astype(np.float32) / 255.0

            # Channel 1: High-resolution Euclidean Distance Transform -> continuous TSDF
            d_in = cv2.distanceTransform(hr_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            d_out = cv2.distanceTransform(255 - hr_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            hr_sdf = np.where(
                hr_mask > 0,
                np.maximum(0.0, d_in - 0.5),
                -np.maximum(0.0, d_out - 0.5),
            ) / float(ss)

            sdf_224 = cv2.resize(
                hr_sdf, (self.canvas_size, self.canvas_size), interpolation=cv2.INTER_AREA
            )
            tsdf = sdf_to_tsdf(sdf_224, sigma=self.tsdf_sigma)

            return np.stack([occ, tsdf], axis=0)
        except Exception:
            # Safe fallback to FreeType raster + binary EDT
            occ_u8 = self._raster_fallback.render_glyph_numpy(font_source, char, weight=weight)
            if occ_u8.max() < 20:
                return np.zeros((2, self.canvas_size, self.canvas_size), dtype=np.float32)
            occ = occ_u8.astype(np.float32) / 255.0
            tsdf = binary_mask_to_tsdf(occ_u8, sigma=self.tsdf_sigma)
            return np.stack([occ, tsdf], axis=0)

    def render_dual_uint8(
        self,
        font_source: Union[str, Path, bytes, io.BytesIO],
        char: str,
        weight: Optional[int] = None,
    ) -> np.ndarray:
        """Render quantized [2, 224, 224] uint8 representation for compact dataset caching."""
        arr = self.render_dual_numpy(font_source, char, weight=weight)
        return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)

    def render_dual_tensor(
        self,
        font_source: Union[str, Path, bytes, io.BytesIO],
        char: str,
        weight: Optional[int] = None,
    ) -> torch.Tensor:
        """Render 2-channel PyTorch float32 tensor [2, 224, 224] in [0.0, 1.0]."""
        arr = self.render_dual_numpy(font_source, char, weight=weight)
        return torch.from_numpy(arr)


def compute_edge_orthogonality(occ_mask: Union[np.ndarray, torch.Tensor]) -> float:
    """Measure fraction of glyph contour pixels aligned to horizontal/vertical 0/90 axes.

    Bitmap/pixel fonts (e.g., Pixel Operator 8, Silkscreen) exhibit ~0.94-0.98 orthogonality
    even on diagonal and curved characters, whereas smooth Bezier sans/serifs drop to
    ~0.35-0.70 on diagonal/curved glyphs.
    """
    if isinstance(occ_mask, torch.Tensor):
        arr = occ_mask.detach().cpu().numpy()
    else:
        arr = np.asarray(occ_mask)
    if arr.ndim == 3:
        arr = arr[0]
    binary = (arr > 0.5).astype(np.float32)
    gx = cv2.Sobel(binary, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(binary, cv2.CV_32F, 0, 1, ksize=3)
    mask = np.hypot(gx, gy) > 0.5
    if int(mask.sum()) < 10:
        return 0.5
    ax = np.abs(gx[mask])
    ay = np.abs(gy[mask])
    ratio = np.minimum(ax, ay) / (ax + ay + 1e-6)
    return float((ratio < 0.08).mean())
