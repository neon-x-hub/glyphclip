"""Sauvola adaptive thresholding and polarity normalization routines."""

from typing import Tuple
import cv2
import numpy as np


def sauvola_threshold(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0,
) -> np.ndarray:
    """Apply Sauvola adaptive binarization to handle uneven lighting and shadows.

    Formula:
        T(x, y) = m(x, y) * (1 + k * (s(x, y) / R - 1))

    Args:
        image: Single-channel 8-bit grayscale image [H, W].
        window_size: Local neighborhood window size (odd integer, default 25).
        k: Dynamic range control parameter (default 0.2).
        r: Dynamic range of standard deviation (default 128.0 for 8-bit image).

    Returns:
        Binary image [H, W] with values in {0, 255}.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if window_size % 2 == 0:
        window_size += 1

    img_f = image.astype(np.float32)

    # Compute local mean m(x, y) and local square mean using box filter
    ksize = (window_size, window_size)
    mean = cv2.boxFilter(img_f, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)
    sq_mean = cv2.boxFilter(img_f * img_f, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    # Compute local standard deviation s(x, y) = sqrt(E[x^2] - (E[x])^2)
    variance = np.maximum(sq_mean - (mean * mean), 0.0)
    std = np.sqrt(variance)

    # Sauvola threshold surface
    threshold = mean * (1.0 + k * ((std / r) - 1.0))

    # Binary output: 255 where image >= threshold, else 0
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[img_f >= threshold] = 255

    return binary


def normalize_polarity(binary_image: np.ndarray, border_width: int = 3) -> Tuple[np.ndarray, bool]:
    """Normalize text polarity so glyph strokes are white (255) on black (0).

    Computes the average intensity along the border of the canvas.
    If border pixels average > 127 (light background), inverts the image.

    Args:
        binary_image: Binary image [H, W] with values in {0, 255}.
        border_width: Pixel depth of boundary band used for background estimation.

    Returns:
        Tuple of (normalized_binary_image, was_inverted).
    """
    h, w = binary_image.shape[:2]
    bw = min(border_width, h // 4, w // 4)

    # Sample top, bottom, left, right borders
    top = binary_image[:bw, :]
    bottom = binary_image[-bw:, :]
    left = binary_image[:, :bw]
    right = binary_image[:, -bw:]

    border_mean = (np.mean(top) + np.mean(bottom) + np.mean(left) + np.mean(right)) / 4.0

    if border_mean > 127.0:
        # Dark text on light background -> invert to white strokes on black background
        return cv2.bitwise_not(binary_image), True

    return binary_image, False


def adaptive_binarize(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
    r: float = 128.0,
) -> Tuple[np.ndarray, bool]:
    """Polarity-aware Sauvola binarization returning white strokes (255) on black (0).

    Sauvola's threshold surface assumes dark text on a brighter background. When applied
    directly to light text on a dark background (e.g., white text on dark green),
    uniform dark regions fall above threshold while transition borders dip below it,
    forming connected edge halos around adjacent glyphs.

    We first detect background polarity via Otsu + border estimation and pre-invert
    light-on-dark images before Sauvola thresholding.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Detect whether background is bright (dark text) or dark (light text)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, is_light_bg = normalize_polarity(otsu)

    work_gray = gray if is_light_bg else (255 - gray)
    binary = sauvola_threshold(work_gray, window_size=window_size, k=k, r=r)
    normalized, _ = normalize_polarity(binary)
    return normalized, (not is_light_bg)
