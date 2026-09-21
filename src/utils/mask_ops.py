"""Mask operations: morphology, bbox clipping, paste-back."""

from __future__ import annotations

import cv2
import numpy as np


def mask_in_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Zero out mask pixels outside the (x1, y1, x2, y2) bbox.

    Returns a new array.
    """
    if mask.size == 0:
        return mask.copy()
    out = mask.copy()
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(out.shape[1] - 1, int(x2)), min(out.shape[0] - 1, int(y2))
    out[:y1, :] = 0
    out[y2 + 1 :, :] = 0
    out[:, :x1] = 0
    out[:, x2 + 1 :] = 0
    return out


def morph_close(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply binary closing (fill small holes) with a square kernel."""
    if mask.size == 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def mask_open(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply binary opening (remove salt noise) with a square kernel."""
    if mask.size == 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def mask_morphology(
    mask: np.ndarray,
    opening_kernel: int = 3,
    closing_kernel: int = 5,
) -> np.ndarray:
    """Run opening then closing in that order (small noise → small holes)."""
    if mask.size == 0:
        return mask
    return morph_close(mask_open(mask, opening_kernel), closing_kernel)


def paste_mask_into_image(
    cropped_mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Paste a cropped binary mask back into a full-image zero mask.

    ``cropped_mask`` is assumed to be aligned with the (possibly padded) crop
    used by the segmentor. ``bbox`` is ``(x1, y1, x2, y2)`` in the full image.
    The function uses ``cv2.resize`` to remap the mask onto the bbox rectangle.
    """
    h_img, w_img = image_shape
    out = np.zeros((h_img, w_img), dtype=np.uint8)

    x1, y1, x2, y2 = bbox
    bw = max(1, int(round(x2 - x1)))
    bh = max(1, int(round(y2 - y1)))

    if cropped_mask.size == 0:
        return out

    if cropped_mask.shape[:2] != (bh, bw):
        resized = cv2.resize(cropped_mask, (bw, bh), interpolation=cv2.INTER_NEAREST)
    else:
        resized = cropped_mask

    # Clip to image bounds (in case the bbox was near the edge).
    src_x1 = max(0, -int(x1))
    src_y1 = max(0, -int(y1))
    dst_x1 = max(0, int(x1))
    dst_y1 = max(0, int(y1))
    dst_x2 = min(w_img, int(x2))
    dst_y2 = min(h_img, int(y2))
    w = max(0, dst_x2 - dst_x1)
    h = max(0, dst_y2 - dst_y1)
    if w <= 0 or h <= 0:
        return out
    out[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y1 + h, src_x1:src_x1 + w]
    return out
