"""High-level visualizer for the pipeline (bbox + mask overlay)."""

from __future__ import annotations

import cv2
import numpy as np


def render_overlay(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    masks: list[np.ndarray],
    scores: np.ndarray | None = None,
    mask_alpha: float = 0.5,
    bbox_thickness: int = 2,
) -> np.ndarray:
    """Combine bboxes and per-face masks into a single BGR overlay."""
    out = image_bgr.copy()
    for i, mask in enumerate(masks):
        color = _palette_color(i)
        out = _overlay_mask(out, mask, color=color, alpha=mask_alpha)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        color = _palette_color(i)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, bbox_thickness)
        if scores is not None and i < len(scores):
            label = f"{float(scores[i]):.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 4)), (x1 + tw, y1), color, -1)
            cv2.putText(
                out,
                label,
                (x1, max(10, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return out


def render_empty_overlay(image_bgr: np.ndarray, message: str = "no face detected") -> np.ndarray:
    """Render a copy of the image with a "no face detected" caption."""
    out = image_bgr.copy()
    cv2.putText(
        out,
        message,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (255, 144, 0), (255, 0, 0), (255, 0, 73), (255, 0, 161),
    (181, 0, 255), (132, 0, 255), (203, 56, 255), (250, 0, 135),
]


def _palette_color(i: int) -> tuple[int, int, int]:
    return _PALETTE[i % len(_PALETTE)]


def _overlay_mask(
    image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float
) -> np.ndarray:
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay_layer = np.zeros_like(image)
    overlay_layer[mask > 0] = color
    mask_bool = (mask > 0).astype(np.float32)[..., None]
    return (
        image * (1 - alpha * mask_bool) + overlay_layer * (alpha * mask_bool)
    ).astype(np.uint8)
