"""Visualization utilities — bbox drawing, mask overlay, combined overlay."""

from __future__ import annotations

import random
from typing import Iterable

import cv2
import numpy as np


_PALETTE: list[tuple[int, int, int]] = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
    (255, 144, 0),
    (255, 0, 0),
    (255, 0, 73),
    (255, 0, 161),
    (181, 0, 255),
    (132, 0, 255),
    (203, 56, 255),
    (250, 0, 135),
]


def _color_for(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


def draw_boxes(
    image: np.ndarray,
    boxes: Iterable[tuple[float, float, float, float]],
    scores: Iterable[float] | None = None,
    color: tuple[int, int, int] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw xyxy boxes onto a BGR image (in place) and return it."""
    out = image if image.flags["WRITEABLE"] else image.copy()
    boxes = list(boxes)
    scores = list(scores) if scores is not None else None
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        c = color or _color_for(i)
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), c, thickness)
        if scores is not None and i < len(scores):
            label = f"{float(scores[i]):.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1i, y1i - th - 4), (x1i + tw, y1i), c, -1)
            cv2.putText(
                out,
                label,
                (x1i, y1i - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return out


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a binary mask onto a BGR image with the given colour/alpha."""
    out = image if image.flags["WRITEABLE"] else image.copy()
    if mask.shape[:2] != out.shape[:2]:
        mask = cv2.resize(mask, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay_layer = np.zeros_like(out)
    overlay_layer[mask > 0] = color
    mask_bool = (mask > 0).astype(np.float32)[..., None]
    out = (out * (1 - alpha * mask_bool) + overlay_layer * (alpha * mask_bool)).astype(np.uint8)
    return out


def overlay(
    image: np.ndarray,
    boxes: np.ndarray,
    masks: list[np.ndarray],
    scores: np.ndarray | None = None,
    box_thickness: int = 2,
    mask_alpha: float = 0.5,
) -> np.ndarray:
    """Combined: draw bboxes, then overlay each mask with its own colour."""
    out = image.copy()
    if boxes.size:
        out = draw_boxes(out, boxes.tolist(), scores=scores.tolist() if scores is not None else None)
    for i, mask in enumerate(masks):
        c = _color_for(i)
        out = draw_mask_overlay(out, mask, color=c, alpha=mask_alpha)
    return out


def random_palette_color(rng: random.Random | None = None) -> tuple[int, int, int]:
    """Return a deterministic-ish colour from the shared palette."""
    rng = rng or random
    return _PALETTE[rng.randint(0, len(_PALETTE) - 1)]
