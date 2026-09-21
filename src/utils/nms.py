"""Pure NumPy / PyTorch NMS implementations."""

from __future__ import annotations

import numpy as np


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Classical greedy NMS.

    Args:
        boxes: ``[N, 4]`` xyxy.
        scores: ``[N]``.
        iou_threshold: drop boxes whose IoU with the current best ≥ this.

    Returns:
        Indices to keep, sorted by descending score.
    """
    if boxes.size == 0:
        return np.zeros(0, dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)

    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        idxs = np.where(iou <= iou_threshold)[0]
        order = order[idxs + 1]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """NMS that does *not* suppress boxes of different classes.

    Args:
        boxes: ``[N, 4]`` xyxy.
        scores: ``[N]``.
        class_ids: ``[N]`` integer class labels.
        iou_threshold: as in ``nms``.

    Returns:
        Indices to keep, sorted by descending score.
    """
    if boxes.size == 0:
        return np.zeros(0, dtype=np.int64)

    max_class = int(class_ids.max()) + 1
    keep_all: list[int] = []
    for c in range(max_class):
        mask = class_ids == c
        if not mask.any():
            continue
        sub_keep = nms(boxes[mask], scores[mask], iou_threshold)
        # Map back to global indices.
        idxs = np.where(mask)[0][sub_keep]
        keep_all.extend(int(i) for i in idxs)

    # Sort by score descending for deterministic output.
    keep_all.sort(key=lambda i: float(scores[i]), reverse=True)
    return np.asarray(keep_all, dtype=np.int64)
