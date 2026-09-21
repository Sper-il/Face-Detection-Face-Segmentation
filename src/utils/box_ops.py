"""Box conversion / encoding utilities used by both detection and the pipeline.

All boxes are float ``np.ndarray`` of shape ``[N, 4]`` in one of two layouts:

- ``xyxy`` — ``[x_min, y_min, x_max, y_max]``
- ``xywh`` — ``[x_center, y_center, width, height]`` (matches the
  WIDER FACE annotation layout).
"""

from __future__ import annotations

import numpy as np


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[N, 4]`` boxes from xyxy to xywh."""
    if boxes.size == 0:
        return boxes.reshape(-1, 4).astype(np.float32)
    out = boxes.astype(np.float32).copy()
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    out[:, 0] = boxes[:, 0] + out[:, 2] * 0.5
    out[:, 1] = boxes[:, 1] + out[:, 3] * 0.5
    return out


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[N, 4]`` boxes from xywh to xyxy."""
    if boxes.size == 0:
        return boxes.reshape(-1, 4).astype(np.float32)
    out = boxes.astype(np.float32).copy()
    half_w = boxes[:, 2] * 0.5
    half_h = boxes[:, 3] * 0.5
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def encode_boxes(
    anchors_xyxy: np.ndarray,
    gt_xyxy: np.ndarray,
    variance: tuple[float, float] = (0.1, 0.2),
) -> np.ndarray:
    """Encode GT boxes relative to anchors (RetinaFace convention).

    Returns ``[N, 4]`` with the standard 4-delta encoding:
        tx = (gx - ax) / aw / var[0]
        ty = (gy - ay) / ah / var[0]
        tw = log(gw / aw) / var[1]
        th = log(gh / ah) / var[1]
    """
    if anchors_xyxy.size == 0 or gt_xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    aw = anchors_xyxy[:, 2] - anchors_xyxy[:, 0]
    ah = anchors_xyxy[:, 3] - anchors_xyxy[:, 1]
    ax = (anchors_xyxy[:, 0] + anchors_xyxy[:, 2]) * 0.5
    ay = (anchors_xyxy[:, 1] + anchors_xyxy[:, 3]) * 0.5

    gw = gt_xyxy[:, 2] - gt_xyxy[:, 0]
    gh = gt_xyxy[:, 3] - gt_xyxy[:, 1]
    gx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) * 0.5
    gy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) * 0.5

    v0, v1 = variance
    tx = (gx - ax) / (aw + 1e-7) / v0
    ty = (gy - ay) / (ah + 1e-7) / v0
    tw = np.log(gw / (aw + 1e-7) + 1e-7) / v1
    th = np.log(gh / (ah + 1e-7) + 1e-7) / v1

    return np.stack([tx, ty, tw, th], axis=1).astype(np.float32)


def decode_boxes(
    anchors_xyxy: np.ndarray,
    deltas: np.ndarray,
    variance: tuple[float, float] = (0.1, 0.2),
) -> np.ndarray:
    """Decode predictions back to xyxy boxes in anchor coordinates."""
    if anchors_xyxy.size == 0 or deltas.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    aw = anchors_xyxy[:, 2] - anchors_xyxy[:, 0]
    ah = anchors_xyxy[:, 3] - anchors_xyxy[:, 1]
    ax = (anchors_xyxy[:, 0] + anchors_xyxy[:, 2]) * 0.5
    ay = (anchors_xyxy[:, 1] + anchors_xyxy[:, 3]) * 0.5

    v0, v1 = variance
    dx = deltas[:, 0] * v0
    dy = deltas[:, 1] * v0
    dw = deltas[:, 2] * v1
    dh = deltas[:, 3] * v1

    px = dx * aw + ax
    py = dy * ah + ay
    pw = np.exp(dw) * aw
    ph = np.exp(dh) * ah

    boxes = np.stack(
        [px - pw * 0.5, py - ph * 0.5, px + pw * 0.5, py + ph * 0.5],
        axis=1,
    ).astype(np.float32)
    return boxes


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip xyxy boxes to the image bounds in-place."""
    if boxes.size == 0:
        return boxes
    out = boxes.astype(np.float32, copy=True)
    out[:, 0] = np.clip(out[:, 0], 0, width - 1)
    out[:, 1] = np.clip(out[:, 1], 0, height - 1)
    out[:, 2] = np.clip(out[:, 2], 0, width - 1)
    out[:, 3] = np.clip(out[:, 3], 0, height - 1)
    return out


def bbox_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of xyxy boxes.

    Returns ``[N, M]`` matrix.
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    a = boxes1.astype(np.float32)
    b = boxes2.astype(np.float32)

    area1 = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area2 = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)
