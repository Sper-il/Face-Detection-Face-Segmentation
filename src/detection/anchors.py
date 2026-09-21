"""Anchor generator + target encoder for RetinaFace-style FPN.

The RetinaFace paper uses three FPN levels (P3, P4, P5) and two extra levels
(P6, P7) when stride-32/64 are needed; for the đề's crowded scenes we keep the
classic 3-level setup:

    P3: stride  8  → feature map / 8
    P4: stride 16  → feature map / 16
    P5: stride 32  → feature map / 32

Two anchor scales per level (`2^0 * s`, `2^(1/3) * s`, `2^(2/3) * s`) and three
aspect ratios (1:1, 1:2, 2:1) give 9 anchors per location.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.utils.box_ops import bbox_iou, decode_boxes, encode_boxes


@dataclass(frozen=True)
class AnchorConfig:
    """Configuration for the multi-scale anchor generator."""

    image_size: int = 640
    strides: tuple[int, ...] = (8, 16, 32)  # feature-map strides — MUST match backbone
    base_sizes: tuple[int, ...] = (16, 64, 256)  # scales per FPN level
    ratios: tuple[float, ...] = (1.0, 2.0, 0.5)
    scales_per_level: int = 3  # octave subdivisions


def generate_anchors(cfg: AnchorConfig) -> np.ndarray:
    """Pre-compute all anchors in image-pixel coordinates.

    Returns ``[N, 4]`` xyxy array.
    """
    all_anchors: list[np.ndarray] = []
    for level_idx, stride in enumerate(cfg.strides):
        base = cfg.base_sizes[level_idx]
        feature_dim = cfg.image_size // stride
        sub_scales = _octave_scales(base, cfg.scales_per_level)

        # Build (cy, cx) grid.
        shifts_x = (np.arange(feature_dim) + 0.5) * stride
        shifts_y = (np.arange(feature_dim) + 0.5) * stride
        cy, cx = np.meshgrid(shifts_y, shifts_x, indexing="ij")
        shifts = np.stack([cx.ravel(), cy.ravel(), cx.ravel(), cy.ravel()], axis=1)  # dummy

        for s in sub_scales:
            for r in cfg.ratios:
                w = s * np.sqrt(r)
                h = s / np.sqrt(r)
                anchors = shifts.copy().astype(np.float32)
                anchors[:, 0] -= w * 0.5
                anchors[:, 1] -= h * 0.5
                anchors[:, 2] = anchors[:, 0] + w
                anchors[:, 3] = anchors[:, 1] + h
                all_anchors.append(anchors)

    return np.concatenate(all_anchors, axis=0).astype(np.float32)


def _octave_scales(base: float, n: int) -> list[float]:
    """Return ``n`` scales centred on ``base`` (RetinaFace style)."""
    return [float(base * (2.0 ** (i / max(1, n)))) for i in range(n)]


# ---------------------------------------------------------------------------
# Target assignment
# ---------------------------------------------------------------------------


def assign_targets(
    anchors: np.ndarray,
    gt_boxes: np.ndarray,
    gt_landmarks: np.ndarray | None,
    image_size: int,
    pos_iou: float = 0.5,
    neg_iou: float = 0.3,
    landmark_match_iou: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign each anchor to a class + regression target.

    Args:
        anchors: ``[N, 4]`` xyxy (image coords).
        gt_boxes: ``[M, 4]`` xyxy.
        gt_landmarks: ``[M, 10]`` (5 landmarks × xy), in image coords. Pass
            ``None`` if not available (landmarks will be left zeroed).
        image_size: used to mark anchors outside the image as ``ignore``.
        pos_iou: anchors with max-IoU ≥ this and whose best match is unique
            become positive.
        neg_iou: anchors with max-IoU < this become negative.
        landmark_match_iou: anchors whose best IoU ≥ this also receive the
            matching GT landmark.

    Returns
        cls_target : ``[N]`` int64 — ``1`` = face, ``0`` = background, ``-1`` = ignore.
        box_target : ``[N, 4]`` float32, encoded.
        lmk_target : ``[N, 10]`` float32, landmark offsets normalised by anchor size.
        lmk_mask   : ``[N]`` float32 — 1 if landmark is assigned, else 0.
    """
    n_anchors = anchors.shape[0]
    cls_target = -np.ones(n_anchors, dtype=np.int64)
    box_target = np.zeros((n_anchors, 4), dtype=np.float32)
    lmk_target = np.zeros((n_anchors, 10), dtype=np.float32)
    lmk_mask = np.zeros(n_anchors, dtype=np.float32)

    if gt_boxes is None or gt_boxes.size == 0:
        cls_target[:] = 0
        return cls_target, box_target, lmk_target, lmk_mask

    ious = bbox_iou(anchors, gt_boxes)  # [N, M]
    max_iou_per_anchor = ious.max(axis=1)
    best_gt_per_anchor = ious.argmax(axis=1)
    max_iou_per_gt = ious.max(axis=0)
    best_anchor_per_gt = ious.argmax(axis=0)

    # Negatives.
    cls_target[max_iou_per_anchor < neg_iou] = 0

    # Positives: every anchor matched to its best GT (standard RetinaFace rule).
    pos_mask = max_iou_per_anchor >= pos_iou
    cls_target[pos_mask] = 1

    # Ensure each GT has at least one positive anchor (its best match).
    for gt_idx, anchor_idx in enumerate(best_anchor_per_gt):
        cls_target[anchor_idx] = 1

    # Encode boxes for positives only.
    if pos_mask.any():
        pos_idx = np.where(pos_mask)[0]
        box_target[pos_idx] = encode_boxes(anchors[pos_idx], gt_boxes[best_gt_per_anchor[pos_idx]])

    # Assign landmarks.
    if gt_landmarks is not None and gt_landmarks.size > 0:
        lmk_anchor_mask = max_iou_per_anchor >= landmark_match_iou
        # Convert absolute landmarks to anchor-normalised offsets (RetinaFace
        # convention: divide by anchor size, like box regression but no
        # variance scaling — matching the original code).
        anchor_w = (anchors[:, 2] - anchors[:, 0]).clip(min=1e-6)
        anchor_h = (anchors[:, 3] - anchors[:, 1]).clip(min=1e-6)
        anchor_cx = (anchors[:, 0] + anchors[:, 2]) * 0.5
        anchor_cy = (anchors[:, 1] + anchors[:, 3]) * 0.5

        for anchor_idx in np.where(lmk_anchor_mask)[0]:
            gt_idx = best_gt_per_anchor[anchor_idx]
            lm = gt_landmarks[gt_idx]
            offsets = np.zeros(10, dtype=np.float32)
            offsets[0::2] = (lm[0::2] - anchor_cx[anchor_idx]) / anchor_w[anchor_idx]
            offsets[1::2] = (lm[1::2] - anchor_cy[anchor_idx]) / anchor_h[anchor_idx]
            lmk_target[anchor_idx] = offsets
            lmk_mask[anchor_idx] = 1.0

    # Mark anchors fully outside the image as ignore. We allow a small padding
    # so anchors whose centres sit near the image edge but extend slightly
    # outside (e.g. ``x_min = -8``) still count as positive.
    pad = 16.0
    inside = (
        (anchors[:, 0] >= -pad)
        & (anchors[:, 1] >= -pad)
        & (anchors[:, 2] < image_size + pad)
        & (anchors[:, 3] < image_size + pad)
    )
    cls_target[~inside] = -1

    return cls_target, box_target, lmk_target, lmk_mask


def decode_predictions(
    anchors: np.ndarray,
    cls_logits: np.ndarray,
    box_deltas: np.ndarray,
    lmk_deltas: np.ndarray | None,
    image_size: int,
    conf_threshold: float = 0.7,
    nms_iou: float = 0.5,
    top_k: int = 5000,
) -> dict[str, np.ndarray]:
    """Decode raw head outputs into (bboxes, scores, landmarks) on the input.

    Returns a dict with:
        boxes: ``[N, 4]`` xyxy, in input-image coords.
        scores: ``[N]`` float.
        landmarks: ``[N, 10]`` (only if head was provided) else ``None``.
    """
    from src.utils.nms import nms

    if cls_logits.size == 0:
        return {"boxes": np.zeros((0, 4), dtype=np.float32), "scores": np.zeros(0, dtype=np.float32),
                "landmarks": None}

    # Sigmoid on classification.
    scores = 1.0 / (1.0 + np.exp(-cls_logits))
    if scores.ndim == 3:
        # [N, 2] → keep face logit only (channel 1).
        scores = scores[..., 1]
    elif scores.ndim == 2 and scores.shape[-1] == 2:
        scores = scores[:, 1]
    scores = scores.reshape(-1)

    # Limit to top-k by score to keep NMS cheap.
    if scores.size > top_k:
        # argpartition(scores, -k)[-k:] gives the indices of the k largest scores.
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        anchors = anchors[top_idx]
        box_deltas = box_deltas[top_idx]
        scores = scores[top_idx]
        if lmk_deltas is not None:
            lmk_deltas = lmk_deltas[top_idx]
        cls_logits = cls_logits[top_idx]

    # Confidence filter.
    keep = scores > conf_threshold
    if not keep.any():
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros(0, dtype=np.float32),
            "landmarks": None,
        }

    anchors = anchors[keep]
    scores = scores[keep]
    box_deltas = box_deltas[keep]
    boxes = decode_boxes(anchors, box_deltas)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, image_size - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, image_size - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, image_size - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, image_size - 1)

    keep_idx = nms(boxes, scores, nms_iou)
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]

    landmarks_out = None
    if lmk_deltas is not None:
        lmk_deltas = lmk_deltas[keep]
        lmk_deltas = lmk_deltas[keep_idx]
        anchor_w = (anchors[keep_idx, 2] - anchors[keep_idx, 0]).clip(min=1e-6)
        anchor_h = (anchors[keep_idx, 3] - anchors[keep_idx, 1]).clip(min=1e-6)
        anchor_cx = (anchors[keep_idx, 0] + anchors[keep_idx, 2]) * 0.5
        anchor_cy = (anchors[keep_idx, 1] + anchors[keep_idx, 3]) * 0.5
        landmarks_out = np.zeros_like(lmk_deltas)
        landmarks_out[:, 0::2] = lmk_deltas[:, 0::2] * anchor_w[:, None] + anchor_cx[:, None]
        landmarks_out[:, 1::2] = lmk_deltas[:, 1::2] * anchor_h[:, None] + anchor_cy[:, None]
        landmarks_out = np.clip(landmarks_out, 0, image_size - 1)

    return {"boxes": boxes, "scores": scores, "landmarks": landmarks_out}


def levels_to_anchor_slices(
    cfg: AnchorConfig,
) -> tuple[tuple[int, int], ...]:
    """Return ``(start, end)`` index slices per FPN level (useful for slicing
    predictions when training).
    """
    slices: list[tuple[int, int]] = []
    cursor = 0
    for level_idx, stride in enumerate(cfg.strides):
        feature_dim = cfg.image_size // stride
        per_loc = cfg.scales_per_level * len(cfg.ratios)
        size = feature_dim * feature_dim * per_loc
        slices.append((cursor, cursor + size))
        cursor += size
    return tuple(slices)


def slice_per_level(arr: np.ndarray, slices: Iterable[tuple[int, int]]) -> list[np.ndarray]:
    """Split a flat ``[N, ...]`` array into per-level chunks."""
    return [arr[s:e] for s, e in slices]
