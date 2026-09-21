"""Evaluation utilities for the RetinaFace detector.

We implement a COCO-style mAP@0.5 + Recall@0.5 protocol that can run on the
WIDER FACE val / test CSVs without requiring ``pycocotools``.

The functions here are pure NumPy so they can be unit-tested without a GPU.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.detection.anchors import AnchorConfig, generate_anchors
from src.utils.box_ops import bbox_iou


@dataclass
class EvalEntry:
    """One ground-truth or prediction entry."""

    image_id: str
    bbox: np.ndarray  # [4] xyxy
    score: float = 1.0
    matched: bool = False


def load_gt(csv_path: str | Path) -> list[EvalEntry]:
    """Group the WIDER FACE CSV into a list of :class:`EvalEntry` (valid only)."""
    entries: list[EvalEntry] = []
    seen: set[str] = set()
    with open(csv_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            image_id = parts[0]
            # Skip header row produced by preprocess_wider.py.
            if image_id.lower() == "image_id":
                continue
            try:
                x1, y1, x2, y2 = (float(parts[i]) for i in range(1, 5))
                valid = int(parts[5])
            except ValueError:
                continue
            if valid == 0:
                continue  # ignore flag
            entries.append(EvalEntry(image_id, np.array([x1, y1, x2, y2], dtype=np.float32)))
            seen.add(image_id)
    return entries


def group_by_image(entries: Iterable[EvalEntry]) -> dict[str, list[EvalEntry]]:
    out: dict[str, list[EvalEntry]] = defaultdict(list)
    for e in entries:
        out[e.image_id].append(e)
    return out


def compute_map_recall(
    gt_entries: list[EvalEntry],
    pred_entries: list[EvalEntry],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute mAP@IoU and Recall@IoU.

    Predictions with no matching image_id are ignored. Predictions are matched
    greedily by score (highest first).
    """
    gt_by_img = group_by_image(gt_entries)
    pred_by_img = group_by_image(pred_entries)

    image_ids = sorted(set(gt_by_img.keys()) | set(pred_by_img.keys()))

    tp: list[float] = []
    fp: list[float] = []
    n_gt = 0

    for img_id in image_ids:
        gt = gt_by_img.get(img_id, [])
        pr = pred_by_img.get(img_id, [])
        n_gt += len(gt)
        if not gt:
            tp.extend([0.0] * len(pr))
            fp.extend([1.0] * len(pr))
            continue
        if not pr:
            continue

        gt_boxes = np.stack([e.bbox for e in gt], axis=0)
        matched = [False] * len(gt)
        # Sort predictions by descending score.
        order = sorted(range(len(pr)), key=lambda i: pr[i].score, reverse=True)
        for pi in order:
            pred_box = pr[pi].bbox.reshape(1, 4)
            ious = bbox_iou(pred_box, gt_boxes)[0]
            best_idx = int(ious.argmax())
            if ious[best_idx] >= iou_threshold and not matched[best_idx]:
                matched[best_idx] = True
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)

    if not tp:
        return {"mAP": 0.0, "recall": 0.0, "n_gt": n_gt, "n_pred": 0}

    tp_arr = np.asarray(tp)
    fp_arr = np.asarray(fp)
    scores = np.asarray([e.score for img in pred_by_img.values() for e in img])
    order = scores.argsort()[::-1]
    tp_arr = tp_arr[order]
    fp_arr = fp_arr[order]
    cum_tp = tp_arr.cumsum()
    cum_fp = fp_arr.cumsum()
    recall = cum_tp / max(1, n_gt)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-7)

    # 11-point interpolated AP (Pascal VOC convention — robust on small sets).
    recall_grid = np.linspace(0.0, 1.0, 11)
    # For every recall threshold r, precision_at_r = max(precision[recall >= r]).
    precision_at = np.zeros_like(recall_grid)
    for i, r in enumerate(recall_grid):
        mask = recall >= r
        precision_at[i] = float(precision[mask].max()) if mask.any() else 0.0
    ap = float(precision_at.mean())
    return {
        "mAP": ap,
        "recall": float(recall[-1]) if recall.size else 0.0,
        "n_gt": n_gt,
        "n_pred": int(len(tp)),
    }


def evaluate_map(
    detector,
    csv_path: str | Path,
    images_dir: str | Path,
    conf_threshold: float = 0.05,
    nms_iou: float = 0.5,
    iou_threshold: float = 0.5,
    max_images: int | None = None,
) -> dict[str, float]:
    """End-to-end detection evaluation.

    Loads the detector, iterates over the CSV's unique images, runs inference,
    and computes mAP + Recall at ``iou_threshold``.
    """
    from src.detection.inference import RetinaFaceDetector

    if not isinstance(detector, RetinaFaceDetector):
        raise TypeError("`detector` must be a RetinaFaceDetector instance")

    gt_entries = load_gt(csv_path)
    gt_by_img = group_by_image(gt_entries)
    image_ids = sorted(gt_by_img.keys())
    if max_images is not None:
        image_ids = image_ids[:max_images]

    import cv2

    pred_entries: list[EvalEntry] = []
    for img_id in image_ids:
        path = Path(images_dir) / img_id
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        pred = detector.predict(img, conf_threshold=conf_threshold, nms_iou=nms_iou)
        for box, score in zip(pred.boxes, pred.scores):
            pred_entries.append(EvalEntry(img_id, box, float(score)))

    return compute_map_recall(gt_entries, pred_entries, iou_threshold=iou_threshold)


def quick_synthetic_check(image_size: int = 320, n_anchors: int = 9) -> dict[str, float]:
    """Generate a tiny random AP sanity check (used by unit tests)."""
    rng = np.random.default_rng(0)
    gt = [
        EvalEntry("img0", np.array([10, 10, 50, 50], dtype=np.float32)),
        EvalEntry("img1", np.array([100, 100, 200, 200], dtype=np.float32)),
    ]
    pred = [
        EvalEntry("img0", np.array([12, 12, 52, 52], dtype=np.float32), score=0.9),
        EvalEntry("img1", np.array([102, 102, 198, 198], dtype=np.float32), score=0.8),
    ]
    return compute_map_recall(gt, pred, iou_threshold=0.5)


def total_anchors(cfg: AnchorConfig | None = None) -> int:
    return generate_anchors(cfg or AnchorConfig()).shape[0]
