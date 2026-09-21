"""Unit tests for the detection utilities (boxes, anchors, NMS)."""

from __future__ import annotations

import numpy as np
import pytest


def test_xyxy_xywh_roundtrip():
    from src.utils.box_ops import xywh_to_xyxy, xyxy_to_xywh

    boxes = np.array([[10, 20, 50, 60], [0, 0, 100, 200]], dtype=np.float32)
    roundtrip = xywh_to_xyxy(xyxy_to_xywh(boxes))
    assert np.allclose(boxes, roundtrip)


def test_clip_boxes_clamps_to_image():
    from src.utils.box_ops import clip_boxes

    boxes = np.array([[-5, -5, 200, 200], [10, 10, 50, 50]], dtype=np.float32)
    clipped = clip_boxes(boxes, width=100, height=100)
    assert clipped[0, 0] == 0
    assert clipped[0, 1] == 0
    assert clipped[0, 2] == 99
    assert clipped[0, 3] == 99
    assert np.array_equal(clipped[1], boxes[1])


def test_bbox_iou_zero_for_disjoint():
    from src.utils.box_ops import bbox_iou

    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[20, 20, 30, 30]], dtype=np.float32)
    assert float(bbox_iou(a, b)[0, 0]) < 1e-6


def test_bbox_iou_one_for_identical():
    from src.utils.box_ops import bbox_iou

    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert float(bbox_iou(a, a)[0, 0]) > 0.999


def test_nms_keeps_highest_score():
    from src.utils.nms import nms

    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2]


def test_batched_nms_does_not_suppress_other_classes():
    from src.utils.nms import batched_nms

    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)
    keep = batched_nms(boxes, scores, classes, iou_threshold=0.5)
    assert keep.tolist() == [0, 1]


def test_anchor_count_matches_config(tiny_anchor_cfg):
    from src.detection.anchors import generate_anchors

    anchors = generate_anchors(tiny_anchor_cfg)
    # 9 anchors × (stride**2) cells per level.
    expected = 9 * (
        (tiny_anchor_cfg.image_size // tiny_anchor_cfg.strides[0]) ** 2
        + (tiny_anchor_cfg.image_size // tiny_anchor_cfg.strides[1]) ** 2
        + (tiny_anchor_cfg.image_size // tiny_anchor_cfg.strides[2]) ** 2
    )
    assert anchors.shape == (expected, 4)
    assert anchors.dtype == np.float32


def test_assign_targets_emits_three_classes(tiny_anchor_cfg):
    from src.detection.anchors import assign_targets, generate_anchors

    anchors = generate_anchors(tiny_anchor_cfg)
    gt = np.array([[20, 20, 60, 60]], dtype=np.float32)
    cls, box, lmk, mask = assign_targets(anchors, gt, None, image_size=tiny_anchor_cfg.image_size)
    assert cls.dtype == np.int64
    assert set(cls.tolist()).issubset({-1, 0, 1})
    assert int((cls == 1).sum()) > 0
    assert box.shape == (anchors.shape[0], 4)
    assert lmk.shape == (anchors.shape[0], 10)
    assert mask.shape == (anchors.shape[0],)


def test_eval_synthetic_high_recall():
    from src.detection.eval import EvalEntry, compute_map_recall

    gt = [EvalEntry("a", np.array([0, 0, 10, 10], dtype=np.float32))]
    pred = [
        EvalEntry("a", np.array([1, 1, 11, 11], dtype=np.float32), score=0.9),
        EvalEntry("a", np.array([100, 100, 110, 110], dtype=np.float32), score=0.5),
    ]
    out = compute_map_recall(gt, pred, iou_threshold=0.5)
    # One correct detection, one false positive → recall = 1.0
    assert out["recall"] == 1.0
    assert out["n_pred"] == 2


def test_eval_synthetic_perfect_precision():
    """Two GTs, two perfect predictions: mAP should be 1.0."""
    from src.detection.eval import EvalEntry, compute_map_recall

    gt = [
        EvalEntry("a", np.array([0, 0, 10, 10], dtype=np.float32)),
        EvalEntry("b", np.array([100, 100, 110, 110], dtype=np.float32)),
    ]
    pred = [
        EvalEntry("a", np.array([1, 1, 11, 11], dtype=np.float32), score=0.9),
        EvalEntry("b", np.array([100, 100, 110, 110], dtype=np.float32), score=0.7),
    ]
    out = compute_map_recall(gt, pred, iou_threshold=0.5)
    # Both predictions are TP — recall = 1.0 and mAP = 1.0
    assert out["recall"] == 1.0
    # mAP at this scale is the precision×recall sum at the single threshold.
    assert out["mAP"] >= 0.99


def test_load_gt_skips_invalid_flag():
    from src.detection.eval import load_gt

    import tempfile, os

    csv = "img1,1,2,3,4,1\nimg1,5,6,7,8,0\nimg2,10,10,20,20,1\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as fh:
        fh.write(csv)
        path = fh.name
    try:
        gt = load_gt(path)
        # Only img1's first row (valid=1) and img2 should remain.
        assert len(gt) == 2
    finally:
        os.unlink(path)
