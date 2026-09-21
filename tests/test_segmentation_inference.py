"""Unit tests for the segmentation module."""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_unet_forward_shape():
    from src.segmentation.unet import UNet, UNetConfig

    model = UNet(UNetConfig(pretrained_encoder=False))
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 128, 128)
    assert torch.all(y >= 0) and torch.all(y <= 1)


def test_unet_at_multiple_sizes():
    from src.segmentation.unet import UNet, UNetConfig

    model = UNet(UNetConfig(pretrained_encoder=False))
    model.eval()
    for size in [128, 256]:
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 1, size, size)


def test_dice_loss_perfect_prediction_zero():
    from src.segmentation.losses import DiceLoss

    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 4:12, 4:12] = 1.0
    # Logits that yield very confident correct prediction.
    logits = (mask * 20) - 10
    loss = DiceLoss()(logits, mask)
    assert float(loss) < 1e-3


def test_bce_dice_loss_accepts_probs_or_logits():
    from src.segmentation.losses import BCEDiceLoss

    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, 4:12, 4:12] = 1.0
    probs = torch.full_like(mask, 0.5)  # uncertain
    logits = torch.zeros_like(mask)
    crit = BCEDiceLoss()
    out_probs = crit(probs, mask)
    out_logits = crit(logits, mask)
    # Both calls should succeed and return finite values.
    assert torch.isfinite(out_probs["total"])
    assert torch.isfinite(out_logits["total"])


def test_compute_segmentation_metrics_perfect_match():
    from src.segmentation.eval import compute_segmentation_metrics

    mask = torch.zeros(1, 1, 32, 32)
    mask[0, 0, 8:24, 8:24] = 1.0
    m = compute_segmentation_metrics(mask, mask)
    assert m["iou"] > 0.999
    assert m["dice"] > 0.999
    assert m["pixel_acc"] > 0.999


def test_compute_segmentation_metrics_no_face():
    from src.segmentation.eval import compute_segmentation_metrics

    empty = torch.zeros(1, 1, 16, 16)
    m = compute_segmentation_metrics(empty, empty)
    # All-zero masks: IoU is degenerate — we use the eps-smoothed version.
    assert 0.0 <= m["iou"] <= 1.0


def test_numpy_metrics_basic():
    from src.segmentation.eval import numpy_metrics

    pred = np.zeros((64, 64), dtype=np.uint8)
    gt = np.zeros((64, 64), dtype=np.uint8)
    pred[10:50, 10:50] = 255
    gt[12:48, 12:48] = 255
    m = numpy_metrics(pred, gt)
    assert 0.5 < m["iou"] < 1.0
    assert 0.5 < m["dice"] < 1.0


def test_segmentor_predict_crop_shape(small_segmentor, small_face_image):
    res = small_segmentor.predict_crop(small_face_image)
    # predict_crop returns a mask at the *crop* resolution (i.e. matching the
    # input crop pixel-for-pixel).
    assert res.mask.shape == small_face_image.shape[:2]
    assert res.mask.dtype.name == "uint8"


def test_segmentor_predict_full_shape(small_segmentor, small_face_image):
    res = small_segmentor.predict_full(small_face_image)
    assert res.mask.shape == small_face_image.shape[:2]
    assert res.mask.dtype.name == "uint8"


def test_segmentor_segment_faces_no_boxes(small_segmentor, small_face_image):
    boxes = np.zeros((0, 4), dtype=np.float32)
    out = small_segmentor.segment_faces(small_face_image, boxes)
    assert out == []


def test_segmentor_segment_faces_with_box(small_segmentor, small_face_image):
    boxes = np.array([[100, 100, 220, 220]], dtype=np.float32)
    out = small_segmentor.segment_faces(small_face_image, boxes, margin=0.1)
    assert len(out) == 1
    # ``segment_faces`` returns a mask at the *crop* resolution (the cropped
    # region of the input image plus the margin). The pipeline orchestrator
    # handles the paste-back into the full image.
    crop_shape = out[0].mask.shape
    # Crop region is bbox + 10% margin on each side. Bbox is 120×120 → crop is
    # 120 + 12 + 12 ≈ 144 on each side.
    assert crop_shape[0] >= 130 and crop_shape[0] <= 150
    assert crop_shape[1] >= 130 and crop_shape[1] <= 150


def test_segmentor_predict_full(small_segmentor, small_face_image):
    res = small_segmentor.predict_full(small_face_image)
    assert res.mask.shape == small_face_image.shape[:2]


def test_mask_in_bbox_zeros_outside():
    from src.utils.mask_ops import mask_in_bbox

    mask = np.ones((32, 32), dtype=np.uint8) * 255
    out = mask_in_bbox(mask, bbox=(8, 8, 24, 24))
    # Corners outside the bbox are zero.
    assert out[0, 0] == 0
    assert out[0, 31] == 0
    assert out[31, 0] == 0
    assert out[31, 31] == 0
    # Center inside bbox is preserved.
    assert out[16, 16] == 255


def test_morphology_pipeline_runs():
    from src.utils.mask_ops import mask_morphology

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[28:36, 28:36] = 255
    out = mask_morphology(mask, opening_kernel=3, closing_kernel=5)
    assert out.shape == mask.shape
