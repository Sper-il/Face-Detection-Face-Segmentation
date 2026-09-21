"""Post-processing unit tests (NMS, morphology, mask ops, viz)."""

from __future__ import annotations

import numpy as np


def test_paste_mask_into_image():
    from src.utils.mask_ops import paste_mask_into_image

    cropped = np.ones((40, 50), dtype=np.uint8) * 255
    out = paste_mask_into_image(cropped, bbox=(10, 20, 60, 60), image_shape=(200, 200))
    assert out.shape == (200, 200)
    # Mask should be present at the bbox rectangle.
    assert out[30:50, 10:60].sum() > 0


def test_paste_mask_into_image_clips_at_edges():
    from src.utils.mask_ops import paste_mask_into_image

    cropped = np.ones((10, 10), dtype=np.uint8) * 255
    out = paste_mask_into_image(cropped, bbox=(-5, -5, 5, 5), image_shape=(20, 20))
    assert out.shape == (20, 20)
    # We must not crash and we must not write outside the image.
    assert out[19, 19] == 0  # bottom-right corner is outside the bbox.


def test_draw_boxes_returns_correct_shape():
    from src.utils.visualizer import draw_boxes

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_boxes(img, [(10, 10, 30, 30)], scores=[0.9])
    assert out.shape == img.shape
    # The label should have been written onto a non-zero pixel area.
    assert out.sum() > 0


def test_overlay_combines_masks_and_boxes():
    from src.utils.visualizer import overlay

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    boxes = np.array([[20, 20, 60, 60]], dtype=np.float32)
    out = overlay(img, boxes, [mask], scores=np.array([0.9]))
    assert out.shape == img.shape
    assert (out != 0).any()


def test_mask_open_removes_small_islands():
    from src.utils.mask_ops import mask_open

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    mask[5:7, 5:7] = 255  # tiny island
    out = mask_open(mask, kernel_size=3)
    assert out[5:7, 5:7].sum() == 0
    assert out[30, 30] == 255


def test_morph_close_fills_small_holes():
    from src.utils.mask_ops import morph_close

    mask = np.ones((40, 40), dtype=np.uint8) * 255
    mask[20, 20] = 0  # single-pixel hole
    out = morph_close(mask, kernel_size=3)
    assert out[20, 20] == 255
