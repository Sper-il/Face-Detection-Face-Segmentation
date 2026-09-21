"""End-to-end pipeline + stage unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def test_pipeline_no_face_returns_failure(pipeline, empty_image, tmp_path):
    out_path = tmp_path / "out.png"
    res = pipeline.run(empty_image, save_overlay_to=str(out_path))
    assert res.failure_reason == "no_face_detected"
    assert res.overlay is not None
    # The overlay file should still exist (graceful degradation).
    assert out_path.exists()


def test_pipeline_single_face_synthetic(pipeline, small_face_image, tmp_path):
    out_path = tmp_path / "overlay.png"
    res = pipeline.run(small_face_image, save_overlay_to=str(out_path))
    # Random weights → may or may not detect a face, but it must not crash.
    assert res.overlay is not None
    assert res.image_path is None
    assert out_path.exists()


def test_pipeline_many_faces_caps_max(pipeline, many_faces_image, tmp_path):
    # Set an aggressive cap; pipeline should still return a valid result.
    pipeline.max_faces = 3
    res = pipeline.run(many_faces_image, save_overlay_to=str(tmp_path / "many.png"))
    assert res.overlay is not None
    assert len(res.masks) <= 3


def test_pipeline_load_failure(pipeline):
    res = pipeline.run("/nonexistent/path.png")
    assert res.failure_reason is not None
    assert "load_error" in res.failure_reason or "runtime_error" in res.failure_reason


def test_pipeline_batch_writes_summary(pipeline, small_image_path, tmp_path):
    res = pipeline.run_batch([str(small_image_path)], output_dir=tmp_path)
    assert len(res) == 1
    summary = tmp_path / "results.json"
    assert summary.exists()
    data = json.loads(summary.read_text())
    assert isinstance(data, list)
    assert "boxes" in data[0]


def test_run_detection_no_face_returns_empty(pipeline, empty_image):
    from src.pipeline.stages import run_detection

    res = run_detection(empty_image, pipeline.detector, conf_threshold=0.99, nms_iou=0.5)
    assert res.boxes.size == 0
    assert res.scores.size == 0


def test_run_segmentation_empty_boxes(pipeline, small_face_image):
    from src.pipeline.stages import DetectionStageResult, run_segmentation

    det = DetectionStageResult(image=small_face_image, boxes=np.zeros((0, 4)), scores=np.zeros(0), landmarks=None)
    res = run_segmentation(det, pipeline.segmentor)
    assert res.masks == []


def test_render_empty_overlay_writes_caption(empty_image):
    from src.pipeline.visualizer import render_empty_overlay

    out = render_empty_overlay(empty_image)
    assert out.shape == empty_image.shape
    assert (out != empty_image).any()  # the caption changed some pixels


def test_render_overlay_includes_mask_colours(small_face_image):
    from src.pipeline.visualizer import render_overlay

    boxes = np.array([[100, 100, 220, 220]], dtype=np.float32)
    mask = np.zeros((small_face_image.shape[0], small_face_image.shape[1]), dtype=np.uint8)
    mask[120:200, 120:200] = 255
    out = render_overlay(small_face_image, boxes, [mask])
    assert out.shape == small_face_image.shape
    # The overlay should have non-black pixels in the bbox region.
    assert (out[100:220, 100:220] != 0).any()


def test_iter_image_paths_single_file(tmp_path):
    from src.pipeline.stages import iter_image_paths

    p = tmp_path / "a.png"
    cv2.imwrite(str(p), np.zeros((8, 8, 3), dtype=np.uint8))
    paths = list(iter_image_paths(str(p)))
    assert paths == [str(p)]


def test_iter_image_paths_directory(tmp_path):
    from src.pipeline.stages import iter_image_paths

    for i in range(3):
        cv2.imwrite(str(tmp_path / f"img_{i}.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    # Create a non-image file to confirm it is skipped.
    (tmp_path / "ignore.txt").write_text("not an image")
    paths = list(iter_image_paths(str(tmp_path)))
    assert len(paths) == 3
    assert all(p.endswith(".png") for p in paths)
