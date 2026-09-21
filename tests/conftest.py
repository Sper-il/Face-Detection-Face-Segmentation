"""Shared pytest fixtures.

These fixtures build *small* models (no pretrained backbone) so the test
suite runs quickly on a CPU machine. Where the production code loads
pretrained weights, we override it explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

# Make `src` importable when pytest is invoked from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_image_path(tmp_path: Path) -> Path:
    """Write a synthetic 320×320 BGR PNG to a temp file and return the path."""
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    cv2.circle(img, (160, 160), 80, (180, 180, 180), -1)
    cv2.rectangle(img, (100, 100), (220, 220), (200, 200, 200), 3)
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def small_face_image() -> np.ndarray:
    """In-memory 320×320 BGR image with a single fake face."""
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    cv2.circle(img, (160, 160), 80, (180, 180, 180), -1)
    return img


@pytest.fixture
def empty_image() -> np.ndarray:
    """All-zero 256×256 BGR image (no faces, should not crash)."""
    return np.zeros((256, 256, 3), dtype=np.uint8)


@pytest.fixture
def many_faces_image() -> np.ndarray:
    """Synthetic image with 20 small gray patches pretending to be faces."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    coords = [(x, y) for x in range(50, 600, 100) for y in range(50, 400, 80)]
    for (x, y) in coords[:20]:
        cv2.circle(img, (x, y), 30, (180, 180, 180), -1)
    return img


@pytest.fixture
def tiny_anchor_cfg():
    """Tiny anchor config used by detection tests."""
    from src.detection.anchors import AnchorConfig

    return AnchorConfig(image_size=160, strides=(8, 16, 32), base_sizes=(8, 32, 128))


@pytest.fixture
def small_detector(tiny_anchor_cfg):
    """A RetinaFaceDetector with random weights and tiny input size."""
    from src.detection.anchors import generate_anchors
    from src.detection.inference import RetinaFaceDetector

    det = RetinaFaceDetector(weights=None, device="cpu", anchor_cfg=tiny_anchor_cfg, pretrained=False)
    # Recompute anchors to match the (smaller, custom) config.
    det._anchors = generate_anchors(tiny_anchor_cfg)
    return det


@pytest.fixture
def small_segmentor():
    """A UNetSegmentor with random weights."""
    from src.segmentation.inference import UNetSegmentor

    return UNetSegmentor(weights=None, device="cpu", image_size=128, threshold=0.5)


@pytest.fixture
def pipeline(small_detector, small_segmentor):
    """End-to-end pipeline with random weights + capped faces for fast tests."""
    from src.pipeline import FaceSegmentationPipeline

    return FaceSegmentationPipeline(
        detector_weights=None,
        segmentor_weights=None,
        device="cpu",
        # High conf_threshold so random-init detections are filtered out.
        conf_threshold=0.9999,
        nms_iou=0.5,
        max_faces=5,
    )
