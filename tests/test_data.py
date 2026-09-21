"""Dataset / split integrity tests.

These do *not* require the full datasets — they only check the
loaders' defensive behaviour and the no-overlap invariant on a
synthetic split.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


def _make_segmentation_split(root: Path, n: int) -> tuple[Path, Path]:
    img_dir = root / "images"
    mask_dir = root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cv2.imwrite(str(img_dir / f"{i}.png"), np.zeros((64, 64, 3), dtype=np.uint8))
        cv2.imwrite(str(mask_dir / f"{i}.png"), np.zeros((64, 64), dtype=np.uint8))
    return img_dir, mask_dir


def test_segmentation_dataset_length_and_item(tmp_path: Path):
    from src.segmentation.dataset import CelebAMaskHQDataset

    img_dir, mask_dir = _make_segmentation_split(tmp_path, n=5)
    ds = CelebAMaskHQDataset(img_dir, mask_dir, image_size=64, augment=False)
    assert len(ds) == 5
    sample = ds[0]
    assert sample.image.shape == (3, 64, 64)
    assert sample.mask.shape == (1, 64, 64)
    assert 0.0 <= float(sample.mask.min()) <= 1.0
    assert 0.0 <= float(sample.mask.max()) <= 1.0


def test_segmentation_dataset_missing_mask_is_zero(tmp_path: Path):
    from src.segmentation.dataset import CelebAMaskHQDataset

    img_dir, mask_dir = _make_segmentation_split(tmp_path, n=3)
    # Delete one mask file.
    (mask_dir / "0.png").unlink()
    ds = CelebAMaskHQDataset(img_dir, mask_dir, image_size=64, augment=False)
    sample = ds[0]
    assert sample.mask.sum() == 0


def test_no_overlap_between_train_and_test(tmp_path: Path):
    """We mock a split manifest as a JSON dict and check disjointness."""
    import json

    manifest = {
        "train": [f"{i}.png" for i in range(10)],
        "val": [f"{i}.png" for i in range(10, 12)],
        "test": [f"{i}.png" for i in range(12, 15)],
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    loaded = json.loads(path.read_text())
    assert set(loaded["train"]).isdisjoint(loaded["val"])
    assert set(loaded["train"]).isdisjoint(loaded["test"])
    assert set(loaded["val"]).isdisjoint(loaded["test"])


def test_detection_csv_grouping_roundtrip(tmp_path: Path):
    """Write a tiny WIDER FACE CSV and verify the loader groups rows correctly."""
    csv_text = (
        "img1.jpg,10,10,200,200,1\n"
        "img1.jpg,5,6,7,8,0\n"  # invalid → ignored
        "img2.jpg,50,50,250,250,1\n"
    )
    p = tmp_path / "annotations.csv"
    p.write_text(csv_text)
    from src.detection.anchors import AnchorConfig
    from src.detection.dataset import WIDERFaceDataset

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "img1.jpg"), np.zeros((320, 320, 3), dtype=np.uint8))
    cv2.imwrite(str(img_dir / "img2.jpg"), np.zeros((320, 320, 3), dtype=np.uint8))

    ds = WIDERFaceDataset(p, img_dir, AnchorConfig(image_size=320), augment=False)
    assert len(ds) == 2  # two images
    sample = ds[0]
    assert sample.cls_target.dtype == torch.int64
    # Some anchors must be marked as face (1) since img1 has a valid 190x190 box.
    assert int((sample.cls_target == 1).sum()) > 0


def test_detection_csv_handles_wider_nested_layout(tmp_path: Path):
    """The WIDER preprocess creates files at
    ``<images>/<category>/<category>_<rest>.jpg`` but stores flat keys in the
    CSV (``<category>_<rest>.png``). The dataset must bridge the two.
    """
    csv_text = (
        "0--Parade_0_Parade_marchingband_1_100.jpg,10,10,200,200,1\n"
        "1--Handshaking_1_Handshaking_Handshake_1_5.jpg,20,20,180,180,1\n"
    )
    p = tmp_path / "annotations.csv"
    p.write_text(csv_text)
    img_root = tmp_path / "images"
    (img_root / "0--Parade").mkdir(parents=True)
    (img_root / "1--Handshaking").mkdir(parents=True)
    cv2.imwrite(
        str(img_root / "0--Parade" / "0--Parade_0_Parade_marchingband_1_100.jpg"),
        np.zeros((320, 320, 3), dtype=np.uint8),
    )
    cv2.imwrite(
        str(img_root / "1--Handshaking" / "1--Handshaking_1_Handshaking_Handshake_1_5.jpg"),
        np.zeros((320, 320, 3), dtype=np.uint8),
    )
    from src.detection.anchors import AnchorConfig
    from src.detection.dataset import WIDERFaceDataset

    ds = WIDERFaceDataset(p, img_root, AnchorConfig(image_size=320), augment=False)
    assert len(ds) == 2
    # Both images should have positive anchors (their boxes are 190x160).
    for i in range(2):
        s = ds[i]
        assert int((s.cls_target == 1).sum()) > 0


# ----------------------------------------------------------------------
# Pre-processing helpers — defensive tests, no real datasets required.
# ----------------------------------------------------------------------


def test_split_indices_is_disjoint_and_correct_size():
    """``split_indices`` must return three disjoint index lists covering [0, n)."""
    from src.data.preprocess_celeb import split_indices

    n = 100
    train, val, test = split_indices(n, val_ratio=0.1, test_ratio=0.1, seed=42)
    union = set(train) | set(val) | set(test)
    assert len(union) == n
    assert set(train).isdisjoint(set(val))
    assert set(train).isdisjoint(set(test))
    assert set(val).isdisjoint(set(test))
    # 80/10/10 within ±1 image (rounding tolerance)
    assert abs(len(train) - 80) <= 1
    assert abs(len(val) - 10) <= 1
    assert abs(len(test) - 10) <= 1


def test_split_indices_is_deterministic():
    """Same seed → same split."""
    from src.data.preprocess_celeb import split_indices

    a_train, a_val, a_test = split_indices(50, 0.1, 0.1, seed=7)
    b_train, b_val, b_test = split_indices(50, 0.1, 0.1, seed=7)
    assert a_train == b_train
    assert a_val == b_val
    assert a_test == b_test


def test_wider_split_train_val_test_is_disjoint():
    """WIDER split must return disjoint lists with the expected sizes."""
    from src.data.preprocess_wider import split_train_val_test

    paths = [Path(f"img_{i}.jpg") for i in range(40)]
    train, val, test = split_train_val_test(paths, val_ratio=0.1, test_ratio=0.1, seed=42)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert len(train) + len(val) + len(test) == len(paths)


def test_wider_parse_bbx_file_groups_boxes_per_image(tmp_path: Path):
    """Parse a tiny ``wider_face_*_bbx_gt.txt`` and verify shape."""
    from src.data.preprocess_wider import parse_bbx_file

    txt = (
        "0--Parade/0_Parade_1.jpg\n"
        "2\n"
        "10 20 30 40 0 0 0 0 0 0\n"
        "1 2 3 4 0 0 0 1 0 0\n"  # invalid=1 → valid=0
        "1--Handshaking/1_Hand_1.jpg\n"
        "1\n"
        "5 5 10 10 0 0 0 0 0 0\n"
    )
    p = tmp_path / "bbx.txt"
    p.write_text(txt)
    out = parse_bbx_file(p)
    assert "0--Parade/0_Parade_1.jpg" in out
    assert len(out["0--Parade/0_Parade_1.jpg"]) == 2
    # The second box is invalid (confidence=0).
    assert out["0--Parade/0_Parade_1.jpg"][1][4] == 0
    assert len(out["1--Handshaking/1_Hand_1.jpg"]) == 1

