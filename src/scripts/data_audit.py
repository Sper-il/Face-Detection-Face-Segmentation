"""Run a complete data audit and write a JSON report.

This script inspects:

- Detection: counts images, boxes, and pure-int image readability per split.
  Skips entries whose image can't be opened or whose box coords are invalid.
- Segmentation: counts image/mask pairs, verifies mask is binary and matches
  image shape.

Output is a single ``data_audit.json`` dict so the v23 notebook can pick it
up and fail fast on any path mismatch.

Usage::

    python -m src.scripts.data_audit --data-root /kaggle/input/face-detection-data
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

# Silence the noisy cv2 decoder warnings on missing files (those are expected
# in local dry-runs where detection images haven't been downloaded yet).
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# Suppress OpenCV's stderr spam (decoder warnings on missing files).
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def audit_detection(data_root: Path, sample_size: int | None = None) -> dict:
    """Audit WIDER FACE processed detection CSV + images.

    Args:
        sample_size: If set, only check the first ``sample_size`` images
            per split (useful for quick local dry-runs where images are
            missing).
    """
    report: dict = {"per_split": {}}
    total_imgs = 0
    total_boxes = 0
    total_dropped = 0

    for split in ("train", "val", "test"):
        csv_path = data_root / "data/processed/detection" / split / "annotations.csv"
        if not csv_path.exists():
            report["per_split"][split] = {"error": f"missing {csv_path}"}
            continue

        img_dir = csv_path.parent / "images"
        img_ids: set[str] = set()
        boxes_per_img: dict[str, list] = defaultdict(list)

        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            for row in reader:
                if len(row) < 6:
                    continue
                img_id = row[0]
                img_ids.add(img_id)
                try:
                    x1, y1, x2, y2 = (float(row[i]) for i in range(1, 5))
                    valid = int(row[5])
                except ValueError:
                    continue
                boxes_per_img[img_id].append([x1, y1, x2, y2, valid])

        all_img_ids = sorted(img_ids)
        if sample_size is not None:
            all_img_ids = all_img_ids[:sample_size]
        n_imgs = len(all_img_ids)
        n_boxes = sum(len(boxes_per_img[iid]) for iid in all_img_ids)
        print(f"  [det/{split}] csv: {n_imgs} imgs / {n_boxes} boxes", flush=True)
        n_dropped_unreadable = 0
        n_dropped_invalid_box = 0
        n_imgs_kept = 0

        for i, img_id in enumerate(all_img_ids):
            img_path = img_dir / img_id
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                n_dropped_unreadable += 1
                continue
            h0, w0 = img.shape[:2]
            valid_flag = True
            for (x1, y1, x2, y2, valid) in boxes_per_img[img_id]:
                if valid == 0:
                    continue  # "ignore" boxes are valid annotations
                if x2 <= x1 or y2 <= y1:
                    n_dropped_invalid_box += 1
                    valid_flag = False
                if x1 < 0 or y1 < 0 or x2 > w0 or y2 > h0:
                    n_dropped_invalid_box += 1
                    valid_flag = False
            if valid_flag:
                n_imgs_kept += 1
            if (i + 1) % 5000 == 0:
                print(f"    ... {i+1}/{n_imgs}", flush=True)

        report["per_split"][split] = {
            "n_imgs_total": n_imgs,
            "n_imgs_readable": n_imgs_kept,
            "n_imgs_unreadable": n_dropped_unreadable,
            "n_boxes": n_boxes,
            "n_invalid_boxes": n_dropped_invalid_box,
        }
        total_imgs += n_imgs
        total_boxes += n_boxes
        total_dropped += n_dropped_unreadable + n_dropped_invalid_box

    report["totals"] = {
        "n_imgs": total_imgs,
        "n_boxes": total_boxes,
        "n_dropped": total_dropped,
    }
    return report


def audit_segmentation(data_root: Path, sample_size: int | None = None) -> dict:
    """Audit CelebAMask-HQ processed segmentation images + masks."""
    report: dict = {"per_split": {}}
    total_pairs = 0
    total_dropped = 0

    for split in ("train", "val", "test"):
        img_dir = data_root / "data/processed/segmentation" / split / "images"
        mask_dir = data_root / "data/processed/segmentation" / split / "masks"
        if not img_dir.exists() or not mask_dir.exists():
            report["per_split"][split] = {
                "error": f"missing dir: img={img_dir.exists()} mask={mask_dir.exists()}"
            }
            continue

        img_files = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        if sample_size is not None:
            img_files = img_files[:sample_size]
        n_imgs = len(img_files)
        n_with_mask = 0
        n_unreadable_img = 0
        n_unreadable_mask = 0
        n_shape_mismatch = 0
        n_non_binary = 0

        print(f"  [seg/{split}] scanning {n_imgs} files...", flush=True)
        for i, ip in enumerate(img_files):
            img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            if img is None:
                n_unreadable_img += 1
                continue
            # Mask may be .png or .jpg; try same stem with each mask ext.
            mp = None
            for ext in (".png", ".jpg", ".jpeg"):
                cand = mask_dir / (ip.stem + ext)
                if cand.exists():
                    mp = cand
                    break
            if mp is None:
                continue
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                n_unreadable_mask += 1
                continue
            if mask.shape != img.shape[:2]:
                n_shape_mismatch += 1
                continue
            uniq = np.unique(mask)
            if not set(uniq.tolist()).issubset({0, 255}):
                # Treat any non-zero pixel as face-related, so we accept other values too.
                n_non_binary += 1
            n_with_mask += 1
            if (i + 1) % 5000 == 0:
                print(f"    ... {i+1}/{n_imgs}", flush=True)

        report["per_split"][split] = {
            "n_imgs": n_imgs,
            "n_pairs": n_with_mask,
            "n_unreadable_img": n_unreadable_img,
            "n_unreadable_mask": n_unreadable_mask,
            "n_shape_mismatch": n_shape_mismatch,
            "n_non_binary": n_non_binary,
        }
        total_pairs += n_with_mask
        total_dropped += n_unreadable_img + n_unreadable_mask + n_shape_mismatch

    report["totals"] = {"n_pairs": total_pairs, "n_dropped": total_dropped}
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("data_audit.json"))
    ap.add_argument(
        "--det-sample-size",
        type=int,
        default=None,
        help="If set, only audit the first N detection images per split.",
    )
    ap.add_argument(
        "--seg-sample-size",
        type=int,
        default=None,
        help="If set, only audit the first N segmentation images per split.",
    )
    args = ap.parse_args()

    report = {
        "data_root": str(args.data_root),
        "detection": audit_detection(args.data_root, sample_size=args.det_sample_size),
        "segmentation": audit_segmentation(args.data_root, sample_size=args.seg_sample_size),
    }

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
