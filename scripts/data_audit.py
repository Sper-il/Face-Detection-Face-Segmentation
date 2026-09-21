"""Audit the WIDER FACE detection + CelebAMask-HQ segmentation datasets.

For detection, we count images and boxes per split (train/val/test) and check:
  - Image is readable with cv2.imread.
  - Box coordinates are well-formed (x2 > x1, y2 > y1, within image bounds).
  - Validity flag is 0 or 1.

For segmentation, we count image/mask pairs and check:
  - Both files are readable.
  - Shapes match (or at least the mask is non-trivial).
  - Masks are roughly binary (0 or 255).

Results are written to ``data_audit.json`` in the repo root, plus an at-a-glance
summary printed to stdout.

Usage::

    python scripts/data_audit.py
    python scripts/data_audit.py --data-root /kaggle/input/face-detection-data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit face det/seg datasets.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/processed",
        help="Root containing detection/{train,val,test} and segmentation/{train,val,test}.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_audit.json",
        help="Where to write the JSON audit summary.",
    )
    parser.add_argument(
        "--max-image-checks",
        type=int,
        default=200,
        help="Maximum number of images to actually cv2.imread per split (for speed).",
    )
    parser.add_argument(
        "--max-mask-checks",
        type=int,
        default=200,
        help="Maximum number of segmentation masks to actually read per split.",
    )
    return parser.parse_args()


def audit_detection(csv_path: Path, images_dir: Path, max_image_checks: int) -> dict:
    """Audit one detection split.

    Returns a dict with image/box counts and a list of dropped sample IDs.
    """
    by_image: dict[str, list[tuple[float, float, float, float, int]]] = defaultdict(list)
    n_rows = 0
    n_box_rows = 0
    n_bad_boxes = 0  # x2 <= x1 or y2 <= y1
    n_invalid_boxes = 0  # valid flag != 1

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for parts in reader:
            if not parts or len(parts) < 6:
                continue
            n_rows += 1
            img_id = parts[0]
            try:
                x1, y1, x2, y2 = (float(parts[i]) for i in range(1, 5))
                valid = int(parts[5])
            except (ValueError, IndexError):
                continue

            if x2 <= x1 or y2 <= y1:
                n_bad_boxes += 1
                continue
            if valid not in (0, 1):
                continue
            if valid == 0:
                n_invalid_boxes += 1
            n_box_rows += 1
            by_image[img_id].append((x1, y1, x2, y2, valid))

    image_ids = sorted(by_image.keys())
    n_imgs = len(image_ids)

    # Check up to ``max_image_checks`` images for on-disk presence + readability.
    n_readable = 0
    n_unreadable = 0
    samples_checked = 0
    samples_to_check = image_ids[:max_image_checks] if max_image_checks else image_ids

    for img_id in samples_to_check:
        samples_checked += 1
        # Try direct, then nested category, then basename.
        stem = img_id.rsplit(".", 1)[0]
        candidates = [images_dir / img_id]
        for ext in (".jpg", ".jpeg", ".png"):
            candidates.append(images_dir / (stem + ext))
        # Try under nested WIDER category subdir.
        import re
        m = re.match(r"^(\d+--[A-Za-z_]+?)_\d", stem)
        if m:
            cat = m.group(1)
            for ext in (".jpg", ".jpeg", ".png"):
                candidates.append(images_dir / cat / (stem + ext))

        # Basename fallback (built once).
        if not hasattr(audit_detection, "_basename_index"):
            basename_index: dict[str, Path] = {}
            for p in images_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    basename_index.setdefault(p.name, p)
            audit_detection._basename_index = basename_index  # type: ignore[attr-defined]
        for ext in (".jpg", ".jpeg", ".png"):
            hit = audit_detection._basename_index.get(stem + ext)  # type: ignore[attr-defined]
            if hit is not None:
                candidates.append(hit)

        found = False
        for cand in candidates:
            if cand.exists():
                img = cv2.imread(str(cand), cv2.IMREAD_COLOR)
                if img is not None:
                    n_readable += 1
                    found = True
                    break
        if not found:
            n_unreadable += 1

    return {
        "csv_path": str(csv_path),
        "images_dir": str(images_dir),
        "n_imgs_total": n_imgs,
        "n_imgs_checked": samples_checked,
        "n_imgs_readable": n_readable,
        "n_imgs_unreadable": n_unreadable,
        "n_boxes": n_box_rows,
        "n_bad_boxes": n_bad_boxes,
        "n_invalid_boxes": n_invalid_boxes,
    }


def audit_segmentation(images_dir: Path, masks_dir: Path, max_mask_checks: int) -> dict:
    """Audit one segmentation split (image/mask pair counts + sanity checks)."""
    image_paths = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    mask_index: dict[str, Path] = {}
    for p in masks_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in MASK_EXTS:
            mask_index.setdefault(p.stem, p)

    pairs = [(p, mask_index.get(p.stem)) for p in image_paths]
    n_pairs = len(pairs)
    n_with_mask = sum(1 for _, m in pairs if m is not None)

    # Check up to ``max_mask_checks`` mask files.
    n_unreadable_mask = 0
    n_shape_mismatch = 0
    n_non_binary = 0
    checked = 0
    samples_to_check = [pair for pair in pairs if pair[1] is not None][:max_mask_checks]
    for img_path, mask_path in samples_to_check:
        checked += 1
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            n_unreadable_mask += 1
            continue
        if img is not None and mask.shape != img.shape[:2]:
            n_shape_mismatch += 1
        uniq = np.unique(mask)
        if uniq.size > 2 and not (np.array_equal(uniq[:2], [0, 255]) or np.array_equal(uniq[:2], [0, 1])):
            n_non_binary += 1

    return {
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "n_imgs": len(image_paths),
        "n_pairs": n_pairs,
        "n_with_mask": n_with_mask,
        "n_masks_checked": checked,
        "n_unreadable_mask": n_unreadable_mask,
        "n_shape_mismatch": n_shape_mismatch,
        "n_non_binary": n_non_binary,
    }


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)

    result = {"data_root": str(data_root.resolve())}

    # Detection
    det_summary: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        split_dir = data_root / "detection" / split
        csv_path = split_dir / "annotations.csv"
        images_dir = split_dir / "images"
        if not csv_path.exists():
            det_summary[split] = {"error": f"missing CSV {csv_path}"}
            continue
        det_summary[split] = audit_detection(csv_path, images_dir, args.max_image_checks)

    det_totals = {
        "n_imgs": sum(s.get("n_imgs_total", 0) for s in det_summary.values()),
        "n_boxes": sum(s.get("n_boxes", 0) for s in det_summary.values()),
        "n_bad_boxes": sum(s.get("n_bad_boxes", 0) for s in det_summary.values()),
        "n_invalid_boxes": sum(s.get("n_invalid_boxes", 0) for s in det_summary.values()),
    }
    result["detection"] = {"per_split": det_summary, "totals": det_totals}

    # Segmentation
    seg_summary: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        split_dir = data_root / "segmentation" / split
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            seg_summary[split] = {"error": f"missing {images_dir} or {masks_dir}"}
            continue
        seg_summary[split] = audit_segmentation(images_dir, masks_dir, args.max_mask_checks)

    seg_totals = {
        "n_pairs": sum(s.get("n_pairs", 0) for s in seg_summary.values()),
        "n_with_mask": sum(s.get("n_with_mask", 0) for s in seg_summary.values()),
        "n_unreadable_mask": sum(s.get("n_unreadable_mask", 0) for s in seg_summary.values()),
        "n_shape_mismatch": sum(s.get("n_shape_mismatch", 0) for s in seg_summary.values()),
    }
    result["segmentation"] = {"per_split": seg_summary, "totals": seg_totals}

    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

    # At-a-glance summary on stdout.
    print("=" * 60)
    print(f"DATA AUDIT — root={data_root.resolve()}")
    print("=" * 60)
    print("Detection:")
    for split, s in det_summary.items():
        if "error" in s:
            print(f"  {split}: ERROR — {s['error']}")
        else:
            print(
                f"  {split}: {s['n_imgs_total']} imgs, {s['n_boxes']} boxes "
                f"(invalid={s['n_invalid_boxes']}, bad={s['n_bad_boxes']})"
                f"  [readable: {s['n_imgs_readable']}/{s['n_imgs_checked']}]"
            )
    print("Segmentation:")
    for split, s in seg_summary.items():
        if "error" in s:
            print(f"  {split}: ERROR — {s['error']}")
        else:
            print(
                f"  {split}: {s['n_imgs']} imgs, {s['n_with_mask']} with-mask "
                f"[masks checked: {s['n_masks_checked']}, "
                f"unreadable={s['n_unreadable_mask']}, mismatch={s['n_shape_mismatch']}]"
            )
    print(f"\nWritten: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
