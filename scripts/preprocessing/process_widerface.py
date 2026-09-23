"""
WIDER FACE Preprocessing Script.

Reads raw WIDER FACE images and annotations, normalizes bounding boxes,
splits into train/val, and writes a layout that mirrors celebamask_hq/:

    data/processed/wider_face/
        train/
            images/<split>_<original_filename>.jpg
            annotations.json   (only train entries)
        val/
            images/<split>_<original_filename>.jpg
            annotations.json   (only val entries)
        statistics/wider_face_stats.json

Each JSON entry has the same schema as the existing
wider_face_annotations.json (already present in the project):
    {
        "image": "images\\train_<original>.jpg",
        "original_image": "<original_relative_path>",
        "split": "train" | "val",
        "width": <W>,
        "height": <H>,
        "faces": [{"bbox": [cx, cy, w, h], "class": "face"}]
    }

Existing `data/processed/wider_face/annotations/wider_face_annotations.json`
is reused as-is if available; the script only verifies and refreshes the
new train/val/annotations.json files.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / 'data' / 'raw' / 'WIDER_FACE'
PROCESSED_ROOT = PROJECT_ROOT / 'data' / 'processed' / 'wider_face'
EXISTING_ANNOT = PROCESSED_ROOT / 'annotations' / 'wider_face_annotations.json'

SPLIT_TO_RAW = {'train': 'WIDER_train', 'val': 'WIDER_val'}


def parse_wider_face_bbx(path: Path) -> Dict[str, List[List[float]]]:
    """Parse wider_face_<split>_bbx_gt.txt file.

    Format:
        <relative/path/to.jpg>
        <num_boxes>
        x y w h blur expression illumination invalid occlusion pose  (one per box)
        ...

    Returns a dict mapping "category/filename.jpg" -> list of bboxes [x, y, w, h].
    """
    out: Dict[str, List[List[float]]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # Path lines always end in .jpg
        if not line.lower().endswith('.jpg'):
            i += 1
            continue
        cur_path = line
        i += 1
        if i >= n:
            break
        try:
            cur_count = int(lines[i])
        except ValueError:
            # Malformed: skip
            continue
        i += 1
        bboxes: List[List[float]] = []
        for _ in range(cur_count):
            if i >= n:
                break
            parts = lines[i].split()
            if len(parts) < 4:
                i += 1
                continue
            x, y, w, h = (float(parts[0]), float(parts[1]),
                           float(parts[2]), float(parts[3]))
            bboxes.append([x, y, w, h])
            i += 1
        out[cur_path] = bboxes
    return out


def load_existing_entries() -> Dict[str, List[dict]]:
    """Load existing combined annotations JSON keyed by split."""
    if not EXISTING_ANNOT.exists():
        return {'train': [], 'val': []}
    with open(EXISTING_ANNOT, 'r', encoding='utf-8') as f:
        data = json.load(f)
    by_split: Dict[str, List[dict]] = {'train': [], 'val': []}
    for entry in data:
        by_split.setdefault(entry.get('split', 'train'), []).append(entry)
    return by_split


def build_split(split: str,
                bbx_map: Dict[str, List[List[float]]],
                existing_entries: List[dict]) -> Tuple[List[dict], int, int]:
    """Resolve images for a split, copy them into train/val/images/,
    and rebuild normalized annotations.

    Returns (entries, copied_count, missing_count).
    """
    raw_dir = RAW_ROOT / SPLIT_TO_RAW[split] / 'images'
    out_dir = PROCESSED_ROOT / split / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: List[dict] = []
    copied = 0
    missing = 0

    # Build lookup by original_image for the existing entries
    by_orig = {e['original_image']: e for e in existing_entries}

    for rel_path, bboxes_xywh in tqdm(sorted(bbx_map.items()),
                                       desc=f"  [{split}] copying images",
                                       total=len(bbx_map)):
        src = raw_dir / rel_path
        if not src.exists():
            missing += 1
            continue

        # Preserve flat filename under processed/.../<split>/images/
        flat_name = f"{split}_{rel_path.replace('/', '_').replace('\\', '_')}"
        dst = out_dir / flat_name
        if not dst.exists():
            shutil.copy2(src, dst)

        # Read dims
        img = cv2.imread(str(dst))
        if img is None:
            missing += 1
            continue
        H, W = img.shape[:2]

        # Normalize bboxes to [0,1] cxcywh
        norm_boxes = []
        for x, y, w, h in bboxes_xywh:
            if W <= 0 or H <= 0 or w <= 0 or h <= 0:
                continue
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            nw = w / W
            nh = h / H
            # clamp to safe range
            cx = min(max(cx, nw / 2), 1 - nw / 2)
            cy = min(max(cy, nh / 2), 1 - nh / 2)
            norm_boxes.append({
                'bbox': [round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)],
                'class': 'face',
            })

        entry = {
            'image': f"images\\{flat_name}",
            'original_image': rel_path,
            'split': split,
            'width': W,
            'height': H,
            'faces': norm_boxes,
        }
        entries.append(entry)
        copied += 1

    return entries, copied, missing


def main():
    print("=" * 70)
    print("  WIDER FACE Preprocessing")
    print("=" * 70)

    # Load existing combined annotations (used for stats/reference only)
    by_split = load_existing_entries()
    print(f"Existing combined JSON -> train={len(by_split.get('train', []))} "
          f"val={len(by_split.get('val', []))}")

    # Parse official WIDER FACE bbx files
    train_bbx = parse_wider_face_bbx(
        RAW_ROOT / 'wider_face_split' / 'wider_face_train_bbx_gt.txt')
    val_bbx = parse_wider_face_bbx(
        RAW_ROOT / 'wider_face_split' / 'wider_face_val_bbx_gt.txt')
    print(f"Official bbx files -> train={len(train_bbx)} val={len(val_bbx)}")

    # Build per-split outputs
    train_entries, train_copied, train_missing = build_split(
        'train', train_bbx, by_split.get('train', []))
    val_entries, val_copied, val_missing = build_split(
        'val', val_bbx, by_split.get('val', []))

    # Write per-split annotations.json (the loader looks for these first)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    for split, entries in [('train', train_entries), ('val', val_entries)]:
        split_dir = PROCESSED_ROOT / split
        split_dir.mkdir(parents=True, exist_ok=True)
        # Also write a "split-name_annotations.json" for backwards compatibility
        (split_dir / 'annotations.json').write_text(
            json.dumps(entries, indent=2, ensure_ascii=False),
            encoding='utf-8')
        print(f"[{split}] wrote {len(entries)} entries -> {split_dir / 'annotations.json'}")

    # Stats
    stats = {
        'splits': {
            'train': {'copied': train_copied, 'missing_in_raw': train_missing},
            'val': {'copied': val_copied, 'missing_in_raw': val_missing},
        },
        'total_copied': train_copied + val_copied,
        'total_missing': train_missing + val_missing,
    }
    stats_path = PROCESSED_ROOT / 'statistics' / 'wider_face_stats.json'
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                          encoding='utf-8')
    print(f"\nStats -> {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
