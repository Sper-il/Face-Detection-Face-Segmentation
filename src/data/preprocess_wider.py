"""Preprocess WIDER FACE -> PNG + CSV annotations + 80/10/10 split.

Input layout (already on disk)::

    data/raw/WIDER_FACE/
    ├── WIDER_train/images/<event>/<file>.jpg
    ├── WIDER_val/images/<event>/<file>.jpg
    ├── WIDER_test/images/<event>/<file>.jpg
    └── wider_face_split/
        ├── wider_face_train_bbx_gt.txt
        └── wider_face_val_bbx_gt.txt

Output layout::

    data/processed/detection/
    ├── train/images/*.png       # 80% of train+val (seed=42)
    ├── train/annotations.csv
    ├── val/images/*.png         # 10%
    ├── val/annotations.csv
    ├── test/images/*.png        # 10% + official val set
    ├── test/annotations.csv
    └── DATASET.md

``annotations.csv`` layout (matches :class:`WIDERFaceDataset`)::

    image_id,x_min,y_min,x_max,y_max,confidence
    0_Parade_marchingband_1_5.png,9,14,64,64,1
    0_Parade_marchingband_1_5.png,75,19,99,47,1

The ``image_id`` is the bare file name (no event subdir), since we flatten
the directory tree on copy.

Usage::

    python -m src.data.preprocess_wider
    python -m src.data.preprocess_wider --max-images 500   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2

from src.utils.io import ensure_dir, set_seed

# Default paths (relative to repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "WIDER_FACE"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "processed" / "detection"

# WIDER FACE bbx files: ``x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose``
# (10 columns). We only need x1, y1, w, h → x_min, y_min, x_max, y_max.
# ``invalid`` flips to ``confidence`` (1 = valid box, 0 = ignore).
BBX_COLUMNS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess WIDER FACE -> PNG + CSV.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="If set, cap the total number of images (smoke test).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def parse_bbx_file(path: Path) -> dict[str, list[tuple[int, int, int, int, int]]]:
    """Parse a WIDER FACE ``*_bbx_gt.txt`` file.

    File format (one image per group)::

        <image_id>\n
        <N>\n
        <x y w h blur expr illum invalid occ pose>\n  (x N times)

    Returns ``{image_id: [(x, y, w, h, valid), ...]}`` where ``valid`` is the
    inverted ``invalid`` flag (1 = use the box, 0 = ignore during eval).
    """
    entries: dict[str, list[tuple[int, int, int, int, int]]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        lines = [line.rstrip("\n") for line in fh]
    i = 0
    while i < len(lines):
        image_id = lines[i].strip()
        i += 1
        if i >= len(lines):
            break
        try:
            n = int(lines[i].strip())
        except ValueError:
            # Defensive: if the count line isn't an int, skip.
            continue
        i += 1
        boxes: list[tuple[int, int, int, int, int]] = []
        for _ in range(n):
            if i >= len(lines):
                break
            parts = lines[i].split()
            i += 1
            if len(parts) < BBX_COLUMNS:
                continue
            x, y, w, h = (int(float(parts[i_])) for i_ in range(4))
            invalid = int(parts[7])
            boxes.append((x, y, w, h, 1 - invalid))
        if boxes:
            entries[image_id] = boxes
    return entries


def iter_train_image_paths(raw_root: Path) -> list[Path]:
    """Return every JPG under WIDER_train/images and WIDER_val/images."""
    paths: list[Path] = []
    for split in ("WIDER_train", "WIDER_val"):
        split_dir = raw_root / split / "images"
        if split_dir.exists():
            paths.extend(p for p in split_dir.rglob("*.jpg"))
    return sorted(paths)


def split_train_val_test(
    image_paths: list[Path], val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    """Stratified-ish 80/10/10 split (random here, since WIDER doesn't stratify)."""
    rng = random.Random(seed)
    indices = list(range(len(image_paths)))
    rng.shuffle(indices)

    n = len(image_paths)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    train = [image_paths[i] for i in train_idx]
    val = [image_paths[i] for i in val_idx]
    test = [image_paths[i] for i in test_idx]
    return train, val, test


def write_split(
    image_paths: list[Path],
    bbx_entries: dict[str, list[tuple[int, int, int, int, int]]],
    split_dir: Path,
    images_dir: Path,
    csv_path: Path,
    max_total: int | None,
) -> tuple[int, int]:
    """Copy images → PNG and write ``annotations.csv``.

    Returns ``(images_written, total_boxes)``.

    Directory layout — preserve WIDER FACE event folders
    ---------------------------------------------------
    The raw WIDER FACE dataset ships images in a *nested* layout::

        WIDER_train/images/0--Parade/0_Parade_marchingband_1_849.jpg
        WIDER_train/images/0--Parade/0_Parade_marchingband_1_851.jpg
        WIDER_train/images/0--Parade/0_Parade_Press_Conference_56_428.jpg
        WIDER_train/images/1--Handshaking/...

    Each event category is one sub-folder under ``<split>/images/``. We
    mirror that nested layout on output (``<split>/images/<event>/.png``)
    instead of flattening it, so the processed tree matches ``data/raw``.

    Why preserve the event folders?
      * **Semantics** — the category name is meaningful (Parade, Handshaking,
        Traffic, ...) and we keep it as a first-class folder rather than
        baking it only into the filename.
      * **Loader compatibility** — ``WIDERFaceDataset._resolve_paths``
        already understands the nested form (regex + basename index), so
        reading the data back is unchanged.
      * **Visual sanity check** — ``ls data/processed/detection/train/images``
        shows 61 event subfolders, exactly the same structure as
        ``data/raw/WIDER_FACE/WIDER_train/images``.

    The ``image_id`` column in ``annotations.csv`` is the **relative path**
    under ``<split>/images/`` (e.g. ``0--Parade/0_Parade_marchingband_1_849.png``),
    so a simple ``Path(images_dir) / image_id`` resolves the on-disk file.

    JPEG → PNG conversion (background for the presentation)
    --------------------------------------------------------
    The raw WIDER FACE dataset ships images as JPEG (``.jpg``). For an
    object-detection pipeline there are three reasons we re-encode them
    as PNG during preprocessing:

      1. **Lossless reproduction** — PNG uses deflate compression
         (Lempel-Ziv + Huffman, the same family as ZIP) which guarantees
         bit-exact reconstruction. JPEG uses an 8×8 DCT with a quantisation
         step that throws away detail; while imperceptible at high quality,
         it does change pixel values, and we want our annotations to stay
         in sync with the *exact* pixels the model was trained on.

      2. **Single-pass I/O** — once converted, every read in training is
         a fast PNG decode (no JPEG IDCT / Huffman tables / chroma
         subsampling to worry about). This matters when a DataLoader has
         4-8 worker processes all hammering the disk.

      3. **Format = extension in OpenCV** — ``cv2.imwrite(path, img)``
         picks the encoder purely from the suffix. So the same ndarray
         becomes JPEG or PNG depending on whether we pass ``"foo.jpg"`` or
         ``"foo.png"``. No codec parameters, no quality flags needed for
         PNG (it is *always* lossless for 8-bit channels).
    """
    ensure_dir(images_dir)
    images_written = 0
    rows: list[tuple[str, int, int, int, int, int]] = []

    for src in image_paths:
        if max_total is not None and images_written >= max_total:
            break

        # src is e.g. "<raw>/WIDER_train/images/0--Parade/0_Parade_marchingband_1_849.jpg"
        # rel_to_grandparent gives "0--Parade/0_Parade_marchingband_1_849.jpg" — exactly
        # the format the WIDER bbx files use as keys.
        rel = src.relative_to(src.parents[1])  # "<event>/<file>.jpg"
        event_dir, fname = rel.parent, rel.name
        # image_id preserves the nested path so the loader can do
        # ``images_dir / image_id`` directly: "0--Parade/0_Parade_marchingband_1_849.png"
        flat_id = Path(rel).as_posix().replace("\\", "/").rsplit(".", 1)[0] + ".png"

        # Mirror the event folder under images_dir.
        dst_dir = images_dir / event_dir
        ensure_dir(dst_dir)
        dst = dst_dir / Path(fname).with_suffix(".png").name

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[wider] skip unreadable: {src}")
            continue
        # Re-encode JPEG-decoded ndarray as PNG. cv2 picks PNG encoder
        # because ``dst`` ends in ".png" — lossless deflate compression,
        # no quality parameter needed.
        cv2.imwrite(str(dst), img)
        images_written += 1

        # WIDER FACE keys are like "0--Parade/0_Parade_marchingband_1_849.jpg"
        wider_id = str(rel).replace("\\", "/")
        for x, y, w, h, valid in bbx_entries.get(wider_id, []):
            x_max = x + w
            y_max = y + h
            rows.append((flat_id, x, y, x_max, y_max, valid))

    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "x_min", "y_min", "x_max", "y_max", "confidence"])
        writer.writerows(rows)

    return images_written, len(rows)


def write_dataset_card(out_root: Path, train_n: int, val_n: int, test_n: int) -> None:
    """Write a DATASET.md card with provenance + counts."""
    card = f"""# WIDER FACE (processed)

- **Source:** http://shuoyang1213.me/WIDERFACE/
- **Format:** PNG (decoded from JPG via OpenCV)
- **Splits (seed=42):** train={train_n}, val={val_n}, test={test_n}
- **Annotation format:** CSV (`image_id, x_min, y_min, x_max, y_max, confidence`)
- **Pre-processing script:** `src/data/preprocess_wider.py`
- **Original notes:** 61 event categories, ~32K images, ~393K faces,
  occlusion / pose / illumination labels.

> Generated by `python -m src.data.preprocess_wider` — do not edit by hand.
"""
    (out_root / "DATASET.md").write_text(card, encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    train_bbx = args.raw_root / "wider_face_split" / "wider_face_train_bbx_gt.txt"
    val_bbx = args.raw_root / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    if not train_bbx.exists() or not val_bbx.exists():
        raise FileNotFoundError(f"Missing bbx files under {args.raw_root}")

    print("[wider] parsing annotations...")
    bbx = parse_bbx_file(train_bbx)
    bbx.update(parse_bbx_file(val_bbx))

    image_paths = iter_train_image_paths(args.raw_root)
    print(f"[wider] found {len(image_paths)} candidate images")
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
        print(f"[wider] capped to {len(image_paths)} images for smoke test")

    train_paths, val_paths, test_paths = split_train_val_test(
        image_paths, args.val_ratio, args.test_ratio, args.seed
    )
    print(f"[wider] split: train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

    train_dir = args.out_root / "train"
    val_dir = args.out_root / "val"
    test_dir = args.out_root / "test"

    n_train, _ = write_split(
        train_paths, bbx, train_dir, train_dir / "images", train_dir / "annotations.csv", args.max_images
    )
    n_val, _ = write_split(
        val_paths, bbx, val_dir, val_dir / "images", val_dir / "annotations.csv", None
    )
    n_test, _ = write_split(
        test_paths, bbx, test_dir, test_dir / "images", test_dir / "annotations.csv", None
    )

    write_dataset_card(args.out_root, n_train, n_val, n_test)
    print(f"[wider] wrote {n_train + n_val + n_test} PNGs to {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
