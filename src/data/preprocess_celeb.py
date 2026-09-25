"""Preprocess CelebAMask-HQ -> PNG images + binary face masks + 80/10/10 split.

Input layout (already on disk)::

    data/raw/CelebAMask-HQ/CelebAMask-HQ/
    ├── CelebA-HQ-img/<id>.jpg                 (30,000 face images)
    └── CelebAMask-HQ-mask-anno/
        ├── 0/  <id>.png    (each subdir holds one component-class per face id)
        ├── 1/
        ...
        └── 14/

CelebAMask-HQ provides 19 component classes (skin, hair, nose, eyes, …) split
across 15 subdirectories per face id. The canonical parsing is::

    id ∈ [0, 29999]
    for sub_idx in 0..14:
        files = sorted(glob(f"{sub_idx}/{id}*.png"))
        # Each component mask has a *single* colour (label) we OR into a single mask.

Output layout::

    data/processed/segmentation/
    ├── train/{images,masks}/*.png
    ├── val/{images,masks}/*.png
    ├── test/{images,masks}/*.png
    └── DATASET.md

Each mask is a *binary* PNG: 0 = background, 255 = any face-related class.
We treat classes ``{skin, nose, eye_g, eye_l, brow, ear, mouth, lip, neck,
cloth}`` as "face-related" — i.e. any non-hair, non-hat, non-background pixel.

Usage::

    python -m src.data.preprocess_celeb
    python -m src.data.preprocess_celeb --max-images 100   # smoke test
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.utils.io import ensure_dir, set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "CelebAMask-HQ" / "CelebAMask-HQ"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "processed" / "segmentation"

# CelebAMask-HQ splits 19 semantic classes across 15 subdirectories. Each mask
# in those subdirs is a single-colour image with the class label. We accept
# any non-zero pixel as "face-related" so the U-Net learns the full face
# silhouette (skin + hair + features + neck + ears). This matches what the
# WIDER FACE pipeline expects downstream.
FACE_LABEL_THRESHOLD = 1  # any pixel with value >= 1 is face


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CelebAMask-HQ -> PNG + masks.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def list_image_ids(raw_root: Path) -> list[int]:
    """Return all CelebAMask-HQ face ids (``int``) found under ``CelebA-HQ-img``.

    Background — Why .jpg → .png matters for ML preprocessing
    -------------------------------------------------------
    CelebAMask-HQ ships its **images as JPEG** (``.jpg``). JPEG is a *lossy*
    format designed for human eyeballs: it throws away high-frequency detail
    that we cannot easily see (blocky 8×8 DCT artefacts at low bit-rates).
    For machine-learning pipelines this is usually fine for the *image* itself,
    but for the **mask** we want a *lossless* format. PNG uses deflate
    compression (zip-style, no quality loss) which guarantees that:

      * every pixel we wrote is the pixel we get back,
      * no decoder drift when downstream code re-reads the mask.

    So even though our **input** images are .jpg, we write both the image AND
    the mask as .png in the processed output. OpenCV's ``cv2.imwrite`` does
    the format-switch automatically based on the file extension we give it:

      cv2.imwrite("foo.png", img)   # PNG → lossless, deflate
      cv2.imwrite("foo.jpg", img)   # JPEG → lossy, default quality=95

    The ``.jpg`` vs ``.png`` decision is one of those tiny details that
    separates a research script from a production-grade pipeline.
    """
    img_dir = raw_root / "CelebA-HQ-img"
    if not img_dir.exists():
        raise FileNotFoundError(f"Missing {img_dir}")
    ids: list[int] = []
    for p in img_dir.glob("*.jpg"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            continue
    return sorted(ids)


def compose_face_mask(raw_root: Path, face_id: int) -> np.ndarray | None:
    """OR together all component masks for ``face_id`` into a single uint8 mask.

    CelebAMask-HQ stores component files as ``<id:05d>_<class>.png`` (zero-padded
    5-digit id). Returns ``None`` if no components exist.
    """
    mask_dir = raw_root / "CelebAMask-HQ-mask-anno"
    composite: np.ndarray | None = None
    fid_str = f"{face_id:05d}"
    for sub_idx in range(15):
        sub_dir = mask_dir / str(sub_idx)
        if not sub_dir.exists():
            continue
        # Each face id appears as "<id>_<class>.png" inside its subdir.
        matches = sorted(sub_dir.glob(f"{fid_str}_*.png"))
        if not matches:
            continue
        for m in matches:
            comp = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            if comp is None:
                continue
            if composite is None:
                composite = (comp > 0).astype(np.uint8) * 255
            else:
                composite = np.maximum(composite, (comp > 0).astype(np.uint8) * 255)
    return composite


def split_indices(
    n: int, val_ratio: float, test_ratio: float, seed: int
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic 80/10/10 split on ``range(n)``.

    Returns ``(train_idx, val_idx, test_idx)``.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    return train_idx, val_idx, test_idx


def write_split(
    face_ids: list[int],
    raw_root: Path,
    images_dir: Path,
    masks_dir: Path,
    max_total: int | None,
) -> tuple[int, int]:
    """Copy images + assembled masks → PNG. Returns ``(written, skipped)``.

    Image format conversion — JPEG in, PNG out
    -----------------------------------------
    The raw CelebAMask-HQ images are JPEG (``cv2.imread(... IMREAD_COLOR)``
    decodes them into a BGR ``uint8`` ``ndarray``). We re-encode that array
    as PNG on disk (``cv2.imwrite(... ".png")``). The rationale:

      * **Lossless:**  PNG = deflate (no quality loss) → downstream mask
         alignment and pixel-level metrics (IoU, Dice) stay exact.
      * **Single channel for masks:**  cv2.imwrite writes 1-channel
         ``uint8`` PNG natively, no fake-RGB trickery needed.
      * **Filename consistency:**  ``flat_id = f"{fid:05d}.png"`` —
         image and mask share the *same* filename so the Dataset class
         can pair them by ``stem``.

    The conversion happens implicitly inside ``cv2.imwrite`` — the format
    is inferred from the suffix we give it. No explicit colour-space
    conversion is needed because the source images are already 8-bit BGR
    and the masks are 8-bit single-channel.
    """
    ensure_dir(images_dir)
    ensure_dir(masks_dir)

    src_img_dir = raw_root / "CelebA-HQ-img"
    written = 0
    skipped = 0

    for fid in face_ids:
        if max_total is not None and written >= max_total:
            break

        img_path = src_img_dir / f"{fid}.jpg"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue
        mask = compose_face_mask(raw_root, fid)
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Resize mask to image dims if needed (component masks are 512×512).
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        flat_id = f"{fid:05d}.png"
        # The .png extension is what tells cv2.imwrite to use lossless
        # deflate compression instead of JPEG's lossy DCT pipeline.
        # Mask stays single-channel uint8 → PNG is perfect for it.
        cv2.imwrite(str(images_dir / flat_id), img)
        cv2.imwrite(str(masks_dir / flat_id), mask)
        written += 1
        if written % 1000 == 0:
            print(f"[celeb] processed {written}/{len(face_ids)}")

    return written, skipped


def write_dataset_card(out_root: Path, train_n: int, val_n: int, test_n: int) -> None:
    card = f"""# CelebAMask-HQ (processed)

- **Source:** https://github.com/switchablenorms/CelebAMask-HQ
- **Format:** PNG images + binary PNG masks (0 = background, 255 = face-region)
- **Splits (seed=42):** train={train_n}, val={val_n}, test={test_n}
- **Mask composition:** OR together all 19 component classes (skin, hair, eyes,
  nose, mouth, ears, neck, clothing, …) — anything that is part of a "face region".
- **Pre-processing script:** `src/data/preprocess_celeb.py`

> Generated by `python -m src.data.preprocess_celeb` — do not edit by hand.
"""
    (out_root / "DATASET.md").write_text(card, encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    face_ids = list_image_ids(args.raw_root)
    print(f"[celeb] found {len(face_ids)} face ids under {args.raw_root}")
    if args.max_images is not None:
        face_ids = face_ids[: args.max_images]
        print(f"[celeb] capped to {len(face_ids)} for smoke test")

    train_idx, val_idx, test_idx = split_indices(
        len(face_ids), args.val_ratio, args.test_ratio, args.seed
    )
    train_ids = [face_ids[i] for i in train_idx]
    val_ids = [face_ids[i] for i in val_idx]
    test_ids = [face_ids[i] for i in test_idx]
    print(f"[celeb] split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    # Ensure dirs exist (don't wipe - allows resuming from partial runs).
    for split in ("train", "val", "test"):
        ensure_dir(args.out_root / split / "images")
        ensure_dir(args.out_root / split / "masks")

    n_train, _ = write_split(
        train_ids,
        args.raw_root,
        args.out_root / "train" / "images",
        args.out_root / "train" / "masks",
        args.max_images,
    )
    n_val, _ = write_split(
        val_ids,
        args.raw_root,
        args.out_root / "val" / "images",
        args.out_root / "val" / "masks",
        None,
    )
    n_test, _ = write_split(
        test_ids,
        args.raw_root,
        args.out_root / "test" / "images",
        args.out_root / "test" / "masks",
        None,
    )

    write_dataset_card(args.out_root, n_train, n_val, n_test)
    print(f"[celeb] wrote {n_train + n_val + n_test} (image, mask) pairs to {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
