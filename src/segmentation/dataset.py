"""CelebAMask-HQ dataset loader for the U-Net segmentor.

Expects a directory layout::

    data/processed/segmentation/
    ├── train/
    │   ├── images/00000.png
    │   └── masks/00000.png
    ├── val/...
    └── test/...

Each mask is single-channel PNG with ``0`` = background and ``255`` = face.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.mask_ops import mask_morphology  # noqa: F401  (re-export)


@dataclass
class SegmentationSample:
    image: torch.Tensor  # [3, H, W]
    mask: torch.Tensor  # [1, H, W] float in [0, 1]
    image_id: str


class CelebAMaskHQDataset(Dataset):
    """CelebAMask-HQ loader (PNG only)."""

    NORMALIZE_MEAN = (0.485, 0.456, 0.406)
    NORMALIZE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        image_size: int = 512,
        augment: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.augment = augment

        self._image_files = sorted(self.images_dir.glob("*.png"))
        if not self._image_files:
            raise FileNotFoundError(f"No PNGs in {self.images_dir}")

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> SegmentationSample:
        img_path = self._image_files[idx]
        mask_path = self.masks_dir / img_path.name
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        else:
            mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

        img_t, mask_t = self._preprocess(img_bgr, mask)
        return SegmentationSample(image=img_t, mask=mask_t, image_id=img_path.name)

    # ------------------------------------------------------------------
    def _preprocess(
        self, img_bgr: np.ndarray, mask: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h0, w0 = img_bgr.shape[:2]
        target = self.image_size

        # Resize keeping aspect ratio + pad to square (so mask alignment is easy).
        ratio = min(target / h0, target / w0)
        new_h, new_w = int(round(h0 * ratio)), int(round(w0 * ratio))
        pad_w = target - new_w
        pad_h = target - new_h

        img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        img_padded = cv2.copyMakeBorder(
            img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        mask_padded = cv2.copyMakeBorder(
            mask_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
        )

        if self.augment:
            img_padded, mask_padded = self._maybe_augment(img_padded, mask_padded)

        rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray(self.NORMALIZE_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(self.NORMALIZE_STD, dtype=np.float32).reshape(3, 1, 1)
        img_t = torch.from_numpy(rgb).permute(2, 0, 1)
        img_t = (img_t - torch.from_numpy(mean)) / torch.from_numpy(std)

        # Binarise: any non-zero pixel in CelebAMask-HQ means "this is a face-class".
        bin_mask = (mask_padded > 0).astype(np.float32)
        mask_t = torch.from_numpy(bin_mask).unsqueeze(0)
        return img_t, mask_t

    def _maybe_augment(
        self, img: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()
        if np.random.rand() < 0.3:
            # Color jitter only.
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + np.random.randint(-10, 11)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] + np.random.randint(-30, 31), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] + np.random.randint(-30, 31), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if np.random.rand() < 0.3:
            scale = float(np.random.uniform(0.8, 1.2))
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # Centre crop / pad back to original size.
            target_h, target_w = img.shape[:2], img.shape[:2]
            # Restore to image_size square.
            target = self.image_size
            ratio = min(target / new_h, target / new_w)
            fh, fw = int(round(new_h * ratio)), int(round(new_w * ratio))
            pad_w = target - fw
            pad_h = target - fh
            img = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        return img, mask


def segmentation_collate(batch: list[SegmentationSample]) -> dict[str, torch.Tensor]:
    return {
        "images": torch.stack([b.image for b in batch], dim=0),
        "masks": torch.stack([b.mask for b in batch], dim=0),
        "image_ids": [b.image_id for b in batch],
    }
