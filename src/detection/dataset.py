"""WIDER FACE dataset loader for the RetinaFace detector.

The CSV layout is:

    image_id, x_min, y_min, x_max, y_max, confidence, mask_path
    0--Parade/0_Parade_marchingband_1_5.jpg, 9,  14, 64, 64, 1, ''
    0--Parade/0_Parade_marchingband_1_5.jpg, 75, 19, 99, 47, 1, ''

Rows for the same image are grouped together. ``confidence`` is the WIDER
FACE ``invalid`` flag inverted (1 = valid box, 0 = ignore). Landmarks are
*not* part of this layout — we train landmark heads only on the GT boxes
whose ``invalid == 0`` and assign landmark targets as zeros when the dataset
doesn't ship them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.detection.anchors import AnchorConfig, generate_anchors
from src.detection.boxes import encode_boxes_torch  # noqa: F401  (re-export)


@dataclass
class DetectionSample:
    """A single training item, already encoded against anchors."""

    image: torch.Tensor  # [3, H, W] float in [0, 1]
    cls_target: torch.Tensor  # [N] int64 (-1 = ignore, 0 = bg, 1 = face)
    box_target: torch.Tensor  # [N, 4] float
    lmk_target: torch.Tensor  # [N, 10] float (zeros when no landmark)
    lmk_mask: torch.Tensor  # [N] float
    image_id: str
    orig_size: tuple[int, int]


class WIDERFaceDataset(Dataset):
    """Load + preprocess WIDER FACE PNGs for the RetinaFace detector."""

    NORMALIZE_MEAN = (0.485, 0.456, 0.406)
    NORMALIZE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        csv_path: str | Path,
        images_dir: str | Path,
        anchor_cfg: AnchorConfig,
        augment: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)
        self.anchor_cfg = anchor_cfg
        self.augment = augment

        self._paths = self._resolve_paths(self.csv_path, self.images_dir)
        self._groups = self._group_by_image(self.csv_path)
        self._anchors = generate_anchors(anchor_cfg)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, idx: int) -> DetectionSample:
        image_id, boxes, valid = self._groups[idx]
        image_path = self._paths[image_id]
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read {image_path}")

        img, (ratio, pad_w, pad_h) = self._preprocess(img_bgr, augment=self.augment)
        boxes_scaled = self._rescale_boxes(boxes, ratio, pad_w, pad_h)

        cls_t, box_t, lmk_t, lmk_mask = self._encode_targets(boxes_scaled, valid)
        return DetectionSample(
            image=img,
            cls_target=torch.from_numpy(cls_t),
            box_target=torch.from_numpy(box_t),
            lmk_target=torch.from_numpy(lmk_t),
            lmk_mask=torch.from_numpy(lmk_mask),
            image_id=image_id,
            orig_size=(img_bgr.shape[0], img_bgr.shape[1]),
        )

    # ------------------------------------------------------------------
    def anchors(self) -> np.ndarray:
        return self._anchors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_paths(
        self, csv_path: Path, images_dir: Path
    ) -> dict[str, Path]:
        """Map every ``image_id`` in ``csv_path`` to an on-disk path.

        The WIDER FACE preprocess writes a nested layout
        (``<images_dir>/<category>/<category>_<rest>.jpg``) but the CSV
        stores flat keys (``<category>_<rest>.png``). This helper bridges
        the two by:

          1. Trying direct join first (works when the dataset is already flat).
          2. Extracting the WIDER category prefix (``<n>--<word>(_<word>)*``)
             and looking for ``<images_dir>/<category>/<flat>.jpg``.
          3. Falling back to a basename index built from ``rglob``.

        Returns ``{image_id: resolved_path}`` for every entry that could be
        located on disk; entries that cannot be located are silently dropped
        (caller's responsibility to log / fail on empty results).
        """
        import re

        # Pre-build basename index once.
        basename_index: dict[str, Path] = {}
        for p in images_dir.rglob("*"):
            if p.is_file():
                basename_index.setdefault(p.name, p)

        out: dict[str, Path] = {}
        seen_ids: set[str] = set()
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                image_id = parts[0]
                if image_id in seen_ids:
                    continue
                seen_ids.add(image_id)
                if image_id.lower() == "image_id":
                    continue
                # 1. Direct join.
                direct = images_dir / image_id
                if direct.exists():
                    out[image_id] = direct
                    continue
                # 2. WIDER FACE nested category layout.
                stem = image_id.rsplit(".", 1)[0]
                m = re.match(r"^(\d+--[A-Za-z_]+?)_\d", stem)
                if m:
                    cat = m.group(1)
                    for ext in (".jpg", ".png", ".jpeg"):
                        cand = images_dir / cat / (stem + ext)
                        if cand.exists():
                            out[image_id] = cand
                            break
                    else:
                        # 3. Fallback: basename lookup.
                        for ext in (".jpg", ".png", ".jpeg"):
                            hit = basename_index.get(stem + ext)
                            if hit is not None:
                                out[image_id] = hit
                                break
                else:
                    for ext in (".jpg", ".png", ".jpeg"):
                        hit = basename_index.get(stem + ext)
                        if hit is not None:
                            out[image_id] = hit
                            break
        return out

    def _group_by_image(
        self, csv_path: Path
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        groups: list[tuple[str, np.ndarray, np.ndarray]] = []
        current_id: Optional[str] = None
        current_boxes: list[list[float]] = []
        current_valid: list[int] = []

        def flush() -> None:
            nonlocal current_id, current_boxes, current_valid
            if current_id is not None:
                boxes = np.asarray(current_boxes, dtype=np.float32).reshape(-1, 4)
                valid = np.asarray(current_valid, dtype=np.int64)
                groups.append((current_id, boxes, valid))
            current_id = None
            current_boxes = []
            current_valid = []

        with open(csv_path, "r", encoding="utf-8") as fh:
            first = True
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                # Skip the header row (always present — written by our preprocessor).
                if first:
                    first = False
                    if line.lower().startswith("image_id"):
                        continue
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                image_id = parts[0]
                if image_id != current_id:
                    flush()
                    current_id = image_id
                x_min, y_min, x_max, y_max = (float(parts[i]) for i in range(1, 5))
                valid = int(parts[5])
                current_boxes.append([x_min, y_min, x_max, y_max])
                current_valid.append(valid)
        flush()
        # Drop entries that we couldn't resolve on disk — caller has no way to
        # load the image otherwise, and we want `len(dataset)` to reflect
        # actually-usable samples.
        return [(iid, boxes, valid) for (iid, boxes, valid) in groups if iid in self._paths]

    def _preprocess(
        self, img_bgr: np.ndarray, augment: bool
    ) -> tuple[torch.Tensor, tuple[float, int, int]]:
        h0, w0 = img_bgr.shape[:2]
        target = self.anchor_cfg.image_size

        ratio = min(target / h0, target / w0)
        new_h, new_w = int(round(h0 * ratio)), int(round(w0 * ratio))
        pad_w = target - new_w
        pad_h = target - new_h

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized,
            top=0,
            bottom=pad_h,
            left=0,
            right=pad_w,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        if augment:
            padded = self._maybe_augment(padded)

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray(self.NORMALIZE_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(self.NORMALIZE_STD, dtype=np.float32).reshape(3, 1, 1)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        tensor = (tensor - torch.from_numpy(mean)) / torch.from_numpy(std)
        return tensor, (ratio, pad_w, pad_h)

    def _maybe_augment(self, padded_bgr: np.ndarray) -> np.ndarray:
        # Lightweight in-pipeline augmentations to keep the dataset dependency-free.
        if np.random.rand() < 0.5:
            padded_bgr = padded_bgr[:, ::-1, :].copy()
        if np.random.rand() < 0.3:
            # Color jitter.
            hsv = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + np.random.randint(-10, 11)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] + np.random.randint(-30, 31), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] + np.random.randint(-30, 31), 0, 255)
            padded_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return padded_bgr

    def _rescale_boxes(
        self,
        boxes: np.ndarray,
        ratio: float,
        pad_w: int,
        pad_h: int,
    ) -> np.ndarray:
        if boxes.size == 0:
            return boxes.copy()
        out = boxes.astype(np.float32).copy()
        out[:, [0, 2]] = out[:, [0, 2]] * ratio
        out[:, [1, 3]] = out[:, [1, 3]] * ratio
        return out

    def _encode_targets(
        self,
        boxes: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from src.detection.anchors import assign_targets

        if boxes.size == 0:
            cls_target = np.zeros(self._anchors.shape[0], dtype=np.int64)
            box_target = np.zeros((self._anchors.shape[0], 4), dtype=np.float32)
            lmk_target = np.zeros((self._anchors.shape[0], 10), dtype=np.float32)
            lmk_mask = np.zeros(self._anchors.shape[0], dtype=np.float32)
            return cls_target, box_target, lmk_target, lmk_mask

        # Only positive (valid) boxes go to assign_targets.
        keep = valid == 1
        pos_boxes = boxes[keep] if keep.any() else boxes
        cls_t, box_t, lmk_t, lmk_mask = assign_targets(
            anchors=self._anchors,
            gt_boxes=pos_boxes,
            gt_landmarks=None,
            image_size=self.anchor_cfg.image_size,
        )

        # Boxes with valid == 0 → force cls=-1 (ignore) so they don't train.
        if (~keep).any():
            ignored_anchors = self._anchors_for_boxes(boxes[~keep])
            for a in ignored_anchors:
                cls_t[a] = -1
                box_t[a] = 0
                lmk_t[a] = 0
                lmk_mask[a] = 0
        return cls_t, box_t, lmk_t, lmk_mask

    def _anchors_for_boxes(self, gt_boxes: np.ndarray) -> list[int]:
        """Return indices of anchors that overlap any of ``gt_boxes`` ≥ 0.5."""
        from src.utils.box_ops import bbox_iou

        if gt_boxes.size == 0:
            return []
        ious = bbox_iou(self._anchors, gt_boxes)
        max_iou = ious.max(axis=1) if ious.size else np.zeros(self._anchors.shape[0])
        return list(np.where(max_iou >= 0.5)[0].tolist())


def detection_collate(batch: list[DetectionSample]) -> dict[str, torch.Tensor]:
    """Default collate — every anchor is shared so we just stack."""
    images = torch.stack([b.image for b in batch], dim=0)
    cls = torch.stack([b.cls_target for b in batch], dim=0)
    box = torch.stack([b.box_target for b in batch], dim=0)
    lmk = torch.stack([b.lmk_target for b in batch], dim=0)
    mask = torch.stack([b.lmk_mask for b in batch], dim=0)
    return {
        "images": images,
        "cls_target": cls,
        "box_target": box,
        "lmk_target": lmk,
        "lmk_mask": mask,
        "image_ids": [b.image_id for b in batch],
        "orig_sizes": [b.orig_size for b in batch],
    }
