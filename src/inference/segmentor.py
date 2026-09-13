"""
Segmentation Inference

Provides the ``FaceSegmentor`` class that wraps a face segmentation model
(FCN-8s / U-Net) for inference on cropped face images.  Supports dependency
injection and falls back to a dummy model when the real model is unavailable.

Typical usage::

    seg = FaceSegmentor("weights/seg.pth", device="cpu")
    crop = cv2.imread("face_crop.jpg")
    result = seg.predict(crop)
    print(result['mask'].shape, result['class_names'][:3])
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except (ImportError, AttributeError, Exception):
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(_handler)
        logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Try to import the real segmentation model from teammate's code
# ---------------------------------------------------------------------------
_REAL_MODEL_AVAILABLE = False
try:
    from src.segmentation.model import build_face_segmentor
    _REAL_MODEL_AVAILABLE = True
    logger.info("Real segmentation model imported successfully.")
except (ImportError, Exception) as exc:
    logger.warning(
        "Could not import real segmentation model (%s). "
        "DummySegmentationModel will be used as fallback.",
        exc,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
CELEBAMASK_CLASSES: List[str] = [
    "background",  # 0
    "skin",         # 1
    "l_brow",       # 2
    "r_brow",       # 3
    "l_eye",        # 4
    "r_eye",        # 5
    "eye_g",        # 6  (glasses)
    "l_ear",        # 7
    "r_ear",        # 8
    "ear_r",        # 9  (earring)
    "nose",         # 10
    "mouth",        # 11
    "u_lip",        # 12
    "l_lip",        # 13
    "neck",         # 14
    "neck_l",       # 15 (necklace)
    "cloth",        # 16
    "hair",         # 17
    "hat",          # 18
]


# ═══════════════════════════════════════════════════════════════════════════
# Dummy segmentation model
# TODO: swap dummy model with real model.py once available
# ═══════════════════════════════════════════════════════════════════════════
class DummySegmentationModel(torch.nn.Module):
    """Stand-in that mimics the segmentation model ``forward()`` interface.

    Returns logits of shape ``(batch, num_classes, H, W)`` filled with
    zeros so that ``argmax`` produces an all-background mask.

    Args:
        num_classes: Number of segmentation classes (default 19).
    """

    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.num_classes = num_classes
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return zero logits → argmax gives class 0 (background)."""
        b, _, h, w = x.shape
        return torch.zeros(b, self.num_classes, h, w, device=x.device)


# ═══════════════════════════════════════════════════════════════════════════
# FaceSegmentor
# ═══════════════════════════════════════════════════════════════════════════
class FaceSegmentor:
    """High-level face segmentation inference wrapper.

    Args:
        checkpoint_path: Path to a ``.pth`` checkpoint.  Ignored when
            *model* is supplied.
        device: ``'cuda'`` or ``'cpu'``.
        input_size: ``(H, W)`` the face crop is resized to before
            inference.  Defaults to ``(512, 512)``.
        model: Pre-built ``nn.Module`` (dependency injection).
        model_type: Architecture when building from scratch
            (``'fcn8s'``, ``'unet'``, ``'lightweight'``).
        num_classes: Number of segmentation classes (default 19 for
            CelebAMask-HQ).

    Example::

        seg = FaceSegmentor(device="cpu")
        crop = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = seg.predict(crop)
        print(result['mask'].shape)
    """

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        input_size: tuple = (512, 512),
        model: Optional[torch.nn.Module] = None,
        model_type: str = "unet",
        num_classes: int = 19,
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.input_size = input_size  # (H, W)
        self.num_classes = num_classes
        self.class_names: List[str] = CELEBAMASK_CLASSES[:num_classes]

        # --- Load or inject model -----------------------------------------
        if model is not None:
            logger.info("Using injected segmentation model.")
            self.model = model
        elif checkpoint_path and os.path.isfile(checkpoint_path):
            logger.info("Loading segmentation checkpoint: %s", checkpoint_path)
            self.model = self._load_from_checkpoint(checkpoint_path, model_type)
        else:
            if checkpoint_path:
                logger.warning(
                    "Checkpoint not found: '%s'. Using dummy.", checkpoint_path
                )
            else:
                logger.warning("No checkpoint provided. Using dummy model.")
            # TODO: swap dummy model with real model.py once available
            self.model = DummySegmentationModel(num_classes)

        self.model.to(self.device)
        self.model.eval()
        logger.info("FaceSegmentor ready on %s.", self.device)

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _load_from_checkpoint(
        self, path: str, model_type: str
    ) -> torch.nn.Module:
        """Load a segmentation model from a checkpoint file.

        Args:
            path: Filesystem path to the ``.pth`` file.
            model_type: Architecture identifier
                (``'fcn8s'``, ``'unet'``, ``'lightweight'``).

        Returns:
            ``torch.nn.Module`` ready for inference.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Handle case where entire model was saved
        if isinstance(checkpoint, torch.nn.Module):
            return checkpoint

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if _REAL_MODEL_AVAILABLE:
            model = build_face_segmentor(
                model_type=model_type,
                num_classes=self.num_classes,
                pretrained=False,
            )
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(
                "Real model class unavailable; cannot load state_dict. "
                "Using dummy."
            )
            model = DummySegmentationModel(self.num_classes)

        return model

    def _preprocess(self, face_crop: np.ndarray) -> torch.Tensor:
        """Resize, normalise, and convert a face crop to a batched tensor.

        Args:
            face_crop: BGR ``numpy.ndarray`` of shape ``(H, W, 3)`` or
                grayscale ``(H, W)``.

        Returns:
            Float tensor ``(1, 3, H', W')``.
        """
        if face_crop.ndim == 2:
            face_crop = np.stack([face_crop] * 3, axis=-1)

        resized = cv2.resize(
            face_crop, (self.input_size[1], self.input_size[0])
        )

        # BGR → RGB, normalise
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - self._MEAN) / self._STD

        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, face_crop: np.ndarray) -> Dict[str, object]:
        """Run segmentation on a single face crop.

        Args:
            face_crop: BGR ``numpy.ndarray`` of shape ``(H, W, 3)``
                (uint8).

        Returns:
            Dict with keys:
                - ``mask``: ``np.ndarray`` of shape ``(H, W)`` with
                  integer values in ``[0, num_classes - 1]``.
                - ``class_names``: ordered list of class-name strings.
        """
        orig_h, orig_w = face_crop.shape[:2]

        tensor = self._preprocess(face_crop).to(self.device)
        logits = self.model(tensor)  # (1, C, H', W')

        # Resize logits back to original crop dimensions
        logits = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        return {
            "mask": mask,
            "class_names": list(self.class_names),
        }

    @torch.no_grad()
    def predict_batch(
        self, face_crops: List[np.ndarray]
    ) -> List[Dict[str, object]]:
        """Run segmentation on multiple face crops.

        Args:
            face_crops: List of BGR numpy arrays (uint8, HWC).

        Returns:
            List of result dicts, one per crop.
        """
        return [self.predict(crop) for crop in face_crops]


# ═══════════════════════════════════════════════════════════════════════════
# Quick self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("FaceSegmentor — smoke test")
    print("=" * 50)

    seg = FaceSegmentor(device="cpu")

    dummy_crop = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    res = seg.predict(dummy_crop)
    print(f"  Mask shape      : {res['mask'].shape}")
    print(f"  Unique values   : {np.unique(res['mask'])}")
    print(f"  Class names (#) : {len(res['class_names'])}")
    print(f"  First 5 classes : {res['class_names'][:5]}")

    # Batch
    batch_res = seg.predict_batch([dummy_crop, dummy_crop])
    print(f"  Batch count     : {len(batch_res)}")

    print("Smoke test passed ✓")
