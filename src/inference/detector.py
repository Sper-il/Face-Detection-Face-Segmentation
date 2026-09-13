"""
Face Detector for Inference

Provides the ``FaceDetector`` class that wraps a detection model (DSFD)
for inference on images.  Supports dependency injection of model objects
and falls back to a dummy model when the real model module is unavailable.

Typical usage::

    detector = FaceDetector("weights/det.pth", device="cpu")
    result = detector.predict("photo.jpg")
    print(result['boxes'], result['scores'])
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

# Try to import torchvision NMS (catch Exception — not just ImportError —
# because a torch / torchvision version mismatch can raise RuntimeError)
try:
    from torchvision.ops import nms as _torchvision_nms
except Exception:
    _torchvision_nms = None

# ---------------------------------------------------------------------------
# Logging: use project logger if available, else stdlib fallback
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
# Try to import the real detection model from teammate's code
# ---------------------------------------------------------------------------
_REAL_MODEL_AVAILABLE = False
try:
    from src.detection.model import DSFDDetector, build_dsfd_detector
    _REAL_MODEL_AVAILABLE = True
    logger.info("Real detection model (DSFDDetector) imported successfully.")
except (ImportError, Exception) as exc:
    logger.warning(
        "Could not import real detection model (%s). "
        "DummyDetectionModel will be used as fallback.",
        exc,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dummy detection model
# TODO: swap dummy model with real model.py once available
# ═══════════════════════════════════════════════════════════════════════════
class DummyDetectionModel(torch.nn.Module):
    """Minimal stand-in that mimics DSFDDetector's eval-mode output.

    The real DSFD model returns ``(batch, num_classes, top_k, 5)`` where
    each entry in the last dimension is ``[score, x1, y1, x2, y2]`` in
    normalised ``[0, 1]`` coordinates.  This dummy returns all zeros
    (i.e. no detections).

    Args:
        num_classes: Number of output classes (2 = background + face).
        top_k: Maximum number of detections per class.
    """

    def __init__(self, num_classes: int = 2, top_k: int = 500):
        super().__init__()
        self.num_classes = num_classes
        self.top_k = top_k
        # Single dummy parameter so torch treats this as a valid Module
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an empty detection tensor."""
        batch_size = x.size(0)
        return torch.zeros(
            batch_size, self.num_classes, self.top_k, 5, device=x.device
        )


# ═══════════════════════════════════════════════════════════════════════════
# FaceDetector
# ═══════════════════════════════════════════════════════════════════════════
class FaceDetector:
    """High-level face detection inference wrapper.

    Wraps a PyTorch detection model (DSFD by default) and exposes simple
    ``predict`` / ``predict_batch`` methods that accept flexible input
    types and return plain Python dicts.

    Args:
        checkpoint_path: Path to a ``.pth`` checkpoint file.  Ignored
            when *model* is supplied.
        device: Target device (``'cuda'`` or ``'cpu'``).
        conf_threshold: Minimum confidence to keep a detection.
        nms_threshold: IoU threshold for Non-Maximum Suppression.
        input_size: ``(H, W)`` tuple the image is resized to before
            feeding the model.  Defaults to ``(640, 640)``.
        model: An already-initialised ``nn.Module`` (dependency
            injection).  When given, *checkpoint_path* is ignored.

    Example::

        detector = FaceDetector("weights/det.pth", device="cpu")
        result = detector.predict("photo.jpg")
        print(result['boxes'], result['scores'])
    """

    # ImageNet normalisation used by ResNet-based backbones
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple = (640, 640),
        model: Optional[torch.nn.Module] = None,
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size  # (H, W)

        # --- Load or inject model -----------------------------------------
        if model is not None:
            logger.info("Using injected detection model.")
            self.model = model
        elif checkpoint_path and os.path.isfile(checkpoint_path):
            logger.info("Loading detection checkpoint: %s", checkpoint_path)
            self.model = self._load_from_checkpoint(checkpoint_path)
        else:
            if checkpoint_path:
                logger.warning(
                    "Checkpoint not found at '%s'. Falling back to dummy.",
                    checkpoint_path,
                )
            else:
                logger.warning("No checkpoint provided. Using dummy model.")
            # TODO: swap dummy model with real model.py once available
            self.model = DummyDetectionModel()

        self.model.to(self.device)
        self.model.eval()
        logger.info("FaceDetector ready on %s.", self.device)

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _load_from_checkpoint(self, path: str) -> torch.nn.Module:
        """Load a model from a checkpoint file.

        Args:
            path: Filesystem path to the ``.pth`` file.

        Returns:
            A ``torch.nn.Module`` ready for inference.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, torch.nn.Module):
            # Entire model was saved with ``torch.save(model, path)``
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
            model = build_dsfd_detector(pretrained=False)
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(
                "Real model class unavailable; cannot load state_dict. "
                "Using dummy."
            )
            model = DummyDetectionModel()

        return model

    @staticmethod
    def _load_image(
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> np.ndarray:
        """Convert various input types to a BGR ``numpy.ndarray`` (uint8, HWC).

        Args:
            image: File path, numpy array, or PIL Image.

        Returns:
            BGR numpy array of shape ``(H, W, 3)``.

        Raises:
            TypeError: If *image* is of an unsupported type.
        """
        if isinstance(image, (str, Path)):
            img = np.array(Image.open(str(image)).convert("RGB"))
            return img[:, :, ::-1].copy()  # RGB → BGR
        if isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
            return img[:, :, ::-1].copy()
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return np.stack([image] * 3, axis=-1)
            return image.copy()
        raise TypeError(f"Unsupported image type: {type(image)}")

    def _preprocess(self, bgr_image: np.ndarray) -> torch.Tensor:
        """Resize, normalise, and convert to a batched NCHW tensor.

        Args:
            bgr_image: BGR ``numpy.ndarray`` of shape ``(H, W, 3)``.

        Returns:
            Float tensor of shape ``(1, 3, H', W')``.
        """
        resized = cv2.resize(
            bgr_image, (self.input_size[1], self.input_size[0])
        )

        # BGR → RGB, uint8 → float32 [0, 1], normalise
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - self._MEAN) / self._STD

        # HWC → CHW → NCHW
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor

    def _postprocess(
        self,
        raw_output: torch.Tensor,
        orig_h: int,
        orig_w: int,
    ) -> Dict[str, list]:
        """Decode model output to boxes & scores in original pixel coords.

        Args:
            raw_output: Raw tensor from the model forward pass.
            orig_h: Original image height.
            orig_w: Original image width.

        Returns:
            Dict with ``'boxes'`` and ``'scores'`` lists.
        """
        # real DSFD output: (batch, num_classes, top_k, 5)
        #   last dim = [score, x1, y1, x2, y2] in normalised coords
        if raw_output.dim() == 4:
            face_preds = raw_output[0, 1]  # class 1 = face → (top_k, 5)
        elif raw_output.dim() == 3:
            face_preds = raw_output[0]
        else:
            return {"boxes": [], "scores": []}

        scores = face_preds[:, 0]
        boxes = face_preds[:, 1:5]

        # Confidence filter
        mask = scores > self.conf_threshold
        scores = scores[mask]
        boxes = boxes[mask]

        if scores.numel() == 0:
            return {"boxes": [], "scores": []}

        # Scale normalised [0,1] boxes → pixel coords
        boxes = boxes.clone()
        boxes[:, 0] *= orig_w  # x1
        boxes[:, 1] *= orig_h  # y1
        boxes[:, 2] *= orig_w  # x2
        boxes[:, 3] *= orig_h  # y2

        # Apply torchvision NMS if available
        if _torchvision_nms is not None:
            keep = _torchvision_nms(boxes, scores, self.nms_threshold)
            boxes = boxes[keep]
            scores = scores[keep]

        # Clamp to image bounds
        boxes[:, 0].clamp_(min=0, max=orig_w)
        boxes[:, 1].clamp_(min=0, max=orig_h)
        boxes[:, 2].clamp_(min=0, max=orig_w)
        boxes[:, 3].clamp_(min=0, max=orig_h)

        return {
            "boxes": boxes.cpu().tolist(),
            "scores": scores.cpu().tolist(),
        }

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict[str, list]:
        """Run face detection on a single image.

        Args:
            image: File path (``str`` / ``Path``), BGR ``numpy.ndarray``
                (uint8, HWC), or ``PIL.Image``.

        Returns:
            Dict with keys:
                - ``boxes``: list of ``[x1, y1, x2, y2]`` (pixel coords).
                - ``scores``: list of ``float`` confidence values.
        """
        bgr = self._load_image(image)
        orig_h, orig_w = bgr.shape[:2]

        tensor = self._preprocess(bgr).to(self.device)
        raw = self.model(tensor)

        return self._postprocess(raw, orig_h, orig_w)

    @torch.no_grad()
    def predict_batch(
        self, images: List[Union[str, Path, np.ndarray, Image.Image]]
    ) -> List[Dict[str, list]]:
        """Run face detection on a list of images.

        Args:
            images: Iterable of images (paths, arrays, or PIL images).

        Returns:
            List of result dicts, one per input image.
        """
        return [self.predict(img) for img in images]


# ═══════════════════════════════════════════════════════════════════════════
# Quick self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("FaceDetector — smoke test")
    print("=" * 50)

    det = FaceDetector(checkpoint_path=None, device="cpu")

    # Random dummy image
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    res = det.predict(dummy)
    print(f"  Boxes : {res['boxes']}")
    print(f"  Scores: {res['scores']}")

    # Batch test
    batch_res = det.predict_batch([dummy, dummy])
    print(f"  Batch results count: {len(batch_res)}")

    print("Smoke test passed ✓")
