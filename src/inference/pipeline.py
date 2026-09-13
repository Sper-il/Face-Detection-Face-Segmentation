"""
Inference Pipeline - Combined Detection + Segmentation

Provides the ``FacePipeline`` class that chains ``FaceDetector`` and
``FaceSegmentor`` to detect faces in an image and produce per-face
segmentation masks.

Typical usage::

    pipeline = create_pipeline(device="cpu")
    result = pipeline.process("photo.jpg")
    for face in result['faces']:
        print(face['bbox'], face['score'], face['mask'].shape)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
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

from src.inference.detector import FaceDetector
from src.inference.segmentor import FaceSegmentor, CELEBAMASK_CLASSES


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════

# Colour palette for the 19 CelebAMask-HQ classes (BGR order for OpenCV)
MASK_PALETTE: List[tuple] = [
    (0,   0,   0),     # 0  background
    (204, 195, 176),   # 1  skin
    (0,   153, 255),   # 2  l_brow
    (0,   153, 255),   # 3  r_brow
    (51,  255, 255),   # 4  l_eye
    (51,  255, 255),   # 5  r_eye
    (204, 153, 255),   # 6  eye_g (glasses)
    (76,  153, 0),     # 7  l_ear
    (76,  153, 0),     # 8  r_ear
    (204, 204, 0),     # 9  ear_r (earring)
    (255, 128, 0),     # 10 nose
    (0,   0,   255),   # 11 mouth
    (255, 0,   85),    # 12 u_lip
    (255, 0,   170),   # 13 l_lip
    (0,   128, 128),   # 14 neck
    (128, 128, 255),   # 15 neck_l (necklace)
    (0,   76,  153),   # 16 cloth
    (0,   255, 0),     # 17 hair
    (255, 0,   0),     # 18 hat
]


def draw_results(
    image: np.ndarray,
    faces: List[Dict],
    mask_alpha: float = 0.45,
    box_color: tuple = (0, 255, 0),
    box_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes and mask overlays on a copy of *image*.

    Args:
        image: BGR ``numpy.ndarray`` (uint8, HWC).
        faces: List of face dicts as returned by
            ``FacePipeline.process()``.
        mask_alpha: Opacity for the mask overlay (0 = transparent,
            1 = opaque).
        box_color: BGR colour for bounding boxes.
        box_thickness: Line thickness for boxes.
        font_scale: Font scale for the score label.

    Returns:
        Annotated BGR image (same shape as input).
    """
    vis = image.copy()

    for face in faces:
        bbox = face["bbox"]
        score = face["score"]
        mask = face.get("mask")

        # --- Bounding box --------------------------------------------------
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, box_thickness)

        label = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(
            vis, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA,
        )

        # --- Mask overlay --------------------------------------------------
        if mask is not None and mask.size > 0:
            h_img, w_img = vis.shape[:2]
            crop_h = y2 - y1
            crop_w = x2 - x1
            if crop_h <= 0 or crop_w <= 0:
                continue

            # Resize mask to match the box region in the image
            mask_resized = cv2.resize(
                mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
            )

            # Build coloured overlay
            overlay = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
            for cls_id in range(1, len(MASK_PALETTE)):  # skip background
                overlay[mask_resized == cls_id] = MASK_PALETTE[cls_id]

            # Blend into the image region
            roi = vis[y1:y2, x1:x2]
            if roi.shape[:2] == overlay.shape[:2]:
                non_bg = mask_resized > 0
                blended = roi.copy()
                blended[non_bg] = cv2.addWeighted(
                    roi, 1 - mask_alpha, overlay, mask_alpha, 0
                )[non_bg]
                vis[y1:y2, x1:x2] = blended

    return vis


# ═══════════════════════════════════════════════════════════════════════════
# FacePipeline
# ═══════════════════════════════════════════════════════════════════════════
class FacePipeline:
    """End-to-end face detection + segmentation pipeline.

    Args:
        detector: A ``FaceDetector`` instance.
        segmentor: A ``FaceSegmentor`` instance.
        padding: Fractional padding added around each detected bbox
            before cropping for segmentation (e.g. ``0.1`` adds 10 %
            on every side).

    Example::

        pipeline = FacePipeline(detector, segmentor)
        result = pipeline.process("photo.jpg")
        for face in result['faces']:
            print(face['bbox'], face['score'])
    """

    def __init__(
        self,
        detector: FaceDetector,
        segmentor: FaceSegmentor,
        padding: float = 0.1,
    ):
        self.detector = detector
        self.segmentor = segmentor
        self.padding = padding
        logger.info("FacePipeline initialised.")

    # ------------------------------------------------------------------
    #  Private helpers
    # ------------------------------------------------------------------

    def _safe_crop(
        self, image: np.ndarray, bbox: List[float]
    ) -> tuple:
        """Crop a padded region from *image* clamped to valid bounds.

        Args:
            image: HWC numpy array.
            bbox: ``[x1, y1, x2, y2]`` in pixel coordinates.

        Returns:
            Tuple of ``(cropped_array, (x1, y1, x2, y2))`` where the
            coordinates are the *clamped* integers actually used.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1
        x1 = x1 - bw * self.padding
        y1 = y1 - bh * self.padding
        x2 = x2 + bw * self.padding
        y2 = y2 + bh * self.padding

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8), (x1, y1, x2, y2)

        return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    @staticmethod
    def _to_numpy_bgr(
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> np.ndarray:
        """Convert heterogeneous input to a BGR numpy array.

        Args:
            image: File path, numpy array, or PIL Image.

        Returns:
            BGR ``numpy.ndarray`` of shape ``(H, W, 3)``.
        """
        if isinstance(image, (str, Path)):
            img = np.array(Image.open(str(image)).convert("RGB"))
            return img[:, :, ::-1].copy()
        if isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"))
            return img[:, :, ::-1].copy()
        if isinstance(image, np.ndarray):
            return image.copy()
        raise TypeError(f"Unsupported image type: {type(image)}")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def process(
        self, image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict:
        """Run the full detect → segment pipeline on a single image.

        Args:
            image: File path, BGR numpy array, or PIL Image.

        Returns:
            Dict with key ``'faces'`` — a list of dicts each containing:

                - ``bbox``: ``[x1, y1, x2, y2]`` (pixel coords)
                - ``score``: ``float``
                - ``mask``: ``np.ndarray (H_crop, W_crop)`` with values
                  ``0 – 18``
                - ``class_names``: list of str
        """
        # 1. Detect faces
        det_result = self.detector.predict(image)
        boxes = det_result["boxes"]
        scores = det_result["scores"]

        if len(boxes) == 0:
            return {"faces": []}

        # 2. Prepare the full image as BGR numpy for cropping
        bgr = self._to_numpy_bgr(image)

        # 3. For each face: crop → segment
        faces: List[Dict] = []
        for bbox, score in zip(boxes, scores):
            crop, clamped_bbox = self._safe_crop(bgr, bbox)
            seg_result = self.segmentor.predict(crop)

            faces.append({
                "bbox": list(clamped_bbox),
                "score": score,
                "mask": seg_result["mask"],
                "class_names": seg_result["class_names"],
            })

        logger.debug("Processed image: %d face(s) found.", len(faces))
        return {"faces": faces}

    def process_batch(
        self, images: List[Union[str, Path, np.ndarray, Image.Image]]
    ) -> List[Dict]:
        """Run the pipeline on multiple images.

        Args:
            images: Iterable of images.

        Returns:
            List of result dicts, one per image.
        """
        return [self.process(img) for img in images]

    def visualize(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        result: Optional[Dict] = None,
        **draw_kwargs,
    ) -> np.ndarray:
        """Convenience wrapper: process (if needed) and draw results.

        Args:
            image: Input image.
            result: Pre-computed result dict.  If ``None`` the pipeline
                runs ``process`` first.
            **draw_kwargs: Forwarded to ``draw_results``.

        Returns:
            Annotated BGR numpy array.
        """
        bgr = self._to_numpy_bgr(image)
        if result is None:
            result = self.process(image)
        return draw_results(bgr, result["faces"], **draw_kwargs)


# Backward-compatible alias used by the existing __init__.py
InferencePipeline = FacePipeline


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════
def create_pipeline(
    det_checkpoint: Optional[str] = None,
    seg_checkpoint: Optional[str] = None,
    device: str = "cuda",
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.4,
) -> FacePipeline:
    """Build a ready-to-use ``FacePipeline``.

    Args:
        det_checkpoint: Path to detection model checkpoint.
        seg_checkpoint: Path to segmentation model checkpoint.
        device: ``'cuda'`` or ``'cpu'``.
        conf_threshold: Detection confidence threshold.
        nms_threshold: NMS IoU threshold.

    Returns:
        Configured ``FacePipeline`` instance.
    """
    detector = FaceDetector(
        checkpoint_path=det_checkpoint,
        device=device,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )
    segmentor = FaceSegmentor(
        checkpoint_path=seg_checkpoint,
        device=device,
    )
    return FacePipeline(detector, segmentor)


# ═══════════════════════════════════════════════════════════════════════════
# Quick self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("FacePipeline — smoke test")
    print("=" * 50)

    pipeline = create_pipeline(device="cpu")

    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = pipeline.process(dummy_img)
    print(f"  Faces found: {len(result['faces'])}")
    for i, face in enumerate(result["faces"]):
        print(
            f"    Face {i}: bbox={face['bbox']}, "
            f"score={face['score']:.4f}, "
            f"mask_shape={face['mask'].shape}"
        )

    # Test visualize
    vis = pipeline.visualize(dummy_img, result)
    print(f"  Visualisation shape: {vis.shape}")

    print("Smoke test passed ✓")
