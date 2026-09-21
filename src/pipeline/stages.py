"""Stage implementations for the 2-stage pipeline.

Each function is independently testable and returns the inputs it was given
plus a ``stage_result`` key (e.g. ``"boxes"``, ``"masks"``). The pipeline's
own :class:`FaceSegmentationPipeline` composes these stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from src.detection.inference import RetinaFaceDetector
from src.segmentation.inference import UNetSegmentor


@dataclass
class DetectionStageResult:
    image: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    landmarks: Optional[np.ndarray]


@dataclass
class SegmentationStageResult:
    image: np.ndarray
    boxes: np.ndarray
    scores: np.ndarray
    masks: list[np.ndarray]
    landmark_used: bool


def run_detection(
    image_bgr: np.ndarray,
    detector: RetinaFaceDetector,
    conf_threshold: float = 0.7,
    nms_iou: float = 0.5,
) -> DetectionStageResult:
    """Stage 1 — detect faces."""
    pred = detector.predict(
        image_bgr, conf_threshold=conf_threshold, nms_iou=nms_iou, return_landmarks=True
    )
    return DetectionStageResult(
        image=image_bgr,
        boxes=pred.boxes,
        scores=pred.scores,
        landmarks=pred.landmarks,
    )


def run_segmentation(
    det_result: DetectionStageResult,
    segmentor: UNetSegmentor,
    margin: float = 0.1,
    use_landmarks: bool = True,
    max_faces: int = 50,
) -> SegmentationStageResult:
    """Stage 2 — segment each detected face.

    Handles the "no face detected" failure mode by returning an empty mask
    list — the caller can still produce a useful overlay of the original image.

    The pipeline caps the number of faces passed to the segmentor to avoid
    catastrophic runtime when the detector is uninitialised (random weights
    → hundreds of false positives). The cap is applied *after sorting by
    detector score* so the most confident detections survive.
    """
    if det_result.boxes.size == 0:
        return SegmentationStageResult(
            image=det_result.image,
            boxes=det_result.boxes,
            scores=det_result.scores,
            masks=[],
            landmark_used=use_landmarks and det_result.landmarks is not None,
        )

    # Keep only the top-scoring boxes so the worst-case runtime is bounded.
    if det_result.boxes.shape[0] > max_faces:
        order = np.argsort(-det_result.scores)[:max_faces]
        boxes = det_result.boxes[order]
        scores = det_result.scores[order]
    else:
        boxes = det_result.boxes
        scores = det_result.scores

    try:
        preds = segmentor.segment_faces(det_result.image, boxes, margin=margin)
    except Exception as exc:  # noqa: BLE001 — we want to keep prior successes
        # Log and continue with empty masks.
        print(f"[pipeline] segmentation failed on one image: {exc}")
        preds = []

    # Skip masks that are empty (degenerate).
    masks: list[np.ndarray] = []
    for p in preds:
        if p.mask.sum() == 0:
            continue
        masks.append(p.mask)

    return SegmentationStageResult(
        image=det_result.image,
        boxes=boxes,
        scores=scores,
        masks=masks,
        landmark_used=use_landmarks and det_result.landmarks is not None,
    )


def collect_stage_outputs(stage: SegmentationStageResult) -> dict[str, object]:
    """Return a plain-dict snapshot for serialization / logging."""
    return {
        "boxes": stage.boxes.tolist() if stage.boxes.size else [],
        "scores": stage.scores.tolist() if stage.scores.size else [],
        "n_masks": len(stage.masks),
    }


def iter_image_paths(path: str | Iterable[str]) -> Iterable[str]:
    """Resolve a CLI ``--image`` argument into a list of file paths.

    Accepts a single path, a directory (recursively searched for ``.png`` /
    ``.jpg`` / ``.jpeg``), or a comma-separated list of paths.
    """
    import os
    from pathlib import Path

    if isinstance(path, str) and "," in path:
        return [p.strip() for p in path.split(",") if p.strip()]

    p = Path(path)
    if p.is_dir():
        return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if p.is_file():
        return [str(p)]
    raise FileNotFoundError(f"Path does not exist or is empty: {path}")
