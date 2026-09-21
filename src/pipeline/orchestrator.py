"""End-to-end pipeline orchestrator.

This module is the public entrypoint used by the CLI and the unit tests. It
composes detection → segmentation → visualization and handles all the
graceful-degradation paths defined in `progress_status.md` §7.2.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from src.detection.inference import RetinaFaceDetector
from src.pipeline.stages import run_detection, run_segmentation
from src.pipeline.visualizer import render_empty_overlay, render_overlay
from src.segmentation.inference import UNetSegmentor
from src.utils.io import ensure_dir, load_image_bgr, save_image_bgr


logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a single-image pipeline run."""

    image_path: Optional[str]
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    masks: list[np.ndarray] = field(default_factory=list)
    overlay: Optional[np.ndarray] = None
    failure_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "boxes": self.boxes.tolist(),
            "scores": self.scores.tolist(),
            "n_masks": len(self.masks),
            "failure_reason": self.failure_reason,
        }


class FaceSegmentationPipeline:
    """Compose RetinaFace + U-Net into a single inference call."""

    def __init__(
        self,
        detector_weights: str | Path | None = None,
        segmentor_weights: str | Path | None = None,
        device: str = "cpu",
        conf_threshold: float = 0.7,
        nms_iou: float = 0.5,
        seg_margin: float = 0.1,
        max_faces: int = 50,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.seg_margin = seg_margin
        self.max_faces = max_faces

        self.detector = RetinaFaceDetector(
            weights=detector_weights, device=device, pretrained=False
        )
        self.segmentor = UNetSegmentor(
            weights=segmentor_weights, device=device
        )

    # ------------------------------------------------------------------
    def run(
        self,
        image: str | Path | np.ndarray,
        save_overlay_to: str | Path | None = None,
    ) -> PipelineResult:
        """Run the pipeline on a single image."""
        if isinstance(image, (str, Path)):
            image_path = str(image)
            try:
                img_bgr = load_image_bgr(image_path)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to load image %s", image_path)
                return PipelineResult(image_path=image_path, failure_reason=f"load_error: {exc}")
        else:
            image_path = None
            img_bgr = image

        try:
            det = run_detection(
                img_bgr,
                self.detector,
                conf_threshold=self.conf_threshold,
                nms_iou=self.nms_iou,
            )
            seg = run_segmentation(det, self.segmentor, margin=self.seg_margin, max_faces=self.max_faces)

            if det.boxes.size == 0:
                overlay = render_empty_overlay(img_bgr)
                if save_overlay_to is not None:
                    save_image_bgr(save_overlay_to, overlay)
                return PipelineResult(
                    image_path=image_path,
                    boxes=det.boxes,
                    scores=det.scores,
                    masks=[],
                    overlay=overlay,
                    failure_reason="no_face_detected",
                )

            overlay = render_overlay(
                img_bgr,
                det.boxes,
                seg.masks,
                scores=det.scores,
            )

            if save_overlay_to is not None:
                save_image_bgr(save_overlay_to, overlay)

            return PipelineResult(
                image_path=image_path,
                boxes=det.boxes,
                scores=det.scores,
                masks=seg.masks,
                overlay=overlay,
                failure_reason=None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline failed on %s", image_path)
            return PipelineResult(image_path=image_path, failure_reason=f"runtime_error: {exc}")

    # ------------------------------------------------------------------
    def run_batch(
        self,
        image_paths: Sequence[str | Path],
        output_dir: str | Path,
    ) -> list[PipelineResult]:
        """Run the pipeline on many images and dump overlays + a JSON summary."""
        output_dir = ensure_dir(output_dir)
        results: list[PipelineResult] = []

        for path in image_paths:
            out_path = output_dir / "vis" / (Path(str(path)).stem + ".png")
            res = self.run(path, save_overlay_to=out_path)
            results.append(res)

        # Save a JSON summary alongside the overlays.
        summary_path = output_dir / "results.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in results], fh, indent=2)

        return results


def build_pipeline_from_args(args) -> FaceSegmentationPipeline:
    """Tiny adapter to build a pipeline from the CLI Namespace."""
    return FaceSegmentationPipeline(
        detector_weights=args.detector_weights,
        segmentor_weights=args.segmentor_weights,
        device=args.device,
        conf_threshold=args.conf_threshold,
        nms_iou=args.nms_iou,
    )
