"""End-to-end pipeline orchestrator.

The main entrypoint is :class:`FaceSegmentationPipeline` in
``src.pipeline.orchestrator``. The CLI lives in ``src.pipeline.run``.
"""

from __future__ import annotations

from src.pipeline.orchestrator import FaceSegmentationPipeline, PipelineResult
from src.pipeline.stages import (
    DetectionStageResult,
    SegmentationStageResult,
    run_detection,
    run_segmentation,
)

__all__ = [
    "FaceSegmentationPipeline",
    "PipelineResult",
    "DetectionStageResult",
    "SegmentationStageResult",
    "run_detection",
    "run_segmentation",
]
