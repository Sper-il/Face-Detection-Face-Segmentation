"""
Inference Module

Public API for face detection, segmentation, and the combined pipeline.
"""

from .detector import FaceDetector
from .segmentor import FaceSegmentor
from .pipeline import FacePipeline, InferencePipeline, create_pipeline
from .batch_inference import BatchProcessor

__all__ = [
    "FaceDetector",
    "FaceSegmentor",
    "FacePipeline",
    "InferencePipeline",   # backward-compatible alias
    "create_pipeline",
    "BatchProcessor",
]
