"""
Face Detection Module
Based on DSFD (Tencent)
Reference: https://github.com/Tencent/FaceDetection-DSFD
"""

from .model import FaceDetector, build_face_detector
from .losses import DetectionLoss, FocalLoss, SmoothL1Loss, GIoULoss

__all__ = [
    'FaceDetector',
    'build_face_detector',
    'DetectionLoss',
    'FocalLoss',
    'SmoothL1Loss',
    'GIoULoss',
]
