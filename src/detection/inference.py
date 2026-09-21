"""Inference helper for the RetinaFace detector.

Wraps a :class:`RetinaFace` model with:

- preprocessing (resize + pad + ImageNet normalisation),
- forward pass,
- post-processing (decode + NMS + clip),
- optional landmark decoding.

Usage::

    detector = RetinaFaceDetector("models/retinaface_best.pth", device="cpu")
    out = detector.predict(image_bgr, conf_threshold=0.7, nms_iou=0.5)
    # out["boxes"], out["scores"], out["landmarks"]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.detection.anchors import AnchorConfig, decode_predictions, generate_anchors
from src.detection.retinaface import RetinaFace, RetinaFaceConfig, flatten_predictions


@dataclass
class DetectorPrediction:
    boxes: np.ndarray  # [N, 4] xyxy in *original* image coords
    scores: np.ndarray  # [N]
    landmarks: Optional[np.ndarray]  # [N, 10] or None
    orig_size: tuple[int, int]


class RetinaFaceDetector:
    """Stateful detector that holds the model + preprocessing config."""

    NORMALIZE_MEAN = (0.485, 0.456, 0.406)
    NORMALIZE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str = "cpu",
        anchor_cfg: AnchorConfig | None = None,
        pretrained: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.anchor_cfg = anchor_cfg or AnchorConfig()
        self.model = RetinaFace(RetinaFaceConfig(pretrained_backbone=pretrained))
        if weights is not None and Path(weights).exists():
            state = torch.load(str(weights), map_location=self.device, weights_only=False)
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "model" in state:
                    state = state["model"]
                elif "state_dict" in state:
                    state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
        self.model.eval().to(self.device)

        self._anchors = generate_anchors(self.anchor_cfg)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float = 0.7,
        nms_iou: float = 0.5,
        return_landmarks: bool = True,
    ) -> DetectorPrediction:
        """Run the detector on a BGR image."""
        h0, w0 = image_bgr.shape[:2]
        tensor, meta = self._preprocess(image_bgr)
        x = tensor.unsqueeze(0).to(self.device)
        outputs = flatten_predictions(self.model(x))

        cls = outputs["cls_logits"][0].cpu().numpy()
        box = outputs["box_deltas"][0].cpu().numpy()
        lmk = outputs["lmk_deltas"][0].cpu().numpy() if return_landmarks else None

        decoded = decode_predictions(
            anchors=self._anchors,
            cls_logits=cls,
            box_deltas=box,
            lmk_deltas=lmk,
            image_size=self.anchor_cfg.image_size,
            conf_threshold=conf_threshold,
            nms_iou=nms_iou,
        )

        boxes = self._rescale_to_original(decoded["boxes"], meta, (h0, w0))
        landmarks = None
        if decoded["landmarks"] is not None:
            landmarks = self._rescale_landmarks_to_original(decoded["landmarks"], meta, (h0, w0))

        return DetectorPrediction(
            boxes=boxes,
            scores=decoded["scores"].astype(np.float32),
            landmarks=landmarks,
            orig_size=(h0, w0),
        )

    # ------------------------------------------------------------------
    def _preprocess(self, image_bgr: np.ndarray) -> tuple[torch.Tensor, dict]:
        h0, w0 = image_bgr.shape[:2]
        target = self.anchor_cfg.image_size
        ratio = min(target / h0, target / w0)
        new_h, new_w = int(round(h0 * ratio)), int(round(w0 * ratio))
        pad_w = target - new_w
        pad_h = target - new_h

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray(self.NORMALIZE_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(self.NORMALIZE_STD, dtype=np.float32).reshape(3, 1, 1)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        tensor = (tensor - torch.from_numpy(mean)) / torch.from_numpy(std)
        meta = {"ratio": ratio, "pad_w": pad_w, "pad_h": pad_h}
        return tensor, meta

    def _rescale_to_original(
        self, boxes: np.ndarray, meta: dict, orig_size: tuple[int, int]
    ) -> np.ndarray:
        if boxes.size == 0:
            return boxes.reshape(0, 4).astype(np.float32)
        ratio = meta["ratio"]
        out = boxes.copy() / ratio
        h0, w0 = orig_size
        out[:, 0] = np.clip(out[:, 0], 0, w0 - 1)
        out[:, 1] = np.clip(out[:, 1], 0, h0 - 1)
        out[:, 2] = np.clip(out[:, 2], 0, w0 - 1)
        out[:, 3] = np.clip(out[:, 3], 0, h0 - 1)
        return out.astype(np.float32)

    def _rescale_landmarks_to_original(
        self, landmarks: np.ndarray, meta: dict, orig_size: tuple[int, int]
    ) -> np.ndarray:
        if landmarks.size == 0:
            return landmarks.reshape(0, 10).astype(np.float32)
        ratio = meta["ratio"]
        out = landmarks.copy() / ratio
        h0, w0 = orig_size
        out[:, 0::2] = np.clip(out[:, 0::2], 0, w0 - 1)
        out[:, 1::2] = np.clip(out[:, 1::2], 0, h0 - 1)
        return out.astype(np.float32)


def load_retinaface(weights: str | Path, device: str = "cpu") -> RetinaFaceDetector:
    """Functional shorthand matching the deployment example in
    ``references/deployment.md``."""
    return RetinaFaceDetector(weights=weights, device=device)
