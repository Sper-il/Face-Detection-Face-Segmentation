"""Inference helper for the U-Net face-mask segmentor.

The model outputs logits with 2 channels (background, face).
This module wraps preprocessing + inference + post-processing.

Usage::

    segmentor = UNetSegmentor("models/unet_final.pth", device="cpu")
    pred = segmentor.predict(image_bgr, conf_threshold=0.5)
    # pred.mask is a uint8 mask in {0, 255}, pred.score is mean confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.segmentation.unet_model import UNet, UNetConfig


@dataclass
class SegmentationPrediction:
    """A prediction for a single image or crop."""

    mask: np.ndarray  # [H, W] uint8 in {0, 255}
    prob: np.ndarray  # [H, W] float32 face probability
    score: float  # mean confidence of face class in the predicted mask


class UNetSegmentor:
    """Segmentor for face mask prediction using U-Net."""

    NORMALIZE_MEAN = (0.485, 0.456, 0.406)
    NORMALIZE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str = "cpu",
        image_size: int = 256,
        threshold: float = 0.5,
        morph_opening: int = 3,
        morph_closing: int = 5,
    ) -> None:
        self.device = torch.device(device)
        self.image_size = image_size
        self.threshold = threshold
        self.morph_opening = morph_opening
        self.morph_closing = morph_closing

        self.model = UNet(UNetConfig())
        if weights is not None and Path(weights).exists():
            state = torch.load(str(weights), map_location=self.device, weights_only=False)

            # Handle different checkpoint formats
            if isinstance(state, dict):
                if "model_state" in state:
                    state = state["model_state"]
                elif "model" in state:
                    state = state["model"]
                elif "state_dict" in state:
                    state = state["state_dict"]

            self.model.load_state_dict(state, strict=False)

        self.model.eval().to(self.device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> SegmentationPrediction:
        """Run segmentation on a BGR image, return binary mask at original size.

        This matches the user's inference logic in the training script:
        ``logits.argmax(dim=1)`` for the mask, softmax for the confidence.
        """
        h0, w0 = image_bgr.shape[:2]
        tensor = self._preprocess(image_bgr, target_size=self.image_size).unsqueeze(0).to(self.device)

        logits = self.model(tensor)  # [1, 2, H, W]
        probs = torch.softmax(logits, dim=1)
        face_prob = probs[0, 1].cpu().numpy()  # [H, W]

        pred_mask_small = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        # Resize both back to original image size
        pred_mask = cv2.resize(pred_mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
        face_prob_full = cv2.resize(face_prob, (w0, h0), interpolation=cv2.INTER_LINEAR)

        # Convert {0, 1} to {0, 255} and apply morphology
        binary = (pred_mask * 255).astype(np.uint8)
        binary = self._morph(binary)

        score = float(face_prob_full[pred_mask == 1].mean()) if (pred_mask == 1).any() else 0.0

        return SegmentationPrediction(mask=binary, prob=face_prob_full, score=score)

    @torch.no_grad()
    def predict_crop(self, crop_bgr: np.ndarray) -> SegmentationPrediction:
        """Same as :meth:`predict` but explicitly named for crop usage."""
        return self.predict(crop_bgr)

    @torch.no_grad()
    def predict_full(self, image_bgr: np.ndarray) -> SegmentationPrediction:
        """Alias for full-image inference."""
        return self.predict(image_bgr)

    # ------------------------------------------------------------------
    def segment_faces(
        self,
        image_bgr: np.ndarray,
        bboxes: np.ndarray,
        margin: float = 0.1,
    ) -> list[SegmentationPrediction]:
        """Run segmentation on each bbox with a small margin.

        Returns one :class:`SegmentationPrediction` per bbox (skipping
        degenerate/empty ones).
        """
        h, w = image_bgr.shape[:2]
        results: list[SegmentationPrediction] = []

        for box in bboxes:
            x1, y1, x2, y2 = box.astype(float).tolist()
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            mx1 = max(0, int(round(x1 - bw * margin)))
            my1 = max(0, int(round(y1 - bh * margin)))
            mx2 = min(w, int(round(x2 + bw * margin)))
            my2 = min(h, int(round(y2 + bh * margin)))
            if mx2 <= mx1 or my2 <= my1:
                continue

            crop = image_bgr[my1:my2, mx1:mx2]
            pred = self.predict(crop)

            # Crop mask back to inner bbox (no margin)
            crop_h = my2 - my1
            crop_w = mx2 - mx1
            iw = int(round((x1 - mx1) / crop_w * pred.mask.shape[1]))
            ih = int(round((y1 - my1) / crop_h * pred.mask.shape[0]))
            jw = int(round((x2 - mx1) / crop_w * pred.mask.shape[1]))
            jh = int(round((y2 - my1) / crop_h * pred.mask.shape[0]))
            iw = max(0, iw); ih = max(0, ih)
            jw = min(pred.mask.shape[1], jw); jh = min(pred.mask.shape[0], jh)

            inner = np.zeros_like(pred.mask)
            inner[ih:jh, iw:jw] = pred.mask[ih:jh, iw:jw]
            pred = SegmentationPrediction(mask=inner, prob=pred.prob, score=pred.score)
            results.append(pred)

        return results

    # ------------------------------------------------------------------
    def _preprocess(self, image_bgr: np.ndarray, target_size: int) -> torch.Tensor:
        """Resize+pad, RGB, normalize (matches user's training script)."""
        h0, w0 = image_bgr.shape[:2]
        # Resize maintaining aspect ratio
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (target_size, target_size))

        img_norm = (img_resized / 255.0 - np.array(self.NORMALIZE_MEAN)) / np.array(self.NORMALIZE_STD)
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        return tensor

    def _morph(self, mask: np.ndarray) -> np.ndarray:
        if self.morph_opening > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_opening, self.morph_opening))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if self.morph_closing > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_closing, self.morph_closing))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask


def load_unet_seg(weights: str | Path, device: str = "cpu") -> UNetSegmentor:
    """Functional shorthand matching the user's training script."""
    return UNetSegmentor(weights=weights, device=device)
