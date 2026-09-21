"""Segmentation evaluation — IoU, Dice, pixel accuracy (binary)."""

from __future__ import annotations

import numpy as np
import torch


def compute_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> dict[str, float]:
    """Compute IoU, Dice and pixel accuracy for binary masks.

    Both ``pred`` and ``target`` are ``[B, 1, H, W]`` float tensors with
    values in ``{0, 1}`` (or probabilities — they are thresholded at 0.5).
    """
    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    pred_f = pred_bin.flatten(1)
    target_f = target_bin.flatten(1)

    intersection = (pred_f * target_f).sum(dim=1)
    pred_sum = pred_f.sum(dim=1)
    target_sum = target_f.sum(dim=1)
    union = pred_sum + target_sum - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
    pixel_acc = (pred_bin == target_bin).float().mean(dim=(1, 2, 3))

    return {
        "iou": float(iou.mean()),
        "dice": float(dice.mean()),
        "pixel_acc": float(pixel_acc.mean()),
    }


def numpy_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> dict[str, float]:
    """Same metrics but for NumPy masks (uint8 0/255)."""
    pred_bin = pred_mask > 127
    gt_bin = gt_mask > 127
    intersection = float((pred_bin & gt_bin).sum())
    pred_sum = float(pred_bin.sum())
    gt_sum = float(gt_bin.sum())
    union = pred_sum + gt_sum - intersection
    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred_sum + gt_sum + eps)
    pixel_acc = float((pred_bin == gt_bin).mean())
    return {"iou": iou, "dice": dice, "pixel_acc": pixel_acc}
