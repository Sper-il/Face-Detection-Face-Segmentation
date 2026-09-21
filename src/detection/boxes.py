"""Re-export of detection-side box utilities (PyTorch-friendly).

This module exists so that the dataset can import a torch-compatible encoder
without forcing the rest of the codebase to depend on it.
"""

from __future__ import annotations

import torch

from src.utils.box_ops import decode_boxes as _decode_boxes_np
from src.utils.box_ops import encode_boxes as _encode_boxes_np


def encode_boxes_torch(
    anchors_xyxy: torch.Tensor,
    gt_xyxy: torch.Tensor,
    variance: tuple[float, float] = (0.1, 0.2),
) -> torch.Tensor:
    """Torch version of ``encode_boxes`` — preserves autograd graph."""
    if anchors_xyxy.numel() == 0 or gt_xyxy.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32, device=anchors_xyxy.device)
    aw = (anchors_xyxy[:, 2] - anchors_xyxy[:, 0]).clamp(min=1e-6)
    ah = (anchors_xyxy[:, 3] - anchors_xyxy[:, 1]).clamp(min=1e-6)
    ax = (anchors_xyxy[:, 0] + anchors_xyxy[:, 2]) * 0.5
    ay = (anchors_xyxy[:, 1] + anchors_xyxy[:, 3]) * 0.5
    gw = gt_xyxy[:, 2] - gt_xyxy[:, 0]
    gh = gt_xyxy[:, 3] - gt_xyxy[:, 1]
    gx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) * 0.5
    gy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) * 0.5

    v0, v1 = variance
    tx = (gx - ax) / aw / v0
    ty = (gy - ay) / ah / v0
    tw = torch.log(gw / aw.clamp(min=1e-6) + 1e-7) / v1
    th = torch.log(gh / ah.clamp(min=1e-6) + 1e-7) / v1
    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes_torch(
    anchors_xyxy: torch.Tensor,
    deltas: torch.Tensor,
    variance: tuple[float, float] = (0.1, 0.2),
) -> torch.Tensor:
    """Torch version of ``decode_boxes``."""
    if anchors_xyxy.numel() == 0 or deltas.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32, device=anchors_xyxy.device)
    aw = anchors_xyxy[:, 2] - anchors_xyxy[:, 0]
    ah = anchors_xyxy[:, 3] - anchors_xyxy[:, 1]
    ax = (anchors_xyxy[:, 0] + anchors_xyxy[:, 2]) * 0.5
    ay = (anchors_xyxy[:, 1] + anchors_xyxy[:, 3]) * 0.5

    v0, v1 = variance
    px = deltas[:, 0] * v0 * aw + ax
    py = deltas[:, 1] * v0 * ah + ay
    pw = torch.exp(deltas[:, 2] * v1) * aw
    ph = torch.exp(deltas[:, 3] * v1) * ah

    return torch.stack(
        [px - pw * 0.5, py - ph * 0.5, px + pw * 0.5, py + ph * 0.5],
        dim=1,
    )


__all__ = ["encode_boxes_torch", "decode_boxes_torch", "_encode_boxes_np", "_decode_boxes_np"]
