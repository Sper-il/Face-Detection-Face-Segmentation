"""Shared utilities (I/O, visualization, NMS, metrics)."""

from __future__ import annotations

from src.utils.box_ops import (
    bbox_iou,
    clip_boxes,
    decode_boxes,
    encode_boxes,
    xyxy_to_xywh,
    xywh_to_xyxy,
)
from src.utils.io import (
    ensure_dir,
    load_image_bgr,
    load_yaml,
    save_image_bgr,
    set_seed,
)
from src.utils.mask_ops import (
    mask_in_bbox,
    morph_close,
    mask_morphology,
    mask_open,
    paste_mask_into_image,
)
from src.utils.nms import batched_nms, nms
from src.utils.visualizer import draw_boxes, draw_mask_overlay, overlay

__all__ = [
    # box_ops
    "bbox_iou",
    "clip_boxes",
    "decode_boxes",
    "encode_boxes",
    "xyxy_to_xywh",
    "xywh_to_xyxy",
    # io
    "ensure_dir",
    "load_image_bgr",
    "load_yaml",
    "save_image_bgr",
    "set_seed",
    # mask_ops
    "mask_in_bbox",
    "morph_close",
    "mask_morphology",
    "mask_open",
    "paste_mask_into_image",
    # nms
    "batched_nms",
    "nms",
    # visualizer
    "draw_boxes",
    "draw_mask_overlay",
    "overlay",
]
