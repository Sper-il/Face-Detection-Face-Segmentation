"""Standalone eval script for RetinaFace detector.

Usage::

    python -m src.scripts.eval_detection \
        --config src/configs/retinaface.yaml \
        --checkpoint runs/retinaface/retinaface_best.pth \
        --output runs/retinaface/eval_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.detection.anchors import AnchorConfig
from src.detection.eval import evaluate_map
from src.detection.inference import RetinaFaceDetector
from src.utils.io import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RetinaFace.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    data_cfg = cfg.get("data", {})
    image_size = int(data_cfg.get("image_size", 640))

    anchor_cfg = AnchorConfig(
        image_size=image_size,
        strides=tuple(cfg.get("model", {}).get("strides", [8, 16, 32])),
    )

    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        detector = RetinaFaceDetector(
            weights=ckpt,
            device="cuda" if _has_cuda() else "cpu",
            anchor_cfg=anchor_cfg,
            pretrained=False,
        )
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print(f"WARNING: Checkpoint not found: {ckpt}, using random weights")
        detector = RetinaFaceDetector(
            weights=None,
            device="cpu",
            anchor_cfg=anchor_cfg,
            pretrained=False,
        )

    csv_path = data_cfg.get("test_csv") or data_cfg.get("val_csv")
    images_dir = Path(csv_path).parent / "images"

    print(f"Evaluating on: {csv_path}")
    metrics = evaluate_map(
        detector=detector,
        csv_path=csv_path,
        images_dir=images_dir,
        conf_threshold=args.conf_threshold,
        nms_iou=args.nms_iou,
        max_images=args.max_images,
    )
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if args.output:
        out_path = Path(args.output)
        ensure_dir(out_path.parent)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(
                metrics,
                fh,
                indent=2,
                default=lambda x: float(x) if isinstance(x, np.floating) else x,
            )
        print(f"Saved: {out_path}")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
