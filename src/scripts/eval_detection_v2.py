"""Standalone eval script for RetinaFace detector with safe defaults.

Usage::

    python -m src.scripts.eval_detection_v2 \
        --config src/configs/retinaface.yaml \
        --checkpoint runs/retinaface/retinaface_best.pth \
        --output runs/retinaface/eval_results.json

This is a *patched* version of ``src/scripts/eval_detection.py`` that:

- lowers the default ``conf_threshold`` to 0.02 (model trained for < 20 epochs
  rarely pushes face scores above 0.7);
- prints raw-score statistics so you can see what the model actually outputs;
- lets you sweep multiple thresholds in one run for diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.detection.anchors import AnchorConfig, generate_anchors
from src.detection.inference import RetinaFaceDetector
from src.utils.io import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RetinaFace (v2 patched).")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--conf-threshold", type=float, default=0.02,
                        help="Confidence threshold (was 0.05/0.7, now 0.02 for undertrained models).")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--raw-stats", action="store_true",
                        help="Print raw score statistics before threshold filtering.")
    parser.add_argument("--sweep", action="store_true",
                        help="Run multiple conf_thresholds and report mAP for each.")
    return parser.parse_args()


def _has_cuda() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def raw_score_stats(detector: RetinaFaceDetector, csv_path: Path, images_dir: Path,
                    max_images: int | None = None) -> dict[str, float]:
    """Run inference on all val images and collect raw-score statistics."""
    import csv

    # Read unique image IDs.
    image_ids: list[str] = []
    seen: set[str] = set()
    with open(csv_path, "r", encoding="utf-8") as fh:
        first = True
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                if line.lower().startswith("image_id"):
                    continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            if parts[0] not in seen:
                seen.add(parts[0])
                image_ids.append(parts[0])

    if max_images is not None:
        image_ids = image_ids[:max_images]

    all_scores: list[float] = []
    n_processed = 0
    for img_id in image_ids:
        path = images_dir / img_id
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        # Use raw logits via a temporary predictor.
        with torch.no_grad():
            tensor, meta = detector._preprocess(img)
            x = tensor.unsqueeze(0).to(detector.device)
            from src.detection.retinaface import flatten_predictions
            outputs = flatten_predictions(detector.model(x))
            cls = outputs["cls_logits"][0].cpu().numpy()
            # Apply sigmoid and take face logit.
            face_logits = cls[:, 1] if cls.ndim == 2 else cls
            face_scores = 1.0 / (1.0 + np.exp(-face_logits))
            all_scores.extend(face_scores.tolist())
        n_processed += 1

    arr = np.asarray(all_scores, dtype=np.float32)
    return {
        "n_images": int(n_processed),
        "n_anchors": int(arr.size),
        "score_min": float(arr.min()) if arr.size else 0.0,
        "score_max": float(arr.max()) if arr.size else 0.0,
        "score_mean": float(arr.mean()) if arr.size else 0.0,
        "score_median": float(np.median(arr)) if arr.size else 0.0,
        "score_p90": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "score_p99": float(np.percentile(arr, 99)) if arr.size else 0.0,
        "frac_above_0.02": float((arr > 0.02).mean()) if arr.size else 0.0,
        "frac_above_0.05": float((arr > 0.05).mean()) if arr.size else 0.0,
        "frac_above_0.10": float((arr > 0.10).mean()) if arr.size else 0.0,
        "frac_above_0.30": float((arr > 0.30).mean()) if arr.size else 0.0,
        "frac_above_0.50": float((arr > 0.50).mean()) if arr.size else 0.0,
        "frac_above_0.70": float((arr > 0.70).mean()) if arr.size else 0.0,
    }


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
    device = "cuda" if _has_cuda() else "cpu"
    if ckpt.exists():
        detector = RetinaFaceDetector(
            weights=ckpt, device=device, anchor_cfg=anchor_cfg, pretrained=False,
        )
        print(f"[INFO] Loaded checkpoint: {ckpt}")
    else:
        print(f"[WARN] Checkpoint not found: {ckpt}, using random weights")
        detector = RetinaFaceDetector(
            weights=None, device=device, anchor_cfg=anchor_cfg, pretrained=False,
        )

    csv_path = data_cfg.get("test_csv") or data_cfg.get("val_csv")
    images_dir = Path(csv_path).parent / "images"
    print(f"[INFO] Evaluating on: {csv_path}")
    print(f"[INFO] conf_threshold={args.conf_threshold}, nms_iou={args.nms_iou}")
    print(f"[INFO] max_images={args.max_images}")

    # Optional raw stats dump.
    if args.raw_stats:
        print("\n=== Raw score statistics ===")
        stats = raw_score_stats(detector, Path(csv_path), images_dir, args.max_images)
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Optional threshold sweep.
    if args.sweep:
        from src.detection.eval import evaluate_map
        print("\n=== Threshold sweep ===")
        thresholds = [0.005, 0.01, 0.02, 0.05, 0.10, 0.30, 0.50, 0.70]
        sweep_results = []
        for thr in thresholds:
            metrics = evaluate_map(
                detector=detector,
                csv_path=csv_path,
                images_dir=images_dir,
                conf_threshold=thr,
                nms_iou=args.nms_iou,
                max_images=args.max_images,
            )
            print(f"  thr={thr:.3f}  mAP={metrics['mAP']:.4f}  recall={metrics['recall']:.4f}  "
                  f"n_pred={metrics['n_pred']}  n_gt={metrics['n_gt']}")
            sweep_results.append({"threshold": thr, **metrics})
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                json.dump({"sweep": sweep_results}, fh, indent=2, default=float)
            print(f"[INFO] Sweep saved: {args.output}")
        return

    # Standard eval.
    from src.detection.eval import evaluate_map
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
                {"metrics": metrics, "config": {
                    "conf_threshold": args.conf_threshold,
                    "nms_iou": args.nms_iou,
                }},
                fh, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x,
            )
        print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()
