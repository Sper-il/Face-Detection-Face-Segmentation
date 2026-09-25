"""Local debug script for the RetinaFace n_pred=0 issue.

Runs on CPU so you can verify what the trained model is actually doing
*before* paying for another Kaggle run.

Usage::

    python -m src.scripts.debug_detection_inference \
        --checkpoint models/retinaface_demo.pth \
        --config src/configs/retinaface.yaml \
        --num-images 5

It will:

1. Load the checkpoint + dataset (CPU).
2. Print raw face-logit / face-score statistics over ``num_images`` validation
   images (this tells us whether the model has actually learned *anything*).
3. Run ``detector.predict`` with ``conf_threshold=0.02`` and report how many
   predictions come out per image.
4. Suggest next steps based on the findings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.detection.anchors import AnchorConfig, generate_anchors
from src.detection.inference import RetinaFaceDetector
from src.detection.retinaface import flatten_predictions
from src.detection.eval import load_gt, group_by_image, EvalEntry, compute_map_recall
from src.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug RetinaFace inference (CPU-friendly).")
    p.add_argument("--checkpoint", type=str, default="models/retinaface_final.pth")
    p.add_argument("--config", type=str, default="src/configs/retinaface.yaml")
    p.add_argument("--num-images", type=int, default=5)
    p.add_argument("--conf-threshold", type=float, default=0.02)
    return p.parse_args()


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
    detector = RetinaFaceDetector(
        weights=ckpt if ckpt.exists() else None,
        device="cpu",
        anchor_cfg=anchor_cfg,
        pretrained=False,
    )
    print(f"[INFO] Checkpoint loaded: {ckpt.exists()} ({ckpt})")

    csv_path = data_cfg.get("val_csv")
    images_dir = Path(csv_path).parent / "images"
    print(f"[INFO] val_csv: {csv_path}")
    print(f"[INFO] images_dir: {images_dir}")

    # Get a few image IDs.
    gt_entries = load_gt(csv_path)
    gt_by_img = group_by_image(gt_entries)
    image_ids = sorted(gt_by_img.keys())[: args.num_images]
    print(f"[INFO] Inspecting {len(image_ids)} validation images")

    all_scores: list[float] = []
    n_preds_per_img: list[int] = []

    for img_id in image_ids:
        path = images_dir / img_id
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] cannot read {path}")
            continue

        # 1) Raw face-score stats.
        with torch.no_grad():
            tensor, meta = detector._preprocess(img)
            x = tensor.unsqueeze(0)
            outputs = flatten_predictions(detector.model(x))
            cls = outputs["cls_logits"][0].cpu().numpy()
            face_logits = cls[:, 1] if cls.ndim == 2 else cls.ravel()
            face_scores = 1.0 / (1.0 + np.exp(-face_logits))
            all_scores.extend(face_scores.tolist())
            score_max = float(face_scores.max())
            score_p99 = float(np.percentile(face_scores, 99))

        # 2) Predict with low threshold.
        pred = detector.predict(img, conf_threshold=args.conf_threshold, nms_iou=0.5)
        n_preds_per_img.append(int(pred.boxes.shape[0]))
        print(f"  [{img_id[:30]:30s}]  size={img.shape[1]}x{img.shape[0]}  "
              f"score_max={score_max:.4f}  p99={score_p99:.4f}  "
              f"n_preds(thr={args.conf_threshold})={pred.boxes.shape[0]}")

    arr = np.asarray(all_scores, dtype=np.float32)
    print("\n=== Aggregate face-score statistics ===")
    print(f"  total anchors scored: {arr.size}")
    if arr.size:
        print(f"  score  min={arr.min():.4f}  max={arr.max():.4f}  mean={arr.mean():.4f}")
        print(f"  score p50={np.percentile(arr,50):.4f}  p90={np.percentile(arr,90):.4f}  "
              f"p99={np.percentile(arr,99):.4f}")
        for thr in [0.005, 0.01, 0.02, 0.05, 0.10, 0.30, 0.50, 0.70]:
            print(f"  frac anchors above {thr}: {(arr > thr).mean():.6f}")

    print(f"\n  avg predictions per image: {np.mean(n_preds_per_img) if n_preds_per_img else 0:.2f}")
    print(f"  max predictions per image: {np.max(n_preds_per_img) if n_preds_per_img else 0}")

    # 3) Verdict.
    print("\n=== Verdict ===")
    if arr.size == 0:
        print("  ❌ No anchors scored — model output is empty.")
    elif arr.max() < 0.05:
        print("  ❌ Model face logits are dead — max score < 0.05.")
        print("     → training has not converged; cls loss is too high.")
        print("     → Action: increase cls_weight to 2-5, train longer (10+ epochs).")
    elif arr.max() < 0.30:
        print("  ⚠️  Model has learned *something* (max score in [0.05, 0.30)) but is undertrained.")
        print("     → Action: lower conf_threshold to 0.02 (already set in patched config).")
        print("     → Action: train 5-10 more epochs.")
    elif arr.max() < 0.70:
        print("  ⚠️  Model face scores peak in [0.30, 0.70).")
        print("     → Action: use conf_threshold ~ 0.05-0.10.")
    else:
        print("  ✅ Model outputs look healthy.")

    if np.mean(n_preds_per_img) == 0:
        print("\n  ❌ Even with conf_threshold=0.02, no predictions survive NMS.")
        print("     → Likely: box head produces degenerate boxes OR NMS too aggressive.")
        print("     → Action: try --conf-threshold 0.005 or disable NMS by setting nms_iou=1.0.")


if __name__ == "__main__":
    main()
