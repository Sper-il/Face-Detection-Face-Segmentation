"""
Comprehensive evaluation on the NEW 512x512 processed dataset.

Run with defaults (CPU, val split, smoke 50 images each)::

    python -m src.scripts.evaluate_models

Run full (no cap)::

    python -m src.scripts.evaluate_models --max-images 0 --device cpu

This replaces the old smoke-train eval; results go to runs/eval/.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from src.detection.anchors import AnchorConfig
from src.detection.eval import EvalEntry, compute_map_recall, group_by_image
from src.detection.inference import RetinaFaceDetector
from src.segmentation.eval import numpy_metrics
from src.segmentation.inference import UNetSegmentor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / "models"
DATA_DIR = REPO / "data" / "processed"

DET_IMAGES = DATA_DIR / "detection" / "val" / "images"
DET_CSV = DATA_DIR / "detection" / "val" / "annotations.csv"
SEG_IMAGES = DATA_DIR / "segmentation" / "val" / "images"
SEG_MASKS = DATA_DIR / "segmentation" / "val" / "masks"

OUT_DIR = REPO / "runs" / "eval"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_detection_gt(csv_path: Path) -> list[EvalEntry]:
    entries: list[EvalEntry] = []
    with open(csv_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6 or parts[0].lower() == "image_id":
                continue
            try:
                x1, y1, x2, y2 = (float(parts[i]) for i in range(1, 5))
                valid = int(parts[5])
            except ValueError:
                continue
            if valid == 0:
                continue
            entries.append(EvalEntry(parts[0], np.array([x1, y1, x2, y2], dtype=np.float32)))
    return entries


def evaluate_detection(
    detector: RetinaFaceDetector,
    gt_entries: list[EvalEntry],
    images_dir: Path,
    conf_threshold: float = 0.05,
    nms_iou: float = 0.5,
    iou_threshold: float = 0.5,
    max_images: int | None = None,
    resize_before_detect: int = 640,
) -> dict:
    """Run detection eval and return metrics dict.

    Args:
        resize_before_detect: images are loaded as 512x512 PNGs but RetinaFace was
            trained at ``anchor_cfg.image_size`` (default 640). We resize to that
            size *before* feeding to the detector so anchor coordinates are calibrated.
            GT boxes are scaled by the same ratio so IoU is consistent.
    """
    gt_by_img = group_by_image(gt_entries)
    image_ids = sorted(gt_by_img.keys())
    if max_images:
        image_ids = image_ids[:max_images]

    # Scale GT boxes from 512-px canvas to ``resize_before_detect`` canvas.
    scale = resize_before_detect / 512.0

    pred_entries: list[EvalEntry] = []
    n_processed = 0
    t0 = time.perf_counter()

    for img_id in image_ids:
        img_path = images_dir / img_id
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Resize to training canvas size so anchors are calibrated.
        img_resized = cv2.resize(img, (resize_before_detect, resize_before_detect),
                                  interpolation=cv2.INTER_LINEAR)

        n_processed += 1
        pred = detector.predict(img_resized, conf_threshold=conf_threshold, nms_iou=nms_iou)
        for box, score in zip(pred.boxes, pred.scores):
            pred_entries.append(EvalEntry(img_id, box, float(score)))

    elapsed = time.perf_counter() - t0

    # Scale GT boxes so they are in the same coordinate space as the resized images.
    scaled_gt = []
    for e in gt_entries:
        if e.image_id in {iid for iid in image_ids}:
            scaled_box = e.bbox * scale
            scaled_gt.append(EvalEntry(e.image_id, scaled_box.astype(np.float32), e.score))

    metrics = compute_map_recall(scaled_gt, pred_entries, iou_threshold=iou_threshold)
    metrics["n_processed"] = n_processed
    metrics["latency_ms"] = (elapsed / max(n_processed, 1)) * 1000
    return metrics


def evaluate_segmentation(
    seg: UNetSegmentor,
    images_dir: Path,
    masks_dir: Path,
    max_images: int | None = None,
) -> dict:
    """Run segmentation eval and return metrics dict."""
    image_files = sorted(list(images_dir.glob("*.png")))
    if max_images:
        image_files = image_files[:max_images]

    ious, dices, accs = [], [], []
    t0 = time.perf_counter()

    for img_path in image_files:
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or gt_mask is None:
            continue
        pred = seg.predict_full(img)
        m = numpy_metrics(pred.mask, gt_mask)
        ious.append(m["iou"])
        dices.append(m["dice"])
        accs.append(m["pixel_acc"])

    elapsed = time.perf_counter() - t0
    n = len(ious)
    return {
        "iou": float(np.mean(ious)) if ious else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "pixel_acc": float(np.mean(accs)) if accs else 0.0,
        "n_images": n,
        "latency_ms": (elapsed / max(n, 1)) * 1000,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RetinaFace + U-Net on 512x512 processed data.")
    p.add_argument("--det-weights", default=str(MODEL_DIR / "retinaface_final.pth"))
    p.add_argument("--seg-weights", default=str(MODEL_DIR / "unet_final.pth"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-images", type=int, default=50,
                   help="0 = no cap (full split). Default 50 for smoke test.")
    p.add_argument("--det-conf", type=float, default=0.05)
    p.add_argument("--det-nms", type=float, default=0.5)
    p.add_argument("--det-iou", type=float, default=0.5)
    p.add_argument("--det-image-size", type=int, default=640,
                   help="RetinaFace input canvas size (default 640, matches training)")
    p.add_argument("--seg-image-size", type=int, default=512,
                   help="U-Net input size (default 512, matches processed dataset)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    max_imgs = args.max_images if args.max_images > 0 else None
    anchor_cfg = AnchorConfig(image_size=args.det_image_size, strides=(8, 16, 32))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # 1. RetinaFace detection
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RETINAFACE DETECTION EVALUATION")
    print("=" * 65)
    print(f"  Checkpoint : {args.det_weights}")
    print(f"  Images dir : {DET_IMAGES}")
    print(f"  Annotations: {DET_CSV}")
    print(f"  Canvas size: {args.det_image_size}")
    print(f"  Conf/NMS   : {args.det_conf} / {args.det_nms}")
    print(f"  Device     : {device}")
    print(f"  Max images : {max_imgs or 'ALL'}")

    if not Path(args.det_weights).exists():
        print(f"  ERROR: checkpoint not found: {args.det_weights}")
        return 1
    if not DET_CSV.exists():
        print(f"  ERROR: annotations not found: {DET_CSV}")
        return 1

    detector = RetinaFaceDetector(
        weights=args.det_weights,
        device=str(device),
        anchor_cfg=anchor_cfg,
        pretrained=False,
    )
    gt_entries = load_detection_gt(DET_CSV)
    print(f"  GT entries : {len(gt_entries)}")

    det_metrics = evaluate_detection(
        detector, gt_entries, DET_IMAGES,
        conf_threshold=args.det_conf,
        nms_iou=args.det_nms,
        iou_threshold=args.det_iou,
        max_images=max_imgs,
        resize_before_detect=args.det_image_size,
    )

    print(f"\n  Results ({det_metrics['n_processed']} images processed):")
    print(f"    mAP@0.5        : {det_metrics['mAP']:.4f}")
    print(f"    Recall@0.5     : {det_metrics['recall']:.4f}")
    print(f"    n_gt           : {det_metrics['n_gt']}")
    print(f"    n_pred         : {det_metrics['n_pred']}")
    print(f"    latency (ms)   : {det_metrics['latency_ms']:.1f}")

    det_out = OUT_DIR / f"detection_{ts}.json"
    with open(det_out, "w") as f:
        json.dump(det_metrics, f, indent=2)
    print(f"  Saved: {det_out}")

    # ------------------------------------------------------------------
    # 2. U-Net segmentation
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  U-NET SEGMENTATION EVALUATION")
    print("=" * 65)
    print(f"  Checkpoint : {args.seg_weights}")
    print(f"  Images dir : {SEG_IMAGES}")
    print(f"  Masks dir  : {SEG_MASKS}")
    print(f"  Input size : {args.seg_image_size}")
    print(f"  Device     : {device}")
    print(f"  Max images : {max_imgs or 'ALL'}")

    if not Path(args.seg_weights).exists():
        print(f"  ERROR: checkpoint not found: {args.seg_weights}")
        return 1
    if not SEG_IMAGES.exists():
        print(f"  ERROR: images not found: {SEG_IMAGES}")
        return 1
    if not SEG_MASKS.exists():
        print(f"  ERROR: masks not found: {SEG_MASKS}")
        return 1

    seg = UNetSegmentor(
        weights=args.seg_weights,
        device=str(device),
        image_size=args.seg_image_size,
    )
    print(f"  UNetSegmentor image_size: {args.seg_image_size} (model trained at seg_size=256 from checkpoint)")
    print(f"  -> Will resize 512x input to {args.seg_image_size}x{args.seg_image_size} before inference")

    seg_metrics = evaluate_segmentation(
        seg, SEG_IMAGES, SEG_MASKS, max_images=max_imgs
    )

    print(f"\n  Results ({seg_metrics['n_images']} images evaluated):")
    print(f"    IoU            : {seg_metrics['iou']:.4f}")
    print(f"    Dice           : {seg_metrics['dice']:.4f}")
    print(f"    Pixel Acc      : {seg_metrics['pixel_acc']:.4f}")
    print(f"    latency (ms)   : {seg_metrics['latency_ms']:.1f}")

    seg_out = OUT_DIR / f"segmentation_{ts}.json"
    with open(seg_out, "w") as f:
        json.dump(seg_metrics, f, indent=2)
    print(f"  Saved: {seg_out}")

    # ------------------------------------------------------------------
    # 3. Combined summary
    # ------------------------------------------------------------------
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_canvas": "512x512",
        "detection": det_metrics,
        "segmentation": seg_metrics,
        "thresholds": {
            "det_conf": args.det_conf,
            "det_nms": args.det_nms,
            "det_iou": args.det_iou,
            "det_image_size": args.det_image_size,
            "seg_image_size": args.seg_image_size,
        },
    }
    summary_out = OUT_DIR / f"summary_{ts}.json"
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_out}")

    print("\n" + "=" * 65)
    print("  COMBINED SUMMARY")
    print("=" * 65)
    print(f"  {'Metric':<22} {'Value':>10}  {'Target':>10}  {'Pass'}")
    print(f"  {'-'*22} {'-'*10}  {'-'*10}  {'-'*4}")

    rows = [
        ("mAP@0.5", det_metrics["mAP"], ">= 0.85", det_metrics["mAP"] >= 0.85),
        ("Recall@0.5", det_metrics["recall"], ">= 0.90", det_metrics["recall"] >= 0.90),
        ("IoU (face)", seg_metrics["iou"], ">= 0.88", seg_metrics["iou"] >= 0.88),
        ("Dice (face)", seg_metrics["dice"], ">= 0.93", seg_metrics["dice"] >= 0.93),
        ("Pixel Acc", seg_metrics["pixel_acc"], ">= 0.96", seg_metrics["pixel_acc"] >= 0.96),
    ]
    for metric, value, target, ok in rows:
        mark = "PASS" if ok else "FAIL"
        print(f"  {metric:<22} {value:>10.4f}  {target:>10}  {mark}")

    print("=" * 65)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
