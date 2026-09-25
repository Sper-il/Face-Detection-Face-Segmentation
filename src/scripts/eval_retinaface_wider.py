"""Evaluate RetinaFace on WIDER FACE (original format).

Usage::

    python -m src.scripts.eval_retinaface_wider --checkpoint models/retinaface_final.pth

Default checkpoint: models/retinaface_final.pth
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.detection.anchors import AnchorConfig
from src.detection.inference import RetinaFaceDetector
from src.utils.box_ops import bbox_iou


class WIDERFaceGT:
    """Parse WIDER_FACE ground truth format."""
    
    def __init__(self, label_file: Path):
        self.entries = []  # list of (image_id, boxes)
        self._parse(label_file)
    
    def _parse(self, label_file: Path):
        with open(label_file, "r") as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Dòng đầu: đường dẫn ảnh
            img_path = lines[i].strip()
            i += 1
            
            if not img_path or i >= len(lines):
                break
            
            # Tên ảnh không có extension
            img_id = img_path.replace(".jpg", "").replace(".png", "")
            
            # Số boxes
            try:
                num_boxes = int(lines[i].strip())
            except ValueError:
                i += 1
                continue
            i += 1
            
            boxes = []
            for _ in range(num_boxes):
                if i >= len(lines):
                    break
                parts = lines[i].strip().split()
                if len(parts) >= 4:
                    x, y, w, h = map(float, parts[:4])
                    # Convert to xyxy
                    boxes.append([x, y, x + w, y + h])
                i += 1
            
            if boxes:
                self.entries.append((img_id, np.array(boxes, dtype=np.float32)))
    
    def get(self, img_id: str) -> np.ndarray | None:
        for eid, boxes in self.entries:
            if eid == img_id:
                return boxes
        return None


def compute_ap_recall(gt_boxes, pred_boxes, pred_scores, iou_thresh=0.5):
    """Compute AP and Recall."""
    if len(gt_boxes) == 0:
        return 0.0, 0.0
    
    if len(pred_boxes) == 0:
        return 0.0, 0.0
    
    # Sort by score
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    
    tp = []
    fp = []
    matched = [False] * len(gt_boxes)
    
    for pred_box in pred_boxes:
        if len(gt_boxes) == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue
        
        ious = bbox_iou(pred_box.reshape(1, 4), gt_boxes)[0]
        best_idx = ious.argmax()
        
        if ious[best_idx] >= iou_thresh and not matched[best_idx]:
            matched[best_idx] = True
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    
    tp = np.array(tp)
    fp = np.array(fp)
    cum_tp = tp.cumsum()
    cum_fp = fp.cumsum()
    
    recall = cum_tp / len(gt_boxes)
    precision = cum_tp / (cum_tp + cum_fp + 1e-7)
    
    # AP (11-point interpolation)
    recall_grid = np.linspace(0, 1, 11)
    ap = 0
    for r in recall_grid:
        mask = recall >= r
        if mask.any():
            ap += float(precision[mask].max())
    ap /= 11
    
    return ap, float(recall[-1]) if len(recall) > 0 else 0.0


def find_image(images_dir: Path, img_id: str) -> Path | None:
    """Tìm ảnh trong thư mục category."""
    # img_id format: "0--Parade/0_Parade_marchingband_1_100"
    parts = img_id.split("/")
    if len(parts) == 2:
        cat_dir = parts[0]
        filename = parts[1]
        search_dir = images_dir / cat_dir
        if search_dir.exists():
            for ext in [".jpg", ".png", ".jpeg"]:
                path = search_dir / f"{filename}{ext}"
                if path.exists():
                    return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/retinaface_final.pth")
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load detector
    anchor_cfg = AnchorConfig(image_size=640, strides=(8, 16, 32))
    detector = RetinaFaceDetector(
        weights=args.checkpoint,
        device=device,
        anchor_cfg=anchor_cfg,
        pretrained=False,
    )
    print(f"Loaded: {args.checkpoint}")
    print(f"Device: {device}")
    
    # Data paths
    wider_base = Path("data/raw/WIDER_FACE")
    images_dir = wider_base / f"WIDER_{args.split.upper()}" / "images"
    label_file = wider_base / "wider_face_split" / f"wider_face_{args.split}_bbx_gt.txt"
    
    print(f"Images: {images_dir}")
    print(f"Labels: {label_file}")
    
    # Load GT
    gt = WIDERFaceGT(label_file)
    print(f"Loaded {len(gt.entries)} images with GT")
    
    # Evaluate
    image_ids = [e[0] for e in gt.entries]
    if args.max_images:
        image_ids = image_ids[:args.max_images]
    
    print(f"Evaluating {len(image_ids)} images...")
    
    all_aps = []
    all_gts = 0
    all_preds = 0
    found = 0
    
    for img_id in image_ids:
        gt_boxes = gt.get(img_id)
        if gt_boxes is None:
            continue
        
        img_path = find_image(images_dir, img_id)
        if img_path is None or not img_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        found += 1
        all_gts += len(gt_boxes)
        
        pred = detector.predict(img, conf_threshold=args.conf_threshold, nms_iou=args.nms_iou)
        
        if len(pred.boxes) > 0:
            all_preds += len(pred.boxes)
            ap, _ = compute_ap_recall(gt_boxes, pred.boxes, pred.scores)
            all_aps.append(ap)
    
    print(f"Processed {found}/{len(image_ids)} images")
    
    # Summary
    mean_ap = np.mean(all_aps) if all_aps else 0.0
    recall = all_gts / max(all_gts, 1)
    
    print("\n" + "="*55)
    print("RETINAFACE DETECTION EVALUATION")
    print("="*55)
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Split:            {args.split.upper()}")
    print(f"Images processed: {found}")
    print(f"Ground truth BB:  {all_gts}")
    print(f"Total predictions:{all_preds}")
    print(f"mAP@0.5:          {mean_ap:.4f}")
    print("="*55)
    
    # Save
    out = Path("runs/eval")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "retinaface_wider.json", "w") as f:
        json.dump({
            "mAP": mean_ap,
            "recall": recall,
            "n_images": found,
            "n_gt": all_gts,
            "n_pred": all_preds
        }, f, indent=2)
    print(f"\nSaved: runs/eval/retinaface_wider.json")


if __name__ == "__main__":
    main()
