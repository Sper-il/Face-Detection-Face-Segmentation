"""
FDDB Evaluation
================
Đánh giá model face detection trên bộ FDDB (Face Detection Data Set and
Benchmark): sinh ROC curve (discrete score) và file kết quả theo đúng
định dạng chuẩn FDDB để có thể chạy tiếp bằng công cụ đánh giá gốc
(``evaluate`` binary của FDDB) nếu cần.

Định dạng annotation FDDB (``FDDB-fold-XX-ellipseList.txt``)::

    2002/08/11/big/img_591
    1
    123.583300 85.549500 1.265839 269.693400 161.781200  1

    mỗi dòng ellipse: major_axis_radius minor_axis_radius angle
    center_x center_y (1)

Định dạng output chuẩn (dùng cho FDDB eval tool)::

    2002/08/11/big/img_591
    1
    x1 y1 w h score

Cách chạy::

    python -m src.evaluation.fddb_eval \\
        --checkpoint weights/detection_best.pth \\
        --images-dir data/processed/fddb/originalPics \\
        --ellipse-file data/processed/fddb/FDDB-fold-01-ellipseList.txt \\
        --output-dir outputs/metrics/fddb

Nếu không có dữ liệu FDDB thật, script tự sinh bộ dữ liệu tí hon để test
(giống convention Dummy*/synthetic của các module khác trong repo).
"""

import argparse
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.evaluation.metrics import compute_iou

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Đọc annotation FDDB (ellipse) & quy đổi sang bounding box
# ---------------------------------------------------------------------------
def load_fddb_annotations(ellipse_file: str) -> Dict[str, List[Tuple[float, float, float, float, float]]]:
    """Đọc file ellipse annotation của FDDB.

    Returns:
        Dict ``{image_relative_path: [(ra, rb, theta, cx, cy), ...]}``.
    """
    annotations: Dict[str, List[Tuple[float, float, float, float, float]]] = {}
    with open(ellipse_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() != ""]

    i = 0
    while i < len(lines):
        img_path = lines[i]
        i += 1
        if i >= len(lines):
            break
        n_faces = int(lines[i])
        i += 1

        ellipses = []
        for _ in range(n_faces):
            if i >= len(lines):
                break
            parts = lines[i].split()
            i += 1
            ra, rb, theta, cx, cy = map(float, parts[:5])
            ellipses.append((ra, rb, theta, cx, cy))
        annotations[img_path] = ellipses

    return annotations


def ellipse_to_bbox(ellipse: Tuple[float, float, float, float, float]) -> List[float]:
    """Quy đổi ellipse (ra, rb, theta, cx, cy) sang bounding box [x1,y1,x2,y2].

    Dùng công thức bounding box của ellipse xoay: nửa chiều rộng/cao của
    axis-aligned bbox bao quanh 1 ellipse xoay góc theta là::

        half_w = sqrt((ra*cos(theta))^2 + (rb*sin(theta))^2)
        half_h = sqrt((ra*sin(theta))^2 + (rb*cos(theta))^2)
    """
    ra, rb, theta, cx, cy = ellipse
    half_w = math.sqrt((ra * math.cos(theta)) ** 2 + (rb * math.sin(theta)) ** 2)
    half_h = math.sqrt((ra * math.sin(theta)) ** 2 + (rb * math.cos(theta)) ** 2)
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


# ---------------------------------------------------------------------------
# Sinh dữ liệu demo khi chưa có FDDB thật
# ---------------------------------------------------------------------------
def _make_synthetic_fddb(root: Path, num_images: int = 6) -> Tuple[Path, Path]:
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    ellipse_file = root / "FDDB-ellipseList.txt"

    rng = np.random.default_rng(7)
    lines = []
    for idx in range(num_images):
        w, h = 320, 240
        img = Image.fromarray(rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8), mode="RGB")
        name = f"synthetic/{idx}"
        (images_dir / "synthetic").mkdir(exist_ok=True)
        img.save(images_dir / f"{name}.jpg")

        n_faces = rng.integers(1, 3)
        lines.append(name)
        lines.append(str(n_faces))
        for _ in range(n_faces):
            ra = rng.uniform(20, 50)
            rb = ra * rng.uniform(0.8, 1.0)
            theta = rng.uniform(-0.3, 0.3)
            cx = rng.uniform(ra, w - ra)
            cy = rng.uniform(rb, h - rb)
            lines.append(f"{ra:.4f} {rb:.4f} {theta:.4f} {cx:.4f} {cy:.4f} 1")

    ellipse_file.write_text("\n".join(lines), encoding="utf-8")
    logger.warning(
        "Không tìm thấy dữ liệu FDDB thật -> đã sinh %d ảnh synthetic tại %s "
        "để test pipeline. Kết quả chỉ có ý nghĩa kiểm thử code.",
        num_images, root,
    )
    return images_dir, ellipse_file


# ---------------------------------------------------------------------------
# Ghi kết quả theo format chuẩn FDDB
# ---------------------------------------------------------------------------
def write_fddb_result_file(
    predictions: Dict[str, Dict[str, list]],
    out_path: str,
) -> None:
    """Ghi file kết quả detection theo format chuẩn FDDB (để dùng với FDDB eval tool gốc).

    Args:
        predictions: ``{image_relative_path_no_ext: {'boxes': [...], 'scores': [...]}}``.
        out_path: đường dẫn file output.
    """
    lines = []
    for img_path, pred in predictions.items():
        boxes, scores = pred["boxes"], pred["scores"]
        lines.append(img_path)
        lines.append(str(len(boxes)))
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            lines.append(f"{x1:.2f} {y1:.2f} {x2 - x1:.2f} {y2 - y1:.2f} {score:.4f}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# ROC curve kiểu "discrete score" (chuẩn FDDB): trục X = số false positive
# tích lũy, trục Y = true positive rate, quét theo ngưỡng confidence giảm dần.
# ---------------------------------------------------------------------------
def compute_roc(
    predictions: Dict[str, Dict[str, list]],
    gt_boxes_by_image: Dict[str, List[List[float]]],
    iou_threshold: float = 0.5,
    num_thresholds: int = 50,
) -> Dict[str, list]:
    """Tính ROC curve (discrete score) cho FDDB.

    Returns:
        Dict với ``thresholds``, ``true_positive_rate``, ``false_positives``
        (số lượng FP tích lũy trên toàn bộ tập ảnh, không phải tỉ lệ).
    """
    total_gt = sum(len(v) for v in gt_boxes_by_image.values())
    thresholds = np.linspace(0.99, 0.0, num_thresholds)

    tpr_list, fp_list = [], []
    for th in thresholds:
        tp, fp = 0, 0
        for img_path, gt_boxes in gt_boxes_by_image.items():
            pred = predictions.get(img_path, {"boxes": [], "scores": []})
            matched_gt = set()
            for box, score in zip(pred["boxes"], pred["scores"]):
                if score < th:
                    continue
                best_iou, best_j = 0.0, -1
                for j, gt in enumerate(gt_boxes):
                    if j in matched_gt:
                        continue
                    iou = compute_iou(box, gt)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_threshold and best_j >= 0:
                    tp += 1
                    matched_gt.add(best_j)
                else:
                    fp += 1
        tpr_list.append(tp / total_gt if total_gt > 0 else 0.0)
        fp_list.append(fp)

    return {
        "thresholds": thresholds.tolist(),
        "true_positive_rate": tpr_list,
        "false_positives": fp_list,
    }


# ---------------------------------------------------------------------------
# Đánh giá tổng hợp
# ---------------------------------------------------------------------------
def evaluate_fddb(
    detector,
    images_dir: str,
    ellipse_file: str,
    output_dir: str,
    iou_threshold: float = 0.5,
) -> Dict[str, object]:
    annotations = load_fddb_annotations(ellipse_file)

    predictions: Dict[str, Dict[str, list]] = {}
    gt_boxes_by_image: Dict[str, List[List[float]]] = {}

    for rel_path, ellipses in annotations.items():
        img_path = os.path.join(images_dir, rel_path + ".jpg")
        if not os.path.isfile(img_path):
            # FDDB gốc không có đuôi .jpg trong annotation nhưng ảnh trên đĩa có
            alt_path = os.path.join(images_dir, rel_path)
            img_path = alt_path if os.path.isfile(alt_path) else img_path
        if not os.path.isfile(img_path):
            logger.warning("Bỏ qua ảnh không tồn tại: %s", img_path)
            continue

        result = detector.predict(img_path)
        predictions[rel_path] = result
        gt_boxes_by_image[rel_path] = [ellipse_to_bbox(e) for e in ellipses]

    roc = compute_roc(predictions, gt_boxes_by_image, iou_threshold=iou_threshold)

    result_file = os.path.join(output_dir, "fddb_detections.txt")
    write_fddb_result_file(predictions, result_file)

    return {
        "num_images": len(predictions),
        "num_gt_faces": sum(len(v) for v in gt_boxes_by_image.values()),
        "roc": roc,
        "result_file": result_file,
        "final_true_positive_rate": roc["true_positive_rate"][-1] if roc["true_positive_rate"] else 0.0,
        "final_false_positives": roc["false_positives"][-1] if roc["false_positives"] else 0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Đánh giá detection model trên FDDB")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--ellipse-file", type=str, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/metrics/fddb")
    args = parser.parse_args()

    from src.inference.detector import FaceDetector

    images_dir, ellipse_file = args.images_dir, args.ellipse_file
    if not images_dir or not ellipse_file or not os.path.isdir(images_dir) or not os.path.isfile(ellipse_file):
        tmp_dir = tempfile.mkdtemp(prefix="fddb_synth_")
        images_dir, ellipse_file = _make_synthetic_fddb(Path(tmp_dir))
        images_dir, ellipse_file = str(images_dir), str(ellipse_file)

    detector = FaceDetector(checkpoint_path=args.checkpoint, device=args.device, conf_threshold=args.conf_threshold)

    logger.info("Bắt đầu đánh giá FDDB: images_dir=%s, ellipse_file=%s", images_dir, ellipse_file)
    results = evaluate_fddb(detector, images_dir, ellipse_file, args.output_dir, iou_threshold=args.iou_threshold)

    print("\n" + "=" * 50)
    print("  FDDB EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Số ảnh              : {results['num_images']}")
    print(f"  Số khuôn mặt GT     : {results['num_gt_faces']}")
    print(f"  TPR @ ngưỡng thấp   : {results['final_true_positive_rate']:.4f}")
    print(f"  Tổng FP @ ngưỡng thấp: {results['final_false_positives']}")
    print(f"  File kết quả (chuẩn FDDB): {results['result_file']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
