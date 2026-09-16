"""
WIDER FACE Evaluation
======================
Đánh giá model face detection trên bộ WIDER FACE, tính mAP@0.5 tổng thể
và theo 3 mức độ khó Easy / Medium / Hard.

Định dạng annotation gốc của WIDER FACE (``wider_face_val_bbx_gt.txt``)::

    0--Parade/0_Parade_marchingband_1_849.jpg
    1
    449 330 122 149 0 0 0 0 0 0
    ...

Mỗi ảnh gồm: 1 dòng đường dẫn, 1 dòng số lượng khuôn mặt, sau đó mỗi dòng
là ``x1 y1 w h blur expression illumination invalid occlusion pose``.

Lưu ý về "Easy / Medium / Hard":
    Bộ công cụ đánh giá chính thức của WIDER FACE (Matlab) dùng các file
    ``.mat`` xác định sẵn theo từng ảnh xem ground-truth nào thuộc tập
    Easy/Medium/Hard. Project này không có các file ``.mat`` đó, nên ta
    dùng một xấp xỉ phổ biến dựa trên **chiều cao khuôn mặt tính bằng
    pixel** (khuôn mặt càng nhỏ càng khó), xem ``DIFFICULTY_THRESHOLDS``.
    Kết quả vì vậy chỉ mang tính tham khảo, không phải con số "chuẩn
    benchmark" để so sánh trực tiếp với các paper khác.

Cách chạy::

    python -m src.evaluation.widerface_eval \\
        --checkpoint weights/detection_best.pth \\
        --images-dir data/processed/wider_face/val/images \\
        --gt-file data/processed/wider_face/val/wider_face_val_bbx_gt.txt \\
        --output outputs/metrics/widerface_eval.json

Nếu không truyền --images-dir/--gt-file (hoặc đường dẫn không tồn tại),
script tự sinh một bộ dữ liệu tí hon (vài ảnh + annotation giả) để có thể
chạy thử toàn bộ pipeline đánh giá ngay cả khi chưa có dữ liệu WIDER FACE
thật — cùng convention với các script train/inference khác trong repo.
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.evaluation.metrics import compute_map

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# Ngưỡng chiều cao box (pixel) dùng để xấp xỉ độ khó — xem docstring ở trên.
DIFFICULTY_THRESHOLDS = {
    "hard": (0, 20),      # mọi kích thước (bao gồm cả Easy + Medium theo protocol gốc)
    "medium": (20, 50),
    "easy": (50, float("inf")),
}


# ---------------------------------------------------------------------------
# Đọc annotation WIDER FACE
# ---------------------------------------------------------------------------
def load_widerface_annotations(gt_file: str) -> Dict[str, List[List[float]]]:
    """Đọc file annotation WIDER FACE dạng ``*_bbx_gt.txt``.

    Args:
        gt_file: đường dẫn tới file annotation.

    Returns:
        Dict ``{image_relative_path: [[x1, y1, x2, y2], ...]}``.
        Các box có cờ ``invalid == 1`` sẽ bị loại bỏ.
    """
    annotations: Dict[str, List[List[float]]] = {}
    with open(gt_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() != ""]

    i = 0
    while i < len(lines):
        img_path = lines[i]
        i += 1
        if i >= len(lines):
            break
        try:
            n_boxes = int(lines[i])
        except ValueError:
            # Một số bản annotation ghi "0" kèm theo 1 dòng box "0 0 0 0 ..."
            # thay vì bỏ qua hẳn — xử lý phòng hờ.
            n_boxes = 0
        i += 1

        boxes = []
        n_lines_to_read = max(n_boxes, 1) if n_boxes == 0 else n_boxes
        for _ in range(n_lines_to_read):
            if i >= len(lines):
                break
            parts = lines[i].split()
            i += 1
            if len(parts) < 4:
                continue
            x, y, w, h = map(float, parts[:4])
            invalid = int(parts[7]) if len(parts) > 7 else 0
            if n_boxes == 0 or w <= 0 or h <= 0 or invalid == 1:
                continue
            boxes.append([x, y, x + w, y + h])

        annotations[img_path] = boxes

    return annotations


def _box_difficulty(box: List[float]) -> str:
    """Xếp 1 box GT vào Easy/Medium/Hard theo chiều cao pixel."""
    height = box[3] - box[1]
    for level, (lo, hi) in DIFFICULTY_THRESHOLDS.items():
        if level == "hard":
            continue
        if lo <= height < hi:
            return level
    return "hard"


# ---------------------------------------------------------------------------
# Sinh dữ liệu demo khi chưa có WIDER FACE thật (giống convention Dummy* của repo)
# ---------------------------------------------------------------------------
def _make_synthetic_widerface(root: Path, num_images: int = 6) -> Tuple[Path, Path]:
    """Sinh vài ảnh + file annotation giả để test được pipeline evaluation.

    KHÔNG dùng để báo cáo số liệu thật — chỉ để đảm bảo script chạy được
    end-to-end khi chưa tải dữ liệu WIDER FACE.
    """
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    gt_file = root / "wider_face_val_bbx_gt.txt"

    rng = np.random.default_rng(42)
    lines = []
    for idx in range(num_images):
        w, h = 320, 240
        img = Image.fromarray(
            rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8), mode="RGB"
        )
        name = f"synthetic/{idx}.jpg"
        (images_dir / "synthetic").mkdir(exist_ok=True)
        img.save(images_dir / name)

        n_faces = rng.integers(1, 4)
        lines.append(name)
        lines.append(str(n_faces))
        for _ in range(n_faces):
            bw = rng.integers(15, 80)
            bh = int(bw * rng.uniform(1.0, 1.3))
            x1 = rng.integers(0, max(1, w - bw))
            y1 = rng.integers(0, max(1, h - bh))
            lines.append(f"{x1} {y1} {bw} {bh} 0 0 0 0 0 0")

    gt_file.write_text("\n".join(lines), encoding="utf-8")
    logger.warning(
        "Không tìm thấy dữ liệu WIDER FACE thật -> đã sinh %d ảnh synthetic "
        "tại %s để test pipeline. Kết quả mAP chỉ có ý nghĩa kiểm thử code, "
        "KHÔNG phản ánh chất lượng model thật.",
        num_images, root,
    )
    return images_dir, gt_file


# ---------------------------------------------------------------------------
# Đánh giá
# ---------------------------------------------------------------------------
def evaluate_widerface(
    detector,
    images_dir: str,
    gt_file: str,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Chạy detector trên toàn bộ WIDER FACE val và tính mAP@0.5.

    Args:
        detector: đối tượng có method ``predict(image_path) -> {'boxes','scores'}``,
            ví dụ ``src.inference.detector.FaceDetector``.
        images_dir: thư mục gốc chứa ảnh (đường dẫn trong gt_file là tương đối
            so với thư mục này).
        gt_file: đường dẫn file annotation ``*_bbx_gt.txt``.
        iou_threshold: ngưỡng IoU để tính mAP (mặc định 0.5 ~ mAP@0.5).

    Returns:
        Dict gồm ``overall``, ``easy``, ``medium``, ``hard`` (mỗi giá trị là mAP),
        và ``num_images`` / ``num_gt_faces``.
    """
    annotations = load_widerface_annotations(gt_file)

    pred_boxes_all, pred_scores_all, gt_boxes_all = [], [], []
    pred_by_level = {"easy": [], "medium": [], "hard": []}
    scores_by_level = {"easy": [], "medium": [], "hard": []}
    gt_by_level = {"easy": [], "medium": [], "hard": []}

    num_gt_faces = 0
    for rel_path, gt_boxes in annotations.items():
        img_path = os.path.join(images_dir, rel_path)
        if not os.path.isfile(img_path):
            logger.warning("Bỏ qua ảnh không tồn tại: %s", img_path)
            continue

        result = detector.predict(img_path)
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])

        pred_boxes_all.append(boxes)
        pred_scores_all.append(scores)
        gt_boxes_all.append(gt_boxes)
        num_gt_faces += len(gt_boxes)

        # Với mAP theo difficulty: dùng cùng dự đoán, nhưng chỉ so khớp với
        # tập con GT thuộc đúng mức độ khó (các GT khác coi như "không tồn tại").
        for level in pred_by_level:
            level_gt = [b for b in gt_boxes if _box_difficulty(b) == level]
            pred_by_level[level].append(boxes)
            scores_by_level[level].append(scores)
            gt_by_level[level].append(level_gt)

    overall_map = compute_map(pred_boxes_all, pred_scores_all, gt_boxes_all, iou_threshold)

    results = {
        "overall_mAP@0.5": overall_map,
        "num_images": len(pred_boxes_all),
        "num_gt_faces": num_gt_faces,
    }
    for level in ("easy", "medium", "hard"):
        results[f"{level}_mAP@0.5"] = compute_map(
            pred_by_level[level], scores_by_level[level], gt_by_level[level], iou_threshold
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Đánh giá detection model trên WIDER FACE")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn checkpoint .pth (bỏ trống -> dummy model)")
    parser.add_argument("--images-dir", type=str, default=None, help="Thư mục ảnh WIDER FACE val")
    parser.add_argument("--gt-file", type=str, default=None, help="File annotation *_bbx_gt.txt")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="outputs/metrics/widerface_eval.json")
    args = parser.parse_args()

    from src.inference.detector import FaceDetector

    images_dir, gt_file = args.images_dir, args.gt_file
    tmp_dir = None
    if not images_dir or not gt_file or not os.path.isdir(images_dir) or not os.path.isfile(gt_file):
        tmp_dir = tempfile.mkdtemp(prefix="widerface_synth_")
        images_dir, gt_file = _make_synthetic_widerface(Path(tmp_dir))
        images_dir = str(images_dir)
        gt_file = str(gt_file)

    detector = FaceDetector(checkpoint_path=args.checkpoint, device=args.device, conf_threshold=args.conf_threshold)

    logger.info("Bắt đầu đánh giá WIDER FACE: images_dir=%s, gt_file=%s", images_dir, gt_file)
    results = evaluate_widerface(detector, images_dir, gt_file, iou_threshold=args.iou_threshold)

    print("\n" + "=" * 50)
    print("  WIDER FACE EVALUATION RESULTS")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k:20s}: {v:.4f}" if isinstance(v, float) else f"  {k:20s}: {v}")
    print("=" * 50)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Đã lưu kết quả tại %s", out_path)


if __name__ == "__main__":
    main()
