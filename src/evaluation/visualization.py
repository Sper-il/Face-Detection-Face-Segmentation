"""
Evaluation Visualization
==========================
Các hàm trực quan hoá kết quả đánh giá cho cả detection và segmentation:

- ``draw_detection_comparison``: vẽ GT (xanh lá) vs Prediction (đỏ) trên ảnh.
- ``draw_segmentation_comparison``: so sánh mask GT vs Prediction (overlay màu theo lớp).
- ``plot_confusion_matrix``: vẽ ma trận nhầm lẫn cho segmentation.
- ``plot_roc_curve``: vẽ ROC curve kiểu FDDB (FP tích lũy vs TPR).
- ``plot_pr_curve``: vẽ Precision-Recall curve cho detection.

Tất cả hàm đều nhận `save_path` optional; nếu không truyền, trả về
``matplotlib.figure.Figure`` để caller tự xử lý (hiển thị trong notebook, ...).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # an toàn khi chạy không có display (server/CLI)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


CELEBAMASK_CLASSES: List[str] = [
    "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g",
    "l_ear", "r_ear", "ear_r", "nose", "mouth", "u_lip", "l_lip", "neck",
    "neck_l", "cloth", "hair", "hat",
]


def generate_color_palette(num_classes: int) -> np.ndarray:
    """Sinh bảng màu cố định (deterministic) cho num_classes lớp, class 0 = đen (nền)."""
    rng = np.random.default_rng(0)
    palette = rng.integers(30, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    return palette


# ---------------------------------------------------------------------------
# Detection: GT vs Prediction
# ---------------------------------------------------------------------------
def draw_detection_comparison(
    image: np.ndarray,
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
    pred_scores: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
    title: str = "GT (xanh) vs Prediction (đỏ)",
):
    """Vẽ ảnh với box GT (xanh lá) và box dự đoán (đỏ) chồng lên nhau.

    Args:
        image: ảnh RGB, ``np.ndarray`` shape (H, W, 3), uint8.
        gt_boxes: list các box ``[x1, y1, x2, y2]`` ground-truth.
        pred_boxes: list các box ``[x1, y1, x2, y2]`` dự đoán.
        pred_scores: điểm confidence tương ứng với pred_boxes (optional, hiển thị làm nhãn).
        save_path: nếu truyền, lưu hình ra file thay vì trả về Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

    for box in gt_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

    for idx, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none", linestyle="--")
        ax.add_patch(rect)
        if pred_scores is not None and idx < len(pred_scores):
            ax.text(x1, max(y1 - 4, 0), f"{pred_scores[idx]:.2f}", color="red", fontsize=8,
                     bbox=dict(facecolor="white", alpha=0.6, pad=0.5))

    handles = [
        patches.Patch(edgecolor="lime", facecolor="none", label="Ground Truth"),
        patches.Patch(edgecolor="red", facecolor="none", label="Prediction"),
    ]
    ax.legend(handles=handles, loc="upper right")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        logger.info("Đã lưu hình so sánh detection tại %s", save_path)
        return save_path
    return fig


# ---------------------------------------------------------------------------
# Segmentation: GT vs Prediction
# ---------------------------------------------------------------------------
def draw_segmentation_comparison(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    alpha: float = 0.5,
):
    """So sánh mask ground-truth và mask dự đoán, hiển thị 3 panel: ảnh gốc | GT overlay | Pred overlay.

    Args:
        image: ảnh RGB gốc (H, W, 3), uint8.
        gt_mask: mask GT (H, W), giá trị nguyên là class id.
        pred_mask: mask dự đoán (H, W), giá trị nguyên là class id.
        class_names: tên lớp (dùng cho chú thích màu), mặc định 19 lớp CelebAMask-HQ.
        save_path: nếu truyền, lưu hình ra file.
    """
    class_names = class_names or CELEBAMASK_CLASSES
    num_classes = len(class_names)
    palette = generate_color_palette(num_classes)

    def colorize(mask):
        h, w = mask.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)
        for c in np.unique(mask):
            if c >= num_classes:
                continue
            color[mask == c] = palette[c]
        return color

    gt_color = colorize(gt_mask)
    pred_color = colorize(pred_mask)

    gt_overlay = (image.astype(np.float32) * (1 - alpha) + gt_color.astype(np.float32) * alpha).astype(np.uint8)
    pred_overlay = (image.astype(np.float32) * (1 - alpha) + pred_color.astype(np.float32) * alpha).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Ảnh gốc")
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")

    present_classes = sorted(set(np.unique(gt_mask).tolist()) | set(np.unique(pred_mask).tolist()))
    handles = [
        patches.Patch(color=palette[c] / 255.0, label=class_names[c] if c < len(class_names) else str(c))
        for c in present_classes if c != 0
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8), fontsize=8, bbox_to_anchor=(0.5, -0.05))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        logger.info("Đã lưu hình so sánh segmentation tại %s", save_path)
        return save_path
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix (segmentation)
# ---------------------------------------------------------------------------
def compute_confusion_matrix(gt_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Tính confusion matrix (num_classes x num_classes) cho 1 cặp mask."""
    mask = (gt_mask >= 0) & (gt_mask < num_classes)
    cm = np.bincount(
        num_classes * gt_mask[mask].astype(int) + pred_mask[mask].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
):
    """Vẽ confusion matrix dạng heatmap.

    Args:
        cm: ma trận (num_classes, num_classes), hàng = GT, cột = Prediction.
        class_names: tên các lớp, mặc định dùng index.
        normalize: nếu True, chuẩn hoá theo hàng (tỉ lệ % trên mỗi lớp GT thật).
    """
    num_classes = cm.shape[0]
    class_names = class_names or [str(i) for i in range(num_classes)]

    cm_display = cm.astype(np.float64)
    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm_display / row_sums

    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.5), max(5, num_classes * 0.45)))
    im = ax.imshow(cm_display, cmap="Blues", vmin=0, vmax=1 if normalize else cm_display.max())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        logger.info("Đã lưu confusion matrix tại %s", save_path)
        return save_path
    return fig


# ---------------------------------------------------------------------------
# ROC curve (FDDB-style) & PR curve (detection)
# ---------------------------------------------------------------------------
def plot_roc_curve(
    false_positives: Sequence[int],
    true_positive_rate: Sequence[float],
    save_path: Optional[str] = None,
    title: str = "FDDB ROC Curve (discrete score)",
):
    """Vẽ ROC curve kiểu FDDB: trục X = số false positive tích luỹ, trục Y = TPR."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(false_positives, true_positive_rate, marker="o", markersize=3, color="tab:blue")
    ax.set_xlabel("Số lượng False Positive (tích lũy)")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        logger.info("Đã lưu ROC curve tại %s", save_path)
        return save_path
    return fig


def plot_pr_curve(
    recalls: Sequence[float],
    precisions: Sequence[float],
    ap: Optional[float] = None,
    save_path: Optional[str] = None,
    title: str = "Precision-Recall Curve",
):
    """Vẽ Precision-Recall curve cho detection (1 class hoặc trung bình các lớp)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recalls, precisions, color="tab:orange", linewidth=2)
    ax.fill_between(recalls, precisions, alpha=0.1, color="tab:orange")
    label = title if ap is None else f"{title} (AP={ap:.3f})"
    ax.set_title(label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        logger.info("Đã lưu PR curve tại %s", save_path)
        return save_path
    return fig


# ---------------------------------------------------------------------------
# Self-test nhanh
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("Evaluation Visualization — smoke test")
    print("=" * 50)

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)

    # Detection comparison
    gt_boxes = [[30, 30, 100, 100], [150, 60, 200, 130]]
    pred_boxes = [[35, 33, 105, 102], [160, 90, 210, 150]]
    pred_scores = [0.92, 0.61]
    out1 = draw_detection_comparison(img, gt_boxes, pred_boxes, pred_scores, save_path="/tmp/det_compare_test.png")
    print("  detection comparison ->", out1)

    # Segmentation comparison
    gt_mask = rng.integers(0, 19, (256, 256)).astype(np.uint8)
    pred_mask = rng.integers(0, 19, (256, 256)).astype(np.uint8)
    out2 = draw_segmentation_comparison(img, gt_mask, pred_mask, save_path="/tmp/seg_compare_test.png")
    print("  segmentation comparison ->", out2)

    # Confusion matrix
    cm = compute_confusion_matrix(gt_mask, pred_mask, num_classes=19)
    out3 = plot_confusion_matrix(cm, CELEBAMASK_CLASSES, save_path="/tmp/cm_test.png")
    print("  confusion matrix ->", out3)

    # ROC & PR
    out4 = plot_roc_curve([50, 30, 10, 2, 0], [0.95, 0.9, 0.8, 0.6, 0.3], save_path="/tmp/roc_test.png")
    print("  ROC curve ->", out4)
    out5 = plot_pr_curve([0.1, 0.3, 0.5, 0.8, 1.0], [1.0, 0.95, 0.85, 0.7, 0.4], ap=0.78, save_path="/tmp/pr_test.png")
    print("  PR curve ->", out5)

    print("Smoke test passed ✓")
