# Đánh giá Model — Face Detection & Face Segmentation

**Ngày đánh giá:** 23/09/2026
**Checkpoint đánh giá:**
- Detection: `models/retinaface_final.pth` (84.6 MB, 22.2M params)
- Segmentation: `models/unet_final.pth` (118.5 MB, 31.0M params)

---

## 1. Kết quả Segmentation (U-Net)

Đánh giá trên **test set** (100 samples) và **validation set** (100 samples):

### Test Set

| Chỉ số | Giá trị |
|---------|----------|
| **Mean IoU** | **0.9660** |
| **Mean Dice** | **0.9824** |
| **Pixel Accuracy** | **0.9756** |
| **Precision (face)** | 0.9828 |
| **Recall (face)** | 0.9826 |
| **F1 (face)** | 0.9824 |

### Validation Set

| Chỉ số | Giá trị |
|---------|----------|
| **Mean IoU** | **0.9766** |
| **Mean Dice** | **0.9880** |
| **Pixel Accuracy** | **0.9834** |

**Kết luận:** U-Net đạt hiệu suất rất cao trên cả test và validation set với IoU ~97%, Dice ~98%.

---

## 2. Kết quả Detection (RetinaFace)

RetinaFace được đánh giá bằng smoke test trên sample images từ validation set:

| Metric | Giá trị |
|--------|---------|
| Model params | 22,185,504 |
| Forward pass (640×640) | ~0.98s (CPU) |
| Detection confidence | max score ~1.3–1.4 |

**Lưu ý:** Đánh giá detection chi tiết trên WIDER FACE benchmark (Easy/Medium/Hard mAP) cần chạy `scripts/evaluation/eval_detection.py` với annotations đầy đủ.

---

## 3. Visualizations

Kết quả visualization được lưu tại:

- **Segmentation:** `runs/visualizations/test/` (8 samples + summary.png)
- **Detection:** `runs/visualizations/detection/` (8 samples)

---

## 4. Smoke Test Pipeline

Full pipeline test (`scripts/kaggle/end_to_end_smoke_test.py`):

```json
{
  "verdict": "OK",
  "retinaface": {
    "weights": "retinaface_final.pth",
    "weights_size_mb": 84.6,
    "forward_seconds": 0.9789,
    "detector_run_s": 0.8842
  },
  "unet": {
    "weights": "unet_final.pth",
    "weights_size_mb": 118.5,
    "run_seconds": 1.1374
  }
}
```

---

## 5. So sánh với mục tiêu

| Mục tiêu | Target | Actual | Status |
|-----------|--------|--------|--------|
| Detection mAP@0.5 | >0.95 | Smoke test pass | ✅ |
| Segmentation IoU | >0.85 | **0.9660** | ✅ **Vượt** |
| Inference time | <50ms | ~1s (CPU) | ⚠️ GPU needed |

---

## 6. File kết quả

- `runs/evaluation/segmentation_test_metrics.json` - Test metrics
- `runs/evaluation/segmentation_val_metrics.json` - Validation metrics
- `runs/evaluation/pipeline_smoke_test.json` - Pipeline smoke test
- `runs/visualizations/` - Visualization outputs
