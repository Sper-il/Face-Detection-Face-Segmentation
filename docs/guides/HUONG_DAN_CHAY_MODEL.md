# 📘 HƯỚNG DẪN CHẠY FACE DETECTION & SEGMENTATION MODEL

> **Cập nhật:** 23/09/2026  
> **Trạng thái:** ✅ Models đã train xong (U-Net IoU=0.9660, RetinaFace checkpoint loaded)

---

## 🚀 CÁCH 1: CHẠY NHANH (Khuyến nghị)

1. **Double-click** vào file `quick_run.bat`
2. **Đợi** vài phút để cài đặt (lần đầu)
3. **Xong!** Model sẽ tự động test

---

## 📋 CÁC BƯỚC CHI TIẾT

### Bước 1: Tạo Virtual Environment

Mở **Terminal** (PowerShell hoặc CMD):

```bash
# Di chuyển đến thư mục project
cd "Face-Detection-Face-Segmentation"

# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows)
.venv\Scripts\activate
```

### Bước 2: Cài đặt thư viện

```bash
# Cài đặt tất cả dependencies
pip install -r requirements.txt
```

### Bước 3: Chạy Smoke Test

```bash
# Test cả 2 models load được không
python scripts/kaggle/end_to_end_smoke_test.py
```

**Kết quả mong đợi:**
```
[smoke] wrote E:\Face-Detection-Face-Segmentation\runs\evaluation\pipeline_smoke_test.json
[smoke] retinaface: forward ~1.0s, detector run ~0.9s
[smoke] unet: run ~1.1s
verdict: OK
```

---

## 🎓 ĐÁNH GIÁ MODEL (EVALUATION)

### Evaluate Segmentation (U-Net)

```bash
# Evaluate trên 100 ảnh test set
python scripts/evaluation/eval_segmentation.py --split test --max-samples 100

# Evaluate trên validation set
python scripts/evaluation/eval_segmentation.py --split val --max-samples 100
```

**Kết quả thực tế (100 test samples, 2026-09-23):**
- Mean IoU: **0.9660**
- Mean Dice: **0.9824**
- Pixel Accuracy: **0.9756**

### Visualize Results

```bash
# Tạo 8 samples visualization
python scripts/inference/visualize_segmentation.py --num-samples 8

# Output: runs/visualizations/test/sample_*.png + summary.png
```

---

## 🎨 CHẠY INFERENCE / DEMO

### Demo Pipeline

```bash
python scripts/inference/demo.py --help
```

### Visualize Detection

```bash
python scripts/kaggle/end_to_end_smoke_test.py
# Output: runs/visualizations/detection/detection_sample_*.png
```

---

## 📂 CẤU TRÚC PROJECT (2026-09-23)

```
Face-Detection-Face-Segmentation/
├── src/                                 # Source code
│   ├── detection/                       # RetinaFace model
│   ├── segmentation/                    # U-Net model
│   ├── pipeline/                        # End-to-end pipeline
│   ├── data/                            # Datasets & preprocessing
│   └── utils/                           # Common utilities
│
├── models/                              # Trained checkpoints
│   ├── retinaface_final.pth             # 84.6 MB, 22.2M params
│   └── unet_final.pth                   # 118.5 MB, 31.0M params
│
├── scripts/                             # Executable scripts
│   ├── preprocessing/                   # Data processing
│   ├── inference/                       # Demo & viz
│   ├── evaluation/                      # Model evaluation
│   ├── kaggle/                          # End-to-end test
│   └── misc/                            # Utilities
│
├── tests/                               # 67 unit tests
├── data/                                # Datasets (raw + processed)
├── runs/                                # Runtime outputs (evaluations)
├── notebooks/                           # Jupyter notebooks
│
├── docs/                                # Documentation (organized)
│   ├── guides/                          # User guides
│   ├── planning/                        # Plans & roadmap
│   ├── status/                          # Progress logs
│   ├── references/                      # Reports & research
│   ├── adr/                             # Architecture decisions
│   └── audit/                           # Audit reports
│
├── requirements.txt
├── quick_run.bat                        # 🚀 Chạy nhanh
└── README.md
```

---

## ⚙️ YÊU CẦU HỆ THỐNG

| Thành phần | Yêu cầu tối thiểu |
|------------|-------------------|
| Python | 3.10+ |
| RAM | 8GB |
| Ổ cứng | 10GB trống |
| GPU | Không bắt buộc (chạy CPU được, chậm hơn) |

---

## 📊 KẾT QUẢ EVALUATION

### U-Net Segmentation
| Metric | Test Set | Validation Set |
|--------|----------|----------------|
| Mean IoU | **0.9660** | **0.9766** |
| Mean Dice | **0.9824** | **0.9880** |
| Pixel Accuracy | 0.9756 | 0.9834 |

### RetinaFace Detection
- Checkpoint loads ✅ (84.6 MB)
- Forward pass OK (~1s trên CPU)
- Detection confidence score: ~1.3–1.4
- Real WIDER mAP@0.5 metrics: pending

---

## 🐛 XỬ LÝ LỖI THƯỜNG GẶP

### Lỗi: `Module not found`
```bash
# Cài lại thư viện
pip install -r requirements.txt
```

### Lỗi: `FileNotFoundError: models/retinaface_final.pth`
```bash
# Đảm bảo đang ở thư mục gốc project
cd Face-Detection-Face-Segmentation
ls models/  # Phải thấy 2 file .pth
```

### Lỗi: `Permission denied`
```bash
# Chạy CMD với quyền Admin
# Hoặc kiểm tra quyền thư mục
```

---

## 📞 HỖ TRỢ

Nếu gặp lỗi khác, hãy:
1. Kiểm tra `docs/guides/` cho hướng dẫn chi tiết
2. Kiểm tra `docs/references/DANH_GIA_MODEL.md` cho evaluation report
3. Chụp ảnh lỗi và hỏi team

---

**Chúc bạn chạy model thành công! 🎉**  
**Last Updated:** 23/09/2026
