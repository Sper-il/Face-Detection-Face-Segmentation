# 📘 HƯỚNG DẪN CHẠY FACE DETECTION & SEGMENTATION MODEL

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
cd "C:\Users\hoait\OneDrive\Tài liệu\Face-Detection-Face-Segmentation"

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường
.\venv\Scripts\activate
```

### Bước 2: Cài đặt thư viện

```bash
# Cài PyTorch (CPU version - không cần GPU mạnh)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài các thư viện khác
pip install opencv-python-headless tqdm PyYAML numpy pillow tensorboard matplotlib
```

### Bước 3: Chạy Test Model

```bash
# Test model hoạt động chưa
python scripts/kaggle/end_to_end_smoke_test.py
```

**Kết quả mong đợi:**
```
============================================================
   MODEL TESTING SCRIPT
============================================================

TESTING FACE DETECTION MODEL
==================================================
[+] Using device: cpu
[+] Detection model created successfully
[OK] Face Detection Model: PASSED

TESTING FACE SEGMENTATION MODEL
==================================================
[+] Segmentation output shape: torch.Size([1, 2, 224, 224])
[OK] Face Segmentation Model: PASSED

============================================================
   TEST SUMMARY
============================================================
  Detection: OK PASSED
  Segmentation: OK PASSED
============================================================

[OK] ALL TESTS PASSED! Models are ready to train.
```

---

## 🎓 HUẤN LUYỆN MODEL (TRAINING)

### Train Face Detection

```bash
python -m src.training.train_detection
```

### Train Face Segmentation

```bash
python -m src.training.train_segmentation
```

---

## 📂 CẤU TRÚC PROJECT

```
Face-Detection-Face-Segmentation/
├── src/
│   ├── detection/           # Model phát hiện khuôn mặt
│   │   ├── model.py        # DSFD Model
│   │   └── losses.py      # Loss functions
│   ├── segmentation/        # Model phân đoạn khuôn mặt
│   │   ├── model.py        # FCN8s/UNet Model
│   │   └── losses.py      # Loss functions
│   └── training/           # Scripts huấn luyện
│       ├── train_detection.py
│       └── train_segmentation.py
├── configs/                 # File cấu hình
│   ├── detection_config.yaml
│   └── segmentation_config.yaml
├── scripts/                 # Scripts (theo mục đích)
│   ├── preprocessing/        # Tiền xử lý dữ liệu
│   ├── inference/             # Inference và demo
│   ├── evaluation/           # Đánh giá model
│   ├── checkpoints/           # Tiện ích checkpoint
│   ├── kaggle/               # Tích hợp Kaggle
│   ├── diagrams/             # Tạo sơ đồ
│   └── misc/                 # Script khác
├── data/                   # Dữ liệu huấn luyện
│   ├── raw/                # Dữ liệu gốc
│   └── processed/          # Dữ liệu đã xử lý
├── quick_run.bat           # 🚀 Chạy nhanh
└── requirements.txt        # Thư viện cần thiết
```

---

## ⚙️ YÊU CẦU HỆ THỐNG

| Thành phần | Yêu cầu tối thiểu |
|------------|-------------------|
| Python | 3.8+ |
| RAM | 8GB |
| Ổ cứng | 10GB trống |
| GPU | Không bắt buộc (chạy CPU được) |

---

## 🐛 XỬ LÝ LỖI THƯỜNG GẶP

### Lỗi: `Module not found`

```bash
# Cài lại thư viện
pip install -r requirements_training.txt
```

### Lỗi: `CUDA out of memory`

→ Model đang dùng GPU hết bộ nhớ
→ Thử giảm batch_size trong config

### Lỗi: `Permission denied`

```bash
# Chạy CMD với quyền Admin
# Hoặc kiểm tra quyền thư mục
```

---

## 📞 HỖ TRỢ

Nếu gặp lỗi khác, hãy:
1. Chụp ảnh lỗi (screenshot)
2. Gửi cho tôi qua chat

---

**Chúc bạn huấn luyện model thành công! 🎉**
