# Hướng Dẫn Tải Dataset - Tiếng Việt

## Tổng Quan

Hướng dẫn này giúp bạn tải xuống và chuẩn bị dữ liệu cho dự án Face Detection & Segmentation.

---

## Datasets Cần Thiết

### 1. WIDER FACE (Cho Detection)
- **Kích thước**: ~2 GB
- **Số ảnh**: 32,203 ảnh
- **Số khuôn mặt**: 393,703 khuôn mặt đã được gán nhãn
- **Mục đích**: Huấn luyện mô hình phát hiện khuôn mặt
- **Đặc điểm**: Chia theo độ khó Easy/Medium/Hard cho các cảnh đông người

### 2. CelebAMask-HQ (Cho Segmentation)
- **Kích thước**: ~20 GB (bản đầy đủ) hoặc ~2 GB (bản resize)
- **Số ảnh**: 30,000 ảnh 1024x1024 (hoặc 256x256 resize)
- **Mục đích**: Huấn luyện mô hình phân đoạn khuôn mặt
- **Đặc điểm**: Mask phân đoạn chi tiết từng pixel

---

## Bắt Đầu Nhanh

### Bước 1: Tải WIDER FACE (Tự động)

```bash
# Di chuyển đến thư mục project
cd "C:\Users\Admin\Downloads\Face Detection & Face Segmentation"

# Kích hoạt môi trường ảo
venv\Scripts\activate

# Cài đặt thư viện cần thiết
pip install requests tqdm gdown

# Chạy script tải WIDER FACE (đang chạy tự động rồi!)
python scripts\download_wider_face.py
```

**Lưu ý**: Script đang chạy trong background và sẽ:
1. Tải xuống 4 file zip (~2 GB tổng cộng)
2. Tự động giải nén
3. Xác minh cấu trúc thư mục

**Thời gian ước tính**: 30-60 phút tùy tốc độ mạng

### Bước 2: Tải CelebAMask-HQ

**Khuyến nghị**: Bắt đầu với bản resize 256x256 (nhỏ hơn, nhanh hơn)

#### Tùy chọn A: Bản Resize từ Kaggle (Khuyến nghị cho bắt đầu)

1. Truy cập: https://www.kaggle.com/datasets/ashishjangra27/celeba-hq-resized-256x256
2. Đăng nhập Kaggle (hoặc tạo tài khoản miễn phí)
3. Click "Download" (khoảng 2 GB)
4. Giải nén vào: `data/raw/CelebAMask-HQ/`

#### Tùy chọn B: Bản Đầy Đủ từ Google Drive

```bash
# Chạy script tự động (có thể chậm do giới hạn Google Drive)
python scripts\download_celebamask.py
```

**Hoặc tải thủ công**:
1. Truy cập: https://github.com/switchablenorms/CelebAMask-HQ
2. Làm theo hướng dẫn trong README để tải từ Google Drive
3. Đặt vào: `data/raw/CelebAMask-HQ/`

---

## Kiểm Tra Dataset Đã Tải

Sau khi tải xong, chạy script kiểm tra:

```bash
python scripts\verify_datasets.py
```

Script này sẽ:
- ✅ Kiểm tra WIDER FACE (số lượng ảnh train/val/test)
- ✅ Kiểm tra CelebAMask-HQ (số lượng ảnh và mask)
- ✅ Hiển thị dung lượng đĩa sử dụng
- ✅ Tóm tắt trạng thái dataset

---

## Cấu Trúc Thư Mục Sau Khi Tải

```
data/raw/
├── WIDER_FACE/
│   ├── WIDER_train/
│   │   └── images/
│   │       ├── 0--Parade/          (các sự kiện)
│   │       ├── 1--Handshaking/
│   │       └── ... (61 thư mục sự kiện)
│   ├── WIDER_val/
│   │   └── images/
│   ├── WIDER_test/
│   │   └── images/
│   └── wider_face_split/
│       ├── wider_face_train_bbx_gt.txt   (annotations)
│       ├── wider_face_val_bbx_gt.txt
│       └── readme.txt
│
└── CelebAMask-HQ/
    ├── CelebA-HQ-img/              (30,000 ảnh)
    │   ├── 0.jpg
    │   ├── 1.jpg
    │   └── ...
    └── CelebAMask-HQ-mask-anno/    (mask phân đoạn)
        ├── 0/
        ├── 1/
        └── ...
```

---

## Xử Lý Sự Cố

### Vấn đề: Tải xuống quá chậm

**Giải pháp**:
1. Sử dụng download manager (IDM, Free Download Manager)
2. Tải qua đêm
3. Dùng bản resize cho CelebAMask-HQ (nhỏ hơn nhiều)
4. Hỏi bạn bè/đồng nghiệp xem có dataset không

### Vấn đề: Hết quota Google Drive

**Giải pháp**:
1. Đợi 24 giờ và thử lại
2. Tải từ nguồn khác:
   - Kaggle: https://www.kaggle.com/datasets/
   - Hỏi bạn bè chia sẻ
3. Dùng tài khoản Google Drive khác

### Vấn đề: Không đủ dung lượng ổ đĩa

**Giải pháp**:
1. Giải phóng dung lượng (cần ~25-30 GB)
2. Sử dụng ổ cứng ngoài
3. Dùng bản resize 256x256 thay vì 1024x1024
4. Xóa file zip sau khi giải nén để tiết kiệm

### Vấn đề: Giải nén bị lỗi

**Giải pháp**:
1. Tải lại file bị hỏng
2. Dùng 7-Zip thay vì WinRAR hoặc Windows Explorer
3. Kiểm tra MD5/SHA checksum nếu có

---

## Thống Kê Dataset

### WIDER FACE

| Phần | Số Ảnh | Số Mặt | Easy | Medium | Hard |
|------|---------|--------|------|--------|------|
| Train | 12,880 | 159,424 | - | - | - |
| Val | 3,226 | 39,720 | 1,288 | 1,026 | 912 |
| Test | 16,097 | 194,559 | - | - | - |

### CelebAMask-HQ

19 lớp phân đoạn khuôn mặt:
- da (skin), mũi (nose), kính (eye_g), mắt trái/phải (l_eye/r_eye)
- lông mày trái/phải (l_brow/r_brow), tai trái/phải (l_ear/r_ear)
- miệng (mouth), môi trên/dưới (u_lip/l_lip), tóc (hair), mũ (hat)
- cổ (neck), quần áo (cloth)

---

## Dung Lượng Đĩa

| Thành Phần | Kích Thước |
|------------|------------|
| WIDER FACE (zip) | 2 GB |
| WIDER FACE (đã giải nén) | 3 GB |
| CelebAMask-HQ full (zip) | 20 GB |
| CelebAMask-HQ full (giải nén) | 25 GB |
| CelebAMask-HQ resize (zip) | 2 GB |
| CelebAMask-HQ resize (giải nén) | 3 GB |
| **Tổng (giữ zip)** | ~50 GB (full) hoặc ~10 GB (resize) |
| **Tổng (xóa zip)** | ~28 GB (full) hoặc ~6 GB (resize) |

**Khuyến nghị**: Xóa file zip sau khi giải nén thành công để tiết kiệm dung lượng.

---

## Thời Gian Tải Ước Tính

| Tốc Độ Mạng | WIDER FACE | CelebAMask (Full) | CelebAMask (Resize) |
|-------------|------------|-------------------|---------------------|
| 10 Mbps | 30 phút | 5 giờ | 30 phút |
| 50 Mbps | 6 phút | 1 giờ | 6 phút |
| 100 Mbps | 3 phút | 30 phút | 3 phút |

*Lưu ý: Thời gian thực tế có thể khác do tốc độ server và điều kiện mạng*

---

## Các Bước Tiếp Theo

Sau khi tải xong dataset:

### 1. Xác minh dataset
```bash
python scripts\verify_datasets.py
```

### 2. Cập nhật file .env
```bash
# Copy .env.example thành .env
copy .env.example .env

# Chỉnh sửa đường dẫn trong .env nếu cần
```

### 3. Khám phá dữ liệu
```bash
# Tạo notebook để khám phá dữ liệu
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 4. Tạo data loaders
```bash
# Xem Phase 1 trong PROGRESS_STATUS.md
# Tạo file: src/data/dataset.py
# Tạo file: src/data/preprocessing.py
```

### 5. Bắt đầu huấn luyện
```bash
# Chuyển sang Phase 2: Detection model development
# Xem chi tiết trong PROGRESS_STATUS.md
```

---

## Trợ Giúp

Nếu gặp vấn đề:
1. Kiểm tra phần "Xử Lý Sự Cố" ở trên
2. Xem `docs/DATASET_DOWNLOAD_GUIDE.md` (tiếng Anh chi tiết hơn)
3. Kiểm tra `PROGRESS_STATUS.md` cho ghi chú về project
4. Hỏi trong nhóm chat hoặc diễn đàn

---

## Ghi Chú Quan Trọng

### ⚠️ WIDER FACE đang tải xuống
Script `download_wider_face.py` đang chạy trong background. Bạn có thể:
- Tiếp tục làm việc với các file khác
- Kiểm tra tiến độ bằng cách chạy `python scripts\verify_datasets.py`
- Đợi cho đến khi hoàn tất (~30-60 phút)

### 💡 Mẹo
1. **Bắt đầu với bản resize**: Dùng CelebAMask-HQ 256x256 để bắt đầu nhanh
2. **Tải qua đêm**: Nếu mạng chậm, để máy tải qua đêm
3. **Backup**: Sau khi tải xong, nên backup lên ổ cứng ngoài
4. **Chia sẻ**: Nếu có nhiều người làm cùng project, chia sẻ dataset cho nhau

---

**Cập nhật lần cuối**: 2026-09-03  
**Người duy trì**: Project Team
