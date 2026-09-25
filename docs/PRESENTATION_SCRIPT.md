# 🎤 Kịch Bản Trình Bày Folder Project — 10 Phút

> **Project:** Face Detection & Face Segmentation
> **Thời lượng:** 10 phút
> **Hình thức:** Trình bày folder project trực tiếp (không slide)
> **Ngày:** 2026-09-25

---

## ⏰ Tổng Quan Timing

| Phần | Thời gian | Folder cần mở |
|------|-----------|---------------|
| 1. Mở đầu — Giới thiệu tổng quan | 1:00 | Folder gốc |
| 2. Cấu trúc project | 1:30 | `src/`, `models/`, `data/` |
| 3. Stage 1 — Detection (RetinaFace) | 2:00 | `src/detection/` |
| 4. Stage 2 — Segmentation (U-Net) | 2:00 | `src/segmentation/` |
| 5. Pipeline & Utils | 1:00 | `src/pipeline/`, `src/utils/` |
| 6. Kết quả & Đánh giá | 1:30 | `runs/` |
| 7. Tests & Documentation | 0:30 | `tests/`, `docs/` |
| 8. Kết luận & Q&A | 0:30 | Folder gốc |

---

## 🎬 PHẦN 1: Mở Đầu (1:00)

### Folder đang mở: `Face-Detection-Face-Segmentation/` (folder gốc)

### Lời nói:

> **[Mở folder gốc trong VS Code Explorer]**
>
> "Xin chào mọi người. Project tôi trình bày hôm nay là **Face Detection & Face Segmentation** — một hệ thống AI 2 giai đoạn có khả năng vừa phát hiện vị trí khuôn mặt, vừa tách vùng khuôn mặt từ ảnh."
>
> **[Chỉ folder gốc trong Explorer]**
>
> "Đây là folder project. Khi các bạn nhìn vào đây, các bạn sẽ thấy toàn bộ những gì tôi đã xây dựng: source code, models đã train, data, kết quả evaluation, tests và documentation."

> **[Chỉ vào từng folder lớn một lượt — overview]**
>
> "Tổng quan có 6 folder chính:"
> "- `src/` — toàn bộ source code"
> "- `models/` — 2 checkpoints đã train"
> "- `data/` — dataset đã xử lý"
> "- `runs/` — kết quả evaluation và visualization"
> "- `tests/` — 67 unit tests"
> "- `docs/` — documentation"

### Lưu ý khi nói:
- ❗ Mở sẵn VS Code Explorer panel bên trái
- ❗ Chỉ folder, đừng mở file
- ❗ Nói chậm, rõ ràng

---

## 📁 PHẦN 2: Cấu Trúc Project (1:30)

### Folder đang mở: `src/`, `models/`, `data/`

### Lời nói:

> **[Click vào folder `src/`, mở rộng]**
>
> "Folder `src/` chứa toàn bộ source code, được tổ chức theo module:"
>
> ```
> src/
> ├── detection/       # Stage 1: RetinaFace
> ├── segmentation/    # Stage 2: U-Net
> ├── pipeline/        # Orchestrator
> ├── utils/           # Box/mask/NMS/I-O helpers
> └── eval.py          # Main evaluation entrypoint
> ```

> **[Click vào `models/`]**
>
> "Folder `models/` chứa 2 file checkpoint đã train:"
> "- `retinaface_final.pth` — 84.6 MB, cho Stage 1"
> "- `unet_final.pth` — 118.5 MB, cho Stage 2"
>
> "Tổng cộng khoảng 200 MB models."

> **[Click vào `data/`]**
>
> "Folder `data/` chứa dataset đã được xử lý sẵn:"
> "- `processed/wider_face/` — WIDER FACE annotations cho detection"
> "- `processed/celeba_mask/` — CelebAMask-HQ cho segmentation"
>
> "Mỗi dataset đã được split sẵn train/val/test."

### Ưu điểm khi hỏi "Tại sao tổ chức như vậy?":
> "Tách module theo chức năng — detection riêng, segmentation riêng — để dễ maintain. Mỗi module có thể chạy độc lập hoặc kết hợp qua pipeline."

---

## 🧠 PHẦN 3: Stage 1 — RetinaFace (2:00)

### Folder đang mở: `src/detection/`

### Lời nói:

> **[Mở folder `src/detection/`, chỉ từng file]**
>
> "Bây giờ vào chi tiết Stage 1 — Detection. Folder `src/detection/` có các file sau:"
>
> ```
> src/detection/
> ├── model.py           # Kiến trúc RetinaFace
> ├── weights.py         # Load/save checkpoint
> ├── boxes.py           # Decode + NMS
> └── infer.py           # Inference cho 1 ảnh
> ```

> **[Mở file `model.py`]**
>
> "File `model.py` định nghĩa kiến trúc RetinaFace:"
> "- Backbone ResNet-34"
> "- FPN — Feature Pyramid Network — 3 levels"
> "- SSH — Single Stage Headless — context module"
> "- 3 multi-task heads: classification, box regression, landmark"

> **[Cuộn xuống, chỉ class chính]**
>
> "Class chính là `RetinaFace` — input là ảnh 640×640, output là classification logits, box deltas, và landmark deltas ở 3 FPN levels."

> **[Mở file `weights.py`]**
>
> "File `weights.py` xử lý load/save checkpoint. Đây là phần quan trọng vì checkpoint phải load đúng key mới chạy được."

> **[Mở file `boxes.py`]**
>
> "File `boxes.py` làm 2 việc:"
> "- **Decode** — chuyển box deltas thành tọa độ thật"
> "- **NMS** — Non-Maximum Suppression để loại box trùng"

> **[Mở file `infer.py`]**
>
> "File `infer.py` là entrypoint chạy detection cho 1 ảnh, trả về list bounding boxes + scores."

### Số liệu:
> "Toàn bộ model có **22.1 triệu tham số**. Trong smoke test, forward pass mất **0.56 giây** trên CPU, top confidence score đạt **1.42**."

### Timing:
- 0:00 → 0:30 — Liệt kê file trong folder
- 0:30 → 1:00 — Mở `model.py`, giải thích kiến trúc
- 1:00 → 1:30 — Mở `weights.py` + `boxes.py`
- 1:30 → 2:00 — Mở `infer.py` + số liệu

---

## 🎨 PHẦN 4: Stage 2 — U-Net (2:00)

### Folder đang mở: `src/segmentation/`

### Lời nói:

> **[Mở folder `src/segmentation/`, chỉ từng file]**
>
> "Sang Stage 2 — Segmentation. Folder `src/segmentation/` có cấu trúc tương tự:"
>
> ```
> src/segmentation/
> ├── model.py           # Kiến trúc U-Net
> ├── weights.py         # Load/save checkpoint
> ├── losses.py          # BCE + Dice loss
> ├── dataset.py         # DataLoader cho CelebAMask-HQ
> ├── train.py           # Training loop
> └── infer.py           # Inference cho 1 face crop
> ```

> **[Mở file `model.py`]**
>
> "File `model.py` định nghĩa kiến trúc U-Net — encoder-decoder với skip connections:"
> "- Encoder: 4 DoubleConv blocks, downsampling bằng MaxPool"
> "- Bottleneck: 1024 channels"
> "- Decoder: 4 upsampling blocks, mỗi block nối với skip từ encoder"
> "- Output: 2 channels (background + face)"

> **[Mở file `losses.py`]**
>
> "File `losses.py` định nghĩa combined loss: **0.5 × BCE + 0.5 × Dice**."
> "- BCE ổn định gradient"
> "- Dice xử lý tốt class imbalance"

> **[Mở file `dataset.py`]**
>
> "File `dataset.py` chứa DataLoader cho CelebAMask-HQ — đọc ảnh + mask, áp dụng augmentation như flip, color jitter."

> **[Mở file `train.py`]**
>
> "File `train.py` là training loop: Adam optimizer, lr=1e-3, batch size=16, 30 epochs, training trên CPU."

> **[Mở file `infer.py`]**
>
> "File `infer.py` là inference cho 1 face crop — input là ảnh 256×256, output là binary mask."

### Số liệu:
> "U-Net có **31 triệu tham số**, checkpoint **118.5 MB**. Input size 256×256."

### Timing:
- 0:00 → 0:30 — Liệt kê file trong folder
- 0:30 → 1:00 — Mở `model.py`, giải thích U-Net
- 1:00 → 1:30 — Mở `losses.py` + `dataset.py`
- 1:30 → 2:00 — Mở `train.py` + `infer.py` + số liệu

---

## 🔗 PHẦN 5: Pipeline & Utils (1:00)

### Folder đang mở: `src/pipeline/`, `src/utils/`

### Lời nói:

> **[Mở folder `src/pipeline/`]**
>
> "Folder `src/pipeline/` chứa orchestrator — file chính kết nối 2 stages:"
> "- `orchestrator.py` — class `FaceSegmentationPipeline` chạy end-to-end"
> "- Khi gọi `pipeline.run('image.jpg')` → chạy RetinaFace → cắt face crops → chạy U-Net → render overlay"

> **[Mở folder `src/utils/`]**
>
> "Folder `src/utils/` chứa các helper:"
> "- `box_utils.py` — IoU computation, box transforms"
> "- `mask_utils.py` — mask resize, overlay rendering"
> "- `io_utils.py` — load/save JSON, PNG"
> "- `visualization.py` — vẽ bbox, mask, summary grid"

> **[Mở file `eval.py` ở root `src/`]**
>
> "Cuối cùng là file `src/eval.py` — main entrypoint để chạy evaluation. File này load 2 models, chạy trên test set, in metrics."

### Timing:
- 0:00 → 0:30 — Pipeline
- 0:30 → 1:00 — Utils + eval entrypoint

---

## 📊 PHẦN 6: Kết Quả & Đánh Giá (1:30)

### Folder đang mở: `runs/`

### Lời nói:

> **[Mở folder `runs/`, chỉ các sub-folder]**
>
> "Bây giờ vào phần quan trọng nhất — kết quả thực tế. Folder `runs/` chứa toàn bộ output của project:"
>
> ```
> runs/
> ├── evaluation/      # Metrics JSON
> └── visualizations/  # 8 sample PNGs
> ```

> **[Mở folder `runs/evaluation/`, chỉ từng file JSON]**
>
> "Folder `evaluation/` có các file JSON chứa metrics:"
> "- `segmentation_test_metrics.json` — kết quả trên test set"
> "- `segmentation_val_metrics.json` — kết quả trên val set"
> "- `pipeline_smoke_test.json` — kết quả smoke test pipeline"

> **[Mở file `segmentation_test_metrics.json`]**
>
> "Đây là kết quả trên test set — 100 samples:"
>
> | Metric | Value | Target |
> |--------|-------|--------|
> | **Mean IoU** | **0.9679** | ≥ 0.90 ✅ |
> | **Mean Dice** | **0.9834** | ≥ 0.95 ✅ |
> | **Pixel Accuracy** | **0.9770** | ≥ 0.97 ✅ |

> **[Nhấn mạnh]**
>
> "IoU đạt **96.79%** — vượt target 90% tận 7.8 điểm. Dice **98.34%** cũng vượt target 95%."
>
> "Trên validation set, IoU là **0.9666** — gần như tương đương test set, chứng tỏ model không bị overfitting."

> **[Mở file `pipeline_smoke_test.json`]**
>
> "File `pipeline_smoke_test.json` cho thấy:"
> "- Verdict: **OK**"
> "- RetinaFace forward: **0.56s**"
> "- Top confidence: **1.42**"
>
> "Tức là end-to-end pipeline chạy ổn định."

> **[Mở folder `runs/visualizations/test/`, chỉ các PNG]**
>
> "Folder `visualizations/test/` có 8 ảnh PNG visualization. Mỗi ảnh có 4 panel:"
> "- Ảnh gốc"
> "- Mask dự đoán"
> "- Overlay mask đỏ"
> "- Ground truth"
>
> **[Mở 1 ảnh PNG bất kỳ]**
>
> "Đây là visualization thực tế. Mask dự đoán gần như khớp hoàn toàn với ground truth."

### Timing:
- 0:00 → 0:30 — Liệt kê folder `runs/`
- 0:30 → 0:50 — Show metrics test set
- 0:50 → 1:10 — Validation + smoke test
- 1:10 → 1:30 — Demo visualization

---

## 🧪 PHẦN 7: Tests & Documentation (0:30)

### Folder đang mở: `tests/`, `docs/`

### Lời nói:

> **[Mở folder `tests/`, chỉ các file test]**
>
> "Folder `tests/` chứa **67 unit tests** — tất cả đều passing. Test cover:"
> "- Model architecture (RetinaFace + U-Net)"
> "- Box decode + NMS"
> "- Mask operations"
> "- Pipeline end-to-end"
> "- I/O helpers"

> **[Mở folder `docs/`, chỉ các file chính]**
>
> "Folder `docs/` chứa documentation:"
> "- `PRESENTATION_SCRIPT.md` — kịch bản trình bày này"
> "- `progress_status.md` — status hiện tại của project"
> "- `ROADMAP.md` — lộ trình phát triển"
> "- `ARCHITECTURE.md` ở root — chi tiết kiến trúc"
>
> "Tổng cộng khoảng 13 markdown files."

### Timing:
- 0:00 → 0:15 — Tests
- 0:15 → 0:30 — Documentation

---

## 🎓 PHẦN 8: Kết Luận & Q&A (0:30)

### Folder đang mở: Folder gốc `Face-Detection-Face-Segmentation/`

### Lời nói:

> **[Quay lại folder gốc]**
>
> "Tóm lại, những gì project đã làm được:"
> "- ✅ Xây dựng pipeline 2-stage: RetinaFace detection + U-Net segmentation"
> "- ✅ Train thành công cả 2 models trên CPU"
> "- ✅ U-Net đạt IoU **96.79%**, vượt target"
> "- ✅ 67 unit tests passing"
> "- ✅ Full documentation + visualizations"

> **[Đóng các file đang mở, show folder gốc sạch sẽ]**
>
> "Tổng project có khoảng 5,000 dòng Python code, 2 trained models (~200MB), và 13 markdown docs."
>
> "Cảm ơn mọi người đã lắng nghe. Tôi sẵn sàng trả lời câu hỏi."

### Q&A thường gặp:

**Q: Tại sao chọn U-Net thay vì DeepLab?**
> "U-Net đơn giản hơn, train nhanh hơn, đủ tốt cho binary face segmentation. CelebAMask-HQ có face chiếm phần lớn ảnh nên U-Net là lựa chọn optimal."

**Q: Có cần GPU không?**
> "Không — inference chạy OK trên CPU, khoảng 1.2 giây cho mỗi ảnh có 1 mặt."

**Q: Tại sao IoU cao vậy?**
> "Dataset chất lượng cao + U-Net skip connections + combined BCE+Dice loss — tất cả kết hợp lại cho kết quả tốt."

### Timing:
- 0:00 → 0:15 — Tổng kết những gì đã làm
- 0:15 → 0:30 — Câu kết + Q&A

---

## 📌 Tips Trình Bày Với Folder

### 🖥️ Chuẩn bị trước:
1. ✅ Mở VS Code với folder project
2. ✅ Mở sẵn Explorer panel bên trái
3. ✅ Tắt các file không liên quan
4. ✅ Zoom terminal/editor vừa phải để khán giả thấy
5. ✅ Mở sẵn file `segmentation_test_metrics.json` trong editor

### 🎤 Kỹ năng trình bày:
- ❗ **Luôn chỉ folder/file** khi nói về nó
- ❗ **Click mở file** để khán giả thấy nội dung thật
- ❗ **Cuộn chậm** khi show code
- ❗ **Đóng file** sau khi xong để Explorer sạch sẽ
- ❗ **Nói số liệu** dựa trên file JSON thật, không nhớ

### ⏰ Quản lý thời gian:
- Tổng 10 phút — đeo đồng hồ
- Mỗi phần có timing rõ — đừng nói quá lâu ở 1 folder
- Nếu hết giờ ở phần 4 → skip phần 7, vào Q&A

### 💬 Câu mở đầu dự phòng:
> "Đây là folder project của tôi. Khi các bạn nhìn vào Explorer bên trái, các bạn sẽ thấy tất cả những gì tôi đã làm trong vài tháng qua."

### 🎤 Câu kết dự phòng:
> "Tổng cộng project có 6 folder chính, 2 models đã train, 67 tests passing, và IoU đạt 96.79%. Cảm ơn mọi người."

---

## 🗂️ Checklist Folder Cần Mở Theo Thứ Tự

```
Phase 1 (0:00 → 1:00)   → Folder gốc (overview)
Phase 2 (1:00 → 2:30)   → src/, models/, data/
Phase 3 (2:30 → 4:30)   → src/detection/
Phase 4 (4:30 → 6:30)   → src/segmentation/
Phase 5 (6:30 → 7:30)   → src/pipeline/, src/utils/
Phase 6 (7:30 → 9:00)   → runs/evaluation/, runs/visualizations/
Phase 7 (9:00 → 9:30)   → tests/, docs/
Phase 8 (9:30 → 10:00)  → Folder gốc (kết luận + Q&A)
```

---

## 🔗 File Tham Khảo Khi Bị Hỏi Sâu

| Câu hỏi | Mở file |
|---------|---------|
| Kiến trúc RetinaFace chi tiết | `src/detection/model.py` |
| Kiến trúc U-Net chi tiết | `src/segmentation/model.py` |
| Loss function | `src/segmentation/losses.py` |
| Metrics gốc | `runs/evaluation/segmentation_test_metrics.json` |
| Pipeline code | `src/pipeline/orchestrator.py` |
| Visualizations | `runs/visualizations/test/` |
| Status project | `docs/status/progress_status.md` |
| Lộ trình | `docs/planning/ROADMAP.md` |
