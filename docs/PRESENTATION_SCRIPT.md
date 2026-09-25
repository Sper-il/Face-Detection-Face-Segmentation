# 🎤 Kịch Bản Trình Bày Project — 10 Phút

> **Project:** Face Detection & Face Segmentation
> **Thời lượng:** 10 phút
> **Ngày:** 2026-09-25

---

## ⏰ Tổng Quan Timing

| Phần | Thời gian | Từ phút |
|------|-----------|---------|
| 1. Mở đầu & Hook | 1:00 | 0:00 → 1:00 |
| 2. Mục tiêu & Phạm vi | 0:30 | 1:00 → 1:30 |
| 3. Pipeline tổng quan | 1:30 | 1:30 → 3:00 |
| 4. Stage 1 — RetinaFace | 2:00 | 3:00 → 5:00 |
| 5. Stage 2 — U-Net | 2:00 | 5:00 → 7:00 |
| 6. Kết quả Evaluation | 1:30 | 7:00 → 8:30 |
| 7. Demo & Kết luận | 1:00 | 8:30 → 9:30 |
| 8. Q&A buffer | 0:30 | 9:30 → 10:00 |

---

## 🎬 PHẦN 1: Mở Đầu & Hook (1:00)

### Lời nói:

> **[Bắt đầu — dứt khoát, tự tin]**
>
> "Xin chào mọi người. Trước khi vào phần chính, tôi muốn hỏi một câu: **Các bạn đã bao giờ thắc mắc filter trên Instagram hoạt động như thế nào chưa?**"

> **[Đợi 1-2 giây]**
>
> "Đằng sau nó là hệ thống 2 giai đoạn mà tôi sẽ trình bày hôm nay — **Face Detection & Face Segmentation**."

> **[Chuyển slide / chỉ sơ đồ]**
>
> "Project này xây dựng pipeline end-to-end: cho 1 ảnh vào → hệ thống sẽ tự động tìm tất cả khuôn mặt bằng **bounding box**, sau đó tách chính xác **vùng da khuôn mặt** ở mức pixel."

> **[Liệt kê nhanh ứng dụng]**
>
> "Ứng dụng thực tế rất nhiều: beauty camera, AR filter, camera an ninh, makeup try-on, age estimation…"

### Lưu ý khi nói:
- ❗ **Nói chậm** ở câu đầu tiên để gây chú ý
- ❗ **Nhìn khán giả**, không đọc slide
- ❗ **Dùng ký hiệu**, đừng đọc bullet points

---

## 🎯 PHẦN 2: Mục Tiêu & Phạm Vi (0:30)

### Lời nói:

> **[Slide bảng mục tiêu]**
>
> "Mục tiêu của project gồm 3 điểm chính:"
>
> "**Thứ nhất** — xây dựng pipeline end-to-end: ảnh vào, overlay ra."
>
> "**Thứ hai** — kết hợp cả detection lẫn segmentation: detection để tìm vị trí, segmentation để tách vùng."
>
> "**Thứ ba** — đạt hiệu năng thực tế: target IoU trên 90%."

> **[Nhấn mạnh phạm vi]**
>
> "Về phạm vi, tôi chỉ tập trung vào **face detection + segmentation** trên ảnh tĩnh. Tôi không làm face recognition — nghĩa là không nhận diện danh tính, chỉ tìm vị trí và tách vùng thôi."

### Timing:
- 0:00 → 0:20 — Liệt kê 3 mục tiêu (mỗi cái ~5s)
- 0:20 → 0:30 — Nói rõ phạm vi

---

## 🔗 PHẦN 3: Pipeline Tổng Quan (1:30)

### Lời nói:

> **[Vẽ/chỉ sơ đồ pipeline]**
>
> "Đây là pipeline tổng quan của hệ thống. Nó gồm 2 giai đoạn cascade:"
>
> ```
> Input → Stage 1 → bboxes → Stage 2 → masks → Overlay
> ```

> **[Giải thích Stage 1]**
>
> "**Stage 1 — RetinaFace** — nhận ảnh đầu vào 640×640, đầu ra là N bounding boxes cùng confidence scores. Mỗi box là một khuôn mặt."
>
> "RetinaFace là paper từ CVPR 2020, dùng ResNet-34 làm backbone kết hợp FPN — Feature Pyramid Network — để phát hiện mặt ở nhiều scale."

> **[Giải thích Stage 2]**
>
> "**Stage 2 — U-Net** — nhận mỗi face crop đã được cắt từ bounding box (cộng thêm margin 10%), resize về 256×256, rồi tách ra binary mask — tức là mask 0/1 cho biết pixel nào thuộc mặt, pixel nào thuộc background."
>
> "U-Net là paper từ MICCAI 2015, vốn nổi tiếng trong medical imaging, nhưng áp dụng cho face segmentation cũng rất hiệu quả nhờ skip connections giữ lại spatial details."

> **[Giải thích Post-processing]**
>
> "Sau cùng, hệ thống render overlay đỏ lên mặt, vẽ bounding box, và lưu kèm file JSON chứa thông tin chi tiết."

### Ưu điểm khi hỏi "Tại sao 2-stage?":
> "Vì segmentation-only sẽ phải post-process tìm connected components để biết có bao nhiêu mặt — rất phức tạp và kém chính xác. Cascade giúp tách bạch 2 bài toán."

---

## 🧠 PHẦN 4: Stage 1 — RetinaFace (2:00)

### Lời nói:

> **[Chỉ sơ đồ kiến trúc]**
>
> "Bây giờ tôi đi vào chi tiết Stage 1. RetinaFace có kiến trúc 4 phần:"

> ```
> Input (3, 640, 640)
>   ↓
> ResNet-34 Backbone → c3, c4, c5
>   ↓
> FPN (Feature Pyramid) → p3, p4, p5 (3 scales)
>   ↓
> SSH Context Module (per FPN level)
>   ↓
> 3 Multi-task Heads:
>   ├── Classification: 12 ch (bg + face × 3 anchors × 2)
>   ├── Box regression: 24 ch (4 coords × 3 anchors × 2)
>   └── Landmark:       60 ch (10 coords × 3 anchors × 2)
> ```

> **[Giải thích Backbone]**
>
> "Đầu tiên là backbone ResNet-34 — 4 stages, mỗi stage giảm spatial resolution một nửa. Output gồm 3 feature maps ở stride 8, 16, 32."

> **[Giải thích FPN]**
>
> "FPN lấy 3 feature maps đó, tạo ra pyramid 3 levels để phát hiện mặt ở nhiều kích thước — mặt nhỏ ở p3, mặt lớn ở p5."

> **[Giải thích Multi-task]**
>
> "Mỗi FPN level có 3 heads chạy song song:"
> "- **Classification** phân loại face/background"
> "- **Box regression** dự đoán tọa độ bounding box"
> "- **Landmark** dự đoán 5 điểm landmark (mắt, mũi, miệng)"

> **[Số liệu]**
>
> "Toàn bộ model có **22.1 triệu tham số**, checkpoint nặng **84.6 MB**. Trong smoke test, forward pass mất **0.56 giây** trên CPU, và top confidence score đạt **1.42** — cho thấy model rất confident với prediction."

### Timing:
- 0:00 → 0:30 — Giải thích sơ đồ tổng thể
- 0:30 → 1:30 — Đi chi tiết từng phần (Backbone + FPN + Heads)
- 1:30 → 2:00 — Đọc số liệu

### ⚠️ Lưu ý:
- ❗ **Không nói quá sâu** về SSH, FPN internal — high-level là đủ
- ❗ Nếu ai hỏi sâu → chỉ vào `ARCHITECTURE.md`

---

## 🎨 PHẦN 5: Stage 2 — U-Net (2:00)

### Lời nói:

> **[Chỉ sơ đồ U-Net]**
>
> "Sang Stage 2 — U-Net. Kiến trúc này có dạng chữ U đối xứng:"
>
> ```
> ENCODER (downsampling)        DECODER (upsampling + skip)
> 3   → 64   (enc1)             up4: 1024 → 512
> 64  → 128  (enc2)             dec4: 1024 → 512  (skip từ enc4)
> 128 → 256  (enc3)             up3: 512 → 256
> 256 → 512  (enc4)             dec3: 512 → 256   (skip từ enc3)
> 512 → 1024 (bottleneck)       up2: 256 → 128
>                                dec2: 256 → 128   (skip từ enc2)
>                                up1: 128 → 64
>                                dec1: 128 → 64    (skip từ enc1)
> Output: 2 channels (bg + face logits)
> ```

> **[Giải thích Encoder]**
>
> "Encoder phía trái gồm 4 DoubleConv blocks, mỗi block có 2 lớp Conv-BN-ReLU 3×3, xen giữa là MaxPool 2×2 để giảm resolution."

> **[Giải thích Skip connections]**
>
> "Điểm đặc biệt của U-Net là **skip connections** — feature map từ encoder được nối thẳng vào decoder cùng level. Điều này cực kỳ quan trọng vì giúp decoder khôi phục spatial details mà encoder đã mất khi downsampling."

> **[Giải thích Output]**
>
> "Output cuối cùng là 2 channels: background và face. Tôi dùng argmax để ra binary mask 0/1."

> **[Loss function]**
>
> "Training dùng combined loss: 0.5×BCE + 0.5×Dice. BCE ổn định gradient, Dice xử lý tốt class imbalance."

> **[Số liệu]**
>
> "Model có **31 triệu tham số**, checkpoint **118.5 MB**."

### Timing:
- 0:00 → 0:30 — Giải thích sơ đồ U-Net
- 0:30 → 1:00 — Nói về DoubleConv block
- 1:00 → 1:30 — Nhấn mạnh skip connections
- 1:30 → 2:00 — Loss + số liệu

---

## 📊 PHẦN 6: Kết Quả Evaluation (1:30)

### Lời nói:

> **[Hiển thị bảng metrics]**
>
> "Bây giờ là phần quan trọng nhất — kết quả evaluation. Tôi chạy trên 100 samples từ test set:"
>
> | Metric | Value | Target |
> |--------|-------|--------|
> | **Mean IoU** | **0.9679** | ≥ 0.90 ✅ |
> | **Mean Dice** | **0.9834** | ≥ 0.95 ✅ |
> | **Pixel Accuracy** | **0.9770** | ≥ 0.97 ✅ |
> | F1 (face) | 0.9834 | — |
> | Precision | 0.9828 | — |
> | Recall | 0.9845 | — |

> **[Nhấn mạnh]**
>
> "IoU đạt **96.79%** — vượt target 90% tận 7.8 điểm phần trăm. Dice **98.34%** cũng vượt target 95%. Pixel Accuracy **97.70%** vượt 97%."
>
> "Điều này có nghĩa: model phân biệt được pixel thuộc mặt vs background với độ chính xác rất cao — gần như không có noise."

> **[Show validation set]**
>
> "Trên validation set, kết quả cũng tương đương:"
> "- IoU: **0.9666**"
> "- Dice: **0.9826**"
>
> "Chứng tỏ model không bị overfitting — generalize tốt."

> **[Kết quả Detection]**
>
> "Về phía RetinaFace, smoke test cho thấy forward pass ổn định ở **0.56 giây**, top confidence score **1.42** — model rất tự tin với prediction."

### Timing:
- 0:00 → 0:30 — Show bảng test set metrics
- 0:30 → 0:50 — Nhấn mạnh IoU vượt target
- 0:50 → 1:10 — So sánh val set
- 1:10 → 1:30 — Detection smoke test

---

## 🖼️ PHẦN 7: Demo & Kết Luận (1:00)

### Lời nói:

> **[Mở file visualization]**
>
> "Để các bạn thấy trực quan, đây là visualization từ test set:"
>
> **[Mở 1 ảnh bất kỳ trong `runs/visualizations/test/`]**
>
> "Ảnh này có 4 panel: ảnh gốc, mask dự đoán, overlay mask đỏ, và ground truth."
>
> "Như các bạn thấy, **mask dự đoán gần như khớp hoàn toàn với ground truth** — đó là lý do IoU=96.79%."

> **[Mở summary.png nếu có]**
>
> "Đây là bảng tổng hợp 8 samples. Đa số mask dự đoán rất sát với ground truth."

### Tổng kết:

> **[Quay lại nhìn khán giả]**
>
> "Tóm lại, project đã đạt được:"
> "- ✅ Pipeline end-to-end hoàn chỉnh"
> "- ✅ U-Net IoU=96.79%, vượt target"
> "- ✅ RetinaFace checkpoint loads + forward OK"
> "- ✅ Full evaluation suite + visualizations"
> "- ✅ 67 unit tests passing"

> **[Hướng phát triển]**
>
> "Hướng phát triển tiếp theo: chạy full WIDER FACE mAP metrics, GPU inference để speed up 5-10 lần, và ONNX export cho production deployment."

> **[Câu kết — dứt khoát]**
>
> "Cảm ơn mọi người đã lắng nghe. Tôi sẵn sàng trả lời câu hỏi."

### Timing:
- 0:00 → 0:30 — Demo visualization
- 0:30 → 0:45 — Tổng kết đạt được
- 0:45 → 0:55 — Hướng phát triển
- 0:55 → 1:00 — Câu kết

---

## 🙋 PHẦN 8: Q&A Buffer (0:30)

### Câu hỏi thường gặp — đã chuẩn bị sẵn:

#### Q1: "Tại sao chọn U-Net thay vì DeepLab hay SegFormer?"

**Trả lời:**
> "U-Net đơn giản hơn, train nhanh hơn, và đủ tốt cho binary face segmentation. DeepLab với atrous convolution phù hợp multi-class hơn. SegFormer là transformer-based mới hơn nhưng tốn nhiều compute hơn. Với bài toán 2-class và dataset CelebAMask-HQ có face chiếm phần lớn ảnh, U-Net là lựa chọn optimal."

#### Q2: "Có cần GPU không?"

**Trả lời:**
> "Inference chạy OK trên CPU — khoảng 1.2 giây cho mỗi ảnh có 1 mặt. Training thì có GPU sẽ nhanh hơn 5-10 lần, nhưng tôi đã train thành công trên CPU với thời gian chấp nhận được."

#### Q3: "Tại sao IoU cao vậy — 96.79%?"

**Trả lời:**
> "CelebAMask-HQ là dataset chất lượng cao, face chiếm phần lớn ảnh nên dễ segment. U-Net với skip connections rất phù hợp cho bài toán có contrast rõ giữa foreground và background. Ngoài ra combined loss BCE+Dice giúp ổn định gradient và xử lý tốt edge cases."

#### Q4: "Có thể detect nhiều mặt trong 1 ảnh không?"

**Trả lời:**
> "Có — RetinaFace detect được nhiều bounding boxes trong 1 ảnh. Mỗi bbox sẽ được segment riêng bằng U-Net. Pipeline đã hỗ trợ xử lý N faces tuần tự, tối đa 50 faces theo config."

#### Q5: "So với face recognition thì khác gì?"

**Trả lời:**
> "Face recognition là bài toán khác — cần embedding vector để so sánh danh tính. Project này chỉ dừng ở detection + segmentation: biết **ở đâu có mặt** và **đâu là vùng mặt**, không biết **mặt của ai**. Để làm recognition, cần thêm ArcFace hoặc FaceNet — bước tiếp theo trong pipeline face analysis."

---

## 📌 Tips Trình Bày

### ⏰ Quản lý thời gian:
- **Tổng:** 10:00 phút
- **Mỗi phần** có timing rõ ràng — đeo đồng hồ
- **Nếu hết giờ ở phần 5** → skip phần 6-7, vào thẳng Q&A
- **Nếu thừa giờ** → mở rộng Q&A hoặc demo thêm ảnh

### 🎨 Visual aids cần chuẩn bị:
1. ✅ **Slide PowerPoint/Keynote** với sơ đồ pipeline (hoặc dùng markdown render)
2. ✅ **Sơ đồ kiến trúc** RetinaFace + U-Net
3. ✅ **Bảng metrics** test/val
4. ✅ **8 visualization PNGs** trong `runs/visualizations/test/`
5. ✅ **`summary.png`** tổng hợp

### 🎤 Kỹ năng nói:
- ❗ **Đừng đọc slide** — chỉ đọc số liệu
- ❗ **Nói chậm** ở phần Pipeline (1:30) — quan trọng nhất
- ❗ **Nhìn khán giả** khi nói kết quả
- ❗ **Dùng ngón tay chỉ sơ đồ** khi giải thích
- ❗ **Câu kết ngắn gọn**, dứt khoát

### 💬 Câu mở đầu dự phòng:
> "Project này được tạo ra trong bối cảnh AI đang bùng nổ, đặc biệt là computer vision. Tôi muốn thử xây dựng một hệ thống thực tế — không chỉ trên paper — mà chạy được end-to-end trên máy thường."

### 🎤 Câu kết dự phòng:
> "Đây là project cá nhân hoàn thiện trong thời gian ngắn, nhưng kết quả cho thấy: với kiến trúc đúng và data tốt, có thể đạt IoU gần 97% — đủ dùng cho nhiều ứng dụng thực tế."

---

## 🔗 Tài Liệu Tham Khảo Khi Bị Hỏi Sâu

| Câu hỏi | File tham khảo |
|---------|----------------|
| RetinaFace kiến trúc chi tiết | [ARCHITECTURE.md](../../ARCHITECTURE.md) dòng 80-150 |
| U-Net kiến trúc chi tiết | [ARCHITECTURE.md](../../ARCHITECTURE.md) dòng 160-220 |
| Pipeline code | `src/pipeline/orchestrator.py` |
| Metrics JSON | `runs/evaluation/segmentation_test_metrics.json` |
| Dataset info | `docs/data/DATA_README.md` |
| Status hiện tại | `docs/status/progress_status.md` |
| Lộ trình | `docs/planning/ROADMAP.md` |
