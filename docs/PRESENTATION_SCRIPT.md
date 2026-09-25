# 🎤 Kịch Bản Trình Bày — 10 Phút

> **Project:** Face Detection & Face Segmentation
> **Thời lượng:** 10 phút
> **Ngày:** 2026-09-25
> **Hình thức:** Nói trực tiếp, không cần slide PowerPoint

---

## 📋 Cấu Trúc 10 Phút

| Phần | Thời gian |
|------|-----------|
| 1. Mở đầu & Giới thiệu | 1:00 |
| 2. Pipeline tổng quan | 1:30 |
| 3. Stage 1: RetinaFace | 2:00 |
| 4. Stage 2: U-Net | 2:00 |
| 5. Dữ liệu & Huấn luyện | 1:00 |
| 6. Kết quả | 1:30 |
| 7. Demo & Kết luận | 1:00 |
| **Tổng** | **10:00** |

---

## 🗣️ Phần 1: Mở Đầu & Giới Thiệu (1:00)

> Xin chào mọi người. Hôm nay tôi xin trình bày project **Face Detection & Face Segmentation** — một hệ thống 2 giai đoạn có khả năng vừa **phát hiện vị trí khuôn mặt**, vừa **tách chính xác vùng da khuôn mặt** từ ảnh đầu vào.

> Bạn đã bao giờ dùng filter trên Instagram hay TikTok? Đằng sau những hiệu ứng đó là hệ thống như thế này — nó phải biết mặt bạn ở đâu, và vùng nào là da mặt để apply filter.

> **Ứng dụng thực tế** rất rộng:
> - Camera an ninh, nhận diện khuôn mặt
> - Beauty camera, AR effects
> - Makeup try-on
> - Medical imaging

> **Bài toán của project:** Cho 1 ảnh đầu vào → trả về **bounding box** cho mỗi khuôn mặt + **binary mask** tách vùng da mặt.

> **Mục tiêu cụ thể:**
> - Pipeline end-to-end: ảnh vào → overlay ra
> - IoU segmentation > 0.90 (chuẩn ngành)
> - Chạy được trên CPU (không cần GPU)

---

## 🔗 Phần 2: Pipeline Tổng Quan (1:30)

> Pipeline của tôi là **2-stage cascade** — 2 giai đoạn nối tiếp nhau:

> **Giai đoạn 1 — RetinaFace Detection:**
> Input là ảnh BGR kích thước bất kỳ. Mình resize về 640×640, đưa qua model RetinaFace, model sẽ trả về N bounding box cùng confidence score. Mỗi box có format `[x1, y1, x2, y2, score]`.

> **Giai đoạn 2 — U-Net Segmentation:**
> Với mỗi bbox tìm được, mình crop vùng mặt (có margin 10%), resize về 256×256, rồi đưa qua U-Net. U-Net trả về binary mask 256×256 — pixel nào là mặt, pixel nào là background.

> **Post-processing:**
> Cuối cùng, resize mask về kích thước crop ban đầu, render overlay đỏ lên ảnh gốc, vẽ bbox, lưu PNG + JSON kết quả.

> **Tại sao tách 2 giai đoạn?**
> - Detection localize trước → segmentation chỉ tập trung vào vùng mặt
> - Chính xác hơn so với segmentation-only
> - Linh hoạt: có thể dùng riêng từng stage

---

## 🧠 Phần 3: Stage 1 — RetinaFace (2:00)

> Giai đoạn 1 dùng **RetinaFace** — paper từ CVPR 2020. Đây là model face detection rất nổi tiếng, cũng là state-of-the-art trên WIDER FACE dataset.

> **Kiến trúc gồm 4 phần chính:**

> **Phần 1 — Backbone:** Mình dùng **ResNet-34** để extract features. ResNet-34 có 4 stages, mình lấy output của 3 stages cuối gọi là c3, c4, c5 — tương ứng với stride 8, 16, 32.

> **Phần 2 — FPN (Feature Pyramid Network):** FPN kết hợp features từ nhiều scale. c3 + upsample c4 → p3, c4 + upsample c5 → p4, c5 → p5. Mục đích là để detect mặt ở nhiều kích thước khác nhau.

> **Phần 3 — SSH Module:** SSH là context module giúp tăng receptive field. Mỗi FPN level đi qua SSH → 128-channel feature map.

> **Phần 4 — Multi-task Heads:** Ở mỗi FPN level, có 3 head song song:
> - Classification: 12 channel — phân loại face / background cho 3 anchors × 2 box
> - Box regression: 24 channel — dự đoán 4 tọa độ box cho 3 anchors × 2
> - Landmark: 60 channel — dự đoán 10 tọa độ landmark cho 3 anchors × 2

> **Output shapes** ở 3 FPN level:
> - p3 (stride 8): 6400 anchors
> - p4 (stride 16): 1600 anchors
> - p5 (stride 32): 400 anchors
> - Tổng: **8400 anchors** trên 1 ảnh

> Model có **22.1M parameters**, file checkpoint **84.6 MB**.

> **Kết quả smoke test:**
> - Forward pass: 0.56 giây trên CPU
> - Top confidence score: **1.42** — rất cao, nghĩa là model tự tin khi detect

---

## 🎨 Phần 4: Stage 2 — U-Net (2:00)

> Giai đoạn 2 dùng **U-Net** — paper từ MICCAI 2015, ban đầu cho medical imaging nhưng cũng rất hiệu quả cho face segmentation.

> **Cấu trúc U-Net có hình chữ U:**

> **Phần Encoder** — đi xuống:
> - enc1: 3 → 64 channels
> - enc2: 64 → 128
> - enc3: 128 → 256
> - enc4: 256 → 512
> - bottleneck: 512 → 1024
>
> Mỗi encoder block là DoubleConv = 2 lần (Conv 3×3 + BatchNorm + ReLU). Giữa các block có MaxPool 2×2 để giảm resolution.

> **Phần Decoder** — đi lên:
> - up4: 1024 → 512, concat với enc4 → dec4
> - up3: 512 → 256, concat với enc3 → dec3
> - up2: 256 → 128, concat với enc2 → dec2
> - up1: 128 → 64, concat với enc1 → dec1
>
> Decoder dùng Upsample + Concat với skip connection từ encoder.

> **Skip connections** là điểm mấu chốt: chúng giữ lại spatial details từ encoder, giúp reconstruct edges chính xác.

> **Output:** 64 → 2 channels (background + face), qua argmax → binary mask 256×256.

> **Loss function:**
> - 0.5 × **BCE Loss** (Binary Cross-Entropy) trên logits
> - 0.5 × **Dice Loss** (1 − |P∩G| / |P∪G|)
>
> Kết hợp BCE + Dice giúp ổn định training và handle class imbalance.

> Model có **31M parameters**, file checkpoint **118.5 MB**.

---

## 📊 Phần 5: Dữ Liệu & Huấn Luyện (1:00)

> Mình dùng 2 dataset:

> **WIDER FACE** cho detection training:
> - 32,000 ảnh, 393,000 khuôn mặt
> - 80/10/10 split cho train/val/test
> - Preprocess: resize về max 1024, lưu CSV annotations

> **CelebAMask-HQ** cho segmentation training:
> - 30,000 ảnh khuôn mặt
> - Split: 24K train / 3K val / 3K test
> - Preprocess: resize 256×256, mask binary 0/255

> **Training config cho U-Net:**
> - Optimizer: Adam, learning rate 1e-3
> - Loss: 0.5 × BCE + 0.5 × Dice
> - Batch size 16, train 30 epochs
> - Augmentation: flip, color jitter

> **Đặc biệt:** Toàn bộ training và inference chạy trên CPU, không cần GPU. Training mất khoảng 2-4 giờ.

---

## 📈 Phần 6: Kết Quả Evaluation (1:30)

> Bây giờ là phần quan trọng nhất — **kết quả thực tế**.

> **U-Net Segmentation — Test Set (100 ảnh, đánh giá lại ngày 25/09/2026):**

> - **Mean IoU: 0.9679** — tức là 96.79% pixel overlap giữa prediction và ground truth
> - **Mean Dice: 0.9834** — F1 score cho segmentation
> - **Pixel Accuracy: 0.9770** — 97.70% pixel được phân loại đúng
> - **F1 (face class): 0.9834**
> - **Precision: 0.9828, Recall: 0.9845**

> So với target đề ra là IoU ≥ 0.90, mình vượt **+7.8%**. Trên validation set cũng đạt **0.9666** — gần như tương đương test set, nghĩa là model **không overfit**.

> **RetinaFace Detection:**
> - Forward pass: 0.56 giây
> - Detector run với post-processing: 0.44 giây
> - Top confidence score: **1.42** — rất confident
> - Smoke test verdict: **OK**

> **Pipeline end-to-end:** ~1.2 giây cho 1 ảnh có 1 khuôn mặt trên CPU.

> Kết quả này cho thấy hệ thống hoạt động ổn định và chính xác, đủ tốt cho ứng dụng thực tế.

---

## 🎬 Phần 7: Demo & Kết Luận (1:00)

> **(Nếu có demo trực tiếp):**
> Tôi sẽ chạy thử trên một ảnh. Đây là ảnh đầu vào... model detect được 2 mặt, đây là overlay với bbox và mask đỏ.

> **(Nếu chỉ có visualization có sẵn):**
> Tôi đã generate sẵn 8 visualization trong folder `runs/visualizations/test/`. Mỗi ảnh có 4 panel: ảnh gốc, predicted mask, overlay, ground truth. Như các bạn thấy, mask dự đoán gần như khớp hoàn toàn với ground truth.

> **Tổng kết — Đã đạt được:**
> - Pipeline end-to-end hoàn chỉnh
> - U-Net IoU = **96.79%** vượt target 90%
> - RetinaFace checkpoint load + forward verified
> - Full evaluation suite + visualizations
> - 67 unit tests, tất cả pass
> - Documentation đầy đủ (ARCHITECTURE.md, progress_status.md, ROADMAP.md)

> **Hướng phát triển tiếp:**
> - Chạy full WIDER FACE mAP evaluation
> - GPU inference để tăng tốc 5-10×
> - Real-time webcam pipeline
> - ONNX export cho production
> - Web demo với Gradio

> Cảm ơn mọi người đã lắng nghe. Tôi sẵn sàng trả lời câu hỏi.

---

## 🙋 Q&A Chuẩn Bị Sẵn

**Hỏi: Tại sao chọn U-Net thay vì DeepLab?**

> U-Net đơn giản hơn, train nhanh hơn, và với bài toán binary segmentation (chỉ 2 class: face/background), U-Net hoàn toàn đủ dùng. DeepLab thường dùng cho multi-class segmentation phức tạp hơn.

**Hỏi: Có cần GPU không?**

> Không bắt buộc. Inference chạy OK trên CPU khoảng 1.2 giây/ảnh. Training thì có GPU sẽ nhanh hơn 5-10 lần, nhưng CPU vẫn train được, chỉ mất thời gian hơn.

**Hỏi: Tại sao IoU cao vậy?**

> Dataset CelebAMask-HQ có khuôn mặt chiếm phần lớn ảnh, ít background phức tạp. U-Net với skip connections rất phù hợp cho loại data này. Ngoài ra loss kết hợp BCE + Dice giúp ổn định training.

**Hỏi: Có thể detect nhiều mặt không?**

> Có. RetinaFace detect được nhiều bounding box, và mỗi box sẽ được segment riêng qua U-Net. Pipeline xử lý tuần tự từng face.

**Hỏi: Input ảnh cần kích thước bao nhiêu?**

> Không giới hạn — pipeline tự resize về 640×640 cho detection và 256×256 cho segmentation. Output mask sẽ được resize về kích thước ảnh gốc.

**Hỏi: Làm sao dùng model?**

> Có 2 cách:
> - **CLI:** `python scripts/kaggle/end_to_end_smoke_test.py` để test pipeline
> - **Python API:** import `FaceSegmentationPipeline` từ `src.pipeline.orchestrator`, khởi tạo với 2 file .pth, gọi `pipeline.run("image.jpg")`

---

## 📌 Ghi Chú Khi Trình Bày

### ⏰ Timing:
- **Tổng 10 phút**, có buffer ~30 giây
- **Nếu hết giờ** → bỏ phần Demo, kết thúc ở phần Kết quả
- **Nếu thừa giờ** → mở rộng phần Q&A hoặc show visualization chi tiết

### 🎨 Visual aids cần chuẩn bị (optional):
- Sơ đồ pipeline (vẽ tay hoặc mở ARCHITECTURE.md)
- Mở file `runs/visualizations/test/summary.png` để show kết quả
- Mở `docs/ARCHITECTURE.md` nếu ai hỏi chi tiết kỹ thuật

### 💬 Tone & Style:
- Nói tự nhiên, không đọc từng chữ
- Dùng ví dụ cụ thể (Instagram filter, AR camera)
- Tránh jargon không cần thiết
- Nếu hỏi chi tiết → chuyển sang tài liệu kỹ thuật

### 🎤 Câu hook:
> "Bạn đã bao giờ thắc mắc filter trên Instagram hoạt động thế nào?"

### 🎤 Câu kết:
> "Project đã đạt IoU=96.79% trên tập test, vượt target 90%. Cảm ơn mọi người đã lắng nghe."

---

## 🔗 Tài Liệu Tham Khảo Nhanh

- 📄 `ARCHITECTURE.md` — Kiến trúc chi tiết
- 📊 `docs/status/progress_status.md` — Status hiện tại
- 📚 `docs/planning/ROADMAP.md` — Lộ trình dự án
- 🧪 `tests/` — 67 unit tests
- 🖼️ `runs/visualizations/` — Demo images
- 📋 `docs/PRESENTATION_SCRIPT.md` — File này
