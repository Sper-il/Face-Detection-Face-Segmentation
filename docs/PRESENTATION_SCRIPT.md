# 🎤 Bài Trình Bày Project — 10 Phút

> **Project:** Face Detection & Face Segmentation
> **Thời lượng:** 10 phút
> **Hình thức:** Trình bày folder project trực tiếp (không slide)
> **Ngày:** 2026-09-25

---

## 🗺️ Bản Đồ Click Chuột — Mở File Nào Lúc Nào

| Thời gian | Click vào đâu | Xem gì |
|-----------|---------------|--------|
| 0:00 → 1:00 | Explorer folder gốc | 6 folder chính |
| 1:00 → 2:30 | `src/`, `models/`, `data/` | Mở rộng xem cấu trúc |
| 2:30 → 4:30 | `src/detection/model.py` → `weights.py` → `boxes.py` → `infer.py` | Class RetinaFace (~dòng 50) |
| 4:30 → 6:30 | `src/segmentation/model.py` → `losses.py` → `dataset.py` → `train.py` → `infer.py` | Class UNet (~dòng 40) |
| 6:30 → 7:30 | `src/pipeline/orchestrator.py` → `src/utils/box_utils.py` → `src/eval.py` | Class Pipeline (~dòng 30) |
| 7:30 → 9:00 | `runs/evaluation/segmentation_test_metrics.json` → `val_metrics.json` → `pipeline_smoke_test.json` → `runs/visualizations/test/sample_000_*.png` | Metrics + viz |
| 9:00 → 9:30 | `tests/` → `docs/PRESENTATION_SCRIPT.md` → `docs/status/progress_status.md` | 67 tests |
| 9:30 → 10:00 | Đóng file, show folder sạch | Tổng kết + Q&A |

---

# 📖 BÀI TRÌNH BÀY HOÀN CHỈNH

> **Hướng dẫn:** Đọc từ đầu đến cuối file này. Mỗi đoạn cách nhau bằng khoảng lặng — nơi bạn cần click chuột hoặc cuộn file. Các ghi chú trong ngoặc vuông `[...]` là hướng dẫn hành động, KHÔNG đọc to.

---

## 🟢 PHẦN 1: Mở Đầu (0:00 → 1:00)

**[Mở VS Code Explorer, đảm bảo folder gốc đang hiển thị. Chưa click vào file nào.]**

Xin chào mọi người. Hôm nay tôi sẽ trình bày project **Face Detection & Face Segmentation** của tôi — một hệ thống AI hai giai đoạn có khả năng vừa phát hiện vị trí khuôn mặt, vừa tách chính xác vùng khuôn mặt từ một bức ảnh.

Trước khi vào chi tiết, cho tôi hỏi một câu: các bạn đã bao giờ thắc mắc filter trên Instagram hoạt động như thế nào chưa? Thì đằng sau nó là một hệ thống rất giống như những gì tôi đang làm.

**[Chỉ vào Explorer panel bên trái]**

Và đây chính là folder project của tôi. Khi các bạn nhìn vào Explorer bên trái, các bạn sẽ thấy tất cả những gì tôi đã xây dựng. Tổng quan có sáu folder chính:

- `src` — toàn bộ source code
- `models` — hai checkpoint đã train
- `data` — dataset đã xử lý
- `runs` — kết quả evaluation và visualization
- `tests` — sáu mươi bảy unit tests
- `docs` — documentation

**[Đợi 1-2 giây để khán giả nhìn theo hướng của bạn]**

Trong mười phút tới, tôi sẽ dẫn các bạn đi qua từng folder, mở từng file quan trọng, và giải thích những gì đã được làm.

---

## 🟢 PHẦN 2: Cấu Trúc Project (1:00 → 2:30)

**[Click vào `src/` để mở rộng]**

Bắt đầu với folder `src` — nơi chứa toàn bộ source code. Folder này được tổ chức theo module chức năng:

- `detection` — Stage 1, dùng model RetinaFace để tìm khuôn mặt
- `segmentation` — Stage 2, dùng model U-Net để tách vùng mặt
- `pipeline` — orchestrator để chạy end-to-end
- `utils` — các hàm helper như IoU, mask processing, visualization

Cách tổ chức này giúp mỗi module có thể chạy độc lập, và kết hợp lại qua pipeline.

**[Click vào `models/` để mở rộng]**

Sang folder `models`. Đây là hai file checkpoint tôi đã train được:

- `retinaface_final.pth` — Stage 1, nặng 84.6 MB
- `unet_final.pth` — Stage 2, nặng 118.5 MB

Tổng cộng khoảng hai trăm megabytes.

**[Click vào `data/` để mở rộng]**

Tiếp theo là folder `data`. Dataset đã được xử lý sẵn, chia thành hai phần:

- `processed/wider_face` — annotations cho detection
- `processed/celeba_mask` — ảnh khuôn mặt kèm mask cho segmentation

Mỗi dataset đã được split sẵn thành train, validation, và test.

**[Đóng các folder vừa mở rộng, chuẩn bị mở `src/detection/`]**

Tổng cộng project có khoảng 5,000 dòng Python code, hai model đã train, và documentation đầy đủ.

---

## 🟢 PHẦN 3: Stage 1 — Detection với RetinaFace (2:30 → 4:30)

**[Click vào `src/detection/` để mở rộng, double-click vào `model.py`]**

Bây giờ tôi sẽ đi vào chi tiết Stage 1 — Detection. Folder `src/detection/` có bốn file chính, và file đầu tiên tôi mở là `model.py`.

**[Cuộn xuống khoảng dòng 50, chỉ class RetinaFace]**

Class chính ở đây là `RetinaFace`. Kiến trúc gồm bốn phần:

**[Đọc chậm từng phần]**

- **Backbone ResNet-34** — trích xuất feature maps ở bốn scales khác nhau
- **FPN — Feature Pyramid Network** — tạo pyramid ba levels để phát hiện mặt ở nhiều kích thước
- **SSH context module** — mở rộng receptive field
- **Ba multi-task heads** — classification, box regression, và landmark prediction

**[Click sang tab `weights.py`]**

Tiếp theo là file `weights.py`. File này xử lý load và save checkpoint. Đây là phần quan trọng vì checkpoint phải được load đúng key thì model mới chạy được.

**[Click sang tab `boxes.py`]**

File `boxes.py` làm hai việc chính: **decode** — chuyển box deltas thành tọa độ thật, và **NMS** — Non-Maximum Suppression để loại các box trùng nhau.

**[Click sang tab `infer.py`]**

Cuối cùng là `infer.py` — entrypoint chạy detection cho một ảnh. Input là ảnh 640×640, output là list bounding boxes cùng confidence scores.

**[Đóng các tab này, nhưng giữ Explorer mở `src/detection/`]**

Toàn bộ Stage 1 có hai mươi hai triệu một trăm nghìn tham số. Trong smoke test, forward pass mất 0.56 giây trên CPU, và top confidence score đạt 1.42 — tức là model rất tự tin với prediction của mình.

---

## 🟢 PHẦN 4: Stage 2 — Segmentation với U-Net (4:30 → 6:30)

**[Click vào `src/segmentation/` để mở rộng, double-click vào `model.py`]**

Sang Stage 2 — Segmentation. Folder `src/segmentation/` có sáu file, nhiều hơn detection vì có thêm phần training.

**[Cuộn xuống khoảng dòng 40, chỉ class UNet]**

File `model.py` định nghĩa kiến trúc U-Net — một mạng encoder-decoder có dạng chữ U đối xứng:

- **Encoder** — bốn DoubleConv blocks, mỗi block có hai lớp Conv-BN-ReLU 3×3, xen giữa là MaxPool 2×2 để giảm resolution
- **Bottleneck** — 1024 channels ở đáy chữ U
- **Decoder** — bốn upsampling blocks, mỗi block nối với skip connection từ encoder cùng level
- **Output** — hai channels: background và face

**[Nhấn mạnh]**

Điểm đặc biệt của U-Net là **skip connections** — feature map từ encoder được nối thẳng vào decoder cùng level. Điều này cực kỳ quan trọng vì giúp decoder khôi phục spatial details mà encoder đã mất khi downsampling.

**[Click sang tab `losses.py`]**

File `losses.py` định nghĩa combined loss: 0.5 nhân BCE cộng 0.5 nhân Dice. BCE ổn định gradient, Dice xử lý tốt class imbalance — vì mặt chiếm phần lớn ảnh nên cần Dice để cân bằng.

**[Click sang tab `dataset.py`]**

File `dataset.py` chứa DataLoader cho CelebAMask-HQ — đọc ảnh kèm mask, áp dụng augmentation như flip và color jitter.

**[Click sang tab `train.py`]**

File `train.py` là training loop. Tôi dùng Adam optimizer, learning rate 1e-3, batch size 16, train 30 epochs trên CPU.

**[Click sang tab `infer.py`]**

Và cuối cùng là `infer.py` — inference cho một face crop. Input là ảnh 256×256, output là binary mask.

**[Đóng các tab này]**

U-Net có ba mươi mốt triệu tham số, checkpoint nặng 118.5 MB. Tại sao chọn U-Net thay vì DeepLab hay SegFormer? Vì U-Net đơn giản hơn, train nhanh hơn, và đủ tốt cho binary face segmentation.

---

## 🟢 PHẦN 5: Pipeline và Utils (6:30 → 7:30)

**[Click vào `src/pipeline/`, double-click `orchestrator.py`]**

Tiếp theo là folder `src/pipeline/`. File quan trọng nhất là `orchestrator.py`.

**[Chỉ class FaceSegmentationPipeline ở khoảng dòng 30]**

Class `FaceSegmentationPipeline` chạy end-to-end. Khi tôi gọi `pipeline.run('image.jpg')`, nó sẽ: chạy RetinaFace trước để lấy bounding boxes, sau đó cắt face crops và chạy U-Net cho từng crop, cuối cùng render overlay đỏ lên mặt và lưu kèm file JSON.

**[Click sang tab `src/utils/box_utils.py`]**

Folder `src/utils/` chứa các helper. File `box_utils.py` có hàm IoU và các box transforms — dùng để tính overlap giữa predicted box và ground truth.

**[Click sang tab `src/eval.py` ở root của `src/`]**

Và cuối cùng là file `src/eval.py` — main entrypoint để chạy evaluation. File này load cả hai models, chạy trên test set, in metrics ra JSON.

**[Đóng các tab]**

---

## 🟢 PHẦN 6: Kết Quả và Đánh Giá (7:30 → 9:00)

**[Click vào `runs/`, mở rộng `runs/evaluation/`, double-click vào `segmentation_test_metrics.json`]**

Bây giờ là phần quan trọng nhất — kết quả thực tế. Tôi mở file `segmentation_test_metrics.json` chạy trên 100 samples từ test set.

**[Đọc chậm các con số]**

- **Mean IoU: 0.9679** — tức là 96.79%
- **Mean Dice: 0.9834** — tức là 98.34%
- **Pixel Accuracy: 0.9770** — tức là 97.70%

Target ban đầu là IoU lớn hơn hoặc bằng 0.90, Dice lớn hơn hoặc bằng 0.95, và Pixel Accuracy lớn hơn hoặc bằng 0.97.

**[Nhấn mạnh]**

IoU đạt 96.79%, vượt target 7.8 điểm phần trăm. Dice 98.34% cũng vượt target. Điều này có nghĩa là model phân biệt được pixel thuộc mặt và pixel thuộc background với độ chính xác rất cao.

**[Click sang tab `segmentation_val_metrics.json`]**

Trên validation set, kết quả cũng tương đương — IoU 96.66%, Dice 98.26%. Chứng tỏ model không bị overfitting, generalize tốt.

**[Click sang tab `pipeline_smoke_test.json`]**

File `pipeline_smoke_test.json` cho thấy verdict là OK — pipeline end-to-end chạy ổn định.

**[Đóng các file JSON, mở `runs/visualizations/test/`, double-click vào một ảnh PNG bất kỳ]**

Để các bạn thấy trực quan, tôi mở một ảnh visualization. Ảnh này có bốn panel:

- Ảnh gốc bên trái
- Mask dự đoán
- Overlay mask đỏ
- Ground truth bên phải

**[Chỉ vào ảnh]**

Như các bạn thấy, mask dự đoán gần như khớp hoàn toàn với ground truth. Đó là lý do IoU đạt 96.79%.

**[Đóng ảnh PNG]**

---

## 🟢 PHẦN 7: Tests và Documentation (9:00 → 9:30)

**[Click vào `tests/` để mở rộng]**

Sang folder `tests/`. Project có tổng cộng sáu mươi bảy unit tests, tất cả đều passing. Test cover:

- Kiến trúc model của cả RetinaFace và U-Net
- Box decode và NMS
- Mask operations
- Pipeline end-to-end
- I-O helpers

**[Click vào `docs/` để mở rộng]**

Và folder `docs/` chứa documentation. File quan trọng nhất là `PRESENTATION_SCRIPT.md` — chính là kịch bản tôi đang đọc. Ngoài ra còn có `progress_status.md` để theo dõi status hiện tại, và `ROADMAP.md` để xem lộ trình phát triển tiếp theo.

**[Đóng folder]**

---

## 🟢 PHẦN 8: Kết Luận và Q&A (9:30 → 10:00)

**[Đóng tất cả file đang mở, chỉ để Explorer show folder gốc sạch sẽ]**

**[Nhìn khán giả]**

Tóm lại, những gì project này đã làm được:

- Xây dựng pipeline hai giai đoạn: RetinaFace detection cộng U-Net segmentation
- Train thành công cả hai models trên CPU
- U-Net đạt IoU 96.79%, vượt target
- Sáu mươi bảy unit tests passing
- Documentation và visualizations đầy đủ

**[Dứt khoát, tự tin]**

Hướng phát triển tiếp theo là chạy full WIDER FACE mAP metrics, GPU inference để speed up năm đến mười lần, và ONNX export cho production deployment.

Cảm ơn mọi người đã lắng nghe. Tôi sẵn sàng trả lời câu hỏi.

**[Đợi câu hỏi]**

---

# 📝 PHỤ LỤC: CÂU TRẢ LỜI Q&A

### Câu hỏi 1: Tại sao chọn U-Net thay vì DeepLab hay SegFormer?

U-Net đơn giản hơn, train nhanh hơn, và đủ tốt cho bài toán binary face segmentation. DeepLab với atrous convolution phù hợp multi-class hơn. SegFormer là transformer-based mới hơn nhưng tốn nhiều compute hơn. Với bài toán hai class và dataset CelebAMask-HQ có mặt chiếm phần lớn ảnh, U-Net là lựa chọn optimal.

### Câu hỏi 2: Có cần GPU không?

Inference chạy OK trên CPU — khoảng 1.2 giây cho mỗi ảnh có một mặt. Training thì có GPU sẽ nhanh hơn năm đến mười lần, nhưng tôi đã train thành công trên CPU với thời gian chấp nhận được.

### Câu hỏi 3: Tại sao IoU cao vậy — 96.79%?

CelebAMask-HQ là dataset chất lượng cao, mặt chiếm phần lớn ảnh nên dễ segment. U-Net với skip connections rất phù hợp cho bài toán có contrast rõ giữa foreground và background. Ngoài ra combined loss BCE cộng Dice giúp ổn định gradient và xử lý tốt edge cases.

### Câu hỏi 4: Có thể detect nhiều mặt trong một ảnh không?

Có. RetinaFace detect được nhiều bounding boxes trong một ảnh. Mỗi box sẽ được segment riêng bằng U-Net. Pipeline đã hỗ trợ xử lý tuần tự N faces, tối đa 50 faces theo config.

### Câu hỏi 5: So với face recognition thì khác gì?

Face recognition là bài toán khác — cần embedding vector để so sánh danh tính. Project này chỉ dừng ở detection cộng segmentation: biết **ở đâu có mặt** và **đâu là vùng mặt**, không biết **mặt của ai**. Để làm recognition, cần thêm ArcFace hoặc FaceNet — bước tiếp theo trong pipeline face analysis.

---

# 🎯 CHECKLIST TRƯỚC KHI TRÌNH BÀY

### Chuẩn bị môi trường:
- [ ] Mở VS Code với folder project
- [ ] Mở sẵn Explorer panel bên trái
- [ ] Đặt Explorer chiếm khoảng 30-40% màn hình
- [ ] Tắt các tab file không liên quan
- [ ] Zoom editor vừa phải để khán giả thấy code

### Chuẩn bị nội dung:
- [ ] Mở sẵn file `PRESENTATION_SCRIPT.md` ở cửa sổ thứ 2 (hoặc in ra giấy)
- [ ] Đọc qua bài trình bày một lần
- [ ] Tập đọc to trước 5 phút
- [ ] Đeo đồng hồ để kiểm tra timing

### Trong khi trình bày:
- [ ] Nhìn khán giả khi nói kết quả
- [ ] Chỉ tay vào file khi đọc code
- [ ] Đóng file sau khi xong để Explorer sạch
- [ ] Nói chậm ở phần Pipeline và Kết quả

### Kết thúc:
- [ ] Đóng tất cả file, show folder gốc
- [ ] Nói "Cảm ơn" rõ ràng
- [ ] Đợi câu hỏi, dùng phụ lục Q&A nếu cần

---

# 🔗 FILE THAM KHẢO NHANH

| Câu hỏi sâu | Mở file này |
|-------------|-------------|
| Kiến trúc RetinaFace | `src/detection/model.py` dòng 50-80 |
| Kiến trúc U-Net | `src/segmentation/model.py` dòng 40-100 |
| Loss function | `src/segmentation/losses.py` dòng 30-60 |
| Training loop | `src/segmentation/train.py` dòng 80-120 |
| Pipeline orchestrator | `src/pipeline/orchestrator.py` dòng 30-80 |
| Metrics gốc | `runs/evaluation/segmentation_test_metrics.json` |
| Status project | `docs/status/progress_status.md` |
| Lộ trình | `docs/planning/ROADMAP.md` |
