# Đánh giá Model — Face Detection & Face Segmentation

**Người thực hiện:** Hậu — phụ trách Evaluation
**Ngày:** 15/9/2026
**Checkpoint đánh giá:**
- Detection: `checkpoints/detection_20260914_014936/best.pth` (720,061 KB)
- Segmentation: `checkpoints/segmentation_fcn8s_20260914_052111/best.pth` (118,522 KB)

---

## 1. Thiết lập training (theo config.yaml đã lưu cùng checkpoint)

### 1.1 Detection

| Tham số | Giá trị |
|---|---|
| Kiến trúc | DSFD, backbone `resnet34` (mặc định — config không có key `backbone` nên script dùng default; key `model_size: n` trong file **không được code sử dụng**, chỉ là field còn sót lại từ template) |
| Ảnh input | 512×512 (khác với default repo là 256×256 → đã được override qua `--img_size` khi chạy) |
| Batch size | 16 (khác default repo = 4 → cũng được override) |
| Epochs | 100 |
| Learning rate | 1e-4 → min 1e-6 |
| Loss weights | box=1.0, cls=1.0, focal α=0.25 γ=2.0 |
| Dataset | `data/processed/wider_face` |
| Pretrained | `false` (train từ đầu, không dùng pretrained backbone) |

### 1.2 Segmentation

| Tham số | Giá trị |
|---|---|
| Kiến trúc | FCN8s (key `backbone: alternative_segmentation` trong config **không được code sử dụng** — script đọc key `model_type`, không có nên dùng default `fcn8s`; khớp với tên thư mục checkpoint `segmentation_fcn8s_...`) |
| Ảnh input | 512×512 |
| Batch size | 16 |
| Epochs | 100, scheduler `cosine_annealing_warm_restarts` (T_0=10, T_mult=2) |
| Loss | CE (w=1.0) + Dice (w=1.0) + Focal (w=0.5) |
| Dataset | `data/processed/celebamask_hq`, 19 lớp |
| Pretrained | `false` |

> **Lưu ý cho báo cáo:** cả 2 file config đều có 1 field "mồ côi" (`model_size` và `backbone`) không khớp với key mà code thực sự đọc (`backbone` cho detection, `model_type` cho segmentation). Đây là lỗi nhỏ về đặt tên field trong config template, nên trình bày rõ trong báo cáo để tránh hiểu nhầm về kiến trúc thật sự đã train.

---

## 2. Đánh giá Detection Model

Dựa trên `outputs/visualizations/detection/detection_20260914_014936/`:

- **Loss** (`loss_curves.png`): giảm đều và hội tụ rất nhanh, về ~0.0001 sau ~40 epoch, train/val bám sát nhau (không overfit theo nghĩa thông thường).
- **mAP@0.5** (`mAP_curve.png`): **phẳng tuyệt đối ở 0.5 suốt 100 epoch.** Kiểm tra code (`train_detection.py`, hàm `validate()`) cho thấy đây là **giá trị hard-code** (`precision, recall = 0.5, 0.5 # Placeholder`), không phải mAP tính thật. → **Không có ý nghĩa đánh giá.**
- **Sample predictions** (`sample_predictions_epoch_*.png`): ảnh input là **nhiễu ngẫu nhiên** (random RGB noise), không phải ảnh khuôn mặt thật. Nguyên nhân: `train_detection.py` có `DummyDetectionDataset` fallback khi không tìm thấy `data/processed/wider_face` thật, và toàn bộ 100 epoch đã chạy trên nhánh fallback này.

**Kết luận:** Loss giảm nhanh chỉ phản ánh việc model khớp (overfit) với box/nhãn ngẫu nhiên có phân phối đơn giản — **không đánh giá được khả năng phát hiện khuôn mặt thật của model.**

---

## 3. Đánh giá Segmentation Model

Dựa trên `outputs/visualizations/segmentation/segmentation_fcn8s_20260914_052111/`:

| Chỉ số | Giá trị hội tụ | Đối chiếu | 
|---|---|---|
| CE Loss | ~2.944 | = **ln(19) = 2.9444** — đúng bằng entropy của phân phối đều trên 19 lớp |
| Dice Loss | ~0.9473 (gần như không giảm suốt 100 epoch) | Dice score tương ứng chỉ ~0.053 |
| Validation Pixel Accuracy | ~5.2–5.3%, dao động ngẫu nhiên không có xu hướng tăng | ≈ **1/19 = 5.26%** — đúng bằng accuracy của đoán ngẫu nhiên đều |
| Validation mIoU | ~0.025–0.0265, **giảm dần** trong nửa đầu training rồi đi ngang | Rất thấp so với FCN8s/U-Net thật (thường 0.5–0.8 mIoU) |
| Per-class IoU (epoch 100) | Tất cả 19 lớp đều nằm trong khoảng 0.02–0.04, không có lớp nào nổi bật | Không có tín hiệu học được đặc trưng riêng cho bất kỳ lớp nào |

**3 con số trùng khớp gần như tuyệt đối với "đoán ngẫu nhiên đều trên 19 lớp"** (CE = ln(19), pixel accuracy ≈ 1/19, per-class IoU đồng đều) là bằng chứng kỹ thuật rõ ràng nhất: model **không học được gì** ngoài việc hội tụ về một phân phối xác suất gần đều giữa các lớp — hệ quả trực tiếp của việc train trên `DummySegDataset` (ảnh + mask ngẫu nhiên, không có tương quan thật giữa input và nhãn nên không có gì để học).

**Kết luận:** Model segmentation hiện tại **tương đương một bộ phân loại ngẫu nhiên**, không có giá trị sử dụng thực tế ở checkpoint này.

---

## 4. Nguyên nhân gốc rễ

Cả 2 vấn đề trên đều bắt nguồn từ cùng 1 nguyên nhân: thư mục `data/processed/wider_face` và `data/processed/celebamask_hq` không tồn tại tại thời điểm chạy training (dataset chưa được tải/tiền xử lý), khiến `train_detection.py` và `train_segmentation.py` tự động rơi vào nhánh `Dummy*Dataset` (sinh ảnh + nhãn bằng `torch.randn`/`randint`) — cơ chế này vốn được thiết kế để **test code chạy được**, không phải để train thật.

## 5. Giới hạn của lần đánh giá này

- Chưa chạy được `widerface_eval.py` / `fddb_eval.py` (đã viết sẵn ở phần 3B) để lấy mAP theo Easy/Medium/Hard và ROC vì: (1) chưa có dữ liệu WIDER FACE/FDDB thật, (2) checkpoint hiện tại train trên dữ liệu random nên số liệu benchmark thật — nếu chạy — cũng sẽ ở mức ngẫu nhiên, không phản ánh gì thêm ngoài kết luận ở mục 2–3.
- Đánh giá này dựa trên log/hình ảnh đã có sẵn (loss curves, mAP curve, per-class IoU, sample predictions) và đối chiếu với source code training, không tự chạy lại model (không có checkpoint `.pth` + PyTorch trong môi trường đánh giá).

## 6. Khuyến nghị

1. Tải và trỏ đúng `data_root` tới dữ liệu WIDER FACE / CelebAMask-HQ đã tiền xử lý thật, train lại từ đầu.
2. Thay `precision, recall = 0.5, 0.5` trong `validate()` bằng lời gọi thật tới `evaluate_detection()` (đã có sẵn, viết đúng, chỉ chưa được nối vào).
3. Sau khi train lại trên dữ liệu thật, chạy `widerface_eval.py` và `fddb_eval.py` để có mAP Easy/Medium/Hard và ROC — đó mới là số liệu nên đưa vào báo cáo cuối.
4. Dọn lại config: đổi `model_size` → `backbone` (detection) và `backbone` → `model_type` (segmentation) cho khớp đúng key mà code đọc, tránh nhầm lẫn khi review.
