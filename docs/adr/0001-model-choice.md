# ADR-0001 — Model choice: RetinaFace + U-Net

- **Status:** Accepted
- **Date:** 2026-09-16
- **Deciders:** AI Engineer (project owner) + Cursor Agent
- **Supersedes:** —
- **Superseded by:** —

## Context

The project (`progress_status.md`, `đề.txt`) requires a 2-stage pipeline:

1. **Detection** — localise faces in crowded photos.
2. **Segmentation** — produce a per-face binary mask.

The candidate model space was surveyed in [`../survey.md`](../survey.md).
We need to lock in:

- one **detection** model (used for training, inference, and ONNX export);
- one **segmentation** model (same constraints);
- a **loss / decoder / backbone** choice for each.

Constraints (from `progress_status.md` §6.3 and §7):

- Detection: mAP@0.5 ≥ 0.90, Recall@0.5 ≥ 0.95, latency ≤ 50 ms (GPU).
- Segmentation: IoU ≥ 0.90, Dice ≥ 0.95, Pixel-acc ≥ 0.97, latency ≤ 50 ms (GPU).
- On-prem deployment only (no third-party upload).

## Decision

1. **Detection: RetinaFace** với backbone **ResNet-34** (custom implementation),
   FPN, SSH modules, và 3 detection head (classification, bbox regression, 5-point landmark regression).
   Custom ResNet-34 (`_ResNet34` class trong `src/detection/retinaface.py`) khớp với checkpoint đã train `models/retinaface_final.pth`.

2. **Segmentation: U-Net** với encoder ResNet-34 pretrained trên ImageNet (torchvision),
   decoder U-Net chuẩn với skip connection, sigmoid head 1 kênh.

3. **Loss (det):** multi-task — focal/softmax cho classification,
   smooth-L1 cho bbox, smooth-L1 cho landmarks. Trọng số 1 : 1 : 0.5.

4. **Loss (seg):** `0.5 * BCE + 0.5 * Dice`.

5. **Optimiser (cả hai):** AdamW, weight decay 1e-4, lr 1e-4, cosine schedule.

6. **Post-processing (det):** ngưỡng conf 0.7, NMS IoU 0.5, clip về ảnh gốc.

7. **Post-processing (seg):** sigmoid → ngưỡng 0.5 → morph (open 3x3,
   close 5x5) → clip về bbox.

8. **Kích thước ảnh:** 640x640 (det), 512x512 (seg).

9. **Augmentation:** Albumentations — horizontal flip, color jitter,
   random crop, (mosaic cho det, scale cho seg).

10. **Mixed precision (AMP):** bật mặc định cho cả hai script training.

11. **Reproducibility:** SEED = 42 cho toàn bộ; chia 80/10/10 cho cả hai dataset.

## Consequences

Tích cực:

- Cả hai model đều được hỗ trợ tốt, có reference implementation, và export sang ONNX sạch sẽ (`references/deployment.md`).
- Thiết kế 2-stage khớp với pipeline của đề và sơ đồ kiến trúc trong `progress_status.md` §3.
- Mỗi model có thể train và evaluate độc lập, đơn giản hoá các milestone GĐ 4 / GĐ 6.

Tiêu cực / rủi ro:

- Training RetinaFace từ đầu khá nặng — giảm thiểu bằng custom ResNet-34 implementation khớp với checkpoint đã train sẵn.
- U-Net mặc định không tách được instance — chấp nhận được vì mỗi khuôn mặt đã đi kèm bbox riêng.
- Ngân sách latency trên CPU khá chật — giảm thiểu bằng ONNX export và cung cấp đường `--device cpu` dùng `CPUExecutionProvider`.

## Các phương án đã xem xét

- **YOLO-Face + BiSeNet** — nhanh hơn nhưng recall thấp hơn trên ảnh đám đông / che lấp. Bị loại cho use case chính (camera an ninh, ưu tiên độ chính xác hơn throughput).
- **Mask R-CNN alone** — một model cho cả hai stage, nhưng nặng hơn, khó debug hơn, và chất lượng bbox/mask bị gắn chặt. Giữ làm **fallback** nếu pipeline 2-stage quá chậm trên CPU.
- **SCRFD-10GF** — detector kiểu RetinaFace mới hơn và nhanh hơn, nhưng cộng đồng nhỏ và không có reference training code. Giữ làm **phương án nâng cấp sau khi hoàn thành đề**.

## References

- [`../survey.md`](../survey.md) — bảng so sánh đầy đủ.
- `src/detection/retinaface.py` — RetinaFace với custom ResNet-34 backbone
- `src/segmentation/unet.py` — U-Net với ResNet-34 encoder (torchvision)
