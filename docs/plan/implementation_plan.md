# Implementation Plan — Face Detection & Face Segmentation

> **Source:** `đề.txt` (Topic: Face Detection & Face Segmentation)
> **Aligned with:** `progress_status.md` + `.cursor/skills/face-detection-segmentation/`
> **Owner:** AI Engineer (you) + Cursor Agent (assistant)
> **Last updated:** 2026-09-16

---

## 0. Đọc đề — Trích yêu cầu gốc

> **Topic:** Face Detection & Face Segmentation
> **Bài toán:**
>   - Detection: khuôn mặt trong ảnh đông người
>   - Segmentation: vùng mặt (face mask)
> **Dataset:**
>   - WIDER FACE
>   - CelebAMask-HQ
> **Mô hình gợi ý:**
>   - RetinaFace
>   - U-Net / Mask R-CNN
> **Ứng dụng:** Camera an ninh, nhận diện khuôn mặt trong đám đông

### Từ khóa đề bài → Feature → AI Solution

| Keyword đề | Feature (output) | AI Solution |
|---|---|---|
| "khuôn mặt trong ảnh đông người" | Bounding boxes, robust với đám đông / che lấp | **RetinaFace** (SOTA WIDER FACE hard set) |
| "vùng mặt (face mask)" | Pixel-level mask cho từng bbox | **U-Net** (binary face mask) — default, hoặc **Mask R-CNN** nếu muốn 1-stage |
| "Camera an ninh" | Real-time, robust 24/7, các điều kiện ánh sáng xấu | Pipeline tối ưu ONNX/TensorRT |
| "nhận diện khuôn mặt trong đám đông" | Precision cao trên ảnh đám đông | Eval trên WIDER FACE hard subset |

---

## 1. Khung tổng quát (theo yêu cầu của bạn)

```
Business Understanding
        ↓
Features  →  Tech Solution  →  AI Solution  →  Implementation (plan → vibe coding)  →  Demo  →  Test
                                                                                          │
                                                                                  ┌───────┴────────┐
                                                                                  ▼                ▼
                                                                          Model-level      Performance full pipeline
                                                                         (mAP/IoU/Dice)   (Acc / Latency / Throughput)
```

**Áp dụng vào dự án:**

| Tầng | Câu hỏi What/Why/Where/When/Which | Câu trả lời (How) |
|---|---|---|
| **Features** | What cần detect/segment? | bbox + face mask trên ảnh đám đông |
| **Tech** | Tech stack nào? | Python + PyTorch + OpenCV + Albumentations + ONNX/TensorRT + FastAPI |
| **AI** | Model nào? Tại sao? | RetinaFace (SOTA crowd) + U-Net (nhẹ, nhanh) |
| **Implementation** | Code ở đâu? | `src/detection/`, `src/segmentation/`, `src/pipeline/` |
| **Demo** | Chạy thế nào? | CLI + ảnh sample + overlay PNG |
| **Test** | Đo gì? | Model-level + Full pipeline |

---

## 2. Roadmap tổng (10 giai đoạn — ánh xạ 14 subtask)

> Mỗi giai đoạn có: **Mục tiêu** | **Input cần có** | **Output cam kết** | **Tiêu chí done** | **Subtask #**

### GĐ 0 — Setup (Subtask: Setup chung)

- **Mục tiêu:** Repo chạy được `python -m src.pipeline.run --help` không lỗi.
- **Input:** Repo trống + đề + `progress_status.md` đã viết.
- **Output:**
  ```
  requirements.txt
  pyproject.toml
  src/__init__.py
  src/{detection,segmentation,pipeline,utils,configs}/__init__.py
  ```
- **Done when:** `pytest tests/test_imports.py` pass + `python -m src.pipeline.run --help` in usage.

---

### GĐ 1 — Research & Survey (#1)

- **Mục tiêu:** Chốt được model & pattern (đã chốt RetinaFace + U-Net, nhưng cần ADR).
- **Input:** Internet / papers / repo InsightFace, Pytorch-UNet.
- **Output:** `docs/survey.md` + `docs/adr/0001-model-choice.md`.
- **Done when:**
  - [ ] Bảng so sánh ≥4 detection model + ≥4 segmentation model.
  - [ ] Có số liệu mAP / FPS tham khảo từ paper.
  - [ ] Quyết định cuối: **RetinaFace + U-Net** (default) + **Mask R-CNN** (alternative).

---

### GĐ 2 — Data Acquisition & Preprocessing (#2, #3)

- **Mục tiêu:** Có dataset PNG sẵn sàng train, split 80/10/10.
- **Input:** URL WIDER FACE + URL CelebAMask-HQ (xem `references/data.md`).
- **Output:**
  ```
  data/processed/detection/{train,val,test}/{images/*.png, annotations.csv}
  data/processed/segmentation/{train,val,test}/{images/*.png, masks/*.png}
  data/processed/*/DATASET.md   # dataset card
  ```
- **Done when:**
  - [ ] WIDER FACE: ≥25K ảnh train, ≥3K val, ≥3K test (PNG).
  - [ ] CelebAMask-HQ: 24K/3K/3K (PNG, masks 0/255).
  - [ ] Mỗi dataset có `DATASET.md` (license, version, số ảnh, sha256 của file index).
  - [ ] Test no-leak (`tests/test_data.py::test_no_overlap`).

---

### GĐ 3 — Detection Module (#4)

- **Mục tiêu:** Code RetinaFace chạy được trên 1 ảnh sample.
- **Input:** WIDER FACE PNG + annotations.csv.
- **Output:**
  ```
  src/detection/
  ├── retinaface.py        # model
  ├── anchors.py
  ├── dataset.py
  ├── losses.py
  ├── inference.py         # forward + NMS + decode
  ├── train.py
  └── eval.py
  models/retinaface_init.pth   # pretrained trên ImageNet (chỉ backbone)
  ```
- **Done when:**
  - [ ] `python -m src.detection.inference --image <png>` in bboxes.
  - [ ] Unit test: `test_detection_inference.py` pass.
  - [ ] Sample overlay lưu tại `data/output/sample_det.png`.

---

### GĐ 4 — Train Detector (#5)

- **Mục tiêu:** Có checkpoint RetinaFace đạt ≥0.85 mAP trên WIDER FACE val.
- **Input:** GĐ 2 + GĐ 3.
- **Output:** `models/retinaface_best.pth` + log TensorBoard + entry vào `eval_results.md`.
- **Done when:**
  - [ ] Train 50 epoch, early stop patience 5.
  - [ ] mAP@0.5 ≥ 0.85 (val) hoặc đạt 80% paper baseline.
  - [ ] Logs committed ở `runs/retinaface_<timestamp>/`.
  - [ ] Eval block append vào `data/output/eval_results.md`.

---

### GĐ 5 — Segmentation Module (#6)

- **Mục tiêu:** Code U-Net chạy được trên 1 ảnh sample.
- **Input:** CelebAMask-HQ PNG.
- **Output:**
  ```
  src/segmentation/
  ├── unet.py
  ├── dataset.py
  ├── losses.py
  ├── inference.py
  ├── train.py
  └── eval.py
  models/unet_init.pth
  ```
- **Done when:**
  - [ ] `python -m src.segmentation.inference --image <png>` in mask.
  - [ ] Unit test pass.
  - [ ] Sample mask lưu tại `data/output/sample_seg.png`.

---

### GĐ 6 — Train Segmentor (#7)

- **Mục tiêu:** U-Net IoU ≥0.85 trên CelebAMask-HQ val.
- **Input:** GĐ 2 + GĐ 5.
- **Output:** `models/unet_best.pth` + logs + eval entry.
- **Done when:**
  - [ ] 30 epoch, BCE+Dice loss.
  - [ ] IoU ≥ 0.85 (val), Dice ≥ 0.90.
  - [ ] Eval block append vào `data/output/eval_results.md`.

---

### GĐ 7 — Pipeline Orchestrator (#8)

- **Mục tiêu:** End-to-end: ảnh → bbox + mask + overlay.
- **Input:** GĐ 4 + GĐ 6.
- **Output:**
  ```
  src/pipeline/
  ├── run.py              # CLI entrypoint
  ├── stages.py
  └── visualizer.py
  ```
- **Done when:**
  - [ ] `python -m src.pipeline.run --image <png> --output <dir>` chạy được.
  - [ ] Failure modes handled (no face → empty, mask degenerate → skip).
  - [ ] `tests/test_pipeline.py` pass.

---

### GĐ 8 — Unit Tests & CI (#9)

- **Mục tiêu:** Coverage ≥ 70% cho pipeline, ≥ 60% cho các module khác.
- **Input:** Toàn bộ code đã viết.
- **Output:** `tests/` + `.github/workflows/ci.yml`.
- **Done when:**
  - [ ] `pytest --cov=src tests/` đạt coverage target.
  - [ ] CI chạy `ruff`, `black --check`, `pytest` mỗi PR.

---

### GĐ 9 — Model-Level + Full-Pipeline Evaluation (#10, #11)

- **Mục tiêu:** Đo đầy đủ metrics, xuất report.
- **Input:** Toàn bộ pipeline + ground truth.
- **Output:** `data/output/eval_results.md` (nhiều entries), `docs/milestones/eval.md`.
- **Done when:**
  - [ ] Detection: mAP@0.5, Recall@0.5 (val + test, WIDER FACE easy/medium/hard).
  - [ ] Segmentation: IoU, Dice, Pixel acc (val + test).
  - [ ] Full pipeline: end-to-end acc/recall, latency GPU/CPU, throughput, robustness subsets.
  - [ ] Failure pipeline rate < 1%.

---

### GĐ 10 — Demo, Export, Docs (#12, #13, #14)

- **Mục tiêu:** Có sản phẩm demo + tài liệu.
- **Output:**
  ```
  scripts/
  ├── inference/demo.py            # Demo CLI (image/video/webcam)
  └── kaggle/                       # Kaggle integration
  README.md
  AI_USAGE.md
  ```
- **Done when:**
  - [ ] Demo CLI runs end-to-end on a sample image.
  - [ ] ONNX export test pass (`test_export.py`).
  - [ ] README có quickstart + screenshot overlay.
  - [ ] AI_USAGE.md log mọi lần dùng AI.

---

## 3. Chi tiết từng subtask (checklist thực thi)

> Subtask #1 → #14 dưới đây **ánh xạ 1-1** với `progress_status.md` §12 và skill `face-detection-segmentation/SKILL.md`.

### #1 — Survey literature → `docs/survey.md`
- [ ] Tải & đọc paper RetinaFace (arxiv 1905.00641).
- [ ] Tải & đọc paper U-Net (arxiv 1505.04597).
- [ ] Đọc paper Mask R-CNN (nếu chọn alternative).
- [ ] Note: dataset WIDER FACE benchmark, CelebAMask-HQ.
- [ ] Viết bảng so sánh speed vs accuracy.
- [ ] Quyết định cuối + viết ADR `docs/adr/0001-model-choice.md`.

### #2 — WIDER FACE preprocessing → `data/processed/detection/`
- [ ] Download zips (train/val annotations).
- [ ] Convert tất cả ảnh → PNG.
- [ ] Parse `wider_face_train_bbx_gt.txt` → `annotations.csv`.
- [ ] Áp dụng split 80/10/10 với `SEED=42`.
- [ ] Viết `DATASET.md` (license, version, sha256, số lượng).
- [ ] Chạy `tests/test_data.py::test_no_overlap`.

### #3 — CelebAMask-HQ preprocessing → `data/processed/segmentation/`
- [ ] Download 30K ảnh + mask zip.
- [ ] Merge 19-class mask → binary face mask (255 if class ∈ face-related).
- [ ] Convert ảnh + mask → PNG.
- [ ] Split 80/10/10 với `SEED=42`.
- [ ] Viết `DATASET.md`.
- [ ] `tests/test_data.py::test_mask_shape`.

### #4 — Implement RetinaFace training loop → `src/detection/`
- [ ] Định nghĩa model: ResNet-34 + FPN + SSH + 3 multi-task heads.
- [ ] Anchor generator (3 scales, 3 ratios).
- [ ] Multi-task loss (cls + box + landmark).
- [ ] Dataset class (load PNG + decode bbox + augmentation).
- [ ] Train script (`python -m src.detection.train`).
- [ ] Inference + NMS + decode.
- [ ] Test forward pass shape.

### #5 — Train RetinaFace → `models/retinaface_*.pth`
- [ ] Chạy train với config từ `progress_status.md §6.3.3`.
- [ ] Save best checkpoint theo mAP val.
- [ ] Log vào TensorBoard.
- [ ] Append kết quả vào `data/output/eval_results.md`.

### #6 — Implement U-Net → `src/segmentation/`
- [ ] Encoder ResNet-34 pretrained.
- [ ] Decoder với skip connections.
- [ ] Loss BCE + Dice (0.5/0.5).
- [ ] Dataset class.
- [ ] Train + inference scripts.
- [ ] Test forward pass shape.

### #7 — Train U-Net → `models/unet_*.pth`
- [ ] Chạy train với config §6.3.3.
- [ ] Save best theo IoU val.
- [ ] TensorBoard logs.
- [ ] Append vào `eval_results.md`.

### #8 — Build pipeline orchestrator → `src/pipeline/`
- [ ] `FaceSegmentationPipeline` class.
- [ ] Stages: detect → crop → segment → post-process → overlay.
- [ ] CLI entrypoint `python -m src.pipeline.run`.
- [ ] Graceful degradation (no face, mask empty).

### #9 — Unit tests → `tests/`
- [ ] `test_imports.py` — module imports.
- [ ] `test_detection_inference.py` — forward + NMS.
- [ ] `test_segmentation_inference.py` — forward + threshold.
- [ ] `test_pipeline.py` — end-to-end + failure paths.
- [ ] `test_data.py` — split no-leak, mask shape.
- [ ] `test_export.py` — ONNX roundtrip.
- [ ] CI config `.github/workflows/ci.yml`.

### #10 — Model-level evaluation → `eval_results.md`
- [ ] Detection metrics: mAP, Recall (val + test).
- [ ] Segmentation metrics: IoU, Dice, Pixel acc.
- [ ] Latency (GPU/CPU) cho từng stage.
- [ ] Append entries (không sửa lịch sử).

### #11 — Full-pipeline evaluation → `eval_results.md`
- [ ] End-to-end accuracy + recall.
- [ ] Latency end-to-end.
- [ ] Throughput (img/s).
- [ ] Robustness subsets (crowded, occluded, low-light, scale).
- [ ] Failure rate.

### #12 — Demo → `scripts/inference/`
- [x] `scripts/inference/demo.py` chạy end-to-end trên 1 ảnh sample.
- [x] `quick_demo.ipynb` merged into `scripts/inference/demo.py`.
- [x] Screenshot overlay lưu vào `README.md`.

### #13 — Export ONNX / TensorRT *(removed — deploy/ directory deleted)*
- [x] `export_onnx.py` was available with ONNX roundtrip test.
- Decision: ONNX export pipeline removed; ONNX tests kept inline in `tests/test_export.py`.
- [ ] (Optional) TensorRT engine build script.
- [ ] (Optional) FastAPI server.

### #14 — Docs → `README.md` + `AI_USAGE.md`
- [ ] README: problem, architecture, quickstart, demo, results.
- [ ] AI_USAGE: log mọi lần dùng Cursor/LLM.
- [ ] `docs/milestones/*.md` cho mỗi milestone.

---

## 4. Lịch trình gợi ý (Work Breakdown)

| Tuần | Giai đoạn | Subtask | Output |
|---|---|---|---|
| W1 | Setup + GĐ 1 + GĐ 2 | #1, #2, #3 | survey.md, datasets PNG |
| W2 | GĐ 3 + GĐ 5 | #4, #6 | src/detection, src/segmentation |
| W3 | GĐ 4 + GĐ 6 | #5, #7 | trained weights + eval entries |
| W4 | GĐ 7 + GĐ 8 + GĐ 9 | #8, #9, #10, #11 | pipeline + tests + full eval |
| W5 | GĐ 10 | #12, #13, #14 | demo + ONNX + docs |

> *Timeline này chỉ là gợi ý; tùy tốc độ & GPU bạn có.*

---

## 5. Risk & Mitigation

| Risk | Mitigation |
|---|---|
| WIDER FACE download chậm/lỗi | Mirror từ Google Drive hoặc HuggingFace; dùng DVC version dataset |
| CelebAMask-HQ parse 19-class phức tạp | Dùng script chuẩn từ repo gốc; visualize mẫu 5 ảnh để verify |
| RetinaFace train không converge | Bắt đầu với pretrained backbone (ResNet-34 ImageNet hoặc custom); giảm LR xuống 1e-5 |
| GPU OOM | Giảm batch (8→4), resize ảnh nhỏ hơn (640→512), dùng mixed precision (AMP) |
| Mask tràn ra ngoài bbox | Post-process: mask = mask & bbox_mask |
| Latency quá cao | Export ONNX + TensorRT FP16 |
| Face data privacy | On-prem only, không push dataset lên cloud |

---

## 6. Decision Log (sẽ cập nhật khi có ADR mới)

| Date | Decision | Rationale | ADR |
|---|---|---|---|
| 2026-09-16 | Chọn **RetinaFace + U-Net** | SOTA cho crowd detection + nhẹ cho face mask | `0001-model-choice.md` |
| 2026-09-16 | Ảnh input **PNG**, split **80/10/10**, SEED=42 | Theo yêu cầu đề (PNG) + reproducibility | `0002-data-format.md` |
| 2026-09-16 | Eval → `data/output/eval_results.md` | Decouple progress narrative khỏi metric logs | `0003-eval-logging.md` |

---

## 7. Quick links

- Đề bài gốc: `đề.txt`
- Tiến độ dự án: `progress_status.md` (root)
- Skill: `.cursor/skills/face-detection-segmentation/SKILL.md`
- References: `.cursor/skills/face-detection-segmentation/references/*.md`
- Eval log: `data/output/eval_results.md` (sẽ tạo khi bắt đầu chạy)