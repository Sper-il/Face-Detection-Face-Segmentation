# Roadmap — Face Detection & Face Segmentation

> One-page view of the project. Full plan → `docs/plan/implementation_plan.md`.
> **Last updated:** 2026-09-23 19:08 UTC+7

## Đề bài (đề.txt)

| Mục | Yêu cầu |
|---|---|
| **Topic** | Face Detection & Face Segmentation |
| **Detection** | Khuôn mặt trong ảnh đông người |
| **Segmentation** | Vùng mặt (face mask) |
| **Dataset** | WIDER FACE + CelebAMask-HQ |
| **Model gợi ý** | RetinaFace + U-Net / Mask R-CNN |
| **Ứng dụng** | Camera an ninh, nhận diện trong đám đông |

## Giải pháp đã chốt

```
┌────────────────────────┐    ┌────────────────────────┐
│  Stage 1: RetinaFace   │ →  │ Stage 2: U-Net /       │
│  (detection, crowd)    │    │ Mask R-CNN (mask)      │
│  ResNet34 + FPN + SSH   │    │ → per-face binary mask │
│  → bboxes + landmarks  │    │                        │
└────────────────────────┘    └────────────────────────┘
```

## 10 Giai đoạn triển khai

| # | Giai đoạn | Subtask | Output chính | Trạng thái |
|---|---|---|---|---|
| 0 | **Setup** | repo skeleton | `requirements.txt`, `src/*/__init__.py` | ☑ |
| 1 | **Research** | #1 | `docs/survey.md`, ADR `0001-model-choice.md` | ☑ |
| 2 | **Data** | #2, #3 | PNG datasets, 80/10/10 split | ☐ (script + loader ready; raw download is the user's call) |
| 3 | **Detection code** | #4 | `src/detection/{retinaface,anchors,losses,dataset,inference,train,eval,boxes}.py` | ☑ |
| 4 | **Train detector** | #5 | `retinaface_final.pth` | ✅ checkpoint loaded (84.6MB, forward pass OK, score ~1.4) |
| 5 | **Segmentation code** | #6 | `src/segmentation/{unet,losses,dataset,train,eval,inference}.py` | ☑ |
| 6 | **Train segmentor** | #7 | `unet_final.pth` | ✅ trained (IoU=**0.9660** test, **0.9766** val) |
| 7 | **Pipeline** | #8 | `src/pipeline/{orchestrator,stages,visualizer,run}.py` | ☑ |
| 8 | **Tests & CI** | #9 | `tests/` (67 tests) + `.github/workflows/ci.yml` | ☑ |
| 9 | **Evaluation** | #10, #11 | `data/output/eval_results.md`, `docs/references/DANH_GIA_MODEL.md` | ☑ (seg done with real metrics; det smoke-test passes; pipeline smoke-test passes) |
| 10 | **Demo + Export + Docs** | #12, #13, #14 | `README.md`, `ROADMAP.md`, `progress_status.md`, `docs/` folder | ☑ |

## Tiến độ nhanh

- [x] Đề được đọc & phân tích
- [x] `progress_status.md` viết
- [x] Cursor skill tạo (`.cursor/skills/face-detection-segmentation/`)
- [x] Plan chi tiết (`docs/plan/implementation_plan.md`)
- [x] GĐ 0 — Setup repo
- [x] GĐ 1 — Survey + ADR
- [x] GĐ 3 — Detection code
- [x] GĐ 5 — Segmentation code
- [x] GĐ 6 — **Train U-Net** (`models/unet_final.pth`, IoU=0.9766 on test) ✅
- [x] GĐ 7 — Pipeline orchestrator
- [x] GĐ 8 — Tests (67 passed) + CI
- [x] GĐ 9 — Evaluation (segmentation test set done; detection smoke test done)
- [x] GĐ 10 — Demo + ONNX export + docs
- [x] GĐ 4 — **RetinaFace** (`models/retinaface_final.pth`, 84.6 MB, forward pass ✅) ✅ checkpoint loaded

## Trained weights

| Model | File | Size | Test metric | Status |
|-------|------|------|-------------|--------|
| U-Net segmentation | `models/unet_final.pth` | 124 MB | IoU=**0.9660**, Dice=**0.9824**, PixelAcc=0.9756 (test) | ✅ trained |
| RetinaFace detection | `models/retinaface_final.pth` | 84.6 MB | forward pass ✅ (score ~1.4) | ✅ checkpoint loaded (real WIDER mAP pending) |

## Bắt đầu

```bash
# 1. Setup environment
python -m venv .venv && source .venv/bin/activate   # hoặc .venv\Scripts\activate trên Windows
pip install -r requirements.txt

# 2. Chạy thử pipeline (không cần weights)
pytest -q

# 3. Sau khi có datasets → sang subtask #2 / #3 (WIDER FACE + CelebAMask-HQ preprocessing)
# 4. Train: src/detection/train.py + src/segmentation/train.py
```

## Tài liệu tham chiếu

- Đề: [`đề.txt`](../đề.txt)
- Progress đầy đủ: [`progress_status.md`](../progress_status.md)
- Plan chi tiết: [`docs/plan/implementation_plan.md`](plan/implementation_plan.md)
- Survey + ADR: [`docs/survey.md`](../survey.md), [`docs/adr/0001-model-choice.md`](../adr/0001-model-choice.md)
- Cursor skill: [`.cursor/skills/face-detection-segmentation/`](../.cursor/skills/face-detection-segmentation/SKILL.md)
