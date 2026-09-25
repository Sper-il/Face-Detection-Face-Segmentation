# 🏗️ Architecture — Face Detection & Face Segmentation

> **Last updated:** 2026-09-25 13:51 UTC+7
> **Status:** ✅ Production-ready 2-stage cascade pipeline

---

## 📐 High-Level Overview

A **2-stage cascade pipeline** that takes a single input image and produces:
1. **Bounding boxes** for every detected face (Stage 1: RetinaFace)
2. **Per-face binary masks** for every detected face (Stage 2: U-Net)
3. **Overlay visualization** combining both outputs

```
┌────────────┐      ┌──────────────────┐      ┌────────────────┐      ┌──────────────┐
│            │      │   Stage 1        │      │   Stage 2      │      │              │
│ Input IMG  │ ───► │   RetinaFace     │ ───► │   U-Net        │ ───► │  Overlay +   │
│ (BGR)      │      │   Detection      │      │   Segmentation │      │  JSON output │
│            │      │                  │      │                │      │              │
└────────────┘      └──────────────────┘      └────────────────┘      └──────────────┘
      │                      │                        │                       │
   HxWx3              bboxes[N,4]               masks[N,H,W]            overlay.png
                     scores[N]                 (binary, 0/1)            results.json
```

---

## 🧠 Stage 1: Face Detection — RetinaFace

**File:** `src/detection/retinaface.py` (354 lines, 22.1M params)
**Checkpoint:** `models/retinaface_final.pth` (84.6 MB)
**Input:** `BGR image, 640×640`
**Output:** `cls_logits`, `box_deltas`, `lmk_deltas` per FPN level

### Architecture (ResNet34 + FPN + SSH)

```
Input (3, 640, 640)
    │
    ▼
┌─────────────────────────────────────────┐
│  Backbone: ResNet-34 (custom)           │
│  ├── stem: 7×7 conv + BN + ReLU + pool │
│  ├── layer1: 3 blocks (stride 1)       │
│  ├── layer2: 4 blocks (stride 2) → c3  │ ──┐
│  ├── layer3: 6 blocks (stride 2) → c4  │ ──┼──► FPN
│  └── layer4: 3 blocks (stride 2) → c5  │ ──┘
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  FPN: 3 levels                          │
│  ├── p3 (stride 8)  ← c3 + c4 upsampled│
│  ├── p4 (stride 16) ← c4 + c5 upsampled│
│  └── p5 (stride 32) ← c5               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  SSH (Single Stage Headless) modules    │
│  Each FPN level → SSH context module    │
│  Output: 128-channel feature maps       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Multi-task Heads (per FPN level)       │
│  ├── Classification: 1×1 conv → 12 ch  │ ──► face / bg score
│  ├── Box regression: 1×1 conv → 24 ch  │ ──► 4 coords × 3 anchors × 2 boxes
│  └── Landmark:       1×1 conv → 60 ch  │ ──► 10 coords × 3 anchors × 2 lmk
└─────────────────────────────────────────┘
```

### Output shapes (per FPN level):

| Level | Stride | # Anchors | cls_logits | box_deltas | lmk_deltas |
|-------|--------|-----------|------------|------------|------------|
| p3    | 8      | 6400      | (1, 6400, 12) | (1, 6400, 24) | (1, 6400, 60) |
| p4    | 16     | 1600      | (1, 1600, 12) | (1, 1600, 24) | (1, 1600, 60) |
| p5    | 32     | 400       | (1,  400, 12) | (1,  400, 24) | (1,  400, 60) |

**12 = 3 anchors × 4 channels** (bg + face × 2)
**24 = 3 anchors × 8 box deltas** (x1, y1, x2, y2 × 2)
**60 = 3 anchors × 20 landmark coords** (5 lmk × 4 deltas × 2)

### Key components:
| File | Purpose |
|------|---------|
| `src/detection/retinaface.py` | Model definition (ResNet34 + FPN + SSH + Heads) |
| `src/detection/anchors.py` | Anchor generation + decode |
| `src/detection/losses.py` | Multi-task loss (cls + box + landmark) |
| `src/detection/inference.py` | Forward + NMS + decode |
| `src/detection/dataset.py` | WIDER FACE dataset loader |
| `src/detection/eval.py` | mAP@0.5 + Recall@0.5 evaluation |
| `src/detection/boxes.py` | Bbox encode/decode utilities |

### Performance:
- **Forward pass:** 0.56s (CPU, 640×640)
- **Detector run:** 0.44s (CPU)
- **Top confidence score:** 1.42

---

## 🎨 Stage 2: Face Segmentation — U-Net

**File:** `src/segmentation/unet_model.py` (124 lines, 31.0M params)
**Checkpoint:** `models/unet_final.pth` (118.5 MB)
**Input:** `RGB image, 256×256`
**Output:** `2-channel logits (background + face)` → `argmax` → binary mask

### Architecture (Standard U-Net with DoubleConv blocks)

```
Input (3, 256, 256)
    │
    ▼
┌─────────────────────────────────────────┐
│  ENCODER                                │
│  enc1: 3   → 64                         │
│  enc2: 64  → 128                        │
│  enc3: 128 → 256                        │
│  enc4: 256 → 512                        │
│  bottleneck: 512 → 1024                 │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  DECODER (with skip connections)        │
│  up4: 1024 → 512  +  dec4: 1024 → 512 │
│  up3: 512  → 256  +  dec3: 512  → 256 │
│  up2: 256  → 128  +  dec2: 256  → 128 │
│  up1: 128  → 64   +  dec1: 128  → 64  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Output head: 64 → 2 (bg + face)        │
│  argmax → binary mask {0, 1}           │
└─────────────────────────────────────────┘
```

### DoubleConv block (per encoder/decoder level):
```python
Conv2d(in_ch, out_ch, 3, padding=1)  # Conv
    ↓
BatchNorm2d(out_ch)
    ↓
ReLU(inplace=True)
    ↓
Conv2d(out_ch, out_ch, 3, padding=1)
    ↓
BatchNorm2d(out_ch)
    ↓
ReLU(inplace=True)
```

### Loss function (combined):
- **BCE (CrossEntropy)** on 2-channel logits
- **Dice loss** on softmax probabilities
- **Combined:** 0.5 × BCE + 0.5 × Dice

### Key components:
| File | Purpose |
|------|---------|
| `src/segmentation/unet_model.py` | U-Net architecture (matches trained checkpoint) |
| `src/segmentation/unet.py` | Alternative U-Net with ResNet encoder |
| `src/segmentation/losses.py` | BCE + Dice combined loss |
| `src/segmentation/inference.py` | Forward + argmax + resize |
| `src/segmentation/dataset.py` | CelebAMask-HQ dataset loader |
| `src/segmentation/eval.py` | IoU + Dice + Pixel-acc evaluation |

### Performance (100 sample evaluation):
| Metric | Test Set | Validation Set |
|--------|----------|----------------|
| **Mean IoU** | **0.9679** | **0.9666** |
| **Mean Dice** | **0.9834** | **0.9826** |
| **Pixel Accuracy** | **0.9770** | **0.9756** |
| **F1 (face)** | 0.9834 | 0.9826 |
| **Inference time** | 0.66s/img | 0.66s/img |

---

## 🔗 Pipeline Orchestrator

**File:** `src/pipeline/orchestrator.py` (167 lines)

### Flow:
```
Image (BGR)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Stage 1: Detection                         │
│  ├── Resize to 640×640 + normalize          │
│  ├── Forward pass → cls/box/lmk per FPN     │
│  ├── Decode boxes from anchors + deltas     │
│  ├── NMS (IoU=0.5, conf=0.7)               │
│  └── Output: N bboxes + N scores            │
└─────────────────────────────────────────────┘
    │
    ▼ (N detected faces)
┌─────────────────────────────────────────────┐
│  Stage 2: Segmentation                      │
│  ├── For each bbox:                         │
│  │   ├── Crop face + add margin (10%)       │
│  │   ├── Resize to 256×256                  │
│  │   ├── Normalize (ImageNet mean/std)      │
│  │   ├── Forward pass → 2-channel logits    │
│  │   ├── argmax → binary mask               │
│  │   └── Resize back to crop size           │
│  └── Output: N binary masks                 │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Post-processing                            │
│  ├── Clip masks to bbox bounds              │
│  ├── Morphology (open 3×3, close 5×5)       │
│  ├── Render overlay (red mask + bbox)       │
│  └── Save PNG + JSON results                │
└─────────────────────────────────────────────┘
```

### Key components:
| File | Purpose |
|------|---------|
| `src/pipeline/orchestrator.py` | Main pipeline class (`FaceSegmentationPipeline`) |
| `src/pipeline/stages.py` | Detection + segmentation stages |
| `src/pipeline/visualizer.py` | Overlay rendering |
| `src/pipeline/run.py` | CLI entrypoint |

### Graceful degradation:
- ✅ Image load failure → returns `PipelineResult(failure_reason="load_error: ...")`
- ✅ No face detected → renders empty overlay + `failure_reason="no_face_detected"`
- ✅ Runtime error → returns `PipelineResult(failure_reason="runtime_error: ...")`
- ✅ Empty masks filtered out before overlay

---

## 🗂️ Project Structure

```
Face-Detection-Face-Segmentation/
├── src/
│   ├── detection/                          # Stage 1: RetinaFace
│   │   ├── retinaface.py                   # Main model (ResNet34+FPN+SSH)
│   │   ├── retinaface_yakhyo.py            # Yakhyo variant
│   │   ├── anchors.py                      # Anchor generation
│   │   ├── boxes.py                        # Box encode/decode
│   │   ├── dataset.py                      # WIDER FACE loader
│   │   ├── eval.py                         # mAP/Recall evaluation
│   │   ├── inference.py                    # Detection inference
│   │   ├── losses.py                       # Multi-task loss
│   │   ├── model.py                        # Backbone factory
│   │   └── train.py                        # Training script
│   │
│   ├── segmentation/                       # Stage 2: U-Net
│   │   ├── unet_model.py                   # Main U-Net (matches checkpoint)
│   │   ├── unet.py                         # ResNet-encoder variant
│   │   ├── model.py                        # Model factory
│   │   ├── dataset.py                      # CelebAMask-HQ loader
│   │   ├── eval.py                         # IoU/Dice/PA evaluation
│   │   ├── inference.py                    # Segmentation inference
│   │   ├── losses.py                       # BCE + Dice loss
│   │   └── train.py                        # Training script
│   │
│   ├── pipeline/                           # Orchestrator
│   │   ├── orchestrator.py                 # FaceSegmentationPipeline
│   │   ├── stages.py                       # Detection + segmentation stages
│   │   ├── visualizer.py                   # Overlay rendering
│   │   └── run.py                          # CLI entrypoint
│   │
│   ├── utils/                              # Shared utilities
│   │   ├── box_ops.py                      # xyxy/xywh, encode/decode, IoU
│   │   ├── mask_ops.py                     # Morphology, paste_mask
│   │   ├── nms.py                          # NMS + batched_NMS
│   │   ├── io.py                           # Image I/O, YAML, seeding
│   │   └── visualizer.py                   # Draw boxes/masks
│   │
│   ├── data/                               # Data preprocessing
│   │   ├── preprocess_wider_face.py
│   │   ├── preprocess_celebamask_hq.py
│   │   ├── widerface.py
│   │   ├── celebamask_hq.py
│   │   ├── augmentation.py
│   │   └── dataset_factory.py
│   │
│   ├── configs/                            # YAML configs
│   │   ├── retinaface.yaml
│   │   └── unet.yaml
│   │
│   ├── eval.py                             # Main eval entrypoint
│   └── training/                           # Training utilities
│
├── models/                                 # Trained checkpoints
│   ├── retinaface_final.pth                # 84.6 MB, 22.1M params
│   └── unet_final.pth                      # 118.5 MB, 31.0M params
│
├── scripts/                                # Executable scripts
│   ├── evaluation/
│   │   └── eval_segmentation.py            # Main eval script
│   ├── inference/
│   │   ├── visualize_segmentation.py       # Generate visualizations
│   │   └── demo.py
│   ├── kaggle/
│   │   └── end_to_end_smoke_test.py        # Pipeline smoke test
│   └── ...
│
├── data/
│   ├── raw/                                # Original datasets
│   └── processed/
│       ├── detection/{train,val,test}/
│       └── segmentation/{train,val,test}/
│
├── runs/                                   # Runtime outputs
│   ├── evaluation/                         # Eval metrics JSONs
│   └── visualizations/                     # Sample overlay PNGs
│
├── tests/                                  # 67 unit tests
├── docs/                                   # Documentation
└── notebooks/                              # Jupyter notebooks
```

---

## 📊 Data Pipeline

### Input datasets:
| Dataset | Purpose | Split | Count |
|---------|---------|-------|-------|
| **WIDER FACE** | Detection training | train/val/test | 32K images, 393K faces |
| **CelebAMask-HQ** | Segmentation training | train/val/test | 30K images |

### Preprocessing:
1. **WIDER FACE** → `data/processed/detection/{split}/{images,annotations.csv}`
   - Images: PNG, 1024×1024 max
   - Annotations: `image_id, x_min, y_min, x_max, y_max, confidence` (CSV)
   - 80/10/10 split, SEED=42
2. **CelebAMask-HQ** → `data/processed/segmentation/{split}/{images,masks}`
   - Images: JPG, 256×256
   - Masks: PNG, single-channel (0/255 binary)
   - 24K/3K/3K split

---

## 🚀 Inference Flow

### CLI usage:
```bash
# Single image inference
python -m src.pipeline.run --image path/to/image.jpg --output output.png

# Pipeline smoke test (both stages)
python scripts/kaggle/end_to_end_smoke_test.py

# Segmentation evaluation
python scripts/evaluation/eval_segmentation.py --split test --max-samples 100

# Visualization generation
python scripts/inference/visualize_segmentation.py --num-samples 8 --summary
```

### Programmatic usage:
```python
from src.pipeline.orchestrator import FaceSegmentationPipeline
import cv2

# Initialize with trained checkpoints
pipeline = FaceSegmentationPipeline(
    detector_weights="models/retinaface_final.pth",
    segmentor_weights="models/unet_final.pth",
    device="cpu",
)

# Run on image
result = pipeline.run("path/to/image.jpg")
print(f"Found {len(result.boxes)} faces")
print(f"Generated {len(result.masks)} masks")
cv2.imwrite("overlay.png", result.overlay)
```

---

## 🎯 Performance Summary

### Segmentation (U-Net, CelebAMask-HQ)
| Metric | Test (100 samples) | Val (100 samples) | Target |
|--------|---------------------|-------------------|--------|
| Mean IoU | **0.9679** | **0.9666** | ≥ 0.90 ✅ |
| Mean Dice | **0.9834** | **0.9826** | ≥ 0.95 ✅ |
| Pixel Accuracy | **0.9770** | **0.9756** | ≥ 0.97 ✅ |
| F1 (face) | 0.9834 | 0.9826 | — |

### Detection (RetinaFace, WIDER FACE)
| Metric | Value | Notes |
|--------|-------|-------|
| Forward pass (640×640) | 0.56s | CPU |
| Detector run (post-process) | 0.44s | CPU |
| Top confidence score | 1.42 | max(cls_logits) |
| Verdict (smoke test) | OK | end-to-end forward pass |

### Pipeline end-to-end
| Stage | Time | Notes |
|-------|------|-------|
| Detection | ~0.56s | per 640×640 image |
| Segmentation | ~0.66s | per face crop |
| Total (1 image, 1 face) | ~1.2s | CPU |

---

## 🧪 Testing

- **67 unit tests** (`tests/`) — all passing
- **CI pipeline** (`.github/workflows/ci.yml`) — lint + format + test on Python 3.10/3.11/3.12
- **Smoke test** — `scripts/kaggle/end_to_end_smoke_test.py`
- **Real eval** — `scripts/evaluation/eval_segmentation.py` (100 samples)

---

## 📚 References

- **RetinaFace paper:** [Deng et al., CVPR 2020](https://arxiv.org/abs/1905.00641)
- **U-Net paper:** [Ronneberger et al., MICCAI 2015](https://arxiv.org/abs/1505.04597)
- **WIDER FACE:** [http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/)
- **CelebAMask-HQ:** [https://github.com/switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
