# Face Detection & Face Segmentation

**Real-time face detection and segmentation system for security cameras and crowd monitoring applications**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Data%20Preprocessing%20Complete-green)](./PREPROCESSING_COMPLETE_FINAL.md)

---

## 📋 Description

This project tackles two critical computer vision tasks:

1. **Face Detection**: Detect multiple faces in crowded scenes with high accuracy
2. **Face Segmentation**: Generate precise face masks for detected faces

**Target Application**: Security camera systems requiring real-time face detection and segmentation in crowds.

---

## 🎯 Goal & Purpose

### Primary Objectives
- Detect faces in crowded scenes with **>95% accuracy**
- Generate precise segmentation masks with **>85% IoU**
- Achieve **real-time inference** (<50ms per frame)
- Handle challenging conditions: occlusion, lighting variations, multiple scales

### Success Criteria
| Metric | Target | Current |
|--------|--------|---------|
| Detection mAP@0.5 | >0.95 | TBD |
| Segmentation IoU | >0.85 | TBD |
| Inference Time | <50ms | TBD |
| FPS (Full Pipeline) | >20 | TBD |

---

## 🏗️ Pipeline Architecture

```
Input Image
    ↓
Preprocessing (Resize, Normalize)
    ↓
Face Detection Model
    ├─→ Bounding Boxes
    └─→ Confidence Scores
    ↓
Face Segmentation Model
    ├─→ Face Masks (19 parts)
    └─→ Segmentation Map
    ↓
Post-processing (NMS, Filtering)
    ↓
Output: Detected Faces + Segmentation Masks
```

### Why This Architecture?

**Detection Stage:**
- Uses state-of-the-art detection model
- Trained on WIDER FACE dataset (diverse scenes)

**Segmentation Stage:**
- Encoder-decoder architecture
- Trained on CelebAMask-HQ (19 facial parts)

---

## 📊 Dataset Summary

### ✅ Preprocessing Complete (Sep 4, 2026)

| Dataset | Images | Splits | Status |
|---------|--------|--------|--------|
| **CelebAMask-HQ** | 30,000 | 70/15/15 | ✅ Complete |
| **WIDER FACE** | 11,030 | Train/Val | ✅ Complete |

**Total Processed**: 41,030 images (~8.5 GB)

#### CelebAMask-HQ Details
- **Purpose**: Face segmentation training
- **Quality**: 100% retention, 71% avg mask coverage
- **Parts**: 19 facial components (skin, eyes, nose, lips, hair, etc.)
- **Splits**: 21,087 train / 4,454 val / 4,459 test

#### WIDER FACE Details
- **Purpose**: Face detection training
- **Quality**: 68.48% retention after aggressive filtering
- **Faces**: 67,088 valid faces (53,616 train / 13,472 val)
- **Filtering**: Removed blur, small faces (<10px), invalid boxes

See [PREPROCESSING_COMPLETE_FINAL.md](./PREPROCESSING_COMPLETE_FINAL.md) for detailed statistics.

---

## 📁 Project Structure

```
Face Detection & Face Segmentation/
│
├── src/
│   ├── detection/                          # Face Detection
│   │   ├── model.py                       # Detection model (DSFD-based)
│   │   ├── losses.py                      # Loss functions (Focal, SmoothL1, GIoU)
│   │   └── __init__.py
│   │
│   ├── segmentation/                       # Face Segmentation
│   │   ├── model.py                       # Segmentation model (FCN8s-based)
│   │   ├── unet.py                        # Alternative segmentation model (U-Net)
│   │   ├── losses.py                      # Loss functions (Dice, Focal, Combined)
│   │   └── __init__.py
│   │
│   ├── training/                          # Training scripts
│   │   ├── train_detection.py             # Train detection model
│   │   ├── train_segmentation.py          # Train segmentation model
│   │   └── __init__.py
│   │
│   ├── evaluation/                        # Evaluation
│   │   ├── metrics.py                     # Metric calculations (mAP, IoU, Dice)
│   │   └── __init__.py
│   │
│   └── inference/                         # Inference scripts
│       ├── detector.py                    # Detection inference
│       ├── segmentor.py                   # Segmentation inference
│       ├── pipeline.py                    # End-to-end pipeline
│       └── batch_inference.py             # Batch processing
│
├── configs/                               # Configuration files
│   ├── detection_config.yaml              # Detection hyperparameters
│   ├── segmentation_config.yaml           # Segmentation hyperparameters
│   └── __init__.py                        # Config loader
│
├── scripts/                               # Executable scripts
│   ├── demo.py                            # Demo inference (image/video/webcam)
│   ├── run_preprocessing.py               # Run data preprocessing
│   ├── validate_preprocessing.py          # Validate preprocessing outputs
│   └── test_preprocessing_quick.py        # Quick preprocessing test
│
├── outputs/                               # Output results
│   ├── logs/                              # Training logs (TensorBoard)
│   ├── metrics/                           # Evaluation metrics (JSON/CSV)
│   └── visualizations/                    # Visualization results
│
├── docs/                                  # Documentation
│   ├── DATA_PREPROCESSING_SUMMARY.md
│   ├── HUONG_DAN_TAI_DATASET.md
│   └── DATASET_DOWNLOAD_GUIDE.md
│
├── data/
│   ├── raw/                               # Original datasets
│   └── processed/                         # Preprocessed data
│       ├── celebamask_hq/
│       └── wider_face/
│
├── requirements.txt                       # Dependencies
├── requirements_preprocessing.txt         # Preprocessing-only dependencies
├── quick_run.bat                          # Quick setup & run script (Windows)
├── HUONG_DAN_CHAY_MODEL.md                # Usage guide (Vietnamese)
├── TEAM_WORK_PLAN.md                      # Team task assignments
└── README.md                              # This file
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd "Face Detection & Face Segmentation"

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing (✅ Already Complete)

```bash
# CelebAMask-HQ preprocessing
python src/data/preprocess_celebamask_hq.py

# WIDER FACE preprocessing
python src/data/preprocess_wider_face.py
```

### 3. Model Training (Next Step)

```bash
# Train detection model
python src/training/train_detection.py --config configs/config.yaml

# Train segmentation model
python src/training/train_segmentation.py --config configs/config.yaml
```

### 4. Evaluation

```bash
# Evaluate detection model
python scripts/evaluate.py --task detection --checkpoint weights/detection_best.pth

# Evaluate segmentation model
python scripts/evaluate.py --task segmentation --checkpoint weights/segmentation_best.pth
```

---

## 🔧 Dependencies

### Core Libraries
- Python 3.8+
- PyTorch 2.0+
- torchvision
- OpenCV (cv2)
- NumPy
- Pillow (PIL)

### Additional Tools
- tqdm (progress bars)
- matplotlib (visualization)
- seaborn (plots)
- tensorboard (training logs)
- albumentations (data augmentation)

See `requirements.txt` for complete list.

---

## 📈 Implementation Plan

### ✅ Phase 1: Data Preparation (COMPLETE)
- [x] Download datasets (CelebAMask-HQ, WIDER FACE)
- [x] Preprocess CelebAMask-HQ (30,000 images + masks)
- [x] Preprocess WIDER FACE (11,030 images)
- [x] Quality filtering (blur, size, validation)
- [x] Create train/val/test splits
- [x] Generate statistics and documentation

### 🔄 Phase 2: Model Development (IN PROGRESS)
- [ ] Implement detection model
  - [ ] Model architecture
  - [ ] Loss functions
  - [ ] Training pipeline
- [ ] Implement segmentation model
  - [ ] Model architecture
  - [ ] Loss functions
  - [ ] Training pipeline

### ⏳ Phase 3: Training & Optimization
- [ ] Train detection model on WIDER FACE
  - [ ] Target: mAP@0.5 > 0.95
- [ ] Train FCN model on CelebAMask-HQ
  - [ ] Target: IoU > 0.90
- [ ] Hyperparameter tuning
- [ ] Model optimization (pruning, quantization)
- [ ] Validation and early stopping

### ⏳ Phase 4: Evaluation & Testing
- [ ] Model-level evaluation
  - Detection: mAP, precision, recall
  - Segmentation: IoU, dice coefficient
- [ ] Pipeline-level evaluation
  - End-to-end latency
  - Throughput (FPS)
  - Resource usage
- [ ] Edge case testing
  - Occlusion, lighting, multiple faces
  - Small faces, profile views

### ⏳ Phase 5: Demo & Deployment
- [ ] Build demo interface
- [ ] Real-time video pipeline
- [ ] Performance profiling
- [ ] Documentation and user guide

---

## 🧪 Evaluation Framework

### Model-Level Metrics

**Detection Model:**
- mAP@0.5, mAP@0.75
- Precision, Recall, F1
- Per-class performance
- Inference time per image

**Segmentation Model:**
- Mean IoU (19 classes)
- Pixel accuracy
- Dice coefficient
- Boundary F-score
- Inference time per image

### Pipeline-Level Metrics

**End-to-End Performance:**
- Total latency (preprocessing + detection + segmentation + postprocessing)
- Throughput (frames per second)
- Resource utilization (CPU, GPU, RAM)
- Failure rate and error handling

**Target Benchmarks:**
| Metric | Target | Hardware |
|--------|--------|----------|
| E2E Latency | <100ms | RTX 3060 |
| FPS | >20 | RTX 3060 |
| GPU Memory | <4GB | RTX 3060 |

---

## 🔬 Technical Decisions & Rationale

### Why This Architecture?

**Detection:**
- Optimized for face detection tasks
- Proven on WIDER FACE benchmark

**Segmentation:**
- Encoder-decoder architecture
- Accurate boundary detection
- **Proven baselines**: Both models have proven results
- **Flexibility**: Can swap models or fine-tune separately
- **Optimization potential**: Can quantize/prune for deployment

---

## 📚 References

### Datasets
- **WIDER FACE**: Yang, S., Luo, P., Loy, C. C., & Tang, X. (2016). WIDER FACE: A Face Detection Benchmark
- **CelebAMask-HQ**: Lee, C. H., Liu, Z., Wu, L., & Luo, P. (2020). MaskGAN: Towards Diverse and Interactive Facial Image Manipulation

---

## 📝 Naming Conventions

### Models
- Format: `{task}_{architecture}_{version}.{ext}`
- Example: `detection_model_v1.pt`, `segmentation_model_v2.pth`

### Datasets
- Format: `{purpose}_{date}.{ext}`
- Example: `training_faces_20260904.zip`

### Scripts
- Format: `{action}_{target}.py`
- Example: `train_model.py`, `evaluate_pipeline.py`

### Configs
- Format: `{component}_config.yaml`
- Example: `detection_config.yaml`, `training_config.yaml`

---

## 🤝 Contributing

This is an academic project for computer vision research and education.

---

## 📄 License

[Add your license here]

---

## 📞 Contact

[Add contact information]

---

## 🗓️ Project Timeline

| Phase | Start | End | Status |
|-------|-------|-----|--------|
| Data Preparation | Sep 3, 2026 | Sep 4, 2026 | ✅ Complete |
| Model Development | Sep 4, 2026 | TBD | 🔄 In Progress |
| Training | TBD | TBD | ⏳ Pending |
| Evaluation | TBD | TBD | ⏳ Pending |
| Demo & Deployment | TBD | TBD | ⏳ Pending |

---

**Last Updated**: Sep 4, 2026 7:37 PM (UTC+7)  
**Status**: ✅ Data Preprocessing Complete → 🔄 Model Development Phase → 👥 Team Assigned

---

## 👥 TEAM STRUCTURE

See [TEAM_WORK_PLAN.md](./TEAM_WORK_PLAN.md) for detailed task assignments.

| Member | Role | Responsibility |
|--------|------|----------------|
| **Member 1** | Detection | Detection Model + Training (WIDER FACE) |
| **Member 2** | Segmentation | Segmentation Model + Training (CelebAMask-HQ) |
| **Member 3** | Data | Data Pipeline + Loaders + Augmentation |
| **Member 4** | Integration | Evaluation + Inference Pipeline + Demo |

### Models
- **Detection**: State-of-the-art face detection
- **Segmentation**: Encoder-decoder segmentation (19 parts)
