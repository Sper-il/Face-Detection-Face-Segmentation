# Face Detection & Face Segmentation

**Real-time face detection and segmentation system using RetinaFace + U-Net for security cameras and crowd monitoring applications.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Models%20Trained-brightgreen)](./docs/status/progress_status.md)

---

## 🎯 Model Performance

### Trained Checkpoints
| Model | File | Size | Params |
|-------|------|------|--------|
| **RetinaFace** (Detection) | `models/retinaface_final.pth` | 84.6 MB | 22.2M |
| **U-Net** (Segmentation) | `models/unet_final.pth` | 118.5 MB | 31.0M |

### Evaluation Results

#### Segmentation (U-Net)
| Metric | Test Set | Validation Set |
|--------|----------|----------------|
| **Mean IoU** | **0.9660** | **0.9766** |
| **Mean Dice** | **0.9824** | **0.9880** |
| **Pixel Accuracy** | 0.9756 | 0.9834 |

#### Detection (RetinaFace)
| Metric | Value |
|--------|-------|
| Forward Pass (640×640) | ~0.98s (CPU) |
| Smoke Test | ✅ Pass |

See [docs/references/DANH_GIA_MODEL.md](./docs/references/DANH_GIA_MODEL.md) for detailed evaluation.

---

## 📁 Project Structure

```
Face-Detection-Face-Segmentation/
│
├── src/                                 # Source code
│   ├── configs/                        # Config loaders
│   ├── data/                           # Dataset classes & loaders
│   ├── detection/                      # RetinaFace detection model
│   │   ├── retinaface.py              # Main model
│   │   ├── backbone.py                # Backbone networks
│   │   ├── anchors.py                 # Anchor generation
│   │   └── losses.py                  # Detection losses
│   ├── segmentation/                   # Segmentation models
│   │   ├── unet.py                    # U-Net implementation
│   │   ├── unet_model.py              # UNet with ResNet encoder
│   │   └── losses.py                  # Segmentation losses
│   ├── pipeline/                       # End-to-end pipeline
│   │   ├── orchestrator.py            # Pipeline coordinator
│   │   ├── stages.py                  # Processing stages
│   │   └── visualizer.py             # Visualization
│   ├── inference/                      # Inference utilities
│   ├── evaluation/                    # Evaluation metrics
│   ├── training/                      # Training utilities
│   └── utils/                         # Common utilities
│
├── models/                             # Trained model checkpoints
│   ├── retinaface_final.pth           # Detection model
│   └── unet_final.pth                 # Segmentation model
│
├── configs/                            # Configuration files
│   ├── detection_config.yaml
│   └── segmentation_config.yaml
│
├── scripts/                            # Executable scripts
│   ├── preprocessing/                 # Data download & preprocessing
│   │   ├── run_preprocessing.py
│   │   ├── validate_preprocessing.py
│   │   └── data_audit.py
│   ├── inference/                     # Inference scripts
│   │   ├── demo.py                    # CLI demo
│   │   ├── visualize_segmentation.py
│   │   └── inference_retinaface_yakhyo.py
│   ├── evaluation/                    # Model evaluation
│   │   ├── eval_segmentation.py
│   │   ├── generate_report_figures.py
│   │   └── test_unet_image.py
│   ├── checkpoints/                   # Checkpoint utilities
│   ├── diagrams/                      # Pipeline diagrams
│   ├── kaggle/                       # Kaggle integration
│   │   ├── sync_results.py
│   │   ├── check_kaggle_status.py
│   │   └── end_to_end_smoke_test.py
│   └── misc/                         # Utilities
│       └── update_progress_status.py
│
├── tests/                              # Unit tests (67 tests)
│   ├── test_detection_inference.py
│   ├── test_segmentation_inference.py
│   ├── test_pipeline.py
│   ├── test_export.py                 # ONNX export tests
│   └── ...
│
├── data/                               # Data directory
│   ├── raw/                          # Original datasets
│   └── processed/                    # Preprocessed data
│       ├── detection/                # WIDER FACE (train/val/test)
│       └── segmentation/              # CelebAMask-HQ (train/val/test)
│
├── runs/                              # Runtime outputs
│   ├── evaluation/                   # Evaluation results
│   │   ├── segmentation_test_metrics.json
│   │   ├── pipeline_smoke_test.json
│   │   └── visualizations/
│   └── visualizations/               # Prediction visualizations
│       ├── detection/
│       └── test/
│
├── notebooks/                         # Jupyter notebooks
│   └── eval_100_samples.ipynb         # 100-sample evaluation
│
├── outputs/                           # Output artifacts
│
├── docs/                              # Documentation
│   ├── guides/                       # User guides
│   │   ├── DATASET_DOWNLOAD_GUIDE.md
│   │   ├── HUONG_DAN_CHAY_MODEL.md
│   │   ├── HUONG_DAN_TAI_DATASET.md
│   │   └── TEMPLATE_HUONG_DAN.md
│   ├── planning/                     # Project plans
│   │   ├── PLAN.md
│   │   ├── ROADMAP.md
│   │   └── TEAM_WORK_PLAN.md
│   ├── status/                       # Status & logs
│   │   ├── progress_status.md
│   │   └── AI_USAGE.md
│   ├── references/                   # Research & references
│   │   ├── DANH_GIA_MODEL.md
│   │   ├── DATA_PREPROCESSING_SUMMARY.md
│   │   └── survey.md
│   ├── adr/                         # Architecture Decision Records
│   ├── audit/                       # Audit reports
│   └── plan/                        # Implementation details
│
├── configs/                          # YAML configs
├── .github/                          # GitHub Actions CI
├── .venv/                           # Python virtual environment
│
├── pyproject.toml                    # Python project config
├── requirements.txt                  # Dependencies
├── requirements_preprocessing.txt    # Preprocessing deps
├── quick_run.bat                     # Quick start script (Windows)
└── README.md                         # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and enter repo
git clone https://github.com/Sper-il/Face-Detection-Face-Segmentation.git
cd Face-Detection-Face-Segmentation

# Create and activate venv (Windows)
python -m venv .venv
.venv\Scripts\activate

# Or on Linux/Mac
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Inference Demo

```bash
# Segment faces on test images
python scripts/inference/demo.py --input data/processed/segmentation/test/images/00007.jpg

# Visualize segmentation results
python scripts/inference/visualize_segmentation.py --num-samples 8

# Run pipeline smoke test
python scripts/kaggle/end_to_end_smoke_test.py
```

### 3. Evaluate Models

```bash
# Evaluate segmentation on test set
python scripts/evaluation/eval_segmentation.py --split test --max-samples 100

# Evaluate on validation set
python scripts/evaluation/eval_segmentation.py --split val
```

---

## 📊 Dataset Summary

| Dataset | Images | Purpose | Status |
|---------|--------|---------|--------|
| **CelebAMask-HQ** | 30,000 | Face Segmentation | ✅ Ready |
| **WIDER FACE** | 11,030 | Face Detection | ✅ Ready |

See [docs/references/DATA_PREPROCESSING_SUMMARY.md](./docs/references/DATA_PREPROCESSING_SUMMARY.md) for details.

---

## 🏗️ Pipeline Architecture

```
Input Image → Preprocessing → RetinaFace (Detection) → Bounding Boxes
                                                        ↓
                                    U-Net (Segmentation) ← Face Crops
                                                        ↓
                                    Output: Bboxes + Face Masks + Overlay
```

### Model Details

**RetinaFace (Detection)**
- Multi-scale feature extraction
- Anchor-based detection heads
- Landmark prediction (5 points per face)

**U-Net (Segmentation)**
- ResNet-34 encoder (pretrained)
- Skip connections for boundary preservation
- 19-class facial part segmentation

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test
pytest tests/test_pipeline.py -v
```

---

## 📈 Training History

| Model | Epochs | Final Loss | Best Metric |
|-------|--------|------------|-------------|
| RetinaFace | 100 | TBD | mAP@0.5 > 0.95 (target) |
| U-Net | 100 | TBD | IoU 0.9766 (val) |

---

## 📝 Documentation

| Document | Path | Description |
|----------|------|-------------|
| Evaluation Report | `docs/references/DANH_GIA_MODEL.md` | Model evaluation results |
| Progress Status | `docs/status/progress_status.md` | Project progress |
| Running Guide | `docs/guides/HUONG_DAN_CHAY_MODEL.md` | How to run models |
| Dataset Guide | `docs/guides/DATASET_DOWNLOAD_GUIDE.md` | Data download instructions |

---

## 🤝 Team

See [docs/planning/TEAM_WORK_PLAN.md](./docs/planning/TEAM_WORK_PLAN.md) for team assignments.

---

**Last Updated**: September 23, 2026  
**Status**: ✅ Models Trained → 📊 Evaluation Complete
