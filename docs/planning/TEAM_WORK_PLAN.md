# 🔬 PHÂN CÔNG CÔNG VIỆC - Face Detection & Face Segmentation

**Dự án:** Face Detection & Face Segmentation  
**Ngày:** 4 tháng 9, 2026  
**Cập nhật:** 5 tháng 9, 2026  
**Trạng thái:** ✅ Tiền xử lý dữ liệu hoàn tất → 🔄 Giai đoạn phát triển mô hình

---

## 📋 PHÂN CÔNG 4 NGƯỜI CÔNG VIỆC

### 👤 Người 1: MODEL - Phát triển kiến trúc mô hình
| Thành viên | File | Trách nhiệm |
|-----------|------|-------------|
| **M1-Det** | `src/detection/model.py` | Detection model architecture |
| **M1-Det** | `src/detection/losses.py` | Detection loss functions |
| **M1-Seg** | `src/segmentation/model.py` | Segmentation model architecture |
| **M1-Seg** | `src/segmentation/unet.py` | U-Net architecture (alternative) |
| **M1-Seg** | `src/segmentation/losses.py` | Segmentation loss functions |

### 👤 Người 2: TRAIN - Script huấn luyện
| Thành viên | File | Trách nhiệm |
|-----------|------|-------------|
| **M2-Det** | `src/training/train_detection.py` | Detection training pipeline |
| **M2-Seg** | `src/training/train_segmentation.py` | Segmentation training pipeline |

### 👤 Người 3: EVALUATE - Đánh giá model
| Thành viên | File | Trách nhiệm |
|-----------|------|-------------|
| **M3** | `src/evaluation/metrics.py` | Metrics (mAP, IoU, Dice, etc.) |

### 👤 Người 4: UI - Giao diện & Demo
| Thành viên | File | Trách nhiệm |
|-----------|------|-------------|
| **M4** | `src/inference/detector.py` | Detection inference |
| **M4** | `src/inference/segmentor.py` | Segmentation inference |
| **M4** | `src/inference/pipeline.py` | End-to-end pipeline |
| **M4** | `src/inference/batch_inference.py` | Batch processing |
| **M4** | `scripts/inference/demo.py` | Demo CLI application |

---

## 📁 CẤU TRÚC FILE PROJECT

```
face_detection_segmentation/
│
├── src/
│   ├── __init__.py
│   │
│   ├── detection/                          # 👤 Người 1 - MODEL
│   │   ├── __init__.py
│   │   ├── model.py                        # Detection model architecture
│   │   └── losses.py                       # Detection loss functions
│   │
│   ├── segmentation/                       # 👤 Người 1 - MODEL
│   │   ├── __init__.py
│   │   ├── model.py                        # Segmentation model
│   │   ├── unet.py                         # U-Net architecture
│   │   └── losses.py                       # Segmentation losses
│   │
│   ├── data/                               # ✅ Đã hoàn thành
│   │   ├── __init__.py
│   │   ├── widerface.py                   # WIDER FACE dataset
│   │   ├── celebamask_hq.py               # CelebAMask-HQ dataset
│   │   ├── augmentation.py                # Data augmentation
│   │   ├── dataset_factory.py             # Dataset factory
│   │   ├── preprocess_wider_face.py       # Preprocessing
│   │   ├── preprocess_celebamask_hq.py    # Preprocessing
│   │   └── visualize_data.py              # Data visualization
│   │
│   ├── training/                           # 👤 Người 2 - TRAIN
│   │   ├── __init__.py
│   │   ├── train_detection.py             # Train detection model
│   │   └── train_segmentation.py          # Train segmentation model
│   │
│   ├── evaluation/                        # 👤 Người 3 - EVALUATE
│   │   ├── __init__.py
│   │   └── metrics.py                    # Metric calculations (mAP, IoU, Dice)
│   │
│   ├── inference/                         # 👤 Người 4 - UI
│   │   ├── __init__.py
│   │   ├── detector.py                   # Detection inference
│   │   ├── segmentor.py                  # Segmentation inference
│   │   ├── pipeline.py                  # End-to-end pipeline
│   │   └── batch_inference.py           # Batch processing
│
├── configs/                               # Cấu hình
│   ├── __init__.py                       # Config loader
│   ├── detection_config.yaml            # Detection hyperparameters
│   └── segmentation_config.yaml         # Segmentation hyperparameters
│
├── scripts/                              # Scripts chạy
│   ├── demo.py                          # Demo inference
│   ├── run_preprocessing.py            # Run preprocessing pipeline
│   ├── validate_preprocessing.py        # Validate preprocessing outputs
│   └── test_preprocessing_quick.py      # Quick preprocessing test
│
├── data/                                 # Data (đã preprocess)
│   ├── processed/
│   │   ├── wider_face/
│   │   └── celebamask_hq/
│   └── raw/                            # Raw data
│
├── notebooks/                           # Jupyter notebooks
│   ├── analysis.ipynb                   # Data analysis
│   └── visualization.ipynb             # Result visualization
│
├── outputs/                            # Output results
│   ├── logs/                          # Training logs
│   ├── metrics/                      # Evaluation metrics
│   └── visualizations/               # Visualization results
│
├── requirements.txt                    # Dependencies
├── setup.py                          # Setup file
├── README.md                         # Documentation
└── TEAM_WORK_PLAN.md                # This file
```

---

## 📋 NHIỆM VỤ CHI TIẾT THEO NGƯỜI

---

### 👤 Người 1: MODEL - Phát triển kiến trúc mô hình

#### 1A. Detection Model (M1-Det)
**Trách nhiệm:** Implement face detection model architecture

**Files cần làm:**
| File | Mô tả | Trạng thái |
|------|-------|------------|
| `src/detection/model.py` | Detection model architecture | ⬜ Chưa làm |
| `src/detection/losses.py` | Loss functions (Focal, L1/L2, GIoU) | ⬜ Chưa làm |

**Yêu cầu:**
- Sử dụng pretrained backbone (ResNet, MobileNet, etc.)
- Output: bounding boxes, confidence scores
- Support multi-scale detection
- Non-Maximum Suppression (NMS) integration

#### 1B. Segmentation Model (M1-Seg)
**Trách nhiệm:** Implement face segmentation model architecture

**Files cần làm:**
| File | Mô tả | Trạng thái |
|------|-------|------------|
| `src/segmentation/model.py` | Main segmentation model | ⬜ Chưa làm |
| `src/segmentation/unet.py` | U-Net architecture (alternative) | ⬜ Chưa làm |
| `src/segmentation/losses.py` | Loss functions (CE, Dice, Focal) | ⬜ Chưa làm |

**Yêu cầu:**
- Output: 19-class segmentation mask (CelebAMask-HQ classes)
- Encoder-Decoder architecture
- Skip connections for fine details
- Multiple loss combination support

---

### 👤 Người 2: TRAIN - Script huấn luyện

#### 2A. Detection Training (M2-Det)
**Trách nhiệm:** Training pipeline cho detection model

**File cần làm:**
| File | Mô tả | Trạng thái |
|------|-------|------------|
| `src/training/train_detection.py` | Detection training script | ⬜ Chưa làm |

**Yêu cầu:**
- Load data từ `src/data/widerface.py`
- Mixed precision training (AMP)
- Learning rate scheduling (Cosine Annealing, etc.)
- Model checkpointing
- TensorBoard/MLFlow logging
- Evaluation on validation set

#### 2B. Segmentation Training (M2-Seg)
**Trách nhiệm:** Training pipeline cho segmentation model

**File cần làm:**
| File | Mô tả | Trạng thái |
|------|-------|------------|
| `src/training/train_segmentation.py` | Segmentation training script | ⬜ Chưa làm |

**Yêu cầu:**
- Load data từ `src/data/celebamask_hq.py`
- Mixed precision training (AMP)
- Learning rate scheduling
- Model checkpointing
- TensorBoard/MLFlow logging
- Evaluation (mIoU calculation)

---

### 👤 Người 3: EVALUATE - Đánh giá model

#### 3A. Metrics (M3)
**Trách nhiệm:** Implementation các metrics đánh giá

**File cần làm:**
| File | Mô tả | Trạng thái |
|------|-------|------------|
| `src/evaluation/metrics.py` | Metrics calculations | ⬜ Chưa làm |

**Metrics cần implement:**
- **Detection:** IoU, mAP@0.5, mAP@[0.5:0.95], Precision, Recall, F1
- **Segmentation:** Pixel Accuracy, mIoU, Dice Coefficient, Confusion Matrix

#### 3B. Dataset Evaluation (M3)
**Trách nhiệm:** Evaluate trên các benchmark datasets

**Files cần làm:**
| File | Mô tả | Trạng thái |
|------|-------|------------|
| `src/evaluation/widerface_eval.py` | WIDER FACE evaluation | ✅ Đã làm |
| `src/evaluation/fddb_eval.py` | FDDB evaluation | ✅ Đã làm |
| `src/evaluation/visualization.py` | Result visualization | ✅ Đã làm |

**Yêu cầu:**
- WIDER FACE: mAP theo difficulty (Easy, Medium, Hard)
- FDDB: ROC curve, format output theo chuẩn
- Visualization: So sánh GT vs Predictions

---

### 👤 Người 4: UI - Giao diện & Demo

#### 4A. Inference (M4)
**Trách nhiệm:** Inference scripts cho model

**Files cần làm:**
| File | Mô tả | Trạng thái |
|------|-------|------------|
| `src/inference/detector.py` | Detection inference | ⬜ Chưa làm |
| `src/inference/segmentor.py` | Segmentation inference | ⬜ Chưa làm |
| `src/inference/pipeline.py` | End-to-end pipeline | ⬜ Chưa làm |
| `src/inference/batch_inference.py` | Batch processing | ⬜ Chưa làm |

**Yêu cầu:**
- Load trained models
- Image/Video input support
- Batch processing
- Progress bar cho batch

#### 4B. Demo Application (M4)
**Trách nhiệm:** Demo CLI application

**File cần làm:**
| File | Mô tả | Trạng thái |
|------|-------|------------|
| `scripts/inference/demo.py` | Demo script | ⬜ Chưa làm |

**Yêu cầu:**
```bash
# Image
python scripts/inference/demo.py --image input.jpg --output output.jpg

# Video
python scripts/inference/demo.py --video input.mp4 --output output.mp4

# Webcam
python scripts/inference/demo.py --webcam

# Batch folder
python scripts/inference/demo.py --folder input/ --output output/
```
---

## 📝 TIMELINE & MILESTONES

### Phase 1: Data Preparation ✅ COMPLETE (Sep 4, 2026)
- Preprocessing scripts
- Dataset validation
- Statistics generation

### Phase 2: Model Development 🔄 IN PROGRESS
- Detection model implementation
- Segmentation model implementation
- Training pipelines

### Phase 3: Training & Optimization ⏳ PENDING
- Model training
- Hyperparameter tuning
- Performance optimization

### Phase 4: Evaluation & Testing ⏳ PENDING
- Model evaluation
- Pipeline testing
- Performance benchmarking

### Phase 5: Demo & Deployment ⏳ PENDING
- Demo application
- Documentation
- Deployment package

---

## 📚 REFERENCES

### Datasets
- WIDER FACE: Face detection benchmark
- CelebAMask-HQ: Face segmentation with 19 parts

---

**Last Updated:** Sep 5, 2026  
**Status:** Data preprocessing complete, model development in progress
weight_decay: 0.0001

# Data
img_size: 512
dataset: celebamask_hq  # or custom
train_split: train
val_split: val

# Loss
loss_type: combined
ce_weight: 0.5
dice_weight: 0.5
```

---

## 📝 TIMELINE & MILESTONES

#### 4.7 `src/inference/pipeline.py` - End-to-End Pipeline
**Chức năng:**
- Combine Detection + Segmentation
- **Pipeline flow:**
  ```
  Image → Detection → [Face 1, Face 2, ...] 
                      ↓
              For each face:
                Crop → Segment → Mask
                      ↓
              Combine results
  ```
- Handle multiple faces
- Return: boxes + masks + scores

```python
class FacePipeline:
    def __init__(self, detector, segmentor):
        self.detector = detector
        self.segmentor = segmentor
    
    def process(self, image):
        # 1. Detect faces
        # 2. For each face: segment
        # 3. Combine and return
        return {
            'faces': [
                {'bbox': [...], 'mask': [...], 'score': ...},
                ...
            ]
        }
```

#### 4.8 `src/inference/batch_inference.py` - Batch Processing
**Chức năng:**
- Process nhiều ảnh/video
- Multi-threaded loading
- Progress bar
- Save results (JSON + images)

#### 4.9 `scripts/inference/demo.py` - Demo Script
**Chức năng:**
- CLI interface cho inference
- Support ảnh và video
- Real-time webcam mode
- Save output

**Cách chạy:**
```bash
# Ảnh đơn
python scripts/inference/demo.py --image input.jpg --output output.jpg

# Video
python scripts/inference/demo.py --video input.mp4 --output output.mp4

# Webcam
python scripts/inference/demo.py --webcam

# Batch folder
python scripts/inference/demo.py --folder input/ --output output/
```

---

## 🗓️ WORKFLOW VÀ DEPENDENCIES

### Dependency Graph
```
┌─────────────┐     ┌─────────────┐
│  Backbone   │────▶│  Detection  │
│  Network    │     │    Model    │
└─────────────┘     └──────┬──────┘
                           │
┌─────────────┐     ┌──────▼──────┐     ┌─────────────┐
│  Encoder    │────▶│Segmentation │────▶│   Output    │
│  Network    │     │    Model    │     │   Masks     │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │   Dataset   │◀─── Thành viên 3
                    │   Loader    │
                    └─────────────┘
```

### Timeline

```
Tuần 1: Setup + Data Pipeline (Member 3)
┌─────────────────────────────────────────────────┐
│ • Implement datasets (WIDER FACE, CelebAMask-HQ) │
│ • Implement transforms và augmentation           │
│ • Verify data loading                           │
│ • Member 1, 2: Setup backbone pretrained        │
└─────────────────────────────────────────────────┘

Tuần 2: Model Development (Member 1, 2)
┌─────────────────────────────────────────────────┐
│ Member 1:                                        │
│ • Implement detection architecture              │
│ • Implement losses                              │
│ • Implement NMS                                │
│ • Start training script                        │
├─────────────────────────────────────────────────┤
│ Member 2:                                        │
│ • Implement segmentation model                  │
│ • Implement alternative model                   │
│ • Implement segmentation losses                │
│ • Start training script                        │
└─────────────────────────────────────────────────┘

Tuần 3: Training + Evaluation (All)
┌─────────────────────────────────────────────────┐
│ • Train detection model                        │
│ • Train segmentation model                     │
│ • Implement evaluation scripts                │
│ • Member 4: Implement inference pipeline        │
│ • Test end-to-end                             │
└─────────────────────────────────────────────────┘

Tuần 4: Optimization + Demo (All)
┌─────────────────────────────────────────────────┐
│ • Hyperparameter tuning                        │
│ • Model optimization (pruning, quantize)       │
│ • Create demo                                  │
│ • Documentation                                │
│ • Final testing                               │
└─────────────────────────────────────────────────┘
```

---

## 🔧 CÁCH CHẠY PROJECT

### 1. Setup Environment
```bash
# Clone repo
git clone <repo_url>
cd face_detection_segmentation

# Create venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data
```bash
# Download pretrained models
python scripts/download_models.py

# Data đã preprocess trong data/processed/
```

### 3. Train Detection Model
```bash
python src/training/train_dsfd.py --config configs/dsfd_config.yaml
```

### 4. Train Segmentation Model
```bash
python src/training/train_fcn.py --config configs/fcn_config.yaml
```

### 5. Evaluate
```bash
# WIDER FACE
python scripts/eval_widerface.py --checkpoint weights/trained/detection_model.pth

# FDDB
python scripts/eval_fddb.py --checkpoint weights/trained/detection_model.pth
```

### 6. Run Demo
```bash
# Image
python scripts/inference/demo.py --image test.jpg --output result.jpg

# Video
python scripts/inference/demo.py --video test.mp4 --output result.mp4

# Webcam
python scripts/inference/demo.py --webcam
```

---

## ✅ CHECKLIST HOÀN THÀNH

### 👤 Người 1: MODEL
- [ ] `src/detection/model.py` - Detection model architecture
- [ ] `src/detection/losses.py` - Detection loss functions
- [ ] `src/segmentation/model.py` - Segmentation model architecture
- [ ] `src/segmentation/unet.py` - U-Net architecture
- [ ] `src/segmentation/losses.py` - Segmentation loss functions

### 👤 Người 2: TRAIN
- [ ] `src/training/train_detection.py` - Detection training script
- [ ] `src/training/train_segmentation.py` - Segmentation training script

### 👤 Người 3: EVALUATE
- [x] `src/evaluation/metrics.py` - Metrics calculations
- [x] `src/evaluation/widerface_eval.py` - WIDER FACE evaluation (mAP theo Easy/Medium/Hard)
- [x] `src/evaluation/fddb_eval.py` - FDDB evaluation (ROC curve + output format chuẩn)
- [x] `src/evaluation/visualization.py` - So sánh GT vs Prediction, confusion matrix, ROC/PR curve

### 👤 Người 4: UI
- [ ] `src/inference/detector.py` - Detection inference
- [ ] `src/inference/segmentor.py` - Segmentation inference
- [ ] `src/inference/pipeline.py` - End-to-end pipeline
- [ ] `src/inference/batch_inference.py` - Batch processing
- [ ] `scripts/inference/demo.py` - Demo application

---

## 📚 TÀI LIỆU THAM KHẢO

### Papers & Resources
```
# Face Detection
Face detection research and benchmarks

# Face Segmentation
Face segmentation with 19 facial parts

# Datasets
WIDER FACE: Face Detection Benchmark
CelebAMask-HQ: Face Segmentation Dataset
```

---

**Cập nhật lần cuối:** 5 tháng 9, 2026  
**Bước tiếp theo:** Bắt đầu implementation theo 4 nhóm công việc 🚀
