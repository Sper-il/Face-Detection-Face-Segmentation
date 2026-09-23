# Dataset Download Guide

## Overview

This guide helps you download and prepare the datasets for the Face Detection & Segmentation project.

---

## Datasets Required

### 1. WIDER FACE (Detection)
- **Size**: ~2 GB
- **Images**: 32,203 images
- **Faces**: 393,703 labeled faces
- **Purpose**: Training face detection model
- **Difficulty**: Easy/Medium/Hard splits for crowded scenes

### 2. CelebAMask-HQ (Segmentation)
- **Size**: ~20 GB (full) or ~2 GB (resized)
- **Images**: 30,000 images at 1024x1024 (or 256x256 resized)
- **Purpose**: Training face segmentation model
- **Features**: Pixel-level facial attribute masks

---

## Quick Start

### Option 1: Automatic Download (Recommended)

```bash
# Navigate to project root
cd "C:\Users\Admin\Downloads\Face Detection & Face Segmentation"

# Activate virtual environment
venv\Scripts\activate

# Install required packages
pip install requests tqdm gdown

# Download WIDER FACE
python scripts\download_wider_face.py

# Download CelebAMask-HQ
python scripts\download_celebamask.py
```

### Option 2: Manual Download

If automatic download fails or is too slow, download manually:

#### WIDER FACE
1. Visit: http://shuoyang1213.me/WIDERFACE/
2. Download:
   - `WIDER_train.zip` (~1.36 GB)
   - `WIDER_val.zip` (~345 MB)
   - `WIDER_test.zip` (~343 MB)
   - `wider_face_split.zip` (~7 MB - annotations)
3. Place in: `data/raw/WIDER_FACE/`
4. Extract all zip files

#### CelebAMask-HQ

**Full Version (20 GB):**
1. Visit: https://github.com/switchablenorms/CelebAMask-HQ
2. Download from Google Drive links in README
3. Place in: `data/raw/CelebAMask-HQ/`
4. Extract

**Resized Version (2 GB - Recommended for faster start):**
1. Visit: https://www.kaggle.com/datasets/ashishjangra27/celeba-hq-resized-256x256
2. Download: `archive.zip` (~2 GB)
3. Place in: `data/raw/CelebAMask-HQ/`
4. Extract

---

## Expected Folder Structure After Download

```
data/raw/
├── WIDER_FACE/
│   ├── WIDER_train/
│   │   └── images/
│   │       ├── 0--Parade/
│   │       ├── 1--Handshaking/
│   │       └── ... (61 event folders)
│   ├── WIDER_val/
│   │   └── images/
│   │       └── ... (event folders)
│   ├── WIDER_test/
│   │   └── images/
│   │       └── ... (event folders)
│   └── wider_face_split/
│       ├── wider_face_train_bbx_gt.txt
│       ├── wider_face_val_bbx_gt.txt
│       └── readme.txt
│
└── CelebAMask-HQ/
    ├── CelebA-HQ-img/
    │   ├── 0.jpg
    │   ├── 1.jpg
    │   └── ... (30,000 images)
    └── CelebAMask-HQ-mask-anno/
        ├── 0/
        ├── 1/
        └── ... (mask folders)
```

---

## Verification

### Check WIDER FACE

```bash
# Count training images
python -c "import os; print(len([f for root, dirs, files in os.walk('data/raw/WIDER_FACE/WIDER_train') for f in files if f.endswith('.jpg')]))"
# Expected: ~12,880 images

# Check annotation file exists
python -c "import os; print(os.path.exists('data/raw/WIDER_FACE/wider_face_split/wider_face_train_bbx_gt.txt'))"
# Expected: True
```

### Check CelebAMask-HQ

```bash
# Count images
python -c "import os; print(len([f for f in os.listdir('data/raw/CelebAMask-HQ/CelebA-HQ-img') if f.endswith('.jpg')]))"
# Expected: 30,000 images

# Check masks exist
python -c "import os; print(os.path.exists('data/raw/CelebAMask-HQ/CelebAMask-HQ-mask-anno'))"
# Expected: True
```

---

## Troubleshooting

### Issue: Download too slow

**Solutions:**
1. Use manual download with a download manager (IDM, Free Download Manager)
2. Download overnight
3. Use Kaggle resized version for CelebAMask-HQ (much smaller)
4. If you have university/institutional access, check if datasets are mirrored

### Issue: Google Drive quota exceeded

**Solutions:**
1. Wait 24 hours and try again
2. Download from alternative sources:
   - Kaggle: https://www.kaggle.com/datasets/
   - Academic Torrents (if available)
3. Ask colleagues if they have the dataset

### Issue: Not enough disk space

**Solutions:**
1. Free up space (you need ~25 GB total)
2. Use external hard drive
3. Use resized version of CelebAMask-HQ (256x256 instead of 1024x1024)
4. Delete zip files after extraction to save space

### Issue: Extraction fails

**Solutions:**
1. Re-download the corrupted file
2. Use 7-Zip instead of Windows built-in extractor
3. Check MD5/SHA checksums if provided

---

## Alternative Datasets (If Main Ones Unavailable)

### For Detection:
- **FDDB**: http://vis-www.cs.umass.edu/fddb/
- **AFW**: https://www.ics.uci.edu/~xzhu/face/
- **PASCAL FACE**: (subset of PASCAL VOC)

### For Segmentation:
- **LFW Parts**: http://vis-www.cs.umass.edu/lfw/part_labels/
- **Helen Face**: http://www.ifp.illinois.edu/~vuongle2/helen/
- **CelebA (original)**: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

---

## Dataset Statistics

### WIDER FACE Distribution

| Split | Images | Faces | Easy | Medium | Hard |
|-------|--------|-------|------|--------|------|
| Train | 12,880 | 159,424 | - | - | - |
| Val | 3,226 | 39,720 | 1,288 | 1,026 | 912 |
| Test | 16,097 | 194,559 | - | - | - |

### CelebAMask-HQ Classes

19 facial attribute classes:
- skin, nose, eye_g (glasses), l_eye, r_eye, l_brow, r_brow
- l_ear, r_ear, mouth, u_lip, l_lip, hair, hat, ear_r, neck_l, neck, cloth

---

## Next Steps After Download

1. ✅ Verify datasets are correctly extracted
2. ✅ Update `.env` file with correct paths
3. ✅ Run data exploration notebooks:
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```
4. ✅ Create data loaders (see Phase 1 in PROGRESS_STATUS.md)
5. ✅ Begin model training

---

## Disk Space Summary

| Component | Size |
|-----------|------|
| WIDER FACE (zip) | 2 GB |
| WIDER FACE (extracted) | 3 GB |
| CelebAMask-HQ (zip) | 20 GB |
| CelebAMask-HQ (extracted) | 25 GB |
| **Total (keep zips)** | ~50 GB |
| **Total (delete zips)** | ~28 GB |

**Recommendation:** Delete zip files after successful extraction to save space.

---

## Download Time Estimates

| Connection | WIDER FACE | CelebAMask-HQ | Total |
|------------|------------|---------------|-------|
| 10 Mbps | 30 min | 5 hours | ~5.5 hours |
| 50 Mbps | 6 min | 1 hour | ~1 hour |
| 100 Mbps | 3 min | 30 min | ~33 min |

*Note: Actual times may vary based on server speed and network conditions*

---

## Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Visit official dataset websites for updates
3. Check `PROGRESS_STATUS.md` for project-specific notes
4. Ask in project discussions or team chat

---

**Last Updated:** 2026-09-03  
**Maintained by:** Project Team
