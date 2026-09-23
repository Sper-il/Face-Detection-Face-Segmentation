"""
Quick Test Script - Process small sample to verify pipeline works
"""

import sys
sys.path.insert(0, 'e:/Face Detection & Face Segmentation')

from src.data.preprocess_wider_face import WIDERFacePreprocessor
from src.data.preprocess_celebamask_hq import CelebAMaskHQPreprocessor
from pathlib import Path
import shutil

print("=" * 70)
print("🧪 QUICK TEST - Processing Small Samples")
print("=" * 70)

# Test 1: WIDER FACE (just val split, smaller)
print("\n[Test 1/2] WIDER FACE - Val Split Only")
print("-" * 70)

try:
    wider_preprocessor = WIDERFacePreprocessor(
        raw_data_dir="e:/Face Detection & Face Segmentation/data/raw/WIDER_FACE",
        processed_data_dir="e:/Face Detection & Face Segmentation/data/processed/wider_face_test",
        target_size=(640, 640),
        min_face_size=10,
        blur_threshold=100.0
    )
    
    # Only process val split (smaller dataset)
    print("Processing validation split only...")
    samples = wider_preprocessor.process_split("val")
    
    if samples:
        print(f"✅ Successfully processed {len(samples)} images")
        print(f"   Total faces: {sum(len(s['faces']) for s in samples)}")
    else:
        print("⚠️ No samples processed")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

# Test 2: CelebAMask-HQ (first 100 images only)
print("\n[Test 2/2] CelebAMask-HQ - First 100 Images")
print("-" * 70)

try:
    celeb_preprocessor = CelebAMaskHQPreprocessor(
        raw_data_dir="e:/Face Detection & Face Segmentation/data/raw/CelebAMask-HQ",
        processed_data_dir="e:/Face Detection & Face Segmentation/data/processed/celebamask_test",
        target_size=(512, 512),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        combine_masks=True
    )
    
    # Locate images
    img_dir = Path("e:/Face Detection & Face Segmentation/data/raw/CelebAMask-HQ/CelebAMask-HQ/CelebA-HQ-img")
    
    if img_dir.exists():
        # Get first 100 images only
        image_files = sorted(img_dir.glob("*.jpg"))[:100]
        print(f"Testing with first {len(image_files)} images...")
        
        # Temporarily replace the process_dataset method
        from tqdm import tqdm
        import cv2
        import numpy as np
        import random
        
        random.seed(42)
        processed = 0
        
        for img_file in tqdm(image_files, desc="Processing"):
            img_idx = int(img_file.stem)
            
            # Read image
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # Find masks
            mask_files = celeb_preprocessor.find_mask_files(img_idx)
            if not mask_files:
                continue
            
            # Load and combine
            mask, _ = celeb_preprocessor.load_and_combine_masks(
                mask_files, 
                celeb_preprocessor.target_size
            )
            
            if mask is None:
                continue
            
            # Resize image
            if image.shape[:2] != celeb_preprocessor.target_size:
                image = cv2.resize(
                    image,
                    celeb_preprocessor.target_size,
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Save to test output
            split = "train" if random.random() < 0.7 else ("val" if random.random() < 0.5 else "test")
            out_dir = celeb_preprocessor.processed_dir / split
            (out_dir / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / "masks").mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(str(out_dir / "images" / f"{img_idx:05d}.jpg"), image)
            cv2.imwrite(str(out_dir / "masks" / f"{img_idx:05d}.png"), mask)
            
            processed += 1
        
        print(f"✅ Successfully processed {processed} images")
    else:
        print(f"❌ Image directory not found: {img_dir}")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ QUICK TEST COMPLETE")
print("=" * 70)
print("\nIf both tests passed, you can run full preprocessing with:")
print("  python src/data/preprocess_wider_face.py")
print("  python src/data/preprocess_celebamask_hq.py")
