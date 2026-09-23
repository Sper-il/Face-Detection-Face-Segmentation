"""Test U-Net model on a single image."""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.segmentation.inference import UNetSegmentor

# Paths
MODEL_PATH = "e:/Face-Detection-Face-Segmentation/models/unet_final.pth"
IMAGE_PATH = "e:/Face-Detection-Face-Segmentation/data/processed/segmentation/test/images/00002.jpg"
OUTPUT_DIR = "e:/Face-Detection-Face-Segmentation/data/output/test_results"

def main():
    # Create output dir
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load image
    print(f"Loading image: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("Error: Could not load image")
        return
    print(f"Image shape: {image.shape}")
    
    # Load model
    print(f"Loading model: {MODEL_PATH}")
    segmentor = UNetSegmentor(weights=MODEL_PATH, device="cpu")
    print("Model loaded successfully")
    
    # Run inference
    print("Running inference...")
    result = segmentor.predict(image)
    
    print(f"Prediction score: {result.score:.4f}")
    print(f"Mask shape: {result.mask.shape}")
    print(f"Face pixels: {(result.mask > 0).sum()}")
    
    # Save results
    # 1. Original image
    cv2.imwrite(f"{OUTPUT_DIR}/01_original.jpg", image)
    
    # 2. Mask only (colored)
    mask_colored = np.zeros_like(image)
    mask_colored[result.mask > 0] = [0, 255, 0]  # Green mask
    cv2.imwrite(f"{OUTPUT_DIR}/02_mask_only.jpg", mask_colored)
    
    # 3. Overlay
    overlay = image.copy()
    mask_bool = result.mask > 0
    overlay[mask_bool] = (0.7 * image[mask_bool] + 0.3 * np.array([0, 255, 0], dtype=np.uint8)).astype(np.uint8)
    cv2.imwrite(f"{OUTPUT_DIR}/03_overlay.jpg", overlay)
    
    # 4. Side by side comparison
    h, w = image.shape[:2]
    comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)
    comparison[:, :w] = image
    comparison[:, w:2*w] = mask_colored
    comparison[:, 2*w:] = overlay
    cv2.imwrite(f"{OUTPUT_DIR}/04_comparison.jpg", comparison)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("  01_original.jpg - Original image")
    print("  02_mask_only.jpg - Segmentation mask (green)")
    print("  03_overlay.jpg - Overlay on original")
    print("  04_comparison.jpg - Side by side comparison")

if __name__ == "__main__":
    main()
