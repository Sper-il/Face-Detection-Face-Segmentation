"""Đánh giá UNet segmentation (hỗ trợ JPG/PNG)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.segmentation.unet import UNet, UNetConfig
from src.segmentation.eval import compute_segmentation_metrics


def load_and_eval(model, images_dir, masks_dir, image_size=512, max_images=None, device="cpu"):
    """Load images and evaluate."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    # Get image files (jpg or png)
    files = list(images_dir.glob("*.jpg")) or list(images_dir.glob("*.png"))
    files = sorted(files)[:max_images] if max_images else sorted(files)
    
    ious, dices, accs = [], [], []
    
    for f in files:
        # Load image
        img = cv2.imread(str(f))
        if img is None:
            continue
        
        # Load mask if exists
        mask_path = masks_dir / f.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Try different extensions
            mask_path_jpg = masks_dir / f.name.replace(".png", ".jpg").replace(".jpg", ".png")
            if mask_path_jpg.exists():
                mask = cv2.imread(str(mask_path_jpg), cv2.IMREAD_GRAYSCALE)
            else:
                mask = None
        
        # Resize
        img = cv2.resize(img, (image_size, image_size))
        if mask is not None:
            mask = cv2.resize(mask, (image_size, image_size))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_rgb - mean) / std
        
        # To tensor [1, 3, H, W]
        img_t = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        
        # Predict
        model.eval()
        with torch.no_grad():
            prob = model(img_t)
        
        # Mask tensor
        if mask is not None:
            mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            m = compute_segmentation_metrics((prob > 0.5).float(), mask_t)
            ious.append(m["iou"])
            dices.append(m["dice"])
            accs.append(m["pixel_acc"])
    
    return {
        "iou": float(np.mean(ious)) if ious else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "pixel_acc": float(np.mean(accs)) if accs else 0.0,
        "n_images": len(files)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/unet_demo.pth")
    parser.add_argument("--images-dir", default="data/processed/segmentation/val/images")
    parser.add_argument("--masks-dir", default="data/processed/segmentation/val/masks")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = UNet(UNetConfig(pretrained_encoder=False)).to(device)
    
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded: {ckpt}")
    else:
        print(f"WARNING: Checkpoint not found: {ckpt}")
    
    print(f"Device: {device}")
    print(f"Images: {args.images_dir}")
    
    # Evaluate
    metrics = load_and_eval(
        model, 
        args.images_dir, 
        args.masks_dir, 
        args.image_size, 
        args.max_images,
        device
    )
    
    print("\n" + "="*55)
    print("UNET SEGMENTATION EVALUATION")
    print("="*55)
    print(f"Checkpoint:       {args.checkpoint}")
    print(f"Images evaluated: {metrics['n_images']}")
    print(f"IoU:              {metrics['iou']:.4f}")
    print(f"Dice:             {metrics['dice']:.4f}")
    print(f"Pixel Accuracy:   {metrics['pixel_acc']:.4f}")
    print("="*55)
    
    # Save
    out = Path("runs/eval")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "unet_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: runs/eval/unet_results.json")


if __name__ == "__main__":
    main()
