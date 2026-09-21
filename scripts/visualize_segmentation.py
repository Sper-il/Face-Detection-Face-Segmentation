"""Generate demo visualizations: overlay predicted mask on original image.

Usage::

    python scripts/visualize_segmentation.py --num-samples 8

Outputs PNGs to runs/visualizations/<split>/sample_XXX.png with 4 panels:
  1. Original image
  2. Predicted mask
  3. Overlay (red mask on image)
  4. Ground truth mask (if available)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import random

import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.segmentation.unet_model import UNet, UNetConfig


def load_unet_model(weights_path: str | Path, device: str = "cpu") -> UNet:
    model = UNet(UNetConfig())
    state = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def predict_mask(model: UNet, img_bgr: np.ndarray, device: str, size: int = 256) -> np.ndarray:
    """Predict a binary mask at the model's resolution."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size))
    img_norm = (img_resized / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
    pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return pred, img_resized


def visualize_samples(
    model: UNet,
    samples: list[dict],
    output_dir: Path,
    device: str = "cpu",
    size: int = 256,
    num_samples: int = 8,
    split_name: str = "test",
):
    """Save 4-panel visualizations for randomly sampled images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = random.sample(samples, min(num_samples, len(samples)))

    print(f"Generating {len(selected)} visualizations -> {output_dir}")
    for i, s in enumerate(selected):
        img = cv2.imread(s["image"])
        if img is None:
            continue

        gt_path = Path(s["mask"])
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else None

        pred_mask, img_resized = predict_mask(model, img, device, size)

        # Overlay
        overlay = img_resized.copy()
        overlay[pred_mask == 1] = (
            0.6 * overlay[pred_mask == 1] + 0.4 * np.array([255, 0, 0])
        ).astype(np.uint8)

        n_panels = 4 if gt is not None else 3
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
        axes[0].imshow(img_resized)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(pred_mask, cmap="gray")
        axes[1].set_title("Predicted mask")
        axes[1].axis("off")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (pred)")
        axes[2].axis("off")
        if gt is not None:
            gt_resized = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
            axes[3].imshow(gt_resized, cmap="gray")
            axes[3].set_title("Ground truth")
            axes[3].axis("off")

        out_path = output_dir / f"sample_{i:03d}_{Path(s['image']).stem}.png"
        plt.suptitle(f"{split_name} — sample {i+1}/{num_samples}: {Path(s['image']).name}", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"  [{i+1}/{num_samples}] {out_path.name}")


def make_summary_figure(
    model: UNet,
    samples: list[dict],
    output_path: Path,
    device: str = "cpu",
    size: int = 256,
    n_samples: int = 6,
):
    """Make a single grid figure with n_samples x 3 panels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = random.sample(samples, min(n_samples, len(samples)))

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i, s in enumerate(selected):
        img = cv2.imread(s["image"])
        if img is None:
            continue

        pred_mask, img_resized = predict_mask(model, img, device, size)
        overlay = img_resized.copy()
        overlay[pred_mask == 1] = (
            0.5 * overlay[pred_mask == 1] + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)

        axes[i, 0].imshow(img_resized)
        axes[i, 0].set_title(f"Image {i+1}", fontsize=11)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(pred_mask, cmap="gray")
        axes[i, 1].set_title("Predicted mask", fontsize=11)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay", fontsize=11)
        axes[i, 2].axis("off")

    plt.suptitle("U-Net Segmentation Demo (CelebAMask-HQ test set)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved summary figure: {output_path}")


def load_segmentation_split(split_dir: Path, max_samples: int | None = None) -> list[dict]:
    images_dir = Path(split_dir) / "images"
    masks_dir = Path(split_dir) / "masks"
    samples = []
    for img_path in sorted(images_dir.iterdir()):
        mask_path = masks_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            samples.append({"image": str(img_path), "mask": str(mask_path)})
    print(f"[{split_dir}] Loaded {len(samples)} samples")
    if max_samples:
        random.shuffle(samples)
        samples = samples[:max_samples]
    return samples


def main():
    parser = argparse.ArgumentParser(description="Visualize U-Net segmentation predictions")
    parser.add_argument("--weights", type=str, default="models/unet_final.pth")
    parser.add_argument("--data-dir", type=str, default="data/processed/segmentation")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default="runs/visualizations")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--summary", action="store_true", help="Also save a 3-panel summary grid")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    split_dir = Path(args.data_dir) / args.split
    output_dir = Path(args.output_dir) / args.split

    print("=" * 60)
    print("U-NET SEGMENTATION VISUALIZATION")
    print("=" * 60)
    print(f"Weights  : {weights_path}")
    print(f"Split    : {args.split}")
    print(f"Output   : {output_dir}")

    print("\nLoading model...")
    model = load_unet_model(weights_path, args.device)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    samples = load_segmentation_split(split_dir, max_samples=200)
    if not samples:
        print("No samples found! Aborting.")
        return

    visualize_samples(
        model, samples, output_dir,
        device=args.device, size=args.size,
        num_samples=args.num_samples, split_name=args.split,
    )

    if args.summary:
        make_summary_figure(
            model, samples, output_dir / "summary.png",
            device=args.device, size=args.size,
            n_samples=min(6, args.num_samples),
        )

    print(f"\nDone. Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
