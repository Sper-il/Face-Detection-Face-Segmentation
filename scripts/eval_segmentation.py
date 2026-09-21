"""Evaluate UNet segmentation model on validation/test set.

Metrics: IoU, Dice, Pixel Accuracy, Precision, Recall, F1 (face class).
Matches the evaluation logic from the user's training script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2
import json
from tqdm import tqdm

from src.segmentation.unet_model import UNet, UNetConfig


def load_segmentation_split(split_dir: Path, max_samples: int | None = None) -> list[dict]:
    """Load image-mask pairs from a split directory."""
    import random
    images_dir = Path(split_dir) / "images"
    masks_dir = Path(split_dir) / "masks"
    samples = []
    missing = 0
    for img_path in sorted(images_dir.iterdir()):
        mask_path = masks_dir / f"{img_path.stem}.png"
        if mask_path.exists():
            samples.append({"image": str(img_path), "mask": str(mask_path)})
        else:
            missing += 1
    print(f"[{split_dir}] Total: {len(samples) + missing} | matched: {len(samples)} | missing: {missing}")
    if max_samples is not None:
        random.shuffle(samples)
        samples = samples[:max_samples]
    return samples


def load_unet_model(weights_path: str | Path, device: str = "cpu") -> UNet:
    """Load UNet model from checkpoint."""
    model = UNet(UNetConfig())
    state = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def evaluate_segmentation(
    model: UNet,
    samples: list[dict],
    device: str = "cpu",
    size: int = 256,
) -> dict:
    """Compute IoU, Dice, Pixel Accuracy, Precision/Recall/F1 for face class."""
    sum_iou = sum_dice = sum_pa = 0.0
    sum_p = sum_r = sum_f1 = 0.0
    n_valid = 0

    for s in tqdm(samples, desc="eval-seg"):
        img = cv2.imread(s["image"])
        gt = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (size, size))
        img_norm = (img_resized / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)

        logits = model(img_tensor)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        gt_resized = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt_resized > 127).astype(np.uint8)

        tp = int(((pred == 1) & (gt_bin == 1)).sum())
        fp = int(((pred == 1) & (gt_bin == 0)).sum())
        fn = int(((pred == 0) & (gt_bin == 1)).sum())
        tn = int(((pred == 0) & (gt_bin == 0)).sum())

        inter = tp
        union = tp + fp + fn
        iou = inter / max(union, 1)
        dice = (2 * inter) / max(2 * inter + fp + fn, 1)
        pa = (tp + tn) / max(tp + fp + fn + tn, 1)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)

        sum_iou += iou
        sum_dice += dice
        sum_pa += pa
        sum_p += p
        sum_r += r
        sum_f1 += f1
        n_valid += 1

    n = max(n_valid, 1)
    return {
        "n_images": n_valid,
        "iou": sum_iou / n,
        "dice": sum_dice / n,
        "pixel_accuracy": sum_pa / n,
        "precision_face": sum_p / n,
        "recall_face": sum_r / n,
        "f1_face": sum_f1 / n,
    }


def visualize_predictions(
    model: UNet,
    samples: list[dict],
    output_dir: Path,
    device: str = "cpu",
    size: int = 256,
    num_samples: int = 5,
):
    """Save prediction visualizations (image | pred_mask | overlay | gt_mask)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(samples[:num_samples]):
        img = cv2.imread(s["image"])
        gt = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (size, size))
        img_norm = (img_resized / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
        pred_mask = logits.argmax(dim=1)[0].cpu().numpy()
        gt_resized = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)

        overlay = img_resized.copy()
        overlay[pred_mask == 1] = (0.6 * overlay[pred_mask == 1] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_resized); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(pred_mask, cmap="gray"); axes[1].set_title("Predicted"); axes[1].axis("off")
        axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
        axes[3].imshow(gt_resized, cmap="gray"); axes[3].set_title("Ground Truth"); axes[3].axis("off")

        out_path = output_dir / f"sample_{i:03d}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate UNet segmentation model")
    parser.add_argument("--weights", type=str, default="models/unet_final.pth")
    parser.add_argument("--data-dir", type=str, default="data/processed/segmentation")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=str, default="runs/evaluation")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num-visualizations", type=int, default=5)
    args = parser.parse_args()

    weights_path = Path(args.weights)
    data_dir = Path(args.data_dir)
    split_dir = data_dir / args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UNET SEGMENTATION EVALUATION")
    print("=" * 60)
    print(f"Weights      : {weights_path}")
    print(f"Split        : {args.split}")
    print(f"Data dir     : {split_dir}")
    print(f"Device       : {args.device}")
    print(f"Image size   : {args.size}")

    # Load model
    print("\nLoading model...")
    model = load_unet_model(weights_path, args.device)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Load dataset
    samples = load_segmentation_split(split_dir, max_samples=args.max_samples)
    if not samples:
        print("No samples found! Aborting.")
        return

    # Evaluate
    print("\nRunning evaluation...")
    metrics = evaluate_segmentation(model, samples, device=args.device, size=args.size)

    print("\n" + "=" * 60)
    print("SEGMENTATION METRICS")
    print("=" * 60)
    print(f"Images evaluated : {metrics['n_images']}")
    print(f"Mean IoU         : {metrics['iou']:.4f}")
    print(f"Mean Dice        : {metrics['dice']:.4f}")
    print(f"Pixel Accuracy   : {metrics['pixel_accuracy']:.4f}")
    print(f"Precision (face) : {metrics['precision_face']:.4f}")
    print(f"Recall (face)    : {metrics['recall_face']:.4f}")
    print(f"F1 (face)        : {metrics['f1_face']:.4f}")

    # Save results
    results_path = output_dir / f"segmentation_{args.split}_metrics.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Visualize
    if args.visualize:
        vis_dir = output_dir / f"visualizations_{args.split}"
        print(f"\nGenerating visualizations in: {vis_dir}")
        visualize_predictions(
            model, samples, vis_dir,
            device=args.device, size=args.size,
            num_samples=args.num_visualizations,
        )


if __name__ == "__main__":
    main()
