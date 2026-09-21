"""Standalone eval script for U-Net segmentor.

Usage::

    python -m src.scripts.eval_segmentation \
        --config src/configs/unet.yaml \
        --checkpoint runs/unet/unet_best.pth \
        --output runs/unet/eval_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.segmentation.dataset import CelebAMaskHQDataset, segmentation_collate
from src.segmentation.eval import compute_segmentation_metrics
from src.segmentation.unet import UNet, UNetConfig
from src.utils.io import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate U-Net.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=None,
                        help="Override image_size from config (e.g. for smoke tests).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    image_size = args.image_size or int(data_cfg.get("image_size", 512))

    # Use val split for evaluation
    val_ds = CelebAMaskHQDataset(
        images_dir=data_cfg["val_images_dir"],
        masks_dir=data_cfg["val_masks_dir"],
        image_size=image_size,
        augment=False,
    )
    if args.max_images is not None:
        val_ds._image_files = val_ds._image_files[: args.max_images]

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=segmentation_collate,
    )

    # Build model
    model = UNet(UNetConfig(pretrained_encoder=False))
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print(f"WARNING: Checkpoint not found: {ckpt}, using random weights")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    ious, dices, accs = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            probs = model(images)
            if probs.shape[1] == 1:
                pred = (probs > 0.5).float()
            else:
                pred = probs.argmax(dim=1, keepdim=True).float()
            m = compute_segmentation_metrics(pred, masks)
            ious.append(m["iou"])
            dices.append(m["dice"])
            accs.append(m["pixel_acc"])

    metrics = {
        "iou": float(np.mean(ious)),
        "dice": float(np.mean(dices)),
        "pixel_acc": float(np.mean(accs)),
        "n_images": len(val_ds),
    }
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if args.output:
        out_path = Path(args.output)
        ensure_dir(out_path.parent)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
