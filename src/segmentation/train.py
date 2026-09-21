"""Training loop for the U-Net segmentor.

Usage::

    python -m src.segmentation.train --config src/configs/unet.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.segmentation.dataset import CelebAMaskHQDataset, segmentation_collate
from src.segmentation.losses import BCEDiceLoss
from src.segmentation.unet import UNet, UNetConfig
from src.utils.io import ensure_dir, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net on CelebAMask-HQ.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default="runs/unet")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the YAML epochs setting.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the YAML batch_size setting.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Disable ImageNet encoder pretraining (smoke tests).")
    return parser.parse_args()


def cosine_lr(epoch: int, total: int, base_lr: float, warmup: int = 1) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: BCEDiceLoss,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    sums = {"total": 0.0, "bce": 0.0, "dice": 0.0}
    n_batches = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                probs = model(images)
                losses = criterion(probs, masks)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            probs = model(images)
            losses = criterion(probs, masks)
            losses["total"].backward()
            optimizer.step()

        for k in sums:
            sums[k] += float(losses[k])
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in sums.items()}


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    from src.segmentation.eval import compute_segmentation_metrics

    model.eval()
    ious, dices, accs = [], [], []
    with torch.no_grad():
        for batch in loader:
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
    return {
        "val_iou": sum(ious) / max(1, len(ious)),
        "val_dice": sum(dices) / max(1, len(dices)),
        "val_pixel_acc": sum(accs) / max(1, len(accs)),
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(args.output_dir)

    data_cfg = cfg.get("data", {})
    image_size = int(data_cfg.get("image_size", 512))

    train_ds = CelebAMaskHQDataset(
        images_dir=data_cfg["train_images_dir"],
        masks_dir=data_cfg["train_masks_dir"],
        image_size=image_size,
        augment=True,
    )
    val_ds = CelebAMaskHQDataset(
        images_dir=data_cfg["val_images_dir"],
        masks_dir=data_cfg["val_masks_dir"],
        image_size=image_size,
        augment=False,
    )

    batch_size = args.batch_size or int(cfg["training"].get("batch_size", 16))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=segmentation_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=segmentation_collate,
    )

    pretrained = not args.no_pretrained
    model = UNet(UNetConfig(pretrained_encoder=pretrained)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    loss_cfg = cfg.get("loss", {})
    criterion = BCEDiceLoss(
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
    )

    epochs = args.epochs or int(cfg["training"].get("epochs", 30))
    patience = int(cfg["training"].get("early_stop_patience", 5))
    base_lr = float(cfg["training"].get("lr", 1e-4))

    scaler = (
        torch.amp.GradScaler(device.type)
        if (cfg["training"].get("amp", True) and device.type == "cuda")
        else None
    )

    best_iou = -1.0
    bad = 0
    history: list[dict[str, float | int]] = []

    print(f"[train] device={device} epochs={epochs} batch_size={batch_size} "
          f"train_n={len(train_ds)} val_n={len(val_ds)} pretrained={pretrained}")

    for epoch in range(epochs):
        # Update LR per epoch (cosine).
        new_lr = cosine_lr(epoch, epochs, base_lr)
        for g in optimizer.param_groups:
            g["lr"] = new_lr

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"[epoch {epoch + 1}/{epochs}] "
            f"train_total={train_metrics['total']:.3f} "
            f"bce={train_metrics['bce']:.3f} dice={train_metrics['dice']:.3f} | "
            f"val_iou={val_metrics['val_iou']:.3f} val_dice={val_metrics['val_dice']:.3f} "
            f"val_pixel_acc={val_metrics['val_pixel_acc']:.3f}"
        )

        history.append({
            "epoch": epoch + 1,
            "lr": new_lr,
            "train": train_metrics,
            "val": val_metrics,
        })

        if val_metrics["val_iou"] > best_iou:
            best_iou = val_metrics["val_iou"]
            bad = 0
            ckpt = output_dir / "unet_best.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_iou": best_iou,
                    "val_metrics": val_metrics,
                },
                ckpt,
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch + 1} (patience={patience}).")
                break

    # Always save the latest checkpoint.
    torch.save(
        {"model": model.state_dict(), "epoch": epoch, "history": history},
        output_dir / "unet_last.pth",
    )

    hist_path = output_dir / "unet_history.json"
    hist_path.write_text(
        json.dumps(history, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Best val IoU: {best_iou:.3f}; history -> {hist_path}")


if __name__ == "__main__":
    main()
