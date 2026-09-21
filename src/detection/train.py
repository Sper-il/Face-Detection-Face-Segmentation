"""Training loop for the RetinaFace detector.

Usage::

    python -m src.detection.train --config src/configs/retinaface.yaml

The script supports:

- AMP (mixed precision) when CUDA is available;
- cosine LR schedule;
- early stopping on validation mAP (best-effort, single-GPU friendly);
- TensorBoard logging (optional — only if `tensorboard` is importable).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.detection.anchors import AnchorConfig
from src.detection.dataset import WIDERFaceDataset, detection_collate
from src.detection.losses import DetectionLossWeights, MultiTaskDetectionLoss
from src.detection.retinaface import RetinaFace, RetinaFaceConfig, flatten_predictions
from src.utils.io import ensure_dir, load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RetinaFace on WIDER FACE.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default="runs/retinaface")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the YAML epochs setting.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the YAML batch_size setting.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Disable ImageNet backbone pretraining (smoke tests).")
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("training", {})
    name = (opt_cfg.get("optimizer") or "adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    wd = float(opt_cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer {name!r}")


def cosine_lr(epoch: int, total: int, base_lr: float, warmup: int = 1) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskDetectionLoss,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    base_lr: float,
    epoch: int,
    total_epochs: int,
) -> tuple[dict[str, float], int]:
    model.train()
    sums = {"cls": 0.0, "box": 0.0, "landmark": 0.0, "total": 0.0}
    n_batches = 0

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        cls_t = batch["cls_target"].to(device, non_blocking=True)
        box_t = batch["box_target"].to(device, non_blocking=True)
        lmk_t = batch["lmk_target"].to(device, non_blocking=True)
        mask_t = batch["lmk_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                outputs = flatten_predictions(model(images))
                losses = criterion(
                    outputs["cls_logits"],
                    outputs["box_deltas"],
                    outputs["lmk_deltas"],
                    cls_t,
                    box_t,
                    lmk_t,
                    mask_t,
                )
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = flatten_predictions(model(images))
            losses = criterion(
                outputs["cls_logits"],
                outputs["box_deltas"],
                outputs["lmk_deltas"],
                cls_t,
                box_t,
                lmk_t,
                mask_t,
            )
            losses["total"].backward()
            optimizer.step()

        for k in sums:
            sums[k] += float(losses[k])
        n_batches += 1

    # Update LR via cosine schedule.
    new_lr = cosine_lr(epoch, total_epochs, base_lr)
    for g in optimizer.param_groups:
        g["lr"] = new_lr
    next_epoch = epoch + 1
    return {k: v / max(1, n_batches) for k, v in sums.items()}, next_epoch


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: MultiTaskDetectionLoss | None = None,
    compute_map: bool = False,
    max_map_images: int | None = 500,
) -> dict[str, float]:
    """Validation pass that matches the training loss exactly.

    Computes val_cls_loss, val_box_loss, val_lmk_loss, val_total_loss using
    the same MultiTaskDetectionLoss (BCE-with-logits + ignore mask) as the
    training loop so the two numbers are directly comparable. When
    ``compute_map=True`` we additionally report val_mAP50 and val_recall50
    on a sub-sample of the validation set (bounded by ``max_map_images``).
    """
    model.eval()
    sums = {"cls": 0.0, "box": 0.0, "landmark": 0.0, "total": 0.0}
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            cls_t = batch["cls_target"].to(device, non_blocking=True)
            box_t = batch["box_target"].to(device, non_blocking=True)
            lmk_t = batch["lmk_target"].to(device, non_blocking=True)
            lmk_mask = batch["lmk_mask"].to(device, non_blocking=True)

            outputs = flatten_predictions(model(images))

            if criterion is not None:
                losses = criterion(
                    outputs["cls_logits"],
                    outputs["box_deltas"],
                    outputs["lmk_deltas"],
                    cls_t,
                    box_t,
                    lmk_t,
                    lmk_mask,
                )
                for k in sums:
                    sums[k] += float(losses[k])
            else:
                # Fallback that mirrors MultiTaskDetectionLoss.
                face_logit = outputs["cls_logits"][..., 1]
                weight = (cls_t >= 0).float()
                safe_target = cls_t.clamp(min=0).float()
                per = torch.nn.functional.binary_cross_entropy_with_logits(
                    face_logit, safe_target, reduction="none"
                )
                cls_loss = (per * weight).sum() / weight.sum().clamp(min=1)
                sums["cls"] += float(cls_loss)
                sums["total"] += float(cls_loss)

            n_batches += 1

    metrics = {f"val_{k}_loss": v / max(1, n_batches) for k, v in sums.items()}

    if compute_map:
        try:
            metrics.update(
                _evaluate_detection_map(
                    model, loader, device, max_images=max_map_images
                )
            )
        except Exception as e:  # noqa: BLE001
            metrics["val_map_error"] = str(e)

    return metrics


def _evaluate_detection_map(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_images: int | None = 500,
) -> dict[str, float]:
    """Light-weight mAP@0.5 + recall@0.5 for the WIDERFaceDataset loader."""
    import cv2 as _cv
    from src.detection.eval import (
        EvalEntry, compute_map_recall, group_by_image, load_gt,
    )
    from src.detection.anchors import AnchorConfig, generate_anchors
    from src.utils.box_ops import decode_boxes

    dataset = loader.dataset
    cfg_path = getattr(dataset, "csv_path", None)
    images_dir = getattr(dataset, "images_dir", None)
    if cfg_path is None or images_dir is None:
        return {}

    anchor_cfg = getattr(dataset, "anchor_cfg", AnchorConfig())
    anchors = generate_anchors(anchor_cfg)
    target = anchor_cfg.image_size

    gt_entries = load_gt(cfg_path)
    gt_by_img = group_by_image(gt_entries)
    img_ids = sorted(gt_by_img.keys())
    if max_images is not None:
        img_ids = img_ids[:max_images]

    mean = np.array([0.485, 0.456, 0.406], dtype="float32").reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype="float32").reshape(3, 1, 1)

    pred_entries: list[EvalEntry] = []
    for img_id in img_ids:
        path = Path(images_dir) / img_id
        img = _cv.imread(str(path), _cv.IMREAD_COLOR)
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        ratio = min(target / h0, target / w0)
        new_h, new_w = int(round(h0 * ratio)), int(round(w0 * ratio))
        pad_w = target - new_w
        pad_h = target - new_h
        resized = _cv.resize(img, (new_w, new_h), interpolation=_cv.INTER_LINEAR)
        padded = _cv.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w, _cv.BORDER_CONSTANT, value=(0, 0, 0)
        )
        rgb = _cv.cvtColor(padded, _cv.COLOR_BGR2RGB).astype("float32") / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1)
        t = (t - torch.from_numpy(mean)) / torch.from_numpy(std)
        x = t.unsqueeze(0).to(device)
        out = flatten_predictions(model(x))
        cls = out["cls_logits"][0].cpu().numpy()
        box = out["box_deltas"][0].cpu().numpy()
        scores = 1.0 / (1.0 + np.exp(-cls[..., 1]))
        keep = scores > 0.02
        if not keep.any():
            continue
        decoded = decode_boxes(anchors[keep], box[keep])
        inv = 1.0 / max(1e-7, ratio)
        decoded[:, [0, 2]] *= inv
        decoded[:, [1, 3]] *= inv
        decoded[:, [0, 2]] = np.clip(decoded[:, [0, 2]], 0, w0 - 1)
        decoded[:, [1, 3]] = np.clip(decoded[:, [1, 3]], 0, h0 - 1)
        kept_scores = scores[keep]
        for i in range(decoded.shape[0]):
            pred_entries.append(EvalEntry(img_id, decoded[i], float(kept_scores[i])))

    m = compute_map_recall(gt_entries, pred_entries, iou_threshold=0.5)
    return {"val_mAP50": float(m["mAP"]), "val_recall50": float(m["recall"])}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(args.output_dir)

    data_cfg = cfg.get("data", {})
    model_cfg_dict = cfg.get("model", {})
    anchor_cfg = AnchorConfig(
        image_size=int(data_cfg.get("image_size", 640)),
        strides=tuple(model_cfg_dict.get("strides", [8, 16, 32])),
    )

    train_ds = WIDERFaceDataset(
        csv_path=data_cfg["train_csv"],
        images_dir=Path(data_cfg["train_csv"]).parent / "images",
        anchor_cfg=anchor_cfg,
        augment=True,
    )
    val_ds = WIDERFaceDataset(
        csv_path=data_cfg["val_csv"],
        images_dir=Path(data_cfg["val_csv"]).parent / "images",
        anchor_cfg=anchor_cfg,
        augment=False,
    )

    batch_size = args.batch_size or int(cfg["training"].get("batch_size", 8))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )

    pretrained = not args.no_pretrained
    model_cfg = RetinaFaceConfig(
        pretrained_backbone=pretrained,
        backbone_type=cfg.get("model", {}).get("backbone_type", "resnet34"),
    )
    model = RetinaFace(model_cfg).to(device)

    opt_cfg = cfg["training"]
    optimizer = build_optimizer(model, cfg)
    criterion = MultiTaskDetectionLoss(
        DetectionLossWeights(
            cls_weight=float(cfg.get("loss", {}).get("cls_weight", 1.0)),
            box_weight=float(cfg.get("loss", {}).get("box_weight", 1.0)),
            landmark_weight=float(cfg.get("loss", {}).get("landmark_weight", 0.5)),
        )
    )

    scaler = torch.amp.GradScaler(device.type) if (opt_cfg.get("amp", True) and device.type == "cuda") else None

    epochs = args.epochs or int(opt_cfg.get("epochs", 50))
    patience = int(opt_cfg.get("early_stop_patience", 5))
    best_val = math.inf
    bad_epochs = 0
    base_lr = float(opt_cfg.get("lr", 1e-4))

    print(f"[train] device={device} epochs={epochs} batch_size={batch_size} "
          f"train_n={len(train_ds)} val_n={len(val_ds)} pretrained={pretrained}")

    history: list[dict[str, float]] = []

    epoch = 0
    while epoch < epochs:
        train_metrics, epoch = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            base_lr, epoch, epochs,
        )
        # Full validation pass (compute_map on every epoch is expensive;
        # we sample 500 images to keep the cost bounded).
        val_metrics = evaluate(
            model, val_loader, device, criterion=criterion,
            compute_map=True, max_map_images=500,
        )
        line = (
            f"[epoch {epoch}/{epochs}] "
            f"train_total={train_metrics['total']:.3f} "
            f"cls={train_metrics['cls']:.3f} box={train_metrics['box']:.3f} "
            f"lmk={train_metrics['landmark']:.3f} | "
            f"val_total={val_metrics.get('val_total_loss', float('nan')):.3f} "
            f"val_cls={val_metrics.get('val_cls_loss', float('nan')):.3f}"
        )
        if "val_mAP50" in val_metrics:
            line += (
                f" val_mAP50={val_metrics['val_mAP50']:.3f}"
                f" val_recall50={val_metrics['val_recall50']:.3f}"
            )
        print(line)

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        # Best checkpoint selection: prefer val_mAP50, fall back to val total loss.
        if "val_mAP50" in val_metrics:
            score = val_metrics["val_mAP50"]
            if score > best_val:
                best_val = score
                bad_epochs = 0
                ckpt = output_dir / "retinaface_best.pth"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_mAP50": best_val,
                        "val_metrics": val_metrics,
                    },
                    ckpt,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch} (patience={patience}).")
                    break
        else:
            current = val_metrics.get("val_total_loss", float("inf"))
            if current < best_val:
                best_val = current
                bad_epochs = 0
                ckpt = output_dir / "retinaface_best.pth"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_total_loss": best_val,
                        "val_metrics": val_metrics,
                    },
                    ckpt,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch} (patience={patience}).")
                    break

    # Always save the latest checkpoint so we never lose progress.
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "history": history,
        },
        output_dir / "retinaface_last.pth",
    )

    # Dump training history as JSON for downstream analysis.
    hist_path = output_dir / "retinaface_history.json"
    hist_path.write_text(
        json.dumps(
            history,
            indent=2,
            default=lambda x: float(x) if isinstance(x, (int, float, np.floating)) else str(x),
        ),
        encoding="utf-8",
    )
    print(f"Best val score: {best_val:.3f}; history -> {hist_path}")


if __name__ == "__main__":
    main()
