"""
Training Script for Face Segmentation (FCN8s / U-Net)
M2-Seg: Người 2 - Training pipeline

Features:
- Mixed precision training (AMP)
- Cosine Annealing with Warm Restarts LR scheduler
- Combined Loss (CE + Dice)
- TensorBoard logging
- Model checkpointing (best by mIoU)
- Per-class IoU tracking
- Auto visualization: loss curves, sample predictions, mIoU per class
- Save results: metrics.json, training_log.csv, sample_predictions/

Usage:
    python src/training/train_segmentation.py --config configs/segmentation_config.yaml
    python src/training/train_segmentation.py --model fcn8s --epochs 20 --batch_size 4
"""

import os
import sys
import time
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from tqdm import tqdm

from src.segmentation.model import build_face_segmentor
from src.segmentation.losses import CombinedLoss, DiceLoss, calculate_miou
from src.data.celebamask_hq import FaceSegmentationDataset


# CelebAMask-HQ class names
CLASS_NAMES = [
    'background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
    'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
    'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth'
]


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int = 42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict:
    """Load YAML config"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_args_with_config(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Override config with CLI args"""
    for key, value in vars(args).items():
        if value is not None and key in cfg:
            cfg[key] = value
    return cfg


def compute_per_class_iou(confusion_matrix: np.ndarray) -> np.ndarray:
    """Compute IoU per class from confusion matrix"""
    intersection = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - intersection
    iou = intersection / (union + 1e-10)
    return iou


def update_confusion_matrix(conf_matrix: np.ndarray, pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    """Update confusion matrix"""
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    valid = (target >= 0) & (target < num_classes)
    for p, t in zip(pred[valid], target[valid]):
        conf_matrix[t, p] += 1


# ============================================================================
# Trainer
# ============================================================================

class SegmentationTrainer:
    """
    Segmentation model trainer with full pipeline:
    - Training loop with AMP
    - Validation with mIoU (per-class + mean)
    - TensorBoard logging
    - Visualization & metrics saving
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = self.config.get('num_classes', 19)

        # Create output dirs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_type = self.config.get('model_type', 'fcn8s')
        self.run_name = f"segmentation_{model_type}_{timestamp}"
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'models/checkpoints')) / self.run_name
        self.log_dir = Path(config.get('log_dir', 'outputs/logs/segmentation')) / self.run_name
        self.metrics_dir = Path('outputs/metrics/segmentation') / self.run_name
        self.viz_dir = Path('outputs/visualizations/segmentation') / self.run_name

        for d in [self.checkpoint_dir, self.log_dir, self.metrics_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.checkpoint_dir / 'config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        # Setup
        self._setup_seed()
        self._setup_datasets()
        self._setup_model()
        self._setup_optimizer()
        self._setup_logging()

        # Tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_ce_loss': [], 'train_dice_loss': [],
            'val_ce_loss': [], 'val_dice_loss': [],
            'lr': [], 'epoch_time': [],
            'val_miou': [], 'val_pixel_acc': [],
            'val_dice': []
        }
        # Per-class IoU history
        self.per_class_iou_history = {i: [] for i in range(self.num_classes)}
        self.best_miou = 0.0

    def _setup_seed(self):
        set_seed(self.config.get('seed', 42))

    def _setup_datasets(self):
        """Setup train/val datasets & loaders"""
        img_size = self.config.get('img_size', 512)
        batch_size = self.config.get('batch_size', 8)
        num_workers = self.config.get('num_workers', 2)

        data_root = self.config.get('data_root', 'data/processed/celebamask_hq')

        print(f"[INFO] Loading segmentation datasets from {data_root}...")
        try:
            self.train_dataset = FaceSegmentationDataset(
                data_root=data_root, split='train', img_size=img_size,
                is_train=True
            )
            self.val_dataset = FaceSegmentationDataset(
                data_root=data_root, split='val', img_size=img_size,
                is_train=False
            )
        except Exception as e:
            print(f"[WARN] Cannot load real data: {e}")
            print("[INFO] Using dummy datasets for demonstration")
            self.train_dataset = DummySegDataset(num_samples=20, img_size=img_size, num_classes=self.num_classes)
            self.val_dataset = DummySegDataset(num_samples=8, img_size=img_size, num_classes=self.num_classes)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        print(f"[INFO] Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")

    def _setup_model(self):
        """Build model & loss"""
        model_type = self.config.get('model_type', 'fcn8s')
        pretrained = self.config.get('pretrained', False)
        dropout = self.config.get('dropout', 0.5)

        print(f"[INFO] Building segmentation model (type={model_type})...")
        self.model = build_face_segmentor(
            model_type=model_type,
            num_classes=self.num_classes,
            pretrained=pretrained,
            dropout=dropout
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] Model parameters: {n_params:,}")

        # Loss
        loss_type = self.config.get('loss_type', 'combined')
        if loss_type == 'combined':
            self.criterion = CombinedLoss(
                num_classes=self.num_classes,
                ce_weight=self.config.get('ce_weight', 1.0),
                dice_weight=self.config.get('dice_weight', 1.0)
            ).to(self.device)
        elif loss_type == 'dice':
            self.criterion = DiceLoss().to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)

    def _setup_optimizer(self):
        """Optimizer, scheduler, AMP scaler"""
        lr = float(self.config.get('learning_rate', 1e-4))
        wd = float(self.config.get('weight_decay', 1e-4))

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd
        )

        epochs = self.config.get('epochs', 50)
        scheduler_name = self.config.get('scheduler', 'cosine_annealing_warm_restarts')

        if scheduler_name == 'cosine_annealing_warm_restarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('T_0', 10),
                T_mult=self.config.get('T_mult', 2)
            )
        elif scheduler_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs,
                eta_min=float(self.config.get('min_lr', 1e-6))
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.1
            )

        self.scaler = GradScaler(enabled=self.config.get('use_amp', True) and self.device.type == 'cuda')

    def _setup_logging(self):
        """TensorBoard logger"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.use_tb = True
        except ImportError:
            print("[WARN] TensorBoard not available")
            self.use_tb = False

    # ------------------------------------------------------------------
    # Training & Validation
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_dice = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [TRAIN]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device).long()

            self.optimizer.zero_grad()

            with autocast(enabled=self.config.get('use_amp', True) and self.device.type == 'cuda'):
                logits = self.model(images)
                # Resize logits to match mask size if needed
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:],
                                            mode='bilinear', align_corners=False)

                loss_dict = self.criterion(logits, masks)
                loss = loss_dict['total_loss'] if isinstance(loss_dict, dict) else loss_dict
                loss_ce = loss_dict.get('loss_ce', torch.tensor(0.0)) if isinstance(loss_dict, dict) else torch.tensor(0.0)
                loss_dice = loss_dict.get('loss_dice', torch.tensor(0.0)) if isinstance(loss_dict, dict) else torch.tensor(0.0)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 10.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()
            epoch_ce += loss_ce.item() if torch.is_tensor(loss_ce) else 0.0
            epoch_dice += loss_dice.item() if torch.is_tensor(loss_dice) else 0.0
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{loss_ce.item() if torch.is_tensor(loss_ce) else 0:.4f}",
                'dice': f"{loss_dice.item() if torch.is_tensor(loss_dice) else 0:.4f}"
            })

        n = max(n_batches, 1)
        return {
            'loss': epoch_loss / n,
            'ce_loss': epoch_ce / n,
            'dice_loss': epoch_dice / n
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate one epoch"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_dice = 0.0
        n_batches = 0

        # Confusion matrix for mIoU
        conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [VAL]  ")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).long()

                with autocast(enabled=self.config.get('use_amp', True) and self.device.type == 'cuda'):
                    logits = self.model(images)
                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = F.interpolate(logits, size=masks.shape[-2:],
                                                mode='bilinear', align_corners=False)
                    loss_dict = self.criterion(logits, masks)
                    loss = loss_dict['total_loss'] if isinstance(loss_dict, dict) else loss_dict
                    loss_ce = loss_dict.get('loss_ce', torch.tensor(0.0)) if isinstance(loss_dict, dict) else torch.tensor(0.0)
                    loss_dice = loss_dict.get('loss_dice', torch.tensor(0.0)) if isinstance(loss_dict, dict) else torch.tensor(0.0)

                # Predictions
                preds = torch.argmax(logits, dim=1)
                update_confusion_matrix(conf_matrix, preds, masks, self.num_classes)

                epoch_loss += loss.item()
                epoch_ce += loss_ce.item() if torch.is_tensor(loss_ce) else 0.0
                epoch_dice += loss_dice.item() if torch.is_tensor(loss_dice) else 0.0
                n_batches += 1

        # Compute mIoU
        iou_per_class = compute_per_class_iou(conf_matrix)
        miou = np.nanmean(iou_per_class)
        pixel_acc = np.diag(conf_matrix).sum() / max(conf_matrix.sum(), 1)

        n = max(n_batches, 1)

        return {
            'loss': epoch_loss / n,
            'ce_loss': epoch_ce / n,
            'dice_loss': epoch_dice / n,
            'miou': float(miou),
            'pixel_acc': float(pixel_acc),
            'dice': float(1 - epoch_dice / n),
            'per_class_iou': iou_per_class.tolist()
        }

    # ------------------------------------------------------------------
    # Save & Visualize
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou': self.best_miou,
            'config': self.config,
            'history': self.history
        }
        torch.save(state, self.checkpoint_dir / 'latest.pth')
        if is_best:
            torch.save(state, self.checkpoint_dir / 'best.pth')
            print(f"[INFO] Saved best model at epoch {epoch+1} (mIoU={self.best_miou:.4f})")

    def save_metrics(self):
        """Save metrics to JSON & CSV"""
        metrics_out = {
            'history': self.history,
            'per_class_iou_history': self.per_class_iou_history,
            'class_names': CLASS_NAMES[:self.num_classes]
        }
        with open(self.metrics_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_out, f, indent=2, ensure_ascii=False)

        # CSV
        csv_path = self.metrics_dir / 'training_log.csv'
        keys = list(self.history.keys())
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            n = len(self.history[keys[0]])
            for i in range(n):
                row = [str(self.history[k][i]) for k in keys]
                f.write(','.join(row) + '\n')
        print(f"[INFO] Metrics saved to {self.metrics_dir}")

    def plot_training_curves(self):
        """Generate training curve plots"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not available, skipping plots")
            return

        epochs = range(1, len(self.history['train_loss']) + 1)

        # Loss plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(epochs, self.history['train_loss'], 'b-o', label='Train', markersize=4)
        axes[0].plot(epochs, self.history['val_loss'], 'r-o', label='Val', markersize=4)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(True)

        axes[1].plot(epochs, self.history['train_ce_loss'], 'b-o', label='Train CE', markersize=4)
        axes[1].plot(epochs, self.history['val_ce_loss'], 'r-o', label='Val CE', markersize=4)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('CE Loss')
        axes[1].set_title('Cross-Entropy Loss'); axes[1].legend(); axes[1].grid(True)

        axes[2].plot(epochs, self.history['train_dice_loss'], 'b-o', label='Train Dice', markersize=4)
        axes[2].plot(epochs, self.history['val_dice_loss'], 'r-o', label='Val Dice', markersize=4)
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Dice Loss')
        axes[2].set_title('Dice Loss'); axes[2].legend(); axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'loss_curves.png', dpi=100, bbox_inches='tight')
        plt.close()

        # mIoU & Accuracy
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, self.history['val_miou'], 'g-o', markersize=4)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('mIoU')
        axes[0].set_title('Validation mIoU'); axes[0].grid(True)
        axes[0].axhline(y=self.best_miou, color='r', linestyle='--', label=f'Best: {self.best_miou:.4f}')
        axes[0].legend()

        axes[1].plot(epochs, self.history['val_pixel_acc'], 'm-o', markersize=4)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Pixel Accuracy')
        axes[1].set_title('Validation Pixel Accuracy'); axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'metrics_curves.png', dpi=100, bbox_inches='tight')
        plt.close()

        # Per-class IoU (final epoch)
        if self.per_class_iou_history and any(len(v) > 0 for v in self.per_class_iou_history.values()):
            final_iou = [self.per_class_iou_history[i][-1] if self.per_class_iou_history[i] else 0
                          for i in range(self.num_classes)]

            plt.figure(figsize=(14, 6))
            class_labels = [f"{i}:{CLASS_NAMES[i][:8]}" for i in range(self.num_classes)]
            colors = plt.cm.viridis(np.linspace(0, 1, self.num_classes))
            bars = plt.bar(range(self.num_classes), final_iou, color=colors)
            plt.xlabel('Class'); plt.ylabel('IoU')
            plt.title(f'Per-Class IoU (Epoch {len(self.history["train_loss"])})')
            plt.xticks(range(self.num_classes), class_labels, rotation=45, ha='right', fontsize=8)
            plt.ylim(0, 1)
            plt.grid(True, axis='y')

            # Annotate values
            for bar, val in zip(bars, final_iou):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                          f'{val:.2f}', ha='center', va='bottom', fontsize=7)

            plt.tight_layout()
            plt.savefig(self.viz_dir / 'per_class_iou.png', dpi=100, bbox_inches='tight')
            plt.close()

        print(f"[INFO] Plots saved to {self.viz_dir}")

    def save_sample_predictions(self, epoch: int, max_images: int = 4):
        """Save sample segmentation predictions"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return

        self.model.eval()
        saved = 0
        fig, axes = plt.subplots(2, max_images, figsize=(4 * max_images, 8))
        if max_images == 1:
            axes = axes.reshape(2, 1)

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                with autocast(enabled=self.device.type == 'cuda'):
                    logits = self.model(images)
                    if logits.shape[-2:] != masks.shape[-2:]:
                        logits = F.interpolate(logits, size=masks.shape[-2:],
                                                mode='bilinear', align_corners=False)
                    preds = torch.argmax(logits, dim=1)

                for i in range(min(max_images - saved, images.size(0))):
                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    img = np.clip(img * np.array([0.229, 0.224, 0.225]) +
                                  np.array([0.485, 0.456, 0.406]), 0, 1)

                    gt_mask = masks[i].cpu().numpy()
                    pred_mask = preds[i].cpu().numpy()

                    axes[0, saved].imshow(img)
                    axes[0, saved].set_title(f'Input {i+1}')
                    axes[0, saved].axis('off')

                    axes[1, saved].imshow(pred_mask, cmap='tab20', vmin=0, vmax=self.num_classes - 1)
                    axes[1, saved].set_title(f'Pred (epoch {epoch+1}) | GT')
                    axes[1, saved].axis('off')

                    # Overlay GT contour
                    from matplotlib.patches import Rectangle
                    axes[1, saved].contour(gt_mask, levels=np.arange(0.5, self.num_classes),
                                            colors='white', linewidths=0.5, alpha=0.4)

                    saved += 1
                    if saved >= max_images:
                        break
                if saved >= max_images:
                    break

        plt.tight_layout()
        plt.savefig(self.viz_dir / f'sample_predictions_epoch_{epoch+1}.png',
                    dpi=100, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        """Full training loop"""
        epochs = self.config.get('epochs', 50)
        print(f"\n{'='*70}")
        print(f"  Starting Segmentation Training | Device: {self.device}")
        print(f"  Epochs: {epochs} | Batch Size: {self.config.get('batch_size')} | LR: {self.config.get('learning_rate')}")
        print(f"  Model: {self.config.get('model_type', 'fcn8s')}")
        print(f"  Outputs: {self.checkpoint_dir}")
        print(f"{'='*70}\n")

        total_start = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()

            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_ce_loss'].append(train_metrics['ce_loss'])
            self.history['train_dice_loss'].append(train_metrics['dice_loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_ce_loss'].append(val_metrics['ce_loss'])
            self.history['val_dice_loss'].append(val_metrics['dice_loss'])
            self.history['lr'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            self.history['val_miou'].append(val_metrics['miou'])
            self.history['val_pixel_acc'].append(val_metrics['pixel_acc'])
            self.history['val_dice'].append(val_metrics['dice'])

            # Per-class IoU
            for i, iou_val in enumerate(val_metrics['per_class_iou']):
                self.per_class_iou_history[i].append(iou_val)

            # TensorBoard
            if self.use_tb:
                self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Loss/ce_train', train_metrics['ce_loss'], epoch)
                self.writer.add_scalar('Loss/ce_val', val_metrics['ce_loss'], epoch)
                self.writer.add_scalar('Loss/dice_train', train_metrics['dice_loss'], epoch)
                self.writer.add_scalar('Loss/dice_val', val_metrics['dice_loss'], epoch)
                self.writer.add_scalar('LR', current_lr, epoch)
                self.writer.add_scalar('Metrics/mIoU', val_metrics['miou'], epoch)
                self.writer.add_scalar('Metrics/PixelAcc', val_metrics['pixel_acc'], epoch)
                for i in range(self.num_classes):
                    self.writer.add_scalar(f'IoU/class_{i}_{CLASS_NAMES[i]}',
                                            val_metrics['per_class_iou'][i], epoch)

            # Save best
            is_best = val_metrics['miou'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['miou']

            self.save_checkpoint(epoch, is_best)

            print(f"\n[Epoch {epoch+1}/{epochs}] "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"mIoU: {val_metrics['miou']:.4f} | "
                  f"PixelAcc: {val_metrics['pixel_acc']:.4f} | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

            # Save sample predictions
            if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
                self.save_sample_predictions(epoch)

        total_time = time.time() - total_start
        print(f"\n{'='*70}")
        print(f"  Training Complete! Total time: {total_time/60:.1f} min")
        print(f"  Best mIoU: {self.best_miou:.4f}")
        print(f"{'='*70}\n")

        # Final outputs
        self.save_metrics()
        self.plot_training_curves()

        # Generate final report
        report = {
            'model': self.config.get('model_type', 'fcn8s'),
            'task': 'segmentation',
            'epochs_trained': epochs,
            'total_time_minutes': round(total_time / 60, 2),
            'best_miou': self.best_miou,
            'final_miou': self.history['val_miou'][-1],
            'final_pixel_acc': self.history['val_pixel_acc'][-1],
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_lr': self.history['lr'][-1],
            'num_classes': self.num_classes,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat(),
            'outputs': {
                'checkpoints': str(self.checkpoint_dir),
                'logs': str(self.log_dir),
                'metrics': str(self.metrics_dir),
                'visualizations': str(self.viz_dir)
            },
            'final_per_class_iou': {
                CLASS_NAMES[i]: self.per_class_iou_history[i][-1] if self.per_class_iou_history[i] else 0.0
                for i in range(self.num_classes)
            }
        }
        with open(self.metrics_dir / 'final_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Final report saved to {self.metrics_dir / 'final_report.json'}")

        if self.use_tb:
            self.writer.close()

        return report


# ============================================================================
# Dummy Dataset (fallback)
# ============================================================================

class DummySegDataset(torch.utils.data.Dataset):
    """Dummy dataset for testing pipeline when real data is unavailable"""

    def __init__(self, num_samples: int = 20, img_size: int = 512, num_classes: int = 19):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.randn(3, self.img_size, self.img_size)
        mask = torch.randint(0, self.num_classes, (self.img_size, self.img_size))
        return img, mask


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Face Segmentation Model')
    parser.add_argument('--config', type=str, default='configs/segmentation_config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--model', type=str, default=None,
                        choices=['fcn8s', 'unet', 'lightweight'],
                        help='Model type')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--loss_type', type=str, default=None,
                        choices=['combined', 'dice', 'ce'])
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config = merge_args_with_config(config, args)

    # Map 'model' arg to 'model_type' config key
    if args.model is not None:
        config['model_type'] = args.model

    trainer = SegmentationTrainer(config)
    report = trainer.train()
    return report


if __name__ == '__main__':
    main()
