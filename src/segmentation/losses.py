"""
Segmentation Loss Functions
Dice Loss, Focal Loss, Cross Entropy, and Combined Losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        num_classes = pred.size(1)
        if target.dim() == 3:
            target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        else:
            target_one_hot = target.float()
        
        pred = F.softmax(pred, dim=1)
        pred_flat = pred.contiguous().view(pred.size(0), pred.size(1), -1)
        target_flat = target_one_hot.contiguous().view(target_one_hot.size(0), target_one_hot.size(1), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=2)
        dice = (2. * intersection + self.smooth) / (pred_flat.sum(dim=2) + target_flat.sum(dim=2) + self.smooth)
        
        if self.reduction == 'mean':
            return 1 - dice.mean()
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss for segmentation"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pred_prob = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.size(1)).permute(0, 3, 1, 2).float()
        p_t = (pred_prob * target_one_hot).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


class CombinedLoss(nn.Module):
    """Combined CE + Dice Loss"""
    def __init__(self, num_classes=19, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        total = self.ce_weight * ce + self.dice_weight * dice
        return {'total_loss': total, 'loss_ce': ce, 'loss_dice': dice}


class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice Loss"""
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        num_classes = pred.size(1)
        if target.dim() == 3:
            target = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
        
        pred = F.softmax(pred, dim=1)
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)
        
        intersection = (pred_flat * target_flat).sum(dim=2)
        fps = (pred_flat * (1 - target_flat)).sum(dim=2)
        fns = ((1 - pred_flat) * target_flat).sum(dim=2)
        
        tversky = (intersection + self.smooth) / (intersection + self.alpha * fps + self.beta * fns + self.smooth)
        return 1 - tversky.mean()


def calculate_miou(pred, target, num_classes=19):
    """Calculate mean IoU"""
    pred = pred.view(-1)
    target = target.view(-1)
    iou = []
    for i in range(1, num_classes):
        intersection = ((pred == i) & (target == i)).sum()
        union = ((pred == i) | (target == i)).sum()
        if union > 0:
            iou.append((intersection / union).item())
    return sum(iou) / len(iou) if iou else 0.0


if __name__ == '__main__':
    B, C, H, W = 2, 19, 256, 256
    pred = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))
    
    dice = DiceLoss()
    loss = dice(pred, target)
    print(f"Dice Loss: {loss.item():.4f}")
    
    combined = CombinedLoss(num_classes=C)
    losses = combined(pred, target)
    print(f"Total Loss: {losses['total_loss'].item():.4f}")
