"""
Detection Loss Functions
Based on DSFD: Focal Loss + Smooth L1 Loss + GIoU Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class FocalLoss(nn.Module):
    """Focal Loss for face detection"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss for bounding box regression"""
    def __init__(self, beta=1.0, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target, weights=None):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta)
        if weights is not None:
            loss = loss * weights.unsqueeze(1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class GIoULoss(nn.Module):
    """Generalized IoU Loss for bounding box regression"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area

        iou = inter_area / (union_area + 1e-7)

        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
        loss = 1 - giou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DetectionLoss(nn.Module):
    """Combined Detection Loss: Focal Loss + Smooth L1 + GIoU"""
    def __init__(self, num_classes=2, alpha=0.25, gamma=2.0,
                 box_weight=1.0, cls_weight=1.0, giou_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.giou_weight = giou_weight
        self.cls_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.loc_loss = SmoothL1Loss(beta=1.0)
        self.giou_loss = GIoULoss()

    def forward(self, predictions, targets):
        loc_preds, conf_preds, priors = predictions
        target_boxes = targets['boxes']
        target_labels = targets['labels']

        # Classification loss
        pos_mask = target_labels > 0
        num_pos = pos_mask.sum()
        if num_pos > 0:
            loss_cls = self.cls_loss(conf_preds[pos_mask], target_labels[pos_mask])
            loss_loc = self.loc_loss(loc_preds[pos_mask], target_boxes[pos_mask])
            loss_giou = self.giou_loss(
                self._decode_boxes(loc_preds[pos_mask], priors[pos_mask]),
                target_boxes[pos_mask])
        else:
            loss_cls = conf_preds.sum() * 0
            loss_loc = loc_preds.sum() * 0
            loss_giou = loc_preds.sum() * 0

        total_loss = self.cls_weight * loss_cls + self.box_weight * loss_loc + self.giou_weight * loss_giou
        return {'total_loss': total_loss, 'loss_cls': loss_cls, 'loss_loc': loss_loc, 'loss_giou': loss_giou}

    def _decode_boxes(self, pred_boxes, priors):
        center = priors[:, :2] + pred_boxes[:, :2] * 0.1 * priors[:, 2:]
        size = priors[:, 2:] * torch.exp(pred_boxes[:, 2:] * 0.2)
        boxes = torch.zeros_like(pred_boxes)
        boxes[:, 0] = center[:, 0] - size[:, 0] / 2
        boxes[:, 1] = center[:, 1] - size[:, 1] / 2
        boxes[:, 2] = center[:, 0] + size[:, 0] / 2
        boxes[:, 3] = center[:, 1] + size[:, 1] / 2
        return boxes


if __name__ == '__main__':
    criterion = FocalLoss()
    inputs = torch.randn(10, 2)
    targets = torch.randint(0, 2, (10,))
    loss = criterion(inputs, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    giou = GIoULoss()
    pred = torch.tensor([[0, 0, 50, 50]], dtype=torch.float32)
    target = torch.tensor([[5, 5, 55, 55]], dtype=torch.float32)
    loss = giou(pred, target)
    print(f"GIoU Loss: {loss.item():.4f}")
