"""
Evaluation Metrics for Detection & Segmentation
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


def compute_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        box1: (N, 4) tensor/array [x1, y1, x2, y2]
        box2: (M, 4) tensor/array [x1, y1, x2, y2]
    
    Returns:
        iou: (N, M) IoU matrix
    """
    if isinstance(box1, np.ndarray):
        box1 = torch.tensor(box1, dtype=torch.float32)
    if isinstance(box2, np.ndarray):
        box2 = torch.tensor(box2, dtype=torch.float32)
    
    if box1.dim() == 1:
        box1 = box1.unsqueeze(0)
    if box2.dim() == 1:
        box2 = box2.unsqueeze(0)
    
    N = box1.size(0)
    M = box2.size(0)
    
    # Intersection
    inter_x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))  # (N, M)
    inter_y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
    inter_x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
    inter_y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    # Union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # (N,)
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # (M,)
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
    
    iou = inter_area / (union + 1e-7)
    return iou


def compute_ap(recalls, precisions):
    """
    Compute Average Precision using 11-point interpolation.
    
    Args:
        recalls: list of recall values
        precisions: list of precision values
    
    Returns:
        ap: Average Precision value
    """
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def compute_map(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Compute mAP@IoU_threshold for detection.
    
    Args:
        pred_boxes: list of (N_i, 4) predicted boxes per image
        pred_scores: list of (N_i,) confidence scores per image
        gt_boxes: list of (M_i, 4) ground truth boxes per image
        iou_threshold: IoU threshold for matching
    
    Returns:
        mAP value
    """
    all_scores = []
    all_matches = []
    total_gt = 0
    
    for preds, scores, gts in zip(pred_boxes, pred_scores, gt_boxes):
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(gts, torch.Tensor):
            gts = gts.cpu().numpy()
        
        total_gt += len(gts)
        
        if len(preds) == 0:
            continue
        
        if len(gts) == 0:
            all_scores.extend(scores.tolist())
            all_matches.extend([False] * len(scores))
            continue
        
        iou_matrix = compute_iou(
            torch.tensor(preds), torch.tensor(gts)
        ).numpy()
        
        # Sort by score
        sorted_idx = np.argsort(-scores)
        matched_gt = set()
        
        for idx in sorted_idx:
            all_scores.append(scores[idx])
            
            best_gt = np.argmax(iou_matrix[idx])
            best_iou = iou_matrix[idx, best_gt]
            
            if best_iou >= iou_threshold and best_gt not in matched_gt:
                all_matches.append(True)
                matched_gt.add(best_gt)
            else:
                all_matches.append(False)
    
    if total_gt == 0:
        return 0.0
    
    # Sort by score
    sorted_idx = np.argsort(-np.array(all_scores))
    all_matches = np.array(all_matches)[sorted_idx]
    
    # Compute precision-recall
    tp_cumsum = np.cumsum(all_matches)
    fp_cumsum = np.cumsum(~all_matches)
    
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    return compute_ap(recalls, precisions)


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy for segmentation"""
    if isinstance(pred, torch.Tensor):
        correct = (pred == target).sum().item()
        total = target.numel()
    else:
        correct = np.sum(pred == target)
        total = target.size
    return correct / (total + 1e-7)


def compute_miou(pred, target, num_classes):
    """
    Compute mean IoU for segmentation.
    
    Args:
        pred: (H, W) predicted class indices
        target: (H, W) ground truth class indices
        num_classes: number of classes
    
    Returns:
        miou: mean IoU
        per_class_iou: dict of per-class IoU
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    ious = {}
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        gt_mask = (target == cls)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            ious[cls] = float('nan')
        else:
            ious[cls] = intersection / union
    
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    
    return miou, ious


def evaluate_detection(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Evaluate detection results.
    
    Args:
        pred_boxes: list of predicted boxes per image
        pred_scores: list of confidence scores per image
        gt_boxes: list of ground truth boxes per image
        iou_threshold: IoU threshold for matching
    
    Returns:
        dict with mAP, precision, recall
    """
    mAP = compute_map(pred_boxes, pred_scores, gt_boxes, iou_threshold)
    return {
        'mAP': mAP,
        'mAP@0.5': mAP,
    }


def evaluate_segmentation(pred, target, num_classes=19):
    """
    Evaluate segmentation results.
    
    Args:
        pred: predicted segmentation map
        target: ground truth segmentation map
        num_classes: number of classes
    
    Returns:
        dict with mIoU, pixel_accuracy, per_class_iou
    """
    miou, per_class_iou = compute_miou(pred, target, num_classes)
    pixel_acc = compute_pixel_accuracy(pred, target)
    return {
        'mIoU': miou,
        'pixel_accuracy': pixel_acc,
        'per_class_iou': per_class_iou,
    }
