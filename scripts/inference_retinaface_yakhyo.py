"""Inference script for RetinaFace yakhyo pretrained model.

Usage:
    python scripts/inference_retinaface_yakhyo.py --image path/to/image.jpg
    python scripts/inference_retinaface_yakhyo.py --image path/to/image.jpg --output result.jpg
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.retinaface_yakhyo import load_retinaface_yakhyo
from src.detection.anchors import AnchorConfig, generate_anchors, decode_predictions


# Prior box configuration (matching yakhyo training)
PRIOR_CONFIG = {
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'image_size': 640,
}


def pre_process(image: np.ndarray, target_size: int = 640) -> tuple:
    """Preprocess image for RetinaFace inference.

    Args:
        image: BGR image
        target_size: Target input size

    Returns:
        Preprocessed tensor and metadata
    """
    h, w = image.shape[:2]
    target = target_size

    # Calculate resize ratio
    ratio = min(target / h, target / w)
    new_h, new_w = int(round(h * ratio)), int(round(w * ratio))

    # Resize with padding
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target - new_w
    pad_h = target - new_h

    # Pad to square
    padded = cv2.copyMakeBorder(
        resized, 0, pad_h, 0, pad_w,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # Convert BGR to RGB and normalize
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    tensor = (rgb - mean) / std

    # Convert to tensor [C, H, W]
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).float()

    return tensor.unsqueeze(0), {
        'ratio': ratio,
        'pad_w': pad_w,
        'pad_h': pad_h,
        'orig_size': (h, w),
    }


def post_process(
    cls_logits: list,
    box_deltas: list,
    lmk_deltas: list,
    anchors: np.ndarray,
    meta: dict,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.4,
) -> dict:
    """Post-process RetinaFace outputs.

    Args:
        cls_logits: Classification logits per level
        box_deltas: Box deltas per level
        lmk_deltas: Landmark deltas per level
        anchors: Prior boxes
        meta: Preprocessing metadata
        conf_threshold: Confidence threshold
        nms_threshold: NMS IoU threshold

    Returns:
        Dictionary with boxes, scores, landmarks
    """
    # Concatenate all levels
    cls = np.concatenate([c.cpu().numpy() for c in cls_logits], axis=1)[0]
    box = np.concatenate([b.cpu().numpy() for b in box_deltas], axis=1)[0]
    lmk = np.concatenate([l.cpu().numpy() for l in lmk_deltas], axis=1)[0]

    # Get face scores (class 1)
    scores = 1.0 / (1.0 + np.exp(-cls[:, 1]))

    # Filter by confidence
    keep = scores > conf_threshold
    if not keep.any():
        return {'boxes': np.array([]), 'scores': np.array([]), 'landmarks': None}

    boxes = box[keep]
    landmarks = lmk[keep]
    scores = scores[keep]
    anchors = anchors[keep]

    # Decode boxes
    boxes = decode_boxes_numpy(anchors, boxes)

    # Apply NMS
    keep_nms = nms(boxes, scores, nms_threshold)
    boxes = boxes[keep_nms]
    scores = scores[keep_nms]
    landmarks = landmarks[keep_nms]

    # Rescale to original size
    ratio = meta['ratio']
    orig_h, orig_w = meta['orig_size']

    boxes = boxes / ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    landmarks = landmarks / ratio
    landmarks[:, 0::2] = np.clip(landmarks[:, 0::2], 0, orig_w)
    landmarks[:, 1::2] = np.clip(landmarks[:, 1::2], 0, orig_h)

    return {
        'boxes': boxes.astype(np.float32),
        'scores': scores.astype(np.float32),
        'landmarks': landmarks.astype(np.float32),
    }


def decode_boxes_numpy(anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Decode box deltas to boxes (xyxy format)."""
    anchors = anchors.astype(np.float32)
    deltas = deltas.astype(np.float32)

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = deltas[:, 0] * 0.1 * widths
    dy = deltas[:, 1] * 0.1 * heights
    dw = deltas[:, 2] * 0.2 * widths
    dh = deltas[:, 3] * 0.2 * heights

    pred_ctr_x = ctr_x + dx
    pred_ctr_y = ctr_y + dy
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    boxes = np.zeros_like(deltas)
    boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
    boxes[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
    boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
    boxes[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2

    return boxes


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Non-maximum suppression."""
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def draw_detections(image: np.ndarray, results: dict) -> np.ndarray:
    """Draw detection results on image."""
    vis = image.copy()

    boxes = results['boxes']
    scores = results['scores']
    landmarks = results.get('landmarks')

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(int, box)

        # Draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f"{score:.2f}"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw landmarks
        if landmarks is not None:
            lmk = landmarks[i]
            for j in range(5):
                x = int(lmk[j * 2])
                y = int(lmk[j * 2 + 1])
                cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

    return vis


def generate_prior_boxes(config: dict, image_size: int = 640) -> np.ndarray:
    """Generate prior boxes for RetinaFace."""
    min_sizes = config['min_sizes']
    steps = config['steps']

    priors = []
    for level, (min_size, step) in enumerate(zip(min_sizes, steps)):
        feature_size = image_size // step
        for h in range(feature_size):
            for w in range(feature_size):
                cx = (w + 0.5) * step / image_size
                cy = (h + 0.5) * step / image_size

                for size in min_size:
                    w_prior = size / image_size
                    h_prior = size / image_size
                    priors.append([
                        cx - 0.5 * w_prior,
                        cy - 0.5 * h_prior,
                        cx + 0.5 * w_prior,
                        cy + 0.5 * h_prior,
                    ])

    return np.array(priors)


def main():
    parser = argparse.ArgumentParser(description="RetinaFace inference with yakhyo weights")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--weights", type=str,
                        default="models/retinaface_resnet34.pth",
                        help="Pretrained weights path")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms-threshold", type=float, default=0.4, help="NMS threshold")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to use")
    parser.add_argument("--no-landmarks", action="store_true", help="Skip landmark detection")
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return

    print(f"Image shape: {image.shape}")
    print(f"Loading model from: {args.weights}")

    # Load model
    device = torch.device(args.device)
    model = load_retinaface_yakhyo(args.weights, backbone_type="resnet34", device=args.device)
    model.to(device)
    model.eval()

    # Generate priors
    priors = generate_prior_boxes(PRIOR_CONFIG, image_size=640)
    priors_pixel = priors * 640  # Convert to pixel coordinates
    priors_pixel = np.concatenate([priors_pixel, np.ones((len(priors_pixel), 4)) * 640], axis=1)

    # Preprocess
    tensor, meta = pre_process(image)
    tensor = tensor.to(device)

    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(tensor)

    # Post-process
    results = post_process(
        outputs['cls_logits'],
        outputs['box_deltas'],
        outputs['lmk_deltas'],
        priors_pixel,
        meta,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
    )

    # Print results
    n_faces = len(results['boxes'])
    print(f"\nDetected {n_faces} faces:")

    for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
        x1, y1, x2, y2 = box
        print(f"  Face {i+1}: conf={score:.3f}, bbox=[({x1:.0f},{y1:.0f}), ({x2:.0f},{y2:.0f})]")

    # Draw and save
    vis = draw_detections(image, results)

    if args.output:
        cv2.imwrite(args.output, vis)
        print(f"\nResult saved to: {args.output}")
    else:
        # Show with OpenCV window
        cv2.imshow("RetinaFace Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
