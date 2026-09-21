"""CLI entrypoint for the 2-stage face detection + segmentation pipeline.

Usage::

    python -m src.pipeline.run --image <path-or-dir> --output-dir data/output/

The CLI defers all real work to :class:`FaceSegmentationPipeline`. If the
model weights files don't exist yet (first run), the pipeline still works —
it just returns an empty overlay (the detector is uninitialised with random
weights and produces no detections in most cases).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline.orchestrator import FaceSegmentationPipeline
from src.pipeline.stages import iter_image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.run",
        description=(
            "Run the 2-stage face detection + segmentation pipeline on an image "
            "or directory of images."
        ),
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a PNG/JPEG image or a directory of images.",
    )
    parser.add_argument(
        "--detector-weights",
        type=str,
        default="models/retinaface_best.pth",
        help="Path to RetinaFace checkpoint.",
    )
    parser.add_argument(
        "--segmentor-weights",
        type=str,
        default="models/unet_best.pth",
        help="Path to U-Net checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output/",
        help="Where to save overlays and predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.7,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--seg-margin",
        type=float,
        default=0.1,
        help="Margin (fraction of bbox size) added before segmentation.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=50,
        help="Maximum number of faces passed to the segmentor per image.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    )

    # Build pipeline (works even without weights files; just uninitialised).
    pipe = FaceSegmentationPipeline(
        detector_weights=args.detector_weights,
        segmentor_weights=args.segmentor_weights,
        device=args.device,
        conf_threshold=args.conf_threshold,
        nms_iou=args.nms_iou,
        seg_margin=args.seg_margin,
        max_faces=args.max_faces,
    )

    # Resolve inputs.
    paths = list(iter_image_paths(args.image))
    if not paths:
        print(f"[pipeline.run] No images found at {args.image}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    results = pipe.run_batch(paths, output_dir)

    n_total = len(results)
    n_failed = sum(1 for r in results if r.failure_reason is not None)
    n_no_face = sum(1 for r in results if r.failure_reason == "no_face_detected")
    print(
        f"[pipeline.run] Done - processed {n_total} image(s); "
        f"{n_no_face} with no face; {n_failed - n_no_face} error(s). "
        f"Overlays -> {output_dir / 'vis'}. Summary -> {output_dir / 'results.json'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
