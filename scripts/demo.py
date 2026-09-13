"""
Demo Script - Face Detection & Segmentation CLI

Run inference on images, videos, webcam feeds, or entire folders.

Usage::

    # Single image
    python scripts/demo.py --image input.jpg --output output.jpg

    # Video file
    python scripts/demo.py --video input.mp4 --output output.mp4

    # Webcam (real-time, press 'q' to quit)
    python scripts/demo.py --webcam

    # Batch folder
    python scripts/demo.py --folder input/ --output output/
"""

import sys
import argparse
import os
from pathlib import Path

# Ensure the project root is on sys.path so that ``src.*`` imports work
# regardless of the working directory.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.inference.detector import FaceDetector
from src.inference.segmentor import FaceSegmentor
from src.inference.pipeline import FacePipeline, create_pipeline, draw_results
from src.inference.batch_inference import BatchProcessor


# ═══════════════════════════════════════════════════════════════════════════
# Mode handlers
# ═══════════════════════════════════════════════════════════════════════════

def demo_image(
    pipeline: FacePipeline,
    image_path: str,
    output_path: str = None,
    show: bool = True,
) -> None:
    """Process a single image.

    Args:
        pipeline: Configured ``FacePipeline``.
        image_path: Path to the input image.
        output_path: Where to save the annotated result (optional).
        show: If ``True``, display the result in a window.
    """
    if not os.path.isfile(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return

    print(f"Processing: {image_path}")
    result = pipeline.process(bgr)
    vis = draw_results(bgr, result["faces"])

    num_faces = len(result["faces"])
    print(f"  Detected {num_faces} face(s)")
    for i, face in enumerate(result["faces"]):
        print(f"    Face {i}: bbox={[int(v) for v in face['bbox']]}, "
              f"score={face['score']:.4f}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, vis)
        print(f"  Saved to: {output_path}")

    if show:
        cv2.imshow("Face Detection & Segmentation", vis)
        print("  Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def demo_video(
    pipeline: FacePipeline,
    video_path: str,
    output_path: str = None,
    show: bool = True,
) -> None:
    """Process a video file frame by frame.

    Args:
        pipeline: Configured ``FacePipeline``.
        video_path: Path to the input video.
        output_path: Where to save the annotated video (optional).
        show: If ``True``, display frames in real time.
    """
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}  ({total} frames, {fps} fps)")
    pbar = tqdm(total=total, desc="Video", unit="frame")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process(frame)
        vis = draw_results(frame, result["faces"])

        if writer:
            writer.write(vis)

        if show:
            cv2.imshow("Video — press 'q' to quit", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n  Stopped by user.")
                break

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"  Processed {frame_count} frames.")
    if output_path:
        print(f"  Saved to: {output_path}")


def demo_webcam(
    pipeline: FacePipeline,
    camera_id: int = 0,
) -> None:
    """Real-time webcam inference.

    Press **q** to stop.

    Args:
        pipeline: Configured ``FacePipeline``.
        camera_id: Camera device index (default ``0``).
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (device {camera_id})")
        return

    print(f"Webcam started (device {camera_id}). Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        result = pipeline.process(frame)
        vis = draw_results(frame, result["faces"])

        # Show face count on screen
        num = len(result["faces"])
        cv2.putText(
            vis, f"Faces: {num}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA,
        )

        cv2.imshow("Webcam — press 'q' to quit", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stopped.")


def demo_folder(
    pipeline: FacePipeline,
    folder_path: str,
    output_dir: str = None,
) -> None:
    """Batch-process all images in a folder.

    Args:
        pipeline: Configured ``FacePipeline``.
        folder_path: Path to the input folder.
        output_dir: Where to save results (default ``outputs/batch``).
    """
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return

    output_dir = output_dir or "outputs/batch"
    processor = BatchProcessor(
        pipeline=pipeline,
        output_dir=output_dir,
        save_visualizations=True,
        save_masks=True,
        save_json=True,
    )
    records = processor.process_folder(folder_path)
    print(f"\nDone. {len(records)} image(s) processed → {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="Face Detection & Segmentation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/demo.py --image photo.jpg --output result.jpg\n"
            "  python scripts/demo.py --video clip.mp4 --output clip_out.mp4\n"
            "  python scripts/demo.py --webcam\n"
            "  python scripts/demo.py --folder imgs/ --output results/\n"
        ),
    )

    # --- Input mode (mutually exclusive) ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image", type=str, default=None,
        help="Path to a single input image.",
    )
    group.add_argument(
        "--video", type=str, default=None,
        help="Path to an input video file.",
    )
    group.add_argument(
        "--webcam", action="store_true", default=False,
        help="Use webcam for real-time inference.",
    )
    group.add_argument(
        "--folder", type=str, default=None,
        help="Path to a folder of images for batch processing.",
    )

    # --- Output ---
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (file for image/video, directory for folder).",
    )

    # --- Model checkpoints ---
    parser.add_argument(
        "--det-checkpoint", type=str,
        default="weights/trained/detection_best.pth",
        help="Path to detection model checkpoint.",
    )
    parser.add_argument(
        "--seg-checkpoint", type=str,
        default="weights/trained/segmentation_best.pth",
        help="Path to segmentation model checkpoint.",
    )

    # --- Device & thresholds ---
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: 'cuda' or 'cpu' (default: cuda).",
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.5,
        help="Detection confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--nms-threshold", type=float, default=0.4,
        help="NMS IoU threshold (default: 0.4).",
    )

    # --- Display ---
    parser.add_argument(
        "--no-show", action="store_true", default=False,
        help="Do not display results in a window.",
    )
    parser.add_argument(
        "--camera-id", type=int, default=0,
        help="Webcam device index (default: 0).",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Build pipeline
    print("=" * 50)
    print(" Face Detection & Segmentation Demo")
    print("=" * 50)
    print(f"Device          : {args.device}")
    print(f"Det checkpoint  : {args.det_checkpoint}")
    print(f"Seg checkpoint  : {args.seg_checkpoint}")
    print(f"Conf threshold  : {args.conf_threshold}")
    print(f"NMS threshold   : {args.nms_threshold}")
    print()

    print("Loading models...")
    pipeline = create_pipeline(
        det_checkpoint=args.det_checkpoint,
        seg_checkpoint=args.seg_checkpoint,
        device=args.device,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
    )
    print("Models loaded.\n")

    # Dispatch to the selected mode
    if args.image:
        demo_image(
            pipeline,
            image_path=args.image,
            output_path=args.output,
            show=not args.no_show,
        )
    elif args.video:
        demo_video(
            pipeline,
            video_path=args.video,
            output_path=args.output,
            show=not args.no_show,
        )
    elif args.webcam:
        demo_webcam(pipeline, camera_id=args.camera_id)
    elif args.folder:
        demo_folder(
            pipeline,
            folder_path=args.folder,
            output_dir=args.output,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
