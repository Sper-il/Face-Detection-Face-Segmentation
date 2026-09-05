"""
Demo Script - Run inference on images/videos
MEMBER 4: Implement and use this for testing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

from src.models.pipeline import create_pipeline


def process_image(
    pipeline,
    image_path: str,
    output_path: str = None,
    show: bool = True
):
    """Process a single image."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Run pipeline
    results = pipeline(image)
    
    # Visualize
    vis = pipeline.visualize(image, results)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"Saved to {output_path}")
    
    # Show if requested
    if show:
        cv2.imshow("Result", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results


def process_video(
    pipeline,
    video_path: str,
    output_path: str = None,
    show: bool = True
):
    """Process a video."""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    pbar = tqdm(desc="Processing video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Run pipeline
        results = pipeline(image)
        
        # Visualize
        vis = pipeline.visualize(image, results)
        
        # Convert back to BGR for cv2
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        
        # Write frame
        if writer:
            writer.write(vis_bgr)
        
        # Show frame
        if show:
            cv2.imshow("Video", vis_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        pbar.update(1)
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    pbar.close()
    
    print(f"Processed {frame_count} frames")


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Detection & Segmentation Demo")
    parser.add_argument("--detection", type=str, default="models/trained/detection_best.pth",
                       help="Path to detection model")
    parser.add_argument("--segmentation", type=str, default="models/trained/segmentation_best.pth",
                       help="Path to segmentation model")
    parser.add_argument("--input", type=str, required=True,
                       help="Input image or video path")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (optional)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display results")
    
    args = parser.parse_args()
    
    # Create pipeline
    print("Loading models...")
    pipeline = create_pipeline(
        detection_checkpoint=args.detection,
        segmentation_checkpoint=args.segmentation,
        device=args.device
    )
    
    # Determine input type
    input_path = Path(args.input)
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Image
            process_image(pipeline, args.input, args.output, show=not args.no_show)
        elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # Video
            process_video(pipeline, args.input, args.output, show=not args.no_show)
    else:
        print(f"Error: {args.input} is not a valid file")


if __name__ == "__main__":
    main()
