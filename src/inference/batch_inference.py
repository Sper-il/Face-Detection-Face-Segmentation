"""
Batch Processing

Provides utilities for processing multiple images, entire folders, and
video files through the ``FacePipeline``.  Features:

- Multi-threaded image loading (``concurrent.futures.ThreadPoolExecutor``)
- Progress bar via ``tqdm``
- Saves results as JSON (boxes + scores + mask-image paths) and
  annotated visualisation images

Typical usage::

    from src.inference.batch_inference import BatchProcessor
    processor = BatchProcessor(pipeline, output_dir="outputs/batch")
    processor.process_folder("data/images")
"""

import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except (ImportError, AttributeError, Exception):
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(_handler)
        logger.setLevel(logging.INFO)

from src.inference.pipeline import FacePipeline, draw_results

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def _load_image_bgr(path: str) -> np.ndarray:
    """Thread-safe image loader.

    Args:
        path: Absolute or relative file path to the image.

    Returns:
        BGR ``numpy.ndarray`` (uint8, HWC).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    return img


def _collect_image_paths(folder: Union[str, Path]) -> List[str]:
    """Recursively gather image file paths from *folder*.

    Args:
        folder: Root directory to search.

    Returns:
        Sorted list of absolute path strings.
    """
    folder = Path(folder)
    paths: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(str(p) for p in folder.rglob(f"*{ext}"))
        paths.extend(str(p) for p in folder.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def _save_mask_image(mask: np.ndarray, path: str) -> None:
    """Save a segmentation mask as a grayscale PNG.

    Args:
        mask: ``(H, W)`` uint8 array with class indices.
        path: Output file path.
    """
    cv2.imwrite(path, mask)


# ═══════════════════════════════════════════════════════════════════════════
# BatchProcessor
# ═══════════════════════════════════════════════════════════════════════════

class BatchProcessor:
    """High-level batch inference engine.

    Args:
        pipeline: Configured ``FacePipeline`` instance.
        output_dir: Root directory for saved results (visualisations,
            masks, JSON).  Created automatically if it does not exist.
        num_workers: Number of threads for concurrent image loading.
        save_visualizations: Whether to save annotated images.
        save_masks: Whether to save per-face mask images.
        save_json: Whether to write a results JSON file.

    Example::

        proc = BatchProcessor(pipeline, output_dir="outputs/batch")
        proc.process_folder("data/images")
    """

    def __init__(
        self,
        pipeline: FacePipeline,
        output_dir: str = "outputs/batch",
        num_workers: int = 4,
        save_visualizations: bool = True,
        save_masks: bool = True,
        save_json: bool = True,
    ):
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.save_visualizations = save_visualizations
        self.save_masks = save_masks
        self.save_json = save_json

        # Sub-directories
        self.vis_dir = self.output_dir / "visualizations"
        self.mask_dir = self.output_dir / "masks"
        self.json_path = self.output_dir / "results.json"

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        """Create output directories if needed."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_visualizations:
            self.vis_dir.mkdir(parents=True, exist_ok=True)
        if self.save_masks:
            self.mask_dir.mkdir(parents=True, exist_ok=True)

    def _process_single(
        self, image_path: str
    ) -> Dict:
        """Process one image and save outputs.

        Args:
            image_path: Path to the image file.

        Returns:
            JSON-serialisable result dict for this image.
        """
        bgr = _load_image_bgr(image_path)
        result = self.pipeline.process(bgr)
        basename = Path(image_path).stem

        record: Dict = {
            "image_path": str(image_path),
            "num_faces": len(result["faces"]),
            "faces": [],
        }

        for idx, face in enumerate(result["faces"]):
            face_record: Dict = {
                "bbox": face["bbox"],
                "score": face["score"],
            }

            # Save mask as a separate image
            if self.save_masks and face.get("mask") is not None:
                mask_filename = f"{basename}_face{idx}_mask.png"
                mask_path = str(self.mask_dir / mask_filename)
                _save_mask_image(face["mask"], mask_path)
                face_record["mask_path"] = mask_path

            record["faces"].append(face_record)

        # Save visualisation
        if self.save_visualizations:
            vis = draw_results(bgr, result["faces"])
            vis_path = str(self.vis_dir / f"{basename}_vis.jpg")
            cv2.imwrite(vis_path, vis)
            record["visualization_path"] = vis_path

        return record

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def process_images(
        self, image_paths: List[str]
    ) -> List[Dict]:
        """Process a list of image paths with progress bar.

        Args:
            image_paths: List of file-path strings.

        Returns:
            List of JSON-serialisable result dicts.
        """
        self._ensure_dirs()
        all_records: List[Dict] = []

        # Multi-threaded loading, sequential processing
        # (GPU inference is typically single-stream anyway)
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {
                pool.submit(_load_image_bgr, p): p for p in image_paths
            }
            loaded: Dict[str, np.ndarray] = {}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    loaded[path] = future.result()
                except Exception as exc:
                    logger.error("Failed to load %s: %s", path, exc)

        # Process in deterministic order
        for path in tqdm(image_paths, desc="Processing images", unit="img"):
            if path not in loaded:
                continue
            try:
                record = self._process_single(path)
                all_records.append(record)
            except Exception as exc:
                logger.error("Error processing %s: %s", path, exc)

        # Save consolidated JSON
        if self.save_json and all_records:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=2, ensure_ascii=False)
            logger.info("Results JSON saved to %s", self.json_path)

        logger.info(
            "Batch complete: %d / %d images processed.",
            len(all_records),
            len(image_paths),
        )
        return all_records

    def process_folder(
        self, folder: Union[str, Path]
    ) -> List[Dict]:
        """Discover and process all images in a folder.

        Args:
            folder: Directory containing image files.

        Returns:
            List of result dicts.
        """
        paths = _collect_image_paths(folder)
        if not paths:
            logger.warning("No images found in %s", folder)
            return []
        logger.info("Found %d images in %s", len(paths), folder)
        return self.process_images(paths)

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> List[Dict]:
        """Process every frame of a video file.

        Args:
            video_path: Path to the input video.
            output_path: Path for the output annotated video.  If
                ``None``, defaults to
                ``<output_dir>/<video_stem>_result.mp4``.

        Returns:
            List of per-frame result dicts.
        """
        self._ensure_dirs()
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path is None:
            stem = Path(video_path).stem
            output_path = str(self.output_dir / f"{stem}_result.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        records: List[Dict] = []

        pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.pipeline.process(frame)
            vis = draw_results(frame, result["faces"])
            writer.write(vis)

            records.append({
                "frame": frame_idx,
                "num_faces": len(result["faces"]),
                "faces": [
                    {"bbox": f["bbox"], "score": f["score"]}
                    for f in result["faces"]
                ],
            })

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()

        # Save frame-level JSON
        if self.save_json:
            json_out = self.output_dir / f"{Path(video_path).stem}_frames.json"
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            logger.info("Video JSON saved to %s", json_out)

        logger.info(
            "Video complete: %d frames processed. Output: %s",
            frame_idx,
            output_path,
        )
        return records


# ═══════════════════════════════════════════════════════════════════════════
# Quick self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.inference.pipeline import create_pipeline

    print("=" * 50)
    print("BatchProcessor — smoke test")
    print("=" * 50)

    pipeline = create_pipeline(device="cpu")
    processor = BatchProcessor(
        pipeline,
        output_dir="outputs/batch_test",
        save_visualizations=False,
        save_masks=False,
        save_json=False,
    )

    # Fake two images via temp files
    import tempfile

    tmp_dir = tempfile.mkdtemp()
    for i in range(2):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmp_dir, f"test_{i}.jpg"), img)

    records = processor.process_folder(tmp_dir)
    print(f"  Processed records: {len(records)}")
    for r in records:
        print(f"    {r['image_path']}: {r['num_faces']} faces")

    print("Smoke test passed ✓")
