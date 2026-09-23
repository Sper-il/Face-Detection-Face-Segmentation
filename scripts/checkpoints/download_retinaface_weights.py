"""Download and load RetinaFace pretrained weights from yakhyo/retinaface-pytorch.

Usage:
    python scripts/download_retinaface_weights.py --backbone resnet34
    python scripts/download_retinaface_weights.py --backbone resnet18
    python scripts/download_retinaface_weights.py --backbone mobilenetv2
"""

import argparse
import os
import urllib.request
from pathlib import Path

# Base URL for yakhyo/retinaface-pytorch releases
RELEASE_BASE = "https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1"

BACKBONE_URLS = {
    "resnet18": f"{RELEASE_BASE}/retinaface_r18.pth",
    "resnet34": f"{RELEASE_BASE}/retinaface_r34.pth",
    "mobilenetv2": f"{RELEASE_BASE}/retinaface_mv2.pth",
    "mobilenetv1": f"{RELEASE_BASE}/retinaface_mv1.pth",
    "mobilenetv1_0.25": f"{RELEASE_BASE}/retinaface_mv1_0.25.pth",
    "mobilenetv1_0.50": f"{RELEASE_BASE}/retinaface_mv1_0.50.pth",
}

BACKBONE_SIZES = {
    "resnet18": "~12 MB",
    "resnet34": "~46 MB",
    "mobilenetv2": "~6 MB",
    "mobilenetv1": "~12 MB",
    "mobilenetv1_0.25": "~2 MB",
    "mobilenetv1_0.50": "~6 MB",
}


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download file with progress bar."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")

    # Create parent directory if needed
    dest.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(block_num: int, block_size: int, total_size: int):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
        bar_len = 40
        filled = int(bar_len * downloaded / total_size) if total_size > 0 else 0
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {percent}% ({downloaded // 1024}KB / {total_size // 1024}KB)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook)
    print()  # Newline after progress
    print(f"Downloaded: {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download RetinaFace pretrained weights")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet34",
        choices=list(BACKBONE_URLS.keys()),
        help="Backbone architecture",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for weights",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: <backbone>.pth)",
    )
    args = parser.parse_args()

    backbone = args.backbone
    output_dir = Path(args.output_dir)
    output_name = args.output_name or f"retinaface_{backbone}.pth"
    output_path = output_dir / output_name

    # Check if file exists
    if output_path.exists():
        print(f"File already exists: {output_path}")
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != "y":
            print("Download cancelled.")
            return

    url = BACKBONE_URLS[backbone]
    size = BACKBONE_SIZES[backbone]

    print(f"\n{'='*60}")
    print(f"Downloading RetinaFace pretrained weights")
    print(f"{'='*60}")
    print(f"Backbone:     {backbone}")
    print(f"URL:          {url}")
    print(f"Expected:     {size}")
    print(f"Output:       {output_path}")
    print(f"{'='*60}\n")

    try:
        download_file(url, output_path)
        print(f"\n✅ Success! Weight saved to: {output_path}")
        print(f"\nUsage in Python:")
        print(f"  from src.detection.retinaface import RetinaFace, RetinaFaceConfig")
        print(f"  model = RetinaFace(RetinaFaceConfig(backbone_type='{backbone}'))")
        print(f"  # Load pretrained weights...")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print(f"\nAlternative: Manually download from:")
        print(f"  {url}")


if __name__ == "__main__":
    main()
