"""I/O helpers — image loading, YAML configs, reproducibility."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def load_image_bgr(path: str | os.PathLike) -> np.ndarray:
    """Read an image from disk as BGR ``np.uint8`` ``[H, W, 3]``."""
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def save_image_bgr(path: str | os.PathLike, image: np.ndarray) -> None:
    """Save a BGR image, creating the parent directory if needed."""
    path = Path(path)
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"cv2.imwrite failed for {path}")


def load_yaml(path: str | os.PathLike) -> dict[str, Any]:
    """Read a YAML config file as a plain dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create the directory (and parents) if missing; return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if available) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
