"""CI smoke test for the U-Net segmentation eval script.

Generates a tiny synthetic checkpoint + dataset and verifies that
``src.scripts.eval_segmentation`` writes a valid ``runs/unet/eval_results.json``
with IoU / Dice in ``[0, 1]``.

Run directly::

    PYTHONPATH=. python tests/test_eval_unet_smoke.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _make_synthetic_seg_split(root: Path, n: int = 4) -> tuple[Path, Path]:
    """Build a tiny ``CelebAMask-HQ``-style split and return ``(imgs, masks)``."""
    imgs = root / "images"
    masks = root / "masks"
    imgs.mkdir(parents=True)
    masks.mkdir(parents=True)
    for i in range(n):
        rng = np.random.default_rng(i)
        img = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        # Add a centered square (face proxy) so IoU is non-trivial.
        mask[32:96, 32:96] = 255
        cv2.imwrite(str(imgs / f"{i:05d}.png"), img)
        cv2.imwrite(str(masks / f"{i:05d}.png"), mask)
    return imgs, masks


def _make_synthetic_unet_checkpoint(path: Path) -> None:
    from src.segmentation.unet import UNet, UNetConfig

    model = UNet(UNetConfig(pretrained_encoder=False))
    torch.save({"model": model.state_dict(), "epoch": 0}, path)


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        imgs_dir, masks_dir = _make_synthetic_seg_split(tmp / "data" / "processed" / "segmentation" / "val", n=4)
        ckpt = tmp / "unet_demo.pth"
        _make_synthetic_unet_checkpoint(ckpt)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.scripts.eval_segmentation",
                "--config",
                str(REPO / "src" / "configs" / "unet.yaml"),
                "--checkpoint",
                str(ckpt),
                "--max-images",
                "4",
                "--batch-size",
                "2",
                "--image-size",
                "128",
                "--output",
                "runs/unet/eval_results.json",
            ],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        if result.returncode != 0:
            print(f"FAILED: stdout={result.stdout}\nstderr={result.stderr}")
            return 1
        out = tmp / "runs" / "unet" / "eval_results.json"
        if not out.exists():
            print(f"Missing output: {out}")
            return 1
        metrics = json.loads(out.read_text(encoding="utf-8"))
        for key in ("iou", "dice", "pixel_acc"):
            assert key in metrics, f"missing metric {key!r}"
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"
        print(f"PASS: UNet eval smoke test OK. metrics={metrics}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
