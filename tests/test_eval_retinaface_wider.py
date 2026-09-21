"""CI smoke test for the RetinaFace WIDER eval script.

The full evaluation requires a real WIDER FACE checkpoint and ~30s of inference
on GPU. The CI version generates a tiny synthetic dataset (5 images + 5 GT
rows) and confirms the script can:
  1. Run end-to-end without crashing.
  2. Write a valid ``runs/eval/retinaface_wider.json`` file with the expected
     keys.
  3. Return mAP / recall in [0, 1].

The synthetic checkpoint is random weights, so we don't assert on the absolute
metric values.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def _make_synthetic_wider(root: Path, n: int = 5) -> tuple[Path, Path]:
    """Create a tiny WIDER-style dataset and return ``(label_path, images_dir)``."""
    import cv2

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for i in range(n):
        cat = f"0--Demo"
        fname = f"0--Demo_0_Demo_img_{i}"
        (images_dir / cat).mkdir(exist_ok=True)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / cat / f"{fname}.jpg"), img)
        # One box per image (10,10 -> 90,90).
        rows.append(f"{fname}.jpg,10,10,90,90,1")
    label = root / "wider_face_val_bbx_gt.txt"
    label.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return label, images_dir


def _make_synthetic_checkpoint(path: Path) -> None:
    """Persist a minimal RetinaFace-shaped state_dict."""
    from src.detection.retinaface import RetinaFace, RetinaFaceConfig

    model = RetinaFace(RetinaFaceConfig(pretrained_backbone=False))
    torch.save({"model": model.state_dict(), "epoch": 0}, path)


def test_eval_retinaface_wider_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test of ``src.scripts.eval_retinaface_wider``."""
    label, images_dir = _make_synthetic_wider(tmp_path, n=5)
    ckpt = tmp_path / "retinaface_demo.pth"
    _make_synthetic_checkpoint(ckpt)

    # The eval script uses hardcoded relative paths:
    #   data/raw/WIDER_FACE/wider_face_split/<file>
    #   data/raw/WIDER_FACE/WIDER_VAL/images/<category>/<file>
    # We need cwd=tmp_path, so build the matching structure under tmp_path.
    wider_root = tmp_path / "data" / "raw" / "WIDER_FACE"
    wider_root.mkdir(parents=True)
    split_dir = wider_root / "wider_face_split"
    split_dir.mkdir(exist_ok=True)
    (split_dir / "wider_face_val_bbx_gt.txt").write_bytes(label.read_bytes())

    val_root = wider_root / "WIDER_VAL"
    val_root.mkdir(exist_ok=True)
    target = val_root / "images"
    target.mkdir(exist_ok=True)
    for cat in (p.name for p in images_dir.iterdir() if p.is_dir()):
        (target / cat).mkdir(exist_ok=True)
        for img in (images_dir / cat).iterdir():
            (target / cat / img.name).write_bytes(img.read_bytes())

    # We need to run the script with cwd=tmp_path so the hardcoded relative
    # paths resolve correctly. The checkpoint is already there.
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.scripts.eval_retinaface_wider",
            "--checkpoint",
            str(ckpt),
            "--split",
            "val",
            "--max-images",
            "5",
            "--conf-threshold",
            "0.0",  # ensure random-init detector emits some predictions.
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, (
        f"eval_retinaface_wider failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    out = tmp_path / "runs" / "eval" / "retinaface_wider.json"
    assert out.exists(), f"Expected output at {out}"
    metrics = json.loads(out.read_text(encoding="utf-8"))
    for key in ("mAP", "recall", "n_images", "n_gt", "n_pred"):
        assert key in metrics, f"missing metric {key!r}"
    assert 0.0 <= metrics["mAP"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
