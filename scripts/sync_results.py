"""Tổng hợp outputs từ Kaggle kernel v23 về repo local.

Chạy sau khi pull các file sau từ Kaggle vào workspace local:
  - models/retinaface_best.pth
  - models/unet_best.pth
  - models/training_history.json
  - runs/eval/retinaface_test_metrics.json
  - runs/eval/unet_test_metrics.json
  - result.png
  - demos/output/vis/test_*.png

Output:
  - models/retinaface_demo.pth + models/unet_demo.pth (hash đã cập nhật)
  - models/training_metrics.json (gộp val + test)
  - models/model_hashes.json
  - progress_status.md (section 14 thay bằng bản v23 cuối)
  - README.md (bảng metrics)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256_short(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def extract_models_zip(models_dir: Path) -> bool:
    """If `models.zip` is present (Kaggle bundles outputs into one archive),
    extract it into a `models/` subdir. Returns True when something was extracted.

    The archive usually contains entries like ``models/retinaface_demo.pth``.
    If we extract into ``models_dir`` directly that produces ``models_dir/models/...``
    (one extra level). To keep the layout flat we strip a leading ``models/``
    prefix from each entry.
    """
    zip_path = models_dir.parent / "models.zip"
    if not zip_path.exists():
        return False
    target = models_dir
    target.mkdir(parents=True, exist_ok=True)
    print(f"[zip] extracting {zip_path} -> {target}")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for m in members:
            stripped = m
            for prefix in ("models/", "./models/"):
                if m.startswith(prefix):
                    stripped = m[len(prefix):]
                    break
            if not stripped:
                continue
            target_path = target / stripped
            if m.endswith("/"):
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"[zip] extracted {len(members)} files")
    return True


def sync_weights(models_dir: Path, repo_models_dir: Path | None = None) -> dict:
    # Source files can have either `_best` (Kaggle notebook convention) or `_demo`
    # (repo convention) suffix depending on which kernel we pulled from.
    pairs = {
        "retinaface_best.pth": "retinaface_demo.pth",
        "unet_best.pth": "unet_demo.pth",
    }
    new_hashes: dict[str, str] = {}
    for src_name, dst_name in pairs.items():
        src = models_dir / src_name
        dst = models_dir / dst_name
        if not src.exists():
            # Fallback: if src not found but dst already exists with newer mtime, use it.
            print(f"[skip] {src} missing - keeping existing {dst_name}")
            continue
        shutil.copy2(src, dst)
        new_hashes[dst_name] = sha256_short(dst)
        print(f"[ok] {src_name} -> {dst_name} "
              f"({dst.stat().st_size / 1024 / 1024:.1f} MB, hash={new_hashes[dst_name]})")
    # If we only have *_demo files (e.g. pulled from a kernel that committed them
    # directly), still recompute hashes for those.
    for name in ("retinaface_demo.pth", "unet_demo.pth"):
        if name in new_hashes:
            continue
        p = models_dir / name
        if not p.exists():
            continue
        new_hashes[name] = sha256_short(p)
        print(f"[ok] using existing {name} "
              f"({p.stat().st_size / 1024 / 1024:.1f} MB, hash={new_hashes[name]})")
    if not new_hashes:
        print("[skip] no new weights found - keeping existing model_hashes.json")
        try:
            return json.loads((models_dir / "model_hashes.json").read_text())
        except FileNotFoundError:
            return {}
    # Merge with previous hashes (preserve history).
    hashes_path = models_dir / "model_hashes.json"
    existing: dict[str, str] = {}
    if hashes_path.exists():
        try:
            existing = json.loads(hashes_path.read_text())
        except Exception:
            existing = {}
    existing.update(new_hashes)
    hashes_path.write_text(json.dumps(existing, indent=2))
    print(f"[ok] wrote {hashes_path}")
    if repo_models_dir is not None:
        for name, _ in new_hashes.items():
            src = models_dir / name
            dst = repo_models_dir / name
            if not src.exists():
                continue
            if dst.exists() and sha256_short(dst) == new_hashes[name]:
                print(f"[copy] {dst} hash already up to date - skip")
                continue
            repo_models_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[copy] {src} -> {dst}")
        # Also propagate the merged hashes to the repo dir.
        repo_hashes = repo_models_dir / "model_hashes.json"
        if hashes_path.resolve() != repo_hashes.resolve():
            shutil.copy2(hashes_path, repo_hashes)
            print(f"[copy] {hashes_path} -> {repo_hashes}")
    return existing


def load_test_metrics(runs_dir: Path) -> tuple[dict | None, dict | None]:
    det_p = runs_dir / "eval" / "retinaface_test_metrics.json"
    seg_p = runs_dir / "eval" / "unet_test_metrics.json"
    det = json.loads(det_p.read_text()) if det_p.exists() else None
    seg = json.loads(seg_p.read_text()) if seg_p.exists() else None
    return det, seg


def load_history(models_dir: Path) -> dict | None:
    p = models_dir / "training_history.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def finalize_training_metrics(models_dir: Path, det_test: dict | None, seg_test: dict | None,
                              history: dict | None) -> dict:
    """Merge val history (last epoch) + test metrics into the JSON consumed by README.

    Bails out cleanly if neither history nor test metrics are present (i.e. Kaggle
    kernel hasn't finished and pulled outputs yet).
    """
    if not history and not (det_test or seg_test):
        print("[skip] no history / no test metrics available yet - "
              "models/training_metrics.json left untouched")
        try:
            return json.loads((models_dir / "training_metrics.json").read_text())
        except FileNotFoundError:
            return {}
    out = {
        "training_run": "kaggle_kernel_v23_2026-09-19",
        "platform": "Kaggle T4 GPU",
        "framework": "PyTorch 2.10+cu128",
        "datasets": {
            "detection": "WIDER FACE",
            "segmentation": "CelebAMask-HQ",
        },
    }
    if history:
        if history.get("detection"):
            last = history["detection"][-1]
            out["detection"] = {
                "epochs": len(history["detection"]),
                "best_val_mAP50": max(
                    (h.get("val", {}).get("mAP", -1) for h in history["detection"]),
                    default=-1,
                ),
                "last_train": last.get("train", {}),
                "last_val": last.get("val", {}),
            }
        if history.get("segmentation"):
            last = history["segmentation"][-1]
            out["segmentation"] = {
                "epochs": len(history["segmentation"]),
                "best_val_iou": max(
                    (h.get("val", {}).get("iou", -1) for h in history["segmentation"]),
                    default=-1,
                ),
                "last_train": last.get("train", {}),
                "last_val": last.get("val", {}),
            }
    if det_test:
        out["detection_test"] = det_test
    if seg_test:
        out["segmentation_test"] = seg_test
    p = models_dir / "training_metrics.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"[ok] wrote {p}")
    return out


def copy_extras_to_repo(models_dir: Path, runs_dir: Path, repo_models_dir: Path) -> None:
    """Propagate training_metrics.json, result.png, and test eval jsons to the repo."""
    extras = [
        (models_dir / "training_metrics.json", repo_models_dir / "training_metrics.json"),
        (models_dir.parent / "result.png", repo_models_dir.parent / "result.png"),
        (runs_dir / "eval" / "retinaface_test_metrics.json",
         runs_dir.parent / "eval" / "retinaface_test_metrics.json"),
        (runs_dir / "eval" / "unet_test_metrics.json",
         runs_dir.parent / "eval" / "unet_test_metrics.json"),
    ]
    for src, dst in extras:
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            print(f"[copy] {dst} already up to date - skip")
            continue
        shutil.copy2(src, dst)
        print(f"[copy] {src} -> {dst}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path,
                        default=REPO / "kaggle_outputs_v23" / "models")
    parser.add_argument("--runs-dir", type=Path,
                        default=REPO / "kaggle_outputs_v23" / "runs")
    parser.add_argument("--repo-models-dir", type=Path,
                        default=REPO / "models",
                        help="Where to copy the final retinaface_demo.pth / "
                             "unet_demo.pth (defaults to repo models/)")
    parser.add_argument("--repo-runs-dir", type=Path,
                        default=REPO / "runs",
                        help="Where to copy runs/eval/*_test_metrics.json "
                             "(defaults to repo runs/)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    extract_models_zip(args.models_dir)
    sync_weights(args.models_dir, args.repo_models_dir)
    det_test, seg_test = load_test_metrics(args.runs_dir)
    history = load_history(args.models_dir)
    finalize_training_metrics(args.models_dir, det_test, seg_test, history)
    copy_extras_to_repo(args.models_dir, args.runs_dir, args.repo_models_dir)
    if args.runs_dir.resolve() != args.repo_runs_dir.resolve():
        for name in ("retinaface_test_metrics.json", "unet_test_metrics.json"):
            src = args.runs_dir / "eval" / name
            if not src.exists():
                continue
            dst = args.repo_runs_dir / "eval" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[copy] {src} -> {dst}")

    print("\nSummary:")
    if det_test:
        print(f"  det test  mAP@0.5={det_test.get('mAP', 'n/a'):.4f}  "
              f"recall@0.5={det_test.get('recall', 'n/a'):.4f}")
    if seg_test:
        print(f"  seg test  IoU={seg_test.get('iou', 'n/a'):.4f}  "
              f"Dice={seg_test.get('dice', 'n/a'):.4f}  "
              f"pixel_acc={seg_test.get('pixel_acc', 'n/a'):.4f}")
    print("\nNext: run `python scripts/update_progress_status.py` to refresh docs.")


if __name__ == "__main__":
    main()

