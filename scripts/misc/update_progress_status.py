"""Cập nhật progress_status.md (section 14) và README.md từ
models/training_metrics.json + runs/eval/*.json.

Chạy SAU khi `sync_results.py` đã hoàn tất.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_metrics(models_dir: Path, runs_dir: Path) -> tuple[dict, dict | None, dict | None]:
    tm_p = models_dir / "training_metrics.json"
    tm = json.loads(tm_p.read_text()) if tm_p.exists() else {}
    det_p = runs_dir / "eval" / "retinaface_test_metrics.json"
    seg_p = runs_dir / "eval" / "unet_test_metrics.json"
    det_test = json.loads(det_p.read_text()) if det_p.exists() else None
    seg_test = json.loads(seg_p.read_text()) if seg_p.exists() else None
    return tm, det_test, seg_test


def fmt(value, default="n/a") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_section_14(tm: dict, det_test: dict | None, seg_test: dict | None) -> str:
    det = tm.get("detection", {})
    seg = tm.get("segmentation", {})
    lines = []
    lines.append("## 14. Latest Training Results (v23, Kaggle T4 GPU, 2026-09-19)")
    lines.append("")
    lines.append(f"**Kernel:** `speril/face-detection-face-segmentation` (v23)")
    lines.append("")
    lines.append("### Detection (RetinaFace)")
    lines.append("")
    lines.append(f"- Epochs: **{det.get('epochs', '?')}**  |  Image size: 640  |  Batch size: 8")
    lines.append(f"- Best val mAP@0.5: **{fmt(det.get('best_val_mAP50'))}**")
    lines.append(f"- Last train loss: total={fmt(det.get('last_train', {}).get('total'))}  "
                 f"cls={fmt(det.get('last_train', {}).get('cls'))}  "
                 f"box={fmt(det.get('last_train', {}).get('box'))}")
    if det_test:
        lines.append(f"- **Test** mAP@0.5: **{fmt(det_test.get('mAP'))}**  "
                     f"recall@0.5: **{fmt(det_test.get('recall'))}**  "
                     f"precision@0.5: **{fmt(det_test.get('precision'))}**")
    lines.append("")
    lines.append("### Segmentation (U-Net)")
    lines.append("")
    lines.append(f"- Epochs: **{seg.get('epochs', '?')}**  |  Image size: 512  |  Batch size: 8")
    lines.append(f"- Best val IoU: **{fmt(seg.get('best_val_iou'))}**")
    lines.append(f"- Last train loss: total={fmt(seg.get('last_train', {}).get('total'))}  "
                 f"bce={fmt(seg.get('last_train', {}).get('bce'))}  "
                 f"dice={fmt(seg.get('last_train', {}).get('dice'))}")
    if seg_test:
        lines.append(f"- **Test** IoU: **{fmt(seg_test.get('iou'))}**  "
                     f"Dice: **{fmt(seg_test.get('dice'))}**  "
                     f"pixel_acc: **{fmt(seg_test.get('pixel_acc'))}**")
    lines.append("")
    lines.append("### Artefacts (in repo)")
    lines.append("")
    lines.append("- `models/retinaface_demo.pth` (~130 MB)")
    lines.append("- `models/unet_demo.pth` (~94 MB)")
    lines.append("- `models/training_metrics.json`")
    lines.append("- `models/model_hashes.json`")
    lines.append("- `runs/eval/retinaface_test_metrics.json`")
    lines.append("- `runs/eval/unet_test_metrics.json`")
    lines.append("- `result.png`")
    lines.append("")
    return "\n".join(lines)


def update_progress_status(tm: dict, det_test: dict | None, seg_test: dict | None,
                            out_path: Path) -> None:
    has_new = bool(det_test or seg_test) and bool(tm)
    if not has_new:
        # Nothing meaningful to write yet — leave the previous "Latest Training Results"
        # section intact so we don't clobber historical results while the Kaggle
        # kernel is still running.
        print(f"[skip] no new test metrics available - {out_path} left untouched")
        return
    text = out_path.read_text(encoding="utf-8", errors="replace")
    new_section = build_section_14(tm, det_test, seg_test)
    # Replace the entire "## 14." section through end-of-file.
    pattern = re.compile(r"## 14\..*$", re.DOTALL)
    updated = pattern.sub(new_section, text)
    out_path.write_text(updated, encoding="utf-8")
    print(f"[ok] updated {out_path}")


def update_readme(tm: dict, det_test: dict | None, seg_test: dict | None,
                   out_path: Path) -> None:
    if not (det_test or seg_test):
        print(f"[skip] no new test metrics available - {out_path} left untouched")
        return
    text = out_path.read_text(encoding="utf-8", errors="replace")
    new_block = (
        f"\n## Latest Training Results (v23, Kaggle T4 GPU, 2026-09-19)\n\n"
        f"- Detection (RetinaFace): 12,880 train / 3,226 val / 16,097 test  |  "
        f"mAP@0.5={fmt(det_test.get('mAP') if det_test else None)}  "
        f"recall@0.5={fmt(det_test.get('recall') if det_test else None)}\n"
        f"- Segmentation (U-Net):   24,000 train / 3,000 val / 3,000 test  |  "
        f"IoU={fmt(seg_test.get('iou') if seg_test else None)}  "
        f"Dice={fmt(seg_test.get('dice') if seg_test else None)}  "
        f"pixel_acc={fmt(seg_test.get('pixel_acc') if seg_test else None)}\n"
    )
    # Replace any "## Latest Training Results" block.
    pattern = re.compile(r"## Latest Training Results.*?(?=\n## |\Z)", re.DOTALL)
    if pattern.search(text):
        updated = pattern.sub(new_block.strip() + "\n", text)
    else:
        updated = text.rstrip() + "\n" + new_block
    out_path.write_text(updated, encoding="utf-8")
    print(f"[ok] updated {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=REPO / "models")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--progress-status", type=Path, default=REPO / "progress_status.md")
    parser.add_argument("--readme", type=Path, default=REPO / "README.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tm, det_test, seg_test = load_metrics(args.models_dir, args.runs_dir)
    update_progress_status(tm, det_test, seg_test, args.progress_status)
    update_readme(tm, det_test, seg_test, args.readme)


if __name__ == "__main__":
    main()
