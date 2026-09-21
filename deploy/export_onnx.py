"""Export the trained models to ONNX.

Usage::

    python -m deploy.export_onnx \
        --detector-weights models/retinaface_best.pth \
        --segmentor-weights models/unet_best.pth \
        --output-dir models/

Writes:

- ``models/retinaface.onnx`` (input: ``[1, 3, 640, 640]``)
- ``models/unet.onnx``       (input: ``[1, 3, 512, 512]``)

The script also runs a `onnxruntime` smoke test on each exported model to
catch shape / dtype mismatches before they reach production.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained models to ONNX.")
    parser.add_argument("--detector-weights", type=str, default="models/retinaface_best.pth")
    parser.add_argument("--segmentor-weights", type=str, default="models/unet_best.pth")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--detector-size", type=int, default=640)
    parser.add_argument("--segmentor-size", type=int, default=512)
    parser.add_argument(
        "--no-smoke-test",
        action="store_true",
        help="Skip the onnxruntime roundtrip test (export only).",
    )
    return parser.parse_args()


def export_detector(weights: str | None, output_path: Path, image_size: int) -> None:
    """Export the RetinaFace detector to ONNX."""
    from src.detection.anchors import AnchorConfig
    from src.detection.inference import RetinaFaceDetector
    from src.detection.retinaface import flatten_predictions

    cfg = AnchorConfig(image_size=image_size, strides=(16, 32, 64))
    detector = RetinaFaceDetector(
        weights=weights, device="cpu", anchor_cfg=cfg, pretrained=False
    )
    detector.model.eval()

    # The detector wraps torchvision preprocessing — bypass it for export.
    dummy = torch.randn(1, 3, image_size, image_size)

    # The torch.onnx.export needs the model to return the raw head outputs so
    # that the consumer can run their own NMS / decode. We expose that via
    # ``flatten_predictions``.
    class _Wrapper(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return flatten_predictions(self.model(x))

    wrapper = _Wrapper(detector.model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["cls_logits", "box_deltas", "lmk_deltas"],
        dynamic_axes={
            "input": {0: "batch", 2: "h", 3: "w"},
            "cls_logits": {0: "batch", 1: "anchors"},
            "box_deltas": {0: "batch", 1: "anchors"},
            "lmk_deltas": {0: "batch", 1: "anchors"},
        },
    )
    print(f"[export] detector -> {output_path}")


def export_segmentor(weights: str | None, output_path: Path, image_size: int) -> None:
    """Export the U-Net segmentor to ONNX."""
    from src.segmentation.inference import UNetSegmentor
    from src.segmentation.unet import UNet

    seg = UNetSegmentor(weights=weights, device="cpu", image_size=image_size)
    seg.model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)

    torch.onnx.export(
        seg.model,
        (dummy,),
        str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["prob"],
        dynamic_axes={
            "input": {0: "batch", 2: "h", 3: "w"},
            "prob": {0: "batch", 2: "h", 3: "w"},
        },
    )
    print(f"[export] segmentor -> {output_path}")


def smoke_test(path: Path, expected_shape: tuple[int, ...]) -> None:
    """Run the exported model through ``onnxruntime`` and check the shape."""
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"[export] onnxruntime not installed — skipping smoke test for {path}")
        return

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    x = np.random.rand(*in_meta.shape).astype(np.float32)
    y = sess.run([out_meta.name], {in_meta.name: x})[0]
    if tuple(y.shape) != expected_shape:
        raise RuntimeError(
            f"Shape mismatch for {path}: expected {expected_shape}, got {tuple(y.shape)}"
        )
    print(f"[export] smoke test OK — {path.name} output shape {y.shape}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.detector_weights).exists():
        detector_out = output_dir / "retinaface.onnx"
        export_detector(args.detector_weights, detector_out, args.detector_size)
        if not args.no_smoke_test:
            smoke_test(
                detector_out,
                expected_shape=(1, 3 * (args.detector_size // 16) ** 2 + 9 * (args.detector_size // 32) ** 2 + 9 * (args.detector_size // 64) ** 2, 2),
            )
    else:
        print(f"[export] detector weights not found at {args.detector_weights} — skipping")

    if Path(args.segmentor_weights).exists():
        segmentor_out = output_dir / "unet.onnx"
        export_segmentor(args.segmentor_weights, segmentor_out, args.segmentor_size)
        if not args.no_smoke_test:
            smoke_test(segmentor_out, expected_shape=(1, 1, args.segmentor_size, args.segmentor_size))
    else:
        print(f"[export] segmentor weights not found at {args.segmentor_weights} — skipping")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
