"""Test loading RetinaFace pretrained weights from yakhyo/retinaface-pytorch.

This script:
1. Downloads the pretrained weights (if not present)
2. Creates a model with the same architecture
3. Attempts to load the weights with best-effort matching
4. Tests inference on a sample image

NOTE: Due to architectural differences between yakhyo's implementation
and this codebase, full weight compatibility is NOT guaranteed.
The yakhyo implementation uses a custom RetinaFace model with SSH modules,
while this codebase uses a simplified FPN-based architecture.

For best results, either:
1. Use this model for training from scratch
2. Fine-tune on the pretrained weights (partial loading)
3. Use the weights with yakhyo's original codebase
"""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.retinaface import RetinaFace, RetinaFaceConfig


def inspect_checkpoint(ckpt_path: Path):
    """Inspect checkpoint structure."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {ckpt_path}")
    print(f"{'='*60}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict):
        print(f"Keys: {list(state.keys())}")
        if "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif "model" in state:
            sd = state["model"]
        elif "state_dict" in state:
            sd = state["state_dict"]
        else:
            sd = state
    else:
        sd = state

    print(f"\nTotal parameters: {sum(v.numel() for v in sd.values()):,}")

    # Group by prefix
    groups = {}
    for k in sd.keys():
        prefix = k.split(".")[0]
        groups.setdefault(prefix, []).append(k)

    for group, keys in sorted(groups.items()):
        print(f"\n{group}: {len(keys)} keys")
        for k in sorted(keys)[:5]:
            v = sd[k]
            shape = list(v.shape) if hasattr(v, "shape") else type(v)
            print(f"  {k}: {shape}")
        if len(keys) > 5:
            print(f"  ... and {len(keys) - 5} more")

    return sd


def create_model_and_compare(backbone: str, checkpoint: dict):
    """Create model and compare parameter names."""
    print(f"\n{'='*60}")
    print(f"Creating model with backbone: {backbone}")
    print(f"{'='*60}")

    cfg = RetinaFaceConfig(backbone_type=backbone, pretrained_backbone=False)
    model = RetinaFace(cfg)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(checkpoint.keys())

    print(f"\nModel parameters: {len(model_keys)}")
    print(f"Checkpoint parameters: {len(ckpt_keys)}")

    # Find matching keys
    matching = model_keys & ckpt_keys
    print(f"Matching keys: {len(matching)}")

    # Keys only in model
    model_only = model_keys - ckpt_keys
    if model_only:
        print(f"\nKeys only in model ({len(model_only)}):")
        for k in sorted(model_only)[:10]:
            print(f"  {k}")

    # Keys only in checkpoint
    ckpt_only = ckpt_keys - model_keys
    if ckpt_only:
        print(f"\nKeys only in checkpoint ({len(ckpt_only)}):")
        for k in sorted(ckpt_only)[:10]:
            print(f"  {k}")

    return model, matching


def partial_load(model: RetinaFace, checkpoint: dict, matching_keys: set):
    """Attempt partial loading of matching weights."""
    print(f"\n{'='*60}")
    print(f"Attempting partial weight loading...")
    print(f"{'='*60}")

    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for k in matching_keys:
        if k in model_state and model_state[k].shape == checkpoint[k].shape:
            model_state[k] = checkpoint[k]
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)

    model.load_state_dict(model_state, strict=False)

    print(f"Loaded: {len(loaded_keys)} parameters")
    if loaded_keys:
        print(f"  Sample: {sorted(loaded_keys)[:5]}")

    if skipped_keys:
        print(f"Skipped (shape mismatch): {len(skipped_keys)}")
        for k in skipped_keys[:5]:
            print(f"  {k}: model={model.state_dict()[k].shape}, ckpt={checkpoint[k].shape}")

    return model


def test_inference(model: RetinaFace):
    """Test model inference."""
    print(f"\n{'='*60}")
    print(f"Testing inference...")
    print(f"{'='*60}")

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nOutput keys: {list(output.keys())}")
    for k, v in output.items():
        print(f"  {k}: {[vv.shape for vv in v]}")

    # Check for NaN/Inf
    has_nan = any(torch.isnan(x).any() for x in output.values() for x in x)
    has_inf = any(torch.isinf(x).any() for x in output.values() for x in x)

    if has_nan or has_inf:
        print(f"\n⚠️  Warning: Output contains {'NaN' if has_nan else ''} {'Inf' if has_inf else ''}")
    else:
        print(f"\n✅ Inference test passed!")

    return output


def main():
    parser = argparse.ArgumentParser(description="Test RetinaFace weight loading")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained .pth file",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet34",
        choices=["resnet18", "resnet34", "resnet50", "mobilenetv2"],
        help="Backbone architecture (must match checkpoint)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        print(f"\nDownload weights using:")
        print(f"  python scripts/download_retinaface_weights.py --backbone {args.backbone}")
        return

    # Inspect checkpoint
    checkpoint = inspect_checkpoint(ckpt_path)

    # Create model and compare
    model, matching = create_model_and_compare(args.backbone, checkpoint)

    # Try partial loading
    model = partial_load(model, checkpoint, matching)

    # Test inference
    test_inference(model)

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Backbone: {args.backbone}")
    print(f"Matching parameters: {len(matching)}")
    print(f"\nNote: Full weight compatibility requires matching architectures.")
    print(f"The yakhyo weights may require the original codebase for best results.")


if __name__ == "__main__":
    main()
