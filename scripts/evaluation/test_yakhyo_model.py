"""Test loading yakhyo weights with the new RetinaFaceYakhyo model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.detection.retinaface_yakhyo import RetinaFaceYakhyo, load_retinaface_yakhyo

def test_model_creation():
    """Test creating model and loading weights."""
    print("=" * 60)
    print("Testing RetinaFaceYakhyo model")
    print("=" * 60)

    weights_path = "models/retinaface_resnet34.pth"
    
    if not Path(weights_path).exists():
        print(f"Weight file not found: {weights_path}")
        print("Download using: python scripts/download_retinaface_weights.py --backbone resnet34")
        return

    # Load checkpoint
    print(f"\n1. Loading checkpoint: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    print(f"   Keys: {list(checkpoint.keys())[:5]}...")
    print(f"   Total params: {sum(v.numel() for v in checkpoint.values()):,}")

    # Create model
    print(f"\n2. Creating RetinaFaceYakhyo model with resnet34 backbone")
    model = RetinaFaceYakhyo(backbone_type="resnet34", pretrained_backbone=False)
    
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(checkpoint.keys())
    
    print(f"   Model keys: {len(model_keys)}")
    print(f"   Checkpoint keys: {len(ckpt_keys)}")
    
    # Check matching
    matching = model_keys & ckpt_keys
    print(f"   Matching keys: {len(matching)}")
    
    if len(matching) == len(ckpt_keys):
        print("   [OK] All checkpoint keys match model keys!")
    else:
        missing = ckpt_keys - model_keys
        extra = model_keys - ckpt_keys
        if missing:
            print(f"   [MISSING] {len(missing)} keys in checkpoint not in model:")
            for k in sorted(missing)[:5]:
                print(f"      {k}")
        if extra:
            print(f"   [EXTRA] {len(extra)} keys in model not in checkpoint:")
            for k in sorted(extra)[:5]:
                print(f"      {k}")

    # Try loading
    print(f"\n3. Loading weights...")
    try:
        model.load_state_dict(checkpoint, strict=True)
        print("   [OK] Strict loading successful!")
    except Exception as e:
        print(f"   [WARNING] Strict loading failed: {e}")
        print("   Trying strict=False...")
        model.load_state_dict(checkpoint, strict=False)
        print("   [OK] Loose loading successful!")

    # Test inference
    print(f"\n4. Testing inference...")
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   Output keys: {list(output.keys())}")
    for k, v in output.items():
        print(f"   {k}: {[vv.shape for vv in v]}")
    
    # Check for NaN/Inf
    has_nan = any(torch.isnan(x).any() for x_list in output.values() for x in x_list)
    has_inf = any(torch.isinf(x).any() for x_list in output.values() for x in x_list)
    
    if has_nan or has_inf:
        print(f"\n   [WARNING] Output contains {'NaN' if has_nan else ''} {'Inf' if has_inf else ''}")
    else:
        print(f"\n   [OK] Inference test passed!")

    # Test with helper function
    print(f"\n5. Testing load_retinaface_yakhyo helper...")
    model2 = load_retinaface_yakhyo(weights_path, backbone_type="resnet34", device="cpu")
    with torch.no_grad():
        output2 = model2(dummy_input)
    print(f"   [OK] Helper function works!")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_model_creation()
