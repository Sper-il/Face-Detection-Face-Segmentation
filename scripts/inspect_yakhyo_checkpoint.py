"""Inspect yakhyo RetinaFace checkpoint in detail - parameter shapes and structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Load checkpoint
ckpt_path = Path("models/retinaface_resnet34.pth")
state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print(f"File: {ckpt_path}")
print(f"Type: {type(state)}")
print(f"Keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")

if isinstance(state, dict):
    sd = state
else:
    sd = state

print(f"\nTotal parameters: {sum(v.numel() for v in sd.values()):,}")
print(f"Total keys: {len(sd)}")

# Detailed structure
print("\n" + "=" * 70)
print("DETAILED PARAMETER INSPECTION")
print("=" * 70)

# Sort keys alphabetically and show all
for key in sorted(sd.keys()):
    v = sd[key]
    shape = list(v.shape) if hasattr(v, 'shape') else type(v)
    dtype = str(v.dtype) if hasattr(v, 'dtype') else 'N/A'
    print(f"{key:60s} {str(shape):30s} {dtype}")

# Summary by component
print("\n" + "=" * 70)
print("SUMMARY BY COMPONENT")
print("=" * 70)

groups = {}
for k in sd.keys():
    prefix = k.split('.')[0]
    groups.setdefault(prefix, {'count': 0, 'params': 0})
    groups[prefix]['count'] += 1
    groups[prefix]['params'] += sd[k].numel()

for group, info in sorted(groups.items()):
    print(f"{group:20s}: {info['count']:3d} layers, {info['params']:12,} params")

print(f"\nTotal:             {sum(g['count'] for g in groups.values()):3d} layers, {sum(g['params'] for g in groups.values()):12,} params")
