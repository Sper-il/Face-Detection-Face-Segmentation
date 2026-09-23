"""Inspect unet_final.pth checkpoint structure - full analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Try loading the checkpoint
ckpt = torch.load('models/unet_final.pth', map_location='cpu', weights_only=False)

print("=" * 60)
print("CHECKPOINT STRUCTURE")
print("=" * 60)
print(f"Keys: {list(ckpt.keys())}")
print(f"seg_size: {ckpt.get('seg_size')}")

model_state = ckpt.get('model_state', {})
print(f"\nTotal parameters: {sum(v.numel() for v in model_state.values()):,}")

# Group keys by prefix
key_groups = {}
for k in model_state.keys():
    prefix = k.split('.')[0]
    if prefix not in key_groups:
        key_groups[prefix] = []
    key_groups[prefix].append(k)

print("\n" + "=" * 60)
print("KEY GROUPS (by first prefix)")
print("=" * 60)
for group, keys in sorted(key_groups.items()):
    print(f"\n{group}: {len(keys)} keys")
    for k in sorted(keys)[:5]:
        v = model_state[k]
        shape = v.shape if hasattr(v, 'shape') else type(v)
        print(f"  {k}: {shape}")
    if len(keys) > 5:
        print(f"  ... and {len(keys) - 5} more")

# Check decoder structure
print("\n" + "=" * 60)
print("DECODER KEYS")
print("=" * 60)
decoder_keys = [k for k in model_state.keys() if k.startswith('dec') or k.startswith('up') or k.startswith('head')]
for k in sorted(decoder_keys):
    v = model_state[k]
    shape = v.shape if hasattr(v, 'shape') else type(v)
    print(f"  {k}: {shape}")
