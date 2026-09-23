import torch

checkpoint = torch.load('e:/Face-Detection-Face-Segmentation/models/unet_final.pth', map_location='cpu')
print("model_state keys (first 30):")
for i, k in enumerate(list(checkpoint['model_state'].keys())[:30]):
    print(f"  {k}")
print(f"\nTotal keys: {len(checkpoint['model_state'])}")
print(f"seg_size: {checkpoint['seg_size']}")
