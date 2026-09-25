"""Create CelebAMask-HQ test fixtures (since real images missing)."""
import cv2
import numpy as np
from pathlib import Path

raw = Path('data/raw/CelebAMask-HQ/CelebAMask-HQ')
img_dir = raw / 'CelebA-HQ-img'
mask_root = raw / 'CelebAMask-HQ-mask-anno'
img_dir.mkdir(exist_ok=True)
(mask_root / '0').mkdir(exist_ok=True)

n_existing = len(list(img_dir.glob('*.jpg')))
target = 30
print(f"Existing images: {n_existing}, target: {target}")

for fid in range(n_existing, target):
    h, w = 128, 128
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / f'{fid}.jpg'), img)
    m = np.zeros((h, w), dtype=np.uint8)
    m[20:80, 20:80] = 255
    cv2.imwrite(str(mask_root / '0' / f'{fid:05d}_skin.png'), m)

n_imgs = len(list(img_dir.glob('*.jpg')))
n_masks = len(list((mask_root / '0').glob('*.png')))
print(f"Total images: {n_imgs}, Total masks in subdir 0: {n_masks}")
