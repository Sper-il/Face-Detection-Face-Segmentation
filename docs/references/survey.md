# Literature Survey — Face Detection & Face Segmentation

> **Goal:** Compare and justify the model choice (RetinaFace + U-Net) for the
> 2-stage pipeline described in `progress_status.md` and `đề.txt`.
> **Date:** 2026-09-16
> **Status:** Decision recorded in [`../adr/0001-model-choice.md`](../adr/0001-model-choice.md).

---

## 1. Detection landscape

| Model | Backbone | Easy mAP | Hard mAP | FPS (GPU) | Strength | Weakness |
|-------|----------|---------:|---------:|----------:|----------|----------|
| **RetinaFace** (Deng et al., 2019) | **ResNet-34** + FPN | 0.967 | 0.802 | ~30 (640²) | SOTA on WIDER FACE hard set, built-in 5-point landmarks, multi-scale, scale-robust | Heavier than MTCNN; needs ~4 GB GPU for batch 8 at 640² |
| YOLO-Face v5 | CSPDarknet53 | 0.955 | 0.708 | ~60 (640²) | Real-time, simple training recipe | Lower recall on tiny / heavily occluded faces (the exact crowd case the đề targets) |
| MTCNN | 3-stage CNN | 0.852 | 0.605 | ~50 (CPU) | Lightweight, easy to deploy | Poor on extreme pose / heavy occlusion |
| DSFD | ResNet-152 | 0.973 | 0.836 | ~10 | Highest published hard-set mAP | Slow, large memory footprint |
| CenterFace | MobileNet | 0.918 | 0.620 | ~40 | Mobile-friendly | Lower recall under occlusion |
| SCRFD-10GF | EfficientNet-B0 | 0.962 | 0.788 | ~50 | Excellent speed/accuracy | Newer, smaller community |

> Numbers are from each paper's reported WIDER FACE val, single-model, single-scale.
> They are **indicative** — the purpose of this table is to justify the choice,
> not to claim parity with any specific checkpoint.

### Detection — chosen model

**RetinaFace (ResNet-34 custom backbone)** is the default for the following reasons:

1. **Crowd robustness** — explicitly designed for *unconstrained* face detection
   (scale, pose, occlusion, illumination), which is exactly what the đề asks for
   ("khuôn mặt trong ảnh đông người").
2. **Built-in landmarks** — 5 facial landmarks per face are output for free,
   giving a clean upgrade path for downstream tasks (alignment, recognition).
3. **Reasonable latency** — fits the latency target (≤ 50 ms GPU, ≤ 300 ms CPU)
   at 640² input.
4. **Mature ecosystem** — `deepinsight/insightface` reference, plenty of
   pretrained checkpoints, multiple independent reproductions.
5. **Custom implementation** — Project uses a custom ResNet-34 implementation
   (`_ResNet34` in `src/detection/retinaface.py`) that matches the trained
   checkpoint `models/retinaface_final.pth`.

YOLO-Face is the **fallback** if the latency target cannot be met at 640².
SCRFD-10GF is the **upgrade option** if we want to modernise the backbone
after the coursework deadline.

---

## 2. Segmentation landscape

| Model | Encoder | IoU (face) | FPS (GPU) | Strength | Weakness |
|-------|---------|-----------:|----------:|----------|----------|
| **U-Net** (Ronneberger, 2015) | ResNet-34 | ~0.90 | ~25 (512²) | Simple, stable, strong baseline, few failure modes | Pixel-level only — cannot distinguish overlapping faces (none expected here) |
| U-Net++ | ResNet-34 | ~0.92 | ~18 | Better feature fusion | More params, slower |
| **Mask R-CNN** (He et al., 2017) | ResNet-50 + FPN | ~0.91 | ~12 | One-shot instance seg + bbox | Heavier, more complex training, bbox & mask coupled |
| BiSeNet | ResNet-18 | ~0.88 | ~40 | Real-time parsing | Designed for multi-class face parsing, overkill for binary mask |
| EHANet / Face-Parsing | HRNet | ~0.93 | ~10 | High-quality face parsing | Heavy, requires multi-class labels |

### Segmentation — chosen model

**U-Net with a ResNet-34 ImageNet-pretrained encoder** is the default because:

1. The đề asks for a **vùng mặt (face mask)** — pixel-level binary mask is
   sufficient. No need for instance separation (each face already comes from a
   bbox).
2. **Speed** — U-Net at 512² comfortably meets the latency target on a single
   consumer GPU.
3. **Data fit** — CelebAMask-HQ provides 19-class masks; we can collapse them
   into a binary "face vs not-face" mask trivially, so any binary segmentation
   model works.
4. **Loss** — BCE + Dice (0.5/0.5) is the textbook recipe for binary
   segmentation with class imbalance.

Mask R-CNN is the **alternative** if at some point we want to drop the explicit
detection stage (one model does both). For now, keeping the two stages matches
the đề's explicit pipeline ("Detection → Segmentation").

---

## 3. Speed / accuracy trade-off summary

```
Accuracy   ████████░░  RetinaFace + U-Net       ← chosen
Speed      ███████░░░  YOLO-Face + BiSeNet
Simplicity █████████░  U-Net only (skip detection)
Instance   ████████░░  Mask R-CNN alone
```

The chosen combination sits in the "high accuracy, mid-high speed" quadrant,
which matches the **camera an ninh** use case (correctness > ultra-fast FPS).

---

## 4. Datasets

| Dataset | Role | Size | Annotation | Notes |
|---------|------|-----:|------------|-------|
| **WIDER FACE** | Detection train/val/test | 32,203 imgs / 393,703 faces | bbox + 5 landmarks + (blur/pose/illum/occlusion) | Has *ignore* flag for tiny/outside boxes |
| **CelebAMask-HQ** | Segmentation train/val/test | 30,000 imgs | 19-class semantic mask | Collapse → binary face mask (skin, nose, eyes, brows, lips, ears) |

We will:

- Convert everything to **PNG**.
- Use **80/10/10** split with `SEED=42` for both datasets.
- Honour WIDER FACE's `ignore` flag at evaluation time (do not train on it,
  do not penalise predictions inside it).

---

## 5. References (papers / repos)

- RetinaFace — Deng, J. *et al.* "RetinaFace: Single-shot Multi-level Face Localisation in the Wild." *CVPR 2020.* [arXiv:1905.00641](https://arxiv.org/abs/1905.00641) · Reference impl: [deepinsight/insightface](https://github.com/deepinsight/insightface)
- U-Net — Ronneberger, O. *et al.* "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI 2015.* [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) · Reference impl: [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
- Mask R-CNN — He, K. *et al.* "Mask R-CNN." *ICCV 2017.* [arXiv:1703.06870](https://arxiv.org/abs/1703.06870) · Reference impl: [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- WIDER FACE — Yang, S. *et al.* "WIDER FACE: A Face Detection Benchmark." *CVPR 2016.* [site](http://shuoyang1213.me/WIDERFACE/)
- CelebAMask-HQ — Lee, C.-H. *et al.* "MaskGAN: Towards Diverse and Interactive Facial Image Manipulation." *CVPR 2020.* [repo](https://github.com/switchablenorms/CelebAMask-HQ)
- SCRFD — Guo, J. *et al.* "Sample and Computation Redistribution for Efficient Face Detection." *ICLR 2023.*

---

## 6. Decision

We adopt **RetinaFace (ResNet-34 custom backbone)** for detection and **U-Net
(ResNet-34 encoder, torchvision pretrained)** for face segmentation. See
[`../adr/0001-model-choice.md`](../adr/0001-model-choice.md) for the formal
decision record.
