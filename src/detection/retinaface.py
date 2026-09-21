"""RetinaFace-style detector.

Two implementations co-exist for backwards compatibility:

1. **Checkpoint RetinaFace** — matches `retinaface_final.pth` exactly.
   Load with :func:`load_retinaface_checkpoint`.

2. **Original RetinaFace** — used by ``RetinaFaceDetector`` in inference.py
   and the unit-test fixtures. Compatible with torchvision ResNet/MobileNet
   backbones via :class:`RetinaFaceConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v2
from torchvision.models.feature_extraction import create_feature_extractor


# ---------------------------------------------------------------------------
# Original RetinaFace (backwards-compatible, used by RetinaFaceDetector)
# ---------------------------------------------------------------------------

@dataclass
class RetinaFaceConfig:
    """Hyperparameters for the RetinaFace detector."""
    num_classes: int = 2
    num_anchors: int = 9
    landmark_count: int = 5
    pretrained_backbone: bool = True
    backbone_type: Literal["resnet18", "resnet34", "resnet50", "mobilenet_v2"] = "resnet34"
    in_channels: int = 3
    feat_channels: int = 256


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(4):
            layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.LeakyReLU(0.1, inplace=True)]
        self.cls = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.cls(out)
        b, _, h, w = out.shape
        out = out.permute(0, 2, 3, 1).contiguous().view(b, h * w * self.num_anchors, self.num_classes)
        return out


class RegressionHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, out_channels: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(4):
            layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.LeakyReLU(0.1, inplace=True)]
        self.reg = nn.Conv2d(in_channels, num_anchors * out_channels, 3, padding=1)
        self.num_anchors = num_anchors
        self.out_channels = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.reg(out)
        b, _, h, w = out.shape
        out = out.permute(0, 2, 3, 1).contiguous().view(b, h * w * self.num_anchors, self.out_channels)
        return out


class FPNOriginal(nn.Module):
    def __init__(self, in_channels_list: list[int], out_channels: int = 256) -> None:
        super().__init__()
        self.lateral_convs = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in in_channels_list)
        self.output_convs = nn.ModuleList(nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list)
        self.out_channels = out_channels

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        out = [laterals[-1]]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(out[-1], size=laterals[i].shape[-2:], mode="nearest")
            out.append(laterals[i] + up)
        out = out[::-1]
        return [conv(o) for conv, o in zip(self.output_convs, out)]


class RetinaFaceOriginal(nn.Module):
    """Original RetinaFace with torchvision backbone (used by RetinaFaceDetector)."""

    def __init__(self, cfg: Optional[RetinaFaceConfig] = None) -> None:
        super().__init__()
        cfg = cfg or RetinaFaceConfig()

        if cfg.backbone_type == "mobilenet_v2":
            weights = "DEFAULT" if cfg.pretrained_backbone else None
            mnv2 = mobilenet_v2(weights=weights)
            return_nodes = {"features.4": "c3", "features.7": "c4", "features": "c5"}
            self.backbone = create_feature_extractor(mnv2, return_nodes=return_nodes)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 64, 64)
                feats = self.backbone(dummy)
                c3_channels = feats["c3"].shape[1]
                c4_channels = feats["c4"].shape[1]
                c5_channels = feats["c5"].shape[1]
        elif cfg.backbone_type == "resnet18":
            weights = "DEFAULT" if cfg.pretrained_backbone else None
            backbone = resnet18(weights=weights)
            return_nodes = {"layer2": "c3", "layer3": "c4", "layer4": "c5"}
            self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 64, 64)
                feats = self.backbone(dummy)
                c3_channels = feats["c3"].shape[1]
                c4_channels = feats["c4"].shape[1]
                c5_channels = feats["c5"].shape[1]
        elif cfg.backbone_type == "resnet34":
            weights = "DEFAULT" if cfg.pretrained_backbone else None
            backbone = resnet34(weights=weights)
            return_nodes = {"layer2": "c3", "layer3": "c4", "layer4": "c5"}
            self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 64, 64)
                feats = self.backbone(dummy)
                c3_channels = feats["c3"].shape[1]
                c4_channels = feats["c4"].shape[1]
                c5_channels = feats["c5"].shape[1]
        else:
            weights = "DEFAULT" if cfg.pretrained_backbone else None
            backbone = resnet50(weights=weights)
            return_nodes = {"layer2": "c3", "layer3": "c4", "layer4": "c5"}
            self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 64, 64)
                feats = self.backbone(dummy)
                c3_channels = feats["c3"].shape[1]
                c4_channels = feats["c4"].shape[1]
                c5_channels = feats["c5"].shape[1]

        self.fpn = FPNOriginal([c3_channels, c4_channels, c5_channels], out_channels=cfg.feat_channels)
        self.cfg = cfg
        self.cls_head = ClassificationHead(cfg.feat_channels, cfg.num_anchors, cfg.num_classes)
        self.box_head = RegressionHead(cfg.feat_channels, cfg.num_anchors, 4)
        self.lmk_head = RegressionHead(cfg.feat_channels, cfg.num_anchors, cfg.landmark_count * 2)

        for layer in [self.cls_head.cls, self.box_head.reg, self.lmk_head.reg]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        feats = self.backbone(x)
        pyramid = self.fpn([feats["c3"], feats["c4"], feats["c5"]])
        cls_logits, box_deltas, lmk_deltas = [], [], []
        for p in pyramid:
            cls_logits.append(self.cls_head(p))
            box_deltas.append(self.box_head(p))
            lmk_deltas.append(self.lmk_head(p))
        return {"cls_logits": cls_logits, "box_deltas": box_deltas, "lmk_deltas": lmk_deltas}


# ---------------------------------------------------------------------------
# Checkpoint-compatible RetinaFace (matches retinaface_final.pth)
# ---------------------------------------------------------------------------

class _BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu2(out)


class _ResNet34(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self._make_layer(_BasicBlock, 64, 3)
        self.layer2 = self._make_layer(_BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(_BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(_BasicBlock, 512, 3, stride=2)

    def _make_layer(self, block: type, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class _FPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output1 = nn.Conv2d(128, 128, 1, bias=True)
        self.output2 = nn.Conv2d(256, 128, 1, bias=True)
        self.output3 = nn.Conv2d(512, 128, 1, bias=True)
        self.merge1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=True), nn.BatchNorm2d(128), nn.ReLU())
        self.merge2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=True), nn.BatchNorm2d(128), nn.ReLU())

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = features
        p5 = self.output3(c5)
        p4 = self.output2(c4)
        p3 = self.output1(c3)
        p5_out = p5
        p4 = p4 + F.interpolate(p5_out, size=p4.shape[-2:], mode="nearest")
        p4_out = self.merge2(p4)
        p3 = p3 + F.interpolate(p4_out, size=p3.shape[-2:], mode="nearest")
        p3_out = self.merge1(p3)
        return p3_out, p4_out, p5_out


class _SSH(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv3X3 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64))
        self.conv5X5_1 = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.conv5X5_2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32))
        self.conv7X7_2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.conv7x7_3 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o1 = self.conv3X3(x)
        o2 = self.conv5X5_1(x)
        o3 = self.conv5X5_2(o2)
        o4 = self.conv7X7_2(o2)
        o5 = self.conv7x7_3(o4)
        return self.relu(torch.cat([o1, o3, o4, o5], dim=1))


class _SSHReduce(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(160, 128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce(x)


class _DetectionHead(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.class_head = nn.ModuleList([nn.Conv2d(128, out_channels, 1) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([conv(x) for conv in self.class_head], dim=1)


class RetinaFace(nn.Module):
    """RetinaFace model — two modes:

    1. **Default (no args)**: checkpoint-compatible, matches ``retinaface_final.pth``.
       Load with :func:`load_retinaface_checkpoint`.

    2. **With cfg=RetinaFaceConfig(...)**: original API used by
       ``RetinaFaceDetector`` / pipeline tests.
    """

    def __init__(self, cfg: Optional[RetinaFaceConfig] = None) -> None:
        super().__init__()
        if cfg is not None:
            self._orig = RetinaFaceOriginal(cfg)
            return

        self.fx = _ResNet34()
        self.fpn = _FPN()
        self.ssh1 = _SSH()
        self.ssh2 = _SSH()
        self.ssh3 = _SSH()
        self.ssh1_reduce = _SSHReduce()
        self.ssh2_reduce = _SSHReduce()
        self.ssh3_reduce = _SSHReduce()
        self.class_head = _DetectionHead(out_channels=4)
        self.bbox_head = _DetectionHead(out_channels=8)
        self.landmark_head = _DetectionHead(out_channels=20)

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        if hasattr(self, "_orig"):
            return self._orig.forward(x)

        c3, c4, c5 = self.fx(x)
        p3, p4, p5 = self.fpn([c3, c4, c5])
        s1 = self.ssh1_reduce(self.ssh1(p3))
        s2 = self.ssh2_reduce(self.ssh2(p4))
        s3 = self.ssh3_reduce(self.ssh3(p5))

        def reshape(t: torch.Tensor) -> torch.Tensor:
            b, c, h, w = t.shape
            return t.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        return {
            "cls_logits": [reshape(self.class_head(s1)), reshape(self.class_head(s2)), reshape(self.class_head(s3))],
            "box_deltas": [reshape(self.bbox_head(s1)), reshape(self.bbox_head(s2)), reshape(self.bbox_head(s3))],
            "lmk_deltas": [reshape(self.landmark_head(s1)), reshape(self.landmark_head(s2)), reshape(self.landmark_head(s3))],
        }


def load_retinaface_checkpoint(checkpoint_path: str, device: str = "cpu") -> RetinaFace:
    """Load ``retinaface_final.pth`` into a checkpoint-compatible RetinaFace."""
    model = RetinaFace()
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state, strict=False)
    return model


def flatten_predictions(outputs: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate per-level outputs into flat ``[B, N, C]`` tensors."""
    return {
        "cls_logits": torch.cat(outputs["cls_logits"], dim=1),
        "box_deltas": torch.cat(outputs["box_deltas"], dim=1),
        "lmk_deltas": torch.cat(outputs["lmk_deltas"], dim=1),
    }
