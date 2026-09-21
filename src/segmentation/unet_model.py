"""U-Net model architecture - matches user's training script.

Standard U-Net with:
- Encoder: enc1 → enc2 → enc3 → enc4 → bottleneck (Ch 64 → 128 → 256 → 512 → 1024)
- Decoder: up4 + dec4 → up3 + dec3 → up2 + dec2 → up1 + dec1
- Output: 2 channels (background + face), logits (no softmax in forward)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    """Configuration for the U-Net segmentor."""
    in_ch: int = 3
    out_ch: int = 2
    base_ch: int = 64


class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2 with bias=True (matches user's training script)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Standard U-Net for face segmentation (matches unet_final.pth)."""

    def __init__(self, cfg: Optional[UNetConfig] = None) -> None:
        super().__init__()
        cfg = cfg or UNetConfig()
        self.cfg = cfg
        chs = [cfg.base_ch, cfg.base_ch * 2, cfg.base_ch * 4, cfg.base_ch * 8, cfg.base_ch * 16]

        # Encoder
        self.enc1 = DoubleConv(cfg.in_ch, chs[0])
        self.enc2 = DoubleConv(chs[0], chs[1])
        self.enc3 = DoubleConv(chs[1], chs[2])
        self.enc4 = DoubleConv(chs[2], chs[3])
        self.bottleneck = DoubleConv(chs[3], chs[4])
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(chs[4], chs[3], 2, stride=2)
        self.dec4 = DoubleConv(chs[4], chs[3])
        self.up3 = nn.ConvTranspose2d(chs[3], chs[2], 2, stride=2)
        self.dec3 = DoubleConv(chs[3], chs[2])
        self.up2 = nn.ConvTranspose2d(chs[2], chs[1], 2, stride=2)
        self.dec2 = DoubleConv(chs[2], chs[1])
        self.up1 = nn.ConvTranspose2d(chs[1], chs[0], 2, stride=2)
        self.dec1 = DoubleConv(chs[1], chs[0])

        self.out_conv = nn.Conv2d(chs[0], cfg.out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)  # logits (no softmax)


def load_unet(weights: str | None = None, device: str = "cpu", strict: bool = False) -> UNet:
    """Load U-Net model with weights."""
    model = UNet(UNetConfig())
    if weights is not None:
        from pathlib import Path
        state = torch.load(weights, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(state, dict):
            if "model_state" in state:
                state = state["model_state"]
            elif "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]

        model.load_state_dict(state, strict=strict)

    return model.to(device).eval()
