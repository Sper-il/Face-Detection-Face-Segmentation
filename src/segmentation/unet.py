"""
U-Net Architecture for Face Segmentation
Based on YuvalNirkin/face_segmentation
Reference: https://github.com/YuvalNirkin/face_segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import torchvision.models as models


class DoubleConv(nn.Module):
    """Double convolution block: (Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention Gate for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    """
    U-Net Architecture for Face Segmentation
    Reference: https://github.com/YuvalNirkin/face_segmentation
    """
    def __init__(self, in_channels=3, num_classes=19, bilinear=False,
                 features=None, use_attention=True, dropout=0.0):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.features = features
        self.use_attention = use_attention

        # Encoder
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)

        # Decoder
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)

        # Attention gates
        if use_attention:
            self.att1 = AttentionGate(features[3] * 2, features[3], features[3] // factor)
            self.att2 = AttentionGate(features[3], features[2], features[2] // factor)
            self.att3 = AttentionGate(features[2], features[1], features[1] // factor)
            self.att4 = AttentionGate(features[1], features[0], features[0] // 2)

        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.dropout:
            x5 = self.dropout(x5)

        if self.use_attention:
            x = self.up1(x5, self.att1(x5, x4))
            x = self.up2(x, self.att2(x, x3))
            x = self.up3(x, self.att3(x, x2))
            x = self.up4(x, self.att4(x, x1))
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        return self.outc(x)


class UNetWithResNet(nn.Module):
    """U-Net with ResNet encoder"""
    def __init__(self, num_classes=19, backbone='resnet34', pretrained=True, dropout=0.0):
        super().__init__()

        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            skip_ch = [64, 64, 128, 256, 512]
        else:
            resnet = models.resnet50(pretrained=pretrained)
            skip_ch = [64, 256, 512, 1024, 2048]

        # Encoder
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # Decoder
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose2d(skip_ch[4], 256, 2, 2),
            DoubleConv(256 + skip_ch[3], 128))
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, 2),
            DoubleConv(128 + skip_ch[2], 64))
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, 2),
            DoubleConv(64 + skip_ch[1], 32))
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, 2),
            DoubleConv(32 + skip_ch[0], 16))

        self.final = nn.Conv2d(16, num_classes, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        if self.dropout:
            e5 = self.dropout(e5)

        d4 = self.decoder4(e5)
        d4 = torch.cat([d4, e4], dim=1)

        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.decoder1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        return self.final(d1)


class LightweightUNet(nn.Module):
    """Lightweight U-Net for limited GPU memory"""
    def __init__(self, num_classes=19, dropout=0.0):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.final = nn.Conv2d(32, num_classes, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)
        if self.dropout:
            b = self.dropout(b)

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


def build_unet(num_classes=19, model_type='standard', pretrained=False, dropout=0.0):
    if model_type == 'standard':
        return UNet(num_classes=num_classes, dropout=dropout)
    elif model_type == 'resnet':
        return UNetWithResNet(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif model_type == 'lightweight':
        return LightweightUNet(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    model = build_unet(num_classes=19, model_type='resnet', pretrained=False)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(1, 3, 512, 512)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Input: {x.shape} -> Output: {output.shape}")
