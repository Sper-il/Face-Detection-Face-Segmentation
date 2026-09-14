"""
Face Segmentation Model - Fully Implemented
Based on YuvalNirkin/face_segmentation
Reference: https://github.com/YuvalNirkin/face_segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import torchvision.models as models


class VGG16(nn.Module):
    """VGG16 backbone for FCN"""
    
    def __init__(self, pretrained=True):
        super(VGG16, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier
        
    def forward(self, x):
        x = self.features(x)
        return x


class FCN8s(nn.Module):
    """
    Fully Convolutional Network 8s for Face Segmentation
    Inspired by: https://github.com/YuvalNirkin/face_segmentation
    
    Architecture:
    - VGG16 encoder
    - FCN-8s decoder with skip connections
    """
    
    def __init__(
        self, 
        num_classes: int = 19,  # CelebAMask-HQ has 19 classes
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super(FCN8s, self).__init__()
        
        self.num_classes = num_classes
        
        # VGG16 backbone
        vgg = models.vgg16(pretrained=pretrained)
        
        # Encoder (VGG16 features)
        # Block 1: conv1_1, conv1_2, pool1 -> 64 channels
        # Block 2: conv2_1, conv2_2, pool2 -> 128 channels  
        # Block 3: conv3_1, conv3_2, conv3_3, pool3 -> 256 channels
        # Block 4: conv4_1, conv4_2, conv4_3, pool4 -> 512 channels
        # Block 5: conv5_1, conv5_2, conv5_3, pool5 -> 512 channels
        
        self.features = vgg.features
        
        # Decoder (FCN-8s)
        # Score layers
        self.score_fr = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # Upconv layers
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 
                                            kernel_size=4, stride=2, padding=1)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 
                                            kernel_size=16, stride=8, padding=4)
        
        # Dropout
        self.dropout = nn.Dropout2d(p=dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # Encoder: sequential feature extraction to save memory and avoid re-computing
        pool3 = self.features[:17](x)           # Layers 0..16 -> pool3 (256 channels)
        pool4 = self.features[17:24](pool3)      # Layers 17..23 -> pool4 (512 channels)
        pool5 = self.features[24:](pool4)        # Layers 24..30 -> pool5 (512 channels)
        
        # Decoder
        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)
        
        upscore2 = self.upscore2(score_pool4)
        upscore2 = upscore2[:, :, :score_pool3.shape[2], :score_pool3.shape[3]] + score_pool3
        
        upscore8 = self.upscore8(upscore2)
        upscore8 = upscore8[:, :, :x.shape[2], :x.shape[3]]
        
        return upscore8


class AttentionGate(nn.Module):
    """Attention Gate for skip connections in U-Net"""
    
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    """Decoder block with skip connection"""
    
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(DecoderBlock, self).__init__()
        
        self.use_attention = use_attention
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2 if not use_attention else out_channels + out_channels // 2, 
                      out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        if use_attention:
            self.attention = AttentionGate(out_channels, out_channels, out_channels // 2)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatch
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        if self.use_attention:
            skip = self.attention(x, skip)
        
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetFaceSeg(nn.Module):
    """
    U-Net for Face Segmentation with ResNet34 encoder
    More modern architecture with attention gates and skip connections
    """
    
    def __init__(
        self,
        num_classes: int = 19,
        pretrained: bool = True,
        dropout: float = 0.5,
        use_attention: bool = True
    ):
        super(UNetFaceSeg, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Encoder with pretrained ResNet34
        resnet = models.resnet34(pretrained=pretrained)
        
        # Encoder
        self.encoder1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu
        )  # 64, H/2, W/2
        self.pool1 = nn.MaxPool2d(2)  # 64, H/4, W/4
        
        self.encoder2 = resnet.layer1  # 64 -> 64
        self.pool2 = nn.MaxPool2d(2)  # 64, H/8, W/8
        
        self.encoder3 = resnet.layer2  # 64 -> 128
        self.pool3 = nn.MaxPool2d(2)  # 128, H/16, W/16
        
        self.encoder4 = resnet.layer3  # 128 -> 256
        self.pool4 = nn.MaxPool2d(2)  # 256, H/32, W/32
        
        self.encoder5 = resnet.layer4  # 256 -> 512
        self.pool5 = nn.MaxPool2d(2)  # 512, H/64, W/64
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )  # 1024, H/64, W/64
        
        # Decoder
        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 512, H/32, W/32
        self.dec5 = DecoderBlock(512, 512, use_attention)  # 512 -> 256
        
        self.up4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)  # 256, H/16, W/16
        self.dec4 = DecoderBlock(256, 256, use_attention)  # 256 -> 128
        
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # 128, H/8, W/8
        self.dec3 = DecoderBlock(128, 128, use_attention)  # 128 -> 64
        
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 64, H/4, W/4
        self.dec2 = DecoderBlock(64, 64, use_attention)  # 64 -> 64
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 32, H/2, W/2
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),  # 32 + 64 (skip from encoder1)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final output
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)  # 64
        e1_pool = self.pool1(e1)  # 64, H/2, W/2
        
        e2 = self.encoder2(e1_pool)  # 64
        e2_pool = self.pool2(e2)  # 64, H/4, W/4
        
        e3 = self.encoder3(e2_pool)  # 128
        e3_pool = self.pool3(e3)  # 128, H/8, W/8
        
        e4 = self.encoder4(e3_pool)  # 256
        e4_pool = self.pool4(e4)  # 256, H/16, W/16
        
        e5 = self.encoder5(e4_pool)  # 512
        e5_pool = self.pool5(e5)  # 512, H/32, W/32
        
        # Bottleneck
        b = self.bottleneck(e5_pool)  # 1024, H/64, W/64
        
        # Decoder
        d5 = self.up5(b)  # 512, H/32, W/32
        d5 = self.dec5(d5, e5)  # 512 -> 256
        
        d4 = self.up4(d5)  # 256, H/16, W/16
        d4 = self.dec4(d4, e4)  # 256 -> 128
        
        d3 = self.up3(d4)  # 128, H/8, W/8
        d3 = self.dec3(d3, e3)  # 128 -> 64
        
        d2 = self.up2(d3)  # 64, H/4, W/4
        d2 = self.dec2(d2, e2)  # 64 -> 64
        
        d1 = self.up1(d2)  # 32, H/2, W/2
        # Handle size mismatch
        diffY = e1.size(2) - d1.size(2)
        diffX = e1.size(3) - d1.size(3)
        d1 = F.pad(d1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # 32 + 64 -> 32
        
        # Output
        out = self.final(d1)  # num_classes, H/2, W/2
        
        # Upsample to original size
        out = F.interpolate(out, size=(x.shape[2], x.shape[3]), 
                           mode='bilinear', align_corners=False)
        
        return out


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net for face segmentation
    Similar to face_seg_fcn8s_300_no_aug model
    Designed for lower resolution images and limited GPU memory
    """
    
    def __init__(
        self,
        num_classes: int = 19,
        dropout: float = 0.0
    ):
        super(LightweightUNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        
        # Output
        self.final = nn.Conv2d(32, num_classes, 1)
        
        # Dropout
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        if self.dropout:
            b = self.dropout(b)
        
        # Decoder
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


def build_face_segmentor(
    model_type: str = 'unet', 
    num_classes: int = 19, 
    pretrained: bool = True, 
    dropout: float = 0.5
):
    """
    Build face segmentation model
    
    Args:
        model_type: 'fcn8s', 'unet', or 'lightweight'
        num_classes: Number of segmentation classes (19 for CelebAMask-HQ)
        pretrained: Use pretrained backbone
        dropout: Dropout rate
        
    Returns:
        Segmentation model
    """
    if model_type == 'fcn8s':
        return FCN8s(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif model_type == 'unet':
        return UNetFaceSeg(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif model_type == 'lightweight':
        return LightweightUNet(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Aliases for package-level imports
# FaceSegmentor wraps the default model (UNetFaceSeg)
FaceSegmentor = UNetFaceSeg
build_segmentation_model = build_face_segmentor


# Quick test
if __name__ == '__main__':
    # Test models
    for model_type in ['fcn8s', 'unet', 'lightweight']:
        model = build_face_segmentor(model_type=model_type, num_classes=19, pretrained=False)
        x = torch.randn(1, 3, 512, 512)
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"{model_type}: Input {x.shape} -> Output {out.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
