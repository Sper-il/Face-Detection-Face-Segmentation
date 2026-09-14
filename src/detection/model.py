"""
DSFD Face Detection Model - Fully Implemented
Based on Tencent/FaceDetection-DSFD
Reference: https://github.com/Tencent/FaceDetection-DSFD
CVPR 2019 Paper: DSFD: Dual Shot Face Detector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from typing import Tuple, List, Optional, Dict
import math


class L2Norm(nn.Module):
    """L2 Normalization layer from DSFD"""
    def __init__(self, n_channels, scale=20.0):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).add_(self.eps).sqrt()
        x = x / norm
        return self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x


class FEM(nn.Module):
    """
    Feature Enhancement Module from DSFD paper
    Expands channels and creates multi-scale features
    """
    def __init__(self, channel_size):
        super(FEM, self).__init__()
        self.cs = channel_size
        # Dilated convolutions for multi-scale feature extraction
        self.cpm1 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm2 = nn.Conv2d(self.cs, 256, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm3 = nn.Conv2d(256, 128, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm4 = nn.Conv2d(256, 128, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm5 = nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1)
        
    def forward(self, x):
        x1_1 = F.relu(self.cpm1(x), inplace=True)
        x1_2 = F.relu(self.cpm2(x), inplace=True)
        x2_1 = F.relu(self.cpm3(x1_2), inplace=True)
        x2_2 = F.relu(self.cpm4(x1_2), inplace=True)
        x3_1 = F.relu(self.cpm5(x2_2), inplace=True)
        return torch.cat([x1_1, x2_1, x3_1], dim=1)


class DeepHeadModule(nn.Module):
    """Deep Head Module for Progressive Anchor"""
    def __init__(self, input_channels, output_channels):
        super(DeepHeadModule, self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._mid_channels = min(self._input_channels, 256)
        
        self.conv1 = nn.Conv2d(self._input_channels, self._mid_channels, 
                                kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self._mid_channels, self._mid_channels, 
                                kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self._mid_channels, self._mid_channels, 
                                kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self._mid_channels, self._output_channels, 
                                kernel_size=1, dilation=1, stride=1, padding=0)
    
    def forward(self, x):
        return self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)), inplace=True))


class PriorBox:
    """Prior Box generation from DSFD"""
    def __init__(self, cfg, min_sizes, max_sizes):
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.image_size = cfg.get('img_size', 640)
        self.feature_maps = cfg.get('feature_maps', [80, 40, 20, 10, 5, 3])
        self.steps = cfg.get('steps', [4, 8, 16, 32, 64, 128])
        self.clip = cfg.get('clip', True)
        self.variance = cfg.get('variance', [0.1, 0.2])
        self.aspect_ratios = cfg.get('aspect_ratios', [[1.5], [1.5], [1.5], [1.5], [1.5], [1.5]])
        
    def forward(self):
        mean = []
        for k in range(len(self.feature_maps)):
            feat_size = self.feature_maps[k]
            step = self.steps[k]
            
            # Handle feat_size as [H, W] list or single int
            if isinstance(feat_size, (list, tuple)):
                feat_h, feat_w = feat_size[0], feat_size[1]
            else:
                feat_h, feat_w = feat_size, feat_size
            
            for i in range(feat_h):
                for j in range(feat_w):
                    # Unit center x, y
                    cx = (j + 0.5) * step / self.image_size
                    cy = (i + 0.5) * step / self.image_size
                    
                    # Aspect ratio: 1, rel size: min_size
                    s_k = self.min_sizes[k] / self.image_size
                    mean += [cx, cy, s_k, s_k]
                    
                    # Additional aspect ratios
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]
                    
                    # Larger box for max_size if available
                    if self.max_sizes and k < len(self.max_sizes):
                        s_k_prime = math.sqrt(self.min_sizes[k] * self.max_sizes[k]) / self.image_size
                        mean += [cx, cy, s_k_prime, s_k_prime]
        
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class Detect:
    """Detection post-processing from DSFD"""
    def __init__(self, num_classes, background_label, num_thresh, 
                 conf_thresh, nms_thresh, variance=[0.1, 0.2], top_k=500):
        self.num_classes = num_classes
        self.background_label = background_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = variance
        self.num_thresh = num_thresh
        
    def forward(self, loc_data, conf_data, prior_data):
        """
        Forward pass for detection
        Args:
            loc_data: (N, num_priors*4) location predictions
            conf_data: (N, num_priors, num_classes) confidence predictions
            prior_data: (num_priors, 4) prior boxes
        """
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        
        # Decode boxes
        decoded_boxes = self._decode_boxes(loc_data, prior_data)
        
        for i in range(num):
            conf_scores = conf_preds[i].clone()
            
            for cl in range(1, self.num_classes):  # Skip background
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                
                if scores.dim() == 0 or scores.numel() == 0:
                    continue
                    
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                
                if boxes.numel() == 0:
                    continue
                
                ids, count = self._nms(boxes, scores, self.nms_thresh, self.top_k)
                
                if count > 0:
                    output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
        
        return output
    
    def _decode_boxes(self, loc, priors):
        """Decode location predictions"""
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * self.variance[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * self.variance[1])
        ), 1)
        # Convert from center to corner format
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes
    
    def _nms(self, boxes, scores, overlap=0.5, top_k=200):
        """Non-Maximum Suppression"""
        keep = scores.new(scores.size(0)).zero_().long()
        if boxes.numel() == 0:
            return keep, 0
            
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1) * (y2 - y1)
        
        v, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        
        count = 0
        while idx.numel() > 0:
            i = idx[-1]
            keep[count] = i
            count += 1
            if idx.size(0) == 1:
                break
            idx = idx[:-1]
            
            xx1 = torch.clamp(torch.index_select(x1, 0, idx), min=x1[i].item())
            yy1 = torch.clamp(torch.index_select(y1, 0, idx), min=y1[i].item())
            xx2 = torch.clamp(torch.index_select(x2, 0, idx), max=x2[i].item())
            yy2 = torch.clamp(torch.index_select(y2, 0, idx), max=y2[i].item())
            
            w = torch.clamp(xx2 - xx1, min=0.0)
            h = torch.clamp(yy2 - yy1, min=0.0)
            inter = w * h
            
            rem_areas = torch.index_select(area, 0, idx)
            union = (rem_areas - inter) + area[i]
            IoU = inter / (union + 1e-7)
            
            idx = idx[IoU.le(overlap)]
            
        return keep, count


class DSFDDetector(nn.Module):
    """
    DSFD (Dual Shot Face Detector) Model
    Adapted from: https://github.com/Tencent/FaceDetection-DSFD
    
    Key Features:
    - ResNet50/ResNet34 backbone
    - Feature Pyramid Network (FPN)
    - Feature Enhancement Module (FEM)
    - Progressive Anchor (PA)
    """
    
    # Default configuration for WIDER FACE 640x640
    default_cfg = {
        'img_size': 640,
        'feature_maps': [80, 40, 20, 10, 5, 3],
        'steps': [4, 8, 16, 32, 64, 128],
        'min_sizes': [16, 32, 64, 128, 256, 512],
        'max_sizes': [32, 64, 128, 256, 512, 1024],
        'variance': [0.1, 0.2],
        'clip': True,
        'backbone': 'resnet50',
        'feature_pyramid_network': True,
        'feature_enhance_module': True,
        'progressive_anchor': True,
        'mbox': [1, 1, 1, 1, 1, 1],
        'aspect_ratios': [[1.5], [1.5], [1.5], [1.5], [1.5], [1.5]],
        'num_thresh': 500,
        'conf_thresh': 0.01,
        'nms_thresh': 0.45,
    }
    
    def __init__(
        self, 
        num_classes: int = 2,  # background + face
        cfg: dict = None,
        pretrained: bool = False
    ):
        super(DSFDDetector, self).__init__()
        
        self.cfg = cfg or self.default_cfg.copy()
        self.num_classes = num_classes
        
        # Build backbone
        self._build_backbone()
        
        # Build FPN
        self._build_fpn()
        
        # Build FEM
        self._build_fem()
        
        # Build detection heads
        self._build_heads()
        
        # Initialize detection module
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(
            num_classes,
            0,
            self.cfg.get('num_thresh', 500),
            self.cfg.get('conf_thresh', 0.01),
            self.cfg.get('nms_thresh', 0.45),
            self.cfg.get('variance', [0.1, 0.2])
        )
        
        # Initialize priors
        self.priors = None
        
    def _build_backbone(self):
        """Build ResNet backbone"""
        backbone = self.cfg.get('backbone', 'resnet50')
        
        if backbone in ['resnet50', 'resnet101', 'resnet152']:
            if backbone == 'resnet50':
                resnet = models.resnet50(pretrained=self.cfg.get('pretrained', False))
            elif backbone == 'resnet101':
                resnet = models.resnet101(pretrained=self.cfg.get('pretrained', False))
            else:
                resnet = models.resnet152(pretrained=self.cfg.get('pretrained', False))
                
            self.layer1 = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
            )
            self.layer2 = nn.Sequential(resnet.layer2)
            self.layer3 = nn.Sequential(resnet.layer3)
            self.layer4 = nn.Sequential(resnet.layer4)
            
            # Extra layers for detection
            self.layer5 = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            self.layer6 = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            
            self.output_channels = [256, 512, 1024, 2048, 512, 256]
            
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=self.cfg.get('pretrained', False))
            self.layer1 = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
            )
            self.layer2 = nn.Sequential(resnet.layer2)
            self.layer3 = nn.Sequential(resnet.layer3)
            self.layer4 = nn.Sequential(resnet.layer4)
            
            self.layer5 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.layer6 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            
            self.output_channels = [64, 128, 256, 512, 256, 128]
    
    def _build_fpn(self):
        """Build Feature Pyramid Network"""
        if not self.cfg.get('feature_pyramid_network', True):
            return
            
        fpn_in = self.output_channels
        
        self.latlayer3 = nn.Conv2d(fpn_in[3], fpn_in[2], kernel_size=1)
        self.latlayer2 = nn.Conv2d(fpn_in[2], fpn_in[1], kernel_size=1)
        self.latlayer1 = nn.Conv2d(fpn_in[1], fpn_in[0], kernel_size=1)
        
        self.smooth3 = nn.Conv2d(fpn_in[2], fpn_in[2], kernel_size=1)
        self.smooth2 = nn.Conv2d(fpn_in[1], fpn_in[1], kernel_size=1)
        self.smooth1 = nn.Conv2d(fpn_in[0], fpn_in[0], kernel_size=1)
    
    def _build_fem(self):
        """Build Feature Enhancement Module"""
        if not self.cfg.get('feature_enhance_module', True):
            return
            
        self.fem_modules = nn.ModuleList([
            FEM(ch) for ch in self.output_channels
        ])
    
    def _build_heads(self):
        """Build detection heads (multibox)"""
        mbox_cfg = self.cfg.get('mbox', [1, 1, 1, 1, 1, 1])
        use_pa = self.cfg.get('progressive_anchor', True)
        
        self.loc = nn.ModuleList()
        self.conf = nn.ModuleList()
        
        for k, ch in enumerate(self.output_channels):
            input_channels = 512 if self.cfg.get('feature_enhance_module', True) else ch
            
            if use_pa:
                # Progressive Anchor heads
                if k == 0:
                    loc_output = 4
                    conf_output = 2
                elif k == 1:
                    loc_output = 8
                    conf_output = 4
                else:
                    loc_output = 12
                    conf_output = 6
                    
                self.loc.append(DeepHeadModule(input_channels, mbox_cfg[k] * loc_output))
                self.conf.append(DeepHeadModule(input_channels, mbox_cfg[k] * conf_output))
            else:
                # Standard multibox
                self.loc.append(
                    nn.Conv2d(input_channels, mbox_cfg[k] * 4, kernel_size=3, padding=1)
                )
                self.conf.append(
                    nn.Conv2d(input_channels, mbox_cfg[k] * self.num_classes, kernel_size=3, padding=1)
                )
    
    def _upsample_add(self, x, y):
        """Upsample and add feature maps"""
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y
    
    def _upsample_product(self, x, y):
        """Upsample and multiply feature maps"""
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) * y
    
    def init_priors(self):
        """Initialize prior boxes"""
        priorbox = PriorBox(
            self.cfg,
            self.cfg.get('min_sizes', [16, 32, 64, 128, 256, 512]),
            self.cfg.get('max_sizes', [32, 64, 128, 256, 512, 1024])
        )
        with torch.no_grad():
            priors = priorbox.forward()
        return priors
        
    def forward(self, x):
        """Forward pass"""
        image_size = [x.shape[2], x.shape[3]]
        
        # Backbone forward
        conv3_3_x = self.layer1(x)
        conv4_3_x = self.layer2(conv3_3_x)
        conv5_3_x = self.layer3(conv4_3_x)
        fc7_x = self.layer4(conv5_3_x)
        conv6_2_x = self.layer5(fc7_x)
        conv7_2_x = self.layer6(conv6_2_x)
        
        sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]
        
        # Apply FPN
        if self.cfg.get('feature_pyramid_network', True):
            lfpn3 = self._upsample_product(self.latlayer3(fc7_x), self.smooth3(conv5_3_x))
            lfpn2 = self._upsample_product(self.latlayer2(lfpn3), self.smooth2(conv4_3_x))
            lfpn1 = self._upsample_product(self.latlayer1(lfpn2), self.smooth1(conv3_3_x))
            
            sources = [lfpn1, lfpn2, lfpn3, fc7_x, conv6_2_x, conv7_2_x]
        
        # Apply FEM
        if self.cfg.get('feature_enhance_module', True):
            for i in range(len(sources)):
                sources[i] = self.fem_modules[i](sources[i])
        
        # Detection heads
        loc = []
        conf = []
        featuremap_size = []
        
        for i, feat in enumerate(sources):
            featuremap_size.append([feat.shape[2], feat.shape[3]])
            loc.append(self.loc[i](feat).permute(0, 2, 3, 1).contiguous())
            conf.append(self.conf[i](feat).permute(0, 2, 3, 1).contiguous())
        
        # Concatenate predictions
        loc_preds = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf_preds = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        if self.training:
            # Training mode: return predictions and priors
            self.cfg['feature_maps'] = featuremap_size
            self.cfg['min_dim'] = image_size
            priors = self.init_priors()
            return loc_preds, conf_preds, priors
        else:
            # Test mode: return detection results
            self.cfg['feature_maps'] = featuremap_size
            self.cfg['min_dim'] = image_size
            priors = self.init_priors()
            
            if self.priors is None or self.priors.device != x.device:
                self.priors = priors.to(x.device)
            
            return self.detect(loc_preds, conf_preds, self.priors)
    
    def load_weights(self, weight_path):
        """Load pretrained weights"""
        print(f'Loading weights from {weight_path}...')
        state_dict = torch.load(weight_path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print('Finished loading weights!')


def build_dsfd_detector(pretrained=False, weight_path=None, backbone='resnet50'):
    """Build DSFD detector"""
    cfg = DSFDDetector.default_cfg.copy()
    cfg['backbone'] = backbone
    cfg['pretrained'] = pretrained
    
    model = DSFDDetector(num_classes=2, cfg=cfg)
    
    if weight_path:
        model.load_weights(weight_path)
    return model


# Aliases for package-level imports
FaceDetector = DSFDDetector
build_face_detector = build_dsfd_detector


# Quick test
if __name__ == '__main__':
    # Test model
    model = build_dsfd_detector(pretrained=False, backbone='resnet34')
    print(f"DSFD Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward
    x = torch.randn(1, 3, 640, 640)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"Detection output shape: {output.shape}")
