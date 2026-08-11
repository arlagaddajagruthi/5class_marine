import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetSegmentation(nn.Module):
    """
    ResNet-18 Multi-Spectral Semantic Segmentation Model.
    Supports variable input channels (C=8 for MADOS, C=11 for MARIDA)
    and produces 5-class pixel-level segmentation maps.
    """
    def __init__(self, in_channels=8, num_classes=5):
        super(ResNetSegmentation, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Load standard ResNet-18 backbone
        backbone = models.resnet18(weights=None)
        
        # Modify initial conv1 to accept in_channels instead of 3 RGB channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1 # 64 ch, stride 1 (H/4, W/4)
        self.layer2 = backbone.layer2 # 128 ch, stride 2 (H/8, W/8)
        self.layer3 = backbone.layer3 # 256 ch, stride 2 (H/16, W/16)
        self.layer4 = backbone.layer4 # 512 ch, stride 2 (H/32, W/32)
        
        # Lateral 1x1 convs for Decoder / FPN fusion
        dec_ch = 128
        self.lat4 = nn.Conv2d(512, dec_ch, kernel_size=1)
        self.lat3 = nn.Conv2d(256, dec_ch, kernel_size=1)
        self.lat2 = nn.Conv2d(128, dec_ch, kernel_size=1)
        self.lat1 = nn.Conv2d(64, dec_ch, kernel_size=1)
        
        # Decoder conv blocks
        self.dec_conv = nn.Sequential(
            nn.Conv2d(dec_ch, dec_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch, dec_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch),
            nn.ReLU(inplace=True)
        )
        
        # Final segmentation head
        self.classifier = nn.Conv2d(dec_ch, num_classes, kernel_size=1)
        
    def forward(self, x):
        input_size = x.shape[2:] # (H, W)
        
        # Encoder Stage
        x0 = self.relu(self.bn1(self.conv1(x)))
        x_mp = self.maxpool(x0)
        
        c1 = self.layer1(x_mp) # (B, 64, H/4, W/4)
        c2 = self.layer2(c1)   # (B, 128, H/8, W/8)
        c3 = self.layer3(c2)   # (B, 256, H/16, W/16)
        c4 = self.layer4(c3)   # (B, 512, H/32, W/32)
        
        # Decoder / FPN Fusion Stage
        p4 = self.lat4(c4)
        
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        
        feat = self.dec_conv(p1)
        logits = self.classifier(feat)
        
        # Upsample back to original resolution (H, W)
        out = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return out
