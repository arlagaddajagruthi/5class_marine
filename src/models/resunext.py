"""
ResUNeXt: ResNeXt-style grouped-conv Encoder + U-Net Decoder with Skip Connections
for Multi-Spectral Marine 5-Class Pixel Segmentation.

Channel layout (base_ch=32):
  stem  : in_ch  -> 32
  enc1  : 32  -> 64   skip=64
  enc2  : 64  -> 128  skip=128
  enc3  : 128 -> 256  skip=256
  enc4  : 256 -> 512  skip=512
  bridge: 512 -> 512
  dec4  : 512 (up) + 256 skip -> 256
  dec3  : 256 (up) + 128 skip -> 128
  dec2  : 128 (up) + 64  skip -> 64
  dec1  : 64  (up) + 32  skip -> 32
  head  : 32 -> num_classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────

def conv_bn_relu(in_ch, out_ch, k=3, stride=1, pad=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResNeXtBlock(nn.Module):
    """
    ResNeXt bottleneck: 1x1 -> 3x3 grouped -> 1x1  +  identity shortcut.
    Uses half-width intermediate channels for efficiency.
    """
    def __init__(self, ch, cardinality=16):
        super().__init__()
        mid = max(ch // 2, cardinality)                      # bottleneck width
        grp = min(cardinality, mid)                          # safe group count
        self.body = nn.Sequential(
            conv_bn_relu(ch,  mid, k=1, pad=0),              # squeeze
            conv_bn_relu(mid, mid, k=3, pad=1, groups=grp),  # group conv
            nn.Conv2d(mid, ch, 1, bias=False),               # expand
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + x)


# ─────────────────────────────────────────────────────────────────
# Encoder / Decoder stages
# ─────────────────────────────────────────────────────────────────

class EncStage(nn.Module):
    """Downsample in_ch->out_ch via stride-2 conv, then 2 ResNeXt blocks."""
    def __init__(self, in_ch, out_ch, cardinality=16):
        super().__init__()
        self.down  = conv_bn_relu(in_ch, out_ch, k=3, stride=2, pad=1)
        self.block = nn.Sequential(
            ResNeXtBlock(out_ch, cardinality),
            ResNeXtBlock(out_ch, cardinality),
        )

    def forward(self, x):
        x = self.down(x)
        return self.block(x)          # return feature map (used as skip + input)


class DecStage(nn.Module):
    """2x bilinear upsample, concat skip, then 2 ConvBnRelu."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            conv_bn_relu(in_ch + skip_ch, out_ch),
            conv_bn_relu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd-size spatial dims
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ─────────────────────────────────────────────────────────────────
# ResUNeXt
# ─────────────────────────────────────────────────────────────────

class ResUNeXt(nn.Module):
    """
    ResUNeXt Segmentation Network.

    Parameters
    ----------
    in_channels : int   spectral bands (8 MADOS / 11 MARIDA)
    num_classes : int   5 marine classes
    base_ch     : int   feature width at first encoder stage (default 32)
    cardinality : int   grouped-conv groups (default 16)
    """
    def __init__(self, in_channels=8, num_classes=5,
                 base_ch=32, cardinality=16):
        super().__init__()
        c = base_ch   # shorthand

        # ── Stem: project spectral bands to base_ch ────────────────
        self.stem = nn.Sequential(
            conv_bn_relu(in_channels, c),
            conv_bn_relu(c, c),
        )                                     # (B, c, H, W)

        # ── Encoder ───────────────────────────────────────────────
        #  Each stage halves spatial res and doubles channels
        self.enc1 = EncStage(c,      c * 2,  cardinality)  # -> (B, 64,  H/2, W/2)
        self.enc2 = EncStage(c * 2,  c * 4,  cardinality)  # -> (B, 128, H/4, W/4)
        self.enc3 = EncStage(c * 4,  c * 8,  cardinality)  # -> (B, 256, H/8, W/8)
        self.enc4 = EncStage(c * 8,  c * 16, cardinality)  # -> (B, 512, H/16,W/16)

        # ── Bottleneck bridge ──────────────────────────────────────
        self.bridge = nn.Sequential(
            ResNeXtBlock(c * 16, cardinality),
            ResNeXtBlock(c * 16, cardinality),
        )                                     # (B, 512, H/16, W/16)

        # ── Decoder ───────────────────────────────────────────────
        #  DecStage(upstream_ch, skip_ch, out_ch)
        self.dec4 = DecStage(c * 16, c * 8,  c * 8)   # 512+256 -> 256
        self.dec3 = DecStage(c * 8,  c * 4,  c * 4)   # 256+128 -> 128
        self.dec2 = DecStage(c * 4,  c * 2,  c * 2)   # 128+64  -> 64
        self.dec1 = DecStage(c * 2,  c,      c)        # 64 +32  -> 32

        # ── Segmentation head ──────────────────────────────────────
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # Stem
        s = self.stem(x)          # (B, 32, H, W)

        # Encode — each stage produces ONE feature map at its output res
        e1 = self.enc1(s)         # (B, 64,  H/2,  W/2)
        e2 = self.enc2(e1)        # (B, 128, H/4,  W/4)
        e3 = self.enc3(e2)        # (B, 256, H/8,  W/8)
        e4 = self.enc4(e3)        # (B, 512, H/16, W/16)

        # Bridge
        b  = self.bridge(e4)      # (B, 512, H/16, W/16)

        # Decode (bridge output fed up, skip from encoder)
        d4 = self.dec4(b,  e3)   # skip=e3 (256 ch at H/8)
        d3 = self.dec3(d4, e2)   # skip=e2 (128 ch at H/4)
        d2 = self.dec2(d3, e1)   # skip=e1 (64  ch at H/2)
        d1 = self.dec1(d2, s)    # skip=s  (32  ch at H)

        # Head + restore original resolution
        out = self.head(d1)
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W),
                                mode='bilinear', align_corners=False)
        return out
