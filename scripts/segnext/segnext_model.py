"""
SegNeXt (Guo et al. 2022) in plain PyTorch: MSCAN encoder + LightHamHead decoder.

Ported from the MADOS repository (gkakogeorgiou/mados, MIT licence, commit
c20a7e9), marinext/mmseg/models/backbones/mscan.py and
decode_heads/ham_head.py, with the mmcv/mmseg helpers replaced by their plain
PyTorch equivalents (SyncBN -> BatchNorm2d as revert_sync_batchnorm does,
ConvModule -> conv + norm + ReLU, mmcv init functions -> torch.nn.init).
The layer code and the initialisation are unchanged.

Tiny configuration, as in marinext/configs/marinext.tiny.240x240.mados.py and
the MADOS paper ("we used the Tiny (SegNeXt-T) version"):
  embed_dims [32, 64, 160, 256], mlp_ratios [8, 8, 4, 4], depths [3, 3, 5, 2],
  drop_path 0.1, head channels 256, ham MD_R 16, GroupNorm(32), dropout 0.1.

first_stage=False: SegNeXt as published, the head uses stages 2-4 (output
                   H/8 x W/8), the MADOS paper's SegNeXt baseline.
first_stage=True : MariNeXt's modelling change, stages 1-4 (output H/4 x W/4).
The output is bilinearly upsampled to the input size, as in the MADOS train.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


# ------------------------------------------------------------- encoder ---

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.dwconv(self.fc1(x))))
        return self.drop(self.fc2(x))


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, 2, 1), nn.BatchNorm2d(out_channels // 2), nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, 2, 1), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        return x.flatten(2).transpose(1, 2), H, W


class AttentionModule(nn.Module):
    """Multi-Scale Convolutional Attention (MSCA)."""

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = (attn + self.conv0_2(self.conv0_1(attn)) + self.conv1_2(self.conv1_1(attn))
                + self.conv2_2(self.conv2_1(attn)))
        return self.conv3(attn) * u


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_2(self.spatial_gating_unit(self.activation(self.proj_1(x))))
        return x + shortcut


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1[:, None, None] * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2[:, None, None] * self.mlp(self.norm2(x)))
        return x.view(B, C, N).permute(0, 2, 1)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride, patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        return self.norm(x).flatten(2).transpose(1, 2), H, W


class MSCAN(nn.Module):
    def __init__(self, in_chans, embed_dims=(32, 64, 160, 256), mlp_ratios=(8, 8, 4, 4), drop_rate=0.,
                 drop_path_rate=0.1, depths=(3, 3, 5, 2)):
        super().__init__()
        self.num_stages = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            patch_embed = (StemConv(in_chans, embed_dims[0]) if i == 0 else
                           OverlapPatchEmbed(3, 2, embed_dims[i - 1], embed_dims[i]))
            block = nn.ModuleList([Block(embed_dims[i], mlp_ratios[i], drop_rate, dpr[cur + j])
                                   for j in range(depths[i])])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", nn.LayerNorm(embed_dims[i]))
        self.init_weights()

    def init_weights(self):
        """MSCAN.init_weights with init_cfg None (training from scratch)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, outs = x.shape[0], []
        for i in range(self.num_stages):
            x, H, W = getattr(self, f"patch_embed{i + 1}")(x)
            for blk in getattr(self, f"block{i + 1}"):
                x = blk(x, H, W)
            x = getattr(self, f"norm{i + 1}")(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


# -------------------------------------------------------------- decoder ---

class ConvModule(nn.Sequential):
    """mmcv ConvModule: conv (bias only without norm) + optional norm + optional ReLU,
    kaiming(fan_out, relu) conv init and constant(1, 0) norm init."""

    def __init__(self, cin, cout, norm=True, act=True):
        layers = [nn.Conv2d(cin, cout, 1, bias=not norm)]
        if norm:
            layers.append(nn.GroupNorm(32, cout))
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)
        nn.init.kaiming_normal_(self[0].weight, mode="fan_out", nonlinearity="relu")
        if self[0].bias is not None:
            nn.init.zeros_(self[0].bias)


class NMF2D(nn.Module):
    """Non-negative matrix factorisation (Hamburger 'ham'), spatial, S=1, inv_t=1, rand_init."""

    def __init__(self, R=16, train_steps=6, eval_steps=7):
        super().__init__()
        self.R, self.train_steps, self.eval_steps = R, train_steps, eval_steps

    @staticmethod
    def _step(x, bases, coef):
        coef = coef * torch.bmm(x.transpose(1, 2), bases) / (coef.bmm(bases.transpose(1, 2).bmm(bases)) + 1e-6)
        bases = bases * torch.bmm(x, coef) / (bases.bmm(coef.transpose(1, 2).bmm(coef)) + 1e-6)
        return bases, coef

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        bases = F.normalize(torch.rand((B, C, self.R), device=x.device), dim=1)
        coef = F.softmax(torch.bmm(x.transpose(1, 2), bases), dim=-1)
        for _ in range(self.train_steps if self.training else self.eval_steps):
            bases, coef = self._step(x, bases, coef)
        coef = coef * torch.bmm(x.transpose(1, 2), bases) / (coef.bmm(bases.transpose(1, 2).bmm(bases)) + 1e-6)
        return torch.bmm(bases, coef.transpose(1, 2)).view(B, C, H, W)


class Hamburger(nn.Module):
    def __init__(self, ch, R):
        super().__init__()
        self.ham_in = ConvModule(ch, ch, norm=False, act=False)
        self.ham = NMF2D(R)
        self.ham_out = ConvModule(ch, ch, norm=True, act=False)

    def forward(self, x):
        enjoy = self.ham_out(self.ham(F.relu(self.ham_in(x), inplace=True)))
        return F.relu(x + enjoy, inplace=True)


class LightHamHead(nn.Module):
    def __init__(self, in_channels, num_classes, channels=256, ham_channels=256, R=16, dropout=0.1):
        super().__init__()
        self.squeeze = ConvModule(sum(in_channels), ham_channels)
        self.hamburger = Hamburger(ham_channels, R)
        self.align = ConvModule(ham_channels, channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
        nn.init.normal_(self.conv_seg.weight, 0, 0.01)
        nn.init.zeros_(self.conv_seg.bias)

    def forward(self, inputs):
        size = inputs[0].shape[2:]
        x = torch.cat([F.interpolate(t, size=size, mode="bilinear", align_corners=False) for t in inputs], 1)
        x = self.align(self.hamburger(self.squeeze(x)))
        return self.conv_seg(self.dropout(x))


class SegNeXt(nn.Module):
    def __init__(self, in_chans, num_classes, first_stage=False):
        super().__init__()
        self.backbone = MSCAN(in_chans)
        self.in_index = [0, 1, 2, 3] if first_stage else [1, 2, 3]
        dims = [32, 64, 160, 256]
        self.decode_head = LightHamHead([dims[i] for i in self.in_index], num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        out = self.decode_head([feats[i] for i in self.in_index])
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
