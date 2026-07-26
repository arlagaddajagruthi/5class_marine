import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. MSCAN Backbone Components
# =====================================================================

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        return self.dwconv(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.proj(x)

class AttentionModule(nn.Module):
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

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)

        return attn * u

class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        x = x + self.layer_scale_1 * self.attn(self.norm1(x))
        x = x + self.layer_scale_2 * self.mlp(self.norm2(x))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class MSCAN(nn.Module):
    def __init__(self, in_chans=11, embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 5, 2]):
        super().__init__()
        self.depths = depths
        self.num_stages = len(depths)

        # Stage 1 Patch Embed
        self.patch_embed1 = StemConv(in_chans, embed_dims[0])

        # Stage 2 Patch Embed
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        # Stage 3 Patch Embed
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        # Stage 4 Patch Embed
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Stages
        self.block1 = nn.ModuleList([Block(dim=embed_dims[0], mlp_ratio=mlp_ratios[0]) for _ in range(depths[0])])
        self.norm1 = nn.BatchNorm2d(embed_dims[0])

        self.block2 = nn.ModuleList([Block(dim=embed_dims[1], mlp_ratio=mlp_ratios[1]) for _ in range(depths[1])])
        self.norm2 = nn.BatchNorm2d(embed_dims[1])

        self.block3 = nn.ModuleList([Block(dim=embed_dims[2], mlp_ratio=mlp_ratios[2]) for _ in range(depths[2])])
        self.norm3 = nn.BatchNorm2d(embed_dims[2])

        self.block4 = nn.ModuleList([Block(dim=embed_dims[3], mlp_ratio=mlp_ratios[3]) for _ in range(depths[3])])
        self.norm4 = nn.BatchNorm2d(embed_dims[3])

    def forward(self, x):
        outs = []

        # Stage 1
        x = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x)
        x = self.norm1(x)
        outs.append(x)

        # Stage 2
        x = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x)
        x = self.norm2(x)
        outs.append(x)

        # Stage 3
        x = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x)
        x = self.norm3(x)
        outs.append(x)

        # Stage 4
        x = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x)
        x = self.norm4(x)
        outs.append(x)

        return outs

# =====================================================================
# 2. LightHamHead Decoder Components
# =====================================================================

class NMF2D(nn.Module):
    def __init__(self, S=1, D=512, R=16, train_steps=6, eval_steps=7):
        super().__init__()
        self.S = S
        self.D = D
        self.R = R
        self.train_steps = train_steps
        self.eval_steps = eval_steps

    def _build_bases(self, B, S, D, R, device):
        bases = torch.rand((B * S, D, R), device=device)
        return F.normalize(bases, dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        D = C // self.S
        N = H * W
        x_flat = x.view(B * self.S, D, N)

        bases = self._build_bases(B, self.S, D, self.R, device=x.device)
        coef = torch.bmm(x_flat.transpose(1, 2), bases)
        coef = F.softmax(coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            numerator = torch.bmm(x_flat.transpose(1, 2), bases)
            denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
            coef = coef * numerator / (denominator + 1e-6)

            numerator = torch.bmm(x_flat, coef)
            denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
            bases = bases * numerator / (denominator + 1e-6)

        x_out = torch.bmm(bases, coef.transpose(1, 2))
        return x_out.view(B, C, H, W)

class Hamburger(nn.Module):
    def __init__(self, ham_channels=256, ham_kwargs=dict(MD_R=16)):
        super().__init__()
        self.ham_in = nn.Conv2d(ham_channels, ham_channels, 1)
        self.ham = NMF2D(S=1, D=ham_channels, R=ham_kwargs.get('MD_R', 16))
        self.ham_out = nn.Conv2d(ham_channels, ham_channels, 1)

    def forward(self, x):
        shortcut = x
        x = self.ham_in(x)
        x = F.relu(x, inplace=True)
        x = self.ham(x)
        x = self.ham_out(x)
        return F.relu(x + shortcut, inplace=True)

class LightHamHead(nn.Module):
    def __init__(self, in_channels=[32, 64, 160, 256], channels=256, num_classes=5):
        super().__init__()
        self.squeeze = nn.ModuleList([
            nn.Conv2d(c, channels, 1) for c in in_channels
        ])
        self.hamburger = Hamburger(ham_channels=channels)
        self.align = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True)
        )
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(self, inputs):
        # Resize multi-stage backbone feature maps to stage 1 size
        h, w = inputs[0].shape[2:]
        outs = []
        for i, feat in enumerate(inputs):
            feat = self.squeeze[i](feat)
            if feat.shape[2:] != (h, w):
                feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            outs.append(feat)

        x = sum(outs) # Feature fusion sum
        x = self.hamburger(x)
        x = self.align(x)
        logits = self.cls_seg(x)
        return logits

# =====================================================================
# 3. Unified MariNeXt Model
# =====================================================================

class MariNeXt(nn.Module):
    def __init__(self, in_chans=11, num_classes=5):
        super().__init__()
        self.backbone = MSCAN(in_chans=in_chans, embed_dims=[32, 64, 160, 256], depths=[3, 3, 5, 2])
        self.decode_head = LightHamHead(in_channels=[32, 64, 160, 256], channels=256, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        # Upsample logits back to input spatial size
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits

if __name__ == "__main__":
    model = MariNeXt(in_chans=11, num_classes=5)
    dummy_input = torch.randn(2, 11, 240, 240)
    out = model(dummy_input)
    print("MariNeXt successfully tested! Output shape:", out.shape)
