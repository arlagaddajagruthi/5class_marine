import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix

# Ensure workspace root is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.datasets.mados import MADOSDataset

# =====================================================================
# Build a MADOS-specific MariNeXt with in_chans=8
# =====================================================================
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
    def forward(self, x):
        return self.dwconv(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.dwconv(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2), nn.GELU(),
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
        attn = attn + self.conv0_2(self.conv0_1(attn)) + self.conv1_2(self.conv1_1(attn)) + self.conv2_2(self.conv2_1(attn))
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
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        lsi = 1e-2
        self.layer_scale_1 = nn.Parameter(lsi * torch.ones((dim, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(lsi * torch.ones((dim, 1, 1)), requires_grad=True)
    def forward(self, x):
        x = x + self.layer_scale_1 * self.attn(self.norm1(x))
        x = x + self.layer_scale_2 * self.mlp(self.norm2(x))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)
    def forward(self, x):
        return self.norm(self.proj(x))

class MSCAN(nn.Module):
    def __init__(self, in_chans=8, embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 5, 2]):
        super().__init__()
        self.patch_embed1 = StemConv(in_chans, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])
        self.block1 = nn.ModuleList([Block(embed_dims[0], mlp_ratios[0]) for _ in range(depths[0])])
        self.norm1 = nn.BatchNorm2d(embed_dims[0])
        self.block2 = nn.ModuleList([Block(embed_dims[1], mlp_ratios[1]) for _ in range(depths[1])])
        self.norm2 = nn.BatchNorm2d(embed_dims[1])
        self.block3 = nn.ModuleList([Block(embed_dims[2], mlp_ratios[2]) for _ in range(depths[2])])
        self.norm3 = nn.BatchNorm2d(embed_dims[2])
        self.block4 = nn.ModuleList([Block(embed_dims[3], mlp_ratios[3]) for _ in range(depths[3])])
        self.norm4 = nn.BatchNorm2d(embed_dims[3])
    def forward(self, x):
        outs = []
        x = self.patch_embed1(x)
        for blk in self.block1: x = blk(x)
        outs.append(self.norm1(x))
        x = self.patch_embed2(x)
        for blk in self.block2: x = blk(x)
        outs.append(self.norm2(x))
        x = self.patch_embed3(x)
        for blk in self.block3: x = blk(x)
        outs.append(self.norm3(x))
        x = self.patch_embed4(x)
        for blk in self.block4: x = blk(x)
        outs.append(self.norm4(x))
        return outs

class NMF2D(nn.Module):
    def __init__(self, ham_channels=256, R=16, train_steps=6, eval_steps=7):
        super().__init__()
        self.R = R; self.train_steps = train_steps; self.eval_steps = eval_steps
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)
        bases = F.normalize(torch.rand((B, C, self.R), device=x.device), dim=1)
        coef = F.softmax(torch.bmm(x_flat.transpose(1, 2), bases), dim=-1)
        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            num = torch.bmm(x_flat.transpose(1, 2), bases)
            den = coef.bmm(bases.transpose(1, 2).bmm(bases))
            coef = coef * num / (den + 1e-6)
            num = torch.bmm(x_flat, coef)
            den = bases.bmm(coef.transpose(1, 2).bmm(coef))
            bases = bases * num / (den + 1e-6)
        return torch.bmm(bases, coef.transpose(1, 2)).view(B, C, H, W)

class Hamburger(nn.Module):
    def __init__(self, ham_channels=256):
        super().__init__()
        self.ham_in = nn.Conv2d(ham_channels, ham_channels, 1)
        self.ham = NMF2D(ham_channels)
        self.ham_out = nn.Conv2d(ham_channels, ham_channels, 1)
    def forward(self, x):
        shortcut = x
        x = F.relu(self.ham_in(x), inplace=True)
        x = self.ham(x)
        return F.relu(self.ham_out(x) + shortcut, inplace=True)

class LightHamHead(nn.Module):
    def __init__(self, in_channels=[32, 64, 160, 256], channels=256, num_classes=5):
        super().__init__()
        self.squeeze = nn.ModuleList([nn.Conv2d(c, channels, 1) for c in in_channels])
        self.hamburger = Hamburger(ham_channels=channels)
        self.align = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(32, channels), nn.ReLU(inplace=True))
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)
    def forward(self, inputs):
        h, w = inputs[0].shape[2:]
        outs = [F.interpolate(self.squeeze[i](f), size=(h, w), mode='bilinear', align_corners=False) for i, f in enumerate(inputs)]
        x = sum(outs)
        return self.cls_seg(self.align(self.hamburger(x)))

class MariNeXtMADOS(nn.Module):
    """MariNeXt model configured specifically for MADOS (8 input channels)"""
    def __init__(self, in_chans=8, num_classes=5):
        super().__init__()
        self.backbone = MSCAN(in_chans=in_chans)
        self.decode_head = LightHamHead(in_channels=[32, 64, 160, 256], channels=256, num_classes=num_classes)
    def forward(self, x):
        logits = self.decode_head(self.backbone(x))
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits

MADOS_MAP = {
    0: 0, 1: 2, 2: 3, 3: 3, 4: 4, 5: 4, 6: 4, 7: 1, 8: 1, 9: 4, 10: 1, 11: 1, 12: 1, 13: 4, 14: 4, 15: 4
}

def remap_target(target_tensor, mapping_dict):
    mapped = target_tensor.clone()
    max_k = max(mapping_dict.keys())
    lut = torch.full((max_k + 256,), -1, dtype=torch.long)
    for k, v in mapping_dict.items():
        lut[k] = v
    valid_mask = (target_tensor >= 0) & (target_tensor < len(lut))
    mapped[valid_mask] = lut[target_tensor[valid_mask]].to(target_tensor.device)
    mapped[~valid_mask] = -1
    return mapped

def run_mados_marinext(dataset_path, batch_size=2, epochs=1):
    print(f"\n{'='*60}")
    print(f"Starting MariNeXt 5-Class Pipeline for MADOS_5Class (8ch)")
    print(f"Dataset Path: {dataset_path}")
    print(f"{'='*60}")

    train_ds = MADOSDataset(dataset_path, split="train")
    val_ds = MADOSDataset(dataset_path, split="val")
    print(f"[INFO] Dataset loaded -> Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    # Detect actual channel count from first sample
    sample_img, _ = train_ds[0]
    actual_channels = sample_img.shape[0]
    print(f"[INFO] Detected input channels: {actual_channels}")

    model = MariNeXtMADOS(in_chans=actual_channels, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    model.train()
    for epoch in range(epochs):
        print(f"[TRAIN] Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        for images, targets in tqdm(train_loader, desc="Training MADOS_5Class"):
            images = images.to(device)
            targets = remap_target(targets.long(), MADOS_MAP).to(device)
            optimizer.zero_grad()
            logits = model(images)
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[TRAIN] Avg Loss: {total_loss / max(1, len(train_loader)):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating MADOS_5Class"):
            images = images.to(device)
            targets_mapped = remap_target(targets.long(), MADOS_MAP)
            logits = model(images)
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            t_np = targets_mapped.numpy()
            mask = (t_np >= 0) & (t_np < 5)
            all_preds.extend(preds[mask])
            all_targets.extend(t_np[mask])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec  = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1   = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    iou  = jaccard_score(all_targets, all_preds, average='macro', zero_division=0)
    cm   = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4])

    print(f"\n{'='*40}")
    print(f"MariNeXt Results for MADOS_5Class ({actual_channels}-channel input):")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"IoU (Jaccard): {iou:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"{'='*40}\n")

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Save result txt (new file)
    txt_path = os.path.join(workspace_root, "MADOS_5Class_marinext_results_new.txt")
    with open(txt_path, "w") as f:
        f.write(f"MariNeXt 5-Class Results for MADOS_5Class (in_chans={actual_channels})\n")
        f.write(f"Precision: {prec:.6f}\nRecall: {rec:.6f}\nF1 Score: {f1:.6f}\nIoU (Jaccard): {iou:.6f}\n")
        f.write(f"\nDetailed Confusion Matrix:\n{cm}\n")
    print(f"[SAVED] {txt_path}")

    # Save per-class markdown (new file)
    log_dir = os.path.join(workspace_root, "outputs", "logs", "MADOS")
    os.makedirs(log_dir, exist_ok=True)
    md_path = os.path.join(log_dir, "per_class_metrics_marinext_new.md")
    class_names = ["Background/Water", "Debris", "Algae", "Other", "Void"]
    pc_iou  = jaccard_score(all_targets, all_preds, average=None, zero_division=0, labels=[0,1,2,3,4])
    pc_f1   = f1_score(all_targets, all_preds, average=None, zero_division=0, labels=[0,1,2,3,4])
    pc_prec = precision_score(all_targets, all_preds, average=None, zero_division=0, labels=[0,1,2,3,4])
    pc_rec  = recall_score(all_targets, all_preds, average=None, zero_division=0, labels=[0,1,2,3,4])
    with open(md_path, "w") as f:
        f.write(f"# Per-Class Metrics for MariNeXt on MADOS_5Class\n\n")
        f.write("| Class | IoU | F1 | Precision | Recall |\n| :--- | ---: | ---: | ---: | ---: |\n")
        for i in range(5):
            f.write(f"| {class_names[i]} | {pc_iou[i]:.4f} | {pc_f1[i]:.4f} | {pc_prec[i]:.4f} | {pc_rec[i]:.4f} |\n")
        f.write(f"| **Average** | **{iou:.4f}** | **{f1:.4f}** | **{prec:.4f}** | **{rec:.4f}** |\n")
    print(f"[SAVED] {md_path}")

    # Save checkpoint (new file)
    ckpt_dir = os.path.join(workspace_root, "outputs", "checkpoints", "marinext_new")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "MADOS_5Class_marinext_weights.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[SAVED] {ckpt_path}\n")

if __name__ == "__main__":
    run_mados_marinext(r"C:\Users\Jagruthi\Downloads\MADOS\MADOS", batch_size=2, epochs=1)
