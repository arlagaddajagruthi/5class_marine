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
from src.datasets.marida import MARIDADataset
from src.datasets.mados import MADOSDataset
from src.models.marinext_model import MariNeXt

MARIDA_MAP = {
    0: 0, 1: 2, 2: 3, 3: 3, 4: 4, 5: 4, 6: 0, 7: 1, 8: 1, 9: 4, 10: 1, 11: 1, 12: 1, 13: 0, 14: 1, 15: 1
}

MADOS_MAP = {
    0: 0, 1: 2, 2: 3, 3: 3, 4: 4, 5: 4, 6: 4, 7: 1, 8: 1, 9: 4, 10: 1, 11: 1, 12: 1, 13: 4, 14: 4, 15: 4
}

def remap_target(target_tensor, mapping_dict):
    mapped = target_tensor.clone()
    # Create lookup table for fast tensor mapping
    max_k = max(mapping_dict.keys())
    lut = torch.full((max_k + 256,), -1, dtype=torch.long)
    for k, v in mapping_dict.items():
        lut[k] = v
    
    # Map valid non-negative indices
    valid_mask = (target_tensor >= 0) & (target_tensor < len(lut))
    mapped[valid_mask] = lut[target_tensor[valid_mask]].to(target_tensor.device)
    mapped[~valid_mask] = -1
    return mapped

def evaluate_and_train_marinext(dataset_name, dataset_path, dataset_class, mapping_dict, in_channels=11, batch_size=2, epochs=1):
    print(f"\n{'='*60}")
    print(f"Starting MariNeXt 5-Class Pipeline for {dataset_name}")
    print(f"Dataset Path: {dataset_path}")
    print(f"{'='*60}")

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        return

    splits_path = os.path.join(dataset_path, "splits")
    if not os.path.exists(splits_path):
        print(f"[ERROR] Splits folder not found at: {splits_path}")
        return

    train_ds = dataset_class(dataset_path, split="train")
    val_ds = dataset_class(dataset_path, split="val")

    print(f"[INFO] Dataset loaded -> Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using Computation Device: {device}")

    model = MariNeXt(in_chans=in_channels, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 1. Training Loop
    model.train()
    for epoch in range(epochs):
        print(f"[TRAIN] Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Training {dataset_name}"):
            images, targets = images.to(device), targets.long().to(device)
            targets = remap_target(targets, mapping_dict)

            optimizer.zero_grad()

            logits = model(images)
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[TRAIN] Epoch {epoch+1} Average Loss: {total_loss / max(1, len(train_loader)):.4f}")

    # 2. Evaluation Loop
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Evaluating {dataset_name}"):
            images, targets = images.to(device), targets.long().to(device)
            targets = remap_target(targets, mapping_dict)

            logits = model(images)
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()

            mask = (targets_np >= 0) & (targets_np < 5)
            all_preds.extend(preds[mask])
            all_targets.extend(targets_np[mask])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    iou = jaccard_score(all_targets, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3, 4])

    print("\n" + "="*40)
    print(f"MariNeXt Evaluation Results for {dataset_name}:")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1 Score:      {f1:.4f}")
    print(f"IoU (Jaccard): {iou:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("="*40 + "\n")

    # 3. Save brand NEW output files
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Save main result text file
    txt_filename = os.path.join(workspace_root, f"{dataset_name}_marinext_results_new.txt")
    with open(txt_filename, "w") as f:
        f.write(f"MariNeXt 5-Class Results for {dataset_name}\n")
        f.write(f"Precision: {prec:.6f}\n")
        f.write(f"Recall: {rec:.6f}\n")
        f.write(f"F1 Score: {f1:.6f}\n")
        f.write(f"IoU (Jaccard): {iou:.6f}\n")
        f.write("\nDetailed Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    print(f"[SAVED] Results file: {txt_filename}")

    # Save per-class markdown log file
    log_dir = os.path.join(workspace_root, "outputs", "logs", dataset_name.split("_")[0])
    os.makedirs(log_dir, exist_ok=True)
    md_filename = os.path.join(log_dir, "per_class_metrics_marinext_new.md")

    class_names = ["Background/Water", "Debris", "Algae", "Other", "Class 4"]
    per_class_iou = jaccard_score(all_targets, all_preds, average=None, zero_division=0, labels=[0,1,2,3,4])
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0, labels=[0,1,2,3,4])
    per_class_prec = precision_score(all_targets, all_preds, average=None, zero_division=0, labels=[0,1,2,3,4])
    per_class_rec = recall_score(all_targets, all_preds, average=None, zero_division=0, labels=[0,1,2,3,4])

    with open(md_filename, "w") as f:
        f.write(f"# Per-Class Metrics for MariNeXt on {dataset_name}\n\n")
        f.write("| Class | IoU | F1 | Precision | Recall |\n")
        f.write("| :--- | ---: | ---: | ---: | ---: |\n")
        for i in range(len(class_names)):
            f.write(f"| {class_names[i]} | {per_class_iou[i]:.4f} | {per_class_f1[i]:.4f} | {per_class_prec[i]:.4f} | {per_class_rec[i]:.4f} |\n")
        f.write(f"| **Average** | **{iou:.4f}** | **{f1:.4f}** | **{prec:.4f}** | **{rec:.4f}** |\n")
    print(f"[SAVED] Markdown Log: {md_filename}")

    # Save Checkpoint
    ckpt_dir = os.path.join(workspace_root, "outputs", "checkpoints", "marinext_new")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_filename = os.path.join(ckpt_dir, f"{dataset_name}_marinext_weights.pth")
    torch.save(model.state_dict(), ckpt_filename)
    print(f"[SAVED] Model Checkpoint: {ckpt_filename}\n")

if __name__ == "__main__":
    datasets = [
        ("MARIDA_5Class", r"C:\Users\Jagruthi\Downloads\MARIDA", MARIDADataset, MARIDA_MAP, 11),
        ("MADOS_5Class", r"C:\Users\Jagruthi\Downloads\MADOS\MADOS", MADOSDataset, MADOS_MAP, 11)
    ]

    for name, path, ds_cls, mapping, chans in datasets:
        evaluate_and_train_marinext(name, path, ds_cls, mapping, in_channels=chans, batch_size=2, epochs=1)
