import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.datasets.mados import MADOSDataset
from src.datasets.marida import MARIDADataset
from src.models.resunext import ResUNeXt
from src.metrics.metrics import calculate_metrics, save_metrics_table

CLASS_NAMES = ["Marine Debris", "Sargassum/Veg", "Natural Phenom/Foam",
               "Ship/Infrastructure", "Water/Other"]
CM_FOLDER   = os.path.join(repo_root, "outputs", "confusion_matrices")


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


def save_dual_heatmap(cm, title, save_path):
    """Save side-by-side raw-count + recall-normalised heatmap."""
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums * 100

    short_names = ["Mar.Debris", "Sarg/Veg", "Nat/Foam", "Ships", "Water"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    sns.heatmap(cm, ax=axes[0], annot=True, fmt='d', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names,
                linewidths=0.5, cbar=True)
    axes[0].set_title('Raw Pixel Counts', fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=10)
    axes[0].set_ylabel('Actual', fontsize=10)
    axes[0].tick_params(axis='x', rotation=30)

    sns.heatmap(cm_norm, ax=axes[1], annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=short_names, yticklabels=short_names,
                linewidths=0.5, cbar=True, vmin=0, vmax=100)
    axes[1].set_title('Recall-Normalised (%)', fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=10)
    axes[1].set_ylabel('Actual', fontsize=10)
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Heatmap saved  → {save_path}")


def save_accuracy_txt(metrics, dataset_name, model_name,
                      num_train, num_test, in_ch,
                      total_params, train_time, inf_time,
                      lat_patch_ms, throughput, model_size_mb,
                      txt_path):
    """Write a human-readable accuracy & metrics text file."""
    ov = metrics['overall']
    pc = metrics['per_class']
    lines = []
    lines.append("=" * 60)
    lines.append(f"  {model_name}  |  {dataset_name}")
    lines.append("=" * 60)
    lines.append(f"\nModel Architecture : {model_name}")
    lines.append(f"Input Channels     : {in_ch}")
    lines.append(f"Total Parameters   : {total_params:,}  ({total_params * 4 / 1e6:.2f} MB weights)")
    lines.append(f"Saved Model Size   : {model_size_mb:.2f} MB")
    lines.append(f"Train Pixels       : {num_train:,}")
    lines.append(f"Test  Pixels       : {num_test:,}")
    lines.append(f"Training Time      : {train_time:.2f} seconds")
    lines.append(f"Inference Time     : {inf_time:.4f} seconds (full test set)")
    lines.append(f"Latency / patch    : {lat_patch_ms:.3f} ms")
    lines.append(f"Throughput         : {throughput:,.0f} pixels / second")
    lines.append("")
    lines.append("─" * 60)
    lines.append("  OVERALL METRICS (Macro-Averaged)")
    lines.append("─" * 60)
    lines.append(f"  Accuracy  : {ov['accuracy']:.6f}  ({ov['accuracy']*100:.2f} %)")
    lines.append(f"  Precision : {ov['precision']:.6f}")
    lines.append(f"  Recall    : {ov['recall']:.6f}")
    lines.append(f"  F1 Score  : {ov['f1']:.6f}")
    lines.append(f"  IoU       : {ov['iou']:.6f}")
    lines.append("")
    lines.append("─" * 60)
    lines.append("  PER-CLASS METRICS")
    lines.append("─" * 60)
    header = f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}"
    lines.append(header)
    lines.append("  " + "-" * 65)
    for i, cn in enumerate(CLASS_NAMES):
        lines.append(
            f"  {cn:<25} {pc['precision'][i]:>10.4f} "
            f"{pc['recall'][i]:>10.4f} {pc['f1'][i]:>10.4f} {pc['iou'][i]:>10.4f}"
        )
    lines.append("  " + "-" * 65)
    lines.append(
        f"  {'Macro Average':<25} {ov['precision']:>10.4f} "
        f"{ov['recall']:>10.4f} {ov['f1']:>10.4f} {ov['iou']:>10.4f}"
    )
    lines.append("")
    lines.append("─" * 60)
    lines.append("  CONFUSION MATRIX (raw counts)")
    lines.append("─" * 60)
    lines.append(str(metrics['confusion_matrix']))
    lines.append("")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Accuracy file saved → {txt_path}")


# ──────────────────────────────────────────────────────────
# Main training + evaluation function
# ──────────────────────────────────────────────────────────

def run_resunext(dataset_name, dataset_dir, in_channels, epochs=3, batch_size=4):
    print(f"\n{'='*55}")
    print(f"  ResUNeXt  |  {dataset_name}")
    print(f"{'='*55}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Datasets ──────────────────────────────────────────
    if dataset_name == "MADOS_5Class":
        train_ds = MADOSDataset(dataset_dir, split="train")
        test_ds  = MADOSDataset(dataset_dir, split="test")
    else:
        train_ds = MARIDADataset(dataset_dir, split="train")
        test_ds  = MARIDADataset(dataset_dir, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────
    model = ResUNeXt(in_channels=in_channels, num_classes=5,
                     base_ch=32, cardinality=16).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}  ({total_params * 4 / 1e6:.2f} MB)")

    # ── Training ──────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    t0_train = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        n_batches = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        print(f"  Epoch [{epoch}/{epochs}]  loss: {running_loss / max(1, n_batches):.4f}")
    train_time = time.time() - t0_train
    print(f"  Training done in {train_time:.1f}s")

    # ── Evaluation ────────────────────────────────────────
    print("  Evaluating …")
    model.eval()
    all_preds, all_tgts = [], []
    t0_inf = time.time()
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            preds = torch.argmax(model(imgs), dim=1).cpu().numpy().flatten()
            tgts  = masks.numpy().flatten()
            valid = (tgts >= 0) & (tgts < 5)
            all_preds.append(preds[valid])
            all_tgts.append(tgts[valid])
    inf_time = time.time() - t0_inf

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_tgts)
    metrics = calculate_metrics(y_true, y_pred, num_classes=5)
    ov = metrics['overall']

    lat_patch_ms = inf_time / len(test_ds) * 1000
    throughput   = len(y_true) / inf_time

    print(f"\n  Accuracy  : {ov['accuracy']:.4f}")
    print(f"  Precision : {ov['precision']:.4f}")
    print(f"  Recall    : {ov['recall']:.4f}")
    print(f"  F1 Score  : {ov['f1']:.4f}")
    print(f"  IoU       : {ov['iou']:.4f}")
    print(f"\n  Confusion Matrix:\n{metrics['confusion_matrix']}")

    # ── Output directories ────────────────────────────────
    os.makedirs(CM_FOLDER, exist_ok=True)
    sub = dataset_name.split("_")[0]                           # MADOS / MARIDA
    log_dir = os.path.join(repo_root, "outputs", "logs", sub)
    os.makedirs(log_dir, exist_ok=True)
    ds_parent = os.path.abspath(os.path.join(dataset_dir, ".."))
    ds_out    = os.path.join(ds_parent, "outputs")
    os.makedirs(ds_out, exist_ok=True)

    # ── Save model ────────────────────────────────────────
    model_name_slug = dataset_name.lower()[:5]
    pth_path = os.path.join(repo_root, "outputs",
                            f"{model_name_slug}_resunext_model.pth")
    torch.save(model.state_dict(), pth_path)
    model_size_mb = os.path.getsize(pth_path) / (1024 ** 2)
    # copy to dataset folder
    import shutil
    shutil.copy(pth_path, os.path.join(ds_out,
                f"{sub.lower()}_resunext_model.pth"))

    # ── Confusion-matrix heatmap → confusion_matrices/ ────
    cm = metrics['confusion_matrix']
    cm_png_name = f"{dataset_name}_ResUNeXt_confusion_heatmap.png"
    cm_png_path = os.path.join(CM_FOLDER, cm_png_name)
    save_dual_heatmap(cm, f"ResUNeXt  |  {dataset_name}  — Confusion Matrix",
                      cm_png_path)
    # extra copy to logs
    shutil.copy(cm_png_path, os.path.join(log_dir,
                f"{dataset_name}_resunext_confusion_matrix.png"))

    # ── Per-class CSV / MD table ──────────────────────────
    table_base = os.path.join(log_dir, f"{dataset_name}_resunext_metrics")
    save_metrics_table(metrics, CLASS_NAMES, table_base)

    # ── Accuracy txt → repo root + dataset folder ─────────
    txt_name = f"{model_name_slug}_resunext_results.txt"
    txt_repo = os.path.join(repo_root, txt_name)
    save_accuracy_txt(
        metrics, dataset_name, "ResUNeXt",
        num_train=len(train_ds), num_test=len(test_ds),
        in_ch=in_channels, total_params=total_params,
        train_time=train_time, inf_time=inf_time,
        lat_patch_ms=lat_patch_ms, throughput=throughput,
        model_size_mb=model_size_mb,
        txt_path=txt_repo
    )
    shutil.copy(txt_repo, os.path.join(ds_parent,
                f"{sub.lower()}_resunext_results.txt"))

    # ── JSON summary ──────────────────────────────────────
    summary = {
        "dataset": dataset_name, "model": "ResUNeXt",
        "in_channels": in_channels, "total_params": total_params,
        "model_size_mb": model_size_mb,
        "train_time_sec": train_time, "inference_time_sec": inf_time,
        "latency_per_patch_ms": lat_patch_ms,
        "throughput_pixels_sec": throughput,
        "metrics": {
            "accuracy":  ov['accuracy'],
            "precision": ov['precision'],
            "recall":    ov['recall'],
            "f1":        ov['f1'],
            "iou":       ov['iou'],
        }
    }
    return summary


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    mados_path  = os.path.abspath("MADOS_5Class/MADOS_5Class")
    marida_path = os.path.abspath("MARIDA_5Class/MARIDA_5Class")

    s_mados  = run_resunext("MADOS_5Class",  mados_path,  in_channels=8,  epochs=3, batch_size=4)
    s_marida = run_resunext("MARIDA_5Class", marida_path, in_channels=11, epochs=3, batch_size=4)

    print("\n" + "=" * 55)
    print("  RESUNEXT EXPERIMENT SUMMARY")
    print("=" * 55)
    print(json.dumps({"MADOS": s_mados, "MARIDA": s_marida}, indent=2))
