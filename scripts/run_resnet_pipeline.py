import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.datasets.mados import MADOSDataset
from src.datasets.marida import MARIDADataset
from src.models.resnet_segmentation import ResNetSegmentation
from src.metrics.metrics import calculate_metrics, plot_confusion_matrix, save_metrics_table

CLASS_NAMES = ["Marine Debris", "Sargassum/Veg", "Natural Phenom/Foam", "Ship/Infrastructure", "Water/Other"]

def collate_fn(batch):
    imgs = [b[0] for b in batch]
    masks = [b[1] for b in batch]
    return torch.stack(imgs, dim=0), torch.stack(masks, dim=0)

def train_and_evaluate_resnet(dataset_name, dataset_dir, in_channels, epochs=5, batch_size=8, lr=1e-3):
    print(f"\n=======================================================")
    print(f" Running ResNet-18 Segmentation on {dataset_name}")
    print(f"=======================================================\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if dataset_name == "MADOS_5Class":
        train_ds = MADOSDataset(dataset_dir, split="train")
        test_ds = MADOSDataset(dataset_dir, split="test")
    else:
        train_ds = MARIDADataset(dataset_dir, split="train")
        test_ds = MARIDADataset(dataset_dir, split="test")
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = ResNetSegmentation(in_channels=in_channels, num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet-18 Model Parameters: {total_params:,} ({total_params * 4 / (1024*1024):.2f} MB)")
    
    # Training Loop
    print(f"Training for {epochs} epochs...")
    start_train_time = time.time()
    
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        num_batches = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs) # (B, 5, H, W)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
        avg_loss = running_loss / max(1, num_batches)
        print(f"  Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")
        
    train_time = time.time() - start_train_time
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Evaluation Loop
    print("Evaluating test set...")
    model.eval()
    all_preds = []
    all_targets = []
    
    start_inf_time = time.time()
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
            targets = masks.numpy().flatten()
            
            # Filter valid labels 0 to 4
            valid_mask = (targets >= 0) & (targets < 5)
            all_preds.append(preds[valid_mask])
            all_targets.append(targets[valid_mask])
            
    inf_time = time.time() - start_inf_time
    
    y_pred = np.concatenate(all_preds)
    y_test = np.concatenate(all_targets)
    
    metrics = calculate_metrics(y_test, y_pred, num_classes=5)
    
    # Latency statistics
    num_test_pixels = len(y_test)
    latency_per_1m_pixels = (inf_time / num_test_pixels) * 1e6
    latency_per_patch_ms = (inf_time / len(test_ds)) * 1000.0
    throughput_pixels_sec = num_test_pixels / inf_time
    
    print(f"\nMetrics for {dataset_name} (ResNet-18):")
    print(f"  Precision    : {metrics['overall']['precision']:.6f}")
    print(f"  Recall       : {metrics['overall']['recall']:.6f}")
    print(f"  F1 Score     : {metrics['overall']['f1']:.6f}")
    print(f"  IoU (Jaccard): {metrics['overall']['iou']:.6f}")
    print(f"  Accuracy     : {metrics['overall']['accuracy']:.6f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Save Model Artifacts
    os.makedirs(os.path.join(repo_root, "outputs"), exist_ok=True)
    sub_dir = dataset_name.split("_")[0]
    out_log_dir = os.path.join(repo_root, "outputs", "logs", sub_dir)
    os.makedirs(out_log_dir, exist_ok=True)
    
    model_pth_path = os.path.join(repo_root, "outputs", f"{dataset_name.lower()[:5]}_resnet_model.pth")
    torch.save(model.state_dict(), model_pth_path)
    model_size_mb = os.path.getsize(model_pth_path) / (1024.0 * 1024.0)
    
    # Save formatted text results
    results_str = f"Loading {dataset_name} Train dataset ({len(train_ds)} patches)...\n"
    results_str += f"Loading {dataset_name} Test dataset ({len(test_ds)} patches)...\n"
    results_str += f"Training ResNet-18 on multi-spectral patches ({in_channels} channels)...\n"
    results_str += f"Evaluating...\n\n"
    results_str += f"Metrics:\n"
    results_str += f"Precision: {metrics['overall']['precision']:.16f}\n"
    results_str += f"Recall: {metrics['overall']['recall']:.16f}\n"
    results_str += f"F1 Score: {metrics['overall']['f1']:.16f}\n"
    results_str += f"IoU (Jaccard): {metrics['overall']['iou']:.16f}\n"
    results_str += f"Accuracy: {metrics['overall']['accuracy']:.16f}\n\n"
    results_str += f"Confusion Matrix:\n{metrics['confusion_matrix']}\n\n"
    results_str += f"Saving model to {os.path.basename(model_pth_path)}...\n"
    
    txt_filename = f"{dataset_name.lower()[:5]}_resnet_results.txt"
    txt_filepath = os.path.join(repo_root, txt_filename)
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write(results_str)
        
    # Save CSV and MD tables
    table_base_path = os.path.join(out_log_dir, f"{dataset_name}_resnet_metrics")
    save_metrics_table(metrics, CLASS_NAMES, table_base_path)
    
    # Save Confusion Matrix Plot
    cm_plot_path = os.path.join(out_log_dir, f"{dataset_name}_resnet_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, cm_plot_path)
    
    # Copy to dataset folders
    ds_target_folder = os.path.abspath(os.path.join(dataset_dir, ".."))
    if os.path.exists(ds_target_folder):
        os.makedirs(os.path.join(ds_target_folder, "outputs"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ds_target_folder, "outputs", f"{sub_dir.lower()}_resnet_model.pth"))
        with open(os.path.join(ds_target_folder, f"{sub_dir.lower()}_resnet_results.txt"), "w", encoding="utf-8") as f:
            f.write(results_str)
            
    summary_dict = {
        "dataset": dataset_name,
        "model_architecture": "ResNet-18 Segmentation",
        "in_channels": in_channels,
        "total_params": total_params,
        "model_size_mb": model_size_mb,
        "train_time_sec": train_time,
        "inference_time_sec": inf_time,
        "latency_per_patch_ms": latency_per_patch_ms,
        "latency_per_1m_pixels_sec": latency_per_1m_pixels,
        "throughput_pixels_sec": throughput_pixels_sec,
        "metrics": {
            "precision": metrics['overall']['precision'],
            "recall": metrics['overall']['recall'],
            "f1": metrics['overall']['f1'],
            "iou": metrics['overall']['iou'],
            "accuracy": metrics['overall']['accuracy']
        }
    }
    
    return summary_dict

if __name__ == "__main__":
    mados_path = os.path.abspath("MADOS_5Class/MADOS_5Class")
    if not os.path.exists(mados_path):
        mados_path = os.path.abspath("MADOS_5Class")
        
    marida_path = os.path.abspath("MARIDA_5Class/MARIDA_5Class")
    if not os.path.exists(marida_path):
        marida_path = os.path.abspath("MARIDA_5Class")
        
    mados_resnet_sum = train_and_evaluate_resnet("MADOS_5Class", mados_path, in_channels=8, epochs=3, batch_size=8)
    marida_resnet_sum = train_and_evaluate_resnet("MARIDA_5Class", marida_path, in_channels=11, epochs=3, batch_size=8)
    
    print("\n=======================================================")
    print(" SUMMARY OF RESNET EXPERIMENTS")
    print("=======================================================")
    print(json.dumps({"MADOS": mados_resnet_sum, "MARIDA": marida_resnet_sum}, indent=2))
