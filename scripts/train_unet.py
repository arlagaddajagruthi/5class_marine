import os
import sys
import yaml
import time
import json
import csv
import numpy as np
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.marida import MARIDADataset
from src.datasets.mados import MADOSDataset
from src.models.unet import UNet
from src.losses.combined_loss import CombinedLoss
from src.metrics.metrics import calculate_metrics, plot_confusion_matrix, save_metrics_table

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def train_and_eval(dataset_name, dataset_class, dataset_cfg, unet_cfg):
    print(f"\n{'='*50}")
    print(f"Starting Training Pipeline for {dataset_name}")
    print(f"{'='*50}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using Device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] CUDA Device Name: {torch.cuda.get_device_name(0)}")
        
    # DataLoaders
    batch_size = dataset_cfg['batch_size']
    num_workers = dataset_cfg['num_workers']
    root_dir = dataset_cfg['root_dir']
    
    print("[INFO] Initializing Datasets...")
    train_ds = dataset_class(root_dir, split="train")
    val_ds = dataset_class(root_dir, split="val")
    test_ds = dataset_class(root_dir, split="test")
    
    print(f"[INFO] Dataset Sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Model
    in_channels = dataset_cfg.get('in_channels', unet_cfg['in_channels'])
    print(f"[INFO] Initializing U-Net Model with {in_channels} input channels...")
    model = UNet(in_channels=in_channels, out_channels=unet_cfg['out_channels']).to(device)
    
    # Print Architecture and MACs/FLOPs
    print("\n[INFO] Model Architecture & Complexity (torchinfo):")

    model_info_dir = os.path.join("outputs", "model_info")
    os.makedirs(model_info_dir, exist_ok=True)

    model_summary_path = os.path.join(
        model_info_dir,
        f"{dataset_name}_model_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    try:

        model_stats = summary(
            model,
            input_size=(1, in_channels, 256, 256),
            verbose=0
        )

        print(model_stats)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = total_params - trainable_params

        fp32_size = total_params * 4 / (1024 ** 2)
        fp16_size = total_params * 2 / (1024 ** 2)

        mult_adds = model_stats.total_mult_adds

        if mult_adds is not None:
            gmacs = mult_adds / 1e9
            gflops = gmacs * 2
        else:
            gmacs = None
            gflops = None

        with open(model_summary_path, "w", encoding="utf-8") as f:

            f.write("=" * 80 + "\n")
            f.write("MODEL INFORMATION\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Dataset               : {dataset_name}\n")
            f.write(f"Timestamp             : {datetime.now()}\n")
            f.write(f"Device                : {device}\n")

            if torch.cuda.is_available():
                f.write(f"GPU                   : {torch.cuda.get_device_name(0)}\n")

            f.write(f"PyTorch               : {torch.__version__}\n")
            f.write(f"CUDA                  : {torch.version.cuda}\n")
            f.write(f"Input Channels        : {in_channels}\n")
            f.write(f"Output Classes        : {unet_cfg['out_channels']}\n")
            f.write(f"Input Resolution      : 256 x 256\n")
            f.write(f"Batch Size            : {batch_size}\n\n")

            f.write("=" * 80 + "\n")
            f.write("MODEL STATISTICS\n")
            f.write("=" * 80 + "\n")

            f.write(f"Total Parameters      : {total_params:,}\n")
            f.write(f"Trainable Parameters  : {trainable_params:,}\n")
            f.write(f"Non-trainable Params  : {non_trainable:,}\n")

            if gmacs is not None:
                f.write(f"MACs                  : {gmacs:.3f} GMAC\n")
                f.write(f"Estimated FLOPs       : {gflops:.3f} GFLOPs\n")

            f.write(f"FP32 Model Size       : {fp32_size:.2f} MB\n")
            f.write(f"FP16 Model Size       : {fp16_size:.2f} MB\n\n")

            f.write("=" * 80 + "\n")
            f.write("TORCHINFO SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(str(model_stats))

        print(f"[INFO] Model summary saved to: {model_summary_path}")

    except Exception as e:

        print(f"Could not calculate model summary: {e}")
        
    # Compute class weights from training set (median-frequency balancing)
    print("[INFO] Computing class weights from training data...")
    class_counts = np.zeros(unet_cfg['out_channels'], dtype=np.float64)
    for _, mask in tqdm(train_loader, desc="Counting class pixels"):
        mask_np = mask.numpy()
        for c in range(unet_cfg['out_channels']):
            class_counts[c] += np.sum(mask_np == c)
    
    # Median-frequency balancing: weight = median(freq) / freq_per_class.
    # This is bounded relative to raw inverse-frequency weighting (which can
    # blow up to the thousands for very rare classes and dominate the
    # gradient), while still upweighting minority classes.
    total_pixels = class_counts.sum()
    class_freq = class_counts / (total_pixels + 1e-12)
    nonzero_freq = class_freq[class_freq > 0]
    median_freq = np.median(nonzero_freq) if len(nonzero_freq) > 0 else 1.0
    class_weights = median_freq / (class_freq + 1e-6)

    # Prevent extremely large weights
    max_weight = unet_cfg.get('class_weight_clip', 10.0)
    class_weights = np.clip(class_weights, a_min=0.0, a_max=max_weight)

    # Normalize weights so their mean is 1
    class_weights = class_weights / class_weights.mean()

    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"[INFO] Class distribution: {dict(zip(['BG', 'Water', 'Debris', 'Algae', 'Other'], class_counts.astype(int)))}")
    print(f"[INFO] Class weights (median-freq, clipped to {max_weight}): {class_weights.cpu().numpy().round(2)}")
    
    focal_gamma = unet_cfg.get('focal_gamma', 2.0)
    print(f"[INFO] Using Focal Loss with gamma={focal_gamma}")
    criterion = CombinedLoss(weight=class_weights, ignore_index=unet_cfg['ignore_index'], alpha=unet_cfg['loss_alpha'], gamma=focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=unet_cfg['learning_rate'], weight_decay=unet_cfg['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6) # max becuase we are now monitoring miou

    best_val_loss = float("inf")      # keep for logging
    best_val_miou = -1.0              # store best miou
    best_epoch = -1
    epochs = unet_cfg['epochs']
    
    # Timestamped run directory for this training session
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", "unet", dataset_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"best_model_{run_timestamp}.pth")
    latest_model_path = os.path.join(checkpoint_dir, f"latest_model_{run_timestamp}.pth")
    
    # Per-epoch training log CSV
    log_dir = os.path.join("outputs", "logs", dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    epoch_log_path = os.path.join(log_dir, f"training_log_{run_timestamp}.csv")
    with open(epoch_log_path, "w", newline="") as lf:
        writer = csv.writer(lf)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_miou",
            "val_f1",
            "val_recall",
            "val_precision",
            "learning_rate",
            "best_val_loss",
            "best_val_miou",
            "best_epoch",
            "best_model_path"
        ])
    print(f"[INFO] Epoch log will be saved to: {epoch_log_path}")
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        pbar = tqdm(train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
                preds_np = torch.argmax(logits, dim=1).cpu().numpy()
                masks_np = masks.cpu().numpy()
                valid = (masks_np != unet_cfg['ignore_index'])
                val_preds.append(preds_np[valid])
                val_targets.append(masks_np[valid])
        
        val_loss /= len(val_loader)
        
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_results = calculate_metrics(val_targets, val_preds, num_classes=unet_cfg['out_channels'])
        val_miou = val_results['overall']['iou']
        val_f1 = val_results['overall']['f1']
        val_recall = val_results['overall']['recall']
        val_precision = val_results['overall']['precision']

        scheduler.step(val_miou)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val mIoU: {val_miou:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f} | Val Precision: {val_precision:.4f}")
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.2e}")

        # Save checkpoints
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_miou": best_val_miou,
        }, best_model_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # checkpoint based on validation mIoU
        if val_miou > best_val_miou:

            best_val_miou = val_miou
            best_epoch = epoch + 1

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_miou": best_val_miou,
            }, best_model_path)

            print(f"[INFO] Saved new best model (mIoU = {best_val_miou:.4f})")
        
        # Log epoch results to CSV
        with open(epoch_log_path, "a", newline="") as lf:
            writer = csv.writer(lf)
            writer.writerow([
                epoch + 1,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_miou:.6f}",
                f"{val_f1:.6f}",
                f"{val_recall:.6f}",
                f"{val_precision:.6f}",
                f"{current_lr:.2e}",
                f"{best_val_loss:.6f}",
                f"{best_val_miou:.6f}",
                best_epoch,
                best_model_path
            ])
            
    # Evaluation on Test Set
    print("\n[INFO] Starting Final Evaluation on Test Set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    inference_start_time = time.time()
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.cpu().numpy()
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Mask out ignore index
            valid = (masks != unet_cfg['ignore_index'])
            all_preds.append(preds[valid])
            all_targets.append(masks[valid])
            
    inference_end_time = time.time()
    total_inference_time = inference_end_time - inference_start_time
    time_per_image = total_inference_time / len(test_ds)

    with open(model_summary_path, "a", encoding="utf-8") as f:

        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("INFERENCE PERFORMANCE\n")
        f.write("=" * 80 + "\n")

        f.write(f"Total Test Images     : {len(test_ds)}\n")
        f.write(f"Total Time            : {total_inference_time:.4f} sec\n")
        f.write(f"Time / Image          : {time_per_image:.6f} sec\n")
        f.write(f"Images / Second       : {1.0 / time_per_image:.2f}\n")
    
    # Concatenate all arrays at once (much faster than list.extend with millions of pixels)
    print("[INFO] Concatenating predictions...")
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    print(f"[INFO] Calculating Comprehensive Metrics on {len(all_preds):,} pixels...")
    results = calculate_metrics(all_targets, all_preds, num_classes=unet_cfg['out_channels'])
    
    # Save Results
    output_dir = os.path.join("outputs", "logs", dataset_name)
    cm_dir = os.path.join("outputs", "confusion_matrices")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    
    class_names = ["Background", "Water", "Debris", "Algae/Sargassum", "Other"]
    cm_path = os.path.join(cm_dir, f"{dataset_name}_cm_{run_timestamp}.png")
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)
    
    table_path = os.path.join(output_dir, f"per_class_metrics_{run_timestamp}")
    save_metrics_table(results, class_names, table_path)
    
    report_dict = {
        "dataset": dataset_name,
        "run_timestamp": run_timestamp,
        "epochs_trained": epochs,
        "best_val_loss": best_val_loss,
        "best_val_miou": best_val_miou,
        "best_epoch": best_epoch,
        "best_model_path": best_model_path,
        "latest_model_path": latest_model_path,
        "inference_time_total_sec": total_inference_time,
        "inference_time_per_image_sec": time_per_image,
        "overall_metrics": results['overall'],
        "per_class_metrics": {
            "precision": results['per_class']['precision'].tolist(),
            "recall": results['per_class']['recall'].tolist(),
            "f1": results['per_class']['f1'].tolist(),
            "iou": results['per_class']['iou'].tolist(),
        }
    }
    
    report_path = os.path.join(output_dir, f"evaluation_report_{run_timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
        
    print(f"\n{'='*50}")
    print(f"Final Evaluation for {dataset_name}:")
    print(f"Accuracy: {results['overall']['accuracy']:.4f}")
    print(f"Mean IoU: {results['overall']['iou']:.4f}")
    print(f"Mean F1:  {results['overall']['f1']:.4f}")
    print(f"Time/Img: {time_per_image:.4f} sec")
    print(f"Full report saved to: {report_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Best model saved to: {best_model_path}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    with open("configs/dataset.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    with open("configs/unet.yaml", "r") as f:
        unet_cfg = yaml.safe_load(f)
        
    # Run MARIDA
    try:
        train_and_eval("MARIDA", MARIDADataset, dataset_cfg['marida'], unet_cfg)
    except Exception as e:
        print(f"Failed to run MARIDA: {e}")
        
    # Run MADOS
    try:
        train_and_eval("MADOS", MADOSDataset, dataset_cfg['mados'], unet_cfg)
    except Exception as e:
        print(f"Failed to run MADOS: {e}")