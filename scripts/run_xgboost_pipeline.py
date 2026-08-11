import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.datasets.mados import MADOSDataset
from src.datasets.marida import MARIDADataset
from src.models.xgb_classifier import extract_pixel_features_21, MarineXGBClassifier
from src.metrics.metrics import calculate_metrics, plot_confusion_matrix, save_metrics_table

CLASS_NAMES = ["Marine Debris", "Sargassum/Veg", "Natural Phenom/Foam", "Ship/Infrastructure", "Water/Other"]

def sample_dataset_features(dataset, max_pixels_per_class=250000, random_seed=42):
    np.random.seed(random_seed)
    X_list = []
    y_list = []
    
    total_patches = len(dataset)
    print(f"Extracting features from {total_patches} patches...")
    
    for idx in range(total_patches):
        img_tensor, mask_tensor = dataset[idx]
        feats = extract_pixel_features_21(img_tensor) # (H*W, 21)
        mask = mask_tensor.numpy().flatten()
        
        # Valid pixels (0 to 4)
        valid_mask = (mask >= 0) & (mask < 5)
        if not np.any(valid_mask):
            continue
            
        feats_valid = feats[valid_mask]
        mask_valid = mask[valid_mask]
        
        X_list.append(feats_valid)
        y_list.append(mask_valid)
        
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    
    # Balance or subsample if total pixels is huge
    if len(y_all) > max_pixels_per_class * 5:
        indices = []
        for c in range(5):
            c_indices = np.where(y_all == c)[0]
            if len(c_indices) > max_pixels_per_class:
                c_indices = np.random.choice(c_indices, max_pixels_per_class, replace=False)
            indices.append(c_indices)
        indices = np.concatenate(indices)
        np.random.shuffle(indices)
        X_all = X_all[indices]
        y_all = y_all[indices]
        
    return X_all, y_all

def evaluate_xgboost_on_dataset(dataset_name, dataset_dir, max_train_pixels=500000):
    print(f"\n=======================================================")
    print(f" Running XGBoost on {dataset_name} Dataset")
    print(f"=======================================================\n")
    
    if dataset_name == "MADOS_5Class":
        train_ds = MADOSDataset(dataset_dir, split="train")
        test_ds = MADOSDataset(dataset_dir, split="test")
    else:
        train_ds = MARIDADataset(dataset_dir, split="train")
        test_ds = MARIDADataset(dataset_dir, split="test")
        
    print(f"Loading {dataset_name} Train dataset ({len(train_ds)} patches)...")
    X_train, y_train = sample_dataset_features(train_ds, max_pixels_per_class=max_train_pixels // 5)
    print(f"Training XGBoost on {len(y_train)} pixels with {X_train.shape[1]} features...")
    
    print(f"Loading {dataset_name} Test dataset ({len(test_ds)} patches)...")
    X_test, y_test = sample_dataset_features(test_ds, max_pixels_per_class=max_train_pixels // 5)
    
    # Initialize XGBoost model
    num_trees = 100
    max_depth = 6
    xgb_clf = MarineXGBClassifier(n_estimators=num_trees, max_depth=max_depth, learning_rate=0.1, num_classes=5)
    
    # Train model
    train_time = xgb_clf.fit(X_train, y_train)
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Evaluate model
    print("Evaluating test set...")
    y_pred, inf_time = xgb_clf.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred, num_classes=5)
    
    # Inference statistics
    num_test_pixels = len(y_test)
    latency_per_1m_pixels = (inf_time / num_test_pixels) * 1e6
    latency_per_patch_ms = (inf_time / len(test_ds)) * 1000.0
    throughput_pixels_sec = num_test_pixels / inf_time
    
    print(f"\nMetrics for {dataset_name}:")
    print(f"  Precision    : {metrics['overall']['precision']:.6f}")
    print(f"  Recall       : {metrics['overall']['recall']:.6f}")
    print(f"  F1 Score     : {metrics['overall']['f1']:.6f}")
    print(f"  IoU (Jaccard): {metrics['overall']['iou']:.6f}")
    print(f"  Accuracy     : {metrics['overall']['accuracy']:.6f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Save Model Artifacts
    os.makedirs(os.path.join(repo_root, "outputs"), exist_ok=True)
    out_log_dir = os.path.join(repo_root, "outputs", "logs", dataset_name.split("_")[0])
    os.makedirs(out_log_dir, exist_ok=True)
    
    model_json_path = os.path.join(repo_root, "outputs", f"{dataset_name.lower()}_xgb_model.json")
    xgb_clf.save(model_json_path)
    model_size_mb = os.path.getsize(model_json_path) / (1024.0 * 1024.0)
    
    # Formatted results text string matching repo standard
    results_str = f"Loading {dataset_name} Train dataset...\n"
    results_str += f"Loading {dataset_name} Test dataset...\n"
    results_str += f"Extracting features for train set...\n"
    results_str += f"Extracting features for test set...\n"
    results_str += f"Training XGBoost on {len(y_train)} pixels with {X_train.shape[1]} features...\n"
    results_str += f"Evaluating...\n\n"
    results_str += f"Metrics:\n"
    results_str += f"Precision: {metrics['overall']['precision']:.16f}\n"
    results_str += f"Recall: {metrics['overall']['recall']:.16f}\n"
    results_str += f"F1 Score: {metrics['overall']['f1']:.16f}\n"
    results_str += f"IoU (Jaccard): {metrics['overall']['iou']:.16f}\n"
    results_str += f"Accuracy: {metrics['overall']['accuracy']:.16f}\n\n"
    results_str += f"Confusion Matrix:\n{metrics['confusion_matrix']}\n\n"
    results_str += f"Saving model to {os.path.basename(model_json_path)}...\n"
    
    # Save txt files in root directory
    txt_filename = f"{dataset_name.lower()[:5]}_xgb_results.txt"
    txt_filepath = os.path.join(repo_root, txt_filename)
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write(results_str)
        
    # Save CSV and MD tables
    table_base_path = os.path.join(out_log_dir, f"{dataset_name}_xgb_metrics")
    save_metrics_table(metrics, CLASS_NAMES, table_base_path)
    
    # Save Confusion Matrix Plot
    cm_plot_path = os.path.join(out_log_dir, f"{dataset_name}_xgb_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, cm_plot_path)
    
    # Copy files into dataset folder as well
    ds_target_folder = os.path.abspath(os.path.join(dataset_dir, ".."))
    if os.path.exists(ds_target_folder):
        ds_txt_path = os.path.join(ds_target_folder, txt_filename)
        with open(ds_txt_path, "w", encoding="utf-8") as f:
            f.write(results_str)
            
    summary_dict = {
        "dataset": dataset_name,
        "train_pixels": len(y_train),
        "test_pixels": len(y_test),
        "num_features": X_train.shape[1],
        "num_trees": num_trees,
        "max_depth": max_depth,
        "num_classes": 5,
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
    
    return summary_dict, txt_filepath

if __name__ == "__main__":
    # Base paths
    mados_path = os.path.abspath("MADOS_5Class/MADOS_5Class")
    if not os.path.exists(mados_path):
        mados_path = os.path.abspath("MADOS_5Class")
        
    marida_path = os.path.abspath("MARIDA_5Class/MARIDA_5Class")
    if not os.path.exists(marida_path):
        marida_path = os.path.abspath("MARIDA_5Class")
        
    mados_summary, mados_txt = evaluate_xgboost_on_dataset("MADOS_5Class", mados_path)
    marida_summary, marida_txt = evaluate_xgboost_on_dataset("MARIDA_5Class", marida_path)
    
    print("\n=======================================================")
    print(" SUMMARY OF XGBOOST EXPERIMENTS")
    print("=======================================================")
    print(json.dumps({"MADOS": mados_summary, "MARIDA": marida_summary}, indent=2))
