import os
import sys
import yaml
import time
import json
import csv
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.marida import MARIDADataset
from src.datasets.mados import MADOSDataset
from src.models.rf_classifier import RFClassifier
from src.metrics.metrics import calculate_metrics, plot_confusion_matrix, save_metrics_table
from src.preprocessing.features import extract_all_features

np.random.seed(42)

def _flatten_dataset_advanced(dataset, dataset_name, ignore_index, desc, rf_cfg):
    all_X = []
    all_y = []
    
    use_si = rf_cfg.get("features", {}).get("use_si", True)
    use_glcm = rf_cfg.get("features", {}).get("use_glcm", True)
    
    for idx in tqdm(range(len(dataset)), desc=desc):
        img_tensor, mask_tensor = dataset[idx]
        img = img_tensor.numpy()
        mask = mask_tensor.numpy()
        
        # Apply Advanced Feature Extraction!
        img_feats = extract_all_features(img, dataset_name, use_si=use_si, use_glcm=use_glcm)

        C, H, W = img_feats.shape
        X = img_feats.reshape(C, H * W).T
        y = mask.reshape(-1)

        valid = y != ignore_index
        if np.any(valid):
            all_X.append(X[valid])
            all_y.append(y[valid])

    if len(all_X) == 0:
        return np.array([]), np.array([])
        
    return np.concatenate(all_X), np.concatenate(all_y)

def _stratified_subsample(X, y, max_pixels, num_classes, rng):
    pixels_per_class = max_pixels // num_classes
    sampled_idx = []
    
    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        if len(idx_c) == 0:
            continue
        
        if len(idx_c) > pixels_per_class:
            idx_c = rng.choice(idx_c, size=pixels_per_class, replace=False)
            
        sampled_idx.append(idx_c)
        
    final_idx = np.concatenate(sampled_idx)
    rng.shuffle(final_idx)
    
    return X[final_idx], y[final_idx]

def train_and_eval(dataset_name, dataset_class, dataset_cfg, rf_cfg):
    print(f"\n{'='*60}")
    print(f"  Advanced RF Pipeline (RFSS+SI+GLCM) — {dataset_name}")
    print(f"{'='*60}\n")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rng = np.random.RandomState(rf_cfg.get("random_state", 42))
    ignore_index = rf_cfg.get("ignore_index", -1)
    num_classes  = rf_cfg.get("num_classes", 5)
    max_train_px = rf_cfg.get("max_train_pixels", 2000000)

    root_dir = dataset_cfg["root_dir"]
    print("[INFO] Initialising datasets …")
    train_ds = dataset_class(root_dir, split="train")
    val_ds   = dataset_class(root_dir, split="val")
    test_ds  = dataset_class(root_dir, split="test")
    print(f"[INFO] Dataset sizes → Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    print("\n[INFO] Flattening & Augmenting training pixels (This WILL take time) …")
    X_train, y_train = _flatten_dataset_advanced(train_ds, dataset_name, ignore_index, desc="Train patches", rf_cfg=rf_cfg)
    
    if len(y_train) == 0:
        print("[ERROR] No valid training pixels found.")
        return
        
    print(f"[INFO] Total training pixels: {len(y_train):,} with {X_train.shape[1]} features")

    if max_train_px and len(y_train) > max_train_px:
        X_train, y_train = _stratified_subsample(X_train, y_train, max_train_px, num_classes, rng)
        print(f"[INFO] After stratified subsampling: {len(y_train):,} pixels")

    unique, counts = np.unique(y_train, return_counts=True)
    class_names = ["Background", "Water", "Debris", "Algae/Sargassum", "Other"]
    print("[INFO] Class distribution in training subset:")
    for cls_id, cnt in zip(unique, counts):
        name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"Class {cls_id}"
        print(f"       {name}: {cnt:,} ({100 * cnt / len(y_train):.2f}%)")

    print("\n[INFO] Training Random Forest …")
    clf = RFClassifier(rf_cfg)
    train_start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"[INFO] Training completed in {train_time:.2f} sec")

    ckpt_dir = os.path.join("checkpoints", "rf_advanced", dataset_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"rf_adv_{run_timestamp}.joblib")
    clf.save(ckpt_path)
    print(f"[INFO] Model saved to: {ckpt_path}")

    model_info_dir = os.path.join("outputs", "model_info")
    os.makedirs(model_info_dir, exist_ok=True)
    summary_path = os.path.join(model_info_dir, f"{dataset_name}_rf_adv_summary_{run_timestamp}.txt")
    clf.write_model_summary(summary_path, dataset_name, len(y_train), train_time)

    print("\n[INFO] Evaluating on test set (This WILL take time due to GLCM) …")
    all_preds = []
    all_targets = []

    use_si = rf_cfg.get("features", {}).get("use_si", True)
    use_glcm = rf_cfg.get("features", {}).get("use_glcm", True)

    inference_start = time.time()
    for idx in tqdm(range(len(test_ds)), desc="Test patches"):
        img_tensor, mask_tensor = test_ds[idx]
        img = img_tensor.numpy()
        mask = mask_tensor.numpy()

        img_feats = extract_all_features(img, dataset_name, use_si=use_si, use_glcm=use_glcm)
        C, H, W = img_feats.shape
        X = img_feats.reshape(C, H * W).T
        y = mask.reshape(-1)

        valid = y != ignore_index
        X_valid = X[valid]
        y_valid = y[valid]

        if len(y_valid) == 0:
            continue

        preds = clf.predict(X_valid)
        all_preds.append(preds)
        all_targets.append(y_valid)

    inference_time = time.time() - inference_start
    time_per_image = inference_time / len(test_ds) if len(test_ds) > 0 else 0.0

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    print(f"[INFO] Calculating metrics on {len(all_preds):,} test pixels …")
    results = calculate_metrics(all_targets, all_preds, num_classes=num_classes)

    print("\n[INFO] Evaluating on validation set (This WILL take time due to GLCM) …")
    val_preds_list = []
    val_targets_list = []
    
    for idx in tqdm(range(len(val_ds)), desc="Val patches"):
        img_tensor, mask_tensor = val_ds[idx]
        img = img_tensor.numpy()
        mask = mask_tensor.numpy()

        img_feats = extract_all_features(img, dataset_name, use_si=use_si, use_glcm=use_glcm)
        C, H, W = img_feats.shape
        X = img_feats.reshape(C, H * W).T
        y = mask.reshape(-1)

        valid = y != ignore_index
        X_valid = X[valid]
        y_valid = y[valid]

        if len(y_valid) == 0:
            continue

        preds = clf.predict(X_valid)
        val_preds_list.append(preds)
        val_targets_list.append(y_valid)
        
    if len(val_targets_list) > 0:
        val_y = np.concatenate(val_targets_list)
        val_preds = np.concatenate(val_preds_list)
        val_results = calculate_metrics(val_y, val_preds, num_classes=num_classes)
        
        # Merge metrics
        results["val_overall"] = val_results["overall"]
        results["val_per_class"] = val_results["per_class"]

    results["time_per_image_sec"] = time_per_image
    results["total_inference_sec"] = inference_time
    results["dataset"] = dataset_name
    results["timestamp"] = run_timestamp

    log_dir = os.path.join("outputs", "logs", dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return super().default(obj)

    report_path = os.path.join(log_dir, f"rf_adv_report_{run_timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
        
    csv_log_path = os.path.join(log_dir, f"rf_adv_training_log_{run_timestamp}.csv")
    fieldnames = ["timestamp", "dataset", "accuracy", "macro_f1", "macro_iou", "macro_precision", "macro_recall",
                  "val_accuracy", "val_macro_f1", "val_macro_iou", "time_per_image_sec"]
    write_header = not os.path.exists(csv_log_path)
    with open(csv_log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": run_timestamp,
            "dataset": dataset_name,
            "accuracy": results['overall']['accuracy'],
            "macro_f1": results['overall']['f1'],
            "macro_iou": results['overall']['iou'],
            "macro_precision": results['overall']['precision'],
            "macro_recall": results['overall']['recall'],
            "val_accuracy": results.get('val_overall', {}).get('accuracy', ''),
            "val_macro_f1": results.get('val_overall', {}).get('f1', ''),
            "val_macro_iou": results.get('val_overall', {}).get('iou', ''),
            "time_per_image_sec": time_per_image
        })

    table_prefix = os.path.join(log_dir, f"rf_adv_per_class_metrics_{run_timestamp}")
    save_metrics_table(results, class_names, table_prefix)
            
    cm_path = os.path.join(log_dir, f"rf_adv_confusion_matrix_{run_timestamp}.png")
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)

    print("\n[INFO] Pipeline Completed!")
    print(f"       Metrics MD   : {table_prefix}.md")
    print(f"       Conf Matrix  : {cm_path}")
    print(f"       Model        : {ckpt_path}")

def main():
    with open("configs/dataset.yaml", "r") as f:
        datasets_cfg = yaml.safe_load(f)

    with open("configs/rf_advanced.yaml", "r") as f:
        rf_cfg = yaml.safe_load(f)

    if "marida" in datasets_cfg:
        train_and_eval("MARIDA", MARIDADataset, datasets_cfg["marida"], rf_cfg)

    if "mados" in datasets_cfg:
        train_and_eval("MADOS", MADOSDataset, datasets_cfg["mados"], rf_cfg)

if __name__ == "__main__":
    main()
