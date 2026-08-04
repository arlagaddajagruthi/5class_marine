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

np.random.seed(42)


def _flatten_dataset(dataset, ignore_index, desc="Loading"):
    all_X = []
    all_y = []
    for idx in tqdm(range(len(dataset)), desc=desc):
        img_tensor, mask_tensor = dataset[idx]
        img = img_tensor.numpy()
        mask = mask_tensor.numpy()

        C, H, W = img.shape
        X = img.reshape(C, H * W).T
        y = mask.reshape(-1)

        valid = y != ignore_index
        all_X.append(X[valid])
        all_y.append(y[valid])

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
    print(f"  Random Forest Pipeline — {dataset_name}")
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

    print("\n[INFO] Flattening training pixels …")
    X_train, y_train = _flatten_dataset(train_ds, ignore_index, desc="Train patches")
    print(f"[INFO] Total training pixels: {len(y_train):,}")

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

    ckpt_dir = os.path.join("checkpoints", "rf", dataset_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"rf_model_{run_timestamp}.joblib")
    clf.save(ckpt_path)
    print(f"[INFO] Model saved to: {ckpt_path}")

    model_info_dir = os.path.join("outputs", "model_info")
    os.makedirs(model_info_dir, exist_ok=True)
    summary_path = os.path.join(
        model_info_dir,
        f"{dataset_name}_rf_model_summary_{run_timestamp}.txt"
    )
    clf.write_model_summary(summary_path, dataset_name, len(y_train), train_time)
    print(f"[INFO] Model summary saved to: {summary_path}")

    print("\n[INFO] Evaluating on test set …")
    all_preds = []
    all_targets = []

    inference_start = time.time()
    for idx in tqdm(range(len(test_ds)), desc="Test patches"):
        img_tensor, mask_tensor = test_ds[idx]
        img = img_tensor.numpy()
        mask = mask_tensor.numpy()

        C, H, W = img.shape
        X = img.reshape(C, H * W).T
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

    print(f"[INFO] Calculating metrics on {len(all_preds):,} pixels …")
    results = calculate_metrics(all_targets, all_preds, num_classes=num_classes)

    print("[INFO] Evaluating on validation set …")
    X_val, y_val = _flatten_dataset(val_ds, ignore_index, desc="Val patches")
    val_preds = clf.predict(X_val)
    val_results = calculate_metrics(y_val, val_preds, num_classes=num_classes)

    train_preds = clf.predict(X_train)
    from sklearn.metrics import accuracy_score
    train_accuracy = accuracy_score(y_train, train_preds)

    log_dir = os.path.join("outputs", "logs", dataset_name)
    cm_dir  = os.path.join("outputs", "confusion_matrices")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"rf_training_log_{run_timestamp}.csv")
    with open(log_path, "w", newline="") as lf:
        writer = csv.writer(lf)
        writer.writerow([
            "model",
            "train_samples",
            "train_time_sec",
            "train_accuracy",
            "val_miou",
            "val_f1",
            "val_recall",
            "val_precision",
            "test_miou",
            "test_f1",
            "test_recall",
            "test_precision",
            "test_accuracy",
            "checkpoint_path"
        ])
        writer.writerow([
            "RandomForest",
            len(y_train),
            f"{train_time:.4f}",
            f"{train_accuracy:.6f}",
            f"{val_results['overall']['iou']:.6f}",
            f"{val_results['overall']['f1']:.6f}",
            f"{val_results['overall']['recall']:.6f}",
            f"{val_results['overall']['precision']:.6f}",
            f"{results['overall']['iou']:.6f}",
            f"{results['overall']['f1']:.6f}",
            f"{results['overall']['recall']:.6f}",
            f"{results['overall']['precision']:.6f}",
            f"{results['overall']['accuracy']:.6f}",
            ckpt_path
        ])
    print(f"[INFO] Training log saved to: {log_path}")

    cm_path = os.path.join(cm_dir, f"{dataset_name}_rf_cm_{run_timestamp}.png")
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path)
    print(f"[INFO] Confusion matrix saved to: {cm_path}")

    table_path = os.path.join(log_dir, f"rf_per_class_metrics_{run_timestamp}")
    save_metrics_table(results, class_names, table_path)
    print(f"[INFO] Per-class metrics saved to: {table_path}.csv / .md")

    report_dict = {
        "dataset": dataset_name,
        "model": "RandomForest",
        "run_timestamp": run_timestamp,
        "hyperparameters": {
            k: rf_cfg.get(k) for k in (
                "n_estimators", "max_depth", "min_samples_split",
                "min_samples_leaf", "max_features", "class_weight",
                "random_state", "max_train_pixels"
            )
        },
        "training_samples": int(len(y_train)),
        "training_time_sec": train_time,
        "train_accuracy": train_accuracy,
        "val_metrics": val_results["overall"],
        "checkpoint_path": ckpt_path,
        "inference_time_total_sec": inference_time,
        "inference_time_per_image_sec": time_per_image,
        "overall_metrics": results["overall"],
        "per_class_metrics": {
            "precision": results["per_class"]["precision"].tolist(),
            "recall":    results["per_class"]["recall"].tolist(),
            "f1":        results["per_class"]["f1"].tolist(),
            "iou":       results["per_class"]["iou"].tolist(),
        }
    }
    report_path = os.path.join(log_dir, f"rf_evaluation_report_{run_timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"[INFO] Evaluation report saved to: {report_path}")

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("INFERENCE PERFORMANCE\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Test Images     : {len(test_ds)}\n")
        f.write(f"Total Time            : {inference_time:.4f} sec\n")
        f.write(f"Time / Image          : {time_per_image:.6f} sec\n")
        if time_per_image > 0:
            f.write(f"Images / Second       : {1.0 / time_per_image:.2f}\n")

    print(f"\n{'='*60}")
    print(f"  Final Results — {dataset_name} Random Forest")
    print(f"{'='*60}")
    print(f"  Accuracy : {results['overall']['accuracy']:.4f}")
    print(f"  Mean IoU : {results['overall']['iou']:.4f}")
    print(f"  Mean F1  : {results['overall']['f1']:.4f}")
    print(f"  Time/Img : {time_per_image:.4f} sec")
    print(f"  Report   : {report_path}")
    print(f"  CM       : {cm_path}")
    print(f"  Model    : {ckpt_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    with open("configs/dataset.yaml", "r") as f:
        dataset_cfg = yaml.safe_load(f)
    with open("configs/rf.yaml", "r") as f:
        rf_cfg = yaml.safe_load(f)

    try:
        train_and_eval("MARIDA", MARIDADataset, dataset_cfg["marida"], rf_cfg)
    except Exception as e:
        print(f"Failed to run MARIDA: {e}")
        import traceback; traceback.print_exc()

    try:
        train_and_eval("MADOS", MADOSDataset, dataset_cfg["mados"], rf_cfg)
    except Exception as e:
        print(f"Failed to run MADOS: {e}")
        import traceback; traceback.print_exc()
