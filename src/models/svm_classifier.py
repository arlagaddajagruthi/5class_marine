import os
import sys
import time
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier

# Feature extraction matching 5class marine repository standard
def extract_pixel_features_21(img_tensor):
    """
    Extract 21 spectral features per pixel from an image tensor (C, H, W).
    Works for both C=8 (MADOS) and C=11 (MARIDA) channels.
    Returns array of shape (H*W, 21).
    """
    if hasattr(img_tensor, 'numpy'):
        img = img_tensor.numpy()
    else:
        img = np.array(img_tensor, dtype=np.float32)
        
    C, H, W = img.shape
    # Flatten spatial dimensions
    pixels = img.reshape(C, -1).T  # (N_pixels, C)
    
    # 1. Base channels (first min(C, 8) or zero-padded)
    if C >= 8:
        base_feats = pixels[:, :8]  # 8 features
    else:
        pad = np.zeros((pixels.shape[0], 8 - C), dtype=np.float32)
        base_feats = np.hstack([pixels, pad])
        
    # Key band picks for ratios/indices
    b_blue = pixels[:, 0]
    b_green = pixels[:, 1] if C > 1 else pixels[:, 0]
    b_red = pixels[:, 2] if C > 2 else pixels[:, 0]
    b_nir = pixels[:, min(6, C - 1)]
    b_swir = pixels[:, min(7, C - 1)]
    
    eps = 1e-6
    # 2. Normalized difference indices (5 features)
    ndvi = (b_nir - b_red) / (b_nir + b_red + eps)
    ndwi = (b_green - b_nir) / (b_green + b_nir + eps)
    fai = b_nir - (b_red + (b_swir - b_red) * 0.5)
    fdi = b_nir - (b_red + (b_swir - b_red) * 0.7)
    ndmi = (b_nir - b_swir) / (b_nir + b_swir + eps)
    
    indices = np.column_stack([ndvi, ndwi, fai, fdi, ndmi])  # 5 features
    
    # 3. Spectral Ratios (4 features)
    r1 = b_red / (b_green + eps)
    r2 = b_nir / (b_red + eps)
    r3 = b_nir / (b_green + eps)
    r4 = b_swir / (b_nir + eps)
    ratios = np.column_stack([r1, r2, r3, r4])  # 4 features
    
    # 4. Statistical summaries across channels per pixel (4 features)
    mean_spec = np.mean(pixels, axis=1)
    std_spec = np.std(pixels, axis=1)
    max_spec = np.max(pixels, axis=1)
    norm_spec = np.linalg.norm(pixels, axis=1)
    stats = np.column_stack([mean_spec, std_spec, max_spec, norm_spec])  # 4 features
    
    # Total: 8 + 5 + 4 + 4 = 21 features
    features = np.hstack([base_feats, indices, ratios, stats])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


class MarineSVMClassifier:
    """
    Support Vector Machine Classifier Pipeline for 5-Class Marine Debris & Water Segmentation.
    Includes built-in feature standardisation (StandardScaler) and supports standard RBF/Linear SVC
    as well as fast SGD/Linear-SVM options for large pixel datasets.
    """
    def __init__(
        self,
        kernel='rbf',
        C=1.0,
        gamma='scale',
        max_iter=2000,
        probability=False,
        num_classes=5,
        random_state=42,
        use_sgd_fast=False,
        class_weight='balanced'
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.probability = probability
        self.num_classes = num_classes
        self.random_state = random_state
        self.use_sgd_fast = use_sgd_fast
        self.class_weight = class_weight
        
        self.scaler = StandardScaler()
        
        if self.use_sgd_fast:
            # High-speed linear SVM using Stochastic Gradient Descent for large datasets
            self.clf = SGDClassifier(
                loss='hinge' if not probability else 'log_loss',
                penalty='l2',
                alpha=1.0 / (self.C * 1000.0),
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_jobs=-1
            )
        elif self.kernel == 'linear' and not self.probability:
            self.clf = LinearSVC(
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
                dual='auto'
            )
        else:
            self.clf = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                max_iter=self.max_iter,
                probability=self.probability,
                random_state=self.random_state,
                class_weight=self.class_weight
            )
            
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', self.clf)
        ])

    def fit(self, X, y):
        start_time = time.time()
        self.pipeline.fit(X, y)
        train_time = time.time() - start_time
        return train_time

    def predict(self, X):
        start_time = time.time()
        preds = self.pipeline.predict(X)
        inf_time = time.time() - start_time
        return preds, inf_time

    def predict_proba(self, X):
        if hasattr(self.pipeline.named_steps['classifier'], "predict_proba"):
            start_time = time.time()
            probs = self.pipeline.predict_proba(X)
            inf_time = time.time() - start_time
            return probs, inf_time
        elif hasattr(self.pipeline.named_steps['classifier'], "decision_function"):
            start_time = time.time()
            decision = self.pipeline.decision_function(X)
            if decision.ndim == 1:
                decision = np.column_stack([-decision, decision])
            # Softmax conversion
            exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            probs = exp_d / np.sum(exp_d, axis=1, keepdims=True)
            inf_time = time.time() - start_time
            return probs, inf_time
        else:
            raise NotImplementedError("Probability estimates not available for this SVM configuration.")

    def save(self, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.pipeline, filepath)

    def load(self, filepath):
        self.pipeline = joblib.load(filepath)
        self.scaler = self.pipeline.named_steps.get('scaler', StandardScaler())
        self.clf = self.pipeline.named_steps.get('classifier', None)


def plot_svm_dual_confusion_matrix(cm, class_names, save_path, title_prefix=""):
    """
    Plots and saves a high-resolution 2-panel confusion matrix heatmap:
    Panel 1: Raw Pixel Counts (Blues)
    Panel 2: Recall-Normalized Percentages % (YlOrRd)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    cm_norm = (cm / row_sums) * 100.0

    short_names = [name.replace(' ', '\n').replace('/', '/\n') for name in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    main_title = f"{title_prefix} - Support Vector Machine (SVM) Confusion Matrix" if title_prefix else "SVM Confusion Matrix"
    fig.suptitle(main_title, fontsize=15, fontweight='bold', y=0.98)

    sns.heatmap(
        cm, ax=axes[0], annot=True, fmt='d', cmap='Blues',
        xticklabels=short_names, yticklabels=short_names,
        linewidths=0.5, cbar=True
    )
    axes[0].set_title('Raw Pixel Counts', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Predicted Class', fontsize=11)
    axes[0].set_ylabel('Actual Class', fontsize=11)
    axes[0].tick_params(axis='x', rotation=25)

    sns.heatmap(
        cm_norm, ax=axes[1], annot=True, fmt='.2f', cmap='YlOrRd',
        xticklabels=short_names, yticklabels=short_names,
        linewidths=0.5, cbar=True, vmin=0, vmax=100
    )
    axes[1].set_title('Recall-Normalized (%)', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Predicted Class', fontsize=11)
    axes[1].set_ylabel('Actual Class', fontsize=11)
    axes[1].tick_params(axis='x', rotation=25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def sample_dataset_features(dataset, max_pixels_per_class=50000, random_seed=42):
    """
    Extracts 21 spectral features across patches in a dataset with stratified subsampling.
    """
    np.random.seed(random_seed)
    X_list = []
    y_list = []
    
    total_patches = len(dataset)
    print(f"Extracting features from {total_patches} patches...")
    
    for idx in range(total_patches):
        img_tensor, mask_tensor = dataset[idx]
        feats = extract_pixel_features_21(img_tensor)  # (H*W, 21)
        if hasattr(mask_tensor, 'numpy'):
            mask = mask_tensor.numpy().flatten()
        else:
            mask = np.array(mask_tensor).flatten()
        
        # Valid pixels (0 to 4)
        valid_mask = (mask >= 0) & (mask < 5)
        if not np.any(valid_mask):
            continue
            
        feats_valid = feats[valid_mask]
        mask_valid = mask[valid_mask]
        
        X_list.append(feats_valid)
        y_list.append(mask_valid)
        
    if len(X_list) == 0:
        return np.empty((0, 21), dtype=np.float32), np.empty((0,), dtype=np.int64)
        
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    
    # Balance / subsample per class
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


def run_svm_pipeline(dataset_name, dataset_dir, max_train_pixels=250000, max_test_pixels=250000, kernel='linear', C=1.0, max_iter=3000):
    """
    End-to-end training, evaluation, and artifact saving pipeline for SVM.
    """
    import csv
    import json
    from datetime import datetime

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from src.datasets.mados import MADOSDataset
    from src.datasets.marida import MARIDADataset
    from src.metrics.metrics import calculate_metrics, plot_confusion_matrix, save_metrics_table

    CLASS_NAMES = ["Marine Debris", "Sargassum/Veg", "Natural Phenom/Foam", "Ship/Infrastructure", "Water/Other"]
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_short = "MADOS" if "MADOS" in dataset_name.upper() else "MARIDA"

    print(f"\n{'='*60}")
    print(f"  Support Vector Machine (SVM) Pipeline — {dataset_name}")
    print(f"{'='*60}\n")
    
    if dataset_short == "MADOS":
        train_ds = MADOSDataset(dataset_dir, split="train")
        val_ds = MADOSDataset(dataset_dir, split="val")
        test_ds = MADOSDataset(dataset_dir, split="test")
    else:
        train_ds = MARIDADataset(dataset_dir, split="train")
        val_ds = MARIDADataset(dataset_dir, split="val")
        test_ds = MARIDADataset(dataset_dir, split="test")
        
    print(f"Loading {dataset_name} Train dataset ({len(train_ds)} patches)...")
    X_train, y_train = sample_dataset_features(train_ds, max_pixels_per_class=max_train_pixels // 5)
    print(f"Training SVM on {len(y_train):,} pixels with {X_train.shape[1]} features...")
    
    unique, counts = np.unique(y_train, return_counts=True)
    print("[INFO] Class distribution in training subset:")
    for cls_id, cnt in zip(unique, counts):
        name = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else f"Class {cls_id}"
        print(f"       {name}: {cnt:,} ({100.0 * cnt / len(y_train):.2f}%)")

    print(f"\nLoading {dataset_name} Test dataset ({len(test_ds)} patches)...")
    X_test, y_test = sample_dataset_features(test_ds, max_pixels_per_class=max_test_pixels // 5)
    print(f"Evaluating SVM on {len(y_test):,} test pixels...")

    # Initialize SVM Model
    svm_clf = MarineSVMClassifier(
        kernel=kernel,
        C=C,
        max_iter=max_iter,
        num_classes=5,
        random_state=42,
        class_weight='balanced'
    )
    
    print("\nTraining SVM model...")
    train_time = svm_clf.fit(X_train, y_train)
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Train set accuracy
    train_preds, _ = svm_clf.predict(X_train)
    from sklearn.metrics import accuracy_score
    train_accuracy = accuracy_score(y_train, train_preds)
    print(f"Train Accuracy: {train_accuracy * 100.0:.2f}%")

    # Evaluation on Test set
    print("Evaluating test set...")
    y_pred, inf_time = svm_clf.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, num_classes=5)
    
    # Validation set evaluation
    print(f"Loading {dataset_name} Validation dataset ({len(val_ds)} patches)...")
    X_val, y_val = sample_dataset_features(val_ds, max_pixels_per_class=max_test_pixels // 5)
    val_preds, _ = svm_clf.predict(X_val)
    val_metrics = calculate_metrics(y_val, val_preds, num_classes=5)

    # Inference statistics
    num_test_pixels = len(y_test)
    latency_per_1m_pixels = (inf_time / num_test_pixels) * 1e6 if num_test_pixels > 0 else 0.0
    latency_per_patch_ms = (inf_time / len(test_ds)) * 1000.0 if len(test_ds) > 0 else 0.0
    throughput_pixels_sec = num_test_pixels / inf_time if inf_time > 0 else 0.0
    
    print(f"\nMetrics for {dataset_name}:")
    print(f"  Precision    : {metrics['overall']['precision']:.6f}")
    print(f"  Recall       : {metrics['overall']['recall']:.6f}")
    print(f"  F1 Score     : {metrics['overall']['f1']:.6f}")
    print(f"  IoU (Jaccard): {metrics['overall']['iou']:.6f}")
    print(f"  Accuracy     : {metrics['overall']['accuracy']:.6f}")
    print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Define and create all output directories
    out_dir = os.path.join(repo_root, "outputs")
    log_dir = os.path.join(out_dir, "logs", dataset_short)
    cm_dir = os.path.join(out_dir, "confusion_matrices")
    model_info_dir = os.path.join(out_dir, "model_info")
    ckpt_dir = os.path.join(repo_root, "checkpoints", "svm", dataset_short.lower())
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    os.makedirs(model_info_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Save Model Artifacts
    model_path = os.path.join(out_dir, f"{dataset_name.lower()}_svm_model.joblib")
    svm_clf.save(model_path)
    ckpt_path = os.path.join(ckpt_dir, f"svm_model_{run_timestamp}.joblib")
    svm_clf.save(ckpt_path)
    model_size_mb = os.path.getsize(model_path) / (1024.0 * 1024.0)
    
    # Save formatted results text in repo root (matching repo convention)
    results_str = f"Loading {dataset_name} Train dataset...\n"
    results_str += f"Loading {dataset_name} Test dataset...\n"
    results_str += f"Extracting features for train set...\n"
    results_str += f"Extracting features for test set...\n"
    results_str += f"Training SVM on {len(y_train)} pixels with {X_train.shape[1]} features...\n"
    results_str += f"Evaluating...\n\n"
    results_str += f"Metrics:\n"
    results_str += f"Precision: {metrics['overall']['precision']:.16f}\n"
    results_str += f"Recall: {metrics['overall']['recall']:.16f}\n"
    results_str += f"F1 Score: {metrics['overall']['f1']:.16f}\n"
    results_str += f"IoU (Jaccard): {metrics['overall']['iou']:.16f}\n"
    results_str += f"Accuracy: {metrics['overall']['accuracy']:.16f}\n\n"
    results_str += f"Confusion Matrix:\n{metrics['confusion_matrix']}\n\n"
    results_str += f"Saving model to {os.path.basename(model_path)}...\n"
    
    txt_filename = f"{dataset_short.lower()}_svm_results.txt"
    txt_filepath = os.path.join(repo_root, txt_filename)
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write(results_str)
        
    # Save per-class metrics tables (.csv and .md)
    table_base_path = os.path.join(log_dir, f"{dataset_name}_svm_metrics")
    save_metrics_table(metrics, CLASS_NAMES, table_base_path)
    table_ts_path = os.path.join(log_dir, f"svm_per_class_metrics_{run_timestamp}")
    save_metrics_table(metrics, CLASS_NAMES, table_ts_path)
    
    # Save Confusion Matrices:
    # 1. Single-panel confusion matrix in log folder
    cm_plot_path = os.path.join(log_dir, f"{dataset_name}_svm_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, cm_plot_path)

    # 2. Single-panel confusion matrix in confusion_matrices folder
    cm_dir_plot_path = os.path.join(cm_dir, f"{dataset_name}_svm_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, cm_dir_plot_path)

    # 3. Dual-panel High-Resolution Heatmap (Raw + Recall-Normalized %)
    cm_heatmap_path = os.path.join(cm_dir, f"{dataset_name}_SVM_confusion_heatmap.png")
    plot_svm_dual_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, cm_heatmap_path, title_prefix=dataset_name)

    # Save CSV Training Log
    log_path = os.path.join(log_dir, f"svm_training_log_{run_timestamp}.csv")
    with open(log_path, "w", newline="") as lf:
        writer = csv.writer(lf)
        writer.writerow([
            "model", "train_samples", "train_time_sec", "train_accuracy",
            "val_miou", "val_f1", "val_recall", "val_precision",
            "test_miou", "test_f1", "test_recall", "test_precision",
            "test_accuracy", "checkpoint_path"
        ])
        writer.writerow([
            "SVM", len(y_train), f"{train_time:.4f}", f"{train_accuracy:.6f}",
            f"{val_metrics['overall']['iou']:.6f}", f"{val_metrics['overall']['f1']:.6f}",
            f"{val_metrics['overall']['recall']:.6f}", f"{val_metrics['overall']['precision']:.6f}",
            f"{metrics['overall']['iou']:.6f}", f"{metrics['overall']['f1']:.6f}",
            f"{metrics['overall']['recall']:.6f}", f"{metrics['overall']['precision']:.6f}",
            f"{metrics['overall']['accuracy']:.6f}", ckpt_path
        ])

    # Save JSON Evaluation Report
    report_dict = {
        "dataset": dataset_name,
        "model": "SupportVectorMachine",
        "run_timestamp": run_timestamp,
        "hyperparameters": {
            "kernel": kernel,
            "C": C,
            "max_iter": max_iter,
            "class_weight": "balanced",
            "max_train_pixels": max_train_pixels
        },
        "training_samples": int(len(y_train)),
        "training_time_sec": train_time,
        "train_accuracy": train_accuracy,
        "val_metrics": val_metrics["overall"],
        "checkpoint_path": ckpt_path,
        "inference_time_total_sec": inf_time,
        "latency_per_patch_ms": latency_per_patch_ms,
        "latency_per_1m_pixels_sec": latency_per_1m_pixels,
        "throughput_pixels_sec": throughput_pixels_sec,
        "overall_metrics": metrics["overall"],
        "per_class_metrics": {
            "precision": metrics["per_class"]["precision"].tolist(),
            "recall": metrics["per_class"]["recall"].tolist(),
            "f1": metrics["per_class"]["f1"].tolist(),
            "iou": metrics["per_class"]["iou"].tolist(),
        }
    }
    report_path = os.path.join(log_dir, f"svm_evaluation_report_{run_timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)

    # Save Model Summary
    summary_path = os.path.join(model_info_dir, f"{dataset_short}_svm_model_summary_{run_timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"SUPPORT VECTOR MACHINE (SVM) MODEL SUMMARY — {dataset_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp             : {run_timestamp}\n")
        f.write(f"Kernel                : {kernel}\n")
        f.write(f"Regularization (C)    : {C}\n")
        f.write(f"Max Iterations        : {max_iter}\n")
        f.write(f"Class Weight          : balanced\n")
        f.write(f"Number of Features    : {X_train.shape[1]}\n")
        f.write(f"Training Pixels       : {len(y_train):,}\n")
        f.write(f"Training Time         : {train_time:.4f} sec\n")
        f.write(f"Model File Size       : {model_size_mb:.4f} MB\n")
        f.write(f"Test Pixels Evaluated : {len(y_test):,}\n")
        f.write(f"Total Inference Time  : {inf_time:.4f} sec\n")
        f.write(f"Latency / Patch       : {latency_per_patch_ms:.4f} ms\n")
        f.write(f"Throughput            : {throughput_pixels_sec:,.2f} pixels/sec\n")
        f.write(f"Overall Accuracy      : {metrics['overall']['accuracy']*100:.2f}%\n")
        f.write(f"Macro F1 Score        : {metrics['overall']['f1']*100:.2f}%\n")
        f.write(f"Macro IoU (mIoU)      : {metrics['overall']['iou']*100:.2f}%\n")
        f.write("=" * 80 + "\n")

    summary = {
        "dataset": dataset_name,
        "model": "SVM",
        "train_pixels": len(y_train),
        "test_pixels": len(y_test),
        "kernel": kernel,
        "C": C,
        "model_size_mb": model_size_mb,
        "train_time_sec": train_time,
        "inference_time_sec": inf_time,
        "latency_per_patch_ms": latency_per_patch_ms,
        "latency_per_1m_pixels_sec": latency_per_1m_pixels,
        "throughput_pixels_sec": throughput_pixels_sec,
        "metrics": metrics['overall']
    }
    return summary, txt_filepath


def resolve_dataset_paths(repo_dir):
    """
    Finds MADOS and MARIDA dataset folders reliably across directory patterns.
    """
    mados_candidates = [
        os.path.join(repo_dir, "datasets", "MADOS_5Class"),
        os.path.join(repo_dir, "data", "MADOS_5Class"),
        os.path.join(repo_dir, "MADOS_5Class"),
        os.path.join(repo_dir, "MADOS_5Class", "MADOS_5Class")
    ]
    marida_candidates = [
        os.path.join(repo_dir, "datasets", "MARIDA_5Class"),
        os.path.join(repo_dir, "data", "MARIDA_5Class"),
        os.path.join(repo_dir, "MARIDA_5Class"),
        os.path.join(repo_dir, "MARIDA_5Class", "MARIDA_5Class")
    ]
    
    mados_path = None
    for p in mados_candidates:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "splits")):
            mados_path = os.path.abspath(p)
            break
            
    marida_path = None
    for p in marida_candidates:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "splits")):
            marida_path = os.path.abspath(p)
            break
            
    return mados_path, marida_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SVM Pipeline on Marine Datasets")
    parser.add_argument("--dataset", type=str, default="both", choices=["MADOS_5Class", "MARIDA_5Class", "both"])
    parser.add_argument("--kernel", type=str, default="linear", choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--max_pixels", type=int, default=250000)
    args = parser.parse_args()

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    mados_path, marida_path = resolve_dataset_paths(repo_dir)

    print(f"MADOS Path : {mados_path}")
    print(f"MARIDA Path: {marida_path}")

    if args.dataset in ["MADOS_5Class", "both"] and mados_path:
        run_svm_pipeline("MADOS_5Class", mados_path, max_train_pixels=args.max_pixels, kernel=args.kernel, C=args.C, max_iter=args.max_iter)

    if args.dataset in ["MARIDA_5Class", "both"] and marida_path:
        run_svm_pipeline("MARIDA_5Class", marida_path, max_train_pixels=args.max_pixels, kernel=args.kernel, C=args.C, max_iter=args.max_iter)
