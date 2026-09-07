"""
Run all 5 ML models (RF, SVM, XGBoost, AdaBoost, LightGBM) on MADOS & MARIDA
using ONLY raw spectral bands — no indices, ratios, or statistical summaries.

Outputs are saved to no_indices subdirectories:
  - Confusion matrices:  outputs/final_cms/no_indices/
  - Results text files:  no_indices_results/
  - Logs/tables:         outputs/logs/no_indices/MADOS/ and .../MARIDA/
  - Model files:         outputs/no_indices/
"""

import os
import sys
import time
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.datasets.mados import MADOSDataset
from src.datasets.marida import MARIDADataset
from src.metrics.metrics import calculate_metrics, save_metrics_table

CLASS_NAMES = ["Marine Debris", "Sargassum/Veg", "Natural Phenom/Foam", "Ship/Infrastructure", "Water/Other"]


# ---------------------------------------------------------------------------
# Feature extraction: raw bands only
# ---------------------------------------------------------------------------
def extract_raw_bands_only(img_tensor):
    """
    Extract raw spectral band values per pixel from an image tensor (C, H, W).
    Returns array of shape (H*W, C) — all C bands, no padding, no indices,
    no ratios, no statistical summaries.
    """
    if hasattr(img_tensor, 'numpy'):
        img = img_tensor.numpy()
    else:
        img = np.array(img_tensor, dtype=np.float32)

    C, H, W = img.shape
    pixels = img.reshape(C, -1).T  # (N_pixels, C)
    pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
    return pixels


def sample_dataset_raw_bands(dataset, max_pixels_per_class=50000, random_seed=42):
    """
    Extracts raw band features across all patches with stratified subsampling.
    """
    np.random.seed(random_seed)
    X_list = []
    y_list = []

    total_patches = len(dataset)
    print(f"Extracting raw band features from {total_patches} patches...")

    for idx in range(total_patches):
        img_tensor, mask_tensor = dataset[idx]
        feats = extract_raw_bands_only(img_tensor)  # (H*W, C)
        if hasattr(mask_tensor, 'numpy'):
            mask = mask_tensor.numpy().flatten()
        else:
            mask = np.array(mask_tensor).flatten()

        # Valid pixels (classes 0 to 4)
        valid_mask = (mask >= 0) & (mask < 5)
        if not np.any(valid_mask):
            continue

        feats_valid = feats[valid_mask]
        mask_valid = mask[valid_mask]

        X_list.append(feats_valid)
        y_list.append(mask_valid)

    if len(X_list) == 0:
        return np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.int64)

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


# ---------------------------------------------------------------------------
# Confusion matrix plotting (saves directly to no_indices dir)
# ---------------------------------------------------------------------------
def plot_confusion_matrix_no_indices(cm, class_names, save_dir, base_name):
    """
    Saves raw + normalized confusion matrix PNGs into the given directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Raw Counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{base_name} — Confusion Matrix (Raw)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_raw.png"), dpi=150)
    plt.close()

    # 2. Normalized
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    cm_norm = (cm / row_sums) * 100.0

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=100)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{base_name} — Confusion Matrix (Normalized %)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{base_name}_norm.png"), dpi=150)
    plt.close()

    print(f"  [CM] Saved to {save_dir}")


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------
def get_output_dirs(dataset_short):
    """Returns dict of output directories for a given dataset (MADOS or MARIDA)."""
    dirs = {
        "cm_dir": os.path.join(REPO_ROOT, "outputs", "final_cms", "no_indices"),
        "results_dir": os.path.join(REPO_ROOT, "no_indices_results"),
        "log_dir": os.path.join(REPO_ROOT, "outputs", "logs", "no_indices", dataset_short),
        "model_dir": os.path.join(REPO_ROOT, "outputs", "no_indices"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ---------------------------------------------------------------------------
# Generic evaluation + artifact saving
# ---------------------------------------------------------------------------
def evaluate_and_save(model_name, model_obj, X_train, y_train, X_test, y_test,
                      dataset_name, dataset_short, dirs, save_func, n_test_patches):
    """
    Train, evaluate, and save all artifacts for one model on one dataset.
    """
    print(f"\n  Training {model_name}...")
    train_time = model_obj.fit(X_train, y_train)
    print(f"  Training completed in {train_time:.2f} seconds.")

    print(f"  Evaluating {model_name} on test set...")
    y_pred, inf_time = model_obj.predict(X_test)

    metrics = calculate_metrics(y_test, y_pred, num_classes=5)

    num_test_pixels = len(y_test)
    latency_per_1m_pixels = (inf_time / num_test_pixels) * 1e6 if num_test_pixels > 0 else 0.0
    latency_per_patch_ms = (inf_time / n_test_patches) * 1000.0 if n_test_patches > 0 else 0.0
    throughput_pixels_sec = num_test_pixels / inf_time if inf_time > 0 else 0.0

    print(f"  Metrics for {dataset_name} [{model_name}]:")
    print(f"    Precision    : {metrics['overall']['precision']:.6f}")
    print(f"    Recall       : {metrics['overall']['recall']:.6f}")
    print(f"    F1 Score     : {metrics['overall']['f1']:.6f}")
    print(f"    IoU (Jaccard): {metrics['overall']['iou']:.6f}")
    print(f"    Accuracy     : {metrics['overall']['accuracy']:.6f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

    # --- Save model ---
    model_tag = model_name.lower().replace(" ", "_")
    save_func(model_obj, dirs["model_dir"], dataset_name, model_tag)

    # --- Save confusion matrix PNGs ---
    cm_base = f"{dataset_name}_{model_tag}_rawbands_confusion_matrix"
    plot_confusion_matrix_no_indices(metrics["confusion_matrix"], CLASS_NAMES, dirs["cm_dir"], cm_base)

    # --- Save metrics tables (CSV + MD) ---
    table_path = os.path.join(dirs["log_dir"], f"{dataset_name}_{model_tag}_rawbands_metrics")
    save_metrics_table(metrics, CLASS_NAMES, table_path)

    # --- Save results text file ---
    results_str = f"=== {model_name} — Raw Bands Only (No Indices) — {dataset_name} ===\n\n"
    results_str += f"Training {model_name} on {len(y_train)} pixels with {X_train.shape[1]} features (raw bands only)...\n"
    results_str += f"Training time: {train_time:.2f} seconds\n"
    results_str += f"Evaluating on {len(y_test)} test pixels...\n\n"
    results_str += f"Metrics:\n"
    results_str += f"Precision: {metrics['overall']['precision']:.16f}\n"
    results_str += f"Recall: {metrics['overall']['recall']:.16f}\n"
    results_str += f"F1 Score: {metrics['overall']['f1']:.16f}\n"
    results_str += f"IoU (Jaccard): {metrics['overall']['iou']:.16f}\n"
    results_str += f"Accuracy: {metrics['overall']['accuracy']:.16f}\n\n"
    results_str += f"Confusion Matrix:\n{metrics['confusion_matrix']}\n"

    txt_filename = f"{dataset_short.lower()}_{model_tag}_rawbands_results.txt"
    txt_filepath = os.path.join(dirs["results_dir"], txt_filename)
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write(results_str)

    summary = {
        "dataset": dataset_name,
        "model": model_name,
        "num_features": int(X_train.shape[1]),
        "feature_type": "raw_bands_only",
        "train_pixels": int(len(y_train)),
        "test_pixels": int(len(y_test)),
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
    return summary


# ---------------------------------------------------------------------------
# Model save helpers
# ---------------------------------------------------------------------------
def save_rf_model(model_obj, model_dir, dataset_name, model_tag):
    path = os.path.join(model_dir, f"{dataset_name.lower()}_{model_tag}_rawbands_model.joblib")
    joblib.dump(model_obj.model, path)
    print(f"  [MODEL] Saved -> {path}")


def save_svm_model(model_obj, model_dir, dataset_name, model_tag):
    path = os.path.join(model_dir, f"{dataset_name.lower()}_{model_tag}_rawbands_model.joblib")
    joblib.dump(model_obj.pipeline, path)
    print(f"  [MODEL] Saved -> {path}")


def save_xgb_model(model_obj, model_dir, dataset_name, model_tag):
    path = os.path.join(model_dir, f"{dataset_name.lower()}_{model_tag}_rawbands_model.json")
    model_obj.model.save_model(path)
    print(f"  [MODEL] Saved -> {path}")


def save_adaboost_model(model_obj, model_dir, dataset_name, model_tag):
    path = os.path.join(model_dir, f"{dataset_name.lower()}_{model_tag}_rawbands_model.joblib")
    joblib.dump(model_obj.model, path)
    print(f"  [MODEL] Saved -> {path}")


def save_lgbm_model(model_obj, model_dir, dataset_name, model_tag):
    path = os.path.join(model_dir, f"{dataset_name.lower()}_{model_tag}_rawbands_model.joblib")
    joblib.dump(model_obj.model, path)
    print(f"  [MODEL] Saved -> {path}")


# ---------------------------------------------------------------------------
# Model factory wrappers (thin wrappers with .fit() and .predict() returning times)
# ---------------------------------------------------------------------------
class RFWrapper:
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=None, n_jobs=-1, random_state=42
        )

    def fit(self, X, y):
        t0 = time.time()
        self.model.fit(X, y)
        return time.time() - t0

    def predict(self, X):
        t0 = time.time()
        preds = self.model.predict(X)
        return preds, time.time() - t0


class SVMWrapper:
    def __init__(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LinearSVC(
                C=1.0, max_iter=3000, random_state=42,
                class_weight='balanced', dual='auto'
            ))
        ])

    def fit(self, X, y):
        t0 = time.time()
        self.pipeline.fit(X, y)
        return time.time() - t0

    def predict(self, X):
        t0 = time.time()
        preds = self.pipeline.predict(X)
        return preds, time.time() - t0


class XGBWrapper:
    def __init__(self):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            objective='multi:softprob', num_class=5,
            tree_method='hist', random_state=42, n_jobs=-1
        )

    def fit(self, X, y):
        t0 = time.time()
        self.model.fit(X, y)
        return time.time() - t0

    def predict(self, X):
        t0 = time.time()
        preds = self.model.predict(X)
        return preds, time.time() - t0


class AdaBoostWrapper:
    def __init__(self):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        import inspect

        base_est = DecisionTreeClassifier(max_depth=3, random_state=42)

        kwargs = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42,
        }

        sig = inspect.signature(AdaBoostClassifier)
        if 'algorithm' in sig.parameters:
            kwargs['algorithm'] = 'SAMME'
        if 'estimator' in sig.parameters:
            kwargs['estimator'] = base_est
        elif 'base_estimator' in sig.parameters:
            kwargs['base_estimator'] = base_est

        self.model = AdaBoostClassifier(**kwargs)

    def fit(self, X, y):
        t0 = time.time()
        self.model.fit(X, y)
        return time.time() - t0

    def predict(self, X):
        t0 = time.time()
        preds = self.model.predict(X)
        return preds, time.time() - t0


class LGBMWrapper:
    def __init__(self):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, num_leaves=31,
            learning_rate=0.1, objective='multiclass', num_class=5,
            random_state=42, n_jobs=-1, verbose=-1
        )

    def fit(self, X, y):
        t0 = time.time()
        self.model.fit(X, y)
        return time.time() - t0

    def predict(self, X):
        t0 = time.time()
        preds = self.model.predict(X)
        return preds, time.time() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_all_models_on_dataset(dataset_name, dataset_dir):
    """Run all 5 models on one dataset using raw bands only."""
    dataset_short = "MADOS" if "MADOS" in dataset_name.upper() else "MARIDA"

    print(f"\n{'='*65}")
    print(f"  RAW BANDS ONLY — {dataset_name}")
    print(f"{'='*65}")

    # Load dataset
    if dataset_short == "MADOS":
        train_ds = MADOSDataset(dataset_dir, split="train")
        test_ds = MADOSDataset(dataset_dir, split="test")
    else:
        train_ds = MARIDADataset(dataset_dir, split="train")
        test_ds = MARIDADataset(dataset_dir, split="test")

    max_train_px = 250000

    print(f"Loading {dataset_name} Train dataset ({len(train_ds)} patches)...")
    X_train, y_train = sample_dataset_raw_bands(train_ds, max_pixels_per_class=max_train_px // 5)
    print(f"  Train: {len(y_train):,} pixels, {X_train.shape[1]} raw band features")

    print(f"Loading {dataset_name} Test dataset ({len(test_ds)} patches)...")
    X_test, y_test = sample_dataset_raw_bands(test_ds, max_pixels_per_class=max_train_px // 5)
    print(f"  Test:  {len(y_test):,} pixels, {X_test.shape[1]} raw band features")

    dirs = get_output_dirs(dataset_short)
    n_test_patches = len(test_ds)

    # Model definitions: (name, wrapper_class, save_function)
    models = [
        ("Random Forest", RFWrapper, save_rf_model),
        ("SVM", SVMWrapper, save_svm_model),
        ("XGBoost", XGBWrapper, save_xgb_model),
        ("AdaBoost", AdaBoostWrapper, save_adaboost_model),
        ("LightGBM", LGBMWrapper, save_lgbm_model),
    ]

    summaries = {}
    for model_name, WrapperCls, save_fn in models:
        print(f"\n{'─'*50}")
        print(f"  [{dataset_short}] {model_name}")
        print(f"{'─'*50}")
        model_obj = WrapperCls()
        summary = evaluate_and_save(
            model_name, model_obj,
            X_train, y_train, X_test, y_test,
            dataset_name, dataset_short, dirs, save_fn, n_test_patches
        )
        summaries[model_name] = summary

    return summaries


def resolve_dataset_paths():
    """Find MADOS and MARIDA dataset directories."""
    mados_path = os.path.join(REPO_ROOT, "data", "MADOS_5Class")
    if not os.path.exists(mados_path):
        mados_path = os.path.join(REPO_ROOT, "MADOS_5Class")

    marida_path = os.path.join(REPO_ROOT, "data", "MARIDA_5Class")
    if not os.path.exists(marida_path):
        marida_path = os.path.join(REPO_ROOT, "MARIDA_5Class")

    return mados_path, marida_path


if __name__ == "__main__":
    print("=" * 65)
    print("  RAW BANDS ONLY — ALL 5 ML MODELS — MADOS & MARIDA")
    print("  (No indices, no ratios, no statistical summaries)")
    print("=" * 65)

    mados_path, marida_path = resolve_dataset_paths()

    all_results = {}

    if os.path.exists(mados_path):
        mados_summaries = run_all_models_on_dataset("MADOS_5Class", mados_path)
        all_results["MADOS"] = mados_summaries
    else:
        print(f"[ERROR] MADOS dataset not found at {mados_path}")

    if os.path.exists(marida_path):
        marida_summaries = run_all_models_on_dataset("MARIDA_5Class", marida_path)
        all_results["MARIDA"] = marida_summaries
    else:
        print(f"[ERROR] MARIDA dataset not found at {marida_path}")

    # Save combined JSON summary
    summary_dir = os.path.join(REPO_ROOT, "no_indices_results")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "all_models_rawbands_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*65}")
    print("  FINAL SUMMARY — RAW BANDS ONLY")
    print(f"{'='*65}")
    print(json.dumps(all_results, indent=2))
    print(f"\n[INFO] Combined summary saved to {summary_path}")
    print("Done!")
