import os
import sys
import time
import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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


class MarineAdaBoostClassifier:
    """
    AdaBoost Classifier Pipeline for 5-Class Marine Debris & Aquatic Target Classification.
    Uses multi-class decision trees as weak learners with boosting iterations.
    """
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        algorithm='SAMME',
        num_classes=5,
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.algorithm = algorithm
        self.num_classes = num_classes
        self.random_state = random_state
        
        self.base_estimator = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        # Scikit-learn compatibility: `estimator` vs `base_estimator` and `algorithm` removal (>=1.6)
        import inspect
        sig = inspect.signature(AdaBoostClassifier)
        
        kwargs = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state
        }
        
        if 'algorithm' in sig.parameters:
            kwargs['algorithm'] = self.algorithm
            
        if 'estimator' in sig.parameters:
            kwargs['estimator'] = self.base_estimator
        elif 'base_estimator' in sig.parameters:
            kwargs['base_estimator'] = self.base_estimator
            
        self.model = AdaBoostClassifier(**kwargs)

    def fit(self, X, y):
        start_time = time.time()
        self.model.fit(X, y)
        train_time = time.time() - start_time
        return train_time

    def predict(self, X):
        start_time = time.time()
        preds = self.model.predict(X)
        inf_time = time.time() - start_time
        return preds, inf_time

    def predict_proba(self, X):
        start_time = time.time()
        probs = self.model.predict_proba(X)
        inf_time = time.time() - start_time
        return probs, inf_time

    def save(self, filepath):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath):
        self.model = joblib.load(filepath)


def sample_dataset_features(dataset, max_pixels_per_class=50000, random_seed=42):
    """
    Extracts 21 spectral features across all patches in a dataset with stratified subsampling.
    """
    np.random.seed(random_seed)
    X_list = []
    y_list = []
    
    total_patches = len(dataset)
    print(f"Extracting features from {total_patches} patches...")
    
    for idx in range(total_patches):
        img_tensor, mask_tensor = dataset[idx]
        feats = extract_pixel_features_21(img_tensor)  # (H*W, 21)
        mask = mask_tensor.numpy().flatten()
        
        # Valid pixels (0 to 4)
        valid_mask = (mask >= 0) & (mask < 5)
        if not np.any(valid_mask):
            continue
            
        feats_valid = feats[valid_mask]
        mask_valid = mask[valid_mask]
        
        X_list.append(feats_valid)
        y_list.append(mask_valid)
        
    if len(X_list) == 0:
        return np.empty((0, 21)), np.empty((0,))
        
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    
    # Subsample if necessary
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


def run_adaboost_pipeline(dataset_name, dataset_dir, max_train_pixels=250000, n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    End-to-end training, evaluation, and artifact saving pipeline for AdaBoost.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from src.datasets.mados import MADOSDataset
    from src.datasets.marida import MARIDADataset
    from src.metrics.metrics import calculate_metrics, plot_confusion_matrix, save_metrics_table

    CLASS_NAMES = ["Marine Debris", "Sargassum/Veg", "Natural Phenom/Foam", "Ship/Infrastructure", "Water/Other"]

    print(f"\n=======================================================")
    print(f" Running AdaBoost Pipeline on {dataset_name} Dataset")
    print(f"=======================================================\n")
    
    if dataset_name.upper().startswith("MADOS"):
        train_ds = MADOSDataset(dataset_dir, split="train")
        test_ds = MADOSDataset(dataset_dir, split="test")
    else:
        train_ds = MARIDADataset(dataset_dir, split="train")
        test_ds = MARIDADataset(dataset_dir, split="test")
        
    print(f"Loading {dataset_name} Train dataset ({len(train_ds)} patches)...")
    X_train, y_train = sample_dataset_features(train_ds, max_pixels_per_class=max_train_pixels // 5)
    print(f"Training AdaBoost on {len(y_train)} pixels with {X_train.shape[1]} features...")
    
    print(f"Loading {dataset_name} Test dataset ({len(test_ds)} patches)...")
    X_test, y_test = sample_dataset_features(test_ds, max_pixels_per_class=max_train_pixels // 5)
    
    # Initialize AdaBoost Model
    ada_clf = MarineAdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_classes=5,
        random_state=42
    )
    
    train_time = ada_clf.fit(X_train, y_train)
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Evaluation
    print("Evaluating test set...")
    y_pred, inf_time = ada_clf.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred, num_classes=5)
    
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
    
    # Output directories
    out_dir = os.path.join(repo_root, "outputs")
    log_dir = os.path.join(out_dir, "logs", dataset_name.split("_")[0])
    os.makedirs(log_dir, exist_ok=True)
    
    model_path = os.path.join(out_dir, f"{dataset_name.lower()}_adaboost_model.joblib")
    ada_clf.save(model_path)
    model_size_mb = os.path.getsize(model_path) / (1024.0 * 1024.0)
    
    # Results text format matching repository benchmark logs
    results_str = f"Loading {dataset_name} Train dataset...\n"
    results_str += f"Loading {dataset_name} Test dataset...\n"
    results_str += f"Extracting features for train set...\n"
    results_str += f"Extracting features for test set...\n"
    results_str += f"Training AdaBoost on {len(y_train)} pixels with {X_train.shape[1]} features...\n"
    results_str += f"Evaluating...\n\n"
    results_str += f"Metrics:\n"
    results_str += f"Precision: {metrics['overall']['precision']:.16f}\n"
    results_str += f"Recall: {metrics['overall']['recall']:.16f}\n"
    results_str += f"F1 Score: {metrics['overall']['f1']:.16f}\n"
    results_str += f"IoU (Jaccard): {metrics['overall']['iou']:.16f}\n"
    results_str += f"Accuracy: {metrics['overall']['accuracy']:.16f}\n\n"
    results_str += f"Confusion Matrix:\n{metrics['confusion_matrix']}\n\n"
    results_str += f"Saving model to {os.path.basename(model_path)}...\n"
    
    txt_filename = f"{dataset_name.lower()[:5]}_adaboost_results.txt"
    txt_filepath = os.path.join(repo_root, txt_filename)
    with open(txt_filepath, "w", encoding="utf-8") as f:
        f.write(results_str)
        
    table_base_path = os.path.join(log_dir, f"{dataset_name}_adaboost_metrics")
    save_metrics_table(metrics, CLASS_NAMES, table_base_path)
    
    cm_plot_path = os.path.join(log_dir, f"{dataset_name}_adaboost_confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES, cm_plot_path)
    
    summary = {
        "dataset": dataset_name,
        "model": "AdaBoost",
        "train_pixels": len(y_train),
        "test_pixels": len(y_test),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "model_size_mb": model_size_mb,
        "train_time_sec": train_time,
        "inference_time_sec": inf_time,
        "latency_per_patch_ms": latency_per_patch_ms,
        "latency_per_1m_pixels_sec": latency_per_1m_pixels,
        "throughput_pixels_sec": throughput_pixels_sec,
        "metrics": metrics['overall']
    }
    return summary, txt_filepath


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run AdaBoost Pipeline on Marine Datasets")
    parser.add_argument("--dataset", type=str, default="both", choices=["MADOS_5Class", "MARIDA_5Class", "both"])
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--max_pixels", type=int, default=250000)
    args = parser.parse_args()

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    mados_path = os.path.join(repo_dir, "data", "MADOS_5Class")
    if not os.path.exists(mados_path):
        mados_path = os.path.join(repo_dir, "MADOS_5Class")
        
    marida_path = os.path.join(repo_dir, "data", "MARIDA_5Class")
    if not os.path.exists(marida_path):
        marida_path = os.path.join(repo_dir, "MARIDA_5Class")

    if args.dataset in ["MADOS_5Class", "both"] and os.path.exists(mados_path):
        run_adaboost_pipeline("MADOS_5Class", mados_path, max_train_pixels=args.max_pixels, n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth)

    if args.dataset in ["MARIDA_5Class", "both"] and os.path.exists(marida_path):
        run_adaboost_pipeline("MARIDA_5Class", marida_path, max_train_pixels=args.max_pixels, n_estimators=args.n_estimators, learning_rate=args.learning_rate, max_depth=args.max_depth)
