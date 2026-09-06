import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


class RFClassifier:
    """Wrapper around sklearn RandomForestClassifier driven by a YAML config dict."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", 30),
            min_samples_split=cfg.get("min_samples_split", 5),
            min_samples_leaf=cfg.get("min_samples_leaf", 2),
            max_features=cfg.get("max_features", "sqrt"),
            class_weight=cfg.get("class_weight", "balanced"),
            n_jobs=cfg.get("n_jobs", -1),
            random_state=cfg.get("random_state", 42),
            verbose=1,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"[INFO] Model saved -> {path}")

    def load(self, path):
        self.model = joblib.load(path)
        print(f"[INFO] Model loaded <- {path}")

    def write_model_summary(self, path, dataset_name, n_train_pixels, train_time):
        """Write a human-readable model summary to a text file."""
        lines = [
            f"Random Forest (Advanced) — Model Summary",
            f"{'=' * 50}",
            f"Dataset           : {dataset_name}",
            f"Training pixels   : {n_train_pixels:,}",
            f"Training time (s) : {train_time:.2f}",
            f"",
            f"Hyperparameters:",
            f"  n_estimators    : {self.cfg.get('n_estimators')}",
            f"  max_depth       : {self.cfg.get('max_depth')}",
            f"  min_samples_split: {self.cfg.get('min_samples_split')}",
            f"  min_samples_leaf : {self.cfg.get('min_samples_leaf')}",
            f"  max_features    : {self.cfg.get('max_features')}",
            f"  class_weight    : {self.cfg.get('class_weight')}",
            f"  n_jobs          : {self.cfg.get('n_jobs')}",
            f"  random_state    : {self.cfg.get('random_state')}",
            f"",
            f"Feature config:",
            f"  use_si          : {self.cfg.get('features', {}).get('use_si')}",
            f"  use_glcm        : {self.cfg.get('features', {}).get('use_glcm')}",
            f"  glcm_win_size   : {self.cfg.get('features', {}).get('glcm_win_size')}",
            f"  glcm_bins       : {self.cfg.get('features', {}).get('glcm_bins')}",
            f"",
            f"Feature importances (top 20):",
        ]

        importances = self.model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:20]
        for rank, idx in enumerate(top_idx, 1):
            lines.append(f"  {rank:>3}. Feature {idx:>4}  importance={importances[idx]:.6f}")

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[INFO] Model summary -> {path}")
