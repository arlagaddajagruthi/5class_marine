import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


class RFClassifier:

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

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str, cfg: dict | None = None):
        instance = cls.__new__(cls)
        instance.cfg = cfg or {}
        instance.model = joblib.load(path)
        return instance

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    def get_model_info(self) -> dict:
        estimators = self.model.estimators_
        n_trees = len(estimators)
        total_nodes = sum(t.tree_.node_count for t in estimators)
        max_actual_depth = max(t.tree_.max_depth for t in estimators)
        avg_depth = np.mean([t.tree_.max_depth for t in estimators])

        import io
        buf = io.BytesIO()
        joblib.dump(self.model, buf)
        model_size_bytes = buf.tell()

        return {
            "n_estimators": n_trees,
            "configured_max_depth": self.cfg.get("max_depth"),
            "actual_max_depth": int(max_actual_depth),
            "actual_avg_depth": float(avg_depth),
            "total_nodes": int(total_nodes),
            "model_size_mb": model_size_bytes / (1024 ** 2),
            "n_features": self.model.n_features_in_,
            "n_classes": len(self.model.classes_),
        }

    def write_model_summary(self, path: str, dataset_name: str,
                            train_samples: int, train_time_sec: float):
        from datetime import datetime

        info = self.get_model_info()
        importances = self.feature_importances_

        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL INFORMATION\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Model Type            : Random Forest Classifier\n")
            f.write(f"Dataset               : {dataset_name}\n")
            f.write(f"Timestamp             : {datetime.now()}\n")
            f.write(f"Scikit-learn Backend  : sklearn.ensemble.RandomForestClassifier\n\n")

            f.write("=" * 80 + "\n")
            f.write("HYPERPARAMETERS\n")
            f.write("=" * 80 + "\n")
            for key in ("n_estimators", "max_depth", "min_samples_split",
                        "min_samples_leaf", "max_features", "class_weight",
                        "random_state"):
                f.write(f"{key:<25s}: {self.cfg.get(key)}\n")
            f.write(f"{'max_train_pixels':<25s}: {self.cfg.get('max_train_pixels')}\n")
            f.write(f"{'num_classes':<25s}: {self.cfg.get('num_classes')}\n\n")

            f.write("=" * 80 + "\n")
            f.write("MODEL STATISTICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Number of Trees       : {info['n_estimators']}\n")
            f.write(f"Configured Max Depth  : {info['configured_max_depth']}\n")
            f.write(f"Actual Max Depth      : {info['actual_max_depth']}\n")
            f.write(f"Actual Avg Depth      : {info['actual_avg_depth']:.1f}\n")
            f.write(f"Total Nodes           : {info['total_nodes']:,}\n")
            f.write(f"Input Features        : {info['n_features']}\n")
            f.write(f"Output Classes        : {info['n_classes']}\n")
            f.write(f"Model Size            : {info['model_size_mb']:.2f} MB\n\n")

            f.write("=" * 80 + "\n")
            f.write("TRAINING STATISTICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Training Samples      : {train_samples:,}\n")
            f.write(f"Training Time         : {train_time_sec:.2f} sec\n\n")

            f.write("=" * 80 + "\n")
            f.write("FEATURE IMPORTANCES (Gini)\n")
            f.write("=" * 80 + "\n")
            for i, imp in enumerate(importances):
                f.write(f"  Band {i:<3d}: {imp:.6f}\n")
