import numpy as np
import xgboost as xgb
import time

from src.features.pixel_features import extract_pixel_features

def extract_pixel_features_21(img_tensor):
    """
    Per-pixel features from an 11-band (C, H, W) image; see
    src/features/pixel_features.py. Kept under its old name for existing
    callers, but now returns 24 correctly indexed features (was 21 with
    shifted band indices). Channels beyond the 11 bands are appended as-is.
    """
    return extract_pixel_features(img_tensor)


class MarineXGBClassifier:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, num_classes=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.random_state = random_state
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective='multi:softprob',
            num_class=self.num_classes,
            tree_method='hist',
            random_state=self.random_state,
            n_jobs=-1
        )

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

    def save(self, filepath):
        self.model.save_model(filepath)

    def load(self, filepath):
        self.model.load_model(filepath)
