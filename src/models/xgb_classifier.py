import numpy as np
import xgboost as xgb
import time

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
    pixels = img.reshape(C, -1).T # (N_pixels, C)
    
    # 1. Base channels (first min(C, 8) or select key bands)
    if C >= 8:
        base_feats = pixels[:, :8] # 8 features
    else:
        pad = np.zeros((pixels.shape[0], 8 - C), dtype=np.float32)
        base_feats = np.hstack([pixels, pad])
        
    # Key band picks for ratios/indices
    b_blue = pixels[:, 0]
    b_green = pixels[:, 1] if C > 1 else pixels[:, 0]
    b_red = pixels[:, 2] if C > 2 else pixels[:, 0]
    b_nir = pixels[:, min(6, C-1)]
    b_swir = pixels[:, min(7, C-1)]
    
    eps = 1e-6
    # 2. Normalized difference indices (5 features)
    ndvi = (b_nir - b_red) / (b_nir + b_red + eps)
    ndwi = (b_green - b_nir) / (b_green + b_nir + eps)
    fai = b_nir - (b_red + (b_swir - b_red) * 0.5)
    fdi = b_nir - (b_red + (b_swir - b_red) * 0.7)
    ndmi = (b_nir - b_swir) / (b_nir + b_swir + eps)
    
    indices = np.column_stack([ndvi, ndwi, fai, fdi, ndmi]) # 5 features
    
    # 3. Spectral Ratios (4 features)
    r1 = b_red / (b_green + eps)
    r2 = b_nir / (b_red + eps)
    r3 = b_nir / (b_green + eps)
    r4 = b_swir / (b_nir + eps)
    ratios = np.column_stack([r1, r2, r3, r4]) # 4 features
    
    # 4. Statistical summaries across channels per pixel (4 features)
    mean_spec = np.mean(pixels, axis=1)
    std_spec = np.std(pixels, axis=1)
    max_spec = np.max(pixels, axis=1)
    norm_spec = np.linalg.norm(pixels, axis=1)
    stats = np.column_stack([mean_spec, std_spec, max_spec, norm_spec]) # 4 features
    
    # Total: 8 + 5 + 4 + 4 = 21 features
    features = np.hstack([base_feats, indices, ratios, stats])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features

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
