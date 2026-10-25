"""
Builds the stage-8 feature matrices exactly as scripts/staged_ml/common.run_stage does
(stage8_ndwi_ratios_glcm.py): 11 bands + NDWI + 4 band ratios from
src/features/pixel_features.py, then the 6 GLCM columns from
scripts/staged_ml/glcm_features.py, classes 1-4 re-encoded to 0-3.

configure() only redirects where the staged code looks for the datasets and the
pixel/GLCM caches; the feature code itself is untouched.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
STAGED = os.path.join(REPO, "scripts", "staged_ml")
if STAGED not in sys.path:
    sys.path.insert(0, STAGED)

import common  # noqa: E402
import glcm_features  # noqa: E402
from stage7_glcm import glcm_columns  # noqa: E402

# The GLCM caches were moved here by the "rearranged output folders" commit
DEFAULT_CACHE = os.path.join(REPO, "outputs", "staged", "balanced_weights", "cache")
STAGE8_FEATURES = list(common.BAND_NAMES) + ["NDWI"] + common.RATIOS


def configure(data_root, cache_dir=DEFAULT_CACHE):
    """Point the staged code at the dataset folders and the existing cache folder."""
    data_root = os.path.abspath(data_root)
    for name, folder in [("MARIDA", "MARIDA_5Class"), ("MADOS", "MADOS_5Class")]:
        cls, _ = common.DATASETS[name]
        common.DATASETS[name] = (cls, os.path.join(data_root, folder))   # same dict object glcm_features uses
    common.CACHE_DIR = cache_dir
    glcm_features.CACHE_DIR = cache_dir


def stage8_data(dataset_name, seed=42):
    """(Xtr, ytr, Xte, yte, feature_names); y encoded 0..3 = classes 1..4."""
    X_tr, y_tr, X_te, y_te = common.load_data(dataset_name, None, seed)
    keep_tr, keep_te = np.isin(y_tr, common.CLASSES), np.isin(y_te, common.CLASSES)
    lab = {c: i for i, c in enumerate(common.CLASSES)}
    cols = common.columns(STAGE8_FEATURES)
    Xtr, ytr = X_tr[keep_tr][:, cols], np.vectorize(lab.get)(y_tr[keep_tr])
    Xte, yte = X_te[keep_te][:, cols], np.vectorize(lab.get)(y_te[keep_te])
    E_tr, E_te, extra = glcm_columns(dataset_name, keep_tr, keep_te)
    Xtr, Xte = np.hstack([Xtr, E_tr]), np.hstack([Xte, E_te])
    return Xtr, ytr.astype(np.int64), Xte, yte.astype(np.int64), STAGE8_FEATURES + list(extra)
