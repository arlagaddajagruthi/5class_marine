"""Run the stage-5 Random Forest on a Scene and describe what it found."""
import os
import threading

import joblib
import numpy as np
from scipy import ndimage

from . import config
from .features import B2, B3, B4, B8, B11, stage5_features, stage8_features

_models = {}
_lock = threading.Lock()

WATER, DEBRIS, ALGAE, OTHER, UNCERTAIN, SCOPE = 1, 2, 3, 4, 5, 6
MIN_SCOPE_PIXELS = 50          # 0.5 ha at 10 m
GROUP_GAP_PX = 2               # gaps of up to ~2 pixels (20 m) stay in one patch


def get_model(key):
    with _lock:
        if key not in _models:
            path = os.path.join(config.MODEL_DIR, config.MODELS[key]["file"])
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file {path} not found. Run train_models.py first.")
            if os.environ.get("ONE_MODEL_IN_MEMORY"):   # small hosts: drop the previous model first
                _models.clear()
            _models[key] = joblib.load(path)
        return _models[key]


def valid_mask(img):
    return np.any(img != 0, axis=0)


def scope_mask(img):
    """Cloud- or land-like pixels the ocean-only model was never trained on.
    Conservative thresholds, chosen so that almost no annotated test pixel of
    MARIDA or MADOS is flagged (see train_models.py report)."""
    vis = (img[B2] + img[B3] + img[B4]) / 3.0
    nir, swir1, red = img[B8], img[B11], img[B4]
    ndvi = (nir - red) / (nir + red + 1e-6)
    cloud = vis > 0.25
    land = (swir1 > 0.12) & (ndvi > 0.2)          # dry vegetation / soil reflect strongly in SWIR
    bare = swir1 > 0.25
    raw = cloud | land | bare
    # Ships and foam are also bright but small; clouds and land are large.
    lab, n = ndimage.label(raw, structure=np.ones((3, 3), bool))
    if n == 0:
        return raw
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= MIN_SCOPE_PIXELS
    return keep[lab]


def predict_proba(img, model_key):
    """Class probabilities (4, H, W) for classes Water, Debris, Algae, Other."""
    model = get_model(model_key)
    c, h, w = img.shape
    valid2d = valid_mask(img)
    valid = valid2d.ravel()
    probs = np.zeros((4, h * w), dtype=np.float32)
    stage = config.MODELS[model_key]["stage"]
    X = (stage8_features(img, valid2d) if stage == "stage8" else stage5_features(img))[valid]
    chunk = 200_000
    out = np.empty((len(X), 4), dtype=np.float32)
    for s in range(0, len(X), chunk):
        out[s:s + chunk] = model.predict_proba(X[s:s + chunk])
    probs[:, valid] = out.T
    return probs.reshape(4, h, w), valid.reshape(h, w)


def label_map(probs, valid, img, threshold, use_scope):
    top = probs.argmax(axis=0).astype(np.uint8) + 1
    conf = probs.max(axis=0)
    labels = np.where(valid, top, 0).astype(np.uint8)
    labels[valid & (conf < threshold)] = UNCERTAIN
    if use_scope:
        labels[valid & scope_mask(img)] = SCOPE
    return labels, conf


def find_regions(labels, conf, scene, max_regions=200):
    """Connected groups (8-neighbour) of debris, algae and other pixels."""
    px_w, px_h = scene.pixel_size_m
    regions = []
    structure = np.ones((3, 3), dtype=bool)
    for cls in (DEBRIS, ALGAE, OTHER):
        mask = labels == cls
        # Pixels up to GROUP_GAP_PX apart belong to the same patch (a windrow of
        # debris is often broken by 1-2 water pixels), but only the class's own
        # pixels are counted.
        grouped, n = ndimage.label(ndimage.binary_dilation(mask, structure, iterations=GROUP_GAP_PX),
                                   structure=structure)
        lab = np.where(mask, grouped, 0)
        if n == 0:
            continue
        present = np.unique(lab[lab > 0])
        remap = np.zeros(n + 1, dtype=np.int32)
        remap[present] = np.arange(1, len(present) + 1)
        lab, n = remap[lab], len(present)
        idx = np.arange(1, n + 1)
        sizes = ndimage.sum_labels(np.ones_like(lab), lab, idx)
        mean_conf = ndimage.mean(conf, lab, idx)
        max_conf = ndimage.maximum(conf, lab, idx)
        centers = ndimage.center_of_mass(np.ones_like(lab), lab, idx)
        slices = ndimage.find_objects(lab)
        for i in range(n):
            sl = slices[i]
            regions.append({
                "class": cls,
                "pixels": int(sizes[i]),
                "area_m2": float(sizes[i] * px_w * px_h),
                "mean_conf": float(mean_conf[i]),
                "max_conf": float(max_conf[i]),
                "row": float(centers[i][0]), "col": float(centers[i][1]),
                "bbox": [int(sl[0].start), int(sl[1].start), int(sl[0].stop), int(sl[1].stop)],
            })
    order = {DEBRIS: 0, ALGAE: 1, OTHER: 2}
    regions.sort(key=lambda r: (order[r["class"]], -r["pixels"], -r["mean_conf"]))
    regions = regions[:max_regions]
    if regions:
        ll = scene.latlon([r["row"] for r in regions], [r["col"] for r in regions])
        for i, r in enumerate(regions):
            r["id"] = i + 1
            if ll is not None:
                r["lat"], r["lon"] = float(ll[0][i]), float(ll[1][i])
    return regions


def class_summary(labels, scene):
    px_w, px_h = scene.pixel_size_m
    counts = np.bincount(labels.ravel(), minlength=7)
    analysed = int(counts[1:].sum())
    out = []
    for cls in range(1, 7):
        n = int(counts[cls])
        out.append({"class": cls, "pixels": n, "area_m2": float(n * px_w * px_h),
                    "percent": 100.0 * n / analysed if analysed else 0.0})
    return out


def gt_metrics(gt, probs, valid):
    """Scores on annotated pixels (classes 1-4), using the raw top class
    (no uncertainty or scope masking), as in the paper."""
    pred = probs.argmax(axis=0) + 1
    m = (gt >= 1) & (gt <= 4) & valid
    if not m.any():
        return None
    y, p = gt[m].astype(int), pred[m].astype(int)
    per = []
    for c in range(1, 5):
        tp = int(np.sum((y == c) & (p == c)))
        fp = int(np.sum((y != c) & (p == c)))
        fn = int(np.sum((y == c) & (p != c)))
        support = tp + fn
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / support if support else None
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else None
        per.append({"class": c, "support": support, "tp": tp, "fp": fp, "fn": fn,
                    "precision": prec, "recall": rec, "f1": f1, "iou": iou})
    cm = np.zeros((4, 4), dtype=int)
    np.add.at(cm, (y - 1, p - 1), 1)
    return {"annotated_pixels": int(m.sum()), "accuracy": float(np.mean(y == p)),
            "per_class": per, "confusion": cm.tolist()}
