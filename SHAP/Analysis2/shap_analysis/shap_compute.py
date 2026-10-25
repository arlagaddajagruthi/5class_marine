"""
SHAP computation for the five saved stage-8 models (see train_stage8_models.py).

Explained outputs (one SHAP vector per class output k = 0..3 <-> labels 1..4):
  rf   TreeSHAP (interventional)    predict_proba               -> probability
  xgb  TreeSHAP (interventional)    raw multi:softprob margin    -> softmax logit
  lgbm TreeSHAP (interventional)    raw multiclass score         -> softmax logit
  ada  TreeSHAP (interventional) on an exact tree-ensemble rewrite of the SAMME
       decision_function (sum of depth-3 trees with per-class leaf values)
                                    decision_function            -> SAMME score
                                    (predict_proba = softmax(score / (K-1)))
  svm  LinearExplainer (interventional, exact for a linear model) on the
       StandardScaler + LinearSVC one-vs-rest decision_function -> margin
       (LinearSVC has no predict_proba); cross-checked with KernelSHAP.

All explainers use the same background: a class-stratified sample of TRAINING
pixels, so base values are the mean output over a class-balanced background and
no test pixel is used to build an explainer. SHAP interaction values (path-dependent
TreeSHAP) are computed for the four tree models on a smaller stratified subset.
"""
import os
import time

import joblib
import numpy as np
import shap

K = 4
MODEL_NAMES = ["rf", "xgb", "lgbm", "ada", "svm"]
MODEL_LABEL = {"rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM", "ada": "AdaBoost", "svm": "Linear SVM"}
OUTPUT_SCALE = {
    "rf": "class probability",
    "xgb": "raw softmax margin (log-odds scale)",
    "lgbm": "raw softmax margin (log-odds scale)",
    "ada": "SAMME decision score",
    "svm": "one-vs-rest SVM margin",
}
EXPLAINER = {
    "rf": "TreeExplainer (interventional TreeSHAP)",
    "xgb": "TreeExplainer (interventional TreeSHAP)",
    "lgbm": "TreeExplainer (interventional TreeSHAP)",
    "ada": "TreeExplainer (interventional TreeSHAP) on exact tree rewrite of SAMME decision_function",
    "svm": "LinearExplainer (interventional; exact for linear margin), validated with KernelExplainer",
}


# ------------------------------------------------------------ model output ---

def raw_output(model, name, X):
    """The exact quantity each explainer explains, shape (n, 4)."""
    if name == "rf":
        return model.predict_proba(X)
    if name == "xgb":
        return model.predict(X, output_margin=True)
    if name == "lgbm":
        return model.predict(X, raw_score=True)
    if name in ("ada", "svm"):
        return model.decision_function(X)
    raise ValueError(name)


def load_models(model_dir):
    return {n: joblib.load(os.path.join(model_dir, f"{n}.joblib")) for n in MODEL_NAMES}


# ------------------------------------------------------- AdaBoost -> trees ---

def ada_as_tree_dict(ada, X_cover=None, w_cover=None):
    """
    Exact rewrite of sklearn SAMME AdaBoostClassifier.decision_function:
        d_k(x) = sum_m (w_m / W) * (1 if tree_m(x) == k else -1/(K-1))
    Each weak learner becomes a tree whose leaf value vector is that term.
    node_sample_weight (the 'cover', only used by path-dependent TreeSHAP, i.e. the
    interaction values) is recomputed from X_cover with weights w_cover, because
    the stored weighted_n_node_samples reflect AdaBoost's re-weighted samples.
    """
    W = ada.estimator_weights_.sum()
    n_cls = ada.n_classes_
    trees = []
    for est, w in zip(ada.estimators_, ada.estimator_weights_):
        t = est.tree_
        pred = est.classes_[np.argmax(t.value[:, 0, :], axis=1)]          # node -> predicted class
        vals = np.where(pred[:, None] == ada.classes_[None, :], w / W, -w / W / (n_cls - 1))
        if X_cover is not None:
            path = est.decision_path(X_cover)                              # (n, nodes) sparse
            cover = np.asarray(path.T @ (w_cover if w_cover is not None else np.ones(len(X_cover)))).ravel()
        else:
            cover = t.weighted_n_node_samples.astype(np.float64)
        cover = np.maximum(cover, 1e-12)
        trees.append({
            "children_left": t.children_left.astype(np.int32),
            "children_right": t.children_right.astype(np.int32),
            "children_default": t.children_left.astype(np.int32),
            "features": t.feature.astype(np.int32),
            "thresholds": t.threshold.astype(np.float64),
            "values": vals.astype(np.float64),
            "node_sample_weight": cover.astype(np.float64),
        })
    return {"trees": trees, "base_offset": 0.0, "tree_output": "raw_value",
            "objective": "squared_error", "input_dtype": np.float32, "internal_dtype": np.float64}


# ---------------------------------------------------------------- explain ---

def make_explainer(model, name, background, ada_cover=None):
    if name == "svm":
        scaler, clf = model.named_steps["scaler"], model.named_steps["clf"]
        return shap.LinearExplainer((clf.coef_, clf.intercept_),
                                    shap.maskers.Independent(scaler.transform(background),
                                                             max_samples=len(background)))
    obj = ada_as_tree_dict(model, *(ada_cover or (None, None))) if name == "ada" else model
    # explicit masker: TreeExplainer would otherwise subsample a background > 100 rows
    return shap.TreeExplainer(obj, data=shap.maskers.Independent(background, max_samples=len(background)),
                              feature_perturbation="interventional", model_output="raw")


def shap_values(model, name, X, background):
    """Returns (phi (n, F, 4), base (4,)) for the raw output of raw_output()."""
    ex = make_explainer(model, name, background)
    Xin = model.named_steps["scaler"].transform(X) if name == "svm" else X
    if name == "svm":
        phi = np.asarray(ex.shap_values(Xin))
        base = np.asarray(ex.expected_value, dtype=np.float64)
    else:
        phi = np.asarray(ex.shap_values(Xin, check_additivity=False))
        base = np.asarray(ex.expected_value, dtype=np.float64).ravel()
    phi = normalise_shape(phi, len(X), X.shape[1])
    return phi.astype(np.float64), base


def normalise_shape(phi, n, F):
    """SHAP has returned lists, (n, F, K) and (K, n, F) across versions; make it (n, F, K)."""
    if isinstance(phi, list):
        phi = np.stack(phi, axis=-1)
    phi = np.asarray(phi)
    if phi.shape == (n, F, K):
        return phi
    if phi.shape == (K, n, F):
        return np.transpose(phi, (1, 2, 0))
    raise ValueError(f"unexpected SHAP shape {phi.shape} for n={n}, F={F}, K={K}")


def additivity_error(model, name, X, phi, base):
    """max |base + sum_j phi_j - f(x)| over samples and classes, max |f|, and the number of
    pixels whose error exceeds 1e-5 (SHAP stores sklearn split thresholds at lower precision,
    so a value lying exactly on a rounded threshold can be routed differently in one tree)."""
    f = raw_output(model, name, X)
    e = np.abs(base[None, :] + phi.sum(axis=1) - f)
    return float(e.max()), float(np.max(np.abs(f))), int((e.max(1) > 1e-5).sum())


def interaction_values(model, name, X, cover_X, cover_w):
    """Path-dependent TreeSHAP interaction values, (n, F, F, 4). Tree models only."""
    if name == "svm":
        raise ValueError("linear margin has no interactions")
    obj = ada_as_tree_dict(model, cover_X, cover_w) if name == "ada" else model
    ex = shap.TreeExplainer(obj, feature_perturbation="tree_path_dependent")
    iv = ex.shap_interaction_values(X)
    if isinstance(iv, list):
        iv = np.stack(iv, axis=-1)
    iv = np.asarray(iv)
    n, F = X.shape
    if iv.shape == (K, n, F, F):
        iv = np.transpose(iv, (1, 2, 3, 0))
    if iv.shape != (n, F, F, K):
        raise ValueError(f"unexpected interaction shape {iv.shape}")
    return iv.astype(np.float64), np.asarray(ex.expected_value, dtype=np.float64).ravel()


def kernel_check(model, X, background, nsamples=4096):
    """KernelSHAP on the SVM pipeline's decision_function (same background as the
    LinearExplainer), to validate it on a small sample."""
    ex = shap.KernelExplainer(model.decision_function, background)
    phi = normalise_shape(ex.shap_values(X, nsamples=nsamples, l1_reg=False, silent=True), len(X), X.shape[1])
    return phi, np.asarray(ex.expected_value).ravel()


# ---------------------------------------------------------------- samples ---

def stratified(y, per_class, rng):
    idx = []
    for k in range(K):
        pool = np.flatnonzero(y == k)
        idx.append(np.sort(rng.choice(pool, min(per_class, len(pool)), replace=False)))
    return np.concatenate(idx)


def explanation_set(y_test, preds, per_class, err_per_pair, seed):
    """
    Stratified test sample (min(per_class, n_k) per true class) plus, for every model and
    every observed (true, predicted) confusion, up to err_per_pair misclassified pixels.
    Returns test indices and a boolean mask marking the stratified part.
    """
    rng = np.random.default_rng(seed)
    strat = stratified(y_test, per_class, rng)
    extra = []
    for name in MODEL_NAMES:
        p = preds[name]
        for a in range(K):
            for b in range(K):
                if a == b:
                    continue
                pool = np.setdiff1d(np.flatnonzero((y_test == a) & (p == b)), strat)
                if len(pool):
                    extra.append(rng.choice(pool, min(err_per_pair, len(pool)), replace=False))
    extra = np.setdiff1d(np.concatenate(extra) if extra else np.array([], int), strat)
    idx = np.concatenate([strat, np.sort(extra)])
    return idx, np.r_[np.ones(len(strat), bool), np.zeros(len(extra), bool)]


def timed(fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    return out, time.time() - t0
