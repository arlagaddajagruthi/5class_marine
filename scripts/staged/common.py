"""
Shared code for the staged feature experiments (scripts/staged/stage*.py).

Every stage trains the same five ML models on the same sampled pixels; only
the feature columns differ, so differences between stages are paired
comparisons. Only annotated classes are used (1 Water, 2 Marine Debris,
3 Algae/Sargassum, 4 Other); class 0 (background/unannotated) is ignored.

Pixels (per dataset, cached under outputs/staged/cache/):
  train : official train split, ALL annotated pixels (optional --train-cap per class)
  test  : official test split, ALL annotated pixels (splits are scene-disjoint)
Water outnumbers debris ~180:1 in MARIDA, so models are trained with balanced
class weights (each class carries equal total weight). --weighting log uses the
MARIDA/MADOS papers' 1/ln(1.03 + class frequency) instead; --weighting none (or
--no-class-weights) disables weighting for all models, the SVM included.

Outputs: <--out-root, default outputs/staged>/<stage>/<dataset>/
  summary.md, metrics.csv (overall + per class), confusion_matrices/,
  feature_importance.csv, run_config.json
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, jaccard_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.datasets.mados import MADOSDataset
from src.datasets.marida import MARIDADataset
from src.features.pixel_features import BAND_NAMES, FEATURE_NAMES, extract_pixel_features

OUT_ROOT = os.path.join(REPO, "outputs", "staged")
CACHE_DIR = os.path.join(OUT_ROOT, "cache")
DATASETS = {
    "MARIDA": (MARIDADataset, os.path.join(REPO, "data", "MARIDA_5Class")),
    "MADOS": (MADOSDataset, os.path.join(REPO, "data", "MADOS_5Class")),
}
CLASSES = [1, 2, 3, 4]
CLASS_NAMES = {1: "Water", 2: "Marine Debris", 3: "Algae/Sargassum", 4: "Other"}
MODELS = ["rf", "xgb", "lgbm", "ada", "svm"]
LOG_WEIGHT_C = 1.03

# Feature groups available to the stages (columns of extract_pixel_features)
INDICES = ["NDVI", "NDWI", "FAI", "FDI", "NDMI"]
RATIOS = ["red/green", "nir/red", "nir/green", "swir1/nir"]
SPEC_STATS = ["spec_mean", "spec_std", "spec_max", "spec_norm"]


# ------------------------------------------------------------------ data ---

def _patch_ids(ds):
    return ds.patch_ids


def sample_split(dataset_name, split, cap, seed):
    """
    Annotated pixels (classes 1-4) of one split with all 24 features.
    cap=None keeps every annotated pixel; otherwise at most `cap` per class.
    Cached as .npz so every stage reuses identical pixels.
    """
    tag = "all" if cap is None else f"cap{cap}"
    path = os.path.join(CACHE_DIR, f"{dataset_name}_{split}_{tag}_seed{seed}_f{len(FEATURE_NAMES)}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["X"], z["y"]

    cls, root = DATASETS[dataset_name]
    ds = cls(root, split=split)
    rng = np.random.default_rng(seed)
    print(f"  [{dataset_name}/{split}] reading {len(ds)} patches...", flush=True)

    # Pass 1: masks only, to count pixels per class
    masks = []
    for i in range(len(ds)):
        _, m = ds[i]
        masks.append(m.numpy().ravel())
    counts = np.array([[np.sum(m == c) for c in CLASSES] for m in masks])

    # Choose global pixel ranks per class, then map back to patches
    picks = [dict() for _ in masks]
    for ci, c in enumerate(CLASSES):
        total = int(counts[:, ci].sum())
        if cap is None or total <= cap:
            chosen = np.arange(total)
        else:
            chosen = np.sort(rng.choice(total, size=cap, replace=False))
        bounds = np.cumsum(counts[:, ci])
        owner = np.searchsorted(bounds, chosen, side="right")
        starts = bounds - counts[:, ci]
        for pi in np.unique(owner):
            picks[pi][c] = chosen[owner == pi] - starts[pi]

    X_list, y_list = [], []
    for pi in range(len(ds)):
        if not picks[pi]:
            continue
        img, _ = ds[pi]
        sel = np.concatenate([np.flatnonzero(masks[pi] == c)[r] for c, r in picks[pi].items()])
        pixels = img.numpy().reshape(img.shape[0], -1)[:, sel][:, :, None]
        X_list.append(extract_pixel_features(pixels))
        y_list.append(masks[pi][sel])

    X, y = np.vstack(X_list), np.concatenate(y_list).astype(np.int64)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def load_data(dataset_name, train_cap, seed):
    X_tr, y_tr = sample_split(dataset_name, "train", train_cap, seed)
    X_te, y_te = sample_split(dataset_name, "test", None, seed)
    return X_tr, y_tr, X_te, y_te


def columns(feature_names):
    missing = [f for f in feature_names if f not in FEATURE_NAMES]
    if missing:
        raise ValueError(f"Unknown features {missing}")
    return [FEATURE_NAMES.index(f) for f in feature_names]


# ---------------------------------------------------------------- models ---

def make_model(name, seed, class_weight=None):
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 objective="multi:softprob", tree_method="hist",
                                 n_jobs=-1, random_state=seed)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=100, max_depth=6, num_leaves=31, learning_rate=0.1,
                                  objective="multiclass", n_jobs=-1, random_state=seed, verbose=-1)
    if name == "ada":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, random_state=seed),
                                  n_estimators=100, learning_rate=0.1, random_state=seed)
    if name == "svm":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LinearSVC(C=1.0, max_iter=3000, class_weight=class_weight,
                                           dual="auto", random_state=seed))])
    raise ValueError(name)


def class_weights(y, scheme):
    """Per-class weights (index = class id 0..3), or None for no weighting.
    balanced: n / (k * n_class), so every class carries equal total weight.
    log: 1 / ln(c + class frequency), c = 1.03, as in the MARIDA U-Net and MADOS MariNeXt code."""
    if scheme == "none":
        return None
    counts = np.clip(np.bincount(y, minlength=len(CLASSES)).astype(np.float64), 1, None)
    if scheme == "balanced":
        return len(y) / (len(CLASSES) * counts)
    if scheme == "log":
        return 1.0 / np.log(LOG_WEIGHT_C + counts / counts.sum())
    raise ValueError(scheme)


def importances(model, name, feature_names):
    est = model.named_steps["clf"] if name == "svm" else model
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=np.float64)
    else:  # linear SVM: mean |coef| on standardised features
        imp = np.abs(est.coef_).mean(axis=0)
    return imp / imp.sum() if imp.sum() > 0 else imp


def evaluate(y_true, y_pred):
    labels = list(range(len(CLASSES)))
    out = {
        "precision": precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "iou": jaccard_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    per = {k: f(y_true, y_pred, average=None, labels=labels, zero_division=0)
           for k, f in [("precision", precision_score), ("recall", recall_score),
                        ("f1", f1_score), ("iou", jaccard_score)]}
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per.update(class_accuracies(cm))
    for i, c in enumerate(CLASSES):
        for k in per:
            out[f"{k}_c{c}"] = float(per[k][i])
    return out, cm


def class_accuracies(cm):
    """acc: TP / class pixels (per-class accuracy = recall = producer's accuracy).
    ovr_acc: (TP + TN) / all pixels, the class as a one-vs-rest yes/no task."""
    tp, total = np.diag(cm), cm.sum()
    tn = total - cm.sum(0) - cm.sum(1) + tp
    return {"acc": tp / np.clip(cm.sum(1), 1, None), "ovr_acc": (tp + tn) / total}


def plot_cm(cm, path, title):
    names = [CLASS_NAMES[c] for c in CLASSES]
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names, ax=ax[0])
    sns.heatmap(cmn * 100, annot=True, fmt=".1f", cmap="Blues", xticklabels=names, yticklabels=names,
                ax=ax[1], vmin=0, vmax=100)
    ax[0].set_title("Counts"); ax[1].set_title("Row-normalised (% recall)")
    for a in ax:
        a.set_xlabel("Predicted"); a.set_ylabel("True")
    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


# ------------------------------------------------------------- run stage ---

def run_stage(stage_id, description, feature_names, args, extra=None):
    """Train/evaluate every model on every dataset for one feature set.
    extra: optional function(dataset_name, keep_tr, keep_te) -> (E_tr, E_te, names) whose
    columns are appended to the selected pixel features (e.g. per-patch environmental data)."""
    if args.no_class_weights:
        args.weighting = "none"
    cols = columns(feature_names)
    all_rows = []
    for dataset_name in args.datasets:
        out_dir = os.path.join(args.out_root, stage_id, dataset_name)
        os.makedirs(os.path.join(out_dir, "confusion_matrices"), exist_ok=True)
        print(f"\n=== {stage_id} | {dataset_name} | {len(cols)} features: {', '.join(feature_names)}", flush=True)

        X_tr, y_tr, X_te, y_te = load_data(dataset_name, args.train_cap, args.seed)
        keep_tr, keep_te = np.isin(y_tr, CLASSES), np.isin(y_te, CLASSES)
        lab = {c: i for i, c in enumerate(CLASSES)}
        Xtr, ytr = X_tr[keep_tr][:, cols], np.vectorize(lab.get)(y_tr[keep_tr])
        Xte, yte = X_te[keep_te][:, cols], np.vectorize(lab.get)(y_te[keep_te])
        names = list(feature_names)
        if extra is not None:
            E_tr, E_te, extra_names = extra(dataset_name, keep_tr, keep_te)
            Xtr, Xte, names = np.hstack([Xtr, E_tr]), np.hstack([Xte, E_te]), names + list(extra_names)
        print(f"  train {len(ytr):,} px {np.bincount(ytr, minlength=4)} | "
              f"test {len(yte):,} px {np.bincount(yte, minlength=4)}", flush=True)

        rows, imp_rows = [], []
        for model_name in args.models:
            sel = np.arange(len(ytr))
            if model_name == "svm" and args.svm_cap:
                rng = np.random.default_rng(args.seed)
                sel = np.sort(np.concatenate([
                    rng.choice(np.flatnonzero(ytr == k), min(args.svm_cap, (ytr == k).sum()), replace=False)
                    for k in range(len(CLASSES))]))
            cw = class_weights(ytr[sel], args.weighting)
            fit_kw = {}
            if cw is None:
                model = make_model(model_name, args.seed)
            elif model_name == "svm":
                model = make_model(model_name, args.seed, dict(enumerate(cw)))
            else:
                model = make_model(model_name, args.seed)
                fit_kw["sample_weight"] = cw[ytr[sel]]
            t0 = time.time(); model.fit(Xtr[sel], ytr[sel], **fit_kw); t_fit = time.time() - t0
            t0 = time.time(); pred = np.asarray(model.predict(Xte)).ravel().astype(int); t_pred = time.time() - t0
            m, cm = evaluate(yte, pred)
            row = {"stage": stage_id, "description": description, "dataset": dataset_name,
                   "model": model_name, "n_features": len(names), "features": "|".join(names),
                   "n_train": len(sel), "n_test": len(yte),
                   "train_time_s": round(t_fit, 2), "predict_time_s": round(t_pred, 2), **m}
            rows.append(row)
            print(f"  {model_name:5s} F1={m['f1']:.4f} IoU={m['iou']:.4f} Acc={m['accuracy']:.4f} "
                  f"debris F1/IoU={m['f1_c2']:.4f}/{m['iou_c2']:.4f} "
                  f"algae F1/IoU={m['f1_c3']:.4f}/{m['iou_c3']:.4f} ({t_fit:.0f}s)", flush=True)

            pd.DataFrame(cm, index=[CLASS_NAMES[c] for c in CLASSES],
                         columns=[CLASS_NAMES[c] for c in CLASSES]).to_csv(
                os.path.join(out_dir, "confusion_matrices", f"{model_name}.csv"))
            plot_cm(cm, os.path.join(out_dir, "confusion_matrices", f"{model_name}.png"),
                    f"{stage_id} | {dataset_name} | {model_name.upper()}")
            for f, v in zip(names, importances(model, model_name, names)):
                imp_rows.append({"model": model_name, "feature": f, "importance": float(v)})

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
        pd.DataFrame(imp_rows).to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)
        with open(os.path.join(out_dir, "run_config.json"), "w") as f:
            json.dump({"stage": stage_id, "description": description, "features": names,
                       **vars(args)}, f, indent=2)
        write_summary(df, os.path.join(out_dir, "summary.md"), stage_id, description, dataset_name, names)
        all_rows += rows
    return pd.DataFrame(all_rows)


def write_summary(df, path, stage_id, description, dataset_name, feature_names):
    lines = [f"# {stage_id}: {description} ({dataset_name})", "",
             f"Features ({len(feature_names)}): {', '.join(feature_names)}", "",
             "Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated "
             "pixels of the official test split.", "",
             "## Overall", "",
             "| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['model']} | {r['f1']:.4f} | {r['iou']:.4f} | {r['precision']:.4f} | "
                     f"{r['recall']:.4f} | {r['accuracy']:.4f} | {r['kappa']:.4f} | "
                     f"{r['n_train']:,} | {r['train_time_s']:.0f} |")
    for metric, label in [("acc", "accuracy (TP / class pixels = recall)"), ("precision", "precision"),
                          ("recall", "recall"), ("f1", "F1"), ("iou", "IoU"),
                          ("ovr_acc", "one-vs-rest accuracy ((TP + TN) / all pixels)")]:
        lines += ["", f"## Per-class {label}", "",
                  "| Model | " + " | ".join(CLASS_NAMES[c] for c in CLASSES) + " |",
                  "|" + "---|" * (1 + len(CLASSES))]
        for _, r in df.iterrows():
            lines.append(f"| {r['model']} | " + " | ".join(f"{r[f'{metric}_c{c}']:.4f}" for c in CLASSES) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def base_parser(doc):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--train-cap", type=int, default=None,
                    help="Max training pixels per class (default: all annotated pixels)")
    ap.add_argument("--svm-cap", type=int, default=None,
                    help="Max SVM training pixels per class (default: all)")
    ap.add_argument("--weighting", choices=["balanced", "log", "none"], default="balanced",
                    help="Class weighting: balanced (default), log = 1/ln(1.03 + freq) as in the "
                         "MARIDA/MADOS papers, or none")
    ap.add_argument("--no-class-weights", action="store_true", help="Same as --weighting none")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", default=OUT_ROOT,
                    help="Folder for stage outputs (default: outputs/staged). Pixel cache stays shared.")
    return ap
