"""
Step 5: do environmental covariates improve the pixel classifiers on MARIDA_5Class?

For every model and feature set the *same* sampled pixels are used; the only
thing that changes is which columns are appended to the 24 per-pixel spectral
features (src/features/pixel_features.py, shared with the baseline scripts).
This gives a paired, like-for-like comparison against 'baseline'.

Feature sets (see src/environmental/features.py):
  baseline       24 spectral features only
  env_all        + all environmental columns
  sst/chla/wind/met/waves   + one group (what each group adds on its own)
  env_all-<g>    all groups except <g> (what removing one group costs)

Evaluation:
  official   train split -> test split (the splits are already scene-disjoint).
             Only 36 train / 15 test scenes, and env values vary mainly
             between scenes, so a single split gives a noisy estimate.
  scene-cv   GroupKFold over all 63 scenes (train+val+test pooled), so every
             scene is tested once; reports mean +/- std and paired deltas.

Label protocols:
  baseline   classes 0-4 exactly as in scripts/baselines/ (comparable numbers).
             Class 0 = background/unannotated (MARIDA: incl. cloud and cloud
             shadow, see scripts/utilities/convert.py).
  labelled   drop class 0 and evaluate only annotated classes 1-4.

Usage:
  python scripts/environmental/run_env_experiments.py
  python scripts/environmental/run_env_experiments.py --eval scene-cv --label-protocol labelled
  python scripts/environmental/run_env_experiments.py --models xgb lgbm --feature-sets baseline env_all sst
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
import tifffile
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.environmental.features import (ENV_TABLE_CSV, FEATURE_GROUPS,
                                        feature_set_columns, load_env_table)
from src.features.pixel_features import FEATURE_NAMES, extract_pixel_features
from src.metrics.metrics import calculate_metrics

MARIDA_ROOT = os.path.join(repo_root, "data", "MARIDA_5Class")
# Names follow the remapping in scripts/utilities/convert.py
CLASS_NAMES = {
    0: "Background/Unannotated",
    1: "Water",
    2: "Marine Debris",
    3: "Algae/Sargassum",
    4: "Other",
}
BASE_FEATURE_NAMES = list(FEATURE_NAMES)
DEFAULT_FEATURE_SETS = ["baseline", "env_all"] + list(FEATURE_GROUPS)
NAN_NATIVE = {"rf", "xgb", "lgbm"}


# ----------------------------------------------------------------- data ---

def read_split(split):
    with open(os.path.join(MARIDA_ROOT, "splits", f"{split}_X.txt")) as f:
        return [line.strip() for line in f if line.strip()]


def patch_paths(patch_id):
    scene = patch_id.rsplit("_", 1)[0]
    d = os.path.join(MARIDA_ROOT, "patches", f"S2_{scene}")
    return os.path.join(d, f"S2_{patch_id}.tif"), os.path.join(d, f"S2_{patch_id}_cl.tif")


def load_image(patch_id):
    """Same preprocessing as src/datasets/marida.py (no extra channels)."""
    img = tifffile.imread(patch_paths(patch_id)[0]).astype(np.float32)
    if img.ndim == 3 and img.shape[-1] < img.shape[0]:
        img = np.transpose(img, (2, 0, 1))
    return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)


def load_mask(patch_id):
    return tifffile.imread(patch_paths(patch_id)[1]).astype(np.int64).ravel()


def sample_pool(patch_ids, classes, cap_per_class, seed, cache_path=None):
    """
    Class-capped random pixel sample (same rule as the baseline scripts: at most
    `cap_per_class` pixels per class, all of them if fewer). Returns
    X_base (n, 24), y (n,), pix_patch (n,) index into patch_ids.
    """
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["X"], z["y"], z["p"]

    rng = np.random.default_rng(seed)
    masks = [load_mask(p) for p in patch_ids]
    counts = np.array([[np.sum(m == c) for c in classes] for m in masks])  # (patches, classes)

    # Global ranks per class -> (patch, rank-within-patch)
    picks = {i: {} for i in range(len(patch_ids))}
    for ci, c in enumerate(classes):
        total = counts[:, ci].sum()
        k = min(cap_per_class, total)
        chosen = np.sort(rng.choice(total, size=k, replace=False)) if k < total else np.arange(total)
        bounds = np.cumsum(counts[:, ci])
        owner = np.searchsorted(bounds, chosen, side="right")
        starts = bounds - counts[:, ci]
        for pi in np.unique(owner):
            picks[pi][c] = chosen[owner == pi] - starts[pi]

    X_list, y_list, p_list = [], [], []
    for pi, pid in enumerate(patch_ids):
        if not picks[pi]:
            continue
        mask = masks[pi]
        sel = np.concatenate([np.flatnonzero(mask == c)[r] for c, r in picks[pi].items()])
        img = load_image(pid)
        pixels = img.reshape(img.shape[0], -1)[:, sel][:, :, None]   # (C, n, 1)
        X_list.append(extract_pixel_features(pixels))
        y_list.append(mask[sel])
        p_list.append(np.full(len(sel), pi, dtype=np.int32))
        if (pi + 1) % 200 == 0:
            print(f"    features: {pi + 1}/{len(patch_ids)} patches")

    X, y, p = np.vstack(X_list), np.concatenate(y_list), np.concatenate(p_list)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, p=p)
    return X, y, p


def with_env(X_base, pix_patch, patch_ids, env_table, columns):
    if not columns:
        return X_base
    env = env_table.loc[patch_ids, columns].to_numpy(dtype=np.float32)
    return np.hstack([X_base, env[pix_patch]])


def subsample_per_class(y, cap, rng):
    idx = []
    for c in np.unique(y):
        ci = np.flatnonzero(y == c)
        idx.append(rng.choice(ci, cap, replace=False) if len(ci) > cap else ci)
    return np.sort(np.concatenate(idx))


# --------------------------------------------------------------- models ---

def make_model(name, seed):
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 objective="multi:softprob", tree_method="hist",
                                 n_jobs=-1, random_state=seed)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=100, max_depth=6, num_leaves=31,
                                  learning_rate=0.1, objective="multiclass",
                                  n_jobs=-1, random_state=seed, verbose=-1)
    if name == "ada":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, random_state=seed),
                                  n_estimators=100, learning_rate=0.1, random_state=seed)
    if name == "svm":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LinearSVC(C=1.0, max_iter=3000, class_weight="balanced",
                                           dual="auto", random_state=seed))])
    raise ValueError(name)


def impute_env(X_tr, X_te, n_base, tr_patch, tr_patch_ids, env_table, columns):
    """
    Median imputation for models that cannot take NaN (AdaBoost, SVM). The
    median is computed over training *patches*, never test data, and the
    number of imputed patches is reported.
    """
    if not columns:
        return X_tr, X_te, 0
    train_patches = [tr_patch_ids[i] for i in np.unique(tr_patch)]
    patch_env = env_table.loc[train_patches, columns]
    med = patch_env.median().to_numpy(dtype=np.float32)
    n_imputed = int(patch_env.isna().any(axis=1).sum())
    for X in (X_tr, X_te):
        env = X[:, n_base:]
        nan = np.isnan(env)
        if nan.any():
            env[nan] = np.take(med, np.nonzero(nan)[1])
    return X_tr, X_te, n_imputed


def importances(model, name, feature_names):
    est = model.named_steps["clf"] if name == "svm" else model
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=np.float64)
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_).mean(axis=0)
    else:
        return None
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    return pd.Series(imp, index=feature_names)


def save_cm(cm, names, path, title):
    cmn = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ----------------------------------------------------------- experiment ---

def run_fold(fold, Xb_tr, y_tr, p_tr, ids_tr, Xb_te, y_te, p_te, ids_te,
             env_table, args, classes, out_dir, imp_rows):
    rows = []
    lab_map = {c: i for i, c in enumerate(classes)}
    ytr = np.vectorize(lab_map.get)(y_tr)
    yte = np.vectorize(lab_map.get)(y_te)
    names = [CLASS_NAMES[c] for c in classes]
    n_base = Xb_tr.shape[1]

    for model_name in args.models:
        rng = np.random.default_rng(args.seed)
        if model_name == "svm":
            sel = subsample_per_class(ytr, args.svm_max_pixels_per_class, rng)
        else:
            sel = np.arange(len(ytr))

        for fs in args.feature_sets:
            cols = feature_set_columns(fs)
            X_tr = with_env(Xb_tr[sel], p_tr[sel], ids_tr, env_table, cols)
            X_te = with_env(Xb_te, p_te, ids_te, env_table, cols)
            n_imputed = 0
            if model_name not in NAN_NATIVE:
                X_tr, X_te, n_imputed = impute_env(X_tr.copy(), X_te.copy(), n_base,
                                                   p_tr[sel], ids_tr, env_table, cols)

            model = make_model(model_name, args.seed)
            t0 = time.time()
            model.fit(X_tr, ytr[sel])
            t_train = time.time() - t0
            t0 = time.time()
            pred = np.asarray(model.predict(X_te)).ravel().astype(int)
            t_pred = time.time() - t0

            m = calculate_metrics(yte, pred, num_classes=len(classes))
            row = {"fold": fold, "model": model_name, "feature_set": fs,
                   "n_features": X_tr.shape[1], "n_train": len(sel), "n_test": len(yte),
                   "train_patches_imputed": n_imputed,
                   "train_time_s": round(t_train, 2), "predict_time_s": round(t_pred, 2),
                   **{k: float(v) for k, v in m["overall"].items()}}
            for ci, c in enumerate(classes):
                row[f"f1_c{c}"] = float(m["per_class"]["f1"][ci])
                row[f"iou_c{c}"] = float(m["per_class"]["iou"][ci])
            rows.append(row)
            print(f"  [fold {fold}] {model_name:5s} {fs:16s} feats={X_tr.shape[1]:2d} "
                  f"F1={row['f1']:.4f} IoU={row['iou']:.4f} Acc={row['accuracy']:.4f} "
                  f"({t_train:.0f}s)", flush=True)

            cm_dir = os.path.join(out_dir, "confusion_matrices")
            os.makedirs(cm_dir, exist_ok=True)
            tag = f"{model_name}_{fs}_fold{fold}"
            pd.DataFrame(m["confusion_matrix"], index=names, columns=names).to_csv(
                os.path.join(cm_dir, f"{tag}.csv"))
            if fold in (0, "official"):
                save_cm(m["confusion_matrix"], names, os.path.join(cm_dir, f"{tag}.png"),
                        f"{model_name.upper()} - {fs}")

            if fs == "env_all":
                imp = importances(model, model_name, BASE_FEATURE_NAMES + cols)
                if imp is not None:
                    for feat, val in imp.items():
                        imp_rows.append({"fold": fold, "model": model_name,
                                         "feature": feat, "importance": float(val)})
    return rows


def summarise(results, out_dir, classes, eval_mode):
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "results_all_folds.csv"), index=False)

    metric_cols = ["precision", "recall", "f1", "iou", "accuracy"] + \
                  [f"f1_c{c}" for c in classes]
    base = df[df.feature_set == "baseline"].set_index(["fold", "model"])
    deltas = []
    for _, r in df.iterrows():
        b = base.loc[(r["fold"], r["model"])]
        deltas.append({f"d_{k}": r[k] - b[k] for k in ["f1", "iou", "accuracy"]})
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(deltas)], axis=1)

    agg = df.groupby(["model", "feature_set"], sort=False).agg(
        **{f"{k}_mean": (k, "mean") for k in metric_cols},
        **{f"{k}_std": (k, "std") for k in ["f1", "iou"]},
        d_f1_mean=("d_f1", "mean"), d_f1_std=("d_f1", "std"),
        d_iou_mean=("d_iou", "mean"),
        folds_improved=("d_f1", lambda s: int((s > 0).sum())),
        folds=("d_f1", "size"),
    ).reset_index()
    agg.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    lines = [f"# Environmental covariates on MARIDA_5Class ({eval_mode})", "",
             "Macro scores on the held-out pixels; deltas are paired against the "
             "`baseline` feature set of the same model on the same pixels.", ""]
    hdr = "| Model | Feature set | F1 | IoU | Acc | dF1 | dIoU | " + \
          " | ".join(f"F1 {CLASS_NAMES[c]}" for c in classes) + \
          (" | folds dF1>0 |" if eval_mode == "scene-cv" else " |")
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for _, r in agg.iterrows():
        pm = (lambda k: f"{r[k + '_mean']:.4f} +/- {r[k + '_std']:.4f}") if eval_mode == "scene-cv" \
            else (lambda k: f"{r[k + '_mean']:.4f}")
        d_f1 = f"{r['d_f1_mean']:+.4f}" + (f" +/- {r['d_f1_std']:.4f}" if eval_mode == "scene-cv" else "")
        cells = [r["model"], r["feature_set"], pm("f1"), pm("iou"), f"{r['accuracy_mean']:.4f}",
                 d_f1, f"{r['d_iou_mean']:+.4f}"] + [f"{r[f'f1_c{c}_mean']:.4f}" for c in classes]
        if eval_mode == "scene-cv":
            cells.append(f"{r['folds_improved']}/{r['folds']}")
        lines.append("| " + " | ".join(str(x) for x in cells) + " |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["rf", "xgb", "lgbm", "ada", "svm"],
                    choices=["rf", "xgb", "lgbm", "ada", "svm"])
    ap.add_argument("--feature-sets", nargs="+", default=DEFAULT_FEATURE_SETS)
    ap.add_argument("--eval", choices=["official", "scene-cv"], default="official")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--label-protocol", choices=["baseline", "labelled"], default="baseline")
    ap.add_argument("--max-pixels-per-class", type=int, default=100000,
                    help="Per-class cap for train and test samples (baseline scripts: 100000)")
    ap.add_argument("--svm-max-pixels-per-class", type=int, default=50000,
                    help="Per-class training cap for the SVM (baseline SVM: 250000 total)")
    ap.add_argument("--env-table", default=ENV_TABLE_CSV)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=os.path.join(repo_root, "outputs", "environmental"))
    args = ap.parse_args()

    for fs in args.feature_sets:
        feature_set_columns(fs)  # validate early
    if "baseline" not in args.feature_sets:
        args.feature_sets = ["baseline"] + args.feature_sets

    classes = [0, 1, 2, 3, 4] if args.label_protocol == "baseline" else [1, 2, 3, 4]
    env_table = load_env_table(args.env_table)
    out_dir = os.path.join(args.out_dir, f"{args.eval}_{args.label_protocol}")
    cache_dir = os.path.join(args.out_dir, "cache")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    used_cols = sorted({c for fs in args.feature_sets for c in feature_set_columns(fs)})
    print("Environmental coverage (patches with value):")
    for c in used_cols:
        print(f"  {c:28s} {env_table[c].notna().sum()}/{len(env_table)}")

    tag = f"{args.label_protocol}_cap{args.max_pixels_per_class}_seed{args.seed}_f{len(FEATURE_NAMES)}"
    results, imp_rows = [], []

    if args.eval == "official":
        ids_tr, ids_te = read_split("train"), read_split("test")
        print(f"\nSampling train pixels from {len(ids_tr)} patches...")
        Xb_tr, y_tr, p_tr = sample_pool(ids_tr, classes, args.max_pixels_per_class, args.seed,
                                        os.path.join(cache_dir, f"official_train_{tag}.npz"))
        print(f"Sampling test pixels from {len(ids_te)} patches...")
        Xb_te, y_te, p_te = sample_pool(ids_te, classes, args.max_pixels_per_class, args.seed + 1,
                                        os.path.join(cache_dir, f"official_test_{tag}.npz"))
        print(f"Train pixels {len(y_tr)} {np.bincount(y_tr, minlength=5)}  "
              f"test pixels {len(y_te)} {np.bincount(y_te, minlength=5)}")
        results += run_fold("official", Xb_tr, y_tr, p_tr, ids_tr, Xb_te, y_te, p_te, ids_te,
                            env_table, args, classes, out_dir, imp_rows)
    else:
        ids_all = read_split("train") + read_split("val") + read_split("test")
        scenes = np.array([p.rsplit("_", 1)[0] for p in ids_all])
        print(f"\nSampling pixels from all {len(ids_all)} patches ({len(set(scenes))} scenes)...")
        Xb, y, p = sample_pool(ids_all, classes, args.max_pixels_per_class, args.seed,
                               os.path.join(cache_dir, f"all_{tag}.npz"))
        pix_scene = scenes[p]
        gkf = GroupKFold(n_splits=args.cv_folds)
        for fold, (tr, te) in enumerate(gkf.split(Xb, y, groups=pix_scene)):
            print(f"\n=== Fold {fold}: {len(set(pix_scene[tr]))} train scenes, "
                  f"{len(set(pix_scene[te]))} test scenes, test classes "
                  f"{np.bincount(y[te], minlength=5)}")
            # Indices into ids_all are kept, so env lookup uses the full list
            results += run_fold(fold, Xb[tr], y[tr], p[tr], ids_all, Xb[te], y[te], p[te], ids_all,
                                env_table, args, classes, out_dir, imp_rows)

    if imp_rows:
        imp = pd.DataFrame(imp_rows).groupby(["model", "feature"])["importance"].mean().reset_index()
        imp = imp.sort_values(["model", "importance"], ascending=[True, False])
        imp.to_csv(os.path.join(out_dir, "feature_importance_env_all.csv"), index=False)
    summarise(results, out_dir, classes, args.eval)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
