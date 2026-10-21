"""
Per-class feature analysis for the five ML models of the staged experiments
(RF, XGBoost, LightGBM, AdaBoost, linear SVM) on MARIDA and MADOS.

Classes 1 Water, 2 Marine Debris, 3 Algae/Sargassum, 4 Other; class 0 is
ignored. Pixels, models, seed and class weighting are exactly those of
scripts/staged_ml (same cache, same make_model), so the models trained here
are the stage-8 models of the paper.

Parts (--parts, all by default):
  staged  How the staged experiments selected features: per-class F1 change of
          every stage against stage 1, for every model (from the existing
          outputs/staged/<weighting>/comparison/all_results.csv, no training).
  dist    Model-free view of all 30 candidate features (24 pixel features +
          6 GLCM): per-class statistics, violin plots, spectral signatures and
          one-vs-rest ROC AUC of every feature for every class.
  shap    Train the five stage-8 models (22 features, full training split) and
          explain them with SHAP, one explanation per class:
            rf, xgb, lgbm : interventional TreeSHAP (shap.TreeExplainer)
            ada           : permutation SHAP on predict_proba (sklearn AdaBoost
                            is not supported by TreeExplainer)
            svm           : exact linear SHAP, coef * (z - mean background z)
                            on the standardised features of each class's
                            one-vs-rest decision function
          All use the same background (--bg-n per class, train) and the same
          explained pixels (--shap-n per class, test). Output units differ
          (probability, log-odds, decision score), so models are compared by
          each feature's share of the class's total mean |SHAP|.
  select  Per-class feature selection from the SHAP ranking: for every model
          and class, retrain on the top-k SHAP features of that class and
          measure that class's F1 on the full test split. The chosen k is the
          smallest k within --tol of the best F1. The union of the per-class
          choices is then compared with all 22 stage-8 features.
          Refits use the full training split, so k = 22 reproduces the stage-8
          scores; AdaBoost (minutes per fit) uses a stratified --ada-frac of it,
          which keeps the class proportions and therefore the log weights.

Usage (repo root):
  python scripts/feature_analysis/per_class_features.py
  python scripts/feature_analysis/per_class_features.py --datasets MARIDA --parts shap select
Output: outputs/feature_analysis/<dataset>/{0_staged,1_distributions,2_shap,3_selection}/
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
from sklearn.metrics import f1_score, roc_auc_score

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "staged_ml"))

from common import (BAND_NAMES, CLASS_NAMES, CLASSES, DATASETS, FEATURE_NAMES, INDICES, MODELS,
                    RATIOS, SPEC_STATS, class_weights, evaluate, importances, load_data, make_model)
from glcm_features import GLCM_NAMES, split_features

OUT_ROOT = os.path.join(REPO, "outputs", "feature_analysis")
ALL_NAMES = list(FEATURE_NAMES) + GLCM_NAMES                              # 30 candidates
STAGE8 = list(BAND_NAMES) + ["NDWI"] + RATIOS + GLCM_NAMES                # 22, the paper's best set
GROUPS = {"bands": BAND_NAMES, "indices": INDICES, "ratios": RATIOS,
          "spectral statistics": SPEC_STATS, "GLCM texture": GLCM_NAMES}
CNAMES = [CLASS_NAMES[c] for c in CLASSES]
PALETTE = dict(zip(CNAMES, ["#1f77b4", "#d62728", "#2ca02c", "#8c8c8c"]))
WAVELENGTH_NM = [443, 490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]
MODEL_LABEL = {"rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM",
               "ada": "AdaBoost", "svm": "Linear SVM"}
WEIGHT_DIR = {"log": "log_weights", "balanced": "balanced_weights", "none": "no_class_weights"}
KS = [1, 2, 3, 4, 6, 8, 11, 15, 22]


def group_of(feature):
    return next(g for g, names in GROUPS.items() if feature in names)


def balanced_idx(y, n, rng):
    """At most n indices per class, sorted."""
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == k), min(n, int((y == k).sum())),
                                              replace=False) for k in range(len(CLASSES))]))


def savefig(fig, path):
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ data ---

def load(dataset, seed):
    """All 30 candidate features of the cached train/test pixels; labels 0..3."""
    X_tr, y_tr, X_te, y_te = load_data(dataset, None, seed)
    G_tr, G_te = split_features(dataset, "train"), split_features(dataset, "test")
    keep_tr, keep_te = np.isin(y_tr, CLASSES), np.isin(y_te, CLASSES)
    Xtr = np.hstack([X_tr, G_tr])[keep_tr].astype(np.float32)
    Xte = np.hstack([X_te, G_te])[keep_te].astype(np.float32)
    return Xtr, y_tr[keep_tr] - 1, Xte, y_te[keep_te] - 1


def cols(names):
    return [ALL_NAMES.index(f) for f in names]


def fit(model_name, X, y, weighting, seed):
    """Same fitting rule as common.run_stage."""
    cw = class_weights(y, weighting)
    if cw is None:
        model = make_model(model_name, seed)
        model.fit(X, y)
    elif model_name == "svm":
        model = make_model(model_name, seed, dict(enumerate(cw)))
        model.fit(X, y)
    else:
        model = make_model(model_name, seed)
        model.fit(X, y, sample_weight=cw[y])
    return model


# ------------------------------------------------------- part 0: staged ---

def part_staged(dataset, args, out):
    path = os.path.join(REPO, "outputs", "staged", WEIGHT_DIR[args.weighting], "comparison", "all_results.csv")
    if not os.path.exists(path):
        print(f"  [staged] {path} missing, skipped")
        return
    d = os.path.join(out, "0_staged"); os.makedirs(d, exist_ok=True)
    df = pd.read_csv(path)
    df = df[(df.dataset == dataset) & df.model.isin(args.models)]
    stages = [s for s in df.stage.unique() if s != "stage1_bands"]
    keep = ["stage", "model", "n_features", "f1"] + [f"f1_c{c}" for c in CLASSES] + \
           ["d_f1"] + [f"d_f1_c{c}" for c in CLASSES]
    df[keep].to_csv(os.path.join(d, "per_class_f1_vs_stage1.csv"), index=False)

    fig, axes = plt.subplots(1, len(args.models), figsize=(3.3 * len(args.models) + 1.5, 0.42 * len(stages) + 2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, args.models):
        t = df[df.model == m].set_index("stage").reindex(stages)
        mat = t[[f"d_f1_c{c}" for c in CLASSES]].to_numpy() * 100
        sns.heatmap(mat, annot=True, fmt=".1f", cmap="RdBu", center=0, vmin=-15, vmax=15, cbar=False, ax=ax,
                    xticklabels=CNAMES, yticklabels=stages, annot_kws={"size": 7})
        ax.set_title(MODEL_LABEL[m]); ax.tick_params(axis="x", rotation=40)
    fig.suptitle(f"{dataset}: per-class F1 change vs stage 1 (bands only), percentage points, "
                 f"{args.weighting} weights")
    savefig(fig, os.path.join(d, "per_class_f1_delta_by_stage.png"))


# -------------------------------------------------- part 1: distributions ---

def part_distributions(dataset, Xtr, ytr, args, out, rng):
    d = os.path.join(out, "1_distributions"); os.makedirs(d, exist_ok=True)
    t0 = time.time()
    rows = []
    for j, f in enumerate(ALL_NAMES):
        x = Xtr[:, j]
        for k, cn in enumerate(CNAMES):
            v = x[ytr == k]
            q25, med, q75 = np.percentile(v, [25, 50, 75])
            auc = roc_auc_score(ytr == k, x)
            rows.append(dict(feature=f, group=group_of(f), cls=cn, n_pixels=len(v), mean=v.mean(), std=v.std(),
                             q25=q25, median=med, q75=q75, auc_ovr=auc, separability=abs(2 * auc - 1),
                             in_stage8=f in STAGE8))
    stats = pd.DataFrame(rows)
    stats.to_csv(os.path.join(d, "per_class_feature_stats.csv"), index=False)

    # one-vs-rest AUC heatmap: >0.5 = feature is higher for this class than for the rest
    auc = stats.pivot(index="feature", columns="cls", values="auc_ovr").reindex(index=ALL_NAMES, columns=CNAMES)
    fig, ax = plt.subplots(figsize=(6.5, 11))
    sns.heatmap(auc, annot=True, fmt=".2f", cmap="RdBu_r", center=0.5, vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "one-vs-rest ROC AUC"})
    ax.set_yticklabels([f"{f}{'' if f in STAGE8 else '  (not in stage 8)'}" for f in ALL_NAMES])
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title(f"{dataset}: how well each feature alone separates each class\n"
                 "(0.5 = no information; far from 0.5 in either direction = strong)")
    savefig(fig, os.path.join(d, "ovr_auc_heatmap.png"))

    # violins, one figure per feature group, on a class-balanced sample
    idx = balanced_idx(ytr, args.plot_n, rng)
    sample = pd.DataFrame(Xtr[idx], columns=ALL_NAMES)
    sample["class"] = [CNAMES[k] for k in ytr[idx]]
    for g, names in GROUPS.items():
        ncol = min(4, len(names)); nrow = int(np.ceil(len(names) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow), squeeze=False)
        for ax, f in zip(axes.ravel(), names):
            lo, hi = np.percentile(sample[f], [0.5, 99.5])
            sns.violinplot(data=sample.assign(**{f: sample[f].clip(lo, hi)}), x="class", y=f, hue="class",
                           order=CNAMES, palette=PALETTE, cut=0, inner="quartile", linewidth=0.8,
                           legend=False, ax=ax)
            ax.set_title(f); ax.set_xlabel(""); ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=25, labelsize=8)
        for ax in axes.ravel()[len(names):]:
            ax.axis("off")
        fig.suptitle(f"{dataset}: {g} per class ({args.plot_n:,} training pixels per class, 0.5-99.5 % clipped)")
        fig.tight_layout()
        savefig(fig, os.path.join(d, f"violins_{g.replace(' ', '_')}.png"))

    # spectral signature: median reflectance per band, IQR shaded
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, cn in enumerate(CNAMES):
        q = np.percentile(Xtr[ytr == k][:, :len(BAND_NAMES)], [25, 50, 75], axis=0)
        ax.plot(WAVELENGTH_NM, q[1], "-o", color=PALETTE[cn], label=cn, ms=4)
        ax.fill_between(WAVELENGTH_NM, q[0], q[2], color=PALETTE[cn], alpha=0.15)
    top = ax.secondary_xaxis("top")
    top.set_xticks(WAVELENGTH_NM)
    top.set_xticklabels(BAND_NAMES, rotation=90, fontsize=7)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("reflectance (median, IQR shaded)")
    ax.set_title(f"{dataset}: spectral signature of each class (training pixels)")
    ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, os.path.join(d, "spectral_signatures.png"))
    print(f"  [dist] done in {time.time() - t0:.0f}s", flush=True)
    return stats


# ----------------------------------------------------------- part 2: SHAP ---

def shap_values(model_name, model, X_exp, X_bg, seed):
    """(n, n_features, n_classes) SHAP values of each class's model output."""
    import shap
    if model_name == "svm":
        scaler, clf = model.named_steps["scaler"], model.named_steps["clf"]
        z, zb = scaler.transform(X_exp), scaler.transform(X_bg)
        return (z - zb.mean(axis=0))[:, :, None] * clf.coef_.T[None]
    if model_name == "ada":
        explainer = shap.PermutationExplainer(model.predict_proba, shap.maskers.Independent(X_bg, len(X_bg)),
                                              seed=seed)
        return explainer(X_exp, max_evals=10 * (2 * X_exp.shape[1] + 1), silent=True).values
    explainer = shap.TreeExplainer(model, data=X_bg, feature_perturbation="interventional")
    return np.asarray(explainer.shap_values(X_exp, check_additivity=False))


def part_shap(dataset, Xtr, ytr, Xte, yte, args, out, rng):
    import shap
    d = os.path.join(out, "2_shap"); os.makedirs(d, exist_ok=True)
    c8 = cols(STAGE8)
    Xtr8, Xte8 = Xtr[:, c8], Xte[:, c8]
    bg = Xtr8[balanced_idx(ytr, args.bg_n, rng)]
    exp_idx = balanced_idx(yte, args.shap_n, rng)
    X_exp, y_exp = Xte8[exp_idx], yte[exp_idx]
    np.savez_compressed(os.path.join(d, "explained_pixels.npz"), X=X_exp, y=y_exp, features=STAGE8)

    metrics, shares, values = [], [], {}
    for m in args.models:
        t0 = time.time()
        model = fit(m, Xtr8, ytr, args.weighting, args.seed)
        t_fit = time.time() - t0
        pred = np.asarray(model.predict(Xte8)).ravel().astype(int)
        met, _ = evaluate(yte, pred)
        metrics.append({"model": m, "n_train": len(ytr), "n_test": len(yte), **met})
        t0 = time.time()
        V = shap_values(m, model, X_exp, bg, args.seed)
        values[m] = V
        builtin = importances(model, m, STAGE8)
        print(f"  [shap] {m:5s} macro F1 {met['f1']:.3f} debris F1 {met['f1_c2']:.3f} "
              f"(fit {t_fit:.0f}s, SHAP {time.time() - t0:.0f}s)", flush=True)
        for k, cn in enumerate(CNAMES):
            mean_abs = np.abs(V[:, :, k]).mean(axis=0)
            own = np.abs(V[y_exp == k][:, :, k]).mean(axis=0)
            mean_signed_own = V[y_exp == k][:, :, k].mean(axis=0)
            for j, f in enumerate(STAGE8):
                shares.append(dict(model=m, cls=cn, feature=f, group=group_of(f), mean_abs_shap=mean_abs[j],
                                   share=mean_abs[j] / mean_abs.sum(), own_class_mean_abs=own[j],
                                   own_class_mean_shap=mean_signed_own[j], builtin_importance=builtin[j]))
    np.savez_compressed(os.path.join(d, "shap_values.npz"), **values)
    metrics = pd.DataFrame(metrics); metrics.to_csv(os.path.join(d, "stage8_metrics.csv"), index=False)
    shares = pd.DataFrame(shares)
    shares["rank"] = shares.groupby(["model", "cls"])["share"].rank(ascending=False, method="first").astype(int)
    shares.to_csv(os.path.join(d, "shap_per_class.csv"), index=False)

    # beeswarm per model: one panel per class
    for m in args.models:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        for ax, (k, cn) in zip(axes.ravel(), enumerate(CNAMES)):
            e = shap.Explanation(values=values[m][:, :, k], data=X_exp, feature_names=STAGE8)
            shap.plots.beeswarm(e, max_display=12, ax=ax, show=False, plot_size=None, s=6)
            ax.set_title(f"{cn}", color=PALETTE[cn], fontweight="bold")
        fig.suptitle(f"{dataset} | {MODEL_LABEL[m]} | SHAP per class (stage-8 features). Right = pushes "
                     f"towards the class; colour = feature value", fontsize=12)
        fig.tight_layout()
        savefig(fig, os.path.join(d, f"beeswarm_{m}.png"))

    # one class across all models
    for k, cn in enumerate(CNAMES):
        n = len(args.models)
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 6))
        for ax, m in zip(np.atleast_1d(axes), args.models):
            e = shap.Explanation(values=values[m][:, :, k], data=X_exp, feature_names=STAGE8)
            shap.plots.beeswarm(e, max_display=10, ax=ax, show=False, plot_size=None, s=5,
                                color_bar=(m == args.models[-1]))
            ax.set_title(MODEL_LABEL[m])
        fig.suptitle(f"{dataset} | {cn}: SHAP of every model", color=PALETTE[cn], fontweight="bold")
        fig.tight_layout()
        savefig(fig, os.path.join(d, f"beeswarm_class{CLASSES[k]}_{cn.split('/')[0].replace(' ', '_')}_all_models.png"))

    # share heatmaps: models x features, one panel per class, features in consensus order
    fig, axes = plt.subplots(4, 1, figsize=(14, 3.0 + 1.6 * len(args.models)), sharex=False)
    for ax, cn in zip(axes, CNAMES):
        t = shares[shares.cls == cn].pivot(index="model", columns="feature", values="share").reindex(args.models)
        order = t.mean().sort_values(ascending=False).index
        sns.heatmap(t[order] * 100, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=40, ax=ax, cbar=False,
                    yticklabels=[MODEL_LABEL[m] for m in args.models], annot_kws={"size": 7})
        ax.set_title(f"{cn}: share of mean |SHAP| (%), features sorted by mean over models",
                     color=PALETTE[cn], fontweight="bold", fontsize=10)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    savefig(fig, os.path.join(d, "shap_share_heatmaps.png"))

    # group view: how much each feature group contributes per class and model
    grp = shares.groupby(["model", "cls", "group"])["share"].sum().unstack("group").fillna(0)
    grp = grp[[g for g in GROUPS if g in grp.columns]]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, cn in zip(axes, CNAMES):
        g = grp.xs(cn, level="cls").reindex(args.models) * 100
        g.index = [MODEL_LABEL[m] for m in g.index]
        g.plot.barh(stacked=True, ax=ax, legend=(cn == CNAMES[-1]), width=0.75,
                    color=["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"][:g.shape[1]])
        ax.set_title(cn, color=PALETTE[cn], fontweight="bold"); ax.set_xlabel("% of mean |SHAP|")
        ax.set_xlim(0, 100)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle(f"{dataset}: contribution of each feature group per class")
    fig.tight_layout()
    savefig(fig, os.path.join(d, "shap_group_shares.png"))
    return shares, metrics


# ------------------------------------------------------ part 3: selection ---

def stratified_idx(y, frac, rng):
    """A fraction of every class, so class proportions (and log weights) stay as in the full split."""
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == k), max(1, int(round(frac * (y == k).sum()))),
                                              replace=False) for k in range(len(CLASSES))]))


def part_selection(dataset, Xtr, ytr, Xte, yte, shares, args, out, rng):
    d = os.path.join(out, "3_selection"); os.makedirs(d, exist_ok=True)
    train_idx = {m: (stratified_idx(ytr, args.ada_frac, rng) if m == "ada" and args.ada_frac < 1
                     else np.arange(len(ytr))) for m in args.models}
    cache = {}

    def scores(m, feats):
        key = (m, tuple(sorted(feats)))
        if key not in cache:
            c = cols(feats)
            model = fit(m, Xtr[np.ix_(train_idx[m], c)], ytr[train_idx[m]], args.weighting, args.seed)
            pred = np.asarray(model.predict(Xte[:, c])).ravel().astype(int)
            cache[key] = f1_score(yte, pred, average=None, labels=list(range(len(CLASSES))), zero_division=0)
        return cache[key]

    curves, chosen, final = [], [], []
    for m in args.models:
        t0 = time.time()
        union = []
        for k, cn in enumerate(CNAMES):
            ranking = shares[(shares.model == m) & (shares.cls == cn)].sort_values("rank").feature.tolist()
            f1s = {}
            for n in KS:
                f1s[n] = scores(m, ranking[:n])[k]
                curves.append(dict(model=m, cls=cn, k=n, f1=f1s[n], features="|".join(ranking[:n])))
            best = max(f1s.values())
            k_sel = min(n for n in KS if f1s[n] >= best - args.tol)
            chosen.append(dict(model=m, cls=cn, k_selected=k_sel, f1_selected=f1s[k_sel], f1_best=best,
                               f1_all22=f1s[22], n_train=len(train_idx[m]), features="|".join(ranking[:k_sel])))
            union += [f for f in ranking[:k_sel] if f not in union]
        for label, feats in [("union of per-class selections", union), ("all 22 stage-8 features", STAGE8)]:
            f = scores(m, feats)
            final.append(dict(model=m, feature_set=label, n_features=len(feats), n_train=len(train_idx[m]),
                              macro_f1=f.mean(),
                              **{f"f1_{cn}": v for cn, v in zip(CNAMES, f)}, features="|".join(feats)))
        print(f"  [select] {m:5s} union {len(union)} features: "
              f"macro F1 {final[-2]['macro_f1']:.3f} vs all 22 {final[-1]['macro_f1']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    curves, chosen, final = pd.DataFrame(curves), pd.DataFrame(chosen), pd.DataFrame(final)
    curves.to_csv(os.path.join(d, "topk_curves.csv"), index=False)
    chosen.to_csv(os.path.join(d, "selected_per_class.csv"), index=False)
    final.to_csv(os.path.join(d, "union_vs_all.csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = dict(zip(MODELS, sns.color_palette("tab10", len(MODELS))))
    for ax, cn in zip(axes.ravel(), CNAMES):
        for m in args.models:
            c = curves[(curves.model == m) & (curves.cls == cn)]
            ax.plot(c.k, c.f1, "-o", ms=4, color=colors[m], label=MODEL_LABEL[m])
            s = chosen[(chosen.model == m) & (chosen.cls == cn)].iloc[0]
            ax.plot(s.k_selected, s.f1_selected, "*", ms=15, color=colors[m], mec="k")
        ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS)
        ax.set_title(cn, color=PALETTE[cn], fontweight="bold"); ax.grid(alpha=0.3)
        ax.set_xlabel("k = number of top SHAP features of this class"); ax.set_ylabel(f"{cn} F1 (test)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{dataset}: per-class F1 when each model keeps only its top-k SHAP features for that class\n"
                 f"(star = smallest k within {args.tol} of the best; full training split, AdaBoost "
                 f"{args.ada_frac:.0%} stratified)")
    fig.tight_layout()
    savefig(fig, os.path.join(d, "topk_curves.png"))
    return chosen, final


# ---------------------------------------------------------------- summary ---

def write_summary(dataset, stats, shares, metrics, chosen, final, args, out):
    L = [f"# Per-class feature analysis: {dataset}", "",
         f"Weighting: {args.weighting}. Generated by `scripts/feature_analysis/per_class_features.py`.", ""]
    if stats is not None:
        L += ["## Strongest single features per class (one-vs-rest ROC AUC, all 30 candidates)", "",
              "| Class | Top 6 features (AUC; >0.5 = higher in this class) |", "|---|---|"]
        for cn in CNAMES:
            t = stats[stats.cls == cn].sort_values("separability", ascending=False).head(6)
            L.append(f"| {cn} | " + ", ".join(f"{r.feature} ({r.auc_ovr:.2f})" for r in t.itertuples()) + " |")
    if metrics is not None:
        L += ["", "## Stage-8 models (full training split)", "",
              "| Model | macro F1 | " + " | ".join(f"{cn} F1" for cn in CNAMES) + " | kappa |",
              "|---|---|" + "---|" * len(CNAMES) + "---|"]
        for r in metrics.itertuples():
            L.append(f"| {MODEL_LABEL[r.model]} | {r.f1:.3f} | " +
                     " | ".join(f"{getattr(r, f'f1_c{c}'):.3f}" for c in CLASSES) + f" | {r.kappa:.3f} |")
    if shares is not None:
        L += ["", "## Top 5 SHAP features per class and model (share of mean |SHAP|)", ""]
        for cn in CNAMES:
            L += [f"### {cn}", "", "| Model | 1 | 2 | 3 | 4 | 5 |", "|---|---|---|---|---|---|"]
            for m in args.models:
                t = shares[(shares.model == m) & (shares.cls == cn)].sort_values("rank").head(5)
                L.append(f"| {MODEL_LABEL[m]} | " + " | ".join(f"{r.feature} ({r.share:.0%})"
                                                               for r in t.itertuples()) + " |")
            L.append("")
    if chosen is not None:
        L += ["## Per-class selection (smallest top-k within tolerance of the best)", "",
              "| Model | Class | k | F1 at k | best F1 | F1 with all 22 | features |", "|---|---|---|---|---|---|---|"]
        for r in chosen.itertuples():
            L.append(f"| {MODEL_LABEL[r.model]} | {r.cls} | {r.k_selected} | {r.f1_selected:.3f} | "
                     f"{r.f1_best:.3f} | {r.f1_all22:.3f} | {r.features.replace('|', ', ')} |")
        L += ["", "## Union of per-class selections vs all 22 features", "",
              "| Model | feature set | n | macro F1 | " + " | ".join(CNAMES) + " |",
              "|---|---|---|---|" + "---|" * len(CNAMES)]
        for _, r in final.iterrows():
            L.append(f"| {MODEL_LABEL[r.model]} | {r.feature_set} | {r.n_features} | {r.macro_f1:.3f} | " +
                     " | ".join(f"{r[f'f1_{cn}']:.3f}" for cn in CNAMES) + " |")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--parts", nargs="+", default=["staged", "dist", "shap", "select"],
                    choices=["staged", "dist", "shap", "select"])
    ap.add_argument("--weighting", choices=["balanced", "log", "none"], default="log",
                    help="Class weighting (default log, the paper's main setting)")
    ap.add_argument("--shap-n", type=int, default=500, help="Explained test pixels per class")
    ap.add_argument("--bg-n", type=int, default=25, help="SHAP background training pixels per class")
    ap.add_argument("--plot-n", type=int, default=3000, help="Training pixels per class in the violin plots")
    ap.add_argument("--ada-frac", type=float, default=0.2,
                    help="Stratified fraction of the training split for the AdaBoost top-k refits (others use all)")
    ap.add_argument("--tol", type=float, default=0.01, help="F1 tolerance for choosing k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", default=OUT_ROOT)
    args = ap.parse_args()

    for dataset in args.datasets:
        out = os.path.join(args.out_root, dataset); os.makedirs(out, exist_ok=True)
        print(f"\n=== {dataset}", flush=True)
        rng = np.random.default_rng(args.seed)
        stats = shares = metrics = chosen = final = None
        if "staged" in args.parts:
            part_staged(dataset, args, out)
        if {"dist", "shap", "select"} & set(args.parts):
            Xtr, ytr, Xte, yte = load(dataset, args.seed)
            print(f"  train {len(ytr):,} px {np.bincount(ytr, minlength=4)} | "
                  f"test {len(yte):,} px {np.bincount(yte, minlength=4)}", flush=True)
        if "dist" in args.parts:
            stats = part_distributions(dataset, Xtr, ytr, args, out, np.random.default_rng(args.seed))
        shap_csv = os.path.join(out, "2_shap", "shap_per_class.csv")
        if "shap" in args.parts:
            shares, metrics = part_shap(dataset, Xtr, ytr, Xte, yte, args, out, np.random.default_rng(args.seed))
        elif os.path.exists(shap_csv):
            shares = pd.read_csv(shap_csv)
            metrics = pd.read_csv(os.path.join(out, "2_shap", "stage8_metrics.csv"))
        if stats is None and os.path.exists(os.path.join(out, "1_distributions", "per_class_feature_stats.csv")):
            stats = pd.read_csv(os.path.join(out, "1_distributions", "per_class_feature_stats.csv"))
        if "select" in args.parts:
            if shares is None:
                raise SystemExit("select needs the SHAP ranking: run --parts shap first")
            chosen, final = part_selection(dataset, Xtr, ytr, Xte, yte, shares, args, out,
                                           np.random.default_rng(args.seed))
        write_summary(dataset, stats, shares, metrics, chosen, final, args, out)
        with open(os.path.join(out, "run_config.json"), "w") as f:
            json.dump({**vars(args), "stage8_features": STAGE8, "candidate_features": ALL_NAMES}, f, indent=2)


if __name__ == "__main__":
    main()
