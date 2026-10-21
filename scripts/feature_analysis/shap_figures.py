"""
SHAP figure set for comparing the five stage-8 ML models, per dataset, per
class and across datasets. Reads the SHAP values saved by
per_class_features.py (part shap); only folder 05 and the interaction
heatmaps in folder 04 refit the models (--with-models).

Output: outputs/feature_analysis/shap_figures/
  01_global_importance/        all classes together, per model
  02_class_specific/           one class at a time
  03_cross_model_comparison/   how much the models agree (and MARIDA vs MADOS)
  04_dependence_and_interactions/  feature value vs SHAP, redundancy, interactions
  05_individual_predictions/   waterfall and decision plots of single pixels
  06_reports/                  index of every figure, tables (CSV)

Every figure exists for MARIDA and MADOS (<dataset>/ subfolders); files at
a folder's top level compare the two datasets.

Usage (repo root):
  python scripts/feature_analysis/shap_figures.py                  # 01-04 + 06 from saved SHAP
  python scripts/feature_analysis/shap_figures.py --with-models    # also 05 + interaction heatmaps
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from per_class_features import (CLASSES, CNAMES, DATASETS, GROUPS, MODEL_LABEL, MODELS, OUT_ROOT, PALETTE,
                                STAGE8, balanced_idx, cols, fit, group_of, load, savefig)

FIG_ROOT = os.path.join(OUT_ROOT, "shap_figures")
FOLDERS = ["01_global_importance", "02_class_specific", "03_cross_model_comparison",
           "04_dependence_and_interactions", "05_individual_predictions", "06_reports"]
SHORT = {"Water": "Water", "Marine Debris": "Debris", "Algae/Sargassum": "Algae", "Other": "Other"}
SLUG = {cn: f"c{c}_{SHORT[cn]}" for c, cn in zip(CLASSES, CNAMES)}
MCOL = dict(zip(MODELS, sns.color_palette("tab10", len(MODELS))))
GCOL = dict(zip(GROUPS, ["#dd8452", "#55a868", "#c44e52", "#937860", "#4c72b0"]))
REDUNDANT = [("NDWI", "nir/green"), ("B8", "B8A"), ("glcm_energy", "glcm_asm"),
             ("glcm_contrast", "glcm_dissimilarity"), ("B11", "B12")]
INDEX = []          # (folder, path, description) for 06_reports/README.md


def long_path(path):
    """Windows refuses paths over 260 characters unless they carry the \\\\?\\ prefix."""
    path = os.path.abspath(path)
    return "\\\\?\\" + path if os.name == "nt" and len(path) > 240 else path


def out(folder, dataset, name):
    d = os.path.join(FIG_ROOT, folder, dataset) if dataset else os.path.join(FIG_ROOT, folder)
    os.makedirs(long_path(d), exist_ok=True)
    return os.path.join(d, name)


def save(fig, folder, dataset, name, desc):
    path = out(folder, dataset, name)
    savefig(fig, long_path(path))
    INDEX.append((folder, os.path.relpath(path, os.path.join(FIG_ROOT, folder)).replace("\\", "/"), desc))


def load_shap(dataset, models):
    d = os.path.join(OUT_ROOT, dataset, "2_shap")
    V = np.load(os.path.join(d, "shap_values.npz"))
    E = np.load(os.path.join(d, "explained_pixels.npz"))
    shares = pd.read_csv(os.path.join(d, "shap_per_class.csv"))
    return {m: V[m] for m in models}, E["X"], E["y"], shares


def mean_abs(V):
    """(n, f, k) -> (f, k) mean |SHAP|."""
    return np.abs(V).mean(axis=0)


def global_share(V):
    """Share of each feature in the model's total mean |SHAP| over all classes."""
    a = mean_abs(V).sum(axis=1)
    return a / a.sum()


def true_class(V, y):
    """SHAP of each pixel's own (true) class output: (n, f)."""
    return V[np.arange(len(y)), :, y]


def beeswarm(ax, values, X, title, n=12, color_bar=True):
    import shap
    shap.plots.beeswarm(shap.Explanation(values=values, data=X, feature_names=STAGE8), max_display=n, ax=ax,
                        show=False, plot_size=None, s=6, color_bar=color_bar)
    ax.set_title(title, fontsize=10)


# ------------------------------------------------- 01 global importance ---

def fig_global(ds, V, X, y, models):
    F = "01_global_importance"
    G = pd.DataFrame({m: global_share(V[m]) * 100 for m in models}, index=STAGE8)
    order = G.mean(axis=1).sort_values(ascending=False).index

    for m in models:
        a = mean_abs(V[m]); a = a / a.sum() * 100
        t = pd.DataFrame(a, index=STAGE8, columns=CNAMES).loc[order[::-1]]
        fig, ax = plt.subplots(figsize=(8, 8))
        t.plot.barh(stacked=True, ax=ax, color=[PALETTE[c] for c in CNAMES], width=0.8)
        ax.set_xlabel("% of total mean |SHAP| (all classes)")
        ax.set_title(f"{ds} | {MODEL_LABEL[m]}: global SHAP importance, split by class output")
        save(fig, F, ds, f"bar_global_by_class_{m}.png",
             f"{MODEL_LABEL[m]}: each feature's share of mean |SHAP| over all four class outputs, coloured by class")

        fig, ax = plt.subplots(figsize=(8, 7))
        beeswarm(ax, true_class(V[m], y), X, f"{ds} | {MODEL_LABEL[m]}: SHAP of each pixel's true class", n=15)
        save(fig, F, ds, f"beeswarm_true_class_{m}.png",
             f"{MODEL_LABEL[m]}: beeswarm of the SHAP value for each pixel's own class (right = supports the correct class)")

    fig, ax = plt.subplots(figsize=(7.5, 9))
    sns.heatmap(G.loc[order], annot=True, fmt=".1f", cmap="Blues", ax=ax,
                xticklabels=[MODEL_LABEL[m] for m in models], cbar_kws={"label": "% of model's total mean |SHAP|"})
    ax.set_title(f"{ds}: normalised global SHAP importance per model\n(columns sum to 100 %)")
    save(fig, F, ds, "heatmap_global_share_models.png",
         "Global share of mean |SHAP| per feature (rows) and model (columns), the other team's Fig. 11 layout")

    fig, ax = plt.subplots(figsize=(11, 6))
    G.loc[order].plot.bar(ax=ax, color=[MCOL[m] for m in models], width=0.85)
    ax.set_ylabel("% of total mean |SHAP|"); ax.legend([MODEL_LABEL[m] for m in models])
    ax.set_title(f"{ds}: global SHAP share, models side by side"); ax.tick_params(axis="x", rotation=60)
    save(fig, F, ds, "bar_global_share_models.png", "Same as the heatmap, as grouped bars")

    grp = G.groupby([group_of(f) for f in G.index]).sum()
    grp = grp.loc[[g for g in GROUPS if g in grp.index]]
    fig, ax = plt.subplots(figsize=(8, 4))
    grp.T.rename(index=MODEL_LABEL).plot.barh(stacked=True, ax=ax, color=[GCOL[g] for g in grp.index], width=0.75)
    ax.set_xlim(0, 100); ax.set_xlabel("% of total mean |SHAP|"); ax.legend(loc="center left", bbox_to_anchor=(1, .5))
    ax.set_title(f"{ds}: global SHAP share by feature group")
    save(fig, F, ds, "bar_global_feature_groups.png", "Global SHAP share per feature group (bands, NDWI, ratios, GLCM) per model")

    cum = G.loc[order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in models:
        s = np.sort(G[m].to_numpy())[::-1].cumsum()
        ax.plot(range(1, len(s) + 1), s, "-o", ms=3, color=MCOL[m], label=MODEL_LABEL[m])
    ax.axhline(80, ls="--", c="gray", lw=.8)
    ax.set_xlabel("number of top features"); ax.set_ylabel("cumulative % of mean |SHAP|"); ax.legend()
    ax.set_title(f"{ds}: how concentrated each model's importance is"); ax.grid(alpha=.3)
    save(fig, F, ds, "cumulative_importance.png",
         "Cumulative global SHAP share against number of features; steep = few features dominate")
    return G


def fig_global_datasets(Gs, models):
    F = "01_global_importance"
    if len(Gs) < 2:
        return
    (d1, G1), (d2, G2) = list(Gs.items())[:2]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.4))
    for ax, m in zip(np.atleast_1d(axes), models):
        ax.scatter(G1[m], G2[m], s=18, color=MCOL[m])
        for f in STAGE8:
            if max(G1.at[f, m], G2.at[f, m]) > 6:
                ax.annotate(f, (G1.at[f, m], G2.at[f, m]), fontsize=7)
        lim = max(G1[m].max(), G2[m].max()) * 1.08
        ax.plot([0, lim], [0, lim], "--", c="gray", lw=.8); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        rho = spearmanr(G1[m], G2[m])[0]
        ax.set_title(f"{MODEL_LABEL[m]} (Spearman {rho:.2f})"); ax.set_xlabel(f"{d1} share %"); ax.set_ylabel(f"{d2} share %")
    fig.suptitle("Global SHAP share: MARIDA vs MADOS (on the diagonal = same importance on both datasets)")
    fig.tight_layout()
    save(fig, F, None, "global_share_MARIDA_vs_MADOS.png", "Each feature's global share on MARIDA (x) vs MADOS (y), per model")


# ------------------------------------------------- 02 class specific ---

def fig_class(ds, V, X, y, models):
    F = "02_class_specific"
    for k, cn in enumerate(CNAMES):
        sl = SLUG[cn]
        S = pd.DataFrame({m: mean_abs(V[m])[:, k] / mean_abs(V[m])[:, k].sum() * 100 for m in models}, index=STAGE8)
        order = S.mean(axis=1).sort_values(ascending=False).index

        fig, ax = plt.subplots(figsize=(11, 5.5))
        S.loc[order].plot.bar(ax=ax, color=[MCOL[m] for m in models], width=0.85)
        ax.set_ylabel(f"% of mean |SHAP| for {cn}"); ax.legend([MODEL_LABEL[m] for m in models])
        ax.set_title(f"{ds} | {cn}: SHAP share per feature, models side by side", color=PALETTE[cn])
        ax.tick_params(axis="x", rotation=60)
        save(fig, F, ds, f"{sl}_bar_share_by_model.png", f"{cn}: each feature's share of mean |SHAP| for this class output, per model")

        fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 6))
        for ax, m in zip(np.atleast_1d(axes), models):
            beeswarm(ax, V[m][:, :, k], X, MODEL_LABEL[m], n=10, color_bar=(m == models[-1]))
        fig.suptitle(f"{ds} | {cn}: SHAP beeswarm of every model (all explained pixels)", color=PALETTE[cn])
        fig.tight_layout()
        save(fig, F, ds, f"{sl}_beeswarm_all_models.png", f"{cn}: beeswarm per model over all explained pixels")

        own = y == k
        fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 6))
        for ax, m in zip(np.atleast_1d(axes), models):
            beeswarm(ax, V[m][own][:, :, k], X[own], MODEL_LABEL[m], n=10, color_bar=(m == models[-1]))
        fig.suptitle(f"{ds} | {cn}: SHAP beeswarm on {cn} pixels only (why true {cn} pixels are recognised)",
                     color=PALETTE[cn])
        fig.tight_layout()
        save(fig, F, ds, f"{sl}_beeswarm_own_pixels.png", f"{cn}: beeswarm restricted to pixels that truly are {cn}")

        # signed mean SHAP on own pixels: which features push true members towards / away from their class
        M = pd.DataFrame({m: V[m][own][:, :, k].mean(axis=0) for m in models}, index=STAGE8)
        M = M.div(M.abs().sum(axis=0), axis=1) * 100
        fig, ax = plt.subplots(figsize=(7.5, 9))
        sns.heatmap(M.loc[order], annot=True, fmt=".0f", cmap="RdBu_r", center=0, ax=ax,
                    xticklabels=[MODEL_LABEL[m] for m in models], cbar_kws={"label": "signed share (%)"})
        ax.set_title(f"{ds} | {cn}: mean signed SHAP on true {cn} pixels\n"
                     "(red = pushes towards the class, blue = against it)", color=PALETTE[cn])
        save(fig, F, ds, f"{sl}_signed_contribution.png",
             f"{cn}: direction of each feature's average push on true {cn} pixels")

        # SHAP value distributions by true class (for this class output, RF-free: every model)
        top = order[:6]
        rows = []
        for m in models:
            for j, f in enumerate(STAGE8):
                if f in top:
                    rows += [dict(model=MODEL_LABEL[m], feature=f, shap=v, true=CNAMES[t])
                             for v, t in zip(V[m][:, j, k] / (np.abs(V[m][:, :, k]).sum(1).mean() + 1e-12), y)]
        df = pd.DataFrame(rows)
        g = sns.catplot(data=df, x="feature", y="shap", hue="true", col="model", kind="box", order=list(top),
                        hue_order=CNAMES, palette=PALETTE, showfliers=False, height=3.6, aspect=1.2, sharey=False)
        g.set_xticklabels(rotation=45); g.set_ylabels("SHAP (scaled)")
        g.figure.suptitle(f"{ds} | {cn} output: SHAP of the top 6 features, split by the pixel's true class",
                          y=1.03, color=PALETTE[cn])
        save(g.figure, F, ds, f"{sl}_shap_by_true_class_box.png",
             f"{cn} output: SHAP distributions of the top 6 features for pixels of each true class")

    for m in models:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        for ax, (k, cn) in zip(axes.ravel(), enumerate(CNAMES)):
            beeswarm(ax, V[m][:, :, k], X, cn)
        fig.suptitle(f"{ds} | {MODEL_LABEL[m]}: SHAP beeswarm per class output")
        fig.tight_layout()
        save(fig, F, ds, f"model_{m}_beeswarm_4_classes.png", f"{MODEL_LABEL[m]}: one beeswarm per class output")

        a = mean_abs(V[m]); a = a / a.sum(axis=0) * 100
        t = pd.DataFrame(a, index=STAGE8, columns=CNAMES)
        fig, ax = plt.subplots(figsize=(6, 9))
        sns.heatmap(t.loc[t.mean(axis=1).sort_values(ascending=False).index], annot=True, fmt=".0f", cmap="Reds",
                    ax=ax, cbar_kws={"label": "% of class output's mean |SHAP|"})
        ax.set_title(f"{ds} | {MODEL_LABEL[m]}: class fingerprint\n(each column sums to 100 %)")
        save(fig, F, ds, f"model_{m}_class_fingerprint.png", f"{MODEL_LABEL[m]}: feature share per class, side by side")

    # the other team's Fig. 53 layout: features x (model, class)
    T = pd.DataFrame({f"{MODEL_LABEL[m]}\n{SHORT[cn]}": mean_abs(V[m])[:, k] / mean_abs(V[m])[:, k].sum() * 100
                      for m in models for k, cn in enumerate(CNAMES)}, index=STAGE8)
    T = T.loc[T.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(17, 10))
    sns.heatmap(T, annot=True, fmt=".0f", cmap="Blues", ax=ax, annot_kws={"size": 7},
                cbar_kws={"label": "% of mean |SHAP| within (model, class output)"})
    for i in range(1, len(models)):
        ax.axvline(4 * i, color="white", lw=3)
    ax.set_title(f"{ds}: class-wise SHAP share across all models and class outputs (each column sums to 100 %)")
    save(fig, F, ds, "heatmap_all_models_all_classes.png", "Features x (model, class) shares, the other team's Fig. 53 layout")


# ---------------------------------------------- 03 cross-model comparison ---

def ranks(V, k):
    a = mean_abs(V)[:, k]
    return pd.Series(a, index=STAGE8).rank(ascending=False)


def fig_cross(ds, V, models, shares):
    F = "03_cross_model_comparison"
    rows = []
    for k, cn in enumerate(CNAMES):
        R = pd.DataFrame({m: ranks(V[m], k) for m in models})
        rho = R.corr(method="spearman")
        top5 = {m: set(R[m].nsmallest(5).index) for m in models}
        jac = pd.DataFrame([[len(top5[a] & top5[b]) / len(top5[a] | top5[b]) for b in models] for a in models],
                           index=models, columns=models)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        lab = [MODEL_LABEL[m] for m in models]
        sns.heatmap(rho, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="RdBu_r", ax=axes[0], xticklabels=lab, yticklabels=lab)
        axes[0].set_title("Spearman correlation of feature ranks")
        sns.heatmap(jac, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Greens", ax=axes[1], xticklabels=lab, yticklabels=lab)
        axes[1].set_title("Overlap of top-5 features (Jaccard)")
        fig.suptitle(f"{ds} | {cn}: do the models agree on which features matter?", color=PALETTE[cn])
        fig.tight_layout()
        save(fig, F, ds, f"{SLUG[cn]}_model_agreement.png", f"{cn}: rank correlation and top-5 overlap between models")
        for a in models:
            for b in models:
                if a < b:
                    rows.append(dict(dataset=ds, cls=cn, model_a=a, model_b=b, spearman=rho.at[a, b], top5_jaccard=jac.at[a, b]))

        # bump chart: rank of the 10 consensus features in each model
        top = R.mean(axis=1).nsmallest(10).index
        fig, ax = plt.subplots(figsize=(9, 6))
        for f in top:
            ax.plot(range(len(models)), R.loc[f, models], "-o", lw=2, label=f)
            ax.annotate(f, (len(models) - 1 + .08, R.at[f, models[-1]]), fontsize=8, va="center")
        ax.set_xticks(range(len(models))); ax.set_xticklabels([MODEL_LABEL[m] for m in models])
        ax.invert_yaxis(); ax.set_ylabel("rank (1 = most important)"); ax.set_xlim(-.3, len(models) + .6)
        ax.set_title(f"{ds} | {cn}: rank of the 10 consensus features in each model", color=PALETTE[cn]); ax.grid(alpha=.3)
        save(fig, F, ds, f"{SLUG[cn]}_rank_bump_chart.png", f"{cn}: how the rank of the top-10 features changes between models")

    # SHAP vs built-in importance (impurity / gain / |coef|)
    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4.3))
    for ax, m in zip(np.atleast_1d(axes), models):
        t = shares[shares.model == m].groupby("feature").agg(shap=("mean_abs_shap", "sum"), builtin=("builtin_importance", "first"))
        t["shap"] = t.shap / t.shap.sum() * 100; t["builtin"] *= 100
        ax.scatter(t.builtin, t.shap, color=MCOL[m], s=18)
        for f, r in t.iterrows():
            if max(r.shap, r.builtin) > 6:
                ax.annotate(f, (r.builtin, r.shap), fontsize=7)
        lim = max(t.max()) * 1.08
        ax.plot([0, lim], [0, lim], "--", c="gray", lw=.8)
        ax.set_title(f"{MODEL_LABEL[m]} (Spearman {spearmanr(t.builtin, t.shap)[0]:.2f})")
        ax.set_xlabel("built-in importance %"); ax.set_ylabel("global SHAP share %")
    fig.suptitle(f"{ds}: SHAP vs the model's built-in importance (impurity / split gain / |coef|)")
    fig.tight_layout()
    save(fig, F, ds, "shap_vs_builtin_importance.png",
         "Global SHAP share vs the model's own importance; off-diagonal features are over/under-rated by the built-in measure")

    # one heatmap per feature: model x class
    S = pd.DataFrame([dict(model=MODEL_LABEL[m], cls=SHORT[cn], feature=f, share=s * 100)
                      for m in models for k, cn in enumerate(CNAMES)
                      for f, s in zip(STAGE8, mean_abs(V[m])[:, k] / mean_abs(V[m])[:, k].sum())])
    order = S.groupby("feature").share.mean().sort_values(ascending=False).index
    fig, axes = plt.subplots(4, 6, figsize=(22, 13))
    for ax, f in zip(axes.ravel(), order):
        t = S[S.feature == f].pivot(index="model", columns="cls", values="share").reindex(
            index=[MODEL_LABEL[m] for m in models], columns=[SHORT[c] for c in CNAMES])
        sns.heatmap(t, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=40, cbar=False, ax=ax, annot_kws={"size": 7})
        ax.set_title(f, fontsize=10); ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    fig.suptitle(f"{ds}: every feature's SHAP share by model (rows) and class (columns), %")
    fig.tight_layout()
    save(fig, F, ds, "per_feature_model_by_class.png", "Small multiples: one model x class heatmap per feature")
    return rows


def fig_cross_datasets(allV, models):
    F = "03_cross_model_comparison"
    if len(allV) < 2:
        return
    (d1, V1), (d2, V2) = list(allV.items())[:2]
    R = pd.DataFrame([[spearmanr(mean_abs(V1[m])[:, k], mean_abs(V2[m])[:, k])[0] for k in range(len(CNAMES))]
                      for m in models], index=[MODEL_LABEL[m] for m in models], columns=CNAMES)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(R, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="Greens", ax=ax)
    ax.set_title(f"Same model and class, {d1} vs {d2}: Spearman correlation of feature ranks")
    save(fig, F, None, "MARIDA_vs_MADOS_rank_correlation.png",
         "How similar each model's per-class ranking is between the two datasets")

    for k, cn in enumerate(CNAMES):
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.3))
        for ax, m in zip(np.atleast_1d(axes), models):
            a, b = (mean_abs(V1[m])[:, k], mean_abs(V2[m])[:, k])
            a, b = a / a.sum() * 100, b / b.sum() * 100
            ax.scatter(a, b, s=18, color=PALETTE[cn])
            for j, f in enumerate(STAGE8):
                if max(a[j], b[j]) > 7:
                    ax.annotate(f, (a[j], b[j]), fontsize=7)
            lim = max(a.max(), b.max()) * 1.08
            ax.plot([0, lim], [0, lim], "--", c="gray", lw=.8)
            ax.set_title(MODEL_LABEL[m]); ax.set_xlabel(f"{d1} %"); ax.set_ylabel(f"{d2} %")
        fig.suptitle(f"{cn}: SHAP share per feature, {d1} vs {d2}", color=PALETTE[cn])
        fig.tight_layout()
        save(fig, F, None, f"{SLUG[cn]}_MARIDA_vs_MADOS.png", f"{cn}: per-feature share on each dataset, per model")


# --------------------------------------- 04 dependence and interactions ---

def interacting(values, X, j):
    import shap
    return int(shap.utils.approximate_interactions(j, values, X)[0])


def fig_dependence(ds, V, X, y, models):
    F = "04_dependence_and_interactions"
    for k, cn in enumerate(CNAMES):
        for m in models:
            vals = V[m][:, :, k]
            top = np.argsort(mean_abs(V[m])[:, k])[::-1][:4]
            fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
            for ax, j in zip(axes, top):
                ci = interacting(vals, X, j)
                lo, hi = np.percentile(X[:, j], [0.5, 99.5])
                sc = ax.scatter(np.clip(X[:, j], lo, hi), vals[:, j], c=X[:, ci], cmap="coolwarm", s=6,
                                vmin=np.percentile(X[:, ci], 2), vmax=np.percentile(X[:, ci], 98))
                ax.axhline(0, c="gray", lw=.6)
                plt.colorbar(sc, ax=ax, label=STAGE8[ci])
                ax.set_xlabel(STAGE8[j]); ax.set_ylabel(f"SHAP for {cn}")
            fig.suptitle(f"{ds} | {MODEL_LABEL[m]} | {cn}: dependence of the top 4 features "
                         "(colour = strongest interacting feature)", color=PALETTE[cn])
            fig.tight_layout()
            save(fig, F, os.path.join(ds, "dependence"), f"{SLUG[cn]}_{m}.png",
                 f"{MODEL_LABEL[m]}, {cn}: feature value vs SHAP for its top 4 features")

        # same feature, all models: the consensus top-3 features of this class
        S = pd.DataFrame({m: mean_abs(V[m])[:, k] / mean_abs(V[m])[:, k].sum() for m in models}, index=STAGE8)
        top3 = S.mean(axis=1).nlargest(3).index
        fig, axes = plt.subplots(3, len(models), figsize=(4.2 * len(models), 10))
        for r, f in enumerate(top3):
            j = STAGE8.index(f)
            lo, hi = np.percentile(X[:, j], [0.5, 99.5])
            for c, m in enumerate(models):
                ax = axes[r, c]
                for t, tn in enumerate(CNAMES):
                    sel = y == t
                    ax.scatter(np.clip(X[sel, j], lo, hi), V[m][sel, j, k], s=4, alpha=.5, color=PALETTE[tn],
                               label=tn if (r, c) == (0, 0) else None)
                ax.axhline(0, c="gray", lw=.6)
                ax.set_xlabel(f); ax.set_title(MODEL_LABEL[m] if r == 0 else "")
                if c == 0:
                    ax.set_ylabel(f"SHAP for {cn}")
        axes[0, 0].legend(markerscale=4, fontsize=7)
        fig.suptitle(f"{ds} | {cn}: dependence of the 3 consensus features in every model "
                     "(colour = the pixel's true class)", color=PALETTE[cn])
        fig.tight_layout()
        save(fig, F, os.path.join(ds, "dependence"), f"{SLUG[cn]}_top3_all_models.png",
             f"{cn}: the three most important features, value vs SHAP, every model, coloured by true class")

    # redundant pairs: how SHAP credit is split between near-duplicate features
    fig, axes = plt.subplots(len(REDUNDANT), len(models), figsize=(3.8 * len(models), 3.4 * len(REDUNDANT)))
    tot = np.zeros(len(REDUNDANT))
    for r, (a, b) in enumerate(REDUNDANT):
        ia, ib = STAGE8.index(a), STAGE8.index(b)
        for c, m in enumerate(models):
            ax = axes[r, c]
            va, vb = true_class(V[m], y)[:, ia], true_class(V[m], y)[:, ib]
            ax.scatter(va, vb, s=4, alpha=.5, c=[PALETTE[CNAMES[t]] for t in y])
            ax.axhline(0, c="gray", lw=.5); ax.axvline(0, c="gray", lw=.5)
            sa, sb = np.abs(va).mean(), np.abs(vb).mean()
            ax.set_title(f"{MODEL_LABEL[m]}: {a} {sa / (sa + sb + 1e-12):.0%} / {b} {sb / (sa + sb + 1e-12):.0%}", fontsize=8)
            ax.set_xlabel(f"SHAP {a}", fontsize=8); ax.set_ylabel(f"SHAP {b}", fontsize=8)
    fig.suptitle(f"{ds}: how SHAP credit is shared between near-duplicate features "
                 "(true-class output; colour = true class)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, F, ds, "redundant_pairs_credit_split.png",
         "SHAP of one feature vs its near-duplicate; titles give each one's share of their combined credit")

    # correlation of feature values vs correlation of SHAP values
    Cx = pd.DataFrame(X, columns=STAGE8).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9.5))
    sns.heatmap(Cx, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".1f", annot_kws={"size": 6}, ax=ax)
    ax.set_title(f"{ds}: Spearman correlation between the 22 features (explained pixels)")
    save(fig, F, ds, "feature_value_correlation.png", "Correlation of the feature values; blocks of |rho| near 1 are near-duplicates")
    try:
        g = sns.clustermap(Cx.abs(), cmap="viridis", vmin=0, vmax=1, figsize=(10, 10))
        g.figure.suptitle(f"{ds}: features clustered by |Spearman| correlation", y=1.01)
        save(g.figure, F, ds, "feature_value_correlation_clustered.png", "Same, hierarchically clustered: each cluster is one kind of information")
    except Exception as e:
        print(f"  [04] clustermap skipped: {e}")
    for m in models:
        Cs = pd.DataFrame(true_class(V[m], y), columns=STAGE8).corr()
        fig, ax = plt.subplots(figsize=(11, 9.5))
        sns.heatmap(Cs, cmap="RdBu_r", vmin=-1, vmax=1, center=0, ax=ax)
        ax.set_title(f"{ds} | {MODEL_LABEL[m]}: correlation between features' SHAP values (true-class output)")
        save(fig, F, os.path.join(ds, "shap_correlation"), f"{m}.png",
             f"{MODEL_LABEL[m]}: which features' contributions move together (shared credit) or against each other")


def fig_interactions(ds, models_fit, X, y, n):
    """Exact TreeSHAP interaction values for XGBoost and LightGBM (fast on depth-6 trees)."""
    import shap
    F = "04_dependence_and_interactions"
    idx = balanced_idx(y, n, np.random.default_rng(0))
    for m in [m for m in ("xgb", "lgbm") if m in models_fit]:
        t0 = time.time()
        I = np.asarray(shap.TreeExplainer(models_fit[m]).shap_interaction_values(X[idx]))
        if I.ndim == 4 and I.shape[0] == len(CNAMES):      # (k, n, f, f) -> (n, f, f, k)
            I = np.moveaxis(I, 0, -1)
        fig, axes = plt.subplots(1, 4, figsize=(26, 6.5))
        for ax, (k, cn) in zip(axes, enumerate(CNAMES)):
            M = np.abs(I[:, :, :, k]).mean(axis=0)
            off = M.copy(); np.fill_diagonal(off, 0)
            off = off / M.sum() * 100
            sns.heatmap(pd.DataFrame(off, index=STAGE8, columns=STAGE8), cmap="Purples", ax=ax, cbar=k == 3,
                        xticklabels=True, yticklabels=(k == 0))
            ax.set_title(cn, color=PALETTE[cn]); ax.tick_params(labelsize=6)
        fig.suptitle(f"{ds} | {MODEL_LABEL[m]}: mean |SHAP interaction| between feature pairs, % of total "
                     f"(diagonal = main effects, removed; {len(idx)} test pixels)")
        fig.tight_layout()
        save(fig, F, ds, f"interaction_heatmap_{m}.png",
             f"{MODEL_LABEL[m]}: exact TreeSHAP interaction strength for every feature pair, per class")
        main = np.abs(np.einsum("nffk->nfk", I)).sum()
        print(f"  [04] {m} interactions {time.time() - t0:.0f}s, off-diagonal share "
              f"{1 - main / np.abs(I).sum():.0%}", flush=True)


# ------------------------------------------ 05 individual predictions ---

def model_output(m, model, X):
    if m in ("rf", "ada"):
        return model.predict_proba(X)
    if m == "xgb":
        return model.predict(X, output_margin=True)
    if m == "lgbm":
        return model.predict(X, raw_score=True)
    return model.decision_function(X)


def fig_individual(ds, V, X, y, models_fit, bg):
    import shap
    F = "05_individual_predictions"
    rows = []
    for m, model in models_fit.items():
        f_exp, base = model_output(m, model, X), model_output(m, model, bg).mean(axis=0)
        pred = f_exp.argmax(axis=1)
        err = np.abs(base[None] + V[m].sum(axis=1) - f_exp).max()
        print(f"  [05] {m} max additivity error {err:.2e}", flush=True)
        for k, cn in enumerate(CNAMES):
            members = np.flatnonzero(y == k)
            right, wrong = members[pred[members] == k], members[pred[members] != k]
            picks = []
            if len(right):
                picks.append(("correct", right[np.argsort(f_exp[right, k])[len(right) // 2]], k))
            if len(wrong):
                w = wrong[np.argmax(f_exp[wrong, pred[wrong]] - f_exp[wrong, k])]
                picks += [("wrong_trueclass", w, k), ("wrong_predclass", w, int(pred[w]))]
            for tag, i, out_k in picks:
                e = shap.Explanation(values=V[m][i, :, out_k], base_values=base[out_k], data=X[i], feature_names=STAGE8)
                shap.plots.waterfall(e, max_display=12, show=False)
                fig = plt.gcf(); fig.set_size_inches(9, 6.5)
                fig.suptitle(f"{ds} | {MODEL_LABEL[m]} | pixel {i}: true {cn}, predicted {CNAMES[pred[i]]}\n"
                             f"explaining the {CNAMES[out_k]} output", fontsize=10)
                save(fig, F, os.path.join(ds, "waterfall", m), f"{SLUG[cn]}_{tag}.png",
                     f"{MODEL_LABEL[m]}: waterfall for a {tag.replace('_', ' ')} {cn} pixel ({CNAMES[out_k]} output)")
                rows.append(dict(dataset=ds, model=m, true=cn, predicted=CNAMES[pred[i]], example=tag, pixel=int(i),
                                 explained_output=CNAMES[out_k], base=float(base[out_k]), output=float(f_exp[i, out_k])))

            # decision plot: up to 60 pixels of the class, misclassified ones dashed
            sel = members[:60]
            shap.decision_plot(base[k], V[m][sel, :, k], features=X[sel], feature_names=STAGE8,
                               highlight=np.flatnonzero(pred[sel] != k), feature_display_range=slice(None, -16, -1),
                               show=False, ignore_warnings=True)
            fig = plt.gcf(); fig.set_size_inches(8, 7)
            fig.suptitle(f"{ds} | {MODEL_LABEL[m]} | {cn} output for {len(sel)} true {cn} pixels\n"
                         "(each line = one pixel; dashed = misclassified)", fontsize=10, color=PALETTE[cn])
            save(fig, F, os.path.join(ds, "decision"), f"{SLUG[cn]}_{m}.png",
                 f"{MODEL_LABEL[m]}: decision plot of the {cn} output for true {cn} pixels")

        # confusion explained: mean SHAP of (predicted - true) output for each wrong (true, predicted) pair
        cm_rows = []
        for t in range(len(CNAMES)):
            for p in range(len(CNAMES)):
                sel = np.flatnonzero((y == t) & (pred == p))
                if t != p and len(sel) >= 5:
                    d = (V[m][sel, :, p] - V[m][sel, :, t]).mean(axis=0)
                    cm_rows.append(pd.Series(d, index=STAGE8, name=f"{SHORT[CNAMES[t]]}→{SHORT[CNAMES[p]]} (n={len(sel)})"))
        if cm_rows:
            D = pd.DataFrame(cm_rows).T
            D = D.loc[D.abs().max(axis=1).sort_values(ascending=False).index[:15]]
            fig, ax = plt.subplots(figsize=(1.3 * D.shape[1] + 4, 7))
            sns.heatmap(D, cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
            ax.set_title(f"{ds} | {MODEL_LABEL[m]}: what pushes pixels into the wrong class\n"
                         "mean SHAP(predicted output) − SHAP(true output) over misclassified pixels")
            save(fig, F, os.path.join(ds, "confusions"), f"{m}.png",
                 f"{MODEL_LABEL[m]}: features responsible for each type of misclassification")
    return rows


# ---------------------------------------------------------- 06 reports ---

def write_reports(tables):
    F = "06_reports"
    d = os.path.join(FIG_ROOT, F); os.makedirs(d, exist_ok=True)
    for name, df in tables.items():
        if df is not None and len(df):
            df.to_csv(os.path.join(d, name), index=False)
    L = ["# SHAP figure index", "",
         "Generated by `scripts/feature_analysis/shap_figures.py` from the stage-8 models "
         "(22 features, log weights) explained in `per_class_features.py`. "
         "The write-up is `docs/feature_analysis.md`.", "",
         "Classes: 1 Water, 2 Marine Debris, 3 Algae/Sargassum, 4 Other. Models: RF, XGBoost, LightGBM, "
         "AdaBoost, linear SVM. SHAP output units differ per model (probability, log-odds, decision score), "
         "so models are compared through shares (% of mean |SHAP|).", "",
         "## Tables", ""]
    L += [f"- `{n}`" for n in tables if tables[n] is not None and len(tables[n])]
    for folder in FOLDERS[:-1]:
        items = [r for r in INDEX if r[0] == folder]
        if not items:
            continue
        L += ["", f"## {folder}", "", "| Figure | What it shows |", "|---|---|"]
        for _, p, desc in sorted(items, key=lambda r: r[1]):
            L.append(f"| [`{p}`](../{folder}/{p}) | {desc.replace('|', chr(92) + '|')} |")
    L += ["", f"{len(INDEX)} figures in total."]
    with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--with-models", action="store_true",
                    help="Refit the stage-8 models for folder 05 and the interaction heatmaps")
    ap.add_argument("--interaction-n", type=int, default=150, help="Test pixels per class for interaction values")
    ap.add_argument("--weighting", default="log")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bg-n", type=int, default=25, help="Must match per_class_features.py")
    args = ap.parse_args()

    Gs, allV, agree, examples = {}, {}, [], []
    for ds in args.datasets:
        t0 = time.time()
        V, X, y, shares = load_shap(ds, args.models)
        allV[ds] = V
        print(f"\n=== {ds}: {len(y)} explained pixels", flush=True)
        Gs[ds] = fig_global(ds, V, X, y, args.models); print(f"  [01] {time.time() - t0:.0f}s", flush=True)
        fig_class(ds, V, X, y, args.models); print(f"  [02] {time.time() - t0:.0f}s", flush=True)
        agree += fig_cross(ds, V, args.models, shares); print(f"  [03] {time.time() - t0:.0f}s", flush=True)
        fig_dependence(ds, V, X, y, args.models); print(f"  [04] {time.time() - t0:.0f}s", flush=True)
        if args.with_models:
            Xtr, ytr, Xte, yte = load(ds, args.seed)
            c8 = cols(STAGE8)
            rng = np.random.default_rng(args.seed)            # same draws as per_class_features.part_shap
            bg = Xtr[balanced_idx(ytr, args.bg_n, rng)][:, c8]
            fitted = {}
            for m in args.models:
                t1 = time.time()
                fitted[m] = fit(m, Xtr[:, c8], ytr, args.weighting, args.seed)
                print(f"  [fit] {m} {time.time() - t1:.0f}s", flush=True)
            del Xtr, ytr
            fig_interactions(ds, fitted, Xte[:, c8], yte, args.interaction_n)
            examples += fig_individual(ds, V, X, y, fitted, bg); print(f"  [05] {time.time() - t0:.0f}s", flush=True)
            del fitted
    fig_global_datasets(Gs, args.models)
    fig_cross_datasets(allV, args.models)
    tables = {"model_agreement.csv": pd.DataFrame(agree), "individual_examples.csv": pd.DataFrame(examples)}
    for ds, G in Gs.items():
        tables[f"global_share_{ds}.csv"] = G.reset_index().rename(columns={"index": "feature"})
    write_reports(tables)
    print(f"\n{len(INDEX)} figures -> {FIG_ROOT}", flush=True)


if __name__ == "__main__":
    main()
