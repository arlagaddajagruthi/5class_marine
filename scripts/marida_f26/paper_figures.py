"""
Paper figures for the 26-feature DL study on MARIDA: only the four runs reported in the
paper (R3 U-Net, R4 SegNeXt-T, R5 MariNeXt architecture, R7 U-Net + RF ensemble), with
short labels, in the style of the ML SHAP figures (SHAP/Analysis1). Reads the outputs of
train.py / rf_ensemble.py (metrics) and xai.py (SHAP).

  dl_f1_marida.png               macro F1 and per-class F1 of the four models
  dl_shap_heatmap_marida.png     share of mean |SHAP| of every feature, per model and class
  dl_importance_marida.png       feature importance (SHAP share) per class, the four models side by side
  dl_groups_marida.png           SHAP share of each feature group per class and model
  dl_beeswarm_<model>_marida.png SHAP beeswarm of all four classes (U-Net, SegNeXt-T, MariNeXt;
                                 the ensemble has no per-pixel SHAP values, only a mix of shares)
  dl_dependence_debris_unet.png  U-Net, Marine Debris: value vs SHAP of the top four features
  dl_shap_maps_debris_unet.png   U-Net, SHAP maps per feature group and Seg-Grad-CAM (copied)

Usage (repo root):  python scripts/marida_f26/paper_figures.py
Output: paper/feature_analysis_paper/ (the folder of the paper .tex)
"""
import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features26 as F  # noqa: E402
from runs import REPO, name  # noqa: E402

STUDY = os.path.join(REPO, "outputs", "marida_f26")
OUT = os.path.join(REPO, "paper", "feature_analysis_paper")   # next to the paper .tex
MODELS = [("R3", "U-Net", "#1f77b4"), ("R4", "SegNeXt-T", "#9467bd"), ("R5", "MariNeXt arch.", "#8c564b"),
          ("R7", "U-Net + RF", "#2ca02c")]
DEEP = MODELS[:3]
CLASSES = ["Water", "Marine Debris", "Algae/Sargassum", "Other"]
SHORT = {"Water": "water", "Marine Debris": "debris", "Algae/Sargassum": "algae", "Other": "other"}
COLLABEL = {"Water": "Water", "Marine Debris": "Debris", "Algae/Sargassum": "Algae", "Other": "Other"}
FILE_TAG = {"R3": "unet", "R4": "segnext", "R5": "marinext"}


def shares(m):
    return pd.read_csv(os.path.join(STUDY, "xai", name(m), "importance_features.csv"))


def fig_f1():
    rows = []
    for m, lab, _ in MODELS:
        met = json.load(open(os.path.join(STUDY, "runs", name(m), "metrics_test.json")))
        rows.append([met["macro_F1"]] + [met[f"F1_{c}"] for c in CLASSES])
    groups = ["Macro F1"] + [f"{COLLABEL[c]} F1" for c in CLASSES]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.2
    for j, (m, lab, col) in enumerate(MODELS):
        b = ax.bar(np.arange(5) + (j - 1.5) * w, rows[j], w, label=lab, color=col)
        ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(range(5), groups)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 (test)")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "dl_f1_marida.png"), dpi=200)
    plt.close()


def fig_heatmap():
    S = {m: shares(m) for m, _, _ in MODELS}
    cols, data = [], []
    for m, lab, _ in MODELS:
        for c in CLASSES:
            s = S[m][S[m]["class"] == c].set_index("feature")["share_%"].reindex(F.CHANNELS)
            cols.append(f"{lab}\n{COLLABEL[c]}")
            data.append(s.values)
    H = pd.DataFrame(np.array(data).T, index=F.CHANNELS, columns=cols)
    H = H.loc[H.mean(1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(H.values, cmap="Blues", aspect="auto", vmin=0)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v = H.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 * H.values.max() else "black")
    for k in range(1, len(MODELS)):
        ax.axvline(4 * k - 0.5, color="white", lw=4)
    ax.set_xticks(range(H.shape[1]), H.columns, rotation=90, fontsize=9)
    ax.set_yticks(range(H.shape[0]), H.index, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, label="% of mean |SHAP| within (model, class output)")
    ax.set_title("MARIDA, 26 features: class-wise SHAP share across DL models and class outputs "
                 "(each column sums to 100 %)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "dl_shap_heatmap_marida.png"), dpi=170)
    plt.close()


def fig_importance(top=12):
    S = {m: shares(m) for m, _, _ in MODELS}
    fig, axes = plt.subplots(1, 4, figsize=(16, 5.6))
    h = 0.2
    for ax, c in zip(axes, CLASSES):
        T = pd.DataFrame({m: S[m][S[m]["class"] == c].set_index("feature")["share_%"] for m, _, _ in MODELS})
        T = T.loc[T.mean(1).sort_values(ascending=False).index[:top]]
        y = np.arange(len(T))
        for j, (m, lab, col) in enumerate(MODELS):
            ax.barh(y + (j - 1.5) * h, T[m], h, color=col, label=lab)
        ax.set_yticks(y, T.index, fontsize=9)
        for t in ax.get_yticklabels():
            t.set_color(F.GROUP_COLORS[F.GROUP_OF[t.get_text()]])
        ax.invert_yaxis()
        ax.set_xlabel("share of mean |SHAP| (%)")
        ax.set_title(c)
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("MARIDA, 26 features: feature importance per class (top 12 features by mean share; "
                 "label colour = group: blue bands, green indices, red ratios, orange GLCM)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "dl_importance_marida.png"), dpi=170)
    plt.close()


def fig_groups():
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    G = {m: pd.read_csv(os.path.join(STUDY, "xai", name(m), "importance_groups.csv")) for m, _, _ in MODELS}
    for a, c in zip(ax, CLASSES):
        bottom = np.zeros(4)
        for g, col in F.GROUP_COLORS.items():
            v = np.array([G[m].loc[(G[m]["class"] == c) & (G[m]["group"] == g), "share_%"].iloc[0]
                          for m, _, _ in MODELS])
            a.bar(range(4), v, bottom=bottom, color=col, label=g)
            for j in range(4):
                if v[j] >= 8:
                    a.text(j, bottom[j] + v[j] / 2, f"{v[j]:.0f}", ha="center", va="center", fontsize=7, color="w")
            bottom += v
        a.set_xticks(range(4), [lab for _, lab, _ in MODELS], rotation=35, ha="right", fontsize=8)
        a.set_title(c)
    ax[0].set_ylabel("share of |SHAP| (%)")
    ax[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "dl_groups_marida.png"), dpi=200)
    plt.close()


def explanation(m, c):
    import shap
    z = np.load(os.path.join(STUDY, "xai", name(m), f"pixels_{SHORT[c]}.npz"))
    return shap.Explanation(values=z["shap"], data=z["values"], feature_names=F.CHANNELS)


def fig_beeswarm(m, lab):
    import shap
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, c in zip(axes.ravel(), CLASSES):
        plt.sca(ax)
        shap.plots.beeswarm(explanation(m, c), max_display=12, show=False, ax=ax, plot_size=None)
        ax.set_title(c, fontsize=11)
    fig.suptitle(f"MARIDA | {lab} (26 features): GradientSHAP beeswarm per class output", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"dl_beeswarm_{FILE_TAG[m]}_marida.png"), dpi=140)
    plt.close()


def fig_dependence(m="R3", c="Marine Debris"):
    import shap
    e = explanation(m, c)
    order = np.argsort(-np.abs(e.values).mean(0))[:4]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    for ax, k in zip(axes, order):
        shap.plots.scatter(e[:, int(k)], color=e, ax=ax, show=False)
        ax.set_title(F.CHANNELS[k])
    fig.suptitle(f"MARIDA | U-Net (26 features), {c}: feature value vs GradientSHAP value, "
                 "top four features; colour = the feature that interacts most strongly", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "dl_dependence_debris_unet.png"), dpi=140)
    plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    for old in ("dl_overall_ci.png", "dl_contrasts_marida.png", "dl_beeswarm_debris_unet.png"):
        p = os.path.join(OUT, old)
        if os.path.exists(p):
            os.remove(p)
    fig_f1()
    fig_heatmap()
    fig_importance()
    fig_groups()
    for m, lab, _ in DEEP:
        fig_beeswarm(m, lab)
    fig_dependence()
    x3 = os.path.join(STUDY, "xai", name("R3"), "figures")
    shutil.copy(os.path.join(x3, "maps_debris.png"), os.path.join(OUT, "dl_shap_maps_debris_unet.png"))
    print("->", OUT, sorted(f for f in os.listdir(OUT) if f.startswith("dl_")))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
