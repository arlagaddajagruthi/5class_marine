"""
Overall SHAP contribution-rate bar charts for all 22 features: MARIDA, MADOS and both together.

contribution rate of feature f = mean over the five models of
    share_m(f) = mean|SHAP_f| / sum_g mean|SHAP_g|    (model m, averaged over the 4 class outputs)
so every model counts equally regardless of its output scale, and the rates sum to 1.
The combined chart averages the MARIDA and MADOS rates with equal weight.

Reads 06_reports/tables/<ds>/<ds>_cross_model_comparison.csv written by run_shap_analysis.py.
Writes 03_cross_model_comparison/overall/ (PNGs, CSV) and 06_reports/overall_contribution.md,
which make_report.py includes in the report.

Usage (repo root): .venv/Scripts/python scripts/shap_analysis/plot_overall_contribution.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MODELS = ["rf", "xgb", "lgbm", "ada", "svm"]


def rates(root, ds):
    df = pd.read_csv(os.path.join(root, "06_reports", "tables", ds, f"{ds}_cross_model_comparison.csv"), index_col=0)
    r = df[[f"{m}_share" for m in MODELS]].mean(axis=1)
    return r / r.sum()


def bar(r, title, path):
    r = r.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    norm = plt.Normalize(r.min(), r.max())
    cmap = plt.get_cmap("viridis")
    ax.barh(range(len(r)), r.values, color=cmap(norm(r.values)), height=0.75)
    for i, v in enumerate(r.values):
        ax.text(v + r.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(r)), r.index, fontsize=10)
    ax.set_xlim(0, r.max() * 1.15)
    ax.set_xlabel("contribution rate (normalised mean |SHAP|, mean of 5 models)")
    ax.spines[["top", "right"]].set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    ax.set_title(title, fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "results", "shap_analysis"))
    a = ap.parse_args()
    out = os.path.join(a.root, "03_cross_model_comparison", "overall")
    os.makedirs(out, exist_ok=True)
    R = {ds: rates(a.root, ds) for ds in ["MARIDA", "MADOS"]}
    R["Combined"] = ((R["MARIDA"] + R["MADOS"]) / 2).pipe(lambda s: s / s.sum())
    tab = pd.DataFrame(R)
    tab["rank_MARIDA"] = tab["MARIDA"].rank(ascending=False).astype(int)
    tab["rank_MADOS"] = tab["MADOS"].rank(ascending=False).astype(int)
    tab["rank_Combined"] = tab["Combined"].rank(ascending=False).astype(int)
    tab.sort_values("Combined", ascending=False).to_csv(os.path.join(out, "overall_contribution_rates.csv"),
                                                        float_format="%.5f")
    md = ["## Overall SHAP contribution rate of all 22 features", "",
          "Contribution rate of a feature = its normalised SHAP importance (mean |SHAP| / sum over the 22 features, "
          "averaged over the 4 class outputs) averaged over the five models, so each model counts equally whatever its "
          "output scale; the rates sum to 1. 'Both datasets' averages the MARIDA and MADOS rates with equal weight. "
          "Values: `03_cross_model_comparison/overall/overall_contribution_rates.csv`.", ""]
    for i, (key, label) in enumerate([("MARIDA", "MARIDA"), ("MADOS", "MADOS"), ("Combined", "MARIDA + MADOS")], 1):
        fid = f"Fig. O-{i}"
        title = f"{fid}. Overall SHAP contribution rate, {label} (5-model mean)"
        name = f"overall_contribution_{key.lower()}.png"
        bar(R[key], title, os.path.join(out, name))
        top = R[key].sort_values(ascending=False)
        top5 = ", ".join(f"**{f}** {v:.3f}" for f, v in top.head(5).items())
        bottom = ", ".join(f"{f} {v:.3f}" for f, v in top.tail(3).items())
        md += [f"#### {title}", "", f"![{fid}](../03_cross_model_comparison/overall/{name})", "",
               f"- **What it shows:** all 22 features ranked by contribution rate for {label}.",
               "- **How to read:** bar length and colour = contribution rate (share of total attribution); labels give the value.",
               f"- **Findings:** top five {top5}; the top five together carry {top.head(5).sum():.3f}. Lowest: {bottom}.",
               "- **Limitations:** unsigned (no direction), averaged over classes and models; per-class and per-model "
               "differences are in sections A1, A3 and B6.", ""]
    with open(os.path.join(a.root, "06_reports", "overall_contribution.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(tab.sort_values("Combined", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
