"""
Deep-learning comparison of the 26-feature study, in the layout of the staged-ML
comparison (outputs/staged/<weighting>/comparison/comparison.md): one section per
dataset (MADOS, MARIDA), one table per metric, the papers' published results on
top of every table, and the change against the U-Net (R3).

Models: R3 U-Net (from scratch), R4 SegNeXt-T, R5 MariNeXt architecture, R7 U-Net
(R3) + RF ensemble; all on the 26 features, test split, 4 classes. 95% confidence
intervals come from each study's scene bootstrap (compare.py); a dataset whose runs
are not finished yet is listed as pending.

Usage (repo root):  python scripts/marida_f26/dl_comparison_f26.py
Output: outputs/dl_comparison_f26/  (comparison.md, all_results.csv, delta_<metric>_<dataset>.png)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compare import PAPERS  # noqa: E402
from runs import REPO, name  # noqa: E402

OUT = os.path.join(REPO, "outputs", "dl_comparison_f26")
DATASETS = ["MADOS", "MARIDA"]                       # the order of the staged-ML comparison
MODELS = [("R3", "U-Net (R3, from scratch)"), ("R4", "SegNeXt-T (R4)"), ("R5", "MariNeXt arch. (R5)"),
          ("R7", "U-Net + RF ensemble (R7)")]
BASE = "R3"
CLASSES = ["Water", "Marine Debris", "Algae/Sargassum", "Other"]
SCHEME = {"MARIDA": "11 classes", "MADOS": "15 classes"}
# (title, key in all_results, paper tuple index or None, bootstrap metric or None)
# paper tuple: (macro F1, mIoU, macro PA, debris F1, debris IoU, debris PA)
METRICS = [
    ("Macro F1", "macro_F1", 0, "macro_F1"),
    ("Macro IoU", "mIoU", 1, "mIoU"),
    ("Cohen's κ", "kappa", None, None),
    ("F1 Marine Debris", "F1_Marine Debris", 3, "F1_Marine Debris"),
    ("IoU Marine Debris", "IoU_Marine Debris", 4, None),
    ("F1 Algae/Sargassum", "F1_Algae/Sargassum", None, None),
    ("IoU Algae/Sargassum", "IoU_Algae/Sargassum", None, None),
    ("F1 Other", "F1_Other", None, None),
    ("Class accuracy Water (TP / class pixels)", "acc_Water", None, None),
    ("Class accuracy Marine Debris (TP / class pixels)", "acc_Marine Debris", 5, None),
    ("Class accuracy Algae/Sargassum (TP / class pixels)", "acc_Algae/Sargassum", None, None),
    ("Class accuracy Other (TP / class pixels)", "acc_Other", None, None),
]
GROUPS = ["bands", "indices", "ratios", "GLCM"]


def study(ds):
    return os.path.join(REPO, "outputs", f"{ds.lower()}_f26")


def load(ds):
    """{model id: metrics dict} for the finished runs, and the bootstrap CIs (if compare.py has run)."""
    res = {}
    for m, _ in MODELS:
        run = os.path.join(study(ds), "runs", name(m))
        if not os.path.exists(os.path.join(run, "metrics_test.json")):
            continue
        r = json.load(open(os.path.join(run, "metrics_test.json")))
        cw = pd.read_csv(os.path.join(run, "classwise_test.csv"), index_col=0)
        for c in CLASSES:
            r[f"acc_{c}"] = cw.loc[c, "recall (class accuracy)"]
        res[m] = r
    p = os.path.join(study(ds), "comparison", "bootstrap_ci.csv")
    ci = pd.read_csv(p) if os.path.exists(p) else None
    return res, ci


def cell(v, base, ci=None):
    s = f"{v:.4f}"
    if ci is not None:
        s += f" [{ci[0]:.3f}–{ci[1]:.3f}]"
    return s + (f" ({v - base:+.4f})" if base is not None else "")


def main():
    os.makedirs(OUT, exist_ok=True)
    L = ["# Deep-learning comparison, 26 features", "",
         "Test split, every annotated pixel, 4 classes (Water, Marine Debris, Algae/Sargassum, Other); macro = mean "
         "over the 4 classes. Each cell: score [95% scene-bootstrap CI where computed] (Δ vs the U-Net, R3, same "
         "dataset). All models use the same 26 input features (11 bands, NDVI, NDWI, FAI, FDI, NDMI, 4 ratios, "
         "6 GLCM); R7 = R3 + the Random Forest (R6) on the same features, weight chosen on validation.", "",
         "Training recipes follow each dataset's paper: MARIDA 45 epochs, no VSCP; MADOS 80 epochs, EMA, VSCP "
         "copy-paste. Single seed; the CIs cover test-scene variation, not training randomness.", "",
         "📄 **Paper rows** (top of each table, values in _italics_) are the published results of the MARIDA "
         "(Table 4) and MADOS (Tables 4 and 5) papers. They use the papers' own class schemes (MARIDA 11 classes, "
         "MADOS 15 classes), so only marine debris is the same class as ours; their macro scores average over "
         "more, and different, classes. Tables without a published value have no paper rows. **Bold** = best of "
         "our models.", ""]
    rows = []
    for ds in DATASETS:
        res, ci = load(ds)
        L += [f"## {ds}", ""]
        if not res:
            L += ["_Runs not finished yet._", ""]
            continue
        missing = [lab for m, lab in MODELS if m not in res]
        if missing:
            L += [f"_Not finished yet: {', '.join(missing)}._", ""]
        for m, lab in MODELS:
            if m in res:
                rows.append({"dataset": ds, "model": m, "label": lab, **res[m]})
        base = res.get(BASE)
        for title, key, pk, bk in METRICS:
            L += [f"### {title}", "", "| Model | score |", "|---|---|"]
            if pk is not None:
                for pname, v in PAPERS[ds].items():
                    if v[pk] is not None:
                        L.append(f"| 📄 **{ds} paper: {pname.replace(' (paper)', '').replace(', paper)', ')')}** "
                                 f"({SCHEME[ds]}) | _{v[pk]:.3f}_ |")
            best = max(res[m][key] for m in res)
            for m, lab in MODELS:
                if m not in res:
                    continue
                c = None
                if ci is not None and bk is not None:
                    q = ci[(ci.run == name(m)) & (ci.metric == bk)]
                    c = (q.ci_low.iloc[0], q.ci_high.iloc[0]) if len(q) else None
                s = cell(res[m][key], None if (base is None or m == BASE) else base[key], c)
                L.append(f"| {lab} | {'**' + s + '**' if res[m][key] == best else s} |")
            L.append("")
        # grouped SHAP
        L += [f"### Feature importance by group (share of \\|SHAP\\|, %), {ds}", "",
              "Deep models: GradientSHAP; R7: weighted mix of its members' shares (approximation). Per-feature "
              f"values in `outputs/{ds.lower()}_f26/xai/<run>/`.", ""]
        G = {}
        for m, _ in MODELS:
            p = os.path.join(study(ds), "xai", name(m), "importance_groups.csv")
            if os.path.exists(p):
                G[m] = pd.read_csv(p)
        if not G:
            L += ["_XAI not run yet._", ""]
        else:
            ms = [m for m, _ in MODELS if m in G]
            L += ["| class | group | " + " | ".join(ms) + " |", "|---" * (len(ms) + 2) + "|"]
            for c in CLASSES:
                for g in GROUPS:
                    v = [G[m].loc[(G[m]["class"] == c) & (G[m]["group"] == g), "share_%"] for m in ms]
                    L.append(f"| {c} | {g} | " + " | ".join(f"{x.iloc[0]:.1f}" if len(x) else "–" for x in v) + " |")
            L.append("")
        # Δ figures, as the staged-ML delta plots
        if base is not None and len(res) > 1:
            for key, fname in [("macro_F1", "f1"), ("mIoU", "iou"), ("F1_Marine Debris", "debris_f1"),
                               ("IoU_Marine Debris", "debris_iou")]:
                ms = [m for m, _ in MODELS if m in res and m != BASE]
                d = [res[m][key] - base[key] for m in ms]
                fig, ax = plt.subplots(figsize=(6.5, 3.6))
                bars = ax.bar([dict(MODELS)[m] for m in ms], d, color=["#d62728" if x < 0 else "#2ca02c" for x in d])
                ax.bar_label(bars, fmt="%+.3f", fontsize=8)
                ax.axhline(0, color="k", lw=0.8)
                ax.set_ylabel(f"Δ {key.replace('_', ' ')} vs U-Net (R3)")
                ax.set_title(f"{ds}: {key.replace('_', ' ')} vs U-Net R3 ({base[key]:.3f})")
                ax.tick_params(axis="x", labelsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(OUT, f"delta_{fname}_{ds}.png"), dpi=120)
                plt.close()
            L += [f"![delta macro F1 {ds}](delta_f1_{ds}.png) ![delta debris F1 {ds}](delta_debris_f1_{ds}.png)", ""]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "all_results.csv"), index=False)
    open(os.path.join(OUT, "comparison.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("->", os.path.join(OUT, "comparison.md"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
