"""
Final comparison of the MARIDA 26-feature study, with the MARIDA paper's results at
the top of every table, scene-bootstrap 95% confidence intervals (G3), the
controlled comparisons between runs, and the cross-model XAI summary.

Bootstrap (G3): the test scenes are resampled with replacement (2,000 times); each
resample's scores are recomputed from per-scene confusion matrices. The same
resamples are used for every run, so differences between runs are paired.

Usage (repo root):  python scripts/marida_f26/compare.py
Output: outputs/marida_f26/comparison/  (comparison.md, all_results.csv, bootstrap_ci.csv,
        differences.csv, figures/)
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
import features26 as F  # noqa: E402
from runs import name  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "outputs", "marida_f26")
RUNS, XAI, OUT = os.path.join(ROOT, "runs"), os.path.join(ROOT, "xai"), os.path.join(ROOT, "comparison")
NAMES = F.CLASS_NAMES
ORDER4 = [name(r) for r in ("R1", "R2", "R3", "R4", "R5", "R6", "R7")]
N_BOOT = 2000
# controlled comparisons: (name, run A, run B, what differs)
CONTRASTS = [
    ("band scaling", name("R1"), name("R2"), "U-Net SSL4EO: raw reflectance vs standardised bands"),
    ("pretraining", name("R2"), name("R3"), "U-Net, standardised bands: SSL4EO weights vs from scratch"),
    ("architecture: U-Net vs SegNeXt-T", name("R3"), name("R4"), "from scratch, standardised bands"),
    ("architecture: U-Net vs MariNeXt", name("R3"), name("R5"), "from scratch, standardised bands"),
    ("architecture: MariNeXt vs SegNeXt-T", name("R5"), name("R4"), "MariNeXt adds stage-1 features (1/4 vs 1/8 output)"),
    ("deep vs RF", name("R1"), name("R6"), "best-setting U-Net vs RF on the same 26 features"),
    ("ensemble vs U-Net", name("R7"), name("R1"), "adding the RF to the U-Net"),
    ("ensemble vs RF", name("R7"), name("R6"), "adding the U-Net to the RF"),
]
# MARIDA paper (MARIDA.pdf Table 4, 11 classes; PA = producer's accuracy = recall)
PAPER = {  # model: (macro F1, mIoU, macro PA, debris F1, debris IoU, debris PA)
    "RF_SS (paper)": (.69, .57, .72, .71, .55, .91),
    "RF_SS+SI (paper)": (.70, .59, .73, .80, .67, .92),
    "RF_SS+SI+GLCM (paper)": (.79, .69, .79, .79, .65, .92),
    "U-Net (paper)": (.69, .57, .69, .50, .33, .70),
}
PAPER_CLASS_F1 = {  # MARIDA Table 4, F1 per class: RF_SS, RF_SS+SI, RF_SS+SI+GLCM, U-Net
    "Marine Debris": (.71, .80, .79, .50), "Dense Sargassum": (.93, .93, .93, .75),
    "Sparse Sargassum": (.69, .82, .91, .79), "Natural Organic Material": (.47, .29, .31, .04),
    "Ship": (.70, .64, .80, .76), "Clouds": (.86, .85, .91, .76), "Marine Water": (.79, .79, .86, .76),
    "Sediment-Laden Water": (1, 1, 1, 1), "Foam": (.37, .47, .75, .71), "Turbid Water": (.85, .89, .94, .91),
    "Shallow Water": (.16, .23, .46, .62),
}


def scene_confusions(run, n):
    z = np.load(os.path.join(RUNS, run, "pred_test.npz"))
    y, yp, patch, ids = z["y"], z["prob"].argmax(1), z["patch"], z["patch_ids"]
    scenes = np.array([p.rsplit("_", 1)[0] for p in ids])[patch]
    us = np.unique(scenes)
    cms = np.stack([np.bincount(y[scenes == s] * n + yp[scenes == s], minlength=n * n).reshape(n, n) for s in us])
    return us, cms


def scores(cm):
    tp = np.diag(cm).astype(float)
    sup, pred = cm.sum(1), cm.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p, r = tp / pred, tp / sup
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
        iou = tp / (sup + pred - tp)
    present = sup > 0
    return {"macro_F1": np.nanmean(np.nan_to_num(f1)[present]), "mIoU": np.nanmean(np.nan_to_num(iou)[present]),
            **{f"F1_{c}": f for c, f in zip(NAMES, f1)}} if len(cm) == 4 else \
        {"macro_F1": np.nanmean(np.nan_to_num(f1)[present]), "mIoU": np.nanmean(np.nan_to_num(iou)[present])}


def bootstrap(runs):
    """Scene-bootstrap CIs per run and paired differences for the contrasts."""
    cm = {r: scene_confusions(r, 4) for r in runs}
    scenes = cm[runs[0]][0]
    for r in runs:
        assert np.array_equal(cm[r][0], scenes), f"{r}: different test scenes"
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(scenes), size=(N_BOOT, len(scenes)))
    keys = ["macro_F1", "mIoU", "F1_Marine Debris"]
    boot = {r: np.array([[scores(cm[r][1][d].sum(0))[k] for k in keys] for d in draws]) for r in runs}
    ci = []
    for r in runs:
        point = scores(cm[r][1].sum(0))
        for j, k in enumerate(keys):
            lo, hi = np.nanpercentile(boot[r][:, j], [2.5, 97.5])
            ci.append({"run": r, "metric": k, "value": point[k], "ci_low": lo, "ci_high": hi})
    diffs = []
    for cname, a, b, what in CONTRASTS:
        if a not in boot or b not in boot:
            continue
        pa, pb = scores(cm[a][1].sum(0)), scores(cm[b][1].sum(0))
        for j, k in enumerate(keys):
            d = boot[a][:, j] - boot[b][:, j]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            diffs.append({"comparison": cname, "A": a, "B": b, "what differs": what, "metric": k,
                          "A - B": pa[k] - pb[k], "ci_low": lo, "ci_high": hi,
                          "P(A > B)": float(np.nanmean(d > 0)), "clear": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(ci), pd.DataFrame(diffs), len(scenes)


def fmt(v, nd=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    runs = [r for r in ORDER4 if os.path.exists(os.path.join(RUNS, r, "metrics_test.json"))]
    if not runs:
        raise SystemExit("no tested runs yet")
    cfg = {r: json.load(open(os.path.join(RUNS, r, "run_config.json"))) for r in runs}
    met = {r: json.load(open(os.path.join(RUNS, r, "metrics_test.json"))) for r in runs}
    cw = {r: pd.read_csv(os.path.join(RUNS, r, "classwise_test.csv"), index_col=0) for r in runs}
    label = {r: f"{r}: {cfg[r].get('title', r)}" for r in runs}
    ci, diffs, n_sc = bootstrap(runs)
    ci.to_csv(os.path.join(OUT, "bootstrap_ci.csv"), index=False)
    diffs.to_csv(os.path.join(OUT, "differences.csv"), index=False)
    cig = lambda r, k: ci[(ci.run == r) & (ci.metric == k)].iloc[0]

    rows = []
    for r in runs:
        rows.append({"run": r, "title": cfg[r].get("title", ""), **met[r]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "all_results.csv"), index=False)

    L = ["# MARIDA, 26 features: comparison of all runs", "",
         "Test split, every annotated pixel, 4 classes (Water, Marine Debris, Algae/Sargassum, Other). "
         f"95% confidence intervals from a scene bootstrap ({n_sc} test scenes, {N_BOOT} resamples). "
         "📄 rows (italic) are the MARIDA paper's published results (Table 4). They are in the paper's 11 "
         "classes, so only Marine Debris is directly comparable; macro scores average over different classes. "
         "Run G2b below compares in the paper's own 11 classes. **Bold** = best of our runs.", ""]
    # overall table
    L += ["## Overall", "", "| Model | Macro F1 [95% CI] | mIoU | κ | OA | Macro P | Macro R | Macro AUPRC | "
          "Debris F1 [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for pname, (mf, mi, pa, df1, di, dpa) in PAPER.items():
        L.append(f"| 📄 **{pname}** (11 classes) | _{mf:.2f}_ | _{mi:.2f}_ | – | – | – | _{pa:.2f}_ | – | _{df1:.2f}_ |")
    best = {k: max(met[r][k] for r in runs) for k in ("macro_F1", "mIoU", "kappa", "OA", "macro_precision",
                                                       "macro_recall", "macro_AUPRC", "F1_Marine Debris")}
    b = lambda r, k: f"**{met[r][k]:.3f}**" if met[r][k] == best[k] else f"{met[r][k]:.3f}"
    for r in runs:
        m1, d1 = cig(r, "macro_F1"), cig(r, "F1_Marine Debris")
        L.append(f"| {label[r]} | {b(r, 'macro_F1')} [{m1.ci_low:.3f}–{m1.ci_high:.3f}] | {b(r, 'mIoU')} | "
                 f"{b(r, 'kappa')} | {b(r, 'OA')} | {b(r, 'macro_precision')} | {b(r, 'macro_recall')} | "
                 f"{b(r, 'macro_AUPRC')} | {b(r, 'F1_Marine Debris')} [{d1.ci_low:.3f}–{d1.ci_high:.3f}] |")
    # per-class tables
    for metric, col in [("F1", "F1"), ("Precision", "precision"), ("Recall (class accuracy)", "recall (class accuracy)"),
                        ("IoU", "IoU"), ("AUPRC", "AUPRC")]:
        L += ["", f"## Per-class {metric}", "", "| Model | " + " | ".join(NAMES) + " |", "|---" * 5 + "|"]
        paper_k = {"F1": 3, "IoU": 4, "Recall (class accuracy)": 5}.get(metric)
        if paper_k is not None:
            for pname, v in PAPER.items():
                L.append(f"| 📄 **{pname}** | – | _{v[paper_k]:.2f}_ | – | – |")
        bc = {c: max(cw[r].loc[c, col] for r in runs) for c in NAMES}
        for r in runs:
            L.append(f"| {label[r]} | " + " | ".join(
                (f"**{cw[r].loc[c, col]:.3f}**" if cw[r].loc[c, col] == bc[c] else f"{cw[r].loc[c, col]:.3f}")
                for c in NAMES) + " |")
    # debris detail
    L += ["", "## Marine debris pixels", "", "| Model | found (TP) | missed (FN) | false alarms (FP) | FP per TP |",
          "|---|---|---|---|---|"]
    for r in runs:
        d = cw[r].loc["Marine Debris"]
        L.append(f"| {label[r]} | {int(d.TP)} / {int(d.support)} | {int(d.FN)} | {int(d.FP)} | {d.FP_per_TP:.2f} |")
    # contrasts
    if len(diffs):
        L += ["", "## Controlled comparisons (paired scene bootstrap)", "",
              "A − B with its 95% CI; **clear** = the CI excludes 0 (single training run per model, so this "
              "covers test-scene sampling, not training randomness).", "",
              "| Comparison | A vs B | what differs | metric | A − B [95% CI] | P(A > B) |", "|---|---|---|---|---|---|"]
        for _, d in diffs.iterrows():
            s = f"{d['A - B']:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}]"
            L.append(f"| {d.comparison} | {d.A} vs {d.B} | {d['what differs']} | {d.metric} | "
                     f"{'**' + s + '**' if d.clear else s} | {d['P(A > B)']:.2f} |")
    # 11-class run vs the paper
    g = os.path.join(RUNS, name("G2b"), "metrics_test.json")
    if os.path.exists(g):
        gm = json.load(open(g))
        gc = pd.read_csv(os.path.join(RUNS, name("G2b"), "classwise_test.csv"), index_col=0)
        cfg_g = json.load(open(os.path.join(RUNS, name("G2b"), "run_config.json")))
        L += ["", "## In the MARIDA paper's 11 classes (run G2b)", "",
              f"G2b: {cfg_g.get('title', '')}, trained and tested on the paper's 11 classes (Waves, Cloud "
              "Shadows, Wakes and Mixed Water merged into Marine Water; Clouds a class), so every number is "
              "directly comparable with the paper.", "",
              "| Model | Macro F1 | mIoU | Macro PA (recall) | " + " | ".join(PAPER_CLASS_F1) + " |",
              "|---" * (4 + len(PAPER_CLASS_F1)) + "|"]
        for k, (pname, v) in enumerate(PAPER.items()):
            L.append(f"| 📄 **{pname}** | _{v[0]:.2f}_ | _{v[1]:.2f}_ | _{v[2]:.2f}_ | "
                     + " | ".join(f"_{PAPER_CLASS_F1[c][k]:.2f}_" for c in PAPER_CLASS_F1) + " |")
        L.append(f"| **G2b (ours)** | {gm['macro_F1']:.3f} | {gm['mIoU']:.3f} | {gm['macro_recall']:.3f} | "
                 + " | ".join(fmt(gc.loc[c, "F1"]) for c in PAPER_CLASS_F1) + " |")
        L += ["", "(per-class values are F1)"]
    # figures
    figs = []
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    for a_, k, t in [(ax[0], "macro_F1", "Macro F1"), (ax[1], "F1_Marine Debris", "Marine Debris F1")]:
        v = [cig(r, k) for r in runs]
        a_.bar(range(len(runs)), [x.value for x in v],
               yerr=[[x.value - x.ci_low for x in v], [x.ci_high - x.value for x in v]], capsize=4,
               color=["#1f77b4"] * 3 + ["#9467bd", "#8c564b", "#7f7f7f", "#2ca02c"][:len(runs) - 3])
        for j, x in enumerate(v):
            a_.text(j, x.ci_high + 0.01, f"{x.value:.3f}", ha="center", fontsize=8)
        if k == "F1_Marine Debris":
            for pname, pv in PAPER.items():
                a_.axhline(pv[3], ls=":", lw=1, color="grey")
                a_.text(len(runs) - 0.5, pv[3], pname, fontsize=6, va="bottom", ha="right")
        a_.set_xticks(range(len(runs)), runs, rotation=30, ha="right", fontsize=8)
        a_.set_ylim(0, 1.05)
        a_.set_title(f"{t} (test, 95% scene-bootstrap CI)" + ("; dotted: MARIDA paper" if "Debris" in t else ""))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figures", "overall_ci.png"), dpi=120)
    plt.close()
    figs.append("overall_ci.png")
    fig, ax = plt.subplots(figsize=(13, 4.8))
    w = 0.8 / len(runs)
    for j, r in enumerate(runs):
        bb = ax.bar(np.arange(4) + (j - len(runs) / 2 + 0.5) * w, cw[r].loc[NAMES, "F1"], w, label=r)
        ax.bar_label(bb, fmt="%.2f", fontsize=6, rotation=90, padding=2)
    ax.set_xticks(range(4), NAMES)
    ax.set_ylim(0, 1.15)
    ax.legend(ncol=len(runs), fontsize=8, loc="upper center")
    ax.set_title("Per-class F1, test")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "figures", "per_class_f1.png"), dpi=120)
    plt.close()
    figs.append("per_class_f1.png")
    if len(diffs):
        d = diffs[diffs.metric.isin(["macro_F1", "F1_Marine Debris"])]
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(d) + 1.5))
        yy = np.arange(len(d))
        ax.errorbar(d["A - B"], yy, xerr=[d["A - B"] - d.ci_low, d.ci_high - d["A - B"]], fmt="o", capsize=3,
                    color="k")
        ax.axvline(0, color="grey")
        ax.set_yticks(yy, [f"{c} ({a} − {b_}), {m.replace('F1_Marine Debris', 'debris F1')}"
                           for c, a, b_, m in zip(d.comparison, d.A, d.B, d.metric)], fontsize=8)
        ax.set_xlabel("difference A − B (95% paired scene-bootstrap CI)")
        ax.set_title("Controlled comparisons")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "figures", "contrasts.png"), dpi=120)
        plt.close()
        figs.append("contrasts.png")
    # cross-model XAI
    xr = [r for r in runs if os.path.exists(os.path.join(XAI, r, "importance_groups.csv"))]
    if xr:
        L += ["", "## Feature importance across models (SHAP)", "",
              "Share of |SHAP| per feature group (%), per class; deep models: GradientSHAP, RF: TreeSHAP, ensemble: "
              "weighted mix of its members (approximation). Details and figures in `xai/<run>/xai_report.md`.", ""]
        G = {r: pd.read_csv(os.path.join(XAI, r, "importance_groups.csv")) for r in xr}
        Fr = {r: pd.read_csv(os.path.join(XAI, r, "importance_features.csv")) for r in xr}
        for c in NAMES:
            L += [f"### {c}", "", "| group | " + " | ".join(xr) + " |", "|---" * (len(xr) + 1) + "|"]
            for gname in F.GROUPS:
                L.append(f"| {gname} | " + " | ".join(
                    fmt(G[r].loc[(G[r]["class"] == c) & (G[r]["group"] == gname), "share_%"].squeeze(), 1)
                    for r in xr) + " |")
            L.append("")
        fig, ax = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
        for a_, c in zip(ax, NAMES):
            bottom = np.zeros(len(xr))
            for gname in F.GROUPS:
                v = np.array([G[r].loc[(G[r]["class"] == c) & (G[r]["group"] == gname), "share_%"].squeeze()
                              for r in xr], dtype=float)
                a_.bar(xr, v, bottom=bottom, color=F.GROUP_COLORS[gname], label=gname)
                bottom += np.nan_to_num(v)
            a_.set_title(c)
            a_.tick_params(axis="x", rotation=45)
        ax[0].set_ylabel("share of |SHAP| (%)")
        ax[0].legend(fontsize=8)
        fig.suptitle("Grouped feature importance per class and model")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "figures", "xai_groups_by_model.png"), dpi=120)
        plt.close()
        figs.append("xai_groups_by_model.png")
        for c in NAMES:
            hm = pd.DataFrame({r: Fr[r][Fr[r]["class"] == c].set_index("feature")["share_%"] for r in xr}
                              ).reindex(F.CHANNELS)
            fig, ax = plt.subplots(figsize=(1.1 * len(xr) + 3, 9))
            im = ax.imshow(hm.values, cmap="viridis")
            ax.set_xticks(range(len(xr)), xr, rotation=45)
            ax.set_yticks(range(len(hm)), hm.index, fontsize=8)
            for t in ax.get_yticklabels():
                t.set_color(F.GROUP_COLORS[F.GROUP_OF[t.get_text()]])
            for i in range(len(hm)):
                for j in range(len(xr)):
                    ax.text(j, i, fmt(hm.values[i, j], 1), ha="center", va="center", fontsize=6, color="w")
            fig.colorbar(im, ax=ax, fraction=0.05, label="share of |SHAP| (%)")
            ax.set_title(f"Feature share, {c}")
            plt.tight_layout()
            f = f"xai_features_{c.split('/')[0].replace(' ', '_').lower()}.png"
            plt.savefig(os.path.join(OUT, "figures", f), dpi=120)
            plt.close()
            figs.append(f)
    L += ["", "## Figures", ""] + [f"![{f}](figures/{f})" for f in figs]
    open(os.path.join(OUT, "comparison.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("->", os.path.join(OUT, "comparison.md"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
