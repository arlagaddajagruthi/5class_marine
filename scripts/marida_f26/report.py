"""
Detailed metrics, stored files and graphs for one run of the MARIDA 26-feature study.

run_report(run_dir) reads what train.py test (or rf_ensemble.py) saved:
  run_config.json, pred_val.npz, pred_test.npz (labels, class probabilities and
  patch index of every annotated pixel), prob_maps_test.npy (optional),
  history.csv and best_epoch.json (deep models only)
and writes, for the validation and the test split:
  metrics_<split>.json        overall: OA, balanced accuracy, kappa, macro / weighted / micro
                              precision, recall, F1, mIoU, macro AUPRC
  classwise_<split>.csv       per class: support, predicted, TP, FP, FN, TN, precision, recall
                              (= class accuracy), F1, IoU, specificity, FP per TP, AUPRC
  confusion_<split>.csv, confusion_rowpct_<split>.csv, confusion_colpct_<split>.csv
  per_scene_<split>.csv       per scene: pixels, OA, macro F1 over the classes present, F1 per class
  pr_curves_<split>.csv       precision-recall curve points per class
  figures/                    training curves, confusion heatmaps, per-class bars, PR curves,
                              per-scene F1, prediction panels
  report.md                   everything above in one page, figures embedded

Usage (repo root):  python scripts/marida_f26/report.py <run_dir>
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, cohen_kappa_score, precision_recall_curve  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CLASS_COLORS4 = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]


def class_colors(n):
    return CLASS_COLORS4 if n == 4 else list(plt.cm.tab20(np.linspace(0, 1, n)))


def confusion(yt, yp, n):
    return np.bincount(yt * n + yp, minlength=n * n).reshape(n, n)


def overall_and_classwise(yt, prob, names):
    n = len(names)
    yp = prob.argmax(1)
    cm = confusion(yt, yp, n)
    tp = np.diag(cm).astype(float)
    sup, pred, tot = cm.sum(1), cm.sum(0), cm.sum()
    fn, fp = sup - tp, pred - tp
    tn = tot - tp - fn - fp
    div = lambda a, b: np.divide(a, b, out=np.full(len(a), np.nan), where=b > 0)
    prec, rec = div(tp, pred), div(tp, sup)
    f1 = div(2 * prec * rec, prec + rec)
    iou = div(tp, tp + fp + fn)
    ap = np.array([average_precision_score(yt == k, prob[:, k]) if sup[k] else np.nan for k in range(n)])
    cw = pd.DataFrame({"support": sup, "share_%": 100 * sup / tot, "predicted": pred, "TP": tp.astype(int),
                       "FP": fp.astype(int), "FN": fn.astype(int), "TN": tn.astype(int),
                       "precision": prec, "recall (class accuracy)": rec, "F1": f1, "IoU": iou,
                       "specificity": div(tn, tn + fp), "FP_per_TP": div(fp, tp), "AUPRC": ap}, index=names)
    present = sup > 0
    w = sup / tot
    p0 = tp.sum() / tot
    pe = (pred * sup).sum() / tot ** 2
    ov = {"pixels": int(tot), "OA": p0, "balanced_accuracy": float(np.nanmean(rec[present])),
          "kappa": float(cohen_kappa_score(yt, yp)) if tot else np.nan,
          "macro_precision": float(np.nanmean(np.nan_to_num(prec[present]))),
          "macro_recall": float(np.nanmean(rec[present])),
          "macro_F1": float(np.nanmean(np.nan_to_num(f1[present]))),
          "mIoU": float(np.nanmean(np.nan_to_num(iou[present]))),
          "weighted_precision": float(np.nansum(np.nan_to_num(prec) * w)),
          "weighted_recall": float(np.nansum(np.nan_to_num(rec) * w)),
          "weighted_F1": float(np.nansum(np.nan_to_num(f1) * w)),
          "micro_F1": p0, "macro_AUPRC": float(np.nanmean(ap[present])),
          **{f"F1_{c}": float(v) for c, v in zip(names, f1)},
          **{f"IoU_{c}": float(v) for c, v in zip(names, iou)}}
    return ov, cw, cm


def per_scene(yt, prob, patch, patch_ids, names):
    scenes = np.array([p.rsplit("_", 1)[0] for p in patch_ids])[patch]
    rows = []
    for s in np.unique(scenes):
        k = scenes == s
        ov, cw, _ = overall_and_classwise(yt[k], prob[k], names)
        rows.append({"scene": s, "pixels": int(k.sum()), "OA": ov["OA"], "macro_F1_present": ov["macro_F1"],
                     **{f"pixels_{c}": int(v) for c, v in zip(names, cw["support"])},
                     **{f"F1_{c}": v for c, v in zip(names, cw["F1"])}})
    return pd.DataFrame(rows).sort_values("pixels", ascending=False)


def md(df, fmt=None):
    fmt = fmt or (lambda c, v: f"{v:,}" if isinstance(v, (int, np.integer)) else
                  ("–" if pd.isna(v) else f"{v:.4f}"))
    cols = list(df.columns)
    out = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, r in df.iterrows():
        out.append(f"| {idx} | " + " | ".join(fmt(c, r[c]) for c in cols) + " |")
    return out


# ------------------------------------------------------------------ figures ---
def fig_history(run, figs, names):
    h = pd.read_csv(os.path.join(run, "history.csv"))
    best = json.load(open(os.path.join(run, "best_epoch.json")))["epoch"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].plot(h.epoch, h.train_loss, "k-")
    ax[0].set_title("training loss (weighted CE)")
    ax[1].plot(h.epoch, h.val_macroF1, "k-", lw=2, label="macro F1")
    for c, col in zip(names, class_colors(len(names))):
        if f"val_F1_{c}" in h:
            ax[1].plot(h.epoch, h[f"val_F1_{c}"], color=col, lw=1, label=c)
    ax[1].set_title("validation F1 (soft TTA)")
    ax[1].legend(fontsize=7, ncol=2)
    ax[2].plot(h.epoch, h.lr, "k-")
    ax[2].set_yscale("log")
    ax[2].set_title("learning rate")
    for a in ax:
        a.axvline(best, color="grey", ls="--", lw=1)
        a.set_xlabel("epoch")
    fig.suptitle(f"Training history (dashed: selected epoch {best})")
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "training_curves.png"), dpi=120)
    plt.close()
    return "training_curves.png"


def fig_confusion(cm, names, figs, split):
    n = len(names)
    rowp = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(6 + 0.9 * n, 2.5 + 0.45 * n))
    for a, mat, title, fmt in [(ax[0], cm, "pixel counts", "{:,}"), (ax[1], rowp, "% of true class (recall)", "{:.1f}")]:
        im = a.imshow(np.log10(mat + 1) if a is ax[0] else mat, cmap="Blues")
        for i in range(n):
            for j in range(n):
                a.text(j, i, fmt.format(int(mat[i, j]) if a is ax[0] else mat[i, j]), ha="center", va="center",
                       fontsize=7 if n > 4 else 9, color="white" if (rowp[i, j] > 50) else "black")
        a.set_xticks(range(n), names, rotation=45, ha="right", fontsize=8)
        a.set_yticks(range(n), names, fontsize=8)
        a.set_xlabel("predicted")
        a.set_ylabel("true")
        a.set_title(title)
    fig.suptitle(f"Confusion matrix, {split}")
    plt.tight_layout()
    f = f"confusion_{split}.png"
    plt.savefig(os.path.join(figs, f), dpi=120)
    plt.close()
    return f


def fig_classbars(cw, figs, split):
    metrics = ["precision", "recall (class accuracy)", "F1", "IoU"]
    n = len(cw)
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * n), 4.5))
    x = np.arange(n)
    for k, m in enumerate(metrics):
        b = ax.bar(x + (k - 1.5) * 0.2, cw[m].values, 0.2, label=m)
        ax.bar_label(b, fmt="%.2f", fontsize=6, rotation=90, padding=2)
    ax.set_xticks(x, cw.index, rotation=30 if n > 4 else 0, ha="right" if n > 4 else "center")
    ax.set_ylim(0, 1.12)
    ax.legend(ncol=4, fontsize=8, loc="upper center")
    ax.set_title(f"Per-class scores, {split}")
    plt.tight_layout()
    f = f"classwise_{split}.png"
    plt.savefig(os.path.join(figs, f), dpi=120)
    plt.close()
    return f


def fig_pr(yt, prob, names, figs, split, rows):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for k, (c, col) in enumerate(zip(names, class_colors(len(names)))):
        if not (yt == k).any():
            continue
        p, r, _ = precision_recall_curve(yt == k, prob[:, k])
        ap = average_precision_score(yt == k, prob[:, k])
        step = max(1, len(p) // 400)
        rows += [{"class": c, "recall": rv, "precision": pv} for rv, pv in zip(r[::step], p[::step])]
        ax.plot(r, p, color=col, label=f"{c} (AUPRC {ap:.3f})")
        base = (yt == k).mean()
        ax.axhline(base, color=col, ls=":", lw=0.8)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    ax.set_title(f"Precision-recall per class, {split}\n(dotted: class share = chance level)")
    plt.tight_layout()
    f = f"pr_curves_{split}.png"
    plt.savefig(os.path.join(figs, f), dpi=120)
    plt.close()
    return f


def fig_scenes(ps, names, figs, split):
    col = f"F1_{names[1]}" if len(names) == 4 else "macro_F1_present"
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(ps)), 4.5))
    x = np.arange(len(ps))
    ax.bar(x - 0.2, ps["macro_F1_present"], 0.4, label="macro F1 (classes present)", color="grey")
    if len(names) == 4:
        ax.bar(x + 0.2, ps[col].fillna(0), 0.4, label="Marine Debris F1", color=CLASS_COLORS4[1])
    ax.set_xticks(x, ps["scene"], rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title(f"Per-scene scores, {split} (scenes ordered by annotated pixels; debris F1 blank = no debris)")
    plt.tight_layout()
    f = f"per_scene_{split}.png"
    plt.savefig(os.path.join(figs, f), dpi=120)
    plt.close()
    return f


def example_patches(run_cfg, n_each=2):
    """Fixed test patches for prediction panels: most pixels of each non-water class (same for every run)."""
    import features26 as F
    from src.datasets.marida import MARIDADataset
    ds = MARIDADataset(F.DATA, split="test")
    counts = np.array([[(ds[i][1].numpy() == c).sum() for c in (1, 2, 3, 4)] for i in range(len(ds))])
    picks = []
    for c in (1, 2, 3):                                  # debris, algae, other
        for i in np.argsort(-counts[:, c]):
            if i not in picks and counts[i, c] > 0:
                picks.append(int(i))
                if sum(1 for p in picks if counts[p, c] > 0) >= n_each:
                    break
    return picks[:6], ds


def fig_panels(run, figs, names):
    path = os.path.join(run, "prob_maps_test.npy")
    if not os.path.exists(path):
        return None
    import features26 as F
    cache = os.path.join(os.path.dirname(os.path.dirname(run)), "example_patches.json")
    if os.path.exists(cache):
        picks = json.load(open(cache))
        from src.datasets.marida import MARIDADataset
        ds = MARIDADataset(F.DATA, split="test")
    else:
        picks, ds = example_patches(None)
        json.dump(picks, open(cache, "w"))
    maps = np.load(path, mmap_mode="r")
    n = len(names)
    cmap = matplotlib.colors.ListedColormap(["#ffffff"] + [matplotlib.colors.to_hex(c) for c in class_colors(n)])
    fig, ax = plt.subplots(len(picks), 5, figsize=(19, 3.8 * len(picks)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(picks):
        img, m = (t.numpy() for t in ds[i])
        if n == 11:                                       # 11-class runs: labels from the run's own masks
            import train
            m = train.RunDataset("test", False, "raw", 11).unscaled(i)[1]
        rgb = np.stack([img[3], img[2], img[1]], -1)
        lo, hi = np.percentile(rgb, [2, 98])
        rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
        p = np.asarray(maps[i], np.float32)
        pred = p.argmax(0) + 1
        lab = m > 0
        err = np.where(~lab, 0, np.where(pred == m, 1, 2))
        ax[r, 0].imshow(rgb)
        ax[r, 0].set_title(f"RGB, {ds.patch_ids[i]}", fontsize=9)
        ax[r, 1].imshow(m, cmap=cmap, vmin=0, vmax=n, interpolation="nearest")
        ax[r, 1].set_title("ground truth (white: unlabelled)", fontsize=9)
        ax[r, 2].imshow(pred, cmap=cmap, vmin=0, vmax=n, interpolation="nearest")
        ax[r, 2].set_title("prediction (every pixel)", fontsize=9)
        ax[r, 3].imshow(err, cmap=matplotlib.colors.ListedColormap(["#ffffff", "#2ca02c", "#d62728"]), vmin=0,
                        vmax=2, interpolation="nearest")
        ax[r, 3].set_title("labelled pixels: green correct, red wrong", fontsize=9)
        k = 1 if n == 4 else 0
        im = ax[r, 4].imshow(p[k], cmap="magma", vmin=0, vmax=1)
        ax[r, 4].set_title(f"P({names[k]})", fontsize=9)
        fig.colorbar(im, ax=ax[r, 4], fraction=0.046)
        for a in ax[r]:
            a.axis("off")
    handles = [matplotlib.patches.Patch(color=matplotlib.colors.to_hex(c), label=nm)
               for c, nm in zip(class_colors(n), names)]
    fig.legend(handles=handles, loc="upper center", ncol=min(n, 6), fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(os.path.join(figs, "prediction_panels_test.png"), dpi=90)
    plt.close()
    return "prediction_panels_test.png"


# ------------------------------------------------------------------- report ---
def run_report(run):
    cfg = json.load(open(os.path.join(run, "run_config.json")))
    names = cfg["class_names"]
    figs = os.path.join(run, "figures")
    os.makedirs(figs, exist_ok=True)
    title = cfg.get("title", f"{cfg['model']} ({cfg['init']} init, {cfg['bands']} bands)")
    lines = [f"# Run {cfg['run']}: {title}, MARIDA, {len(names)} classes", "",
             f"Inputs: {len(cfg['channels'])} channels ({', '.join(cfg['channels'])}). "
             "See `architecture.md` for the model and how the features are fed in.", ""]
    if os.path.exists(os.path.join(run, "best_epoch.json")):
        b = json.load(open(os.path.join(run, "best_epoch.json")))
        lines += [f"Selected checkpoint: epoch {b['epoch']} of {cfg['recipe']['epochs']} (best validation macro "
                  f"F1 with soft TTA, {b['val_macroF1_tta']:.4f}).", "",
                  f"![training curves](figures/{fig_history(run, figs, names)})", ""]
    if cfg.get("notes"):
        lines += [cfg["notes"], ""]
    summary = {}
    for split in ("test", "val"):
        z = np.load(os.path.join(run, f"pred_{split}.npz"))
        yt, prob, patch, pids = z["y"], z["prob"].astype(np.float64), z["patch"], z["patch_ids"]
        ov, cw, cm = overall_and_classwise(yt, prob, names)
        summary[split] = ov
        json.dump(ov, open(os.path.join(run, f"metrics_{split}.json"), "w"), indent=2)
        cw.to_csv(os.path.join(run, f"classwise_{split}.csv"), index_label="class")
        pd.DataFrame(cm, index=names, columns=names).to_csv(os.path.join(run, f"confusion_{split}.csv"))
        rowp = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
        colp = 100 * cm / np.maximum(cm.sum(0, keepdims=True), 1)
        pd.DataFrame(rowp, index=names, columns=names).to_csv(os.path.join(run, f"confusion_rowpct_{split}.csv"))
        pd.DataFrame(colp, index=names, columns=names).to_csv(os.path.join(run, f"confusion_colpct_{split}.csv"))
        ps = per_scene(yt, prob, patch, pids, names)
        ps.to_csv(os.path.join(run, f"per_scene_{split}.csv"), index=False)
        pr_rows = []
        f_pr = fig_pr(yt, prob, names, figs, split, pr_rows)
        pd.DataFrame(pr_rows).to_csv(os.path.join(run, f"pr_curves_{split}.csv"), index=False)
        f_cm, f_cw, f_ps = fig_confusion(cm, names, figs, split), fig_classbars(cw, figs, split), \
            fig_scenes(ps, names, figs, split)
        lines += [f"## {split.capitalize()} split ({ov['pixels']:,} annotated pixels)", "", "### Overall", "",
                  "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {v:.4f} |" for k, v in ov.items() if k != "pixels" and not k.startswith(("F1_", "IoU_"))]
        lines += ["", "### Per class", ""] + md(cw)
        lines += ["", "### Confusion matrix (rows true, columns predicted): counts / % of true class / % of "
                  "predicted class", ""]
        lines += md(pd.DataFrame(cm, index=names, columns=names), lambda c, v: f"{int(v):,}") + [""]
        lines += md(pd.DataFrame(rowp, index=names, columns=names), lambda c, v: f"{v:.2f}") + [""]
        lines += md(pd.DataFrame(colp, index=names, columns=names), lambda c, v: f"{v:.2f}") + [""]
        lines += [f"![confusion](figures/{f_cm})", "", f"![per-class](figures/{f_cw})", "",
                  f"![PR curves](figures/{f_pr})", "", "### Per scene", "",
                  f"`per_scene_{split}.csv` has every scene; the figure shows macro F1 (classes present) and "
                  "debris F1.", "", f"![per scene](figures/{f_ps})", ""]
    f_pan = fig_panels(run, figs, names)
    if f_pan:
        lines += ["## Example test patches", "", "The same 6 test patches (most debris, algae and other pixels) "
                  "in every run.", "", f"![panels](figures/{f_pan})", ""]
    open(os.path.join(run, "report.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"report -> {os.path.join(run, 'report.md')}  (test macro F1 {summary['test']['macro_F1']:.4f}, "
          f"mIoU {summary['test']['mIoU']:.4f}, kappa {summary['test']['kappa']:.4f})")
    return summary


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run_report(sys.argv[1])
