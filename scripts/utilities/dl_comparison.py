"""Comparison of all deep-learning models (and the RF / ensembles they are compared with),
with the MARIDA and MADOS paper results at the top of every table.

Every pixel-level score of our models is recomputed from the saved test confusion
matrices, so all models are scored the same way (classes 1-4, all annotated test pixels).

Inputs (from the repo root):
    outputs/deeplabv3_resnet50/<ds>/confusion_model_best_val_loss.csv
    outputs/{unet_ssl4eo,segnext,marinext_arch}/<run>/confusion_model_best_val_f1.csv
    outputs/staged/log_weights/stage{5,8}_*/<ds>/confusion_matrices/rf.csv
    outputs/ensemble_unet_rf[_stage8]/<ds>/{confusion_ensemble.csv,report.md}
    outputs/resnet_paper/<ds>/classwise_model_best_val_loss.csv  (patch-level, separate table)
Paper values: MARIDA.pdf Table 4, MADOS.pdf Tables 4 and 5 (typed in below).

Output: outputs/dl_comparison/comparison.md and all_results.csv

    python scripts/utilities/dl_comparison.py
"""
import os
import re

import numpy as np
import pandas as pd

OUT = os.path.join("outputs", "dl_comparison")
CLASSES = ["Water", "Marine Debris", "Algae/Sargassum", "Other"]
SHORT = ["water", "debris", "algae", "other"]
NAN = float("nan")

# Published results in the papers' own class schemes. MARIDA Table 4 (11 classes,
# PA = recall, no precision/kappa/OA); MADOS Table 4 (15 classes, mean of 5 runs)
# and Table 5 (MariNeXt per class). Only marine debris maps 1:1 onto our classes.
PUBLISHED = {
    "MARIDA": [
        ("RF, spectral (RF_SS)", "11 classes", dict(macro_f1=.69, miou=.57, macro_rec=.72, f1_debris=.71, rec_debris=.91, iou_debris=.55)),
        ("RF + indices (RF_SS+SI)", "11 classes", dict(macro_f1=.70, miou=.59, macro_rec=.73, f1_debris=.80, rec_debris=.92, iou_debris=.67)),
        ("RF + indices + GLCM", "11 classes", dict(macro_f1=.79, miou=.69, macro_rec=.79, f1_debris=.79, rec_debris=.92, iou_debris=.65)),
        ("U-Net", "11 classes", dict(macro_f1=.69, miou=.57, macro_rec=.69, f1_debris=.50, rec_debris=.70, iou_debris=.33)),
    ],
    "MADOS": [
        ("MariNeXt", "15 classes", dict(macro_f1=.760, miou=.643, oa=.891, f1_debris=.513, prec_debris=.406, rec_debris=.704, iou_debris=.346)),
        ("SegNeXt-T", "15 classes", dict(macro_f1=.606, miou=.492, oa=.866)),
        ("U-Net", "15 classes", dict(macro_f1=.638, miou=.510, oa=.829)),
        ("RF++", "15 classes", dict(macro_f1=.644, miou=.524, oa=.838)),
        ("RF", "15 classes", dict(macro_f1=.566, miou=.439, oa=.671)),
    ],
}

# MARIDA paper Table 4, F1 per original class: (RF_SS, RF_SS+SI, RF_SS+SI+GLCM, U-Net)
MARIDA_PAPER_CLASS_F1 = [
    ("Water", "Marine Water", (.79, .79, .86, .76)),
    ("Water", "Sediment-Laden Water", (1, 1, 1, 1)),
    ("Water", "Turbid Water", (.85, .89, .94, .91)),
    ("Water", "Shallow Water", (.16, .23, .46, .62)),
    ("Marine Debris", "Marine Debris", (.71, .80, .79, .50)),
    ("Algae/Sargassum", "Dense Sargassum", (.93, .93, .93, .75)),
    ("Algae/Sargassum", "Sparse Sargassum", (.69, .82, .91, .79)),
    ("Other", "Natural Organic Material", (.47, .29, .31, .04)),
    ("Other", "Ship", (.70, .64, .80, .76)),
    ("Other", "Foam", (.37, .47, .75, .71)),
    ("ignored (class 0)", "Clouds", (.86, .85, .91, .76)),
]


def read_cm(path):
    return pd.read_csv(path, index_col=0).loc[CLASSES, CLASSES].to_numpy(dtype=np.int64)


def scores(cm):
    tp = np.diag(cm).astype(float)
    pred, true, n = cm.sum(0), cm.sum(1), cm.sum()
    prec = np.divide(tp, pred, out=np.zeros(4), where=pred > 0)
    rec = np.divide(tp, true, out=np.zeros(4), where=true > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0)
    iou = tp / (pred + true - tp)
    oa = tp.sum() / n
    pe = (pred * true).sum() / n ** 2
    r = {"macro_f1": f1.mean(), "miou": iou.mean(), "kappa": (oa - pe) / (1 - pe), "oa": oa,
         "macro_prec": prec.mean(), "macro_rec": rec.mean()}
    for i, s in enumerate(SHORT):
        r[f"prec_{s}"], r[f"rec_{s}"], r[f"f1_{s}"], r[f"iou_{s}"] = prec[i], rec[i], f1[i], iou[i]
    r["debris_tp"], r["debris_fp"], r["debris_fn"] = int(tp[1]), int(pred[1] - tp[1]), int(true[1] - tp[1])
    r["water_to_other"], r["other_to_water"] = int(cm[0, 3]), int(cm[3, 0])
    r["n_water"], r["n_other"] = int(true[0]), int(true[3])
    return r


def staged_rf_dir(stage):
    base = os.path.join("outputs", "staged", "log_weights")
    return os.path.join(base, next(d for d in os.listdir(base) if d.startswith(f"stage{stage}_")))


def ensemble_setting(folder, ds):
    """Chosen U-Net run and weight w from the ensemble report."""
    text = open(os.path.join(folder, ds, "report.md"), encoding="utf-8").read()
    m = re.search(r"\*\*(\S+), w_unet = ([\d.]+)\*\*", text)
    return f"U-Net {m.group(1).split('_', 1)[1]}, w_unet = {float(m.group(2)):.2f}"


def collect():
    rows = []

    def add(ds, model, kind, setting, r):
        rows.append({"dataset": ds, "model": model, "kind": kind, "setting": setting, **r})

    # paper rows first, so they sit at the top of every table
    for ds, entries in PUBLISHED.items():
        for model, classes, vals in entries:
            add(ds, model, "paper", f"{ds} paper, {classes}", vals)

    for ds in ["MARIDA", "MADOS"]:
        for st in (5, 8):
            add(ds, f"Random Forest, stage {st}", "ML", "log weights, per-pixel features",
                scores(read_cm(os.path.join(staged_rf_dir(st), ds, "confusion_matrices", "rf.csv"))))

        add(ds, "DeepLabv3-ResNet50", "DL", "11 bands, from scratch",
            scores(read_cm(os.path.join("outputs", "deeplabv3_resnet50", ds, "confusion_model_best_val_loss.csv"))))

        for folder, name, inp in [("unet_ssl4eo", "U-Net (SSL4EO encoder)", "16 ch, pretrained"),
                                  ("segnext", "SegNeXt-T", "16 ch, from scratch"),
                                  ("marinext_arch", "MariNeXt architecture", "16 ch, from scratch")]:
            runs = ["MARIDA_novscp", "MARIDA_vscp"] if ds == "MARIDA" else ["MADOS_vscp"]
            for run in runs:
                vscp = "VSCP" if run.endswith("_vscp") else "no VSCP"
                add(ds, name, "DL", f"{inp}, {vscp}",
                    scores(read_cm(os.path.join("outputs", folder, run, "confusion_model_best_val_f1.csv"))))

        for folder, st in [("ensemble_unet_rf", 5), ("ensemble_unet_rf_stage8", 8)]:
            folder = os.path.join("outputs", folder)
            add(ds, f"U-Net + RF ensemble (RF stage {st})", "DL + ML", ensemble_setting(folder, ds),
                scores(read_cm(os.path.join(folder, ds, "confusion_ensemble.csv"))))

    return pd.DataFrame(rows)


def label(r):
    if r.kind == "paper":
        return f"📄 **{r.model}** ({r.setting})"
    return f"{r.model} ({r.setting})"


def fmt_col(d, col):
    """Paper values in italics; the best of our own models in bold."""
    ours = d.loc[d.kind != "paper", col].dropna()
    best = ours.max() if len(ours) else None
    out = []
    for v, k in zip(d[col], d.kind):
        if pd.isna(v):
            out.append("–")
        elif k == "paper":
            out.append(f"_{v:.3f}_")
        else:
            out.append(f"**{v:.3f}**" if v == best else f"{v:.3f}")
    return out


def table(d, cols, headers):
    # drop paper rows that have none of these metrics
    d = d[(d.kind != "paper") | d[cols].notna().any(axis=1)].reset_index(drop=True)
    lines = ["| Model | " + " | ".join(headers) + " |", "|---" * (len(cols) + 1) + "|"]
    formatted = {c: fmt_col(d, c) for c in cols}
    for i, (_, r) in enumerate(d.iterrows()):
        lines.append(f"| {label(r)} | " + " | ".join(formatted[c][i] for c in cols) + " |")
    return lines


def marida_paper_class_section():
    lines = ["### MARIDA paper, F1 per original class (Table 4)", "",
             "The MARIDA paper gives no confusion matrix, so its classes cannot be merged into ours. "
             "These are its per-class F1 scores, grouped by the class of ours they belong to.", "",
             "| Our class | MARIDA class | RF_SS | RF_SS+SI | RF_SS+SI+GLCM | U-Net |", "|---|---|---|---|---|---|"]
    for ours, theirs, v in MARIDA_PAPER_CLASS_F1:
        lines.append(f"| {ours} | {theirs} | " + " | ".join(f"_{x:.2f}_" for x in v) + " |")
    return lines + [""]


def resnet_section():
    lines = ["## ResNet-50 patch classifier (MARIDA paper's multi-label baseline)", "",
             "Different task: it predicts which classes are *present in a patch* (multi-label, "
             "threshold 0.5), not a class per pixel, so it is not comparable with the tables above. "
             "Scores are per patch.", "",
             "| Dataset | Class | Patches with class | Precision | Recall | F1 |", "|---|---|---|---|---|---|"]
    for ds in ["MARIDA", "MADOS"]:
        c = pd.read_csv(os.path.join("outputs", "resnet_paper", ds, "classwise_model_best_val_loss.csv"))
        for _, r in c.iterrows():
            lines.append(f"| {ds} | {r['class']} | {int(r.support)} | {r.precision:.3f} | {r.recall:.3f} | {r.F1:.3f} |")
        lines.append(f"| {ds} | **macro** | | {c.precision.mean():.3f} | {c.recall.mean():.3f} | {c.F1.mean():.3f} |")
    return lines + [""]


def main():
    df = collect()
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "all_results.csv"), index=False)

    L = ["# Deep-learning model comparison", "",
         "Our scores are recomputed from the saved **test** confusion matrices: classes 1-4, every "
         "annotated pixel of the official test split, macro = mean over the 4 classes. "
         "The Random Forest rows are the per-pixel baselines from the staged study (log class weights).", "",
         "📄 **Paper rows** (top of each table, values in _italics_) are the published results. "
         "**Bold** = best of our models in the column.", "",
         "The paper results are in the papers' own class schemes (MARIDA 11 classes, MADOS 15 "
         "classes), so only marine debris is the same class; their macro scores average over more, "
         "and different, classes. The MARIDA paper reports no precision, kappa or OA, and its 'PA' "
         "is recall.", "",
         "Inputs: DeepLabv3 uses the 11 Sentinel-2 bands; U-Net, SegNeXt-T and the MariNeXt "
         "architecture use 16 channels (11 bands + NDWI + red/green, NIR/red, NIR/green, SWIR1/NIR). "
         "The ensemble averages U-Net and RF class probabilities, p = w·p_U-Net + (1−w)·p_RF.", ""]

    for ds in ["MARIDA", "MADOS"]:
        d = df[df.dataset == ds].reset_index(drop=True)
        L += [f"## {ds}", "", "### Overall", ""]
        L += table(d, ["macro_f1", "miou", "kappa", "oa", "macro_prec", "macro_rec"],
                   ["Macro F1", "mIoU", "κ", "OA", "Macro P", "Macro R"])
        for metric, name in [("f1", "F1"), ("prec", "Precision"), ("rec", "Recall"), ("iou", "IoU")]:
            L += ["", f"### Per-class {name}", ""]
            L += table(d, [f"{metric}_{s}" for s in SHORT], CLASSES)
        if ds == "MARIDA":
            L += [""] + marida_paper_class_section()
        L += ["", "### Marine debris pixels and the water / other confusion", "",
              "| Model | Debris found (TP) | Debris missed (FN) | False alarms (FP) | FP per TP | "
              "Water → Other | Other → Water |", "|---|---|---|---|---|---|---|"]
        for _, r in d[d.debris_tp.notna()].iterrows():
            tp, fn, fp = int(r.debris_tp), int(r.debris_fn), int(r.debris_fp)
            L.append(f"| {label(r)} | {tp} / {tp + fn} | {fn} | {fp} | {fp / max(tp, 1):.2f} | "
                     f"{int(r.water_to_other):,} ({100 * r.water_to_other / r.n_water:.1f} %) | "
                     f"{int(r.other_to_water):,} ({100 * r.other_to_water / r.n_other:.1f} %) |")
        L.append("")

    L += resnet_section()

    path = os.path.join(OUT, "comparison.md")
    open(path, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", path, "and all_results.csv")


if __name__ == "__main__":
    main()
