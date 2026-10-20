"""
Fine-to-coarse experiment: does merging the original 15 classes into 4 before
training cost accuracy?

Same pixels, same stage-5 features (11 bands + NDWI + 4 ratios), same Random
Forest and seed as the staged runs. Only the training labels change:

  coarse        : train on the 4 merged classes (= stage 5, log weights)
  fine_logw     : train on the original classes; log weights 1/ln(1.03 + freq)
                  computed over the original classes
  fine_coarsew  : train on the original classes; each pixel keeps the weight of
                  its merged class (isolates the effect of the labels alone)

Fine predictions are mapped back to the 4 classes with the mapping used to build
data/*_5Class (scripts/utilities/convert.py), so all three variants are scored on exactly
the same 4-class test labels.

The original masks are read from --orig-root (default
C:/PROJECT/marine_pollution_detection/{MARIDA,MADOS}); the cached stage pixels
are matched to them pixel by pixel and the merge is checked.

Output: outputs/staged/fine_to_coarse/<dataset>/
  summary.md, metrics.csv, per_original_class.csv, confusion_matrices/
"""
import os

import numpy as np
import pandas as pd
import tifffile

from common import (BAND_NAMES, CLASSES, CLASS_NAMES, DATASETS, LOG_WEIGHT_C, OUT_ROOT, RATIOS,
                    base_parser, columns, evaluate, load_data, make_model, plot_cm)

MAPS = {  # original class -> merged class (scripts/utilities/convert.py)
    "MARIDA": {0: 0, 1: 2, 2: 3, 3: 3, 4: 4, 5: 4, 6: 0, 7: 1, 8: 1, 9: 4, 10: 1, 11: 1, 12: 1, 13: 0,
               14: 1, 15: 1},
    "MADOS": {0: 0, 1: 2, 2: 3, 3: 3, 4: 4, 5: 4, 6: 4, 7: 1, 8: 1, 9: 4, 10: 1, 11: 1, 12: 1, 13: 4,
              14: 4, 15: 4},
}
ORIG_NAMES = {
    "MARIDA": {1: "Marine Debris", 2: "Dense Sargassum", 3: "Sparse Sargassum", 4: "Natural Organic Material",
               5: "Ship", 6: "Clouds", 7: "Marine Water", 8: "Sediment-Laden Water", 9: "Foam",
               10: "Turbid Water", 11: "Shallow Water", 12: "Waves", 13: "Cloud Shadows", 14: "Wakes",
               15: "Mixed Water"},
    "MADOS": {1: "Marine Debris", 2: "Dense Sargassum", 3: "Sparse Floating Algae",
              4: "Natural Organic Material", 5: "Ship", 6: "Oil Spill", 7: "Marine Water",
              8: "Sediment-Laden Water", 9: "Foam", 10: "Turbid Water", 11: "Shallow Water",
              12: "Waves & Wakes", 13: "Oil Platform", 14: "Jellyfish", 15: "Sea snarl"},
}
FEATURES = list(BAND_NAMES) + ["NDWI"] + RATIOS
VARIANTS = ["coarse", "fine_logw", "fine_coarsew"]


def mask_rel(dataset_name, pid):
    if dataset_name == "MARIDA":
        scene = pid.rsplit("_", 1)[0]
        return os.path.join("patches", f"S2_{scene}", f"S2_{pid}_cl.tif")
    scene, k = pid.rsplit("_", 1)
    return os.path.join(scene, "10", f"{scene}_L2R_cl_{k}.tif")


def fine_labels(dataset_name, split, orig_root, y_cached):
    """Original-class label of every cached pixel, in the cache's order
    (sample_split with cap=None: per patch, classes 1-4 in turn, pixels in raster order)."""
    root = DATASETS[dataset_name][1]
    lut = np.zeros(16, dtype=np.int64)
    for k, v in MAPS[dataset_name].items():
        lut[k] = v
    with open(os.path.join(root, "splits", f"{split}_X.txt")) as f:
        ids = [l.strip() for l in f if l.strip()]
    out = []
    for pid in ids:
        m4 = tifffile.imread(os.path.join(root, mask_rel(dataset_name, pid))).astype(np.int64).ravel()
        mo = tifffile.imread(os.path.join(orig_root, dataset_name, mask_rel(dataset_name, pid))).astype(np.int64).ravel()
        if not np.array_equal(lut[mo], m4):
            raise ValueError(f"{dataset_name} {pid}: original mask does not merge to the 4-class mask")
        sel = np.concatenate([np.flatnonzero(m4 == c) for c in CLASSES])
        out.append(mo[sel])
    fine = np.concatenate(out)
    if len(fine) != len(y_cached) or not np.array_equal(lut[fine], y_cached):
        raise ValueError(f"{dataset_name}/{split}: fine labels do not line up with the cached pixels")
    return fine


def log_weights(labels):
    vals, counts = np.unique(labels, return_counts=True)
    w = 1.0 / np.log(LOG_WEIGHT_C + counts / counts.sum())
    return dict(zip(vals, w))


def run(dataset_name, args):
    out_dir = os.path.join(args.out_root, "fine_to_coarse", dataset_name)
    os.makedirs(os.path.join(out_dir, "confusion_matrices"), exist_ok=True)
    X_tr, y_tr, X_te, y_te = load_data(dataset_name, None, args.seed)
    f_tr = fine_labels(dataset_name, "train", args.orig_root, y_tr)
    f_te = fine_labels(dataset_name, "test", args.orig_root, y_te)
    cols = columns(FEATURES)
    Xtr, Xte = X_tr[:, cols], X_te[:, cols]
    to_idx = {c: i for i, c in enumerate(CLASSES)}
    ytr, yte = np.vectorize(to_idx.get)(y_tr), np.vectorize(to_idx.get)(y_te)
    fine_to_idx = {k: to_idx[v] for k, v in MAPS[dataset_name].items() if v in to_idx}
    print(f"\n=== {dataset_name}: train {len(ytr):,} px, test {len(yte):,} px", flush=True)
    print("  original classes, train px:", dict(zip(*np.unique(f_tr, return_counts=True))), flush=True)

    wc = log_weights(ytr)
    rows, per_orig, preds = [], [], {}
    for v in VARIANTS:
        if v == "coarse":
            target, sw = ytr, np.vectorize(wc.get)(ytr)
        elif v == "fine_logw":
            wf = log_weights(f_tr)
            target, sw = f_tr, np.vectorize(wf.get)(f_tr)
        else:
            target, sw = f_tr, np.vectorize(wc.get)(ytr)
        model = make_model("rf", args.seed)
        model.fit(Xtr, target, sample_weight=sw)
        raw = np.asarray(model.predict(Xte)).astype(int)
        pred = raw if v == "coarse" else np.vectorize(fine_to_idx.get)(raw)
        preds[v] = pred
        m, cm = evaluate(yte, pred)
        rows.append({"dataset": dataset_name, "variant": v, "n_train_classes": len(np.unique(target)), **m})
        print(f"  {v:13s} macroF1={m['f1']:.4f} IoU={m['iou']:.4f} debris F1={m['f1_c2']:.4f} "
              f"algae F1={m['f1_c3']:.4f} other F1={m['f1_c4']:.4f}", flush=True)
        names = [CLASS_NAMES[c] for c in CLASSES]
        pd.DataFrame(cm, index=names, columns=names).to_csv(os.path.join(out_dir, "confusion_matrices", f"{v}.csv"))
        plot_cm(cm, os.path.join(out_dir, "confusion_matrices", f"{v}.png"), f"fine_to_coarse | {dataset_name} | {v}")

    # Where each ORIGINAL test class ends up (merged predictions), per variant
    for k in np.unique(f_te):
        sel = f_te == k
        r = {"orig_class": int(k), "orig_name": ORIG_NAMES[dataset_name][k],
             "merged_into": CLASS_NAMES[MAPS[dataset_name][k]], "test_px": int(sel.sum())}
        for v in VARIANTS:
            counts = np.bincount(preds[v][sel], minlength=len(CLASSES))
            r[f"{v}_correct_%"] = 100 * counts[to_idx[MAPS[dataset_name][k]]] / sel.sum()
            for i, c in enumerate(CLASSES):
                r[f"{v}_as_{CLASS_NAMES[c]}"] = int(counts[i])
        per_orig.append(r)

    df, po = pd.DataFrame(rows), pd.DataFrame(per_orig)
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    po.to_csv(os.path.join(out_dir, "per_original_class.csv"), index=False)
    write_summary(df, po, os.path.join(out_dir, "summary.md"), dataset_name)
    return df


def write_summary(df, po, path, dataset_name):
    L = [f"# Fine-to-coarse training, {dataset_name} (Random Forest, stage-5 features)", "",
         f"Features ({len(FEATURES)}): {', '.join(FEATURES)}", "",
         "All variants are scored on the same 4-class test labels (annotated pixels of the official "
         "test split). Fine models predict an original class, which is then merged to the 4 classes.", "",
         "- coarse: trained on the 4 merged classes, log weights (identical to stage 5, log weights)",
         "- fine_logw: trained on the original classes, log weights over the original classes",
         "- fine_coarsew: trained on the original classes, each pixel weighted as its merged class", "",
         "## Overall", "",
         "| variant | train classes | macro F1 | IoU | precision | recall | accuracy | kappa |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r['variant']} | {r['n_train_classes']} | {r['f1']:.4f} | {r['iou']:.4f} | "
                 f"{r['precision']:.4f} | {r['recall']:.4f} | {r['accuracy']:.4f} | {r['kappa']:.4f} |")
    base = df[df.variant == "coarse"].iloc[0]
    for metric, label in [("acc", "accuracy (TP / class pixels = recall)"), ("precision", "precision"),
                          ("recall", "recall"), ("f1", "F1"), ("iou", "IoU"),
                          ("ovr_acc", "one-vs-rest accuracy ((TP + TN) / all pixels)")]:
        L += ["", f"## Per-class {label}", "",
              "| variant | " + " | ".join(CLASS_NAMES[c] for c in CLASSES) + " |", "|" + "---|" * (1 + len(CLASSES))]
        for _, r in df.iterrows():
            cells = []
            for c in CLASSES:
                val, d = r[f"{metric}_c{c}"], r[f"{metric}_c{c}"] - base[f"{metric}_c{c}"]
                cells.append(f"{val:.4f}" if r["variant"] == "coarse" else f"{val:.4f} ({d:+.4f})")
            L.append(f"| {r['variant']} | " + " | ".join(cells) + " |")
    L += ["", "Values in brackets: change from the coarse model.", "",
          "## Original classes in the test split: % of pixels given the correct merged class", "",
          "| original class | merged into | test px | " + " | ".join(VARIANTS) + " | coarse model's main error |",
          "|---|---|---|" + "---|" * len(VARIANTS) + "---|"]
    for _, r in po.iterrows():
        wrong = {CLASS_NAMES[c]: r[f"coarse_as_{CLASS_NAMES[c]}"] for c in CLASSES if CLASS_NAMES[c] != r["merged_into"]}
        top = max(wrong, key=wrong.get)
        err = f"{top} {100 * wrong[top] / r['test_px']:.1f}%" if wrong[top] else "-"
        L.append(f"| {r['orig_name']} | {r['merged_into']} | {r['test_px']:,} | "
                 + " | ".join(f"{r[f'{v}_correct_%']:.1f}" for v in VARIANTS) + f" | {err} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = base_parser(__doc__)
    ap.add_argument("--orig-root", default="C:/PROJECT/marine_pollution_detection",
                    help="Folder holding the original MARIDA/ and MADOS/ datasets")
    args = ap.parse_args()
    args.out_root = args.out_root or OUT_ROOT
    for d in args.datasets:
        run(d, args)
