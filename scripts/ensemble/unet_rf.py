"""
Ensemble of the U-Net (SSL4EO, scripts/unet_ssl4eo) and the stage-5 or stage-8
Random Forest (log weights, scripts/staged_ml): per-pixel class probabilities are averaged,

    p = w * p_unet + (1 - w) * p_rf,   prediction = argmax p

U-Net probabilities: softmax averaged over the same 8 TTA views (4 rotations x
flip). RF: retrained exactly as stage 5 (or stage 8 = stage 5 + 6 GLCM texture
features, --rf-stage 8) with log weights (same cached pixels, seed 42; it
reproduces the staged test scores).

The weight w and, for MARIDA, which U-Net run to use are chosen on the
VALIDATION split (highest macro F1); the test split is only used for the final
report. The full test curve over w is written for reference, not for selection.

Usage (repo root): python scripts/ensemble/unet_rf.py --datasets MARIDA MADOS [--rf-stage 8]
Output: outputs/ensemble_unet_rf/<dataset>/ (stage 5) or outputs/ensemble_unet_rf_stage8/<dataset>/
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in ("staged_ml", "unet_ssl4eo", "deeplabv3_resnet50"):
    sys.path.insert(0, os.path.join(HERE, "..", p))
import common  # noqa: E402  (staged_ml)
from glcm_features import split_features  # noqa: E402  (staged_ml)
import unet  # noqa: E402
from deeplab import CLASS_NAMES, classwise_section, md_table, metrics, seed_all  # noqa: E402

OUT_ROOT = os.path.join(REPO, "outputs", "ensemble_unet_rf")
UNET_RUNS = {"MARIDA": ["MARIDA_novscp", "MARIDA_vscp"], "MADOS": ["MADOS_vscp"]}
STAGE5 = list(common.BAND_NAMES) + ["NDWI"] + common.RATIOS
WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)


def tta_probs(model, image):
    """Softmax averaged over the 8 TTA views, each mapped back to the original orientation."""
    views = []
    for k in range(4):
        im = torch.rot90(image, k, dims=[-2, -1])
        views += [im, torch.flip(im, dims=[-1])]
    p = torch.softmax(model(torch.cat(views)), 1)
    back = []
    for k in range(4):
        back.append(torch.rot90(p[2 * k], -k, dims=[-2, -1]))
        back.append(torch.rot90(torch.flip(p[2 * k + 1], dims=[-1]), -k, dims=[-2, -1]))
    return torch.stack(back).mean(0)                                 # (4, H, W)


def unet_probs(dataset, run, split, out):
    path = os.path.join(out, f"unet_{run}_{split}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["p"], z["y"]
    run_dir = os.path.join(REPO, "outputs", "unet_ssl4eo", run)
    st = json.load(open(os.path.join(run_dir, "extra_channel_stats.json")))
    device = torch.device("cuda")
    model = unet.build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model_best_val_f1.pth"), map_location=device))
    model.eval()
    ds = unet.Seg16(dataset, split, False, st)
    P, Y = [], []
    seed_all()
    with torch.no_grad():
        for i in range(len(ds)):
            img, t = ds[i]
            t = t.numpy().ravel()
            # same pixel order as the staged cache: classes 1-4 in turn, raster order within each
            sel = np.concatenate([np.flatnonzero(t == c) for c in range(4)])
            if len(sel) == 0:
                continue
            p = tta_probs(model, img[None].to(device)).reshape(4, -1).cpu().numpy()
            P.append(p[:, sel].T)
            Y.append(t[sel])
    P, Y = np.concatenate(P).astype(np.float32), np.concatenate(Y)
    np.savez_compressed(path, p=P, y=Y)
    return P, Y


def rf_probs(dataset, out, stage, seed=42):
    path = os.path.join(out, f"rf_stage{stage}_logw.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {s: (z[f"p_{s}"], z[f"y_{s}"]) for s in ("val", "test")}
    cols = common.columns(STAGE5)
    lab = {c: i for i, c in enumerate(common.CLASSES)}
    def feats(X, split):  # stage 8 appends the GLCM columns (same pixel order as the cache)
        X = X[:, cols]
        return np.hstack([X, split_features(dataset, split).astype(X.dtype)]) if stage == 8 else X

    Xtr, ytr = common.sample_split(dataset, "train", None, seed)
    ytr = np.vectorize(lab.get)(ytr)
    w = common.class_weights(ytr, "log")
    rf = common.make_model("rf", seed)
    rf.fit(feats(Xtr, "train"), ytr, sample_weight=w[ytr])
    del Xtr
    res = {}
    for s in ("val", "test"):
        X, y = common.sample_split(dataset, s, None, seed)
        res[s] = (rf.predict_proba(feats(X, s)).astype(np.float32), np.vectorize(lab.get)(y))
    np.savez_compressed(path, **{f"{k}_{s}": v for s, (p, y) in res.items() for k, v in (("p", p), ("y", y))})
    return res


def run(dataset, stage):
    out = os.path.join(OUT_ROOT + ("_stage8" if stage == 8 else ""), dataset)
    os.makedirs(out, exist_ok=True)
    rf = rf_probs(dataset, out, stage)
    curves, unets = [], {}
    for r in UNET_RUNS[dataset]:
        for s in ("val", "test"):
            p, y = unet_probs(dataset, r, s, out)
            if not np.array_equal(y, rf[s][1]):
                raise ValueError(f"{dataset} {r} {s}: U-Net and RF pixels are not aligned")
            unets[(r, s)] = p
        for w in WEIGHTS:
            row = {"unet_run": r, "w_unet": w}
            for s in ("val", "test"):
                pred = (w * unets[(r, s)] + (1 - w) * rf[s][0]).argmax(1)
                acc, _ = metrics(rf[s][1], pred)
                row[f"{s}_macroF1"], row[f"{s}_F1_debris"] = acc["macroF1"], acc["F1_Marine Debris"]
            curves.append(row)
            print(f"{dataset} {r} w={w:.2f} val F1 {row['val_macroF1']:.4f} test F1 {row['test_macroF1']:.4f} "
                  f"(debris {row['test_F1_debris']:.4f})", flush=True)
    cur = pd.DataFrame(curves)
    cur.to_csv(os.path.join(out, "weight_curve.csv"), index=False)
    best = cur.loc[cur.val_macroF1.idxmax()]
    r, w = best.unet_run, float(best.w_unet)

    yt = rf["test"][1]
    variants = {"Random Forest": rf["test"][0].argmax(1), f"U-Net ({r}, soft TTA)": unets[(r, "test")].argmax(1),
                f"Ensemble (w_unet = {w:.2f})": (w * unets[(r, "test")] + (1 - w) * rf["test"][0]).argmax(1)}
    lines = [f"# U-Net + Random Forest (stage {stage}, log weights) ensemble, {dataset}", "",
             "p = w * p_unet + (1 - w) * p_rf. The U-Net run and w were chosen on the validation split "
             f"(highest macro F1): **{r}, w_unet = {w:.2f}** (val macro F1 {best.val_macroF1:.4f}). "
             "Test pixels: all annotated pixels of the official test split.", "",
             "U-Net probabilities are the softmax averaged over 8 TTA views, so the U-Net-only row can differ "
             "slightly from its own test report (majority vote over views).", "", "## Overall (test)", "",
             "| model | macro F1 | IoU | accuracy | kappa | macro precision | macro recall |", "|---|---|---|---|---|---|---|"]
    summary, sections = [], []
    for name, yp in variants.items():
        acc, cm = metrics(yt, yp)
        lines.append(f"| {name} | {acc['macroF1']:.4f} | {acc['IoU']:.4f} | {acc['microF1']:.4f} | "
                     f"{cohen_kappa_score(yt, yp):.4f} | {acc['macroPrec']:.4f} | {acc['macroRec']:.4f} |")
        tag = {"Random Forest": "rf", "Ensemble": "ensemble"}.get(name.split(" (")[0], "unet")
        _, _, cw, sec = classwise_section(yt, yp, out, tag)
        summary.append((name, cw))
        sections += [f"## {name}, test split, full class-wise analysis", ""] + sec
    for m in ("precision", "recall", "F1", "IoU"):
        lines += ["", f"## Per-class {m} (test)", ""]
        lines += md_table(pd.DataFrame({n: cw[m] for n, cw in summary}), lambda c, v: f"{v:.4f}")
    lines += ["", "## Validation macro F1 by weight (used for the choice)", ""]
    piv = cur.pivot(index="w_unet", columns="unet_run", values="val_macroF1")
    lines += md_table(piv, lambda c, v: f"{v:.4f}")
    lines += ["", "Test-split curve (reference only, not used for selection): weight_curve.csv", ""] + sections
    open(os.path.join(out, "report.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["MARIDA", "MADOS"], choices=["MARIDA", "MADOS"])
    ap.add_argument("--rf-stage", type=int, default=5, choices=[5, 8],
                    help="5 = bands + NDWI + ratios; 8 = stage 5 + 6 GLCM texture features")
    a = ap.parse_args()
    for d in a.datasets:
        run(d, a.rf_stage)
