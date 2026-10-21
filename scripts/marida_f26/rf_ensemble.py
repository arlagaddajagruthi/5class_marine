"""
MARIDA 26-feature study: R6 Random Forest on the same 26 features, and R7 the
U-Net + RF probability ensemble.

R6  Random Forest exactly as the staged ML (scripts/staged_ml/common.py): 100 trees,
    seed 42, log class weights 1/ln(1.03 + p_c) as sample weights, trained on every
    annotated train pixel (358,442) with the 26 features (staged caches, identical
    to features26.py). It predicts val/test pixels in the same order as the deep
    models, plus full probability maps of the test patches.
R7  p = w * p_U-Net + (1 - w) * p_RF, w in {0, 0.05, ..., 1} chosen by validation
    macro F1 (the test split is used only for the report). U-Net = run R1 by default.

Usage (repo root), after R1 has been tested:
  python scripts/marida_f26/rf_ensemble.py rf
  python scripts/marida_f26/rf_ensemble.py ensemble [--unet R1]
Outputs: outputs/marida_f26/runs/R6_rf/, outputs/marida_f26/runs/R7_ensemble_unet_rf/ (same files as the deep runs)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(REPO, "scripts", "staged_ml")]
import features26 as F  # noqa: E402
import report  # noqa: E402
from runs import name  # noqa: E402

RUNS = os.path.join(REPO, "outputs", "marida_f26", "runs")
NAMES = F.CLASS_NAMES


def base_config(run, model, title, notes):
    return {"run": run, "dataset": "MARIDA", "model": model, "init": "-", "bands": "unscaled", "classes": 4,
            "class_names": NAMES, "channels": F.CHANNELS, "groups": F.GROUPS, "title": title, "notes": notes,
            "recipe": {"epochs": "-"}}


def pixels_in_dl_order(split):
    """Unscaled 26 features and labels of the annotated pixels, patch by patch in raster order."""
    ds = F.MaridaF26(split, train=False, band_scaling="raw")
    X, y, pid = [], [], []
    for i in range(len(ds)):
        x, m = ds.unscaled(i)
        keep = m > 0
        X.append(x[:, keep].T)
        y.append(m[keep] - 1)
        pid.append(np.full(keep.sum(), i, np.int32))
    return np.vstack(X), np.concatenate(y), np.concatenate(pid), ds


def cmd_rf(a):
    import joblib
    from common import class_weights, make_model
    out = os.path.join(RUNS, name("R6"))
    os.makedirs(out, exist_ok=True)
    Xtr, ytr = F.ml_features("train")
    ytr = ytr - 1
    cw = class_weights(ytr, "log")
    rf = make_model("rf", 42)
    t0 = time.time()
    rf.fit(Xtr, ytr, sample_weight=cw[ytr])
    fit_s = time.time() - t0
    joblib.dump(rf, os.path.join(out, "rf.joblib"), compress=3)
    imp = pd.DataFrame({"feature": F.CHANNELS, "group": [F.GROUP_OF[c] for c in F.CHANNELS],
                        "impurity_importance": rf.feature_importances_}).sort_values("impurity_importance",
                                                                                     ascending=False)
    imp.to_csv(os.path.join(out, "impurity_importance.csv"), index=False)
    for split in ("val", "test"):
        X, y, pid, ds = pixels_in_dl_order(split)
        prob = rf.predict_proba(X).astype(np.float32)
        np.savez_compressed(os.path.join(out, f"pred_{split}.npz"), y=y, prob=prob, patch=pid,
                            patch_ids=np.array(ds.ids))
        if split == "test":                              # full maps, for the prediction panels and the ensemble
            maps = np.zeros((len(ds), 4, 256, 256), np.float16)
            for i in range(len(ds)):
                x, _ = ds.unscaled(i)
                maps[i] = rf.predict_proba(x.reshape(F.N_CH, -1).T).T.reshape(4, 256, 256)
            np.save(os.path.join(out, "prob_maps_test.npy"), maps)
    cfg = base_config(name("R6"), "rf", "Random Forest on the 26 features",
                      f"Random Forest as in the staged ML: 100 trees, seed 42, log class weights "
                      f"({', '.join(f'{n} {w:.3f}' for n, w in zip(NAMES, cw))}) as sample weights, all "
                      f"{len(ytr):,} annotated train pixels, fitted in {fit_s:.0f} s. Impurity importance in "
                      "`impurity_importance.csv` (SHAP in the XAI step).")
    cfg["ce_weight"] = dict(zip(NAMES, np.round(cw, 4).tolist()))
    json.dump(cfg, open(os.path.join(out, "run_config.json"), "w"), indent=2)
    report.run_report(out)


def cmd_ensemble(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(RUNS, name("R7"))
    os.makedirs(out, exist_ok=True)
    unet = name(a.unet)
    u, r = os.path.join(RUNS, unet), os.path.join(RUNS, name("R6"))
    P = {}
    for split in ("val", "test"):
        zu, zr = np.load(os.path.join(u, f"pred_{split}.npz")), np.load(os.path.join(r, f"pred_{split}.npz"))
        if not (np.array_equal(zu["y"], zr["y"]) and np.array_equal(zu["patch"], zr["patch"])):
            raise SystemExit(f"{split}: U-Net and RF pixels are not in the same order")
        P[split] = (zu["y"], zu["prob"].astype(np.float64), zr["prob"].astype(np.float64), zu["patch"],
                    zu["patch_ids"])
    rows = []
    for w in np.round(np.arange(0, 1.0001, 0.05), 2):
        row = {"w_unet": w}
        for split, (y, pu, pr, _, _) in P.items():
            ov, _, _ = report.overall_and_classwise(y, w * pu + (1 - w) * pr, NAMES)
            row[f"{split}_macro_F1"], row[f"{split}_debris_F1"] = ov["macro_F1"], ov["F1_Marine Debris"]
        rows.append(row)
    curve = pd.DataFrame(rows)
    curve.to_csv(os.path.join(out, "weight_curve.csv"), index=False)
    w = float(curve.loc[curve.val_macro_F1.idxmax(), "w_unet"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for col, st in [("val_macro_F1", "k-"), ("test_macro_F1", "k--"), ("val_debris_F1", "r-"), ("test_debris_F1", "r--")]:
        ax.plot(curve.w_unet, curve[col], st, label=col.replace("_", " "))
    ax.axvline(w, color="grey", ls=":")
    ax.set_xlabel("w (U-Net weight); 0 = RF only, 1 = U-Net only")
    ax.set_ylabel("F1")
    ax.legend(fontsize=8)
    ax.set_title(f"Ensemble weight, chosen on validation: w = {w:.2f}")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "weight_curve.png"), dpi=120)
    plt.close()
    for split, (y, pu, pr, pid, ids) in P.items():
        np.savez_compressed(os.path.join(out, f"pred_{split}.npz"), y=y, prob=(w * pu + (1 - w) * pr).astype(np.float32),
                            patch=pid, patch_ids=ids)
    mu = np.load(os.path.join(u, "prob_maps_test.npy"), mmap_mode="r")
    mr = np.load(os.path.join(r, "prob_maps_test.npy"), mmap_mode="r")
    np.save(os.path.join(out, "prob_maps_test.npy"),
            (w * mu.astype(np.float32) + (1 - w) * mr.astype(np.float32)).astype(np.float16))
    cfg = base_config(name("R7"), "ensemble", f"U-Net ({unet}) + Random Forest ({name('R6')}) ensemble",
                      f"p = w · p_U-Net + (1 − w) · p_RF with w = {w:.2f}, chosen by validation macro F1 over "
                      "w = 0, 0.05, …, 1 (`weight_curve.csv`).\n\n![weight curve](figures/weight_curve.png)")
    cfg["w_unet"], cfg["members"] = w, [unet, name("R6")]
    json.dump(cfg, open(os.path.join(out, "run_config.json"), "w"), indent=2)
    report.run_report(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["rf", "ensemble"])
    ap.add_argument("--unet", default="R1")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    {"rf": cmd_rf, "ensemble": cmd_ensemble}[a.command](a)
