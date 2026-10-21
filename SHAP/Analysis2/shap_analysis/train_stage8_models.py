"""
Re-fit and SAVE the five stage-8 models (scripts/staged_ml/stage8_ndwi_ratios_glcm.py,
log weighting = the paper's main setting) so they can be explained with SHAP.

run_stage() in scripts/staged_ml/common.py trains the models in memory and never
writes them to disk. This script repeats exactly the same steps with the same
functions (load_data, columns, glcm_columns, class_weights, make_model, evaluate),
seed 42, then saves each fitted model and checks that its test metrics match the
recorded outputs/staged/log_weights/stage8_ndwi_ratios_glcm/<dataset>/metrics.csv.
The classification pipeline itself is not modified.

Saved to models/stage8_log/<dataset>/:
  rf.joblib, xgb.joblib, lgbm.joblib, ada.joblib, svm.joblib  (sklearn-API objects)
  meta.json        feature names/order, class encoding, weights, metrics, recorded metrics
  test_pred.npz    test-set predictions (index order = staged test cache order)

Usage (repo root):
  .venv/Scripts/python scripts/shap_analysis/train_stage8_models.py --data-root ../5_class
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "staged_ml"))
sys.path.insert(0, HERE)

from SHAP.Analysis2.shap_analysis.stage8_setup import configure, stage8_data  # noqa: E402
import common  # noqa: E402

STAGE = "stage8_ndwi_ratios_glcm"
RECORDED = os.path.join(REPO, "outputs", "staged", "log_weights", STAGE)
MODEL_ROOT = os.path.join(REPO, "models", "stage8_log")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(REPO, "data"),
                    help="Folder that contains MARIDA_5Class/ and MADOS_5Class/")
    ap.add_argument("--datasets", nargs="+", default=["MARIDA", "MADOS"])
    ap.add_argument("--models", nargs="+", default=common.MODELS)
    ap.add_argument("--weighting", default="log")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    configure(a.data_root)

    for ds in a.datasets:
        Xtr, ytr, Xte, yte, names = stage8_data(ds, a.seed)
        out = os.path.join(MODEL_ROOT, ds)
        os.makedirs(out, exist_ok=True)
        rec = pd.read_csv(os.path.join(RECORDED, ds, "metrics.csv")).set_index("model")
        print(f"\n=== {ds}: train {len(ytr):,} {np.bincount(ytr, minlength=4)} | "
              f"test {len(yte):,} {np.bincount(yte, minlength=4)} | {len(names)} features", flush=True)
        meta_path = os.path.join(out, "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"models": {}}
        meta.update({
            "stage": STAGE, "dataset": ds, "weighting": a.weighting, "seed": a.seed,
            "features": names,
            "class_encoding": {str(i): {"label": c, "name": common.CLASS_NAMES[c]}
                               for i, c in enumerate(common.CLASSES)},
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "train_counts": np.bincount(ytr, minlength=4).tolist(),
            "test_counts": np.bincount(yte, minlength=4).tolist(),
            "versions": versions(),
        })
        preds = dict(np.load(os.path.join(out, "test_pred.npz"))) if os.path.exists(
            os.path.join(out, "test_pred.npz")) else {}
        for name in a.models:
            # identical to common.run_stage (svm_cap / train_cap are None in the recorded run)
            cw = common.class_weights(ytr, a.weighting)
            fit_kw = {}
            if name == "svm":
                model = common.make_model(name, a.seed, dict(enumerate(cw)))
            else:
                model = common.make_model(name, a.seed)
                fit_kw["sample_weight"] = cw[ytr]
            t0 = time.time(); model.fit(Xtr, ytr, **fit_kw); t_fit = time.time() - t0
            pred = np.asarray(model.predict(Xte)).ravel().astype(int)
            m, cm = common.evaluate(yte, pred)
            r = rec.loc[name]
            diff = {k: float(m[k] - r[k]) for k in ["f1", "accuracy", "kappa", "f1_c1", "f1_c2", "f1_c3", "f1_c4"]}
            worst = max(abs(v) for v in diff.values())
            print(f"  {name:5s} fit {t_fit:6.0f}s  F1 {m['f1']:.4f} (recorded {r['f1']:.4f})  "
                  f"debris F1 {m['f1_c2']:.4f} (recorded {r['f1_c2']:.4f})  max|diff| {worst:.2e}", flush=True)
            joblib.dump(model, os.path.join(out, f"{name}.joblib"))
            preds[name] = pred
            meta["models"][name] = {
                "class": type(model).__name__ if name != "svm" else "Pipeline(StandardScaler, LinearSVC)",
                "fit_seconds": round(t_fit, 1), "metrics": {k: float(v) for k, v in m.items()},
                "recorded_metrics": {k: float(r[k]) for k in diff}, "diff_vs_recorded": diff,
                "max_abs_diff_vs_recorded": worst, "confusion_matrix": cm.tolist(),
            }
        preds["y_true"] = yte
        np.savez_compressed(os.path.join(out, "test_pred.npz"), **preds)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def versions():
    import lightgbm, numba, shap, sklearn, xgboost
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "shap": shap.__version__,
            "numba": numba.__version__}


if __name__ == "__main__":
    main()
