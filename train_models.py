"""
Train the Random Forests served by the website, exactly as in the staged study
(scripts/staged_ml/common.py): RandomForestClassifier(n_estimators=100,
random_state=42), every annotated train pixel, log class weights
w_c = 1 / ln(1.03 + p_c) passed as sample weights.

    python train_models.py                 # both datasets, stage 5 and stage 8
    python train_models.py --datasets MARIDA

Writes models/rf_<key>.joblib, models/report.json (test scores, to compare with
outputs/dl_comparison/comparison.md) and models/samples.json (gallery).
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from marine import config
from marine.features import stage5_features
from marine.glcm import glcm_at
from marine.inference import scope_mask
from marine.readers import load_sample

CLASSES = [1, 2, 3, 4]
NAMES = {1: "Water", 2: "Marine Debris", 3: "Algae/Sargassum", 4: "Other"}


def split_ids(dataset, split):
    if dataset == "MARIDA":
        path = os.path.join(config.MARIDA_SPLITS, f"{split}_X.txt")
    else:
        path = os.path.join(config.MADOS_ROOT, "splits", f"{split}_X.txt")
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def collect(dataset, split):
    """Stage-8 features (stage 5 = first 16 columns) of every annotated pixel."""
    X, y, info, flagged = [], [], [], np.zeros(5, dtype=np.int64)
    ids = split_ids(dataset, split)
    t0 = time.time()
    for i, pid in enumerate(ids):
        try:
            sc = load_sample({"dataset": dataset, "id": pid}, config.MARIDA_ROOT, config.MADOS_ROOT)
        except Exception as e:                       # noqa: BLE001  (report and skip broken patches)
            print(f"  skip {pid}: {e}")
            continue
        gt = sc.gt.ravel()
        sel = np.concatenate([np.flatnonzero(gt == c) for c in CLASSES])
        counts = np.bincount(gt, minlength=5)
        info.append({"dataset": dataset, "id": pid, "counts": counts[:5].tolist()})
        if len(sel) == 0:
            continue
        rows, cols = np.divmod(sel, sc.img.shape[2])
        f5 = stage5_features(sc.img)[sel]
        g = glcm_at(sc.img, rows, cols)
        X.append(np.hstack([f5, g]))
        y.append(gt[sel])
        flagged += np.bincount(gt[sel][scope_mask(sc.img).ravel()[sel]], minlength=5)[:5]
        if (i + 1) % 200 == 0:
            print(f"  [{dataset}/{split}] {i + 1}/{len(ids)} patches, {time.time() - t0:.0f}s", flush=True)
    X, y = np.vstack(X).astype(np.float32), np.concatenate(y).astype(np.int64)
    print(f"  [{dataset}/{split}] {len(y):,} annotated pixels, per class {np.bincount(y, minlength=5)[1:]}, "
          f"{time.time() - t0:.0f}s", flush=True)
    return X, y, info, flagged


def log_weights(y):
    counts = np.clip(np.bincount(y, minlength=5)[1:].astype(np.float64), 1, None)
    return 1.0 / np.log(1.03 + counts / counts.sum())


def scores(y, p):
    out, f1s, ious = {}, [], []
    for c in CLASSES:
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1); iou = tp / max(tp + fp + fn, 1)
        out[NAMES[c]] = {"precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)),
                         "f1": float(f1), "iou": float(iou)}
        f1s.append(f1); ious.append(iou)
    return {"macro_f1": float(np.mean(f1s)), "miou": float(np.mean(ious)),
            "accuracy": float(np.mean(y == p)), "per_class": out}


def pick_samples(info, dataset, n_debris, n_algae, n_other):
    c = lambda r, k: r["counts"][k]
    chosen, seen = [], set()

    def take(rows, n, why):
        for r in rows:
            if len([s for s in chosen if s["why"] == why]) >= n:
                break
            if r["id"] not in seen:
                seen.add(r["id"])
                chosen.append({"dataset": dataset, "id": r["id"], "why": why,
                               "counts": {NAMES[k]: c(r, k) for k in CLASSES}})

    take(sorted(info, key=lambda r: -c(r, 2)), n_debris, "Marine debris")
    take(sorted(info, key=lambda r: -c(r, 3)), n_algae, "Algae / Sargassum")
    take(sorted(info, key=lambda r: -c(r, 4)), n_other, "Other floating features")
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["MARIDA", "MADOS"])
    args = ap.parse_args()
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    rpath = os.path.join(config.MODEL_DIR, "report.json")
    report = json.load(open(rpath)) if os.path.exists(rpath) else {}
    samples = json.load(open(config.SAMPLES_FILE)) if os.path.exists(config.SAMPLES_FILE) else []

    for ds in args.datasets:
        print(f"=== {ds}", flush=True)
        Xtr, ytr, _, _ = collect(ds, "train")
        Xte, yte, info, flagged = collect(ds, "test")
        w = log_weights(ytr)[ytr - 1]
        for stage, ncol in (("stage5", 16), ("stage8", 22)):
            key = f"{ds.lower()}_{stage[0]}{stage[-1]}"
            t0 = time.time()
            rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            rf.fit(Xtr[:, :ncol], ytr, sample_weight=w)
            pred = rf.predict(Xte[:, :ncol])
            s = scores(yte, pred)
            path = os.path.join(config.MODEL_DIR, config.MODELS[key]["file"])
            joblib.dump(rf, path, compress=3)
            s.update({"train_pixels": int(len(ytr)), "test_pixels": int(len(yte)),
                      "train_seconds": round(time.time() - t0, 1),
                      "size_mb": round(os.path.getsize(path) / 1e6, 1)})
            report[key] = s
            print(f"  {key}: macro F1 {s['macro_f1']:.4f}, debris F1 {s['per_class']['Marine Debris']['f1']:.4f}, "
                  f"mIoU {s['miou']:.4f}  ({s['size_mb']} MB)", flush=True)
        tot = np.bincount(yte, minlength=5)
        report[f"{ds.lower()}_scope_guard"] = {NAMES[c]: {"flagged": int(flagged[c]), "of": int(tot[c])}
                                               for c in CLASSES}
        print(f"  scope guard flags on annotated test pixels: {flagged[1:]} of {tot[1:]}", flush=True)
        samples = [s for s in samples if s["dataset"] != ds]
        samples += pick_samples(info, ds, 8, 3, 3) if ds == "MARIDA" else pick_samples(info, ds, 6, 3, 3)
        json.dump(report, open(rpath, "w"), indent=2)
        json.dump(samples, open(config.SAMPLES_FILE, "w"), indent=2)


if __name__ == "__main__":
    main()
