"""
Validation-based selection of feature set and class weighting (see README.md
for the pre-registered rules).

  prepare  build the val pixel cache (and MARIDA val patch ids for env features)
  worker   train/evaluate pending jobs (safe to run several workers at once)
  decide   rule 2: which feature groups help on val -> stage 5v / 6v
  report   rules 3-5 -> outputs/staged_valselect/report.md, all_jobs.csv

Models are trained on the train split only and scored on both val and test;
selection uses val scores only.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "staged"))
import common as C  # noqa: E402
from common import (BAND_NAMES, CLASSES, INDICES, RATIOS, SPEC_STATS,  # noqa: E402
                    class_weights, columns, evaluate, make_model)
from stage6_env import ENV_CSV, ENV_FEATURES  # noqa: E402

SEED = 42
OUT = os.path.join(REPO, "outputs", "staged_valselect")
VCACHE = os.path.join(OUT, "cache")
JOBS = os.path.join(OUT, "jobs")
OLD_CACHE = os.path.join(REPO, "outputs", "staged", "cache")
OLD_ROOTS = {"balanced": os.path.join(REPO, "outputs", "staged"),
             "log": os.path.join(REPO, "outputs", "staged", "log_weights"),
             "none": os.path.join(REPO, "outputs", "staged", "no_class_weights")}
DATASETS = ["MARIDA", "MADOS"]
WEIGHTINGS = ["balanced", "log", "none"]
MODELS = C.MODELS
STALE_LOCK_S = 6 * 3600

B = list(BAND_NAMES)
GROUPS = {"NDVI": ["NDVI"], "NDWI": ["NDWI"], "FAI": ["FAI"], "FDI": ["FDI"], "NDMI": ["NDMI"],
          "ratios": RATIOS, "specstats": SPEC_STATS}
GROUP_STAGE = {"NDVI": "stage2a_ndvi", "NDWI": "stage2b_ndwi", "FAI": "stage2c_fai", "FDI": "stage2d_fdi",
               "NDMI": "stage2e_ndmi", "ratios": "stage2f_ratios", "specstats": "stage2g_specstats"}
BASE_SETS = {  # stage -> (pixel features, add env features)
    "stage1_bands": (B, False),
    **{GROUP_STAGE[g]: (B + f, False) for g, f in GROUPS.items()},
    "stage3_all_indices": (B + INDICES, False),
    "stage4_indices_ratios": (B + INDICES + RATIOS, False),
    "stage5_ndwi_ratios": (B + ["NDWI"] + RATIOS, False),
    "stage6_env": (B + ["NDWI"] + RATIOS, True),
}
DECISION = os.path.join(OUT, "decision.json")


def feature_sets():
    sets = dict(BASE_SETS)
    if os.path.exists(DECISION):
        d = json.load(open(DECISION))
        if not d["same_as_stage5"]:
            sets["stage5v_valselected"] = (d["features"], False)
            sets["stage6v_valselected_env"] = (d["features"], True)
    return sets


def all_jobs():
    jobs = []
    for stage, (_, env) in feature_sets().items():
        for ds in DATASETS:
            if env and ds != "MARIDA":
                continue
            for w in WEIGHTINGS:
                for m in MODELS:
                    jobs.append((ds, stage, w, m))
    # slowest last: AdaBoost on MADOS
    return sorted(jobs, key=lambda j: (j[0] == "MADOS" and j[3] == "ada", j[0], j[1], j[2], j[3]))


def job_id(j):
    return "__".join(j).replace("/", "-")


# ------------------------------------------------------------------ data ---

def val_patch_ids(n_pixels):
    """MARIDA val patch id per cached pixel (same order logic as stage6_env.pixel_patch_ids)."""
    path = os.path.join(VCACHE, "MARIDA_val_all_patchids.npy")
    if os.path.exists(path):
        ids = np.load(path)
    else:
        cls, root = C.DATASETS["MARIDA"]
        ds = cls(root, split="val")
        ids, labels = [], []
        for i in range(len(ds)):
            m = ds[i][1].numpy().ravel()
            for c in CLASSES:
                n = int(np.sum(m == c))
                ids += [ds.patch_ids[i]] * n
                labels += [c] * n
        ids = np.array(ids)
        y = np.load(os.path.join(VCACHE, "MARIDA_val_all_seed42_f24.npz"))["y"]
        if not np.array_equal(np.array(labels), y):
            raise RuntimeError("val: rebuilt pixel order does not match the cached labels")
        np.save(path, ids)
    if len(ids) != n_pixels:
        raise RuntimeError("val patch ids do not match the val pixel cache")
    return ids


_DATA = {}


def data(ds):
    if ds in _DATA:
        return _DATA[ds]
    d = {}
    for split, folder in (("train", OLD_CACHE), ("val", VCACHE), ("test", OLD_CACHE)):
        z = np.load(os.path.join(folder, f"{ds}_{split}_all_seed42_f24.npz"))
        X, y = z["X"], z["y"]
        if not np.isin(y, CLASSES).all():
            raise RuntimeError(f"{ds}/{split}: unexpected labels")
        d[split] = (X, (y - CLASSES[0]).astype(np.int64))
    if ds == "MARIDA":
        ids = {"train": np.load(os.path.join(OLD_CACHE, "MARIDA_train_all_patchids.npy")),
               "test": np.load(os.path.join(OLD_CACHE, "MARIDA_test_all_patchids.npy")),
               "val": val_patch_ids(len(d["val"][1]))}
        for s in ids:
            if len(ids[s]) != len(d[s][1]):
                raise RuntimeError(f"MARIDA {s}: patch ids out of date")
        env = pd.read_csv(ENV_CSV).set_index("patch_id")[ENV_FEATURES]
        env = env.fillna(env.loc[np.unique(ids["train"])].median())   # train patches only
        d["env"] = {s: env.loc[ids[s]].to_numpy(dtype=np.float32) for s in ids}
    _DATA[ds] = d
    return d


def matrices(ds, feats, use_env):
    d, cols = data(ds), columns(feats)
    out = {}
    for s in ("train", "val", "test"):
        X = d[s][0][:, cols]
        if use_env:
            X = np.hstack([X, d["env"][s]])
        out[s] = (X, d[s][1])
    return out


# ------------------------------------------------------------- commands ---

def cmd_prepare(_):
    os.makedirs(VCACHE, exist_ok=True)
    C.CACHE_DIR = VCACHE           # sample_split writes the val cache here, not in outputs/staged/cache
    for ds in DATASETS:
        X, y = C.sample_split(ds, "val", None, SEED)
        print(f"{ds} val: {len(y):,} px, per class {np.bincount(y, minlength=5)[1:]}", flush=True)
    ids = val_patch_ids(len(np.load(os.path.join(VCACHE, "MARIDA_val_all_seed42_f24.npz"))["y"]))
    print(f"MARIDA val patch ids verified: {len(ids):,} px, {len(set(ids))} patches", flush=True)


def claim(path):
    lock = path + ".lock"
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        if time.time() - os.path.getmtime(lock) > STALE_LOCK_S:
            os.remove(lock)
            return claim(path)
        return False


def run_job(j):
    ds, stage, w, model = j
    feats, use_env = feature_sets()[stage]
    M = matrices(ds, feats, use_env)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = M["train"], M["val"], M["test"]
    cw = class_weights(ytr, w)
    fit_kw = {}
    if cw is None:
        clf = make_model(model, SEED)
    elif model == "svm":
        clf = make_model(model, SEED, dict(enumerate(cw)))
    else:
        clf = make_model(model, SEED)
        fit_kw["sample_weight"] = cw[ytr]
    t0 = time.time(); clf.fit(Xtr, ytr, **fit_kw); t_fit = time.time() - t0
    res = {"dataset": ds, "stage": stage, "weighting": w, "model": model,
           "features": feats + (ENV_FEATURES if use_env else []), "n_train": int(len(ytr)),
           "n_val": int(len(yva)), "n_test": int(len(yte)), "train_time_s": round(t_fit, 2)}
    for split, X, y in (("val", Xva, yva), ("test", Xte, yte)):
        pred = np.asarray(clf.predict(X)).ravel().astype(int)
        m, cm = evaluate(y, pred)
        res[split] = {k: float(v) for k, v in m.items()}
        res[f"{split}_confusion"] = cm.tolist()
    return res


def cmd_worker(args):
    os.makedirs(JOBS, exist_ok=True)
    jobs = all_jobs()
    if args.heavy_only:
        jobs = [j for j in jobs if j[0] == "MADOS" and j[3] == "ada"]
    done = 0
    for j in jobs:
        path = os.path.join(JOBS, job_id(j) + ".json")
        if os.path.exists(path) or not claim(path):
            continue
        try:
            res = run_job(j)
            with open(path + ".tmp", "w") as f:
                json.dump(res, f)
            os.replace(path + ".tmp", path)
            done += 1
            print(f"{job_id(j)}: val F1={res['val']['f1']:.4f} test F1={res['test']['f1']:.4f} "
                  f"({res['train_time_s']:.0f}s)", flush=True)
        except Exception as e:
            print(f"{job_id(j)}: FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            if os.path.exists(path + ".lock"):
                os.remove(path + ".lock")
    print(f"worker finished; {done} jobs run", flush=True)


def load_results():
    rows = []
    for f in os.listdir(JOBS):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(JOBS, f)))
            row = {k: r[k] for k in ("dataset", "stage", "weighting", "model", "train_time_s")}
            row["n_features"] = len(r["features"])
            for s in ("val", "test"):
                row.update({f"{s}_{k}": v for k, v in r[s].items()})
            rows.append(row)
    return pd.DataFrame(rows)


def cmd_decide(_):
    df = load_results()
    need = [(ds, s, w, m) for ds in DATASETS for s in ["stage1_bands"] + list(GROUP_STAGE.values())
            for w in WEIGHTINGS for m in MODELS]
    have = set(map(tuple, df[["dataset", "stage", "weighting", "model"]].values))
    missing = [j for j in need if j not in have]
    if missing:
        raise SystemExit(f"{len(missing)} stage 1/2 jobs not finished yet, e.g. {missing[:3]}")
    base = df[df.stage == "stage1_bands"].set_index(["dataset", "weighting", "model"])
    rows = []
    for g, st in GROUP_STAGE.items():
        sub = df[df.stage == st].set_index(["dataset", "weighting", "model"])
        for ds in DATASETS:
            dv = (sub.loc[ds, "val_f1"] - base.loc[ds, "val_f1"]).mean()
            dt = (sub.loc[ds, "test_f1"] - base.loc[ds, "test_f1"]).mean()
            dd = (sub.loc[ds, "val_f1_c2"] - base.loc[ds, "val_f1_c2"]).mean()
            rows.append({"group": g, "dataset": ds, "val_dF1": dv, "val_d_debrisF1": dd, "test_dF1_reference": dt})
    t = pd.DataFrame(rows)
    piv = t.pivot(index="group", columns="dataset", values="val_dF1")
    kept = [g for g in GROUPS if (piv.loc[g] > 0).all()]
    feats = B + [f for g in kept for f in GROUPS[g]]
    stage5 = B + ["NDWI"] + RATIOS
    dec = {"kept_groups": kept, "features": feats, "same_as_stage5": sorted(feats) == sorted(stage5)}
    json.dump(dec, open(DECISION, "w"), indent=2)
    t.to_csv(os.path.join(OUT, "decision_group_deltas.csv"), index=False)
    print(t.pivot(index="group", columns="dataset", values=["val_dF1", "test_dF1_reference"]).round(4).to_string())
    print("\nDecision:", json.dumps(dec, indent=2))


def fmt(x):
    return f"{x:.3f}"


def cmd_report(_):
    df = load_results()
    expected = len(all_jobs())
    lines = ["# Validation-based selection: report", "",
             f"Jobs finished: {len(df)} / {expected}. Models trained on the train split only; "
             "selection uses validation macro F1 (ties: val debris F1). Test is reported once, for the chosen configurations.", ""]
    df.to_csv(os.path.join(OUT, "all_jobs.csv"), index=False)

    # Rule 5: reproducibility against the saved staged results
    diffs = []
    for _, r in df.iterrows():
        p = os.path.join(OLD_ROOTS[r.weighting], r.stage, r.dataset, "metrics.csv")
        if not os.path.exists(p):
            continue
        old = pd.read_csv(p).set_index("model")
        if r.model in old.index:
            for k in ("f1", "iou", "f1_c2", "iou_c2", "f1_c3"):
                diffs.append({"key": k, "diff": abs(r[f"test_{k}"] - old.loc[r.model, k]),
                              "job": f"{r.dataset}/{r.stage}/{r.weighting}/{r.model}"})
    dd = pd.DataFrame(diffs)
    lines += ["## Reproducibility (rule 5)", ""]
    if len(dd):
        worst = dd.sort_values("diff").iloc[-1]
        lines += [f"Retrained test scores compared with the saved staged results: {dd.job.nunique()} runs, "
                  f"{len(dd)} values; max |difference| = {dd['diff'].max():.2e} ({worst.job}, {worst.key}); "
                  f"runs differing by more than 1e-6: {dd[dd['diff'] > 1e-6].job.nunique()}.", ""]

    # Rule 2 decision
    if os.path.exists(DECISION):
        dec = json.load(open(DECISION))
        t = pd.read_csv(os.path.join(OUT, "decision_group_deltas.csv"))
        lines += ["## Feature groups on validation (rule 2)", "",
                  "Mean Δ macro F1 vs bands only (5 models × 3 weightings). Kept if > 0 on both datasets.", "",
                  "| Group | MARIDA val Δ | MADOS val Δ | MARIDA test Δ (ref.) | MADOS test Δ (ref.) | Kept |",
                  "|---|---|---|---|---|---|"]
        for g in GROUPS:
            r = t[t.group == g].set_index("dataset")
            lines.append(f"| {g} | {r.loc['MARIDA','val_dF1']:+.4f} | {r.loc['MADOS','val_dF1']:+.4f} | "
                         f"{r.loc['MARIDA','test_dF1_reference']:+.4f} | {r.loc['MADOS','test_dF1_reference']:+.4f} | "
                         f"{'yes' if g in dec['kept_groups'] else 'no'} |")
        lines += ["", f"Val-selected feature set: {', '.join(dec['features'])} "
                  f"({'identical to stage 5' if dec['same_as_stage5'] else 'differs from stage 5 -> stage5v / stage6v'}).", ""]

    # Rule 3: selection
    key = df.assign(_k=list(zip(df.val_f1, df.val_f1_c2)))
    lines += ["## Selected configuration per dataset and model (rule 3)", "",
              "| Dataset | Model | Stage | Weighting | val F1 | test F1 | test IoU | test debris F1 / IoU | test algae F1 |",
              "|---|---|---|---|---|---|---|---|---|"]
    best_rows = []
    for (ds, m), g in key.groupby(["dataset", "model"]):
        r = g.loc[g._k.idxmax()]
        best_rows.append(r)
        lines.append(f"| {ds} | {m} | {r.stage} | {r.weighting} | {fmt(r.val_f1)} | {fmt(r.test_f1)} | {fmt(r.test_iou)} | "
                     f"{fmt(r.test_f1_c2)} / {fmt(r.test_iou_c2)} | {fmt(r.test_f1_c3)} |")
    lines += ["", "## Overall choice per dataset vs the earlier test-based choice (rule 4)", "",
              "| Dataset | Chosen by | Model / stage / weighting | val F1 | test F1 | test IoU | test debris F1 |",
              "|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        g = key[key.dataset == ds]
        if g.empty:
            continue
        v = g.loc[g._k.idxmax()]
        t = g.loc[g.test_f1.idxmax()]
        earlier = g[(g.model == "rf") & (g.stage == "stage5_ndwi_ratios") & (g.weighting == "log")]
        choices = [("validation (this run)", v), ("test (best possible, optimistic)", t)]
        if len(earlier):
            choices.append(("earlier headline: RF, stage 5, log", earlier.iloc[0]))
        for label, r in choices:
            lines.append(f"| {ds} | {label} | {r.model} / {r.stage} / {r.weighting} | {fmt(r.val_f1)} | {fmt(r.test_f1)} | "
                         f"{fmt(r.test_iou)} | {fmt(r.test_f1_c2)} |")
        lines.append(f"| {ds} | optimism of test-based choice | | | {t.test_f1 - v.test_f1:+.3f} | | |")
    lines += ["", "## Weighting on validation", "", "Mean val macro F1 over all feature sets.", "",
              "| Dataset | Model | balanced | log | none | best on val |", "|---|---|---|---|---|---|"]
    for (ds, m), g in df.groupby(["dataset", "model"]):
        s = g.groupby("weighting").val_f1.mean()
        lines.append(f"| {ds} | {m} | " + " | ".join(fmt(s.get(w, np.nan)) for w in WEIGHTINGS) + f" | {s.idxmax()} |")
    env = df[(df.dataset == "MARIDA") & df.stage.isin(["stage5_ndwi_ratios", "stage6_env"])]
    if env.stage.nunique() == 2:
        p = env.pivot_table(index=["model", "weighting"], columns="stage", values=["val_f1", "test_f1"])
        dv = p["val_f1"]["stage6_env"] - p["val_f1"]["stage5_ndwi_ratios"]
        dt = p["test_f1"]["stage6_env"] - p["test_f1"]["stage5_ndwi_ratios"]
        lines += ["", "## Environmental features (MARIDA): stage 6 − stage 5", "",
                  f"Validation: mean Δ macro F1 {dv.mean():+.4f}, improved in {int((dv > 0).sum())}/{len(dv)} model×weighting runs. "
                  f"Test: mean Δ {dt.mean():+.4f}, improved in {int((dt > 0).sum())}/{len(dt)}.", ""]
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    w = sub.add_parser("worker")
    w.add_argument("--heavy-only", action="store_true", help="Only AdaBoost on MADOS (the slowest jobs)")
    sub.add_parser("decide")
    sub.add_parser("report")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "worker": cmd_worker, "decide": cmd_decide, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
