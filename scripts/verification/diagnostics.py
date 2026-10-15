"""
Direct tests of two interpretations made about the staged results (MARIDA).

A. "The environmental features mostly identify the scene."
   A1 How well can the scene be predicted from the 7 env features of a patch?
   A2 Effect of adding env features (stage 6 vs stage 5) under two CV schemes on
      the train split: random pixel folds (the model has seen every scene) vs
      scene-grouped folds (held-out scenes are unseen). Memorisation predicts
      env helps (or is neutral) with random folds but hurts with grouped folds.
B. "Unweighted LightGBM collapsed because of instability."
   Training-loss curve, train vs test macro F1, and predicted class counts for
   LightGBM on stage 5 features, unweighted vs log weights.
C. "Debris scenes follow heavy rain" (Honduras): 7-day rainfall before scenes
   with vs without debris labels, from the real ERA5 rainfall already in
   data/environmental/marida_env_features.csv (used here only as evidence).

Output: outputs/verification/diagnostics.md (+ .csv tables)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts", "staged"))
import common as C  # noqa: E402
from stage6_env import ENV_CSV, ENV_FEATURES  # noqa: E402

OUT = os.path.join(REPO, "outputs", "verification")
CACHE = os.path.join(REPO, "outputs", "staged", "cache")
VCACHE = os.path.join(REPO, "outputs", "staged_valselect", "cache")
SEED = 42
S5 = list(C.BAND_NAMES) + ["NDWI"] + C.RATIOS


def load(split, folder):
    z = np.load(os.path.join(folder, f"MARIDA_{split}_all_seed42_f24.npz"))
    ids = np.load(os.path.join(folder, f"MARIDA_{split}_all_patchids.npy"))
    return z["X"], z["y"] - 1, ids


def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=[0, 1, 2, 3], zero_division=0)


def part_a(lines):
    env = pd.read_csv(ENV_CSV)
    X = env[ENV_FEATURES].fillna(env[ENV_FEATURES].median()).to_numpy()
    scenes = env.scene.to_numpy()
    keep = pd.Series(scenes).map(pd.Series(scenes).value_counts()) >= 5   # StratifiedKFold needs >= 5 per scene
    pred = cross_val_predict(RandomForestClassifier(200, random_state=SEED, n_jobs=-1), X[keep], scenes[keep],
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    acc = float(np.mean(pred == scenes[keep]))
    uniq = len(np.unique(np.round(X, 6), axis=0))
    lines += ["## A1. Can the scene be identified from the environmental features?", "",
              f"- Patches: {len(X)} in {len(np.unique(scenes))} scenes; distinct environmental vectors: {uniq}.",
              f"- Random forest predicting the scene from the 7 env features (5-fold, patches of scenes with ≥5 patches, "
              f"{keep.sum()} patches / {len(np.unique(scenes[keep]))} scenes): accuracy **{acc:.3f}** "
              f"(chance ≈ {1 / len(np.unique(scenes[keep])):.3f}).", ""]

    Xtr, ytr, ids = load("train", CACHE)
    E = env.set_index("patch_id")[ENV_FEATURES]
    E = E.fillna(E.loc[np.unique(ids)].median()).loc[ids].to_numpy(dtype=np.float32)
    groups = np.array([i.rsplit("_", 1)[0] for i in ids])
    X5 = Xtr[:, C.columns(S5)]
    X6 = np.hstack([X5, E])
    schemes = {"random pixel folds (scenes seen)": StratifiedKFold(5, shuffle=True, random_state=SEED).split(X5, ytr),
               "scene-grouped folds (scenes unseen)": GroupKFold(5).split(X5, ytr, groups)}
    rows = []
    for scheme, splits in schemes.items():
        splits = list(splits)
        for model in ("rf", "lgbm"):
            res = {}
            for name, X in (("stage5", X5), ("stage6", X6)):
                pred = np.empty_like(ytr)
                for tr, te in splits:
                    w = C.class_weights(ytr[tr], "log")
                    m = C.make_model(model, SEED)
                    m.fit(X[tr], ytr[tr], sample_weight=w[ytr[tr]])
                    pred[te] = np.asarray(m.predict(X[te])).ravel()
                res[name] = macro_f1(ytr, pred)
                res[name + "_debris"] = f1_score(ytr, pred, labels=[1], average="macro", zero_division=0)
            rows.append({"scheme": scheme, "model": model, **res,
                         "delta_env": res["stage6"] - res["stage5"],
                         "delta_env_debris": res["stage6_debris"] - res["stage5_debris"]})
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "A2_env_cv.csv"), index=False)
    lines += ["## A2. Adding env features under two cross-validation schemes (MARIDA train split, log weights)", "",
              "| CV scheme | Model | stage 5 F1 | stage 6 F1 | Δ env | Δ env (debris F1) |", "|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r.scheme} | {r.model} | {r.stage5:.4f} | {r.stage6:.4f} | {r.delta_env:+.4f} | {r.delta_env_debris:+.4f} |")
    lines.append("")


def part_b(lines):
    import lightgbm as lgb
    Xtr, ytr, _ = load("train", CACHE)
    z = np.load(os.path.join(CACHE, "MARIDA_test_all_seed42_f24.npz"))
    Xte, yte = z["X"][:, C.columns(S5)], z["y"] - 1
    Xtr = Xtr[:, C.columns(S5)]
    lines += ["## B. LightGBM, stage 5 features: unweighted vs log weights", "",
              "| Weighting | train F1 | test F1 | train logloss at iter 1 / 10 / 50 / 100 (min, at iter) | "
              "predicted test pixels per class (water, debris, algae, other) |", "|---|---|---|---|---|"]
    for w in ("none", "log"):
        cw = C.class_weights(ytr, w)
        m, ev = C.make_model("lgbm", SEED), {}
        kw = {"sample_weight": cw[ytr]} if cw is not None else {}
        m.fit(Xtr, ytr, eval_set=[(Xtr, ytr)], eval_names=["train"], eval_metric="multi_logloss",
              callbacks=[lgb.record_evaluation(ev)], **kw)
        curve = np.array(ev["train"]["multi_logloss"])
        ptr, pte = m.predict(Xtr), m.predict(Xte)
        lines.append(f"| {w} | {macro_f1(ytr, ptr):.4f} | {macro_f1(yte, pte):.4f} | "
                     f"{curve[0]:.4f} / {curve[9]:.4f} / {curve[49]:.4f} / {curve[99]:.4f} "
                     f"(min {curve.min():.4f} at {curve.argmin() + 1}) | {np.bincount(pte, minlength=4).tolist()} |")
        pd.DataFrame({"iteration": np.arange(1, len(curve) + 1), "train_logloss": curve}).to_csv(
            os.path.join(OUT, f"B_lgbm_curve_{w}.csv"), index=False)
    lines += ["", f"True test pixels per class: {np.bincount(yte, minlength=4).tolist()}", ""]


def part_c(lines):
    old = pd.read_csv(os.path.join(REPO, "data", "environmental", "marida_env_features.csv"))
    debris = {}
    for split, folder in (("train", CACHE), ("test", CACHE), ("val", VCACHE)):
        z = np.load(os.path.join(folder, f"MARIDA_{split}_all_seed42_f24.npz"))
        ids = np.load(os.path.join(folder, f"MARIDA_{split}_all_patchids.npy"))
        for s, n in pd.Series(ids[z["y"] == 2]).str.rsplit("_", n=1).str[0].value_counts().items():
            debris[s] = debris.get(s, 0) + n
    sc = old.groupby("scene").agg(tile=("tile", "first"), precip_7d=("era5_precip_7d_mm", "mean"),
                                  precip_72h=("era5_precip_72h_mm", "mean")).reset_index()
    sc["debris_px"] = sc.scene.map(debris).fillna(0).astype(int)
    sc["honduras"] = sc.tile.str.startswith("16P") | sc.tile.str.startswith("16Q")
    sc.to_csv(os.path.join(OUT, "C_scene_rain_debris.csv"), index=False)
    lines += ["## C. Rainfall before scenes with and without labelled debris (ERA5, 7 days before the image)", "",
              "| Region | Group | scenes | median 7-day rain (mm) | median 72 h rain (mm) |", "|---|---|---|---|---|"]
    for region, g in (("Honduras/Guatemala tiles (16P*, 16Q*)", sc[sc.honduras]), ("all MARIDA", sc)):
        for label, h in (("with debris", g[g.debris_px > 0]), ("without debris", g[g.debris_px == 0])):
            lines.append(f"| {region} | {label} | {len(h)} | {h.precip_7d.median():.1f} | {h.precip_72h.median():.1f} |")
    lines.append("")


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["# Verification diagnostics (MARIDA)", ""]
    part_c(lines)
    part_b(lines)
    part_a(lines)
    with open(os.path.join(OUT, "diagnostics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
