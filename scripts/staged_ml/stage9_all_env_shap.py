"""
Stage 9: every spectral feature (11 bands + NDVI, NDWI, FAI, FDI, NDMI + 4 band
ratios, i.e. stage 4) plus every environmental covariate in data/, explained
with SHAP. MARIDA only (MADOS has no location or date). Log class weights.

Environmental features (per patch, given to every pixel of the patch):
  data/env_sst_chl_wind/marida_env_sst_chl_wind.csv (scripts/env_sst_chl_wind/)
    sst_c, sst_grad_c_per_km             MUR SST, 0.01 deg
    log10_chl                            OLCI 300 m chlorophyll-a (gap-free 4 km fallback)
    wind_speed, wind_u, wind_v,          Copernicus Marine L4 wind, 0.125 deg
    wind_speed_mean_24h, wind_speed_mean_72h, wind_speed_max_72h
  data/environmental/marida_env_features.csv (scripts/environmental/)
    modis_log10_chla                     MODIS chlorophyll-a
    era5_u10, era5_v10, era5_wind_speed, era5_wind_speed_mean_24h,
    era5_wind_speed_mean_72h, era5_wind_speed_max_72h,
    era5_u10_mean_72h, era5_v10_mean_72h ERA5 10 m wind
    era5_msl_hpa                         ERA5 mean sea-level pressure
    era5_precip_24h_mm, era5_precip_72h_mm, era5_precip_7d_mm
    era5_swh, era5_swh_mean_72h          ERA5 significant wave height
    era5_stokes_u, era5_stokes_v, era5_stokes_speed   ERA5 Stokes drift
mur_sst_c and mur_sst_grad_c_per_km in the second file are identical to sst_c
and sst_grad_c_per_km and are left out. Location/geometry columns (lon, lat,
epsg, bounds, size) and quality columns (chl_source, chl_olci_valid_frac) are
not features. Unlike stage 6, wind direction (u/v) is kept, as every
environmental variable is included; with 63 scenes it can act as a scene id,
which the SHAP results make visible.

Missing values are filled with the median over the training patches; test
patches never contribute to the fill value.

SHAP (explainers from SHAP/Analysis2/shap_analysis/shap_compute.py): TreeSHAP
(interventional) for RF, XGBoost, LightGBM and AdaBoost (exact tree rewrite),
LinearExplainer for the SVM. Background: class-stratified TRAINING pixels.
Explained: class-stratified TEST pixels (same pixels for every model).

Output: <--out-root, default outputs/staged/log_weights>/stage9_all_env_shap/
  training/MARIDA/   metrics, summary, confusion matrices, model feature importance
  shap/MARIDA/       00_samples .. 07_validation, REPORT.md
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import (BAND_NAMES, CACHE_DIR, CLASS_NAMES, CLASSES, INDICES, OUT_ROOT, RATIOS, REPO,
                    base_parser, run_stage)
from stage6_env import pixel_patch_ids

sys.path.insert(0, os.path.join(REPO, "SHAP", "Analysis2", "shap_analysis"))
import shap  # noqa: E402
import shap_compute as sc  # noqa: E402

STAGE = "stage9_all_env_shap"
ENV_SOURCES = {
    os.path.join(REPO, "data", "env_sst_chl_wind", "marida_env_sst_chl_wind.csv"):
        ["sst_c", "sst_grad_c_per_km", "log10_chl", "wind_speed", "wind_u", "wind_v",
         "wind_speed_mean_24h", "wind_speed_mean_72h", "wind_speed_max_72h"],
    os.path.join(REPO, "data", "environmental", "marida_env_features.csv"):
        ["modis_log10_chla", "era5_u10", "era5_v10", "era5_wind_speed", "era5_wind_speed_mean_24h",
         "era5_wind_speed_mean_72h", "era5_wind_speed_max_72h", "era5_u10_mean_72h", "era5_v10_mean_72h",
         "era5_msl_hpa", "era5_precip_24h_mm", "era5_precip_72h_mm", "era5_precip_7d_mm",
         "era5_swh", "era5_swh_mean_72h", "era5_stokes_u", "era5_stokes_v", "era5_stokes_speed"],
}
ENV_FEATURES = [f for cols in ENV_SOURCES.values() for f in cols]
SPECTRAL = list(BAND_NAMES) + INDICES + RATIOS

GROUPS = {"Bands": list(BAND_NAMES), "Indices": INDICES, "Ratios": RATIOS, "Environment": ENV_FEATURES}
ENV_GROUPS = {
    "SST": ["sst_c", "sst_grad_c_per_km"],
    "Chlorophyll-a": ["log10_chl", "modis_log10_chla"],
    "Wind speed": ["wind_speed", "wind_speed_mean_24h", "wind_speed_mean_72h", "wind_speed_max_72h",
                   "era5_wind_speed", "era5_wind_speed_mean_24h", "era5_wind_speed_mean_72h",
                   "era5_wind_speed_max_72h"],
    "Wind direction (u/v)": ["wind_u", "wind_v", "era5_u10", "era5_v10", "era5_u10_mean_72h",
                             "era5_v10_mean_72h"],
    "Pressure / precipitation": ["era5_msl_hpa", "era5_precip_24h_mm", "era5_precip_72h_mm",
                                 "era5_precip_7d_mm"],
    "Waves / Stokes drift": ["era5_swh", "era5_swh_mean_72h", "era5_stokes_u", "era5_stokes_v",
                             "era5_stokes_speed"],
}
K = len(CLASSES)
CNAMES = [CLASS_NAMES[c] for c in CLASSES]
SHORT = {1: "water", 2: "debris", 3: "algae", 4: "other"}
SHAP_DIRS = ["00_samples", "01_shap_values", "02_global_importance", "03_per_class", "04_feature_groups",
             "05_cross_model", "06_dependence", "07_validation"]


# ------------------------------------------------------------ environment ---

def env_columns(dataset_name, keep_tr, keep_te):
    if dataset_name != "MARIDA":
        raise SystemExit("Stage 9 is MARIDA only (MADOS has no georeferencing).")
    y_tr = np.load(os.path.join(CACHE_DIR, "MARIDA_train_all_seed42_f24.npz"))["y"]
    y_te = np.load(os.path.join(CACHE_DIR, "MARIDA_test_all_seed42_f24.npz"))["y"]
    ids_tr, ids_te = pixel_patch_ids("train", len(y_tr)), pixel_patch_ids("test", len(y_te))
    for split, y in (("train", y_tr), ("test", y_te)):
        lab = np.load(os.path.join(CACHE_DIR, f"MARIDA_{split}_all_patchlabels.npy"))
        if not np.array_equal(lab, y):
            raise RuntimeError(f"{split}: rebuilt pixel order does not match the cached labels")

    env = pd.concat([pd.read_csv(p).set_index("patch_id")[cols] for p, cols in ENV_SOURCES.items()], axis=1)
    fill = env.loc[np.unique(ids_tr)].median()
    for split, ids in (("train", ids_tr), ("test", ids_te)):
        missing = env.loc[np.unique(ids)].isna().sum()
        print(f"  env {split}: patches with a filled value: "
              + (", ".join(f"{k}={v}" for k, v in missing.items() if v) or "none"), flush=True)
    env = env.fillna(fill)
    E_tr = env.loc[ids_tr[keep_tr]].to_numpy(dtype=np.float32)
    E_te = env.loc[ids_te[keep_te]].to_numpy(dtype=np.float32)
    return E_tr, E_te, ENV_FEATURES


# ------------------------------------------------------------------- SHAP ---

class ShapRun:
    """Collects the fitted models (run_stage on_model hook) and explains each one right away,
    on the same background and test pixels for every model."""

    def __init__(self, args):
        self.args = args
        self.res = {}           # model -> dict(phi, base, pred, err, n_bad, max_f, time)
        self.D = None

    def setup(self, dataset_name, data):
        a, rng = self.args, np.random.default_rng(self.args.seed)
        out = os.path.join(a.out_root, STAGE, "shap", dataset_name)
        for d in SHAP_DIRS:
            os.makedirs(os.path.join(out, d), exist_ok=True)
        bg_idx = sc.stratified(data["ytr"], a.bg_per_class, rng)
        idx = sc.stratified(data["yte"], a.per_class, rng)
        self.D = {"out": out, "names": data["names"], "bg": data["Xtr"][bg_idx], "bg_idx": bg_idx,
                  "idx": idx, "X": data["Xte"][idx], "y": data["yte"][idx], "dataset": dataset_name}

    def __call__(self, dataset_name, model_name, model, data):
        if self.D is None:
            self.setup(dataset_name, data)
        D = self.D
        print(f"  [{dataset_name}] SHAP {model_name} on {len(D['X'])} test pixels "
              f"(background {len(D['bg'])})...", flush=True)
        t0 = time.time()
        phi, base = sc.shap_values(model, model_name, D["X"], D["bg"])
        t = time.time() - t0
        err, max_f, n_bad = sc.additivity_error(model, model_name, D["X"], phi, base)
        print(f"    done in {t:.0f}s, additivity max error {err:.2e} (max |f| {max_f:.2f}, "
              f"{n_bad} pixels > 1e-5)", flush=True)
        self.res[model_name] = {"phi": phi, "base": base, "pred": data["pred"][D["idx"]],
                                "err": err, "max_f": max_f, "n_bad": n_bad, "time": t}


# ---------------------------------------------------------------- outputs ---

def share(v):
    s = v.sum()
    return v / s if s > 0 else v


def group_of(f):
    return next(g for g, fs in GROUPS.items() if f in fs)


def write_samples(D, res):
    df = pd.DataFrame(D["X"], columns=D["names"])
    df.insert(0, "test_index", D["idx"])
    df.insert(1, "true_class", [CNAMES[k] for k in D["y"]])
    for i, (n, r) in enumerate(res.items()):
        df.insert(2 + i, f"pred_{n}", [CNAMES[k] for k in r["pred"]])
    df.to_csv(os.path.join(D["out"], "00_samples", "explained_test_pixels.csv"), index=False)
    pd.DataFrame(D["bg"], columns=D["names"]).assign(train_index=D["bg_idx"]).to_csv(
        os.path.join(D["out"], "00_samples", "background_train_pixels.csv"), index=False)


def write_shap_values(D, res):
    for n, r in res.items():
        rows = []
        for k in range(K):
            f = pd.DataFrame(r["phi"][:, :, k], columns=D["names"])
            f.insert(0, "test_index", D["idx"])
            f.insert(1, "true_class", [CNAMES[c] for c in D["y"]])
            f.insert(2, "pred_class", [CNAMES[c] for c in r["pred"]])
            f.insert(3, "explained_output", CNAMES[k])
            f.insert(4, "base_value", r["base"][k])
            rows.append(f)
        pd.concat(rows).to_csv(os.path.join(D["out"], "01_shap_values", f"{n}_shap_values.csv"), index=False)


def global_importance(D, res):
    """Mean |SHAP| over the explained pixels and the four class outputs; share = normalised."""
    names, out = D["names"], os.path.join(D["out"], "02_global_importance")
    tab = pd.DataFrame({"feature": names, "group": [group_of(f) for f in names]})
    for n, r in res.items():
        tab[f"{n}_mean_abs_shap"] = np.abs(r["phi"]).mean(axis=(0, 2))
        tab[f"{n}_share"] = share(tab[f"{n}_mean_abs_shap"].to_numpy())
    share_cols = [f"{n}_share" for n in res]
    tab["mean_share"] = tab[share_cols].mean(axis=1)
    tab["mean_rank"] = tab[share_cols].rank(ascending=False).mean(axis=1)
    tab = tab.sort_values("mean_share", ascending=False)
    tab.to_csv(os.path.join(out, "global_importance_all_models.csv"), index=False)

    pal = dict(zip(GROUPS, sns.color_palette("Set2", len(GROUPS))))
    for n in res:
        t = tab.sort_values(f"{n}_share", ascending=False).head(25)[::-1]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh(t.feature, t[f"{n}_share"], color=[pal[g] for g in t.group])
        ax.set_xlabel("share of mean |SHAP| (all classes)")
        ax.set_title(f"{sc.MODEL_LABEL[n]}: top 25 features\n({sc.OUTPUT_SCALE[n]})")
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in pal.values()], labels=list(pal),
                  loc="lower right")
        plt.tight_layout(); plt.savefig(os.path.join(out, f"{n}_global_bar.png"), dpi=120); plt.close(fig)

    hm = tab.set_index("feature")[share_cols].rename(columns=lambda c: c.replace("_share", "").upper())
    fig, ax = plt.subplots(figsize=(7, 13))
    sns.heatmap(hm * 100, annot=True, fmt=".1f", cmap="viridis", cbar_kws={"label": "% of mean |SHAP|"}, ax=ax)
    ax.set_title("Global SHAP share per model (%), sorted by mean share")
    plt.tight_layout(); plt.savefig(os.path.join(out, "global_share_heatmap.png"), dpi=120); plt.close(fig)
    return tab


def per_class(D, res):
    """For class k: mean |SHAP| of the class-k output over the pixels whose TRUE class is k,
    plus beeswarm plots (top 20) of the class-k output over all explained pixels."""
    names, out = D["names"], os.path.join(D["out"], "03_per_class")
    rows = []
    for n, r in res.items():
        for k in range(K):
            on_k = r["phi"][D["y"] == k][:, :, k]
            imp = np.abs(on_k).mean(0)
            mean_signed = on_k.mean(0)
            t = pd.DataFrame({"feature": names, "group": [group_of(f) for f in names],
                              "mean_abs_shap": imp, "share": share(imp), "mean_shap": mean_signed})
            t = t.sort_values("mean_abs_shap", ascending=False)
            t.insert(0, "rank", np.arange(1, len(t) + 1))
            t.to_csv(os.path.join(out, f"{n}_c{CLASSES[k]}_{SHORT[CLASSES[k]]}_ranking.csv"), index=False)
            rows += [{"model": n, "class": CNAMES[k], "feature": f, "share": s}
                     for f, s in zip(t.feature, t.share)]

            plt.figure()
            shap.summary_plot(r["phi"][:, :, k], D["X"], feature_names=names, max_display=20, show=False,
                              plot_size=(9, 8))
            plt.title(f"{sc.MODEL_LABEL[n]}: {CNAMES[k]} output ({sc.OUTPUT_SCALE[n]})")
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"{n}_c{CLASSES[k]}_{SHORT[CLASSES[k]]}_beeswarm.png"), dpi=110)
            plt.close("all")
    long = pd.DataFrame(rows)
    long.to_csv(os.path.join(out, "per_class_share_all_models.csv"), index=False)
    wide = long.pivot_table(index="feature", columns="class", values="share", aggfunc="mean")[CNAMES]
    wide = wide.loc[wide.mean(axis=1).sort_values(ascending=False).index]
    wide.to_csv(os.path.join(out, "per_class_share_mean_over_models.csv"))
    fig, ax = plt.subplots(figsize=(7, 13))
    sns.heatmap(wide * 100, annot=True, fmt=".1f", cmap="magma", cbar_kws={"label": "% of mean |SHAP|"}, ax=ax)
    ax.set_title("Per-class SHAP share (%), mean over models\n(class-k output on true class-k pixels)")
    plt.tight_layout(); plt.savefig(os.path.join(out, "per_class_share_heatmap.png"), dpi=120); plt.close(fig)


def feature_groups(D, res):
    out = os.path.join(D["out"], "04_feature_groups")
    names = D["names"]
    rows = []
    for n, r in res.items():
        a = np.abs(r["phi"])
        scopes = [("All classes", a.mean(axis=(0, 2)))] + \
                 [(CNAMES[k], a[D["y"] == k][:, :, k].mean(0)) for k in range(K)]
        for scope, imp in scopes:
            s = share(imp)
            for kind, groups in (("group", GROUPS), ("env_subgroup", ENV_GROUPS)):
                for g, fs in groups.items():
                    rows.append({"model": n, "scope": scope, "kind": kind, "group": g,
                                 "share": float(s[[names.index(f) for f in fs]].sum())})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "group_share.csv"), index=False)

    for kind, groups, fname in (("group", GROUPS, "group_share"), ("env_subgroup", ENV_GROUPS, "env_subgroup_share")):
        g = t[t.kind == kind]
        scopes = ["All classes"] + CNAMES
        fig, axes = plt.subplots(1, len(scopes), figsize=(4 * len(scopes), 4.5), sharey=True)
        for ax, scope in zip(axes, scopes):
            p = g[g.scope == scope].pivot(index="model", columns="group", values="share")[list(groups)]
            p = p.reindex([m for m in sc.MODEL_NAMES if m in p.index])
            (p * 100).plot(kind="bar", stacked=True, ax=ax, legend=False, colormap="Set2" if kind == "group" else "tab10")
            ax.set_title(scope); ax.set_xlabel(""); ax.set_ylabel("% of mean |SHAP|")
        axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.suptitle("Feature-group share of SHAP" if kind == "group" else
                     "Environmental sub-groups: share of total SHAP")
        plt.tight_layout(); plt.savefig(os.path.join(out, f"{fname}.png"), dpi=120); plt.close(fig)
    return t


def cross_model(D, res, tab):
    out = os.path.join(D["out"], "05_cross_model")
    share_cols = [f"{n}_share" for n in res]
    ranks = tab.set_index("feature")[share_cols].rank(ascending=False).astype(int)
    ranks.columns = [c.replace("_share", "") for c in share_cols]
    ranks.sort_values(ranks.columns[0]).to_csv(os.path.join(out, "feature_rank_per_model.csv"))
    rho = ranks.corr(method="spearman")
    rho.to_csv(os.path.join(out, "rank_spearman_correlation.csv"))
    top10 = {n: set(ranks[n].nsmallest(10).index) for n in ranks}
    overlap = pd.DataFrame([[len(top10[a] & top10[b]) for b in ranks] for a in ranks], index=ranks.columns,
                           columns=ranks.columns)
    overlap.to_csv(os.path.join(out, "top10_overlap.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(rho, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax[0])
    ax[0].set_title("Spearman correlation of global SHAP rankings")
    sns.heatmap(overlap, annot=True, fmt="d", cmap="Blues", vmin=0, vmax=10, ax=ax[1])
    ax[1].set_title("Shared features in the top 10")
    plt.tight_layout(); plt.savefig(os.path.join(out, "model_agreement.png"), dpi=120); plt.close(fig)

    # Agreement between SHAP and each model's built-in importance (training/feature_importance.csv)
    fi_path = os.path.join(D["out"], "..", "..", "training", D["dataset"], "feature_importance.csv")
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
        rows = []
        for n in res:
            m = fi[fi.model == n].set_index("feature")["importance"].reindex(tab.feature)
            rows.append({"model": n, "spearman_shap_vs_builtin":
                         float(pd.Series(tab.set_index("feature")[f"{n}_share"]).corr(m, method="spearman"))})
        pd.DataFrame(rows).to_csv(os.path.join(out, "shap_vs_builtin_importance.csv"), index=False)
    return rho, overlap


def dependence(D, res, tab, top_env=4):
    """Debris-output SHAP vs feature value for each model's top environmental features
    and its top spectral feature, coloured by true class."""
    out = os.path.join(D["out"], "06_dependence")
    names, k = D["names"], CLASSES.index(2)
    colors = sns.color_palette("tab10", K)
    for n, r in res.items():
        t = tab.sort_values(f"{n}_share", ascending=False)
        feats = list(t[t.group == "Environment"].feature[:top_env]) + list(t[t.group != "Environment"].feature[:2])
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        for ax, f in zip(axes.ravel(), feats):
            j = names.index(f)
            for c in range(K):
                m = D["y"] == c
                ax.scatter(D["X"][m, j], r["phi"][m, j, k], s=6, alpha=0.5, color=colors[c], label=CNAMES[c])
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xlabel(f); ax.set_ylabel("SHAP (Marine Debris output)")
            ax.set_title(f"{f} [{group_of(f)}]")
        axes[0, 0].legend(fontsize=8, markerscale=2)
        fig.suptitle(f"{sc.MODEL_LABEL[n]}: dependence of the Marine Debris output ({sc.OUTPUT_SCALE[n]})")
        plt.tight_layout(); plt.savefig(os.path.join(out, f"{n}_debris_dependence.png"), dpi=110); plt.close(fig)


def validation(D, res, args):
    out = os.path.join(D["out"], "07_validation")
    v = {n: {"explainer": sc.EXPLAINER[n], "explained_output": sc.OUTPUT_SCALE[n],
             "additivity_max_abs_error": r["err"], "max_abs_output": r["max_f"],
             "pixels_error_gt_1e-5": r["n_bad"], "shap_seconds": round(r["time"], 1),
             "base_values": dict(zip(CNAMES, map(float, r["base"])))} for n, r in res.items()}
    meta = {"explained_test_pixels": {"n": int(len(D["y"])), "per_class": dict(zip(CNAMES, np.bincount(D["y"], minlength=K).tolist())),
                                      "per_class_cap": args.per_class},
            "background_train_pixels": {"n": int(len(D["bg"])), "per_class_cap": args.bg_per_class},
            "seed": args.seed, "n_features": len(D["names"]), "features": D["names"],
            "models": v, "shap_version": shap.__version__}
    with open(os.path.join(out, "validation.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def report(D, res, tab, groups, rho, meta, args):
    names = D["names"]
    train_dir = os.path.join(args.out_root, STAGE, "training", D["dataset"])
    met = pd.read_csv(os.path.join(train_dir, "metrics.csv")).set_index("model")
    L = [f"# Stage 9 SHAP: all spectral + all environmental features ({D['dataset']}, log weights)", "",
         f"{len(names)} features: {len(BAND_NAMES)} bands, {len(INDICES)} indices, {len(RATIOS)} ratios, "
         f"{len(ENV_FEATURES)} environmental covariates (both env CSVs in `data/`). "
         f"Explained: {meta['explained_test_pixels']['n']} class-stratified test pixels "
         f"({meta['explained_test_pixels']['per_class']}); background: "
         f"{meta['background_train_pixels']['n']} class-stratified training pixels.", "",
         "## Test scores (all annotated test pixels)", "",
         "| Model | macro F1 | IoU | Debris F1 | Algae F1 | Kappa |", "|---|---|---|---|---|---|"]
    for n in res:
        r = met.loc[n]
        L.append(f"| {n} | {r.f1:.4f} | {r.iou:.4f} | {r.f1_c2:.4f} | {r.f1_c3:.4f} | {r.kappa:.4f} |")

    ref = []
    for st in ("stage4_indices_ratios", "stage6_env"):
        p = os.path.join(args.out_root, st, D["dataset"], "metrics.csv")
        if os.path.exists(p):
            ref.append((st, pd.read_csv(p).set_index("model")))
    if ref:
        L += ["", "### Compared with earlier log-weight stages (Δ = stage 9 − reference)", "",
              "| Model | " + " | ".join(f"Δ F1 vs {s} | Δ Debris F1 vs {s}" for s, _ in ref) + " |",
              "|---|" + "---|---|" * len(ref)]
        for n in res:
            cells = []
            for _, m in ref:
                cells += [f"{met.loc[n].f1 - m.loc[n].f1:+.4f}", f"{met.loc[n].f1_c2 - m.loc[n].f1_c2:+.4f}"] \
                    if n in m.index else ["", ""]
            L.append(f"| {n} | " + " | ".join(cells) + " |")

    L += ["", "## Feature-group share of mean |SHAP| (all classes)", "",
          "| Model | " + " | ".join(GROUPS) + " |", "|---|" + "---|" * len(GROUPS)]
    g = groups[(groups.kind == "group") & (groups.scope == "All classes")]
    for n in res:
        s = g[g.model == n].set_index("group").share
        L.append(f"| {n} | " + " | ".join(f"{100 * s[x]:.1f}%" for x in GROUPS) + " |")
    L += ["", "## Environmental share per class (Marine Debris / Algae rows matter most)", "",
          "| Model | " + " | ".join(CNAMES) + " |", "|---|" + "---|" * K]
    for n in res:
        s = groups[(groups.kind == "group") & (groups.group == "Environment") & (groups.model == n)].set_index("scope").share
        L.append(f"| {n} | " + " | ".join(f"{100 * s[c]:.1f}%" for c in CNAMES) + " |")
    L += ["", "## Environmental sub-groups (share of total SHAP, all classes)", "",
          "| Model | " + " | ".join(ENV_GROUPS) + " |", "|---|" + "---|" * len(ENV_GROUPS)]
    e = groups[(groups.kind == "env_subgroup") & (groups.scope == "All classes")]
    for n in res:
        s = e[e.model == n].set_index("group").share
        L.append(f"| {n} | " + " | ".join(f"{100 * s[x]:.1f}%" for x in ENV_GROUPS) + " |")

    L += ["", f"## Top 15 features (mean share over the {len(res)} models)", "",
          "| # | Feature | Group | Mean share | " + " | ".join(res) + " |", "|---|---|---|---|" + "---|" * len(res)]
    for i, (_, r) in enumerate(tab.head(15).iterrows(), 1):
        L.append(f"| {i} | {r.feature} | {r.group} | {100 * r.mean_share:.1f}% | "
                 + " | ".join(f"{100 * r[f'{n}_share']:.1f}%" for n in res) + " |")
    env_rank = tab[tab.group == "Environment"].head(10)
    L += ["", "## Top 10 environmental features", "", "| Feature | Mean share | Mean rank (of %d) |" % len(names),
          "|---|---|---|"]
    for _, r in env_rank.iterrows():
        L.append(f"| {r.feature} | {100 * r.mean_share:.2f}% | {r.mean_rank:.1f} |")

    off = rho.where(~np.eye(len(rho), dtype=bool)).stack()
    L += ["", "## Model agreement", "",
          f"Spearman correlation between the models' global SHAP rankings: mean {off.mean():.2f} "
          f"(min {off.min():.2f}). See `05_cross_model/`.", "",
          "## SHAP validation", "", "| Model | explainer | additivity max error | max |output| | seconds |",
          "|---|---|---|---|---|"]
    for n, v in meta["models"].items():
        L.append(f"| {n} | {v['explainer']} | {v['additivity_max_abs_error']:.2e} | {v['max_abs_output']:.2f} | "
                 f"{v['shap_seconds']:.0f} |")
    L += ["", "SHAP values are on each model's own output scale (RF: probability; XGB/LGBM: softmax margin; "
          "AdaBoost: SAMME score; SVM: one-vs-rest margin), so compare models through the normalised "
          "shares, not raw magnitudes.", "",
          "Environmental values are constant within a 256×256 patch and MARIDA has only 63 scenes, so a "
          "high environmental share can mean the model uses them to recognise the scene rather than "
          "the conditions; check the test scores against stage 4 before reading it as a physical effect.", "",
          "## Folders", "",
          "- `00_samples/`: explained test pixels (features, true and predicted class) and the background pixels",
          "- `01_shap_values/`: raw SHAP values per model, one row per pixel and class output",
          "- `02_global_importance/`: global mean |SHAP| and share per feature and model, bar charts, heatmap",
          "- `03_per_class/`: per-class rankings and beeswarm plots for every model",
          "- `04_feature_groups/`: bands / indices / ratios / environment shares, environmental sub-groups",
          "- `05_cross_model/`: ranking agreement between models and with built-in importance",
          "- `06_dependence/`: Marine Debris dependence plots for the top environmental features",
          "- `07_validation/`: additivity check, explainer settings, timings",
          "- `../../training/MARIDA/`: test metrics, confusion matrices, built-in feature importance"]
    with open(os.path.join(D["out"], "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = base_parser(__doc__)
    ap.set_defaults(datasets=["MARIDA"], weighting="log", out_root=os.path.join(OUT_ROOT, "log_weights"))
    ap.add_argument("--per-class", type=int, default=300, help="stratified test pixels per class to explain")
    ap.add_argument("--bg-per-class", type=int, default=50, help="background training pixels per class")
    args = ap.parse_args()
    if args.datasets != ["MARIDA"]:
        raise SystemExit("Stage 9 is MARIDA only (MADOS has no georeferencing).")

    t0 = time.time()
    runner = ShapRun(args)
    run_stage(STAGE, "bands + 5 indices + 4 band ratios + all environmental covariates", SPECTRAL, args,
              extra=env_columns, subdir="training", on_model=runner)
    D, res = runner.D, runner.res
    print("\n  writing SHAP outputs...", flush=True)
    write_samples(D, res)
    write_shap_values(D, res)
    tab = global_importance(D, res)
    per_class(D, res)
    groups = feature_groups(D, res)
    rho, _ = cross_model(D, res, tab)
    dependence(D, res, tab)
    meta = validation(D, res, args)
    report(D, res, tab, groups, rho, meta, args)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. SHAP outputs: {D['out']}", flush=True)
