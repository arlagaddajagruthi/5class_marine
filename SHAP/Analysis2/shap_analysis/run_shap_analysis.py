"""
Complete SHAP analysis of the five saved stage-8 models (RF, XGBoost, LightGBM,
AdaBoost, linear SVM; 22 features = 11 S2 bands + NDWI + 4 band ratios from
src/features/pixel_features.py + 6 GLCM textures from scripts/staged_ml/glcm_features.py;
classes 1 Water, 2 Marine Debris, 3 Sargassum/algae, 4 Other, encoded 0..3).

Run train_stage8_models.py first. Then (repo root):
  .venv/Scripts/python scripts/shap_analysis/run_shap_analysis.py --data-root ../5_class
  .venv/Scripts/python scripts/shap_analysis/make_report.py

Writes results/shap_analysis/ (see README.md there). Figures are numbered per dataset
in report order and every figure gets a facts entry (figures_<dataset>.json) that
make_report.py turns into the Markdown report.
"""
import argparse
import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from SHAP.Analysis2.shap_analysis.stage8_setup import configure, stage8_data  # noqa: E402
import SHAP.Analysis2.shap_analysis.shap_compute as sc  # noqa: E402
import common  # noqa: E402  (scripts/staged_ml, added to sys.path by stage8_setup)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

K = sc.K
MODELS = sc.MODEL_NAMES
CLS = ["Water", "Marine Debris", "Sargassum", "Other"]                 # encoded 0..3 = labels 1..4
CLS_FULL = [f"Class {i + 1} {n}" for i, n in enumerate(CLS)]
CLS_TAG = ["c1_water", "c2_debris", "c3_sargassum", "c4_other"]
CLASS_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]               # water, debris, sargassum, other
MODEL_COLOR = {"rf": "#2a78d6", "xgb": "#eb6834", "lgbm": "#1baf7a", "ada": "#eda100", "svm": "#e87ba4"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5",
                                                     "#256abf", "#184f95", "#0d366b"])
DPI = 200
SECTIONS = {
    "01": "01_global_importance", "02": "02_class_specific", "03": "03_cross_model_comparison",
    "04": "04_dependence_and_interactions", "05": "05_individual_predictions", "06": "06_reports",
    "00": "00_shap_values",
}


# ======================================================================= utils ==

class FigureLog:
    """Numbers figures in report order and records the computed facts for each."""

    def __init__(self, dataset, root):
        self.ds, self.root, self.items, self.n = dataset, root, [], 0

    def path(self, sec, name):
        d = os.path.join(self.root, SECTIONS[sec], self.ds)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self.ds}_{name}.png")

    def next_title(self, text):
        self.n += 1
        return f"Fig. {self.ds}-{self.n}", f"Fig. {self.ds}-{self.n}. {text}"

    def add(self, fig_id, title, path, section, item, **facts):
        self.items.append({"id": fig_id, "title": title, "section": section, "item": item,
                           "path": os.path.relpath(path, self.root).replace("\\", "/"), **facts})


def save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if os.path.getsize(path) == 0:
        raise RuntimeError(f"empty figure {path}")


def shap_fig(draw, title, path, size=None, xlabel=None):
    """Run a shap.plots.* call that draws on the current figure, then title + save."""
    plt.close("all")
    draw()
    fig = plt.gcf()
    if size:
        fig.set_size_inches(*size)
    if xlabel:
        fig.axes[0].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=11, y=1.0, ha="center")
    save(fig, path)


def expl(values, X, names, base=None):
    return shap.Explanation(values=values, base_values=base, data=X, feature_names=names)


def top(vec, names, n):
    order = np.argsort(-vec)[:n]
    return [(names[i], float(vec[i])) for i in order]


def fmt_top(pairs, unit="", pct=False):
    if pct:
        return ", ".join(f"{f} ({v * 100:.1f}%)" for f, v in pairs)
    return ", ".join(f"{f} ({v:.3g}{unit})" for f, v in pairs)


def direction_word(rho):
    if rho is None or np.isnan(rho):
        return "no consistent monotonic direction"
    if rho >= 0.5:
        return "higher values push the output up"
    if rho <= -0.5:
        return "higher values push the output down"
    if rho >= 0.2:
        return "weakly, higher values push the output up"
    if rho <= -0.2:
        return "weakly, higher values push the output down"
    return "no consistent monotonic direction (|rho| < 0.2; non-monotonic or interaction-driven)"


def spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


# ===================================================================== compute ==

def compute(ds, a, root):
    """SHAP values for every model on one explanation set, cached under 00_shap_values/<ds>/."""
    vdir = os.path.join(root, SECTIONS["00"], ds)
    os.makedirs(vdir, exist_ok=True)
    cache = os.path.join(vdir, f"{ds}_explanation_set.npz")
    Xtr, ytr, Xte, yte, names = stage8_data(ds, a.seed)
    model_dir = os.path.join(REPO, "models", "stage8_log", ds)
    models = sc.load_models(model_dir)
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    saved = np.load(os.path.join(model_dir, "test_pred.npz"))

    # ---------------- validation of models / features / labels -----------------
    val = {"dataset": ds, "n_features": len(names), "features": names,
           "feature_order_matches_meta": names == meta["features"],
           "class_encoding": meta["class_encoding"],
           "train_counts": np.bincount(ytr, minlength=K).tolist(),
           "test_counts": np.bincount(yte, minlength=K).tolist(), "models": {}}
    assert np.array_equal(saved["y_true"], yte), "saved test labels do not match the staged test pixels"
    for n, m in models.items():
        nf = m.named_steps["scaler"].n_features_in_ if n == "svm" else m.n_features_in_
        classes = (m.named_steps["clf"].classes_ if n == "svm" else m.classes_).tolist()
        probe = Xte[:: max(1, len(Xte) // 5000)]
        f = sc.raw_output(m, n, probe)
        pred_probe = np.asarray(m.predict(probe)).ravel()
        val["models"][n] = {
            "n_features_in": int(nf), "classes_": classes, "raw_output_shape_probe": list(f.shape),
            "argmax_raw_equals_predict": bool(np.array_equal(f.argmax(1), pred_probe)),
            "predict_matches_saved": bool(np.array_equal(pred_probe, saved[n][:: max(1, len(Xte) // 5000)])),
            "explained_output": sc.OUTPUT_SCALE[n], "explainer": sc.EXPLAINER[n],
            "test_macro_f1": meta["models"][n]["metrics"]["f1"],
            "recorded_macro_f1": meta["models"][n]["recorded_metrics"]["f1"],
        }
        assert nf == len(names) and classes == [0, 1, 2, 3], (n, nf, classes)
        assert val["models"][n]["argmax_raw_equals_predict"] and val["models"][n]["predict_matches_saved"], n
    preds = {n: saved[n] for n in MODELS}

    rng = np.random.default_rng(a.seed)
    bg_idx = sc.stratified(ytr, a.bg_per_class, rng)                    # TRAINING pixels only
    idx, strat = sc.explanation_set(yte, preds, a.per_class, a.err_per_pair, a.seed)
    X, y = Xte[idx], yte[idx]
    bg = Xtr[bg_idx]
    int_rng = np.random.default_rng(a.seed + 1)
    int_sel = np.flatnonzero(strat)[sc.stratified(y[strat], a.int_per_class, int_rng)]
    val.update({"explanation_set": {"n_total": int(len(idx)), "n_stratified": int(strat.sum()),
                                    "stratified_per_class": np.bincount(y[strat], minlength=K).tolist(),
                                    "n_extra_misclassified": int((~strat).sum()),
                                    "per_class_cap": a.per_class, "err_per_pair_cap": a.err_per_pair},
                "background": {"source": "train split", "per_class": np.bincount(ytr[bg_idx], minlength=K).tolist()},
                "interaction_subset": {"n": int(len(int_sel)),
                                       "per_class": np.bincount(y[int_sel], minlength=K).tolist()}})

    out = {"idx": idx, "strat": strat, "X": X, "y": y, "bg_idx": bg_idx, "int_sel": int_sel,
           "names": np.array(names)}
    for n in MODELS:
        out[f"pred_{n}"] = preds[n][idx]
    old = dict(np.load(cache, allow_pickle=False)) if (os.path.exists(cache) and not a.recompute) else {}
    same = bool(old) and np.array_equal(old.get("idx", []), idx) and np.array_equal(old.get("bg_idx", []), bg_idx)
    if not same:
        old = {}                                                        # stale cache: recompute everything
    timings = {}
    for n in MODELS:
        m = models[n]
        if same and f"phi_{n}" in old:
            phi, base = old[f"phi_{n}"], old[f"base_{n}"]
        else:
            print(f"  [{ds}] SHAP {n} on {len(X)} pixels (background {len(bg)})...", flush=True)
            (phi, base), timings[n] = sc.timed(sc.shap_values, m, n, X, bg)
        err, scale, n_off = sc.additivity_error(m, n, X, phi, base)
        assert phi.shape == (len(X), len(names), K), phi.shape
        # tolerate a single-tree routing difference on a few pixels, never a systematic error
        assert err <= 0.02 * max(1.0, scale) and n_off <= 0.01 * len(X), f"{n}: additivity error {err} on {n_off} px"
        val["models"][n].update({"shap_shape": list(phi.shape), "base_values": base.tolist(),
                                 "additivity_max_abs_error": err, "max_abs_output": scale,
                                 "additivity_pixels_above_1e-5": n_off,
                                 "shap_seconds": round(timings.get(n, float("nan")), 1)})
        out[f"phi_{n}"], out[f"base_{n}"] = phi, base
        old.update({f"phi_{n}": phi, f"base_{n}": base, "idx": idx, "bg_idx": bg_idx})
        np.savez_compressed(cache, **old)
        same = True
        out[f"f_{n}"] = sc.raw_output(m, n, X)

        if n == "svm":
            ksel = np.flatnonzero(strat)[sc.stratified(y[strat], max(1, a.kernel_check // K), np.random.default_rng(7))]
            if same and "kernel_phi" in old and np.array_equal(old["kernel_sel"], ksel):
                kphi = old["kernel_phi"]
            else:
                print(f"  [{ds}] KernelSHAP check on {len(ksel)} pixels...", flush=True)
                kphi, _ = sc.kernel_check(m, X[ksel], bg)
            out["kernel_phi"], out["kernel_sel"] = kphi, ksel
            d = np.abs(kphi - phi[ksel])
            val["models"][n]["kernel_check"] = {"n": int(len(ksel)), "max_abs_diff": float(d.max()),
                                                "max_abs_phi": float(np.abs(phi[ksel]).max()),
                                                "corr": float(np.corrcoef(kphi.ravel(), phi[ksel].ravel())[0, 1])}
            continue
        if same and f"iv_{n}" in old:
            iv = old[f"iv_{n}"]
            # re-check additivity from the cache: sum_ij phi_ij + E[f] (path-dependent) = f(x)
            # (recompute 2 pixels: gives the explainer's base value, which for XGBoost is only set
            # correctly by shap_interaction_values, and confirms the cached values)
            cw = common.class_weights(ytr, "log")[ytr] if n == "ada" else None
            iv2, ib = sc.interaction_values(m, n, X[int_sel[:2]], Xtr if n == "ada" else None, cw)
            val["models"][n]["interaction_cache_max_abs_diff"] = float(np.abs(iv2 - iv[:2]).max())
            val["models"][n]["interaction_additivity_max_abs_error"] = float(
                np.abs(iv.sum(axis=(1, 2)) + ib[None] - out[f"f_{n}"][int_sel]).max())
        else:
            print(f"  [{ds}] interaction values {n} on {len(int_sel)} pixels...", flush=True)
            cw = common.class_weights(ytr, "log")[ytr] if n == "ada" else None
            (res, t) = sc.timed(sc.interaction_values, m, n, X[int_sel],
                                Xtr if n == "ada" else None, cw)
            iv, ib = res
            timings[f"iv_{n}"] = t
            ierr = float(np.abs(iv.sum(axis=(1, 2)) + ib[None] - out[f"f_{n}"][int_sel]).max())
            val["models"][n]["interaction_additivity_max_abs_error"] = ierr
        out[f"iv_{n}"] = iv
    prev_t = json.loads(str(old["timings_json"])) if "timings_json" in old else {}
    timings = {**prev_t, **timings}
    out["timings_json"] = np.array(json.dumps(timings))
    np.savez_compressed(cache, **out)
    val["timings_s"] = {k: round(v, 1) for k, v in timings.items()}
    for n in MODELS:
        val["models"][n]["shap_seconds"] = round(timings[n], 1) if n in timings else None
    return out, val, names, meta


# ============================================================= tables / stats ==

def write_tables(ds, D, names, root):
    tdir = os.path.join(root, SECTIONS["06"], "tables", ds)
    os.makedirs(tdir, exist_ok=True)
    s, X, y = D["strat"], D["X"], D["y"]
    Xs, ys = X[s], y[s]
    rows_g, rows_c = [], []
    S = {}
    for n in MODELS:
        phi = D[f"phi_{n}"][s]
        g = np.abs(phi).mean(axis=(0, 2))
        c = np.abs(phi).mean(axis=0)                                          # (F, K)
        ct = np.stack([np.abs(phi[ys == k, :, k]).mean(0) for k in range(K)], 1)
        rho = np.array([[spearman(Xs[:, j], phi[:, j, k]) for k in range(K)] for j in range(len(names))])
        rho_t = np.array([spearman(Xs[:, j], phi[np.arange(len(ys)), j, ys]) for j in range(len(names))])
        S[n] = {"global": g, "class": c, "class_true": ct, "rho": rho, "rho_true": rho_t}
        for j, f in enumerate(names):
            rows_g.append({"model": n, "feature": f, "mean_abs_shap": g[j], "share": g[j] / g.sum(),
                           "rank": int((g > g[j]).sum() + 1), "spearman_value_vs_trueclass_shap": rho_t[j],
                           "output_scale": sc.OUTPUT_SCALE[n]})
            for k in range(K):
                rows_c.append({"model": n, "class": CLS_FULL[k], "feature": f,
                               "mean_abs_shap": c[j, k], "share": c[j, k] / c[:, k].sum(),
                               "rank": int((c[:, k] > c[j, k]).sum() + 1),
                               "mean_abs_shap_on_true_class_pixels": ct[j, k],
                               "mean_shap_on_true_class_pixels": phi[ys == k, j, k].mean(),
                               "spearman_value_vs_shap": rho[j, k], "output_scale": sc.OUTPUT_SCALE[n]})
    G, C = pd.DataFrame(rows_g), pd.DataFrame(rows_c)
    G.to_csv(os.path.join(tdir, f"{ds}_global_importance_all_models.csv"), index=False)
    C.to_csv(os.path.join(tdir, f"{ds}_class_importance_all_models.csv"), index=False)
    for n in MODELS:
        G[G.model == n].sort_values("rank").to_csv(os.path.join(tdir, f"{ds}_{n}_global_ranking.csv"), index=False)
        for k in range(K):
            C[(C.model == n) & (C["class"] == CLS_FULL[k])].sort_values("rank").to_csv(
                os.path.join(tdir, f"{ds}_{n}_{CLS_TAG[k]}_ranking.csv"), index=False)
    share = G.pivot(index="feature", columns="model", values="share").loc[names, MODELS]
    rank = G.pivot(index="feature", columns="model", values="rank").loc[names, MODELS]
    cmp_ = share.add_suffix("_share").join(rank.add_suffix("_rank"))
    cmp_["mean_share"] = share.mean(1)
    cmp_.sort_values("mean_share", ascending=False).to_csv(os.path.join(tdir, f"{ds}_cross_model_comparison.csv"))
    cshare = C.pivot_table(index="feature", columns=["model", "class"], values="share").loc[names]
    cshare.to_csv(os.path.join(tdir, f"{ds}_class_wise_comparison_share.csv"))

    # raw SHAP values (stratified + error pixels) as CSV, one file per model
    vdir = os.path.join(root, SECTIONS["00"], ds)
    for n in MODELS:
        phi = D[f"phi_{n}"]
        cols = {f"shap__{f}__{CLS_TAG[k]}": phi[:, j, k] for k in range(K) for j, f in enumerate(names)}
        df = pd.DataFrame({"test_index": D["idx"], "in_stratified_sample": D["strat"],
                           "true_class": D["y"] + 1, "pred_class": D[f"pred_{n}"] + 1,
                           **{f"value__{f}": X[:, j] for j, f in enumerate(names)},
                           **{f"base__{CLS_TAG[k]}": D[f"base_{n}"][k] for k in range(K)},
                           **{f"output__{CLS_TAG[k]}": D[f"f_{n}"][:, k] for k in range(K)}, **cols})
        df.to_csv(os.path.join(vdir, f"{ds}_{n}_shap_values.csv"), index=False, float_format="%.6g")
    return S, G, C, tdir


# ======================================================================= plots ==

def section_A(ds, D, S, names, L):
    s = D["strat"]
    X, y = D["X"][s], D["y"][s]
    F = len(names)
    for n in MODELS:                                                        # A1 bar
        phi = D[f"phi_{n}"][s]
        fid, title = L.next_title(f"{sc.MODEL_LABEL[n]}: global SHAP importance (mean |SHAP| over the 4 class outputs)")
        p = L.path("01", f"{n}_global_bar")
        shap_fig(lambda: shap.plots.bar(expl(np.abs(phi).mean(2), X, names), max_display=F, show=False),
                 title, p, size=(8, 7.5), xlabel=f"mean(|SHAP|), averaged over class outputs  [{sc.OUTPUT_SCALE[n]}]")
        g = S[n]["global"]
        L.add(fid, title, p, "A", "1", model=n, kind="global_bar",
              top5=top(g / g.sum(), names, 5), bottom3=top(-g / g.sum(), names, 3),
              top3_share=float(np.sort(g)[::-1][:3].sum() / g.sum()), scale=sc.OUTPUT_SCALE[n])
    for n in MODELS:                                                        # A2 beeswarm
        phi = D[f"phi_{n}"][s]
        pt = phi[np.arange(len(y)), :, y]
        fid, title = L.next_title(f"{sc.MODEL_LABEL[n]}: SHAP beeswarm of each pixel's TRUE-class output")
        p = L.path("01", f"{n}_global_beeswarm_trueclass")
        shap_fig(lambda: shap.plots.beeswarm(expl(pt, X, names), max_display=F, show=False),
                 title, p, size=(9, 8), xlabel=f"SHAP value for the true-class output  [{sc.OUTPUT_SCALE[n]}]")
        imp = np.abs(pt).mean(0)
        order = np.argsort(-imp)[:5]
        L.add(fid, title, p, "A", "2", model=n, kind="global_beeswarm",
              top5=[(names[j], float(imp[j]), float(S[n]["rho_true"][j]), float(pt[:, j].mean())) for j in order],
              scale=sc.OUTPUT_SCALE[n])

    # A3 cross-model heatmap + rank agreement
    share = np.stack([S[n]["global"] / S[n]["global"].sum() for n in MODELS], 1)        # (F, M)
    order = np.argsort(-share.mean(1))
    fig, ax = plt.subplots(figsize=(7.5, 9))
    im = ax.imshow(share[order] * 100, cmap=SEQ, aspect="auto")
    for i in range(F):
        for j in range(len(MODELS)):
            v = share[order][i, j] * 100
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > share.max() * 60 else "#0b0b0b")
    ax.set_xticks(range(len(MODELS)), [sc.MODEL_LABEL[n] for n in MODELS], rotation=20)
    ax.set_yticks(range(F), [names[i] for i in order])
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("share of the model's total mean |SHAP| (%)")
    fid, title = L.next_title("Cross-model comparison of normalised global SHAP importance")
    ax.set_title(title + "\nshare_j = mean|SHAP_j| / sum_f mean|SHAP_f| (per model, columns sum to 100%)", fontsize=10)
    p = L.path("03", "cross_model_global_importance_heatmap")
    save(fig, p)
    rk = pd.DataFrame(share, index=names, columns=MODELS).rank(ascending=False)
    L.add(fid, title, p, "A", "3", kind="cross_model_heatmap",
          top_by_mean=[(names[i], float(share.mean(1)[i])) for i in order[:6]],
          top1={n: names[int(np.argmax(share[:, j]))] for j, n in enumerate(MODELS)},
          per_model_top3={n: [names[i] for i in np.argsort(-share[:, j])[:3]] for j, n in enumerate(MODELS)},
          rank_spread=[(names[i], int(rk.loc[names[i]].min()), int(rk.loc[names[i]].max())) for i in order[:8]],
          glcm_share={n: float(share[[names.index(g) for g in names if g.startswith("glcm")], j].sum())
                      for j, n in enumerate(MODELS)},
          band_share={n: float(share[:11, j].sum()) for j, n in enumerate(MODELS)},
          index_ratio_share={n: float(share[11:16, j].sum()) for j, n in enumerate(MODELS)})

    rho = np.array([[spearman(share[:, i], share[:, j]) for j in range(len(MODELS))] for i in range(len(MODELS))])
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(rho, cmap=SEQ, vmin=min(0, rho.min()), vmax=1)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if rho[i, j] > 0.75 else "#0b0b0b")
    lab = [sc.MODEL_LABEL[n] for n in MODELS]
    ax.set_xticks(range(len(MODELS)), lab, rotation=25); ax.set_yticks(range(len(MODELS)), lab)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Spearman rank correlation of feature importance")
    fid, title = L.next_title("Agreement of global feature rankings between models")
    ax.set_title(title, fontsize=10)
    p = L.path("03", "cross_model_rank_agreement")
    save(fig, p)
    off = [(MODELS[i], MODELS[j], float(rho[i, j])) for i in range(len(MODELS)) for j in range(i + 1, len(MODELS))]
    L.add(fid, title, p, "A", "3", kind="rank_agreement", pairs=sorted(off, key=lambda t: -t[2]))
    return share


def section_B(ds, D, S, names, L):
    s = D["strat"]
    X, y = D["X"][s], D["y"][s]
    F = len(names)
    for n in MODELS:                                                        # B4 class bars
        for k in range(K):
            phi = D[f"phi_{n}"][s][:, :, k]
            fid, title = L.next_title(f"{sc.MODEL_LABEL[n]}: SHAP importance for the {CLS_FULL[k]} output")
            p = L.path("02", f"{n}_{CLS_TAG[k]}_bar")
            shap_fig(lambda: shap.plots.bar(expl(phi, X, names), max_display=F, show=False), title, p,
                     size=(8, 7.5), xlabel=f"mean(|SHAP|) for {CLS[k]} output  [{sc.OUTPUT_SCALE[n]}]")
            c = S[n]["class"][:, k]
            ct = S[n]["class_true"][:, k]
            L.add(fid, title, p, "B", "4", model=n, cls=k, kind="class_bar",
                  top5=top(c / c.sum(), names, 5), top3_true=top(ct / ct.sum(), names, 3),
                  scale=sc.OUTPUT_SCALE[n])
    for n in MODELS:                                                        # B5 class beeswarms
        for k in range(K):
            phi = D[f"phi_{n}"][s][:, :, k]
            fid, title = L.next_title(f"{sc.MODEL_LABEL[n]}: SHAP beeswarm for the {CLS_FULL[k]} output")
            p = L.path("02", f"{n}_{CLS_TAG[k]}_beeswarm")
            shap_fig(lambda: shap.plots.beeswarm(expl(phi, X, names), max_display=F, show=False), title, p,
                     size=(9, 8), xlabel=f"SHAP value for {CLS[k]} output  [{sc.OUTPUT_SCALE[n]}]")
            c = S[n]["class"][:, k]
            order = np.argsort(-c)[:5]
            L.add(fid, title, p, "B", "5", model=n, cls=k, kind="class_beeswarm",
                  top5=[(names[j], float(c[j]), float(S[n]["rho"][j, k]),
                         float(phi[y == k, j].mean()), float(phi[y != k, j].mean())) for j in order],
                  scale=sc.OUTPUT_SCALE[n])

    # B6 class-wise heatmap: features x (model, class)
    M = np.stack([S[n]["class"][:, k] / S[n]["class"][:, k].sum() for n in MODELS for k in range(K)], 1)
    order = np.argsort(-M.mean(1))
    fig, ax = plt.subplots(figsize=(15, 9))
    im = ax.imshow(M[order] * 100, cmap=SEQ, aspect="auto")
    for i in range(F):
        for j in range(M.shape[1]):
            v = M[order][i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > M.max() * 60 else "#0b0b0b")
    ax.set_xticks(range(M.shape[1]), [f"{sc.MODEL_LABEL[n]}\n{CLS[k]}" for n in MODELS for k in range(K)],
                  rotation=90, fontsize=7.5)
    ax.set_yticks(range(F), [names[i] for i in order])
    for b in range(1, len(MODELS)):
        ax.axvline(b * K - 0.5, color="white", lw=3)
    fig.colorbar(im, ax=ax, shrink=0.6).set_label("share of mean |SHAP| within (model, class output), %")
    fid, title = L.next_title("Class-wise SHAP importance across all five models and four class outputs")
    ax.set_title(title + "\nshare = mean|SHAP_j,k| / sum_f mean|SHAP_f,k| (each column sums to 100%)", fontsize=10)
    p = L.path("03", "class_wise_importance_heatmap")
    save(fig, p)
    per_class_top = {}
    for k in range(K):
        mk = np.stack([S[n]["class"][:, k] / S[n]["class"][:, k].sum() for n in MODELS], 1)
        per_class_top[k] = {"mean_top4": top(mk.mean(1), names, 4),
                            "model_top1": {n: names[int(np.argmax(mk[:, j]))] for j, n in enumerate(MODELS)}}
    L.add(fid, title, p, "B", "6", kind="class_heatmap", per_class=per_class_top)

    # B7 class-versus-rest contrasts
    contrast_facts = {}
    for n in MODELS:
        phi = D[f"phi_{n}"][s]
        C = phi - (phi.sum(2, keepdims=True) - phi) / (K - 1)            # phi_k - mean_{j!=k} phi_j
        if n in ("rf", "ada"):
            ratio = float(np.max(np.abs(C - phi * K / (K - 1))))
            contrast_facts[n] = {"reduces_to_scaled_class_shap": ratio < 1e-6, "max_dev": ratio}
        else:
            r = [spearman(np.abs(C[:, :, k]).mean(0), np.abs(phi[:, :, k]).mean(0)) for k in range(K)]
            contrast_facts[n] = {"reduces_to_scaled_class_shap": False, "rank_corr_with_class_shap": r}
    for n in ("xgb", "lgbm"):
        phi = D[f"phi_{n}"][s]
        C = phi - (phi.sum(2, keepdims=True) - phi) / (K - 1)
        for k in range(K):
            fid, title = L.next_title(f"{sc.MODEL_LABEL[n]}: class-versus-rest SHAP for {CLS_FULL[k]} "
                                      f"(log[p_k / geometric mean of other p])")
            p = L.path("02", f"{n}_{CLS_TAG[k]}_vs_rest_beeswarm")
            shap_fig(lambda: shap.plots.beeswarm(expl(C[:, :, k], X, names), max_display=F, show=False), title, p,
                     size=(9, 8), xlabel=f"SHAP contribution to log(p_{CLS[k]} / geo-mean p_rest)")
            ck, cc = np.abs(C[:, :, k]).mean(0), np.abs(phi[:, :, k]).mean(0)
            L.add(fid, title, p, "B", "7", model=n, cls=k, kind="vs_rest",
                  top5=top(ck / ck.sum(), names, 5), top5_class=top(cc / cc.sum(), names, 5),
                  rank_corr=spearman(ck, cc),
                  rho=[(names[j], spearman(X[:, j], C[:, j, k])) for j in np.argsort(-ck)[:3]])
    return contrast_facts


def section_C(ds, D, S, names, L, a):
    s = D["strat"]
    X, y = D["X"][s], D["y"][s]
    F = len(names)
    for n in MODELS:                                                        # C8 dependence
        phi = D[f"phi_{n}"][s]
        for k in range(K):
            c = S[n]["class"][:, k]
            feats = np.argsort(-c)[:3]
            fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
            facts = []
            ex_k = expl(phi[:, :, k], X, names)
            for ax, j in zip(axes, feats):
                inter = shap.utils.approximate_interactions(int(j), phi[:, :, k], X)
                partner = int(inter[0]) if inter[0] != j else int(inter[1])
                shap.plots.scatter(ex_k[:, int(j)], color=ex_k[:, partner], ax=ax, show=False)
                ax.set_ylabel(f"SHAP for {CLS[k]} output")
                facts.append((names[j], names[partner], float(S[n]["rho"][j, k]),
                              float(np.quantile(X[:, j], 0.05)), float(np.quantile(X[:, j], 0.95))))
            fid, title = L.next_title(f"{sc.MODEL_LABEL[n]}: dependence plots of the 3 most important features "
                                      f"for the {CLS_FULL[k]} output")
            fig.suptitle(title + f"  [{sc.OUTPUT_SCALE[n]}]; colour = feature with the strongest approximate interaction",
                         fontsize=10)
            plt.tight_layout()
            p = L.path("04", f"{n}_{CLS_TAG[k]}_dependence_top3")
            save(fig, p)
            L.add(fid, title, p, "C", "8", model=n, cls=k, kind="dependence", panels=facts,
                  scale=sc.OUTPUT_SCALE[n])

    inter_facts = {}
    for n in MODELS:                                                        # C9 interactions
        if n == "svm":
            inter_facts[n] = None
            continue
        iv = D[f"iv_{n}"]
        Xi = D["X"][D["int_sel"]]
        absiv = np.abs(iv).mean(0)                                          # (F, F, K)
        fig, axes = plt.subplots(1, K, figsize=(22, 5.6))
        fk = {}
        for k, ax in enumerate(axes):
            Mk = absiv[:, :, k].copy()
            main = np.diag(Mk).copy()
            off = Mk + Mk.T
            np.fill_diagonal(off, 0)
            tot = np.abs(iv[:, :, :, k]).sum(axis=(1, 2)).mean()
            offmass = (np.abs(iv[:, :, :, k]).sum(axis=(1, 2)) - np.abs(np.einsum("nii->ni", iv[:, :, :, k])).sum(1)).mean()
            sel = np.argsort(-(main + off.sum(1)))[:8]
            sub = off[np.ix_(sel, sel)]
            im = ax.imshow(sub, cmap=SEQ)
            ax.set_xticks(range(8), [names[i] for i in sel], rotation=90, fontsize=8)
            ax.set_yticks(range(8), [names[i] for i in sel], fontsize=8)
            ax.set_title(f"{CLS[k]} output", fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.7)
            iu = np.triu_indices(F, 1)
            o = np.argsort(-off[iu])[:3]
            fk[k] = {"offdiag_fraction": float(offmass / tot) if tot > 0 else 0.0,
                     "top_pairs": [(names[iu[0][t]], names[iu[1][t]], float(off[iu][t])) for t in o],
                     "top_main": top(main, names, 3)}
        fid, title = L.next_title(f"{sc.MODEL_LABEL[n]}: mean |SHAP interaction| between the top features, per class output")
        fig.suptitle(title + f"\n(path-dependent TreeSHAP, n={len(Xi)} stratified test pixels; cell = mean|phi_ij|+mean|phi_ji|, "
                     f"diagonal (main effects) set to 0)  [{sc.OUTPUT_SCALE[n]}]", fontsize=10)
        plt.tight_layout()
        p = L.path("04", f"{n}_interaction_matrices")
        save(fig, p)
        L.add(fid, title, p, "C", "9", model=n, kind="interaction_matrix", per_class=fk, n=int(len(Xi)))
        inter_facts[n] = fk

        k = 1                                                               # debris: strongest pair
        i, j = names.index(fk[k]["top_pairs"][0][0]), names.index(fk[k]["top_pairs"][0][1])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, (u, v) in zip(axes, [(i, j), (j, i)]):
            sca = ax.scatter(Xi[:, u], 2 * iv[:, u, v, k], c=Xi[:, v], cmap=shap.plots.colors.red_blue, s=10)
            ax.set_xlabel(names[u]); ax.set_ylabel(f"SHAP interaction {names[u]} x {names[v]} (x2)")
            plt.colorbar(sca, ax=ax).set_label(names[v])
            ax.axhline(0, color="#c3c2b7", lw=0.8)
        fid, title = L.next_title(f"{sc.MODEL_LABEL[n]}: strongest interaction pair for the Marine Debris output "
                                  f"({names[i]} x {names[j]})")
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        p = L.path("04", f"{n}_c2_debris_top_interaction_pair")
        save(fig, p)
        pr = 2 * iv[:, i, j, k]
        L.add(fid, title, p, "C", "9", model=n, cls=k, kind="interaction_pair", pair=(names[i], names[j]),
              rho_i=spearman(Xi[:, i], pr), rho_j=spearman(Xi[:, j], pr),
              range=(float(pr.min()), float(pr.max())), main_range=(float(iv[:, i, i, k].min()), float(iv[:, i, i, k].max())))

    # C10 distribution plots: normalised SHAP of top features per class across models
    for k in range(K):
        shares = np.stack([S[n]["class"][:, k] / S[n]["class"][:, k].sum() for n in MODELS], 1).mean(1)
        feats = np.argsort(-shares)[:6]
        fig, axes = plt.subplots(1, len(feats), figsize=(20, 4.8), sharey=True)
        spread = {}
        for ax, j in zip(axes, feats):
            data = []
            for n in MODELS:
                phi = D[f"phi_{n}"][s][:, :, k]
                norm = np.abs(phi).sum(1).mean()                             # mean total attribution mass
                data.append(phi[:, j] / norm)
            parts = ax.violinplot(data, showmedians=True, widths=0.85)
            for b, n in zip(parts["bodies"], MODELS):
                b.set_facecolor(MODEL_COLOR[n]); b.set_alpha(0.75); b.set_edgecolor("white")
            for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                parts[key].set_color("#52514e"); parts[key].set_linewidth(1)
            ax.axhline(0, color="#c3c2b7", lw=0.8)
            ax.set_xticks(range(1, len(MODELS) + 1), [sc.MODEL_LABEL[n].replace(" ", "\n") for n in MODELS], fontsize=7.5)
            ax.set_title(names[j], fontsize=10)
            spread[names[j]] = {n: (float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))) for n, d in zip(MODELS, data)}
        axes[0].set_ylabel(f"normalised SHAP for {CLS[k]} output\n(SHAP / mean total |SHAP| of that model)")
        fid, title = L.next_title(f"Distribution of normalised SHAP values of the 6 most important features for the "
                                  f"{CLS_FULL[k]} output, all models")
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()
        p = L.path("04", f"all_models_{CLS_TAG[k]}_shap_distribution")
        save(fig, p)
        L.add(fid, title, p, "C", "10", cls=k, kind="distribution", spread=spread)
    return inter_facts


def pick_exemplars(D, names):
    """Per class: a stratified-sample pixel correctly classified by ALL models, nearest to the class median."""
    s = np.flatnonzero(D["strat"])
    X, y = D["X"], D["y"]
    ok = np.all([D[f"pred_{n}"] == y for n in MODELS], 0)
    mu, sd = X[s].mean(0), X[s].std(0) + 1e-9
    ex = {}
    for k in range(K):
        cand = s[(y[s] == k) & ok[s]]
        if len(cand) == 0:
            continue
        z = (X[cand] - mu) / sd
        med = np.median(z, 0)
        ex[k] = int(cand[np.argmin(((z - med) ** 2).sum(1))])
    return ex


def confusion_pairs(meta, min_count):
    """(true, pred) pairs ordered by total test errors over the five models."""
    tot = np.zeros((K, K), int)
    per = {}
    for n in MODELS:
        cm = np.array(meta["models"][n]["confusion_matrix"])
        per[n] = cm
        tot += cm
    pairs = [(a, b, int(tot[a, b])) for a in range(K) for b in range(K) if a != b and tot[a, b] >= min_count]
    return sorted(pairs, key=lambda t: -t[2]), per


def section_D(ds, D, S, names, L, meta, a):
    X, y = D["X"], D["y"]
    ex = pick_exemplars(D, names)
    for k, i in ex.items():                                                 # D11 correct exemplars
        for n in MODELS:
            phi, base = D[f"phi_{n}"][i, :, k], D[f"base_{n}"][k]
            fid, title = L.next_title(f"{sc.MODEL_LABEL[n]}: waterfall for a correctly classified {CLS[k]} pixel "
                                      f"(explaining the {CLS[k]} output)")
            p = L.path("05", f"{n}_{CLS_TAG[k]}_correct_waterfall")
            shap_fig(lambda: shap.plots.waterfall(shap.Explanation(phi, base, X[i], feature_names=names),
                                                  max_display=12, show=False), title, p, size=(9, 6.5))
            order = np.argsort(-np.abs(phi))[:3]
            L.add(fid, title, p, "D", "11", model=n, cls=k, kind="waterfall_correct", test_index=int(D["idx"][i]),
                  base=float(base), fx=float(base + phi.sum()), top3=[(names[j], float(X[i, j]), float(phi[j])) for j in order],
                  scale=sc.OUTPUT_SCALE[n])

    pairs, per_cm = confusion_pairs(meta, a.min_pair_errors)
    D["_pairs"] = pairs
    rng = np.random.default_rng(a.seed + 2)
    chosen_mis = {}
    for n in MODELS:                                                        # D11 misclassified
        cm = per_cm[n].copy()
        np.fill_diagonal(cm, 0)
        top_pairs = [divmod(int(t), K) for t in np.argsort(-cm.ravel())[:2] if cm.ravel()[t] > 0]
        for (ta, pb) in top_pairs:
            cand = np.flatnonzero((y == ta) & (D[f"pred_{n}"] == pb))
            if len(cand) == 0:
                continue
            # most typical error: the candidate whose margin f_pb - f_ta is the median
            marg = D[f"f_{n}"][cand, pb] - D[f"f_{n}"][cand, ta]
            i = int(cand[np.argsort(marg)[len(cand) // 2]])
            chosen_mis[(n, ta, pb)] = i
            for k, role in [(pb, "predicted"), (ta, "true")]:
                phi, base = D[f"phi_{n}"][i, :, k], D[f"base_{n}"][k]
                fid, title = L.next_title(f"{sc.MODEL_LABEL[n]}: misclassified pixel (true {CLS[ta]}, predicted {CLS[pb]}), "
                                          f"explaining the {role}-class output ({CLS[k]})")
                p = L.path("05", f"{n}_true-{CLS_TAG[ta]}_pred-{CLS_TAG[pb]}_{role}-output_waterfall")
                shap_fig(lambda: shap.plots.waterfall(shap.Explanation(phi, base, X[i], feature_names=names),
                                                      max_display=12, show=False), title, p, size=(9, 6.5))
                order = np.argsort(-np.abs(phi))[:3]
                L.add(fid, title, p, "D", "11", model=n, cls=k, kind="waterfall_mis", role=role, true=ta, pred=pb,
                      test_index=int(D["idx"][i]), n_errors_test=int(per_cm[n][ta, pb]),
                      base=float(base), fx=float(base + phi.sum()),
                      top3=[(names[j], float(X[i, j]), float(phi[j])) for j in order], scale=sc.OUTPUT_SCALE[n])

    for n in MODELS:                                                        # D12 decision plot
        keys = [kk for kk in chosen_mis if kk[0] == n]
        deb = [kk for kk in keys if 1 in kk[1:]]
        key = (deb or keys or [None])[0]
        if key is None:
            continue
        i = chosen_mis[key]
        plt.close("all")
        shap.multioutput_decision_plot(list(D[f"base_{n}"]), [D[f"phi_{n}"][[i], :, k] for k in range(K)],
                                       row_index=0, feature_names=names, legend_labels=CLS,
                                       legend_location="lower right", show=False, feature_display_range=slice(None, -13, -1))
        fid, title = L.next_title(f"{sc.MODEL_LABEL[n]}: multi-output decision plot for the misclassified pixel "
                                  f"of its waterfalls (true {CLS[key[1]]}, predicted {CLS[key[2]]})")
        fig = plt.gcf()
        ax0 = fig.axes[0]
        # SHAP's lines carry no labels: identify each class line by its end point (= that class's model
        # output at the top of the plot) and give it a fixed, distinct class colour
        fx = D[f"f_{n}"][i]
        handles = []
        for line in ax0.get_lines():
            xs, ys = np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)
            if len(xs) < 3 or line.get_linestyle() != "-":
                continue
            kk = int(np.argmin(np.abs(fx - xs[np.argmax(ys)])))
            line.set_color(CLASS_COLOR[kk]); line.set_linewidth(2); line.set_label(CLS[kk])
            handles.append(line)
        handles.sort(key=lambda h: CLS.index(h.get_label()))
        ax0.legend(handles=handles, loc="lower right", fontsize=9)
        ax0.set_xlabel(f"model output [{sc.OUTPUT_SCALE[n]}]")
        fig.suptitle(title, fontsize=10, y=1.0)
        p = L.path("05", f"{n}_true-{CLS_TAG[key[1]]}_pred-{CLS_TAG[key[2]]}_decision_plot")
        save(fig, p)
        L.add(fid, title, p, "D", "12", model=n, kind="decision", true=key[1], pred=key[2],
              test_index=int(D["idx"][i]), outputs=[float(v) for v in D[f"f_{n}"][i]],
              base=[float(v) for v in D[f"base_{n}"]], scale=sc.OUTPUT_SCALE[n])

    # D13 confusion case studies
    tdir = os.path.join(L.root, SECTIONS["06"], "tables", ds)
    rows = []
    for (ta, pb, tot) in pairs[: a.max_case_pairs]:
        fig, axes = plt.subplots(1, len(MODELS), figsize=(23, 5.2))
        facts = {}
        for ax, n in zip(axes, MODELS):
            mis = np.flatnonzero((y == ta) & (D[f"pred_{n}"] == pb))
            cor = np.flatnonzero(D["strat"] & (y == ta) & (D[f"pred_{n}"] == ta))
            if len(mis) < a.min_case_n:
                ax.text(0.5, 0.5, f"{len(mis)} explained pixels\n(< {a.min_case_n}); not shown",
                        ha="center", va="center", transform=ax.transAxes, color="#52514e")
                ax.set_title(sc.MODEL_LABEL[n]); ax.set_axis_off()
                facts[n] = {"n_mis": int(len(mis)), "n_test_errors": int(per_cm[n][ta, pb])}
                continue
            dphi = D[f"phi_{n}"][:, :, pb] - D[f"phi_{n}"][:, :, ta]           # evidence for pred over true
            dm, dc = dphi[mis].mean(0), dphi[cor].mean(0)
            order = np.argsort(-np.abs(dm - dc))[:8][::-1]
            ax.barh(range(8), dm[order], color=MODEL_COLOR[n], height=0.6, label="misclassified")
            ax.scatter(dc[order], range(8), color="#0b0b0b", marker="D", s=22, zorder=3,
                       label=f"correct {CLS[ta]}")
            ax.axvline(0, color="#c3c2b7", lw=0.8)
            ax.set_yticks(range(8), [names[j] for j in order], fontsize=8)
            ax.set_title(f"{sc.MODEL_LABEL[n]} (n={len(mis)} of {per_cm[n][ta, pb]} test errors)", fontsize=9)
            ax.set_xlabel(f"mean SHAP({CLS[pb]}) - SHAP({CLS[ta]})", fontsize=8)
            ax.legend(fontsize=7, loc="lower right")
            facts[n] = {"n_mis": int(len(mis)), "n_test_errors": int(per_cm[n][ta, pb]),
                        "top_shift": [(names[j], float(dm[j]), float(dc[j])) for j in order[::-1][:4]],
                        "margin_mis_mean": float((D[f"f_{n}"][mis, pb] - D[f"f_{n}"][mis, ta]).mean())}
            for j, f in enumerate(names):
                rows.append({"true": CLS_FULL[ta], "pred": CLS_FULL[pb], "model": n, "feature": f,
                             "mean_dshap_misclassified": dm[j], "mean_dshap_correct_true": dc[j],
                             "n_mis_explained": len(mis), "n_test_errors": int(per_cm[n][ta, pb]),
                             "mean_value_misclassified": X[mis, j].mean(),
                             "mean_value_all_true_class_px": X[D["strat"] & (y == ta), j].mean(),
                             "mean_value_all_pred_class_px": X[D["strat"] & (y == pb), j].mean(),
                             "output_scale": sc.OUTPUT_SCALE[n]})
        fid, title = L.next_title(f"Confusion case study: true {CLS_FULL[ta]} predicted as {CLS_FULL[pb]} "
                                  f"({tot} test errors summed over the 5 models)")
        fig.suptitle(title + "\nbars: mean SHAP difference (predicted-class output minus true-class output) over the "
                     "misclassified pixels; diamonds: same quantity for correctly classified pixels of the true class",
                     fontsize=10)
        plt.tight_layout()
        p = L.path("05", f"case_true-{CLS_TAG[ta]}_pred-{CLS_TAG[pb]}")
        save(fig, p)
        # feature-value context for the most-shifted features
        L.add(fid, title, p, "D", "13", kind="case_study", true=ta, pred=pb, total=tot, per_model=facts)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(tdir, f"{ds}_confusion_case_studies.csv"), index=False)
    return pairs


# ======================================================================== main ==

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(REPO, "data"))
    ap.add_argument("--datasets", nargs="+", default=["MARIDA", "MADOS"])
    ap.add_argument("--out", default=os.path.join(REPO, "results", "shap_analysis"))
    ap.add_argument("--per-class", type=int, default=300, help="stratified test pixels per class")
    ap.add_argument("--err-per-pair", type=int, default=30, help="extra misclassified pixels per model and confusion pair")
    ap.add_argument("--bg-per-class", type=int, default=50, help="background training pixels per class")
    ap.add_argument("--int-per-class", type=int, default=40, help="pixels per class for interaction values")
    ap.add_argument("--kernel-check", type=int, default=40, help="pixels for the SVM KernelSHAP check")
    ap.add_argument("--min-pair-errors", type=int, default=20, help="min total test errors for a case-study pair")
    ap.add_argument("--max-case-pairs", type=int, default=10)
    ap.add_argument("--min-case-n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recompute", action="store_true", help="ignore cached SHAP values")
    a = ap.parse_args()
    configure(a.data_root)
    os.makedirs(a.out, exist_ok=True)
    for ds in a.datasets:
        t0 = time.time()
        print(f"\n===== {ds} =====", flush=True)
        D, val, names, meta = compute(ds, a, a.out)
        S, G, C, tdir = write_tables(ds, D, names, a.out)
        L = FigureLog(ds, a.out)
        section_A(ds, D, S, names, L)
        contrast = section_B(ds, D, S, names, L)
        inter = section_C(ds, D, S, names, L, a)
        pairs = section_D(ds, D, S, names, L, meta, a)
        val["contrast_check"] = contrast
        val["confusion_pairs"] = [(CLS_FULL[x], CLS_FULL[yy], t) for x, yy, t in pairs]
        val["test_metrics"] = {n: {k: meta["models"][n]["metrics"][k] for k in
                                   ["f1", "accuracy", "kappa", "f1_c1", "f1_c2", "f1_c3", "f1_c4",
                                    "recall_c1", "recall_c2", "recall_c3", "recall_c4"]} for n in MODELS}
        val["confusion_matrices"] = {n: meta["models"][n]["confusion_matrix"] for n in MODELS}
        val["n_figures"] = len(L.items)
        val["runtime_s"] = round(time.time() - t0, 1)
        with open(os.path.join(tdir, f"{ds}_validation.json"), "w") as f:
            json.dump(val, f, indent=2, default=float)
        with open(os.path.join(tdir, f"{ds}_figures.json"), "w") as f:
            json.dump(L.items, f, indent=1, default=float)
        print(f"  [{ds}] {len(L.items)} figures, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
