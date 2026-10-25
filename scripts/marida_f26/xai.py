"""
Explainability for the 26-feature study (MARIDA or MADOS, `--dataset`): GradientSHAP (all 26 features and
the 4 feature groups, per class) and Seg-Grad-CAM figures.

What is explained (the same for every model)
  For class c and a test patch, the explained output is the class-c score of the
  patch: S_c = sum over the pixels labelled c of P(c). GradientSHAP gives every
  input value (channel k at pixel j) a share of S_c(x) - E_b[S_c(b)], where b are
  reference inputs (20 random training patches). Reported per class c:
    - pixel importance  mean over the class-c pixels of |attribution at that pixel|
                        (what the pixel's own features contribute; directly comparable
                        with the RF's TreeSHAP, whose output at a pixel depends only on
                        that pixel)
    - share             each feature's (group's) importance / the sum over features
    - direction         Spearman correlation between a feature's value and its SHAP
                        value at the class-c pixels (+: high values push towards c)
    - context share     deep models only: share of the total |attribution| that lies
                        on pixels that are NOT class c (how much the model uses the
                        surroundings)
  Group values sum the signed attributions of a group's features at each pixel
  before taking |.|, as SHAP values are additive.

Deep models (R1-R5): GradientSHAP implemented directly (expected gradients):
  attribution = mean over n samples of (x - b) * dS_c/dx at b + alpha (x - b),
  alpha ~ U(0, 1); one forward pass without TTA; inputs in the model's scaled space.
  Completeness is checked per patch: sum of attributions vs S_c(x) - mean S_c(b).
Random Forest (R6): shap.TreeExplainer (exact TreeSHAP of the class probabilities).
Ensemble (R7): w x (U-Net share) + (1 - w) x (RF share), an approximation that is
  labelled as such (the two members are explained by different estimators).
Seg-Grad-CAM (deep models): gradients of the class-c score w.r.t. the last feature
  map before the classifier (U-Net: decoder output; SegNeXt/MariNeXt: input of the
  1x1 classifier), on the fixed example patches.

Usage (repo root):
  python scripts/marida_f26/xai.py --run R1            # one deep model
  python scripts/marida_f26/xai.py --run R6_rf         # RF (needs: pip install shap)
  python scripts/marida_f26/xai.py --run R7_ensemble   # after its members
  (add --dataset MADOS for MADOS)
Outputs: outputs/<dataset>_f26/xai/<run>/
"""
import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import features26 as F  # noqa: E402

ROOT = F.STUDY
RUNS = os.path.join(ROOT, "runs")
XAI = os.path.join(ROOT, "xai")
NAMES = F.CLASS_NAMES
SHORT = ["water", "debris", "algae", "other"]
GROUPS = list(F.GROUPS)
GIDX = {g: [F.CHANNELS.index(c) for c in cs] for g, cs in F.GROUPS.items()}
N_BASELINES, N_SAMPLES, CHUNK = 20, 32, 4
MAX_PATCHES = {0: 60}           # water is in almost every patch: 60 patches are enough; other classes: all
MAX_PIXELS = 4000               # pixels per class kept for the beeswarm / dependence plots


# ------------------------------------------------------------------ helpers ---
def example_patches():
    path = os.path.join(ROOT, "example_patches.json")
    if not os.path.exists(path):
        import report
        picks, _ = report.example_patches(None)
        json.dump(picks, open(path, "w"))
    return json.load(open(path))


def group_attr(attr):
    """attr (..., 26) -> (..., 4) signed group sums."""
    return np.stack([attr[..., GIDX[g]].sum(-1) for g in GROUPS], -1)


def summarise(pix_attr, pix_vals, out, tag):
    """Per-class pixel importance, shares and directions for features and groups -> two CSVs."""
    feat_rows, group_rows = [], []
    for c in range(4):
        a, v = pix_attr.get(c), pix_vals.get(c)
        if a is None or len(a) == 0:
            continue
        imp = np.abs(a).mean(0)
        share = imp / imp.sum()
        for k, ch in enumerate(F.CHANNELS):
            rho = spearmanr(v[:, k], a[:, k]).statistic if np.std(v[:, k]) > 0 and np.std(a[:, k]) > 0 else np.nan
            feat_rows.append({"class": NAMES[c], "feature": ch, "group": F.GROUP_OF[ch], "mean_abs_shap": imp[k],
                              "share_%": 100 * share[k], "mean_signed_shap": a[:, k].mean(),
                              "direction_spearman": rho, "pixels": len(a)})
        ga = group_attr(a)
        gimp = np.abs(ga).mean(0)
        for k, g in enumerate(GROUPS):
            group_rows.append({"class": NAMES[c], "group": g, "mean_abs_shap": gimp[k],
                               "share_%": 100 * gimp[k] / gimp.sum(), "mean_signed_shap": ga[:, k].mean(),
                               "features": len(GIDX[g]), "pixels": len(a)})
    fr, gr = pd.DataFrame(feat_rows), pd.DataFrame(group_rows)
    fr.to_csv(os.path.join(out, f"importance_features{tag}.csv"), index=False)
    gr.to_csv(os.path.join(out, f"importance_groups{tag}.csv"), index=False)
    return fr, gr


# --------------------------------------------------------------- deep models ---
def load_deep(run):
    import torch
    import train
    cfg = json.load(open(os.path.join(RUNS, run, "run_config.json")))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = train.build_model(cfg["model"], "scratch", F.N_CH, cfg["classes"]).to(dev)
    net.load_state_dict(torch.load(os.path.join(RUNS, run, "model_best_val_f1.pth"), map_location=dev))
    net.eval()
    return net, cfg, dev


def gradient_shap(net, x, mask, c, baselines, dev, rng):
    """x (26, H, W) scaled, mask (H, W) bool -> attribution (26, H, W), completeness (sum attr, delta)."""
    import torch
    x = x.to(dev)
    m = torch.from_numpy(mask).to(dev)
    total = torch.zeros_like(x)
    idx = rng.integers(0, len(baselines), N_SAMPLES)
    alphas = torch.from_numpy(rng.random(N_SAMPLES).astype(np.float32)).to(dev)
    for s in range(0, N_SAMPLES, CHUNK):
        b = baselines[idx[s:s + CHUNK]].to(dev)
        al = alphas[s:s + CHUNK].view(-1, 1, 1, 1)
        xs = (b + al * (x[None] - b)).requires_grad_(True)
        p = torch.softmax(net(xs), 1)[:, c]
        score = (p * m).sum()
        g, = torch.autograd.grad(score, xs)
        total += ((x[None] - b) * g).sum(0)
    attr = total / N_SAMPLES
    with torch.no_grad():
        fx = (torch.softmax(net(x[None]), 1)[0, c] * m).sum().item()
        sb = {i: (torch.softmax(net(baselines[i:i + 1].to(dev)), 1)[0, c] * m).sum().item() for i in np.unique(idx)}
        fb = float(np.mean([sb[i] for i in idx]))           # weighted by how often each baseline was drawn
    return attr.detach().cpu().numpy(), float(attr.sum().item()), fx - fb


def seg_grad_cam(net, cfg, x, mask, c, dev):
    """Seg-Grad-CAM map (H, W) in [0, 1] for class c over the pixels in mask."""
    import torch
    import torch.nn.functional as Fn
    store = {}
    if cfg["model"] == "unet":
        layer = net.segmentation_head
        h = layer.register_forward_pre_hook(lambda mod, inp: store.__setitem__("A", inp[0]))
    else:
        layer = net.decode_head.conv_seg
        h = layer.register_forward_pre_hook(lambda mod, inp: store.__setitem__("A", inp[0]))
    logits = net(x[None].to(dev))
    h.remove()
    A = store["A"]
    m = torch.from_numpy(mask).to(dev)
    score = (logits[0, c] * m).sum()
    g, = torch.autograd.grad(score, A)
    wts = g.mean((2, 3), keepdim=True)
    cam = torch.relu((wts * A).sum(1, keepdim=True))
    cam = Fn.interpolate(cam, size=x.shape[1:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-12)
    return cam.detach().cpu().numpy()


def run_deep(run):
    import torch
    import train
    net, cfg, dev = load_deep(run)
    out = os.path.join(XAI, run)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    rng = np.random.default_rng(0)
    tr = train.RunDataset("train", False, cfg["bands"], 4)
    baselines = torch.stack([tr[i][0] for i in rng.choice(len(tr), N_BASELINES, replace=False)])
    te = train.RunDataset("test", False, cfg["bands"], 4)
    masks = [te.unscaled(i)[1] for i in range(len(te))]
    pix_attr, pix_vals, comp, ctx, patch_tot = {}, {}, [], [], []
    ex = example_patches()
    ex_maps = {}
    t0 = time.time()
    for c in range(4):
        cand = [i for i in range(len(te)) if (masks[i] == c + 1).any()]
        if c in MAX_PATCHES and len(cand) > MAX_PATCHES[c]:
            cand = sorted(rng.choice(cand, MAX_PATCHES[c], replace=False).tolist())
        keep = sorted(set(cand) | {i for i in ex if (masks[i] == c + 1).any()})
        A, V = [], []
        for i in keep:
            x, _ = te[i]
            mask = masks[i] == c + 1
            attr, s_attr, delta = gradient_shap(net, x, mask, c, baselines, dev, rng)
            if i in ex:
                ex_maps[(i, c)] = attr.astype(np.float16)
            if i not in cand:
                continue
            comp.append({"class": NAMES[c], "patch": te.ids[i], "sum_attr": s_attr, "delta_score": delta})
            absall = np.abs(attr).sum()
            ctx.append({"class": NAMES[c], "patch": te.ids[i],
                        "context_share_%": 100 * (1 - np.abs(attr[:, mask]).sum() / max(absall, 1e-12))})
            patch_tot.append({"class": NAMES[c], "patch": te.ids[i], **dict(zip(F.CHANNELS, attr.sum((1, 2))))})
            A.append(attr[:, mask].T)
            V.append(te.unscaled(i)[0][:, mask].T)
        A, V = np.vstack(A), np.vstack(V)
        if len(A) > MAX_PIXELS:
            sel = rng.choice(len(A), MAX_PIXELS, replace=False)
            A_s, V_s = A[sel], V[sel]
        else:
            A_s, V_s = A, V
        pix_attr[c], pix_vals[c] = A, V
        np.savez_compressed(os.path.join(out, f"pixels_{SHORT[c]}.npz"), shap=A_s, values=V_s)
        print(f"  {run} {NAMES[c]}: {len(cand)} patches, {len(A):,} pixels ({time.time() - t0:.0f} s)", flush=True)
    comp, ctx = pd.DataFrame(comp), pd.DataFrame(ctx)
    comp.to_csv(os.path.join(out, "completeness.csv"), index=False)
    ctx.to_csv(os.path.join(out, "context_share.csv"), index=False)
    pd.DataFrame(patch_tot).to_csv(os.path.join(out, "patch_totals.csv"), index=False)
    np.savez_compressed(os.path.join(out, "example_shap_maps.npz"),
                        **{f"p{i}_c{c}": a for (i, c), a in ex_maps.items()})
    fr, gr = summarise(pix_attr, pix_vals, out, "")
    # Seg-Grad-CAM on the example patches
    cams = {}
    for i in ex:
        x, _ = te[i]
        for c in range(4):
            if (masks[i] == c + 1).any():
                cams[(i, c)] = seg_grad_cam(net, cfg, x, masks[i] == c + 1, c, dev)
    np.savez_compressed(os.path.join(out, "seg_grad_cam.npz"), **{f"p{i}_c{c}": v for (i, c), v in cams.items()})
    figures(run, out, fr, gr, pix_attr, pix_vals, comp=comp, ctx=ctx, ex_maps=ex_maps, cams=cams, te=te,
            masks=masks)


# ------------------------------------------------------------------------ RF ---
def run_rf(run):
    import joblib
    try:
        import shap
    except ImportError:
        raise SystemExit("TreeSHAP needs the shap package: pip install shap")
    rf = joblib.load(os.path.join(RUNS, run, "rf.joblib"))
    out = os.path.join(XAI, run)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    import rf_ensemble
    X, y, pid, ds = rf_ensemble.pixels_in_dl_order("test")
    rng = np.random.default_rng(0)
    explainer = shap.TreeExplainer(rf)
    pix_attr, pix_vals = {}, {}
    t0 = time.time()
    for c in range(4):
        idx = np.flatnonzero(y == c)
        if len(idx) > MAX_PIXELS:
            idx = np.sort(rng.choice(idx, MAX_PIXELS, replace=False))
        sv = explainer.shap_values(X[idx], check_additivity=False)
        sv = sv[c] if isinstance(sv, list) else sv[..., c]
        pix_attr[c], pix_vals[c] = sv, X[idx]
        np.savez_compressed(os.path.join(out, f"pixels_{SHORT[c]}.npz"), shap=sv, values=X[idx])
        print(f"  {run} {NAMES[c]}: {len(idx):,} pixels ({time.time() - t0:.0f} s)", flush=True)
    fr, gr = summarise(pix_attr, pix_vals, out, "")
    figures(run, out, fr, gr, pix_attr, pix_vals)


# ------------------------------------------------------------------ ensemble ---
def run_ensemble(run):
    cfg = json.load(open(os.path.join(RUNS, run, "run_config.json")))
    w = cfg["w_unet"]
    unet, rf = cfg["members"]
    out = os.path.join(XAI, run)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    for tag in ("features", "groups"):
        key = "feature" if tag == "features" else "group"
        a = pd.read_csv(os.path.join(XAI, unet, f"importance_{tag}.csv"))
        b = pd.read_csv(os.path.join(XAI, rf, f"importance_{tag}.csv"))
        m = a.merge(b, on=["class", key], suffixes=("_unet", "_rf"))
        m["share_%"] = w * m["share_%_unet"] + (1 - w) * m["share_%_rf"]
        if tag == "features":
            m["group"] = m["group_unet"]
        m.to_csv(os.path.join(out, f"importance_{tag}.csv"), index=False)
    fr = pd.read_csv(os.path.join(out, "importance_features.csv"))
    gr = pd.read_csv(os.path.join(out, "importance_groups.csv"))
    fig_bars(fr, gr, out, f"{run} (approximation: {w:.2f} x {unet} share + {1 - w:.2f} x {rf} share)")
    lines = [f"# XAI, {run}", "", f"Approximation: the ensemble's feature shares are w × ({unet} share) + (1 − w) × "
             f"({rf} share) with w = {w:.2f}. The members are explained with different estimators "
             "(GradientSHAP, TreeSHAP), so these shares are indicative; see the members' reports for detail.", "",
             "![importance](figures/importance_by_class.png)", "", "![groups](figures/group_importance.png)", ""]
    lines += share_tables(fr, gr)
    open(os.path.join(out, "xai_report.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("->", os.path.join(out, "xai_report.md"))


# ------------------------------------------------------------------- figures ---
def fig_bars(fr, gr, out, title):
    classes = [n for n in NAMES if n in set(fr["class"])]
    fig, ax = plt.subplots(1, len(classes), figsize=(5 * len(classes), 7.5), sharey=False)
    ax = np.atleast_1d(ax)
    for a, c in zip(ax, classes):
        d = fr[fr["class"] == c].sort_values("share_%")
        a.barh(d.feature, d["share_%"], color=[F.GROUP_COLORS[F.GROUP_OF[f]] for f in d.feature])
        a.set_title(c)
        a.set_xlabel("share of |SHAP| (%)")
        a.tick_params(axis="y", labelsize=8)
    handles = [matplotlib.patches.Patch(color=v, label=k) for k, v in F.GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    fig.suptitle(f"Feature importance per class: {title}")
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    plt.savefig(os.path.join(out, "figures", "importance_by_class.png"), dpi=110)
    plt.close()
    piv = gr.pivot(index="group", columns="class", values="share_%").reindex(index=GROUPS, columns=classes)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xx = np.arange(len(classes))
    for k, g in enumerate(GROUPS):
        b = ax.bar(xx + (k - 1.5) * 0.2, piv.loc[g], 0.2, label=g, color=F.GROUP_COLORS[g])
        ax.bar_label(b, fmt="%.0f", fontsize=7)
    ax.set_xticks(xx, classes)
    ax.set_ylabel("share of |SHAP| (%)")
    ax.legend()
    ax.set_title(f"Grouped importance per class: {title}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "group_importance.png"), dpi=120)
    plt.close()
    # feature x class heatmap of shares
    hm = fr.pivot(index="feature", columns="class", values="share_%").reindex(index=F.CHANNELS, columns=classes)
    fig, ax = plt.subplots(figsize=(6, 9))
    im = ax.imshow(hm.values, cmap="viridis")
    ax.set_xticks(range(len(classes)), classes, rotation=20)
    ax.set_yticks(range(len(hm)), hm.index, fontsize=8)
    for t in ax.get_yticklabels():
        t.set_color(F.GROUP_COLORS[F.GROUP_OF[t.get_text()]])
    for i in range(len(hm)):
        for j in range(len(classes)):
            ax.text(j, i, f"{hm.values[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if hm.values[i, j] < np.nanmax(hm.values) * 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.05, label="share of |SHAP| (%)")
    ax.set_title("Feature share per class", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "importance_heatmap.png"), dpi=120)
    plt.close()


def fig_beeswarm(a, v, c, out, run, top=None):
    imp = np.abs(a).mean(0)
    order = np.argsort(imp)
    if top:
        order = order[-top:]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 0.33 * len(order) + 1.5))
    lo, hi = np.percentile(a[:, order], [0.5, 99.5])
    for r, k in enumerate(order):
        x = np.clip(a[:, k], lo, hi)
        # beeswarm-like vertical spread: wider where many points share similar SHAP values
        hist, edges = np.histogram(x, bins=60)
        dens = hist[np.clip(np.digitize(x, edges) - 1, 0, 59)] / max(hist.max(), 1)
        yy = r + (rng.random(len(x)) - 0.5) * 0.8 * dens
        col = (pd.Series(v[:, k]).rank(pct=True).values if np.std(v[:, k]) > 0 else np.full(len(x), 0.5))
        ax.scatter(x, yy, c=col, cmap="coolwarm", s=3, alpha=0.6, linewidths=0, vmin=0, vmax=1)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(order)), [F.CHANNELS[k] for k in order], fontsize=8)
    for t in ax.get_yticklabels():
        t.set_color(F.GROUP_COLORS[F.GROUP_OF[t.get_text()]])
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cb.set_ticks([0, 1], labels=["low", "high"])
    cb.set_label("feature value (rank)")
    ax.set_xlabel(f"SHAP value for {NAMES[c]} (positive = pushes towards {NAMES[c]})")
    ax.set_title(f"{run}: SHAP beeswarm, {NAMES[c]} ({len(a):,} pixels; features by importance)")
    plt.tight_layout()
    f = f"beeswarm_{SHORT[c]}.png"
    plt.savefig(os.path.join(out, "figures", f), dpi=120)
    plt.close()
    return f


def fig_dependence(a, v, c, out, run):
    imp = np.abs(a).mean(0)
    top = np.argsort(-imp)[:4]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for j, k in enumerate(top):
        lo, hi = np.percentile(v[:, k], [0.5, 99.5])
        keep = (v[:, k] >= lo) & (v[:, k] <= hi)
        ax[j].scatter(v[keep, k], a[keep, k], s=3, alpha=0.4, color=F.GROUP_COLORS[F.GROUP_OF[F.CHANNELS[k]]])
        ax[j].axhline(0, color="grey", lw=0.8)
        ax[j].set_xlabel(f"{F.CHANNELS[k]} (unscaled value)")
        ax[j].set_ylabel("SHAP value")
        ax[j].set_title(F.CHANNELS[k])
    fig.suptitle(f"{run}: dependence of the SHAP value on the feature value, top 4 features for {NAMES[c]}")
    plt.tight_layout()
    f = f"dependence_{SHORT[c]}.png"
    plt.savefig(os.path.join(out, "figures", f), dpi=110)
    plt.close()
    return f


def fig_maps(run, out, ex_maps, cams, te, masks):
    files = []
    for c in range(4):
        pats = sorted({i for (i, cc) in ex_maps if cc == c})
        if not pats:
            continue
        rows = pats[:3]
        fig, ax = plt.subplots(len(rows), 7, figsize=(24, 3.6 * len(rows)))
        ax = np.atleast_2d(ax)
        for r, i in enumerate(rows):
            img = te.unscaled(i)[0][:11]
            rgb = np.stack([img[3], img[2], img[1]], -1)
            lo, hi = np.percentile(rgb, [2, 98])
            rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
            ax[r, 0].imshow(rgb)
            ax[r, 0].contour(masks[i] == c + 1, levels=[0.5], colors="yellow", linewidths=0.8)
            ax[r, 0].set_title(f"{te.ids[i]}\n(yellow: {NAMES[c]} pixels)", fontsize=8)
            ga = group_attr(np.moveaxis(ex_maps[(i, c)].astype(np.float32), 0, -1))
            lim = np.percentile(np.abs(ga), 99.5) + 1e-12
            for k, g in enumerate(GROUPS):
                im = ax[r, k + 1].imshow(ga[..., k], cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax[r, k + 1].set_title(f"SHAP, {g}", fontsize=9)
            fig.colorbar(im, ax=ax[r, 4], fraction=0.046)
            tot = ga.sum(-1)
            ax[r, 5].imshow(tot, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[r, 5].set_title("SHAP, all 26 features", fontsize=9)
            if (i, c) in cams:
                ax[r, 6].imshow(rgb)
                ax[r, 6].imshow(cams[(i, c)], cmap="jet", alpha=0.45, vmin=0, vmax=1)
                ax[r, 6].set_title("Seg-Grad-CAM", fontsize=9)
            for a in ax[r]:
                a.axis("off")
        fig.suptitle(f"{run}: where the {NAMES[c]} score comes from (red: pushes towards {NAMES[c]}, "
                     "blue: away)")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        f = f"maps_{SHORT[c]}.png"
        plt.savefig(os.path.join(out, "figures", f), dpi=90)
        plt.close()
        files.append(f)
    return files


def share_tables(fr, gr):
    lines = ["## Grouped importance (share of |SHAP|, %)", ""]
    classes = [n for n in NAMES if n in set(gr["class"])]
    piv = gr.pivot(index="group", columns="class", values="share_%").reindex(index=GROUPS, columns=classes)
    lines += ["| group | " + " | ".join(classes) + " |", "|---" * (len(classes) + 1) + "|"]
    lines += [f"| {g} | " + " | ".join(f"{v:.1f}" for v in piv.loc[g]) + " |" for g in GROUPS]
    lines += ["", "## Feature importance (share of |SHAP|, %; direction = Spearman of value vs SHAP)", ""]
    piv = fr.pivot(index="feature", columns="class", values="share_%").reindex(index=F.CHANNELS, columns=classes)
    has_dir = "direction_spearman" in fr
    if has_dir:
        dr = fr.pivot(index="feature", columns="class", values="direction_spearman").reindex(index=F.CHANNELS,
                                                                                            columns=classes)
    lines += ["| feature | group | " + " | ".join(classes) + " |", "|---" * (len(classes) + 2) + "|"]
    for f in F.CHANNELS:
        cells = [f"{piv.loc[f, c]:.1f}" + (f" ({dr.loc[f, c]:+.2f})" if has_dir and not pd.isna(dr.loc[f, c]) else "")
                 for c in classes]
        lines.append(f"| {f} | {F.GROUP_OF[f]} | " + " | ".join(cells) + " |")
    return lines + [""]


def figures(run, out, fr, gr, pix_attr, pix_vals, comp=None, ctx=None, ex_maps=None, cams=None, te=None,
            masks=None):
    fig_bars(fr, gr, out, run)
    lines = [f"# XAI, run {run}", "",
             "Explained output: for class c, the sum of P(c) over the pixels labelled c in a test patch. "
             "Importance = mean |SHAP| at the class-c pixels; share = importance / sum over features. "
             "Direction (in brackets) = Spearman correlation between the feature value and its SHAP value at the "
             "class-c pixels: + means high values push towards the class. Group values add the SHAP values of a "
             "group's features at each pixel first. Correlated features share credit (see "
             "`features/correlation/`), so read single features together with the grouped results.", "",
             "![importance](figures/importance_by_class.png)", "", "![groups](figures/group_importance.png)", "",
             "![heatmap](figures/importance_heatmap.png)", ""]
    lines += share_tables(fr, gr)
    if comp is not None and len(comp):
        r = np.corrcoef(comp.sum_attr, comp.delta_score)[0, 1]
        rel = np.median(np.abs(comp.sum_attr - comp.delta_score) / np.maximum(np.abs(comp.delta_score), 1e-6))
        lines += ["## Completeness check (GradientSHAP)", "",
                  f"Sum of attributions vs S_c(x) − mean S_c(baseline) over {len(comp)} patch × class cases: "
                  f"correlation {r:.3f}, median relative error {100 * rel:.1f} % ({N_SAMPLES} samples, "
                  f"{N_BASELINES} baselines). `completeness.csv` has every case.", ""]
    if ctx is not None and len(ctx):
        cs = ctx.groupby("class")["context_share_%"].median().reindex(NAMES).dropna()
        lines += ["## Context share", "", "Median share of |attribution| that lies on pixels that are not the "
                  "explained class (the model using the surroundings):", "", "| class | context share (%) |",
                  "|---|---|"] + [f"| {c} | {v:.1f} |" for c, v in cs.items()] + [""]
    for c in range(4):
        if c in pix_attr and len(pix_attr[c]):
            a, v = pix_attr[c], pix_vals[c]
            if len(a) > MAX_PIXELS:
                sel = np.random.default_rng(1).choice(len(a), MAX_PIXELS, replace=False)
                a, v = a[sel], v[sel]
            fb, fd = fig_beeswarm(a, v, c, out, run), fig_dependence(a, v, c, out, run)
            lines += [f"## {NAMES[c]}", "", f"![beeswarm](figures/{fb})", "", f"![dependence](figures/{fd})", ""]
    if ex_maps:
        lines += ["## SHAP maps and Seg-Grad-CAM on example test patches", ""]
        lines += [f"![maps](figures/{f})" for f in fig_maps(run, out, ex_maps, cams, te, masks)]
    open(os.path.join(out, "xai_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("->", os.path.join(out, "xai_report.md"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    from runs import name
    a.run = name(a.run)
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = json.load(open(os.path.join(RUNS, a.run, "run_config.json")))
    {"rf": run_rf, "ensemble": run_ensemble}.get(cfg["model"], run_deep)(a.run)
