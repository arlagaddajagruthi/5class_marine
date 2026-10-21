"""
Step 2 of the MARIDA 26-feature study: the 26 input channels, their scaling, and
the feature-correlation analysis.

The 26 channels are every staged-ML feature except the 4 spectral statistics, in
this order, identical to the staged-ML features at every pixel:
  bands (11)    B1 B2 B3 B4 B5 B6 B7 B8 B8A B11 B12
  indices (5)   NDVI, NDWI, FAI (Hu 2009), FDI (Biermann et al. 2020), NDMI
  ratios (4)    red/green (B4/B3), nir/red (B8/B4), nir/green (B8/B3), swir1/nir (B11/B8)
  GLCM (6)      contrast, dissimilarity, homogeneity, energy, correlation, ASM

How a patch is turned into the model input
  1. read the 11 Rayleigh-corrected bands (MARIDADataset, NaN -> 0, as the staged ML;
     only 7 of 1,381 patches contain NaN)
  2. the 5 indices and 4 ratios: src/features/pixel_features.extract_pixel_features at
     every pixel (the function the staged ML used)
  3. the 6 GLCM maps: precomputed by glcm_maps.py (identical to the ML GLCM)
  4. scaling, with statistics from the train split only:
       bands, "raw"  -> reflectance unchanged (SSL4EO pretraining scale)
       bands, "std"  -> (x - mean) / std with the MARIDA paper's band statistics
       other 15      -> clipped to the train 1st-99th percentile, then (x - mean) / std
  5. training only: the 26 channels and the label are rotated (-90/0/90/180 deg)
     and flipped together, as in the MARIDA U-Net dataloader. Every feature is
     per pixel or averaged over the 4 GLCM directions, so these transforms do not
     change the feature values.

Usage (repo root):
  python scripts/marida_f26/features26.py stats        # train statistics -> channel_stats.json, channel_spec.md
  python scripts/marida_f26/features26.py verify       # features == staged-ML features at every annotated pixel
  python scripts/marida_f26/features26.py correlation  # correlation matrices, heatmaps, per-class distributions
Outputs: outputs/marida_f26/features/
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "resnet_paper")]
from src.datasets.marida import MARIDADataset  # noqa: E402
from src.features.pixel_features import FEATURE_NAMES, extract_pixel_features  # noqa: E402

DATA = os.path.join(REPO, "data", "MARIDA_5Class")
OUT = os.path.join(REPO, "outputs", "marida_f26", "features")
GLCM_DIR = os.path.join(OUT, "glcm")
ML_CACHE = os.path.join(REPO, "outputs", "staged", "balanced_weights", "cache")

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
INDICES = ["NDVI", "NDWI", "FAI", "FDI", "NDMI"]
RATIOS = ["red/green", "nir/red", "nir/green", "swir1/nir"]
GLCM_NAMES = ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy",
              "glcm_correlation", "glcm_asm"]
CHANNELS = BANDS + INDICES + RATIOS + GLCM_NAMES                      # 26
SPECTRAL = BANDS + INDICES + RATIOS                                   # the 20 from extract_pixel_features
SPECTRAL_IDX = [FEATURE_NAMES.index(c) for c in SPECTRAL]
GROUPS = {"bands": BANDS, "indices": INDICES, "ratios": RATIOS, "GLCM": GLCM_NAMES}
GROUP_OF = {c: g for g, cs in GROUPS.items() for c in cs}
GROUP_COLORS = {"bands": "#1f77b4", "indices": "#2ca02c", "ratios": "#d62728", "GLCM": "#ff7f0e"}
N_CH, N_BANDS = len(CHANNELS), len(BANDS)
FORMULA = {
    "B1": "coastal aerosol, 443 nm", "B2": "blue, 490 nm", "B3": "green, 560 nm", "B4": "red, 665 nm",
    "B5": "red edge 1, 705 nm", "B6": "red edge 2, 740 nm", "B7": "red edge 3, 783 nm", "B8": "NIR, 842 nm",
    "B8A": "narrow NIR, 865 nm", "B11": "SWIR 1, 1610 nm", "B12": "SWIR 2, 2190 nm",
    "NDVI": "(B8 - B4) / (B8 + B4)", "NDWI": "(B3 - B8) / (B3 + B8)",
    "FAI": "B8 - [B4 + (B11 - B4)(842 - 665)/(1610 - 665)]  (Hu 2009)",
    "FDI": "B8 - [B6 + (B11 - B6)(842 - 665)/(1610 - 665) x 10]  (Biermann et al. 2020)",
    "NDMI": "(B8 - B11) / (B8 + B11)",
    "red/green": "B4 / B3", "nir/red": "B8 / B4", "nir/green": "B8 / B3", "swir1/nir": "B11 / B8",
    **{g: f"GLCM {g[5:]}, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged"
       for g in GLCM_NAMES},
}
# MARIDA paper band statistics (marine-debris.github.io), as used by every earlier MARIDA model
BAND_MEAN = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
                      0.03875053, 0.03436435, 0.0392113, 0.02358126, 0.01588816], np.float32)
BAND_STD = np.array([0.04725893, 0.04743808, 0.04699043, 0.04967381, 0.04946782, 0.06458357,
                     0.07594915, 0.07120246, 0.08251058, 0.05111466, 0.03524419], np.float32)
SPLITS = ["train", "val", "test"]
CLASS_NAMES = ["Water", "Marine Debris", "Algae/Sargassum", "Other"]
IDENTITY_STATS = {"p1": [-np.inf] * N_CH, "p99": [np.inf] * N_CH, "mean": [0.0] * N_CH, "std": [1.0] * N_CH}


def stats_path():
    return os.path.join(OUT, "channel_stats.json")


def patch_features(img, glcm):
    """img (11, H, W) reflectance with NaN -> 0, glcm (6, H, W) -> (26, H, W) unscaled float32."""
    h, w = img.shape[1:]
    f24 = extract_pixel_features(img.reshape(11, h * w, 1))            # (h*w, 24), the staged-ML features
    spec = f24[:, SPECTRAL_IDX].T.reshape(len(SPECTRAL), h, w)
    return np.concatenate([spec, glcm], 0).astype(np.float32)


class MaridaF26(Dataset):
    """MARIDA patches as (26, H, W) scaled channels + target (classes 1-4 -> 0-3, unlabelled -> -1)."""

    def __init__(self, split, train, band_scaling, stats=None):
        assert band_scaling in ("raw", "std")
        self.ds = MARIDADataset(DATA, split=split)
        self.ids = self.ds.patch_ids
        self.train, self.band_scaling = train, band_scaling
        self.glcm = np.load(os.path.join(GLCM_DIR, f"MARIDA_{split}_glcm6.npy"), mmap_mode="r")
        st = stats or json.load(open(stats_path()))
        self.lo, self.hi, self.mu, self.sd = (np.array(st[k], np.float32)[:, None, None]
                                              for k in ("p1", "p99", "mean", "std"))
        if train:
            from resnet_ml import RandomRotationTransform
            from torchvision import transforms
            self.aug = transforms.Compose([RandomRotationTransform([-90, 0, 90, 180]),
                                           transforms.RandomHorizontalFlip()])

    def __len__(self):
        return len(self.ids)

    def unscaled(self, i):
        img, mask = self.ds[i]
        return patch_features(img.numpy(), np.asarray(self.glcm[i])), mask.numpy()

    def scale(self, x):
        x = x.copy()
        if self.band_scaling == "std":
            x[:N_BANDS] = (x[:N_BANDS] - BAND_MEAN[:, None, None]) / BAND_STD[:, None, None]
        x[N_BANDS:] = (np.clip(x[N_BANDS:], self.lo[N_BANDS:], self.hi[N_BANDS:]) - self.mu[N_BANDS:]) \
            / self.sd[N_BANDS:]
        return x

    def __getitem__(self, i):
        x, mask = self.unscaled(i)
        x = torch.from_numpy(self.scale(x))
        target = torch.from_numpy(mask.astype(np.int64) - 1)
        if self.train:
            stack = self.aug(torch.cat([x, target[None].float()]))
            x, target = stack[:-1], stack[-1].round().long()
        return x.float(), target


def cmd_stats():
    """Train-split percentiles, mean and std of every channel (2000 random pixels per patch, all pixels)."""
    ds = MaridaF26("train", train=False, band_scaling="raw", stats=IDENTITY_STATS)
    rng = np.random.default_rng(0)
    samples = []
    for i in range(len(ds)):
        x, _ = ds.unscaled(i)
        x = x.reshape(N_CH, -1)
        samples.append(x[:, rng.choice(x.shape[1], 2000, replace=False)])
    a = np.concatenate(samples, axis=1)
    p1, p99 = np.percentile(a, 1, axis=1), np.percentile(a, 99, axis=1)
    c = np.clip(a, p1[:, None], p99[:, None])
    mean, std = c.mean(1), c.std(1)
    # a channel whose 1st and 99th percentiles coincide would collapse to a constant: keep its full range
    flat = (p99 - p1) < 1e-6
    p1[flat], p99[flat] = a[flat].min(1), a[flat].max(1)
    mean[flat], std[flat] = a[flat].mean(1), a[flat].std(1)
    std = np.maximum(std, 1e-6)
    st = {"channels": CHANNELS, "p1": p1.tolist(), "p99": p99.tolist(), "mean": mean.tolist(), "std": std.tolist(),
          "unclipped": [CHANNELS[k] for k in np.flatnonzero(flat)],
          "note": "train split, 2000 random pixels per patch (all pixels, annotated or not); bands are not "
                  "clipped (raw: unchanged; std: MARIDA paper mean/std)"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(st, open(stats_path(), "w"), indent=2)

    lines = ["# MARIDA 26-channel input specification", "",
             "The 26 channels are every staged-ML feature except the 4 spectral statistics, in the model's "
             "input order. Every value is "
             "computed per pixel from the 11 Rayleigh-corrected bands (NaN → 0, as in the staged ML; 7 of 1,381 "
             "patches contain NaN), with the same code as the staged ML, so the features equal the RF's "
             "features (checked in `verify_features.json`).", "",
             "Scaling (train-split statistics only):", "",
             "- **bands, raw** (U-Net with SSL4EO weights: R1, G2b): reflectance unchanged, the scale the "
             "SSL4EO encoder was pretrained on (L1C / 10000).",
             "- **bands, std** (R2, R3, SegNeXt-T, MariNeXt arch.): (x − mean) / std with the MARIDA paper's "
             "band statistics.",
             "- **the 15 other channels** (all models): clipped to the train 1st–99th percentile, then "
             "(x − mean) / std of the clipped values"
             + (f"; {', '.join(st['unclipped'])} keep their full range because their 1st and 99th "
                "percentiles are equal." if st["unclipped"] else "."), "",
             "Augmentation (training only): the 26 channels and the label are rotated by −90/0/90/180° and "
             "flipped horizontally together. Every feature is per pixel or averaged over the 4 GLCM "
             "directions, so these transforms do not change the feature values.", "",
             "| # | Channel | Group | Definition | 1st pct | 99th pct | mean | std |",
             "|---|---|---|---|---|---|---|---|"]
    for k, c in enumerate(CHANNELS):
        lines.append(f"| {k + 1} | {c} | {GROUP_OF[c]} | {FORMULA[c]} | {p1[k]:.4g} | {p99[k]:.4g} | "
                     f"{mean[k]:.4g} | {std[k]:.4g} |")
    open(os.path.join(OUT, "channel_spec.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def ml_features(split):
    """The 26 staged-ML features of every annotated pixel of a split (staged caches), and the labels 1-4."""
    z = np.load(os.path.join(ML_CACHE, f"MARIDA_{split}_all_seed42_f24.npz"))
    g = np.load(os.path.join(ML_CACHE, f"MARIDA_{split}_all_glcm6.npy"))
    return np.hstack([z["X"][:, SPECTRAL_IDX], g]), z["y"]


def cmd_verify():
    """The 26 unscaled features at the annotated pixels vs the staged-ML caches."""
    res = {}
    for split in SPLITS:
        ds = MaridaF26(split, train=False, band_scaling="raw", stats=IDENTITY_STATS)
        rows = []
        for i in range(len(ds)):
            x, m = ds.unscaled(i)
            m = m.ravel()
            sel = np.concatenate([np.flatnonzero(m == c) for c in (1, 2, 3, 4)])
            if len(sel):
                rows.append(x.reshape(N_CH, -1)[:, sel].T)
        ours, (ref, _) = np.vstack(rows), ml_features(split)
        d = np.abs(ours - ref).max(0)
        res[split] = {"pixels": len(ref), "max_abs_diff": dict(zip(CHANNELS, d.tolist()))}
        print(f"{split}: {len(ref):,} pixels, max |ours - ML| over the 26 features = {d.max():.2e}")
    json.dump(res, open(os.path.join(OUT, "verify_features.json"), "w"), indent=2)


def cmd_correlation():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    out = os.path.join(OUT, "correlation")
    os.makedirs(out, exist_ok=True)
    X, y = ml_features("train")
    df = pd.DataFrame(X, columns=CHANNELS)

    def heatmap(c, title, path, order):
        c = c.iloc[order, order]
        fig, ax = plt.subplots(figsize=(11, 9.5))
        im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=9)
        ax.set_yticks(range(len(c)), c.index, fontsize=9)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_color(GROUP_COLORS[GROUP_OF[t.get_text()]])
        for i in range(len(c)):
            for j in range(len(c)):
                v = c.values[i, j]
                if i != j:
                    ax.text(j, i, f"{v:.2f}".replace("0.", ".", 1), ha="center", va="center", fontsize=5.5,
                            color="white" if abs(v) > 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.04)
        handles = [matplotlib.patches.Patch(color=v, label=k) for k, v in GROUP_COLORS.items()]
        ax.legend(handles=handles, bbox_to_anchor=(1.15, 1.0), loc="upper left", fontsize=8, title="group")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close()

    summary = ["# Feature correlation, MARIDA train split (annotated pixels)", "",
               f"{len(df):,} pixels with the 26 features. Heatmaps are ordered by hierarchical "
               "clustering on 1 − |Spearman ρ| (the all-classes order is used for every heatmap, so they can be "
               "compared), tick colours show the feature group, and every cell shows ρ. Strongly correlated "
               "features carry largely the same information, so SHAP can split credit between them; read "
               "feature importance together with these clusters and with the grouped results.", ""]
    order = None
    for method in ("spearman", "pearson"):
        for cls in [None, 1, 2, 3, 4]:
            sub = df if cls is None else df[y == cls]
            name = "all classes" if cls is None else CLASS_NAMES[cls - 1]
            c = sub.corr(method=method).fillna(0.0)
            tag = "all" if cls is None else CLASS_NAMES[cls - 1].split("/")[0].replace(" ", "_").lower()
            c.to_csv(os.path.join(out, f"{method}_{tag}.csv"))
            if order is None:
                order = leaves_list(linkage(squareform(1 - np.abs(c.values), checks=False), "average"))
            if method == "spearman":
                pairs = [(c.columns[i], c.columns[j], c.values[i, j]) for i in range(N_CH)
                         for j in range(i + 1, N_CH) if abs(c.values[i, j]) >= 0.9]
                summary += [f"## {name}: {len(pairs)} feature pairs with |Spearman ρ| ≥ 0.90", ""]
                if pairs:
                    summary += ["| Feature | Feature | ρ |", "|---|---|---|"]
                    summary += [f"| {a} | {b} | {v:+.3f} |" for a, b, v in sorted(pairs, key=lambda t: -abs(t[2]))]
                summary += [""]
            heatmap(c, f"{method.capitalize()} correlation, {name} ({len(sub):,} train pixels)",
                    os.path.join(out, f"{method}_{tag}.png"), order)
    # group-level: mean |Spearman| between and within groups
    c = df.corr(method="spearman").abs()
    gnames = list(GROUPS)
    gm = pd.DataFrame([[c.loc[GROUPS[a], GROUPS[b]].values[~np.eye(len(GROUPS[a]), dtype=bool)].mean()
                        if a == b and len(GROUPS[a]) > 1 else
                        (np.nan if a == b else c.loc[GROUPS[a], GROUPS[b]].values.mean())
                        for b in gnames] for a in gnames], index=gnames, columns=gnames)
    gm.to_csv(os.path.join(out, "spearman_groups.csv"))
    summary += ["## Grouped: mean |Spearman ρ| within and between groups (all classes)", "",
                "| | " + " | ".join(gnames) + " |", "|---" * (len(gnames) + 1) + "|"]
    summary += [f"| {a} | " + " | ".join("–" if np.isnan(v) else f"{v:.2f}" for v in gm.loc[a]) + " |"
                for a in gnames]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(gm.values, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(4), gnames)
    ax.set_yticks(range(4), gnames)
    for i in range(4):
        for j in range(4):
            v = gm.values[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Mean |Spearman ρ| within / between groups")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "spearman_groups.png"), dpi=130)
    plt.close()
    summary += ["", "## Files", "",
                "- `spearman_<class>.png/.csv`, `pearson_<class>.png/.csv`: all classes and each class "
                "(water, marine_debris, algae, other)",
                "- `spearman_groups.png/.csv`: group-level correlation",
                "- `distributions.png`: per-class distribution of every feature", ""]
    open(os.path.join(out, "correlation_summary.md"), "w", encoding="utf-8").write("\n".join(summary))

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(6, 5, figsize=(20, 23))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    for k, ax in enumerate(axes.ravel()):
        if k >= N_CH:
            ax.axis("off")
            continue
        c = CHANNELS[k]
        v = X[:, k]
        lo, hi = np.percentile(v, [1, 99])
        data = []
        for cls in range(1, 5):
            vc = v[y == cls]
            vc = vc[rng.choice(len(vc), min(len(vc), 20000), replace=False)]
            data.append(np.clip(vc, lo, hi))
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for b, col in zip(parts["bodies"], colors):
            b.set_facecolor(col)
            b.set_alpha(0.6)
        ax.set_xticks(range(1, 5), ["water", "debris", "algae", "other"], fontsize=9)
        ax.set_title(f"{c} ({GROUP_OF[c]})", fontsize=11, color=GROUP_COLORS[GROUP_OF[c]])
    fig.suptitle("Per-class distribution of each feature, MARIDA train (clipped to the 1st-99th percentile; "
                 "line = median)", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(os.path.join(out, "distributions.png"), dpi=100)
    plt.close()
    print("\n".join(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["stats", "verify", "correlation"])
    sys.stdout.reconfigure(encoding="utf-8")
    {"stats": cmd_stats, "verify": cmd_verify, "correlation": cmd_correlation}[ap.parse_args().command]()
