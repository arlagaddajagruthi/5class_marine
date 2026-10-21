"""
Step 1 of the MARIDA 26-feature study: full-image GLCM texture maps.

The deep models need the 6 GLCM features at every pixel of every patch, not only
at the annotated pixels used by the staged ML. This script runs the staged-ML
GLCM kernel (scripts/staged_ml/glcm_features.py, MARIDA reference settings:
B4/B3/B2 clipped to [0, 0.15], grey, 17 levels, 13 x 13 window with reflect
padding, distance 1, 0/45/90/135 deg averaged) on all 256 x 256 pixels.

The image is read exactly as the staged ML read it (MARIDADataset: NaN -> 0), so
the maps equal the ML GLCM features at the annotated pixels; `--verify` checks
this against the staged cache, pixel by pixel.

The 4-direction average makes the maps invariant to the training augmentation
(90-degree rotations and flips), so they can be computed once, before augmentation.

Usage (repo root):
  python scripts/marida_f26/glcm_maps.py --time 20      # time 20 patches, no output
  python scripts/marida_f26/glcm_maps.py                # all splits
  python scripts/marida_f26/glcm_maps.py --verify       # compare with the staged ML cache
  python scripts/marida_f26/glcm_maps.py --figure       # example maps for one test patch
Output: outputs/marida_f26/features/glcm/MARIDA_<split>_glcm6.npy
        float32 (n_patches, 6, 256, 256), patches in splits/<split>_X.txt order (~2.2 GB in total)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "staged_ml")]
from glcm_features import GLCM_NAMES, OFFSETS, LEVELS, WIN, _glcm_at, quantise  # noqa: E402
from src.datasets.marida import MARIDADataset  # noqa: E402

DATA = os.path.join(REPO, "data", "MARIDA_5Class")
OUT = os.path.join(REPO, "outputs", "marida_f26", "features", "glcm")
ML_CACHE = os.path.join(REPO, "outputs", "staged", "balanced_weights", "cache")
SPLITS = ["train", "val", "test"]
CLASSES = [1, 2, 3, 4]


def map_path(split):
    return os.path.join(OUT, f"MARIDA_{split}_glcm6.npy")


def glcm_map(img):
    """img: (11, H, W) reflectance, NaN already 0 -> (6, H, W) float32."""
    q = quantise(img)
    h, w = q.shape
    qpad = np.pad(q, WIN // 2, mode="reflect")
    rows, cols = np.divmod(np.arange(h * w, dtype=np.int64), w)
    g = _glcm_at(qpad, rows, cols, OFFSETS, LEVELS, WIN)          # (h*w, 6)
    return g.T.reshape(6, h, w).astype(np.float32)


def build(split, limit=None):
    ds = MARIDADataset(DATA, split=split)
    n = len(ds) if limit is None else min(limit, len(ds))
    path = map_path(split)
    if limit is None and os.path.exists(path):
        print(f"{split}: exists, skipped ({path})")
        return
    h, w = ds[0][1].shape
    G = None if limit else np.lib.format.open_memmap(path + ".tmp.npy", mode="w+", dtype=np.float32,
                                                      shape=(n, 6, h, w))
    t0 = time.time()
    for i in range(n):
        img, _ = ds[i]
        g = glcm_map(img.numpy())
        if G is not None:
            G[i] = g
        if (i + 1) % 50 == 0 or i + 1 == n:
            el = time.time() - t0
            print(f"  {split}: {i + 1}/{n} patches, {el:.0f} s ({el / (i + 1):.2f} s/patch)", flush=True)
    if G is not None:
        G.flush()
        del G
        os.replace(path + ".tmp.npy", path)
        print(f"{split}: {n} patches in {time.time() - t0:.0f} s -> {path}")
    return (time.time() - t0) / n


def verify():
    """Annotated pixels of the maps vs the staged ML cache (same pixel order as glcm_features.split_features)."""
    report = {}
    for split in SPLITS:
        ds = MARIDADataset(DATA, split=split)
        G = np.load(map_path(split), mmap_mode="r")
        rows = []
        for i in range(len(ds)):
            m = ds[i][1].numpy().ravel()
            sel = np.concatenate([np.flatnonzero(m == c) for c in CLASSES])
            if len(sel):
                rows.append(G[i].reshape(6, -1)[:, sel].T)
        ours = np.vstack(rows)
        ref = np.load(os.path.join(ML_CACHE, f"MARIDA_{split}_all_glcm6.npy"))
        if ours.shape != ref.shape:
            raise SystemExit(f"{split}: shape {ours.shape} vs staged cache {ref.shape}")
        diff = np.abs(ours - ref).max(axis=0)
        report[split] = {"pixels": int(len(ref)), **{n: float(d) for n, d in zip(GLCM_NAMES, diff)}}
        print(f"{split}: {len(ref):,} annotated pixels, max |map - ML cache| per feature: "
              + ", ".join(f"{n.replace('glcm_', '')} {d:.1e}" for n, d in zip(GLCM_NAMES, diff)))
    os.makedirs(OUT, exist_ok=True)
    json.dump(report, open(os.path.join(OUT, "verify.json"), "w"), indent=2)
    worst = max(max(v for k, v in r.items() if k != "pixels") for r in report.values())
    print(f"worst difference: {worst:.1e} -> {'OK' if worst < 1e-5 else 'MISMATCH'}")
    return worst


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = MARIDADataset(DATA, split="test")
    G = np.load(map_path("test"), mmap_mode="r")
    # the test patch with the most debris pixels
    idx = int(np.argmax([(ds[i][1].numpy() == 2).sum() for i in range(len(ds))]))
    img, m = (t.numpy() for t in ds[idx])
    rgb = np.stack([img[3], img[2], img[1]], -1)
    lo, hi = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    ax[0, 0].imshow(rgb); ax[0, 0].set_title(f"RGB (B4/B3/B2)\n{ds.patch_ids[idx]}")
    cmap = matplotlib.colors.ListedColormap(["#ffffff", "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"])
    ax[0, 1].imshow(m, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax[0, 1].set_title("labels (white unlabelled, blue water,\nred debris, green algae, orange other)")
    for k, a in enumerate(ax.ravel()[2:]):
        im = a.imshow(G[idx, k], cmap="viridis")
        a.set_title(GLCM_NAMES[k])
        fig.colorbar(im, ax=a, fraction=0.046)
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    path = os.path.join(OUT, "example_test_patch.png")
    plt.savefig(path, dpi=110)
    print("figure ->", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time", type=int, help="time N train patches without saving")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.time:
        s = build("train", limit=a.time)
        total = sum(len(MARIDADataset(DATA, split=sp)) for sp in SPLITS)
        print(f"{s:.2f} s/patch -> about {s * total / 3600:.1f} h for all {total} patches")
    elif a.verify:
        raise SystemExit(0 if verify() < 1e-5 else 1)
    elif a.figure:
        figure()
    else:
        for sp in a.splits:
            build(sp)
