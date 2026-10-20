"""
GLCM texture features for every cached pixel, computed exactly as in the MARIDA
reference code (marine-debris.github.io, semantic_segmentation/random_forest/
engineering_patches.py, texture()), which the MARIDA and MADOS papers use for
their RF_SS+SI+GLCM / RF++ baselines:

  - RGB composite of the Rayleigh-corrected B4, B3, B2, clipped to [0, 0.15]
    and divided by 0.15
  - skimage rgb2gray (0.2125 R + 0.7154 G + 0.0721 B)
  - quantised with np.digitize(gray, linspace(0, 1, 16)) -> 17 grey levels
  - 13 x 13 window centred on the pixel, image padded by reflection
  - co-occurrence at distance 1 in 4 directions (0, 45, 90, 135 deg),
    symmetric and normalised per direction, then averaged over directions
  - the 6 skimage graycoprops: contrast, dissimilarity, homogeneity, energy,
    correlation, ASM

The reference loops over every pixel with skimage; here a numba kernel computes
the same matrices only at the annotated pixels held in the staged pixel cache
(common.sample_split, no cap), in the same order. The labels are checked
against the cache, so the columns line up with the cached features.
`python glcm_features.py --check` compares the kernel with skimage.

Usage (repo root): python scripts/staged_ml/glcm_features.py [--datasets MARIDA MADOS] [--splits train test val]
Output: outputs/staged/cache/<dataset>_<split>_all_glcm6.npy  (float32, n_pixels x 6)
"""
import argparse
import os
import time

import numpy as np
from numba import njit, prange

from common import CACHE_DIR, CLASSES, DATASETS

GLCM_NAMES = ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy",
              "glcm_correlation", "glcm_asm"]
WIN, LEVELS = 13, 17
# (row, col) offsets for 0, 45, 90, 135 degrees; with symmetric matrices the sign is irrelevant
OFFSETS = np.array([[0, 1], [1, 1], [1, 0], [1, -1]], dtype=np.int64)


def quantise(img):
    """img: (C>=4, H, W) reflectance with bands B1..B12 -> (H, W) int grey levels 0..16."""
    rgb = np.stack([img[3], img[2], img[1]], axis=-1).astype(np.float64)   # B4, B3, B2
    rgb = np.clip(rgb, 0.0, 0.15) / 0.15
    gray = rgb @ np.array([0.2125, 0.7154, 0.0721])
    return np.digitize(gray, np.linspace(0.0, 1.0, LEVELS - 1)).astype(np.int64)


@njit(parallel=True, cache=True)
def _glcm_at(qpad, rows, cols, offsets, levels, win):
    """qpad: grey levels padded by win//2 (reflect); rows/cols: pixel coordinates in the unpadded image."""
    half = win // 2
    n = rows.shape[0]
    out = np.zeros((n, 6), dtype=np.float64)
    for k in prange(n):
        r0, c0 = rows[k], cols[k]                      # window top-left in padded coords
        P = np.zeros((levels, levels), dtype=np.float64)
        for a in range(offsets.shape[0]):
            dr, dc = offsets[a, 0], offsets[a, 1]
            C = np.zeros((levels, levels), dtype=np.float64)
            tot = 0.0
            for i in range(win):
                i2 = i + dr
                if i2 < 0 or i2 >= win:
                    continue
                for j in range(win):
                    j2 = j + dc
                    if j2 < 0 or j2 >= win:
                        continue
                    u = qpad[r0 + i, c0 + j]
                    v = qpad[r0 + i2, c0 + j2]
                    C[u, v] += 1.0
                    C[v, u] += 1.0
                    tot += 2.0
            for u in range(levels):
                for v in range(levels):
                    P[u, v] += C[u, v] / tot / offsets.shape[0]
        con = dis = hom = asm = mi = mj = 0.0
        for u in range(levels):
            for v in range(levels):
                p = P[u, v]
                d = u - v
                con += p * d * d
                dis += p * abs(d)
                hom += p / (1.0 + d * d)
                asm += p * p
                mi += u * p
                mj += v * p
        si = sj = cov = 0.0
        for u in range(levels):
            for v in range(levels):
                p = P[u, v]
                si += p * (u - mi) ** 2
                sj += p * (v - mj) ** 2
                cov += p * (u - mi) * (v - mj)
        si, sj = np.sqrt(si), np.sqrt(sj)
        cor = 1.0 if (si < 1e-15 or sj < 1e-15) else cov / (si * sj)
        out[k, 0], out[k, 1], out[k, 2] = con, dis, hom
        out[k, 3], out[k, 4], out[k, 5] = np.sqrt(asm), cor, asm
    return out


def glcm_at(q, rows, cols):
    half = WIN // 2
    qpad = np.pad(q, half, mode="reflect")
    # pixel (r, c) -> its window starts at (r, c) in padded coordinates
    return _glcm_at(qpad, rows.astype(np.int64), cols.astype(np.int64), OFFSETS, LEVELS, WIN)


def split_features(dataset_name, split):
    """GLCM features for every annotated pixel of one split, in the staged cache order."""
    path = os.path.join(CACHE_DIR, f"{dataset_name}_{split}_all_glcm6.npy")
    cache = os.path.join(CACHE_DIR, f"{dataset_name}_{split}_all_seed42_f24.npz")
    y_cache = np.load(cache)["y"]
    if os.path.exists(path):
        G = np.load(path)
        if len(G) != len(y_cache):
            raise RuntimeError(f"{path}: {len(G)} rows for {len(y_cache)} cached pixels; delete it and rerun")
        return G
    cls, root = DATASETS[dataset_name]
    ds = cls(root, split=split)
    t0, G, Y = time.time(), [], []
    for i in range(len(ds)):
        img, m = ds[i]
        img, m = img.numpy(), m.numpy()
        flat = m.ravel()
        # same order as common.sample_split: classes in CLASSES order, raster order within each
        sel = np.concatenate([np.flatnonzero(flat == c) for c in CLASSES])
        if len(sel) == 0:
            continue
        rows, cols = np.divmod(sel, m.shape[1])
        G.append(glcm_at(quantise(img), rows, cols).astype(np.float32))
        Y.append(flat[sel])
        if (i + 1) % 200 == 0:
            print(f"  [{dataset_name}/{split}] {i + 1}/{len(ds)} patches, {time.time() - t0:.0f}s", flush=True)
    G, Y = np.vstack(G), np.concatenate(Y)
    if not np.array_equal(Y, y_cache):
        raise RuntimeError(f"{dataset_name}/{split}: GLCM pixel order does not match the cached labels")
    np.save(path, G)
    print(f"  [{dataset_name}/{split}] {len(G):,} pixels in {time.time() - t0:.0f}s -> {path}", flush=True)
    return G


def check(n=300, seed=0):
    """Compare the kernel with skimage (the reference implementation) on random windows."""
    from skimage.feature import graycomatrix, graycoprops
    rng = np.random.default_rng(seed)
    q = rng.integers(0, LEVELS, size=(40, 40))
    q[5:20, 5:20] = 3                                  # include flat areas (correlation = 1 case)
    rows, cols = rng.integers(0, 40, n), rng.integers(0, 40, n)
    ours = glcm_at(q, rows, cols)
    qpad = np.pad(q, WIN // 2, mode="reflect")
    worst = 0.0
    for k in range(n):
        w = qpad[rows[k]:rows[k] + WIN, cols[k]:cols[k] + WIN]
        M = graycomatrix(w, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=LEVELS,
                         normed=True, symmetric=True).mean(3)[:, :, :, None]
        ref = [graycoprops(M, p).item() for p in
               ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")]
        worst = max(worst, float(np.max(np.abs(np.array(ref) - ours[k]))))
    print(f"max |ours - skimage| over {n} windows x 6 features: {worst:.2e}")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--splits", nargs="+", default=["train", "test", "val"])
    ap.add_argument("--check", action="store_true", help="only compare the kernel with skimage")
    a = ap.parse_args()
    if a.check:
        raise SystemExit(0 if check() < 1e-9 else 1)
    for d in a.datasets:
        for s in a.splits:
            split_features(d, s)
