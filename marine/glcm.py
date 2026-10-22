"""
GLCM texture features, ported from scripts/staged_ml/glcm_features.py (dl branch)
of the 5class_marine repository, which follows the MARIDA reference code:

  RGB (B4, B3, B2) clipped to [0, 0.15] / 0.15 -> grey (0.2125 R + 0.7154 G + 0.0721 B)
  -> 17 grey levels -> 13 x 13 window (reflect padding) -> symmetric, normalised
  co-occurrence at distance 1 in 4 directions, averaged
  -> contrast, dissimilarity, homogeneity, energy, correlation, ASM
"""
import numpy as np
from numba import njit, prange

GLCM_NAMES = ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy",
              "glcm_correlation", "glcm_asm"]
WIN, LEVELS = 13, 17
OFFSETS = np.array([[0, 1], [1, 1], [1, 0], [1, -1]], dtype=np.int64)


def quantise(img):
    rgb = np.stack([img[3], img[2], img[1]], axis=-1).astype(np.float64)
    rgb = np.clip(rgb, 0.0, 0.15) / 0.15
    gray = rgb @ np.array([0.2125, 0.7154, 0.0721])
    return np.digitize(gray, np.linspace(0.0, 1.0, LEVELS - 1)).astype(np.int64)


@njit(parallel=True, cache=True)
def _glcm_at(qpad, rows, cols, offsets, levels, win):
    n = rows.shape[0]
    out = np.zeros((n, 6), dtype=np.float64)
    for k in prange(n):
        r0, c0 = rows[k], cols[k]
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


def glcm_at(img, rows, cols):
    """GLCM features at pixel coordinates (rows, cols) of an (11, H, W) image."""
    q = quantise(img)
    qpad = np.pad(q, WIN // 2, mode="reflect")
    return _glcm_at(qpad, np.asarray(rows, np.int64), np.asarray(cols, np.int64),
                    OFFSETS, LEVELS, WIN).astype(np.float32)


def glcm_image(img, valid=None):
    """(H*W, 6) GLCM features for every pixel (zeros where not valid)."""
    h, w = img.shape[1:]
    flat = np.arange(h * w) if valid is None else np.flatnonzero(valid.ravel())
    rows, cols = np.divmod(flat, w)
    out = np.zeros((h * w, 6), dtype=np.float32)
    if len(flat):
        out[flat] = glcm_at(img, rows, cols)
    return out
