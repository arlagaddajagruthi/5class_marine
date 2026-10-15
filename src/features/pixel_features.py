"""
Per-pixel spectral features for the tree / SVM classifiers.

Both datasets are loaded as 11 Sentinel-2 bands in the same order
(src/datasets/marida.py and src/datasets/mados.py):

    idx  band  centre (nm)
     0   B1    443   coastal aerosol
     1   B2    490   blue
     2   B3    560   green
     3   B4    665   red
     4   B5    705   red edge 1
     5   B6    740   red edge 2
     6   B7    783   red edge 3
     7   B8    842   NIR
     8   B8A   865   narrow NIR
     9   B11   1610  SWIR 1
    10   B12   2190  SWIR 2

Features (24): the 11 bands, 5 indices, 4 band ratios, 4 spectral statistics.
"""
import numpy as np

N_BANDS = 11
B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 = range(N_BANDS)
WAVELENGTH_NM = {B4: 665.0, B8: 842.0, B11: 1610.0}

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
FEATURE_NAMES = (BAND_NAMES
                 + ["NDVI", "NDWI", "FAI", "FDI", "NDMI"]
                 + ["red/green", "nir/red", "nir/green", "swir1/nir"]
                 + ["spec_mean", "spec_std", "spec_max", "spec_norm"])
N_FEATURES = len(FEATURE_NAMES)

EPS = 1e-6


def extract_pixel_features(img):
    """
    img: (C, H, W) array or tensor with C >= 11 (first 11 = S2 bands above).
    Returns (H*W, 24) float32. Any channels after the first 11 (e.g. extra
    covariates appended by a dataset) are passed through as extra columns.
    """
    img = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    img = img.astype(np.float32, copy=False)
    C = img.shape[0]
    if C < N_BANDS:
        raise ValueError(f"Expected at least {N_BANDS} Sentinel-2 bands, got {C}")
    px = img.reshape(C, -1).T                      # (N, C)
    bands = px[:, :N_BANDS]

    green, red = bands[:, B3], bands[:, B4]
    re2, nir, swir1 = bands[:, B6], bands[:, B8], bands[:, B11]
    l_red, l_nir, l_swir = (WAVELENGTH_NM[b] for b in (B4, B8, B11))

    ndvi = (nir - red) / (nir + red + EPS)
    ndwi = (green - nir) / (green + nir + EPS)                       # McFeeters
    # Floating Algae Index (Hu 2009): NIR above the red-SWIR1 baseline
    fai = nir - (red + (swir1 - red) * (l_nir - l_red) / (l_swir - l_red))
    # Floating Debris Index (Biermann et al. 2020)
    fdi = nir - (re2 + (swir1 - re2) * (l_nir - l_red) / (l_swir - l_red) * 10.0)
    ndmi = (nir - swir1) / (nir + swir1 + EPS)
    indices = np.column_stack([ndvi, ndwi, fai, fdi, ndmi])

    ratios = np.column_stack([red / (green + EPS), nir / (red + EPS),
                              nir / (green + EPS), swir1 / (nir + EPS)])

    stats = np.column_stack([bands.mean(axis=1), bands.std(axis=1),
                             bands.max(axis=1), np.linalg.norm(bands, axis=1)])

    feats = [bands, indices, ratios, stats]
    if C > N_BANDS:
        feats.append(px[:, N_BANDS:])
    out = np.hstack(feats).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
