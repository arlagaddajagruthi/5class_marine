"""
Per-pixel spectral features, identical to src/features/pixel_features.py in the
5class_marine repository (era-5 branch), so the website reproduces the
stage-5 experiment exactly.

Band order (both datasets): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.
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

# Stage 5 of the staged study: bands + NDWI + 4 band ratios (16 features).
STAGE5 = BAND_NAMES + ["NDWI", "red/green", "nir/red", "nir/green", "swir1/nir"]
STAGE5_COLS = [FEATURE_NAMES.index(f) for f in STAGE5]

EPS = 1e-6


def extract_pixel_features(img):
    """img: (11, H, W) reflectance -> (H*W, 24) float32."""
    img = np.asarray(img, dtype=np.float32)
    px = img.reshape(img.shape[0], -1).T
    bands = px[:, :N_BANDS]

    green, red = bands[:, B3], bands[:, B4]
    re2, nir, swir1 = bands[:, B6], bands[:, B8], bands[:, B11]
    l_red, l_nir, l_swir = (WAVELENGTH_NM[b] for b in (B4, B8, B11))

    ndvi = (nir - red) / (nir + red + EPS)
    ndwi = (green - nir) / (green + nir + EPS)
    fai = nir - (red + (swir1 - red) * (l_nir - l_red) / (l_swir - l_red))
    fdi = nir - (re2 + (swir1 - re2) * (l_nir - l_red) / (l_swir - l_red) * 10.0)
    ndmi = (nir - swir1) / (nir + swir1 + EPS)
    indices = np.column_stack([ndvi, ndwi, fai, fdi, ndmi])

    ratios = np.column_stack([red / (green + EPS), nir / (red + EPS),
                              nir / (green + EPS), swir1 / (nir + EPS)])
    stats = np.column_stack([bands.mean(axis=1), bands.std(axis=1),
                             bands.max(axis=1), np.linalg.norm(bands, axis=1)])

    out = np.hstack([bands, indices, ratios, stats]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def stage5_features(img):
    return extract_pixel_features(img)[:, STAGE5_COLS]


# Stage 8: stage 5 + 6 GLCM texture features (22 features).
def stage8_features(img, valid=None):
    from .glcm import glcm_image
    return np.hstack([stage5_features(img), glcm_image(img, valid)])


FEATURE_SETS = {"stage5": stage5_features, "stage8": stage8_features}
