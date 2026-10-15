"""
Environmental Feature Engineering Module for ERA5 Integration.

Extracts and engineers physical oceanographic variables from ERA5 reanalysis data:
  - 10m U-wind component (zonal velocity, m/s)
  - 10m V-wind component (meridional velocity, m/s)
  - Wind speed magnitude (m/s): dictates wave breaking, whitecaps, and debris aggregation/dispersion
  - Wind direction cosine and sine: provides continuous directional orientation
  - Wind stress proxy (m^2/s^2): quadratic wind drag driving surface drift
  - Sea Surface Temperature in Celsius (°C)
  - Air-sea thermodynamic interaction term (SST_celsius * wind_speed)

Works in combination with the 21 spectral/index features from Sentinel-2 imagery.
"""

import numpy as np


BASE_SPECTRAL_NAMES = [
    "Band_Coastal", "Band_Blue", "Band_Green", "Band_Red",
    "Band_RE1", "Band_RE2", "Band_RE3", "Band_NIR",
    "Index_NDVI", "Index_NDWI", "Index_FAI", "Index_FDI", "Index_NDMI",
    "Ratio_Red_Green", "Ratio_NIR_Red", "Ratio_NIR_Green", "Ratio_SWIR_NIR",
    "Stat_Spectral_Mean", "Stat_Spectral_Std", "Stat_Spectral_Max", "Stat_Spectral_Norm"
]

ERA5_STANDARD_NAMES = [
    "ERA5_u10", "ERA5_v10", "ERA5_sst_celsius"
]

ERA5_ENGINEERED_NAMES = [
    "ERA5_u10", "ERA5_v10", "ERA5_wind_speed",
    "ERA5_wind_dir_cos", "ERA5_wind_dir_sin",
    "ERA5_wind_stress_proxy", "ERA5_sst_celsius",
    "ERA5_sst_wind_interaction"
]


def extract_era5_vector(u10: float, v10: float, sst: float, mode: str = "engineered") -> np.ndarray:
    """
    Construct environmental feature vector from raw ERA5 variables.

    Args:
        u10: Eastward 10m wind component (m/s)
        v10: Northward 10m wind component (m/s)
        sst: Sea surface temperature in Kelvin (or already in Celsius if < 100)
        mode: "standard" (3 feats) or "engineered" (8 feats)

    Returns:
        np.ndarray of shape (num_era5_features,)
    """
    # Convert SST to Celsius if in Kelvin
    sst_c = sst - 273.15 if sst > 100.0 else sst

    if mode == "standard":
        return np.array([u10, v10, sst_c], dtype=np.float32)

    # Engineered physical oceanographic variables
    wind_speed = np.sqrt(u10**2 + v10**2)
    eps = 1e-6
    wind_dir_cos = u10 / (wind_speed + eps)
    wind_dir_sin = v10 / (wind_speed + eps)
    wind_stress_proxy = wind_speed**2
    sst_wind_interaction = sst_c * wind_speed

    return np.array([
        u10,
        v10,
        wind_speed,
        wind_dir_cos,
        wind_dir_sin,
        wind_stress_proxy,
        sst_c,
        sst_wind_interaction
    ], dtype=np.float32)


def get_feature_names(era5_mode: str = None) -> list:
    """
    Return ordered feature names for model interpretability.

    Args:
        era5_mode: None (21 baseline features), "standard" (24 features), or "engineered" (29 features)
    """
    if era5_mode is None or era5_mode.lower() == "none":
        return list(BASE_SPECTRAL_NAMES)
    elif era5_mode.lower() == "standard":
        return list(BASE_SPECTRAL_NAMES) + list(ERA5_STANDARD_NAMES)
    elif era5_mode.lower() == "engineered":
        return list(BASE_SPECTRAL_NAMES) + list(ERA5_ENGINEERED_NAMES)
    else:
        raise ValueError(f"Unknown era5_mode: {era5_mode}")


def extract_pixel_features_era5(img_tensor, era5_dict=None, era5_mode="engineered", pixel_indices=None) -> np.ndarray:
    """
    Extract pixel-level features from Sentinel-2 image tensor and optionally append
    ERA5 environmental features.

    Args:
        img_tensor: Tensor or ndarray of shape (C, H, W)
        era5_dict: Optional dict with {"u10": ..., "v10": ..., "sst": ...}
        era5_mode: None, "standard", or "engineered"
        pixel_indices: Optional 1D array of flattened spatial indices to extract.
                       If provided, only those pixels are processed, saving massive memory & compute.

    Returns:
        np.ndarray of shape (N_pixels, num_features)
    """
    if hasattr(img_tensor, 'numpy'):
        img = img_tensor.numpy()
    else:
        img = np.array(img_tensor, dtype=np.float32)

    C, H, W = img.shape
    pixels = img.reshape(C, -1).T  # (H*W, C)

    if pixel_indices is not None:
        pixels = pixels[pixel_indices]

    num_pixels = pixels.shape[0]

    # 1. Base channels (first min(C, 8) bands)
    if C >= 8:
        base_feats = pixels[:, :8]
    else:
        pad = np.zeros((num_pixels, 8 - C), dtype=np.float32)
        base_feats = np.hstack([pixels, pad])

    # Key band picks for ratios/indices
    b_blue = pixels[:, 0]
    b_green = pixels[:, 1] if C > 1 else pixels[:, 0]
    b_red = pixels[:, 2] if C > 2 else pixels[:, 0]
    b_nir = pixels[:, min(6, C - 1 if C < 11 else 6)]
    b_swir = pixels[:, min(7, C - 1 if C < 11 else 7)]

    eps = 1e-6
    # 2. Normalized difference indices (5 features)
    ndvi = (b_nir - b_red) / (b_nir + b_red + eps)
    ndwi = (b_green - b_nir) / (b_green + b_nir + eps)
    fai = b_nir - (b_red + (b_swir - b_red) * 0.5)
    fdi = b_nir - (b_red + (b_swir - b_red) * 0.7)
    ndmi = (b_nir - b_swir) / (b_nir + b_swir + eps)
    indices = np.column_stack([ndvi, ndwi, fai, fdi, ndmi])

    # 3. Spectral Ratios (4 features)
    r1 = b_red / (b_green + eps)
    r2 = b_nir / (b_red + eps)
    r3 = b_nir / (b_green + eps)
    r4 = b_swir / (b_nir + eps)
    ratios = np.column_stack([r1, r2, r3, r4])

    # 4. Statistical summaries across channels per pixel (4 features)
    spectral_pixels = pixels[:, :min(C, 11)]
    mean_spec = np.mean(spectral_pixels, axis=1)
    std_spec = np.std(spectral_pixels, axis=1)
    max_spec = np.max(spectral_pixels, axis=1)
    norm_spec = np.linalg.norm(spectral_pixels, axis=1)
    stats = np.column_stack([mean_spec, std_spec, max_spec, norm_spec])

    # Combine 21 base spectral features
    features = np.hstack([base_feats, indices, ratios, stats])

    # 5. Optional ERA5 integration
    if era5_dict is not None and era5_mode is not None and era5_mode.lower() != "none":
        u10 = float(era5_dict.get("u10", 0.0))
        v10 = float(era5_dict.get("v10", 0.0))
        sst = float(era5_dict.get("sst", 300.0))

        env_vec = extract_era5_vector(u10, v10, sst, mode=era5_mode)
        # Broadcast across all pixels in patch
        env_broadcast = np.tile(env_vec, (num_pixels, 1))
        features = np.hstack([features, env_broadcast])

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features
