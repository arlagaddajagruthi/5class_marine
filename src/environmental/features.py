"""
Environmental feature table for MARIDA patches and helpers to attach it to
per-pixel feature matrices.

Every environmental value is a real observation/analysis that varies at
patch or scene scale, so within a patch it is broadcast to all pixels.
"""
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_DIR = os.path.join(REPO_ROOT, "data", "environmental")
PATCH_METADATA_CSV = os.path.join(ENV_DIR, "marida_patch_metadata.csv")
ENV_TABLE_CSV = os.path.join(ENV_DIR, "marida_env_features.csv")
RAW_DIR = os.path.join(ENV_DIR, "raw")

FEATURE_GROUPS = {
    # GHRSST MUR, 0.01 deg: patch-scale SST and thermal fronts (fronts aggregate floating matter)
    "sst": ["mur_sst_c", "mur_sst_grad_c_per_km"],
    # MODIS-Aqua, ~4.6 km: phytoplankton / organic matter proxy (Sargassum, NOM)
    "chla": ["modis_log10_chla"],
    # ERA5 10 m wind now and antecedent (windrows, foam, wind-driven drift)
    "wind": ["era5_u10", "era5_v10", "era5_wind_speed", "era5_wind_speed_mean_24h",
             "era5_wind_speed_mean_72h", "era5_wind_speed_max_72h",
             "era5_u10_mean_72h", "era5_v10_mean_72h"],
    # ERA5 pressure and antecedent rainfall (river run-off flushes debris)
    "met": ["era5_msl_hpa", "era5_precip_24h_mm", "era5_precip_72h_mm", "era5_precip_7d_mm"],
    # ERA5 waves and Stokes drift (sea state, surface transport of floating litter)
    "waves": ["era5_swh", "era5_swh_mean_72h", "era5_stokes_u", "era5_stokes_v", "era5_stokes_speed"],
}
ALL_ENV_COLUMNS = [c for cols in FEATURE_GROUPS.values() for c in cols]


def feature_set_columns(name):
    """
    'baseline' -> [], 'env_all' -> every env column, a group name -> that group,
    'a+b' -> union of groups, 'env_all-x' -> all groups except x (ablation).
    """
    if name == "baseline":
        return []
    if name == "env_all":
        return list(ALL_ENV_COLUMNS)
    if name.startswith("env_all-"):
        drop = set(FEATURE_GROUPS[name[len("env_all-"):]])
        return [c for c in ALL_ENV_COLUMNS if c not in drop]
    cols = []
    for g in name.split("+"):
        if g not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group '{g}'. Known: {list(FEATURE_GROUPS)}")
        cols += FEATURE_GROUPS[g]
    return cols


def load_env_table(path=ENV_TABLE_CSV):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the scripts in scripts/environmental/ first "
            "(build_patch_metadata -> fetch_ocean_erddap -> fetch_era5 -> build_env_table).")
    return pd.read_csv(path).set_index("patch_id")


def env_matrix(env_table, patch_ids, columns):
    """(n_patches, n_columns) float32 matrix aligned to patch_ids."""
    missing = [p for p in patch_ids if p not in env_table.index]
    if missing:
        raise KeyError(f"{len(missing)} patches missing from env table, e.g. {missing[:3]}")
    return env_table.loc[list(patch_ids), columns].to_numpy(dtype=np.float32)
