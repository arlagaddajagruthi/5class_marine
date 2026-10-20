"""
Stage 6: stage 5 (11 bands + NDWI + 4 band ratios) plus environmental data,
MARIDA only (MADOS has no location or date).

Environmental features (per patch, given to every pixel of the patch), from
data/env_sst_chl_wind/marida_env_sst_chl_wind.csv (scripts/env_sst_chl_wind/):
  sst_c, sst_grad_c_per_km          MUR SST, 0.01 deg
  log10_chl                         OLCI 300 m chlorophyll-a (gap-free 4 km fallback)
  wind_speed, wind_speed_mean_24h,  Copernicus Marine L4 wind, 0.125 deg
  wind_speed_mean_72h, wind_speed_max_72h
Wind direction (u/v) is left out: its meaning depends on the site's coastline,
so with only 63 scenes it mostly identifies the scene rather than the conditions.

Missing values are filled with the median over the training patches (counts are
printed); test patches never contribute to the fill value.

Output: <--out-root, default outputs/staged>/stage6_env/MARIDA/
"""
import os

import numpy as np
import pandas as pd

from common import (BAND_NAMES, CACHE_DIR, CLASSES, DATASETS, RATIOS, REPO, base_parser, run_stage)

ENV_CSV = os.path.join(REPO, "data", "env_sst_chl_wind", "marida_env_sst_chl_wind.csv")
ENV_FEATURES = ["sst_c", "sst_grad_c_per_km", "log10_chl",
                "wind_speed", "wind_speed_mean_24h", "wind_speed_mean_72h", "wind_speed_max_72h"]


def pixel_patch_ids(split, n_pixels):
    """
    Patch id of every cached pixel. The pixel cache (common.sample_split, no cap)
    holds, patch by patch in dataset order, every annotated pixel grouped by class
    in CLASSES order; this rebuilds that order from the masks and checks the labels.
    """
    path = os.path.join(CACHE_DIR, f"MARIDA_{split}_all_patchids.npy")
    if os.path.exists(path):
        ids = np.load(path, allow_pickle=False)
    else:
        cls, root = DATASETS["MARIDA"]
        ds = cls(root, split=split)
        ids, labels = [], []
        for i in range(len(ds)):
            m = ds[i][1].numpy().ravel()
            for c in CLASSES:
                n = int(np.sum(m == c))
                ids += [ds.patch_ids[i]] * n
                labels += [c] * n
        ids = np.array(ids)
        np.save(path, ids)
        np.save(path.replace("patchids", "patchlabels"), np.array(labels, dtype=np.int64))
    if len(ids) != n_pixels:
        raise RuntimeError(f"{split}: {len(ids)} patch ids for {n_pixels} cached pixels; pixel cache out of date")
    return ids


def env_columns(dataset_name, keep_tr, keep_te):
    if dataset_name != "MARIDA":
        raise SystemExit("Stage 6 is MARIDA only (MADOS has no georeferencing).")
    z_tr = np.load(os.path.join(CACHE_DIR, "MARIDA_train_all_seed42_f24.npz"))["y"]
    z_te = np.load(os.path.join(CACHE_DIR, "MARIDA_test_all_seed42_f24.npz"))["y"]
    ids_tr, ids_te = pixel_patch_ids("train", len(z_tr)), pixel_patch_ids("test", len(z_te))
    for split, y in (("train", z_tr), ("test", z_te)):
        lab = np.load(os.path.join(CACHE_DIR, f"MARIDA_{split}_all_patchlabels.npy"))
        if not np.array_equal(lab, y):
            raise RuntimeError(f"{split}: rebuilt pixel order does not match the cached labels")

    env = pd.read_csv(ENV_CSV).set_index("patch_id")[ENV_FEATURES]
    train_patches = env.loc[np.unique(ids_tr)]
    fill = train_patches.median()
    for split, ids in (("train", ids_tr), ("test", ids_te)):
        missing = env.loc[np.unique(ids)].isna().sum()
        print(f"  env {split}: patches with a filled value: "
              + (", ".join(f"{k}={v}" for k, v in missing.items() if v) or "none"), flush=True)
    env = env.fillna(fill)
    E_tr = env.loc[ids_tr[keep_tr]].to_numpy(dtype=np.float32)
    E_te = env.loc[ids_te[keep_te]].to_numpy(dtype=np.float32)
    return E_tr, E_te, ENV_FEATURES


if __name__ == "__main__":
    ap = base_parser(__doc__)
    ap.set_defaults(datasets=["MARIDA"])
    args = ap.parse_args()
    run_stage("stage6_env", "bands + NDWI + 4 band ratios + SST, chlorophyll-a, wind",
              list(BAND_NAMES) + ["NDWI"] + RATIOS, args, extra=env_columns)
