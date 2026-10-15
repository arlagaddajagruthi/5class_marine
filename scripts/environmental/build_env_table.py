"""
Step 4: turn the cached raw downloads into one row of environmental features
per MARIDA patch.

  MUR SST        mean over the 2.56 km patch footprint (bilinear lattice) and
                 mean SST-gradient magnitude over the footprint (deg C / km)
  MODIS chl-a    log10 of the footprint mean; if clouds/coast leave the 8-day
                 composite empty there, the nearest valid 8-day cell within
                 10 km, then the monthly composite, are used (column chla_source)
  ERA5           bilinear at the patch centroid, linear in time to the
                 estimated S2 overpass, plus 24 h / 72 h / 7 d antecedent stats

Remaining gaps are left as NaN (no values are invented); the experiment script
reports how many there are and handles them explicitly.

Output: data/environmental/marida_env_features.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.environmental import era5, erddap, interp
from src.environmental.features import (ALL_ENV_COLUMNS, ENV_TABLE_CSV,
                                        PATCH_METADATA_CSV, RAW_DIR)

CHLA_NN_KM = 10.0
SST_NN_KM = 5.0


def load_grid(key, scene):
    path = os.path.join(RAW_DIR, key, f"{scene}.csv")
    if not os.path.exists(path):
        return None
    return erddap.to_grid(pd.read_csv(path))


def box(r):
    return r["lat_min"], r["lat_max"], r["lon_min"], r["lon_max"]


def sst_features(grid, r):
    if grid is None:
        return {"mur_sst_c": np.nan, "mur_sst_grad_c_per_km": np.nan, "mur_time": None}
    v, la, lo, t = grid
    sst = interp.footprint_mean(v, la, lo, *box(r))
    if not np.isfinite(sst):
        sst = interp.sample(v, la, lo, r["lat"], r["lon"], max_km=SST_NN_KM)
    grad = interp.footprint_mean(interp.gradient_magnitude(v, la, lo), la, lo, *box(r))
    return {"mur_sst_c": sst, "mur_sst_grad_c_per_km": grad, "mur_time": t}


def chla_features(grid8, gridm, r):
    for grid, name in ((grid8, "8day"), (gridm, "month")):
        if grid is None:
            continue
        v, la, lo, t = grid
        c = interp.footprint_mean(v, la, lo, *box(r))
        src = name
        if not np.isfinite(c):
            c = interp.sample(v, la, lo, r["lat"], r["lon"], max_km=CHLA_NN_KM)
            src = f"{name}_nn{int(CHLA_NN_KM)}km"
        if np.isfinite(c) and c > 0:
            return {"modis_log10_chla": float(np.log10(c)), "chla_source": src, "chla_time": t}
    return {"modis_log10_chla": np.nan, "chla_source": "missing", "chla_time": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=PATCH_METADATA_CSV)
    parser.add_argument("--out", default=ENV_TABLE_CSV)
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    rows = []
    for scene, grp in meta.groupby("scene", sort=True):
        mur = load_grid("mur_sst", scene)
        c8 = load_grid("chla_8day", scene)
        cm = load_grid("chla_month", scene)
        era5_fields = era5.load_scene(os.path.join(RAW_DIR, "era5"), scene)
        if not era5_fields:
            print(f"  [warn] no ERA5 data for {scene}")
        for _, r in grp.iterrows():
            row = {"patch_id": r["patch_id"]}
            row.update(sst_features(mur, r))
            row.update(chla_features(c8, cm, r))
            if era5_fields:
                row.update(era5.point_features(era5_fields, r["lat"], r["lon"],
                                               pd.Timestamp(r["overpass_utc"])))
            rows.append(row)
        print(f"  {scene:16s} {len(grp):3d} patches")

    env = pd.DataFrame(rows)
    for c in ALL_ENV_COLUMNS:
        if c not in env:
            env[c] = np.nan
    table = meta.merge(env, on="patch_id", how="left")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    table.to_csv(args.out, index=False)

    print("\nCoverage (patches with a value / total):")
    for c in ALL_ENV_COLUMNS:
        print(f"  {c:28s} {table[c].notna().sum():5d} / {len(table)}")
    print("\nchl-a source:\n" + table["chla_source"].value_counts().to_string())
    print("\nSummary statistics:")
    print(table[ALL_ENV_COLUMNS].describe().T[["mean", "std", "min", "max"]].to_string())
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
