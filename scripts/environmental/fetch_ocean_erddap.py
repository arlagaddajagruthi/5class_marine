"""
Step 2: download MUR SST (0.01 deg, daily) and MODIS-Aqua chlorophyll-a
(~4.6 km, 8-day and monthly composites) from NOAA ERDDAP for every MARIDA scene.

One request per scene and product covers all of that scene's patches plus a
margin (needed for SST gradients and the chlorophyll cloud-gap fallback).
Raw subsets are cached under data/environmental/raw/<product>/<scene>.csv.
No account is needed.
"""
import argparse
import os
import sys

import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.environmental import erddap
from src.environmental.features import PATCH_METADATA_CSV, RAW_DIR

# Margins in degrees around the union of a scene's patch footprints
MARGINS = {"mur_sst": 0.05, "chla_8day": 0.15, "chla_month": 0.15}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=PATCH_METADATA_CSV)
    parser.add_argument("--products", nargs="+", default=list(MARGINS), choices=list(MARGINS))
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    scenes = meta.groupby("scene").agg(
        date=("date", "first"), lat_min=("lat_min", "min"), lat_max=("lat_max", "max"),
        lon_min=("lon_min", "min"), lon_max=("lon_max", "max"))

    for key in args.products:
        m = MARGINS[key]
        print(f"\n== {key} ({erddap.DATASETS[key]['id']}) for {len(scenes)} scenes")
        for i, (scene, r) in enumerate(scenes.iterrows(), 1):
            out = os.path.join(RAW_DIR, key, f"{scene}.csv")
            cached = os.path.exists(out)
            df = erddap.fetch_subset(
                key, f"{r['date']}T12:00:00Z",
                r["lat_min"] - m, r["lat_max"] + m, r["lon_min"] - m, r["lon_max"] + m, out)
            n_valid = int(df["value"].notna().sum()) if len(df) else 0
            used_time = df["time"].iloc[0] if len(df) else "-"
            print(f"  [{i:2d}/{len(scenes)}] {scene:16s} date={r['date']} used={used_time} "
                  f"cells={len(df):5d} valid={n_valid:5d}{' (cached)' if cached else ''}")


if __name__ == "__main__":
    main()
