"""
Step 3: download ERA5 hourly single-level data from the Copernicus CDS for
every MARIDA scene (wind, MSL pressure, precipitation, waves, Stokes drift).

Each request covers the scene's patches plus a 1 deg margin, for the
acquisition day and the HISTORY_DAYS days before it (for antecedent wind and
rainfall). Needs a CDS account and ~/.cdsapirc (url: https://cds.climate.copernicus.eu/api),
and the ERA5 licence accepted on the CDS website.

Output: data/environmental/raw/era5/<scene>.nc (or <scene>_parts/*.nc)
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.environmental import era5
from src.environmental.features import PATCH_METADATA_CSV, RAW_DIR

ERA5_RAW = os.path.join(RAW_DIR, "era5")


def fetch_one(scene, r):
    import cdsapi
    if era5.scene_files(ERA5_RAW, scene):
        return scene, "cached"
    overpass = pd.Timestamp(r["overpass_utc"]).to_pydatetime()
    req = era5.scene_request(era5.history_dates(overpass),
                             r["lat_min"], r["lat_max"], r["lon_min"], r["lon_max"])
    client = cdsapi.Client(quiet=True, progress=False)
    path = era5.download_scene(client, req, os.path.join(ERA5_RAW, f"{scene}.nc"))
    return scene, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=PATCH_METADATA_CSV)
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent CDS requests (CDS queues them server-side)")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    scenes = meta.groupby("scene").agg(
        overpass_utc=("overpass_utc", "first"), lat_min=("lat_min", "min"),
        lat_max=("lat_max", "max"), lon_min=("lon_min", "min"), lon_max=("lon_max", "max"))
    print(f"Requesting ERA5 for {len(scenes)} scenes with {args.workers} workers...")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, s, r): s for s, r in scenes.iterrows()}
        for i, fut in enumerate(as_completed(futures), 1):
            scene = futures[fut]
            try:
                _, status = fut.result()
                print(f"  [{i:2d}/{len(scenes)}] {scene:16s} -> {status}", flush=True)
            except Exception as e:
                failures.append(scene)
                print(f"  [{i:2d}/{len(scenes)}] {scene:16s} FAILED: {e}", flush=True)

    if failures:
        print(f"\n{len(failures)} scenes failed; re-run this script to retry them: {failures}")
        sys.exit(1)
    print("All ERA5 scenes downloaded.")


if __name__ == "__main__":
    main()
