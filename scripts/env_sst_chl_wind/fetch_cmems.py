"""
Download chlorophyll-a and wind for every MARIDA scene from Copernicus Marine
(needs a Copernicus Marine login: run `copernicusmarine login` once).

Per scene, a small box around its patches is cut out:
  chl_olci300m   OLCI L3 daily CHL, 300 m      cmems_obs-oc_glo_bgc-plankton_my_l3-olci-300m_P1D
                 (scene date; has cloud gaps; available from 2016-04-25)
  chl_gapfree4km GlobColour L4 gap-free daily CHL, 4 km
                 cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D  (fallback; scene date)
  wind_l4        L4 hourly 10 m wind, 0.125 deg (scatterometer-corrected model wind)
                 cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H  (72 h before the overpass to 1 h after)

SST is not downloaded here: the MUR SST already extracted by
scripts/environmental/ (data/environmental/marida_env_features.csv) is reused.

Input : data/environmental/marida_patch_metadata.csv
Output: data/env_sst_chl_wind/raw/<product>/<scene>.nc
Existing files are skipped, so the script can be re-run after an interruption.
"""
import argparse
import os
import sys
import time

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
META = os.path.join(REPO, "data", "environmental", "marida_patch_metadata.csv")
RAW = os.path.join(REPO, "data", "env_sst_chl_wind", "raw")

# name: (dataset_id, variables, spatial buffer in degrees)
PRODUCTS = {
    "chl_olci300m": ("cmems_obs-oc_glo_bgc-plankton_my_l3-olci-300m_P1D", ["CHL"], 0.05),
    "chl_gapfree4km": ("cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D", ["CHL"], 0.1),
    "wind_l4": ("cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H", ["eastward_wind", "northward_wind"], 0.25),
}
OLCI_START = pd.Timestamp("2016-04-25")
WIND_HISTORY_H = 72


def scene_table(meta):
    meta = meta.assign(overpass=pd.to_datetime(meta.overpass_utc, utc=True).dt.tz_localize(None))
    return meta.groupby("scene").agg(lat_min=("lat_min", "min"), lat_max=("lat_max", "max"),
                                     lon_min=("lon_min", "min"), lon_max=("lon_max", "max"),
                                     overpass=("overpass", "mean")).reset_index()


def time_window(name, overpass):
    if name == "wind_l4":
        t1 = overpass.ceil("h") + pd.Timedelta(hours=1)
        return overpass.floor("h") - pd.Timedelta(hours=WIND_HISTORY_H), t1
    day = overpass.normalize()
    return day, day


def fetch(cm, name, s, retries=3):
    dataset_id, variables, buf = PRODUCTS[name]
    out_dir = os.path.join(RAW, name)
    out = os.path.join(out_dir, f"{s.scene}.nc")
    if os.path.exists(out):
        return "skip"
    if name == "chl_olci300m" and s.overpass < OLCI_START:
        return "before OLCI"
    t0, t1 = time_window(name, s.overpass)
    for attempt in range(retries):
        try:
            cm.subset(dataset_id=dataset_id, variables=variables,
                      minimum_longitude=s.lon_min - buf, maximum_longitude=s.lon_max + buf,
                      minimum_latitude=s.lat_min - buf, maximum_latitude=s.lat_max + buf,
                      start_datetime=t0.strftime("%Y-%m-%dT%H:%M:%S"),
                      end_datetime=t1.strftime("%Y-%m-%dT%H:%M:%S"),
                      output_directory=out_dir, output_filename=f"{s.scene}.nc",
                      disable_progress_bar=True)
            return "ok"
        except Exception as e:  # network hiccups: retry, then report and continue
            err = e
            time.sleep(5 * (attempt + 1))
    return f"FAILED: {str(err)[:200]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=list(PRODUCTS))
    ap.add_argument("--scenes", nargs="+", help="Only these scenes (default: all)")
    args = ap.parse_args()

    import copernicusmarine as cm
    scenes = scene_table(pd.read_csv(META))
    if args.scenes:
        scenes = scenes[scenes.scene.isin(args.scenes)]
    failed = 0
    for name in args.products:
        os.makedirs(os.path.join(RAW, name), exist_ok=True)
        for i, s in enumerate(scenes.itertuples(), 1):
            status = fetch(cm, name, s)
            failed += status.startswith("FAILED")
            print(f"[{name}] {i}/{len(scenes)} {s.scene} {s.overpass:%Y-%m-%d %H:%M}Z: {status}", flush=True)
    print(f"Done; {failed} failed downloads.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
