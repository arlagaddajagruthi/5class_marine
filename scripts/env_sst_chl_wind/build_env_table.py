"""
Build the per-patch environmental table (SST, chlorophyll-a, wind) for MARIDA.

  SST    MUR SST and SST-gradient magnitude, reused from
         data/environmental/marida_env_features.csv (GHRSST MUR L4, 0.01 deg, daily)
  chl    log10 chlorophyll-a, in order of preference:
           1. OLCI 300 m, mean over the patch footprint (cells whose centre is inside it)
           2. OLCI 300 m, nearest valid cell within 3 km (same day)
           3. GlobColour gap-free 4 km, mean over the footprint (nearest valid cell within 10 km)
         chl_source records which one was used; chl_olci_valid_frac is the share of
         valid 300 m cells in the footprint (0 = cloud-covered).
  wind   L4 hourly 0.125 deg wind, bilinear at the patch centre (coast-aware, see
         src/environmental/interp.py), linearly interpolated to the overpass time;
         plus mean over 24 h / 72 h and max over 72 h before the overpass.

Input : data/environmental/marida_patch_metadata.csv, data/env_sst_chl_wind/raw/
Output: data/env_sst_chl_wind/marida_env_sst_chl_wind.csv
Nothing is synthesised; values that no real source covers stay NaN.
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from src.environmental.interp import KM_PER_DEG, footprint_mean, nearest_valid, sample  # noqa: E402

META = os.path.join(REPO, "data", "environmental", "marida_patch_metadata.csv")
OLD_ENV = os.path.join(REPO, "data", "environmental", "marida_env_features.csv")
ROOT = os.path.join(REPO, "data", "env_sst_chl_wind")
RAW = os.path.join(ROOT, "raw")
OUT = os.path.join(ROOT, "marida_env_sst_chl_wind.csv")

FEATURES = ["sst_c", "sst_grad_c_per_km", "log10_chl",
            "wind_speed", "wind_u", "wind_v",
            "wind_speed_mean_24h", "wind_speed_mean_72h", "wind_speed_max_72h"]


def load(product, scene):
    path = os.path.join(RAW, product, f"{scene}.nc")
    return xr.open_dataset(path).load() if os.path.exists(path) else None


def chl_features(p, olci, gapfree):
    """Returns (log10_chl, source, valid fraction of 300 m footprint cells)."""
    if olci is not None:
        da = olci.CHL.isel(time=0)
        lats, lons = da.latitude.values, da.longitude.values
        inside = da.sel(latitude=lats[(lats >= p.lat_min) & (lats <= p.lat_max)],
                        longitude=lons[(lons >= p.lon_min) & (lons <= p.lon_max)]).values
        valid = np.isfinite(inside) & (inside > 0)
        frac = float(valid.mean()) if inside.size else 0.0
        if valid.any():
            return float(np.mean(np.log10(inside[valid]))), "olci300m_footprint", frac
        logv = np.log10(np.where(da.values > 0, da.values, np.nan))
        nn = float(nearest_valid(logv, lats, lons, p.lat, p.lon, max_km=3.0))
        if np.isfinite(nn):
            return nn, "olci300m_nearest3km", frac
    else:
        frac = np.nan
    if gapfree is not None:
        da = gapfree.CHL.isel(time=0)
        logv = np.log10(np.where(da.values > 0, da.values, np.nan))
        v = footprint_mean(logv, da.latitude.values, da.longitude.values,
                           p.lat_min, p.lat_max, p.lon_min, p.lon_max, max_km=10.0)
        if np.isfinite(v):
            return v, "gapfree4km", frac
    return np.nan, "none", frac


def wind_features(p, wind):
    keys = ["wind_speed", "wind_u", "wind_v", "wind_speed_mean_24h", "wind_speed_mean_72h", "wind_speed_max_72h"]
    if wind is None:
        return dict.fromkeys(keys, np.nan)
    lats, lons = wind.latitude.values, wind.longitude.values
    u = sample(wind.eastward_wind.values, lats, lons, p.lat, p.lon)   # (time,)
    v = sample(wind.northward_wind.values, lats, lons, p.lat, p.lon)
    t = pd.to_datetime(wind.time.values)
    tq = pd.Timestamp(p.overpass_utc).tz_convert(None)
    hours = (t - tq) / pd.Timedelta(hours=1)
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 2:
        return dict.fromkeys(keys, np.nan)
    hours, u, v = np.asarray(hours[ok]), u[ok], v[ok]
    speed = np.hypot(u, v)
    u0, v0 = np.interp(0.0, hours, u), np.interp(0.0, hours, v)
    past24, past72 = (hours <= 0) & (hours > -24), (hours <= 0) & (hours > -72)
    return {"wind_speed": float(np.hypot(u0, v0)), "wind_u": float(u0), "wind_v": float(v0),
            "wind_speed_mean_24h": float(speed[past24].mean()) if past24.any() else np.nan,
            "wind_speed_mean_72h": float(speed[past72].mean()) if past72.any() else np.nan,
            "wind_speed_max_72h": float(speed[past72].max()) if past72.any() else np.nan}


def main():
    meta = pd.read_csv(META)
    sst = pd.read_csv(OLD_ENV, usecols=["patch_id", "mur_sst_c", "mur_sst_grad_c_per_km"]).rename(
        columns={"mur_sst_c": "sst_c", "mur_sst_grad_c_per_km": "sst_grad_c_per_km"})
    rows = []
    for scene, g in meta.groupby("scene"):
        olci, gapfree, wind = load("chl_olci300m", scene), load("chl_gapfree4km", scene), load("wind_l4", scene)
        for p in g.itertuples():
            chl, src, frac = chl_features(p, olci, gapfree)
            rows.append({"patch_id": p.patch_id, "log10_chl": chl, "chl_source": src,
                         "chl_olci_valid_frac": frac, **wind_features(p, wind)})
        print(f"{scene}: {len(g)} patches", flush=True)
    df = meta.merge(sst, on="patch_id", how="left").merge(pd.DataFrame(rows), on="patch_id", how="left")
    df.to_csv(OUT, index=False)

    print(f"\nSaved {len(df)} patches -> {OUT}\n\nCoverage (non-NaN patches):")
    for f in FEATURES:
        print(f"  {f:22s} {df[f].notna().sum():5d} / {len(df)}")
    print("\nChlorophyll source:\n" + df.chl_source.value_counts().to_string())


if __name__ == "__main__":
    main()
