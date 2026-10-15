"""
ERA5 hourly single-level reanalysis (Copernicus CDS, dataset
'reanalysis-era5-single-levels'), requested per MARIDA scene.

Resolution: 0.25 deg atmosphere (~28 km), 0.5 deg ocean waves. This is far
coarser than a 2.56 km patch, but there is no gap-free global wind/wave/
precipitation product at patch scale for 2015-2021 (ASCAT on ERDDAP is also
0.25 deg and has swath gaps). Values are bilinearly interpolated to each patch
centroid (NaN-aware near land) and linearly in time to the S2 overpass, so
neighbouring patches of a scene get slightly different, physically consistent
values rather than one identical grid-cell value.

Each scene request covers [overpass date - HISTORY_DAYS, overpass date] so
that antecedent conditions (72 h wind, 7-day rainfall) can be derived.
"""
import glob
import os
import zipfile
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr

from src.environmental.interp import sample

DATASET = "reanalysis-era5-single-levels"
HISTORY_DAYS = 7
AREA_MARGIN_DEG = 1.0

# CDS long name -> short name inside the NetCDF
VARIABLES = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "total_precipitation": "tp",
    "significant_height_of_combined_wind_waves_and_swell": "swh",
    "u_component_stokes_drift": "ust",
    "v_component_stokes_drift": "vst",
}


def scene_request(dates, lat_min, lat_max, lon_min, lon_max):
    """CDS request covering every day in `dates` over a padded bounding box."""
    days = sorted(set(dates))
    north = np.ceil((lat_max + AREA_MARGIN_DEG) * 4) / 4
    south = np.floor((lat_min - AREA_MARGIN_DEG) * 4) / 4
    west = np.floor((lon_min - AREA_MARGIN_DEG) * 4) / 4
    east = np.ceil((lon_max + AREA_MARGIN_DEG) * 4) / 4
    return {
        "product_type": ["reanalysis"],
        "variable": list(VARIABLES.keys()),
        # CDS takes the cartesian product of year/month/day; extra days are harmless
        "year": sorted({f"{d.year}" for d in days}),
        "month": sorted({f"{d.month:02d}" for d in days}),
        "day": sorted({f"{d.day:02d}" for d in days}),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [float(north), float(west), float(south), float(east)],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def history_dates(overpass):
    d0 = overpass.date()
    return [d0 - timedelta(days=k) for k in range(HISTORY_DAYS, -1, -1)]


def download_scene(client, request, target_nc):
    """Retrieve one scene; CDS may return a zip when variables differ in grid/step type."""
    if os.path.exists(target_nc):
        return target_nc
    os.makedirs(os.path.dirname(target_nc), exist_ok=True)
    tmp = target_nc + ".part"
    client.retrieve(DATASET, request, tmp)
    if zipfile.is_zipfile(tmp):
        out_dir = target_nc[:-3] + "_parts"
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(out_dir)
        os.remove(tmp)
        return out_dir
    os.replace(tmp, target_nc)
    return target_nc


def scene_files(raw_dir, scene):
    nc = os.path.join(raw_dir, f"{scene}.nc")
    if os.path.exists(nc):
        return [nc]
    return sorted(glob.glob(os.path.join(raw_dir, f"{scene}_parts", "*.nc")))


def load_scene(raw_dir, scene):
    """
    Return {short_name: (values[time, lat, lon], times, lats, lons)} keeping
    each variable on its native grid (waves are on a different grid).
    """
    out = {}
    for f in scene_files(raw_dir, scene):
        with xr.open_dataset(f) as ds:
            tdim = "valid_time" if "valid_time" in ds.dims else "time"
            for name in VARIABLES.values():
                if name not in ds:
                    continue
                da = ds[name]
                # Drop singleton/ensemble/expver dims if present
                for extra in [d for d in da.dims if d not in (tdim, "latitude", "longitude")]:
                    da = da.isel({extra: 0}) if da.sizes[extra] == 1 else da.max(extra, skipna=True)
                da = da.transpose(tdim, "latitude", "longitude").sortby(tdim)
                out[name] = (
                    da.values.astype(np.float64),
                    pd.to_datetime(da[tdim].values).tz_localize(None),
                    da["latitude"].values.astype(np.float64),
                    da["longitude"].values.astype(np.float64),
                )
    return out


def _series(field, lat, lon):
    values, times, lats, lons = field
    return pd.Series(sample(values, lats, lons, lat, lon, max_km=60.0), index=times)


def _at(series, t):
    """Linear interpolation of an hourly series at time t."""
    s = series.dropna()
    if s.empty:
        return np.nan
    x = (s.index - t).total_seconds().to_numpy()
    return float(np.interp(0.0, x, s.to_numpy(), left=np.nan, right=np.nan))


def _window(series, t, hours):
    """Values with t - hours < valid_time <= t (ERA5 accumulations end at valid_time)."""
    lo = t - timedelta(hours=hours)
    return series[(series.index > lo) & (series.index <= t)]


def point_features(fields, lat, lon, overpass):
    """ERA5 features for one patch centroid at the S2 overpass time."""
    t = pd.Timestamp(overpass).tz_localize(None) if pd.Timestamp(overpass).tzinfo else pd.Timestamp(overpass)
    s = {k: _series(f, lat, lon) for k, f in fields.items()}
    feats = {}

    u, v = s.get("u10"), s.get("v10")
    if u is not None and v is not None:
        ws = np.hypot(u, v)
        feats["era5_u10"] = _at(u, t)
        feats["era5_v10"] = _at(v, t)
        feats["era5_wind_speed"] = float(np.hypot(feats["era5_u10"], feats["era5_v10"]))
        feats["era5_wind_speed_mean_24h"] = float(_window(ws, t, 24).mean())
        feats["era5_wind_speed_mean_72h"] = float(_window(ws, t, 72).mean())
        feats["era5_wind_speed_max_72h"] = float(_window(ws, t, 72).max())
        feats["era5_u10_mean_72h"] = float(_window(u, t, 72).mean())
        feats["era5_v10_mean_72h"] = float(_window(v, t, 72).mean())
    if "msl" in s:
        feats["era5_msl_hpa"] = _at(s["msl"], t) / 100.0
    if "tp" in s:
        # tp is metres accumulated over the hour ending at valid_time
        feats["era5_precip_24h_mm"] = float(_window(s["tp"], t, 24).sum() * 1000.0)
        feats["era5_precip_72h_mm"] = float(_window(s["tp"], t, 72).sum() * 1000.0)
        feats["era5_precip_7d_mm"] = float(_window(s["tp"], t, 24 * 7).sum() * 1000.0)
    if "swh" in s:
        feats["era5_swh"] = _at(s["swh"], t)
        feats["era5_swh_mean_72h"] = float(_window(s["swh"], t, 72).mean())
    if "ust" in s and "vst" in s:
        feats["era5_stokes_u"] = _at(s["ust"], t)
        feats["era5_stokes_v"] = _at(s["vst"], t)
        feats["era5_stokes_speed"] = float(np.hypot(feats["era5_stokes_u"], feats["era5_stokes_v"]))
    return feats
