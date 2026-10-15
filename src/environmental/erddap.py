"""
NOAA CoastWatch ERDDAP griddap client (no account required).

Datasets used
-------------
jplMURSST41     GHRSST MUR L4 SST v4.1, 0.01 deg (~1.1 km), daily, gap-free.
                Finest global daily SST product; a 2.56 km MARIDA patch spans
                ~2-3 MUR cells, so SST is resolved at roughly patch scale.
erdMH1chla8day  MODIS-Aqua L3 chlorophyll-a, 0.0417 deg (~4.6 km), 8-day composite.
erdMH1chlamday  Same sensor, monthly composite (only used where the 8-day
                composite has no valid pixel near the patch, e.g. persistent cloud).

All values are real observations/analyses; nothing is generated. Raw subsets
are cached as CSV so the downloads are reproducible and only happen once.
"""
import io
import os
import time
import urllib.parse

import numpy as np
import pandas as pd
import requests

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"

DATASETS = {
    "mur_sst": {"id": "jplMURSST41", "var": "analysed_sst", "res_deg": 0.01},
    "chla_8day": {"id": "erdMH1chla8day", "var": "chlorophyll", "res_deg": 0.0417},
    "chla_month": {"id": "erdMH1chlamday", "var": "chlorophyll", "res_deg": 0.0417},
}

_LAT_DESCENDING = {}


def _get(url, timeout=180, retries=4):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text
            # 404 with "Your query produced no matching results" is a real empty answer
            if r.status_code == 404 and "no matching results" in r.text.lower():
                return None
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except requests.RequestException as e:
            last = e
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ERDDAP request failed: {url}\n{last}")


def lat_is_descending(dataset_id):
    """ERDDAP needs lat ranges in storage order, so check it once per dataset."""
    if dataset_id not in _LAT_DESCENDING:
        txt = _get(f"{ERDDAP_BASE}/{dataset_id}.csv?latitude%5B0:1:1%5D")
        df = pd.read_csv(io.StringIO(txt), skiprows=[1])
        _LAT_DESCENDING[dataset_id] = bool(df["latitude"].iloc[1] < df["latitude"].iloc[0])
    return _LAT_DESCENDING[dataset_id]


def build_query(key, time_iso, lat_min, lat_max, lon_min, lon_max):
    ds = DATASETS[key]
    lat_a, lat_b = (lat_max, lat_min) if lat_is_descending(ds["id"]) else (lat_min, lat_max)
    q = (f"{ds['var']}[({time_iso})]"
         f"[({lat_a:.4f}):1:({lat_b:.4f})]"
         f"[({lon_min:.4f}):1:({lon_max:.4f})]")
    return f"{ERDDAP_BASE}/{ds['id']}.csv?" + urllib.parse.quote(q, safe="():,")


def fetch_subset(key, time_iso, lat_min, lat_max, lon_min, lon_max, cache_csv):
    """
    Download (or load from cache) a single-time lat/lon subset as a long-format
    DataFrame with columns time, latitude, longitude, value.
    ERDDAP snaps the requested time to the nearest available time step; the
    returned `time` column records which one was actually used.
    """
    if os.path.exists(cache_csv):
        return pd.read_csv(cache_csv)
    url = build_query(key, time_iso, lat_min, lat_max, lon_min, lon_max)
    txt = _get(url)
    if txt is None:
        df = pd.DataFrame(columns=["time", "latitude", "longitude", "value"])
    else:
        df = pd.read_csv(io.StringIO(txt), skiprows=[1])
        df = df.rename(columns={DATASETS[key]["var"]: "value"})
    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


def to_grid(df):
    """Long-format subset -> (values[nlat, nlon], lats asc, lons asc, time string)."""
    if df.empty:
        return None
    g = df.pivot_table(index="latitude", columns="longitude", values="value", dropna=False)
    g = g.sort_index().sort_index(axis=1)
    return (g.to_numpy(dtype=np.float64), g.index.to_numpy(dtype=np.float64),
            g.columns.to_numpy(dtype=np.float64), str(df["time"].iloc[0]))
