"""
Test ERA5 extraction on a single MARIDA GeoTIFF patch.

Usage:
    python scripts/test_era5_single.py --tif data/MARIDA_5Class/patches/S2_1-12-19_48MYU/S2_1-12-19_48MYU_0.tif

This script validates:
  1. MARIDA metadata parsing (date, CRS, WGS84 reprojection)
  2. CDS API connectivity and credential validity
  3. ERA5 data extraction (wind U/V, SST)
"""

import os
import sys
import argparse

# Ensure project root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing.era5_integration import (
    MARIDAMetadataParser,
    ERA5FeatureExtractor,
)


def test_single_patch(tif_path, cache_dir="data/era5_cache"):
    """Test full ERA5 extraction pipeline on one MARIDA patch."""

    tif_path = os.path.abspath(tif_path)
    cache_dir = os.path.join(REPO_ROOT, cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("  ERA5 Single-Patch Test")
    print("=" * 60)
    print(f"\nInput TIF: {tif_path}")

    # Step 1: Parse metadata
    print("\n--- Step 1: Parsing MARIDA Metadata ---")
    parser = MARIDAMetadataParser()
    meta = parser.parse_patch(tif_path)

    print(f"  Scene:        {os.path.basename(os.path.dirname(tif_path))}")
    print(f"  Date:         {meta['date_str']}")
    print(f"  Tile:         {meta['tile']}")
    print(f"  Native CRS:   {meta['native_crs']}")
    print(f"  Center (lat): {meta['center_lat']:.6f}")
    print(f"  Center (lon): {meta['center_lon']:.6f}")
    print(f"  BBox WGS84:   {meta['bbox_wgs84']}")
    print(f"  Est. UTC:     {meta['overpass_utc']}")

    # Step 2: Download ERA5 data for this single patch
    print("\n--- Step 2: Downloading ERA5 Data (single request) ---")
    import cdsapi

    nc_file = os.path.join(cache_dir, "era5_test_single.nc")

    year = meta["year"]
    month = f"{meta['month']:02d}"
    day = f"{meta['day']:02d}"
    lon_min, lat_min, lon_max, lat_max = meta["bbox_wgs84"]

    # Pad bounding box to ensure we capture nearest ERA5 grid nodes
    pad = 0.5
    area = [
        min(lat_max + pad, 90.0),    # North
        max(lon_min - pad, -180.0),  # West
        max(lat_min - pad, -90.0),   # South
        min(lon_max + pad, 180.0),   # East
    ]

    print(f"  CDS Request: {year}-{month}-{day} at {meta['overpass_utc']} UTC")
    print(f"  Area (N,W,S,E): {area}")

    if os.path.exists(nc_file):
        print(f"  -> Using cached file: {nc_file}")
    else:
        try:
            c = cdsapi.Client()
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "format": "netcdf",
                    "variable": [
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "sea_surface_temperature",
                    ],
                    "year": year,
                    "month": month,
                    "day": day,
                    "time": meta["overpass_utc"],
                    "area": area,
                },
                nc_file,
            )
            print(f"  -> Downloaded: {nc_file} ({os.path.getsize(nc_file)} bytes)")
        except Exception as e:
            print(f"\n  [ERROR] CDS API request failed: {e}")
            print("  Check your ~/.cdsapirc configuration and API key validity.")
            print("  CDS v2 URL: https://cds.climate.copernicus.eu/api")
            sys.exit(1)

    # Step 3: Extract values
    print("\n--- Step 3: Extracting ERA5 Values ---")
    import xarray as xr
    import numpy as np

    ds = xr.open_dataset(nc_file)
    print(f"  Dataset vars:   {list(ds.data_vars)}")
    print(f"  Dataset dims:   {dict(ds.dims)}")
    print(f"  Dataset coords: {list(ds.coords)}")

    # Detect coordinate names
    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"
    time_dim = "time" if "time" in ds.dims else "valid_time"

    print(f"  Using dims:     lat={lat_dim}, lon={lon_dim}, time={time_dim}")

    hour = int(meta["overpass_utc"].split(":")[0])
    target_time = np.datetime64(f"{year}-{month}-{day}T{hour:02d}:00:00")

    point = ds.sel(
        **{lat_dim: meta["center_lat"], lon_dim: meta["center_lon"], time_dim: target_time},
        method="nearest",
    )

    u10_var = "u10" if "u10" in point else "U10"
    v10_var = "v10" if "v10" in point else "V10"
    sst_var = "sst" if "sst" in point else "SST"

    u10 = float(point[u10_var].values)
    v10 = float(point[v10_var].values)
    sst = float(point[sst_var].values)

    wind_speed = np.sqrt(u10**2 + v10**2)

    print(f"\n  ---- Extracted ERA5 Vector ----")
    print(f"  Wind U (10m):    {u10:.4f} m/s")
    print(f"  Wind V (10m):    {v10:.4f} m/s")
    print(f"  Wind Speed:      {wind_speed:.4f} m/s")
    print(f"  SST:             {sst:.4f} K", end="")
    if not np.isnan(sst):
        print(f"  ({sst - 273.15:.2f} °C)")
    else:
        print("  [NaN — coastal masking]")

    # Sanity checks
    print("\n--- Sanity Checks ---")
    checks_passed = True

    if np.isnan(u10) or np.isnan(v10):
        print("  [WARN] Wind component is NaN — unusual")
        checks_passed = False
    elif abs(u10) > 50 or abs(v10) > 50:
        print("  [WARN] Wind component exceeds 50 m/s — check units")
        checks_passed = False
    else:
        print(f"  [OK] Wind values in normal range")

    if np.isnan(sst):
        print("  [WARN] SST is NaN — this patch may be over land/coast")
        print("         The bulk pipeline will handle this with nearest-valid-pixel fallback")
        checks_passed = False
    elif sst < 250 or sst > 320:
        print(f"  [WARN] SST {sst:.1f} K outside expected ocean range (270-305 K)")
        checks_passed = False
    else:
        print(f"  [OK] SST in normal ocean range")

    if checks_passed:
        print("\n  All checks PASSED. CDS API and extraction pipeline working correctly.")
    else:
        print("\n  Some checks had warnings — review output above.")

    ds.close()

    return {"u10": u10, "v10": v10, "sst": sst, "wind_speed": wind_speed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ERA5 extraction on a single MARIDA patch")
    parser.add_argument(
        "--tif",
        type=str,
        default=os.path.join(
            REPO_ROOT, "data", "MARIDA_5Class", "patches",
            "S2_1-12-19_48MYU", "S2_1-12-19_48MYU_0.tif"
        ),
        help="Path to a MARIDA GeoTIFF patch",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/era5_cache",
        help="Directory to cache downloaded ERA5 NetCDF files",
    )
    args = parser.parse_args()

    result = test_single_patch(args.tif, args.cache_dir)
    print(f"\nFinal vector: {result}")
