"""
ERA5 Environmental Data Integration for MARIDA Marine Pollution Pipeline.

Provides:
  - MARIDAMetadataParser: Extracts date, CRS, and WGS84 coordinates from MARIDA GeoTIFFs
  - ERA5BulkDownloader: Downloads ERA5 reanalysis data in monthly batches via CDS API
  - ERA5FeatureExtractor: Extracts per-patch environmental vectors from cached NetCDFs
  - build_era5_lookup: Master function that produces a JSON lookup keyed by patch_id

Requires: rasterio, cdsapi, xarray, netCDF4, scipy
"""

import os
import re
import json
import logging
import numpy as np
from datetime import datetime

import rasterio
from rasterio.warp import transform_bounds

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Default ERA5 variables to download
DEFAULT_ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "sea_surface_temperature",
]

# Sentinel-2 overpass UTC hours to download (covers ~09:00-12:00 UTC globally)
OVERPASS_HOURS = ["09:00", "10:00", "11:00", "12:00"]


# ---------------------------------------------------------------------------
# 1. Metadata Parser
# ---------------------------------------------------------------------------

class MARIDAMetadataParser:
    """
    Parses acquisition metadata from MARIDA folder/file naming convention.

    MARIDA scene folders follow the pattern: S2_DD-MM-YY_TILE
    Example: S2_1-12-19_48MYU  ->  day=1, month=12, year=2019, tile=48MYU
    """

    # Regex for scene folder name: S2_<day>-<month>-<2-digit-year>_<tile>
    SCENE_PATTERN = re.compile(r"S2_(\d{1,2})-(\d{1,2})-(\d{2})_([A-Z0-9]+)")

    @staticmethod
    def parse_scene_name(scene_name):
        """
        Parse a MARIDA scene folder name into date components and tile ID.

        Args:
            scene_name: e.g. "S2_1-12-19_48MYU"

        Returns:
            dict with keys: day, month, year (4-digit str), year_short, tile,
                            date_str (YYYY-MM-DD)
        """
        match = MARIDAMetadataParser.SCENE_PATTERN.match(scene_name)
        if not match:
            raise ValueError(f"Cannot parse scene name: '{scene_name}'. "
                             f"Expected format S2_DD-MM-YY_TILE")

        day, month, year_short, tile = match.groups()
        day = int(day)
        month = int(month)
        year_2d = int(year_short)

        # Convert 2-digit year: 00-49 -> 2000s, 50-99 -> 1900s
        year_4d = 2000 + year_2d if year_2d < 50 else 1900 + year_2d

        return {
            "day": day,
            "month": month,
            "year": str(year_4d),
            "year_short": year_short,
            "tile": tile,
            "date_str": f"{year_4d}-{month:02d}-{day:02d}",
        }

    @staticmethod
    def get_patch_wgs84(tif_path):
        """
        Open a MARIDA GeoTIFF and return its center point and bounding box in WGS84.

        Args:
            tif_path: Absolute path to a MARIDA patch .tif file

        Returns:
            dict with keys: center_lat, center_lon, bbox_wgs84 (lon_min, lat_min, lon_max, lat_max),
                            native_crs (str)
        """
        with rasterio.open(tif_path) as src:
            native_crs = str(src.crs)
            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                src.crs, "EPSG:4326", *src.bounds
            )

        center_lon = (lon_min + lon_max) / 2.0
        center_lat = (lat_min + lat_max) / 2.0

        return {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "bbox_wgs84": (lon_min, lat_min, lon_max, lat_max),
            "native_crs": native_crs,
        }

    @staticmethod
    def estimate_overpass_utc(center_lon):
        """
        Estimate Sentinel-2 overpass time in UTC from longitude.
        Sentinel-2 crosses the equator at ~10:30 local solar time (descending node).
        Local solar time ~ UTC + longitude/15 hours.

        Returns:
            str: Nearest whole hour in "HH:00" format
        """
        local_solar_hour = 10.5  # 10:30 AM local solar time
        utc_offset_hours = center_lon / 15.0
        utc_hour = local_solar_hour - utc_offset_hours

        # Wrap to 0-23
        utc_hour = utc_hour % 24.0
        nearest_hour = int(round(utc_hour))
        if nearest_hour == 24:
            nearest_hour = 0

        return f"{nearest_hour:02d}:00"

    @classmethod
    def parse_patch(cls, tif_path, scene_name=None):
        """
        Full metadata extraction for a single MARIDA patch.

        Args:
            tif_path: Path to the GeoTIFF
            scene_name: Optional scene folder name. If None, inferred from path.

        Returns:
            dict with all date, coordinate, and overpass metadata
        """
        if scene_name is None:
            # Infer from path: .../patches/S2_DD-MM-YY_TILE/S2_DD-MM-YY_TILE_N.tif
            scene_name = os.path.basename(os.path.dirname(tif_path))

        date_info = cls.parse_scene_name(scene_name)
        geo_info = cls.get_patch_wgs84(tif_path)
        overpass_utc = cls.estimate_overpass_utc(geo_info["center_lon"])

        return {**date_info, **geo_info, "overpass_utc": overpass_utc}


# ---------------------------------------------------------------------------
# 2. Bulk Downloader
# ---------------------------------------------------------------------------

class ERA5BulkDownloader:
    """
    Downloads ERA5 reanalysis data from the Copernicus Climate Data Store
    in monthly batches covering all MARIDA scenes.
    """

    def __init__(self, dataset_dir, cache_dir, variables=None):
        """
        Args:
            dataset_dir: Root of MARIDA_5Class (contains patches/ folder)
            cache_dir: Directory to store downloaded NetCDF files
            variables: List of ERA5 variable names. Defaults to wind U/V + SST.
        """
        self.dataset_dir = dataset_dir
        self.patches_dir = os.path.join(dataset_dir, "patches")
        self.cache_dir = cache_dir
        self.variables = variables or DEFAULT_ERA5_VARIABLES

        os.makedirs(self.cache_dir, exist_ok=True)

    def scan_scenes(self):
        """
        Scan all MARIDA scene folders and build a manifest of metadata.

        Returns:
            list of dicts, one per scene, each with date/geo info and list of patch TIFs
        """
        scenes = []
        parser = MARIDAMetadataParser()

        for scene_name in sorted(os.listdir(self.patches_dir)):
            scene_dir = os.path.join(self.patches_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue

            try:
                date_info = parser.parse_scene_name(scene_name)
            except ValueError as e:
                logger.warning(f"Skipping folder: {e}")
                continue

            # Get geo info from the first patch in this scene
            patch_tifs = sorted([
                f for f in os.listdir(scene_dir)
                if f.endswith(".tif") and "_cl" not in f and "_conf" not in f
            ])
            if not patch_tifs:
                logger.warning(f"No patch TIFs in {scene_name}, skipping")
                continue

            first_tif = os.path.join(scene_dir, patch_tifs[0])
            geo_info = parser.get_patch_wgs84(first_tif)

            scenes.append({
                "scene_name": scene_name,
                **date_info,
                **geo_info,
                "patch_tifs": patch_tifs,
            })

        logger.info(f"Scanned {len(scenes)} MARIDA scenes")
        return scenes

    def build_monthly_requests(self, scenes):
        """
        Group scenes by (year, month) and compute the bounding box for each group.

        Returns:
            dict: {(year, month): {"days": [...], "area": [N, W, S, E]}}
        """
        monthly = {}

        for scene in scenes:
            key = (scene["year"], f"{scene['month']:02d}")

            if key not in monthly:
                monthly[key] = {
                    "days": set(),
                    "lats": [],
                    "lons": [],
                }

            monthly[key]["days"].add(f"{scene['day']:02d}")
            lon_min, lat_min, lon_max, lat_max = scene["bbox_wgs84"]
            monthly[key]["lats"].extend([lat_min, lat_max])
            monthly[key]["lons"].extend([lon_min, lon_max])

        # Build CDS-format area bounding boxes with 0.5 degree padding
        requests = {}
        for key, info in monthly.items():
            pad = 0.5
            north = max(info["lats"]) + pad
            south = min(info["lats"]) - pad
            west = min(info["lons"]) - pad
            east = max(info["lons"]) + pad

            # Clamp to valid ranges
            north = min(north, 90.0)
            south = max(south, -90.0)
            west = max(west, -180.0)
            east = min(east, 180.0)

            requests[key] = {
                "days": sorted(info["days"]),
                "area": [north, west, south, east],  # CDS format: N, W, S, E
            }

        logger.info(f"Built {len(requests)} monthly download requests")
        return requests

    def _nc_path(self, year, month):
        """Path for a cached monthly NetCDF."""
        return os.path.join(self.cache_dir, f"era5_{year}_{month}.nc")

    def download_all(self, dry_run=False):
        """
        Execute all monthly CDS API downloads. Skips already-cached files.

        Args:
            dry_run: If True, only prints what would be downloaded without calling CDS.

        Returns:
            list of paths to downloaded (or pre-existing) NetCDF files
        """
        import cdsapi

        scenes = self.scan_scenes()
        monthly_requests = self.build_monthly_requests(scenes)

        downloaded = []
        total = len(monthly_requests)

        for idx, ((year, month), req_info) in enumerate(sorted(monthly_requests.items()), 1):
            nc_path = self._nc_path(year, month)

            if os.path.exists(nc_path):
                logger.info(f"[{idx}/{total}] CACHED: {os.path.basename(nc_path)}")
                downloaded.append(nc_path)
                continue

            logger.info(
                f"[{idx}/{total}] Downloading ERA5 for {year}-{month} "
                f"(days: {req_info['days']}, area: {req_info['area']})"
            )

            if dry_run:
                logger.info("  -> DRY RUN: skipping actual download")
                continue

            request_params = {
                "product_type": ["reanalysis"],
                "format": "netcdf",
                "variable": self.variables,
                "year": year,
                "month": month,
                "day": req_info["days"],
                "time": OVERPASS_HOURS,
                "area": req_info["area"],
            }

            try:
                c = cdsapi.Client()
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    request_params,
                    nc_path,
                )
                logger.info(f"  -> Saved: {nc_path} ({os.path.getsize(nc_path) / 1024:.1f} KB)")
                downloaded.append(nc_path)
            except Exception as e:
                logger.error(f"  -> FAILED: {e}")
                # Remove partial file if exists
                if os.path.exists(nc_path):
                    os.remove(nc_path)

        return downloaded


# ---------------------------------------------------------------------------
# 3. Feature Extractor
# ---------------------------------------------------------------------------

class ERA5FeatureExtractor:
    """
    Extracts per-patch environmental vectors from cached ERA5 NetCDF files.
    Handles SST NaN values near coastlines with nearest-valid-pixel fallback.
    """

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self._ds_cache = {}  # {nc_path: xarray.Dataset}

    def _load_dataset(self, nc_path):
        """Load and cache an xarray Dataset."""
        if nc_path not in self._ds_cache:
            import xarray as xr
            self._ds_cache[nc_path] = xr.open_dataset(nc_path)
        return self._ds_cache[nc_path]

    def _nc_path(self, year, month):
        return os.path.join(self.cache_dir, f"era5_{year}_{int(month):02d}.nc")

    def extract(self, year, month, day, center_lat, center_lon, overpass_utc="10:00"):
        """
        Extract ERA5 environmental vector for a specific patch.

        Args:
            year: 4-digit year string
            month: month (int or str)
            day: day (int or str)
            center_lat: Patch center latitude (WGS84)
            center_lon: Patch center longitude (WGS84)
            overpass_utc: Estimated overpass time "HH:00"

        Returns:
            dict: {"u10": float, "v10": float, "sst": float}
        """
        import xarray as xr

        nc_path = self._nc_path(year, month)
        if not os.path.exists(nc_path):
            raise FileNotFoundError(
                f"ERA5 cache not found: {nc_path}. Run download_era5.py first."
            )

        ds = self._load_dataset(nc_path)

        # Build target timestamp
        hour = int(overpass_utc.split(":")[0])
        target_time = np.datetime64(f"{year}-{int(month):02d}-{int(day):02d}T{hour:02d}:00:00")

        # Select nearest grid point and time
        # ERA5 uses 'latitude' and 'longitude' as coordinate names
        # Handle both 'latitude'/'longitude' and 'lat'/'lon' conventions
        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"
        time_dim = "time" if "time" in ds.dims else "valid_time"

        point = ds.sel(
            **{
                lat_dim: center_lat,
                lon_dim: center_lon,
                time_dim: target_time,
            },
            method="nearest",
        )

        # Extract wind components (these are always over ocean and land -- rarely NaN)
        u10_var = "u10" if "u10" in point else "U10"
        v10_var = "v10" if "v10" in point else "V10"
        sst_var = "sst" if "sst" in point else "SST"

        u10 = float(point[u10_var].values)
        v10 = float(point[v10_var].values)
        sst_raw = float(point[sst_var].values)

        # Handle SST NaN (common near coastlines)
        if np.isnan(sst_raw):
            sst_raw = self._find_nearest_valid_sst(
                ds, center_lat, center_lon, target_time, lat_dim, lon_dim, time_dim, sst_var
            )

        return {"u10": u10, "v10": v10, "sst": sst_raw}

    def _find_nearest_valid_sst(self, ds, center_lat, center_lon, target_time,
                                lat_dim, lon_dim, time_dim, sst_var,
                                max_radius_deg=2.0):
        """
        Search for the nearest non-NaN SST value by expanding radius around the patch center.
        ERA5 grid spacing is 0.25 degrees, so we search all valid pixels using a KD-tree.

        Returns:
            float: Nearest valid SST value, or NaN if none found within max_radius_deg.
        """
        from scipy.spatial import cKDTree

        # Get the SST field for the target time
        time_slice = ds.sel(**{time_dim: target_time}, method="nearest")
        sst_field = time_slice[sst_var].values  # 2D array (lat, lon)
        lats = time_slice[lat_dim].values
        lons = time_slice[lon_dim].values

        # Create meshgrid of valid (non-NaN) SST coordinates
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        valid_mask = ~np.isnan(sst_field)

        if not np.any(valid_mask):
            logger.warning(
                f"No valid SST values in entire monthly file near "
                f"({center_lat:.2f}, {center_lon:.2f}). Returning NaN."
            )
            return float("nan")

        valid_lats = lat_grid[valid_mask]
        valid_lons = lon_grid[valid_mask]
        valid_sst = sst_field[valid_mask]

        # Use KD-tree for efficient nearest-neighbor search
        tree = cKDTree(np.column_stack([valid_lats, valid_lons]))
        dist, idx = tree.query([center_lat, center_lon])

        if dist > max_radius_deg:
            logger.warning(
                f"Nearest valid SST is {dist:.2f} deg away from "
                f"({center_lat:.2f}, {center_lon:.2f}) -- exceeds {max_radius_deg} deg limit. "
                f"Returning NaN."
            )
            return float("nan")

        nearest_sst = float(valid_sst[idx])
        logger.info(
            f"SST NaN fallback: used pixel {dist:.3f} deg away "
            f"({center_lat:.2f}, {center_lon:.2f}) -> SST={nearest_sst:.2f} K"
        )
        return nearest_sst

    def close(self):
        """Close all cached datasets."""
        for ds in self._ds_cache.values():
            ds.close()
        self._ds_cache.clear()


# ---------------------------------------------------------------------------
# 4. Master Lookup Builder
# ---------------------------------------------------------------------------

def build_era5_lookup(dataset_dir, cache_dir, output_json=None):
    """
    Build a pre-computed JSON lookup of ERA5 features for all MARIDA patches.

    This is the main entry point: call it once after bulk downloading to create
    a fast lookup file that the training pipeline reads at runtime.

    Args:
        dataset_dir: Root of MARIDA_5Class
        cache_dir: Directory containing cached ERA5 NetCDF files
        output_json: Path for the output JSON. Defaults to cache_dir/era5_features.json

    Returns:
        dict: {patch_id: {"u10": float, "v10": float, "sst": float}}
    """
    if output_json is None:
        output_json = os.path.join(cache_dir, "era5_features.json")

    parser = MARIDAMetadataParser()
    extractor = ERA5FeatureExtractor(cache_dir)
    patches_dir = os.path.join(dataset_dir, "patches")

    # Collect all patch IDs from all splits
    splits_dir = os.path.join(dataset_dir, "splits")
    all_patch_ids = set()
    for split_file in ["train_X.txt", "val_X.txt", "test_X.txt"]:
        split_path = os.path.join(splits_dir, split_file)
        if os.path.exists(split_path):
            with open(split_path, "r") as f:
                for line in f:
                    pid = line.strip()
                    if pid:
                        all_patch_ids.add(pid)

    logger.info(f"Building ERA5 lookup for {len(all_patch_ids)} patches...")

    lookup = {}
    errors = []
    nan_sst_count = 0

    for i, patch_id in enumerate(sorted(all_patch_ids)):
        # patch_id example: "1-12-19_48MYU_0"
        # scene name: "S2_" + patch_id without the last "_N" suffix
        scene_part = patch_id.rsplit("_", 1)[0]
        scene_name = f"S2_{scene_part}"
        tif_name = f"S2_{patch_id}.tif"
        tif_path = os.path.join(patches_dir, scene_name, tif_name)

        if not os.path.exists(tif_path):
            logger.warning(f"TIF not found: {tif_path}")
            errors.append(patch_id)
            continue

        try:
            meta = parser.parse_patch(tif_path, scene_name)
            vector = extractor.extract(
                year=meta["year"],
                month=meta["month"],
                day=meta["day"],
                center_lat=meta["center_lat"],
                center_lon=meta["center_lon"],
                overpass_utc=meta["overpass_utc"],
            )

            if np.isnan(vector["sst"]):
                nan_sst_count += 1
                vector["sst"] = 0.0  # Final fallback: zero-fill
                vector["sst_nan_fallback"] = True

            lookup[patch_id] = vector

        except Exception as e:
            logger.error(f"Error processing {patch_id}: {e}")
            errors.append(patch_id)

        if (i + 1) % 100 == 0 or (i + 1) == len(all_patch_ids):
            logger.info(f"  Processed {i + 1}/{len(all_patch_ids)} patches")

    extractor.close()

    # Save lookup
    with open(output_json, "w") as f:
        json.dump(lookup, f, indent=2)

    logger.info(
        f"ERA5 lookup saved to {output_json}\n"
        f"  Total patches: {len(all_patch_ids)}\n"
        f"  Successful: {len(lookup)}\n"
        f"  Errors: {len(errors)}\n"
        f"  SST NaN fallbacks: {nan_sst_count}"
    )

    if errors:
        logger.warning(f"Failed patch IDs: {errors}")

    return lookup
