"""
Geolocation and acquisition time of MARIDA patches.

MARIDA scene folders are named  S2_<D>-<M>-<YY>_<MGRS tile>, e.g. S2_1-12-19_48MYU
(1 Dec 2019, tile 48MYU). Each patch GeoTIFF is 256 x 256 px at 10 m in the
tile's UTM CRS, i.e. a 2.56 km x 2.56 km footprint.

The exact sensing time is not stored in the patches. Sentinel-2 crosses the
equator on a descending node at 10:30 local mean solar time, so the UTC
acquisition time is estimated as 10.5 h - longitude / 15. The error is
typically < 20 min, well inside the 1-hour step of ERA5.
"""
import os
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

SCENE_RE = re.compile(r"^(?:S2_)?(\d{1,2})-(\d{1,2})-(\d{2})_([0-9]{2}[A-Z]{3})$")
S2_LOCAL_SOLAR_TIME_H = 10.5


def parse_scene(scene):
    """'1-12-19_48MYU' or 'S2_1-12-19_48MYU' -> (date, tile)."""
    m = SCENE_RE.match(scene)
    if not m:
        raise ValueError(f"Unrecognised MARIDA scene name: {scene}")
    day, month, yy, tile = m.groups()
    return datetime(2000 + int(yy), int(month), int(day), tzinfo=timezone.utc), tile


def s2_overpass_utc(date, lon):
    """Estimated Sentinel-2 sensing time (UTC) for a given date and longitude."""
    return date + timedelta(hours=S2_LOCAL_SOLAR_TIME_H - lon / 15.0)


def patch_footprint(tif_path):
    """Return centroid and lat/lon bounding box of a georeferenced patch."""
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"{tif_path} has no CRS")
        b = src.bounds
        to_wgs84 = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        xs = np.array([b.left, b.right, b.right, b.left, (b.left + b.right) / 2])
        ys = np.array([b.top, b.top, b.bottom, b.bottom, (b.top + b.bottom) / 2])
        lons, lats = to_wgs84.transform(xs, ys)
        return {
            "epsg": src.crs.to_epsg(),
            "lon": float(lons[4]),
            "lat": float(lats[4]),
            "lon_min": float(lons[:4].min()),
            "lon_max": float(lons[:4].max()),
            "lat_min": float(lats[:4].min()),
            "lat_max": float(lats[:4].max()),
            "width_m": float(b.right - b.left),
            "height_m": float(b.top - b.bottom),
        }


def build_patch_metadata(marida_root, splits=("train", "val", "test")):
    """One row per patch listed in the MARIDA split files."""
    rows = []
    for split in splits:
        split_file = os.path.join(marida_root, "splits", f"{split}_X.txt")
        with open(split_file, "r") as f:
            patch_ids = [line.strip() for line in f if line.strip()]
        for patch_id in patch_ids:
            scene = patch_id.rsplit("_", 1)[0]
            tif = os.path.join(marida_root, "patches", f"S2_{scene}", f"S2_{patch_id}.tif")
            date, tile = parse_scene(scene)
            fp = patch_footprint(tif)
            rows.append({
                "patch_id": patch_id,
                "split": split,
                "scene": scene,
                "tile": tile,
                "date": date.strftime("%Y-%m-%d"),
                "overpass_utc": s2_overpass_utc(date, fp["lon"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                **fp,
            })
    return pd.DataFrame(rows)
