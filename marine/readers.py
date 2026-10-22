"""
Reading satellite images into the 11-band reflectance stack the models expect
(B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12), plus optional ground truth and
georeference.

Supported inputs
  * MARIDA patch: one 11-band GeoTIFF (+ optional *_cl.tif ground truth)
  * MADOS patch: the 11 *_L2R_rhorc_<wavelength>_<i>.tif files (+ optional *_L2R_cl_<i>.tif)
  * Sentinel-2 multi-band GeoTIFF with 11, 12 (L2A, with B9) or 13 (L1C) bands
  * Sentinel-2 single-band files named with their band (B01 ... B12, B8A),
    e.g. from Copernicus Browser or a .SAFE folder (.tif / .tiff / .jp2)
  * any of the above inside a .zip
"""
import glob
import os
import re
import tempfile
import warnings
import zipfile

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform

from .features import BAND_NAMES, N_BANDS

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

RASTER_EXT = (".tif", ".tiff", ".jp2")
PHOTO_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".heic")

_MADOS_BAND = re.compile(r"_L2R_rhorc_(\d+)_\d+\.tif$", re.I)
_MADOS_CL = re.compile(r"_L2R_cl_\d+\.tif$", re.I)
_BAND_NAME = re.compile(r"(?<![A-Za-z0-9])B(8A|0?[1-9]|1[0-2])(?![0-9A-Za-z])", re.I)

# Index of each of our 11 bands in common multi-band layouts.
LAYOUTS = {
    11: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],          # B1..B8, B8A, B11, B12 (MARIDA order)
    12: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11],         # L2A: B1..B8, B8A, B9, B11, B12
    13: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12],         # L1C: B1..B8, B8A, B9, B10, B11, B12
}


class InputError(ValueError):
    """Problem with the user's file that should be shown to them as-is."""


class Scene:
    def __init__(self, img, source, name, crs=None, transform=None, gt=None, notes=None):
        self.img = img                      # (11, H, W) float32 reflectance
        self.source = source                # "marida" | "mados" | "sentinel2"
        self.name = name
        # MADOS band files declare EPSG:4326 but carry an identity transform,
        # i.e. no real location: treat such files as not georeferenced.
        if transform is None or crs is None or transform.is_identity:
            crs, transform = None, None
            if source == "mados":
                (notes := notes if notes is not None else []).append(
                    "MADOS patches carry no location or date, so the map view and lat/lon are not "
                    "available; patch positions are given in pixels (10 m each).")
        self.crs = crs
        self.transform = transform
        self.gt = gt                        # (H, W) uint8 labels 0..4 or None
        self.notes = notes or []

    @property
    def pixel_size_m(self):
        if self.transform is not None and self.crs is not None and self.crs.is_projected:
            return abs(self.transform.a), abs(self.transform.e)
        return 10.0, 10.0

    def latlon(self, rows, cols):
        """Pixel centres -> (lat, lon) arrays, or None without georeference."""
        if self.crs is None or self.transform is None:
            return None
        xs, ys = rasterio.transform.xy(self.transform, rows, cols)
        lons, lats = warp_transform(self.crs, "EPSG:4326", np.atleast_1d(xs), np.atleast_1d(ys))
        return np.asarray(lats), np.asarray(lons)

    def bounds_latlon(self):
        h, w = self.img.shape[1:]
        r = self.latlon([0, 0, h - 1, h - 1], [0, w - 1, 0, w - 1])
        if r is None:
            return None
        lats, lons = r
        # [[lat, lon] corners]: top-left, top-right, bottom-left, bottom-right
        return [[float(a), float(o)] for a, o in zip(lats, lons)]


# ------------------------------------------------------------------ helpers

def _read(path, out_shape=None):
    with rasterio.open(path) as ds:
        if out_shape is None:
            arr = ds.read()
        else:
            arr = ds.read(out_shape=(ds.count,) + tuple(out_shape), resampling=Resampling.nearest)
        return arr, ds.crs, ds.transform, ds.count


def _clean(img):
    img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return img


def _to_reflectance(img, notes, boa_offset):
    """Sentinel-2 digital numbers (0-10000) -> reflectance (0-1)."""
    valid = img[:, np.any(img != 0, axis=0)]
    if valid.size and np.percentile(valid, 99) > 1.5:
        if boa_offset:
            img = img - 1000.0
            notes.append("Subtracted the L2A BOA offset (1000) before scaling.")
        img = img / 10000.0
        notes.append("Values looked like Sentinel-2 digital numbers; divided by 10,000 to get reflectance.")
    return img


def _read_mask(path, shape):
    arr, *_ = _read(path)
    m = arr[0]
    if m.shape != tuple(shape):
        return None
    return m.astype(np.uint8)


def _band_key(fname):
    m = _BAND_NAME.search(os.path.basename(fname))
    if not m:
        return None
    b = m.group(1).upper()
    return "B8A" if b == "8A" else f"B{int(b)}"


# -------------------------------------------------------------- loaders

def load_marida(tif_path, cl_path=None, name=None):
    arr, crs, tr, count = _read(tif_path)
    if count != N_BANDS:
        raise InputError(f"Expected an 11-band MARIDA patch, found {count} bands.")
    img = _clean(arr)
    gt = _read_mask(cl_path, img.shape[1:]) if cl_path and os.path.exists(cl_path) else None
    return Scene(img, "marida", name or os.path.basename(tif_path), crs, tr, gt)


def load_mados(band_paths, cl_path=None, name=None):
    files = sorted(band_paths, key=lambda f: int(_MADOS_BAND.search(f).group(1)))
    if len(files) != N_BANDS:
        raise InputError(f"A MADOS patch needs 11 *_L2R_rhorc_* band files, found {len(files)}.")
    target = None
    for f in files:                                   # finest grid = 10 m bands
        with rasterio.open(f) as ds:
            if target is None or ds.height > target[0]:
                target = (ds.height, ds.width)
    bands, crs, tr = [], None, None
    for f in files:
        arr, c, t, _ = _read(f, out_shape=target)
        bands.append(arr[0])
        if arr.shape[1:] == target and crs is None and c is not None:
            crs, tr = c, t
    img = _clean(np.stack(bands))
    gt = _read_mask(cl_path, img.shape[1:]) if cl_path else None
    return Scene(img, "mados", name or os.path.basename(files[0]).split("_L2R")[0], crs, tr, gt)


def load_multiband(path, boa_offset=False):
    arr, crs, tr, count = _read(path)
    notes = []
    if count not in LAYOUTS:
        raise InputError(
            f"This GeoTIFF has {count} band(s). The model needs Sentinel-2 bands B1–B8, B8A, B11 and B12 "
            "(11, 12 or 13 bands in standard order). An RGB image (3 bands) does not contain the "
            "infrared bands that reveal floating material.")
    img = arr[LAYOUTS[count]]
    if count != 11:
        notes.append(f"{count}-band file: assumed standard Sentinel-2 order and dropped "
                     f"{'B9' if count == 12 else 'B9 and B10'}.")
    img = _to_reflectance(_clean(img), notes, boa_offset)
    return Scene(img, "sentinel2", os.path.basename(path), crs, tr, None, notes)


def load_band_files(paths, boa_offset=False):
    by_band = {}
    for p in paths:
        k = _band_key(p)
        if k in BAND_NAMES:
            by_band[k] = p
    missing = [b for b in BAND_NAMES if b not in by_band]
    if missing:
        raise InputError("Missing Sentinel-2 band file(s): " + ", ".join(missing) +
                         ". Upload all of B01–B08, B8A, B11, B12 (B09/B10 are not needed).")
    target, crs, tr = None, None, None
    for b in BAND_NAMES:
        with rasterio.open(by_band[b]) as ds:
            if target is None or ds.height * ds.width > target[0] * target[1]:
                target, crs, tr = (ds.height, ds.width), ds.crs, ds.transform
    bands = [_read(by_band[b], out_shape=target)[0][0] for b in BAND_NAMES]
    notes = ["Resampled all bands to the finest (10 m) grid by nearest neighbour."]
    img = _to_reflectance(_clean(np.stack(bands)), notes, boa_offset)
    return Scene(img, "sentinel2", os.path.basename(by_band["B2"]), crs, tr, None, notes)


def load_files(paths, boa_offset=False):
    """Decide what the user uploaded and load it."""
    expanded = []
    tmp = None
    for p in paths:
        if p.lower().endswith(".zip"):
            tmp = tmp or tempfile.mkdtemp(prefix="marine_zip_")
            with zipfile.ZipFile(p) as z:
                for member in z.namelist():
                    if member.lower().endswith(RASTER_EXT) and ".." not in member:
                        expanded.append(z.extract(member, tmp))
        else:
            expanded.append(p)

    photos = [p for p in expanded if p.lower().endswith(PHOTO_EXT)]
    rasters = [p for p in expanded if p.lower().endswith(RASTER_EXT)]
    if photos and not rasters:
        raise InputError(
            "Ordinary photos (JPG/PNG) only have red, green and blue. The model needs Sentinel-2's "
            "near- and short-wave-infrared bands, because floating material shows up mainly in "
            "infrared. Please upload a Sentinel-2 GeoTIFF or a MARIDA/MADOS patch.")
    if not rasters:
        raise InputError("No .tif/.tiff/.jp2 file found in the upload.")

    mados_bands = [p for p in rasters if _MADOS_BAND.search(p)]
    mados_cl = [p for p in rasters if _MADOS_CL.search(p)]
    if mados_bands:
        return load_mados(mados_bands, mados_cl[0] if mados_cl else None)

    cl = [p for p in rasters if p.lower().endswith(("_cl.tif", "_cl.tiff"))]
    conf = [p for p in rasters if p.lower().endswith(("_conf.tif", "_conf.tiff"))]
    images = [p for p in rasters if p not in cl and p not in conf]

    if len(images) == 1:
        with rasterio.open(images[0]) as ds:
            count = ds.count
        if count == 11 and re.search(r"S2_\d+-\d+-\d+_\w+_\d+\.tif$", images[0], re.I):
            return load_marida(images[0], cl[0] if cl else None)
        return load_multiband(images[0], boa_offset)
    if len(images) > 1:
        return load_band_files(images, boa_offset)
    raise InputError("Only label/confidence masks were uploaded; please include the image itself.")


# --------------------------------------------------------- dataset samples

def load_sample(sample, marida_root, mados_root):
    if sample["dataset"] == "MARIDA":
        pid = sample["id"]
        scene = pid.rsplit("_", 1)[0]
        d = os.path.join(marida_root, "patches", f"S2_{scene}")
        return load_marida(os.path.join(d, f"S2_{pid}.tif"), os.path.join(d, f"S2_{pid}_cl.tif"), pid)
    scene, idx = sample["id"].rsplit("_", 1)
    d = os.path.join(mados_root, scene)
    bands = glob.glob(os.path.join(d, "*", f"{scene}_L2R_rhorc_*_{idx}.tif"))
    return load_mados(bands, os.path.join(d, "10", f"{scene}_L2R_cl_{idx}.tif"), sample["id"])
