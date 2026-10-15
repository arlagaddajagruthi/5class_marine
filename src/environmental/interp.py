"""
NaN-aware sampling of regular lat/lon grids.

Ocean products are NaN over land (and, for ocean colour, under cloud). Near
the coast a plain bilinear interpolation would return NaN, so corners that are
NaN are dropped and the remaining weights renormalised. If all four corners
are NaN, the nearest valid cell within `max_km` is used instead.
"""
import numpy as np

KM_PER_DEG = 111.32


def _axis_bracket(axis, q):
    """Indices (i0, i1) and fraction t such that q = axis[i0] + t*(axis[i1]-axis[i0])."""
    ascending = axis[-1] > axis[0]
    a = axis if ascending else axis[::-1]
    j = int(np.clip(np.searchsorted(a, q) - 1, 0, len(a) - 2)) if len(a) > 1 else 0
    if len(a) == 1:
        return 0, 0, 0.0
    t = float(np.clip((q - a[j]) / (a[j + 1] - a[j]), 0.0, 1.0))
    if ascending:
        return j, j + 1, t
    n = len(axis) - 1
    return n - j, n - (j + 1), t


def sample(values, lats, lons, qlat, qlon, max_km=30.0):
    """
    Sample `values` (..., nlat, nlon) at one point. Leading axes (e.g. time) are
    preserved. Returns an array of shape values.shape[:-2] (or a float).
    """
    values = np.asarray(values, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)

    i0, i1, ty = _axis_bracket(lats, qlat)
    j0, j1, tx = _axis_bracket(lons, qlon)
    corners = [(i0, j0, (1 - ty) * (1 - tx)), (i0, j1, (1 - ty) * tx),
               (i1, j0, ty * (1 - tx)), (i1, j1, ty * tx)]

    num = np.zeros(values.shape[:-2])
    den = np.zeros(values.shape[:-2])
    for i, j, w in corners:
        v = values[..., i, j]
        ok = np.isfinite(v)
        num += np.where(ok, v * w, 0.0)
        den += np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    if np.any(~np.isfinite(out)):
        nn = nearest_valid(values, lats, lons, qlat, qlon, max_km)
        out = np.where(np.isfinite(out), out, nn)
    return float(out) if out.ndim == 0 else out


def nearest_valid(values, lats, lons, qlat, qlon, max_km=30.0):
    """Value of the nearest cell that is finite anywhere along leading axes."""
    valid = np.isfinite(values)
    if values.ndim > 2:
        valid = valid.reshape(-1, *values.shape[-2:]).any(axis=0)
    if not valid.any():
        return np.full(values.shape[:-2], np.nan)
    glat, glon = np.meshgrid(lats, lons, indexing="ij")
    d = np.hypot((glat - qlat) * KM_PER_DEG, (glon - qlon) * KM_PER_DEG * np.cos(np.radians(qlat)))
    d = np.where(valid, d, np.inf)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    if d[i, j] > max_km:
        return np.full(values.shape[:-2], np.nan)
    return values[..., i, j]


def footprint_mean(values, lats, lons, lat_min, lat_max, lon_min, lon_max, n=5, max_km=0.0):
    """
    Mean of the field over a patch footprint, from an n x n lattice of bilinear
    samples inside the box. With max_km=0 no nearest-cell fallback is used, so
    the result is NaN only if no part of the footprint touches valid data.
    """
    qlats = np.linspace(lat_min, lat_max, n)
    qlons = np.linspace(lon_min, lon_max, n)
    s = [sample(values, lats, lons, a, b, max_km=max_km) for a in qlats for b in qlons]
    s = np.array(s, dtype=np.float64)
    return float(np.nanmean(s)) if np.isfinite(s).any() else np.nan


def gradient_magnitude(values, lats, lons):
    """|grad| of a 2-D lat/lon field in units per km (central differences)."""
    lat_c = float(np.mean(lats))
    dy = np.gradient(lats) * KM_PER_DEG
    dx = np.gradient(lons) * KM_PER_DEG * np.cos(np.radians(lat_c))
    gy = np.gradient(values, axis=0) / dy[:, None]
    gx = np.gradient(values, axis=1) / dx[None, :]
    return np.hypot(gx, gy)
