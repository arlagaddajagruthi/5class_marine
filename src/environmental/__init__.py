"""
Environmental-covariate integration for MARIDA_5Class.

Only MARIDA patches carry georeferencing (CRS + geotransform) and an
acquisition date (in the scene name), so environmental data can only be
attached to MARIDA. MADOS_5Class tiles have an identity transform and no CRS.

Modules
-------
patch_geo  : per-patch footprint (lat/lon), acquisition date, S2 overpass time
interp     : NaN-aware bilinear sampling of regular lat/lon grids
erddap     : NOAA ERDDAP griddap client (MUR SST 0.01 deg, MODIS-Aqua chl-a 4 km)
era5       : Copernicus CDS ERA5 hourly single-level download + point features
features   : environmental feature table, feature groups, joining to pixels
"""
