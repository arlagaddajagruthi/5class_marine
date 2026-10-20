# Data

Not tracked in git except the two environmental tables (see `.gitignore`).

| Path | Contents | Made by |
|---|---|---|
| `MARIDA_5Class/` | MARIDA with 4-class masks: `patches/S2_<scene>/S2_<patch>.tif` (11 bands) and `_cl.tif` (labels), `splits/{train,val,test}_X.txt` | `scripts/utilities/convert.py` from the original MARIDA in `data/MARIDA` |
| `MADOS_5Class/` | MADOS with 4-class masks, one folder per scene (`Scene_<n>/`) | `scripts/utilities/convert.py` from the original MADOS in `data/MADOS` |
| `environmental/marida_patch_metadata.csv` | date, overpass time and footprint of every MARIDA patch | `scripts/environmental/build_patch_metadata.py` |
| `environmental/marida_env_features.csv` | MUR SST, MODIS chlorophyll-a and ERA5 covariates per patch | `scripts/environmental/build_env_table.py` |
| `env_sst_chl_wind/marida_env_sst_chl_wind.csv` | SST, OLCI chlorophyll-a and Copernicus wind per patch, used by stage 6 | `scripts/env_sst_chl_wind/build_env_table.py` |
| `environmental/raw/`, `env_sst_chl_wind/raw/` | downloaded NetCDF files (cached, can be re-fetched) | the fetch scripts in those folders |

The original 15-class datasets are not in this folder; they are kept in
`C:/PROJECT/marine_pollution_detection/{MARIDA,MADOS}` (also the default
`--orig-root` of `scripts/staged_ml/fine_to_coarse.py`). To rebuild the
`*_5Class` folders, copy or link them to `data/MARIDA` and `data/MADOS` first.

Label mapping (15 original classes → 4): see the table in the main `README.md`.
Class 0 (unlabelled, and cloud / cloud shadow in MARIDA) is ignored by every
model.

MADOS patches carry no georeferencing or date, so environmental covariates
exist for MARIDA only.
