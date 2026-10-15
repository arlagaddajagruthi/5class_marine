# Environmental covariates: SST, chlorophyll-a, wind (MARIDA only)

This is a smaller replacement for `scripts/environmental/`. It keeps only
three variables and uses **no ERA5**. MADOS can't be used because its tiles
have no location or date.

| Variable | Source | Resolution | Time used | Columns |
|---|---|---|---|---|
| SST, SST gradient | GHRSST MUR L4 v4.1 (NASA JPL), reused from `data/environmental/marida_env_features.csv` | 0.01° ≈ 1.1 km, daily | scene day | `sst_c`, `sst_grad_c_per_km` |
| Chlorophyll-a | Copernicus Marine OLCI L3 300 m daily; fallback: GlobColour L4 gap-free 4 km daily | 300 m / 4 km | scene day | `log10_chl`, `chl_source`, `chl_olci_valid_frac` |
| Wind | Copernicus Marine L4 hourly sea-surface wind (scatterometer-corrected model wind) | 0.125° ≈ 14 km, hourly | overpass time, plus 24 h and 72 h history | `wind_speed`, `wind_u`, `wind_v`, `wind_speed_mean_24h`, `wind_speed_mean_72h`, `wind_speed_max_72h` |

A patch is 2.56 km across.

## Chlorophyll-a

Each patch takes the first of these that has data:

1. **`olci300m_footprint`**: mean of the valid 300 m cells inside the patch.
2. **`olci300m_nearest3km`**: the nearest valid 300 m cell within 3 km, if the
   footprint is cloud-covered.
3. **`gapfree4km`**: the 4 km gap-free product.

The values are averaged in log10 space, because chlorophyll is log-normally
distributed. OLCI starts on 2016-04-25, so the one earlier scene always uses
the 4 km product.

## Wind

Wind is taken at the patch centre with bilinear interpolation. Near the coast,
land cells are skipped and the nearest sea cell within 30 km is used instead.
The value is then interpolated in time to the Sentinel-2 overpass.

## Missing data

Nothing is synthesised. A value that no real source covers stays NaN.

## Run (from the repo root)

You need a Copernicus Marine login first (`copernicusmarine login`).

```powershell
python scripts/env_sst_chl_wind/fetch_cmems.py      # -> data/env_sst_chl_wind/raw/{chl_olci300m,chl_gapfree4km,wind_l4}/<scene>.nc
python scripts/env_sst_chl_wind/build_env_table.py  # -> data/env_sst_chl_wind/marida_env_sst_chl_wind.csv
```

The fetch script skips files it has already downloaded, so you can re-run it
after an interruption. The table needs
`data/environmental/marida_patch_metadata.csv` and the MUR SST columns of
`data/environmental/marida_env_features.csv`. Both come from
`scripts/environmental/` and are already built.
