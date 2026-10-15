# Environmental covariates for MARIDA_5Class

Tests whether real environmental data improves the pixel classifiers in
`scripts/baselines/`. Applies to **MARIDA only**: MADOS_5Class tiles have no
CRS or geotransform and no acquisition date, so they can't be matched to
environmental data.

## Data sources

| Group | Product | Native resolution | How it's matched to a 2.56 km patch |
|---|---|---|---|
| `sst` | GHRSST MUR L4 v4.1 (NOAA ERDDAP `jplMURSST41`) | 0.01° ≈ 1.1 km, daily | Mean over the patch footprint; SST-gradient magnitude (fronts) |
| `chla` | MODIS-Aqua chlorophyll-a (ERDDAP `erdMH1chla8day`, `erdMH1chlamday`) | 0.042° ≈ 4.6 km, 8-day / monthly | Footprint mean; fallbacks: nearest valid 8-day cell within 10 km, then the monthly composite |
| `wind` | ERA5 hourly (Copernicus CDS) | 0.25° ≈ 28 km | Bilinear at the centroid, interpolated to the S2 overpass time; 24 h / 72 h history |
| `met` | ERA5 MSL pressure, total precipitation | 0.25° | Same; rainfall summed over 24 h / 72 h / 7 days |
| `waves` | ERA5 significant wave height, Stokes drift | 0.5° | Same; nearest sea cell when the coast masks the corners |

MUR is the finest gap-free global daily SST product, close to patch scale.
No gap-free wind, wave or rain product exists at patch scale for 2015–2021, so
ERA5 is the best available real source for those groups.

The S2 overpass time is estimated as 10:30 local mean solar time
(`10.5 h − lon/15` UTC), which is accurate to well under ERA5's 1-hour step.

No values are synthesised. Gaps that the documented fallbacks can't fill stay
NaN. RF, XGBoost and LightGBM handle NaN natively. For AdaBoost and the SVM,
NaN is replaced by the training-patch median, and the number of imputed
patches is reported in the results.

## Run order (from the repo root)

```powershell
python scripts/environmental/build_patch_metadata.py   # -> data/environmental/marida_patch_metadata.csv
python scripts/environmental/fetch_ocean_erddap.py     # -> data/environmental/raw/{mur_sst,chla_*}/  (no account needed)
python scripts/environmental/fetch_era5.py             # -> data/environmental/raw/era5/  (needs ~/.cdsapirc)
python scripts/environmental/build_env_table.py        # -> data/environmental/marida_env_features.csv
python scripts/environmental/run_env_experiments.py    # -> outputs/environmental/<eval>_<labels>/
```

Or run everything with `scripts/environmental/run_env_pipeline.ps1`.

## Experiments

`run_env_experiments.py` samples the pixels once. Each feature set (baseline,
`env_all`, each group alone, `env_all-<group>`) is then trained and tested on
exactly those pixels, so every delta is a paired comparison.

- `--eval official`: train split → test split. These splits are already
  scene-disjoint, but there are only 15 test scenes, so the result is noisy.
- `--eval scene-cv`: GroupKFold over all 63 scenes. Gives mean ± std and the
  number of folds in which the env features helped.
- `--label-protocol baseline`: classes 0–4, as in `scripts/baselines/`.
  Class 0 here means unlabelled + cloud + cloud shadow (see
  `scripts/utilities/convert.py`).
- `--label-protocol labelled`: annotated classes 1–4 only.

Outputs: `summary.md`, `summary.csv`, `results_all_folds.csv`,
`feature_importance_env_all.csv`, and confusion matrices.

**Interpreting results:** the environmental values only vary between scenes
(and slightly between patches), so the training set holds only about 36–50
distinct environmental "situations". Treat the scene-CV numbers as the main
evidence. A gain that shows up only on the single official split is weak
evidence.
