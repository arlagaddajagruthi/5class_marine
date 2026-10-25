# Stage 9 SHAP: all spectral + all environmental features (MARIDA, log weights)

47 features: 11 bands, 5 indices, 4 ratios, 27 environmental covariates (both env CSVs in `data/`). Explained: 1200 class-stratified test pixels ({'Water': 300, 'Marine Debris': 300, 'Algae/Sargassum': 300, 'Other': 300}); background: 200 class-stratified training pixels.

## Test scores (all annotated test pixels)

| Model | macro F1 | IoU | Debris F1 | Algae F1 | Kappa |
|---|---|---|---|---|---|
| rf | 0.8508 | 0.7616 | 0.7033 | 0.9525 | 0.8580 |
| xgb | 0.6909 | 0.5975 | 0.7318 | 0.8751 | 0.3387 |
| lgbm | 0.6946 | 0.6143 | 0.7159 | 0.9482 | 0.2989 |
| ada | 0.7602 | 0.6655 | 0.6779 | 0.9487 | 0.6198 |
| svm | 0.6640 | 0.5468 | 0.3247 | 0.7281 | 0.6276 |

### Compared with earlier log-weight stages (Δ = stage 9 − reference)

| Model | Δ F1 vs stage4_indices_ratios | Δ Debris F1 vs stage4_indices_ratios | Δ F1 vs stage6_env | Δ Debris F1 vs stage6_env |
|---|---|---|---|---|
| rf | -0.0473 | -0.0817 | -0.0413 | -0.0878 |
| xgb | -0.1793 | -0.0008 | -0.1127 | -0.0174 |
| lgbm | -0.1821 | -0.0327 | -0.0378 | +0.2531 |
| ada | -0.0518 | +0.1625 | -0.0193 | +0.1158 |
| svm | -0.1439 | -0.3935 | -0.2130 | -0.3963 |

## Feature-group share of mean |SHAP| (all classes)

| Model | Bands | Indices | Ratios | Environment |
|---|---|---|---|---|
| rf | 27.7% | 24.1% | 16.9% | 31.3% |
| xgb | 24.8% | 24.7% | 12.4% | 38.1% |
| lgbm | 26.7% | 21.6% | 14.3% | 37.4% |
| ada | 20.4% | 25.7% | 22.6% | 31.3% |
| svm | 34.0% | 17.1% | 14.0% | 34.8% |

## Environmental share per class (Marine Debris / Algae rows matter most)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 26.1% | 32.5% | 28.2% | 30.7% |
| xgb | 19.8% | 23.6% | 57.9% | 39.8% |
| lgbm | 21.0% | 26.8% | 47.0% | 40.0% |
| ada | 20.8% | 28.9% | 35.0% | 34.1% |
| svm | 17.4% | 30.7% | 45.9% | 22.1% |

## Environmental sub-groups (share of total SHAP, all classes)

| Model | SST | Chlorophyll-a | Wind speed | Wind direction (u/v) | Pressure / precipitation | Waves / Stokes drift |
|---|---|---|---|---|---|---|
| rf | 2.4% | 5.9% | 6.8% | 7.9% | 3.0% | 5.2% |
| xgb | 2.3% | 11.6% | 7.1% | 8.6% | 3.7% | 4.7% |
| lgbm | 2.6% | 10.7% | 6.4% | 9.2% | 4.6% | 4.0% |
| ada | 3.0% | 11.5% | 6.3% | 6.2% | 2.0% | 2.3% |
| svm | 1.9% | 1.6% | 15.2% | 5.3% | 4.2% | 6.7% |

## Top 15 features (mean share over the 5 models)

| # | Feature | Group | Mean share | rf | xgb | lgbm | ada | svm |
|---|---|---|---|---|---|---|---|---|
| 1 | nir/red | Ratios | 6.9% | 6.6% | 3.9% | 5.8% | 14.5% | 3.6% |
| 2 | B11 | Bands | 6.6% | 6.9% | 8.3% | 6.4% | 5.4% | 5.9% |
| 3 | NDVI | Indices | 5.9% | 7.9% | 8.3% | 4.2% | 5.9% | 3.2% |
| 4 | log10_chl | Environment | 5.7% | 2.8% | 8.0% | 8.0% | 9.1% | 0.9% |
| 5 | FAI | Indices | 5.7% | 5.1% | 6.4% | 7.0% | 7.1% | 2.6% |
| 6 | NDWI | Indices | 4.2% | 4.7% | 3.7% | 3.7% | 2.5% | 6.5% |
| 7 | FDI | Indices | 3.8% | 3.3% | 4.1% | 3.3% | 6.2% | 2.4% |
| 8 | nir/green | Ratios | 3.8% | 5.5% | 2.2% | 2.4% | 1.9% | 6.8% |
| 9 | B2 | Bands | 3.6% | 3.2% | 3.8% | 4.2% | 4.5% | 2.3% |
| 10 | NDMI | Indices | 3.1% | 3.1% | 2.2% | 3.4% | 4.0% | 2.5% |
| 11 | red/green | Ratios | 3.0% | 1.9% | 4.4% | 4.3% | 3.7% | 0.5% |
| 12 | B3 | Bands | 2.5% | 2.3% | 1.8% | 2.8% | 2.7% | 3.2% |
| 13 | modis_log10_chla | Environment | 2.5% | 3.2% | 3.6% | 2.7% | 2.4% | 0.7% |
| 14 | swir1/nir | Ratios | 2.4% | 2.9% | 1.9% | 1.7% | 2.5% | 3.1% |
| 15 | B5 | Bands | 2.3% | 1.2% | 2.2% | 2.2% | 1.4% | 4.6% |

## Top 10 environmental features

| Feature | Mean share | Mean rank (of 47) |
|---|---|---|
| log10_chl | 5.75% | 10.4 |
| modis_log10_chla | 2.52% | 16.8 |
| sst_c | 1.69% | 21.4 |
| era5_v10 | 1.67% | 22.0 |
| era5_u10 | 1.48% | 23.6 |
| era5_msl_hpa | 1.46% | 24.6 |
| era5_wind_speed | 1.35% | 25.6 |
| wind_u | 1.32% | 27.4 |
| wind_speed | 1.21% | 28.6 |
| era5_v10_mean_72h | 1.20% | 29.4 |

## Model agreement

Spearman correlation between the models' global SHAP rankings: mean 0.57 (min 0.16). See `05_cross_model/`.

## SHAP validation

| Model | explainer | additivity max error | max |output| | seconds |
|---|---|---|---|---|
| rf | TreeExplainer (interventional TreeSHAP) | 7.96e-09 | 1.00 | 296 |
| xgb | TreeExplainer (interventional TreeSHAP) | 2.45e-06 | 6.31 | 164 |
| lgbm | TreeExplainer (interventional TreeSHAP) | 2.56e-07 | 13.55 | 98 |
| ada | TreeExplainer (interventional TreeSHAP) on exact tree rewrite of SAMME decision_function | 5.21e-09 | 0.75 | 17 |
| svm | LinearExplainer (interventional; exact for linear margin) | 1.85e-06 | 55.89 | 0 |

SHAP values are on each model's own output scale (RF: probability; XGB/LGBM: softmax margin; AdaBoost: SAMME score; SVM: one-vs-rest margin), so compare models through the normalised shares, not raw magnitudes.

Environmental values are constant within a 256×256 patch and MARIDA has only 63 scenes, so a high environmental share can mean the model uses them to recognise the scene rather than the conditions; check the test scores against stage 4 before reading it as a physical effect.

## Folders

- `00_samples/`: explained test pixels (features, true and predicted class) and the background pixels
- `01_shap_values/`: raw SHAP values per model, one row per pixel and class output
- `02_global_importance/`: global mean |SHAP| and share per feature and model, bar charts, heatmap
- `03_per_class/`: per-class rankings and beeswarm plots for every model
- `04_feature_groups/`: bands / indices / ratios / environment shares, environmental sub-groups
- `05_cross_model/`: ranking agreement between models and with built-in importance
- `06_dependence/`: Marine Debris dependence plots for the top environmental features
- `07_validation/`: additivity check, explainer settings, timings
- `../../training/MARIDA/`: test metrics, confusion matrices, built-in feature importance
