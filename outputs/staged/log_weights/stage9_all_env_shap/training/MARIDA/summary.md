# stage9_all_env_shap: bands + 5 indices + 4 band ratios + all environmental covariates (MARIDA)

Features (47): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI, red/green, nir/red, nir/green, swir1/nir, sst_c, sst_grad_c_per_km, log10_chl, wind_speed, wind_u, wind_v, wind_speed_mean_24h, wind_speed_mean_72h, wind_speed_max_72h, modis_log10_chla, era5_u10, era5_v10, era5_wind_speed, era5_wind_speed_mean_24h, era5_wind_speed_mean_72h, era5_wind_speed_max_72h, era5_u10_mean_72h, era5_v10_mean_72h, era5_msl_hpa, era5_precip_24h_mm, era5_precip_72h_mm, era5_precip_7d_mm, era5_swh, era5_swh_mean_72h, era5_stokes_u, era5_stokes_v, era5_stokes_speed

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8508 | 0.7616 | 0.8549 | 0.8483 | 0.9936 | 0.8580 | 358,442 | 31 |
| xgb | 0.6909 | 0.5975 | 0.6817 | 0.8640 | 0.9263 | 0.3387 | 358,442 | 25 |
| lgbm | 0.6946 | 0.6143 | 0.6695 | 0.8863 | 0.9106 | 0.2989 | 358,442 | 5 |
| ada | 0.7602 | 0.6655 | 0.7034 | 0.9118 | 0.9750 | 0.6198 | 358,442 | 335 |
| svm | 0.6640 | 0.5468 | 0.5690 | 0.8905 | 0.9757 | 0.6276 | 358,442 | 24 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9974 | 0.7060 | 0.9098 | 0.7801 |
| xgb | 0.9286 | 0.8845 | 0.7812 | 0.8615 |
| lgbm | 0.9115 | 0.8898 | 0.9086 | 0.8354 |
| ada | 0.9771 | 0.9003 | 0.9129 | 0.8571 |
| svm | 0.9778 | 0.7874 | 0.9659 | 0.8311 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9973 | 0.7005 | 0.9993 | 0.7223 |
| xgb | 0.9987 | 0.6241 | 0.9946 | 0.1095 |
| lgbm | 0.9985 | 0.5989 | 0.9914 | 0.0894 |
| ada | 0.9989 | 0.5436 | 0.9875 | 0.2837 |
| svm | 0.9990 | 0.2045 | 0.5842 | 0.4881 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9974 | 0.7060 | 0.9098 | 0.7801 |
| xgb | 0.9286 | 0.8845 | 0.7812 | 0.8615 |
| lgbm | 0.9115 | 0.8898 | 0.9086 | 0.8354 |
| ada | 0.9771 | 0.9003 | 0.9129 | 0.8571 |
| svm | 0.9778 | 0.7874 | 0.9659 | 0.8311 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9974 | 0.7033 | 0.9525 | 0.7501 |
| xgb | 0.9624 | 0.7318 | 0.8751 | 0.1943 |
| lgbm | 0.9530 | 0.7159 | 0.9482 | 0.1615 |
| ada | 0.9879 | 0.6779 | 0.9487 | 0.4263 |
| svm | 0.9883 | 0.3247 | 0.7281 | 0.6150 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9948 | 0.5423 | 0.9093 | 0.6001 |
| xgb | 0.9275 | 0.5771 | 0.7779 | 0.1076 |
| lgbm | 0.9102 | 0.5576 | 0.9015 | 0.0878 |
| ada | 0.9760 | 0.5127 | 0.9024 | 0.2709 |
| svm | 0.9768 | 0.1938 | 0.5724 | 0.4441 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9949 | 0.9986 | 0.9991 | 0.9947 |
| xgb | 0.9291 | 0.9984 | 0.9977 | 0.9274 |
| lgbm | 0.9122 | 0.9983 | 0.9990 | 0.9118 |
| ada | 0.9765 | 0.9979 | 0.9990 | 0.9765 |
| svm | 0.9773 | 0.9921 | 0.9925 | 0.9894 |
