# stage6_env: bands + NDWI + 4 band ratios + SST, chlorophyll-a, wind (MARIDA)

Features (23): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir, sst_c, sst_grad_c_per_km, log10_chl, wind_speed, wind_speed_mean_24h, wind_speed_mean_72h, wind_speed_max_72h

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.9033 | 0.8351 | 0.8915 | 0.9180 | 0.9956 | 0.9026 | 358,442 | 63 |
| xgb | 0.7705 | 0.6747 | 0.6945 | 0.9480 | 0.9809 | 0.6890 | 358,442 | 17 |
| lgbm | 0.7192 | 0.6081 | 0.6765 | 0.8846 | 0.9732 | 0.5978 | 358,442 | 12 |
| ada | 0.6025 | 0.5287 | 0.5568 | 0.9323 | 0.9305 | 0.3713 | 358,442 | 509 |
| svm | 0.8441 | 0.7590 | 0.7961 | 0.9383 | 0.9934 | 0.8636 | 358,442 | 19 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.8101 | 0.9815 | 0.8234 |
| xgb | 0.9907 | 0.6030 | 0.9636 | 0.5248 |
| lgbm | 0.9878 | 0.6127 | 0.8565 | 0.4199 |
| ada | 0.9638 | 0.0929 | 0.9281 | 0.4250 |
| svm | 0.9972 | 0.6190 | 0.9662 | 0.7941 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.6808 | 0.9636 | 0.6998 |
| xgb | 0.9815 | 0.4316 | 0.9298 | 0.3557 |
| lgbm | 0.9759 | 0.4417 | 0.7490 | 0.2657 |
| ada | 0.9302 | 0.0487 | 0.8659 | 0.2698 |
| svm | 0.9945 | 0.4482 | 0.9345 | 0.6586 |
