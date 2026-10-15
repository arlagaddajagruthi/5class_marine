# stage6_env: bands + NDWI + 4 band ratios + SST, chlorophyll-a, wind (MARIDA)

Features (23): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir, sst_c, sst_grad_c_per_km, log10_chl, wind_speed, wind_speed_mean_24h, wind_speed_mean_72h, wind_speed_max_72h

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8921 | 0.8196 | 0.8737 | 0.9148 | 0.9949 | 0.8894 | 358,442 | 78 |
| xgb | 0.8036 | 0.7067 | 0.7516 | 0.9155 | 0.9842 | 0.7205 | 358,442 | 18 |
| lgbm | 0.7324 | 0.6322 | 0.6731 | 0.9090 | 0.9808 | 0.6798 | 358,442 | 14 |
| ada | 0.7795 | 0.6820 | 0.7120 | 0.9207 | 0.9857 | 0.7435 | 358,442 | 500 |
| svm | 0.8770 | 0.7999 | 0.8462 | 0.9270 | 0.9950 | 0.8919 | 358,442 | 20 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9977 | 0.7910 | 0.9827 | 0.7971 |
| xgb | 0.9928 | 0.7492 | 0.9307 | 0.5419 |
| lgbm | 0.9910 | 0.4629 | 0.9309 | 0.5449 |
| ada | 0.9933 | 0.5620 | 0.9555 | 0.6070 |
| svm | 0.9979 | 0.7210 | 0.9790 | 0.8102 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9955 | 0.6543 | 0.9660 | 0.6626 |
| xgb | 0.9856 | 0.5990 | 0.8704 | 0.3716 |
| lgbm | 0.9822 | 0.3011 | 0.8708 | 0.3745 |
| ada | 0.9867 | 0.3909 | 0.9147 | 0.4358 |
| svm | 0.9959 | 0.5637 | 0.9590 | 0.6810 |
