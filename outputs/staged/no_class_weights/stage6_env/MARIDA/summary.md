# stage6_env: bands + NDWI + 4 band ratios + SST, chlorophyll-a, wind (MARIDA)

Features (23): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir, sst_c, sst_grad_c_per_km, log10_chl, wind_speed, wind_speed_mean_24h, wind_speed_mean_72h, wind_speed_max_72h

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7987 | 0.7236 | 0.7625 | 0.9188 | 0.9901 | 0.8063 | 358,442 | 120 |
| xgb | 0.7664 | 0.6668 | 0.7420 | 0.8559 | 0.9777 | 0.6325 | 358,442 | 17 |
| lgbm | 0.6781 | 0.5922 | 0.6232 | 0.8741 | 0.9700 | 0.5681 | 358,442 | 12 |
| ada | 0.7994 | 0.7024 | 0.8136 | 0.8167 | 0.9930 | 0.8396 | 358,442 | 435 |
| svm | 0.8830 | 0.8077 | 0.9033 | 0.8686 | 0.9955 | 0.8953 | 358,442 | 15 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9923 | 0.8793 | 0.9707 | 0.8329 |
| xgb | 0.9815 | 0.8189 | 0.8251 | 0.7981 |
| lgbm | 0.9726 | 0.7848 | 0.9324 | 0.8068 |
| ada | 0.9983 | 0.7244 | 0.8391 | 0.7050 |
| svm | 0.9991 | 0.7769 | 0.9781 | 0.7205 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9983 | 0.2619 | 0.9938 | 0.7958 |
| xgb | 0.9981 | 0.6842 | 0.9919 | 0.2938 |
| lgbm | 0.9982 | 0.2347 | 0.9884 | 0.2716 |
| ada | 0.9967 | 0.3966 | 0.9885 | 0.8724 |
| svm | 0.9970 | 0.7202 | 0.9877 | 0.9084 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9923 | 0.8793 | 0.9707 | 0.8329 |
| xgb | 0.9815 | 0.8189 | 0.8251 | 0.7981 |
| lgbm | 0.9726 | 0.7848 | 0.9324 | 0.8068 |
| ada | 0.9983 | 0.7244 | 0.8391 | 0.7050 |
| svm | 0.9991 | 0.7769 | 0.9781 | 0.7205 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9953 | 0.4036 | 0.9821 | 0.8140 |
| xgb | 0.9897 | 0.7455 | 0.9009 | 0.4296 |
| lgbm | 0.9852 | 0.3613 | 0.9595 | 0.4064 |
| ada | 0.9975 | 0.5125 | 0.9077 | 0.7798 |
| svm | 0.9980 | 0.7475 | 0.9829 | 0.8036 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9906 | 0.2528 | 0.9649 | 0.6863 |
| xgb | 0.9797 | 0.5943 | 0.8196 | 0.2735 |
| lgbm | 0.9709 | 0.2205 | 0.9222 | 0.2550 |
| ada | 0.9950 | 0.3446 | 0.8310 | 0.6391 |
| svm | 0.9961 | 0.5968 | 0.9663 | 0.6717 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9908 | 0.9937 | 0.9996 | 0.9961 |
| xgb | 0.9801 | 0.9987 | 0.9981 | 0.9784 |
| lgbm | 0.9715 | 0.9933 | 0.9992 | 0.9760 |
| ada | 0.9951 | 0.9967 | 0.9982 | 0.9960 |
| svm | 0.9962 | 0.9987 | 0.9996 | 0.9964 |
