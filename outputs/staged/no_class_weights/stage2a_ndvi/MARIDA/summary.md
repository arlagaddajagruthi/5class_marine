# stage2a_ndvi: bands + NDVI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8132 | 0.7357 | 0.7774 | 0.9071 | 0.9918 | 0.8327 | 358,442 | 46 |
| xgb | 0.8664 | 0.7834 | 0.8523 | 0.8845 | 0.9947 | 0.8837 | 358,442 | 8 |
| lgbm | 0.8474 | 0.7542 | 0.8082 | 0.8986 | 0.9933 | 0.8593 | 358,442 | 5 |
| ada | 0.7311 | 0.6549 | 0.7953 | 0.6885 | 0.9917 | 0.7961 | 358,442 | 190 |
| svm | 0.6888 | 0.6359 | 0.7901 | 0.6579 | 0.9928 | 0.8202 | 358,442 | 3 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9940 | 0.8294 | 0.9683 | 0.8366 |
| xgb | 0.9977 | 0.7848 | 0.9549 | 0.8006 |
| lgbm | 0.9959 | 0.8136 | 0.9464 | 0.8385 |
| ada | 0.9984 | 0.2362 | 0.9634 | 0.5559 |
| svm | 0.9991 | 0.0210 | 0.9519 | 0.6596 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.3147 | 0.9870 | 0.8095 |
| xgb | 0.9980 | 0.6362 | 0.9613 | 0.8138 |
| lgbm | 0.9984 | 0.5678 | 0.8725 | 0.7941 |
| ada | 0.9937 | 0.3147 | 0.9894 | 0.8835 |
| svm | 0.9939 | 0.2963 | 0.9501 | 0.9203 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9940 | 0.8294 | 0.9683 | 0.8366 |
| xgb | 0.9977 | 0.7848 | 0.9549 | 0.8006 |
| lgbm | 0.9959 | 0.8136 | 0.9464 | 0.8385 |
| ada | 0.9984 | 0.2362 | 0.9634 | 0.5559 |
| svm | 0.9991 | 0.0210 | 0.9519 | 0.6596 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.4563 | 0.9775 | 0.8228 |
| xgb | 0.9978 | 0.7027 | 0.9581 | 0.8071 |
| lgbm | 0.9971 | 0.6688 | 0.9079 | 0.8157 |
| ada | 0.9960 | 0.2699 | 0.9762 | 0.6824 |
| svm | 0.9965 | 0.0392 | 0.9510 | 0.7685 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9923 | 0.2956 | 0.9561 | 0.6990 |
| xgb | 0.9957 | 0.5417 | 0.9196 | 0.6766 |
| lgbm | 0.9943 | 0.5024 | 0.8314 | 0.6888 |
| ada | 0.9921 | 0.1560 | 0.9536 | 0.5179 |
| svm | 0.9930 | 0.0200 | 0.9066 | 0.6240 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9924 | 0.9952 | 0.9995 | 0.9963 |
| xgb | 0.9958 | 0.9984 | 0.9991 | 0.9961 |
| lgbm | 0.9944 | 0.9981 | 0.9980 | 0.9961 |
| ada | 0.9923 | 0.9969 | 0.9995 | 0.9947 |
| svm | 0.9931 | 0.9975 | 0.9990 | 0.9960 |
