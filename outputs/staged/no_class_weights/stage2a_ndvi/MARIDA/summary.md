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
