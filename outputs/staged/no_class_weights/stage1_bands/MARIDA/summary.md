# stage1_bands: 11 Sentinel-2 bands only (MARIDA)

Features (11): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8865 | 0.8119 | 0.8639 | 0.9161 | 0.9954 | 0.8992 | 358,442 | 14 |
| xgb | 0.8632 | 0.7815 | 0.8389 | 0.9024 | 0.9947 | 0.8846 | 358,442 | 5 |
| lgbm | 0.7260 | 0.6199 | 0.6766 | 0.8376 | 0.9863 | 0.7383 | 358,442 | 2 |
| ada | 0.7310 | 0.6381 | 0.8261 | 0.6671 | 0.9912 | 0.7754 | 358,442 | 94 |
| svm | 0.6120 | 0.5277 | 0.6469 | 0.6259 | 0.9850 | 0.6700 | 358,442 | 2 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7469 | 0.9770 | 0.8239 |
| xgb | 0.9980 | 0.6862 | 0.9723 | 0.7963 |
| lgbm | 0.9941 | 0.3562 | 0.8221 | 0.7315 |
| ada | 0.9959 | 0.3259 | 0.9136 | 0.6887 |
| svm | 0.9925 | 0.0276 | 0.6887 | 0.7392 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9962 | 0.5960 | 0.9550 | 0.7006 |
| xgb | 0.9959 | 0.5223 | 0.9461 | 0.6615 |
| lgbm | 0.9882 | 0.2167 | 0.6979 | 0.5767 |
| ada | 0.9918 | 0.1947 | 0.8409 | 0.5251 |
| svm | 0.9851 | 0.0140 | 0.5252 | 0.5863 |
