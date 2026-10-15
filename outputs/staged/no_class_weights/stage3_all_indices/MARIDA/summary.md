# stage3_all_indices: bands + all 5 indices (MARIDA)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8909 | 0.8192 | 0.8732 | 0.9125 | 0.9955 | 0.9006 | 358,442 | 131 |
| xgb | 0.8834 | 0.8063 | 0.8639 | 0.9089 | 0.9952 | 0.8948 | 358,442 | 15 |
| lgbm | 0.7329 | 0.6079 | 0.7218 | 0.7736 | 0.9864 | 0.7098 | 358,442 | 10 |
| ada | 0.8154 | 0.7275 | 0.8468 | 0.7976 | 0.9936 | 0.8477 | 358,442 | 425 |
| svm | 0.8140 | 0.7326 | 0.8891 | 0.7647 | 0.9944 | 0.8660 | 358,442 | 8 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7509 | 0.9861 | 0.8285 |
| xgb | 0.9980 | 0.7494 | 0.9676 | 0.8185 |
| lgbm | 0.9945 | 0.5955 | 0.7439 | 0.5977 |
| ada | 0.9972 | 0.5566 | 0.9753 | 0.7326 |
| svm | 0.9974 | 0.4917 | 0.9801 | 0.7866 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.6011 | 0.9727 | 0.7072 |
| xgb | 0.9961 | 0.5993 | 0.9372 | 0.6928 |
| lgbm | 0.9891 | 0.4240 | 0.5923 | 0.4262 |
| ada | 0.9945 | 0.3856 | 0.9517 | 0.5780 |
| svm | 0.9949 | 0.3260 | 0.9611 | 0.6483 |
