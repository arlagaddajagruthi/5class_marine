# stage1_bands: 11 Sentinel-2 bands only (MARIDA)

Features (11): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8848 | 0.8100 | 0.8717 | 0.9021 | 0.9954 | 0.8982 | 358,442 | 8 |
| xgb | 0.7967 | 0.7014 | 0.7314 | 0.9365 | 0.9899 | 0.8078 | 358,442 | 5 |
| lgbm | 0.8179 | 0.7240 | 0.7485 | 0.9532 | 0.9905 | 0.8185 | 358,442 | 2 |
| ada | 0.5770 | 0.4746 | 0.5214 | 0.8958 | 0.9225 | 0.3396 | 358,442 | 91 |
| svm | 0.5657 | 0.4702 | 0.5470 | 0.7943 | 0.9481 | 0.4135 | 358,442 | 3 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9979 | 0.8215 | 0.9677 | 0.8211 |
| xgb | 0.9918 | 0.9475 | 0.9488 | 0.8578 |
| lgbm | 0.9920 | 0.9816 | 0.9555 | 0.8839 |
| ada | 0.9239 | 0.9580 | 0.9738 | 0.7273 |
| svm | 0.9516 | 0.5748 | 0.9805 | 0.6702 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.6660 | 0.9851 | 0.8378 |
| xgb | 0.9997 | 0.3599 | 0.9323 | 0.6338 |
| lgbm | 0.9997 | 0.4284 | 0.9457 | 0.6200 |
| ada | 0.9998 | 0.0368 | 0.4631 | 0.5861 |
| svm | 0.9972 | 0.0630 | 0.2695 | 0.8584 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9979 | 0.8215 | 0.9677 | 0.8211 |
| xgb | 0.9918 | 0.9475 | 0.9488 | 0.8578 |
| lgbm | 0.9920 | 0.9816 | 0.9555 | 0.8839 |
| ada | 0.9239 | 0.9580 | 0.9738 | 0.7273 |
| svm | 0.9516 | 0.5748 | 0.9805 | 0.6702 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7356 | 0.9763 | 0.8294 |
| xgb | 0.9957 | 0.5217 | 0.9405 | 0.7290 |
| lgbm | 0.9958 | 0.5965 | 0.9506 | 0.7288 |
| ada | 0.9604 | 0.0708 | 0.6277 | 0.6491 |
| svm | 0.9739 | 0.1136 | 0.4228 | 0.7527 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.5818 | 0.9538 | 0.7085 |
| xgb | 0.9915 | 0.3529 | 0.8877 | 0.5735 |
| lgbm | 0.9917 | 0.4250 | 0.9058 | 0.5733 |
| ada | 0.9238 | 0.0367 | 0.4574 | 0.4805 |
| svm | 0.9490 | 0.0602 | 0.2681 | 0.6035 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.9986 | 0.9995 | 0.9966 |
| xgb | 0.9917 | 0.9958 | 0.9988 | 0.9935 |
| lgbm | 0.9919 | 0.9968 | 0.9990 | 0.9933 |
| ada | 0.9255 | 0.9395 | 0.9880 | 0.9920 |
| svm | 0.9501 | 0.9784 | 0.9723 | 0.9955 |
