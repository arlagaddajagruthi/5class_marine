# stage2f_ratios: bands + 4 band ratios (MARIDA)

Features (15): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8165 | 0.7409 | 0.7792 | 0.9261 | 0.9916 | 0.8311 | 358,442 | 50 |
| xgb | 0.8843 | 0.8082 | 0.8647 | 0.9101 | 0.9951 | 0.8924 | 358,442 | 7 |
| lgbm | 0.7106 | 0.6059 | 0.6438 | 0.8854 | 0.9819 | 0.6851 | 358,442 | 6 |
| ada | 0.8318 | 0.7452 | 0.8558 | 0.8261 | 0.9939 | 0.8568 | 358,442 | 265 |
| svm | 0.8122 | 0.7313 | 0.8820 | 0.7647 | 0.9943 | 0.8633 | 358,442 | 4 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.4601 | 0.9862 | 0.8236 |
| xgb | 0.9980 | 0.7592 | 0.9759 | 0.8041 |
| lgbm | 0.9914 | 0.3433 | 0.8601 | 0.6476 |
| ada | 0.9973 | 0.6020 | 0.9754 | 0.7526 |
| svm | 0.9974 | 0.4861 | 0.9838 | 0.7814 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9921 | 0.2988 | 0.9727 | 0.7001 |
| xgb | 0.9959 | 0.6118 | 0.9529 | 0.6723 |
| lgbm | 0.9830 | 0.2072 | 0.7545 | 0.4788 |
| ada | 0.9947 | 0.4306 | 0.9519 | 0.6034 |
| svm | 0.9947 | 0.3211 | 0.9681 | 0.6412 |
