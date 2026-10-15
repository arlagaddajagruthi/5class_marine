# stage1_bands: 11 Sentinel-2 bands only (MADOS)

Features (11): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7022 | 0.5562 | 0.7441 | 0.7160 | 0.7253 | 0.3596 | 761,887 | 34 |
| xgb | 0.6018 | 0.4582 | 0.5585 | 0.8051 | 0.7507 | 0.4274 | 761,887 | 6 |
| lgbm | 0.6163 | 0.4693 | 0.5786 | 0.8005 | 0.7342 | 0.4029 | 761,887 | 3 |
| ada | 0.5024 | 0.3675 | 0.4642 | 0.7549 | 0.7057 | 0.3585 | 761,887 | 319 |
| svm | 0.5370 | 0.4131 | 0.5370 | 0.6953 | 0.7108 | 0.3197 | 761,887 | 6 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8114 | 0.6788 | 0.8228 | 0.4956 |
| xgb | 0.8314 | 0.3085 | 0.7181 | 0.5493 |
| lgbm | 0.8175 | 0.3677 | 0.7497 | 0.5304 |
| ada | 0.7987 | 0.1468 | 0.5591 | 0.5051 |
| svm | 0.8053 | 0.1120 | 0.7561 | 0.4747 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6827 | 0.5137 | 0.6990 | 0.3294 |
| xgb | 0.7114 | 0.1824 | 0.5602 | 0.3787 |
| lgbm | 0.6913 | 0.2253 | 0.5996 | 0.3609 |
| ada | 0.6649 | 0.0792 | 0.3880 | 0.3378 |
| svm | 0.6740 | 0.0593 | 0.6079 | 0.3112 |
