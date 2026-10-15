# stage2d_fdi: bands + FDI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FDI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7048 | 0.5590 | 0.7414 | 0.7296 | 0.7222 | 0.3675 | 761,887 | 35 |
| xgb | 0.6035 | 0.4577 | 0.5754 | 0.7905 | 0.7189 | 0.3783 | 761,887 | 6 |
| lgbm | 0.6218 | 0.4773 | 0.5996 | 0.7931 | 0.7205 | 0.3807 | 761,887 | 4 |
| ada | 0.5508 | 0.4157 | 0.5407 | 0.7546 | 0.7026 | 0.3504 | 761,887 | 360 |
| svm | 0.5370 | 0.4131 | 0.5370 | 0.6953 | 0.7108 | 0.3197 | 761,887 | 7 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8072 | 0.6771 | 0.8310 | 0.5040 |
| xgb | 0.8059 | 0.3414 | 0.7551 | 0.5116 |
| lgbm | 0.8064 | 0.3734 | 0.7941 | 0.5134 |
| ada | 0.7976 | 0.1903 | 0.7276 | 0.4878 |
| svm | 0.8053 | 0.1120 | 0.7561 | 0.4747 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6767 | 0.5118 | 0.7108 | 0.3369 |
| xgb | 0.6749 | 0.2058 | 0.6066 | 0.3437 |
| lgbm | 0.6757 | 0.2296 | 0.6585 | 0.3454 |
| ada | 0.6634 | 0.1052 | 0.5718 | 0.3225 |
| svm | 0.6740 | 0.0593 | 0.6079 | 0.3112 |
