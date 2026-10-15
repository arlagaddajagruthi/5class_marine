# stage1_bands: 11 Sentinel-2 bands only (MADOS)

Features (11): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7145 | 0.5704 | 0.7552 | 0.7332 | 0.7259 | 0.3705 | 761,887 | 81 |
| xgb | 0.6998 | 0.5552 | 0.7145 | 0.7404 | 0.7753 | 0.4493 | 761,887 | 15 |
| lgbm | 0.6780 | 0.5354 | 0.6858 | 0.7314 | 0.7750 | 0.4494 | 761,887 | 11 |
| ada | 0.5733 | 0.4383 | 0.7273 | 0.5542 | 0.7572 | 0.3992 | 761,887 | 532 |
| svm | 0.4917 | 0.3849 | 0.5700 | 0.4559 | 0.7852 | 0.3212 | 761,887 | 8 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8104 | 0.7133 | 0.8287 | 0.5057 |
| xgb | 0.8499 | 0.5669 | 0.8207 | 0.5616 |
| lgbm | 0.8502 | 0.4791 | 0.8217 | 0.5609 |
| ada | 0.8367 | 0.2126 | 0.7119 | 0.5319 |
| svm | 0.8673 | 0.0000 | 0.6586 | 0.4411 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6813 | 0.5543 | 0.7075 | 0.3384 |
| xgb | 0.7390 | 0.3955 | 0.6960 | 0.3904 |
| lgbm | 0.7395 | 0.3150 | 0.6973 | 0.3898 |
| ada | 0.7192 | 0.1189 | 0.5527 | 0.3624 |
| svm | 0.7656 | 0.0000 | 0.4910 | 0.2830 |
