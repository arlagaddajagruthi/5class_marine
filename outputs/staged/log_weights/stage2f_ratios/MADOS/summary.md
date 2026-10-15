# stage2f_ratios: bands + 4 band ratios (MADOS)

Features (15): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7217 | 0.5786 | 0.7553 | 0.7480 | 0.7326 | 0.3836 | 761,887 | 142 |
| xgb | 0.6833 | 0.5399 | 0.6700 | 0.7956 | 0.7709 | 0.4472 | 761,887 | 15 |
| lgbm | 0.6962 | 0.5533 | 0.6825 | 0.8000 | 0.7638 | 0.4397 | 761,887 | 11 |
| ada | 0.5678 | 0.4377 | 0.5487 | 0.7577 | 0.7353 | 0.3903 | 761,887 | 39293 |
| svm | 0.6647 | 0.5254 | 0.6620 | 0.7005 | 0.7510 | 0.3552 | 761,887 | 15 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8155 | 0.7286 | 0.8275 | 0.5152 |
| xgb | 0.8472 | 0.5037 | 0.8245 | 0.5579 |
| lgbm | 0.8407 | 0.5475 | 0.8429 | 0.5535 |
| ada | 0.8197 | 0.1684 | 0.7526 | 0.5304 |
| svm | 0.8364 | 0.4893 | 0.8525 | 0.4805 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6885 | 0.5731 | 0.7057 | 0.3469 |
| xgb | 0.7349 | 0.3367 | 0.7014 | 0.3869 |
| lgbm | 0.7252 | 0.3769 | 0.7285 | 0.3827 |
| ada | 0.6945 | 0.0919 | 0.6033 | 0.3609 |
| svm | 0.7188 | 0.3239 | 0.7429 | 0.3162 |
