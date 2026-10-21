# stage5_ndwi_ratios: bands + NDWI + 4 band ratios (MADOS)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7165 | 0.5725 | 0.7429 | 0.7489 | 0.7357 | 0.3854 | 761,887 | 53 |
| xgb | 0.6254 | 0.4805 | 0.5894 | 0.8079 | 0.7460 | 0.4132 | 761,887 | 7 |
| lgbm | 0.6569 | 0.5125 | 0.6260 | 0.8121 | 0.7488 | 0.4188 | 761,887 | 4 |
| ada | 0.5207 | 0.3884 | 0.4869 | 0.7816 | 0.7120 | 0.3540 | 761,887 | 754 |
| svm | 0.6621 | 0.5255 | 0.6241 | 0.7935 | 0.7519 | 0.3803 | 761,887 | 12 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7212 | 0.7575 | 0.7129 | 0.8039 |
| xgb | 0.7285 | 0.9147 | 0.7643 | 0.8243 |
| lgbm | 0.7297 | 0.9044 | 0.7805 | 0.8338 |
| ada | 0.7022 | 0.9096 | 0.7627 | 0.7519 |
| svm | 0.7619 | 0.8613 | 0.8521 | 0.6988 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9462 | 0.6557 | 0.9906 | 0.3792 |
| xgb | 0.9583 | 0.2326 | 0.7693 | 0.3976 |
| lgbm | 0.9596 | 0.2949 | 0.8504 | 0.3991 |
| ada | 0.9440 | 0.0719 | 0.5549 | 0.3767 |
| svm | 0.9274 | 0.3035 | 0.8782 | 0.3872 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7212 | 0.7575 | 0.7129 | 0.8039 |
| xgb | 0.7285 | 0.9147 | 0.7643 | 0.8243 |
| lgbm | 0.7297 | 0.9044 | 0.7805 | 0.8338 |
| ada | 0.7022 | 0.9096 | 0.7627 | 0.7519 |
| svm | 0.7619 | 0.8613 | 0.8521 | 0.6988 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8185 | 0.7029 | 0.8291 | 0.5153 |
| xgb | 0.8277 | 0.3708 | 0.7668 | 0.5364 |
| lgbm | 0.8290 | 0.4448 | 0.8140 | 0.5398 |
| ada | 0.8053 | 0.1333 | 0.6424 | 0.5020 |
| svm | 0.8365 | 0.4488 | 0.8649 | 0.4983 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6928 | 0.5419 | 0.7080 | 0.3471 |
| xgb | 0.7061 | 0.2276 | 0.6218 | 0.3665 |
| lgbm | 0.7080 | 0.2860 | 0.6863 | 0.3697 |
| ada | 0.6741 | 0.0714 | 0.4732 | 0.3351 |
| svm | 0.7190 | 0.2894 | 0.7620 | 0.3318 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7397 | 0.9983 | 0.9975 | 0.7359 |
| xgb | 0.7532 | 0.9916 | 0.9960 | 0.7512 |
| lgbm | 0.7550 | 0.9939 | 0.9969 | 0.7518 |
| ada | 0.7236 | 0.9681 | 0.9927 | 0.7394 |
| svm | 0.7576 | 0.9943 | 0.9977 | 0.7542 |
