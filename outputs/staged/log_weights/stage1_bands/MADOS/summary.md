# stage1_bands: 11 Sentinel-2 bands only (MADOS)

Features (11): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7097 | 0.5638 | 0.7521 | 0.7279 | 0.7290 | 0.3765 | 761,887 | 142 |
| xgb | 0.6625 | 0.5188 | 0.6550 | 0.7827 | 0.7478 | 0.4186 | 761,887 | 15 |
| lgbm | 0.6813 | 0.5383 | 0.6734 | 0.7890 | 0.7552 | 0.4299 | 761,887 | 9 |
| ada | 0.6124 | 0.4663 | 0.6305 | 0.6671 | 0.7202 | 0.3721 | 761,887 | 597 |
| svm | 0.5321 | 0.4131 | 0.5596 | 0.5735 | 0.7294 | 0.3186 | 761,887 | 7 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7127 | 0.6927 | 0.6989 | 0.8073 |
| xgb | 0.7272 | 0.8273 | 0.7326 | 0.8436 |
| lgbm | 0.7354 | 0.8294 | 0.7446 | 0.8467 |
| ada | 0.6975 | 0.4687 | 0.6698 | 0.8323 |
| svm | 0.7507 | 0.2580 | 0.6436 | 0.6418 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9459 | 0.7006 | 0.9890 | 0.3730 |
| xgb | 0.9611 | 0.3141 | 0.9469 | 0.3978 |
| lgbm | 0.9618 | 0.3603 | 0.9660 | 0.4054 |
| ada | 0.9518 | 0.2959 | 0.9043 | 0.3701 |
| svm | 0.9054 | 0.0506 | 0.9153 | 0.3671 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7127 | 0.6927 | 0.6989 | 0.8073 |
| xgb | 0.7272 | 0.8273 | 0.7326 | 0.8436 |
| lgbm | 0.7354 | 0.8294 | 0.7446 | 0.8467 |
| ada | 0.6975 | 0.4687 | 0.6698 | 0.8323 |
| svm | 0.7507 | 0.2580 | 0.6436 | 0.6418 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8129 | 0.6966 | 0.8190 | 0.5103 |
| xgb | 0.8279 | 0.4553 | 0.8261 | 0.5406 |
| lgbm | 0.8335 | 0.5023 | 0.8410 | 0.5483 |
| ada | 0.8050 | 0.3628 | 0.7696 | 0.5123 |
| svm | 0.8208 | 0.0846 | 0.7557 | 0.4670 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6848 | 0.5345 | 0.6935 | 0.3425 |
| xgb | 0.7064 | 0.2948 | 0.7037 | 0.3704 |
| lgbm | 0.7145 | 0.3354 | 0.7256 | 0.3777 |
| ada | 0.6737 | 0.2216 | 0.6255 | 0.3444 |
| svm | 0.6961 | 0.0442 | 0.6074 | 0.3047 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7329 | 0.9984 | 0.9974 | 0.7294 |
| xgb | 0.7539 | 0.9947 | 0.9974 | 0.7497 |
| lgbm | 0.7608 | 0.9956 | 0.9976 | 0.7564 |
| ada | 0.7250 | 0.9956 | 0.9966 | 0.7233 |
| svm | 0.7332 | 0.9850 | 0.9964 | 0.7442 |
