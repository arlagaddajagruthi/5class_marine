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
