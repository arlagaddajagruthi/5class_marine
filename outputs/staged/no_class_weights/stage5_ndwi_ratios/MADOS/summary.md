# stage5_ndwi_ratios: bands + NDWI + 4 band ratios (MADOS)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7238 | 0.5814 | 0.7495 | 0.7581 | 0.7359 | 0.3905 | 761,887 | 79 |
| xgb | 0.7182 | 0.5764 | 0.7249 | 0.7585 | 0.7856 | 0.4542 | 761,887 | 9 |
| lgbm | 0.5050 | 0.3669 | 0.4457 | 0.7385 | 0.7445 | 0.4028 | 761,887 | 5 |
| ada | 0.6506 | 0.5176 | 0.7376 | 0.6108 | 0.7977 | 0.4015 | 761,887 | 622 |
| svm | 0.6550 | 0.5143 | 0.7985 | 0.5949 | 0.7731 | 0.3260 | 761,887 | 11 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8179 | 0.7217 | 0.8355 | 0.5201 |
| xgb | 0.8593 | 0.6208 | 0.8329 | 0.5597 |
| lgbm | 0.8265 | 0.2961 | 0.3479 | 0.5495 |
| ada | 0.8734 | 0.3915 | 0.8340 | 0.5036 |
| svm | 0.8568 | 0.5078 | 0.8106 | 0.4446 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6919 | 0.5646 | 0.7175 | 0.3515 |
| xgb | 0.7534 | 0.4502 | 0.7137 | 0.3886 |
| lgbm | 0.7043 | 0.1738 | 0.2106 | 0.3789 |
| ada | 0.7752 | 0.2434 | 0.7153 | 0.3366 |
| svm | 0.7495 | 0.3403 | 0.6815 | 0.2858 |
