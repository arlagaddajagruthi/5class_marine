# stage4_indices_ratios: bands + all 5 indices + 4 band ratios (MADOS)

Features (20): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7125 | 0.5676 | 0.7352 | 0.7543 | 0.7321 | 0.3865 | 761,887 | 346 |
| xgb | 0.7155 | 0.5726 | 0.7262 | 0.7528 | 0.7772 | 0.4432 | 761,887 | 27 |
| lgbm | 0.6772 | 0.5229 | 0.6569 | 0.7389 | 0.7641 | 0.4290 | 761,887 | 19 |
| ada | 0.6656 | 0.5325 | 0.7059 | 0.6480 | 0.7989 | 0.4344 | 761,887 | 1184 |
| svm | 0.6959 | 0.5577 | 0.7728 | 0.6525 | 0.7612 | 0.3161 | 761,887 | 22 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8148 | 0.6872 | 0.8305 | 0.5176 |
| xgb | 0.8524 | 0.6261 | 0.8300 | 0.5533 |
| lgbm | 0.8416 | 0.6163 | 0.7031 | 0.5479 |
| ada | 0.8726 | 0.4075 | 0.8470 | 0.5351 |
| svm | 0.8477 | 0.6579 | 0.8403 | 0.4378 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6875 | 0.5235 | 0.7102 | 0.3492 |
| xgb | 0.7428 | 0.4557 | 0.7094 | 0.3825 |
| lgbm | 0.7266 | 0.4454 | 0.5421 | 0.3774 |
| ada | 0.7740 | 0.2559 | 0.7346 | 0.3653 |
| svm | 0.7356 | 0.4902 | 0.7246 | 0.2803 |
