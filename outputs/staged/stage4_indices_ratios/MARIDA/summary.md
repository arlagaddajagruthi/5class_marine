# stage4_indices_ratios: bands + all 5 indices + 4 band ratios (MARIDA)

Features (20): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8995 | 0.8300 | 0.8867 | 0.9173 | 0.9955 | 0.9012 | 358,442 | 42 |
| xgb | 0.8367 | 0.7424 | 0.7611 | 0.9621 | 0.9907 | 0.8234 | 358,442 | 10 |
| lgbm | 0.8458 | 0.7543 | 0.7732 | 0.9626 | 0.9915 | 0.8367 | 358,442 | 7 |
| ada | 0.6713 | 0.6019 | 0.6310 | 0.9345 | 0.9464 | 0.4371 | 358,442 | 475 |
| svm | 0.7661 | 0.6464 | 0.6872 | 0.9351 | 0.9822 | 0.6994 | 358,442 | 9 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.8950 | 0.9756 | 0.8006 |
| xgb | 0.9918 | 0.9790 | 0.9707 | 0.9068 |
| lgbm | 0.9926 | 0.9711 | 0.9726 | 0.9143 |
| ada | 0.9470 | 0.9501 | 0.9762 | 0.8646 |
| svm | 0.9841 | 0.9685 | 0.9872 | 0.8006 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7209 | 0.9901 | 0.8376 |
| xgb | 0.9996 | 0.5188 | 0.9108 | 0.6153 |
| lgbm | 0.9996 | 0.5316 | 0.9172 | 0.6442 |
| ada | 0.9995 | 0.0484 | 0.9352 | 0.5408 |
| svm | 0.9988 | 0.4817 | 0.4544 | 0.8138 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.8950 | 0.9756 | 0.8006 |
| xgb | 0.9918 | 0.9790 | 0.9707 | 0.9068 |
| lgbm | 0.9926 | 0.9711 | 0.9726 | 0.9143 |
| ada | 0.9470 | 0.9501 | 0.9762 | 0.8646 |
| svm | 0.9841 | 0.9685 | 0.9872 | 0.8006 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7986 | 0.9828 | 0.8187 |
| xgb | 0.9957 | 0.6782 | 0.9398 | 0.7331 |
| lgbm | 0.9961 | 0.6871 | 0.9441 | 0.7558 |
| ada | 0.9725 | 0.0921 | 0.9553 | 0.6654 |
| svm | 0.9914 | 0.6434 | 0.6224 | 0.8071 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.6647 | 0.9662 | 0.6930 |
| xgb | 0.9915 | 0.5131 | 0.8865 | 0.5787 |
| lgbm | 0.9922 | 0.5233 | 0.8941 | 0.6075 |
| ada | 0.9465 | 0.0483 | 0.9144 | 0.4986 |
| svm | 0.9830 | 0.4743 | 0.4518 | 0.6766 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.9989 | 0.9996 | 0.9964 |
| xgb | 0.9917 | 0.9978 | 0.9987 | 0.9933 |
| lgbm | 0.9924 | 0.9979 | 0.9988 | 0.9940 |
| ada | 0.9477 | 0.9549 | 0.9991 | 0.9912 |
| svm | 0.9834 | 0.9974 | 0.9876 | 0.9961 |
