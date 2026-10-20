# stage8_ndwi_ratios_glcm: bands + NDWI + 4 band ratios + 6 GLCM texture features (MARIDA)

Features (22): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.9165 | 0.8551 | 0.9025 | 0.9337 | 0.9963 | 0.9174 | 358,442 | 11 |
| xgb | 0.8828 | 0.8036 | 0.8314 | 0.9522 | 0.9940 | 0.8776 | 358,442 | 5 |
| lgbm | 0.8855 | 0.8081 | 0.8343 | 0.9539 | 0.9941 | 0.8796 | 358,442 | 2 |
| ada | 0.7988 | 0.7105 | 0.7362 | 0.9360 | 0.9896 | 0.8029 | 358,442 | 221 |
| svm | 0.8472 | 0.7466 | 0.7800 | 0.9409 | 0.9909 | 0.8226 | 358,442 | 4 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.9108 | 0.9799 | 0.8460 |
| xgb | 0.9953 | 0.9344 | 0.9640 | 0.9149 |
| lgbm | 0.9954 | 0.9396 | 0.9744 | 0.9062 |
| ada | 0.9912 | 0.9081 | 0.9701 | 0.8745 |
| svm | 0.9924 | 0.9108 | 0.9951 | 0.8652 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9984 | 0.7643 | 0.9938 | 0.8534 |
| xgb | 0.9994 | 0.6605 | 0.9524 | 0.7133 |
| lgbm | 0.9993 | 0.6642 | 0.9546 | 0.7191 |
| ada | 0.9991 | 0.3337 | 0.9608 | 0.6512 |
| svm | 0.9989 | 0.6298 | 0.6994 | 0.7919 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.9108 | 0.9799 | 0.8460 |
| xgb | 0.9953 | 0.9344 | 0.9640 | 0.9149 |
| lgbm | 0.9954 | 0.9396 | 0.9744 | 0.9062 |
| ada | 0.9912 | 0.9081 | 0.9701 | 0.8745 |
| svm | 0.9924 | 0.9108 | 0.9951 | 0.8652 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9983 | 0.8311 | 0.9868 | 0.8497 |
| xgb | 0.9973 | 0.7739 | 0.9582 | 0.8016 |
| lgbm | 0.9973 | 0.7783 | 0.9644 | 0.8019 |
| ada | 0.9952 | 0.4880 | 0.9654 | 0.7466 |
| svm | 0.9956 | 0.7446 | 0.8214 | 0.8270 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9966 | 0.7111 | 0.9740 | 0.7386 |
| xgb | 0.9947 | 0.6312 | 0.9198 | 0.6689 |
| lgbm | 0.9947 | 0.6370 | 0.9313 | 0.6693 |
| ada | 0.9904 | 0.3228 | 0.9332 | 0.5956 |
| svm | 0.9913 | 0.5932 | 0.6970 | 0.7050 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9967 | 0.9991 | 0.9997 | 0.9970 |
| xgb | 0.9948 | 0.9987 | 0.9991 | 0.9954 |
| lgbm | 0.9948 | 0.9987 | 0.9993 | 0.9954 |
| ada | 0.9906 | 0.9954 | 0.9993 | 0.9940 |
| svm | 0.9915 | 0.9985 | 0.9955 | 0.9963 |
