# stage8_ndwi_ratios_glcm: bands + NDWI + 4 band ratios + 6 GLCM texture features (MARIDA)

Features (22): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.9144 | 0.8516 | 0.9033 | 0.9282 | 0.9960 | 0.9118 | 358,442 | 34 |
| xgb | 0.8424 | 0.7516 | 0.7696 | 0.9634 | 0.9914 | 0.8342 | 358,442 | 12 |
| lgbm | 0.8568 | 0.7699 | 0.7928 | 0.9540 | 0.9926 | 0.8535 | 358,442 | 8 |
| ada | 0.6559 | 0.5824 | 0.6069 | 0.9469 | 0.9466 | 0.4390 | 358,442 | 442 |
| svm | 0.7802 | 0.6617 | 0.6848 | 0.9547 | 0.9846 | 0.7343 | 358,442 | 9 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.9081 | 0.9793 | 0.8273 |
| xgb | 0.9923 | 0.9659 | 0.9744 | 0.9211 |
| lgbm | 0.9938 | 0.9396 | 0.9738 | 0.9087 |
| ada | 0.9469 | 0.9843 | 0.9799 | 0.8764 |
| svm | 0.9856 | 0.9606 | 0.9982 | 0.8745 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.7775 | 0.9883 | 0.8489 |
| xgb | 0.9996 | 0.5090 | 0.9286 | 0.6412 |
| lgbm | 0.9995 | 0.5559 | 0.9389 | 0.6770 |
| ada | 0.9995 | 0.0539 | 0.9147 | 0.4593 |
| svm | 0.9993 | 0.4716 | 0.5404 | 0.7280 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.9081 | 0.9793 | 0.8273 |
| xgb | 0.9923 | 0.9659 | 0.9744 | 0.9211 |
| lgbm | 0.9938 | 0.9396 | 0.9738 | 0.9087 |
| ada | 0.9469 | 0.9843 | 0.9799 | 0.8764 |
| svm | 0.9856 | 0.9606 | 0.9982 | 0.8745 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.8378 | 0.9838 | 0.8380 |
| xgb | 0.9960 | 0.6667 | 0.9509 | 0.7561 |
| lgbm | 0.9966 | 0.6985 | 0.9560 | 0.7759 |
| ada | 0.9725 | 0.1022 | 0.9462 | 0.6027 |
| svm | 0.9924 | 0.6327 | 0.7012 | 0.7946 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9964 | 0.7208 | 0.9681 | 0.7212 |
| xgb | 0.9920 | 0.5000 | 0.9065 | 0.6078 |
| lgbm | 0.9933 | 0.5367 | 0.9158 | 0.6339 |
| ada | 0.9464 | 0.0539 | 0.8978 | 0.4314 |
| svm | 0.9849 | 0.4627 | 0.5399 | 0.6592 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9965 | 0.9992 | 0.9997 | 0.9967 |
| xgb | 0.9922 | 0.9977 | 0.9990 | 0.9940 |
| lgbm | 0.9934 | 0.9980 | 0.9991 | 0.9947 |
| ada | 0.9477 | 0.9584 | 0.9988 | 0.9883 |
| svm | 0.9853 | 0.9973 | 0.9912 | 0.9954 |
