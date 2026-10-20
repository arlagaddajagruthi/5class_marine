# stage8_ndwi_ratios_glcm: bands + NDWI + 4 band ratios + 6 GLCM texture features (MADOS)

Features (22): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7609 | 0.6271 | 0.7754 | 0.7663 | 0.8375 | 0.5296 | 761,887 | 51 |
| xgb | 0.7360 | 0.6062 | 0.7021 | 0.8323 | 0.8481 | 0.5707 | 761,887 | 8 |
| lgbm | 0.7364 | 0.6058 | 0.7053 | 0.8277 | 0.8448 | 0.5628 | 761,887 | 5 |
| ada | 0.6016 | 0.4805 | 0.5769 | 0.7686 | 0.8108 | 0.4820 | 761,887 | 1168 |
| svm | 0.7140 | 0.5777 | 0.7262 | 0.7229 | 0.7768 | 0.3995 | 761,887 | 38 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8646 | 0.7636 | 0.7187 | 0.7181 |
| xgb | 0.8645 | 0.8808 | 0.8109 | 0.7728 |
| lgbm | 0.8611 | 0.8767 | 0.8028 | 0.7700 |
| ada | 0.8346 | 0.7708 | 0.7663 | 0.7029 |
| svm | 0.8014 | 0.6351 | 0.7915 | 0.6636 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9348 | 0.6593 | 0.9797 | 0.5278 |
| xgb | 0.9498 | 0.3792 | 0.9292 | 0.5502 |
| lgbm | 0.9487 | 0.3904 | 0.9390 | 0.5429 |
| ada | 0.9336 | 0.0926 | 0.7728 | 0.5087 |
| svm | 0.9164 | 0.5965 | 0.9753 | 0.4165 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8646 | 0.7636 | 0.7187 | 0.7181 |
| xgb | 0.8645 | 0.8808 | 0.8109 | 0.7728 |
| lgbm | 0.8611 | 0.8767 | 0.8028 | 0.7700 |
| ada | 0.8346 | 0.7708 | 0.7663 | 0.7029 |
| svm | 0.8014 | 0.6351 | 0.7915 | 0.6636 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8984 | 0.7076 | 0.8291 | 0.6084 |
| xgb | 0.9052 | 0.5302 | 0.8660 | 0.6428 |
| lgbm | 0.9028 | 0.5402 | 0.8656 | 0.6368 |
| ada | 0.8813 | 0.1654 | 0.7695 | 0.5902 |
| svm | 0.8550 | 0.6152 | 0.8738 | 0.5118 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8155 | 0.5475 | 0.7081 | 0.4372 |
| xgb | 0.8268 | 0.3607 | 0.7637 | 0.4736 |
| lgbm | 0.8228 | 0.3701 | 0.7631 | 0.4671 |
| ada | 0.7878 | 0.0902 | 0.6254 | 0.4187 |
| svm | 0.7468 | 0.4443 | 0.7759 | 0.3439 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8407 | 0.9983 | 0.9975 | 0.8386 |
| xgb | 0.8525 | 0.9958 | 0.9979 | 0.8500 |
| lgbm | 0.8491 | 0.9960 | 0.9979 | 0.8466 |
| ada | 0.8170 | 0.9790 | 0.9961 | 0.8296 |
| svm | 0.7787 | 0.9979 | 0.9980 | 0.7789 |
