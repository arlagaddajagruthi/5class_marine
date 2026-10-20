# stage8_ndwi_ratios_glcm: bands + NDWI + 4 band ratios + 6 GLCM texture features (MADOS)

Features (22): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7661 | 0.6341 | 0.7781 | 0.7749 | 0.8305 | 0.5215 | 761,887 | 166 |
| xgb | 0.7563 | 0.6243 | 0.7559 | 0.7816 | 0.8601 | 0.5806 | 761,887 | 17 |
| lgbm | 0.6678 | 0.5478 | 0.6567 | 0.7873 | 0.8546 | 0.5727 | 761,887 | 12 |
| ada | 0.6963 | 0.5754 | 0.7778 | 0.6429 | 0.8766 | 0.5339 | 761,887 | 1369 |
| svm | 0.6909 | 0.5579 | 0.8439 | 0.6140 | 0.8532 | 0.4683 | 761,887 | 29 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8516 | 0.7677 | 0.7430 | 0.7371 |
| xgb | 0.8901 | 0.7729 | 0.7355 | 0.7278 |
| lgbm | 0.8820 | 0.7831 | 0.7507 | 0.7335 |
| ada | 0.9650 | 0.3350 | 0.7948 | 0.4770 |
| svm | 0.9392 | 0.3433 | 0.7061 | 0.4677 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9380 | 0.6809 | 0.9812 | 0.5122 |
| xgb | 0.9387 | 0.5226 | 0.9780 | 0.5841 |
| lgbm | 0.9400 | 0.1680 | 0.9276 | 0.5911 |
| ada | 0.8938 | 0.5086 | 0.9665 | 0.7422 |
| svm | 0.8882 | 0.8789 | 0.9896 | 0.6189 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8516 | 0.7677 | 0.7430 | 0.7371 |
| xgb | 0.8901 | 0.7729 | 0.7355 | 0.7278 |
| lgbm | 0.8820 | 0.7831 | 0.7507 | 0.7335 |
| ada | 0.9650 | 0.3350 | 0.7948 | 0.4770 |
| svm | 0.9392 | 0.3433 | 0.7061 | 0.4677 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8927 | 0.7217 | 0.8456 | 0.6044 |
| xgb | 0.9138 | 0.6235 | 0.8396 | 0.6481 |
| lgbm | 0.9101 | 0.2766 | 0.8298 | 0.6547 |
| ada | 0.9280 | 0.4040 | 0.8723 | 0.5808 |
| svm | 0.9129 | 0.4937 | 0.8241 | 0.5328 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8063 | 0.5646 | 0.7325 | 0.4331 |
| xgb | 0.8412 | 0.4530 | 0.7236 | 0.4794 |
| lgbm | 0.8350 | 0.1605 | 0.7092 | 0.4866 |
| ada | 0.8657 | 0.2531 | 0.7735 | 0.4092 |
| svm | 0.8398 | 0.3278 | 0.7008 | 0.3631 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8334 | 0.9984 | 0.9977 | 0.8315 |
| xgb | 0.8632 | 0.9975 | 0.9976 | 0.8620 |
| lgbm | 0.8581 | 0.9890 | 0.9974 | 0.8649 |
| ada | 0.8781 | 0.9973 | 0.9980 | 0.8797 |
| svm | 0.8542 | 0.9981 | 0.9974 | 0.8568 |
