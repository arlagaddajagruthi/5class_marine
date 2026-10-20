# stage8_ndwi_ratios_glcm: bands + NDWI + 4 band ratios + 6 GLCM texture features (MARIDA)

Features (22): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.9016 | 0.8348 | 0.8744 | 0.9382 | 0.9959 | 0.9114 | 358,442 | 49 |
| xgb | 0.8862 | 0.8087 | 0.8582 | 0.9196 | 0.9950 | 0.8934 | 358,442 | 10 |
| lgbm | 0.7741 | 0.6596 | 0.7149 | 0.8710 | 0.9889 | 0.7790 | 358,442 | 6 |
| ada | 0.8444 | 0.7655 | 0.8481 | 0.8412 | 0.9947 | 0.8811 | 358,442 | 461 |
| svm | 0.8946 | 0.8256 | 0.8961 | 0.8948 | 0.9961 | 0.9115 | 358,442 | 7 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9976 | 0.9003 | 0.9768 | 0.8783 |
| xgb | 0.9971 | 0.8556 | 0.9567 | 0.8689 |
| lgbm | 0.9926 | 0.8294 | 0.8800 | 0.7820 |
| ada | 0.9979 | 0.5853 | 0.9634 | 0.8180 |
| svm | 0.9985 | 0.7743 | 0.9902 | 0.8161 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9987 | 0.6609 | 0.9950 | 0.8432 |
| xgb | 0.9988 | 0.6936 | 0.9591 | 0.7811 |
| lgbm | 0.9981 | 0.4185 | 0.7612 | 0.6816 |
| ada | 0.9974 | 0.5603 | 0.9931 | 0.8415 |
| svm | 0.9979 | 0.7057 | 0.9848 | 0.8957 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9976 | 0.9003 | 0.9768 | 0.8783 |
| xgb | 0.9971 | 0.8556 | 0.9567 | 0.8689 |
| lgbm | 0.9926 | 0.8294 | 0.8800 | 0.7820 |
| ada | 0.9979 | 0.5853 | 0.9634 | 0.8180 |
| svm | 0.9985 | 0.7743 | 0.9902 | 0.8161 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7622 | 0.9859 | 0.8604 |
| xgb | 0.9979 | 0.7662 | 0.9579 | 0.8227 |
| lgbm | 0.9953 | 0.5563 | 0.8163 | 0.7284 |
| ada | 0.9976 | 0.5725 | 0.9780 | 0.8296 |
| svm | 0.9982 | 0.7384 | 0.9875 | 0.8541 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9963 | 0.6158 | 0.9721 | 0.7549 |
| xgb | 0.9959 | 0.6210 | 0.9192 | 0.6988 |
| lgbm | 0.9907 | 0.3854 | 0.6896 | 0.5728 |
| ada | 0.9952 | 0.4011 | 0.9570 | 0.7088 |
| svm | 0.9965 | 0.5853 | 0.9754 | 0.7453 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9964 | 0.9986 | 0.9997 | 0.9971 |
| xgb | 0.9960 | 0.9987 | 0.9991 | 0.9962 |
| lgbm | 0.9909 | 0.9968 | 0.9959 | 0.9941 |
| ada | 0.9953 | 0.9979 | 0.9996 | 0.9966 |
| svm | 0.9965 | 0.9987 | 0.9997 | 0.9972 |
