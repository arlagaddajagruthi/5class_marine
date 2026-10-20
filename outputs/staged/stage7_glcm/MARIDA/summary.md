# stage7_glcm: bands + 6 GLCM texture features (MARIDA)

Features (17): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8789 | 0.8034 | 0.8691 | 0.8929 | 0.9955 | 0.9008 | 358,442 | 27 |
| xgb | 0.8040 | 0.7137 | 0.7406 | 0.9501 | 0.9902 | 0.8137 | 358,442 | 10 |
| lgbm | 0.8207 | 0.7312 | 0.7567 | 0.9528 | 0.9911 | 0.8284 | 358,442 | 7 |
| ada | 0.6110 | 0.5166 | 0.5505 | 0.9089 | 0.9361 | 0.3896 | 358,442 | 296 |
| svm | 0.5774 | 0.4756 | 0.5190 | 0.8444 | 0.9525 | 0.4488 | 358,442 | 8 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.7848 | 0.9665 | 0.8224 |
| xgb | 0.9917 | 0.9711 | 0.9512 | 0.8863 |
| lgbm | 0.9926 | 0.9711 | 0.9555 | 0.8919 |
| ada | 0.9373 | 0.9475 | 0.9665 | 0.7845 |
| svm | 0.9548 | 0.6745 | 0.9890 | 0.7590 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.6335 | 0.9857 | 0.8592 |
| xgb | 0.9997 | 0.3468 | 0.9630 | 0.6531 |
| lgbm | 0.9997 | 0.4035 | 0.9739 | 0.6498 |
| ada | 1.0000 | 0.0431 | 0.6712 | 0.4878 |
| svm | 0.9980 | 0.0785 | 0.3069 | 0.6927 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.7848 | 0.9665 | 0.8224 |
| xgb | 0.9917 | 0.9711 | 0.9512 | 0.8863 |
| lgbm | 0.9926 | 0.9711 | 0.9555 | 0.8919 |
| ada | 0.9373 | 0.9475 | 0.9665 | 0.7845 |
| svm | 0.9548 | 0.6745 | 0.9890 | 0.7590 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7011 | 0.9760 | 0.8404 |
| xgb | 0.9957 | 0.5110 | 0.9571 | 0.7520 |
| lgbm | 0.9961 | 0.5701 | 0.9646 | 0.7518 |
| ada | 0.9676 | 0.0825 | 0.7922 | 0.6016 |
| svm | 0.9759 | 0.1407 | 0.4684 | 0.7244 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9963 | 0.5397 | 0.9531 | 0.7247 |
| xgb | 0.9915 | 0.3432 | 0.9177 | 0.6026 |
| lgbm | 0.9923 | 0.3987 | 0.9317 | 0.6023 |
| ada | 0.9373 | 0.0430 | 0.6559 | 0.4302 |
| svm | 0.9530 | 0.0757 | 0.3058 | 0.5678 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9963 | 0.9984 | 0.9995 | 0.9968 |
| xgb | 0.9916 | 0.9955 | 0.9991 | 0.9941 |
| lgbm | 0.9924 | 0.9965 | 0.9993 | 0.9940 |
| ada | 0.9387 | 0.9493 | 0.9947 | 0.9894 |
| svm | 0.9540 | 0.9802 | 0.9767 | 0.9941 |
