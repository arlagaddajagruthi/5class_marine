# stage7_glcm: bands + 6 GLCM texture features (MARIDA)

Features (17): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8872 | 0.8149 | 0.8767 | 0.9023 | 0.9958 | 0.9070 | 358,442 | 27 |
| xgb | 0.8454 | 0.7587 | 0.7916 | 0.9399 | 0.9932 | 0.8618 | 358,442 | 10 |
| lgbm | 0.8612 | 0.7792 | 0.8097 | 0.9439 | 0.9937 | 0.8715 | 358,442 | 7 |
| ada | 0.7415 | 0.6591 | 0.6933 | 0.8544 | 0.9869 | 0.7562 | 358,442 | 295 |
| svm | 0.6134 | 0.5122 | 0.5585 | 0.8004 | 0.9668 | 0.5333 | 358,442 | 7 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.8084 | 0.9640 | 0.8385 |
| xgb | 0.9950 | 0.9370 | 0.9512 | 0.8764 |
| lgbm | 0.9953 | 0.9344 | 0.9628 | 0.8832 |
| ada | 0.9894 | 0.6142 | 0.9433 | 0.8708 |
| svm | 0.9699 | 0.4724 | 0.9817 | 0.7776 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.6484 | 0.9950 | 0.8654 |
| xgb | 0.9994 | 0.4811 | 0.9702 | 0.7155 |
| lgbm | 0.9993 | 0.5378 | 0.9875 | 0.7142 |
| ada | 0.9987 | 0.1915 | 0.9479 | 0.6350 |
| svm | 0.9972 | 0.1065 | 0.3526 | 0.7776 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.8084 | 0.9640 | 0.8385 |
| xgb | 0.9950 | 0.9370 | 0.9512 | 0.8764 |
| lgbm | 0.9953 | 0.9344 | 0.9628 | 0.8832 |
| ada | 0.9894 | 0.6142 | 0.9433 | 0.8708 |
| svm | 0.9699 | 0.4724 | 0.9817 | 0.7776 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.7196 | 0.9793 | 0.8517 |
| xgb | 0.9972 | 0.6358 | 0.9606 | 0.7878 |
| lgbm | 0.9973 | 0.6826 | 0.9750 | 0.7898 |
| ada | 0.9940 | 0.2920 | 0.9456 | 0.7344 |
| svm | 0.9833 | 0.1738 | 0.5188 | 0.7776 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9964 | 0.5620 | 0.9594 | 0.7418 |
| xgb | 0.9944 | 0.4661 | 0.9242 | 0.6499 |
| lgbm | 0.9947 | 0.5182 | 0.9512 | 0.6526 |
| ada | 0.9882 | 0.1709 | 0.8969 | 0.5803 |
| svm | 0.9672 | 0.0952 | 0.3503 | 0.6362 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9965 | 0.9985 | 0.9996 | 0.9970 |
| xgb | 0.9946 | 0.9974 | 0.9992 | 0.9952 |
| lgbm | 0.9948 | 0.9979 | 0.9995 | 0.9952 |
| ada | 0.9884 | 0.9928 | 0.9989 | 0.9936 |
| svm | 0.9679 | 0.9892 | 0.9811 | 0.9955 |
