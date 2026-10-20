# stage7_glcm: bands + 6 GLCM texture features (MADOS)

Features (17): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7547 | 0.6222 | 0.7730 | 0.7512 | 0.8501 | 0.5528 | 761,887 | 133 |
| xgb | 0.7085 | 0.5769 | 0.6790 | 0.8048 | 0.8485 | 0.5691 | 761,887 | 16 |
| lgbm | 0.6860 | 0.5572 | 0.6647 | 0.7969 | 0.8332 | 0.5378 | 761,887 | 11 |
| ada | 0.6685 | 0.5377 | 0.6725 | 0.6871 | 0.8614 | 0.5689 | 761,887 | 935 |
| svm | 0.5712 | 0.4517 | 0.5977 | 0.6026 | 0.7648 | 0.3732 | 761,887 | 28 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8819 | 0.6742 | 0.7388 | 0.7099 |
| xgb | 0.8672 | 0.8109 | 0.7753 | 0.7658 |
| lgbm | 0.8490 | 0.8068 | 0.7692 | 0.7628 |
| ada | 0.9037 | 0.4872 | 0.6785 | 0.6791 |
| svm | 0.7920 | 0.2980 | 0.6708 | 0.6497 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9338 | 0.6108 | 0.9883 | 0.5589 |
| xgb | 0.9479 | 0.3125 | 0.9004 | 0.5553 |
| lgbm | 0.9464 | 0.2506 | 0.9373 | 0.5245 |
| ada | 0.9300 | 0.2764 | 0.8836 | 0.6001 |
| svm | 0.9109 | 0.0907 | 0.9801 | 0.4089 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8819 | 0.6742 | 0.7388 | 0.7099 |
| xgb | 0.8672 | 0.8109 | 0.7753 | 0.7658 |
| lgbm | 0.8490 | 0.8068 | 0.7692 | 0.7628 |
| ada | 0.9037 | 0.4872 | 0.6785 | 0.6791 |
| svm | 0.7920 | 0.2980 | 0.6708 | 0.6497 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9071 | 0.6409 | 0.8455 | 0.6254 |
| xgb | 0.9057 | 0.4511 | 0.8332 | 0.6438 |
| lgbm | 0.8951 | 0.3825 | 0.8450 | 0.6216 |
| ada | 0.9167 | 0.3527 | 0.7676 | 0.6371 |
| svm | 0.8473 | 0.1391 | 0.7965 | 0.5019 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8300 | 0.4716 | 0.7323 | 0.4550 |
| xgb | 0.8277 | 0.2913 | 0.7141 | 0.4747 |
| lgbm | 0.8101 | 0.2364 | 0.7315 | 0.4510 |
| ada | 0.8462 | 0.2141 | 0.6229 | 0.4675 |
| svm | 0.7351 | 0.0747 | 0.6618 | 0.3351 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8530 | 0.9980 | 0.9977 | 0.8515 |
| xgb | 0.8530 | 0.9947 | 0.9973 | 0.8520 |
| lgbm | 0.8379 | 0.9930 | 0.9976 | 0.8378 |
| ada | 0.8663 | 0.9952 | 0.9965 | 0.8649 |
| svm | 0.7676 | 0.9901 | 0.9971 | 0.7748 |
