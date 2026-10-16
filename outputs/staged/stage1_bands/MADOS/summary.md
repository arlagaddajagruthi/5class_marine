# stage1_bands: 11 Sentinel-2 bands only (MADOS)

Features (11): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7022 | 0.5562 | 0.7441 | 0.7160 | 0.7253 | 0.3596 | 761,887 | 34 |
| xgb | 0.6018 | 0.4582 | 0.5585 | 0.8051 | 0.7507 | 0.4274 | 761,887 | 6 |
| lgbm | 0.6163 | 0.4693 | 0.5786 | 0.8005 | 0.7342 | 0.4029 | 761,887 | 3 |
| ada | 0.5024 | 0.3675 | 0.4642 | 0.7549 | 0.7057 | 0.3585 | 761,887 | 319 |
| svm | 0.5370 | 0.4131 | 0.5370 | 0.6953 | 0.7108 | 0.3197 | 761,887 | 6 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7158 | 0.6721 | 0.7044 | 0.7715 |
| xgb | 0.7301 | 0.8787 | 0.7679 | 0.8439 |
| lgbm | 0.7098 | 0.8798 | 0.7685 | 0.8440 |
| ada | 0.6859 | 0.8006 | 0.7378 | 0.7951 |
| svm | 0.7170 | 0.6691 | 0.7132 | 0.6821 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9366 | 0.6855 | 0.9891 | 0.3650 |
| xgb | 0.9654 | 0.1871 | 0.6744 | 0.4072 |
| lgbm | 0.9637 | 0.2324 | 0.7318 | 0.3867 |
| ada | 0.9560 | 0.0808 | 0.4500 | 0.3701 |
| svm | 0.9183 | 0.0611 | 0.8046 | 0.3640 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7158 | 0.6721 | 0.7044 | 0.7715 |
| xgb | 0.7301 | 0.8787 | 0.7679 | 0.8439 |
| lgbm | 0.7098 | 0.8798 | 0.7685 | 0.8440 |
| ada | 0.6859 | 0.8006 | 0.7378 | 0.7951 |
| svm | 0.7170 | 0.6691 | 0.7132 | 0.6821 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8114 | 0.6788 | 0.8228 | 0.4956 |
| xgb | 0.8314 | 0.3085 | 0.7181 | 0.5493 |
| lgbm | 0.8175 | 0.3677 | 0.7497 | 0.5304 |
| ada | 0.7987 | 0.1468 | 0.5591 | 0.5051 |
| svm | 0.8053 | 0.1120 | 0.7561 | 0.4747 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6827 | 0.5137 | 0.6990 | 0.3294 |
| xgb | 0.7114 | 0.1824 | 0.5602 | 0.3787 |
| lgbm | 0.6913 | 0.2253 | 0.5996 | 0.3609 |
| ada | 0.6649 | 0.0792 | 0.3880 | 0.3378 |
| svm | 0.6740 | 0.0593 | 0.6079 | 0.3112 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7291 | 0.9983 | 0.9974 | 0.7257 |
| xgb | 0.7589 | 0.9894 | 0.9948 | 0.7582 |
| lgbm | 0.7420 | 0.9918 | 0.9956 | 0.7390 |
| ada | 0.7186 | 0.9749 | 0.9900 | 0.7279 |
| svm | 0.7177 | 0.9714 | 0.9961 | 0.7364 |
