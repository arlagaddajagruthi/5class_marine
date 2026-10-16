# stage2d_fdi: bands + FDI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FDI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7048 | 0.5590 | 0.7414 | 0.7296 | 0.7222 | 0.3675 | 761,887 | 35 |
| xgb | 0.6035 | 0.4577 | 0.5754 | 0.7905 | 0.7189 | 0.3783 | 761,887 | 6 |
| lgbm | 0.6218 | 0.4773 | 0.5996 | 0.7931 | 0.7205 | 0.3807 | 761,887 | 4 |
| ada | 0.5508 | 0.4157 | 0.5407 | 0.7546 | 0.7026 | 0.3504 | 761,887 | 360 |
| svm | 0.5370 | 0.4131 | 0.5370 | 0.6953 | 0.7108 | 0.3197 | 761,887 | 7 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7040 | 0.6906 | 0.7161 | 0.8074 |
| xgb | 0.6944 | 0.8890 | 0.7498 | 0.8290 |
| lgbm | 0.6952 | 0.8828 | 0.7604 | 0.8337 |
| ada | 0.6842 | 0.8376 | 0.7103 | 0.7862 |
| svm | 0.7170 | 0.6691 | 0.7132 | 0.6821 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9457 | 0.6640 | 0.9897 | 0.3663 |
| xgb | 0.9600 | 0.2112 | 0.7606 | 0.3699 |
| lgbm | 0.9600 | 0.2368 | 0.8309 | 0.3709 |
| ada | 0.9562 | 0.1073 | 0.7458 | 0.3536 |
| svm | 0.9183 | 0.0611 | 0.8046 | 0.3640 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7040 | 0.6906 | 0.7161 | 0.8074 |
| xgb | 0.6944 | 0.8890 | 0.7498 | 0.8290 |
| lgbm | 0.6952 | 0.8828 | 0.7604 | 0.8337 |
| ada | 0.6842 | 0.8376 | 0.7103 | 0.7862 |
| svm | 0.7170 | 0.6691 | 0.7132 | 0.6821 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8072 | 0.6771 | 0.8310 | 0.5040 |
| xgb | 0.8059 | 0.3414 | 0.7551 | 0.5116 |
| lgbm | 0.8064 | 0.3734 | 0.7941 | 0.5134 |
| ada | 0.7976 | 0.1903 | 0.7276 | 0.4878 |
| svm | 0.8053 | 0.1120 | 0.7561 | 0.4747 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6767 | 0.5118 | 0.7108 | 0.3369 |
| xgb | 0.6749 | 0.2058 | 0.6066 | 0.3437 |
| lgbm | 0.6757 | 0.2296 | 0.6585 | 0.3454 |
| ada | 0.6634 | 0.1052 | 0.5718 | 0.3225 |
| svm | 0.6740 | 0.0593 | 0.6079 | 0.3112 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7261 | 0.9982 | 0.9975 | 0.7225 |
| xgb | 0.7277 | 0.9908 | 0.9958 | 0.7236 |
| lgbm | 0.7283 | 0.9920 | 0.9966 | 0.7240 |
| ada | 0.7174 | 0.9808 | 0.9955 | 0.7116 |
| svm | 0.7177 | 0.9714 | 0.9961 | 0.7364 |
