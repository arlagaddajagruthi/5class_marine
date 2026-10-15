# stage2c_fai: bands + FAI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FAI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8936 | 0.8223 | 0.8834 | 0.9049 | 0.9954 | 0.8991 | 358,442 | 31 |
| xgb | 0.8502 | 0.7626 | 0.7960 | 0.9321 | 0.9929 | 0.8569 | 358,442 | 7 |
| lgbm | 0.8601 | 0.7746 | 0.8078 | 0.9354 | 0.9933 | 0.8629 | 358,442 | 5 |
| ada | 0.7714 | 0.6910 | 0.7580 | 0.8524 | 0.9899 | 0.7926 | 358,442 | 221 |
| svm | 0.5830 | 0.4873 | 0.5609 | 0.7380 | 0.9624 | 0.4842 | 358,442 | 5 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9979 | 0.7685 | 0.9846 | 0.8234 |
| xgb | 0.9969 | 0.6713 | 0.9574 | 0.7752 |
| lgbm | 0.9971 | 0.7044 | 0.9592 | 0.7797 |
| ada | 0.9956 | 0.3558 | 0.9641 | 0.7699 |
| svm | 0.9812 | 0.1233 | 0.4625 | 0.7651 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9959 | 0.6240 | 0.9697 | 0.6998 |
| xgb | 0.9939 | 0.5052 | 0.9184 | 0.6329 |
| lgbm | 0.9941 | 0.5437 | 0.9217 | 0.6390 |
| ada | 0.9911 | 0.2164 | 0.9307 | 0.6259 |
| svm | 0.9630 | 0.0657 | 0.3008 | 0.6196 |
