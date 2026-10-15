# stage1_bands: 11 Sentinel-2 bands only (MARIDA)

Features (11): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8849 | 0.8102 | 0.8621 | 0.9170 | 0.9953 | 0.8964 | 358,442 | 27 |
| xgb | 0.8441 | 0.7575 | 0.7915 | 0.9342 | 0.9929 | 0.8568 | 358,442 | 7 |
| lgbm | 0.8614 | 0.7786 | 0.8095 | 0.9439 | 0.9935 | 0.8680 | 358,442 | 11 |
| ada | 0.7975 | 0.7156 | 0.7661 | 0.8957 | 0.9914 | 0.8258 | 358,442 | 187 |
| svm | 0.5831 | 0.4873 | 0.5609 | 0.7380 | 0.9624 | 0.4842 | 358,442 | 4 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7420 | 0.9800 | 0.8197 |
| xgb | 0.9971 | 0.6454 | 0.9695 | 0.7644 |
| lgbm | 0.9973 | 0.6977 | 0.9738 | 0.7768 |
| ada | 0.9964 | 0.4318 | 0.9643 | 0.7974 |
| svm | 0.9812 | 0.1232 | 0.4627 | 0.7651 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.5898 | 0.9607 | 0.6944 |
| xgb | 0.9942 | 0.4764 | 0.9408 | 0.6186 |
| lgbm | 0.9946 | 0.5357 | 0.9489 | 0.6351 |
| ada | 0.9929 | 0.2754 | 0.9311 | 0.6631 |
| svm | 0.9630 | 0.0657 | 0.3010 | 0.6196 |
