# stage2g_specstats: bands + 4 spectral statistics (MARIDA)

Features (15): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, spec_mean, spec_std, spec_max, spec_norm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8846 | 0.8092 | 0.8754 | 0.8978 | 0.9954 | 0.8974 | 358,442 | 9 |
| xgb | 0.8109 | 0.7186 | 0.7457 | 0.9467 | 0.9905 | 0.8190 | 358,442 | 3 |
| lgbm | 0.8241 | 0.7338 | 0.7588 | 0.9507 | 0.9909 | 0.8256 | 358,442 | 2 |
| ada | 0.5345 | 0.4307 | 0.4716 | 0.8828 | 0.9085 | 0.3004 | 358,442 | 293 |
| svm | 0.7257 | 0.6650 | 0.7184 | 0.8842 | 0.9751 | 0.6139 | 358,442 | 6 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.8241 | 0.9616 | 0.8075 |
| xgb | 0.9922 | 0.9659 | 0.9537 | 0.8752 |
| lgbm | 0.9925 | 0.9685 | 0.9598 | 0.8820 |
| ada | 0.9098 | 0.9265 | 0.9665 | 0.7286 |
| svm | 0.9777 | 0.8268 | 0.9933 | 0.7391 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.6738 | 0.9875 | 0.8425 |
| xgb | 0.9997 | 0.3898 | 0.9572 | 0.6361 |
| lgbm | 0.9996 | 0.4388 | 0.9752 | 0.6214 |
| ada | 0.9998 | 0.0321 | 0.3909 | 0.4636 |
| svm | 0.9980 | 0.0864 | 0.9702 | 0.8190 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.8241 | 0.9616 | 0.8075 |
| xgb | 0.9922 | 0.9659 | 0.9537 | 0.8752 |
| lgbm | 0.9925 | 0.9685 | 0.9598 | 0.8820 |
| ada | 0.9098 | 0.9265 | 0.9665 | 0.7286 |
| svm | 0.9777 | 0.8268 | 0.9933 | 0.7391 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7414 | 0.9744 | 0.8246 |
| xgb | 0.9959 | 0.5555 | 0.9554 | 0.7367 |
| lgbm | 0.9960 | 0.6039 | 0.9674 | 0.7291 |
| ada | 0.9527 | 0.0621 | 0.5567 | 0.5667 |
| svm | 0.9878 | 0.1564 | 0.9816 | 0.7770 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.5891 | 0.9500 | 0.7016 |
| xgb | 0.9919 | 0.3845 | 0.9147 | 0.5832 |
| lgbm | 0.9921 | 0.4326 | 0.9369 | 0.5737 |
| ada | 0.9096 | 0.0321 | 0.3857 | 0.3953 |
| svm | 0.9758 | 0.0848 | 0.9639 | 0.6353 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9962 | 0.9986 | 0.9995 | 0.9965 |
| xgb | 0.9921 | 0.9963 | 0.9991 | 0.9936 |
| lgbm | 0.9923 | 0.9969 | 0.9993 | 0.9933 |
| ada | 0.9117 | 0.9327 | 0.9841 | 0.9887 |
| svm | 0.9763 | 0.9785 | 0.9996 | 0.9957 |
