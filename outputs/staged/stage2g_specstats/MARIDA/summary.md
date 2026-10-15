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
