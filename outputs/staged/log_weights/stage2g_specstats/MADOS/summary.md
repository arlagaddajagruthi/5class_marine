# stage2g_specstats: bands + 4 spectral statistics (MADOS)

Features (15): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, spec_mean, spec_std, spec_max, spec_norm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.6864 | 0.5396 | 0.7289 | 0.6942 | 0.7163 | 0.3261 | 761,887 | 141 |
| xgb | 0.6725 | 0.5293 | 0.6670 | 0.7903 | 0.7550 | 0.4318 | 761,887 | 26 |
| lgbm | 0.6891 | 0.5462 | 0.6818 | 0.7933 | 0.7664 | 0.4496 | 761,887 | 19 |
| ada | 0.6558 | 0.5156 | 0.6638 | 0.6927 | 0.7938 | 0.4836 | 761,887 | 809 |
| svm | 0.6931 | 0.5589 | 0.7080 | 0.7056 | 0.7536 | 0.3718 | 761,887 | 13 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7179 | 0.6526 | 0.6957 | 0.7106 |
| xgb | 0.7341 | 0.8458 | 0.7290 | 0.8522 |
| lgbm | 0.7470 | 0.8335 | 0.7355 | 0.8571 |
| ada | 0.7870 | 0.4584 | 0.6895 | 0.8360 |
| svm | 0.7671 | 0.5324 | 0.8330 | 0.6899 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9211 | 0.6567 | 0.9899 | 0.3478 |
| xgb | 0.9641 | 0.3304 | 0.9678 | 0.4058 |
| lgbm | 0.9651 | 0.3722 | 0.9709 | 0.4188 |
| ada | 0.9578 | 0.3323 | 0.9103 | 0.4550 |
| svm | 0.9186 | 0.5487 | 0.9765 | 0.3883 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7179 | 0.6526 | 0.6957 | 0.7106 |
| xgb | 0.7341 | 0.8458 | 0.7290 | 0.8522 |
| lgbm | 0.7470 | 0.8335 | 0.7355 | 0.8571 |
| ada | 0.7870 | 0.4584 | 0.6895 | 0.8360 |
| svm | 0.7671 | 0.5324 | 0.8330 | 0.6899 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8069 | 0.6546 | 0.8171 | 0.4670 |
| xgb | 0.8335 | 0.4752 | 0.8316 | 0.5498 |
| lgbm | 0.8422 | 0.5146 | 0.8370 | 0.5627 |
| ada | 0.8640 | 0.3853 | 0.7847 | 0.5893 |
| svm | 0.8361 | 0.5404 | 0.8990 | 0.4969 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6763 | 0.4866 | 0.6908 | 0.3046 |
| xgb | 0.7146 | 0.3116 | 0.7118 | 0.3791 |
| lgbm | 0.7274 | 0.3464 | 0.7197 | 0.3915 |
| ada | 0.7606 | 0.2386 | 0.6457 | 0.4177 |
| svm | 0.7183 | 0.3703 | 0.8166 | 0.3306 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7203 | 0.9981 | 0.9973 | 0.7168 |
| xgb | 0.7613 | 0.9950 | 0.9975 | 0.7563 |
| lgbm | 0.7720 | 0.9958 | 0.9975 | 0.7674 |
| ada | 0.7983 | 0.9961 | 0.9968 | 0.7965 |
| svm | 0.7551 | 0.9976 | 0.9984 | 0.7561 |
