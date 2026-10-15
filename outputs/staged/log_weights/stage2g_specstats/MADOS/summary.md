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
