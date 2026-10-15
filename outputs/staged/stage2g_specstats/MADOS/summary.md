# stage2g_specstats: bands + 4 spectral statistics (MADOS)

Features (15): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, spec_mean, spec_std, spec_max, spec_norm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.6875 | 0.5410 | 0.7306 | 0.6931 | 0.7192 | 0.3285 | 761,887 | 96 |
| xgb | 0.6162 | 0.4732 | 0.5814 | 0.8047 | 0.7529 | 0.4306 | 761,887 | 7 |
| lgbm | 0.6378 | 0.4975 | 0.6196 | 0.8043 | 0.7420 | 0.4153 | 761,887 | 4 |
| ada | 0.5183 | 0.3872 | 0.4796 | 0.7719 | 0.7383 | 0.4154 | 761,887 | 457 |
| svm | 0.6332 | 0.5122 | 0.6105 | 0.7814 | 0.7341 | 0.3629 | 761,887 | 8 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8095 | 0.6535 | 0.8187 | 0.4681 |
| xgb | 0.8335 | 0.3308 | 0.7512 | 0.5493 |
| lgbm | 0.8237 | 0.3683 | 0.8206 | 0.5384 |
| ada | 0.8249 | 0.1272 | 0.5689 | 0.5523 |
| svm | 0.8202 | 0.2995 | 0.9180 | 0.4949 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6800 | 0.4853 | 0.6931 | 0.3056 |
| xgb | 0.7145 | 0.1982 | 0.6016 | 0.3787 |
| lgbm | 0.7002 | 0.2257 | 0.6958 | 0.3683 |
| ada | 0.7020 | 0.0679 | 0.3975 | 0.3815 |
| svm | 0.6952 | 0.1762 | 0.8485 | 0.3288 |
