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

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7225 | 0.6465 | 0.6973 | 0.7062 |
| xgb | 0.7328 | 0.8880 | 0.7536 | 0.8445 |
| lgbm | 0.7181 | 0.8849 | 0.7640 | 0.8503 |
| ada | 0.7178 | 0.8078 | 0.7284 | 0.8336 |
| svm | 0.7341 | 0.7790 | 0.8867 | 0.7259 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9204 | 0.6607 | 0.9913 | 0.3500 |
| xgb | 0.9663 | 0.2033 | 0.7488 | 0.4071 |
| lgbm | 0.9657 | 0.2326 | 0.8862 | 0.3939 |
| ada | 0.9697 | 0.0690 | 0.4667 | 0.4129 |
| svm | 0.9293 | 0.1854 | 0.9517 | 0.3754 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7225 | 0.6465 | 0.6973 | 0.7062 |
| xgb | 0.7328 | 0.8880 | 0.7536 | 0.8445 |
| lgbm | 0.7181 | 0.8849 | 0.7640 | 0.8503 |
| ada | 0.7178 | 0.8078 | 0.7284 | 0.8336 |
| svm | 0.7341 | 0.7790 | 0.8867 | 0.7259 |

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

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7232 | 0.9982 | 0.9974 | 0.7197 |
| xgb | 0.7616 | 0.9903 | 0.9957 | 0.7580 |
| lgbm | 0.7497 | 0.9918 | 0.9971 | 0.7454 |
| ada | 0.7519 | 0.9701 | 0.9906 | 0.7640 |
| svm | 0.7380 | 0.9902 | 0.9986 | 0.7413 |
