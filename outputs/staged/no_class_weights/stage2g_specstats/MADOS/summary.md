# stage2g_specstats: bands + 4 spectral statistics (MADOS)

Features (15): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, spec_mean, spec_std, spec_max, spec_norm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.6988 | 0.5535 | 0.7386 | 0.7081 | 0.7236 | 0.3404 | 761,887 | 146 |
| xgb | 0.7063 | 0.5647 | 0.7167 | 0.7452 | 0.7972 | 0.4864 | 761,887 | 15 |
| lgbm | 0.6564 | 0.5098 | 0.6540 | 0.7162 | 0.7779 | 0.4540 | 761,887 | 11 |
| ada | 0.5986 | 0.4785 | 0.8148 | 0.5671 | 0.8327 | 0.5221 | 761,887 | 1127 |
| svm | 0.5882 | 0.4681 | 0.7805 | 0.5486 | 0.7627 | 0.3090 | 761,887 | 28 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8123 | 0.6842 | 0.8212 | 0.4775 |
| xgb | 0.8667 | 0.5446 | 0.8260 | 0.5877 |
| lgbm | 0.8530 | 0.4465 | 0.7614 | 0.5646 |
| ada | 0.8936 | 0.1508 | 0.7329 | 0.6171 |
| svm | 0.8493 | 0.2254 | 0.8437 | 0.4345 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6839 | 0.5199 | 0.6966 | 0.3136 |
| xgb | 0.7648 | 0.3742 | 0.7036 | 0.4162 |
| lgbm | 0.7438 | 0.2874 | 0.6148 | 0.3934 |
| ada | 0.8077 | 0.0816 | 0.5784 | 0.4462 |
| svm | 0.7380 | 0.1270 | 0.7296 | 0.2776 |
