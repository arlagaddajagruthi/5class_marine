# stage2g_specstats: bands + 4 spectral statistics (MARIDA)

Features (15): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, spec_mean, spec_std, spec_max, spec_norm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8791 | 0.8021 | 0.8595 | 0.9055 | 0.9952 | 0.8944 | 358,442 | 47 |
| xgb | 0.8543 | 0.7698 | 0.8285 | 0.8961 | 0.9943 | 0.8768 | 358,442 | 7 |
| lgbm | 0.7458 | 0.6219 | 0.6759 | 0.8652 | 0.9845 | 0.7181 | 358,442 | 6 |
| ada | 0.7372 | 0.6537 | 0.7953 | 0.6908 | 0.9922 | 0.8099 | 358,442 | 253 |
| svm | 0.7566 | 0.6784 | 0.8789 | 0.6964 | 0.9931 | 0.8288 | 358,442 | 5 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9977 | 0.8346 | 0.9598 | 0.8298 |
| xgb | 0.9973 | 0.8478 | 0.9476 | 0.7919 |
| lgbm | 0.9880 | 0.7874 | 0.9196 | 0.7658 |
| ada | 0.9989 | 0.2362 | 0.8379 | 0.6901 |
| svm | 0.9994 | 0.2152 | 0.9488 | 0.6224 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9983 | 0.6373 | 0.9881 | 0.8141 |
| xgb | 0.9984 | 0.5503 | 0.9898 | 0.7755 |
| lgbm | 0.9982 | 0.4518 | 0.5284 | 0.7253 |
| ada | 0.9944 | 0.3093 | 0.9978 | 0.8797 |
| svm | 0.9940 | 0.6165 | 0.9943 | 0.9109 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9977 | 0.8346 | 0.9598 | 0.8298 |
| xgb | 0.9973 | 0.8478 | 0.9476 | 0.7919 |
| lgbm | 0.9880 | 0.7874 | 0.9196 | 0.7658 |
| ada | 0.9989 | 0.2362 | 0.8379 | 0.6901 |
| svm | 0.9994 | 0.2152 | 0.9488 | 0.6224 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7227 | 0.9737 | 0.8219 |
| xgb | 0.9978 | 0.6674 | 0.9682 | 0.7837 |
| lgbm | 0.9931 | 0.5742 | 0.6711 | 0.7450 |
| ada | 0.9966 | 0.2679 | 0.9109 | 0.7734 |
| svm | 0.9967 | 0.3191 | 0.9710 | 0.7395 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.5658 | 0.9488 | 0.6977 |
| xgb | 0.9957 | 0.5008 | 0.9384 | 0.6443 |
| lgbm | 0.9863 | 0.4027 | 0.5050 | 0.5936 |
| ada | 0.9933 | 0.1546 | 0.8364 | 0.6305 |
| svm | 0.9934 | 0.1898 | 0.9436 | 0.5867 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.9985 | 0.9995 | 0.9963 |
| xgb | 0.9957 | 0.9980 | 0.9994 | 0.9956 |
| lgbm | 0.9866 | 0.9972 | 0.9907 | 0.9947 |
| ada | 0.9934 | 0.9969 | 0.9983 | 0.9959 |
| svm | 0.9936 | 0.9978 | 0.9994 | 0.9955 |
