# stage2a_ndvi: bands + NDVI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7157 | 0.5710 | 0.7527 | 0.7381 | 0.7343 | 0.3883 | 761,887 | 147 |
| xgb | 0.7211 | 0.5780 | 0.7499 | 0.7385 | 0.7757 | 0.4480 | 761,887 | 20 |
| lgbm | 0.6840 | 0.5379 | 0.6985 | 0.7212 | 0.7774 | 0.4530 | 761,887 | 13 |
| ada | 0.5688 | 0.4570 | 0.5952 | 0.5748 | 0.7720 | 0.4046 | 761,887 | 640 |
| svm | 0.5072 | 0.4027 | 0.5628 | 0.4803 | 0.7665 | 0.3037 | 761,887 | 10 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7160 | 0.7020 | 0.7135 | 0.8209 |
| xgb | 0.7685 | 0.6588 | 0.7125 | 0.8142 |
| lgbm | 0.7695 | 0.6084 | 0.6857 | 0.8213 |
| ada | 0.7881 | 0.0966 | 0.7038 | 0.7108 |
| svm | 0.8286 | 0.0000 | 0.5950 | 0.4974 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9499 | 0.6977 | 0.9835 | 0.3798 |
| xgb | 0.9512 | 0.6365 | 0.9852 | 0.4266 |
| lgbm | 0.9539 | 0.4575 | 0.9528 | 0.4300 |
| ada | 0.9239 | 0.0732 | 0.9641 | 0.4197 |
| svm | 0.8788 | 0.0000 | 0.9903 | 0.3823 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7160 | 0.7020 | 0.7135 | 0.8209 |
| xgb | 0.7685 | 0.6588 | 0.7125 | 0.8142 |
| lgbm | 0.7695 | 0.6084 | 0.6857 | 0.8213 |
| ada | 0.7881 | 0.0966 | 0.7038 | 0.7108 |
| svm | 0.8286 | 0.0000 | 0.5950 | 0.4974 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8165 | 0.6998 | 0.8270 | 0.5194 |
| xgb | 0.8502 | 0.6475 | 0.8270 | 0.5599 |
| lgbm | 0.8518 | 0.5223 | 0.7974 | 0.5645 |
| ada | 0.8506 | 0.0833 | 0.8136 | 0.5278 |
| svm | 0.8530 | 0.0000 | 0.7434 | 0.4323 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6899 | 0.5382 | 0.7051 | 0.3508 |
| xgb | 0.7394 | 0.4787 | 0.7050 | 0.3888 |
| lgbm | 0.7418 | 0.3534 | 0.6631 | 0.3932 |
| ada | 0.7401 | 0.0434 | 0.6858 | 0.3585 |
| svm | 0.7436 | 0.0000 | 0.5916 | 0.2758 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7380 | 0.9984 | 0.9974 | 0.7347 |
| xgb | 0.7795 | 0.9981 | 0.9974 | 0.7765 |
| lgbm | 0.7820 | 0.9970 | 0.9970 | 0.7787 |
| ada | 0.7747 | 0.9943 | 0.9972 | 0.7779 |
| svm | 0.7674 | 0.9973 | 0.9965 | 0.7719 |
