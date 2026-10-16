# stage3_all_indices: bands + all 5 indices (MARIDA)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8969 | 0.8270 | 0.8827 | 0.9150 | 0.9956 | 0.9022 | 358,442 | 47 |
| xgb | 0.8676 | 0.7823 | 0.8126 | 0.9457 | 0.9933 | 0.8628 | 358,442 | 8 |
| lgbm | 0.8680 | 0.7845 | 0.8127 | 0.9464 | 0.9934 | 0.8661 | 358,442 | 6 |
| ada | 0.7246 | 0.6546 | 0.6919 | 0.9134 | 0.9765 | 0.6357 | 358,442 | 313 |
| svm | 0.7983 | 0.6860 | 0.7490 | 0.8907 | 0.9894 | 0.7901 | 358,442 | 5 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9979 | 0.8661 | 0.9750 | 0.8211 |
| xgb | 0.9949 | 0.9475 | 0.9604 | 0.8801 |
| lgbm | 0.9949 | 0.9291 | 0.9726 | 0.8888 |
| ada | 0.9784 | 0.8819 | 0.9628 | 0.8304 |
| svm | 0.9923 | 0.8425 | 0.9933 | 0.7348 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.7112 | 0.9963 | 0.8252 |
| xgb | 0.9992 | 0.6150 | 0.9381 | 0.6980 |
| lgbm | 0.9993 | 0.6031 | 0.9477 | 0.7008 |
| ada | 0.9988 | 0.1028 | 0.9422 | 0.7239 |
| svm | 0.9981 | 0.4871 | 0.6573 | 0.8535 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9979 | 0.8661 | 0.9750 | 0.8211 |
| xgb | 0.9949 | 0.9475 | 0.9604 | 0.8801 |
| lgbm | 0.9949 | 0.9291 | 0.9726 | 0.8888 |
| ada | 0.9784 | 0.8819 | 0.9628 | 0.8304 |
| svm | 0.9923 | 0.8425 | 0.9933 | 0.7348 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7811 | 0.9855 | 0.8232 |
| xgb | 0.9970 | 0.7459 | 0.9491 | 0.7786 |
| lgbm | 0.9971 | 0.7314 | 0.9600 | 0.7837 |
| ada | 0.9885 | 0.1842 | 0.9524 | 0.7735 |
| svm | 0.9952 | 0.6173 | 0.7911 | 0.7897 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.6408 | 0.9715 | 0.6995 |
| xgb | 0.9941 | 0.5947 | 0.9032 | 0.6374 |
| lgbm | 0.9941 | 0.5765 | 0.9231 | 0.6443 |
| ada | 0.9772 | 0.1014 | 0.9091 | 0.6307 |
| svm | 0.9905 | 0.4465 | 0.6544 | 0.6525 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9962 | 0.9988 | 0.9997 | 0.9964 |
| xgb | 0.9942 | 0.9984 | 0.9989 | 0.9949 |
| lgbm | 0.9943 | 0.9984 | 0.9992 | 0.9950 |
| ada | 0.9777 | 0.9812 | 0.9990 | 0.9951 |
| svm | 0.9907 | 0.9975 | 0.9946 | 0.9960 |
