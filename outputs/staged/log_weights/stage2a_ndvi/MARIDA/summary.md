# stage2a_ndvi: bands + NDVI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8995 | 0.8310 | 0.8921 | 0.9085 | 0.9959 | 0.9080 | 358,442 | 10 |
| xgb | 0.8469 | 0.7571 | 0.7942 | 0.9285 | 0.9929 | 0.8562 | 358,442 | 3 |
| lgbm | 0.8573 | 0.7707 | 0.8032 | 0.9367 | 0.9933 | 0.8642 | 358,442 | 2 |
| ada | 0.7038 | 0.6488 | 0.7171 | 0.8379 | 0.9561 | 0.4617 | 358,442 | 151 |
| svm | 0.6438 | 0.5375 | 0.6079 | 0.7791 | 0.9752 | 0.5978 | 358,442 | 4 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.8268 | 0.9762 | 0.8329 |
| xgb | 0.9949 | 0.8976 | 0.9433 | 0.8783 |
| lgbm | 0.9950 | 0.9081 | 0.9665 | 0.8770 |
| ada | 0.9590 | 0.6903 | 0.9738 | 0.7286 |
| svm | 0.9792 | 0.4488 | 0.9848 | 0.7037 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.7309 | 0.9969 | 0.8423 |
| xgb | 0.9992 | 0.5278 | 0.9468 | 0.7031 |
| lgbm | 0.9992 | 0.5554 | 0.9379 | 0.7204 |
| ada | 0.9971 | 0.0402 | 0.9756 | 0.8556 |
| svm | 0.9967 | 0.1557 | 0.4091 | 0.8702 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.8268 | 0.9762 | 0.8329 |
| xgb | 0.9949 | 0.8976 | 0.9433 | 0.8783 |
| lgbm | 0.9950 | 0.9081 | 0.9665 | 0.8770 |
| ada | 0.9590 | 0.6903 | 0.9738 | 0.7286 |
| svm | 0.9792 | 0.4488 | 0.9848 | 0.7037 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.7759 | 0.9865 | 0.8376 |
| xgb | 0.9970 | 0.6647 | 0.9451 | 0.7810 |
| lgbm | 0.9971 | 0.6892 | 0.9520 | 0.7910 |
| ada | 0.9776 | 0.0760 | 0.9747 | 0.7870 |
| svm | 0.9879 | 0.2312 | 0.5781 | 0.7782 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9963 | 0.6338 | 0.9733 | 0.7206 |
| xgb | 0.9940 | 0.4978 | 0.8958 | 0.6407 |
| lgbm | 0.9942 | 0.5258 | 0.9084 | 0.6543 |
| ada | 0.9562 | 0.0395 | 0.9506 | 0.6488 |
| svm | 0.9760 | 0.1307 | 0.4065 | 0.6369 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9964 | 0.9989 | 0.9997 | 0.9967 |
| xgb | 0.9942 | 0.9978 | 0.9989 | 0.9950 |
| lgbm | 0.9944 | 0.9980 | 0.9990 | 0.9953 |
| ada | 0.9571 | 0.9596 | 0.9995 | 0.9960 |
| svm | 0.9765 | 0.9928 | 0.9851 | 0.9959 |
