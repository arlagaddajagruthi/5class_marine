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
