# stage2b_ndwi: bands + NDWI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8987 | 0.8294 | 0.8792 | 0.9232 | 0.9954 | 0.9003 | 358,442 | 43 |
| xgb | 0.8623 | 0.7795 | 0.8079 | 0.9492 | 0.9933 | 0.8641 | 358,442 | 10 |
| lgbm | 0.8665 | 0.7845 | 0.8131 | 0.9476 | 0.9934 | 0.8668 | 358,442 | 6 |
| ada | 0.8273 | 0.7424 | 0.8008 | 0.8920 | 0.9931 | 0.8523 | 358,442 | 291 |
| svm | 0.7622 | 0.6455 | 0.7200 | 0.8496 | 0.9872 | 0.7530 | 358,442 | 8 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7911 | 0.9865 | 0.8192 |
| xgb | 0.9971 | 0.7013 | 0.9732 | 0.7776 |
| lgbm | 0.9971 | 0.7167 | 0.9723 | 0.7798 |
| ada | 0.9971 | 0.5536 | 0.9705 | 0.7880 |
| svm | 0.9942 | 0.5287 | 0.7428 | 0.7830 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9959 | 0.6544 | 0.9733 | 0.6938 |
| xgb | 0.9941 | 0.5401 | 0.9477 | 0.6362 |
| lgbm | 0.9942 | 0.5585 | 0.9461 | 0.6391 |
| ada | 0.9942 | 0.3827 | 0.9426 | 0.6501 |
| svm | 0.9884 | 0.3594 | 0.5908 | 0.6434 |
