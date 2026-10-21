# stage2a_ndvi: bands + NDVI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.9019 | 0.8336 | 0.8997 | 0.9049 | 0.9958 | 0.9062 | 358,442 | 8 |
| xgb | 0.7238 | 0.6453 | 0.6753 | 0.9393 | 0.9784 | 0.6614 | 358,442 | 5 |
| lgbm | 0.7792 | 0.6905 | 0.7115 | 0.9538 | 0.9876 | 0.7755 | 358,442 | 2 |
| ada | 0.6839 | 0.6230 | 0.6734 | 0.9062 | 0.9272 | 0.3542 | 358,442 | 109 |
| svm | 0.6144 | 0.5073 | 0.5775 | 0.8353 | 0.9645 | 0.5180 | 358,442 | 2 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9983 | 0.8268 | 0.9683 | 0.8261 |
| xgb | 0.9798 | 0.9528 | 0.9439 | 0.8807 |
| lgbm | 0.9888 | 0.9606 | 0.9714 | 0.8944 |
| ada | 0.9281 | 0.9291 | 0.9756 | 0.7919 |
| svm | 0.9678 | 0.6982 | 0.9896 | 0.6857 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7609 | 0.9987 | 0.8412 |
| xgb | 0.9995 | 0.1310 | 0.9480 | 0.6227 |
| lgbm | 0.9996 | 0.2740 | 0.9388 | 0.6335 |
| ada | 0.9990 | 0.0318 | 0.9697 | 0.6929 |
| svm | 0.9978 | 0.1367 | 0.3222 | 0.8532 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9983 | 0.8268 | 0.9683 | 0.8261 |
| xgb | 0.9798 | 0.9528 | 0.9439 | 0.8807 |
| lgbm | 0.9888 | 0.9606 | 0.9714 | 0.8944 |
| ada | 0.9281 | 0.9291 | 0.9756 | 0.7919 |
| svm | 0.9678 | 0.6982 | 0.9896 | 0.6857 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7925 | 0.9833 | 0.8336 |
| xgb | 0.9896 | 0.2303 | 0.9460 | 0.7296 |
| lgbm | 0.9941 | 0.4263 | 0.9548 | 0.7417 |
| ada | 0.9623 | 0.0615 | 0.9727 | 0.7391 |
| svm | 0.9826 | 0.2286 | 0.4862 | 0.7603 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9962 | 0.6562 | 0.9671 | 0.7147 |
| xgb | 0.9794 | 0.1301 | 0.8975 | 0.5743 |
| lgbm | 0.9884 | 0.2709 | 0.9135 | 0.5894 |
| ada | 0.9273 | 0.0317 | 0.9468 | 0.5862 |
| svm | 0.9658 | 0.1291 | 0.3211 | 0.6133 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9963 | 0.9990 | 0.9997 | 0.9966 |
| xgb | 0.9798 | 0.9847 | 0.9989 | 0.9934 |
| lgbm | 0.9886 | 0.9938 | 0.9990 | 0.9937 |
| ada | 0.9289 | 0.9318 | 0.9994 | 0.9943 |
| svm | 0.9665 | 0.9887 | 0.9783 | 0.9956 |
