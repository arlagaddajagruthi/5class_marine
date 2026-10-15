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
