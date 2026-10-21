# stage2d_fdi: bands + FDI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FDI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8906 | 0.8183 | 0.8729 | 0.9143 | 0.9956 | 0.9023 | 358,442 | 8 |
| xgb | 0.7826 | 0.6764 | 0.6991 | 0.9418 | 0.9863 | 0.7556 | 358,442 | 3 |
| lgbm | 0.7986 | 0.6961 | 0.7160 | 0.9481 | 0.9868 | 0.7639 | 358,442 | 2 |
| ada | 0.6743 | 0.5907 | 0.6258 | 0.9107 | 0.9700 | 0.5770 | 358,442 | 107 |
| svm | 0.5657 | 0.4702 | 0.5470 | 0.7943 | 0.9481 | 0.4135 | 358,442 | 3 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9979 | 0.8609 | 0.9817 | 0.8168 |
| xgb | 0.9879 | 0.9580 | 0.9531 | 0.8683 |
| lgbm | 0.9882 | 0.9606 | 0.9665 | 0.8770 |
| ada | 0.9724 | 0.9606 | 0.9476 | 0.7621 |
| svm | 0.9516 | 0.5748 | 0.9805 | 0.6702 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9983 | 0.6694 | 0.9841 | 0.8397 |
| xgb | 0.9996 | 0.4294 | 0.8641 | 0.5034 |
| lgbm | 0.9996 | 0.4822 | 0.8865 | 0.4958 |
| ada | 0.9995 | 0.0991 | 0.8927 | 0.5119 |
| svm | 0.9972 | 0.0631 | 0.2692 | 0.8584 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9979 | 0.8609 | 0.9817 | 0.8168 |
| xgb | 0.9879 | 0.9580 | 0.9531 | 0.8683 |
| lgbm | 0.9882 | 0.9606 | 0.9665 | 0.8770 |
| ada | 0.9724 | 0.9606 | 0.9476 | 0.7621 |
| svm | 0.9516 | 0.5748 | 0.9805 | 0.6702 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7532 | 0.9829 | 0.8281 |
| xgb | 0.9937 | 0.5930 | 0.9064 | 0.6373 |
| lgbm | 0.9939 | 0.6421 | 0.9248 | 0.6335 |
| ada | 0.9858 | 0.1797 | 0.9193 | 0.6124 |
| svm | 0.9739 | 0.1138 | 0.4225 | 0.7527 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9962 | 0.6041 | 0.9664 | 0.7066 |
| xgb | 0.9875 | 0.4215 | 0.8288 | 0.4677 |
| lgbm | 0.9878 | 0.4729 | 0.8601 | 0.4636 |
| ada | 0.9720 | 0.0987 | 0.8507 | 0.4414 |
| svm | 0.9490 | 0.0603 | 0.2678 | 0.6035 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9963 | 0.9986 | 0.9996 | 0.9966 |
| xgb | 0.9878 | 0.9968 | 0.9980 | 0.9900 |
| lgbm | 0.9881 | 0.9974 | 0.9984 | 0.9897 |
| ada | 0.9726 | 0.9789 | 0.9983 | 0.9902 |
| svm | 0.9501 | 0.9785 | 0.9722 | 0.9955 |
