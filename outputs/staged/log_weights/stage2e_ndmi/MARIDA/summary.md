# stage2e_ndmi: bands + NDMI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8830 | 0.8082 | 0.8601 | 0.9150 | 0.9953 | 0.8966 | 358,442 | 29 |
| xgb | 0.8322 | 0.7413 | 0.7744 | 0.9297 | 0.9914 | 0.8301 | 358,442 | 7 |
| lgbm | 0.8521 | 0.7662 | 0.8018 | 0.9308 | 0.9933 | 0.8627 | 358,442 | 6 |
| ada | 0.8071 | 0.7210 | 0.7722 | 0.9119 | 0.9921 | 0.8391 | 358,442 | 209 |
| svm | 0.6178 | 0.5117 | 0.5838 | 0.7718 | 0.9671 | 0.5249 | 358,442 | 5 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7309 | 0.9822 | 0.8208 |
| xgb | 0.9963 | 0.6553 | 0.9644 | 0.7129 |
| lgbm | 0.9972 | 0.6752 | 0.9669 | 0.7689 |
| ada | 0.9970 | 0.4854 | 0.9585 | 0.7872 |
| svm | 0.9836 | 0.2481 | 0.4597 | 0.7798 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.5759 | 0.9649 | 0.6961 |
| xgb | 0.9927 | 0.4874 | 0.9312 | 0.5538 |
| lgbm | 0.9945 | 0.5096 | 0.9360 | 0.6246 |
| ada | 0.9940 | 0.3205 | 0.9204 | 0.6491 |
| svm | 0.9677 | 0.1416 | 0.2985 | 0.6391 |
