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

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9976 | 0.8661 | 0.9726 | 0.8236 |
| xgb | 0.9933 | 0.9108 | 0.9488 | 0.8658 |
| lgbm | 0.9952 | 0.9029 | 0.9531 | 0.8720 |
| ada | 0.9949 | 0.9186 | 0.9299 | 0.8043 |
| svm | 0.9710 | 0.4331 | 0.9726 | 0.7106 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9984 | 0.6322 | 0.9919 | 0.8180 |
| xgb | 0.9994 | 0.5118 | 0.9805 | 0.6058 |
| lgbm | 0.9993 | 0.5392 | 0.9812 | 0.6876 |
| ada | 0.9991 | 0.3299 | 0.9890 | 0.7708 |
| svm | 0.9964 | 0.1739 | 0.3010 | 0.8640 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9976 | 0.8661 | 0.9726 | 0.8236 |
| xgb | 0.9933 | 0.9108 | 0.9488 | 0.8658 |
| lgbm | 0.9952 | 0.9029 | 0.9531 | 0.8720 |
| ada | 0.9949 | 0.9186 | 0.9299 | 0.8043 |
| svm | 0.9710 | 0.4331 | 0.9726 | 0.7106 |

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

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.9985 | 0.9996 | 0.9963 |
| xgb | 0.9928 | 0.9977 | 0.9993 | 0.9929 |
| lgbm | 0.9946 | 0.9979 | 0.9993 | 0.9947 |
| ada | 0.9941 | 0.9953 | 0.9992 | 0.9956 |
| svm | 0.9683 | 0.9937 | 0.9763 | 0.9959 |
