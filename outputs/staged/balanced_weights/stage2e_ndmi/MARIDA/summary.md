# stage2e_ndmi: bands + NDMI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8884 | 0.8151 | 0.8691 | 0.9147 | 0.9954 | 0.8991 | 358,442 | 8 |
| xgb | 0.8081 | 0.7090 | 0.7360 | 0.9413 | 0.9901 | 0.8114 | 358,442 | 3 |
| lgbm | 0.8187 | 0.7196 | 0.7448 | 0.9447 | 0.9904 | 0.8169 | 358,442 | 2 |
| ada | 0.6795 | 0.6003 | 0.6376 | 0.9145 | 0.9674 | 0.5582 | 358,442 | 104 |
| svm | 0.5997 | 0.4937 | 0.5628 | 0.8237 | 0.9580 | 0.4719 | 358,442 | 3 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9978 | 0.8688 | 0.9726 | 0.8199 |
| xgb | 0.9918 | 0.9475 | 0.9531 | 0.8727 |
| lgbm | 0.9920 | 0.9475 | 0.9604 | 0.8789 |
| ada | 0.9696 | 0.9633 | 0.9519 | 0.7733 |
| svm | 0.9611 | 0.6509 | 0.9835 | 0.6994 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9983 | 0.6607 | 0.9919 | 0.8255 |
| xgb | 0.9997 | 0.4222 | 0.8978 | 0.6244 |
| lgbm | 0.9996 | 0.4676 | 0.8854 | 0.6267 |
| ada | 0.9997 | 0.0848 | 0.9071 | 0.5588 |
| svm | 0.9975 | 0.1549 | 0.2546 | 0.8441 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9978 | 0.8688 | 0.9726 | 0.8199 |
| xgb | 0.9918 | 0.9475 | 0.9531 | 0.8727 |
| lgbm | 0.9920 | 0.9475 | 0.9604 | 0.8789 |
| ada | 0.9696 | 0.9633 | 0.9519 | 0.7733 |
| svm | 0.9611 | 0.6509 | 0.9835 | 0.6994 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7506 | 0.9822 | 0.8227 |
| xgb | 0.9957 | 0.5841 | 0.9246 | 0.7280 |
| lgbm | 0.9958 | 0.6262 | 0.9214 | 0.7316 |
| ada | 0.9844 | 0.1559 | 0.9289 | 0.6488 |
| svm | 0.9790 | 0.2503 | 0.4045 | 0.7649 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.6007 | 0.9649 | 0.6988 |
| xgb | 0.9915 | 0.4126 | 0.8598 | 0.5723 |
| lgbm | 0.9916 | 0.4558 | 0.8542 | 0.5768 |
| ada | 0.9693 | 0.0845 | 0.8673 | 0.4801 |
| svm | 0.9588 | 0.1430 | 0.2535 | 0.6194 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.9986 | 0.9996 | 0.9964 |
| xgb | 0.9917 | 0.9968 | 0.9984 | 0.9934 |
| lgbm | 0.9918 | 0.9973 | 0.9983 | 0.9934 |
| ada | 0.9700 | 0.9749 | 0.9985 | 0.9915 |
| svm | 0.9597 | 0.9906 | 0.9700 | 0.9956 |
