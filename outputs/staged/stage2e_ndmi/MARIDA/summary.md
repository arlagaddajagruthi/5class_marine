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
