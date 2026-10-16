# stage2e_ndmi: bands + NDMI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7075 | 0.5622 | 0.7430 | 0.7312 | 0.7250 | 0.3673 | 761,887 | 135 |
| xgb | 0.6455 | 0.5063 | 0.6422 | 0.7859 | 0.7374 | 0.4049 | 761,887 | 14 |
| lgbm | 0.6682 | 0.5291 | 0.6578 | 0.7971 | 0.7444 | 0.4138 | 761,887 | 9 |
| ada | 0.5989 | 0.4581 | 0.6006 | 0.7117 | 0.7370 | 0.3976 | 761,887 | 742 |
| svm | 0.5553 | 0.4349 | 0.5720 | 0.6145 | 0.7420 | 0.3389 | 761,887 | 13 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7101 | 0.7081 | 0.7116 | 0.7949 |
| xgb | 0.7145 | 0.8397 | 0.7472 | 0.8423 |
| lgbm | 0.7229 | 0.8489 | 0.7753 | 0.8413 |
| ada | 0.7195 | 0.6351 | 0.6685 | 0.8236 |
| svm | 0.7649 | 0.3649 | 0.6847 | 0.6436 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9427 | 0.6709 | 0.9910 | 0.3675 |
| xgb | 0.9618 | 0.2526 | 0.9669 | 0.3874 |
| lgbm | 0.9608 | 0.3051 | 0.9716 | 0.3937 |
| ada | 0.9570 | 0.1902 | 0.8691 | 0.3862 |
| svm | 0.9085 | 0.0763 | 0.9228 | 0.3806 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7101 | 0.7081 | 0.7116 | 0.7949 |
| xgb | 0.7145 | 0.8397 | 0.7472 | 0.8423 |
| lgbm | 0.7229 | 0.8489 | 0.7753 | 0.8413 |
| ada | 0.7195 | 0.6351 | 0.6685 | 0.8236 |
| svm | 0.7649 | 0.3649 | 0.6847 | 0.6436 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8101 | 0.6890 | 0.8283 | 0.5026 |
| xgb | 0.8199 | 0.3883 | 0.8430 | 0.5307 |
| lgbm | 0.8250 | 0.4489 | 0.8624 | 0.5364 |
| ada | 0.8214 | 0.2927 | 0.7557 | 0.5259 |
| svm | 0.8306 | 0.1262 | 0.7861 | 0.4783 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6808 | 0.5256 | 0.7070 | 0.3357 |
| xgb | 0.6947 | 0.2409 | 0.7285 | 0.3612 |
| lgbm | 0.7022 | 0.2894 | 0.7582 | 0.3665 |
| ada | 0.6970 | 0.1714 | 0.6074 | 0.3567 |
| svm | 0.7102 | 0.0673 | 0.6476 | 0.3143 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7289 | 0.9983 | 0.9975 | 0.7253 |
| xgb | 0.7444 | 0.9929 | 0.9976 | 0.7399 |
| lgbm | 0.7504 | 0.9944 | 0.9979 | 0.7460 |
| ada | 0.7453 | 0.9917 | 0.9963 | 0.7407 |
| svm | 0.7459 | 0.9864 | 0.9968 | 0.7548 |
