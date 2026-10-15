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
