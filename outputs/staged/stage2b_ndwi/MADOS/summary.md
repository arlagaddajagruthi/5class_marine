# stage2b_ndwi: bands + NDWI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7203 | 0.5773 | 0.7631 | 0.7323 | 0.7372 | 0.3828 | 761,887 | 35 |
| xgb | 0.6484 | 0.5001 | 0.6076 | 0.8105 | 0.7603 | 0.4407 | 761,887 | 6 |
| lgbm | 0.6630 | 0.5162 | 0.6385 | 0.8084 | 0.7488 | 0.4245 | 761,887 | 4 |
| ada | 0.5084 | 0.3696 | 0.4794 | 0.7571 | 0.6810 | 0.3308 | 761,887 | 365 |
| svm | 0.5966 | 0.4711 | 0.5688 | 0.7613 | 0.7376 | 0.3583 | 761,887 | 6 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8202 | 0.7203 | 0.8280 | 0.5129 |
| xgb | 0.8382 | 0.4399 | 0.7586 | 0.5568 |
| lgbm | 0.8282 | 0.4741 | 0.8040 | 0.5456 |
| ada | 0.7732 | 0.1649 | 0.6039 | 0.4915 |
| svm | 0.8253 | 0.2275 | 0.8428 | 0.4909 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6952 | 0.5628 | 0.7065 | 0.3449 |
| xgb | 0.7214 | 0.2820 | 0.6111 | 0.3858 |
| lgbm | 0.7067 | 0.3107 | 0.6722 | 0.3752 |
| ada | 0.6303 | 0.0899 | 0.4325 | 0.3258 |
| svm | 0.7025 | 0.1284 | 0.7283 | 0.3253 |
