# stage2e_ndmi: bands + NDMI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7022 | 0.5561 | 0.7372 | 0.7259 | 0.7247 | 0.3651 | 761,887 | 35 |
| xgb | 0.6076 | 0.4646 | 0.5729 | 0.8069 | 0.7307 | 0.3965 | 761,887 | 6 |
| lgbm | 0.6368 | 0.4979 | 0.6129 | 0.8119 | 0.7426 | 0.4146 | 761,887 | 4 |
| ada | 0.4951 | 0.3640 | 0.4647 | 0.7446 | 0.6974 | 0.3539 | 761,887 | 364 |
| svm | 0.5518 | 0.4284 | 0.5477 | 0.7252 | 0.7277 | 0.3381 | 761,887 | 7 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8102 | 0.6713 | 0.8266 | 0.5006 |
| xgb | 0.8155 | 0.3236 | 0.7664 | 0.5246 |
| lgbm | 0.8243 | 0.3596 | 0.8261 | 0.5374 |
| ada | 0.7921 | 0.1040 | 0.5772 | 0.5071 |
| svm | 0.8200 | 0.1310 | 0.7731 | 0.4830 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6809 | 0.5052 | 0.7045 | 0.3339 |
| xgb | 0.6885 | 0.1931 | 0.6213 | 0.3556 |
| lgbm | 0.7012 | 0.2192 | 0.7037 | 0.3674 |
| ada | 0.6557 | 0.0549 | 0.4057 | 0.3397 |
| svm | 0.6950 | 0.0701 | 0.6301 | 0.3184 |
