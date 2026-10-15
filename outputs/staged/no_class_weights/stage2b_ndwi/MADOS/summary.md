# stage2b_ndwi: bands + NDWI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7279 | 0.5867 | 0.7690 | 0.7430 | 0.7357 | 0.3829 | 761,887 | 146 |
| xgb | 0.7106 | 0.5658 | 0.7276 | 0.7464 | 0.7862 | 0.4695 | 761,887 | 17 |
| lgbm | 0.7004 | 0.5534 | 0.7220 | 0.7326 | 0.7757 | 0.4513 | 761,887 | 22 |
| ada | 0.6388 | 0.5015 | 0.7914 | 0.6069 | 0.7805 | 0.4208 | 761,887 | 673 |
| svm | 0.5181 | 0.4153 | 0.6874 | 0.4926 | 0.7655 | 0.3017 | 761,887 | 9 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8185 | 0.7513 | 0.8282 | 0.5136 |
| xgb | 0.8583 | 0.5996 | 0.8087 | 0.5758 |
| lgbm | 0.8503 | 0.5913 | 0.7970 | 0.5632 |
| ada | 0.8562 | 0.3570 | 0.8040 | 0.5379 |
| svm | 0.8522 | 0.0102 | 0.7809 | 0.4291 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6928 | 0.6017 | 0.7067 | 0.3456 |
| xgb | 0.7518 | 0.4282 | 0.6788 | 0.4043 |
| lgbm | 0.7395 | 0.4198 | 0.6625 | 0.3919 |
| ada | 0.7486 | 0.2173 | 0.6723 | 0.3679 |
| svm | 0.7424 | 0.0051 | 0.6406 | 0.2731 |
