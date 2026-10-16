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

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7225 | 0.7390 | 0.7122 | 0.7982 |
| xgb | 0.7786 | 0.6927 | 0.6863 | 0.8279 |
| lgbm | 0.7665 | 0.6639 | 0.6750 | 0.8248 |
| ada | 0.7962 | 0.2271 | 0.6834 | 0.7209 |
| svm | 0.8275 | 0.0051 | 0.6439 | 0.4940 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9440 | 0.7641 | 0.9892 | 0.3786 |
| xgb | 0.9562 | 0.5286 | 0.9842 | 0.4414 |
| lgbm | 0.9545 | 0.5330 | 0.9729 | 0.4275 |
| ada | 0.9261 | 0.8340 | 0.9764 | 0.4291 |
| svm | 0.8784 | 0.5000 | 0.9920 | 0.3792 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7225 | 0.7390 | 0.7122 | 0.7982 |
| xgb | 0.7786 | 0.6927 | 0.6863 | 0.8279 |
| lgbm | 0.7665 | 0.6639 | 0.6750 | 0.8248 |
| ada | 0.7962 | 0.2271 | 0.6834 | 0.7209 |
| svm | 0.8275 | 0.0051 | 0.6439 | 0.4940 |

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

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7392 | 0.9987 | 0.9975 | 0.7360 |
| xgb | 0.7907 | 0.9975 | 0.9972 | 0.7870 |
| lgbm | 0.7802 | 0.9975 | 0.9971 | 0.7765 |
| ada | 0.7823 | 0.9978 | 0.9972 | 0.7838 |
| svm | 0.7663 | 0.9973 | 0.9969 | 0.7704 |
