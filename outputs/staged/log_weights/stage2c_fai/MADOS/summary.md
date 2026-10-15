# stage2c_fai: bands + FAI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FAI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7031 | 0.5575 | 0.7397 | 0.7226 | 0.7263 | 0.3637 | 761,887 | 131 |
| xgb | 0.6799 | 0.5373 | 0.6697 | 0.7897 | 0.7428 | 0.4114 | 761,887 | 15 |
| lgbm | 0.6883 | 0.5436 | 0.6799 | 0.7885 | 0.7485 | 0.4217 | 761,887 | 9 |
| ada | 0.5506 | 0.4338 | 0.5601 | 0.7198 | 0.6914 | 0.3416 | 761,887 | 712 |
| svm | 0.5321 | 0.4131 | 0.5597 | 0.5735 | 0.7294 | 0.3186 | 761,887 | 12 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8120 | 0.6737 | 0.8279 | 0.4988 |
| xgb | 0.8230 | 0.5079 | 0.8529 | 0.5360 |
| lgbm | 0.8274 | 0.5433 | 0.8390 | 0.5437 |
| ada | 0.7801 | 0.0795 | 0.8302 | 0.5127 |
| svm | 0.8208 | 0.0845 | 0.7559 | 0.4671 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6834 | 0.5080 | 0.7063 | 0.3323 |
| xgb | 0.6992 | 0.3404 | 0.7435 | 0.3661 |
| lgbm | 0.7056 | 0.3730 | 0.7227 | 0.3733 |
| ada | 0.6395 | 0.0414 | 0.7097 | 0.3447 |
| svm | 0.6961 | 0.0441 | 0.6076 | 0.3047 |
