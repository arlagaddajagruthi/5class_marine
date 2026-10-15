# stage2a_ndvi: bands + NDVI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7095 | 0.5645 | 0.7536 | 0.7244 | 0.7227 | 0.3615 | 761,887 | 134 |
| xgb | 0.6807 | 0.5361 | 0.6722 | 0.7823 | 0.7552 | 0.4296 | 761,887 | 14 |
| lgbm | 0.6930 | 0.5478 | 0.6829 | 0.7894 | 0.7548 | 0.4316 | 761,887 | 10 |
| ada | 0.5639 | 0.4360 | 0.5473 | 0.7081 | 0.7381 | 0.3975 | 761,887 | 929 |
| svm | 0.5717 | 0.4532 | 0.5850 | 0.6160 | 0.7445 | 0.3432 | 761,887 | 23 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8084 | 0.7071 | 0.8239 | 0.4985 |
| xgb | 0.8334 | 0.5112 | 0.8295 | 0.5485 |
| lgbm | 0.8324 | 0.5563 | 0.8323 | 0.5508 |
| ada | 0.8196 | 0.1450 | 0.7510 | 0.5402 |
| svm | 0.8316 | 0.1480 | 0.8273 | 0.4800 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6785 | 0.5469 | 0.7005 | 0.3320 |
| xgb | 0.7144 | 0.3434 | 0.7087 | 0.3779 |
| lgbm | 0.7129 | 0.3854 | 0.7128 | 0.3800 |
| ada | 0.6944 | 0.0782 | 0.6012 | 0.3700 |
| svm | 0.7118 | 0.0799 | 0.7054 | 0.3158 |
