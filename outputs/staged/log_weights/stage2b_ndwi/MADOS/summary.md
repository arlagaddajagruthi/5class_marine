# stage2b_ndwi: bands + NDWI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7226 | 0.5802 | 0.7660 | 0.7372 | 0.7309 | 0.3757 | 761,887 | 134 |
| xgb | 0.7056 | 0.5616 | 0.6980 | 0.7990 | 0.7653 | 0.4470 | 761,887 | 14 |
| lgbm | 0.7067 | 0.5632 | 0.7015 | 0.7963 | 0.7634 | 0.4445 | 761,887 | 9 |
| ada | 0.6637 | 0.5151 | 0.6454 | 0.7374 | 0.7764 | 0.4530 | 761,887 | 725 |
| svm | 0.6378 | 0.5026 | 0.6489 | 0.6549 | 0.7486 | 0.3483 | 761,887 | 10 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7170 | 0.7276 | 0.7073 | 0.7969 |
| xgb | 0.7459 | 0.8592 | 0.7352 | 0.8558 |
| lgbm | 0.7431 | 0.8458 | 0.7384 | 0.8578 |
| ada | 0.7667 | 0.6495 | 0.7064 | 0.8270 |
| svm | 0.7709 | 0.4347 | 0.7653 | 0.6485 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9433 | 0.7564 | 0.9909 | 0.3736 |
| xgb | 0.9637 | 0.4437 | 0.9672 | 0.4173 |
| lgbm | 0.9639 | 0.4492 | 0.9777 | 0.4152 |
| ada | 0.9536 | 0.3803 | 0.8145 | 0.4332 |
| svm | 0.9095 | 0.3516 | 0.9567 | 0.3777 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7170 | 0.7276 | 0.7073 | 0.7969 |
| xgb | 0.7459 | 0.8592 | 0.7352 | 0.8558 |
| lgbm | 0.7431 | 0.8458 | 0.7384 | 0.8578 |
| ada | 0.7667 | 0.6495 | 0.7064 | 0.8270 |
| svm | 0.7709 | 0.4347 | 0.7653 | 0.6485 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8147 | 0.7417 | 0.8255 | 0.5087 |
| xgb | 0.8409 | 0.5852 | 0.8354 | 0.5610 |
| lgbm | 0.8392 | 0.5868 | 0.8414 | 0.5595 |
| ada | 0.8500 | 0.4797 | 0.7566 | 0.5685 |
| svm | 0.8345 | 0.3888 | 0.8504 | 0.4774 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6873 | 0.5895 | 0.7028 | 0.3411 |
| xgb | 0.7255 | 0.4137 | 0.7173 | 0.3899 |
| lgbm | 0.7230 | 0.4152 | 0.7262 | 0.3884 |
| ada | 0.7391 | 0.3155 | 0.6085 | 0.3972 |
| svm | 0.7160 | 0.2413 | 0.7397 | 0.3135 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7345 | 0.9986 | 0.9974 | 0.7312 |
| xgb | 0.7702 | 0.9967 | 0.9975 | 0.7661 |
| lgbm | 0.7682 | 0.9968 | 0.9976 | 0.7642 |
| ada | 0.7797 | 0.9962 | 0.9961 | 0.7808 |
| svm | 0.7511 | 0.9963 | 0.9977 | 0.7521 |
