# stage3_all_indices: bands + all 5 indices (MADOS)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7091 | 0.5636 | 0.7335 | 0.7491 | 0.7294 | 0.3809 | 761,887 | 55 |
| xgb | 0.6400 | 0.4971 | 0.6171 | 0.8061 | 0.7472 | 0.4183 | 761,887 | 6 |
| lgbm | 0.6465 | 0.5016 | 0.6247 | 0.8033 | 0.7349 | 0.4001 | 761,887 | 4 |
| ada | 0.5490 | 0.4186 | 0.5289 | 0.7877 | 0.7201 | 0.3794 | 761,887 | 999 |
| svm | 0.6193 | 0.4915 | 0.5867 | 0.7860 | 0.7429 | 0.3630 | 761,887 | 18 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7107 | 0.7564 | 0.7125 | 0.8167 |
| xgb | 0.7274 | 0.9085 | 0.7517 | 0.8368 |
| lgbm | 0.7123 | 0.9044 | 0.7598 | 0.8368 |
| ada | 0.7007 | 0.8931 | 0.7504 | 0.8066 |
| svm | 0.7553 | 0.8489 | 0.8621 | 0.6778 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9492 | 0.6190 | 0.9914 | 0.3743 |
| xgb | 0.9619 | 0.2478 | 0.8606 | 0.3981 |
| lgbm | 0.9602 | 0.2811 | 0.8728 | 0.3846 |
| ada | 0.9568 | 0.0776 | 0.6936 | 0.3874 |
| svm | 0.9244 | 0.1805 | 0.8646 | 0.3775 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7107 | 0.7564 | 0.7125 | 0.8167 |
| xgb | 0.7274 | 0.9085 | 0.7517 | 0.8368 |
| lgbm | 0.7123 | 0.9044 | 0.7598 | 0.8368 |
| ada | 0.7007 | 0.8931 | 0.7504 | 0.8066 |
| svm | 0.7553 | 0.8489 | 0.8621 | 0.6778 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8128 | 0.6809 | 0.8292 | 0.5134 |
| xgb | 0.8284 | 0.3894 | 0.8025 | 0.5395 |
| lgbm | 0.8178 | 0.4288 | 0.8124 | 0.5270 |
| ada | 0.8090 | 0.1428 | 0.7209 | 0.5234 |
| svm | 0.8313 | 0.2977 | 0.8633 | 0.4849 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6847 | 0.5161 | 0.7082 | 0.3453 |
| xgb | 0.7071 | 0.2418 | 0.6701 | 0.3694 |
| lgbm | 0.6918 | 0.2730 | 0.6841 | 0.3578 |
| ada | 0.6792 | 0.0769 | 0.5636 | 0.3545 |
| svm | 0.7113 | 0.1749 | 0.7596 | 0.3201 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7335 | 0.9981 | 0.9975 | 0.7296 |
| xgb | 0.7546 | 0.9923 | 0.9968 | 0.7506 |
| lgbm | 0.7417 | 0.9935 | 0.9970 | 0.7377 |
| ada | 0.7306 | 0.9711 | 0.9950 | 0.7435 |
| svm | 0.7505 | 0.9892 | 0.9977 | 0.7486 |
