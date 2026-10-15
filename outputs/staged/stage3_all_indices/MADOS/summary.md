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
