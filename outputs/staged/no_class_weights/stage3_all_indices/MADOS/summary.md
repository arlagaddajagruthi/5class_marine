# stage3_all_indices: bands + all 5 indices (MADOS)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7104 | 0.5656 | 0.7350 | 0.7498 | 0.7272 | 0.3796 | 761,887 | 385 |
| xgb | 0.7129 | 0.5696 | 0.7210 | 0.7548 | 0.7752 | 0.4426 | 761,887 | 27 |
| lgbm | 0.7113 | 0.5684 | 0.7192 | 0.7563 | 0.7601 | 0.4237 | 761,887 | 18 |
| ada | 0.6719 | 0.5379 | 0.6951 | 0.6651 | 0.8106 | 0.4517 | 761,887 | 1373 |
| svm | 0.5565 | 0.4412 | 0.7565 | 0.5238 | 0.7562 | 0.2972 | 761,887 | 25 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8106 | 0.6820 | 0.8361 | 0.5130 |
| xgb | 0.8506 | 0.6172 | 0.8297 | 0.5539 |
| lgbm | 0.8381 | 0.6209 | 0.8442 | 0.5418 |
| ada | 0.8813 | 0.4232 | 0.8372 | 0.5461 |
| svm | 0.8451 | 0.1406 | 0.8140 | 0.4264 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6816 | 0.5175 | 0.7184 | 0.3450 |
| xgb | 0.7400 | 0.4463 | 0.7090 | 0.3831 |
| lgbm | 0.7213 | 0.4502 | 0.7305 | 0.3715 |
| ada | 0.7878 | 0.2684 | 0.7200 | 0.3756 |
| svm | 0.7317 | 0.0756 | 0.6864 | 0.2710 |
