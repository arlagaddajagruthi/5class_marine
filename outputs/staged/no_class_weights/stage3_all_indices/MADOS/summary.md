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

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7068 | 0.7461 | 0.7242 | 0.8222 |
| xgb | 0.7714 | 0.7307 | 0.7209 | 0.7962 |
| lgbm | 0.7497 | 0.7205 | 0.7449 | 0.8100 |
| ada | 0.8482 | 0.4358 | 0.7316 | 0.6450 |
| svm | 0.8118 | 0.0771 | 0.6957 | 0.5108 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9503 | 0.6280 | 0.9889 | 0.3728 |
| xgb | 0.9478 | 0.5342 | 0.9772 | 0.4247 |
| lgbm | 0.9502 | 0.5455 | 0.9742 | 0.4070 |
| ada | 0.9171 | 0.4113 | 0.9784 | 0.4736 |
| svm | 0.8812 | 0.7979 | 0.9808 | 0.3660 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7068 | 0.7461 | 0.7242 | 0.8222 |
| xgb | 0.7714 | 0.7307 | 0.7209 | 0.7962 |
| lgbm | 0.7497 | 0.7205 | 0.7449 | 0.8100 |
| ada | 0.8482 | 0.4358 | 0.7316 | 0.6450 |
| svm | 0.8118 | 0.0771 | 0.6957 | 0.5108 |

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

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7312 | 0.9981 | 0.9976 | 0.7274 |
| xgb | 0.7793 | 0.9976 | 0.9975 | 0.7761 |
| lgbm | 0.7642 | 0.9976 | 0.9976 | 0.7607 |
| ada | 0.8140 | 0.9968 | 0.9976 | 0.8128 |
| svm | 0.7577 | 0.9975 | 0.9973 | 0.7601 |
