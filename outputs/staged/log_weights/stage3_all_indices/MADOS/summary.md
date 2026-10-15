# stage3_all_indices: bands + all 5 indices (MADOS)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7114 | 0.5660 | 0.7389 | 0.7482 | 0.7294 | 0.3827 | 761,887 | 186 |
| xgb | 0.6783 | 0.5354 | 0.6638 | 0.7971 | 0.7572 | 0.4309 | 761,887 | 15 |
| lgbm | 0.6886 | 0.5442 | 0.6771 | 0.7957 | 0.7564 | 0.4311 | 761,887 | 12 |
| ada | 0.6426 | 0.4997 | 0.6279 | 0.7558 | 0.7498 | 0.4097 | 761,887 | 913 |
| svm | 0.6721 | 0.5332 | 0.6685 | 0.7096 | 0.7497 | 0.3545 | 761,887 | 16 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8125 | 0.6900 | 0.8277 | 0.5153 |
| xgb | 0.8354 | 0.4919 | 0.8373 | 0.5486 |
| lgbm | 0.8343 | 0.5367 | 0.8343 | 0.5492 |
| ada | 0.8312 | 0.4019 | 0.8050 | 0.5323 |
| svm | 0.8352 | 0.5126 | 0.8608 | 0.4798 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6842 | 0.5267 | 0.7061 | 0.3471 |
| xgb | 0.7173 | 0.3262 | 0.7201 | 0.3780 |
| lgbm | 0.7157 | 0.3668 | 0.7157 | 0.3785 |
| ada | 0.7111 | 0.2515 | 0.6737 | 0.3626 |
| svm | 0.7170 | 0.3447 | 0.7556 | 0.3156 |
