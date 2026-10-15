# stage2b_ndwi: bands + NDWI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.9008 | 0.8322 | 0.8870 | 0.9177 | 0.9956 | 0.9025 | 358,442 | 8 |
| xgb | 0.8055 | 0.7049 | 0.7245 | 0.9600 | 0.9883 | 0.7869 | 358,442 | 3 |
| lgbm | 0.8231 | 0.7268 | 0.7462 | 0.9591 | 0.9897 | 0.8071 | 358,442 | 2 |
| ada | 0.6439 | 0.5545 | 0.5829 | 0.9356 | 0.9539 | 0.4730 | 358,442 | 107 |
| svm | 0.7095 | 0.5872 | 0.6491 | 0.8953 | 0.9795 | 0.6611 | 358,442 | 2 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7952 | 0.9852 | 0.8248 |
| xgb | 0.9946 | 0.6074 | 0.9269 | 0.6933 |
| lgbm | 0.9952 | 0.6423 | 0.9412 | 0.7137 |
| ada | 0.9767 | 0.1238 | 0.8579 | 0.6171 |
| svm | 0.9904 | 0.4737 | 0.6077 | 0.7661 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.6600 | 0.9709 | 0.7018 |
| xgb | 0.9892 | 0.4362 | 0.8637 | 0.5306 |
| lgbm | 0.9904 | 0.4731 | 0.8889 | 0.5548 |
| ada | 0.9545 | 0.0660 | 0.7512 | 0.4462 |
| svm | 0.9810 | 0.3103 | 0.4365 | 0.6209 |
