# stage4_indices_ratios: bands + all 5 indices + 4 band ratios (MADOS)

Features (20): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7115 | 0.5665 | 0.7375 | 0.7476 | 0.7323 | 0.3833 | 761,887 | 206 |
| xgb | 0.6426 | 0.4988 | 0.6187 | 0.8062 | 0.7390 | 0.4055 | 761,887 | 17 |
| lgbm | 0.6543 | 0.5102 | 0.6319 | 0.8072 | 0.7424 | 0.4109 | 761,887 | 13 |
| ada | 0.5439 | 0.4143 | 0.5201 | 0.7954 | 0.7101 | 0.3673 | 761,887 | 1437 |
| svm | 0.6476 | 0.5142 | 0.6127 | 0.7972 | 0.7518 | 0.3799 | 761,887 | 36 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7155 | 0.7523 | 0.7109 | 0.8117 |
| xgb | 0.7179 | 0.9096 | 0.7637 | 0.8337 |
| lgbm | 0.7211 | 0.9065 | 0.7630 | 0.8382 |
| ada | 0.6879 | 0.8962 | 0.7909 | 0.8068 |
| svm | 0.7628 | 0.8849 | 0.8475 | 0.6936 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9480 | 0.6338 | 0.9914 | 0.3767 |
| xgb | 0.9602 | 0.2635 | 0.8622 | 0.3888 |
| lgbm | 0.9606 | 0.2899 | 0.8848 | 0.3924 |
| ada | 0.9557 | 0.0704 | 0.6732 | 0.3811 |
| svm | 0.9278 | 0.2502 | 0.8857 | 0.3871 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7155 | 0.7523 | 0.7109 | 0.8117 |
| xgb | 0.7179 | 0.9096 | 0.7637 | 0.8337 |
| lgbm | 0.7211 | 0.9065 | 0.7630 | 0.8382 |
| ada | 0.6879 | 0.8962 | 0.7909 | 0.8068 |
| svm | 0.7628 | 0.8849 | 0.8475 | 0.6936 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8155 | 0.6880 | 0.8281 | 0.5146 |
| xgb | 0.8215 | 0.4087 | 0.8100 | 0.5303 |
| lgbm | 0.8238 | 0.4394 | 0.8194 | 0.5345 |
| ada | 0.8000 | 0.1306 | 0.7273 | 0.5177 |
| svm | 0.8372 | 0.3901 | 0.8662 | 0.4969 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6885 | 0.5244 | 0.7066 | 0.3464 |
| xgb | 0.6971 | 0.2568 | 0.6806 | 0.3608 |
| lgbm | 0.7004 | 0.2815 | 0.6941 | 0.3648 |
| ada | 0.6667 | 0.0698 | 0.5715 | 0.3493 |
| svm | 0.7200 | 0.2423 | 0.7639 | 0.3306 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7364 | 0.9982 | 0.9975 | 0.7326 |
| xgb | 0.7461 | 0.9929 | 0.9969 | 0.7421 |
| lgbm | 0.7489 | 0.9938 | 0.9971 | 0.7451 |
| ada | 0.7200 | 0.9678 | 0.9949 | 0.7375 |
| svm | 0.7585 | 0.9925 | 0.9978 | 0.7547 |
