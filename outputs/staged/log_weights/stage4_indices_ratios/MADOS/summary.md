# stage4_indices_ratios: bands + all 5 indices + 4 band ratios (MADOS)

Features (20): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7096 | 0.5639 | 0.7346 | 0.7482 | 0.7320 | 0.3849 | 761,887 | 195 |
| xgb | 0.6876 | 0.5430 | 0.6758 | 0.7944 | 0.7617 | 0.4359 | 761,887 | 16 |
| lgbm | 0.6949 | 0.5506 | 0.6851 | 0.7973 | 0.7563 | 0.4314 | 761,887 | 13 |
| ada | 0.6371 | 0.4938 | 0.6240 | 0.7553 | 0.7426 | 0.4024 | 761,887 | 1238 |
| svm | 0.7002 | 0.5629 | 0.6882 | 0.7531 | 0.7549 | 0.3699 | 761,887 | 36 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7136 | 0.7513 | 0.7096 | 0.8183 |
| xgb | 0.7462 | 0.8633 | 0.7342 | 0.8340 |
| lgbm | 0.7366 | 0.8613 | 0.7439 | 0.8472 |
| ada | 0.7270 | 0.7451 | 0.7332 | 0.8158 |
| svm | 0.7722 | 0.7616 | 0.8071 | 0.6716 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9498 | 0.6216 | 0.9901 | 0.3771 |
| xgb | 0.9590 | 0.3853 | 0.9474 | 0.4115 |
| lgbm | 0.9614 | 0.4124 | 0.9603 | 0.4064 |
| ada | 0.9530 | 0.2667 | 0.8851 | 0.3911 |
| svm | 0.9171 | 0.4859 | 0.9629 | 0.3869 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7136 | 0.7513 | 0.7096 | 0.8183 |
| xgb | 0.7462 | 0.8633 | 0.7342 | 0.8340 |
| lgbm | 0.7366 | 0.8613 | 0.7439 | 0.8472 |
| ada | 0.7270 | 0.7451 | 0.7332 | 0.8158 |
| svm | 0.7722 | 0.7616 | 0.8071 | 0.6716 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8149 | 0.6803 | 0.8267 | 0.5163 |
| xgb | 0.8393 | 0.5328 | 0.8273 | 0.5511 |
| lgbm | 0.8341 | 0.5577 | 0.8384 | 0.5493 |
| ada | 0.8248 | 0.3928 | 0.8021 | 0.5287 |
| svm | 0.8385 | 0.5933 | 0.8781 | 0.4910 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6877 | 0.5155 | 0.7046 | 0.3480 |
| xgb | 0.7231 | 0.3632 | 0.7054 | 0.3804 |
| lgbm | 0.7155 | 0.3867 | 0.7217 | 0.3787 |
| ada | 0.7019 | 0.2444 | 0.6695 | 0.3593 |
| svm | 0.7219 | 0.4217 | 0.7827 | 0.3254 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7361 | 0.9981 | 0.9975 | 0.7323 |
| xgb | 0.7673 | 0.9959 | 0.9974 | 0.7628 |
| lgbm | 0.7615 | 0.9963 | 0.9975 | 0.7573 |
| ada | 0.7486 | 0.9938 | 0.9969 | 0.7460 |
| svm | 0.7578 | 0.9972 | 0.9981 | 0.7568 |
