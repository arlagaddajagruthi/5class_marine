# stage5_ndwi_ratios: bands + NDWI + 4 band ratios (MADOS)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7190 | 0.5750 | 0.7504 | 0.7480 | 0.7342 | 0.3867 | 761,887 | 173 |
| xgb | 0.6850 | 0.5420 | 0.6696 | 0.8001 | 0.7710 | 0.4502 | 761,887 | 16 |
| lgbm | 0.6928 | 0.5491 | 0.6808 | 0.7964 | 0.7621 | 0.4350 | 761,887 | 11 |
| ada | 0.6238 | 0.4767 | 0.5946 | 0.7518 | 0.7322 | 0.3768 | 761,887 | 700 |
| svm | 0.7150 | 0.5764 | 0.7221 | 0.7358 | 0.7565 | 0.3680 | 761,887 | 12 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7171 | 0.7533 | 0.7067 | 0.8150 |
| xgb | 0.7579 | 0.8705 | 0.7397 | 0.8322 |
| lgbm | 0.7477 | 0.8664 | 0.7430 | 0.8287 |
| ada | 0.7218 | 0.7544 | 0.7520 | 0.7792 |
| svm | 0.7758 | 0.7256 | 0.7757 | 0.6660 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9487 | 0.6844 | 0.9896 | 0.3790 |
| xgb | 0.9597 | 0.3544 | 0.9415 | 0.4228 |
| lgbm | 0.9577 | 0.3980 | 0.9563 | 0.4112 |
| ada | 0.9436 | 0.2736 | 0.7848 | 0.3763 |
| svm | 0.9144 | 0.6248 | 0.9607 | 0.3887 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7171 | 0.7533 | 0.7067 | 0.8150 |
| xgb | 0.7579 | 0.8705 | 0.7397 | 0.8322 |
| lgbm | 0.7477 | 0.8664 | 0.7430 | 0.8287 |
| ada | 0.7218 | 0.7544 | 0.7520 | 0.7792 |
| svm | 0.7758 | 0.7256 | 0.7757 | 0.6660 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8168 | 0.7172 | 0.8246 | 0.5174 |
| xgb | 0.8469 | 0.5037 | 0.8285 | 0.5607 |
| lgbm | 0.8398 | 0.5455 | 0.8362 | 0.5497 |
| ada | 0.8179 | 0.4015 | 0.7681 | 0.5075 |
| svm | 0.8394 | 0.6714 | 0.8583 | 0.4909 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6903 | 0.5591 | 0.7015 | 0.3490 |
| xgb | 0.7345 | 0.3366 | 0.7072 | 0.3895 |
| lgbm | 0.7238 | 0.3750 | 0.7185 | 0.3790 |
| ada | 0.6919 | 0.2512 | 0.6235 | 0.3400 |
| svm | 0.7232 | 0.5054 | 0.7518 | 0.3253 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7381 | 0.9984 | 0.9974 | 0.7345 |
| xgb | 0.7770 | 0.9954 | 0.9974 | 0.7723 |
| lgbm | 0.7678 | 0.9961 | 0.9975 | 0.7629 |
| ada | 0.7384 | 0.9939 | 0.9961 | 0.7359 |
| svm | 0.7583 | 0.9981 | 0.9978 | 0.7588 |
