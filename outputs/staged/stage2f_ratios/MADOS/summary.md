# stage2f_ratios: bands + 4 band ratios (MADOS)

Features (15): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7185 | 0.5748 | 0.7473 | 0.7490 | 0.7344 | 0.3851 | 761,887 | 39 |
| xgb | 0.6241 | 0.4798 | 0.5908 | 0.8055 | 0.7413 | 0.4044 | 761,887 | 7 |
| lgbm | 0.6507 | 0.5070 | 0.6211 | 0.8096 | 0.7501 | 0.4191 | 761,887 | 4 |
| ada | 0.5208 | 0.3885 | 0.4869 | 0.7816 | 0.7121 | 0.3542 | 761,887 | 556 |
| svm | 0.6232 | 0.4883 | 0.5828 | 0.7816 | 0.7430 | 0.3674 | 761,887 | 8 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7185 | 0.7564 | 0.7122 | 0.8090 |
| xgb | 0.7240 | 0.9147 | 0.7656 | 0.8177 |
| lgbm | 0.7327 | 0.9044 | 0.7734 | 0.8278 |
| ada | 0.7023 | 0.9096 | 0.7627 | 0.7519 |
| svm | 0.7526 | 0.8263 | 0.8559 | 0.6914 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9473 | 0.6740 | 0.9897 | 0.3785 |
| xgb | 0.9563 | 0.2299 | 0.7852 | 0.3917 |
| lgbm | 0.9584 | 0.2758 | 0.8499 | 0.4004 |
| ada | 0.9440 | 0.0719 | 0.5549 | 0.3768 |
| svm | 0.9277 | 0.2145 | 0.8118 | 0.3772 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7185 | 0.7564 | 0.7122 | 0.8090 |
| xgb | 0.7240 | 0.9147 | 0.7656 | 0.8177 |
| lgbm | 0.7327 | 0.9044 | 0.7734 | 0.8278 |
| ada | 0.7023 | 0.9096 | 0.7627 | 0.7519 |
| svm | 0.7526 | 0.8263 | 0.8559 | 0.6914 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8172 | 0.7128 | 0.8283 | 0.5157 |
| xgb | 0.8241 | 0.3675 | 0.7753 | 0.5297 |
| lgbm | 0.8305 | 0.4227 | 0.8098 | 0.5398 |
| ada | 0.8054 | 0.1333 | 0.6424 | 0.5020 |
| svm | 0.8310 | 0.3405 | 0.8333 | 0.4881 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6909 | 0.5538 | 0.7069 | 0.3474 |
| xgb | 0.7008 | 0.2251 | 0.6330 | 0.3603 |
| lgbm | 0.7101 | 0.2680 | 0.6804 | 0.3696 |
| ada | 0.6742 | 0.0714 | 0.4732 | 0.3352 |
| svm | 0.7109 | 0.2052 | 0.7142 | 0.3228 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7383 | 0.9984 | 0.9975 | 0.7346 |
| xgb | 0.7484 | 0.9915 | 0.9962 | 0.7464 |
| lgbm | 0.7565 | 0.9933 | 0.9969 | 0.7535 |
| ada | 0.7237 | 0.9681 | 0.9927 | 0.7395 |
| svm | 0.7508 | 0.9914 | 0.9971 | 0.7467 |
