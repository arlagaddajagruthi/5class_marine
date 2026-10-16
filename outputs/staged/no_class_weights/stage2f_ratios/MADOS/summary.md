# stage2f_ratios: bands + 4 band ratios (MADOS)

Features (15): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7252 | 0.5825 | 0.7554 | 0.7544 | 0.7374 | 0.3922 | 761,887 | 147 |
| xgb | 0.7162 | 0.5743 | 0.7218 | 0.7588 | 0.7866 | 0.4558 | 761,887 | 14 |
| lgbm | 0.6485 | 0.5055 | 0.6442 | 0.7395 | 0.7682 | 0.4295 | 761,887 | 11 |
| ada | 0.6506 | 0.5176 | 0.7376 | 0.6108 | 0.7977 | 0.4015 | 761,887 | 847 |
| svm | 0.5423 | 0.4344 | 0.6475 | 0.5093 | 0.7792 | 0.3222 | 761,887 | 14 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7203 | 0.7688 | 0.7106 | 0.8181 |
| xgb | 0.7893 | 0.7492 | 0.7187 | 0.7779 |
| lgbm | 0.7652 | 0.7143 | 0.6918 | 0.7865 |
| ada | 0.8465 | 0.2826 | 0.7329 | 0.5811 |
| svm | 0.8459 | 0.0401 | 0.6656 | 0.4855 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9496 | 0.7010 | 0.9883 | 0.3826 |
| xgb | 0.9449 | 0.5181 | 0.9858 | 0.4385 |
| lgbm | 0.9459 | 0.2875 | 0.9251 | 0.4184 |
| ada | 0.9020 | 0.6366 | 0.9675 | 0.4444 |
| svm | 0.8796 | 0.3197 | 0.9889 | 0.4018 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7203 | 0.7688 | 0.7106 | 0.8181 |
| xgb | 0.7893 | 0.7492 | 0.7187 | 0.7779 |
| lgbm | 0.7652 | 0.7143 | 0.6918 | 0.7865 |
| ada | 0.8465 | 0.2826 | 0.7329 | 0.5811 |
| svm | 0.8459 | 0.0401 | 0.6656 | 0.4855 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8192 | 0.7333 | 0.8267 | 0.5214 |
| xgb | 0.8601 | 0.6126 | 0.8313 | 0.5608 |
| lgbm | 0.8460 | 0.4100 | 0.7916 | 0.5462 |
| ada | 0.8734 | 0.3915 | 0.8340 | 0.5036 |
| svm | 0.8624 | 0.0712 | 0.7957 | 0.4397 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6938 | 0.5789 | 0.7047 | 0.3526 |
| xgb | 0.7545 | 0.4416 | 0.7113 | 0.3897 |
| lgbm | 0.7331 | 0.2579 | 0.6551 | 0.3757 |
| ada | 0.7752 | 0.2434 | 0.7153 | 0.3366 |
| svm | 0.7581 | 0.0369 | 0.6607 | 0.2818 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7412 | 0.9985 | 0.9975 | 0.7377 |
| xgb | 0.7910 | 0.9974 | 0.9975 | 0.7872 |
| lgbm | 0.7732 | 0.9945 | 0.9969 | 0.7718 |
| ada | 0.8002 | 0.9976 | 0.9975 | 0.8000 |
| svm | 0.7803 | 0.9972 | 0.9971 | 0.7839 |
