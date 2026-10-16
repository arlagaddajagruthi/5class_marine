# stage2c_fai: bands + FAI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FAI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7130 | 0.5690 | 0.7480 | 0.7331 | 0.7333 | 0.3771 | 761,887 | 142 |
| xgb | 0.7130 | 0.5704 | 0.7325 | 0.7373 | 0.7938 | 0.4785 | 761,887 | 15 |
| lgbm | 0.6924 | 0.5483 | 0.7094 | 0.7269 | 0.7739 | 0.4469 | 761,887 | 10 |
| ada | 0.6160 | 0.4819 | 0.7844 | 0.5635 | 0.7737 | 0.3251 | 761,887 | 638 |
| svm | 0.4919 | 0.3851 | 0.5701 | 0.4560 | 0.7852 | 0.3212 | 761,887 | 9 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7215 | 0.7040 | 0.7174 | 0.7897 |
| xgb | 0.7903 | 0.6362 | 0.7057 | 0.8169 |
| lgbm | 0.7654 | 0.6146 | 0.7086 | 0.8187 |
| ada | 0.8349 | 0.2271 | 0.6908 | 0.5011 |
| svm | 0.8581 | 0.0000 | 0.4943 | 0.4717 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9420 | 0.6850 | 0.9897 | 0.3752 |
| xgb | 0.9535 | 0.5387 | 0.9864 | 0.4516 |
| lgbm | 0.9530 | 0.4788 | 0.9807 | 0.4249 |
| ada | 0.8849 | 0.9020 | 0.9617 | 0.3889 |
| svm | 0.8766 | 0.0000 | 0.9896 | 0.4143 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7215 | 0.7040 | 0.7174 | 0.7897 |
| xgb | 0.7903 | 0.6362 | 0.7057 | 0.8169 |
| lgbm | 0.7654 | 0.6146 | 0.7086 | 0.8187 |
| ada | 0.8349 | 0.2271 | 0.6908 | 0.5011 |
| svm | 0.8581 | 0.0000 | 0.4943 | 0.4717 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8171 | 0.6944 | 0.8318 | 0.5088 |
| xgb | 0.8643 | 0.5834 | 0.8228 | 0.5816 |
| lgbm | 0.8490 | 0.5383 | 0.8228 | 0.5594 |
| ada | 0.8592 | 0.3629 | 0.8041 | 0.4379 |
| svm | 0.8673 | 0.0000 | 0.6593 | 0.4411 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6908 | 0.5318 | 0.7121 | 0.3412 |
| xgb | 0.7610 | 0.4118 | 0.6989 | 0.4101 |
| lgbm | 0.7376 | 0.3682 | 0.6989 | 0.3884 |
| ada | 0.7531 | 0.2217 | 0.6723 | 0.2803 |
| svm | 0.7656 | 0.0000 | 0.4918 | 0.2830 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7371 | 0.9983 | 0.9975 | 0.7337 |
| xgb | 0.7979 | 0.9976 | 0.9974 | 0.7948 |
| lgbm | 0.7783 | 0.9972 | 0.9974 | 0.7748 |
| ada | 0.7771 | 0.9979 | 0.9971 | 0.7754 |
| svm | 0.7862 | 0.9973 | 0.9956 | 0.7913 |
