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
