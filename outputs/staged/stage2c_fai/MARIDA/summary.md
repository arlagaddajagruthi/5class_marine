# stage2c_fai: bands + FAI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FAI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8975 | 0.8278 | 0.8930 | 0.9029 | 0.9957 | 0.9043 | 358,442 | 8 |
| xgb | 0.8147 | 0.7191 | 0.7403 | 0.9520 | 0.9899 | 0.8096 | 358,442 | 3 |
| lgbm | 0.8234 | 0.7289 | 0.7508 | 0.9538 | 0.9905 | 0.8187 | 358,442 | 2 |
| ada | 0.6596 | 0.5908 | 0.6466 | 0.8940 | 0.8869 | 0.2555 | 358,442 | 110 |
| svm | 0.5658 | 0.4702 | 0.5470 | 0.7944 | 0.9481 | 0.4137 | 358,442 | 3 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.8136 | 0.9750 | 0.8248 |
| xgb | 0.9912 | 0.9554 | 0.9689 | 0.8925 |
| lgbm | 0.9918 | 0.9580 | 0.9634 | 0.9019 |
| ada | 0.8868 | 0.9213 | 0.9817 | 0.7863 |
| svm | 0.9516 | 0.5748 | 0.9811 | 0.6702 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7416 | 0.9950 | 0.8373 |
| xgb | 0.9996 | 0.4344 | 0.9228 | 0.6045 |
| lgbm | 0.9996 | 0.4574 | 0.9284 | 0.6179 |
| ada | 0.9989 | 0.0203 | 0.9285 | 0.6387 |
| svm | 0.9972 | 0.0631 | 0.2694 | 0.8584 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.8136 | 0.9750 | 0.8248 |
| xgb | 0.9912 | 0.9554 | 0.9689 | 0.8925 |
| lgbm | 0.9918 | 0.9580 | 0.9634 | 0.9019 |
| ada | 0.8868 | 0.9213 | 0.9817 | 0.7863 |
| svm | 0.9516 | 0.5748 | 0.9811 | 0.6702 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7760 | 0.9849 | 0.8310 |
| xgb | 0.9954 | 0.5972 | 0.9453 | 0.7208 |
| lgbm | 0.9957 | 0.6192 | 0.9456 | 0.7333 |
| ada | 0.9395 | 0.0397 | 0.9544 | 0.7049 |
| svm | 0.9739 | 0.1138 | 0.4227 | 0.7527 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9962 | 0.6339 | 0.9703 | 0.7109 |
| xgb | 0.9909 | 0.4257 | 0.8963 | 0.5635 |
| lgbm | 0.9914 | 0.4484 | 0.8968 | 0.5789 |
| ada | 0.8860 | 0.0203 | 0.9127 | 0.5443 |
| svm | 0.9490 | 0.0603 | 0.2680 | 0.6035 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9962 | 0.9989 | 0.9997 | 0.9966 |
| xgb | 0.9911 | 0.9969 | 0.9988 | 0.9930 |
| lgbm | 0.9916 | 0.9972 | 0.9989 | 0.9933 |
| ada | 0.8885 | 0.8929 | 0.9990 | 0.9933 |
| svm | 0.9501 | 0.9785 | 0.9722 | 0.9955 |
