# stage5_ndwi_ratios: bands + NDWI + 4 band ratios (MARIDA)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8996 | 0.8302 | 0.8827 | 0.9221 | 0.9954 | 0.8990 | 358,442 | 12 |
| xgb | 0.8260 | 0.7299 | 0.7488 | 0.9599 | 0.9904 | 0.8178 | 358,442 | 5 |
| lgbm | 0.8410 | 0.7507 | 0.7706 | 0.9597 | 0.9914 | 0.8342 | 358,442 | 2 |
| ada | 0.6677 | 0.5961 | 0.6257 | 0.9370 | 0.9451 | 0.4293 | 358,442 | 198 |
| svm | 0.7890 | 0.6722 | 0.7110 | 0.9359 | 0.9859 | 0.7462 | 358,442 | 3 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9978 | 0.9108 | 0.9762 | 0.8037 |
| xgb | 0.9915 | 0.9685 | 0.9726 | 0.9068 |
| lgbm | 0.9926 | 0.9685 | 0.9732 | 0.9043 |
| ada | 0.9457 | 0.9764 | 0.9829 | 0.8429 |
| svm | 0.9877 | 0.9580 | 0.9909 | 0.8068 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7214 | 0.9913 | 0.8200 |
| xgb | 0.9996 | 0.4695 | 0.8992 | 0.6271 |
| lgbm | 0.9996 | 0.5041 | 0.9416 | 0.6372 |
| ada | 0.9993 | 0.0485 | 0.9118 | 0.5430 |
| svm | 0.9989 | 0.5028 | 0.5395 | 0.8028 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9978 | 0.9108 | 0.9762 | 0.8037 |
| xgb | 0.9915 | 0.9685 | 0.9726 | 0.9068 |
| lgbm | 0.9926 | 0.9685 | 0.9732 | 0.9043 |
| ada | 0.9457 | 0.9764 | 0.9829 | 0.8429 |
| svm | 0.9877 | 0.9580 | 0.9909 | 0.8068 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.8051 | 0.9837 | 0.8118 |
| xgb | 0.9955 | 0.6324 | 0.9344 | 0.7415 |
| lgbm | 0.9961 | 0.6631 | 0.9571 | 0.7476 |
| ada | 0.9718 | 0.0924 | 0.9460 | 0.6605 |
| svm | 0.9933 | 0.6594 | 0.6986 | 0.8048 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9959 | 0.6738 | 0.9680 | 0.6832 |
| xgb | 0.9911 | 0.4624 | 0.8769 | 0.5892 |
| lgbm | 0.9921 | 0.4960 | 0.9178 | 0.5970 |
| ada | 0.9451 | 0.0484 | 0.8976 | 0.4931 |
| svm | 0.9867 | 0.4919 | 0.5368 | 0.6734 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.9989 | 0.9997 | 0.9962 |
| xgb | 0.9913 | 0.9973 | 0.9986 | 0.9936 |
| lgbm | 0.9923 | 0.9976 | 0.9991 | 0.9938 |
| ada | 0.9463 | 0.9538 | 0.9988 | 0.9912 |
| svm | 0.9870 | 0.9976 | 0.9911 | 0.9960 |
