# stage5_ndwi_ratios: bands + NDWI + 4 band ratios (MARIDA)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8788 | 0.8034 | 0.8483 | 0.9269 | 0.9949 | 0.8890 | 358,442 | 22 |
| xgb | 0.8822 | 0.8051 | 0.8635 | 0.9064 | 0.9950 | 0.8910 | 358,442 | 5 |
| lgbm | 0.5861 | 0.4766 | 0.5109 | 0.8523 | 0.9600 | 0.4860 | 358,442 | 2 |
| ada | 0.8317 | 0.7459 | 0.8577 | 0.8209 | 0.9940 | 0.8600 | 358,442 | 208 |
| svm | 0.8845 | 0.8093 | 0.9127 | 0.8628 | 0.9953 | 0.8904 | 358,442 | 2 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9970 | 0.9108 | 0.9756 | 0.8242 |
| xgb | 0.9976 | 0.8556 | 0.9573 | 0.8149 |
| lgbm | 0.9633 | 0.8058 | 0.8818 | 0.7584 |
| ada | 0.9987 | 0.6667 | 0.9628 | 0.6553 |
| svm | 0.9991 | 0.7717 | 0.9793 | 0.7012 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9983 | 0.5871 | 0.9932 | 0.8146 |
| xgb | 0.9983 | 0.6749 | 0.9905 | 0.7904 |
| lgbm | 0.9977 | 0.0946 | 0.4241 | 0.5272 |
| ada | 0.9961 | 0.5270 | 0.9887 | 0.9190 |
| svm | 0.9967 | 0.7636 | 0.9853 | 0.9054 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9970 | 0.9108 | 0.9756 | 0.8242 |
| xgb | 0.9976 | 0.8556 | 0.9573 | 0.8149 |
| lgbm | 0.9633 | 0.8058 | 0.8818 | 0.7584 |
| ada | 0.9987 | 0.6667 | 0.9628 | 0.6553 |
| svm | 0.9991 | 0.7717 | 0.9793 | 0.7012 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9977 | 0.7140 | 0.9843 | 0.8194 |
| xgb | 0.9979 | 0.7546 | 0.9737 | 0.8024 |
| lgbm | 0.9802 | 0.1694 | 0.5727 | 0.6220 |
| ada | 0.9974 | 0.5886 | 0.9756 | 0.7650 |
| svm | 0.9979 | 0.7676 | 0.9823 | 0.7903 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9954 | 0.5552 | 0.9691 | 0.6940 |
| xgb | 0.9959 | 0.6059 | 0.9487 | 0.6701 |
| lgbm | 0.9612 | 0.0925 | 0.4013 | 0.4514 |
| ada | 0.9948 | 0.4171 | 0.9524 | 0.6195 |
| svm | 0.9958 | 0.6229 | 0.9652 | 0.6534 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9955 | 0.9982 | 0.9997 | 0.9963 |
| xgb | 0.9960 | 0.9987 | 0.9995 | 0.9959 |
| lgbm | 0.9620 | 0.9810 | 0.9864 | 0.9906 |
| ada | 0.9949 | 0.9978 | 0.9995 | 0.9959 |
| svm | 0.9959 | 0.9989 | 0.9996 | 0.9962 |
