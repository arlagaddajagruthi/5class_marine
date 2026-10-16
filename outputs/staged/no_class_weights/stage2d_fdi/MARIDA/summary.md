# stage2d_fdi: bands + FDI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FDI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8886 | 0.8158 | 0.8649 | 0.9189 | 0.9954 | 0.9010 | 358,442 | 46 |
| xgb | 0.8636 | 0.7818 | 0.8381 | 0.9003 | 0.9949 | 0.8889 | 358,442 | 6 |
| lgbm | 0.7053 | 0.5786 | 0.6167 | 0.8534 | 0.9801 | 0.6646 | 358,442 | 5 |
| ada | 0.7883 | 0.6939 | 0.8233 | 0.7785 | 0.9924 | 0.8194 | 358,442 | 189 |
| svm | 0.6117 | 0.5273 | 0.6465 | 0.6259 | 0.9850 | 0.6693 | 358,442 | 4 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9976 | 0.8609 | 0.9811 | 0.8360 |
| xgb | 0.9975 | 0.8215 | 0.9640 | 0.8180 |
| lgbm | 0.9833 | 0.7060 | 0.9427 | 0.7814 |
| ada | 0.9984 | 0.6194 | 0.9062 | 0.5901 |
| svm | 0.9923 | 0.0157 | 0.8751 | 0.6205 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9986 | 0.6626 | 0.9871 | 0.8113 |
| xgb | 0.9985 | 0.5722 | 0.9623 | 0.8195 |
| lgbm | 0.9982 | 0.4063 | 0.5805 | 0.4818 |
| ada | 0.9952 | 0.4207 | 0.9920 | 0.8854 |
| svm | 0.9926 | 0.1132 | 0.5660 | 0.9140 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9976 | 0.8609 | 0.9811 | 0.8360 |
| xgb | 0.9975 | 0.8215 | 0.9640 | 0.8180 |
| lgbm | 0.9833 | 0.7060 | 0.9427 | 0.7814 |
| ada | 0.9984 | 0.6194 | 0.9062 | 0.5901 |
| svm | 0.9923 | 0.0157 | 0.8751 | 0.6205 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7489 | 0.9841 | 0.8235 |
| xgb | 0.9980 | 0.6746 | 0.9632 | 0.8188 |
| lgbm | 0.9907 | 0.5158 | 0.7185 | 0.5961 |
| ada | 0.9968 | 0.5011 | 0.9471 | 0.7082 |
| svm | 0.9925 | 0.0276 | 0.6874 | 0.7392 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9962 | 0.5985 | 0.9687 | 0.6999 |
| xgb | 0.9960 | 0.5089 | 0.9289 | 0.6932 |
| lgbm | 0.9815 | 0.3475 | 0.5607 | 0.4246 |
| ada | 0.9936 | 0.3343 | 0.8996 | 0.5482 |
| svm | 0.9851 | 0.0140 | 0.5237 | 0.5863 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9962 | 0.9986 | 0.9997 | 0.9964 |
| xgb | 0.9961 | 0.9981 | 0.9992 | 0.9963 |
| lgbm | 0.9819 | 0.9968 | 0.9923 | 0.9892 |
| ada | 0.9938 | 0.9970 | 0.9990 | 0.9951 |
| svm | 0.9853 | 0.9973 | 0.9918 | 0.9955 |
