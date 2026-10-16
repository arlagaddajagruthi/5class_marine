# stage3_all_indices: bands + all 5 indices (MARIDA)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8909 | 0.8192 | 0.8732 | 0.9125 | 0.9955 | 0.9006 | 358,442 | 131 |
| xgb | 0.8834 | 0.8063 | 0.8639 | 0.9089 | 0.9952 | 0.8948 | 358,442 | 15 |
| lgbm | 0.7329 | 0.6079 | 0.7218 | 0.7736 | 0.9864 | 0.7098 | 358,442 | 10 |
| ada | 0.8154 | 0.7275 | 0.8468 | 0.7976 | 0.9936 | 0.8477 | 358,442 | 425 |
| svm | 0.8140 | 0.7326 | 0.8891 | 0.7647 | 0.9944 | 0.8660 | 358,442 | 8 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9977 | 0.8346 | 0.9762 | 0.8416 |
| xgb | 0.9977 | 0.8556 | 0.9543 | 0.8280 |
| lgbm | 0.9936 | 0.7612 | 0.6453 | 0.6944 |
| ada | 0.9987 | 0.6063 | 0.9616 | 0.6236 |
| svm | 0.9992 | 0.3885 | 0.9775 | 0.6938 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9983 | 0.6824 | 0.9963 | 0.8158 |
| xgb | 0.9984 | 0.6667 | 0.9812 | 0.8094 |
| lgbm | 0.9955 | 0.4890 | 0.8781 | 0.5246 |
| ada | 0.9958 | 0.5145 | 0.9893 | 0.8877 |
| svm | 0.9957 | 0.6697 | 0.9828 | 0.9081 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9977 | 0.8346 | 0.9762 | 0.8416 |
| xgb | 0.9977 | 0.8556 | 0.9543 | 0.8280 |
| lgbm | 0.9936 | 0.7612 | 0.6453 | 0.6944 |
| ada | 0.9987 | 0.6063 | 0.9616 | 0.6236 |
| svm | 0.9992 | 0.3885 | 0.9775 | 0.6938 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7509 | 0.9861 | 0.8285 |
| xgb | 0.9980 | 0.7494 | 0.9676 | 0.8185 |
| lgbm | 0.9945 | 0.5955 | 0.7439 | 0.5977 |
| ada | 0.9972 | 0.5566 | 0.9753 | 0.7326 |
| svm | 0.9974 | 0.4917 | 0.9801 | 0.7866 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.6011 | 0.9727 | 0.7072 |
| xgb | 0.9961 | 0.5993 | 0.9372 | 0.6928 |
| lgbm | 0.9891 | 0.4240 | 0.5923 | 0.4262 |
| ada | 0.9945 | 0.3856 | 0.9517 | 0.5780 |
| svm | 0.9949 | 0.3260 | 0.9611 | 0.6483 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.9987 | 0.9997 | 0.9965 |
| xgb | 0.9961 | 0.9986 | 0.9993 | 0.9963 |
| lgbm | 0.9893 | 0.9975 | 0.9954 | 0.9905 |
| ada | 0.9946 | 0.9977 | 0.9995 | 0.9954 |
| svm | 0.9950 | 0.9981 | 0.9996 | 0.9962 |
