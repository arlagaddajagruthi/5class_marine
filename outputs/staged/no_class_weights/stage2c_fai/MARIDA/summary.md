# stage2c_fai: bands + FAI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FAI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7832 | 0.7138 | 0.7496 | 0.9032 | 0.9888 | 0.7857 | 358,442 | 45 |
| xgb | 0.8547 | 0.7707 | 0.8391 | 0.8741 | 0.9943 | 0.8765 | 358,442 | 6 |
| lgbm | 0.8078 | 0.6953 | 0.7524 | 0.8844 | 0.9891 | 0.7876 | 358,442 | 5 |
| ada | 0.7715 | 0.6815 | 0.8330 | 0.7291 | 0.9924 | 0.8135 | 358,442 | 192 |
| svm | 0.6124 | 0.5282 | 0.6458 | 0.6264 | 0.9851 | 0.6708 | 358,442 | 4 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9909 | 0.7953 | 0.9787 | 0.8478 |
| xgb | 0.9974 | 0.7454 | 0.9555 | 0.7981 |
| lgbm | 0.9917 | 0.7664 | 0.9616 | 0.8180 |
| ada | 0.9989 | 0.4199 | 0.9378 | 0.5596 |
| svm | 0.9924 | 0.0157 | 0.8769 | 0.6205 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9983 | 0.2075 | 0.9811 | 0.8115 |
| xgb | 0.9981 | 0.6094 | 0.9905 | 0.7581 |
| lgbm | 0.9982 | 0.5935 | 0.6459 | 0.7720 |
| ada | 0.9943 | 0.4638 | 0.9897 | 0.8842 |
| svm | 0.9927 | 0.1071 | 0.5694 | 0.9140 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9909 | 0.7953 | 0.9787 | 0.8478 |
| xgb | 0.9974 | 0.7454 | 0.9555 | 0.7981 |
| lgbm | 0.9917 | 0.7664 | 0.9616 | 0.8180 |
| ada | 0.9989 | 0.4199 | 0.9378 | 0.5596 |
| svm | 0.9924 | 0.0157 | 0.8769 | 0.6205 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9946 | 0.3292 | 0.9799 | 0.8293 |
| xgb | 0.9978 | 0.6706 | 0.9727 | 0.7776 |
| lgbm | 0.9950 | 0.6690 | 0.7728 | 0.7943 |
| ada | 0.9966 | 0.4408 | 0.9631 | 0.6854 |
| svm | 0.9925 | 0.0275 | 0.6905 | 0.7392 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9892 | 0.1970 | 0.9605 | 0.7084 |
| xgb | 0.9955 | 0.5044 | 0.9469 | 0.6361 |
| lgbm | 0.9900 | 0.5026 | 0.6297 | 0.6588 |
| ada | 0.9932 | 0.2827 | 0.9288 | 0.5214 |
| svm | 0.9852 | 0.0139 | 0.5273 | 0.5863 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9894 | 0.9922 | 0.9996 | 0.9965 |
| xgb | 0.9956 | 0.9982 | 0.9994 | 0.9954 |
| lgbm | 0.9902 | 0.9982 | 0.9941 | 0.9957 |
| ada | 0.9933 | 0.9974 | 0.9993 | 0.9948 |
| svm | 0.9854 | 0.9973 | 0.9919 | 0.9955 |
