# stage2f_ratios: bands + 4 band ratios (MARIDA)

Features (15): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.9059 | 0.8398 | 0.8902 | 0.9267 | 0.9957 | 0.9052 | 358,442 | 34 |
| xgb | 0.8687 | 0.7859 | 0.8172 | 0.9444 | 0.9934 | 0.8652 | 358,442 | 8 |
| lgbm | 0.8737 | 0.7926 | 0.8213 | 0.9480 | 0.9937 | 0.8709 | 358,442 | 6 |
| ada | 0.8210 | 0.7360 | 0.7775 | 0.9256 | 0.9922 | 0.8414 | 358,442 | 308 |
| svm | 0.8380 | 0.7449 | 0.7988 | 0.9045 | 0.9931 | 0.8558 | 358,442 | 6 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.9134 | 0.9799 | 0.8155 |
| xgb | 0.9951 | 0.9449 | 0.9580 | 0.8795 |
| lgbm | 0.9952 | 0.9396 | 0.9689 | 0.8882 |
| ada | 0.9944 | 0.9213 | 0.9738 | 0.8130 |
| svm | 0.9959 | 0.8661 | 0.9951 | 0.7609 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7311 | 0.9975 | 0.8342 |
| xgb | 0.9991 | 0.6050 | 0.9716 | 0.6931 |
| lgbm | 0.9992 | 0.6194 | 0.9625 | 0.7041 |
| ada | 0.9987 | 0.3619 | 0.9433 | 0.8060 |
| svm | 0.9984 | 0.5015 | 0.8649 | 0.8305 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.9134 | 0.9799 | 0.8155 |
| xgb | 0.9951 | 0.9449 | 0.9580 | 0.8795 |
| lgbm | 0.9952 | 0.9396 | 0.9689 | 0.8882 |
| ada | 0.9944 | 0.9213 | 0.9738 | 0.8130 |
| svm | 0.9959 | 0.8661 | 0.9951 | 0.7609 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.8121 | 0.9886 | 0.8247 |
| xgb | 0.9971 | 0.7377 | 0.9647 | 0.7753 |
| lgbm | 0.9972 | 0.7466 | 0.9657 | 0.7855 |
| ada | 0.9965 | 0.5196 | 0.9583 | 0.8095 |
| svm | 0.9971 | 0.6352 | 0.9255 | 0.7942 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.6837 | 0.9775 | 0.7018 |
| xgb | 0.9942 | 0.5844 | 0.9318 | 0.6330 |
| lgbm | 0.9944 | 0.5957 | 0.9336 | 0.6468 |
| ada | 0.9931 | 0.3510 | 0.9200 | 0.6800 |
| svm | 0.9942 | 0.4654 | 0.8613 | 0.6586 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9962 | 0.9990 | 0.9998 | 0.9965 |
| xgb | 0.9943 | 0.9984 | 0.9993 | 0.9948 |
| lgbm | 0.9945 | 0.9985 | 0.9993 | 0.9951 |
| ada | 0.9933 | 0.9959 | 0.9991 | 0.9961 |
| svm | 0.9944 | 0.9976 | 0.9983 | 0.9960 |
