# stage2e_ndmi: bands + NDMI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7028 | 0.5566 | 0.7311 | 0.7381 | 0.7234 | 0.3709 | 761,887 | 141 |
| xgb | 0.6828 | 0.5395 | 0.6924 | 0.7339 | 0.7685 | 0.4349 | 761,887 | 13 |
| lgbm | 0.6706 | 0.5275 | 0.6737 | 0.7237 | 0.7688 | 0.4362 | 761,887 | 9 |
| ada | 0.5609 | 0.4475 | 0.7195 | 0.5524 | 0.7752 | 0.4102 | 761,887 | 620 |
| svm | 0.4989 | 0.3931 | 0.5595 | 0.4720 | 0.7586 | 0.2915 | 761,887 | 10 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7044 | 0.7215 | 0.7145 | 0.8121 |
| xgb | 0.7618 | 0.6578 | 0.7119 | 0.8042 |
| lgbm | 0.7619 | 0.6064 | 0.7206 | 0.8060 |
| ada | 0.7914 | 0.0401 | 0.6614 | 0.7168 |
| svm | 0.8189 | 0.0000 | 0.5711 | 0.4980 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9472 | 0.6207 | 0.9884 | 0.3681 |
| xgb | 0.9502 | 0.4159 | 0.9865 | 0.4169 |
| lgbm | 0.9509 | 0.3821 | 0.9440 | 0.4178 |
| ada | 0.9253 | 0.5652 | 0.9669 | 0.4205 |
| svm | 0.8783 | 0.0000 | 0.9910 | 0.3685 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7044 | 0.7215 | 0.7145 | 0.8121 |
| xgb | 0.7618 | 0.6578 | 0.7119 | 0.8042 |
| lgbm | 0.7619 | 0.6064 | 0.7206 | 0.8060 |
| ada | 0.7914 | 0.0401 | 0.6614 | 0.7168 |
| svm | 0.8189 | 0.0000 | 0.5711 | 0.4980 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8080 | 0.6673 | 0.8294 | 0.5066 |
| xgb | 0.8456 | 0.5096 | 0.8270 | 0.5492 |
| lgbm | 0.8460 | 0.4688 | 0.8173 | 0.5503 |
| ada | 0.8531 | 0.0749 | 0.7855 | 0.5301 |
| svm | 0.8476 | 0.0000 | 0.7246 | 0.4236 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6778 | 0.5007 | 0.7085 | 0.3392 |
| xgb | 0.7325 | 0.3419 | 0.7050 | 0.3785 |
| lgbm | 0.7330 | 0.3062 | 0.6911 | 0.3796 |
| ada | 0.7439 | 0.0389 | 0.6467 | 0.3606 |
| svm | 0.7355 | 0.0000 | 0.5681 | 0.2687 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7274 | 0.9981 | 0.9975 | 0.7238 |
| xgb | 0.7735 | 0.9966 | 0.9975 | 0.7694 |
| lgbm | 0.7741 | 0.9963 | 0.9972 | 0.7700 |
| ada | 0.7782 | 0.9973 | 0.9969 | 0.7780 |
| svm | 0.7602 | 0.9973 | 0.9963 | 0.7633 |
