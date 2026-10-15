# stage2d_fdi: bands + FDI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FDI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8910 | 0.8200 | 0.8687 | 0.9208 | 0.9956 | 0.9034 | 358,442 | 30 |
| xgb | 0.8508 | 0.7629 | 0.7936 | 0.9430 | 0.9929 | 0.8573 | 358,442 | 7 |
| lgbm | 0.8543 | 0.7666 | 0.7971 | 0.9418 | 0.9930 | 0.8582 | 358,442 | 5 |
| ada | 0.8033 | 0.7121 | 0.7687 | 0.8924 | 0.9921 | 0.8365 | 358,442 | 219 |
| svm | 0.5831 | 0.4873 | 0.5609 | 0.7380 | 0.9624 | 0.4842 | 358,442 | 5 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7503 | 0.9902 | 0.8255 |
| xgb | 0.9970 | 0.6773 | 0.9553 | 0.7738 |
| lgbm | 0.9970 | 0.6952 | 0.9549 | 0.7699 |
| ada | 0.9971 | 0.5073 | 0.9462 | 0.7628 |
| svm | 0.9812 | 0.1236 | 0.4623 | 0.7651 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9962 | 0.6004 | 0.9806 | 0.7028 |
| xgb | 0.9940 | 0.5121 | 0.9145 | 0.6310 |
| lgbm | 0.9940 | 0.5328 | 0.9137 | 0.6259 |
| ada | 0.9942 | 0.3398 | 0.8980 | 0.6165 |
| svm | 0.9630 | 0.0659 | 0.3007 | 0.6196 |
