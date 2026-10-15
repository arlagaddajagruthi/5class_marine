# stage2e_ndmi: bands + NDMI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8861 | 0.8119 | 0.8633 | 0.9156 | 0.9953 | 0.8980 | 358,442 | 42 |
| xgb | 0.8450 | 0.7604 | 0.8255 | 0.8744 | 0.9944 | 0.8773 | 358,442 | 7 |
| lgbm | 0.6532 | 0.5500 | 0.5924 | 0.8142 | 0.9744 | 0.5933 | 358,442 | 6 |
| ada | 0.7698 | 0.6733 | 0.8303 | 0.7332 | 0.9918 | 0.7974 | 358,442 | 179 |
| svm | 0.6173 | 0.5312 | 0.6417 | 0.6418 | 0.9843 | 0.6673 | 358,442 | 4 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7426 | 0.9801 | 0.8236 |
| xgb | 0.9979 | 0.6304 | 0.9736 | 0.7782 |
| lgbm | 0.9878 | 0.3031 | 0.8456 | 0.4765 |
| ada | 0.9963 | 0.4683 | 0.9390 | 0.6754 |
| svm | 0.9921 | 0.0473 | 0.6725 | 0.7574 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.5906 | 0.9609 | 0.7002 |
| xgb | 0.9959 | 0.4603 | 0.9486 | 0.6369 |
| lgbm | 0.9760 | 0.1786 | 0.7325 | 0.3127 |
| ada | 0.9927 | 0.3057 | 0.8850 | 0.5099 |
| svm | 0.9844 | 0.0242 | 0.5065 | 0.6095 |
