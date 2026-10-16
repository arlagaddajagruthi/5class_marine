# stage5_ndwi_ratios: bands + NDWI + 4 band ratios (MARIDA)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.9042 | 0.8373 | 0.8865 | 0.9270 | 0.9956 | 0.9027 | 358,442 | 14 |
| xgb | 0.8669 | 0.7837 | 0.8147 | 0.9445 | 0.9934 | 0.8656 | 358,442 | 5 |
| lgbm | 0.8747 | 0.7941 | 0.8211 | 0.9517 | 0.9936 | 0.8700 | 358,442 | 2 |
| ada | 0.8208 | 0.7356 | 0.7778 | 0.9256 | 0.9923 | 0.8427 | 358,442 | 223 |
| svm | 0.8444 | 0.7426 | 0.7900 | 0.9242 | 0.9912 | 0.8232 | 358,442 | 6 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9978 | 0.9134 | 0.9799 | 0.8168 |
| xgb | 0.9951 | 0.9423 | 0.9604 | 0.8801 |
| lgbm | 0.9951 | 0.9528 | 0.9695 | 0.8894 |
| ada | 0.9945 | 0.9186 | 0.9701 | 0.8193 |
| svm | 0.9934 | 0.9186 | 0.9902 | 0.7944 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9982 | 0.7326 | 0.9969 | 0.8183 |
| xgb | 0.9991 | 0.5905 | 0.9681 | 0.7011 |
| lgbm | 0.9993 | 0.6205 | 0.9660 | 0.6985 |
| ada | 0.9987 | 0.3579 | 0.9332 | 0.8213 |
| svm | 0.9984 | 0.6295 | 0.7071 | 0.8252 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9978 | 0.9134 | 0.9799 | 0.8168 |
| xgb | 0.9951 | 0.9423 | 0.9604 | 0.8801 |
| lgbm | 0.9951 | 0.9528 | 0.9695 | 0.8894 |
| ada | 0.9945 | 0.9186 | 0.9701 | 0.8193 |
| svm | 0.9934 | 0.9186 | 0.9902 | 0.7944 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.8131 | 0.9883 | 0.8175 |
| xgb | 0.9971 | 0.7260 | 0.9642 | 0.7805 |
| lgbm | 0.9972 | 0.7516 | 0.9678 | 0.7825 |
| ada | 0.9966 | 0.5151 | 0.9513 | 0.8203 |
| svm | 0.9959 | 0.7471 | 0.8251 | 0.8095 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.6850 | 0.9769 | 0.6914 |
| xgb | 0.9942 | 0.5698 | 0.9309 | 0.6400 |
| lgbm | 0.9943 | 0.6020 | 0.9375 | 0.6427 |
| ada | 0.9932 | 0.3469 | 0.9071 | 0.6953 |
| svm | 0.9918 | 0.5963 | 0.7022 | 0.6800 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.9990 | 0.9998 | 0.9963 |
| xgb | 0.9943 | 0.9983 | 0.9993 | 0.9950 |
| lgbm | 0.9944 | 0.9985 | 0.9993 | 0.9950 |
| ada | 0.9934 | 0.9958 | 0.9990 | 0.9964 |
| svm | 0.9920 | 0.9985 | 0.9956 | 0.9962 |
