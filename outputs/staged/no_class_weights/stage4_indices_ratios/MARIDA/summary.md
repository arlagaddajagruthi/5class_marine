# stage4_indices_ratios: bands + all 5 indices + 4 band ratios (MARIDA)

Features (20): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8354 | 0.7577 | 0.7976 | 0.9235 | 0.9930 | 0.8546 | 358,442 | 136 |
| xgb | 0.8869 | 0.8121 | 0.8673 | 0.9116 | 0.9952 | 0.8946 | 358,442 | 17 |
| lgbm | 0.7193 | 0.5931 | 0.6359 | 0.8760 | 0.9811 | 0.6724 | 358,442 | 12 |
| ada | 0.8196 | 0.7339 | 0.8461 | 0.8064 | 0.9937 | 0.8529 | 358,442 | 543 |
| svm | 0.8770 | 0.7998 | 0.8876 | 0.8805 | 0.9952 | 0.8883 | 358,442 | 11 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9951 | 0.8898 | 0.9768 | 0.8323 |
| xgb | 0.9976 | 0.8583 | 0.9653 | 0.8255 |
| lgbm | 0.9843 | 0.8268 | 0.9171 | 0.7758 |
| ada | 0.9985 | 0.6115 | 0.9634 | 0.6522 |
| svm | 0.9988 | 0.8478 | 0.9823 | 0.6932 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9984 | 0.3844 | 0.9950 | 0.8126 |
| xgb | 0.9983 | 0.6827 | 0.9900 | 0.7982 |
| lgbm | 0.9975 | 0.3777 | 0.5100 | 0.6584 |
| ada | 0.9959 | 0.4936 | 0.9918 | 0.9028 |
| svm | 0.9970 | 0.6525 | 0.9847 | 0.9163 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9951 | 0.8898 | 0.9768 | 0.8323 |
| xgb | 0.9976 | 0.8583 | 0.9653 | 0.8255 |
| lgbm | 0.9843 | 0.8268 | 0.9171 | 0.7758 |
| ada | 0.9985 | 0.6115 | 0.9634 | 0.6522 |
| svm | 0.9988 | 0.8478 | 0.9823 | 0.6932 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9967 | 0.5368 | 0.9859 | 0.8223 |
| xgb | 0.9980 | 0.7605 | 0.9775 | 0.8116 |
| lgbm | 0.9909 | 0.5185 | 0.6555 | 0.7123 |
| ada | 0.9972 | 0.5463 | 0.9774 | 0.7573 |
| svm | 0.9979 | 0.7374 | 0.9835 | 0.7893 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9935 | 0.3669 | 0.9721 | 0.6983 |
| xgb | 0.9959 | 0.6135 | 0.9559 | 0.6829 |
| lgbm | 0.9819 | 0.3500 | 0.4875 | 0.5531 |
| ada | 0.9945 | 0.3758 | 0.9559 | 0.6094 |
| svm | 0.9958 | 0.5841 | 0.9676 | 0.6519 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9936 | 0.9963 | 0.9997 | 0.9963 |
| xgb | 0.9960 | 0.9987 | 0.9995 | 0.9961 |
| lgbm | 0.9823 | 0.9963 | 0.9900 | 0.9936 |
| ada | 0.9946 | 0.9976 | 0.9995 | 0.9958 |
| svm | 0.9959 | 0.9985 | 0.9997 | 0.9962 |
