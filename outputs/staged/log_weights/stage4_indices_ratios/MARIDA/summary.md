# stage4_indices_ratios: bands + all 5 indices + 4 band ratios (MARIDA)

Features (20): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8981 | 0.8284 | 0.8855 | 0.9152 | 0.9956 | 0.9029 | 358,442 | 51 |
| xgb | 0.8702 | 0.7875 | 0.8164 | 0.9493 | 0.9937 | 0.8705 | 358,442 | 9 |
| lgbm | 0.8768 | 0.7974 | 0.8251 | 0.9495 | 0.9938 | 0.8740 | 358,442 | 7 |
| ada | 0.8120 | 0.7235 | 0.7633 | 0.9101 | 0.9917 | 0.8323 | 358,442 | 414 |
| svm | 0.8079 | 0.6949 | 0.7422 | 0.9261 | 0.9870 | 0.7587 | 358,442 | 7 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.8766 | 0.9768 | 0.8093 |
| xgb | 0.9952 | 0.9528 | 0.9634 | 0.8857 |
| lgbm | 0.9952 | 0.9344 | 0.9726 | 0.8957 |
| ada | 0.9940 | 0.8556 | 0.9689 | 0.8217 |
| svm | 0.9892 | 0.9396 | 0.9854 | 0.7901 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7106 | 0.9932 | 0.8401 |
| xgb | 0.9992 | 0.5951 | 0.9564 | 0.7148 |
| lgbm | 0.9993 | 0.6246 | 0.9696 | 0.7069 |
| ada | 0.9987 | 0.3688 | 0.9375 | 0.7483 |
| svm | 0.9985 | 0.5812 | 0.5500 | 0.8391 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.8766 | 0.9768 | 0.8093 |
| xgb | 0.9952 | 0.9528 | 0.9634 | 0.8857 |
| lgbm | 0.9952 | 0.9344 | 0.9726 | 0.8957 |
| ada | 0.9940 | 0.8556 | 0.9689 | 0.8217 |
| svm | 0.9892 | 0.9396 | 0.9854 | 0.7901 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7850 | 0.9849 | 0.8244 |
| xgb | 0.9972 | 0.7326 | 0.9599 | 0.7911 |
| lgbm | 0.9972 | 0.7487 | 0.9711 | 0.7901 |
| ada | 0.9963 | 0.5154 | 0.9530 | 0.7833 |
| svm | 0.9938 | 0.7182 | 0.7060 | 0.8138 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.6460 | 0.9703 | 0.7013 |
| xgb | 0.9945 | 0.5780 | 0.9229 | 0.6544 |
| lgbm | 0.9945 | 0.5983 | 0.9438 | 0.6531 |
| ada | 0.9927 | 0.3472 | 0.9101 | 0.6438 |
| svm | 0.9877 | 0.5603 | 0.5455 | 0.6861 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9962 | 0.9988 | 0.9997 | 0.9965 |
| xgb | 0.9946 | 0.9983 | 0.9992 | 0.9952 |
| lgbm | 0.9946 | 0.9985 | 0.9994 | 0.9952 |
| ada | 0.9929 | 0.9961 | 0.9990 | 0.9954 |
| svm | 0.9879 | 0.9982 | 0.9915 | 0.9963 |
