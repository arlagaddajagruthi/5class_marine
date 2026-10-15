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
