# stage2b_ndwi: bands + NDWI (MARIDA)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.9013 | 0.8329 | 0.8784 | 0.9300 | 0.9956 | 0.9044 | 358,442 | 42 |
| xgb | 0.8677 | 0.7846 | 0.8505 | 0.8903 | 0.9946 | 0.8828 | 358,442 | 6 |
| lgbm | 0.7320 | 0.6308 | 0.6757 | 0.8737 | 0.9852 | 0.7274 | 358,442 | 5 |
| ada | 0.7965 | 0.7090 | 0.8726 | 0.7427 | 0.9933 | 0.8365 | 358,442 | 190 |
| svm | 0.7339 | 0.6716 | 0.8728 | 0.6896 | 0.9937 | 0.8437 | 358,442 | 2 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7931 | 0.9849 | 0.8290 |
| xgb | 0.9979 | 0.7192 | 0.9594 | 0.7944 |
| lgbm | 0.9931 | 0.3459 | 0.8591 | 0.7297 |
| ada | 0.9969 | 0.4858 | 0.9712 | 0.7319 |
| svm | 0.9970 | 0.1826 | 0.9745 | 0.7817 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.6571 | 0.9703 | 0.7079 |
| xgb | 0.9958 | 0.5615 | 0.9220 | 0.6590 |
| lgbm | 0.9864 | 0.2091 | 0.7530 | 0.5745 |
| ada | 0.9938 | 0.3208 | 0.9440 | 0.5772 |
| svm | 0.9939 | 0.1005 | 0.9503 | 0.6416 |
