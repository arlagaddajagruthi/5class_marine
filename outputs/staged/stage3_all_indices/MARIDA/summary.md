# stage3_all_indices: bands + all 5 indices (MARIDA)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8996 | 0.8303 | 0.8851 | 0.9189 | 0.9956 | 0.9028 | 358,442 | 12 |
| xgb | 0.8298 | 0.7318 | 0.7517 | 0.9587 | 0.9899 | 0.8106 | 358,442 | 5 |
| lgbm | 0.8418 | 0.7492 | 0.7682 | 0.9594 | 0.9907 | 0.8234 | 358,442 | 2 |
| ada | 0.6670 | 0.5939 | 0.6240 | 0.9341 | 0.9466 | 0.4376 | 358,442 | 174 |
| svm | 0.7467 | 0.6253 | 0.6814 | 0.9110 | 0.9835 | 0.7105 | 358,442 | 3 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.7930 | 0.9840 | 0.8233 |
| xgb | 0.9953 | 0.6814 | 0.9252 | 0.7170 |
| lgbm | 0.9957 | 0.6998 | 0.9461 | 0.7255 |
| ada | 0.9727 | 0.0938 | 0.9375 | 0.6641 |
| svm | 0.9924 | 0.5508 | 0.6637 | 0.7801 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.6570 | 0.9685 | 0.6996 |
| xgb | 0.9907 | 0.5168 | 0.8609 | 0.5589 |
| lgbm | 0.9914 | 0.5382 | 0.8978 | 0.5693 |
| ada | 0.9469 | 0.0492 | 0.8823 | 0.4971 |
| svm | 0.9849 | 0.3801 | 0.4967 | 0.6394 |
