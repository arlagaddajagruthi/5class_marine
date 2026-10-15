# stage2f_ratios: bands + 4 band ratios (MARIDA)

Features (15): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, red/green, nir/red, nir/green, swir1/nir

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.9056 | 0.8386 | 0.8916 | 0.9236 | 0.9957 | 0.9040 | 358,442 | 9 |
| xgb | 0.8258 | 0.7296 | 0.7488 | 0.9599 | 0.9904 | 0.8178 | 358,442 | 3 |
| lgbm | 0.8416 | 0.7510 | 0.7714 | 0.9579 | 0.9915 | 0.8355 | 358,442 | 2 |
| ada | 0.6677 | 0.5961 | 0.6257 | 0.9370 | 0.9451 | 0.4293 | 358,442 | 157 |
| svm | 0.7946 | 0.6853 | 0.7315 | 0.9245 | 0.9895 | 0.7980 | 358,442 | 3 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9981 | 0.8189 | 0.9837 | 0.8215 |
| xgb | 0.9955 | 0.6325 | 0.9335 | 0.7417 |
| lgbm | 0.9961 | 0.6661 | 0.9540 | 0.7503 |
| ada | 0.9718 | 0.0924 | 0.9460 | 0.6605 |
| svm | 0.9954 | 0.5692 | 0.8172 | 0.7965 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.6934 | 0.9679 | 0.6971 |
| xgb | 0.9911 | 0.4625 | 0.8753 | 0.5895 |
| lgbm | 0.9922 | 0.4993 | 0.9121 | 0.6003 |
| ada | 0.9451 | 0.0484 | 0.8976 | 0.4931 |
| svm | 0.9909 | 0.3978 | 0.6908 | 0.6619 |
