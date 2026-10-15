# stage2g_specstats: bands + 4 spectral statistics (MARIDA)

Features (15): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, spec_mean, spec_std, spec_max, spec_norm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8852 | 0.8104 | 0.8662 | 0.9120 | 0.9953 | 0.8973 | 358,442 | 31 |
| xgb | 0.8486 | 0.7594 | 0.7910 | 0.9393 | 0.9927 | 0.8535 | 358,442 | 7 |
| lgbm | 0.8627 | 0.7799 | 0.8102 | 0.9450 | 0.9935 | 0.8672 | 358,442 | 7 |
| ada | 0.7907 | 0.7051 | 0.7622 | 0.8787 | 0.9913 | 0.8214 | 358,442 | 262 |
| svm | 0.7440 | 0.6789 | 0.7337 | 0.8438 | 0.9850 | 0.7194 | 358,442 | 6 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.7432 | 0.9781 | 0.8215 |
| xgb | 0.9969 | 0.6800 | 0.9524 | 0.7651 |
| lgbm | 0.9972 | 0.7038 | 0.9729 | 0.7767 |
| ada | 0.9965 | 0.4324 | 0.9547 | 0.7793 |
| svm | 0.9927 | 0.2206 | 0.9846 | 0.7780 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.5914 | 0.9572 | 0.6970 |
| xgb | 0.9938 | 0.5152 | 0.9092 | 0.6195 |
| lgbm | 0.9944 | 0.5430 | 0.9472 | 0.6349 |
| ada | 0.9929 | 0.2758 | 0.9133 | 0.6383 |
| svm | 0.9855 | 0.1240 | 0.9696 | 0.6367 |
