# stage2d_fdi: bands + FDI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FDI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7065 | 0.5607 | 0.7416 | 0.7339 | 0.7211 | 0.3659 | 761,887 | 128 |
| xgb | 0.6602 | 0.5184 | 0.6522 | 0.7877 | 0.7510 | 0.4237 | 761,887 | 15 |
| lgbm | 0.6738 | 0.5297 | 0.6694 | 0.7867 | 0.7394 | 0.4091 | 761,887 | 10 |
| ada | 0.6251 | 0.4793 | 0.6176 | 0.7287 | 0.7487 | 0.4128 | 761,887 | 743 |
| svm | 0.5320 | 0.4131 | 0.5596 | 0.5735 | 0.7294 | 0.3186 | 761,887 | 13 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8064 | 0.6911 | 0.8259 | 0.5027 |
| xgb | 0.8307 | 0.4339 | 0.8322 | 0.5438 |
| lgbm | 0.8203 | 0.4985 | 0.8418 | 0.5346 |
| ada | 0.8303 | 0.3772 | 0.7569 | 0.5361 |
| svm | 0.8208 | 0.0845 | 0.7557 | 0.4670 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6755 | 0.5280 | 0.7035 | 0.3358 |
| xgb | 0.7104 | 0.2770 | 0.7127 | 0.3735 |
| lgbm | 0.6953 | 0.3320 | 0.7268 | 0.3648 |
| ada | 0.7098 | 0.2324 | 0.6088 | 0.3662 |
| svm | 0.6961 | 0.0441 | 0.6074 | 0.3047 |
