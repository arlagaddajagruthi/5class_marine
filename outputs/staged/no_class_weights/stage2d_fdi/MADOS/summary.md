# stage2d_fdi: bands + FDI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FDI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7093 | 0.5641 | 0.7407 | 0.7426 | 0.7216 | 0.3720 | 761,887 | 79 |
| xgb | 0.6931 | 0.5465 | 0.7081 | 0.7416 | 0.7497 | 0.4068 | 761,887 | 14 |
| lgbm | 0.5998 | 0.4607 | 0.5641 | 0.7277 | 0.7523 | 0.4128 | 761,887 | 9 |
| ada | 0.6402 | 0.5010 | 0.6933 | 0.6191 | 0.7910 | 0.4217 | 761,887 | 572 |
| svm | 0.4918 | 0.3849 | 0.5700 | 0.4559 | 0.7852 | 0.3212 | 761,887 | 9 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8059 | 0.6897 | 0.8334 | 0.5083 |
| xgb | 0.8304 | 0.5921 | 0.8201 | 0.5297 |
| lgbm | 0.8326 | 0.2781 | 0.7492 | 0.5391 |
| ada | 0.8667 | 0.3810 | 0.7832 | 0.5298 |
| svm | 0.8673 | 0.0000 | 0.6586 | 0.4411 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6749 | 0.5264 | 0.7143 | 0.3407 |
| xgb | 0.7100 | 0.4205 | 0.6951 | 0.3603 |
| lgbm | 0.7132 | 0.1615 | 0.5990 | 0.3691 |
| ada | 0.7648 | 0.2353 | 0.6436 | 0.3604 |
| svm | 0.7656 | 0.0000 | 0.4910 | 0.2830 |
