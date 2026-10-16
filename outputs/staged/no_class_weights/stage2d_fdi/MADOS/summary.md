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

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6998 | 0.7276 | 0.7197 | 0.8232 |
| xgb | 0.7382 | 0.7205 | 0.7018 | 0.8060 |
| lgbm | 0.7414 | 0.5827 | 0.7825 | 0.8042 |
| ada | 0.8206 | 0.3330 | 0.6565 | 0.6665 |
| svm | 0.8581 | 0.0000 | 0.4937 | 0.4717 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9499 | 0.6556 | 0.9898 | 0.3676 |
| xgb | 0.9489 | 0.5025 | 0.9864 | 0.3945 |
| lgbm | 0.9494 | 0.1827 | 0.7187 | 0.4055 |
| ada | 0.9183 | 0.4451 | 0.9703 | 0.4397 |
| svm | 0.8766 | 0.0000 | 0.9890 | 0.4143 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6998 | 0.7276 | 0.7197 | 0.8232 |
| xgb | 0.7382 | 0.7205 | 0.7018 | 0.8060 |
| lgbm | 0.7414 | 0.5827 | 0.7825 | 0.8042 |
| ada | 0.8206 | 0.3330 | 0.6565 | 0.6665 |
| svm | 0.8581 | 0.0000 | 0.4937 | 0.4717 |

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

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7255 | 0.9982 | 0.9975 | 0.7218 |
| xgb | 0.7545 | 0.9973 | 0.9974 | 0.7501 |
| lgbm | 0.7573 | 0.9919 | 0.9955 | 0.7599 |
| ada | 0.7945 | 0.9971 | 0.9969 | 0.7934 |
| svm | 0.7862 | 0.9973 | 0.9956 | 0.7913 |
