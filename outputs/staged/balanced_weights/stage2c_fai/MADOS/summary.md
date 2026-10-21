# stage2c_fai: bands + FAI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, FAI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7004 | 0.5542 | 0.7394 | 0.7182 | 0.7259 | 0.3642 | 761,887 | 37 |
| xgb | 0.6154 | 0.4669 | 0.5678 | 0.8026 | 0.7479 | 0.4210 | 761,887 | 6 |
| lgbm | 0.6405 | 0.4953 | 0.6087 | 0.8065 | 0.7441 | 0.4165 | 761,887 | 3 |
| ada | 0.5300 | 0.4092 | 0.5329 | 0.7692 | 0.6656 | 0.3181 | 761,887 | 364 |
| svm | 0.5370 | 0.4131 | 0.5369 | 0.6953 | 0.7108 | 0.3197 | 761,887 | 6 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7142 | 0.6680 | 0.7083 | 0.7823 |
| xgb | 0.7271 | 0.8726 | 0.7689 | 0.8421 |
| lgbm | 0.7212 | 0.8849 | 0.7724 | 0.8474 |
| ada | 0.6358 | 0.8448 | 0.8009 | 0.7953 |
| svm | 0.7170 | 0.6691 | 0.7132 | 0.6820 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9395 | 0.6619 | 0.9892 | 0.3669 |
| xgb | 0.9624 | 0.2370 | 0.6692 | 0.4027 |
| lgbm | 0.9634 | 0.2626 | 0.8124 | 0.3964 |
| ada | 0.9536 | 0.0369 | 0.7822 | 0.3591 |
| svm | 0.9183 | 0.0611 | 0.8043 | 0.3640 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7142 | 0.6680 | 0.7083 | 0.7823 |
| xgb | 0.7271 | 0.8726 | 0.7689 | 0.8421 |
| lgbm | 0.7212 | 0.8849 | 0.7724 | 0.8474 |
| ada | 0.6358 | 0.8448 | 0.8009 | 0.7953 |
| svm | 0.7170 | 0.6691 | 0.7132 | 0.6820 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8115 | 0.6650 | 0.8255 | 0.4996 |
| xgb | 0.8284 | 0.3728 | 0.7156 | 0.5448 |
| lgbm | 0.8249 | 0.4050 | 0.7919 | 0.5401 |
| ada | 0.7629 | 0.0708 | 0.7914 | 0.4947 |
| svm | 0.8053 | 0.1120 | 0.7560 | 0.4747 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6828 | 0.4981 | 0.7029 | 0.3329 |
| xgb | 0.7070 | 0.2291 | 0.5571 | 0.3744 |
| lgbm | 0.7020 | 0.2539 | 0.6555 | 0.3700 |
| ada | 0.6167 | 0.0367 | 0.6548 | 0.3287 |
| svm | 0.6740 | 0.0593 | 0.6077 | 0.3112 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7299 | 0.9982 | 0.9974 | 0.7263 |
| xgb | 0.7547 | 0.9921 | 0.9948 | 0.7543 |
| lgbm | 0.7507 | 0.9930 | 0.9965 | 0.7480 |
| ada | 0.6783 | 0.9402 | 0.9964 | 0.7163 |
| svm | 0.7177 | 0.9714 | 0.9961 | 0.7364 |
