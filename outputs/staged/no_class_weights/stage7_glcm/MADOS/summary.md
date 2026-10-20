# stage7_glcm: bands + 6 GLCM texture features (MADOS)

Features (17): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7471 | 0.6116 | 0.7678 | 0.7475 | 0.8223 | 0.5020 | 761,887 | 147 |
| xgb | 0.7170 | 0.5841 | 0.7269 | 0.7311 | 0.8503 | 0.5599 | 761,887 | 15 |
| lgbm | 0.7137 | 0.5817 | 0.7132 | 0.7391 | 0.8514 | 0.5634 | 761,887 | 13 |
| ada | 0.6345 | 0.5125 | 0.7890 | 0.5665 | 0.8817 | 0.5837 | 761,887 | 924 |
| svm | 0.5442 | 0.4411 | 0.7039 | 0.4907 | 0.8494 | 0.4344 | 761,887 | 22 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8441 | 0.6804 | 0.7381 | 0.7273 |
| xgb | 0.8775 | 0.5920 | 0.7209 | 0.7342 |
| lgbm | 0.8781 | 0.6033 | 0.7388 | 0.7364 |
| ada | 0.9490 | 0.1501 | 0.5720 | 0.5948 |
| svm | 0.9460 | 0.0164 | 0.5753 | 0.4251 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9347 | 0.6522 | 0.9874 | 0.4969 |
| xgb | 0.9381 | 0.4195 | 0.9889 | 0.5611 |
| lgbm | 0.9390 | 0.3898 | 0.9596 | 0.5643 |
| ada | 0.9111 | 0.5407 | 0.9944 | 0.7097 |
| svm | 0.8796 | 0.3200 | 0.9938 | 0.6222 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8441 | 0.6804 | 0.7381 | 0.7273 |
| xgb | 0.8775 | 0.5920 | 0.7209 | 0.7342 |
| lgbm | 0.8781 | 0.6033 | 0.7388 | 0.7364 |
| ada | 0.9490 | 0.1501 | 0.5720 | 0.5948 |
| svm | 0.9460 | 0.0164 | 0.5753 | 0.4251 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8871 | 0.6660 | 0.8448 | 0.5904 |
| xgb | 0.9068 | 0.4910 | 0.8339 | 0.6361 |
| lgbm | 0.9075 | 0.4736 | 0.8348 | 0.6390 |
| ada | 0.9296 | 0.2349 | 0.7263 | 0.6472 |
| svm | 0.9116 | 0.0313 | 0.7287 | 0.5051 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7971 | 0.4992 | 0.7312 | 0.4189 |
| xgb | 0.8294 | 0.3254 | 0.7152 | 0.4664 |
| lgbm | 0.8307 | 0.3103 | 0.7165 | 0.4695 |
| ada | 0.8685 | 0.1331 | 0.5702 | 0.4784 |
| svm | 0.8376 | 0.0159 | 0.5732 | 0.3379 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8250 | 0.9982 | 0.9977 | 0.8238 |
| xgb | 0.8531 | 0.9967 | 0.9975 | 0.8533 |
| lgbm | 0.8543 | 0.9964 | 0.9975 | 0.8547 |
| ada | 0.8831 | 0.9974 | 0.9963 | 0.8867 |
| svm | 0.8506 | 0.9973 | 0.9963 | 0.8545 |
