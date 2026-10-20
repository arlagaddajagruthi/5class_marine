# stage7_glcm: bands + 6 GLCM texture features (MARIDA)

Features (17): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.8777 | 0.8034 | 0.8567 | 0.9067 | 0.9954 | 0.9003 | 358,442 | 41 |
| xgb | 0.8578 | 0.7782 | 0.8318 | 0.9029 | 0.9949 | 0.8894 | 358,442 | 10 |
| lgbm | 0.6821 | 0.5944 | 0.6486 | 0.8577 | 0.9740 | 0.5981 | 358,442 | 6 |
| ada | 0.6617 | 0.5757 | 0.7964 | 0.6113 | 0.9896 | 0.7325 | 358,442 | 304 |
| svm | 0.6505 | 0.5679 | 0.6375 | 0.6884 | 0.9843 | 0.6870 | 358,442 | 6 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9976 | 0.8031 | 0.9647 | 0.8615 |
| xgb | 0.9975 | 0.8320 | 0.9598 | 0.8224 |
| lgbm | 0.9774 | 0.7900 | 0.8598 | 0.8037 |
| ada | 0.9983 | 0.0735 | 0.5344 | 0.8391 |
| svm | 0.9896 | 0.0892 | 0.9299 | 0.7447 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9984 | 0.5930 | 0.9931 | 0.8421 |
| xgb | 0.9986 | 0.5138 | 0.9943 | 0.8203 |
| lgbm | 0.9980 | 0.0960 | 0.8852 | 0.6153 |
| ada | 0.9916 | 0.3684 | 0.9989 | 0.8268 |
| svm | 0.9948 | 0.0702 | 0.5793 | 0.9056 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9976 | 0.8031 | 0.9647 | 0.8615 |
| xgb | 0.9975 | 0.8320 | 0.9598 | 0.8224 |
| lgbm | 0.9774 | 0.7900 | 0.8598 | 0.8037 |
| ada | 0.9983 | 0.0735 | 0.5344 | 0.8391 |
| svm | 0.9896 | 0.0892 | 0.9299 | 0.7447 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9980 | 0.6823 | 0.9787 | 0.8517 |
| xgb | 0.9980 | 0.6353 | 0.9767 | 0.8213 |
| lgbm | 0.9876 | 0.1713 | 0.8723 | 0.6970 |
| ada | 0.9949 | 0.1225 | 0.6963 | 0.8329 |
| svm | 0.9922 | 0.0786 | 0.7139 | 0.8173 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9960 | 0.5178 | 0.9582 | 0.7417 |
| xgb | 0.9961 | 0.4655 | 0.9545 | 0.6968 |
| lgbm | 0.9755 | 0.0937 | 0.7736 | 0.5349 |
| ada | 0.9899 | 0.0653 | 0.5341 | 0.7137 |
| svm | 0.9845 | 0.0409 | 0.5551 | 0.6911 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9961 | 0.9982 | 0.9996 | 0.9970 |
| xgb | 0.9962 | 0.9977 | 0.9995 | 0.9964 |
| lgbm | 0.9760 | 0.9816 | 0.9974 | 0.9929 |
| ada | 0.9901 | 0.9975 | 0.9952 | 0.9966 |
| svm | 0.9848 | 0.9950 | 0.9923 | 0.9966 |
