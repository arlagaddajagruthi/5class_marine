# stage2a_ndvi: bands + NDVI (MADOS)

Features (12): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7064 | 0.5609 | 0.7474 | 0.7224 | 0.7280 | 0.3687 | 761,887 | 36 |
| xgb | 0.6262 | 0.4796 | 0.5852 | 0.8052 | 0.7445 | 0.4167 | 761,887 | 6 |
| lgbm | 0.6367 | 0.4906 | 0.6106 | 0.8002 | 0.7351 | 0.4051 | 761,887 | 3 |
| ada | 0.5446 | 0.4244 | 0.5395 | 0.7804 | 0.6915 | 0.3510 | 761,887 | 341 |
| svm | 0.5606 | 0.4416 | 0.5459 | 0.7355 | 0.7233 | 0.3395 | 761,887 | 6 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7155 | 0.6773 | 0.7090 | 0.7878 |
| xgb | 0.7222 | 0.8746 | 0.7795 | 0.8444 |
| lgbm | 0.7093 | 0.8828 | 0.7569 | 0.8518 |
| ada | 0.6629 | 0.8273 | 0.8148 | 0.8166 |
| svm | 0.7299 | 0.7061 | 0.8181 | 0.6881 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9409 | 0.6886 | 0.9905 | 0.3697 |
| xgb | 0.9631 | 0.2435 | 0.7366 | 0.3977 |
| lgbm | 0.9645 | 0.2648 | 0.8261 | 0.3869 |
| ada | 0.9565 | 0.0411 | 0.7778 | 0.3828 |
| svm | 0.9207 | 0.0670 | 0.8175 | 0.3783 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7155 | 0.6773 | 0.7090 | 0.7878 |
| xgb | 0.7222 | 0.8746 | 0.7795 | 0.8444 |
| lgbm | 0.7093 | 0.8828 | 0.7569 | 0.8518 |
| ada | 0.6629 | 0.8273 | 0.8148 | 0.8166 |
| svm | 0.7299 | 0.7061 | 0.8181 | 0.6881 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8129 | 0.6829 | 0.8264 | 0.5033 |
| xgb | 0.8255 | 0.3809 | 0.7575 | 0.5407 |
| lgbm | 0.8174 | 0.4074 | 0.7900 | 0.5321 |
| ada | 0.7831 | 0.0783 | 0.7959 | 0.5212 |
| svm | 0.8143 | 0.1223 | 0.8178 | 0.4882 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.6847 | 0.5185 | 0.7042 | 0.3362 |
| xgb | 0.7028 | 0.2353 | 0.6096 | 0.3706 |
| lgbm | 0.6913 | 0.2558 | 0.6529 | 0.3625 |
| ada | 0.6435 | 0.0408 | 0.6610 | 0.3525 |
| svm | 0.6867 | 0.0651 | 0.6918 | 0.3229 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.7318 | 0.9983 | 0.9975 | 0.7284 |
| xgb | 0.7513 | 0.9923 | 0.9957 | 0.7495 |
| lgbm | 0.7421 | 0.9931 | 0.9966 | 0.7384 |
| ada | 0.7010 | 0.9475 | 0.9964 | 0.7380 |
| svm | 0.7289 | 0.9727 | 0.9969 | 0.7480 |
