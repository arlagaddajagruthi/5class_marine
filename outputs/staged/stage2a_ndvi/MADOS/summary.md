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
