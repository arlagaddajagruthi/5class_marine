# stage7_glcm: bands + 6 GLCM texture features (MADOS)

Features (17): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7476 | 0.6132 | 0.7695 | 0.7417 | 0.8480 | 0.5476 | 761,887 | 122 |
| xgb | 0.6691 | 0.5393 | 0.6217 | 0.8278 | 0.8413 | 0.5578 | 761,887 | 16 |
| lgbm | 0.6747 | 0.5437 | 0.6286 | 0.8282 | 0.8396 | 0.5538 | 761,887 | 14 |
| ada | 0.5409 | 0.4178 | 0.4952 | 0.7623 | 0.8117 | 0.4715 | 761,887 | 927 |
| svm | 0.5596 | 0.4334 | 0.5427 | 0.7007 | 0.7524 | 0.3740 | 761,887 | 28 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8799 | 0.6567 | 0.7219 | 0.7083 |
| xgb | 0.8579 | 0.8705 | 0.8184 | 0.7646 |
| lgbm | 0.8557 | 0.8808 | 0.8109 | 0.7652 |
| ada | 0.8500 | 0.7945 | 0.7692 | 0.6355 |
| svm | 0.7693 | 0.6300 | 0.7274 | 0.6762 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9333 | 0.6034 | 0.9872 | 0.5540 |
| xgb | 0.9505 | 0.2104 | 0.7810 | 0.5449 |
| lgbm | 0.9497 | 0.2270 | 0.7975 | 0.5402 |
| ada | 0.9282 | 0.0698 | 0.4525 | 0.5303 |
| svm | 0.9218 | 0.0859 | 0.7596 | 0.4036 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8799 | 0.6567 | 0.7219 | 0.7083 |
| xgb | 0.8579 | 0.8705 | 0.8184 | 0.7646 |
| lgbm | 0.8557 | 0.8808 | 0.8109 | 0.7652 |
| ada | 0.8500 | 0.7945 | 0.7692 | 0.6355 |
| svm | 0.7693 | 0.6300 | 0.7274 | 0.6762 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9058 | 0.6289 | 0.8340 | 0.6217 |
| xgb | 0.9018 | 0.3389 | 0.7992 | 0.6363 |
| lgbm | 0.9002 | 0.3609 | 0.8042 | 0.6333 |
| ada | 0.8874 | 0.1282 | 0.5698 | 0.5782 |
| svm | 0.8387 | 0.1511 | 0.7432 | 0.5055 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8278 | 0.4587 | 0.7152 | 0.4511 |
| xgb | 0.8212 | 0.2040 | 0.6656 | 0.4666 |
| lgbm | 0.8186 | 0.2202 | 0.6725 | 0.4634 |
| ada | 0.7976 | 0.0685 | 0.3984 | 0.4066 |
| svm | 0.7222 | 0.0817 | 0.5913 | 0.3383 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8510 | 0.9979 | 0.9975 | 0.8495 |
| xgb | 0.8479 | 0.9908 | 0.9965 | 0.8474 |
| lgbm | 0.8456 | 0.9916 | 0.9966 | 0.8453 |
| ada | 0.8244 | 0.9709 | 0.9901 | 0.8381 |
| svm | 0.7591 | 0.9809 | 0.9957 | 0.7690 |
