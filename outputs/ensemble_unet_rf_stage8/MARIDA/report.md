# U-Net + Random Forest (stage 8, log weights) ensemble, MARIDA

p = w * p_unet + (1 - w) * p_rf. The U-Net run and w were chosen on the validation split (highest macro F1): **MARIDA_novscp, w_unet = 0.45** (val macro F1 0.9421). Test pixels: all annotated pixels of the official test split.

U-Net probabilities are the softmax averaged over 8 TTA views, so the U-Net-only row can differ slightly from its own test report (majority vote over views).

## Overall (test)

| model | macro F1 | IoU | accuracy | kappa | macro precision | macro recall |
|---|---|---|---|---|---|---|
| Random Forest | 0.9165 | 0.8551 | 0.9963 | 0.9174 | 0.9025 | 0.9337 |
| U-Net (MARIDA_novscp, soft TTA) | 0.8620 | 0.7843 | 0.9943 | 0.8794 | 0.8233 | 0.9147 |
| Ensemble (w_unet = 0.45) | 0.9234 | 0.8664 | 0.9966 | 0.9256 | 0.9046 | 0.9444 |

## Per-class precision (test)

| class | Random Forest | U-Net (MARIDA_novscp, soft TTA) | Ensemble (w_unet = 0.45) |
|---|---|---|---|
| Water | 0.9984 | 0.9988 | 0.9989 |
| Marine Debris | 0.7643 | 0.5528 | 0.7826 |
| Algae/Sargassum | 0.9938 | 0.9927 | 0.9957 |
| Other | 0.8534 | 0.7491 | 0.8413 |

## Per-class recall (test)

| class | Random Forest | U-Net (MARIDA_novscp, soft TTA) | Ensemble (w_unet = 0.45) |
|---|---|---|---|
| Water | 0.9982 | 0.9961 | 0.9980 |
| Marine Debris | 0.9108 | 0.8110 | 0.8976 |
| Algae/Sargassum | 0.9799 | 0.9933 | 0.9933 |
| Other | 0.8460 | 0.8584 | 0.8888 |

## Per-class F1 (test)

| class | Random Forest | U-Net (MARIDA_novscp, soft TTA) | Ensemble (w_unet = 0.45) |
|---|---|---|---|
| Water | 0.9983 | 0.9975 | 0.9984 |
| Marine Debris | 0.8311 | 0.6574 | 0.8362 |
| Algae/Sargassum | 0.9868 | 0.9930 | 0.9945 |
| Other | 0.8497 | 0.8000 | 0.8644 |

## Per-class IoU (test)

| class | Random Forest | U-Net (MARIDA_novscp, soft TTA) | Ensemble (w_unet = 0.45) |
|---|---|---|---|
| Water | 0.9966 | 0.9949 | 0.9969 |
| Marine Debris | 0.7111 | 0.4897 | 0.7185 |
| Algae/Sargassum | 0.9740 | 0.9861 | 0.9891 |
| Other | 0.7386 | 0.6667 | 0.7612 |

## Validation macro F1 by weight (used for the choice)

| class | MARIDA_novscp | MARIDA_vscp |
|---|---|---|
| 0.0 | 0.8776 | 0.8776 |
| 0.05 | 0.8847 | 0.8860 |
| 0.1 | 0.8931 | 0.8947 |
| 0.15 | 0.9039 | 0.9043 |
| 0.2 | 0.9110 | 0.9120 |
| 0.25 | 0.9174 | 0.9219 |
| 0.3 | 0.9283 | 0.9318 |
| 0.35 | 0.9384 | 0.9393 |
| 0.4 | 0.9418 | 0.9408 |
| 0.45 | 0.9421 | 0.9420 |
| 0.5 | 0.9409 | 0.9419 |
| 0.55 | 0.9390 | 0.9413 |
| 0.6 | 0.9338 | 0.9393 |
| 0.65 | 0.9298 | 0.9376 |
| 0.7 | 0.9243 | 0.9348 |
| 0.75 | 0.9197 | 0.9302 |
| 0.8 | 0.9156 | 0.9264 |
| 0.85 | 0.9111 | 0.9235 |
| 0.9 | 0.9074 | 0.9206 |
| 0.95 | 0.9027 | 0.9166 |
| 1.0 | 0.8962 | 0.9143 |

Test-split curve (reference only, not used for selection): weight_curve.csv

## Random Forest, test split, full class-wise analysis

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9963 |
| balanced accuracy (macro recall) | 0.9337 |
| Cohen's kappa | 0.9174 |
| macroPrec | 0.9025 |
| macroRec | 0.9337 |
| macroF1 | 0.9165 |
| IoU | 0.8551 |
| microPrec | 0.9963 |
| microRec | 0.9963 |
| microF1 | 0.9963 |
| weightPrec | 0.9964 |
| weightRec | 0.9963 |
| weightF1 | 0.9963 |

Annotated pixels: 158,371. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 154,739 | 97.71 | 154,703 | 154,461 | 242 | 278 | 3,390 | 0.9982 | 0.9967 | 0.9984 | 0.9982 | 0.9983 | 0.9966 | 0.9334 | 0.00 |
| Marine Debris | 381 | 0.24 | 454 | 347 | 107 | 34 | 157,883 | 0.9108 | 0.9991 | 0.7643 | 0.9108 | 0.8311 | 0.7111 | 0.9993 | 0.31 |
| Algae/Sargassum | 1,641 | 1.04 | 1,618 | 1,608 | 10 | 33 | 156,720 | 0.9799 | 0.9997 | 0.9938 | 0.9799 | 0.9868 | 0.9740 | 0.9999 | 0.01 |
| Other | 1,610 | 1.02 | 1,596 | 1,362 | 234 | 248 | 156,527 | 0.8460 | 0.9970 | 0.8534 | 0.8460 | 0.8497 | 0.7386 | 0.9985 | 0.17 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 154,461 | 50 | 9 | 219 | 154,739 |
| Marine Debris | 29 | 347 | 1 | 4 | 381 |
| Algae/Sargassum | 17 | 5 | 1,608 | 11 | 1,641 |
| Other | 196 | 52 | 0 | 1,362 | 1,610 |
| total predicted | 154,703 | 454 | 1,618 | 1,596 | 158,371 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.82 | 0.03 | 0.01 | 0.14 |
| Marine Debris | 7.61 | 91.08 | 0.26 | 1.05 |
| Algae/Sargassum | 1.04 | 0.30 | 97.99 | 0.67 |
| Other | 12.17 | 3.23 | 0.00 | 84.60 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.84 | 11.01 | 0.56 | 13.72 |
| Marine Debris | 0.02 | 76.43 | 0.06 | 0.25 |
| Algae/Sargassum | 0.01 | 1.10 | 99.38 | 0.69 |
| Other | 0.13 | 11.45 | 0.00 | 85.34 |

### Where the errors go

- **Water**: 278 missed pixels went to: Other 219 (78.8%), Marine Debris 50 (18.0%), Algae/Sargassum 9 (3.2%).
  242 false-positive pixels came from: Other 196 (81.0%), Marine Debris 29 (12.0%), Algae/Sargassum 17 (7.0%).
- **Marine Debris**: 34 missed pixels went to: Water 29 (85.3%), Other 4 (11.8%), Algae/Sargassum 1 (2.9%).
  107 false-positive pixels came from: Other 52 (48.6%), Water 50 (46.7%), Algae/Sargassum 5 (4.7%).
- **Algae/Sargassum**: 33 missed pixels went to: Water 17 (51.5%), Other 11 (33.3%), Marine Debris 5 (15.2%).
  10 false-positive pixels came from: Water 9 (90.0%), Marine Debris 1 (10.0%).
- **Other**: 248 missed pixels went to: Water 196 (79.0%), Marine Debris 52 (21.0%).
  234 false-positive pixels came from: Water 219 (93.6%), Algae/Sargassum 11 (4.7%), Marine Debris 4 (1.7%).

## U-Net (MARIDA_novscp, soft TTA), test split, full class-wise analysis

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9943 |
| balanced accuracy (macro recall) | 0.9147 |
| Cohen's kappa | 0.8794 |
| macroPrec | 0.8233 |
| macroRec | 0.9147 |
| macroF1 | 0.8620 |
| IoU | 0.7843 |
| microPrec | 0.9943 |
| microRec | 0.9943 |
| microF1 | 0.9943 |
| weightPrec | 0.9951 |
| weightRec | 0.9943 |
| weightF1 | 0.9946 |

Annotated pixels: 158,371. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 154,739 | 97.71 | 154,325 | 154,140 | 185 | 599 | 3,447 | 0.9961 | 0.9950 | 0.9988 | 0.9961 | 0.9975 | 0.9949 | 0.9491 | 0.00 |
| Marine Debris | 381 | 0.24 | 559 | 309 | 250 | 72 | 157,740 | 0.8110 | 0.9980 | 0.5528 | 0.8110 | 0.6574 | 0.4897 | 0.9984 | 0.81 |
| Algae/Sargassum | 1,641 | 1.04 | 1,642 | 1,630 | 12 | 11 | 156,718 | 0.9933 | 0.9999 | 0.9927 | 0.9933 | 0.9930 | 0.9861 | 0.9999 | 0.01 |
| Other | 1,610 | 1.02 | 1,845 | 1,382 | 463 | 228 | 156,298 | 0.8584 | 0.9956 | 0.7491 | 0.8584 | 0.8000 | 0.6667 | 0.9970 | 0.34 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 154,140 | 155 | 4 | 440 | 154,739 |
| Marine Debris | 46 | 309 | 3 | 23 | 381 |
| Algae/Sargassum | 9 | 2 | 1,630 | 0 | 1,641 |
| Other | 130 | 93 | 5 | 1,382 | 1,610 |
| total predicted | 154,325 | 559 | 1,642 | 1,845 | 158,371 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.61 | 0.10 | 0.00 | 0.28 |
| Marine Debris | 12.07 | 81.10 | 0.79 | 6.04 |
| Algae/Sargassum | 0.55 | 0.12 | 99.33 | 0.00 |
| Other | 8.07 | 5.78 | 0.31 | 85.84 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.88 | 27.73 | 0.24 | 23.85 |
| Marine Debris | 0.03 | 55.28 | 0.18 | 1.25 |
| Algae/Sargassum | 0.01 | 0.36 | 99.27 | 0.00 |
| Other | 0.08 | 16.64 | 0.30 | 74.91 |

### Where the errors go

- **Water**: 599 missed pixels went to: Other 440 (73.5%), Marine Debris 155 (25.9%), Algae/Sargassum 4 (0.7%).
  185 false-positive pixels came from: Other 130 (70.3%), Marine Debris 46 (24.9%), Algae/Sargassum 9 (4.9%).
- **Marine Debris**: 72 missed pixels went to: Water 46 (63.9%), Other 23 (31.9%), Algae/Sargassum 3 (4.2%).
  250 false-positive pixels came from: Water 155 (62.0%), Other 93 (37.2%), Algae/Sargassum 2 (0.8%).
- **Algae/Sargassum**: 11 missed pixels went to: Water 9 (81.8%), Marine Debris 2 (18.2%).
  12 false-positive pixels came from: Other 5 (41.7%), Water 4 (33.3%), Marine Debris 3 (25.0%).
- **Other**: 228 missed pixels went to: Water 130 (57.0%), Marine Debris 93 (40.8%), Algae/Sargassum 5 (2.2%).
  463 false-positive pixels came from: Water 440 (95.0%), Marine Debris 23 (5.0%).

## Ensemble (w_unet = 0.45), test split, full class-wise analysis

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9966 |
| balanced accuracy (macro recall) | 0.9444 |
| Cohen's kappa | 0.9256 |
| macroPrec | 0.9046 |
| macroRec | 0.9444 |
| macroF1 | 0.9234 |
| IoU | 0.8664 |
| microPrec | 0.9966 |
| microRec | 0.9966 |
| microF1 | 0.9966 |
| weightPrec | 0.9967 |
| weightRec | 0.9966 |
| weightF1 | 0.9966 |

Annotated pixels: 158,371. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 154,739 | 97.71 | 154,596 | 154,426 | 170 | 313 | 3,462 | 0.9980 | 0.9970 | 0.9989 | 0.9980 | 0.9984 | 0.9969 | 0.9532 | 0.00 |
| Marine Debris | 381 | 0.24 | 437 | 342 | 95 | 39 | 157,895 | 0.8976 | 0.9992 | 0.7826 | 0.8976 | 0.8362 | 0.7185 | 0.9994 | 0.28 |
| Algae/Sargassum | 1,641 | 1.04 | 1,637 | 1,630 | 7 | 11 | 156,723 | 0.9933 | 0.9999 | 0.9957 | 0.9933 | 0.9945 | 0.9891 | 1.0000 | 0.00 |
| Other | 1,610 | 1.02 | 1,701 | 1,431 | 270 | 179 | 156,491 | 0.8888 | 0.9972 | 0.8413 | 0.8888 | 0.8644 | 0.7612 | 0.9983 | 0.19 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 154,426 | 46 | 2 | 265 | 154,739 |
| Marine Debris | 36 | 342 | 1 | 2 | 381 |
| Algae/Sargassum | 8 | 0 | 1,630 | 3 | 1,641 |
| Other | 126 | 49 | 4 | 1,431 | 1,610 |
| total predicted | 154,596 | 437 | 1,637 | 1,701 | 158,371 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.80 | 0.03 | 0.00 | 0.17 |
| Marine Debris | 9.45 | 89.76 | 0.26 | 0.52 |
| Algae/Sargassum | 0.49 | 0.00 | 99.33 | 0.18 |
| Other | 7.83 | 3.04 | 0.25 | 88.88 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.89 | 10.53 | 0.12 | 15.58 |
| Marine Debris | 0.02 | 78.26 | 0.06 | 0.12 |
| Algae/Sargassum | 0.01 | 0.00 | 99.57 | 0.18 |
| Other | 0.08 | 11.21 | 0.24 | 84.13 |

### Where the errors go

- **Water**: 313 missed pixels went to: Other 265 (84.7%), Marine Debris 46 (14.7%), Algae/Sargassum 2 (0.6%).
  170 false-positive pixels came from: Other 126 (74.1%), Marine Debris 36 (21.2%), Algae/Sargassum 8 (4.7%).
- **Marine Debris**: 39 missed pixels went to: Water 36 (92.3%), Other 2 (5.1%), Algae/Sargassum 1 (2.6%).
  95 false-positive pixels came from: Other 49 (51.6%), Water 46 (48.4%).
- **Algae/Sargassum**: 11 missed pixels went to: Water 8 (72.7%), Other 3 (27.3%).
  7 false-positive pixels came from: Other 4 (57.1%), Water 2 (28.6%), Marine Debris 1 (14.3%).
- **Other**: 179 missed pixels went to: Water 126 (70.4%), Marine Debris 49 (27.4%), Algae/Sargassum 4 (2.2%).
  270 false-positive pixels came from: Water 265 (98.1%), Algae/Sargassum 3 (1.1%), Marine Debris 2 (0.7%).
