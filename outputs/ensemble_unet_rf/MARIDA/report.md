# U-Net + Random Forest ensemble, MARIDA

p = w * p_unet + (1 - w) * p_rf. The U-Net run and w were chosen on the validation split (highest macro F1): **MARIDA_novscp, w_unet = 0.45** (val macro F1 0.9419). Test pixels: all annotated pixels of the official test split.

U-Net probabilities are the softmax averaged over 8 TTA views, so the U-Net-only row can differ slightly from its own test report (majority vote over views).

## Overall (test)

| model | macro F1 | IoU | accuracy | kappa | macro precision | macro recall |
|---|---|---|---|---|---|---|
| Random Forest | 0.9042 | 0.8373 | 0.9956 | 0.9027 | 0.8865 | 0.9270 |
| U-Net (MARIDA_novscp, soft TTA) | 0.8620 | 0.7843 | 0.9943 | 0.8794 | 0.8233 | 0.9147 |
| Ensemble (w_unet = 0.45) | 0.9201 | 0.8617 | 0.9965 | 0.9243 | 0.9012 | 0.9420 |

## Per-class precision (test)

| class | Random Forest | U-Net (MARIDA_novscp, soft TTA) | Ensemble (w_unet = 0.45) |
|---|---|---|---|
| Water | 0.9982 | 0.9988 | 0.9988 |
| Marine Debris | 0.7326 | 0.5528 | 0.7600 |
| Algae/Sargassum | 0.9969 | 0.9927 | 0.9945 |
| Other | 0.8183 | 0.7491 | 0.8516 |

## Per-class recall (test)

| class | Random Forest | U-Net (MARIDA_novscp, soft TTA) | Ensemble (w_unet = 0.45) |
|---|---|---|---|
| Water | 0.9978 | 0.9961 | 0.9980 |
| Marine Debris | 0.9134 | 0.8110 | 0.8976 |
| Algae/Sargassum | 0.9799 | 0.9933 | 0.9951 |
| Other | 0.8168 | 0.8584 | 0.8770 |

## Per-class F1 (test)

| class | Random Forest | U-Net (MARIDA_novscp, soft TTA) | Ensemble (w_unet = 0.45) |
|---|---|---|---|
| Water | 0.9980 | 0.9975 | 0.9984 |
| Marine Debris | 0.8131 | 0.6574 | 0.8231 |
| Algae/Sargassum | 0.9883 | 0.9930 | 0.9948 |
| Other | 0.8175 | 0.8000 | 0.8641 |

## Per-class IoU (test)

| class | Random Forest | U-Net (MARIDA_novscp, soft TTA) | Ensemble (w_unet = 0.45) |
|---|---|---|---|
| Water | 0.9960 | 0.9949 | 0.9968 |
| Marine Debris | 0.6850 | 0.4897 | 0.6994 |
| Algae/Sargassum | 0.9769 | 0.9861 | 0.9897 |
| Other | 0.6914 | 0.6667 | 0.7608 |

## Validation macro F1 by weight (used for the choice)

| class | MARIDA_novscp | MARIDA_vscp |
|---|---|---|
| 0.0 | 0.9181 | 0.9181 |
| 0.05 | 0.9207 | 0.9203 |
| 0.1 | 0.9234 | 0.9230 |
| 0.15 | 0.9258 | 0.9259 |
| 0.2 | 0.9273 | 0.9284 |
| 0.25 | 0.9304 | 0.9303 |
| 0.3 | 0.9330 | 0.9319 |
| 0.35 | 0.9365 | 0.9347 |
| 0.4 | 0.9394 | 0.9387 |
| 0.45 | 0.9419 | 0.9405 |
| 0.5 | 0.9410 | 0.9404 |
| 0.55 | 0.9403 | 0.9400 |
| 0.6 | 0.9370 | 0.9397 |
| 0.65 | 0.9305 | 0.9369 |
| 0.7 | 0.9256 | 0.9338 |
| 0.75 | 0.9215 | 0.9305 |
| 0.8 | 0.9162 | 0.9276 |
| 0.85 | 0.9111 | 0.9241 |
| 0.9 | 0.9071 | 0.9216 |
| 0.95 | 0.9027 | 0.9179 |
| 1.0 | 0.8962 | 0.9143 |

Test-split curve (reference only, not used for selection): weight_curve.csv

## Random Forest, test split, full class-wise analysis

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9956 |
| balanced accuracy (macro recall) | 0.9270 |
| Cohen's kappa | 0.9027 |
| macroPrec | 0.8865 |
| macroRec | 0.9270 |
| macroF1 | 0.9042 |
| IoU | 0.8373 |
| microPrec | 0.9956 |
| microRec | 0.9956 |
| microF1 | 0.9956 |
| weightPrec | 0.9957 |
| weightRec | 0.9956 |
| weightF1 | 0.9956 |

Annotated pixels: 158,371. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 154,739 | 97.71 | 154,676 | 154,399 | 277 | 340 | 3,355 | 0.9978 | 0.9961 | 0.9982 | 0.9978 | 0.9980 | 0.9960 | 0.9237 | 0.00 |
| Marine Debris | 381 | 0.24 | 475 | 348 | 127 | 33 | 157,863 | 0.9134 | 0.9990 | 0.7326 | 0.9134 | 0.8131 | 0.6850 | 0.9992 | 0.36 |
| Algae/Sargassum | 1,641 | 1.04 | 1,613 | 1,608 | 5 | 33 | 156,725 | 0.9799 | 0.9998 | 0.9969 | 0.9799 | 0.9883 | 0.9769 | 1.0000 | 0.00 |
| Other | 1,610 | 1.02 | 1,607 | 1,315 | 292 | 295 | 156,469 | 0.8168 | 0.9963 | 0.8183 | 0.8168 | 0.8175 | 0.6914 | 0.9981 | 0.22 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 154,399 | 64 | 4 | 272 | 154,739 |
| Marine Debris | 29 | 348 | 0 | 4 | 381 |
| Algae/Sargassum | 14 | 3 | 1,608 | 16 | 1,641 |
| Other | 234 | 60 | 1 | 1,315 | 1,610 |
| total predicted | 154,676 | 475 | 1,613 | 1,607 | 158,371 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.78 | 0.04 | 0.00 | 0.18 |
| Marine Debris | 7.61 | 91.34 | 0.00 | 1.05 |
| Algae/Sargassum | 0.85 | 0.18 | 97.99 | 0.98 |
| Other | 14.53 | 3.73 | 0.06 | 81.68 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.82 | 13.47 | 0.25 | 16.93 |
| Marine Debris | 0.02 | 73.26 | 0.00 | 0.25 |
| Algae/Sargassum | 0.01 | 0.63 | 99.69 | 1.00 |
| Other | 0.15 | 12.63 | 0.06 | 81.83 |

### Where the errors go

- **Water**: 340 missed pixels went to: Other 272 (80.0%), Marine Debris 64 (18.8%), Algae/Sargassum 4 (1.2%).
  277 false-positive pixels came from: Other 234 (84.5%), Marine Debris 29 (10.5%), Algae/Sargassum 14 (5.1%).
- **Marine Debris**: 33 missed pixels went to: Water 29 (87.9%), Other 4 (12.1%).
  127 false-positive pixels came from: Water 64 (50.4%), Other 60 (47.2%), Algae/Sargassum 3 (2.4%).
- **Algae/Sargassum**: 33 missed pixels went to: Other 16 (48.5%), Water 14 (42.4%), Marine Debris 3 (9.1%).
  5 false-positive pixels came from: Water 4 (80.0%), Other 1 (20.0%).
- **Other**: 295 missed pixels went to: Water 234 (79.3%), Marine Debris 60 (20.3%), Algae/Sargassum 1 (0.3%).
  292 false-positive pixels came from: Water 272 (93.2%), Algae/Sargassum 16 (5.5%), Marine Debris 4 (1.4%).

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
| overall accuracy | 0.9965 |
| balanced accuracy (macro recall) | 0.9420 |
| Cohen's kappa | 0.9243 |
| macroPrec | 0.9012 |
| macroRec | 0.9420 |
| macroF1 | 0.9201 |
| IoU | 0.8617 |
| microPrec | 0.9965 |
| microRec | 0.9965 |
| microF1 | 0.9965 |
| weightPrec | 0.9967 |
| weightRec | 0.9965 |
| weightF1 | 0.9966 |

Annotated pixels: 158,371. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 154,739 | 97.71 | 154,621 | 154,434 | 187 | 305 | 3,445 | 0.9980 | 0.9969 | 0.9988 | 0.9980 | 0.9984 | 0.9968 | 0.9485 | 0.00 |
| Marine Debris | 381 | 0.24 | 450 | 342 | 108 | 39 | 157,882 | 0.8976 | 0.9991 | 0.7600 | 0.8976 | 0.8231 | 0.6994 | 0.9993 | 0.32 |
| Algae/Sargassum | 1,641 | 1.04 | 1,642 | 1,633 | 9 | 8 | 156,721 | 0.9951 | 0.9999 | 0.9945 | 0.9951 | 0.9948 | 0.9897 | 0.9999 | 0.01 |
| Other | 1,610 | 1.02 | 1,658 | 1,412 | 246 | 198 | 156,515 | 0.8770 | 0.9972 | 0.8516 | 0.8770 | 0.8641 | 0.7608 | 0.9984 | 0.17 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 154,434 | 60 | 2 | 243 | 154,739 |
| Marine Debris | 36 | 342 | 1 | 2 | 381 |
| Algae/Sargassum | 7 | 0 | 1,633 | 1 | 1,641 |
| Other | 144 | 48 | 6 | 1,412 | 1,610 |
| total predicted | 154,621 | 450 | 1,642 | 1,658 | 158,371 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.80 | 0.04 | 0.00 | 0.16 |
| Marine Debris | 9.45 | 89.76 | 0.26 | 0.52 |
| Algae/Sargassum | 0.43 | 0.00 | 99.51 | 0.06 |
| Other | 8.94 | 2.98 | 0.37 | 87.70 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.88 | 13.33 | 0.12 | 14.66 |
| Marine Debris | 0.02 | 76.00 | 0.06 | 0.12 |
| Algae/Sargassum | 0.00 | 0.00 | 99.45 | 0.06 |
| Other | 0.09 | 10.67 | 0.37 | 85.16 |

### Where the errors go

- **Water**: 305 missed pixels went to: Other 243 (79.7%), Marine Debris 60 (19.7%), Algae/Sargassum 2 (0.7%).
  187 false-positive pixels came from: Other 144 (77.0%), Marine Debris 36 (19.3%), Algae/Sargassum 7 (3.7%).
- **Marine Debris**: 39 missed pixels went to: Water 36 (92.3%), Other 2 (5.1%), Algae/Sargassum 1 (2.6%).
  108 false-positive pixels came from: Water 60 (55.6%), Other 48 (44.4%).
- **Algae/Sargassum**: 8 missed pixels went to: Water 7 (87.5%), Other 1 (12.5%).
  9 false-positive pixels came from: Other 6 (66.7%), Water 2 (22.2%), Marine Debris 1 (11.1%).
- **Other**: 198 missed pixels went to: Water 144 (72.7%), Marine Debris 48 (24.2%), Algae/Sargassum 6 (3.0%).
  246 false-positive pixels came from: Water 243 (98.8%), Marine Debris 2 (0.8%), Algae/Sargassum 1 (0.4%).
