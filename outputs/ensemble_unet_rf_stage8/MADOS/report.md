# U-Net + Random Forest (stage 8, log weights) ensemble, MADOS

p = w * p_unet + (1 - w) * p_rf. The U-Net run and w were chosen on the validation split (highest macro F1): **MADOS_vscp, w_unet = 0.45** (val macro F1 0.8751). Test pixels: all annotated pixels of the official test split.

U-Net probabilities are the softmax averaged over 8 TTA views, so the U-Net-only row can differ slightly from its own test report (majority vote over views).

## Overall (test)

| model | macro F1 | IoU | accuracy | kappa | macro precision | macro recall |
|---|---|---|---|---|---|---|
| Random Forest | 0.7609 | 0.6271 | 0.8375 | 0.5296 | 0.7754 | 0.7663 |
| U-Net (MADOS_vscp, soft TTA) | 0.8240 | 0.7268 | 0.9250 | 0.7273 | 0.8334 | 0.8393 |
| Ensemble (w_unet = 0.45) | 0.8202 | 0.7119 | 0.8720 | 0.6078 | 0.8122 | 0.8330 |

## Per-class precision (test)

| class | Random Forest | U-Net (MADOS_vscp, soft TTA) | Ensemble (w_unet = 0.45) |
|---|---|---|---|
| Water | 0.9348 | 0.9290 | 0.9391 |
| Marine Debris | 0.6593 | 0.5147 | 0.7045 |
| Algae/Sargassum | 0.9797 | 0.9834 | 0.9901 |
| Other | 0.5278 | 0.9063 | 0.6150 |

## Per-class recall (test)

| class | Random Forest | U-Net (MADOS_vscp, soft TTA) | Ensemble (w_unet = 0.45) |
|---|---|---|---|
| Water | 0.8646 | 0.9859 | 0.9032 |
| Marine Debris | 0.7636 | 0.7729 | 0.7965 |
| Algae/Sargassum | 0.7187 | 0.9566 | 0.9064 |
| Other | 0.7181 | 0.6420 | 0.7256 |

## Per-class F1 (test)

| class | Random Forest | U-Net (MADOS_vscp, soft TTA) | Ensemble (w_unet = 0.45) |
|---|---|---|---|
| Water | 0.8984 | 0.9566 | 0.9208 |
| Marine Debris | 0.7076 | 0.6179 | 0.7477 |
| Algae/Sargassum | 0.8291 | 0.9698 | 0.9464 |
| Other | 0.6084 | 0.7516 | 0.6657 |

## Per-class IoU (test)

| class | Random Forest | U-Net (MADOS_vscp, soft TTA) | Ensemble (w_unet = 0.45) |
|---|---|---|---|
| Water | 0.8155 | 0.9169 | 0.8532 |
| Marine Debris | 0.5475 | 0.4471 | 0.5971 |
| Algae/Sargassum | 0.7081 | 0.9414 | 0.8983 |
| Other | 0.4372 | 0.6021 | 0.4989 |

## Validation macro F1 by weight (used for the choice)

| class | MADOS_vscp |
|---|---|
| 0.0 | 0.8246 |
| 0.05 | 0.8329 |
| 0.1 | 0.8427 |
| 0.15 | 0.8512 |
| 0.2 | 0.8592 |
| 0.25 | 0.8665 |
| 0.3 | 0.8701 |
| 0.35 | 0.8710 |
| 0.4 | 0.8734 |
| 0.45 | 0.8751 |
| 0.5 | 0.8741 |
| 0.55 | 0.8726 |
| 0.6 | 0.8699 |
| 0.65 | 0.8673 |
| 0.7 | 0.8646 |
| 0.75 | 0.8622 |
| 0.8 | 0.8596 |
| 0.85 | 0.8577 |
| 0.9 | 0.8564 |
| 0.95 | 0.8543 |
| 1.0 | 0.8521 |

Test-split curve (reference only, not used for selection): weight_curve.csv

## Random Forest, test split, full class-wise analysis

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.8375 |
| balanced accuracy (macro recall) | 0.7663 |
| Cohen's kappa | 0.5296 |
| macroPrec | 0.7754 |
| macroRec | 0.7663 |
| macroF1 | 0.7609 |
| IoU | 0.6271 |
| microPrec | 0.8375 |
| microRec | 0.8375 |
| microF1 | 0.8375 |
| weightPrec | 0.8634 |
| weightRec | 0.8375 |
| weightF1 | 0.8466 |

Annotated pixels: 361,147. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 294,020 | 81.41 | 271,945 | 254,224 | 17,721 | 39,796 | 49,406 | 0.8646 | 0.8407 | 0.9348 | 0.8646 | 0.8984 | 0.8155 | 0.7360 | 0.07 |
| Marine Debris | 973 | 0.27 | 1,127 | 743 | 384 | 230 | 359,790 | 0.7636 | 0.9983 | 0.6593 | 0.7636 | 0.7076 | 0.5475 | 0.9989 | 0.52 |
| Algae/Sargassum | 3,089 | 0.86 | 2,266 | 2,220 | 46 | 869 | 358,012 | 0.7187 | 0.9975 | 0.9797 | 0.7187 | 0.8291 | 0.7081 | 0.9999 | 0.02 |
| Other | 63,065 | 17.46 | 85,809 | 45,287 | 40,522 | 17,778 | 257,560 | 0.7181 | 0.8386 | 0.5278 | 0.7181 | 0.6084 | 0.4372 | 0.8641 | 0.89 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 254,224 | 21 | 27 | 39,748 | 294,020 |
| Marine Debris | 39 | 743 | 8 | 183 | 973 |
| Algae/Sargassum | 249 | 29 | 2,220 | 591 | 3,089 |
| Other | 17,433 | 334 | 11 | 45,287 | 63,065 |
| total predicted | 271,945 | 1,127 | 2,266 | 85,809 | 361,147 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 86.46 | 0.01 | 0.01 | 13.52 |
| Marine Debris | 4.01 | 76.36 | 0.82 | 18.81 |
| Algae/Sargassum | 8.06 | 0.94 | 71.87 | 19.13 |
| Other | 27.64 | 0.53 | 0.02 | 71.81 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 93.48 | 1.86 | 1.19 | 46.32 |
| Marine Debris | 0.01 | 65.93 | 0.35 | 0.21 |
| Algae/Sargassum | 0.09 | 2.57 | 97.97 | 0.69 |
| Other | 6.41 | 29.64 | 0.49 | 52.78 |

### Where the errors go

- **Water**: 39,796 missed pixels went to: Other 39,748 (99.9%), Algae/Sargassum 27 (0.1%), Marine Debris 21 (0.1%).
  17,721 false-positive pixels came from: Other 17,433 (98.4%), Algae/Sargassum 249 (1.4%), Marine Debris 39 (0.2%).
- **Marine Debris**: 230 missed pixels went to: Other 183 (79.6%), Water 39 (17.0%), Algae/Sargassum 8 (3.5%).
  384 false-positive pixels came from: Other 334 (87.0%), Algae/Sargassum 29 (7.6%), Water 21 (5.5%).
- **Algae/Sargassum**: 869 missed pixels went to: Other 591 (68.0%), Water 249 (28.7%), Marine Debris 29 (3.3%).
  46 false-positive pixels came from: Water 27 (58.7%), Other 11 (23.9%), Marine Debris 8 (17.4%).
- **Other**: 17,778 missed pixels went to: Water 17,433 (98.1%), Marine Debris 334 (1.9%), Algae/Sargassum 11 (0.1%).
  40,522 false-positive pixels came from: Water 39,748 (98.1%), Algae/Sargassum 591 (1.5%), Marine Debris 183 (0.5%).

## U-Net (MADOS_vscp, soft TTA), test split, full class-wise analysis

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9250 |
| balanced accuracy (macro recall) | 0.8393 |
| Cohen's kappa | 0.7273 |
| macroPrec | 0.8334 |
| macroRec | 0.8393 |
| macroF1 | 0.8240 |
| IoU | 0.7268 |
| microPrec | 0.9250 |
| microRec | 0.9250 |
| microF1 | 0.9250 |
| weightPrec | 0.9244 |
| weightRec | 0.9250 |
| weightF1 | 0.9200 |

Annotated pixels: 361,147. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 294,020 | 81.41 | 312,009 | 289,871 | 22,138 | 4,149 | 44,989 | 0.9859 | 0.9272 | 0.9290 | 0.9859 | 0.9566 | 0.9169 | 0.6702 | 0.08 |
| Marine Debris | 973 | 0.27 | 1,461 | 752 | 709 | 221 | 359,465 | 0.7729 | 0.9974 | 0.5147 | 0.7729 | 0.6179 | 0.4471 | 0.9980 | 0.94 |
| Algae/Sargassum | 3,089 | 0.86 | 3,005 | 2,955 | 50 | 134 | 358,008 | 0.9566 | 0.9995 | 0.9834 | 0.9566 | 0.9698 | 0.9414 | 0.9999 | 0.02 |
| Other | 63,065 | 17.46 | 44,672 | 40,488 | 4,184 | 22,577 | 293,898 | 0.6420 | 0.9259 | 0.9063 | 0.6420 | 0.7516 | 0.6021 | 0.9860 | 0.10 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 289,871 | 125 | 19 | 4,005 | 294,020 |
| Marine Debris | 22 | 752 | 20 | 179 | 973 |
| Algae/Sargassum | 107 | 27 | 2,955 | 0 | 3,089 |
| Other | 22,009 | 557 | 11 | 40,488 | 63,065 |
| total predicted | 312,009 | 1,461 | 3,005 | 44,672 | 361,147 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 98.59 | 0.04 | 0.01 | 1.36 |
| Marine Debris | 2.26 | 77.29 | 2.06 | 18.40 |
| Algae/Sargassum | 3.46 | 0.87 | 95.66 | 0.00 |
| Other | 34.90 | 0.88 | 0.02 | 64.20 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 92.90 | 8.56 | 0.63 | 8.97 |
| Marine Debris | 0.01 | 51.47 | 0.67 | 0.40 |
| Algae/Sargassum | 0.03 | 1.85 | 98.34 | 0.00 |
| Other | 7.05 | 38.12 | 0.37 | 90.63 |

### Where the errors go

- **Water**: 4,149 missed pixels went to: Other 4,005 (96.5%), Marine Debris 125 (3.0%), Algae/Sargassum 19 (0.5%).
  22,138 false-positive pixels came from: Other 22,009 (99.4%), Algae/Sargassum 107 (0.5%), Marine Debris 22 (0.1%).
- **Marine Debris**: 221 missed pixels went to: Other 179 (81.0%), Water 22 (10.0%), Algae/Sargassum 20 (9.0%).
  709 false-positive pixels came from: Other 557 (78.6%), Water 125 (17.6%), Algae/Sargassum 27 (3.8%).
- **Algae/Sargassum**: 134 missed pixels went to: Water 107 (79.9%), Marine Debris 27 (20.1%).
  50 false-positive pixels came from: Marine Debris 20 (40.0%), Water 19 (38.0%), Other 11 (22.0%).
- **Other**: 22,577 missed pixels went to: Water 22,009 (97.5%), Marine Debris 557 (2.5%), Algae/Sargassum 11 (0.0%).
  4,184 false-positive pixels came from: Water 4,005 (95.7%), Marine Debris 179 (4.3%).

## Ensemble (w_unet = 0.45), test split, full class-wise analysis

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.8720 |
| balanced accuracy (macro recall) | 0.8330 |
| Cohen's kappa | 0.6078 |
| macroPrec | 0.8122 |
| macroRec | 0.8330 |
| macroF1 | 0.8202 |
| IoU | 0.7119 |
| microPrec | 0.8720 |
| microRec | 0.8720 |
| microF1 | 0.8720 |
| weightPrec | 0.8823 |
| weightRec | 0.8720 |
| weightF1 | 0.8760 |

Annotated pixels: 361,147. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 294,020 | 81.41 | 282,807 | 265,573 | 17,234 | 28,447 | 49,893 | 0.9032 | 0.8735 | 0.9391 | 0.9032 | 0.9208 | 0.8532 | 0.7433 | 0.06 |
| Marine Debris | 973 | 0.27 | 1,100 | 775 | 325 | 198 | 359,849 | 0.7965 | 0.9986 | 0.7045 | 0.7965 | 0.7477 | 0.5971 | 0.9991 | 0.42 |
| Algae/Sargassum | 3,089 | 0.86 | 2,828 | 2,800 | 28 | 289 | 358,030 | 0.9064 | 0.9991 | 0.9901 | 0.9064 | 0.9464 | 0.8983 | 0.9999 | 0.01 |
| Other | 63,065 | 17.46 | 74,412 | 45,761 | 28,651 | 17,304 | 269,431 | 0.7256 | 0.8728 | 0.6150 | 0.7256 | 0.6657 | 0.4989 | 0.9039 | 0.63 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 265,573 | 28 | 12 | 28,407 | 294,020 |
| Marine Debris | 29 | 775 | 10 | 159 | 973 |
| Algae/Sargassum | 198 | 6 | 2,800 | 85 | 3,089 |
| Other | 17,007 | 291 | 6 | 45,761 | 63,065 |
| total predicted | 282,807 | 1,100 | 2,828 | 74,412 | 361,147 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 90.32 | 0.01 | 0.00 | 9.66 |
| Marine Debris | 2.98 | 79.65 | 1.03 | 16.34 |
| Algae/Sargassum | 6.41 | 0.19 | 90.64 | 2.75 |
| Other | 26.97 | 0.46 | 0.01 | 72.56 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 93.91 | 2.55 | 0.42 | 38.18 |
| Marine Debris | 0.01 | 70.45 | 0.35 | 0.21 |
| Algae/Sargassum | 0.07 | 0.55 | 99.01 | 0.11 |
| Other | 6.01 | 26.45 | 0.21 | 61.50 |

### Where the errors go

- **Water**: 28,447 missed pixels went to: Other 28,407 (99.9%), Marine Debris 28 (0.1%), Algae/Sargassum 12 (0.0%).
  17,234 false-positive pixels came from: Other 17,007 (98.7%), Algae/Sargassum 198 (1.1%), Marine Debris 29 (0.2%).
- **Marine Debris**: 198 missed pixels went to: Other 159 (80.3%), Water 29 (14.6%), Algae/Sargassum 10 (5.1%).
  325 false-positive pixels came from: Other 291 (89.5%), Water 28 (8.6%), Algae/Sargassum 6 (1.8%).
- **Algae/Sargassum**: 289 missed pixels went to: Water 198 (68.5%), Other 85 (29.4%), Marine Debris 6 (2.1%).
  28 false-positive pixels came from: Water 12 (42.9%), Marine Debris 10 (35.7%), Other 6 (21.4%).
- **Other**: 17,304 missed pixels went to: Water 17,007 (98.3%), Marine Debris 291 (1.7%), Algae/Sargassum 6 (0.0%).
  28,651 false-positive pixels came from: Water 28,407 (99.1%), Marine Debris 159 (0.6%), Algae/Sargassum 85 (0.3%).
