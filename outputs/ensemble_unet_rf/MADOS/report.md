# U-Net + Random Forest ensemble, MADOS

p = w * p_unet + (1 - w) * p_rf. The U-Net run and w were chosen on the validation split (highest macro F1): **MADOS_vscp, w_unet = 0.55** (val macro F1 0.8812). Test pixels: all annotated pixels of the official test split.

U-Net probabilities are the softmax averaged over 8 TTA views, so the U-Net-only row can differ slightly from its own test report (majority vote over views).

## Overall (test)

| model | macro F1 | IoU | accuracy | kappa | macro precision | macro recall |
|---|---|---|---|---|---|---|
| Random Forest | 0.7190 | 0.5750 | 0.7342 | 0.3867 | 0.7504 | 0.7480 |
| U-Net (MADOS_vscp, soft TTA) | 0.8240 | 0.7268 | 0.9250 | 0.7273 | 0.8334 | 0.8393 |
| Ensemble (w_unet = 0.55) | 0.8384 | 0.7365 | 0.8874 | 0.6572 | 0.8272 | 0.8543 |

## Per-class precision (test)

| class | Random Forest | U-Net (MADOS_vscp, soft TTA) | Ensemble (w_unet = 0.55) |
|---|---|---|---|
| Water | 0.9487 | 0.9290 | 0.9509 |
| Marine Debris | 0.6844 | 0.5147 | 0.7196 |
| Algae/Sargassum | 0.9896 | 0.9834 | 0.9893 |
| Other | 0.3790 | 0.9063 | 0.6492 |

## Per-class recall (test)

| class | Random Forest | U-Net (MADOS_vscp, soft TTA) | Ensemble (w_unet = 0.55) |
|---|---|---|---|
| Water | 0.7171 | 0.9859 | 0.9107 |
| Marine Debris | 0.7533 | 0.7729 | 0.8016 |
| Algae/Sargassum | 0.7067 | 0.9566 | 0.9268 |
| Other | 0.8150 | 0.6420 | 0.7781 |

## Per-class F1 (test)

| class | Random Forest | U-Net (MADOS_vscp, soft TTA) | Ensemble (w_unet = 0.55) |
|---|---|---|---|
| Water | 0.8168 | 0.9566 | 0.9303 |
| Marine Debris | 0.7172 | 0.6179 | 0.7584 |
| Algae/Sargassum | 0.8246 | 0.9698 | 0.9570 |
| Other | 0.5174 | 0.7516 | 0.7078 |

## Per-class IoU (test)

| class | Random Forest | U-Net (MADOS_vscp, soft TTA) | Ensemble (w_unet = 0.55) |
|---|---|---|---|
| Water | 0.6903 | 0.9169 | 0.8697 |
| Marine Debris | 0.5591 | 0.4471 | 0.6108 |
| Algae/Sargassum | 0.7015 | 0.9414 | 0.9176 |
| Other | 0.3490 | 0.6021 | 0.5478 |

## Validation macro F1 by weight (used for the choice)

| class | MADOS_vscp |
|---|---|
| 0.0 | 0.7821 |
| 0.05 | 0.7920 |
| 0.1 | 0.8052 |
| 0.15 | 0.8192 |
| 0.2 | 0.8373 |
| 0.25 | 0.8506 |
| 0.3 | 0.8582 |
| 0.35 | 0.8628 |
| 0.4 | 0.8689 |
| 0.45 | 0.8741 |
| 0.5 | 0.8802 |
| 0.55 | 0.8812 |
| 0.6 | 0.8767 |
| 0.65 | 0.8727 |
| 0.7 | 0.8694 |
| 0.75 | 0.8654 |
| 0.8 | 0.8625 |
| 0.85 | 0.8586 |
| 0.9 | 0.8571 |
| 0.95 | 0.8546 |
| 1.0 | 0.8521 |

Test-split curve (reference only, not used for selection): weight_curve.csv

## Random Forest, test split, full class-wise analysis

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.7342 |
| balanced accuracy (macro recall) | 0.7480 |
| Cohen's kappa | 0.3867 |
| macroPrec | 0.7504 |
| macroRec | 0.7480 |
| macroF1 | 0.7190 |
| IoU | 0.5750 |
| microPrec | 0.7342 |
| microRec | 0.7342 |
| microF1 | 0.7342 |
| weightPrec | 0.8488 |
| weightRec | 0.7342 |
| weightF1 | 0.7643 |

Annotated pixels: 361,147. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 294,020 | 81.41 | 222,256 | 210,846 | 11,410 | 83,174 | 55,717 | 0.7171 | 0.7381 | 0.9487 | 0.7171 | 0.8168 | 0.6903 | 0.8300 | 0.05 |
| Marine Debris | 973 | 0.27 | 1,071 | 733 | 338 | 240 | 359,836 | 0.7533 | 0.9984 | 0.6844 | 0.7533 | 0.7172 | 0.5591 | 0.9991 | 0.46 |
| Algae/Sargassum | 3,089 | 0.86 | 2,206 | 2,183 | 23 | 906 | 358,035 | 0.7067 | 0.9974 | 0.9896 | 0.7067 | 0.8246 | 0.7015 | 0.9999 | 0.01 |
| Other | 63,065 | 17.46 | 135,614 | 51,399 | 84,215 | 11,666 | 213,867 | 0.8150 | 0.7345 | 0.3790 | 0.8150 | 0.5174 | 0.3490 | 0.7175 | 1.64 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 210,846 | 29 | 15 | 83,130 | 294,020 |
| Marine Debris | 36 | 733 | 3 | 201 | 973 |
| Algae/Sargassum | 20 | 2 | 2,183 | 884 | 3,089 |
| Other | 11,354 | 307 | 5 | 51,399 | 63,065 |
| total predicted | 222,256 | 1,071 | 2,206 | 135,614 | 361,147 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 71.71 | 0.01 | 0.01 | 28.27 |
| Marine Debris | 3.70 | 75.33 | 0.31 | 20.66 |
| Algae/Sargassum | 0.65 | 0.06 | 70.67 | 28.62 |
| Other | 18.00 | 0.49 | 0.01 | 81.50 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 94.87 | 2.71 | 0.68 | 61.30 |
| Marine Debris | 0.02 | 68.44 | 0.14 | 0.15 |
| Algae/Sargassum | 0.01 | 0.19 | 98.96 | 0.65 |
| Other | 5.11 | 28.66 | 0.23 | 37.90 |

### Where the errors go

- **Water**: 83,174 missed pixels went to: Other 83,130 (99.9%), Marine Debris 29 (0.0%), Algae/Sargassum 15 (0.0%).
  11,410 false-positive pixels came from: Other 11,354 (99.5%), Marine Debris 36 (0.3%), Algae/Sargassum 20 (0.2%).
- **Marine Debris**: 240 missed pixels went to: Other 201 (83.8%), Water 36 (15.0%), Algae/Sargassum 3 (1.2%).
  338 false-positive pixels came from: Other 307 (90.8%), Water 29 (8.6%), Algae/Sargassum 2 (0.6%).
- **Algae/Sargassum**: 906 missed pixels went to: Other 884 (97.6%), Water 20 (2.2%), Marine Debris 2 (0.2%).
  23 false-positive pixels came from: Water 15 (65.2%), Other 5 (21.7%), Marine Debris 3 (13.0%).
- **Other**: 11,666 missed pixels went to: Water 11,354 (97.3%), Marine Debris 307 (2.6%), Algae/Sargassum 5 (0.0%).
  84,215 false-positive pixels came from: Water 83,130 (98.7%), Algae/Sargassum 884 (1.0%), Marine Debris 201 (0.2%).

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

## Ensemble (w_unet = 0.55), test split, full class-wise analysis

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.8874 |
| balanced accuracy (macro recall) | 0.8543 |
| Cohen's kappa | 0.6572 |
| macroPrec | 0.8272 |
| macroRec | 0.8543 |
| macroF1 | 0.8384 |
| IoU | 0.7365 |
| microPrec | 0.8874 |
| microRec | 0.8874 |
| microF1 | 0.8874 |
| weightPrec | 0.8979 |
| weightRec | 0.8874 |
| weightF1 | 0.8912 |

Annotated pixels: 361,147. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 294,020 | 81.41 | 281,590 | 267,753 | 13,837 | 26,267 | 53,290 | 0.9107 | 0.8890 | 0.9509 | 0.9107 | 0.9303 | 0.8697 | 0.7939 | 0.05 |
| Marine Debris | 973 | 0.27 | 1,084 | 780 | 304 | 193 | 359,870 | 0.8016 | 0.9986 | 0.7196 | 0.8016 | 0.7584 | 0.6108 | 0.9992 | 0.39 |
| Algae/Sargassum | 3,089 | 0.86 | 2,894 | 2,863 | 31 | 226 | 358,027 | 0.9268 | 0.9993 | 0.9893 | 0.9268 | 0.9570 | 0.9176 | 0.9999 | 0.01 |
| Other | 63,065 | 17.46 | 75,579 | 49,069 | 26,510 | 13,996 | 271,572 | 0.7781 | 0.8878 | 0.6492 | 0.7781 | 0.7078 | 0.5478 | 0.9111 | 0.54 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 267,753 | 37 | 15 | 26,215 | 294,020 |
| Marine Debris | 27 | 780 | 10 | 156 | 973 |
| Algae/Sargassum | 87 | 0 | 2,863 | 139 | 3,089 |
| Other | 13,723 | 267 | 6 | 49,069 | 63,065 |
| total predicted | 281,590 | 1,084 | 2,894 | 75,579 | 361,147 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 91.07 | 0.01 | 0.01 | 8.92 |
| Marine Debris | 2.77 | 80.16 | 1.03 | 16.03 |
| Algae/Sargassum | 2.82 | 0.00 | 92.68 | 4.50 |
| Other | 21.76 | 0.42 | 0.01 | 77.81 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 95.09 | 3.41 | 0.52 | 34.69 |
| Marine Debris | 0.01 | 71.96 | 0.35 | 0.21 |
| Algae/Sargassum | 0.03 | 0.00 | 98.93 | 0.18 |
| Other | 4.87 | 24.63 | 0.21 | 64.92 |

### Where the errors go

- **Water**: 26,267 missed pixels went to: Other 26,215 (99.8%), Marine Debris 37 (0.1%), Algae/Sargassum 15 (0.1%).
  13,837 false-positive pixels came from: Other 13,723 (99.2%), Algae/Sargassum 87 (0.6%), Marine Debris 27 (0.2%).
- **Marine Debris**: 193 missed pixels went to: Other 156 (80.8%), Water 27 (14.0%), Algae/Sargassum 10 (5.2%).
  304 false-positive pixels came from: Other 267 (87.8%), Water 37 (12.2%).
- **Algae/Sargassum**: 226 missed pixels went to: Other 139 (61.5%), Water 87 (38.5%).
  31 false-positive pixels came from: Water 15 (48.4%), Marine Debris 10 (32.3%), Other 6 (19.4%).
- **Other**: 13,996 missed pixels went to: Water 13,723 (98.0%), Marine Debris 267 (1.9%), Algae/Sargassum 6 (0.0%).
  26,510 false-positive pixels came from: Water 26,215 (98.9%), Marine Debris 156 (0.6%), Algae/Sargassum 139 (0.5%).
