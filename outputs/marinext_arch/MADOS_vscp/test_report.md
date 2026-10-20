# MariNeXt-architecture SegNeXt-T (from scratch), MADOS, run MADOS_vscp, 4 classes (annotated pixels only)

Recipe: {"epochs": 80, "batch": 5, "lr": 0.0002, "decay": 0.0, "lr_steps": [45, 65], "gamma": 0.1, "weight_param": 1.03, "aug": "mados", "vscp": true, "ema": true, "ema_decay": 0.999, "tta": true, "drop_last": true, "select": "val macroF1", "encoder": "SegNeXt-T (MSCAN-T + LightHamHead), from scratch, stages 1-4 (H/4, MariNeXt modelling)", "inputs": "11 bands (standardised) + NDWI, red/green, nir/red, nir/green, swir1/nir"}

Selected checkpoint: epoch 45 of 80 (ema weights), highest val macro F1 with TTA 0.8442. Test-time augmentation: True. VSCP: True.

## Class distribution (annotated pixels per split)

| class | train pixels | val pixels | test pixels | train % | val % | test % |
|---|---|---|---|---|---|---|
| Water | 591,224 | 300,952 | 294,020 | 77.60 | 84.04 | 81.41 |
| Marine Debris | 2,561 | 1,162 | 973 | 0.34 | 0.32 | 0.27 |
| Algae/Sargassum | 4,397 | 1,794 | 3,089 | 0.58 | 0.50 | 0.86 |
| Other | 163,705 | 54,213 | 63,065 | 21.49 | 15.14 | 17.46 |

## model_best_val_f1, test split

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9236 |
| balanced accuracy (macro recall) | 0.8976 |
| Cohen's kappa | 0.7700 |
| macroPrec | 0.7548 |
| macroRec | 0.8976 |
| macroF1 | 0.7976 |
| IoU | 0.7070 |
| microPrec | 0.9236 |
| microRec | 0.9236 |
| microF1 | 0.9236 |
| weightPrec | 0.9347 |
| weightRec | 0.9236 |
| weightF1 | 0.9273 |

Annotated pixels: 361,147. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 294,020 | 81.41 | 280,085 | 274,017 | 6,068 | 20,003 | 61,059 | 0.9320 | 0.9278 | 0.9783 | 0.9320 | 0.9546 | 0.9131 | 0.9096 | 0.02 |
| Marine Debris | 973 | 0.27 | 2,465 | 792 | 1,673 | 181 | 358,501 | 0.8140 | 0.9949 | 0.3213 | 0.8140 | 0.4607 | 0.2993 | 0.9954 | 2.11 |
| Algae/Sargassum | 3,089 | 0.86 | 3,021 | 2,964 | 57 | 125 | 358,001 | 0.9595 | 0.9995 | 0.9811 | 0.9595 | 0.9702 | 0.9421 | 0.9998 | 0.02 |
| Other | 63,065 | 17.46 | 75,576 | 55,799 | 19,777 | 7,266 | 278,305 | 0.8848 | 0.9251 | 0.7383 | 0.8848 | 0.8049 | 0.6736 | 0.9337 | 0.35 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 274,017 | 322 | 41 | 19,640 | 294,020 |
| Marine Debris | 37 | 792 | 7 | 137 | 973 |
| Algae/Sargassum | 66 | 59 | 2,964 | 0 | 3,089 |
| Other | 5,965 | 1,292 | 9 | 55,799 | 63,065 |
| total predicted | 280,085 | 2,465 | 3,021 | 75,576 | 361,147 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 93.20 | 0.11 | 0.01 | 6.68 |
| Marine Debris | 3.80 | 81.40 | 0.72 | 14.08 |
| Algae/Sargassum | 2.14 | 1.91 | 95.95 | 0.00 |
| Other | 9.46 | 2.05 | 0.01 | 88.48 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 97.83 | 13.06 | 1.36 | 25.99 |
| Marine Debris | 0.01 | 32.13 | 0.23 | 0.18 |
| Algae/Sargassum | 0.02 | 2.39 | 98.11 | 0.00 |
| Other | 2.13 | 52.41 | 0.30 | 73.83 |

### Where the errors go

- **Water**: 20,003 missed pixels went to: Other 19,640 (98.2%), Marine Debris 322 (1.6%), Algae/Sargassum 41 (0.2%).
  6,068 false-positive pixels came from: Other 5,965 (98.3%), Algae/Sargassum 66 (1.1%), Marine Debris 37 (0.6%).
- **Marine Debris**: 181 missed pixels went to: Other 137 (75.7%), Water 37 (20.4%), Algae/Sargassum 7 (3.9%).
  1,673 false-positive pixels came from: Other 1,292 (77.2%), Water 322 (19.2%), Algae/Sargassum 59 (3.5%).
- **Algae/Sargassum**: 125 missed pixels went to: Water 66 (52.8%), Marine Debris 59 (47.2%).
  57 false-positive pixels came from: Water 41 (71.9%), Other 9 (15.8%), Marine Debris 7 (12.3%).
- **Other**: 7,266 missed pixels went to: Water 5,965 (82.1%), Marine Debris 1,292 (17.8%), Algae/Sargassum 9 (0.1%).
  19,777 false-positive pixels came from: Water 19,640 (99.3%), Marine Debris 137 (0.7%).

## model_best_val_f1, val split

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9410 |
| balanced accuracy (macro recall) | 0.8869 |
| Cohen's kappa | 0.7520 |
| macroPrec | 0.8474 |
| macroRec | 0.8869 |
| macroF1 | 0.8441 |
| IoU | 0.7535 |
| microPrec | 0.9410 |
| microRec | 0.9410 |
| microF1 | 0.9410 |
| weightPrec | 0.9433 |
| weightRec | 0.9410 |
| weightF1 | 0.9363 |

Annotated pixels: 358,121. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 300,952 | 84.04 | 318,243 | 299,425 | 18,818 | 1,527 | 38,351 | 0.9949 | 0.9432 | 0.9409 | 0.9949 | 0.9671 | 0.9364 | 0.6708 | 0.06 |
| Marine Debris | 1,162 | 0.32 | 2,088 | 1,073 | 1,015 | 89 | 355,944 | 0.9234 | 0.9969 | 0.5139 | 0.9234 | 0.6603 | 0.4929 | 0.9972 | 0.95 |
| Algae/Sargassum | 1,794 | 0.50 | 1,830 | 1,774 | 56 | 20 | 356,271 | 0.9889 | 0.9998 | 0.9694 | 0.9889 | 0.9790 | 0.9589 | 0.9998 | 0.03 |
| Other | 54,213 | 15.14 | 35,960 | 34,712 | 1,248 | 19,501 | 302,660 | 0.6403 | 0.9421 | 0.9653 | 0.6403 | 0.7699 | 0.6259 | 0.9959 | 0.04 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 299,425 | 293 | 40 | 1,194 | 300,952 |
| Marine Debris | 19 | 1,073 | 16 | 54 | 1,162 |
| Algae/Sargassum | 0 | 20 | 1,774 | 0 | 1,794 |
| Other | 18,799 | 702 | 0 | 34,712 | 54,213 |
| total predicted | 318,243 | 2,088 | 1,830 | 35,960 | 358,121 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.49 | 0.10 | 0.01 | 0.40 |
| Marine Debris | 1.64 | 92.34 | 1.38 | 4.65 |
| Algae/Sargassum | 0.00 | 1.11 | 98.89 | 0.00 |
| Other | 34.68 | 1.29 | 0.00 | 64.03 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 94.09 | 14.03 | 2.19 | 3.32 |
| Marine Debris | 0.01 | 51.39 | 0.87 | 0.15 |
| Algae/Sargassum | 0.00 | 0.96 | 96.94 | 0.00 |
| Other | 5.91 | 33.62 | 0.00 | 96.53 |

### Where the errors go

- **Water**: 1,527 missed pixels went to: Other 1,194 (78.2%), Marine Debris 293 (19.2%), Algae/Sargassum 40 (2.6%).
  18,818 false-positive pixels came from: Other 18,799 (99.9%), Marine Debris 19 (0.1%).
- **Marine Debris**: 89 missed pixels went to: Other 54 (60.7%), Water 19 (21.3%), Algae/Sargassum 16 (18.0%).
  1,015 false-positive pixels came from: Other 702 (69.2%), Water 293 (28.9%), Algae/Sargassum 20 (2.0%).
- **Algae/Sargassum**: 20 missed pixels went to: Marine Debris 20 (100.0%).
  56 false-positive pixels came from: Water 40 (71.4%), Marine Debris 16 (28.6%).
- **Other**: 19,501 missed pixels went to: Water 18,799 (96.4%), Marine Debris 702 (3.6%).
  1,248 false-positive pixels came from: Water 1,194 (95.7%), Marine Debris 54 (4.3%).

## model_last, test split

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9369 |
| balanced accuracy (macro recall) | 0.9017 |
| Cohen's kappa | 0.8033 |
| macroPrec | 0.7671 |
| macroRec | 0.9017 |
| macroF1 | 0.8062 |
| IoU | 0.7213 |
| microPrec | 0.9369 |
| microRec | 0.9369 |
| microF1 | 0.9369 |
| weightPrec | 0.9428 |
| weightRec | 0.9369 |
| weightF1 | 0.9391 |

Annotated pixels: 361,147. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 294,020 | 81.41 | 286,351 | 279,543 | 6,808 | 14,477 | 60,319 | 0.9508 | 0.9411 | 0.9762 | 0.9508 | 0.9633 | 0.9292 | 0.8986 | 0.02 |
| Marine Debris | 973 | 0.27 | 2,516 | 795 | 1,721 | 178 | 358,453 | 0.8171 | 0.9947 | 0.3160 | 0.8171 | 0.4557 | 0.2951 | 0.9952 | 2.16 |
| Algae/Sargassum | 3,089 | 0.86 | 3,042 | 2,985 | 57 | 104 | 358,001 | 0.9663 | 0.9996 | 0.9813 | 0.9663 | 0.9737 | 0.9488 | 0.9998 | 0.02 |
| Other | 63,065 | 17.46 | 69,238 | 55,028 | 14,210 | 8,037 | 283,872 | 0.8726 | 0.9384 | 0.7948 | 0.8726 | 0.8318 | 0.7121 | 0.9523 | 0.26 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 279,543 | 369 | 40 | 14,068 | 294,020 |
| Marine Debris | 30 | 795 | 6 | 142 | 973 |
| Algae/Sargassum | 65 | 39 | 2,985 | 0 | 3,089 |
| Other | 6,713 | 1,313 | 11 | 55,028 | 63,065 |
| total predicted | 286,351 | 2,516 | 3,042 | 69,238 | 361,147 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 95.08 | 0.13 | 0.01 | 4.78 |
| Marine Debris | 3.08 | 81.71 | 0.62 | 14.59 |
| Algae/Sargassum | 2.10 | 1.26 | 96.63 | 0.00 |
| Other | 10.64 | 2.08 | 0.02 | 87.26 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 97.62 | 14.67 | 1.31 | 20.32 |
| Marine Debris | 0.01 | 31.60 | 0.20 | 0.21 |
| Algae/Sargassum | 0.02 | 1.55 | 98.13 | 0.00 |
| Other | 2.34 | 52.19 | 0.36 | 79.48 |

### Where the errors go

- **Water**: 14,477 missed pixels went to: Other 14,068 (97.2%), Marine Debris 369 (2.5%), Algae/Sargassum 40 (0.3%).
  6,808 false-positive pixels came from: Other 6,713 (98.6%), Algae/Sargassum 65 (1.0%), Marine Debris 30 (0.4%).
- **Marine Debris**: 178 missed pixels went to: Other 142 (79.8%), Water 30 (16.9%), Algae/Sargassum 6 (3.4%).
  1,721 false-positive pixels came from: Other 1,313 (76.3%), Water 369 (21.4%), Algae/Sargassum 39 (2.3%).
- **Algae/Sargassum**: 104 missed pixels went to: Water 65 (62.5%), Marine Debris 39 (37.5%).
  57 false-positive pixels came from: Water 40 (70.2%), Other 11 (19.3%), Marine Debris 6 (10.5%).
- **Other**: 8,037 missed pixels went to: Water 6,713 (83.5%), Marine Debris 1,313 (16.3%), Algae/Sargassum 11 (0.1%).
  14,210 false-positive pixels came from: Water 14,068 (99.0%), Marine Debris 142 (1.0%).

## Validation vs test, per class (model_best_val_f1)

| class | precision val | recall val | F1 val | IoU val | precision test | recall test | F1 test | IoU test | F1 test - val |
|---|---|---|---|---|---|---|---|---|---|
| Water | 0.9409 | 0.9949 | 0.9671 | 0.9364 | 0.9783 | 0.9320 | 0.9546 | 0.9131 | -0.0126 |
| Marine Debris | 0.5139 | 0.9234 | 0.6603 | 0.4929 | 0.3213 | 0.8140 | 0.4607 | 0.2993 | -0.1996 |
| Algae/Sargassum | 0.9694 | 0.9889 | 0.9790 | 0.9589 | 0.9811 | 0.9595 | 0.9702 | 0.9421 | -0.0088 |
| Other | 0.9653 | 0.6403 | 0.7699 | 0.6259 | 0.7383 | 0.8848 | 0.8049 | 0.6736 | +0.0350 |

Large negative F1 test - val means the class generalises worse to the test scenes than validation suggested.
