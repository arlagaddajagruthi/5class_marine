# MariNeXt-architecture SegNeXt-T (from scratch), MARIDA, run MARIDA_vscp, 4 classes (annotated pixels only)

Recipe: {"epochs": 45, "batch": 5, "lr": 0.0002, "decay": 0.0, "lr_steps": [40], "gamma": 0.1, "weight_param": 1.03, "aug": "marida", "vscp": true, "ema": false, "tta": true, "drop_last": false, "select": "val macroF1", "encoder": "SegNeXt-T (MSCAN-T + LightHamHead), from scratch, stages 1-4 (H/4, MariNeXt modelling)", "inputs": "11 bands (standardised) + NDWI, red/green, nir/red, nir/green, swir1/nir"}

Selected checkpoint: epoch 37 of 45 (model weights), highest val macro F1 with TTA 0.8931. Test-time augmentation: True. VSCP: True.

## Class distribution (annotated pixels per split)

| class | train pixels | val pixels | test pixels | train % | val % | test % |
|---|---|---|---|---|---|---|
| Water | 350,057 | 187,008 | 154,739 | 97.66 | 97.69 | 97.71 |
| Marine Debris | 1,943 | 1,075 | 381 | 0.54 | 0.56 | 0.24 |
| Algae/Sargassum | 1,961 | 1,552 | 1,641 | 0.55 | 0.81 | 1.04 |
| Other | 4,481 | 1,801 | 1,610 | 1.25 | 0.94 | 1.02 |

## model_best_val_f1, test split

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9937 |
| balanced accuracy (macro recall) | 0.9301 |
| Cohen's kappa | 0.8710 |
| macroPrec | 0.7909 |
| macroRec | 0.9301 |
| macroF1 | 0.8347 |
| IoU | 0.7611 |
| microPrec | 0.9937 |
| microRec | 0.9937 |
| microF1 | 0.9937 |
| weightPrec | 0.9959 |
| weightRec | 0.9937 |
| weightF1 | 0.9945 |

Annotated pixels: 158,371. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 154,739 | 97.71 | 154,088 | 153,997 | 91 | 742 | 3,541 | 0.9952 | 0.9947 | 0.9994 | 0.9952 | 0.9973 | 0.9946 | 0.9749 | 0.00 |
| Marine Debris | 381 | 0.24 | 916 | 325 | 591 | 56 | 157,399 | 0.8530 | 0.9959 | 0.3548 | 0.8530 | 0.5012 | 0.3344 | 0.9963 | 1.82 |
| Algae/Sargassum | 1,641 | 1.04 | 1,676 | 1,625 | 51 | 16 | 156,679 | 0.9902 | 0.9996 | 0.9696 | 0.9902 | 0.9798 | 0.9604 | 0.9997 | 0.03 |
| Other | 1,610 | 1.02 | 1,691 | 1,420 | 271 | 190 | 156,490 | 0.8820 | 0.9971 | 0.8397 | 0.8820 | 0.8603 | 0.7549 | 0.9983 | 0.19 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 153,997 | 448 | 40 | 254 | 154,739 |
| Marine Debris | 35 | 325 | 4 | 17 | 381 |
| Algae/Sargassum | 3 | 13 | 1,625 | 0 | 1,641 |
| Other | 53 | 130 | 7 | 1,420 | 1,610 |
| total predicted | 154,088 | 916 | 1,676 | 1,691 | 158,371 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.52 | 0.29 | 0.03 | 0.16 |
| Marine Debris | 9.19 | 85.30 | 1.05 | 4.46 |
| Algae/Sargassum | 0.18 | 0.79 | 99.02 | 0.00 |
| Other | 3.29 | 8.07 | 0.43 | 88.20 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.94 | 48.91 | 2.39 | 15.02 |
| Marine Debris | 0.02 | 35.48 | 0.24 | 1.01 |
| Algae/Sargassum | 0.00 | 1.42 | 96.96 | 0.00 |
| Other | 0.03 | 14.19 | 0.42 | 83.97 |

### Where the errors go

- **Water**: 742 missed pixels went to: Marine Debris 448 (60.4%), Other 254 (34.2%), Algae/Sargassum 40 (5.4%).
  91 false-positive pixels came from: Other 53 (58.2%), Marine Debris 35 (38.5%), Algae/Sargassum 3 (3.3%).
- **Marine Debris**: 56 missed pixels went to: Water 35 (62.5%), Other 17 (30.4%), Algae/Sargassum 4 (7.1%).
  591 false-positive pixels came from: Water 448 (75.8%), Other 130 (22.0%), Algae/Sargassum 13 (2.2%).
- **Algae/Sargassum**: 16 missed pixels went to: Marine Debris 13 (81.2%), Water 3 (18.8%).
  51 false-positive pixels came from: Water 40 (78.4%), Other 7 (13.7%), Marine Debris 4 (7.8%).
- **Other**: 190 missed pixels went to: Marine Debris 130 (68.4%), Water 53 (27.9%), Algae/Sargassum 7 (3.7%).
  271 false-positive pixels came from: Water 254 (93.7%), Marine Debris 17 (6.3%).

## model_best_val_f1, val split

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9946 |
| balanced accuracy (macro recall) | 0.9398 |
| Cohen's kappa | 0.8902 |
| macroPrec | 0.8566 |
| macroRec | 0.9398 |
| macroF1 | 0.8930 |
| IoU | 0.8207 |
| microPrec | 0.9946 |
| microRec | 0.9946 |
| microF1 | 0.9946 |
| weightPrec | 0.9956 |
| weightRec | 0.9946 |
| weightF1 | 0.9950 |

Annotated pixels: 191,436. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 187,008 | 97.69 | 186,375 | 186,311 | 64 | 697 | 4,364 | 0.9963 | 0.9960 | 0.9997 | 0.9963 | 0.9980 | 0.9959 | 0.9855 | 0.00 |
| Marine Debris | 1,075 | 0.56 | 1,439 | 942 | 497 | 133 | 189,864 | 0.8763 | 0.9967 | 0.6546 | 0.8763 | 0.7494 | 0.5992 | 0.9974 | 0.53 |
| Algae/Sargassum | 1,552 | 0.81 | 1,529 | 1,494 | 35 | 58 | 189,849 | 0.9626 | 0.9995 | 0.9771 | 0.9626 | 0.9698 | 0.9414 | 0.9998 | 0.02 |
| Other | 1,801 | 0.94 | 2,093 | 1,664 | 429 | 137 | 189,206 | 0.9239 | 0.9970 | 0.7950 | 0.9239 | 0.8546 | 0.7462 | 0.9977 | 0.26 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 186,311 | 323 | 33 | 341 | 187,008 |
| Marine Debris | 43 | 942 | 2 | 88 | 1,075 |
| Algae/Sargassum | 1 | 57 | 1,494 | 0 | 1,552 |
| Other | 20 | 117 | 0 | 1,664 | 1,801 |
| total predicted | 186,375 | 1,439 | 1,529 | 2,093 | 191,436 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.63 | 0.17 | 0.02 | 0.18 |
| Marine Debris | 4.00 | 87.63 | 0.19 | 8.19 |
| Algae/Sargassum | 0.06 | 3.67 | 96.26 | 0.00 |
| Other | 1.11 | 6.50 | 0.00 | 92.39 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.97 | 22.45 | 2.16 | 16.29 |
| Marine Debris | 0.02 | 65.46 | 0.13 | 4.20 |
| Algae/Sargassum | 0.00 | 3.96 | 97.71 | 0.00 |
| Other | 0.01 | 8.13 | 0.00 | 79.50 |

### Where the errors go

- **Water**: 697 missed pixels went to: Other 341 (48.9%), Marine Debris 323 (46.3%), Algae/Sargassum 33 (4.7%).
  64 false-positive pixels came from: Marine Debris 43 (67.2%), Other 20 (31.2%), Algae/Sargassum 1 (1.6%).
- **Marine Debris**: 133 missed pixels went to: Other 88 (66.2%), Water 43 (32.3%), Algae/Sargassum 2 (1.5%).
  497 false-positive pixels came from: Water 323 (65.0%), Other 117 (23.5%), Algae/Sargassum 57 (11.5%).
- **Algae/Sargassum**: 58 missed pixels went to: Marine Debris 57 (98.3%), Water 1 (1.7%).
  35 false-positive pixels came from: Water 33 (94.3%), Marine Debris 2 (5.7%).
- **Other**: 137 missed pixels went to: Marine Debris 117 (85.4%), Water 20 (14.6%).
  429 false-positive pixels came from: Water 341 (79.5%), Marine Debris 88 (20.5%).

## model_last, test split

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9935 |
| balanced accuracy (macro recall) | 0.9405 |
| Cohen's kappa | 0.8676 |
| macroPrec | 0.7952 |
| macroRec | 0.9405 |
| macroF1 | 0.8337 |
| IoU | 0.7612 |
| microPrec | 0.9935 |
| microRec | 0.9935 |
| microF1 | 0.9935 |
| weightPrec | 0.9963 |
| weightRec | 0.9935 |
| weightF1 | 0.9945 |

Annotated pixels: 158,371. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 154,739 | 97.71 | 154,079 | 153,990 | 89 | 749 | 3,543 | 0.9952 | 0.9947 | 0.9994 | 0.9952 | 0.9973 | 0.9946 | 0.9755 | 0.00 |
| Marine Debris | 381 | 0.24 | 1,045 | 352 | 693 | 29 | 157,297 | 0.9239 | 0.9954 | 0.3368 | 0.9239 | 0.4937 | 0.3277 | 0.9956 | 1.97 |
| Algae/Sargassum | 1,641 | 1.04 | 1,653 | 1,621 | 32 | 20 | 156,698 | 0.9878 | 0.9997 | 0.9806 | 0.9878 | 0.9842 | 0.9689 | 0.9998 | 0.02 |
| Other | 1,610 | 1.02 | 1,594 | 1,377 | 217 | 233 | 156,544 | 0.8553 | 0.9972 | 0.8639 | 0.8553 | 0.8596 | 0.7537 | 0.9986 | 0.16 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 153,990 | 510 | 27 | 212 | 154,739 |
| Marine Debris | 20 | 352 | 4 | 5 | 381 |
| Algae/Sargassum | 0 | 20 | 1,621 | 0 | 1,641 |
| Other | 69 | 163 | 1 | 1,377 | 1,610 |
| total predicted | 154,079 | 1,045 | 1,653 | 1,594 | 158,371 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.52 | 0.33 | 0.02 | 0.14 |
| Marine Debris | 5.25 | 92.39 | 1.05 | 1.31 |
| Algae/Sargassum | 0.00 | 1.22 | 98.78 | 0.00 |
| Other | 4.29 | 10.12 | 0.06 | 85.53 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.94 | 48.80 | 1.63 | 13.30 |
| Marine Debris | 0.01 | 33.68 | 0.24 | 0.31 |
| Algae/Sargassum | 0.00 | 1.91 | 98.06 | 0.00 |
| Other | 0.04 | 15.60 | 0.06 | 86.39 |

### Where the errors go

- **Water**: 749 missed pixels went to: Marine Debris 510 (68.1%), Other 212 (28.3%), Algae/Sargassum 27 (3.6%).
  89 false-positive pixels came from: Other 69 (77.5%), Marine Debris 20 (22.5%).
- **Marine Debris**: 29 missed pixels went to: Water 20 (69.0%), Other 5 (17.2%), Algae/Sargassum 4 (13.8%).
  693 false-positive pixels came from: Water 510 (73.6%), Other 163 (23.5%), Algae/Sargassum 20 (2.9%).
- **Algae/Sargassum**: 20 missed pixels went to: Marine Debris 20 (100.0%).
  32 false-positive pixels came from: Water 27 (84.4%), Marine Debris 4 (12.5%), Other 1 (3.1%).
- **Other**: 233 missed pixels went to: Marine Debris 163 (70.0%), Water 69 (29.6%), Algae/Sargassum 1 (0.4%).
  217 false-positive pixels came from: Water 212 (97.7%), Marine Debris 5 (2.3%).

## Validation vs test, per class (model_best_val_f1)

| class | precision val | recall val | F1 val | IoU val | precision test | recall test | F1 test | IoU test | F1 test - val |
|---|---|---|---|---|---|---|---|---|---|
| Water | 0.9997 | 0.9963 | 0.9980 | 0.9959 | 0.9994 | 0.9952 | 0.9973 | 0.9946 | -0.0007 |
| Marine Debris | 0.6546 | 0.8763 | 0.7494 | 0.5992 | 0.3548 | 0.8530 | 0.5012 | 0.3344 | -0.2482 |
| Algae/Sargassum | 0.9771 | 0.9626 | 0.9698 | 0.9414 | 0.9696 | 0.9902 | 0.9798 | 0.9604 | +0.0100 |
| Other | 0.7950 | 0.9239 | 0.8546 | 0.7462 | 0.8397 | 0.8820 | 0.8603 | 0.7549 | +0.0057 |

Large negative F1 test - val means the class generalises worse to the test scenes than validation suggested.
