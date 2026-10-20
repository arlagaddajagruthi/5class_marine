# MariNeXt-architecture SegNeXt-T (from scratch), MARIDA, run MARIDA_novscp, 4 classes (annotated pixels only)

Recipe: {"epochs": 45, "batch": 5, "lr": 0.0002, "decay": 0.0, "lr_steps": [40], "gamma": 0.1, "weight_param": 1.03, "aug": "marida", "vscp": false, "ema": false, "tta": true, "drop_last": false, "select": "val macroF1", "encoder": "SegNeXt-T (MSCAN-T + LightHamHead), from scratch, stages 1-4 (H/4, MariNeXt modelling)", "inputs": "11 bands (standardised) + NDWI, red/green, nir/red, nir/green, swir1/nir"}

Selected checkpoint: epoch 23 of 45 (model weights), highest val macro F1 with TTA 0.8904. Test-time augmentation: True. VSCP: False.

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
| overall accuracy | 0.9938 |
| balanced accuracy (macro recall) | 0.8660 |
| Cohen's kappa | 0.8674 |
| macroPrec | 0.7926 |
| macroRec | 0.8660 |
| macroF1 | 0.8091 |
| IoU | 0.7306 |
| microPrec | 0.9938 |
| microRec | 0.9938 |
| microF1 | 0.9938 |
| weightPrec | 0.9957 |
| weightRec | 0.9938 |
| weightF1 | 0.9945 |

Annotated pixels: 158,371. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 154,739 | 97.71 | 154,451 | 154,272 | 179 | 467 | 3,453 | 0.9970 | 0.9959 | 0.9988 | 0.9970 | 0.9979 | 0.9958 | 0.9507 | 0.00 |
| Marine Debris | 381 | 0.24 | 864 | 275 | 589 | 106 | 157,401 | 0.7218 | 0.9956 | 0.3183 | 0.7218 | 0.4418 | 0.2835 | 0.9963 | 2.14 |
| Algae/Sargassum | 1,641 | 1.04 | 1,604 | 1,560 | 44 | 81 | 156,686 | 0.9506 | 0.9992 | 0.9726 | 0.9506 | 0.9615 | 0.9258 | 0.9997 | 0.03 |
| Other | 1,610 | 1.02 | 1,452 | 1,279 | 173 | 331 | 156,588 | 0.7944 | 0.9968 | 0.8809 | 0.7944 | 0.8354 | 0.7173 | 0.9989 | 0.14 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 154,272 | 301 | 13 | 153 | 154,739 |
| Marine Debris | 60 | 275 | 26 | 20 | 381 |
| Algae/Sargassum | 8 | 73 | 1,560 | 0 | 1,641 |
| Other | 111 | 215 | 5 | 1,279 | 1,610 |
| total predicted | 154,451 | 864 | 1,604 | 1,452 | 158,371 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.70 | 0.19 | 0.01 | 0.10 |
| Marine Debris | 15.75 | 72.18 | 6.82 | 5.25 |
| Algae/Sargassum | 0.49 | 4.45 | 95.06 | 0.00 |
| Other | 6.89 | 13.35 | 0.31 | 79.44 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.88 | 34.84 | 0.81 | 10.54 |
| Marine Debris | 0.04 | 31.83 | 1.62 | 1.38 |
| Algae/Sargassum | 0.01 | 8.45 | 97.26 | 0.00 |
| Other | 0.07 | 24.88 | 0.31 | 88.09 |

### Where the errors go

- **Water**: 467 missed pixels went to: Marine Debris 301 (64.5%), Other 153 (32.8%), Algae/Sargassum 13 (2.8%).
  179 false-positive pixels came from: Other 111 (62.0%), Marine Debris 60 (33.5%), Algae/Sargassum 8 (4.5%).
- **Marine Debris**: 106 missed pixels went to: Water 60 (56.6%), Algae/Sargassum 26 (24.5%), Other 20 (18.9%).
  589 false-positive pixels came from: Water 301 (51.1%), Other 215 (36.5%), Algae/Sargassum 73 (12.4%).
- **Algae/Sargassum**: 81 missed pixels went to: Marine Debris 73 (90.1%), Water 8 (9.9%).
  44 false-positive pixels came from: Marine Debris 26 (59.1%), Water 13 (29.5%), Other 5 (11.4%).
- **Other**: 331 missed pixels went to: Marine Debris 215 (65.0%), Water 111 (33.5%), Algae/Sargassum 5 (1.5%).
  173 false-positive pixels came from: Water 153 (88.4%), Marine Debris 20 (11.6%).

## model_best_val_f1, val split

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9951 |
| balanced accuracy (macro recall) | 0.9165 |
| Cohen's kappa | 0.8954 |
| macroPrec | 0.8678 |
| macroRec | 0.9165 |
| macroF1 | 0.8896 |
| IoU | 0.8131 |
| microPrec | 0.9951 |
| microRec | 0.9951 |
| microF1 | 0.9951 |
| weightPrec | 0.9956 |
| weightRec | 0.9951 |
| weightF1 | 0.9953 |

Annotated pixels: 191,436. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 187,008 | 97.69 | 186,665 | 186,529 | 136 | 479 | 4,292 | 0.9974 | 0.9968 | 0.9993 | 0.9974 | 0.9984 | 0.9967 | 0.9693 | 0.00 |
| Marine Debris | 1,075 | 0.56 | 1,332 | 910 | 422 | 165 | 189,939 | 0.8465 | 0.9969 | 0.6832 | 0.8465 | 0.7561 | 0.6079 | 0.9978 | 0.46 |
| Algae/Sargassum | 1,552 | 0.81 | 1,508 | 1,441 | 67 | 111 | 189,817 | 0.9285 | 0.9991 | 0.9556 | 0.9285 | 0.9418 | 0.8901 | 0.9996 | 0.05 |
| Other | 1,801 | 0.94 | 1,931 | 1,609 | 322 | 192 | 189,313 | 0.8934 | 0.9973 | 0.8332 | 0.8934 | 0.8623 | 0.7579 | 0.9983 | 0.20 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 186,529 | 209 | 29 | 241 | 187,008 |
| Marine Debris | 49 | 910 | 38 | 78 | 1,075 |
| Algae/Sargassum | 39 | 69 | 1,441 | 3 | 1,552 |
| Other | 48 | 144 | 0 | 1,609 | 1,801 |
| total predicted | 186,665 | 1,332 | 1,508 | 1,931 | 191,436 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.74 | 0.11 | 0.02 | 0.13 |
| Marine Debris | 4.56 | 84.65 | 3.53 | 7.26 |
| Algae/Sargassum | 2.51 | 4.45 | 92.85 | 0.19 |
| Other | 2.67 | 8.00 | 0.00 | 89.34 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.93 | 15.69 | 1.92 | 12.48 |
| Marine Debris | 0.03 | 68.32 | 2.52 | 4.04 |
| Algae/Sargassum | 0.02 | 5.18 | 95.56 | 0.16 |
| Other | 0.03 | 10.81 | 0.00 | 83.32 |

### Where the errors go

- **Water**: 479 missed pixels went to: Other 241 (50.3%), Marine Debris 209 (43.6%), Algae/Sargassum 29 (6.1%).
  136 false-positive pixels came from: Marine Debris 49 (36.0%), Other 48 (35.3%), Algae/Sargassum 39 (28.7%).
- **Marine Debris**: 165 missed pixels went to: Other 78 (47.3%), Water 49 (29.7%), Algae/Sargassum 38 (23.0%).
  422 false-positive pixels came from: Water 209 (49.5%), Other 144 (34.1%), Algae/Sargassum 69 (16.4%).
- **Algae/Sargassum**: 111 missed pixels went to: Marine Debris 69 (62.2%), Water 39 (35.1%), Other 3 (2.7%).
  67 false-positive pixels came from: Marine Debris 38 (56.7%), Water 29 (43.3%).
- **Other**: 192 missed pixels went to: Marine Debris 144 (75.0%), Water 48 (25.0%).
  322 false-positive pixels came from: Water 241 (74.8%), Marine Debris 78 (24.2%), Algae/Sargassum 3 (0.9%).

## model_last, test split

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9924 |
| balanced accuracy (macro recall) | 0.9259 |
| Cohen's kappa | 0.8490 |
| macroPrec | 0.7696 |
| macroRec | 0.9259 |
| macroF1 | 0.8173 |
| IoU | 0.7380 |
| microPrec | 0.9924 |
| microRec | 0.9924 |
| microF1 | 0.9924 |
| weightPrec | 0.9953 |
| weightRec | 0.9924 |
| weightF1 | 0.9935 |

Annotated pixels: 158,371. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 154,739 | 97.71 | 153,932 | 153,855 | 77 | 884 | 3,555 | 0.9943 | 0.9939 | 0.9995 | 0.9943 | 0.9969 | 0.9938 | 0.9788 | 0.00 |
| Marine Debris | 381 | 0.24 | 986 | 333 | 653 | 48 | 157,337 | 0.8740 | 0.9956 | 0.3377 | 0.8740 | 0.4872 | 0.3221 | 0.9959 | 1.96 |
| Algae/Sargassum | 1,641 | 1.04 | 1,635 | 1,610 | 25 | 31 | 156,705 | 0.9811 | 0.9996 | 0.9847 | 0.9811 | 0.9829 | 0.9664 | 0.9998 | 0.02 |
| Other | 1,610 | 1.02 | 1,818 | 1,375 | 443 | 235 | 156,318 | 0.8540 | 0.9957 | 0.7563 | 0.8540 | 0.8022 | 0.6698 | 0.9972 | 0.32 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 153,855 | 431 | 19 | 434 | 154,739 |
| Marine Debris | 37 | 333 | 2 | 9 | 381 |
| Algae/Sargassum | 2 | 29 | 1,610 | 0 | 1,641 |
| Other | 38 | 193 | 4 | 1,375 | 1,610 |
| total predicted | 153,932 | 986 | 1,635 | 1,818 | 158,371 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.43 | 0.28 | 0.01 | 0.28 |
| Marine Debris | 9.71 | 87.40 | 0.52 | 2.36 |
| Algae/Sargassum | 0.12 | 1.77 | 98.11 | 0.00 |
| Other | 2.36 | 11.99 | 0.25 | 85.40 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.95 | 43.71 | 1.16 | 23.87 |
| Marine Debris | 0.02 | 33.77 | 0.12 | 0.50 |
| Algae/Sargassum | 0.00 | 2.94 | 98.47 | 0.00 |
| Other | 0.02 | 19.57 | 0.24 | 75.63 |

### Where the errors go

- **Water**: 884 missed pixels went to: Other 434 (49.1%), Marine Debris 431 (48.8%), Algae/Sargassum 19 (2.1%).
  77 false-positive pixels came from: Other 38 (49.4%), Marine Debris 37 (48.1%), Algae/Sargassum 2 (2.6%).
- **Marine Debris**: 48 missed pixels went to: Water 37 (77.1%), Other 9 (18.8%), Algae/Sargassum 2 (4.2%).
  653 false-positive pixels came from: Water 431 (66.0%), Other 193 (29.6%), Algae/Sargassum 29 (4.4%).
- **Algae/Sargassum**: 31 missed pixels went to: Marine Debris 29 (93.5%), Water 2 (6.5%).
  25 false-positive pixels came from: Water 19 (76.0%), Other 4 (16.0%), Marine Debris 2 (8.0%).
- **Other**: 235 missed pixels went to: Marine Debris 193 (82.1%), Water 38 (16.2%), Algae/Sargassum 4 (1.7%).
  443 false-positive pixels came from: Water 434 (98.0%), Marine Debris 9 (2.0%).

## Validation vs test, per class (model_best_val_f1)

| class | precision val | recall val | F1 val | IoU val | precision test | recall test | F1 test | IoU test | F1 test - val |
|---|---|---|---|---|---|---|---|---|---|
| Water | 0.9993 | 0.9974 | 0.9984 | 0.9967 | 0.9988 | 0.9970 | 0.9979 | 0.9958 | -0.0004 |
| Marine Debris | 0.6832 | 0.8465 | 0.7561 | 0.6079 | 0.3183 | 0.7218 | 0.4418 | 0.2835 | -0.3144 |
| Algae/Sargassum | 0.9556 | 0.9285 | 0.9418 | 0.8901 | 0.9726 | 0.9506 | 0.9615 | 0.9258 | +0.0196 |
| Other | 0.8332 | 0.8934 | 0.8623 | 0.7579 | 0.8809 | 0.7944 | 0.8354 | 0.7173 | -0.0269 |

Large negative F1 test - val means the class generalises worse to the test scenes than validation suggested.
