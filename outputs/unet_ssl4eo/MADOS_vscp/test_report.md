# U-Net (ResNet-50, SSL4EO-S12 pretrained), MADOS, run MADOS_vscp, 4 classes (annotated pixels only)

Recipe: {"epochs": 80, "batch": 5, "lr": 0.0002, "decay": 0.0, "lr_steps": [45, 65], "gamma": 0.1, "weight_param": 1.03, "aug": "mados", "vscp": true, "ema": true, "ema_decay": 0.999, "tta": true, "drop_last": true, "select": "val macroF1", "encoder": "resnet50 SSL4EO-S12 MoCo", "inputs": "11 bands (reflectance) + NDWI, red/green, nir/red, nir/green, swir1/nir"}

Selected checkpoint: epoch 79 of 80 (ema weights), highest val macro F1 with TTA 0.8489. Test-time augmentation: True. VSCP: True.

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
| overall accuracy | 0.9234 |
| balanced accuracy (macro recall) | 0.8372 |
| Cohen's kappa | 0.7196 |
| macroPrec | 0.8307 |
| macroRec | 0.8372 |
| macroF1 | 0.8200 |
| IoU | 0.7220 |
| microPrec | 0.9234 |
| microRec | 0.9234 |
| microF1 | 0.9234 |
| weightPrec | 0.9230 |
| weightRec | 0.9234 |
| weightF1 | 0.9179 |

Annotated pixels: 361,147. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 294,020 | 81.41 | 312,958 | 290,063 | 22,895 | 3,957 | 44,232 | 0.9865 | 0.9256 | 0.9268 | 0.9865 | 0.9558 | 0.9153 | 0.6589 | 0.08 |
| Marine Debris | 973 | 0.27 | 1,502 | 756 | 746 | 217 | 359,428 | 0.7770 | 0.9973 | 0.5033 | 0.7770 | 0.6109 | 0.4398 | 0.9979 | 0.99 |
| Algae/Sargassum | 3,089 | 0.86 | 3,000 | 2,952 | 48 | 137 | 358,010 | 0.9556 | 0.9995 | 0.9840 | 0.9556 | 0.9696 | 0.9410 | 0.9999 | 0.02 |
| Other | 63,065 | 17.46 | 43,687 | 39,698 | 3,989 | 23,367 | 294,093 | 0.6295 | 0.9243 | 0.9087 | 0.6295 | 0.7437 | 0.5920 | 0.9866 | 0.10 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 290,063 | 123 | 19 | 3,815 | 294,020 |
| Marine Debris | 23 | 756 | 20 | 174 | 973 |
| Algae/Sargassum | 111 | 26 | 2,952 | 0 | 3,089 |
| Other | 22,761 | 597 | 9 | 39,698 | 63,065 |
| total predicted | 312,958 | 1,502 | 3,000 | 43,687 | 361,147 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 98.65 | 0.04 | 0.01 | 1.30 |
| Marine Debris | 2.36 | 77.70 | 2.06 | 17.88 |
| Algae/Sargassum | 3.59 | 0.84 | 95.56 | 0.00 |
| Other | 36.09 | 0.95 | 0.01 | 62.95 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 92.68 | 8.19 | 0.63 | 8.73 |
| Marine Debris | 0.01 | 50.33 | 0.67 | 0.40 |
| Algae/Sargassum | 0.04 | 1.73 | 98.40 | 0.00 |
| Other | 7.27 | 39.75 | 0.30 | 90.87 |

### Where the errors go

- **Water**: 3,957 missed pixels went to: Other 3,815 (96.4%), Marine Debris 123 (3.1%), Algae/Sargassum 19 (0.5%).
  22,895 false-positive pixels came from: Other 22,761 (99.4%), Algae/Sargassum 111 (0.5%), Marine Debris 23 (0.1%).
- **Marine Debris**: 217 missed pixels went to: Other 174 (80.2%), Water 23 (10.6%), Algae/Sargassum 20 (9.2%).
  746 false-positive pixels came from: Other 597 (80.0%), Water 123 (16.5%), Algae/Sargassum 26 (3.5%).
- **Algae/Sargassum**: 137 missed pixels went to: Water 111 (81.0%), Marine Debris 26 (19.0%).
  48 false-positive pixels came from: Marine Debris 20 (41.7%), Water 19 (39.6%), Other 9 (18.8%).
- **Other**: 23,367 missed pixels went to: Water 22,761 (97.4%), Marine Debris 597 (2.6%), Algae/Sargassum 9 (0.0%).
  3,989 false-positive pixels came from: Water 3,815 (95.6%), Marine Debris 174 (4.4%).

## model_best_val_f1, val split

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9300 |
| balanced accuracy (macro recall) | 0.8493 |
| Cohen's kappa | 0.6949 |
| macroPrec | 0.8809 |
| macroRec | 0.8493 |
| macroF1 | 0.8489 |
| IoU | 0.7571 |
| microPrec | 0.9300 |
| microRec | 0.9300 |
| microF1 | 0.9300 |
| weightPrec | 0.9320 |
| weightRec | 0.9300 |
| weightF1 | 0.9225 |

Annotated pixels: 358,121. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 300,952 | 84.04 | 322,617 | 299,560 | 23,057 | 1,392 | 34,112 | 0.9954 | 0.9317 | 0.9285 | 0.9954 | 0.9608 | 0.9245 | 0.5967 | 0.08 |
| Marine Debris | 1,162 | 0.32 | 1,532 | 1,002 | 530 | 160 | 356,429 | 0.8623 | 0.9981 | 0.6540 | 0.8623 | 0.7439 | 0.5922 | 0.9985 | 0.53 |
| Algae/Sargassum | 1,794 | 0.50 | 1,769 | 1,744 | 25 | 50 | 356,302 | 0.9721 | 0.9998 | 0.9859 | 0.9721 | 0.9790 | 0.9588 | 0.9999 | 0.01 |
| Other | 54,213 | 15.14 | 32,203 | 30,763 | 1,440 | 23,450 | 302,468 | 0.5674 | 0.9305 | 0.9553 | 0.5674 | 0.7120 | 0.5528 | 0.9953 | 0.05 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 299,560 | 70 | 13 | 1,309 | 300,952 |
| Marine Debris | 18 | 1,002 | 12 | 130 | 1,162 |
| Algae/Sargassum | 10 | 39 | 1,744 | 1 | 1,794 |
| Other | 23,029 | 421 | 0 | 30,763 | 54,213 |
| total predicted | 322,617 | 1,532 | 1,769 | 32,203 | 358,121 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.54 | 0.02 | 0.00 | 0.43 |
| Marine Debris | 1.55 | 86.23 | 1.03 | 11.19 |
| Algae/Sargassum | 0.56 | 2.17 | 97.21 | 0.06 |
| Other | 42.48 | 0.78 | 0.00 | 56.74 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 92.85 | 4.57 | 0.73 | 4.06 |
| Marine Debris | 0.01 | 65.40 | 0.68 | 0.40 |
| Algae/Sargassum | 0.00 | 2.55 | 98.59 | 0.00 |
| Other | 7.14 | 27.48 | 0.00 | 95.53 |

### Where the errors go

- **Water**: 1,392 missed pixels went to: Other 1,309 (94.0%), Marine Debris 70 (5.0%), Algae/Sargassum 13 (0.9%).
  23,057 false-positive pixels came from: Other 23,029 (99.9%), Marine Debris 18 (0.1%), Algae/Sargassum 10 (0.0%).
- **Marine Debris**: 160 missed pixels went to: Other 130 (81.2%), Water 18 (11.2%), Algae/Sargassum 12 (7.5%).
  530 false-positive pixels came from: Other 421 (79.4%), Water 70 (13.2%), Algae/Sargassum 39 (7.4%).
- **Algae/Sargassum**: 50 missed pixels went to: Marine Debris 39 (78.0%), Water 10 (20.0%), Other 1 (2.0%).
  25 false-positive pixels came from: Water 13 (52.0%), Marine Debris 12 (48.0%).
- **Other**: 23,450 missed pixels went to: Water 23,029 (98.2%), Marine Debris 421 (1.8%).
  1,440 false-positive pixels came from: Water 1,309 (90.9%), Marine Debris 130 (9.0%), Algae/Sargassum 1 (0.1%).

## model_last, test split

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9231 |
| balanced accuracy (macro recall) | 0.8366 |
| Cohen's kappa | 0.7182 |
| macroPrec | 0.8310 |
| macroRec | 0.8366 |
| macroF1 | 0.8199 |
| IoU | 0.7218 |
| microPrec | 0.9231 |
| microRec | 0.9231 |
| microF1 | 0.9231 |
| weightPrec | 0.9227 |
| weightRec | 0.9231 |
| weightF1 | 0.9175 |

Annotated pixels: 361,147. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 294,020 | 81.41 | 313,123 | 290,089 | 23,034 | 3,931 | 44,093 | 0.9866 | 0.9253 | 0.9264 | 0.9866 | 0.9556 | 0.9150 | 0.6569 | 0.08 |
| Marine Debris | 973 | 0.27 | 1,498 | 756 | 742 | 217 | 359,432 | 0.7770 | 0.9973 | 0.5047 | 0.7770 | 0.6119 | 0.4408 | 0.9979 | 0.98 |
| Algae/Sargassum | 3,089 | 0.86 | 3,000 | 2,952 | 48 | 137 | 358,010 | 0.9556 | 0.9995 | 0.9840 | 0.9556 | 0.9696 | 0.9410 | 0.9999 | 0.02 |
| Other | 63,065 | 17.46 | 43,526 | 39,562 | 3,964 | 23,503 | 294,118 | 0.6273 | 0.9239 | 0.9089 | 0.6273 | 0.7423 | 0.5902 | 0.9867 | 0.10 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 290,089 | 122 | 19 | 3,790 | 294,020 |
| Marine Debris | 23 | 756 | 20 | 174 | 973 |
| Algae/Sargassum | 111 | 26 | 2,952 | 0 | 3,089 |
| Other | 22,900 | 594 | 9 | 39,562 | 63,065 |
| total predicted | 313,123 | 1,498 | 3,000 | 43,526 | 361,147 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 98.66 | 0.04 | 0.01 | 1.29 |
| Marine Debris | 2.36 | 77.70 | 2.06 | 17.88 |
| Algae/Sargassum | 3.59 | 0.84 | 95.56 | 0.00 |
| Other | 36.31 | 0.94 | 0.01 | 62.73 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 92.64 | 8.14 | 0.63 | 8.71 |
| Marine Debris | 0.01 | 50.47 | 0.67 | 0.40 |
| Algae/Sargassum | 0.04 | 1.74 | 98.40 | 0.00 |
| Other | 7.31 | 39.65 | 0.30 | 90.89 |

### Where the errors go

- **Water**: 3,931 missed pixels went to: Other 3,790 (96.4%), Marine Debris 122 (3.1%), Algae/Sargassum 19 (0.5%).
  23,034 false-positive pixels came from: Other 22,900 (99.4%), Algae/Sargassum 111 (0.5%), Marine Debris 23 (0.1%).
- **Marine Debris**: 217 missed pixels went to: Other 174 (80.2%), Water 23 (10.6%), Algae/Sargassum 20 (9.2%).
  742 false-positive pixels came from: Other 594 (80.1%), Water 122 (16.4%), Algae/Sargassum 26 (3.5%).
- **Algae/Sargassum**: 137 missed pixels went to: Water 111 (81.0%), Marine Debris 26 (19.0%).
  48 false-positive pixels came from: Marine Debris 20 (41.7%), Water 19 (39.6%), Other 9 (18.8%).
- **Other**: 23,503 missed pixels went to: Water 22,900 (97.4%), Marine Debris 594 (2.5%), Algae/Sargassum 9 (0.0%).
  3,964 false-positive pixels came from: Water 3,790 (95.6%), Marine Debris 174 (4.4%).

## Validation vs test, per class (model_best_val_f1)

| class | precision val | recall val | F1 val | IoU val | precision test | recall test | F1 test | IoU test | F1 test - val |
|---|---|---|---|---|---|---|---|---|---|
| Water | 0.9285 | 0.9954 | 0.9608 | 0.9245 | 0.9268 | 0.9865 | 0.9558 | 0.9153 | -0.0050 |
| Marine Debris | 0.6540 | 0.8623 | 0.7439 | 0.5922 | 0.5033 | 0.7770 | 0.6109 | 0.4398 | -0.1330 |
| Algae/Sargassum | 0.9859 | 0.9721 | 0.9790 | 0.9588 | 0.9840 | 0.9556 | 0.9696 | 0.9410 | -0.0093 |
| Other | 0.9553 | 0.5674 | 0.7120 | 0.5528 | 0.9087 | 0.6295 | 0.7437 | 0.5920 | +0.0318 |

Large negative F1 test - val means the class generalises worse to the test scenes than validation suggested.
