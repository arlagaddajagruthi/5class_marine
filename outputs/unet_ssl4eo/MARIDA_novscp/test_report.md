# U-Net (ResNet-50, SSL4EO-S12 pretrained), MARIDA, run MARIDA_novscp, 4 classes (annotated pixels only)

Recipe: {"epochs": 45, "batch": 5, "lr": 0.0002, "decay": 0.0, "lr_steps": [40], "gamma": 0.1, "weight_param": 1.03, "aug": "marida", "vscp": false, "ema": false, "tta": true, "drop_last": false, "select": "val macroF1", "encoder": "resnet50 SSL4EO-S12 MoCo", "inputs": "11 bands (reflectance) + NDWI, red/green, nir/red, nir/green, swir1/nir"}

Selected checkpoint: epoch 43 of 45 (model weights), highest val macro F1 with TTA 0.8961. Test-time augmentation: True. VSCP: False.

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
| overall accuracy | 0.9941 |
| balanced accuracy (macro recall) | 0.9093 |
| Cohen's kappa | 0.8755 |
| macroPrec | 0.8216 |
| macroRec | 0.9093 |
| macroF1 | 0.8576 |
| IoU | 0.7788 |
| microPrec | 0.9941 |
| microRec | 0.9941 |
| microF1 | 0.9941 |
| weightPrec | 0.9949 |
| weightRec | 0.9941 |
| weightF1 | 0.9944 |

Annotated pixels: 158,371. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 154,739 | 97.71 | 154,391 | 154,166 | 225 | 573 | 3,407 | 0.9963 | 0.9950 | 0.9985 | 0.9963 | 0.9974 | 0.9949 | 0.9381 | 0.00 |
| Marine Debris | 381 | 0.24 | 583 | 313 | 270 | 68 | 157,720 | 0.8215 | 0.9979 | 0.5369 | 0.8215 | 0.6494 | 0.4808 | 0.9983 | 0.86 |
| Algae/Sargassum | 1,641 | 1.04 | 1,638 | 1,627 | 11 | 14 | 156,719 | 0.9915 | 0.9998 | 0.9933 | 0.9915 | 0.9924 | 0.9849 | 0.9999 | 0.01 |
| Other | 1,610 | 1.02 | 1,759 | 1,333 | 426 | 277 | 156,335 | 0.8280 | 0.9956 | 0.7578 | 0.8280 | 0.7913 | 0.6547 | 0.9973 | 0.32 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 154,166 | 162 | 4 | 407 | 154,739 |
| Marine Debris | 47 | 313 | 2 | 19 | 381 |
| Algae/Sargassum | 10 | 4 | 1,627 | 0 | 1,641 |
| Other | 168 | 104 | 5 | 1,333 | 1,610 |
| total predicted | 154,391 | 583 | 1,638 | 1,759 | 158,371 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.63 | 0.10 | 0.00 | 0.26 |
| Marine Debris | 12.34 | 82.15 | 0.52 | 4.99 |
| Algae/Sargassum | 0.61 | 0.24 | 99.15 | 0.00 |
| Other | 10.43 | 6.46 | 0.31 | 82.80 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.85 | 27.79 | 0.24 | 23.14 |
| Marine Debris | 0.03 | 53.69 | 0.12 | 1.08 |
| Algae/Sargassum | 0.01 | 0.69 | 99.33 | 0.00 |
| Other | 0.11 | 17.84 | 0.31 | 75.78 |

### Where the errors go

- **Water**: 573 missed pixels went to: Other 407 (71.0%), Marine Debris 162 (28.3%), Algae/Sargassum 4 (0.7%).
  225 false-positive pixels came from: Other 168 (74.7%), Marine Debris 47 (20.9%), Algae/Sargassum 10 (4.4%).
- **Marine Debris**: 68 missed pixels went to: Water 47 (69.1%), Other 19 (27.9%), Algae/Sargassum 2 (2.9%).
  270 false-positive pixels came from: Water 162 (60.0%), Other 104 (38.5%), Algae/Sargassum 4 (1.5%).
- **Algae/Sargassum**: 14 missed pixels went to: Water 10 (71.4%), Marine Debris 4 (28.6%).
  11 false-positive pixels came from: Other 5 (45.5%), Water 4 (36.4%), Marine Debris 2 (18.2%).
- **Other**: 277 missed pixels went to: Water 168 (60.6%), Marine Debris 104 (37.5%), Algae/Sargassum 5 (1.8%).
  426 false-positive pixels came from: Water 407 (95.5%), Marine Debris 19 (4.5%).

## model_best_val_f1, val split

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9955 |
| balanced accuracy (macro recall) | 0.9145 |
| Cohen's kappa | 0.9033 |
| macroPrec | 0.8808 |
| macroRec | 0.9145 |
| macroF1 | 0.8961 |
| IoU | 0.8237 |
| microPrec | 0.9955 |
| microRec | 0.9955 |
| microF1 | 0.9955 |
| weightPrec | 0.9959 |
| weightRec | 0.9955 |
| weightF1 | 0.9956 |

Annotated pixels: 191,436. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 187,008 | 97.69 | 186,771 | 186,627 | 144 | 381 | 4,284 | 0.9980 | 0.9973 | 0.9992 | 0.9980 | 0.9986 | 0.9972 | 0.9675 | 0.00 |
| Marine Debris | 1,075 | 0.56 | 1,264 | 906 | 358 | 169 | 190,003 | 0.8428 | 0.9972 | 0.7168 | 0.8428 | 0.7747 | 0.6322 | 0.9981 | 0.40 |
| Algae/Sargassum | 1,552 | 0.81 | 1,485 | 1,464 | 21 | 88 | 189,863 | 0.9433 | 0.9994 | 0.9859 | 0.9433 | 0.9641 | 0.9307 | 0.9999 | 0.01 |
| Other | 1,801 | 0.94 | 1,916 | 1,574 | 342 | 227 | 189,293 | 0.8740 | 0.9970 | 0.8215 | 0.8740 | 0.8469 | 0.7345 | 0.9982 | 0.22 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 186,627 | 140 | 6 | 235 | 187,008 |
| Marine Debris | 58 | 906 | 15 | 96 | 1,075 |
| Algae/Sargassum | 22 | 55 | 1,464 | 11 | 1,552 |
| Other | 64 | 163 | 0 | 1,574 | 1,801 |
| total predicted | 186,771 | 1,264 | 1,485 | 1,916 | 191,436 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.80 | 0.07 | 0.00 | 0.13 |
| Marine Debris | 5.40 | 84.28 | 1.40 | 8.93 |
| Algae/Sargassum | 1.42 | 3.54 | 94.33 | 0.71 |
| Other | 3.55 | 9.05 | 0.00 | 87.40 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.92 | 11.08 | 0.40 | 12.27 |
| Marine Debris | 0.03 | 71.68 | 1.01 | 5.01 |
| Algae/Sargassum | 0.01 | 4.35 | 98.59 | 0.57 |
| Other | 0.03 | 12.90 | 0.00 | 82.15 |

### Where the errors go

- **Water**: 381 missed pixels went to: Other 235 (61.7%), Marine Debris 140 (36.7%), Algae/Sargassum 6 (1.6%).
  144 false-positive pixels came from: Other 64 (44.4%), Marine Debris 58 (40.3%), Algae/Sargassum 22 (15.3%).
- **Marine Debris**: 169 missed pixels went to: Other 96 (56.8%), Water 58 (34.3%), Algae/Sargassum 15 (8.9%).
  358 false-positive pixels came from: Other 163 (45.5%), Water 140 (39.1%), Algae/Sargassum 55 (15.4%).
- **Algae/Sargassum**: 88 missed pixels went to: Marine Debris 55 (62.5%), Water 22 (25.0%), Other 11 (12.5%).
  21 false-positive pixels came from: Marine Debris 15 (71.4%), Water 6 (28.6%).
- **Other**: 227 missed pixels went to: Marine Debris 163 (71.8%), Water 64 (28.2%).
  342 false-positive pixels came from: Water 235 (68.7%), Marine Debris 96 (28.1%), Algae/Sargassum 11 (3.2%).

## model_last, test split

### Overall

| metric | value |
|---|---|
| overall accuracy | 0.9941 |
| balanced accuracy (macro recall) | 0.9107 |
| Cohen's kappa | 0.8750 |
| macroPrec | 0.8090 |
| macroRec | 0.9107 |
| macroF1 | 0.8412 |
| IoU | 0.7637 |
| microPrec | 0.9941 |
| microRec | 0.9941 |
| microF1 | 0.9941 |
| weightPrec | 0.9955 |
| weightRec | 0.9941 |
| weightF1 | 0.9946 |

Annotated pixels: 158,371. Macro scores weight every class equally; micro/weighted scores are dominated by Water.

### Per class

| class | support | share_% | predicted | TP | FP | FN | TN | class_acc | OvR_acc | precision | recall | F1 | IoU | specificity | FP_per_TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 154,739 | 97.71 | 154,396 | 154,196 | 200 | 543 | 3,432 | 0.9965 | 0.9953 | 0.9987 | 0.9965 | 0.9976 | 0.9952 | 0.9449 | 0.00 |
| Marine Debris | 381 | 0.24 | 777 | 326 | 451 | 55 | 157,539 | 0.8556 | 0.9968 | 0.4196 | 0.8556 | 0.5630 | 0.3918 | 0.9971 | 1.38 |
| Algae/Sargassum | 1,641 | 1.04 | 1,644 | 1,629 | 15 | 12 | 156,715 | 0.9927 | 0.9998 | 0.9909 | 0.9927 | 0.9918 | 0.9837 | 0.9999 | 0.01 |
| Other | 1,610 | 1.02 | 1,554 | 1,285 | 269 | 325 | 156,492 | 0.7981 | 0.9962 | 0.8269 | 0.7981 | 0.8123 | 0.6839 | 0.9983 | 0.21 |

class_acc = TP / support: share of that class's pixels classified correctly (per-class accuracy, = recall = producer's accuracy; their mean is the balanced accuracy).
OvR_acc = (TP + TN) / all pixels: one-vs-rest accuracy of the class as a yes/no task (inflated by the many pixels of other classes; shown for completeness).
FP_per_TP = false positives per correct pixel (above 1 means most predictions of that class are wrong).

### Confusion matrix, pixel counts (rows true, columns predicted)

| class | Water | Marine Debris | Algae/Sargassum | Other | total true |
|---|---|---|---|---|---|
| Water | 154,196 | 274 | 5 | 264 | 154,739 |
| Marine Debris | 45 | 326 | 5 | 5 | 381 |
| Algae/Sargassum | 9 | 3 | 1,629 | 0 | 1,641 |
| Other | 146 | 174 | 5 | 1,285 | 1,610 |
| total predicted | 154,396 | 777 | 1,644 | 1,554 | 158,371 |

### Confusion matrix, % of each TRUE class (row-normalised; diagonal = recall)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.65 | 0.18 | 0.00 | 0.17 |
| Marine Debris | 11.81 | 85.56 | 1.31 | 1.31 |
| Algae/Sargassum | 0.55 | 0.18 | 99.27 | 0.00 |
| Other | 9.07 | 10.81 | 0.31 | 79.81 |

### Confusion matrix, % of each PREDICTED class (column-normalised; diagonal = precision)

| class | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.87 | 35.26 | 0.30 | 16.99 |
| Marine Debris | 0.03 | 41.96 | 0.30 | 0.32 |
| Algae/Sargassum | 0.01 | 0.39 | 99.09 | 0.00 |
| Other | 0.09 | 22.39 | 0.30 | 82.69 |

### Where the errors go

- **Water**: 543 missed pixels went to: Marine Debris 274 (50.5%), Other 264 (48.6%), Algae/Sargassum 5 (0.9%).
  200 false-positive pixels came from: Other 146 (73.0%), Marine Debris 45 (22.5%), Algae/Sargassum 9 (4.5%).
- **Marine Debris**: 55 missed pixels went to: Water 45 (81.8%), Algae/Sargassum 5 (9.1%), Other 5 (9.1%).
  451 false-positive pixels came from: Water 274 (60.8%), Other 174 (38.6%), Algae/Sargassum 3 (0.7%).
- **Algae/Sargassum**: 12 missed pixels went to: Water 9 (75.0%), Marine Debris 3 (25.0%).
  15 false-positive pixels came from: Water 5 (33.3%), Marine Debris 5 (33.3%), Other 5 (33.3%).
- **Other**: 325 missed pixels went to: Marine Debris 174 (53.5%), Water 146 (44.9%), Algae/Sargassum 5 (1.5%).
  269 false-positive pixels came from: Water 264 (98.1%), Marine Debris 5 (1.9%).

## Validation vs test, per class (model_best_val_f1)

| class | precision val | recall val | F1 val | IoU val | precision test | recall test | F1 test | IoU test | F1 test - val |
|---|---|---|---|---|---|---|---|---|---|
| Water | 0.9992 | 0.9980 | 0.9986 | 0.9972 | 0.9985 | 0.9963 | 0.9974 | 0.9949 | -0.0012 |
| Marine Debris | 0.7168 | 0.8428 | 0.7747 | 0.6322 | 0.5369 | 0.8215 | 0.6494 | 0.4808 | -0.1253 |
| Algae/Sargassum | 0.9859 | 0.9433 | 0.9641 | 0.9307 | 0.9933 | 0.9915 | 0.9924 | 0.9849 | +0.0283 |
| Other | 0.8215 | 0.8740 | 0.8469 | 0.7345 | 0.7578 | 0.8280 | 0.7913 | 0.6547 | -0.0556 |

Large negative F1 test - val means the class generalises worse to the test scenes than validation suggested.
