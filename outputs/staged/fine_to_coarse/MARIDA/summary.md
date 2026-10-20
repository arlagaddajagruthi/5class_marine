# Fine-to-coarse training, MARIDA (Random Forest, stage-5 features)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir

All variants are scored on the same 4-class test labels (annotated pixels of the official test split). Fine models predict an original class, which is then merged to the 4 classes.

- coarse: trained on the 4 merged classes, log weights (identical to stage 5, log weights)
- fine_logw: trained on the original classes, log weights over the original classes
- fine_coarsew: trained on the original classes, each pixel weighted as its merged class

## Overall

| variant | train classes | macro F1 | IoU | precision | recall | accuracy | kappa |
|---|---|---|---|---|---|---|---|
| coarse | 4 | 0.9042 | 0.8373 | 0.8865 | 0.9270 | 0.9956 | 0.9027 |
| fine_logw | 13 | 0.8904 | 0.8182 | 0.8531 | 0.9426 | 0.9950 | 0.8937 |
| fine_coarsew | 13 | 0.8966 | 0.8266 | 0.8638 | 0.9405 | 0.9952 | 0.8977 |

## Per-class accuracy (TP / class pixels = recall)

| variant | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| coarse | 0.9978 | 0.9134 | 0.9799 | 0.8168 |
| fine_logw | 0.9968 (-0.0010) | 0.9396 (+0.0262) | 0.9811 (+0.0012) | 0.8528 (+0.0360) |
| fine_coarsew | 0.9971 (-0.0007) | 0.9396 (+0.0262) | 0.9854 (+0.0055) | 0.8398 (+0.0230) |

## Per-class precision

| variant | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| coarse | 0.9982 | 0.7326 | 0.9969 | 0.8183 |
| fine_logw | 0.9988 (+0.0006) | 0.6427 (-0.0899) | 0.9920 (-0.0049) | 0.7788 (-0.0395) |
| fine_coarsew | 0.9987 (+0.0004) | 0.6767 (-0.0559) | 0.9902 (-0.0067) | 0.7897 (-0.0286) |

## Per-class recall

| variant | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| coarse | 0.9978 | 0.9134 | 0.9799 | 0.8168 |
| fine_logw | 0.9968 (-0.0010) | 0.9396 (+0.0262) | 0.9811 (+0.0012) | 0.8528 (+0.0360) |
| fine_coarsew | 0.9971 (-0.0007) | 0.9396 (+0.0262) | 0.9854 (+0.0055) | 0.8398 (+0.0230) |

## Per-class F1

| variant | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| coarse | 0.9980 | 0.8131 | 0.9883 | 0.8175 |
| fine_logw | 0.9978 (-0.0002) | 0.7633 (-0.0498) | 0.9865 (-0.0018) | 0.8141 (-0.0034) |
| fine_coarsew | 0.9979 (-0.0001) | 0.7868 (-0.0263) | 0.9878 (-0.0005) | 0.8140 (-0.0036) |

## Per-class IoU

| variant | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| coarse | 0.9960 | 0.6850 | 0.9769 | 0.6914 |
| fine_logw | 0.9955 (-0.0005) | 0.6172 (-0.0678) | 0.9734 (-0.0035) | 0.6865 (-0.0049) |
| fine_coarsew | 0.9958 (-0.0003) | 0.6486 (-0.0365) | 0.9759 (-0.0011) | 0.6863 (-0.0051) |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| variant | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| coarse | 0.9961 | 0.9990 | 0.9998 | 0.9963 |
| fine_logw | 0.9956 (-0.0005) | 0.9986 (-0.0004) | 0.9997 (-0.0000) | 0.9960 (-0.0003) |
| fine_coarsew | 0.9958 (-0.0003) | 0.9988 (-0.0002) | 0.9997 (-0.0000) | 0.9961 (-0.0002) |

Values in brackets: change from the coarse model.

## Original classes in the test split: % of pixels given the correct merged class

| original class | merged into | test px | coarse | fine_logw | fine_coarsew | coarse model's main error |
|---|---|---|---|---|---|---|
| Marine Debris | Marine Debris | 381 | 91.3 | 94.0 | 94.0 | Water 7.6% |
| Dense Sargassum | Algae/Sargassum | 760 | 99.3 | 100.0 | 100.0 | Other 0.7% |
| Sparse Sargassum | Algae/Sargassum | 881 | 96.8 | 96.5 | 97.3 | Water 1.6% |
| Natural Organic Material | Other | 49 | 57.1 | 55.1 | 46.9 | Marine Debris 32.7% |
| Ship | Other | 1,174 | 81.4 | 84.3 | 83.4 | Water 14.7% |
| Marine Water | Water | 23,443 | 100.0 | 100.0 | 100.0 | Algae/Sargassum 0.0% |
| Sediment-Laden Water | Water | 93,037 | 100.0 | 100.0 | 100.0 | Other 0.0% |
| Foam | Other | 387 | 85.5 | 92.0 | 90.4 | Water 14.5% |
| Turbid Water | Water | 32,226 | 99.9 | 99.9 | 99.9 | Other 0.1% |
| Shallow Water | Water | 2,506 | 99.8 | 99.8 | 99.8 | Other 0.2% |
| Waves | Water | 1,865 | 89.3 | 84.0 | 85.7 | Other 8.6% |
| Wakes | Water | 1,570 | 95.1 | 93.6 | 93.6 | Other 3.9% |
| Mixed Water | Water | 92 | 94.6 | 88.0 | 84.8 | Marine Debris 4.3% |
