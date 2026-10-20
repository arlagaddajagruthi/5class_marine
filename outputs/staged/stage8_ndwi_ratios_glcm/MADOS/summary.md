# stage8_ndwi_ratios_glcm: bands + NDWI + 4 band ratios + 6 GLCM texture features (MADOS)

Features (22): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm

Annotated classes only (1-4). Macro scores over the 4 classes; test = all annotated pixels of the official test split.

## Overall

| Model | F1 | IoU | Precision | Recall | Acc | Kappa | train px | train s |
|---|---|---|---|---|---|---|---|---|
| rf | 0.7695 | 0.6376 | 0.7846 | 0.7729 | 0.8529 | 0.5631 | 761,887 | 148 |
| xgb | 0.6818 | 0.5478 | 0.6345 | 0.8311 | 0.8496 | 0.5766 | 761,887 | 18 |
| lgbm | 0.7000 | 0.5656 | 0.6596 | 0.8273 | 0.8478 | 0.5738 | 761,887 | 13 |
| ada | 0.5823 | 0.4621 | 0.5479 | 0.8086 | 0.8122 | 0.4889 | 761,887 | 1294 |
| svm | 0.6702 | 0.5327 | 0.6311 | 0.7764 | 0.7740 | 0.4117 | 761,887 | 17 |

## Per-class accuracy (TP / class pixels = recall)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8827 | 0.7729 | 0.7138 | 0.7224 |
| xgb | 0.8659 | 0.9044 | 0.7779 | 0.7761 |
| lgbm | 0.8625 | 0.8931 | 0.7711 | 0.7825 |
| ada | 0.8396 | 0.9198 | 0.7912 | 0.6839 |
| svm | 0.7905 | 0.7790 | 0.8420 | 0.6941 |

## Per-class precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9372 | 0.6533 | 0.9844 | 0.5635 |
| xgb | 0.9527 | 0.2515 | 0.7732 | 0.5605 |
| lgbm | 0.9533 | 0.2956 | 0.8370 | 0.5524 |
| ada | 0.9385 | 0.0700 | 0.6545 | 0.5285 |
| svm | 0.9273 | 0.3237 | 0.8570 | 0.4164 |

## Per-class recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8827 | 0.7729 | 0.7138 | 0.7224 |
| xgb | 0.8659 | 0.9044 | 0.7779 | 0.7761 |
| lgbm | 0.8625 | 0.8931 | 0.7711 | 0.7825 |
| ada | 0.8396 | 0.9198 | 0.7912 | 0.6839 |
| svm | 0.7905 | 0.7790 | 0.8420 | 0.6941 |

## Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.9091 | 0.7081 | 0.8275 | 0.6332 |
| xgb | 0.9072 | 0.3936 | 0.7755 | 0.6509 |
| lgbm | 0.9056 | 0.4442 | 0.8027 | 0.6476 |
| ada | 0.8863 | 0.1301 | 0.7164 | 0.5963 |
| svm | 0.8534 | 0.4573 | 0.8494 | 0.5205 |

## Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8334 | 0.5481 | 0.7058 | 0.4632 |
| xgb | 0.8302 | 0.2450 | 0.6334 | 0.4825 |
| lgbm | 0.8275 | 0.2855 | 0.6704 | 0.4789 |
| ada | 0.7958 | 0.0696 | 0.5581 | 0.4248 |
| svm | 0.7443 | 0.2964 | 0.7383 | 0.3518 |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| rf | 0.8563 | 0.9983 | 0.9975 | 0.8538 |
| xgb | 0.8558 | 0.9925 | 0.9961 | 0.8546 |
| lgbm | 0.8536 | 0.9940 | 0.9968 | 0.8513 |
| ada | 0.8246 | 0.9669 | 0.9946 | 0.8383 |
| svm | 0.7789 | 0.9950 | 0.9974 | 0.7767 |
