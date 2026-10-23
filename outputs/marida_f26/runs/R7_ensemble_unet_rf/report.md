# Run R7_ensemble_unet_rf: U-Net (R3_unet_scratch_std) + Random Forest (R6_rf) ensemble, MARIDA, 4 classes

Inputs: 26 channels (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, NDWI, FAI, FDI, NDMI, red/green, nir/red, nir/green, swir1/nir, glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_asm). See `architecture.md` for the model and how the features are fed in.

p = w · p_U-Net + (1 − w) · p_RF with w = 0.40, chosen by validation macro F1 over w = 0, 0.05, …, 1 (`weight_curve.csv`).

![weight curve](figures/weight_curve.png)

## Test split (158,371 annotated pixels)

### Overall

| metric | value |
|---|---|
| OA | 0.9965 |
| balanced_accuracy | 0.9519 |
| kappa | 0.9245 |
| macro_precision | 0.8919 |
| macro_recall | 0.9519 |
| macro_F1 | 0.9186 |
| mIoU | 0.8590 |
| weighted_precision | 0.9968 |
| weighted_recall | 0.9965 |
| weighted_F1 | 0.9966 |
| micro_F1 | 0.9965 |
| macro_AUPRC | 0.9615 |

### Per class

| | support | share_% | predicted | TP | FP | FN | TN | precision | recall (class accuracy) | F1 | IoU | specificity | FP_per_TP | AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 154739.0000 | 97.7067 | 154557.0000 | 154412.0000 | 145.0000 | 327.0000 | 3487.0000 | 0.9991 | 0.9979 | 0.9985 | 0.9970 | 0.9601 | 0.0009 | 1.0000 |
| Marine Debris | 381.0000 | 0.2406 | 487.0000 | 356.0000 | 131.0000 | 25.0000 | 157859.0000 | 0.7310 | 0.9344 | 0.8203 | 0.6953 | 0.9992 | 0.3680 | 0.9068 |
| Algae/Sargassum | 1641.0000 | 1.0362 | 1622.0000 | 1617.0000 | 5.0000 | 24.0000 | 156725.0000 | 0.9969 | 0.9854 | 0.9911 | 0.9824 | 1.0000 | 0.0031 | 0.9996 |
| Other | 1610.0000 | 1.0166 | 1705.0000 | 1433.0000 | 272.0000 | 177.0000 | 156489.0000 | 0.8405 | 0.8901 | 0.8646 | 0.7614 | 0.9983 | 0.1898 | 0.9396 |

### Confusion matrix (rows true, columns predicted): counts / % of true class / % of predicted class

| | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 154,412 | 54 | 5 | 268 |
| Marine Debris | 25 | 356 | 0 | 0 |
| Algae/Sargassum | 6 | 14 | 1,617 | 4 |
| Other | 114 | 63 | 0 | 1,433 |

| | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.79 | 0.03 | 0.00 | 0.17 |
| Marine Debris | 6.56 | 93.44 | 0.00 | 0.00 |
| Algae/Sargassum | 0.37 | 0.85 | 98.54 | 0.24 |
| Other | 7.08 | 3.91 | 0.00 | 89.01 |

| | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.91 | 11.09 | 0.31 | 15.72 |
| Marine Debris | 0.02 | 73.10 | 0.00 | 0.00 |
| Algae/Sargassum | 0.00 | 2.87 | 99.69 | 0.23 |
| Other | 0.07 | 12.94 | 0.00 | 84.05 |

![confusion](figures/confusion_test.png)

![per-class](figures/classwise_test.png)

![PR curves](figures/pr_curves_test.png)

### Per scene

`per_scene_test.csv` has every scene; the figure shows macro F1 (classes present) and debris F1.

![per scene](figures/per_scene_test.png)

## Val split (191,436 annotated pixels)

### Overall

| metric | value |
|---|---|
| OA | 0.9971 |
| balanced_accuracy | 0.9265 |
| kappa | 0.9354 |
| macro_precision | 0.9384 |
| macro_recall | 0.9265 |
| macro_F1 | 0.9322 |
| mIoU | 0.8781 |
| weighted_precision | 0.9971 |
| weighted_recall | 0.9971 |
| weighted_F1 | 0.9971 |
| micro_F1 | 0.9971 |
| macro_AUPRC | 0.9586 |

### Per class

| | support | share_% | predicted | TP | FP | FN | TN | precision | recall (class accuracy) | F1 | IoU | specificity | FP_per_TP | AUPRC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Water | 187008.0000 | 97.6870 | 187092.0000 | 186874.0000 | 218.0000 | 134.0000 | 4210.0000 | 0.9988 | 0.9993 | 0.9991 | 0.9981 | 0.9508 | 0.0012 | 1.0000 |
| Marine Debris | 1075.0000 | 0.5615 | 1093.0000 | 940.0000 | 153.0000 | 135.0000 | 190208.0000 | 0.8600 | 0.8744 | 0.8672 | 0.7655 | 0.9992 | 0.1628 | 0.9096 |
| Algae/Sargassum | 1552.0000 | 0.8107 | 1473.0000 | 1471.0000 | 2.0000 | 81.0000 | 189882.0000 | 0.9986 | 0.9478 | 0.9726 | 0.9466 | 1.0000 | 0.0014 | 0.9977 |
| Other | 1801.0000 | 0.9408 | 1778.0000 | 1593.0000 | 185.0000 | 208.0000 | 189450.0000 | 0.8960 | 0.8845 | 0.8902 | 0.8021 | 0.9990 | 0.1161 | 0.9272 |

### Confusion matrix (rows true, columns predicted): counts / % of true class / % of predicted class

| | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 186,874 | 21 | 0 | 113 |
| Marine Debris | 65 | 940 | 2 | 68 |
| Algae/Sargassum | 23 | 54 | 1,471 | 4 |
| Other | 130 | 78 | 0 | 1,593 |

| | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.93 | 0.01 | 0.00 | 0.06 |
| Marine Debris | 6.05 | 87.44 | 0.19 | 6.33 |
| Algae/Sargassum | 1.48 | 3.48 | 94.78 | 0.26 |
| Other | 7.22 | 4.33 | 0.00 | 88.45 |

| | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Water | 99.88 | 1.92 | 0.00 | 6.36 |
| Marine Debris | 0.03 | 86.00 | 0.14 | 3.82 |
| Algae/Sargassum | 0.01 | 4.94 | 99.86 | 0.22 |
| Other | 0.07 | 7.14 | 0.00 | 89.60 |

![confusion](figures/confusion_val.png)

![per-class](figures/classwise_val.png)

![PR curves](figures/pr_curves_val.png)

### Per scene

`per_scene_val.csv` has every scene; the figure shows macro F1 (classes present) and debris F1.

![per scene](figures/per_scene_val.png)

## Example test patches

The same 6 test patches (most debris, algae and other pixels) in every run.

![panels](figures/prediction_panels_test.png)
