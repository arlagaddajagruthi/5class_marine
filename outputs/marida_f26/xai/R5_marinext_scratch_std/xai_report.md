# XAI, run R5_marinext_scratch_std

Explained output: for class c, the sum of P(c) over the pixels labelled c in a test patch. Importance = mean |SHAP| at the class-c pixels; share = importance / sum over features. Direction (in brackets) = Spearman correlation between the feature value and its SHAP value at the class-c pixels: + means high values push towards the class. Group values add the SHAP values of a group's features at each pixel first. Correlated features share credit (see `features/correlation/`), so read single features together with the grouped results.

![importance](figures/importance_by_class.png)

![groups](figures/group_importance.png)

![heatmap](figures/importance_heatmap.png)

## Grouped importance (share of |SHAP|, %)

| group | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| bands | 25.3 | 11.3 | 14.0 | 31.5 |
| indices | 34.5 | 46.7 | 55.0 | 12.4 |
| ratios | 22.6 | 23.9 | 13.5 | 11.2 |
| GLCM | 17.6 | 18.1 | 17.4 | 44.9 |

## Feature importance (share of |SHAP|, %; direction = Spearman of value vs SHAP)

| feature | group | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|---|
| B1 | bands | 1.8 (-0.10) | 1.4 (-0.23) | 0.9 (-0.08) | 1.2 (-0.11) |
| B2 | bands | 2.0 (+0.02) | 1.6 (+0.26) | 0.7 (-0.17) | 2.9 (+0.21) |
| B3 | bands | 4.5 (+0.01) | 2.3 (-0.02) | 1.6 (+0.31) | 7.0 (+0.17) |
| B4 | bands | 4.3 (+0.07) | 2.1 (-0.22) | 0.9 (-0.22) | 4.9 (+0.07) |
| B5 | bands | 3.7 (-0.11) | 1.3 (+0.49) | 3.4 (+0.53) | 3.4 (+0.24) |
| B6 | bands | 2.8 (+0.15) | 1.5 (-0.07) | 3.3 (-0.49) | 2.8 (+0.04) |
| B7 | bands | 2.8 (+0.09) | 1.4 (-0.23) | 2.9 (+0.11) | 3.0 (-0.20) |
| B8 | bands | 2.2 (+0.04) | 1.8 (-0.36) | 2.0 (+0.14) | 2.7 (-0.27) |
| B8A | bands | 2.7 (-0.01) | 0.9 (+0.09) | 2.1 (+0.06) | 2.1 (+0.09) |
| B11 | bands | 2.6 (-0.03) | 0.9 (+0.11) | 0.7 (-0.08) | 2.2 (+0.19) |
| B12 | bands | 1.9 (+0.09) | 1.6 (-0.42) | 0.7 (-0.35) | 2.5 (-0.50) |
| NDVI | indices | 8.6 (-0.03) | 5.8 (+0.17) | 12.8 (+0.32) | 2.2 (+0.19) |
| NDWI | indices | 5.1 (+0.05) | 8.2 (-0.09) | 8.4 (+0.07) | 2.4 (-0.03) |
| FAI | indices | 5.0 (+0.03) | 1.2 (-0.38) | 4.2 (-0.41) | 1.0 (-0.33) |
| FDI | indices | 4.3 (-0.00) | 3.4 (+0.25) | 6.1 (-0.07) | 2.6 (-0.03) |
| NDMI | indices | 8.0 (-0.19) | 16.3 (+0.23) | 10.1 (+0.09) | 2.2 (+0.33) |
| red/green | ratios | 5.0 (+0.11) | 9.7 (+0.25) | 1.8 (-0.12) | 3.7 (+0.06) |
| nir/red | ratios | 3.9 (+0.06) | 1.0 (-0.22) | 3.6 (+0.06) | 0.6 (-0.29) |
| nir/green | ratios | 3.7 (-0.00) | 2.9 (+0.04) | 5.3 (+0.23) | 1.4 (+0.18) |
| swir1/nir | ratios | 4.9 (-0.18) | 6.3 (-0.15) | 4.7 (+0.06) | 2.9 (+0.43) |
| glcm_contrast | GLCM | 3.2 (+0.03) | 3.2 (-0.17) | 2.4 (-0.34) | 15.7 (+0.12) |
| glcm_dissimilarity | GLCM | 3.3 (+0.00) | 5.3 (-0.17) | 5.4 (-0.19) | 10.5 (-0.04) |
| glcm_homogeneity | GLCM | 3.4 (+0.14) | 5.4 (-0.39) | 3.8 (-0.03) | 8.3 (-0.11) |
| glcm_energy | GLCM | 4.1 (-0.08) | 5.8 (-0.19) | 6.4 (-0.09) | 6.3 (+0.13) |
| glcm_correlation | GLCM | 3.0 (+0.02) | 3.5 (+0.00) | 1.3 (-0.02) | 1.5 (+0.08) |
| glcm_asm | GLCM | 3.0 (+0.13) | 5.0 (+0.05) | 4.6 (-0.05) | 4.0 (-0.08) |

## Completeness check (GradientSHAP)

Sum of attributions vs S_c(x) − mean S_c(baseline) over 267 patch × class cases: correlation 0.992, median relative error 21.3 % (32 samples, 20 baselines). `completeness.csv` has every case.

## Context share

Median share of |attribution| that lies on pixels that are not the explained class (the model using the surroundings):

| class | context share (%) |
|---|---|
| Water | 96.3 |
| Marine Debris | 99.2 |
| Algae/Sargassum | 93.9 |
| Other | 98.2 |

## Water

![beeswarm](figures/beeswarm_water.png)

![dependence](figures/dependence_water.png)

## Marine Debris

![beeswarm](figures/beeswarm_debris.png)

![dependence](figures/dependence_debris.png)

## Algae/Sargassum

![beeswarm](figures/beeswarm_algae.png)

![dependence](figures/dependence_algae.png)

## Other

![beeswarm](figures/beeswarm_other.png)

![dependence](figures/dependence_other.png)

## SHAP maps and Seg-Grad-CAM on example test patches

![maps](figures/maps_water.png)
![maps](figures/maps_debris.png)
![maps](figures/maps_algae.png)
![maps](figures/maps_other.png)
