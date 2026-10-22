# XAI, run R4_segnext_scratch_std

Explained output: for class c, the sum of P(c) over the pixels labelled c in a test patch. Importance = mean |SHAP| at the class-c pixels; share = importance / sum over features. Direction (in brackets) = Spearman correlation between the feature value and its SHAP value at the class-c pixels: + means high values push towards the class. Group values add the SHAP values of a group's features at each pixel first. Correlated features share credit (see `features/correlation/`), so read single features together with the grouped results.

![importance](figures/importance_by_class.png)

![groups](figures/group_importance.png)

![heatmap](figures/importance_heatmap.png)

## Grouped importance (share of |SHAP|, %)

| group | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| bands | 23.8 | 11.2 | 13.4 | 22.8 |
| indices | 34.5 | 36.6 | 57.5 | 12.6 |
| ratios | 22.2 | 26.3 | 13.6 | 11.3 |
| GLCM | 19.5 | 25.9 | 15.4 | 53.3 |

## Feature importance (share of |SHAP|, %; direction = Spearman of value vs SHAP)

| feature | group | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|---|
| B1 | bands | 1.8 (-0.02) | 2.1 (-0.09) | 1.6 (+0.06) | 1.3 (-0.20) |
| B2 | bands | 1.8 (+0.06) | 1.2 (+0.05) | 0.9 (-0.09) | 2.4 (+0.13) |
| B3 | bands | 4.1 (-0.01) | 1.8 (+0.01) | 1.7 (+0.15) | 5.2 (+0.09) |
| B4 | bands | 3.6 (-0.05) | 1.9 (-0.07) | 1.3 (-0.09) | 3.7 (+0.03) |
| B5 | bands | 3.8 (-0.16) | 1.6 (+0.41) | 3.3 (+0.36) | 3.4 (+0.17) |
| B6 | bands | 2.7 (+0.04) | 1.3 (-0.04) | 3.2 (-0.37) | 2.6 (+0.24) |
| B7 | bands | 2.7 (+0.07) | 1.1 (-0.18) | 2.1 (-0.04) | 2.9 (-0.18) |
| B8 | bands | 2.3 (+0.00) | 2.1 (-0.24) | 1.8 (-0.02) | 2.1 (-0.14) |
| B8A | bands | 2.9 (+0.02) | 0.9 (-0.18) | 2.2 (+0.06) | 2.1 (+0.12) |
| B11 | bands | 2.8 (+0.02) | 1.0 (+0.01) | 0.9 (-0.08) | 2.9 (-0.01) |
| B12 | bands | 2.3 (+0.02) | 1.2 (-0.17) | 1.0 (-0.17) | 2.5 (-0.46) |
| NDVI | indices | 8.6 (+0.02) | 6.4 (+0.15) | 12.9 (+0.22) | 2.3 (+0.13) |
| NDWI | indices | 5.1 (+0.00) | 7.9 (+0.09) | 7.8 (+0.04) | 2.2 (+0.10) |
| FAI | indices | 4.7 (-0.02) | 1.3 (-0.23) | 4.3 (-0.34) | 1.1 (-0.15) |
| FDI | indices | 3.7 (+0.10) | 3.7 (+0.07) | 5.9 (-0.05) | 3.2 (-0.17) |
| NDMI | indices | 7.4 (-0.09) | 11.9 (+0.07) | 10.4 (+0.08) | 2.0 (-0.05) |
| red/green | ratios | 4.5 (+0.09) | 8.7 (+0.03) | 1.8 (-0.18) | 3.7 (-0.06) |
| nir/red | ratios | 3.9 (+0.01) | 1.2 (-0.01) | 4.4 (-0.01) | 0.7 (-0.12) |
| nir/green | ratios | 3.8 (+0.00) | 3.1 (+0.07) | 5.6 (+0.15) | 1.5 (+0.06) |
| swir1/nir | ratios | 5.6 (-0.13) | 9.2 (+0.01) | 4.4 (+0.10) | 3.1 (+0.32) |
| glcm_contrast | GLCM | 4.1 (-0.08) | 5.1 (+0.15) | 2.5 (-0.21) | 21.6 (+0.11) |
| glcm_dissimilarity | GLCM | 3.7 (-0.00) | 5.4 (-0.12) | 4.9 (-0.15) | 9.0 (+0.02) |
| glcm_homogeneity | GLCM | 3.8 (+0.05) | 6.0 (-0.26) | 3.0 (-0.10) | 8.7 (-0.01) |
| glcm_energy | GLCM | 3.7 (-0.08) | 4.7 (+0.14) | 5.7 (+0.03) | 5.0 (+0.01) |
| glcm_correlation | GLCM | 3.5 (-0.04) | 4.3 (+0.01) | 1.3 (+0.03) | 1.9 (+0.10) |
| glcm_asm | GLCM | 3.2 (+0.03) | 5.1 (+0.03) | 5.2 (-0.13) | 3.1 (-0.05) |

## Completeness check (GradientSHAP)

Sum of attributions vs S_c(x) − mean S_c(baseline) over 267 patch × class cases: correlation 0.971, median relative error 28.8 % (32 samples, 20 baselines). `completeness.csv` has every case.

## Context share

Median share of |attribution| that lies on pixels that are not the explained class (the model using the surroundings):

| class | context share (%) |
|---|---|
| Water | 97.4 |
| Marine Debris | 99.6 |
| Algae/Sargassum | 95.3 |
| Other | 99.0 |

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
