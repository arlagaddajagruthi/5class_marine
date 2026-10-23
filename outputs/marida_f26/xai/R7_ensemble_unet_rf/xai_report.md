# XAI, R7_ensemble_unet_rf

Approximation: the ensemble's feature shares are w × (R3_unet_scratch_std share) + (1 − w) × (R6_rf share) with w = 0.40. The members are explained with different estimators (GradientSHAP, TreeSHAP), so these shares are indicative; see the members' reports for detail.

![importance](figures/importance_by_class.png)

![groups](figures/group_importance.png)

## Grouped importance (share of |SHAP|, %)

| group | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| bands | 21.2 | 19.4 | 9.5 | 35.7 |
| indices | 39.9 | 44.6 | 59.1 | 10.9 |
| ratios | 18.5 | 29.6 | 26.7 | 9.0 |
| GLCM | 20.5 | 6.5 | 4.7 | 44.4 |

## Feature importance (share of |SHAP|, %; direction = Spearman of value vs SHAP)

| feature | group | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|---|
| B1 | bands | 2.2 | 1.6 | 0.8 | 1.0 |
| B2 | bands | 1.5 | 0.9 | 1.9 | 3.0 |
| B3 | bands | 2.3 | 1.4 | 1.1 | 4.4 |
| B4 | bands | 2.8 | 2.8 | 1.3 | 5.3 |
| B5 | bands | 3.8 | 2.5 | 1.6 | 2.7 |
| B6 | bands | 1.7 | 1.6 | 0.8 | 1.2 |
| B7 | bands | 1.9 | 1.5 | 1.0 | 1.7 |
| B8 | bands | 2.1 | 1.8 | 0.7 | 2.6 |
| B8A | bands | 2.5 | 1.6 | 0.9 | 2.7 |
| B11 | bands | 3.1 | 2.4 | 2.2 | 7.6 |
| B12 | bands | 1.1 | 2.0 | 1.2 | 2.9 |
| NDVI | indices | 14.7 | 9.0 | 24.0 | 2.8 |
| NDWI | indices | 6.8 | 9.5 | 10.2 | 2.7 |
| FAI | indices | 8.0 | 6.3 | 9.2 | 0.9 |
| FDI | indices | 5.4 | 7.5 | 4.6 | 2.8 |
| NDMI | indices | 2.8 | 10.7 | 4.3 | 2.6 |
| red/green | ratios | 3.3 | 8.4 | 2.1 | 3.2 |
| nir/red | ratios | 8.1 | 6.2 | 15.7 | 1.6 |
| nir/green | ratios | 4.7 | 4.7 | 6.1 | 1.6 |
| swir1/nir | ratios | 2.7 | 9.0 | 3.0 | 2.9 |
| glcm_contrast | GLCM | 8.5 | 2.6 | 2.0 | 22.6 |
| glcm_dissimilarity | GLCM | 3.6 | 1.5 | 1.5 | 11.3 |
| glcm_homogeneity | GLCM | 2.0 | 1.2 | 0.9 | 5.3 |
| glcm_energy | GLCM | 1.3 | 1.0 | 1.4 | 1.6 |
| glcm_correlation | GLCM | 2.1 | 1.2 | 0.5 | 1.2 |
| glcm_asm | GLCM | 1.0 | 1.1 | 1.1 | 1.7 |
