# XAI, R7_ensemble_unet_rf

Approximation: the ensemble's feature shares are w × (R3_unet_scratch_std share) + (1 − w) × (R6_rf share) with w = 0.55. The members are explained with different estimators (GradientSHAP, TreeSHAP), so these shares are indicative; see the members' reports for detail.

![importance](figures/importance_by_class.png)

![groups](figures/group_importance.png)

## Grouped importance (share of |SHAP|, %)

| group | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| bands | 31.1 | 20.8 | 10.9 | 40.8 |
| indices | 25.2 | 44.4 | 62.4 | 19.4 |
| ratios | 18.0 | 25.7 | 22.2 | 18.2 |
| GLCM | 25.6 | 9.0 | 4.4 | 21.6 |

## Feature importance (share of |SHAP|, %; direction = Spearman of value vs SHAP)

| feature | group | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|---|
| B1 | bands | 10.1 | 3.2 | 1.3 | 9.9 |
| B2 | bands | 6.3 | 1.9 | 2.1 | 7.0 |
| B3 | bands | 8.3 | 2.4 | 0.9 | 3.9 |
| B4 | bands | 5.0 | 2.4 | 1.1 | 2.9 |
| B5 | bands | 3.7 | 2.9 | 1.4 | 2.6 |
| B6 | bands | 1.7 | 1.0 | 1.2 | 2.8 |
| B7 | bands | 1.1 | 1.0 | 0.9 | 1.2 |
| B8 | bands | 1.4 | 1.2 | 1.5 | 2.1 |
| B8A | bands | 1.5 | 1.0 | 1.3 | 1.4 |
| B11 | bands | 2.7 | 2.3 | 2.0 | 3.9 |
| B12 | bands | 2.2 | 4.0 | 1.1 | 3.8 |
| NDVI | indices | 4.1 | 6.8 | 28.0 | 3.5 |
| NDWI | indices | 4.0 | 7.7 | 10.8 | 4.5 |
| FAI | indices | 3.8 | 4.2 | 7.7 | 1.7 |
| FDI | indices | 3.6 | 10.8 | 5.4 | 5.0 |
| NDMI | indices | 3.7 | 11.9 | 5.3 | 3.6 |
| red/green | ratios | 5.3 | 6.7 | 1.0 | 8.5 |
| nir/red | ratios | 2.5 | 3.0 | 11.4 | 1.6 |
| nir/green | ratios | 2.5 | 3.9 | 6.2 | 2.6 |
| swir1/nir | ratios | 5.5 | 10.9 | 3.6 | 5.3 |
| glcm_contrast | GLCM | 4.7 | 4.2 | 1.9 | 5.3 |
| glcm_dissimilarity | GLCM | 3.2 | 1.7 | 0.8 | 3.8 |
| glcm_homogeneity | GLCM | 3.0 | 1.3 | 0.6 | 3.6 |
| glcm_energy | GLCM | 2.7 | 1.2 | 1.1 | 3.3 |
| glcm_correlation | GLCM | 4.8 | 1.0 | 0.7 | 3.2 |
| glcm_asm | GLCM | 2.8 | 1.1 | 0.9 | 2.9 |
