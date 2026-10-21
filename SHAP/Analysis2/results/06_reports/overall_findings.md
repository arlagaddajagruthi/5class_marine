## Overall findings (computed results only)

All shares below are **normalised SHAP importance**: a feature's mean |SHAP| divided by the sum over all 22 features, for one model (and one class output where stated). They are computed on 300 stratified test pixels per class. Raw SHAP magnitudes are never compared across models. "Direction" is the Spearman correlation (rho) between a feature's value and its SHAP value on the stated output.

### 1. Most influential features globally

| Rank (mean over 5 models) | MARIDA | MADOS |
|---|---|---|
| 1 | nir/red 16.4% | nir/red 16.3% |
| 2 | glcm_contrast 11.7% | swir1/nir 11.9% |
| 3 | swir1/nir 6.7% | NDWI 6.7% |
| 4 | NDWI 6.3% | glcm_contrast 6.0% |
| 5 | B11 5.6% | nir/green 5.5% |

- **nir/red (B8/B4) is the top feature for all four tree models in both datasets.** Its share is 14.5–28.0% on MARIDA and 13.2–29.5% on MADOS, with AdaBoost the most concentrated. For the linear SVM it carries only 3.1% (MARIDA) and 6.0% (MADOS).
- **GLCM texture matters more on MARIDA than on MADOS.** The six GLCM features together take 18–22% for the tree models on MARIDA and 12–14% on MADOS. In both datasets nearly all of that is `glcm_contrast`. The SVM spreads more weight over the GLCM features: 28% (MARIDA) and 37% (MADOS).
- **`glcm_asm` is effectively unused by XGBoost and LightGBM** (0.00–0.01% share in both datasets). Since `glcm_energy` = sqrt(`glcm_asm`), the two are perfectly rank-correlated, and these split-based models only ever split on energy. Any importance difference between these two features is therefore arbitrary and should not be interpreted.

### 2. Most influential features per class (tree models; SVM listed separately in section 3)

| Class output | MARIDA | MADOS | Direction (both datasets) |
|---|---|---|---|
| **1 Water** | nir/red top for RF, XGB, LGBM, AdaBoost (16–33%); then glcm_contrast (10–22%) | nir/red top for all four tree models (13–37%); then NDWI (RF), glcm_contrast (XGB, LGBM) or swir1/nir (AdaBoost) | high nir/red lowers the Water score (rho -0.90 to -0.97); high NDWI raises it (rho +0.86 to +0.96); high glcm_contrast lowers it (rho -0.70 to -0.90) |
| **2 Marine Debris** | nir/red top for all four (13–26%); swir1/nir second (10–24%) | swir1/nir top for all four (15–42%); nir/red second (11–18%) | **nir/red acts as a window, not a slope**: debris SHAP is positive only for nir/red ≈ 0.8–1.5 and peaks at 1.0–1.2 in every tree model and both datasets. It is negative below 0.8 and above 1.5. The median debris pixel has nir/red 0.93 (MARIDA) / 0.99 (MADOS), against 0.50 / 0.75 for water and 2.57 / 2.33 for Sargassum. **Low swir1/nir raises the debris score** (rho -0.52 to -0.61 on MARIDA, -0.72 to -0.82 on MADOS). |
| **3 Sargassum** | nir/red top for all four (23–42%) | nir/red top for all four (23–51%) | high nir/red raises the Sargassum score (rho +0.79 to +0.94); for XGB and LightGBM, high B2/B3 (blue, green) lower it (rho -0.81 to -0.95) |
| **4 Other** (ships, natural organic material, foam; MADOS adds oil spill, oil platform, jellyfish, sea snot) | **glcm_contrast top for all four (19–35%)**, then B11 (RF, XGB, LGBM) or swir1/nir (AdaBoost) | swir1/nir top for XGB, LGBM, AdaBoost (13–24%); B12 top for RF (10%); glcm_contrast 8–11% | high glcm_contrast, high B11/B12 and high swir1/nir raise the Other score (rho +0.74 to +0.96; the one exception is AdaBoost's B11 on MARIDA, +0.49) |

What this does and does not show:
- The models separate Water, Debris and Sargassum mainly along nir/red. Debris is the intermediate range between water (low) and Sargassum (high). swir1/nir is the second axis for debris.
- "Other" is the class where neighbourhood texture (`glcm_contrast` over the 13 x 13 window of the visible-band grey image) carries the most weight in MARIDA. Its median is 5.59 for Other pixels against 0.13 for water. On MADOS that texture gap is much smaller (median 0.28 vs 0.10), and the models rely on SWIR instead. One possible reason is that MADOS's class 4 contains different materials, but that is a hypothesis.
- These are attributions of fitted models. They do not show that a physical property of debris or Sargassum causes the values. A texture statistic describes the grey-level pattern around a pixel, not the material.

### 3. How the five models use features differently

- **XGBoost and LightGBM are nearly interchangeable**: Spearman rank agreement of global importance is 0.95 in both datasets.
- **RF and AdaBoost broadly agree with them** (rank agreement 0.61–0.79 MARIDA, 0.65–0.77 MADOS). AdaBoost is the most concentrated model: on MARIDA nir/red + glcm_contrast take 47% of its attribution, and on MADOS nir/red + swir1/nir take 51%. Its depth-3 trees use few features.
- **The linear SVM ranks features very differently from all tree models**: rank agreement -0.19 to 0.40 on MARIDA and -0.37 to -0.10 on MADOS. Its top feature is B4 (MARIDA) and glcm_dissimilarity (MADOS). It assigns large, opposing linear weights to strongly correlated features. For example, nir/green and NDWI both have rho = -1.00 on its debris output, and their contributions partly cancel. This is a property of a linear fit on collinear inputs, not evidence that those features matter more for the SVM than nir/red does for the trees.
- **Interactions**: in the tree models, 36–70% of the absolute SHAP-interaction mass is off-diagonal. AdaBoost has the smallest interaction share (0.36–0.50) and LightGBM the largest (0.59–0.70), so none of the tree models behaves additively. For the debris output the strongest pair is **nir/red x swir1/nir** for XGBoost, LightGBM and AdaBoost in both datasets, and for RF on MADOS. For RF on MARIDA it is NDWI x nir/red. The linear SVM has no interactions by construction.

### 4. What drives the main confusions

The confusion pairs below are ranked by total test errors over the five models. Values are mean feature values of the explained misclassified pixels against all stratified pixels of the true and predicted class (XGBoost case-study table).

- **Water → Other** is the most frequent error in both datasets: 2,393 errors in MARIDA and 220,773 in MADOS, summed over models.
  - On MARIDA every tree model's error is driven by `glcm_contrast`. The misclassified water pixels have mean contrast 2.94, against 0.17 for water and 5.44 for Other, so they sit between the two.
  - On MADOS the dominant terms are B3, B1 and nir/red for XGBoost and LightGBM, and nir/red for RF and AdaBoost. Texture is not the main factor there.
- **Water → Debris** (MARIDA 966, MADOS 5,722; on MADOS 5,423 of these come from AdaBoost).
  - On MARIDA nir/red is the main term in every tree model. On MADOS it is swir1/nir for XGBoost and LightGBM (nir/red second) and nir/red for RF and AdaBoost.
  - The misclassified water pixels have nir/red 1.00 (MARIDA) / 0.97 (MADOS), inside the debris window, against 0.55 / 0.69 for typical water.
- **Debris → Water** (MARIDA 101, MADOS 244). These debris pixels have nir/red 0.76 / 0.85 and NDWI 0.16 / 0.16. That is lower nir/red and higher NDWI than typical debris (0.95 / 0.98 and 0.05 / 0.03), so they look more water-like on both main axes.
- **Water ↔ Sargassum** (MARIDA water→Sargassum 904; MADOS Sargassum→water 1,322, water→Sargassum 996).
  - Again nir/red dominates. The Sargassum pixels predicted as water have nir/red 0.90 (MARIDA) / 0.93 (MADOS), against 2.56 / 2.16 for typical Sargassum.
  - The water pixels predicted as Sargassum have nir/red 1.16 / 1.12. For XGBoost and LightGBM, B2/B3 also push towards Sargassum.

Overall, most errors come from pixels whose nir/red (and, for Other on MARIDA, glcm_contrast) value lies in the range the models associate with the other class. This is consistent with the class-specific attribution patterns above. It does not establish whether those pixels are mixed or ambiguous or whether the labels are wrong.
