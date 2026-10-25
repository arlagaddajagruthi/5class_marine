# MARIDA 26-channel input specification

The 26 channels are every staged-ML feature except the 4 spectral statistics, in the model's input order. Every value is computed per pixel from the 11 Rayleigh-corrected bands (NaN → 0, as in the staged ML; 7 of 1,381 patches contain NaN), with the same code as the staged ML, so the features equal the RF's features (checked in `verify_features.json`).

Scaling (train-split statistics only):

- **bands, raw** (U-Net with SSL4EO weights: R1, G2b): reflectance unchanged, the scale the SSL4EO encoder was pretrained on (L1C / 10000).
- **bands, std** (R2, R3, SegNeXt-T, MariNeXt arch.): (x − mean) / std with the MARIDA paper's band statistics.
- **the 15 other channels** (all models): clipped to the train 1st–99th percentile, then (x − mean) / std of the clipped values.

Augmentation (training only): the 26 channels and the label are rotated by −90/0/90/180° and flipped horizontally together. Every feature is per pixel or averaged over the 4 GLCM directions, so these transforms do not change the feature values.

| # | Channel | Group | Definition | 1st pct | 99th pct | mean | std |
|---|---|---|---|---|---|---|---|
| 1 | B1 | bands | coastal aerosol, 443 nm | 0.02191 | 0.2191 | 0.05008 | 0.0284 |
| 2 | B2 | bands | blue, 490 nm | 0.01782 | 0.2103 | 0.04587 | 0.02801 |
| 3 | B3 | bands | green, 560 nm | 0.007268 | 0.1952 | 0.03862 | 0.0291 |
| 4 | B4 | bands | red, 665 nm | 0.003473 | 0.1993 | 0.02956 | 0.03038 |
| 5 | B5 | bands | red edge 1, 705 nm | 0.002628 | 0.1946 | 0.02771 | 0.03132 |
| 6 | B6 | bands | red edge 2, 740 nm | 0.002843 | 0.304 | 0.03316 | 0.05377 |
| 7 | B7 | bands | red edge 3, 783 nm | 0.003229 | 0.3857 | 0.03753 | 0.06717 |
| 8 | B8 | bands | NIR, 842 nm | 0.00235 | 0.3635 | 0.03324 | 0.06308 |
| 9 | B8A | bands | narrow NIR, 865 nm | 0.00271 | 0.4291 | 0.03809 | 0.07462 |
| 10 | B11 | bands | SWIR 1, 1610 nm | 0.0006669 | 0.2522 | 0.02234 | 0.04161 |
| 11 | B12 | bands | SWIR 2, 2190 nm | 0.0003422 | 0.1813 | 0.01484 | 0.02628 |
| 12 | NDVI | indices | (B8 - B4) / (B8 + B4) | -0.5476 | 0.8383 | -0.137 | 0.2086 |
| 13 | NDWI | indices | (B3 - B8) / (B3 + B8) | -0.7337 | 0.7673 | 0.3064 | 0.2628 |
| 14 | FAI | indices | B8 - [B4 + (B11 - B4)(842 - 665)/(1610 - 665)]  (Hu 2009) | -0.0406 | 0.2773 | 0.00394 | 0.04348 |
| 15 | FDI | indices | B8 - [B6 + (B11 - B6)(842 - 665)/(1610 - 665) x 10]  (Biermann et al. 2020) | -0.0181 | 0.2846 | 0.02016 | 0.04317 |
| 16 | NDMI | indices | (B8 - B11) / (B8 + B11) | -0.09053 | 0.7913 | 0.2504 | 0.1785 |
| 17 | red/green | ratios | B4 / B3 | 0.301 | 1.162 | 0.6927 | 0.1797 |
| 18 | nir/red | ratios | B8 / B4 | 0.2783 | 11.37 | 0.9797 | 1.449 |
| 19 | nir/green | ratios | B8 / B3 | 0.1269 | 6.511 | 0.6818 | 0.9055 |
| 20 | swir1/nir | ratios | B11 / B8 | 0.09814 | 1.199 | 0.6291 | 0.2275 |
| 21 | glcm_contrast | GLCM | GLCM contrast, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged | 0 | 2.834 | 0.1628 | 0.3758 |
| 22 | glcm_dissimilarity | GLCM | GLCM dissimilarity, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged | 0 | 1.059 | 0.1242 | 0.1885 |
| 23 | glcm_homogeneity | GLCM | GLCM homogeneity, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged | 0.6102 | 1 | 0.9425 | 0.08227 |
| 24 | glcm_energy | GLCM | GLCM energy, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged | 0.2341 | 1 | 0.8504 | 0.1993 |
| 25 | glcm_correlation | GLCM | GLCM correlation, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged | -0.01781 | 1 | 0.6057 | 0.3989 |
| 26 | glcm_asm | GLCM | GLCM asm, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged | 0.05481 | 1 | 0.7626 | 0.2943 |
