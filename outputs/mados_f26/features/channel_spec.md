# MADOS 26-channel input specification

The 26 channels are every staged-ML feature except the 4 spectral statistics, in the model's input order. Every value is computed per pixel from the 11 Rayleigh-corrected bands (NaN → 0, as in the staged ML; 20/60 m bands nearest-neighbour upsampled to 10 m, as in MADOSDataset), with the same code as the staged ML, so the features equal the RF's features (checked in `verify_features.json`).

Scaling (train-split statistics only):

- **bands, raw** (U-Net with SSL4EO weights: R1, G2b): reflectance unchanged, the scale the SSL4EO encoder was pretrained on (L1C / 10000).
- **bands, std** (R2, R3, SegNeXt-T, MariNeXt arch.): (x − mean) / std with the MADOS paper's band statistics.
- **the 15 other channels** (all models): clipped to the train 1st–99th percentile, then (x − mean) / std of the clipped values.

Augmentation (training only): the 26 channels and the label are transformed together, as in the MADOS code (one of horizontal / vertical / both / no flip, then with p = 0.8 a rotation by k × 90°; VSCP copy-paste on top, in train.py). Every feature is per pixel or averaged over the 4 GLCM directions, so these transforms do not change the feature values.

| # | Channel | Group | Definition | 1st pct | 99th pct | mean | std |
|---|---|---|---|---|---|---|---|
| 1 | B1 | bands | coastal aerosol, 443 nm | 0.02524 | 0.1221 | 0.05696 | 0.02025 |
| 2 | B2 | bands | blue, 490 nm | 0.01859 | 0.1288 | 0.05104 | 0.02273 |
| 3 | B3 | bands | green, 560 nm | 0.007991 | 0.1317 | 0.04255 | 0.02505 |
| 4 | B4 | bands | red, 665 nm | 0.00375 | 0.1405 | 0.0344 | 0.02591 |
| 5 | B5 | bands | red edge 1, 705 nm | 0.003097 | 0.143 | 0.0329 | 0.02667 |
| 6 | B6 | bands | red edge 2, 740 nm | 0.003098 | 0.272 | 0.03595 | 0.0426 |
| 7 | B7 | bands | red edge 3, 783 nm | 0.003702 | 0.348 | 0.03912 | 0.05208 |
| 8 | B8 | bands | NIR, 842 nm | 0.002666 | 0.3273 | 0.03484 | 0.04899 |
| 9 | B8A | bands | narrow NIR, 865 nm | 0.002848 | 0.39 | 0.03882 | 0.05762 |
| 10 | B11 | bands | SWIR 1, 1610 nm | 0.0007619 | 0.2137 | 0.02563 | 0.03378 |
| 11 | B12 | bands | SWIR 2, 2190 nm | 0.0004415 | 0.1313 | 0.01874 | 0.02163 |
| 12 | NDVI | indices | (B8 - B4) / (B8 + B4) | -0.4844 | 0.7965 | -0.1077 | 0.1621 |
| 13 | NDWI | indices | (B3 - B8) / (B3 + B8) | -0.6902 | 0.7242 | 0.2478 | 0.2201 |
| 14 | FAI | indices | B8 - [B4 + (B11 - B4)(842 - 665)/(1610 - 665)]  (Hu 2009) | -0.03242 | 0.2453 | 0.001233 | 0.03223 |
| 15 | FDI | indices | B8 - [B6 + (B11 - B6)(842 - 665)/(1610 - 665) x 10]  (Biermann et al. 2020) | -0.008209 | 0.2473 | 0.01744 | 0.03135 |
| 16 | NDMI | indices | (B8 - B11) / (B8 + B11) | -0.07657 | 0.7276 | 0.2119 | 0.1715 |
| 17 | red/green | ratios | B4 / B3 | 0.3231 | 1.137 | 0.748 | 0.1759 |
| 18 | nir/red | ratios | B8 / B4 | 0.3286 | 8.829 | 0.9237 | 0.9503 |
| 19 | nir/green | ratios | B8 / B3 | 0.1506 | 5.457 | 0.6953 | 0.6414 |
| 20 | swir1/nir | ratios | B11 / B8 | 0.1312 | 1.166 | 0.6787 | 0.2231 |
| 21 | glcm_contrast | GLCM | GLCM contrast, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged | 0 | 2.38 | 0.1753 | 0.3226 |
| 22 | glcm_dissimilarity | GLCM | GLCM dissimilarity, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged | 0 | 0.9563 | 0.1447 | 0.1836 |
| 23 | glcm_homogeneity | GLCM | GLCM homogeneity, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged | 0.6364 | 1 | 0.9329 | 0.08298 |
| 24 | glcm_energy | GLCM | GLCM energy, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged | 0.2639 | 1 | 0.8305 | 0.1969 |
| 25 | glcm_correlation | GLCM | GLCM correlation, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged | -0.03454 | 1 | 0.5102 | 0.4108 |
| 26 | glcm_asm | GLCM | GLCM asm, grey B4/B3/B2 in [0, 0.15], 17 levels, 13x13 window, d = 1, 4 directions averaged | 0.06965 | 1 | 0.7275 | 0.2959 |
