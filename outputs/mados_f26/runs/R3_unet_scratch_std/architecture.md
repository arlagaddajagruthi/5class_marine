# Architecture and input pipeline: run R3_unet_scratch_std

**Model**: U-Net (segmentation_models_pytorch) with a ResNet-50 encoder: 5 encoder stages (strides 2-32), a decoder of 5 upsampling blocks (256, 128, 64, 32, 16 channels) with skip connections from every encoder stage, and a 3x3 convolution to the class scores at full resolution.

**Initialisation**: All weights randomly initialised (PyTorch / timm defaults); trained from scratch.

**Parameters**: 32,593,668 (32,593,668 trainable).

**Output**: 4 class scores per pixel (Water, Marine Debris, Algae/Sargassum, Other); softmax → probabilities; prediction = argmax. Unlabelled pixels are ignored by the loss and the scores.

## How the features are fed in

```
MADOS patch (11 bands, 240×240, Rayleigh-corrected reflectance, NaN → 0)
   ├─ 20 spectral features per pixel (src/features/pixel_features.py, as the staged ML)
   │     11 bands │ NDVI NDWI FAI FDI NDMI │ red/green nir/red nir/green swir1/nir
   └─ 6 GLCM maps (glcm_maps.py, 13×13 window, as the staged ML)
         contrast dissimilarity homogeneity energy correlation ASM
        ↓  stacked in that order → tensor 26 × 240 × 240
   bands (ch 1–11): (x − mean) / std with the MADOS paper's band statistics
   other 15 channels: clipped to the train 1st–99th percentile, standardised
        ↓  training only: h / v / both / no flip, then p = 0.8 rotation by k x 90 (MADOS code) (channels and labels together); VSCP copy-paste adds batch // 2 samples
   unet → 4 × 240 × 240 class scores
        ↓  test: softmax averaged over 8 views (4 rotations × 2 flips)
   per-pixel class probabilities → argmax
```

All 26 channels enter the first layer together (early fusion): every first-layer filter sees all 26 channels of its 7×7 (U-Net) or patch-embedding (SegNeXt/MariNeXt) window, so the network can combine bands, indices and texture from the first layer on.

## Input channels

| # | Channel | Group | Scaling | Train mean | Train std |
|---|---|---|---|---|---|
| 1 | B1 | bands | standardised (paper stats) | 0.05827 | 0.03241 |
| 2 | B2 | bands | standardised (paper stats) | 0.05223 | 0.03432 |
| 3 | B3 | bands | standardised (paper stats) | 0.04381 | 0.03548 |
| 4 | B4 | bands | standardised (paper stats) | 0.03571 | 0.03758 |
| 5 | B5 | bands | standardised (paper stats) | 0.03413 | 0.03785 |
| 6 | B6 | bands | standardised (paper stats) | 0.0368 | 0.04992 |
| 7 | B7 | bands | standardised (paper stats) | 0.03999 | 0.05884 |
| 8 | B8 | bands | standardised (paper stats) | 0.03567 | 0.05546 |
| 9 | B8A | bands | standardised (paper stats) | 0.03965 | 0.06424 |
| 10 | B11 | bands | standardised (paper stats) | 0.0268 | 0.04211 |
| 11 | B12 | bands | standardised (paper stats) | 0.01979 | 0.03019 |
| 12 | NDVI | indices | clip 1–99 %, standardised | -0.1077 | 0.1621 |
| 13 | NDWI | indices | clip 1–99 %, standardised | 0.2478 | 0.2201 |
| 14 | FAI | indices | clip 1–99 %, standardised | 0.001233 | 0.03223 |
| 15 | FDI | indices | clip 1–99 %, standardised | 0.01744 | 0.03135 |
| 16 | NDMI | indices | clip 1–99 %, standardised | 0.2119 | 0.1715 |
| 17 | red/green | ratios | clip 1–99 %, standardised | 0.748 | 0.1759 |
| 18 | nir/red | ratios | clip 1–99 %, standardised | 0.9237 | 0.9503 |
| 19 | nir/green | ratios | clip 1–99 %, standardised | 0.6953 | 0.6414 |
| 20 | swir1/nir | ratios | clip 1–99 %, standardised | 0.6787 | 0.2231 |
| 21 | glcm_contrast | GLCM | clip 1–99 %, standardised | 0.1753 | 0.3226 |
| 22 | glcm_dissimilarity | GLCM | clip 1–99 %, standardised | 0.1447 | 0.1836 |
| 23 | glcm_homogeneity | GLCM | clip 1–99 %, standardised | 0.9329 | 0.08298 |
| 24 | glcm_energy | GLCM | clip 1–99 %, standardised | 0.8305 | 0.1969 |
| 25 | glcm_correlation | GLCM | clip 1–99 %, standardised | 0.5102 | 0.4108 |
| 26 | glcm_asm | GLCM | clip 1–99 %, standardised | 0.7275 | 0.2959 |

## Training

| setting | value |
|---|---|
| epochs | 80 |
| batch | 5 |
| lr | 0.0002 |
| decay | 0.0 |
| lr_steps | [45, 65] |
| gamma | 0.1 |
| weight_param | 1.03 |
| seed | 0 |
| augmentation | h / v / both / no flip, then p = 0.8 rotation by k x 90 (MADOS code) |
| vscp | True |
| ema | True |
| ema_decay | 0.999 |
| drop_last | True |
| tta | soft, 8 views (4 rotations x 2 flips), softmax averaged |
| select | best validation macro F1 of the EMA model with soft TTA |
| class weights | Water 1.6917, Marine Debris 30.472, Algae/Sargassum 28.4525, Other 4.5656 |

## Layer summary (torchinfo, depth 4)

See `model_summary.txt`.
