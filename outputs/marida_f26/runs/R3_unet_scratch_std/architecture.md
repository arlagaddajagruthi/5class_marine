# Architecture and input pipeline: run R3_unet_scratch_std

**Model**: U-Net (segmentation_models_pytorch) with a ResNet-50 encoder: 5 encoder stages (strides 2-32), a decoder of 5 upsampling blocks (256, 128, 64, 32, 16 channels) with skip connections from every encoder stage, and a 3x3 convolution to the class scores at full resolution.

**Initialisation**: All weights randomly initialised (PyTorch / timm defaults); trained from scratch.

**Parameters**: 32,593,668 (32,593,668 trainable).

**Output**: 4 class scores per pixel (Water, Marine Debris, Algae/Sargassum, Other); softmax → probabilities; prediction = argmax. Unlabelled pixels are ignored by the loss and the scores.

## How the features are fed in

```
MARIDA patch (11 bands, 256×256, Rayleigh-corrected reflectance, NaN → 0)
   ├─ 20 spectral features per pixel (src/features/pixel_features.py, as the staged ML)
   │     11 bands │ NDVI NDWI FAI FDI NDMI │ red/green nir/red nir/green swir1/nir
   └─ 6 GLCM maps (glcm_maps.py, 13×13 window, as the staged ML)
         contrast dissimilarity homogeneity energy correlation ASM
        ↓  stacked in that order → tensor 26 × 256 × 256
   bands (ch 1–11): (x − mean) / std with the MARIDA paper's band statistics
   other 15 channels: clipped to the train 1st–99th percentile, standardised
        ↓  training only: rotate −90/0/90/180°, horizontal flip (channels and labels together)
   unet → 4 × 256 × 256 class scores
        ↓  test: softmax averaged over 8 views (4 rotations × 2 flips)
   per-pixel class probabilities → argmax
```

All 26 channels enter the first layer together (early fusion): every first-layer filter sees all 26 channels of its 7×7 (U-Net) or patch-embedding (SegNeXt/MariNeXt) window, so the network can combine bands, indices and texture from the first layer on.

## Input channels

| # | Channel | Group | Scaling | Train mean | Train std |
|---|---|---|---|---|---|
| 1 | B1 | bands | standardised (paper stats) | 0.05198 | 0.04726 |
| 2 | B2 | bands | standardised (paper stats) | 0.04784 | 0.04744 |
| 3 | B3 | bands | standardised (paper stats) | 0.04057 | 0.04699 |
| 4 | B4 | bands | standardised (paper stats) | 0.03164 | 0.04967 |
| 5 | B5 | bands | standardised (paper stats) | 0.02973 | 0.04947 |
| 6 | B6 | bands | standardised (paper stats) | 0.03457 | 0.06458 |
| 7 | B7 | bands | standardised (paper stats) | 0.03875 | 0.07595 |
| 8 | B8 | bands | standardised (paper stats) | 0.03436 | 0.0712 |
| 9 | B8A | bands | standardised (paper stats) | 0.03921 | 0.08251 |
| 10 | B11 | bands | standardised (paper stats) | 0.02358 | 0.05111 |
| 11 | B12 | bands | standardised (paper stats) | 0.01589 | 0.03524 |
| 12 | NDVI | indices | clip 1–99 %, standardised | -0.137 | 0.2086 |
| 13 | NDWI | indices | clip 1–99 %, standardised | 0.3064 | 0.2628 |
| 14 | FAI | indices | clip 1–99 %, standardised | 0.00394 | 0.04348 |
| 15 | FDI | indices | clip 1–99 %, standardised | 0.02016 | 0.04317 |
| 16 | NDMI | indices | clip 1–99 %, standardised | 0.2504 | 0.1785 |
| 17 | red/green | ratios | clip 1–99 %, standardised | 0.6927 | 0.1797 |
| 18 | nir/red | ratios | clip 1–99 %, standardised | 0.9797 | 1.449 |
| 19 | nir/green | ratios | clip 1–99 %, standardised | 0.6818 | 0.9055 |
| 20 | swir1/nir | ratios | clip 1–99 %, standardised | 0.6291 | 0.2275 |
| 21 | glcm_contrast | GLCM | clip 1–99 %, standardised | 0.1628 | 0.3758 |
| 22 | glcm_dissimilarity | GLCM | clip 1–99 %, standardised | 0.1242 | 0.1885 |
| 23 | glcm_homogeneity | GLCM | clip 1–99 %, standardised | 0.9425 | 0.08227 |
| 24 | glcm_energy | GLCM | clip 1–99 %, standardised | 0.8504 | 0.1993 |
| 25 | glcm_correlation | GLCM | clip 1–99 %, standardised | 0.6057 | 0.3989 |
| 26 | glcm_asm | GLCM | clip 1–99 %, standardised | 0.7626 | 0.2943 |

## Training

| setting | value |
|---|---|
| epochs | 45 |
| batch | 5 |
| lr | 0.0002 |
| decay | 0.0 |
| lr_steps | [40] |
| gamma | 0.1 |
| weight_param | 1.03 |
| seed | 0 |
| vscp | False |
| ema | False |
| tta | soft, 8 views (4 rotations x 2 flips), softmax averaged |
| select | best validation macro F1 with soft TTA |
| class weights | Water 1.4359, Marine Debris 28.7292, Algae/Sargassum 28.6892, Other 24.0252 |

## Layer summary (torchinfo, depth 4)

See `model_summary.txt`.
