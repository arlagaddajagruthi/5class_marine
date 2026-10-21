# Architecture and input pipeline: run G2b_unet_ssl4eo_raw_11class

**Model**: U-Net (segmentation_models_pytorch) with a ResNet-50 encoder: 5 encoder stages (strides 2-32), a decoder of 5 upsampling blocks (256, 128, 64, 32, 16 channels) with skip connections from every encoder stage, and a 3x3 convolution to the class scores at full resolution.

**Initialisation**: Encoder initialised from SSL4EO-S12 MoCo ResNet-50 weights (torchgeo `ResNet50_Weights.SENTINEL2_ALL_MOCO`, pretrained on 13-band Sentinel-2 L1C / 10000). First convolution (64 filters, 7×7, stride 2): the pretrained filters of B1–B8, B8A, B11, B12 are copied to input channels 1–11 (B9 and B10 dropped); the filters of channels 12–26 start at zero, so at the start the network sees only the bands and learns how to use the 15 extra channels. The decoder is randomly initialised.

**Parameters**: 32,594,683 (32,594,683 trainable).

**Output**: 11 class scores per pixel (Marine Debris, Dense Sargassum, Sparse Sargassum, Natural Organic Material, Ship, Clouds, Marine Water, Sediment-Laden Water, Foam, Turbid Water, Shallow Water); softmax → probabilities; prediction = argmax. Unlabelled pixels are ignored by the loss and the scores.

## How the features are fed in

```
MARIDA patch (11 bands, 256×256, Rayleigh-corrected reflectance, NaN → 0)
   ├─ 20 spectral features per pixel (src/features/pixel_features.py, as the staged ML)
   │     11 bands │ NDVI NDWI FAI FDI NDMI │ red/green nir/red nir/green swir1/nir
   └─ 6 GLCM maps (glcm_maps.py, 13×13 window, as the staged ML)
         contrast dissimilarity homogeneity energy correlation ASM
        ↓  stacked in that order → tensor 26 × 256 × 256
   bands (ch 1–11): reflectance unchanged (the SSL4EO pretraining scale)
   other 15 channels: clipped to the train 1st–99th percentile, standardised
        ↓  training only: rotate −90/0/90/180°, horizontal flip (channels and labels together)
   unet → 11 × 256 × 256 class scores
        ↓  test: softmax averaged over 8 views (4 rotations × 2 flips)
   per-pixel class probabilities → argmax
```

All 26 channels enter the first layer together (early fusion): every first-layer filter sees all 26 channels of its 7×7 (U-Net) or patch-embedding (SegNeXt/MariNeXt) window, so the network can combine bands, indices and texture from the first layer on.

## Input channels

| # | Channel | Group | Scaling | Train mean | Train std |
|---|---|---|---|---|---|
| 1 | B1 | bands | raw reflectance | 0.05008 | 0.0284 |
| 2 | B2 | bands | raw reflectance | 0.04587 | 0.02801 |
| 3 | B3 | bands | raw reflectance | 0.03862 | 0.0291 |
| 4 | B4 | bands | raw reflectance | 0.02956 | 0.03038 |
| 5 | B5 | bands | raw reflectance | 0.02771 | 0.03132 |
| 6 | B6 | bands | raw reflectance | 0.03316 | 0.05377 |
| 7 | B7 | bands | raw reflectance | 0.03753 | 0.06717 |
| 8 | B8 | bands | raw reflectance | 0.03324 | 0.06308 |
| 9 | B8A | bands | raw reflectance | 0.03809 | 0.07462 |
| 10 | B11 | bands | raw reflectance | 0.02234 | 0.04161 |
| 11 | B12 | bands | raw reflectance | 0.01484 | 0.02628 |
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
| class weights | Marine Debris 29.4619, Dense Sargassum 31.722, Sparse Sargassum 31.2281, Natural Organic Material 32.0594, Ship 27.0508, Clouds 5.9789, Marine Water 4.2602, Sediment-Laden Water 3.0406, Foam 32.6599, Turbid Water 4.7896, Shallow Water 16.5571 |

## Layer summary (torchinfo, depth 4)

See `model_summary.txt`.
