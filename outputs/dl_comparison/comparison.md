# Deep-learning model comparison

Our scores are recomputed from the saved **test** confusion matrices: classes 1-4, every annotated pixel of the official test split, macro = mean over the 4 classes. The Random Forest rows are the per-pixel baselines from the staged study (log class weights).

📄 **Paper rows** (top of each table, values in _italics_) are the published results. **Bold** = best of our models in the column.

The paper results are in the papers' own class schemes (MARIDA 11 classes, MADOS 15 classes), so only marine debris is the same class; their macro scores average over more, and different, classes. The MARIDA paper reports no precision, kappa or OA, and its 'PA' is recall.

Inputs: DeepLabv3 uses the 11 Sentinel-2 bands; U-Net, SegNeXt-T and the MariNeXt architecture use 16 channels (11 bands + NDWI + red/green, NIR/red, NIR/green, SWIR1/NIR). The ensemble averages U-Net and RF class probabilities, p = w·p_U-Net + (1−w)·p_RF.

## MARIDA

### Overall

| Model | Macro F1 | mIoU | κ | OA | Macro P | Macro R |
|---|---|---|---|---|---|---|
| 📄 **RF, spectral (RF_SS)** (MARIDA paper, 11 classes) | _0.690_ | _0.570_ | – | – | – | _0.720_ |
| 📄 **RF + indices (RF_SS+SI)** (MARIDA paper, 11 classes) | _0.700_ | _0.590_ | – | – | – | _0.730_ |
| 📄 **RF + indices + GLCM** (MARIDA paper, 11 classes) | _0.790_ | _0.690_ | – | – | – | _0.790_ |
| 📄 **U-Net** (MARIDA paper, 11 classes) | _0.690_ | _0.570_ | – | – | – | _0.690_ |
| Random Forest, stage 5 (log weights, per-pixel features) | 0.904 | 0.837 | 0.903 | 0.996 | 0.887 | 0.927 |
| Random Forest, stage 8 (log weights, per-pixel features) | 0.916 | 0.855 | 0.917 | 0.996 | 0.902 | 0.934 |
| DeepLabv3-ResNet50 (11 bands, from scratch) | 0.662 | 0.565 | 0.598 | 0.974 | 0.633 | 0.841 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, no VSCP) | 0.858 | 0.779 | 0.875 | 0.994 | 0.822 | 0.909 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, VSCP) | 0.848 | 0.772 | 0.881 | 0.994 | 0.816 | 0.924 |
| SegNeXt-T (16 ch, from scratch, no VSCP) | 0.803 | 0.722 | 0.819 | 0.991 | 0.747 | 0.931 |
| SegNeXt-T (16 ch, from scratch, VSCP) | 0.800 | 0.720 | 0.822 | 0.991 | 0.746 | 0.914 |
| MariNeXt architecture (16 ch, from scratch, no VSCP) | 0.809 | 0.731 | 0.867 | 0.994 | 0.793 | 0.866 |
| MariNeXt architecture (16 ch, from scratch, VSCP) | 0.835 | 0.761 | 0.871 | 0.994 | 0.791 | 0.930 |
| U-Net + RF ensemble (RF stage 5) (U-Net novscp, w_unet = 0.45) | 0.920 | 0.862 | 0.924 | 0.997 | 0.901 | 0.942 |
| U-Net + RF ensemble (RF stage 8) (U-Net novscp, w_unet = 0.45) | **0.923** | **0.866** | **0.926** | **0.997** | **0.905** | **0.944** |

### Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| 📄 **RF, spectral (RF_SS)** (MARIDA paper, 11 classes) | – | _0.710_ | – | – |
| 📄 **RF + indices (RF_SS+SI)** (MARIDA paper, 11 classes) | – | _0.800_ | – | – |
| 📄 **RF + indices + GLCM** (MARIDA paper, 11 classes) | – | _0.790_ | – | – |
| 📄 **U-Net** (MARIDA paper, 11 classes) | – | _0.500_ | – | – |
| Random Forest, stage 5 (log weights, per-pixel features) | 0.998 | 0.813 | 0.988 | 0.818 |
| Random Forest, stage 8 (log weights, per-pixel features) | 0.998 | 0.831 | 0.987 | 0.850 |
| DeepLabv3-ResNet50 (11 bands, from scratch) | 0.989 | 0.177 | 0.801 | 0.683 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, no VSCP) | 0.997 | 0.649 | 0.992 | 0.791 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, VSCP) | 0.998 | 0.571 | 0.989 | 0.833 |
| SegNeXt-T (16 ch, from scratch, no VSCP) | 0.996 | 0.452 | 0.979 | 0.784 |
| SegNeXt-T (16 ch, from scratch, VSCP) | 0.996 | 0.444 | 0.980 | 0.782 |
| MariNeXt architecture (16 ch, from scratch, no VSCP) | 0.998 | 0.442 | 0.961 | 0.835 |
| MariNeXt architecture (16 ch, from scratch, VSCP) | 0.997 | 0.501 | 0.980 | 0.860 |
| U-Net + RF ensemble (RF stage 5) (U-Net novscp, w_unet = 0.45) | 0.998 | 0.823 | 0.995 | 0.864 |
| U-Net + RF ensemble (RF stage 8) (U-Net novscp, w_unet = 0.45) | **0.998** | **0.836** | **0.995** | **0.864** |

### Per-class Precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| Random Forest, stage 5 (log weights, per-pixel features) | 0.998 | 0.733 | **0.997** | 0.818 |
| Random Forest, stage 8 (log weights, per-pixel features) | 0.998 | 0.764 | 0.994 | 0.853 |
| DeepLabv3-ResNet50 (11 bands, from scratch) | 0.999 | 0.099 | 0.817 | 0.618 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, no VSCP) | 0.999 | 0.537 | 0.993 | 0.758 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, VSCP) | 0.999 | 0.418 | 0.985 | 0.862 |
| SegNeXt-T (16 ch, from scratch, no VSCP) | 0.999 | 0.306 | 0.986 | 0.697 |
| SegNeXt-T (16 ch, from scratch, VSCP) | 0.999 | 0.308 | 0.976 | 0.700 |
| MariNeXt architecture (16 ch, from scratch, no VSCP) | 0.999 | 0.318 | 0.973 | **0.881** |
| MariNeXt architecture (16 ch, from scratch, VSCP) | **0.999** | 0.355 | 0.970 | 0.840 |
| U-Net + RF ensemble (RF stage 5) (U-Net novscp, w_unet = 0.45) | 0.999 | 0.760 | 0.995 | 0.852 |
| U-Net + RF ensemble (RF stage 8) (U-Net novscp, w_unet = 0.45) | 0.999 | **0.783** | 0.996 | 0.841 |

### Per-class Recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| 📄 **RF, spectral (RF_SS)** (MARIDA paper, 11 classes) | – | _0.910_ | – | – |
| 📄 **RF + indices (RF_SS+SI)** (MARIDA paper, 11 classes) | – | _0.920_ | – | – |
| 📄 **RF + indices + GLCM** (MARIDA paper, 11 classes) | – | _0.920_ | – | – |
| 📄 **U-Net** (MARIDA paper, 11 classes) | – | _0.700_ | – | – |
| Random Forest, stage 5 (log weights, per-pixel features) | 0.998 | **0.913** | 0.980 | 0.817 |
| Random Forest, stage 8 (log weights, per-pixel features) | **0.998** | 0.911 | 0.980 | 0.846 |
| DeepLabv3-ResNet50 (11 bands, from scratch) | 0.979 | 0.837 | 0.786 | 0.763 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, no VSCP) | 0.996 | 0.822 | 0.991 | 0.828 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, VSCP) | 0.997 | 0.900 | 0.994 | 0.806 |
| SegNeXt-T (16 ch, from scratch, no VSCP) | 0.992 | 0.866 | 0.973 | **0.895** |
| SegNeXt-T (16 ch, from scratch, VSCP) | 0.992 | 0.795 | 0.985 | 0.884 |
| MariNeXt architecture (16 ch, from scratch, no VSCP) | 0.997 | 0.722 | 0.951 | 0.794 |
| MariNeXt architecture (16 ch, from scratch, VSCP) | 0.995 | 0.853 | 0.990 | 0.882 |
| U-Net + RF ensemble (RF stage 5) (U-Net novscp, w_unet = 0.45) | 0.998 | 0.898 | **0.995** | 0.877 |
| U-Net + RF ensemble (RF stage 8) (U-Net novscp, w_unet = 0.45) | 0.998 | 0.898 | 0.993 | 0.889 |

### Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| 📄 **RF, spectral (RF_SS)** (MARIDA paper, 11 classes) | – | _0.550_ | – | – |
| 📄 **RF + indices (RF_SS+SI)** (MARIDA paper, 11 classes) | – | _0.670_ | – | – |
| 📄 **RF + indices + GLCM** (MARIDA paper, 11 classes) | – | _0.650_ | – | – |
| 📄 **U-Net** (MARIDA paper, 11 classes) | – | _0.330_ | – | – |
| Random Forest, stage 5 (log weights, per-pixel features) | 0.996 | 0.685 | 0.977 | 0.691 |
| Random Forest, stage 8 (log weights, per-pixel features) | 0.997 | 0.711 | 0.974 | 0.739 |
| DeepLabv3-ResNet50 (11 bands, from scratch) | 0.978 | 0.097 | 0.668 | 0.518 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, no VSCP) | 0.995 | 0.481 | 0.985 | 0.655 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, VSCP) | 0.995 | 0.399 | 0.979 | 0.714 |
| SegNeXt-T (16 ch, from scratch, no VSCP) | 0.991 | 0.292 | 0.959 | 0.645 |
| SegNeXt-T (16 ch, from scratch, VSCP) | 0.992 | 0.286 | 0.961 | 0.641 |
| MariNeXt architecture (16 ch, from scratch, no VSCP) | 0.996 | 0.284 | 0.926 | 0.717 |
| MariNeXt architecture (16 ch, from scratch, VSCP) | 0.995 | 0.334 | 0.960 | 0.755 |
| U-Net + RF ensemble (RF stage 5) (U-Net novscp, w_unet = 0.45) | 0.997 | 0.699 | **0.990** | 0.761 |
| U-Net + RF ensemble (RF stage 8) (U-Net novscp, w_unet = 0.45) | **0.997** | **0.718** | 0.989 | **0.761** |

### MARIDA paper, F1 per original class (Table 4)

The MARIDA paper gives no confusion matrix, so its classes cannot be merged into ours. These are its per-class F1 scores, grouped by the class of ours they belong to.

| Our class | MARIDA class | RF_SS | RF_SS+SI | RF_SS+SI+GLCM | U-Net |
|---|---|---|---|---|---|
| Water | Marine Water | _0.79_ | _0.79_ | _0.86_ | _0.76_ |
| Water | Sediment-Laden Water | _1.00_ | _1.00_ | _1.00_ | _1.00_ |
| Water | Turbid Water | _0.85_ | _0.89_ | _0.94_ | _0.91_ |
| Water | Shallow Water | _0.16_ | _0.23_ | _0.46_ | _0.62_ |
| Marine Debris | Marine Debris | _0.71_ | _0.80_ | _0.79_ | _0.50_ |
| Algae/Sargassum | Dense Sargassum | _0.93_ | _0.93_ | _0.93_ | _0.75_ |
| Algae/Sargassum | Sparse Sargassum | _0.69_ | _0.82_ | _0.91_ | _0.79_ |
| Other | Natural Organic Material | _0.47_ | _0.29_ | _0.31_ | _0.04_ |
| Other | Ship | _0.70_ | _0.64_ | _0.80_ | _0.76_ |
| Other | Foam | _0.37_ | _0.47_ | _0.75_ | _0.71_ |
| ignored (class 0) | Clouds | _0.86_ | _0.85_ | _0.91_ | _0.76_ |


### Marine debris pixels and the water / other confusion

| Model | Debris found (TP) | Debris missed (FN) | False alarms (FP) | FP per TP | Water → Other | Other → Water |
|---|---|---|---|---|---|---|
| Random Forest, stage 5 (log weights, per-pixel features) | 348 / 381 | 33 | 127 | 0.36 | 272 (0.2 %) | 234 (14.5 %) |
| Random Forest, stage 8 (log weights, per-pixel features) | 347 / 381 | 34 | 107 | 0.31 | 219 (0.1 %) | 196 (12.2 %) |
| DeepLabv3-ResNet50 (11 bands, from scratch) | 319 / 381 | 62 | 2906 | 9.11 | 754 (0.5 %) | 77 (4.8 %) |
| U-Net (SSL4EO encoder) (16 ch, pretrained, no VSCP) | 313 / 381 | 68 | 270 | 0.86 | 407 (0.3 %) | 168 (10.4 %) |
| U-Net (SSL4EO encoder) (16 ch, pretrained, VSCP) | 343 / 381 | 38 | 478 | 1.39 | 205 (0.1 %) | 141 (8.8 %) |
| SegNeXt-T (16 ch, from scratch, no VSCP) | 330 / 381 | 51 | 749 | 2.27 | 620 (0.4 %) | 32 (2.0 %) |
| SegNeXt-T (16 ch, from scratch, VSCP) | 303 / 381 | 78 | 680 | 2.24 | 592 (0.4 %) | 66 (4.1 %) |
| MariNeXt architecture (16 ch, from scratch, no VSCP) | 275 / 381 | 106 | 589 | 2.14 | 153 (0.1 %) | 111 (6.9 %) |
| MariNeXt architecture (16 ch, from scratch, VSCP) | 325 / 381 | 56 | 591 | 1.82 | 254 (0.2 %) | 53 (3.3 %) |
| U-Net + RF ensemble (RF stage 5) (U-Net novscp, w_unet = 0.45) | 342 / 381 | 39 | 108 | 0.32 | 243 (0.2 %) | 144 (8.9 %) |
| U-Net + RF ensemble (RF stage 8) (U-Net novscp, w_unet = 0.45) | 342 / 381 | 39 | 95 | 0.28 | 265 (0.2 %) | 126 (7.8 %) |

## MADOS

### Overall

| Model | Macro F1 | mIoU | κ | OA | Macro P | Macro R |
|---|---|---|---|---|---|---|
| 📄 **MariNeXt** (MADOS paper, 15 classes) | _0.760_ | _0.643_ | – | _0.891_ | – | – |
| 📄 **SegNeXt-T** (MADOS paper, 15 classes) | _0.606_ | _0.492_ | – | _0.866_ | – | – |
| 📄 **U-Net** (MADOS paper, 15 classes) | _0.638_ | _0.510_ | – | _0.829_ | – | – |
| 📄 **RF++** (MADOS paper, 15 classes) | _0.644_ | _0.524_ | – | _0.838_ | – | – |
| 📄 **RF** (MADOS paper, 15 classes) | _0.566_ | _0.439_ | – | _0.671_ | – | – |
| Random Forest, stage 5 (log weights, per-pixel features) | 0.719 | 0.575 | 0.387 | 0.734 | 0.750 | 0.748 |
| Random Forest, stage 8 (log weights, per-pixel features) | 0.761 | 0.627 | 0.530 | 0.838 | 0.775 | 0.766 |
| DeepLabv3-ResNet50 (11 bands, from scratch) | 0.671 | 0.546 | 0.674 | 0.911 | 0.665 | 0.755 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, VSCP) | 0.820 | 0.722 | 0.720 | 0.923 | **0.831** | 0.837 |
| SegNeXt-T (16 ch, from scratch, VSCP) | 0.807 | 0.720 | **0.793** | **0.931** | 0.762 | **0.902** |
| MariNeXt architecture (16 ch, from scratch, VSCP) | 0.798 | 0.707 | 0.770 | 0.924 | 0.755 | 0.898 |
| U-Net + RF ensemble (RF stage 5) (U-Net vscp, w_unet = 0.55) | **0.838** | **0.736** | 0.657 | 0.887 | 0.827 | 0.854 |
| U-Net + RF ensemble (RF stage 8) (U-Net vscp, w_unet = 0.45) | 0.820 | 0.712 | 0.608 | 0.872 | 0.812 | 0.833 |

### Per-class F1

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| 📄 **MariNeXt** (MADOS paper, 15 classes) | – | _0.513_ | – | – |
| Random Forest, stage 5 (log weights, per-pixel features) | 0.817 | 0.717 | 0.825 | 0.517 |
| Random Forest, stage 8 (log weights, per-pixel features) | 0.898 | 0.708 | 0.829 | 0.608 |
| DeepLabv3-ResNet50 (11 bands, from scratch) | 0.950 | 0.332 | 0.685 | 0.717 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, VSCP) | 0.956 | 0.611 | 0.970 | 0.744 |
| SegNeXt-T (16 ch, from scratch, VSCP) | **0.959** | 0.470 | **0.973** | **0.826** |
| MariNeXt architecture (16 ch, from scratch, VSCP) | 0.955 | 0.461 | 0.970 | 0.805 |
| U-Net + RF ensemble (RF stage 5) (U-Net vscp, w_unet = 0.55) | 0.930 | **0.758** | 0.957 | 0.708 |
| U-Net + RF ensemble (RF stage 8) (U-Net vscp, w_unet = 0.45) | 0.921 | 0.748 | 0.946 | 0.666 |

### Per-class Precision

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| 📄 **MariNeXt** (MADOS paper, 15 classes) | – | _0.406_ | – | – |
| Random Forest, stage 5 (log weights, per-pixel features) | 0.949 | 0.684 | 0.990 | 0.379 |
| Random Forest, stage 8 (log weights, per-pixel features) | 0.935 | 0.659 | 0.980 | 0.528 |
| DeepLabv3-ResNet50 (11 bands, from scratch) | 0.920 | 0.223 | 0.596 | **0.921** |
| U-Net (SSL4EO encoder) (16 ch, pretrained, VSCP) | 0.927 | 0.503 | 0.984 | 0.909 |
| SegNeXt-T (16 ch, from scratch, VSCP) | **0.983** | 0.334 | 0.975 | 0.756 |
| MariNeXt architecture (16 ch, from scratch, VSCP) | 0.978 | 0.321 | 0.981 | 0.738 |
| U-Net + RF ensemble (RF stage 5) (U-Net vscp, w_unet = 0.55) | 0.951 | **0.720** | 0.989 | 0.649 |
| U-Net + RF ensemble (RF stage 8) (U-Net vscp, w_unet = 0.45) | 0.939 | 0.705 | **0.990** | 0.615 |

### Per-class Recall

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| 📄 **MariNeXt** (MADOS paper, 15 classes) | – | _0.704_ | – | – |
| Random Forest, stage 5 (log weights, per-pixel features) | 0.717 | 0.753 | 0.707 | 0.815 |
| Random Forest, stage 8 (log weights, per-pixel features) | 0.865 | 0.764 | 0.719 | 0.718 |
| DeepLabv3-ResNet50 (11 bands, from scratch) | 0.983 | 0.646 | 0.804 | 0.587 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, VSCP) | **0.987** | 0.777 | 0.956 | 0.629 |
| SegNeXt-T (16 ch, from scratch, VSCP) | 0.936 | 0.791 | **0.971** | **0.909** |
| MariNeXt architecture (16 ch, from scratch, VSCP) | 0.932 | **0.814** | 0.960 | 0.885 |
| U-Net + RF ensemble (RF stage 5) (U-Net vscp, w_unet = 0.55) | 0.911 | 0.802 | 0.927 | 0.778 |
| U-Net + RF ensemble (RF stage 8) (U-Net vscp, w_unet = 0.45) | 0.903 | 0.797 | 0.906 | 0.726 |

### Per-class IoU

| Model | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| 📄 **MariNeXt** (MADOS paper, 15 classes) | – | _0.346_ | – | – |
| Random Forest, stage 5 (log weights, per-pixel features) | 0.690 | 0.559 | 0.701 | 0.349 |
| Random Forest, stage 8 (log weights, per-pixel features) | 0.815 | 0.548 | 0.708 | 0.437 |
| DeepLabv3-ResNet50 (11 bands, from scratch) | 0.906 | 0.199 | 0.521 | 0.559 |
| U-Net (SSL4EO encoder) (16 ch, pretrained, VSCP) | 0.915 | 0.440 | 0.941 | 0.592 |
| SegNeXt-T (16 ch, from scratch, VSCP) | **0.921** | 0.307 | **0.947** | **0.703** |
| MariNeXt architecture (16 ch, from scratch, VSCP) | 0.913 | 0.299 | 0.942 | 0.674 |
| U-Net + RF ensemble (RF stage 5) (U-Net vscp, w_unet = 0.55) | 0.870 | **0.611** | 0.918 | 0.548 |
| U-Net + RF ensemble (RF stage 8) (U-Net vscp, w_unet = 0.45) | 0.853 | 0.597 | 0.898 | 0.499 |

### Marine debris pixels and the water / other confusion

| Model | Debris found (TP) | Debris missed (FN) | False alarms (FP) | FP per TP | Water → Other | Other → Water |
|---|---|---|---|---|---|---|
| Random Forest, stage 5 (log weights, per-pixel features) | 733 / 973 | 240 | 338 | 0.46 | 83,130 (28.3 %) | 11,354 (18.0 %) |
| Random Forest, stage 8 (log weights, per-pixel features) | 743 / 973 | 230 | 384 | 0.52 | 39,748 (13.5 %) | 17,433 (27.6 %) |
| DeepLabv3-ResNet50 (11 bands, from scratch) | 629 / 973 | 344 | 2188 | 3.48 | 2,940 (1.0 %) | 24,658 (39.1 %) |
| U-Net (SSL4EO encoder) (16 ch, pretrained, VSCP) | 756 / 973 | 217 | 746 | 0.99 | 3,815 (1.3 %) | 22,761 (36.1 %) |
| SegNeXt-T (16 ch, from scratch, VSCP) | 770 / 973 | 203 | 1536 | 1.99 | 18,319 (6.2 %) | 4,646 (7.4 %) |
| MariNeXt architecture (16 ch, from scratch, VSCP) | 792 / 973 | 181 | 1673 | 2.11 | 19,640 (6.7 %) | 5,965 (9.5 %) |
| U-Net + RF ensemble (RF stage 5) (U-Net vscp, w_unet = 0.55) | 780 / 973 | 193 | 304 | 0.39 | 26,215 (8.9 %) | 13,723 (21.8 %) |
| U-Net + RF ensemble (RF stage 8) (U-Net vscp, w_unet = 0.45) | 775 / 973 | 198 | 325 | 0.42 | 28,407 (9.7 %) | 17,007 (27.0 %) |

