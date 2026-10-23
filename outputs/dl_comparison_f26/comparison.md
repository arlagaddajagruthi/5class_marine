# Deep-learning comparison, 26 features

Test split, every annotated pixel, 4 classes (Water, Marine Debris, Algae/Sargassum, Other); macro = mean over the 4 classes. Each cell: score [95% scene-bootstrap CI where computed] (Δ vs the U-Net, R3, same dataset). All models use the same 26 input features (11 bands, NDVI, NDWI, FAI, FDI, NDMI, 4 ratios, 6 GLCM); R7 = R3 + the Random Forest (R6) on the same features, weight chosen on validation.

Training recipes follow each dataset's paper: MARIDA 45 epochs, no VSCP; MADOS 80 epochs, EMA, VSCP copy-paste. Single seed; the CIs cover test-scene variation, not training randomness.

📄 **Paper rows** (top of each table, values in _italics_) are the published results of the MARIDA (Table 4) and MADOS (Tables 4 and 5) papers. They use the papers' own class schemes (MARIDA 11 classes, MADOS 15 classes), so only marine debris is the same class as ours; their macro scores average over more, and different, classes. Tables without a published value have no paper rows. **Bold** = best of our models.

## MADOS

### Macro F1

| Model | score |
|---|---|
| 📄 **MADOS paper: RF** (15 classes) | _0.566_ |
| 📄 **MADOS paper: RF++ (Mikeli et al. 2022)** (15 classes) | _0.644_ |
| 📄 **MADOS paper: U-Net** (15 classes) | _0.638_ |
| 📄 **MADOS paper: SegNeXt-T** (15 classes) | _0.606_ |
| 📄 **MADOS paper: MariNeXt** (15 classes) | _0.760_ |
| U-Net (R3, from scratch) | 0.7478 [0.668–0.826] |
| SegNeXt-T (R4) | 0.7497 [0.673–0.838] (+0.0020) |
| MariNeXt arch. (R5) | 0.7619 [0.682–0.858] (+0.0141) |
| U-Net + RF ensemble (R7) | **0.7867 [0.704–0.869] (+0.0389)** |

### Macro IoU

| Model | score |
|---|---|
| 📄 **MADOS paper: RF** (15 classes) | _0.439_ |
| 📄 **MADOS paper: RF++ (Mikeli et al. 2022)** (15 classes) | _0.524_ |
| 📄 **MADOS paper: U-Net** (15 classes) | _0.510_ |
| 📄 **MADOS paper: SegNeXt-T** (15 classes) | _0.492_ |
| 📄 **MADOS paper: MariNeXt** (15 classes) | _0.643_ |
| U-Net (R3, from scratch) | 0.6279 [0.537–0.728] |
| SegNeXt-T (R4) | 0.6465 [0.560–0.754] (+0.0186) |
| MariNeXt arch. (R5) | 0.6632 [0.573–0.786] (+0.0353) |
| U-Net + RF ensemble (R7) | **0.6677 [0.568–0.785] (+0.0398)** |

### Cohen's κ

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.5354 |
| SegNeXt-T (R4) | 0.6704 (+0.1350) |
| MariNeXt arch. (R5) | **0.7029 (+0.1675)** |
| U-Net + RF ensemble (R7) | 0.5348 (-0.0006) |

### F1 Marine Debris

| Model | score |
|---|---|
| 📄 **MADOS paper: MariNeXt** (15 classes) | _0.513_ |
| U-Net (R3, from scratch) | 0.5437 [0.311–0.733] |
| SegNeXt-T (R4) | 0.3976 [0.178–0.643] (-0.1461) |
| MariNeXt arch. (R5) | 0.4015 [0.195–0.630] (-0.1422) |
| U-Net + RF ensemble (R7) | **0.7077 [0.456–0.882] (+0.1640)** |

### IoU Marine Debris

| Model | score |
|---|---|
| 📄 **MADOS paper: MariNeXt** (15 classes) | _0.346_ |
| U-Net (R3, from scratch) | 0.3733 |
| SegNeXt-T (R4) | 0.2481 (-0.1252) |
| MariNeXt arch. (R5) | 0.2512 (-0.1222) |
| U-Net + RF ensemble (R7) | **0.5476 (+0.1743)** |

### F1 Algae/Sargassum

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.9381 |
| SegNeXt-T (R4) | 0.9472 (+0.0091) |
| MariNeXt arch. (R5) | **0.9577 (+0.0196)** |
| U-Net + RF ensemble (R7) | 0.9303 (-0.0078) |

### IoU Algae/Sargassum

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.8834 |
| SegNeXt-T (R4) | 0.8996 (+0.0162) |
| MariNeXt arch. (R5) | **0.9189 (+0.0355)** |
| U-Net + RF ensemble (R7) | 0.8696 (-0.0138) |

### F1 Other

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.6118 |
| SegNeXt-T (R4) | 0.7214 (+0.1096) |
| MariNeXt arch. (R5) | **0.7521 (+0.1403)** |
| U-Net + RF ensemble (R7) | 0.6123 (+0.0005) |

### Class accuracy Water (TP / class pixels)

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.8596 |
| SegNeXt-T (R4) | **0.9065 (+0.0469)** |
| MariNeXt arch. (R5) | 0.8956 (+0.0360) |
| U-Net + RF ensemble (R7) | 0.8578 (-0.0018) |

### Class accuracy Marine Debris (TP / class pixels)

| Model | score |
|---|---|
| 📄 **MADOS paper: MariNeXt** (15 classes) | _0.704_ |
| U-Net (R3, from scratch) | 0.7801 |
| SegNeXt-T (R4) | 0.7533 (-0.0267) |
| MariNeXt arch. (R5) | **0.8232 (+0.0432)** |
| U-Net + RF ensemble (R7) | 0.8037 (+0.0236) |

### Class accuracy Algae/Sargassum (TP / class pixels)

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.9003 |
| SegNeXt-T (R4) | **0.9660 (+0.0657)** |
| MariNeXt arch. (R5) | 0.9466 (+0.0463) |
| U-Net + RF ensemble (R7) | 0.8789 (-0.0214) |

### Class accuracy Other (TP / class pixels)

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.7293 |
| SegNeXt-T (R4) | 0.8054 (+0.0761) |
| MariNeXt arch. (R5) | **0.8921 (+0.1628)** |
| U-Net + RF ensemble (R7) | 0.7355 (+0.0063) |

### Feature importance by group (share of \|SHAP\|, %), MADOS

Deep models: GradientSHAP; R7: weighted mix of its members' shares (approximation). Per-feature values in `outputs/mados_f26/xai/<run>/`.

| class | group | R3 | R4 | R5 | R7 |
|---|---|---|---|---|---|
| Water | bands | 29.2 | 24.8 | 28.0 | 31.1 |
| Water | indices | 26.1 | 28.7 | 25.8 | 25.2 |
| Water | ratios | 21.0 | 18.6 | 17.4 | 18.0 |
| Water | GLCM | 23.6 | 28.0 | 28.8 | 25.6 |
| Marine Debris | bands | 12.4 | 19.9 | 21.4 | 20.8 |
| Marine Debris | indices | 50.0 | 36.8 | 37.7 | 44.4 |
| Marine Debris | ratios | 27.4 | 16.9 | 18.8 | 25.7 |
| Marine Debris | GLCM | 10.2 | 26.5 | 22.2 | 9.0 |
| Algae/Sargassum | bands | 6.4 | 13.8 | 12.5 | 10.9 |
| Algae/Sargassum | indices | 74.6 | 56.3 | 55.0 | 62.4 |
| Algae/Sargassum | ratios | 13.4 | 15.5 | 16.8 | 22.2 |
| Algae/Sargassum | GLCM | 5.6 | 14.3 | 15.7 | 4.4 |
| Other | bands | 43.9 | 30.2 | 32.0 | 40.8 |
| Other | indices | 17.7 | 20.2 | 17.4 | 19.4 |
| Other | ratios | 20.0 | 19.5 | 17.2 | 18.2 |
| Other | GLCM | 18.4 | 30.0 | 33.4 | 21.6 |

![delta macro F1 MADOS](delta_f1_MADOS.png) ![delta debris F1 MADOS](delta_debris_f1_MADOS.png)

## MARIDA

### Macro F1

| Model | score |
|---|---|
| 📄 **MARIDA paper: RF_SS** (11 classes) | _0.690_ |
| 📄 **MARIDA paper: RF_SS+SI** (11 classes) | _0.700_ |
| 📄 **MARIDA paper: RF_SS+SI+GLCM** (11 classes) | _0.790_ |
| 📄 **MARIDA paper: U-Net** (11 classes) | _0.690_ |
| U-Net (R3, from scratch) | 0.8536 [0.780–0.896] |
| SegNeXt-T (R4) | 0.7368 [0.691–0.780] (-0.1167) |
| MariNeXt arch. (R5) | 0.8025 [0.753–0.838] (-0.0510) |
| U-Net + RF ensemble (R7) | **0.9186 [0.859–0.954] (+0.0651)** |

### Macro IoU

| Model | score |
|---|---|
| 📄 **MARIDA paper: RF_SS** (11 classes) | _0.570_ |
| 📄 **MARIDA paper: RF_SS+SI** (11 classes) | _0.590_ |
| 📄 **MARIDA paper: RF_SS+SI+GLCM** (11 classes) | _0.690_ |
| 📄 **MARIDA paper: U-Net** (11 classes) | _0.570_ |
| U-Net (R3, from scratch) | 0.7751 [0.668–0.832] |
| SegNeXt-T (R4) | 0.6578 [0.577–0.692] (-0.1173) |
| MariNeXt arch. (R5) | 0.7206 [0.639–0.757] (-0.0545) |
| U-Net + RF ensemble (R7) | **0.8590 [0.772–0.917] (+0.0839)** |

### Cohen's κ

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.8806 |
| SegNeXt-T (R4) | 0.7120 (-0.1687) |
| MariNeXt arch. (R5) | 0.8183 (-0.0624) |
| U-Net + RF ensemble (R7) | **0.9245 (+0.0438)** |

### F1 Marine Debris

| Model | score |
|---|---|
| 📄 **MARIDA paper: RF_SS** (11 classes) | _0.710_ |
| 📄 **MARIDA paper: RF_SS+SI** (11 classes) | _0.800_ |
| 📄 **MARIDA paper: RF_SS+SI+GLCM** (11 classes) | _0.790_ |
| 📄 **MARIDA paper: U-Net** (11 classes) | _0.500_ |
| U-Net (R3, from scratch) | 0.6087 [0.524–0.709] |
| SegNeXt-T (R4) | 0.2753 [0.173–0.441] (-0.3334) |
| MariNeXt arch. (R5) | 0.4575 [0.351–0.584] (-0.1512) |
| U-Net + RF ensemble (R7) | **0.8203 [0.752–0.892] (+0.2116)** |

### IoU Marine Debris

| Model | score |
|---|---|
| 📄 **MARIDA paper: RF_SS** (11 classes) | _0.550_ |
| 📄 **MARIDA paper: RF_SS+SI** (11 classes) | _0.670_ |
| 📄 **MARIDA paper: RF_SS+SI+GLCM** (11 classes) | _0.650_ |
| 📄 **MARIDA paper: U-Net** (11 classes) | _0.330_ |
| U-Net (R3, from scratch) | 0.4375 |
| SegNeXt-T (R4) | 0.1596 (-0.2779) |
| MariNeXt arch. (R5) | 0.2966 (-0.1409) |
| U-Net + RF ensemble (R7) | **0.6953 (+0.2578)** |

### F1 Algae/Sargassum

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.9809 |
| SegNeXt-T (R4) | 0.9631 (-0.0179) |
| MariNeXt arch. (R5) | 0.9771 (-0.0038) |
| U-Net + RF ensemble (R7) | **0.9911 (+0.0102)** |

### IoU Algae/Sargassum

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.9626 |
| SegNeXt-T (R4) | 0.9288 (-0.0338) |
| MariNeXt arch. (R5) | 0.9553 (-0.0072) |
| U-Net + RF ensemble (R7) | **0.9824 (+0.0198)** |

### F1 Other

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.8270 |
| SegNeXt-T (R4) | 0.7174 (-0.1096) |
| MariNeXt arch. (R5) | 0.7798 (-0.0472) |
| U-Net + RF ensemble (R7) | **0.8646 (+0.0375)** |

### Class accuracy Water (TP / class pixels)

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.9959 |
| SegNeXt-T (R4) | 0.9838 (-0.0121) |
| MariNeXt arch. (R5) | 0.9918 (-0.0042) |
| U-Net + RF ensemble (R7) | **0.9979 (+0.0020)** |

### Class accuracy Marine Debris (TP / class pixels)

| Model | score |
|---|---|
| 📄 **MARIDA paper: RF_SS** (11 classes) | _0.910_ |
| 📄 **MARIDA paper: RF_SS+SI** (11 classes) | _0.920_ |
| 📄 **MARIDA paper: RF_SS+SI+GLCM** (11 classes) | _0.920_ |
| 📄 **MARIDA paper: U-Net** (11 classes) | _0.700_ |
| U-Net (R3, from scratch) | 0.8635 |
| SegNeXt-T (R4) | 0.8189 (-0.0446) |
| MariNeXt arch. (R5) | 0.8898 (+0.0262) |
| U-Net + RF ensemble (R7) | **0.9344 (+0.0709)** |

### Class accuracy Algae/Sargassum (TP / class pixels)

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.9714 |
| SegNeXt-T (R4) | **0.9933 (+0.0219)** |
| MariNeXt arch. (R5) | 0.9902 (+0.0189) |
| U-Net + RF ensemble (R7) | 0.9854 (+0.0140) |

### Class accuracy Other (TP / class pixels)

| Model | score |
|---|---|
| U-Net (R3, from scratch) | 0.8863 |
| SegNeXt-T (R4) | **0.9019 (+0.0155)** |
| MariNeXt arch. (R5) | 0.8888 (+0.0025) |
| U-Net + RF ensemble (R7) | 0.8901 (+0.0037) |

### Feature importance by group (share of \|SHAP\|, %), MARIDA

Deep models: GradientSHAP; R7: weighted mix of its members' shares (approximation). Per-feature values in `outputs/marida_f26/xai/<run>/`.

| class | group | R3 | R4 | R5 | R7 |
|---|---|---|---|---|---|
| Water | bands | 20.2 | 23.8 | 25.3 | 21.2 |
| Water | indices | 53.5 | 34.5 | 34.5 | 39.9 |
| Water | ratios | 14.6 | 22.2 | 22.6 | 18.5 |
| Water | GLCM | 11.6 | 19.5 | 17.6 | 20.5 |
| Marine Debris | bands | 7.3 | 11.2 | 11.3 | 19.4 |
| Marine Debris | indices | 53.2 | 36.6 | 46.7 | 44.6 |
| Marine Debris | ratios | 35.6 | 26.3 | 23.9 | 29.6 |
| Marine Debris | GLCM | 3.9 | 25.9 | 18.1 | 6.5 |
| Algae/Sargassum | bands | 5.6 | 13.4 | 14.0 | 9.5 |
| Algae/Sargassum | indices | 77.3 | 57.5 | 55.0 | 59.1 |
| Algae/Sargassum | ratios | 9.6 | 13.6 | 13.5 | 26.7 |
| Algae/Sargassum | GLCM | 7.5 | 15.4 | 17.4 | 4.7 |
| Other | bands | 51.3 | 22.8 | 31.5 | 35.7 |
| Other | indices | 13.0 | 12.6 | 12.4 | 10.9 |
| Other | ratios | 11.5 | 11.3 | 11.2 | 9.0 |
| Other | GLCM | 24.3 | 53.3 | 44.9 | 44.4 |

![delta macro F1 MARIDA](delta_f1_MARIDA.png) ![delta debris F1 MARIDA](delta_debris_f1_MARIDA.png)

