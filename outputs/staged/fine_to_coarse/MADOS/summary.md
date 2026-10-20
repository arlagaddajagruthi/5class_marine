# Fine-to-coarse training, MADOS (Random Forest, stage-5 features)

Features (16): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDWI, red/green, nir/red, nir/green, swir1/nir

All variants are scored on the same 4-class test labels (annotated pixels of the official test split). Fine models predict an original class, which is then merged to the 4 classes.

- coarse: trained on the 4 merged classes, log weights (identical to stage 5, log weights)
- fine_logw: trained on the original classes, log weights over the original classes
- fine_coarsew: trained on the original classes, each pixel weighted as its merged class

## Overall

| variant | train classes | macro F1 | IoU | precision | recall | accuracy | kappa |
|---|---|---|---|---|---|---|---|
| coarse | 4 | 0.7190 | 0.5750 | 0.7504 | 0.7480 | 0.7342 | 0.3867 |
| fine_logw | 15 | 0.7160 | 0.5721 | 0.7293 | 0.7711 | 0.7330 | 0.3922 |
| fine_coarsew | 15 | 0.7161 | 0.5719 | 0.7320 | 0.7668 | 0.7360 | 0.3946 |

## Per-class accuracy (TP / class pixels = recall)

| variant | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| coarse | 0.7171 | 0.7533 | 0.7067 | 0.8150 |
| fine_logw | 0.7108 (-0.0064) | 0.8047 (+0.0514) | 0.7336 (+0.0269) | 0.8353 (+0.0203) |
| fine_coarsew | 0.7159 (-0.0012) | 0.7986 (+0.0452) | 0.7229 (+0.0162) | 0.8297 (+0.0147) |

## Per-class precision

| variant | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| coarse | 0.9487 | 0.6844 | 0.9896 | 0.3790 |
| fine_logw | 0.9544 (+0.0057) | 0.5982 (-0.0862) | 0.9844 (-0.0052) | 0.3802 (+0.0012) |
| fine_coarsew | 0.9532 (+0.0045) | 0.6051 (-0.0793) | 0.9872 (-0.0024) | 0.3826 (+0.0036) |

## Per-class recall

| variant | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| coarse | 0.7171 | 0.7533 | 0.7067 | 0.8150 |
| fine_logw | 0.7108 (-0.0064) | 0.8047 (+0.0514) | 0.7336 (+0.0269) | 0.8353 (+0.0203) |
| fine_coarsew | 0.7159 (-0.0012) | 0.7986 (+0.0452) | 0.7229 (+0.0162) | 0.8297 (+0.0147) |

## Per-class F1

| variant | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| coarse | 0.8168 | 0.7172 | 0.8246 | 0.5174 |
| fine_logw | 0.8148 (-0.0020) | 0.6862 (-0.0310) | 0.8407 (+0.0161) | 0.5225 (+0.0051) |
| fine_coarsew | 0.8176 (+0.0009) | 0.6885 (-0.0287) | 0.8346 (+0.0101) | 0.5237 (+0.0063) |

## Per-class IoU

| variant | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| coarse | 0.6903 | 0.5591 | 0.7015 | 0.3490 |
| fine_logw | 0.6874 (-0.0029) | 0.5223 (-0.0368) | 0.7251 (+0.0236) | 0.3537 (+0.0047) |
| fine_coarsew | 0.6915 (+0.0012) | 0.5250 (-0.0341) | 0.7162 (+0.0147) | 0.3547 (+0.0057) |

## Per-class one-vs-rest accuracy ((TP + TN) / all pixels)

| variant | Water | Marine Debris | Algae/Sargassum | Other |
|---|---|---|---|---|
| coarse | 0.7381 | 0.9984 | 0.9974 | 0.7345 |
| fine_logw | 0.7369 (-0.0012) | 0.9980 (-0.0004) | 0.9976 (+0.0002) | 0.7334 (-0.0011) |
| fine_coarsew | 0.7400 (+0.0019) | 0.9981 (-0.0003) | 0.9975 (+0.0001) | 0.7364 (+0.0019) |

Values in brackets: change from the coarse model.

## Original classes in the test split: % of pixels given the correct merged class

| original class | merged into | test px | coarse | fine_logw | fine_coarsew | coarse model's main error |
|---|---|---|---|---|---|---|
| Marine Debris | Marine Debris | 973 | 75.3 | 80.5 | 79.9 | Other 20.7% |
| Dense Sargassum | Algae/Sargassum | 1,303 | 94.6 | 98.5 | 98.3 | Other 5.4% |
| Sparse Floating Algae | Algae/Sargassum | 1,786 | 53.2 | 55.0 | 53.3 | Other 45.6% |
| Natural Organic Material | Other | 278 | 91.0 | 90.6 | 90.6 | Marine Debris 4.7% |
| Ship | Other | 3,574 | 96.1 | 94.9 | 94.9 | Water 3.3% |
| Oil Spill | Other | 51,528 | 78.7 | 81.5 | 81.0 | Water 21.3% |
| Marine Water | Water | 139,465 | 48.9 | 48.0 | 49.0 | Other 51.1% |
| Sediment-Laden Water | Water | 93,254 | 100.0 | 100.0 | 100.0 | Other 0.0% |
| Foam | Other | 387 | 81.9 | 83.5 | 81.9 | Water 18.1% |
| Turbid Water | Water | 42,168 | 79.8 | 79.8 | 79.8 | Other 20.2% |
| Shallow Water | Water | 12,732 | 91.9 | 90.4 | 90.9 | Other 8.1% |
| Waves & Wakes | Water | 6,401 | 63.6 | 56.4 | 58.3 | Other 35.9% |
| Oil Platform | Other | 4,790 | 99.6 | 99.7 | 99.5 | Water 0.4% |
| Jellyfish | Other | 274 | 97.4 | 92.3 | 94.2 | Marine Debris 2.6% |
| Sea snarl | Other | 2,234 | 80.8 | 75.4 | 72.8 | Marine Debris 11.5% |
