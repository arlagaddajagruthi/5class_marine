# Marine debris detection on MARIDA and MADOS (4 classes)

Pixel classification of Sentinel-2 patches from the MARIDA and MADOS benchmarks,
with both benchmarks harmonised to four classes:

| Class | MARIDA | MADOS |
|---|---|---|
| 1 Water | marine, sediment-laden, turbid, shallow and mixed water; waves; wakes | marine, sediment-laden, turbid and shallow water; waves & wakes |
| 2 Marine debris | marine debris | marine debris |
| 3 Algae/Sargassum | dense and sparse Sargassum | dense Sargassum, sparse floating algae |
| 4 Other | natural organic material, ship, foam | natural organic material, ship, foam, oil spill, oil platform, jellyfish, sea snot |
| 0 ignored | unlabelled, cloud, cloud shadow | unlabelled |

All models are trained on the official, scene-disjoint train splits and
evaluated on every annotated pixel of the official test splits. Every report
gives overall and per-class precision, recall, F1, IoU and accuracy, plus kappa.

## Layout

| Path | What it is |
|---|---|
| `data/` | the 4-class datasets and the environmental covariates (see `data/README.md`) |
| `src/datasets/`, `src/features/`, `src/metrics/`, `src/environmental/` | dataset loaders, per-pixel features, metrics, environmental-data helpers |
| `scripts/utilities/convert.py` | builds `data/*_5Class` from the original MARIDA/MADOS (15 classes → 4) |
| `scripts/staged_ml/` | **staged feature ablation** (RF, XGBoost, LightGBM, AdaBoost, SVM; stages 1–8) |
| `scripts/env_sst_chl_wind/` | SST, chlorophyll-a and wind covariates for stage 6 (MARIDA) |
| `scripts/environmental/` | patch metadata, MUR SST and ERA5 downloads; first environmental experiments |
| `scripts/deeplabv3_resnet50/` | DeepLabv3-ResNet50; also the shared training code for the models below |
| `scripts/unet_ssl4eo/` | U-Net with an SSL4EO-S12 pretrained ResNet-50 encoder |
| `scripts/segnext/` | SegNeXt-T and the MariNeXt architecture (`--first-stage`) |
| `scripts/ensemble/` | U-Net + Random Forest probability ensemble |
| `scripts/resnet_paper/` | MARIDA's multi-label ResNet baseline (patch classifier) |
| `outputs/<pipeline>/` | results of each pipeline, with `logs/` holding the runner scripts |
| `outputs/paper_compare/` | MADOS paper's MariNeXt results collapsed to our 4 classes |
| `paper/main.tex` | paper 1: staged feature ablation (ML only) |
| `paper/dl_main.tex` | paper 2: deep-learning models compared with the MARIDA and MADOS papers |
| `MARIDA.pdf`, `MADOS.pdf` | the two benchmark papers |

Each `scripts/<pipeline>/` folder has its own README with run commands and
outputs.

## Quick start (from the repo root)

```powershell
python scripts/utilities/check_cuda.py
python scripts/utilities/convert.py                       # once: data/MARIDA, data/MADOS -> data/*_5Class
powershell -File scripts/staged_ml/run_all_stages.ps1     # staged ML, all stages and weightings
python scripts/unet_ssl4eo/unet.py train --dataset MADOS --run MADOS_vscp --vscp
python scripts/unet_ssl4eo/unet.py test  --dataset MADOS --run MADOS_vscp --vscp
python scripts/ensemble/unet_rf.py --datasets MARIDA MADOS
```

Main packages: torch (CUDA), torchgeo, segmentation-models-pytorch, timm,
scikit-learn, xgboost, lightgbm, numba, scikit-image, rasterio, tifffile,
pandas; copernicusmarine and cdsapi for the environmental downloads.
Developed on a 4 GB RTX 2050 laptop.

## Results on the test splits (macro F1 / debris F1)

| Model | MARIDA | MADOS |
|---|---|---|
| RF, stage 5 (bands + NDWI + 4 ratios, log weights) | 0.904 / 0.813 | 0.719 / 0.717 |
| RF, stage 8 (stage 5 + GLCM texture) | 0.916 / 0.831 | 0.761 / 0.708 |
| DeepLabv3-ResNet50 | 0.662 / 0.177 | 0.671 / 0.332 |
| U-Net, SSL4EO encoder | 0.848 / 0.571 | 0.820 / 0.611 |
| SegNeXt-T | 0.800 / 0.444 | 0.807 / 0.470 |
| MariNeXt architecture | 0.835 / 0.501 | 0.798 / 0.461 |
| **U-Net + RF ensemble** | **0.923 / 0.836** (RF stage 8) | **0.838 / 0.758** (RF stage 5) |
| MADOS paper, MariNeXt (collapsed to 4 classes) | – | 0.814 / 0.509 |

The deep models' MARIDA rows use the VSCP runs, which the validation split
preferred for all three architectures (without VSCP: U-Net 0.858 / 0.649,
SegNeXt 0.803 / 0.452, MariNeXt architecture 0.809 / 0.442). Per-class results are in each pipeline's reports.

Main findings:
- NDWI and simple band ratios carry most of the added information; FAI, FDI,
  NDVI and NDMI add little.
- GLCM texture hurts debris when added to the bands alone, but on top of NDWI
  and the ratios it gives the best RF.
- Log class weights (1/ln(1.03 + p)) beat both balanced and no weights.
- Patch-level environmental covariates identify the scene rather than the
  conditions, and reduce performance on unseen scenes.
- On MADOS, the ensemble beats the MADOS paper's MariNeXt on macro F1, mIoU and
  debris F1, but not on kappa or overall accuracy (water/other confusion).
