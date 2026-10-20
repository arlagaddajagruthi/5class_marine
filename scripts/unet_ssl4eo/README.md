# U-Net with an SSL4EO-S12 pretrained ResNet-50 encoder, 4 classes

`unet.py` trains a segmentation_models_pytorch U-Net whose ResNet-50 encoder is
initialised from the SSL4EO-S12 MoCo Sentinel-2 weights (torchgeo
`ResNet50_Weights.SENTINEL2_ALL_MOCO`). It predicts water, debris,
algae/sargassum or other for every annotated pixel; unannotated pixels are
ignored, as in the ML experiments.

- Input: 16 channels: the 11 bands as reflectance plus NDWI and the 4 band
  ratios of the staged ML (`scripts/staged_ml/`).
- Recipe per dataset as in `scripts/deeplabv3_resnet50/deeplab.py` (which this
  script imports): MARIDA 45 epochs, MADOS 80 epochs with EMA; Adam 2e-4,
  batch 5, weighted CE 1/ln(1.03 + p_c).
- Checkpoint chosen by validation macro F1; 8-view test-time augmentation.
- VSCP copy-paste on or off per run (`--vscp` / `--no-vscp`).

The docstring of `unet.py` lists every setting.

## Run (from the repo root)
```bash
python scripts/unet_ssl4eo/unet.py check --dataset MARIDA --run MARIDA_vscp --vscp
python scripts/unet_ssl4eo/unet.py train --dataset MARIDA --run MARIDA_vscp --vscp   # add --resume after an interruption
python scripts/unet_ssl4eo/unet.py test  --dataset MARIDA --run MARIDA_vscp --vscp
```
Runs used: `MARIDA_novscp`, `MARIDA_vscp`, `MADOS_vscp`.
`bash outputs/unet_ssl4eo/logs/run_all.sh` runs all three.

## Outputs: `outputs/unet_ssl4eo/<run>/`
- `recipe_and_weights.json`, `extra_channel_stats.json`
- `history.csv`, `best_epoch.json`, `log.txt`
- `model_best_val_f1.pth`, `model_last.pth`
- `test_report.md`, `test_metrics.csv`, `classwise_*.csv`, `confusion_*.csv`
  (counts, row % and column %), for test and validation

## Test results (macro F1 / debris F1)

| Run | macro F1 | mIoU | kappa | debris F1 |
|---|---|---|---|---|
| MARIDA_novscp | 0.858 | 0.779 | 0.875 | 0.649 |
| MARIDA_vscp | 0.848 | 0.772 | 0.881 | 0.571 |
| MADOS_vscp | 0.820 | 0.722 | 0.720 | 0.611 |

Full per-class results are in each `test_report.md`.
