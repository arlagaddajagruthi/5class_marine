# DeepLabv3-ResNet50, pixel-level, 4 classes

`deeplab.py` trains torchvision's DeepLabv3-ResNet50 from scratch on 11 bands.
It predicts one class for every annotated pixel: water, debris,
algae/sargassum or other. Unannotated pixels are ignored, as in both papers
and in the ML experiments, so the scores are comparable with RF and XGBoost.

Neither paper used DeepLabv3. Each dataset is trained with its own paper's
official training procedure:
- **MARIDA**: the MARIDA U-Net recipe: weighted CE with c = 1.03, Adam 2e-4,
  45 epochs, batch 5, learning-rate drop at epoch 40, rotation + flip, early
  stopping on validation loss.
- **MADOS**: the MADOS/MariNeXt recipe: weighted CE with c = 1.03, Adam 2e-4,
  80 epochs, batch 5, learning-rate drops at 45 and 65, flip/rotation, VSCP
  copy-paste, EMA weights, and 8-view test-time augmentation.

The docstring of `deeplab.py` lists every setting and its source file.

## Run (from the repo root)
```bash
python scripts/deeplabv3_resnet50/deeplab.py cache --dataset MADOS   # stacks MADOS band files once (~7 GB, outputs/deeplabv3_resnet50/cache/)
python scripts/deeplabv3_resnet50/deeplab.py check --dataset MARIDA  # GPU memory and speed
python scripts/deeplabv3_resnet50/deeplab.py train --dataset MARIDA
python scripts/deeplabv3_resnet50/deeplab.py test  --dataset MARIDA
```
`outputs/deeplabv3_resnet50/logs/run.sh` runs everything in that order (MARIDA,
then MADOS) once the ResNet run has finished.

## Outputs: `outputs/deeplabv3_resnet50/<dataset>/`
- `recipe_and_weights.json`
- `history.csv`: per epoch
- `best_epoch.json`
- `model_best_val_loss.pth`, `model_last.pth`
- `test_report.md`, `test_metrics.csv`, `confusion_*.csv`
- `log.txt`
