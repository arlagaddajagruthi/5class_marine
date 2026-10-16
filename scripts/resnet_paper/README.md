# ResNet baseline as in the MARIDA paper (4-class remapping)

`resnet_ml.py` re-runs the ResNet baseline from Kikaki et al. 2022 (MARIDA,
PLOS ONE). It follows the official code in
[marine-debris.github.io](https://github.com/marine-debris/marine-debris.github.io),
`multi-label/resnet/`. The only change is the classes: water, debris,
algae/sargassum and other, instead of MARIDA's 11.

**This ResNet is a patch classifier, not a pixel classifier.** For each
256×256 patch (240×240 for MADOS) it predicts which classes appear anywhere in
the patch. Its scores therefore can't be compared directly with the pixel F1 of
RF, SVM, XGBoost, LightGBM or AdaBoost.

The MADOS paper and its code (MariNeXt) have no ResNet. MADOS is run here with
the same MARIDA ResNet recipe, using MADOS's own band mean and standard
deviation from the MADOS repo.

All settings, and the few changes that don't affect the results, are listed in
the docstring of `resnet_ml.py`.

## Run (from the repo root)
```bash
python scripts/resnet_paper/resnet_ml.py check --dataset MARIDA   # labels, loss weights, GPU memory and speed
python scripts/resnet_paper/resnet_ml.py train --dataset MARIDA
python scripts/resnet_paper/resnet_ml.py test  --dataset MARIDA
```
Or run everything: `bash outputs/resnet_paper/logs/run.sh` (MARIDA, then MADOS).

## Outputs: `outputs/resnet_paper/<dataset>/`
- `labels_and_weights.json`: patch label counts and BCE weights
- `history.csv`: training loss, plus validation loss and metrics, per epoch
- `best_epoch.json`
- `model_best_val_loss.pth`, `model_last.pth`
- `test_report.md`, `test_metrics_*.csv`, `test_predictions_*.json`
- `log.txt`

The runner's logs and `status.txt` are in `outputs/resnet_paper/logs/`.
