# U-Net + Random Forest ensemble

`unet_rf.py` averages the per-pixel class probabilities of the U-Net
(`scripts/unet_ssl4eo/`, softmax averaged over 8 TTA views) and the staged
Random Forest (`scripts/staged_ml/`, log weights, retrained with the same
pixels and seed, so it reproduces the staged test scores):

    p = w * p_unet + (1 - w) * p_rf,   prediction = argmax p

The weight `w` (0 to 1 in steps of 0.05) and, for MARIDA, which U-Net run to
use are chosen on the **validation** split. The test split is used only for the
final report.

## Run (from the repo root)
```bash
python scripts/ensemble/unet_rf.py --datasets MARIDA MADOS                # RF stage 5 (bands + NDWI + ratios)
python scripts/ensemble/unet_rf.py --datasets MARIDA MADOS --rf-stage 8   # RF stage 8 (+ GLCM texture)
```
The U-Net probabilities are cached as `unet_<run>_<split>.npz` and the RF
probabilities as `rf_stage<N>_logw.npz`, so re-runs are fast.

## Outputs
`outputs/ensemble_unet_rf/<dataset>/` (stage 5) and
`outputs/ensemble_unet_rf_stage8/<dataset>/` (stage 8):
- `report.md`: overall and per-class precision, recall, F1 and IoU for the RF,
  the U-Net and the ensemble, plus the full class-wise analysis
- `weight_curve.csv`: validation and test scores for every `w`
- `classwise_*.csv`, `confusion_*.csv`

## Test results (macro F1 / debris F1)

| Dataset | RF | U-Net | Ensemble, RF stage 5 | Ensemble, RF stage 8 |
|---|---|---|---|---|
| MARIDA | 0.904 / 0.813 (st. 5), 0.916 / 0.831 (st. 8) | 0.862 / 0.657 | 0.920 / 0.823 | **0.923 / 0.836** |
| MADOS | 0.719 / 0.717 (st. 5), 0.761 / 0.708 (st. 8) | 0.824 / 0.618 | **0.838 / 0.758** | 0.820 / 0.748 |

On MADOS the stage-8 RF is better alone but worse in the ensemble, and the
validation split also prefers stage 5 (0.881 vs 0.875), so the stage-5
ensemble is the main result.
