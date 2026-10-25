# MARIDA and MADOS, 26 features: deep models, RF, ensemble and explainability

U-Net, SegNeXt-T and the MariNeXt architecture trained on MARIDA with 26 input
channels, compared with a Random Forest on the same features, their ensemble and
the MARIDA paper, with SHAP feature importance (per feature and per group, per
class) and Seg-Grad-CAM. Single seed, no VSCP, 4 classes (Water, Marine Debris,
Algae/Sargassum, Other) unless stated.

Every script takes `--dataset MARIDA|MADOS` (default MARIDA; the runner: `-Dataset MADOS`).
Outputs: `outputs/marida_f26/` and `outputs/mados_f26/`. MADOS runs: R3, R4, R5, R6, R7.
The deep-learning comparison of both datasets (R3, R4, R5, R7, layout of the staged-ML
comparison): `dl_comparison_f26.py` -> `outputs/dl_comparison_f26/comparison.md`.

## Input features (26, `features26.py`)

| Group | Channels |
|---|---|
| bands (11) | B1 B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 |
| indices (5) | NDVI, NDWI, FAI, FDI, NDMI |
| ratios (4) | red/green, nir/red, nir/green, swir1/nir |
| GLCM (6) | contrast, dissimilarity, homogeneity, energy, correlation, ASM (13×13 window) |

Every feature equals the staged-ML feature at every annotated pixel (checked:
`outputs/marida_f26/features/verify_features.json`, maximum difference 0). The
GLCM maps are computed once for every pixel (`glcm_maps.py`, about 4 minutes).
Scaling and the full specification: `outputs/marida_f26/features/channel_spec.md`.

## Runs

Run folders carry the model name (`runs.py`); the runner's steps use the short IDs.

| Run (folder) | Model | Initialisation | Bands | Purpose |
|---|---|---|---|---|
| R1 (`R1_unet_ssl4eo_raw`) | U-Net (ResNet-50) | SSL4EO-S12 | raw reflectance | main U-Net |
| R2 (`R2_unet_ssl4eo_std`) | U-Net (ResNet-50) | SSL4EO-S12 | standardised | effect of band scaling (R1 vs R2) |
| R3 (`R3_unet_scratch_std`) | U-Net (ResNet-50) | scratch | standardised | effect of pretraining (R2 vs R3); like-for-like with R4/R5 |
| R4 (`R4_segnext_scratch_std`) | SegNeXt-T | scratch | standardised | |
| R5 (`R5_marinext_scratch_std`) | MariNeXt architecture | scratch | standardised | |
| R6 (`R6_rf`) | Random Forest (staged-ML settings) | – | unscaled | ML on the same 26 features |
| R7 (`R7_ensemble_unet_rf`) | R3 + R6 (from-scratch U-Net + RF) | – | – | w chosen on validation |
| G2b (`G2b_unet_ssl4eo_raw_11class`) | U-Net as R1 | SSL4EO-S12 | raw | the MARIDA paper's 11 classes, for a direct comparison |

Recipe: each dataset's paper. MARIDA: the MARIDA U-Net recipe, 45 epochs, Adam 2e-4,
batch 5, LR / 10 at epoch 40, weighted CE 1/ln(1.03 + p), rotation + flip augmentation,
no VSCP. MADOS: the MADOS/MariNeXt recipe, 80 epochs, Adam 2e-4, batch 5, LR / 10 at
epochs 45 and 65, the same weighted CE, flip + rotation augmentation, VSCP copy-paste,
EMA of the weights (decay 0.999). Checkpoint = best validation macro F1 with soft
8-view TTA (MADOS: of the EMA model), seed 0. The earlier R1 + RF ensemble is kept in
`outputs/marida_f26/runs/R7_ensemble_unet_rf_R1/` for reference only.

## Run it, one step at a time (repo root)

```powershell
powershell -File scripts/marida_f26/run_marida_f26.ps1 -Step features   # done already; rerun is quick
powershell -File scripts/marida_f26/run_marida_f26.ps1 -Step arch       # architecture docs (CPU)
powershell -File scripts/marida_f26/run_marida_f26.ps1 -Step check      # GPU memory / time estimate
powershell -File scripts/marida_f26/run_marida_f26.ps1 -Step R1         # ~1.5-2 h; then R2 R3 R4 R5 G2b
powershell -File scripts/marida_f26/run_marida_f26.ps1 -Step rf
powershell -File scripts/marida_f26/run_marida_f26.ps1 -Step ensemble
pip install shap                                                         # once, for xai-rf
powershell -File scripts/marida_f26/run_marida_f26.ps1 -Step xai-R1     # then xai-R2 ... xai-R5, xai-rf, xai-ensemble
powershell -File scripts/marida_f26/run_marida_f26.ps1 -Step compare
```
`-Step all` runs everything in order; `-Resume` continues an interrupted training
run. Logs: `outputs/marida_f26/logs/`.

## Outputs (`outputs/marida_f26/`)

- `features/`: GLCM maps, `channel_spec.md`, `channel_stats.json`, `correlation/`
  (Spearman/Pearson heatmaps overall and per class, group-level correlation,
  per-class feature distributions)
- `runs/<run>/`: `architecture.md`, `model_summary.txt`, `run_config.json`,
  `history.csv`, `best_epoch.json`, checkpoints, `report.md`, and for val and test:
  `metrics_<split>.json`, `classwise_<split>.csv`, confusion matrices (counts,
  row %, column %), `per_scene_<split>.csv`, `pr_curves_<split>.csv`,
  predictions (`pred_<split>.npz`, `prob_maps_test.npy`), `figures/`
- `xai/<run>/`: `xai_report.md`, `importance_features.csv`,
  `importance_groups.csv`, completeness and context checks, pixel SHAP samples,
  `figures/` (importance bars, grouped bars, heatmap, beeswarm and dependence
  plots per class, SHAP maps and Seg-Grad-CAM)
- `comparison/`: `comparison.md` (MARIDA paper rows on top), bootstrap CIs,
  paired differences, cross-model figures

## Notes

- GradientSHAP is implemented in `xai.py` (expected gradients, 32 samples, 20
  training patches as references); its completeness is checked per patch. The RF
  uses exact TreeSHAP (`shap`). The ensemble's importance is a weighted mix of its
  members' and is labelled as an approximation.
- Correlated features share SHAP credit; read single features together with the
  grouped results and `features/correlation/`.
- Confidence intervals come from resampling MARIDA's 15 test scenes. They cover
  test-scene variation, not training randomness (single seed).
