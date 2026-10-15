# Staged feature experiments (ML models only)

These experiments measure how much each feature group adds. Every stage trains the
same five models (RF, XGBoost, LightGBM, AdaBoost, linear SVM) on **MARIDA**
and **MADOS** using exactly the same sampled pixels. Only the feature columns
change between stages.

- Classes: 1 Water, 2 Marine Debris, 3 Algae/Sargassum, 4 Other.
  Class 0 (background/unannotated) is ignored.
- Train: **all** annotated pixels of the official train split. Water
  outnumbers debris about 180:1 in MARIDA, so every model is trained with
  balanced class weights: each class carries the same total weight. The
  `--train-cap` and `--no-class-weights` options can change this.
- Test: **every** annotated pixel of the official test split. The splits are
  scene-disjoint.
- Metrics: macro and per-class F1, IoU, precision and recall, plus accuracy
  and Cohen's kappa.
- Features come from `src/features/pixel_features.py`, and pixel samples are
  cached in `outputs/staged/cache/`.

| Stage | Script | Features |
|---|---|---|
| 1 | `stage1_bands.py` | 11 Sentinel-2 bands only |
| 2a–2e | `stage2_single_index.py` | bands + one index: NDVI, NDWI, FAI, FDI, NDMI |
| 2f, 2g | `stage2_single_index.py` | bands + 4 band ratios / bands + 4 spectral statistics |
| 3 | `stage3_all_indices.py` | bands + all 5 indices |
| 4 | `stage4_indices_ratios.py` | bands + all 5 indices + 4 band ratios (no environmental data yet) |
| later | *(not yet created)* | stage 4 + environmental features (MARIDA only) |

Run all stages with `scripts/staged/run_stages_1_to_3.ps1`, then use
`compare_stages.py` to build the comparison table.

Outputs go to `outputs/staged/<stage>/<dataset>/`:
- `summary.md` and `metrics.csv`: overall and per-class precision, recall,
  F1, IoU, accuracy and kappa
- `confusion_matrices/`
- `feature_importance.csv`

`outputs/staged/comparison/` holds every stage side by side, with the change
versus stage 1 for each model.
