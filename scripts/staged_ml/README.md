# Staged feature experiments (ML models only)

These experiments measure how much each feature group adds. Every stage trains
the same five models (RF, XGBoost, LightGBM, AdaBoost, linear SVM) on **MARIDA**
and **MADOS** with exactly the same pixels, models and seed (42). Only the
feature columns change between stages. This is the code behind `paper/main.tex`.

- Classes: 1 Water, 2 Marine Debris, 3 Algae/Sargassum, 4 Other.
  Class 0 (unannotated, cloud, cloud shadow) is ignored.
- Train: **all** annotated pixels of the official train split.
- Test: **every** annotated pixel of the official test split. The splits are
  scene-disjoint.
- Metrics: macro and per-class precision, recall, F1, IoU and accuracy, plus
  overall accuracy and Cohen's kappa.
- Features come from `src/features/pixel_features.py`; pixel samples are
  cached in `outputs/staged/cache/`.

## Stages

| Stage | Script | Features | # |
|---|---|---|---|
| 1 | `stage1_bands.py` | 11 Sentinel-2 bands | 11 |
| 2a–2e | `stage2_single_index.py` | bands + one index: NDVI, NDWI, FAI, FDI, NDMI | 12 |
| 2f, 2g | `stage2_single_index.py` | bands + 4 band ratios / bands + 4 spectral statistics | 15 |
| 3 | `stage3_all_indices.py` | bands + all 5 indices | 16 |
| 4 | `stage4_indices_ratios.py` | bands + 5 indices + 4 ratios | 20 |
| 5 | `stage5_ndwi_ratios.py` | bands + NDWI + 4 ratios (the two best stage-2 additions) | 16 |
| 6 | `stage6_env.py` | stage 5 + 7 environmental covariates (SST, SST gradient, chlorophyll-a, wind); **MARIDA only** | 23 |
| 7 | `stage7_glcm.py` | bands + 6 GLCM texture features | 17 |
| 8 | `stage8_ndwi_ratios_glcm.py` | stage 5 + 6 GLCM texture features | 22 |
| 9 | `stage9_all_env_shap.py` | bands + 5 indices + 4 ratios + all 27 environmental covariates from both `data/` env tables, explained with SHAP; **MARIDA only, log weights** | 47 |

The 4 ratios are red/green, NIR/red, NIR/green and SWIR1/NIR.

**GLCM texture** (`glcm_features.py`) follows the MARIDA reference code: grey
image from RGB (B4, B3, B2) clipped to [0, 0.15], 17 grey levels, 13×13 window
with reflect padding, distance 1, angles 0/45/90/135° averaged; contrast,
dissimilarity, homogeneity, energy, correlation and ASM. The features are
cached per split as `outputs/staged/cache/<dataset>_<split>_all_glcm6.npy`.
`python scripts/staged_ml/glcm_features.py --check` compares the kernel with
scikit-image.

**Stage 8 rule** (`glcm_decision.py`, fixed before stage 7 was run): stage 8
runs only if, for at least one weighting and one dataset, stage 7 beats stage
1 by more than 0.01 in the five-model mean of macro F1 or debris F1. Exit code
0 = run, 2 = skip. The decision is written to `outputs/staged/glcm_decision.md`
(result: run).

## Stage 9: all features + environment, with SHAP

`stage9_all_env_shap.py` trains the five models on stage 4 (bands, all 5
indices, 4 ratios) plus every environmental covariate in
`data/env_sst_chl_wind/marida_env_sst_chl_wind.csv` and
`data/environmental/marida_env_features.csv` (SST and SST gradient,
chlorophyll-a from OLCI and MODIS, CMEMS and ERA5 wind speed and direction,
pressure, precipitation, wave height, Stokes drift). The MUR SST columns of the
second file duplicate the first and are dropped. Each model is then explained
with the SHAP explainers of `SHAP/Analysis2/shap_analysis/shap_compute.py`
(TreeSHAP for the tree models, LinearExplainer for the SVM) on 300
class-stratified test pixels per class, with 50 training pixels per class as
background. It runs with log weights by default:

```powershell
python scripts/staged_ml/stage9_all_env_shap.py   # writes outputs/staged/log_weights/stage9_all_env_shap/
```

Outputs, `outputs/staged/log_weights/stage9_all_env_shap/`:
- `training/MARIDA/`: the usual stage outputs (metrics, summary, confusion
  matrices, built-in feature importance); `compare_stages.py` picks these up
- `shap/MARIDA/REPORT.md`: scores vs stages 4 and 6, group shares, top features
- `shap/MARIDA/00_samples` explained and background pixels, `01_shap_values`
  raw values, `02_global_importance`, `03_per_class` (rankings, beeswarms),
  `04_feature_groups` (bands / indices / ratios / environment and
  environmental sub-groups), `05_cross_model` (ranking agreement),
  `06_dependence` (debris dependence plots), `07_validation` (additivity check)
- `logs/`: run log

## Class weighting

Every stage is run with three per-pixel class weightings (`--weighting`):

| Weighting | Flag | Output root |
|---|---|---|
| balanced (default): N / (K n_c) | `--weighting balanced` | `outputs/staged/` |
| log: 1 / ln(1.03 + p_c), as in the MARIDA/MADOS code | `--weighting log` | `outputs/staged/log_weights/` |
| none | `--no-class-weights` | `outputs/staged/no_class_weights/` |

Log weighting is the main setting in the paper.

## Run (from the repo root)

```powershell
powershell -File scripts/staged_ml/run_all_stages.ps1                 # stages 1-8, all three weightings
powershell -File scripts/staged_ml/run_all_stages.ps1 -Weighting log  # one weighting
```

Single stage, for example:
```powershell
python scripts/staged_ml/stage8_ndwi_ratios_glcm.py --weighting log --out-root outputs/staged/log_weights
python scripts/staged_ml/compare_stages.py --out-root outputs/staged/log_weights
```

Other options: `--datasets`, `--models`, `--train-cap`, `--svm-cap`, `--seed`.

`fine_to_coarse.py` is a separate check: it trains the stage-5 RF on the
original 15 classes and maps the predictions back to 4 classes, to test
whether merging the classes before training costs accuracy (it does not). It
needs the original datasets (`--orig-root`).

## Outputs

`<root>/<stage>/<dataset>/`:
- `summary.md` and `metrics.csv`: overall and per-class precision, recall,
  F1, IoU and accuracy, plus kappa
- `confusion_matrices/`: one CSV and PNG per model
- `feature_importance.csv`, `run_config.json`

`<root>/comparison/` holds every stage side by side (`comparison.md`,
`all_results.csv`) and plots of the change versus stage 1 for each model.

## Main results (log weights, RF; macro F1 / debris F1)

| Stage | MARIDA | MADOS |
|---|---|---|
| 1 bands | 0.885 / 0.742 | 0.710 / 0.697 |
| 5 NDWI + ratios | 0.904 / 0.813 | 0.719 / 0.717 |
| 7 bands + GLCM | 0.887 / 0.720 | 0.755 / 0.641 |
| 8 NDWI + ratios + GLCM | **0.916 / 0.831** | **0.761** / 0.708 |

Per-class results for every model, stage and weighting are in the
`summary.md` files.
