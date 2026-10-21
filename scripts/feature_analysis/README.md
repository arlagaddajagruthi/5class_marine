# Per-class feature analysis (ML models)

`per_class_features.py` explains, class by class, which features the five ML
models of `scripts/staged_ml` (RF, XGBoost, LightGBM, AdaBoost, linear SVM)
rely on, using SHAP, and checks the choices by retraining on per-class
top-k feature subsets. The results and their write-up are in
[`SHAP/Analysis1/`](../../SHAP/Analysis1/README.md).

Classes 1 Water, 2 Marine Debris, 3 Algae/Sargassum, 4 Other (class 0 ignored).
Pixels, models, seed (42) and log class weighting are those of the staged
experiments, so the models are the stage-8 models (11 bands + NDWI + 4 ratios
+ 6 GLCM = 22 features).

## Run (from the repo root)

Needs `data/MARIDA_5Class`, `data/MADOS_5Class`, and `pip install shap numba`.
The pixel and GLCM caches of `scripts/staged_ml` are built on first use.

```powershell
python scripts/feature_analysis/per_class_features.py                        # everything, both datasets
python scripts/feature_analysis/per_class_features.py --datasets MARIDA --parts dist shap
python scripts/feature_analysis/per_class_features.py --parts select         # reuses the saved SHAP ranking
```

Options: `--models`, `--weighting` (default `log`), `--shap-n` explained test
pixels per class (500), `--bg-n` background pixels per class (25), `--ada-frac`
stratified share of the training split for the AdaBoost top-k refits (0.2; the
other models refit on the full split), `--tol` (0.01).

## Parts and outputs

`outputs/feature_analysis/<dataset>/`

| Part | Folder | What |
|---|---|---|
| `staged` | `0_staged/` | Per-class F1 change of every stage vs stage 1, every model (from the existing staged results) |
| `dist` | `1_distributions/` | All 30 candidate features: per-class statistics, violins per feature group, spectral signatures, one-vs-rest ROC AUC heatmap |
| `shap` | `2_shap/` | Stage-8 models on the full training split; per-class SHAP beeswarms per model and per class across models, share-of-SHAP heatmaps, feature-group shares, `shap_per_class.csv` |
| `select` | `3_selection/` | Per-class top-k curves, the chosen k per class and model, union of the per-class choices vs all 22 features |

`summary.md` in each dataset folder collects the tables.

## Full SHAP figure set (`shap_figures.py`)

`shap_figures.py` reads the SHAP values saved by the `shap` part and draws
every comparison figure for both datasets and all five models into
`outputs/feature_analysis/shap_figures/`:

```
01_global_importance/            02_class_specific/
03_cross_model_comparison/       04_dependence_and_interactions/
05_individual_predictions/       06_reports/   (README.md = index of every figure, CSV tables)
```

```powershell
python scripts/feature_analysis/shap_figures.py                  # 01-04 + 06 from the saved SHAP values (~25 min)
python scripts/feature_analysis/shap_figures.py --with-models    # also refits the models for 05 and the interaction heatmaps
```

## Publishing a run (`export_analysis.py`)

The scripts write to `outputs/feature_analysis/` (git-ignored).
`export_analysis.py` copies everything into one self-contained folder in the
six-folder layout, `SHAP/<name>/`, with the write-up as its `README.md`
(`--report`, links rewritten):

```powershell
python scripts/feature_analysis/export_analysis.py --name Analysis1
```

The per-class distributions, staged deltas and selection results are placed
in the matching folders: `02_class_specific/<dataset>/feature_distributions`,
`02_class_specific/<dataset>/shap_summary`,
`03_cross_model_comparison/<dataset>/staged_experiments` and
`06_reports/<dataset>/feature_selection`.

## SHAP per model

| Model | Explainer | Output explained |
|---|---|---|
| RF | interventional TreeSHAP | class probability |
| XGBoost, LightGBM | interventional TreeSHAP | class log-odds (margin) |
| AdaBoost | permutation SHAP (sklearn AdaBoost is not supported by TreeSHAP) | class probability |
| Linear SVM | exact linear SHAP, coef × (z − mean background z) | one-vs-rest decision score |

All models share the same background pixels and the same explained pixels.
Because the output units differ, models are compared through each feature's
share of a class's total mean |SHAP|.
