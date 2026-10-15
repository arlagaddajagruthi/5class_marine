# Validation-based selection (re-check of the staged experiments)

In `scripts/staged/`, two choices were made by looking at **test** scores:
- the class weighting (log was preferred);
- the stage 5 feature set (bands + NDWI + 4 ratios).

That makes the reported test numbers optimistic. This folder repeats both
choices using only the official **validation** split. The test split is used
once, to report the chosen configurations.

Nothing in `scripts/staged/` or `outputs/staged/` is modified.
- Train and test pixels are read (read-only) from `outputs/staged/cache/`.
- Validation pixels and every output go to `outputs/staged_valselect/`.

## Data
- Same pixels as the staged runs: all annotated pixels (classes 1–4) of the
  official train / val / test splits. The splits are scene-disjoint.
- Same features (`src/features/pixel_features.py`) and the same five models
  with the same settings and seed (`scripts/staged/common.py`).
- Models are trained on the **train split only**, as in the staged runs, so
  the test scores stay comparable.

## Pre-registered rules (written before any validation result was seen)
1. **Candidates.** All feature sets from the staged experiments × 3 weightings
   (balanced, log, none) × 5 models:
   - stage 1, 2a–2g, 3, 4, 5;
   - MARIDA only: stage 6 = stage 5 + SST, chlorophyll-a, wind.
2. **Feature groups to keep (re-deriving stage 5 on validation).** For each
   group G in {NDVI, NDWI, FAI, FDI, NDMI, ratios, spectral stats}:
   - Δ(G) = val macro F1(bands + G) − val macro F1(bands only), averaged over
     the 5 models and 3 weightings, computed separately per dataset;
   - G is kept if Δ(G) > 0 on **both** datasets.
   - "Stage 5v" = 11 bands + the kept groups. On MARIDA, "stage 6v" = stage
     5v + the environmental features.
   - If 5v equals stage 5, no new runs are needed.
3. **Configuration selection.** For each dataset and model, the configuration
   (feature set, weighting) with the highest **val macro F1** is chosen. The
   overall choice per dataset is the best model + configuration by val macro
   F1. Ties are broken by val debris F1.
4. **Reporting.** Test metrics of the val-selected configurations are compared
   with the configurations previously chosen on test. The difference estimates
   how optimistic the earlier numbers were.
5. **Reproducibility check.** Every candidate is retrained, so its test score
   can be compared with the saved `outputs/staged/**/metrics.csv`.

## Run (from the repo root)
```bash
python scripts/staged_valselect/valselect.py prepare   # val pixel cache (+ val patch ids for env)
python scripts/staged_valselect/valselect.py worker    # trains/evaluates every pending job; several workers may run at once
python scripts/staged_valselect/valselect.py decide    # applies rule 2, adds stage 5v/6v jobs if needed
python scripts/staged_valselect/valselect.py worker    # runs the new jobs, if any
python scripts/staged_valselect/valselect.py report    # rules 3–5 -> outputs/staged_valselect/report.md
```
Each job writes its own JSON file to `outputs/staged_valselect/jobs/`. Jobs
that are already done are skipped, so the steps can be resumed after an
interruption.
