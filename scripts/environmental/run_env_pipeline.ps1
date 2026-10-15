# Environmental-covariate pipeline for MARIDA_5Class (run from the repo root).
# Downloads are cached, so re-running only fetches what is missing.
$ErrorActionPreference = "Stop"
$here = "scripts/environmental"

python "$here/build_patch_metadata.py";  if (-not $?) { exit 1 }
python "$here/fetch_ocean_erddap.py";    if (-not $?) { exit 1 }
python "$here/fetch_era5.py";            if (-not $?) { exit 1 }
python "$here/build_env_table.py";       if (-not $?) { exit 1 }

# 1) Official scene-disjoint split, same labels as scripts/baselines (comparable numbers)
python "$here/run_env_experiments.py" --eval official --label-protocol baseline
# 2) Scene-grouped 5-fold CV on annotated classes only (more robust estimate)
python "$here/run_env_experiments.py" --eval scene-cv --label-protocol labelled
