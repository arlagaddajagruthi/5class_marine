# Staged ML experiments (stages 1-5; file name kept from when it ran 1-3) on MARIDA and MADOS, annotated classes 1-4 only.
# Run from the repo root. Pixel samples are cached, so stages after the first start quickly.
$ErrorActionPreference = "Stop"
$here = "scripts/staged"
python "$here/stage1_bands.py";        if (-not $?) { exit 1 }
python "$here/stage2_single_index.py"; if (-not $?) { exit 1 }
python "$here/stage3_all_indices.py";  if (-not $?) { exit 1 }
python "$here/stage4_indices_ratios.py"; if (-not $?) { exit 1 }
python "$here/stage5_ndwi_ratios.py";   if (-not $?) { exit 1 }
python "$here/compare_stages.py"
