# Staged ML experiments, stages 1-8, on MARIDA and MADOS (annotated classes 1-4 only).
# Run from the repo root:
#   powershell -File scripts/staged_ml/run_all_stages.ps1                 # all three weightings
#   powershell -File scripts/staged_ml/run_all_stages.ps1 -Weighting log  # one weighting
# Pixel samples and GLCM features are cached (outputs/staged/cache/), so only the first run reads the patches.
# Stage 6 runs on MARIDA only. Stage 8 runs only if glcm_decision.py says GLCM adds something
# (exit code 0; 2 = skip); the rule looks at stage 7 of all three weightings, so with a single -Weighting
# the other two must already have been run.
param (
    [ValidateSet("all", "balanced", "log", "none")]
    [string]$Weighting = "all"
)
$ErrorActionPreference = "Stop"
$here = "scripts/staged_ml"
$roots = @{ balanced = "outputs/staged"; log = "outputs/staged/log_weights"; none = "outputs/staged/no_class_weights" }
$flags = @{ balanced = @("--weighting", "balanced"); log = @("--weighting", "log"); none = @("--no-class-weights") }
$todo = if ($Weighting -eq "all") { @("balanced", "log", "none") } else { @($Weighting) }

function Run($script, $extra) {
    python "$here/$script" @extra
    if (-not $?) { throw "$script failed" }
}

python "$here/glcm_features.py"; if (-not $?) { exit 1 }

foreach ($w in $todo) {
    $a = $flags[$w] + @("--out-root", $roots[$w])
    Write-Host "=== weighting: $w -> $($roots[$w])" -ForegroundColor Cyan
    Run "stage1_bands.py" $a
    Run "stage2_single_index.py" $a
    Run "stage3_all_indices.py" $a
    Run "stage4_indices_ratios.py" $a
    Run "stage5_ndwi_ratios.py" $a
    Run "stage6_env.py" $a
    Run "stage7_glcm.py" $a
}

python "$here/glcm_decision.py"    # 0 = run stage 8, 2 = skip, anything else = error
$code = $LASTEXITCODE
if ($code -ne 0 -and $code -ne 2) { exit 1 }
$runStage8 = ($code -eq 0)
foreach ($w in $todo) {
    $a = $flags[$w] + @("--out-root", $roots[$w])
    if ($runStage8) { Run "stage8_ndwi_ratios_glcm.py" $a }
    Run "compare_stages.py" @("--out-root", $roots[$w])
}
