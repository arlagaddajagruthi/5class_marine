# 26-feature study (MARIDA or MADOS): run one step at a time, check its results, then run the next.
#
#   powershell -File scripts/marida_f26/run_marida_f26.ps1 -Step <step> [-Dataset MADOS]
#
# Steps, in order (each prints where its results are):
#   features   GLCM maps (skipped if present), channel stats + spec, feature check, correlation
#   arch       architecture.md + model_summary.txt for every run (CPU, no training)
#   check      10 training batches per model on the GPU: memory and time estimate
#   R1 R2 R3 R4 R5 G2b    train + test one run (MARIDA ~2 h each; MADOS: R3 R4 R5 only, ~6-7.5 h each)
#   rf         R6 Random Forest on the same 26 features (CPU)
#   ensemble   R7 U-Net (R3, from scratch) + RF ensemble (needs R3 and rf)
#   xai-R1 ... xai-R5, xai-rf, xai-ensemble   SHAP + Seg-Grad-CAM for one model
#   compare    comparison of every finished run, bootstrap CIs, cross-model XAI, and the
#              cross-dataset DL comparison (outputs/dl_comparison_f26/)
#   all        everything above in order (stops at the first error)
#
# -Dataset MARIDA (default) or MADOS; outputs go to outputs/<dataset>_f26/. Each dataset uses its
# paper's training recipe (train.py RECIPES). Run folders carry the model name (runs.py), e.g.
# outputs/marida_f26/runs/R3_unet_scratch_std/. Every step writes its logs to outputs/<dataset>_f26/logs/
# (named after the run). An interrupted training run continues with:  -Step R3 -Resume
param(
    [Parameter(Mandatory = $true)][string]$Step,
    [ValidateSet("MARIDA", "MADOS")][string]$Dataset = "MARIDA",
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$S = "scripts/marida_f26"
$Out = "outputs/$($Dataset.ToLower())_f26"
$Logs = "$Out/logs"
$D = "--dataset $Dataset"
New-Item -ItemType Directory -Force $Logs | Out-Null

# step -> (run folder name, train.py arguments); folder names as in runs.py
$Runs = @{
    "R1"  = @("R1_unet_ssl4eo_raw", "--model unet --init ssl4eo --bands raw");
    "R2"  = @("R2_unet_ssl4eo_std", "--model unet --init ssl4eo --bands std");
    "R3"  = @("R3_unet_scratch_std", "--model unet --init scratch --bands std");
    "R4"  = @("R4_segnext_scratch_std", "--model segnext --init scratch --bands std");
    "R5"  = @("R5_marinext_scratch_std", "--model marinext --init scratch --bands std");
    "G2b" = @("G2b_unet_ssl4eo_raw_11class", "--model unet --init ssl4eo --bands raw --classes 11")
}
if ($Dataset -eq "MARIDA") {
    $Deep = @("R1", "R2", "R3", "R4", "R5", "G2b")
    $Checks = @("R1", "R4", "R5")
} else {
    $Deep = @("R3", "R4", "R5")
    $Checks = @("R3", "R4", "R5")
}
$Order = @("features", "arch", "check") + $Deep + @("rf", "ensemble") +
         ($Deep | Where-Object { $_ -ne "G2b" } | ForEach-Object { "xai-$_" }) + @("xai-rf", "xai-ensemble", "compare")

function Py([string]$ArgLine, [string]$Log) {
    Write-Host ">> python $ArgLine"
    $p = Start-Process -FilePath "python" -ArgumentList $ArgLine -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$Logs/$Log.out.log" -RedirectStandardError "$Logs/$Log.err.log"
    Get-Content "$Logs/$Log.out.log" | Select-Object -Last 15
    if ($p.ExitCode -ne 0) {
        Get-Content "$Logs/$Log.err.log" | Select-Object -Last 20
        throw "step failed: python $ArgLine (exit $($p.ExitCode)); see $Logs/$Log.err.log"
    }
}

function Run-Step([string]$St) {
    switch -Regex ($St) {
        "^features$" {
            Py "$S/glcm_maps.py $D" "features_glcm"
            Py "$S/glcm_maps.py --verify $D" "features_glcm_verify"
            Py "$S/features26.py stats $D" "features_stats"
            Py "$S/features26.py verify $D" "features_verify"
            Py "$S/features26.py correlation $D" "features_correlation"
            Write-Host "results: $Out/features/ (channel_spec.md, correlation/correlation_summary.md)"
        }
        "^arch$" {
            foreach ($r in $Deep) { $n = $Runs[$r][0]; Py "$S/train.py arch --run $n $($Runs[$r][1]) $D" "arch_$n" }
            Write-Host "results: $Out/runs/<run>/architecture.md"
        }
        "^check$" {
            foreach ($r in $Checks) { $n = $Runs[$r][0]; Py "$S/train.py check --run $n $($Runs[$r][1]) $D" "check_$n" }
            Write-Host "results: $Out/runs/<run>/check.json"
        }
        "^(R[1-5]|G2b)$" {
            if ($Deep -notcontains $St) { throw "$St is not a $Dataset run (runs: $($Deep -join ', '))" }
            $res = ""
            if ($Resume) { $res = "--resume" }
            $n = $Runs[$St][0]
            Py "$S/train.py train --run $n $($Runs[$St][1]) $res $D" "train_$n"
            Py "$S/train.py test --run $n $($Runs[$St][1]) $D" "test_$n"
            Write-Host "results: $Out/runs/$n/report.md"
        }
        "^rf$" { Py "$S/rf_ensemble.py rf $D" "R6_rf"; Write-Host "results: $Out/runs/R6_rf/report.md" }
        "^ensemble$" {
            Py "$S/rf_ensemble.py ensemble --unet R3_unet_scratch_std $D" "R7_ensemble_unet_rf"
            Write-Host "results: $Out/runs/R7_ensemble_unet_rf/report.md"
        }
        "^xai-(R[1-5])$" { $n = $Runs[$Matches[1]][0]; Py "$S/xai.py --run $n $D" "xai_$n"; Write-Host "results: $Out/xai/$n/xai_report.md" }
        "^xai-rf$" { Py "$S/xai.py --run R6_rf $D" "xai_R6_rf"; Write-Host "results: $Out/xai/R6_rf/xai_report.md" }
        "^xai-ensemble$" { Py "$S/xai.py --run R7_ensemble_unet_rf $D" "xai_R7_ensemble_unet_rf"; Write-Host "results: $Out/xai/R7_ensemble_unet_rf/xai_report.md" }
        "^compare$" {
            Py "$S/compare.py $D" "compare"
            Py "$S/dl_comparison_f26.py" "dl_comparison_f26"
            Write-Host "results: $Out/comparison/comparison.md, outputs/dl_comparison_f26/comparison.md"
        }
        default { throw "unknown step '$St'. Steps: $($Order -join ', '), all" }
    }
}

if ($Step -eq "all") { foreach ($st in $Order) { Write-Host "=== $st ==="; Run-Step $st } }
else { Run-Step $Step }
