# MARIDA 26-feature study: run one step at a time, check its results, then run the next.
#
#   powershell -File scripts/marida_f26/run_marida_f26.ps1 -Step <step>
#
# Steps, in order (each prints where its results are):
#   features   GLCM maps (skipped if present), channel stats + spec, feature check, correlation
#   arch       architecture.md + model_summary.txt for every run (CPU, no training)
#   check      10 training batches per model on the GPU: memory and time estimate
#   R1 R2 R3 R4 R5 G2b    train + test one run (each ~1.5-2 h on the RTX 2050)
#   rf         R6 Random Forest on the same 26 features (CPU)
#   ensemble   R7 U-Net (R1) + RF ensemble (needs R1 and rf)
#   xai-R1 ... xai-R5, xai-rf, xai-ensemble   SHAP + Seg-Grad-CAM for one model
#   compare    comparison of every finished run, bootstrap CIs, cross-model XAI
#   all        everything above in order (stops at the first error)
#
# Run folders carry the model name (runs.py), e.g. outputs/marida_f26/runs/R1_unet_ssl4eo_raw/.
# Every step writes its logs to outputs/marida_f26/logs/ (named after the run). An interrupted
# training run continues with:  -Step R1 -Resume
param(
    [Parameter(Mandatory = $true)][string]$Step,
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$S = "scripts/marida_f26"
$Logs = "outputs/marida_f26/logs"
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
$Order = @("features", "arch", "check", "R1", "R2", "R3", "R4", "R5", "G2b", "rf", "ensemble",
           "xai-R1", "xai-R2", "xai-R3", "xai-R4", "xai-R5", "xai-rf", "xai-ensemble", "compare")

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
            Py "$S/glcm_maps.py" "features_glcm"
            Py "$S/glcm_maps.py --verify" "features_glcm_verify"
            Py "$S/features26.py stats" "features_stats"
            Py "$S/features26.py verify" "features_verify"
            Py "$S/features26.py correlation" "features_correlation"
            Write-Host "results: outputs/marida_f26/features/ (channel_spec.md, correlation/correlation_summary.md)"
        }
        "^arch$" {
            foreach ($r in $Runs.Keys) { $n = $Runs[$r][0]; Py "$S/train.py arch --run $n $($Runs[$r][1])" "arch_$n" }
            Write-Host "results: outputs/marida_f26/runs/<run>/architecture.md"
        }
        "^check$" {
            foreach ($r in @("R1", "R4", "R5")) { $n = $Runs[$r][0]; Py "$S/train.py check --run $n $($Runs[$r][1])" "check_$n" }
            Write-Host "results: outputs/marida_f26/runs/<run>/check.json"
        }
        "^(R[1-5]|G2b)$" {
            $res = ""
            if ($Resume) { $res = "--resume" }
            $n = $Runs[$St][0]
            Py "$S/train.py train --run $n $($Runs[$St][1]) $res" "train_$n"
            Py "$S/train.py test --run $n $($Runs[$St][1])" "test_$n"
            Write-Host "results: outputs/marida_f26/runs/$n/report.md"
        }
        "^rf$" { Py "$S/rf_ensemble.py rf" "R6_rf"; Write-Host "results: outputs/marida_f26/runs/R6_rf/report.md" }
        "^ensemble$" {
            Py "$S/rf_ensemble.py ensemble --unet R1_unet_ssl4eo_raw" "R7_ensemble_unet_rf"
            Write-Host "results: outputs/marida_f26/runs/R7_ensemble_unet_rf/report.md"
        }
        "^xai-(R[1-5])$" { $n = $Runs[$Matches[1]][0]; Py "$S/xai.py --run $n" "xai_$n"; Write-Host "results: outputs/marida_f26/xai/$n/xai_report.md" }
        "^xai-rf$" { Py "$S/xai.py --run R6_rf" "xai_R6_rf"; Write-Host "results: outputs/marida_f26/xai/R6_rf/xai_report.md" }
        "^xai-ensemble$" { Py "$S/xai.py --run R7_ensemble_unet_rf" "xai_R7_ensemble_unet_rf"; Write-Host "results: outputs/marida_f26/xai/R7_ensemble_unet_rf/xai_report.md" }
        "^compare$" { Py "$S/compare.py" "compare"; Write-Host "results: outputs/marida_f26/comparison/comparison.md" }
        default { throw "unknown step '$St'. Steps: $($Order -join ', '), all" }
    }
}

if ($Step -eq "all") { foreach ($st in $Order) { Write-Host "=== $st ==="; Run-Step $st } }
else { Run-Step $Step }
