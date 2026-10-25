# Runs the MADOS 26-feature study steps in order (launched detached); stops at the first failure.
# Progress: outputs/mados_f26/logs/sequence_status.txt (one line per finished step).
$Runner = "C:\PROJECT\5class_marine_pollution\scripts\marida_f26\run_marida_f26.ps1"
$Status = "C:\PROJECT\5class_marine_pollution\outputs\mados_f26\logs\sequence_status.txt"
$Steps = @("R3", "rf", "ensemble", "xai-R3", "R4", "R5", "xai-R4", "xai-R5", "xai-rf", "xai-ensemble", "compare")
Set-Content $Status "started $(Get-Date -Format s)"
foreach ($st in $Steps) {
    Add-Content $Status "running $st $(Get-Date -Format s)"
    & powershell -ExecutionPolicy Bypass -File $Runner -Step $st -Dataset MADOS *> "C:\PROJECT\5class_marine_pollution\outputs\mados_f26\logs\step_$st.log"
    if ($LASTEXITCODE -ne 0) { Add-Content $Status "FAILED $st $(Get-Date -Format s)"; exit 1 }
    Add-Content $Status "done $st $(Get-Date -Format s)"
}
Add-Content $Status "ALL DONE $(Get-Date -Format s)"
