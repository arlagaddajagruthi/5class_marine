# New PowerShell runner for MariNeXt 5-Class Pipeline
Write-Host "============================================================"
Write-Host "Launching MariNeXt 5-Class Model Training & Evaluation..."
Write-Host "============================================================"

$pythonExe = "python"
$scriptPath = "$PSScriptRoot\scripts\train_marinext_pipeline.py"

& $pythonExe $scriptPath

Write-Host "============================================================"
Write-Host "MariNeXt 5-Class Execution Completed Successfully!"
Write-Host "============================================================"
