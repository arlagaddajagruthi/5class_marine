param (
    [string]$Model = "all",
    [string]$Modes = "baseline standard engineered",
    [int]$MaxPixels = 500000,
    [string]$DatasetPath = "data/MARIDA_5Class",
    [string]$Era5Lookup = "data/era5_cache/era5_features.json",
    [string]$OutputDir = "outputs/era5_integrated"
)

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " 5-Class Marine Debris & Pollution — ERA5 SST & Wind Benchmark" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "Target Model       : $Model"
Write-Host "Feature Modes      : $Modes"
Write-Host "Max Train Pixels   : $MaxPixels"
Write-Host "Dataset Directory  : $DatasetPath"
Write-Host "Output Directory   : $OutputDir"
Write-Host "=================================================================`n"

$PythonExe = "python"
$modesArray = $Modes.Split(" ")

if ($Model -eq "all") {
    Write-Host "[INFO] Executing comprehensive ERA5 benchmark across ALL ML models..." -ForegroundColor Green
    $cmd = "$PythonExe scripts/run_all_era5_benchmark.py --models all --modes $Modes --dataset_dir `"$DatasetPath`" --era5_json `"$Era5Lookup`" --out_dir `"$OutputDir`" --max_train_pixels $MaxPixels"
    Invoke-Expression $cmd
}
elseif ($Model -eq "rf") {
    Write-Host "[INFO] Executing Random Forest ERA5 pipeline..." -ForegroundColor Green
    $cmd = "$PythonExe scripts/run_rf_era5_pipeline.py --modes $Modes --dataset_dir `"$DatasetPath`" --era5_json `"$Era5Lookup`" --out_dir `"$OutputDir`" --max_train_pixels $MaxPixels"
    Invoke-Expression $cmd
}
elseif ($Model -eq "xgb") {
    Write-Host "[INFO] Executing XGBoost ERA5 pipeline..." -ForegroundColor Green
    $cmd = "$PythonExe scripts/run_xgboost_era5_pipeline.py --modes $Modes --dataset_dir `"$DatasetPath`" --era5_json `"$Era5Lookup`" --out_dir `"$OutputDir`" --max_train_pixels $MaxPixels"
    Invoke-Expression $cmd
}
elseif ($Model -eq "lgbm") {
    Write-Host "[INFO] Executing LightGBM ERA5 pipeline..." -ForegroundColor Green
    $cmd = "$PythonExe scripts/run_lightgbm_era5_pipeline.py --modes $Modes --dataset_dir `"$DatasetPath`" --era5_json `"$Era5Lookup`" --out_dir `"$OutputDir`" --max_train_pixels $MaxPixels"
    Invoke-Expression $cmd
}
elseif ($Model -eq "adaboost") {
    Write-Host "[INFO] Executing AdaBoost ERA5 pipeline..." -ForegroundColor Green
    $cmd = "$PythonExe scripts/run_adaboost_era5_pipeline.py --modes $Modes --dataset_dir `"$DatasetPath`" --era5_json `"$Era5Lookup`" --out_dir `"$OutputDir`" --max_train_pixels $MaxPixels"
    Invoke-Expression $cmd
}
elseif ($Model -eq "svm") {
    Write-Host "[INFO] Executing Support Vector Machine (SVM) ERA5 pipeline..." -ForegroundColor Green
    $cmd = "$PythonExe scripts/run_svm_era5_pipeline.py --modes $Modes --dataset_dir `"$DatasetPath`" --era5_json `"$Era5Lookup`" --out_dir `"$OutputDir`" --max_train_pixels $MaxPixels"
    Invoke-Expression $cmd
}
else {
    Write-Host "[ERROR] Unknown Model option: '$Model'. Supported options: all, rf, xgb, lgbm, adaboost, svm" -ForegroundColor Red
}

Write-Host "`n[DONE] Execution completed! Check '$OutputDir/comparisons/' for reports and plots." -ForegroundColor Cyan
