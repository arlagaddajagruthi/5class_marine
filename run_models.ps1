param (
    [string]$Model = "rf",
    [string]$Dataset = "MARIDA_5Class",
    [int]$BatchSize = 5
)

# Handle DatasetPath logic based on dataset
if ($Dataset -eq "MARIDA_5Class") {
    $DatasetPath = "c:\Users\CB.SC.U4CSE23709\Desktop\Marine Datasets\$Dataset\patches"
}
else {
    $DatasetPath = "c:\Users\CB.SC.U4CSE23709\Desktop\Marine Datasets\$Dataset"
}

$PythonExe = "python"

if ($Model -eq "rf") {
    Write-Host "Running Random Forest pipeline on $Dataset..."
    if ($Dataset -eq "MARIDA_5Class") {
        $cmd = "$PythonExe mados-master/utils/rf_pipeline.py --path `"$DatasetPath`""
    }
    else {
        $cmd = "$PythonExe mados-master/utils/mados_rf_pipeline.py --path `"$DatasetPath`""
    }
    Invoke-Expression $cmd
}
elseif ($Model -eq "unet") {
    Write-Host "Running U-Net training on $Dataset..."
    $cmd = "$PythonExe mados-master/marinext/train.py --path `"$DatasetPath`" --config mados-master/marinext/configs/unet_5class.py --batch $BatchSize --tensorboard tsboard_unet_$Dataset"
    Invoke-Expression $cmd
}
elseif ($Model -eq "segnext") {
    Write-Host "Running SegNeXt training on $Dataset..."
    $cmd = "$PythonExe mados-master/marinext/train.py --path `"$DatasetPath`" --config mados-master/marinext/configs/segnext_5class.py --batch $BatchSize --tensorboard tsboard_segnext_$Dataset"
    Invoke-Expression $cmd
}
elseif ($Model -eq "marinext") {
    Write-Host "Running MariNeXt training on $Dataset..."
    $cmd = "$PythonExe mados-master/marinext/train.py --path `"$DatasetPath`" --config mados-master/marinext/configs/marinext_5class.py --batch $BatchSize --tensorboard tsboard_marinext_$Dataset"
    Invoke-Expression $cmd
}
elseif ($Model -eq "all_dl") {
    Write-Host "Running all Deep Learning models sequentially on $Dataset..."
    
    Write-Host "Starting U-Net..."
    $cmd1 = "$PythonExe mados-master/marinext/train.py --path `"$DatasetPath`" --config mados-master/marinext/configs/unet_5class.py --batch $BatchSize --tensorboard tsboard_unet_$Dataset"
    Invoke-Expression $cmd1
    
    Write-Host "Starting SegNeXt..."
    $cmd2 = "$PythonExe mados-master/marinext/train.py --path `"$DatasetPath`" --config mados-master/marinext/configs/segnext_5class.py --batch $BatchSize --tensorboard tsboard_segnext_$Dataset"
    Invoke-Expression $cmd2
    
    Write-Host "Starting MariNeXt..."
    $cmd3 = "$PythonExe mados-master/marinext/train.py --path `"$DatasetPath`" --config mados-master/marinext/configs/marinext_5class.py --batch $BatchSize --tensorboard tsboard_marinext_$Dataset"
    Invoke-Expression $cmd3
    
    Write-Host "Finished all DL models!"
}
else {
    Write-Host "Unknown model: $Model. Please specify rf, unet, segnext, marinext, or all_dl."
}
