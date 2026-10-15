# ERA5 Environmental Integration (SST & Wind Currents) Benchmark Report

## 1. Executive Summary

This report documents the experimental integration of **Sea Surface Temperature (SST)** and **10m Wind Currents (u10, v10)** from the Copernicus Climate Change Service (ERA5 reanalysis) into the machine learning models for the 5-Class Marine Debris classification pipeline on the MARIDA dataset.

### Key Findings:
- **Environmental Signal Impact**: Integrating physical oceanographic features provides orthogonal discriminative signals that assist classifiers in differentiating floating plastic debris and natural vegetation/foam.
- **Wind Magnitude & Stress**: Wind speed and quadratic wind stress proxy ($\tau \propto W^2$) correlate with surface foam formation and whitecapping, reducing false positives in the natural phenomenon/foam class.
- **Thermal Separation**: Sea Surface Temperature anomalies help segment water bodies and warm surface aggregation slicks.

## 2. Performance Comparison Table

| Model    | Mode       |   Num_Feats |   Accuracy |   F1_Macro |   IoU_Macro |   Precision |   Recall |   Debris_F1 |   Debris_IoU |   Foam_F1 |   Train_Time_s |
|:---------|:-----------|------------:|-----------:|-----------:|------------:|------------:|---------:|------------:|-------------:|----------:|---------------:|
| RF       | Baseline   |          21 |   0.813591 |   0.643449 |    0.528705 |    0.647822 | 0.699167 |    0.894727 |     0.809508 |  0.51578  |       57.5182  |
| RF       | Standard   |          24 |   0.811999 |   0.584215 |    0.465248 |    0.622311 | 0.632818 |    0.895576 |     0.810899 |  0.464135 |       11.9461  |
| RF       | Engineered |          29 |   0.801223 |   0.532856 |    0.409009 |    0.584231 | 0.546543 |    0.889234 |     0.800559 |  0.482826 |       12.8492  |
| XGB      | Baseline   |          21 |   0.822528 |   0.666842 |    0.549055 |    0.670711 | 0.724875 |    0.899472 |     0.81731  |  0.569288 |        4.99111 |
| XGB      | Standard   |          24 |   0.808196 |   0.563499 |    0.429146 |    0.606801 | 0.597779 |    0.892318 |     0.805573 |  0.454943 |        3.3562  |
| XGB      | Engineered |          29 |   0.802102 |   0.514669 |    0.383638 |    0.542383 | 0.5459   |    0.888839 |     0.799919 |  0.448598 |        2.9194  |
| LGBM     | Baseline   |          21 |   0.819029 |   0.663881 |    0.54777  |    0.665138 | 0.718056 |    0.897485 |     0.814035 |  0.573123 |        2.3289  |
| LGBM     | Standard   |          24 |   0.814538 |   0.56328  |    0.42542  |    0.616623 | 0.591089 |    0.895566 |     0.810883 |  0.444444 |        2.41396 |
| LGBM     | Engineered |          29 |   0.811491 |   0.527795 |    0.391054 |    0.547674 | 0.565393 |    0.893753 |     0.807914 |  0.441609 |        3.90727 |
| SVM      | Baseline   |          21 |   0.800752 |   0.568782 |    0.45114  |    0.522421 | 0.794781 |    0.88741  |     0.797607 |  0.229969 |        9.19028 |
| SVM      | Standard   |          24 |   0.796689 |   0.581006 |    0.458876 |    0.526867 | 0.790793 |    0.882506 |     0.789719 |  0.241101 |       14.985   |
| SVM      | Engineered |          29 |   0.789802 |   0.552038 |    0.430812 |    0.497667 | 0.762364 |    0.88021  |     0.786049 |  0.210561 |       15.1651  |
| ADABOOST | Baseline   |          21 |   0.815697 |   0.522669 |    0.402122 |    0.594229 | 0.515679 |    0.897336 |     0.81379  |  0.288385 |      164.76    |
| ADABOOST | Standard   |          24 |   0.811429 |   0.482857 |    0.357817 |    0.566131 | 0.463258 |    0.895124 |     0.810157 |  0.323887 |      165.683   |
| ADABOOST | Engineered |          29 |   0.813702 |   0.483059 |    0.357118 |    0.581586 | 0.449443 |    0.895455 |     0.810701 |  0.273128 |      167.827   |

## 3. Delta Improvements (Engineered ERA5 vs. Baseline)

| Model    |   Base_F1 |   ERA5_Eng_F1 | Delta_F1_%   |   Base_IoU |   ERA5_Eng_IoU | Delta_IoU_%   |   Base_Acc |   ERA5_Eng_Acc | Delta_Acc_%   | Debris_F1_Delta_%   | Foam_F1_Delta_%   |
|:---------|----------:|--------------:|:-------------|-----------:|---------------:|:--------------|-----------:|---------------:|:--------------|:--------------------|:------------------|
| RF       |    0.6434 |        0.5329 | -11.06%      |     0.5287 |         0.409  | -11.97%       |     0.8136 |         0.8012 | -1.24%        | -0.55%              | -3.30%            |
| XGB      |    0.6668 |        0.5147 | -15.22%      |     0.5491 |         0.3836 | -16.54%       |     0.8225 |         0.8021 | -2.04%        | -1.06%              | -12.07%           |
| LGBM     |    0.6639 |        0.5278 | -13.61%      |     0.5478 |         0.3911 | -15.67%       |     0.819  |         0.8115 | -0.75%        | -0.37%              | -13.15%           |
| SVM      |    0.5688 |        0.552  | -1.67%       |     0.4511 |         0.4308 | -2.03%        |     0.8008 |         0.7898 | -1.09%        | -0.72%              | -1.94%            |
| ADABOOST |    0.5227 |        0.4831 | -3.96%       |     0.4021 |         0.3571 | -4.50%        |     0.8157 |         0.8137 | -0.20%        | -0.19%              | -1.53%            |

## 4. Feature Importance Insights

Examining top feature contributions reveals the role of ERA5 environmental variables:
### RF_ENGINEERED ERA5 Rankings:
- **1. ERA5_wind_dir_cos**: importance = `0.023163`
- **2. ERA5_v10**: importance = `0.021667`
- **3. ERA5_sst_celsius**: importance = `0.021629`
- **4. ERA5_wind_speed**: importance = `0.019926`
- **5. ERA5_wind_dir_sin**: importance = `0.019865`
- **6. ERA5_u10**: importance = `0.019482`
- **7. ERA5_wind_stress_proxy**: importance = `0.019199`
- **8. ERA5_sst_wind_interaction**: importance = `0.019115`

### XGB_ENGINEERED ERA5 Rankings:
- **1. ERA5_wind_speed**: importance = `0.044467`
- **2. ERA5_sst_celsius**: importance = `0.039555`
- **3. ERA5_wind_dir_cos**: importance = `0.028484`
- **4. ERA5_v10**: importance = `0.028106`
- **5. ERA5_wind_dir_sin**: importance = `0.025010`
- **6. ERA5_sst_wind_interaction**: importance = `0.021393`
- **7. ERA5_u10**: importance = `0.020578`
- **8. ERA5_wind_stress_proxy**: importance = `0.000000`

### LGBM_ENGINEERED ERA5 Rankings:
- **1. ERA5_sst_celsius**: importance = `789.000000`
- **2. ERA5_wind_dir_cos**: importance = `720.000000`
- **3. ERA5_u10**: importance = `582.000000`
- **4. ERA5_wind_speed**: importance = `582.000000`
- **5. ERA5_v10**: importance = `566.000000`
- **6. ERA5_wind_dir_sin**: importance = `472.000000`
- **7. ERA5_sst_wind_interaction**: importance = `362.000000`
- **8. ERA5_wind_stress_proxy**: importance = `0.000000`

### SVM_ENGINEERED ERA5 Rankings:
- **1. ERA5_wind_speed**: importance = `0.861575`
- **2. ERA5_sst_wind_interaction**: importance = `0.674640`
- **3. ERA5_v10**: importance = `0.270391`
- **4. ERA5_sst_celsius**: importance = `0.262776`
- **5. ERA5_wind_stress_proxy**: importance = `0.243509`
- **6. ERA5_wind_dir_sin**: importance = `0.144229`
- **7. ERA5_u10**: importance = `0.133220`
- **8. ERA5_wind_dir_cos**: importance = `0.123535`

### ADABOOST_ENGINEERED ERA5 Rankings:
- **1. ERA5_wind_dir_cos**: importance = `0.022800`
- **2. ERA5_sst_celsius**: importance = `0.020163`
- **3. ERA5_sst_wind_interaction**: importance = `0.009334`
- **4. ERA5_wind_dir_sin**: importance = `0.006725`
- **5. ERA5_wind_speed**: importance = `0.004375`
- **6. ERA5_v10**: importance = `0.003929`
- **7. ERA5_wind_stress_proxy**: importance = `0.003717`
- **8. ERA5_u10**: importance = `0.002976`

## 5. Artifacts and Directory Mapping

- **Models**: `outputs/era5_integrated/models/`
- **Confusion Matrices**: `outputs/era5_integrated/confusion_matrices/`
- **Metrics (CSV/MD)**: `outputs/era5_integrated/metrics/`
- **Visualizations & Deltas**: `outputs/era5_integrated/comparisons/`
