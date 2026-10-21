# Feature correlation, MARIDA train split (annotated pixels)

358,442 pixels with the 26 features. Heatmaps are ordered by hierarchical clustering on 1 − |Spearman ρ| (the all-classes order is used for every heatmap, so they can be compared), tick colours show the feature group, and every cell shows ρ. Strongly correlated features carry largely the same information, so SHAP can split credit between them; read feature importance together with these clusters and with the grouped results.

## all classes: 44 feature pairs with |Spearman ρ| ≥ 0.90

| Feature | Feature | ρ |
|---|---|---|
| glcm_energy | glcm_asm | +1.000 |
| NDWI | nir/green | -1.000 |
| NDMI | swir1/nir | -1.000 |
| NDVI | nir/red | +1.000 |
| glcm_dissimilarity | glcm_homogeneity | -1.000 |
| glcm_contrast | glcm_dissimilarity | +0.999 |
| glcm_contrast | glcm_homogeneity | -0.999 |
| B6 | B7 | +0.998 |
| B6 | B8 | +0.994 |
| B7 | B8 | +0.994 |
| B8 | B8A | +0.994 |
| B7 | B8A | +0.993 |
| B6 | B8A | +0.993 |
| B4 | B5 | +0.990 |
| glcm_homogeneity | glcm_asm | +0.984 |
| glcm_homogeneity | glcm_energy | +0.984 |
| glcm_dissimilarity | glcm_energy | -0.983 |
| glcm_dissimilarity | glcm_asm | -0.983 |
| glcm_contrast | glcm_energy | -0.981 |
| glcm_contrast | glcm_asm | -0.981 |
| B5 | B7 | +0.978 |
| B5 | B6 | +0.978 |
| B11 | B12 | +0.974 |
| B5 | B8 | +0.968 |
| B5 | B8A | +0.961 |
| B2 | B3 | +0.958 |
| B3 | B4 | +0.956 |
| B4 | B7 | +0.955 |
| B4 | B6 | +0.954 |
| B1 | B2 | +0.952 |
| B4 | B8 | +0.945 |
| B7 | red/green | +0.939 |
| B6 | red/green | +0.939 |
| B4 | B8A | +0.934 |
| B8 | red/green | +0.934 |
| B2 | B4 | +0.933 |
| B3 | B5 | +0.931 |
| B8A | red/green | +0.930 |
| B5 | red/green | +0.927 |
| B2 | B5 | +0.917 |
| B5 | FDI | +0.909 |
| B4 | red/green | +0.909 |
| B4 | FDI | +0.901 |
| B6 | FDI | +0.901 |

## Water: 46 feature pairs with |Spearman ρ| ≥ 0.90

| Feature | Feature | ρ |
|---|---|---|
| glcm_energy | glcm_asm | +1.000 |
| NDWI | nir/green | -1.000 |
| NDMI | swir1/nir | -1.000 |
| NDVI | nir/red | +1.000 |
| glcm_dissimilarity | glcm_homogeneity | -1.000 |
| glcm_contrast | glcm_dissimilarity | +1.000 |
| glcm_contrast | glcm_homogeneity | -0.999 |
| B6 | B7 | +0.998 |
| B6 | B8 | +0.997 |
| B7 | B8 | +0.996 |
| B8 | B8A | +0.995 |
| B7 | B8A | +0.994 |
| B6 | B8A | +0.994 |
| B4 | B5 | +0.993 |
| glcm_homogeneity | glcm_energy | +0.985 |
| glcm_homogeneity | glcm_asm | +0.985 |
| glcm_dissimilarity | glcm_energy | -0.984 |
| glcm_dissimilarity | glcm_asm | -0.984 |
| glcm_contrast | glcm_energy | -0.983 |
| glcm_contrast | glcm_asm | -0.983 |
| B5 | B7 | +0.980 |
| B5 | B6 | +0.979 |
| B5 | B8 | +0.973 |
| B11 | B12 | +0.973 |
| B5 | B8A | +0.966 |
| B2 | B3 | +0.960 |
| B4 | B7 | +0.959 |
| B4 | B6 | +0.958 |
| B1 | B2 | +0.958 |
| B3 | B4 | +0.956 |
| B4 | B8 | +0.953 |
| B7 | red/green | +0.943 |
| B6 | red/green | +0.943 |
| B4 | B8A | +0.942 |
| B8 | red/green | +0.941 |
| B8A | red/green | +0.937 |
| B2 | B4 | +0.936 |
| B3 | B5 | +0.934 |
| B5 | red/green | +0.929 |
| B2 | B5 | +0.924 |
| B5 | FDI | +0.916 |
| B4 | red/green | +0.911 |
| B4 | FDI | +0.910 |
| B6 | FDI | +0.910 |
| B7 | FDI | +0.908 |
| B8 | FDI | +0.906 |

## Marine Debris: 39 feature pairs with |Spearman ρ| ≥ 0.90

| Feature | Feature | ρ |
|---|---|---|
| glcm_energy | glcm_asm | +1.000 |
| NDWI | nir/green | -1.000 |
| NDMI | swir1/nir | -1.000 |
| NDVI | nir/red | +1.000 |
| glcm_dissimilarity | glcm_homogeneity | -0.988 |
| B7 | B8A | +0.987 |
| B6 | B7 | +0.985 |
| B5 | B6 | +0.981 |
| B6 | B8A | +0.981 |
| B5 | B7 | +0.981 |
| glcm_contrast | glcm_dissimilarity | +0.978 |
| B3 | B4 | +0.977 |
| B11 | B12 | +0.977 |
| B5 | B8A | +0.975 |
| B4 | B8 | +0.967 |
| B2 | B3 | +0.958 |
| B3 | B8 | +0.956 |
| B4 | B5 | +0.942 |
| glcm_contrast | glcm_homogeneity | -0.942 |
| B8 | B8A | +0.941 |
| glcm_homogeneity | glcm_energy | +0.941 |
| glcm_homogeneity | glcm_asm | +0.941 |
| B4 | B6 | +0.939 |
| B4 | B7 | +0.938 |
| B3 | B5 | +0.937 |
| B7 | B8 | +0.936 |
| B6 | B8 | +0.934 |
| B4 | B8A | +0.934 |
| B5 | B8 | +0.932 |
| B2 | B4 | +0.931 |
| B3 | B6 | +0.930 |
| B3 | B7 | +0.930 |
| B2 | B8 | +0.928 |
| B8 | FDI | +0.924 |
| B3 | B8A | +0.924 |
| B2 | B5 | +0.918 |
| B2 | B7 | +0.917 |
| B2 | B6 | +0.914 |
| B2 | B8A | +0.910 |

## Algae/Sargassum: 39 feature pairs with |Spearman ρ| ≥ 0.90

| Feature | Feature | ρ |
|---|---|---|
| glcm_energy | glcm_asm | +1.000 |
| NDWI | nir/green | -1.000 |
| NDVI | nir/red | +1.000 |
| NDMI | swir1/nir | -1.000 |
| glcm_dissimilarity | glcm_homogeneity | -0.998 |
| B7 | B8A | +0.993 |
| B6 | B7 | +0.991 |
| B6 | B8A | +0.990 |
| B5 | B6 | +0.986 |
| B3 | B4 | +0.986 |
| B5 | B7 | +0.985 |
| glcm_contrast | glcm_dissimilarity | +0.984 |
| FAI | FDI | +0.983 |
| B5 | B8A | +0.980 |
| B11 | B12 | +0.975 |
| B8 | FDI | +0.972 |
| glcm_contrast | glcm_homogeneity | -0.972 |
| B8 | FAI | +0.961 |
| NDWI | nir/red | -0.957 |
| nir/red | nir/green | +0.957 |
| NDVI | NDWI | -0.957 |
| NDVI | nir/green | +0.957 |
| B8A | FDI | +0.946 |
| B7 | FDI | +0.946 |
| B8 | B8A | +0.945 |
| B6 | FDI | +0.943 |
| B7 | B8 | +0.943 |
| B6 | B8 | +0.941 |
| B5 | B8 | +0.930 |
| glcm_homogeneity | glcm_energy | +0.920 |
| glcm_homogeneity | glcm_asm | +0.920 |
| B5 | FDI | +0.918 |
| FAI | nir/green | +0.915 |
| NDWI | FAI | -0.915 |
| glcm_dissimilarity | glcm_energy | -0.914 |
| glcm_dissimilarity | glcm_asm | -0.914 |
| B1 | B2 | +0.914 |
| B4 | B11 | +0.913 |
| B3 | B11 | +0.900 |

## Other: 14 feature pairs with |Spearman ρ| ≥ 0.90

| Feature | Feature | ρ |
|---|---|---|
| glcm_energy | glcm_asm | +1.000 |
| NDWI | nir/green | -1.000 |
| NDMI | swir1/nir | -1.000 |
| NDVI | nir/red | +1.000 |
| B6 | B7 | +0.980 |
| B11 | B12 | +0.975 |
| B5 | B6 | +0.971 |
| glcm_contrast | glcm_dissimilarity | +0.967 |
| B2 | B3 | +0.966 |
| B7 | B8A | +0.964 |
| B5 | B7 | +0.961 |
| B6 | B8A | +0.945 |
| B3 | B4 | +0.912 |
| B5 | B8A | +0.902 |

## Grouped: mean |Spearman ρ| within and between groups (all classes)

| | bands | indices | ratios | GLCM |
|---|---|---|---|---|
| bands | 0.78 | 0.53 | 0.56 | 0.36 |
| indices | 0.53 | 0.45 | 0.56 | 0.16 |
| ratios | 0.56 | 0.56 | 0.48 | 0.19 |
| GLCM | 0.36 | 0.16 | 0.19 | 0.87 |

## Files

- `spearman_<class>.png/.csv`, `pearson_<class>.png/.csv`: all classes and each class (water, marine_debris, algae, other)
- `spearman_groups.png/.csv`: group-level correlation
- `distributions.png`: per-class distribution of every feature
