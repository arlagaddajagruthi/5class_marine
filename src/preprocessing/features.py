import numpy as np
import numba
from numba import njit, prange

def compute_spectral_indices(img, dataset_name):
    """
    img: (C, H, W) numpy array
    """
    eps = 1e-8
    indices = []
    
    if dataset_name.upper() == "MARIDA":
        # MARIDA: 11 bands
        # 0:Coastal, 1:Blue, 2:Green, 3:Red, 4:RE1, 5:RE2, 6:RE3, 7:NIR, 8:NNIR, 9:SWIR1, 10:SWIR2
        blue, green, red = img[1], img[2], img[3]
        re2 = img[5]
        nir = img[7]
        swir1 = img[9]
        
        # 1. NDVI
        ndvi = (nir - red) / (nir + red + eps)
        indices.append(ndvi)
        
        # 2. NDWI
        ndwi = (green - nir) / (green + nir + eps)
        indices.append(ndwi)
        
        # 3. FAI = NIR - [Red + (SWIR1 - Red) * ((842 - 665)/(1610 - 665))]
        lambda_nir, lambda_red, lambda_swir1 = 842.0, 665.0, 1610.0
        fai = nir - (red + (swir1 - red) * ((lambda_nir - lambda_red) / (lambda_swir1 - lambda_red)))
        indices.append(fai)
        
        # 4. FDI = NIR - [RE2 + (SWIR1 - RE2) * 10 * (842 - 740) / 1000] -> typically scaled by wavelength diff in nm
        lambda_re2 = 740.0
        fdi = nir - (re2 + (swir1 - re2) * 10 * (lambda_nir - lambda_re2) / 1000.0)
        indices.append(fdi)
        
        # 5. SI (Shadow Index)
        si = np.power(np.clip((1.0 - blue) * (1.0 - green) * (1.0 - red), 0, None), 1.0/3.0)
        indices.append(si)
        
        # 6. NDMI
        ndmi = (nir - swir1) / (nir + swir1 + eps)
        indices.append(ndmi)
        
        # 7. BSI
        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)
        indices.append(bsi)
        
        # 8. NRD
        nrd = (red - green) / (red + green + eps)
        indices.append(nrd)
        
    elif dataset_name.upper() == "MADOS":
        # MADOS: 8 bands (first 2 are masks)
        # 2:Blue, 3:Green, 4:Red, 5:NIR
        blue, green, red = img[2], img[3], img[4]
        nir = img[5]
        
        # 1. NDVI
        ndvi = (nir - red) / (nir + red + eps)
        indices.append(ndvi)
        
        # 2. NDWI
        ndwi = (green - nir) / (green + nir + eps)
        indices.append(ndwi)
        
        # 3. SI
        si = np.power(np.clip((1.0 - blue) * (1.0 - green) * (1.0 - red), 0, None), 1.0/3.0)
        indices.append(si)
        
        # 4. NRD
        nrd = (red - green) / (red + green + eps)
        indices.append(nrd)
        
    else:
        raise ValueError(f"Unknown dataset for indices: {dataset_name}")
        
    # Stack indices and replace NaNs/Infs
    indices_arr = np.stack(indices, axis=0)
    indices_arr = np.nan_to_num(indices_arr, nan=0.0, posinf=0.0, neginf=0.0)
    return indices_arr


@njit(parallel=True, fastmath=True)
def _compute_glcm_numba(img, h, w, win_size, num_bins):
    """
    Computes 6 GLCM features (CON, DIS, HOMO, ENER, COR, ASM) for a 13x13 window.
    Offset: d=1, angle=0 (horizontal right).
    """
    out = np.zeros((6, h, w), dtype=np.float32)
    pad = win_size // 2
    
    for r in prange(h):
        for c in range(w):
            # Compute GLCM matrix for this window (16x16)
            glcm = np.zeros((num_bins, num_bins), dtype=np.float32)
            total = 0.0
            
            # Window boundaries
            r_start = max(0, r - pad)
            r_end = min(h, r + pad + 1)
            c_start = max(0, c - pad)
            c_end = min(w, c + pad + 1)
            
            # Fill GLCM (horizontal offset +1)
            for wr in range(r_start, r_end):
                for wc in range(c_start, c_end - 1):
                    i = img[wr, wc]
                    j = img[wr, wc + 1]
                    glcm[i, j] += 1.0
                    total += 1.0
            
            if total > 0:
                glcm /= total
                
            # Extract features
            con = 0.0
            dis = 0.0
            homo = 0.0
            ener = 0.0
            asm = 0.0
            
            # For correlation
            mu_i = 0.0
            mu_j = 0.0
            for i in range(num_bins):
                for j in range(num_bins):
                    p = glcm[i, j]
                    if p > 0:
                        con += p * ((i - j) ** 2)
                        dis += p * abs(i - j)
                        homo += p / (1.0 + (i - j) ** 2)
                        ener += p ** 2
                        mu_i += i * p
                        mu_j += j * p
            
            asm = ener
            ener = np.sqrt(ener)
            
            std_i = 0.0
            std_j = 0.0
            cor = 0.0
            for i in range(num_bins):
                for j in range(num_bins):
                    p = glcm[i, j]
                    if p > 0:
                        std_i += p * ((i - mu_i) ** 2)
                        std_j += p * ((j - mu_j) ** 2)
            
            std_i = np.sqrt(std_i)
            std_j = np.sqrt(std_j)
            
            if std_i > 0 and std_j > 0:
                for i in range(num_bins):
                    for j in range(num_bins):
                        p = glcm[i, j]
                        if p > 0:
                            cor += p * (i - mu_i) * (j - mu_j)
                cor /= (std_i * std_j)
                
            out[0, r, c] = con
            out[1, r, c] = dis
            out[2, r, c] = homo
            out[3, r, c] = ener
            out[4, r, c] = cor
            out[5, r, c] = asm
            
    return out

def compute_glcm_features(img, dataset_name, win_size=13, num_bins=16):
    """
    Extracts RGB, converts to grayscale, quantizes to 16 bins, runs GLCM.
    """
    if dataset_name.upper() == "MARIDA":
        r, g, b = img[3], img[2], img[1]
    elif dataset_name.upper() == "MADOS":
        r, g, b = img[4], img[3], img[2]
    else:
        raise ValueError("Unknown dataset")
        
    # RGB to Gray
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    # Quantize to [0, num_bins-1]
    gray = np.clip(gray, 0.0, 1.0)
    quantized = np.floor(gray * (num_bins - 1)).astype(np.int32)
    
    h, w = quantized.shape
    glcm_feats = _compute_glcm_numba(quantized, h, w, win_size, num_bins)
    
    return glcm_feats

def extract_all_features(img, dataset_name, use_si=True, use_glcm=True):
    """
    Given an image tensor (C, H, W), returns (C', H, W)
    """
    bands = [img]
    
    if use_si:
        si = compute_spectral_indices(img, dataset_name)
        bands.append(si)
        
    if use_glcm:
        glcm = compute_glcm_features(img, dataset_name)
        bands.append(glcm)
        
    return np.concatenate(bands, axis=0)
