import os
import torch
import numpy as np
import tifffile
import json
from torch.utils.data import Dataset

class MARIDADataset(Dataset):
    def __init__(self, root_dir, split="train", era5_features_path=None):
        self.root_dir = root_dir
        self.split = split
        self.split_file = os.path.join(root_dir, "splits", f"{split}_X.txt")
        self.era5_features = None
        
        with open(self.split_file, "r") as f:
            self.patch_ids = [line.strip() for line in f if line.strip()]
            
        if era5_features_path and os.path.exists(era5_features_path):
            with open(era5_features_path, "r") as f:
                self.era5_features = json.load(f)
            
    def __len__(self):
        return len(self.patch_ids)
        
    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        # patch_id is like 1-12-19_48MYU_0
        scene = patch_id.rsplit('_', 1)[0]
        
        img_path = os.path.join(self.root_dir, "patches", f"S2_{scene}", f"S2_{patch_id}.tif")
        mask_path = os.path.join(self.root_dir, "patches", f"S2_{scene}", f"S2_{patch_id}_cl.tif")
        
        img = tifffile.imread(img_path).astype(np.float32)
        mask = tifffile.imread(mask_path).astype(np.int64)
        
        # If image is (H, W, C), transpose to (C, H, W) for PyTorch
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=0)
            
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Inject ERA5 features as 3 additional channels if available
        if self.era5_features is not None and patch_id in self.era5_features:
            era5 = self.era5_features[patch_id]
            u10 = np.full((1, img.shape[1], img.shape[2]), era5.get("u10", 0.0), dtype=np.float32)
            v10 = np.full((1, img.shape[1], img.shape[2]), era5.get("v10", 0.0), dtype=np.float32)
            sst = np.full((1, img.shape[1], img.shape[2]), era5.get("sst", 0.0), dtype=np.float32)
            img = np.concatenate([img, u10, v10, sst], axis=0)
        
        return torch.tensor(img), torch.tensor(mask)
