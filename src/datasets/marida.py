import os
import torch
import numpy as np
import tifffile
from torch.utils.data import Dataset

class MARIDADataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split
        self.split_file = os.path.join(root_dir, "splits", f"{split}_X.txt")
        
        with open(self.split_file, "r") as f:
            self.patch_ids = [line.strip() for line in f if line.strip()]
            
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
        
        return torch.tensor(img), torch.tensor(mask)
