import os
import glob
import torch
import numpy as np
import tifffile
from torch.utils.data import Dataset

class MADOSDataset(Dataset):
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
        # patch_id is like Scene_0_1. So scene = Scene_0, patch_idx = 1
        parts = patch_id.split('_')
        scene = f"{parts[0]}_{parts[1]}"
        patch_idx = parts[2]
        
        scene_dir = os.path.join(self.root_dir, scene)
        
        # Find the mask file (e.g. Scene_0_L2R_cl_1.tif)
        mask_pattern = os.path.join(scene_dir, "**", f"*_cl_{patch_idx}.tif")
        mask_files = glob.glob(mask_pattern, recursive=True)
        if not mask_files:
            raise FileNotFoundError(f"Mask not found for {patch_id}")
        mask_path = mask_files[0]
        
        # Find all other TIFs in the same directory (10/, 20/, etc) to stack as bands
        patch_dir = os.path.dirname(mask_path)
        img_pattern = os.path.join(patch_dir, f"*_{patch_idx}.tif")
        img_files = sorted(glob.glob(img_pattern))
        
        # Remove the mask from the image bands
        img_files = [f for f in img_files if f != mask_path]
        
        bands = []
        for f in img_files:
            band = tifffile.imread(f).astype(np.float32)
            bands.append(band)
            
        img = np.stack(bands, axis=0) # (C, H, W)
        mask = tifffile.imread(mask_path).astype(np.int64)
        
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        
        return torch.tensor(img), torch.tensor(mask)
