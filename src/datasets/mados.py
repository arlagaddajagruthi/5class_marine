import os
import re
import glob
import torch
import numpy as np
import tifffile
from torch.utils.data import Dataset

# MADOS stores each Sentinel-2 band as a separate ACOLITE rhorc file in the
# 10/, 20/ and 60/ folders. Wavelength labels differ slightly between S2A and
# S2B (e.g. 443 vs 442), so bands are ordered by wavelength rather than matched
# by name. The resulting 11-band order is the same as MARIDA:
# B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12.
# Only reflectance (L2R_rhorc) files are used, as in the official MADOS
# stack_patches.py. The _conf_ / _rep_ files are annotation metadata (non-zero
# exactly where pixels are labelled) and must never be model inputs.
N_BANDS = 11
_RHORC_RE = re.compile(r"_L2R_rhorc_(\d+)_\d+\.tif$")


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
        scene, patch_idx = patch_id.rsplit('_', 1)
        scene_dir = os.path.join(self.root_dir, scene)

        mask_path = os.path.join(scene_dir, "10", f"{scene}_L2R_cl_{patch_idx}.tif")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for {patch_id}")
        mask = tifffile.imread(mask_path).astype(np.int64)
        h, w = mask.shape

        band_files = glob.glob(os.path.join(scene_dir, "*", f"{scene}_L2R_rhorc_*_{patch_idx}.tif"))
        band_files = sorted(band_files, key=lambda f: int(_RHORC_RE.search(f).group(1)))
        if len(band_files) != N_BANDS:
            raise ValueError(f"{patch_id}: expected {N_BANDS} rhorc bands, found {len(band_files)}")

        bands = []
        for f in band_files:
            band = tifffile.imread(f).astype(np.float32)
            # 20 m / 60 m bands -> 10 m grid by nearest neighbour (MADOS default)
            fy, fx = h // band.shape[0], w // band.shape[1]
            if (fy, fx) != (1, 1):
                band = np.repeat(np.repeat(band, fy, axis=0), fx, axis=1)
            bands.append(band)

        img = np.stack(bands, axis=0) # (11, H, W)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.tensor(img), torch.tensor(mask)
