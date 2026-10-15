"""
Dataset Loader for MARIDA with ERA5 Environmental Integration.

Loads 5-class Sentinel-2 patches alongside matched ERA5 environmental reanalysis data
(Sea Surface Temperature and 10m Wind components).
"""

import os
import json
import torch
import numpy as np
import tifffile
from torch.utils.data import Dataset


class MARIDAEra5Dataset(Dataset):
    """
    MARIDA dataset loader with seamless ERA5 environmental vector attachment.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        era5_json_path: str = None
    ):
        """
        Args:
            root_dir: Path to MARIDA_5Class directory (containing splits/ and patches/)
            split: "train", "val", or "test"
            era5_json_path: Path to era5_features.json lookup. If None, defaults to data/era5_cache/era5_features.json
        """
        self.root_dir = os.path.abspath(root_dir)
        self.split = split
        self.split_file = os.path.join(self.root_dir, "splits", f"{split}_X.txt")

        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        with open(self.split_file, "r") as f:
            self.patch_ids = [line.strip() for line in f if line.strip()]

        # Resolve ERA5 lookup
        if era5_json_path is None:
            # Check default locations
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            default_path = os.path.join(repo_root, "data", "era5_cache", "era5_features.json")
            if os.path.exists(default_path):
                era5_json_path = default_path

        self.era5_data = {}
        if era5_json_path and os.path.exists(era5_json_path):
            with open(era5_json_path, "r") as f:
                self.era5_data = json.load(f)
            print(f"[MARIDAEra5Dataset] Loaded {len(self.era5_data)} ERA5 entries from {era5_json_path}")
        else:
            print(f"[MARIDAEra5Dataset] WARNING: ERA5 JSON not found at {era5_json_path}. Environmental values will be default.")

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        scene = patch_id.rsplit('_', 1)[0]

        img_path = os.path.join(self.root_dir, "patches", f"S2_{scene}", f"S2_{patch_id}.tif")
        mask_path = os.path.join(self.root_dir, "patches", f"S2_{scene}", f"S2_{patch_id}_cl.tif")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image patch missing: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask patch missing: {mask_path}")

        img = tifffile.imread(img_path).astype(np.float32)
        mask = tifffile.imread(mask_path).astype(np.int64)

        # Ensure (C, H, W) format
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=0)

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # Retrieve matched ERA5 features
        era5_meta = self.era5_data.get(patch_id, {
            "u10": 0.0,
            "v10": 0.0,
            "sst": 300.0  # ~27 deg C standard ocean default if missing
        })
        era5_meta["patch_id"] = patch_id

        return torch.tensor(img), torch.tensor(mask), era5_meta
