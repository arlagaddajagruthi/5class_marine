import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from torch.utils.data import DataLoader

from src.datasets.marida import MARIDADataset

ROOT = "C:/PROJECT/5class_marine_pollution/data/MARIDA_5Class"

dataset = MARIDADataset(ROOT, split="train")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

num_classes = 5

pixel_counts = np.zeros(num_classes, dtype=np.int64)

for _, mask in loader:

    mask = mask.numpy()

    for c in range(num_classes):
        pixel_counts[c] += np.sum(mask == c)

total = pixel_counts.sum()

print("\nPixel Distribution\n")

for c in range(num_classes):
    pct = 100 * pixel_counts[c] / total
    print(f"Class {c}: {pixel_counts[c]:12,d} pixels ({pct:.4f}%)")