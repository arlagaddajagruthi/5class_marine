"""
AdaBoost Pipeline with Sea Surface Temperature (SST) & Wind Currents (ERA5) Integration.

Trains and evaluates AdaBoost on MARIDA_5Class with:
  - 21 Baseline spectral features
  - 24 Standard ERA5 features (+ u10, v10, SST)
  - 29 Engineered ERA5 features (+ wind speed, wind dir, wind stress, SST, wind-SST interaction)

Outputs are saved in `outputs/era5_integrated/`.
"""

import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_all_era5_benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run AdaBoost with ERA5 Integration")
    parser.add_argument("--dataset_dir", type=str, default="data/MARIDA_5Class", help="Dataset path")
    parser.add_argument("--era5_json", type=str, default="data/era5_cache/era5_features.json", help="ERA5 JSON path")
    parser.add_argument("--modes", nargs="+", default=["baseline", "standard", "engineered"], help="Modes to run")
    parser.add_argument("--out_dir", type=str, default="outputs/era5_integrated", help="Output directory")
    parser.add_argument("--max_train_pixels", type=int, default=500000, help="Max train pixels")
    args = parser.parse_args()

    run_benchmark(
        dataset_dir=args.dataset_dir,
        era5_json_path=args.era5_json,
        models=["adaboost"],
        modes=args.modes,
        output_base_dir=args.out_dir,
        max_train_pixels=args.max_train_pixels
    )


if __name__ == "__main__":
    main()
