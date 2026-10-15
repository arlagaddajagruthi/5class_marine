"""
Step 1: geolocate every MARIDA_5Class patch and estimate its acquisition time.

Output: data/environmental/marida_patch_metadata.csv
"""
import argparse
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.environmental.features import PATCH_METADATA_CSV
from src.environmental.patch_geo import build_patch_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marida-root", default=os.path.join(repo_root, "data", "MARIDA_5Class"))
    parser.add_argument("--out", default=PATCH_METADATA_CSV)
    args = parser.parse_args()

    df = build_patch_metadata(args.marida_root)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Patches: {len(df)}  scenes: {df['scene'].nunique()}  dates: {df['date'].nunique()}")
    print(f"Patch footprint: {df['width_m'].iloc[0]:.0f} m x {df['height_m'].iloc[0]:.0f} m")
    print(df.groupby("split")["scene"].nunique().rename("scenes per split").to_string())
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
