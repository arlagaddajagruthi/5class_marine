"""
Bulk ERA5 Download and Lookup Builder for MARIDA Marine Pollution Pipeline.

This script:
  1. Scans all MARIDA scene folders to determine required (year, month) batches
  2. Downloads ERA5 reanalysis data from CDS API in monthly batches (~20 requests)
  3. Extracts per-patch environmental vectors (wind U/V, SST)
  4. Saves a JSON lookup file for use during model training

Usage:
    python scripts/download_era5.py --dataset_dir data/MARIDA_5Class --cache_dir data/era5_cache
    python scripts/download_era5.py --dry_run   # Preview downloads without calling CDS API

Prerequisites:
    - Valid ~/.cdsapirc with CDS API v2 credentials
    - pip install cdsapi rasterio xarray netCDF4 scipy
"""

import os
import sys
import argparse
import time

# Ensure project root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing.era5_integration import (
    ERA5BulkDownloader,
    build_era5_lookup,
)


def main():
    parser = argparse.ArgumentParser(
        description="Bulk download ERA5 data and build feature lookup for MARIDA"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "MARIDA_5Class"),
        help="Root directory of MARIDA_5Class dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "era5_cache"),
        help="Directory to store cached ERA5 NetCDF files",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview what would be downloaded without calling CDS API",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download phase and only build the lookup from existing cache",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path for output JSON lookup (default: cache_dir/era5_features.json)",
    )
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    cache_dir = os.path.abspath(args.cache_dir)

    print("=" * 60)
    print("  ERA5 Bulk Download & Lookup Builder")
    print("=" * 60)
    print(f"\n  Dataset:    {dataset_dir}")
    print(f"  Cache:      {cache_dir}")
    print(f"  Dry run:    {args.dry_run}")
    print(f"  Skip DL:    {args.skip_download}")
    print()

    # Validate dataset dir
    patches_dir = os.path.join(dataset_dir, "patches")
    if not os.path.isdir(patches_dir):
        print(f"[ERROR] Patches directory not found: {patches_dir}")
        sys.exit(1)

    # Phase 1: Download
    if not args.skip_download:
        print("-" * 60)
        print("  Phase 1: Bulk ERA5 Download")
        print("-" * 60)

        downloader = ERA5BulkDownloader(dataset_dir, cache_dir)

        # Show scan summary
        scenes = downloader.scan_scenes()
        monthly = downloader.build_monthly_requests(scenes)

        print(f"\n  Scenes found:     {len(scenes)}")
        print(f"  Monthly batches:  {len(monthly)}")
        print(f"  Date range:       "
              f"{min(s['date_str'] for s in scenes)} to "
              f"{max(s['date_str'] for s in scenes)}")

        print("\n  Monthly download plan:")
        for (year, month), info in sorted(monthly.items()):
            nc_path = downloader._nc_path(year, month)
            cached = os.path.exists(nc_path)
            status = "CACHED" if cached else "PENDING"
            print(f"    {year}-{month}: {len(info['days'])} days, "
                  f"area={[f'{x:.1f}' for x in info['area']]} [{status}]")

        start_time = time.time()
        downloaded = downloader.download_all(dry_run=args.dry_run)
        elapsed = time.time() - start_time

        print(f"\n  Download phase complete: {len(downloaded)} files in {elapsed:.1f}s")

        if args.dry_run:
            print("\n  [DRY RUN] No files were actually downloaded.")
            print("  Remove --dry_run to execute the downloads.")
            return
    else:
        print("  Skipping download phase (--skip_download)")

    # Phase 2: Build lookup
    print("\n" + "-" * 60)
    print("  Phase 2: Building ERA5 Feature Lookup")
    print("-" * 60)

    start_time = time.time()
    output_json = args.output_json or os.path.join(cache_dir, "era5_features.json")
    lookup = build_era5_lookup(dataset_dir, cache_dir, output_json=output_json)
    elapsed = time.time() - start_time

    print(f"\n  Lookup built in {elapsed:.1f}s")
    print(f"  Total entries: {len(lookup)}")
    print(f"  Output file:   {output_json}")

    # Print sample entries
    sample_keys = sorted(lookup.keys())[:5]
    print(f"\n  Sample entries:")
    for key in sample_keys:
        v = lookup[key]
        sst_c = v['sst'] - 273.15 if v['sst'] > 0 else float('nan')
        fallback = " [NaN fallback]" if v.get("sst_nan_fallback") else ""
        print(f"    {key}: u10={v['u10']:.3f}, v10={v['v10']:.3f}, "
              f"sst={v['sst']:.2f}K ({sst_c:.1f}C){fallback}")

    print("\n" + "=" * 60)
    print("  Done! You can now use ERA5 features in your training pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()
