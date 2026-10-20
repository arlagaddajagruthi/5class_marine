"""
Stage 5: the 11 bands plus only the additions that helped on their own in
stage 2: NDWI and the 4 band ratios (red/green, nir/red, nir/green, swir1/nir).
NDVI, FAI, FDI and NDMI are dropped; adding them alone gave no consistent gain.

Output: <--out-root, default outputs/staged>/stage5_ndwi_ratios/<dataset>/
"""
from common import BAND_NAMES, RATIOS, base_parser, run_stage

if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    run_stage("stage5_ndwi_ratios", "bands + NDWI + 4 band ratios",
              list(BAND_NAMES) + ["NDWI"] + RATIOS, args)
