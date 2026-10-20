"""
Stage 4: stage 3 (11 bands + NDVI, NDWI, FAI, FDI, NDMI) plus the 4 band
ratios (red/green, nir/red, nir/green, swir1/nir), the best single addition
in stage 2. No environmental data yet; that will be added on top of this set.

Output: outputs/staged/stage4_indices_ratios/<dataset>/
"""
from common import BAND_NAMES, INDICES, RATIOS, base_parser, run_stage

if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    run_stage("stage4_indices_ratios", "bands + all 5 indices + 4 band ratios",
              list(BAND_NAMES) + INDICES + RATIOS, args)
