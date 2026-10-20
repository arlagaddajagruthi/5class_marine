"""
Stage 3: the 11 bands plus all five spectral indices (NDVI, NDWI, FAI, FDI, NDMI).

Output: outputs/staged/stage3_all_indices/<dataset>/
(Stage 4, adding environmental features, will build on this.)
"""
from common import BAND_NAMES, INDICES, base_parser, run_stage

if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    run_stage("stage3_all_indices", "bands + all 5 indices", list(BAND_NAMES) + INDICES, args)
