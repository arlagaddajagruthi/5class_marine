"""
Stage 1: only the information stored in the dataset - the 11 Sentinel-2
bands (B1-B12), no spectral indices and no environmental data.

Output: outputs/staged/stage1_bands/<dataset>/
"""
from common import BAND_NAMES, base_parser, run_stage

if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    run_stage("stage1_bands", "11 Sentinel-2 bands only", list(BAND_NAMES), args)
