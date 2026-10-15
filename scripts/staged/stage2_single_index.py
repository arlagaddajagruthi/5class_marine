"""
Stage 2: the 11 bands plus ONE spectral index at a time, to measure how much
each index adds on top of the raw bands (compare with stage 1).

  2a  + NDVI   (B8, B4)                   vegetation
  2b  + NDWI   (B3, B8)                   open water
  2c  + FAI    (B4, B8, B11; Hu 2009)     floating algae
  2d  + FDI    (B6, B8, B11; Biermann 2020) floating debris
  2e  + NDMI   (B8, B11)                  moisture

Also run by default, as they are part of the previous 24-feature set:
  2f  + band ratios (red/green, nir/red, nir/green, swir1/nir)
  2g  + spectral statistics (mean, std, max, norm over the 11 bands)

Output: outputs/staged/stage2<letter>_<name>/<dataset>/
Use --only 2a 2d to run a subset.
"""
from common import BAND_NAMES, RATIOS, SPEC_STATS, base_parser, run_stage

SUBSTAGES = {
    "2a": ("ndvi", "bands + NDVI", ["NDVI"]),
    "2b": ("ndwi", "bands + NDWI", ["NDWI"]),
    "2c": ("fai", "bands + FAI", ["FAI"]),
    "2d": ("fdi", "bands + FDI", ["FDI"]),
    "2e": ("ndmi", "bands + NDMI", ["NDMI"]),
    "2f": ("ratios", "bands + 4 band ratios", RATIOS),
    "2g": ("specstats", "bands + 4 spectral statistics", SPEC_STATS),
}

if __name__ == "__main__":
    ap = base_parser(__doc__)
    ap.add_argument("--only", nargs="+", choices=list(SUBSTAGES), default=list(SUBSTAGES))
    args = ap.parse_args()
    for key in args.only:
        name, desc, extra = SUBSTAGES[key]
        run_stage(f"stage{key}_{name}", desc, list(BAND_NAMES) + extra, args)
