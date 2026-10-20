"""
Stage 8: stage 5 (11 bands + NDWI + 4 band ratios) plus the 6 GLCM texture
features. Run only if stage 7 shows that GLCM adds something over the bands
alone (rule in scripts/staged_ml/glcm_decision.py, fixed before stage 7 ran).

Output: <--out-root, default outputs/staged>/stage8_ndwi_ratios_glcm/<dataset>/
"""
from common import BAND_NAMES, RATIOS, base_parser, run_stage
from stage7_glcm import glcm_columns

if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    run_stage("stage8_ndwi_ratios_glcm", "bands + NDWI + 4 band ratios + 6 GLCM texture features",
              list(BAND_NAMES) + ["NDWI"] + RATIOS, args, extra=glcm_columns)
