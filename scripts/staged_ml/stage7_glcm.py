"""
Stage 7: the 11 bands plus the 6 GLCM texture features of the MARIDA/MADOS
random-forest baselines (contrast, dissimilarity, homogeneity, energy,
correlation, ASM; 13 x 13 window on the RGB greyscale, see glcm_features.py).
Compared with stage 1 (bands only), it measures what spatial texture adds.

Output: <--out-root, default outputs/staged>/stage7_glcm/<dataset>/
"""
import numpy as np

from common import BAND_NAMES, base_parser, run_stage
from glcm_features import GLCM_NAMES, split_features


def glcm_columns(dataset_name, keep_tr, keep_te):
    G_tr, G_te = split_features(dataset_name, "train"), split_features(dataset_name, "test")
    return G_tr[keep_tr].astype(np.float32), G_te[keep_te].astype(np.float32), GLCM_NAMES


if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    run_stage("stage7_glcm", "bands + 6 GLCM texture features", list(BAND_NAMES), args, extra=glcm_columns)
