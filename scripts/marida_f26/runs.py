"""Run IDs of the 26-feature study, their folder names, and the dataset being studied.

Every script of the study takes `--dataset MARIDA|MADOS` (default MARIDA). Importing
this module reads it from the command line (or the F26_DATASET environment variable),
so it must be imported before features26:
  MARIDA -> outputs/marida_f26/,  MADOS -> outputs/mados_f26/
"""
import os
import sys

RUN_NAMES = {
    "R1": "R1_unet_ssl4eo_raw",           # U-Net, SSL4EO-S12 encoder, raw-reflectance bands
    "R2": "R2_unet_ssl4eo_std",           # U-Net, SSL4EO-S12 encoder, standardised bands
    "R3": "R3_unet_scratch_std",          # U-Net from scratch, standardised bands
    "R4": "R4_segnext_scratch_std",       # SegNeXt-T from scratch, standardised bands
    "R5": "R5_marinext_scratch_std",      # MariNeXt architecture from scratch, standardised bands
    "R6": "R6_rf",                        # Random Forest on the same 26 features
    "R7": "R7_ensemble_unet_rf",          # R3 + R6 probability ensemble
    "G2b": "G2b_unet_ssl4eo_raw_11class",  # as R1, in the MARIDA paper's 11 classes
}
DATASETS = ("MARIDA", "MADOS")


def _dataset():
    if "--dataset" in sys.argv:
        i = sys.argv.index("--dataset")
        os.environ["F26_DATASET"] = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    d = os.environ.get("F26_DATASET", "MARIDA")
    if d not in DATASETS:
        raise SystemExit(f"--dataset must be one of {DATASETS}, not {d}")
    return d


DATASET = _dataset()
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
STUDY = os.path.join(REPO, "outputs", f"{DATASET.lower()}_f26")


def name(run_id):
    """'R1' -> 'R1_unet_ssl4eo_raw'; full names are returned unchanged."""
    return RUN_NAMES.get(run_id, run_id)
