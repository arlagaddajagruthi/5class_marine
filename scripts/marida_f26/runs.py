"""Run IDs of the MARIDA 26-feature study and their folder names (outputs/marida_f26/runs/<name>/)."""

RUN_NAMES = {
    "R1": "R1_unet_ssl4eo_raw",           # U-Net, SSL4EO-S12 encoder, raw-reflectance bands
    "R2": "R2_unet_ssl4eo_std",           # U-Net, SSL4EO-S12 encoder, standardised bands
    "R3": "R3_unet_scratch_std",          # U-Net from scratch, standardised bands
    "R4": "R4_segnext_scratch_std",       # SegNeXt-T from scratch, standardised bands
    "R5": "R5_marinext_scratch_std",      # MariNeXt architecture from scratch, standardised bands
    "R6": "R6_rf",                        # Random Forest on the same 26 features
    "R7": "R7_ensemble_unet_rf",          # R1 + R6 probability ensemble
    "G2b": "G2b_unet_ssl4eo_raw_11class",  # as R1, in the MARIDA paper's 11 classes
}


def name(run_id):
    """'R1' -> 'R1_unet_ssl4eo_raw'; full names are returned unchanged."""
    return RUN_NAMES.get(run_id, run_id)
