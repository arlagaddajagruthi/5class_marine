# SegNeXt-T and the MariNeXt architecture, 4 classes

`segnext.py` trains SegNeXt-T (MSCAN-T encoder + LightHamHead,
`segnext_model.py`) exactly like the U-Net in `scripts/unet_ssl4eo/`: the same
16 input channels (11 bands + NDWI + 4 ratios), recipe per dataset, weighted
CE, augmentation, optional VSCP, checkpoint by validation macro F1 and 8-view
TTA. Only the model, the band scaling and the output folder differ:

- The model is trained from scratch (no Sentinel-2 pretrained MSCAN exists; the
  MADOS paper also trained it from scratch).
- The bands are standardised with the dataset mean and std, as in the MADOS
  code.
- `--first-stage` adds MariNeXt's stage-1 features (H/4 output instead of
  H/8). This is the MariNeXt architecture of the MADOS paper, and its outputs
  go to `outputs/marinext_arch/` instead of `outputs/segnext/`.

## Run (from the repo root)
```bash
python scripts/segnext/segnext.py train --dataset MADOS --run MADOS_vscp --vscp                 # SegNeXt
python scripts/segnext/segnext.py test  --dataset MADOS --run MADOS_vscp --vscp
python scripts/segnext/segnext.py train --dataset MADOS --run MADOS_vscp --vscp --first-stage   # MariNeXt architecture
python scripts/segnext/segnext.py test  --dataset MADOS --run MADOS_vscp --vscp --first-stage
```
`check` tests GPU memory and speed; `--resume` continues an interrupted run.
`bash outputs/segnext/logs/run_all.sh` and `bash outputs/marinext_arch/logs/run_all.sh`
run all three runs (`MARIDA_novscp`, `MARIDA_vscp`, `MADOS_vscp`).

Outputs have the same layout as `outputs/unet_ssl4eo/<run>/`.

## Test results

| Model | Run | macro F1 | mIoU | kappa | debris F1 |
|---|---|---|---|---|---|
| SegNeXt | MARIDA_novscp | 0.803 | 0.722 | 0.819 | 0.452 |
| SegNeXt | MARIDA_vscp | 0.800 | 0.720 | 0.822 | 0.444 |
| SegNeXt | MADOS_vscp | 0.807 | 0.720 | 0.793 | 0.470 |
| MariNeXt arch. | MARIDA_novscp | 0.809 | 0.731 | 0.867 | 0.442 |
| MariNeXt arch. | MARIDA_vscp | 0.835 | 0.761 | 0.871 | 0.501 |
| MariNeXt arch. | MADOS_vscp | 0.798 | 0.707 | 0.770 | 0.461 |

For comparison, the MADOS paper's MariNeXt, collapsed to our 4 classes
(`outputs/paper_compare/`), reaches macro F1 0.814, mIoU 0.724, kappa 0.791
and debris F1 0.509.
