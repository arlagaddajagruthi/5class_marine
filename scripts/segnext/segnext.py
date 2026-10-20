"""
SegNeXt-T (MSCAN-T + LightHamHead, segnext_model.py), run exactly like the
U-Net in scripts/unet_ssl4eo/unet.py: same 16 input channels (11 bands + NDWI
+ 4 ratios), same recipe per dataset (MARIDA 45 epochs; MADOS 80 epochs + EMA),
same weighted CE, augmentation, optional VSCP, checkpoint by validation macro F1
with TTA, 8-view TTA at test, same class-wise reports. unet.py's commands are
reused unchanged; only the model, the band scaling and the output folder differ:

  - model: SegNeXt-T trained FROM SCRATCH (no Sentinel-2 pretrained MSCAN
    exists; the MADOS paper also trained it from scratch). --first-stage adds
    MariNeXt's stage-1 features (H/4 output) instead of the published H/8.
  - bands: standardised with the dataset mean/std (as in the MADOS code), since
    there are no pretrained filters whose input scale must be matched.

Usage (repo root):
  python scripts/segnext/segnext.py check --dataset MARIDA --run MARIDA_vscp --vscp
  python scripts/segnext/segnext.py train --dataset MARIDA --run MARIDA_vscp --vscp [--resume]
  python scripts/segnext/segnext.py test  --dataset MARIDA --run MARIDA_vscp --vscp
Outputs: outputs/segnext/<run>/   (outputs/marinext_arch/<run>/ with --first-stage)
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "unet_ssl4eo"))
import unet  # noqa: E402
from segnext_model import SegNeXt  # noqa: E402

FIRST_STAGE = False


class Seg16Std(unet.Seg16):
    """unet.Seg16 with the 11 bands standardised again (extra channels unchanged)."""

    def __getitem__(self, i):
        x, t = super().__getitem__(i)
        x[:11] = (x[:11] - self.m) / self.s
        return x, t


def build_model(pretrained=False):
    return SegNeXt(11 + len(unet.EXTRA_NAMES), 4, first_stage=FIRST_STAGE)


def recipe(dataset, use_vscp):
    r = _unet_recipe(dataset, use_vscp)
    r.update(encoder="SegNeXt-T (MSCAN-T + LightHamHead), from scratch, "
                     + ("stages 1-4 (H/4, MariNeXt modelling)" if FIRST_STAGE else "stages 2-4 (H/8, as published)"),
             inputs="11 bands (standardised) + " + ", ".join(unet.EXTRA_NAMES))
    return r


_unet_recipe = unet.recipe

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "train", "test"])
    ap.add_argument("--dataset", choices=["MARIDA", "MADOS"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--vscp", action=argparse.BooleanOptionalAction, required=True)
    ap.add_argument("--first-stage", action="store_true", help="MariNeXt's stage-1 features (H/4 output)")
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    FIRST_STAGE = a.first_stage
    repo = os.path.abspath(os.path.join(HERE, "..", ".."))
    unet.OUT_ROOT = os.path.join(repo, "outputs", "marinext_arch" if FIRST_STAGE else "segnext")
    unet.TITLE = "MariNeXt-architecture SegNeXt-T (from scratch)" if FIRST_STAGE else "SegNeXt-T (from scratch)"
    unet.build_model, unet.Seg16, unet.recipe = build_model, Seg16Std, recipe
    torch.backends.cudnn.benchmark = False
    {"check": unet.cmd_check, "train": unet.cmd_train, "test": unet.cmd_test}[a.command](a)
