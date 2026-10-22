"""PNG layers for the browser: true colour, prediction, ground truth, confidence."""
import base64
import io

import numpy as np
from PIL import Image

from . import config
from .features import B2, B3, B4


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def true_colour(img, valid):
    rgb = np.stack([img[B4], img[B3], img[B2]], axis=-1)
    vals = rgb[valid] if valid.any() else rgb.reshape(-1, 3)
    # One stretch for all three channels keeps natural colours (a per-channel
    # stretch turns the near-uniform dark water into colour noise).
    lo, hi = float(np.percentile(vals, 0.5)), float(np.percentile(vals, 99.8))
    out = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) ** 0.75
    out = (out * 255).astype(np.uint8)
    out[~valid] = 0
    return _png(out)


PALETTE = np.zeros((256, 4), dtype=np.uint8)
for k, v in config.CLASSES.items():
    PALETTE[k] = v["color"]


def class_layer(labels):
    return _png(PALETTE[labels])


def ground_truth_layer(gt):
    pal = PALETTE.copy()
    pal[1] = [30, 90, 200, 110]
    return _png(pal[gt])


# compact "magma"-like ramp: low confidence = dark purple, high = yellow
_RAMP = np.array([[0, 0, 4], [80, 18, 123], [182, 54, 121], [251, 136, 97], [252, 253, 191]], float)


def confidence_layer(conf, valid):
    x = np.clip(conf, 0.25, 1.0)
    t = (x - 0.25) / 0.75 * (len(_RAMP) - 1)
    i = np.clip(t.astype(int), 0, len(_RAMP) - 2)
    f = (t - i)[..., None]
    rgb = _RAMP[i] * (1 - f) + _RAMP[i + 1] * f
    a = np.where(valid, 230, 0)[..., None]
    return _png(np.concatenate([rgb, a], axis=-1).astype(np.uint8))


def raw_b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()
