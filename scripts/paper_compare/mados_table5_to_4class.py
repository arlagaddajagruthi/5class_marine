"""
Collapse the MariNeXt confusion matrix published in the MADOS paper (Kikaki et
al. 2024, ISPRS J. 210:39-54, Table 5: row percentages + test pixels per class)
to our four classes, so that the paper's model can be compared with ours on the
same label space and the same test pixels.

Pixel counts are reconstructed as row% / 100 * pixels, so each cell carries a
rounding error of at most 0.05 % of its row (the table gives one decimal).
Check: the published per-class pixel counts sum to our MADOS test supports
exactly (Water 294,020 / Debris 973 / Algae 3,089 / Other 63,065); the
reconstructed row sums differ from them by under 0.1 % (row-% rounding), and
the 15-class scores recomputed from the reconstruction match the paper's.

Usage (repo root): python scripts/paper_compare/mados_table5_to_4class.py
Output: outputs/paper_compare/mados_marinext_table5_4class.md
"""
import os

import numpy as np

CLASSES = ["MD", "DenS", "SpFA", "NatM", "Ship", "Oil", "MWater", "SLWater", "Foam", "TWater", "SWater",
           "Waves", "OilPlat", "Jellyfish", "Sea snot"]
# rows = reference, columns = predicted (%), transcribed from Table 5
ROW_PCT = [
    [70.4, 0.0, 2.1, 0.0, 11.1, 0.1, 12.7, 0.0, 0.0, 0.0, 0.0, 3.1, 0.4, 0.0, 0.0],
    [0.5, 70.0, 28.2, 1.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.2, 1.3, 90.4, 0.2, 0.0, 0.0, 6.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [22.7, 0.0, 5.3, 46.1, 17.6, 0.1, 5.7, 0.0, 0.0, 0.2, 0.0, 2.3, 0.0, 0.0, 0.0],
    [3.1, 0.0, 0.0, 0.0, 52.7, 2.1, 0.3, 0.2, 0.0, 0.4, 0.0, 1.6, 39.3, 0.3, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 70.6, 28.3, 0.0, 0.0, 0.1, 0.2, 0.7, 0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.0, 0.0, 2.6, 95.7, 0.0, 0.0, 1.1, 0.2, 0.4, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.3, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0, 4.9, 0.1, 0.3, 0.0, 80.6, 1.1, 0.0, 7.4, 0.0, 0.0, 5.2],
    [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.5, 0.1, 0.0, 89.8, 7.6, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.1, 0.0, 0.0, 12.4, 61.2, 0.3, 0.0, 0.0, 0.0],
    [2.9, 0.0, 0.0, 0.0, 2.2, 17.3, 41.7, 0.0, 0.1, 1.2, 6.1, 28.4, 0.1, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.0, 3.5, 1.1, 0.3, 0.0, 0.0, 0.0, 0.0, 0.5, 94.6, 0.0, 0.0],
    [1.1, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 13.1, 0.0, 84.4, 0.0],
    [22.4, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 69.6],
]
PIXELS = [973, 1303, 1786, 278, 3574, 51528, 139465, 93254, 387, 42168, 12732, 6401, 4790, 274, 2234]
# MADOS class -> our class (same as scripts/convert.py): 0 water, 1 debris, 2 algae, 3 other
TO4 = {"MD": 1, "DenS": 2, "SpFA": 2, "NatM": 3, "Ship": 3, "Oil": 3, "MWater": 0, "SLWater": 0, "Foam": 3,
       "TWater": 0, "SWater": 0, "Waves": 0, "OilPlat": 3, "Jellyfish": 3, "Sea snot": 3}
NAMES4 = ["Water", "Marine Debris", "Algae/Sargassum", "Other"]


def scores(cm):
    tp = np.diag(cm)
    p, r = tp / cm.sum(0), tp / cm.sum(1)
    f1, iou = 2 * p * r / (p + r), tp / (cm.sum(0) + cm.sum(1) - tp)
    n = cm.sum()
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    kappa = (tp.sum() / n - pe) / (1 - pe)
    return p, r, f1, iou, tp.sum() / n, kappa


def main():
    cm15 = np.array(ROW_PCT) / 100 * np.array(PIXELS)[:, None]
    p15, r15, f15, i15, oa15, _ = scores(cm15)
    m = np.zeros((15, 4))
    m[np.arange(15), [TO4[c] for c in CLASSES]] = 1
    cm4 = m.T @ cm15 @ m
    p, r, f1, iou, oa, kappa = scores(cm4)
    lines = ["# MADOS paper, MariNeXt (Table 5) collapsed to our four classes", "",
             "Reconstructed from row percentages x test pixels (rounding error <= 0.05 % of a row per cell).", "",
             f"Sanity check, 15 classes: macro F1 {100 * f15.mean():.1f} (paper 76.0), mIoU {100 * i15.mean():.1f} "
             f"(paper 64.3), OA {100 * oa15:.1f} (paper 89.1).", "",
             f"Four classes: macro F1 **{f1.mean():.4f}**, mIoU **{iou.mean():.4f}**, OA {oa:.4f}, kappa {kappa:.4f}, "
             f"macro precision {p.mean():.4f}, macro recall {r.mean():.4f}.", "",
             "| class | support | precision | recall | F1 | IoU |", "|---|---|---|---|---|---|"]
    for k in range(4):
        lines.append(f"| {NAMES4[k]} | {cm4[k].sum():,.0f} | {p[k]:.4f} | {r[k]:.4f} | {f1[k]:.4f} | {iou[k]:.4f} |")
    lines += ["", "Confusion matrix (pixels, rows = reference, columns = predicted)", "",
              "| | " + " | ".join(NAMES4) + " |", "|---" * 5 + "|"]
    for k in range(4):
        lines.append(f"| {NAMES4[k]} | " + " | ".join(f"{v:,.0f}" for v in cm4[k]) + " |")
    out = os.path.join("outputs", "paper_compare")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "mados_marinext_table5_4class.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
