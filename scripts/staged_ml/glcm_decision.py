"""
Does GLCM texture add anything? Decides whether stage 8 runs.

Rule (fixed before stage 7 was run): stage 8 runs if, for at least one class
weighting (balanced, log, none) and at least one dataset, stage 7 (bands + GLCM)
beats stage 1 (bands only) by more than 0.01 in the mean over the five models
of either the macro F1 or the marine-debris F1. 0.01 is the single-seed noise
level stated in the paper.

Exit code 0 = run stage 8, 2 = do not (1 would be indistinguishable from a crash). Prints the deltas either way and writes
<outputs/staged>/glcm_decision.md.

Usage (repo root): python scripts/staged_ml/glcm_decision.py
"""
import os
import sys

import pandas as pd

from common import OUT_ROOT

ROOTS = {"balanced": OUT_ROOT, "log": os.path.join(OUT_ROOT, "log_weights"),
         "none": os.path.join(OUT_ROOT, "no_class_weights")}
THRESHOLD = 0.01


def main():
    rows = []
    for w, root in ROOTS.items():
        for ds in ("MARIDA", "MADOS"):
            s1 = pd.read_csv(os.path.join(root, "stage1_bands", ds, "metrics.csv"))
            s7 = pd.read_csv(os.path.join(root, "stage7_glcm", ds, "metrics.csv"))
            r = {"weighting": w, "dataset": ds}
            for k, name in (("f1", "macroF1"), ("f1_c2", "debrisF1")):
                r[f"stage1_{name}"], r[f"stage7_{name}"] = s1[k].mean(), s7[k].mean()
                r[f"delta_{name}"] = r[f"stage7_{name}"] - r[f"stage1_{name}"]
            rows.append(r)
    d = pd.DataFrame(rows)
    run = bool(((d.delta_macroF1 > THRESHOLD) | (d.delta_debrisF1 > THRESHOLD)).any())
    lines = ["# Does GLCM add anything? (stage 7 vs stage 1, mean of 5 models)", "",
             f"Rule: run stage 8 if any delta > {THRESHOLD} (any weighting, any dataset, macro or debris F1).", "",
             d.round(4).to_markdown(index=False), "", f"Decision: **{'run stage 8' if run else 'do not run stage 8'}**"]
    open(os.path.join(OUT_ROOT, "glcm_decision.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if run else 2)


if __name__ == "__main__":
    main()
