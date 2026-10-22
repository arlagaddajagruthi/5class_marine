"""
How many annotated test pixels does the cloud/land guard mask by mistake?
(Every annotated pixel is ocean, so ideally none.)

    python eval_guard.py --datasets MARIDA MADOS
"""
import argparse
import json
import os

import numpy as np

from marine import config
from marine.inference import scope_mask
from marine.readers import load_sample
from train_models import NAMES, split_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["MARIDA", "MADOS"])
    args = ap.parse_args()
    rpath = os.path.join(config.MODEL_DIR, "report.json")
    report = json.load(open(rpath)) if os.path.exists(rpath) else {}
    for ds in args.datasets:
        flagged, total = np.zeros(5, np.int64), np.zeros(5, np.int64)
        for pid in split_ids(ds, "test"):
            sc = load_sample({"dataset": ds, "id": pid}, config.MARIDA_ROOT, config.MADOS_ROOT)
            gt = sc.gt.ravel()
            m = scope_mask(sc.img).ravel()
            total += np.bincount(gt, minlength=5)[:5]
            flagged += np.bincount(gt[m], minlength=5)[:5]
        report[f"{ds.lower()}_scope_guard"] = {NAMES[c]: {"flagged": int(flagged[c]), "of": int(total[c])}
                                               for c in NAMES}
        print(ds, {NAMES[c]: f"{flagged[c]}/{total[c]}" for c in NAMES}, flush=True)
    json.dump(report, open(rpath, "w"), indent=2)


if __name__ == "__main__":
    main()
