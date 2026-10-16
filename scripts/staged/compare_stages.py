"""
Collect every finished stage into one comparison, with deltas against stage 1
(bands only) for the same dataset and model.

Output: <--out-root, default outputs/staged>/comparison/
  all_results.csv, comparison.md, delta_{f1,iou,debris_f1,debris_iou}_<dataset>.png
"""
import argparse
import glob
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import CLASS_NAMES, CLASSES, OUT_ROOT

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=OUT_ROOT, help="Folder holding the stage outputs (default: outputs/staged)")
    root = ap.parse_args().out_root
    OUT = os.path.join(root, "comparison")
    files = sorted(glob.glob(os.path.join(root, "stage*", "*", "metrics.csv")))
    if not files:
        raise SystemExit("No stage results found; run the stage scripts first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    base = df[df.stage == "stage1_bands"].set_index(["dataset", "model"])
    keys = ["f1", "iou", "accuracy"] + [f"{m}_c{c}" for c in CLASSES for m in ("f1", "iou", "acc")]
    for k in keys:
        df[f"d_{k}"] = [r[k] - base.loc[(r.dataset, r.model), k] if (r.dataset, r.model) in base.index else float("nan")
                        for _, r in df.iterrows()]
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "all_results.csv"), index=False)

    tables = [("f1", "Macro F1"), ("iou", "Macro IoU")]
    for c in (2, 3):
        tables += [(f"f1_c{c}", f"F1 {CLASS_NAMES[c]}"), (f"iou_c{c}", f"IoU {CLASS_NAMES[c]}")]
    tables += [(f"acc_c{c}", f"Class accuracy {CLASS_NAMES[c]} (TP / class pixels)") for c in CLASSES]
    stage_order = sorted(df.stage.unique())
    lines = ["# Staged feature comparison", "",
             "Scores over classes 1-4 on all annotated test pixels (macro = mean over the 4 classes). "
             "Each cell: score (Δ vs stage 1, bands only, same model and dataset).", ""]
    for ds, g in df.groupby("dataset"):
        lines += [f"## {ds}", ""]
        piv = g.pivot(index="stage", columns="model", values="f1").reindex(stage_order).dropna(how="all")
        models = list(piv.columns)
        for key, title in tables:
            p = g.pivot(index="stage", columns="model", values=key).reindex(piv.index)
            d = g.pivot(index="stage", columns="model", values=f"d_{key}").reindex(piv.index)
            lines += [f"### {title}", "", "| Stage | " + " | ".join(models) + " | mean Δ |",
                      "|" + "---|" * (len(models) + 2)]
            for st in p.index:
                lines.append(f"| {st} | " + " | ".join(f"{p.loc[st, m]:.4f} ({d.loc[st, m]:+.4f})" for m in models)
                             + f" | {d.loc[st].mean():+.4f} |")
            lines.append("")

        for key, label, fname in [("d_f1", "Δ macro F1 vs stage 1", "delta_f1"),
                                  ("d_iou", "Δ macro IoU vs stage 1", "delta_iou"),
                                  ("d_f1_c2", "Δ Marine Debris F1 vs stage 1", "delta_debris_f1"),
                                  ("d_iou_c2", "Δ Marine Debris IoU vs stage 1", "delta_debris_iou")]:
            dp = g.pivot(index="stage", columns="model", values=key).reindex(piv.index)
            dp = dp.drop(index="stage1_bands", errors="ignore").dropna(how="all")
            if dp.empty:
                continue
            ax = dp.plot.barh(figsize=(10, 6))
            ax.axvline(0, color="k", lw=0.8); ax.set_xlabel(label); ax.set_title(f"{ds}: {label}")
            ax.invert_yaxis(); plt.tight_layout()
            plt.savefig(os.path.join(OUT, f"{fname}_{ds}.png"), dpi=110); plt.close()

    with open(os.path.join(OUT, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
