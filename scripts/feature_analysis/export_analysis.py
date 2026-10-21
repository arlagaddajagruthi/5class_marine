"""
Copy the outputs of per_class_features.py and shap_figures.py into one
self-contained folder, SHAP/<name>/, in the six-folder layout:

  01_global_importance/            <- shap_figures/01_global_importance
  02_class_specific/               <- shap_figures/02_class_specific
      <dataset>/feature_distributions/   <- <dataset>/1_distributions (model-free, all 30 features)
      <dataset>/shap_summary/            <- <dataset>/2_shap (per-class SHAP tables and summary plots)
  03_cross_model_comparison/       <- shap_figures/03_cross_model_comparison
      <dataset>/staged_experiments/      <- <dataset>/0_staged (per-class F1 change of every stage)
  04_dependence_and_interactions/  <- shap_figures/04_dependence_and_interactions
  05_individual_predictions/       <- shap_figures/05_individual_predictions
  06_reports/                      <- shap_figures/06_reports (figure index, CSV tables)
      <dataset>/feature_selection/       <- <dataset>/3_selection (per-class top-k SHAP selection)
      <dataset>/summary.md, run_config.json
  README.md                        <- docs/feature_analysis.md, links rewritten to this layout

The SHAP value arrays (*.npz) are not copied (they are git-ignored).

Usage (repo root): python scripts/feature_analysis/export_analysis.py --name Analysis1
"""
import argparse
import os
import re
import shutil

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = "outputs/feature_analysis"
FIG = f"{SRC}/shap_figures"
DATASETS = ["MARIDA", "MADOS"]
FOLDERS = ["01_global_importance", "02_class_specific", "03_cross_model_comparison",
           "04_dependence_and_interactions", "05_individual_predictions", "06_reports"]
LAYOUT = """## Folder layout

Every figure exists for both datasets (`MARIDA/`, `MADOS/` subfolders) and all
five ML models (RF, XGBoost, LightGBM, AdaBoost, linear SVM). Files at the top
of a folder compare the two datasets.
[`06_reports/README.md`](06_reports/README.md) lists every figure with a
one-line description.

| Folder | What it answers |
|---|---|
| [`01_global_importance/`](01_global_importance) | Which features does each model use overall? |
| [`02_class_specific/`](02_class_specific) | Which features make a pixel Water, Debris, Algae or Other? Also: per-class feature distributions (`<dataset>/feature_distributions/`) and SHAP tables (`<dataset>/shap_summary/`). |
| [`03_cross_model_comparison/`](03_cross_model_comparison) | Do the models (and the two datasets) agree? Also: per-class effect of every staged experiment (`<dataset>/staged_experiments/`). |
| [`04_dependence_and_interactions/`](04_dependence_and_interactions) | How does each feature's value change the prediction, and which features act together? |
| [`05_individual_predictions/`](05_individual_predictions) | Why was this pixel classified as it was, and what causes the errors? |
| [`06_reports/`](06_reports) | Figure index, tables, per-class feature selection (`<dataset>/feature_selection/`), per-dataset summaries. |

Reproduce: `scripts/feature_analysis/` (see its README). This folder is built
by `python scripts/feature_analysis/export_analysis.py --name Analysis1`.
"""


def mapping():
    """(source, destination) pairs, repo-relative, most specific first."""
    pairs = [(f"{FIG}/{f}", f) for f in FOLDERS]
    for ds in DATASETS:
        pairs += [(f"{SRC}/{ds}/1_distributions", f"02_class_specific/{ds}/feature_distributions"),
                  (f"{SRC}/{ds}/2_shap", f"02_class_specific/{ds}/shap_summary"),
                  (f"{SRC}/{ds}/0_staged", f"03_cross_model_comparison/{ds}/staged_experiments"),
                  (f"{SRC}/{ds}/3_selection", f"06_reports/{ds}/feature_selection"),
                  (f"{SRC}/{ds}/summary.md", f"06_reports/{ds}/summary.md"),
                  (f"{SRC}/{ds}/run_config.json", f"06_reports/{ds}/run_config.json")]
    return pairs


def long_path(path):
    """Windows refuses paths over 260 characters unless they carry the \\\\?\\ prefix."""
    path = os.path.abspath(path)
    return "\\\\?\\" + path if os.name == "nt" else path


def copy(src, dst):
    if os.path.isdir(src):
        shutil.copytree(long_path(src), long_path(dst), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.npz"))
    else:
        os.makedirs(long_path(os.path.dirname(dst)), exist_ok=True)
        shutil.copy2(long_path(src), long_path(dst))


def rewrite_links(text, pairs, dest_rel):
    """Point the report's links (written relative to docs/) at the exported copies."""
    def fix(m):
        target = os.path.normpath(os.path.join("docs", m.group(2))).replace("\\", "/")
        for src, dst in sorted(pairs, key=lambda p: -len(p[0])):
            if target == src or target.startswith(src + "/"):
                return f"{m.group(1)}({dst}{target[len(src):]})"
        return f"{m.group(1)}({os.path.relpath(target, dest_rel).replace(os.sep, '/')})"
    return re.sub(r"(\]|\[[^\]]*\])\((\.\./[^)\s]+)\)", fix, text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="Analysis1", help="Folder name under SHAP/")
    ap.add_argument("--report", default="docs/feature_analysis.md",
                    help="Markdown write-up (links relative to its folder) to export as README.md")
    args = ap.parse_args()
    dest_rel = f"SHAP/{args.name}"
    dest = os.path.join(REPO, dest_rel)
    pairs = mapping()
    n = 0
    for src, dst in pairs:
        s = os.path.join(REPO, src)
        if os.path.exists(long_path(s)):
            copy(s, os.path.join(dest, dst))
            n += 1
        else:
            print(f"  missing, skipped: {src}")
    report_src = os.path.join(REPO, args.report)
    if os.path.exists(report_src):
        with open(report_src, encoding="utf-8") as f:
            title, _, body = rewrite_links(f.read(), pairs, dest_rel).partition("\n")
        with open(os.path.join(dest, "README.md"), "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n{LAYOUT}\n{body}")
    else:
        print(f"  no report at {args.report}, README.md not written")
    print(f"{n} sources -> {dest_rel}/")


if __name__ == "__main__":
    main()
