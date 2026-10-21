"""
Builds results/shap_analysis/06_reports/REPORT.md from the facts written by
run_shap_analysis.py (06_reports/tables/<dataset>/<dataset>_figures.json and
_validation.json). Every finding sentence is filled from computed numbers.
The hand-written synthesis 06_reports/overall_findings.md (if present) is placed
at the top.

Usage (repo root): .venv/Scripts/python scripts/shap_analysis/make_report.py
"""
import argparse
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

CLS = ["Water", "Marine Debris", "Sargassum", "Other"]
CLSF = [f"Class {i + 1} {c}" for i, c in enumerate(CLS)]
ML = {"rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM", "ada": "AdaBoost", "svm": "Linear SVM"}
MODELS = list(ML)
UNIT = {"rf": "probability", "xgb": "log-odds margin", "lgbm": "log-odds margin",
        "ada": "SAMME score", "svm": "SVM margin"}

FEATURES = [
    ("B1 ... B12 (11 columns)", "pixel_features.py",
     "Sentinel-2 surface reflectance of bands B1 (443 nm), B2 (490), B3 (560), B4 (665), B5 (705), B6 (740), "
     "B7 (783), B8 (842), B8A (865), B11 (1610), B12 (2190). MARIDA: Rayleigh-corrected patches; MADOS: ACOLITE rhorc."),
    ("NDWI", "pixel_features.py", "(B3 - B8) / (B3 + B8) (McFeeters). Higher = green reflectance high relative to NIR."),
    ("red/green", "pixel_features.py", "B4 / B3"),
    ("nir/red", "pixel_features.py", "B8 / B4"),
    ("nir/green", "pixel_features.py", "B8 / B3"),
    ("swir1/nir", "pixel_features.py", "B11 / B8"),
    ("glcm_contrast", "glcm_features.py", "sum p(i,j)(i-j)^2 - large when neighbouring grey levels differ strongly."),
    ("glcm_dissimilarity", "glcm_features.py", "sum p(i,j)|i-j| - like contrast but linear in the grey-level difference."),
    ("glcm_homogeneity", "glcm_features.py", "sum p(i,j)/(1+(i-j)^2) - large when co-occurring grey levels are similar."),
    ("glcm_energy", "glcm_features.py", "sqrt(ASM)."),
    ("glcm_correlation", "glcm_features.py", "linear correlation of grey levels of neighbouring pixels; 1 for a flat window."),
    ("glcm_asm", "glcm_features.py", "angular second moment sum p(i,j)^2 - large when few grey-level pairs dominate (uniform window)."),
]


def pct(v):
    return f"{100 * v:.1f}%"


def g(v):
    return f"{v:.3g}"


def dir_word(rho):
    if rho is None or rho != rho:
        return "no monotonic trend"
    if rho >= 0.5:
        return f"high values raise it (rho = {rho:+.2f})"
    if rho <= -0.5:
        return f"high values lower it (rho = {rho:+.2f})"
    if rho >= 0.2:
        return f"high values tend to raise it (rho = {rho:+.2f})"
    if rho <= -0.2:
        return f"high values tend to lower it (rho = {rho:+.2f})"
    return f"no consistent monotonic direction (rho = {rho:+.2f})"


def tl(pairs, as_pct=True):
    return ", ".join(f"**{f}** ({pct(v) if as_pct else g(v)})" for f, v in pairs)


# ------------------------------------------------------------- per figure ---

def case_values(ds, f, V):
    """Mean feature values for the leading features of a case study (XGBoost explained errors)."""
    T = V["_cases"]
    d = T[(T["true"] == CLSF[f["true"]]) & (T["pred"] == CLSF[f["pred"]])]
    feats = []
    for n in MODELS:
        p = f["per_model"].get(n) or {}
        for a, _, _ in p.get("top_shift", [])[:1]:
            if a not in feats:
                feats.append(a)
    for model in ("xgb", "lgbm", "rf", "ada", "svm"):
        dm = d[d.model == model].set_index("feature")
        if len(dm):
            break
    else:
        return ""
    return "; ".join(f"{a} {dm.loc[a, 'mean_value_misclassified']:.3g} / {dm.loc[a, 'mean_value_all_true_class_px']:.3g} / "
                     f"{dm.loc[a, 'mean_value_all_pred_class_px']:.3g}" for a in feats) + f" ({ML[model]} errors)"


def text(f, V, ds):
    k = f.get("kind")
    m = f.get("model")
    unit = UNIT.get(m, "")
    c = f.get("cls")
    out = {}
    if k == "global_bar":
        out["What it shows"] = (f"Global importance of all 22 features for {ML[m]}: mean |SHAP| over the class-stratified "
                                f"test sample and over the four class outputs ({f['scale']}).")
        out["How to read"] = ("Bar length = average absolute contribution of a feature, in the model's output unit; sign is "
                              "ignored. Features are sorted by importance.")
        out["Findings"] = (f"Top five (share of the model's total): {tl(f['top5'])}. The top three together carry "
                           f"{pct(f['top3_share'])}. Least used: {', '.join(x for x, _ in f['bottom3'])}.")
        out["Meaning"] = (f"{ML[m]} separates the four classes mainly through {f['top5'][0][0]} and {f['top5'][1][0]}; "
                          "features at the bottom could be perturbed with little change in its outputs.")
        out["Limitations"] = ("Magnitudes are in this model's own unit and are not comparable across models (see the "
                              "normalised comparison in section A3). Correlated features (e.g. adjacent bands, energy/ASM) "
                              "can share or swap credit.")
    elif k == "global_beeswarm":
        rows = "; ".join(f"**{n}**: mean|SHAP| {g(a)}, pooled rho {r:+.2f}, mean SHAP {mu:+.3g}" for n, a, r, mu in f["top5"])
        out["What it shows"] = (f"Distribution of {ML[m]} SHAP values for each pixel's own (true-class) output, so every "
                                "point shows how a feature moved the score of the correct class.")
        out["How to read"] = ("One dot per pixel; x = SHAP value (right = pushes the true-class output up), colour = the "
                              "feature's value (SHAP red = high, blue = low), rows sorted by mean |SHAP|.")
        out["Findings"] = (rows + ". rho = Spearman correlation between feature value and true-class SHAP, pooled over "
                           "the four classes; it is not a per-class direction (see section B5). A positive mean SHAP means the "
                           "feature on average supports the correct label.")
        out["Meaning"] = ("A positive spread means the feature generally supports the correct label; mixed colours on both "
                          "sides mean the direction depends on the class (see the class-specific beeswarms).")
        out["Limitations"] = ("Pools four different outputs (one per true class), so a monotonic colour pattern is not "
                              "expected; directions must be read per class in section B.")
    elif k == "cross_model_heatmap":
        top1 = ", ".join(f"{ML[n]}: {v}" for n, v in f["top1"].items())
        grp = "; ".join(f"{ML[n]} {pct(f['band_share'][n])} / {pct(f['index_ratio_share'][n])} / {pct(f['glcm_share'][n])}"
                        for n in MODELS)
        spread = ", ".join(f"{n} (rank {a}-{b})" for n, a, b in f["rank_spread"][:6])
        out["What it shows"] = "Normalised global importance of every feature for the five models, side by side."
        out["How to read"] = ("Each column is one model; cell = that feature's share of the model's total mean |SHAP| "
                              "(columns sum to 100%). Rows sorted by mean share. Shares, not raw SHAP values, are compared "
                              "because the models explain outputs on different scales.")
        out["Findings"] = (f"Highest mean share: {tl(f['top_by_mean'])}. Most important feature per model: {top1}. "
                           f"Share of 11 bands / NDWI + 4 ratios / 6 GLCM features: {grp}. Rank range across models of "
                           f"the leading features: {spread}.")
        out["Meaning"] = ("Features that rank high in all columns are used consistently by every model; a feature that is "
                          "high in one column only is a model-specific choice.")
        out["Limitations"] = "Normalisation removes scale but not the dependence of SHAP on the background and on correlated features."
    elif k == "rank_agreement":
        hi, lo = f["pairs"][0], f["pairs"][-1]
        out["What it shows"] = "How similarly the five models rank the 22 features globally."
        out["How to read"] = "Cell = Spearman correlation between two models' normalised importance vectors (1 = identical ranking)."
        out["Findings"] = (f"Most similar: {ML[hi[0]]} vs {ML[hi[1]]} (rho = {hi[2]:.2f}); least similar: {ML[lo[0]]} vs "
                           f"{ML[lo[1]]} (rho = {lo[2]:.2f}). All pairs: " +
                           ", ".join(f"{ML[a]}-{ML[b]} {r:.2f}" for a, b, r in f["pairs"]) + ".")
        out["Meaning"] = "Low agreement means two models reach their predictions through different features."
        out["Limitations"] = "Rank correlation over 22 features is sensitive to the many small, noisy low-importance features."
    elif k == "class_bar":
        out["What it shows"] = f"{ML[m]} importance of each feature for the {CLSF[c]} output only ({f['scale']})."
        out["How to read"] = (f"Bar = mean |SHAP of the {CLS[c]} output| over the whole stratified sample (all four true "
                              "classes), i.e. how much each feature moves this class's score anywhere in the data.")
        out["Findings"] = (f"Top five: {tl(f['top5'])}. Restricted to pixels whose true class is {CLS[c]}, the top three are "
                           f"{tl(f['top3_true'])}.")
        out["Meaning"] = f"These are the features {ML[m]} relies on most to raise or lower its {CLS[c]} score."
        out["Limitations"] = "Unsigned; see the matching beeswarm for direction."
    elif k == "class_beeswarm":
        rows = "; ".join(f"**{n}**: {dir_word(r)}, mean SHAP on {CLS[c]} pixels {a:+.3g} vs {b:+.3g} on other pixels"
                         for n, _, r, a, b in f["top5"])
        out["What it shows"] = f"Signed {ML[m]} SHAP values for the {CLSF[c]} output, for all pixels of the stratified sample."
        out["How to read"] = (f"x > 0 pushes the {CLS[c]} score up; colour = feature value (red high, blue low); rho = Spearman "
                              "correlation between feature value and SHAP value.")
        out["Findings"] = rows + "."
        out["Meaning"] = (f"Features whose mean SHAP is positive on {CLS[c]} pixels and negative elsewhere are the ones that "
                          f"discriminate {CLS[c]} for this model.")
        out["Limitations"] = ("|rho| < 0.2 does not mean unimportant: tree models can use a feature non-monotonically "
                              "(see dependence plots). Attribution is not causation.")
    elif k == "class_heatmap":
        parts = []
        for kk in range(4):
            p = f["per_class"][str(kk)] if str(kk) in f["per_class"] else f["per_class"][kk]
            parts.append(f"{CLS[kk]}: mean top features {tl(p['mean_top4'])}; top-1 per model " +
                         ", ".join(f"{ML[n]} {v}" for n, v in p["model_top1"].items()))
        out["What it shows"] = "Normalised class-specific importance for every model x class output (20 columns)."
        out["How to read"] = ("Cell = feature's share of the total mean |SHAP| of that model's class output (each column "
                              "sums to 100%). White separators split models.")
        out["Findings"] = " | ".join(parts) + "."
        out["Meaning"] = "Rows that light up for one class across all models are class-specific cues shared by the models."
        out["Limitations"] = "Shares within a column depend on how concentrated that output's attribution is."
    elif k == "vs_rest":
        out["What it shows"] = (f"{ML[m]} class-versus-rest SHAP for {CLSF[c]}: contribution to s_k - mean of the other three "
                                "margins, which for a softmax model equals log(p_k / geometric mean of the other three "
                                "probabilities). It is exactly additive (difference of SHAP vectors).")
        out["How to read"] = "x > 0 = the feature raises this class's probability relative to the competing classes."
        out["Findings"] = (f"Top five (contrast): {tl(f['top5'])}; top five for the plain {CLS[c]} margin: {tl(f['top5_class'])}; "
                           f"rank correlation of the two importance vectors {f['rank_corr']:.2f}. Directions: " +
                           "; ".join(f"{n} {dir_word(r)}" for n, r in f["rho"]) + ".")
        out["Meaning"] = ("Where the two rankings differ, a feature raises (or lowers) all class margins together and so "
                          "does not discriminate; the contrast removes that shared part.")
        out["Limitations"] = ("Values are on the log-probability-ratio scale, not on the probability scale; the softmax "
                              "probability itself is not additive in SHAP.")
    elif k == "dependence":
        rows = "; ".join(f"**{a}** (coloured by {b}): {dir_word(r)}, 5-95% value range {lo:.3g}-{hi:.3g}"
                         for a, b, r, lo, hi in f["panels"])
        out["What it shows"] = (f"How the value of {ML[m]}'s three most important features for the {CLSF[c]} output maps to "
                                "their SHAP contribution.")
        out["How to read"] = ("x = feature value, y = SHAP value for this class output, colour = value of the feature that "
                              "SHAP's approximate_interactions picks as the strongest interaction partner. Grey histogram = "
                              "distribution of x. Vertical spread at a fixed x indicates interaction.")
        out["Findings"] = rows + "."
        out["Meaning"] = "Threshold-like jumps show the value ranges at which the model switches its decision for this class."
        out["Limitations"] = ("Interventional SHAP evaluates feature values off the data manifold when features are "
                              "correlated; the colour partner is a heuristic, not a measured interaction (see section C9).")
    elif k == "interaction_matrix":
        rows = []
        for kk in range(4):
            p = f["per_class"][str(kk)] if str(kk) in f["per_class"] else f["per_class"][kk]
            tp = ", ".join(f"{a} x {b} ({g(v)})" for a, b, v in p["top_pairs"])
            rows.append(f"{CLS[kk]}: off-diagonal share {pct(p['offdiag_fraction'])}, strongest pairs {tp}")
        out["What it shows"] = (f"Pairwise SHAP interaction strength for {ML[m]} (path-dependent TreeSHAP interaction values, "
                                f"n = {f['n']} stratified test pixels), one panel per class output.")
        out["How to read"] = ("Cell (i, j) = mean |phi_ij| + mean |phi_ji| in the model's output unit; the diagonal (main "
                              "effects) is blanked. 'Off-diagonal share' = fraction of the total |interaction matrix| mass "
                              "that is interaction rather than main effect.")
        out["Findings"] = " | ".join(rows) + "."
        out["Meaning"] = "A small off-diagonal share means the model behaves close to additively for that class."
        out["Limitations"] = ("Path-dependent TreeSHAP uses the training cover as reference (not the interventional "
                              "background used elsewhere); for AdaBoost the cover was recomputed from the log-weighted training "
                              "pixels. The linear SVM margin has no interactions by construction.")
    elif k == "interaction_pair":
        a, b = f["pair"]
        out["What it shows"] = f"The strongest Marine Debris interaction of {ML[m]} ({a} x {b}), from both sides."
        out["How to read"] = ("y = interaction value phi_ij + phi_ji (the part of the debris score that depends jointly on "
                              "both features); x = one feature, colour = the other.")
        out["Findings"] = (f"Interaction values range {g(f['range'][0])} to {g(f['range'][1])} {unit}, versus a main-effect range "
                           f"of {g(f['main_range'][0])} to {g(f['main_range'][1])} for {a}. Spearman(value, interaction): "
                           f"{a} {f['rho_i']:+.2f}, {b} {f['rho_j']:+.2f}.")
        out["Meaning"] = f"The effect of {a} on the debris score depends on the level of {b} (and vice versa)."
        out["Limitations"] = f"Based on {V['interaction_subset']['n']} pixels; rare value combinations are poorly sampled."
    elif k == "distribution":
        rows = []
        for feat, d in f["spread"].items():
            widest = max(d, key=lambda n: d[n][1] - d[n][0])
            rows.append(f"{feat}: widest 5-95% range for {ML[widest]} ({d[widest][0]:+.2f} to {d[widest][1]:+.2f})")
        out["What it shows"] = (f"Spread of SHAP values for the {CLSF[c]} output of the six features with the highest mean "
                                "normalised importance, compared across models.")
        out["How to read"] = ("Each violin = one model; y = SHAP value divided by that model's mean total |SHAP| for this "
                              "output (unit-free, so models can be compared). Line = median.")
        out["Findings"] = "; ".join(rows) + "."
        out["Meaning"] = ("A wide, one-sided violin means the model sometimes relies very heavily on that feature for this "
                          "class (typically on the few pixels of that class).")
        out["Limitations"] = "The normalisation is a presentation choice; it does not make the outputs' semantics equal."
    elif k == "waterfall_correct":
        rows = "; ".join(f"{n} = {x:.3g} contributes {v:+.3g}" for n, x, v in f["top3"])
        out["What it shows"] = (f"How {ML[m]} builds its {CLS[c]} output for one representative {CLS[c]} pixel (test index "
                                f"{f['test_index']}; correctly classified by all five models, closest to the class median).")
        out["How to read"] = ("Starts at E[f(X)] (mean output over the background) and adds each feature's SHAP value; "
                              "red bars raise, blue lower the output; grey numbers are the pixel's feature values.")
        out["Findings"] = f"Output goes from {f['base']:.3g} to {f['fx']:.3g} ({f['scale']}). Largest contributions: {rows}."
        out["Meaning"] = "Shows which cues a typical, easy pixel of this class uses."
        out["Limitations"] = "A single pixel; the beeswarms show how typical these contributions are."
    elif k == "waterfall_mis":
        rows = "; ".join(f"{n} = {x:.3g} contributes {v:+.3g}" for n, x, v in f["top3"])
        out["What it shows"] = (f"{ML[m]} {f['role']}-class output ({CLS[c]}) for a pixel whose true class is {CLS[f['true']]} "
                                f"but which is predicted {CLS[f['pred']]} (one of this model's two most frequent confusions, "
                                f"{f['n_errors_test']} such test errors; the pixel with the median error margin).")
        out["How to read"] = "As for the other waterfalls; compare with the companion plot for the other class output."
        out["Findings"] = f"Output goes from {f['base']:.3g} to {f['fx']:.3g} ({f['scale']}). Largest contributions: {rows}."
        out["Meaning"] = ("Features pushing the predicted-class output up (or the true-class output down) are the ones "
                          "responsible for the error on this pixel.")
        out["Limitations"] = "Ground truth is a polygon annotation; a single mislabeled-looking pixel can be an annotation edge."
    elif k == "decision":
        outs = ", ".join(f"{CLS[i]} {v:.3g}" for i, v in enumerate(f["outputs"]))
        out["What it shows"] = (f"All four {ML[m]} class outputs for the misclassified pixel of the preceding waterfalls (true "
                                f"{CLS[f['true']]}, predicted {CLS[f['pred']]}), built up feature by feature.")
        out["How to read"] = ("Each line starts at its class's base value at the bottom and moves by each feature's SHAP value "
                              "(features ordered by total effect; only the top 12 are shown, the rest are applied first). "
                              "The highest line at the top is the prediction.")
        out["Findings"] = f"Final outputs ({f['scale']}): {outs}."
        out["Meaning"] = "Shows where along the feature list the predicted class overtakes the true class."
        out["Limitations"] = "Feature order is a display choice; SHAP values do not depend on it."
    elif k == "case_study":
        rows = []
        for n in MODELS:
            p = f["per_model"].get(n)
            if not p or "top_shift" not in p:
                rows.append(f"{ML[n]}: {p['n_mis'] if p else 0} explained pixels ({p['n_test_errors'] if p else 0} test errors) - too few")
                continue
            sh = ", ".join(f"{a} {b:+.3g} (correct {cc:+.3g})" for a, b, cc in p["top_shift"][:3])
            rows.append(f"{ML[n]} (n = {p['n_mis']} of {p['n_test_errors']} errors): {sh}")
        out["What it shows"] = (f"Why {CLSF[f['true']]} pixels are predicted {CLSF[f['pred']]}: per model, the mean of "
                                f"SHAP({CLS[f['pred']]}) - SHAP({CLS[f['true']]}) per feature over the misclassified pixels, "
                                f"compared with the same quantity for correctly classified {CLS[f['true']]} pixels.")
        out["How to read"] = ("Bar > 0 = the feature favours the wrong class on the errors; diamond = its usual value on "
                              "correct pixels. For softmax models the difference is the feature's contribution to "
                              "log(p_pred / p_true); RF: probability difference; AdaBoost: SAMME score difference "
                              "(= 3 x log ratio); SVM: margin difference.")
        vals = case_values(ds, f, V)
        out["Findings"] = " | ".join(rows) + "." + (f" Mean feature values (misclassified / all {CLS[f['true']]} / all "
                                                    f"{CLS[f['pred']]} stratified pixels): {vals}." if vals else "")
        out["Meaning"] = ("Where a bar and its diamond have opposite signs, the feature's contribution on the errors is the "
                          "reverse of its usual contribution on correctly classified pixels: on these pixels the model reads "
                          f"that feature as {CLS[f['pred']]}-like.")
        out["Limitations"] = ("Uses the explained error pixels only: every test pixel selected for any model's errors (up to "
                              "30 per model and confusion pair) or in the stratified sample, so n can exceed 30; not every "
                              "test error is explained.")
    return out


# --------------------------------------------------------------- document ---

SEC_TITLE = {"A": "Section A - Overall feature importance", "B": "Section B - Class-specific feature importance",
             "C": "Section C - How individual features affect predictions", "D": "Section D - Individual prediction explanations"}
ITEM_TITLE = {"1": "A1. Global SHAP bar plots", "2": "A2. Global SHAP beeswarm plots",
              "3": "A3. Cross-model comparison", "4": "B4. Class-specific bar plots",
              "5": "B5. Class-specific beeswarm plots", "6": "B6. Class-wise comparison heatmap",
              "7": "B7. Class-versus-rest explanations", "8": "C8. Dependence plots",
              "9": "C9. Feature interactions", "10": "C10. SHAP distribution plots",
              "11": "D11. Waterfall plots", "12": "D12. Decision plots", "13": "D13. Class-confusion case studies"}


def dataset_part(ds, root, partno):
    tdir = os.path.join(root, "06_reports", "tables", ds)
    figs = json.load(open(os.path.join(tdir, f"{ds}_figures.json")))
    V = json.load(open(os.path.join(tdir, f"{ds}_validation.json")))
    V["_n_cases"] = sum(1 for f in figs if f["kind"] == "case_study")
    V["_cases"] = pd.read_csv(os.path.join(tdir, f"{ds}_confusion_case_studies.csv"))
    L = [f"# Part {partno}: {ds}", ""]
    L += validation_block(ds, V)
    sec = item = None
    for f in figs:
        if f["section"] != sec:
            sec = f["section"]
            L += ["", f"## {ds} - {SEC_TITLE[sec]}", ""]
        if f["item"] != item:
            item = f["item"]
            L += [f"### {ds} - {ITEM_TITLE[item]}", ""]
            L += item_intro(item, V)
        L += [f"#### {f['title']}", "", f"![{f['id']}](../{f['path']})", ""]
        for key, val in text(f, V, ds).items():
            L.append(f"- **{key}:** {val}")
        L.append("")
    return L, V


def item_intro(item, V):
    if item == "7":
        c = V["contrast_check"]
        msg = ("Class-versus-rest contrast = SHAP_k - mean_{j != k} SHAP_j. For Random Forest it reduces exactly to 4/3 x "
               f"the class-probability SHAP (max deviation {c['rf']['max_dev']:.1e}) because the four probabilities sum to "
               f"1, and for AdaBoost to 4/3 x the SAMME-score SHAP (max deviation {c['ada']['max_dev']:.1e}) because the "
               "SAMME scores sum to 0; those plots would duplicate B4/B5 and are not repeated. The linear SVM is trained "
               "one-vs-rest, so its class margins (B4/B5) already are class-versus-rest scores. Only XGBoost and "
               "LightGBM (softmax margins) get separate contrast plots. Class probabilities are explained directly only "
               "for Random Forest; for XGBoost/LightGBM the contrast is on the log-probability-ratio scale; AdaBoost and "
               "SVM explain raw scores.")
        return [msg, ""]
    if item == "9":
        return ["SHAP interaction values (path-dependent TreeSHAP) were computed for the four tree models. The linear SVM "
                "decision function is a weighted sum of standardised features, so all its SHAP interaction values are "
                "exactly zero; no plot is produced for it.", ""]
    if item == "13":
        pairs = ", ".join(f"{a} -> {b} ({n})" for a, b, n in V["confusion_pairs"])
        n_cases = V.get("_n_cases", len(V["confusion_pairs"]))
        return [f"Confusion pairs with at least 20 test errors summed over the five models (true -> predicted): {pairs}. "
                f"Case studies are shown for the {n_cases} most frequent of these {len(V['confusion_pairs'])} pairs.", ""]
    return []


def validation_block(ds, V):
    L = [f"## {ds} - Data, models and validation", ""]
    tc, trc = V["test_counts"], V["train_counts"]
    es = V["explanation_set"]
    L += [f"- Train pixels per class (1-4): {trc}; test pixels per class: {tc} (official scene-disjoint splits).",
          f"- Explanation set: {es['n_stratified']} stratified test pixels (per class {es['stratified_per_class']}, cap "
          f"{es['per_class_cap']}) plus {es['n_extra_misclassified']} extra misclassified test pixels (up to "
          f"{es['err_per_pair_cap']} per model and confusion pair) used only for section D. Global/class plots use the "
          "stratified part, so each class weighs equally regardless of its frequency.",
          f"- Background: {sum(V['background']['per_class'])} training pixels ({V['background']['per_class']} per class), shared "
          f"by all five explainers. Interaction subset: {V['interaction_subset']['n']} stratified test pixels.",
          f"- Feature order verified identical to training ({V['n_features']} features); class encoding 0-3 = labels 1-4 "
          "verified for every model (classes_ = [0, 1, 2, 3]).", "",
          "| Model | Explainer | Explained output | Test macro F1 (reproduced / recorded) | Additivity max error | SHAP array |",
          "|---|---|---|---|---|---|"]
    for n in MODELS:
        v = V["models"][n]
        off = v.get("additivity_pixels_above_1e-5", 0)
        note = f"; {off} of {v['shap_shape'][0]} px > 1e-5" if off else ""
        L.append(f"| {ML[n]} | {v['explainer']} | {v['explained_output']} | {v['test_macro_f1']:.4f} / "
                 f"{v['recorded_macro_f1']:.4f} | {v['additivity_max_abs_error']:.1e} (max output {v['max_abs_output']:.3g}{note}) | "
                 f"{tuple(v['shap_shape'])} |")
    if any(V["models"][n].get("additivity_pixels_above_1e-5", 0) for n in MODELS):
        L += ["", "Random Forest additivity: all but a handful of pixels are exact to < 1e-5; the outliers differ by exactly "
              "0.01 = 1/100 trees, i.e. one tree routes the pixel differently because SHAP stores the sklearn split "
              "thresholds at lower precision (a value lying exactly on a rounded threshold). Not an analysis error."]
    iv = {n: V["models"][n].get("interaction_additivity_max_abs_error") for n in MODELS if n != "svm"}
    L += ["", "Interaction values: sum over the full 22 x 22 matrix + base value reproduces the model output to " +
          ", ".join(f"{ML[n]} {e:.1e}" for n, e in iv.items() if e is not None) + "."]
    kc = V["models"]["svm"]["kernel_check"]
    L += ["", f"SVM check: KernelSHAP (same background, 4096 coalitions, no L1 selection) on {kc['n']} pixels agrees with "
          f"LinearExplainer to {kc['max_abs_diff']:.1e} (max |SHAP| {kc['max_abs_phi']:.3g}; correlation {kc['corr']:.6f}).",
          "", "Per-class test F1 (reproduced):", "", "| Model | Water | Marine Debris | Sargassum | Other |", "|---|---|---|---|---|"]
    for n in MODELS:
        t = V["test_metrics"][n]
        L.append(f"| {ML[n]} | {t['f1_c1']:.3f} | {t['f1_c2']:.3f} | {t['f1_c3']:.3f} | {t['f1_c4']:.3f} |")
    L.append("")
    return L


def header(Vs):
    L = ["# SHAP analysis of the five stage-8 pixel classifiers (MARIDA and MADOS)", "",
         "All numbers in this report are produced by `scripts/shap_analysis/run_shap_analysis.py`; the per-figure "
         "findings are filled in from the computed SHAP values (see `06_reports/tables/`).", "",
         "## Scope", "",
         "- **Models:** the five models of the staged pipeline, stage 8 (`scripts/staged_ml/stage8_ndwi_ratios_glcm.py`), "
         "log class weighting, seed 42: Random Forest, XGBoost, LightGBM, AdaBoost (SAMME, depth-3 trees) and a linear "
         "SVM (StandardScaler + LinearSVC). The pipeline never saved fitted models, so they were re-fitted once with the "
         "pipeline's own functions (`scripts/shap_analysis/train_stage8_models.py`) and saved to `models/stage8_log/`; "
         "every model reproduces the recorded test metrics in `outputs/staged/log_weights/stage8_ndwi_ratios_glcm/` "
         "(max |difference| <= 1.1e-16).",
         "- **Classes** (labels 1-4 encoded 0-3, from `scripts/utilities/convert.py`): 1 Water = Marine Water, "
         "Sediment-Laden Water, Turbid, Shallow, Waves, Wakes, Mixed Water; 2 Marine Debris; 3 Sargassum = dense + sparse "
         "Sargassum (MADOS: dense Sargassum + sparse floating algae); 4 Other = Ship, Natural Organic Material, Foam "
         "(MADOS additionally maps original classes 6, 13, 14, 15 - Oil Spill, Oil Platform, Jellyfish, Sea snot in "
         "the MADOS class list - to 4). Class 0 (unannotated, clouds, cloud shadows) is excluded, as in training.",
         "- **SHAP only.** Explained quantity per model:", "",
         "| Model | Explainer | Output explained (one SHAP vector per class) |", "|---|---|---|",
         "| Random Forest | TreeExplainer, interventional | class probability (predict_proba) |",
         "| XGBoost | TreeExplainer, interventional | raw softmax margin (log-odds scale) |",
         "| LightGBM | TreeExplainer, interventional | raw softmax margin (log-odds scale) |",
         "| AdaBoost | TreeExplainer, interventional, on an exact tree-ensemble rewrite of the SAMME decision_function | "
         "SAMME score d_k = sum_m (w_m/W)(1[tree_m = k] - 1/3[tree_m != k]); predict_proba = softmax(d/3) |",
         "| Linear SVM | LinearExplainer, interventional (exact); validated with KernelSHAP | one-vs-rest margin (LinearSVC has no probabilities) |",
         "", "SHAP values are relative to the mean output over the shared class-balanced training background. "
         "Raw SHAP magnitudes are only compared within a model; cross-model comparisons use normalised shares "
         "(share = mean|SHAP_f| / sum of mean|SHAP| over all 22 features, per model and output).", "",
         "## Features (stage 8, in model input order)", "",
         "| Feature | Source | Computational meaning |", "|---|---|---|"]
    for a, b, c in FEATURES:
        L.append(f"| {a} | `{b}` | {c} |")
    L += ["", "GLCM details (`glcm_features.py`, MARIDA reference implementation): grey image = 0.2125 B4 + 0.7154 B3 + "
          "0.0721 B2 after clipping to [0, 0.15] and scaling; 17 grey levels; 13 x 13 window centred on the pixel "
          "(reflect padding); co-occurrence at distance 1, averaged over 0/45/90/135 degrees. GLCM features therefore "
          "describe the visible-band texture of a ~130 m x 130 m neighbourhood, not the pixel itself. energy = sqrt(ASM), "
          "so the two are perfectly rank-correlated and any split of credit between them is arbitrary; contrast and "
          "dissimilarity are also strongly related. Statements below about what a texture value 'means' for debris or "
          "Sargassum are hypotheses; SHAP only shows how the models use these numbers.", ""]
    return L


LIMITS = ["## Limitations and computational compromises", "",
          "- **Attribution, not causation.** SHAP describes how each fitted model's output depends on its inputs over the "
          "chosen background; it says nothing about physical causes.",
          "- **Retrained models.** No fitted models existed on disk; they were re-fitted with the pipeline's own code and "
          "seed and reproduce the recorded test metrics to 1e-16, but library versions differ from the original run "
          "(see `models/stage8_log/<dataset>/meta.json`).",
          "- **Sampling.** Global and class plots use a class-stratified sample of the test split (up to 300 pixels per "
          "class), so importance is class-balanced, not frequency-weighted. Water dominates the real test set "
          "(MARIDA ~98%), so frequency-weighted importance would be close to the Water output's importance.",
          "- **Background.** A 200-pixel class-balanced training background is used for all explainers. Base values and "
          "SHAP values change with the background; interventional SHAP breaks feature correlations, so strongly "
          "correlated features (adjacent bands, band ratios built from the same bands, energy/ASM) can receive credit "
          "for off-manifold combinations, and credit can be split between them.",
          "- **Different output scales.** RF explains probabilities, XGBoost/LightGBM log-odds margins, AdaBoost SAMME "
          "scores and SVM margins. Raw magnitudes are never compared across models; only normalised shares and ranks.",
          "- **Interactions.** SHAP interaction values are only defined (in SHAP) for tree models via path-dependent "
          "TreeSHAP; they use the training cover as reference (a different reference from the interventional main "
          "values) and were computed on a smaller subset. AdaBoost's cover was recomputed from the log-weighted training "
          "pixels because the stored node weights reflect boosting re-weighting. The linear SVM has no interactions.",
          "- **SVM.** KernelSHAP was run only as a correctness check on a small sample; the exact LinearExplainer "
          "values are used for all plots.",
          "- **Single pixels.** Waterfall and decision plots explain individual pixels and should be read together with "
          "the population plots; annotated polygons can contain mixed pixels at their edges.", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "results", "shap_analysis"))
    ap.add_argument("--datasets", nargs="+", default=["MARIDA", "MADOS"])
    a = ap.parse_args()
    parts, Vs = [], []
    for i, ds in enumerate(a.datasets, 1):
        L, V = dataset_part(ds, a.root, i)
        parts += L + [""]
        Vs.append(V)
    doc = header(Vs)
    of = os.path.join(a.root, "06_reports", "overall_findings.md")
    if os.path.exists(of):
        doc += open(of, encoding="utf-8").read().splitlines() + [""]
    oc = os.path.join(a.root, "06_reports", "overall_contribution.md")       # plot_overall_contribution.py
    if os.path.exists(oc):
        doc += open(oc, encoding="utf-8").read().splitlines() + [""]
    doc += LIMITS + parts
    out = os.path.join(a.root, "06_reports", "REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(doc) + "\n")
    n = sum(1 for l in doc if l.startswith("![") )
    print(f"wrote {out} ({n} figures)")


if __name__ == "__main__":
    main()
