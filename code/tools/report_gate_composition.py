#!/usr/bin/env python3
"""
report_gate_composition.py — Disclosure diagnostic for the `both`-feature lock.

The TIER 7 post-hoc gate only tests features that rank in the top-N (default 5) of
the TIER 6 importance ranking for each cancer type (`shap_utils.identify_top_features`).
When the model is trained on `feature_set=both`, technical features (SUBJECT_CENTER,
number_of_genes, ...) compete for those top-N slots and can push demographic features
below the gate — so a demographic gene-stratified test is silently skipped for that
cancer type.

This script quantifies that effect for a locked run so it can be disclosed in the
paper (a supplement table + a one-line limitation). It reuses the EXACT gate logic
(`shap_utils.get_cancer_types_with_top_feature` / `load_shap_rankings_adaptive`), so
the numbers here match what TIER 7 actually did.

It reports, per cancer type:
  - the top-N features and the category each belongs to,
  - how many top-N slots are technical vs demographic,
  - which demographic categories CLEAR the gate vs are GATED OUT.

Usage:
    # auto-locate the importance file inside a run dir
    python code/tools/report_gate_composition.py --run-dir runs/run_X_rf --pathway rf
    python code/tools/report_gate_composition.py --run-dir runs/run_X_lr --pathway lr --lr-significant-only

    # or point at the CSV directly
    python code/tools/report_gate_composition.py --shap-file <combined_feature_importance.csv>

Writes a per-cancer CSV (--output, default <run-dir>/gate_composition.csv) and prints
a summary. Read-only w.r.t. run inputs.
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shap_utils import (  # noqa: E402
    FEATURE_PATTERNS,
    DEFAULT_TOP_N_FEATURES,
    load_shap_rankings_adaptive,
    get_cancer_types_with_top_feature,
)

# Demographic categories the paper reports on (the gate is about whether THESE
# clear top-N). TECHNICAL is the competitor that consumes slots.
DEMOGRAPHIC_CATEGORIES = ["SEX", "AGE", "ANCESTRY", "ETHNICITY", "SAMPLE_TYPE_DETAILED"]
MISSING_RE = r"_missing|Missing \(Indicator\)"


def classify_feature(name):
    """Best-effort single-category label for a feature, matching gate semantics.

    Missingness indicators are checked first (they would otherwise match the bare
    'ANCESTRY'/'ETHNICITY' substrings). Uses the same regex=False substring match as
    shap_utils.identify_top_features.
    """
    if pd.Series([name]).str.contains(MISSING_RE, case=False, regex=True).iloc[0]:
        return "MISSING_INDICATOR"
    for cat in DEMOGRAPHIC_CATEGORIES + ["TECHNICAL"]:
        for pat in FEATURE_PATTERNS[cat]:
            if pat.lower() in str(name).lower():
                return cat
    return "OTHER"


def locate_importance_file(run_dir, pathway):
    if pathway == "rf":
        # The orchestrator (TIER 6a) writes with --no-timestamp-subdir, so the file
        # lands directly under primary_values/ (Gotcha #7). Manual
        # compute_shap_primary_values.py runs instead use a run_*/ subdir. Support both,
        # preferring the orchestrator's canonical no-subdir output.
        direct = os.path.join(run_dir, "tier6_shap_analysis/primary_values/combined_feature_importance.csv")
        if os.path.isfile(direct):
            return direct
        subdir = sorted(glob.glob(os.path.join(
            run_dir, "tier6_shap_analysis/primary_values/run_*/combined_feature_importance.csv")))
        if subdir:
            return subdir[-1]
        sys.exit(f"ERROR: no RF SHAP importance file under {run_dir}; looked for:\n"
                 f"  tier6_shap_analysis/primary_values/combined_feature_importance.csv\n"
                 f"  tier6_shap_analysis/primary_values/run_*/combined_feature_importance.csv")
    else:
        pat = os.path.join(run_dir, "tier6_shap_analysis/lr_analysis/combined_lr_grouped_importance.csv")
        matches = sorted(glob.glob(pat))
        if not matches:
            sys.exit(f"ERROR: no importance file under {run_dir} matching:\n  {pat}")
        return matches[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shap-file", help="Path to the TIER 6 importance CSV directly.")
    src.add_argument("--run-dir", help="Run directory; auto-locates the importance CSV.")
    ap.add_argument("--pathway", choices=["rf", "lr"],
                    help="Required with --run-dir to pick the right importance file.")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N_FEATURES,
                    help=f"Gate size (default {DEFAULT_TOP_N_FEATURES}).")
    ap.add_argument("--lr-significant-only", action="store_true",
                    help="LR only: restrict to Wald-significant features before ranking (mirrors the locked LR gate).")
    ap.add_argument("--output", default=None, help="Per-cancer CSV output path.")
    args = ap.parse_args()

    if args.run_dir:
        if not args.pathway:
            ap.error("--pathway is required with --run-dir")
        shap_file = locate_importance_file(args.run_dir, args.pathway)
    else:
        shap_file = args.shap_file

    df = load_shap_rankings_adaptive(shap_file, lr_significant_only=args.lr_significant_only)
    top_n = args.top_n

    cancers = sorted(df["ONCOTREE_CODE"].unique())

    # Per-category gate membership using the SAME helper TIER 7 uses.
    clears = {cat: set(get_cancer_types_with_top_feature(df, cat, top_n=top_n))
              for cat in DEMOGRAPHIC_CATEGORIES + ["TECHNICAL"]}

    rows = []
    for ct in cancers:
        top = (df[(df["ONCOTREE_CODE"] == ct) & (df["feature_rank"] <= top_n)]
               .sort_values("feature_rank"))
        cats = [classify_feature(f) for f in top["feature"]]
        n_tech = sum(c == "TECHNICAL" for c in cats)
        n_demo = sum(c in DEMOGRAPHIC_CATEGORIES for c in cats)
        in_gate = [c for c in DEMOGRAPHIC_CATEGORIES if ct in clears[c]]
        gated_out = [c for c in DEMOGRAPHIC_CATEGORIES if ct not in clears[c]]
        rows.append({
            "ONCOTREE_CODE": ct,
            "n_technical_slots": n_tech,
            "n_demographic_slots": n_demo,
            "demographic_categories_in_gate": "|".join(in_gate),
            "demographic_categories_gated_out": "|".join(gated_out),
            "top_n_features": " ; ".join(f"{r}:{f} [{c}]"
                                         for r, f, c in zip(top["feature_rank"], top["feature"], cats)),
        })
    out = pd.DataFrame(rows)

    output = args.output or (os.path.join(args.run_dir, "gate_composition.csv")
                             if args.run_dir else "gate_composition.csv")
    out.to_csv(output, index=False)

    # ---- summary ----
    n = len(cancers)
    print(f"\nGate composition (top-{top_n}) — {os.path.basename(shap_file)}")
    print(f"  {n} cancer types\n")
    print("  Demographic category gate clearance (tests that RUN vs are SKIPPED):")
    for cat in DEMOGRAPHIC_CATEGORIES:
        c = len(clears[cat])
        print(f"    {cat:22} clears gate in {c:2}/{n}  (skipped in {n - c})")
    tech_ge2 = int((out["n_technical_slots"] >= 2).sum())
    tech_ge3 = int((out["n_technical_slots"] >= 3).sum())
    print(f"\n  Technical-feature crowding:")
    print(f"    cancers with >=2 technical slots in top-{top_n}: {tech_ge2}/{n}")
    print(f"    cancers with >=3 technical slots in top-{top_n}: {tech_ge3}/{n}")
    print(f"\n  Per-cancer CSV -> {output}\n")


if __name__ == "__main__":
    main()
