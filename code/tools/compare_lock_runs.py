#!/usr/bin/env python3
"""
compare_lock_runs.py — Concordance checker for the canonical publication lock.

Diffs two run directories that are SUPPOSED to be identical (e.g. a local `idunn`
run vs. the same run reproduced on HPC) for a single pathway (rf or lr). Emits a
per-check PASS/FAIL/SKIP report and exits non-zero if any non-skipped check fails.

The acceptance criteria come from ROADMAP.md ("Reproducibility / concordance
protocol"):

  - Train/test split:        IDENTICAL patient-id -> split mapping (hard equality).
  - TIER 3 hyperparameters:  IDENTICAL best_params_consensus (deterministic given
                             seed + folds + sample-fraction=1.0).
  - TIER 5 per-cancer AUC:   |test_auc_a - test_auc_b| < AUC_TOL  (default 0.005).
  - TIER 6 RF SHAP ranks:    Spearman of top-N feature ranks > SHAP_SPEARMAN_MIN
                             (default 0.95), per cancer type (rf pathway only).
  - TIER 6 LR coefficients:  |log_odds_a - log_odds_b| <= LR_COEF_TOL (lr pathway
                             only; LR is convex/deterministic).
  - TIER 7 hit set:          IDENTICAL set of FDR<0.05 gene-stratified hits, or the
                             symmetric difference is listed.

Usage:
    python code/tools/compare_lock_runs.py \
        --run-a runs/run_LOCAL_rf --run-b runs/run_HPC_rf --pathway rf

    python code/tools/compare_lock_runs.py \
        --run-a runs/run_LOCAL_lr --run-b runs/run_HPC_lr --pathway lr [--json report.json]

A run compared against itself returns all-PASS, which is the smoke test for this
script. Checks whose inputs are missing in either run (e.g. RF SHAP under --quick)
are reported as SKIP and do not fail the overall result.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# ---- Acceptance tolerances (ROADMAP.md) -----------------------------------
AUC_TOL = 0.005           # TIER 5 per-cancer test-AUC absolute tolerance
SHAP_SPEARMAN_MIN = 0.95  # TIER 6 RF SHAP top-N rank Spearman floor
SHAP_TOP_N = 20           # number of top features per cancer to correlate
LR_COEF_TOL = 1e-4        # TIER 6 LR log-odds absolute tolerance

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class CheckResult:
    def __init__(self, name, status, detail=""):
        self.name = name
        self.status = status
        self.detail = detail


# ---- file location helpers ------------------------------------------------

def _find_one(run_dir, pattern, label):
    """Return the single file matching `pattern` (glob) under run_dir.

    `pattern` may be a single glob or a list of globs tried in order (first that
    matches wins) — used where a file may live either directly in a tier dir or in
    a timestamped run_* subdir (e.g. RF SHAP: the orchestrator writes with
    --no-timestamp-subdir, manual runs don't — see Gotcha #7).

    Returns (path, None) on success or (None, reason) if zero/ambiguous matches.
    Newest match wins when several exist (timestamped run_* subdirs), with a note.
    """
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    matches = []
    for pat in patterns:
        matches = sorted(glob.glob(os.path.join(run_dir, pat)))
        if matches:
            break
    if not matches:
        return None, f"{label}: no file matching {patterns}"
    if len(matches) > 1:
        # Multiple timestamped subdirs — pick the lexicographically newest and note it.
        return matches[-1], f"{label}: {len(matches)} matches, used newest ({os.path.relpath(matches[-1], run_dir)})"
    return matches[0], None


# ---- individual checks ----------------------------------------------------

def check_split(run_a, run_b):
    rel = "tier2_data_setup/train_test_split/train_test_split.csv"
    pa, pb = os.path.join(run_a, rel), os.path.join(run_b, rel)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        missing = [p for p in (pa, pb) if not os.path.exists(p)]
        return CheckResult("train/test split", SKIP, f"missing: {missing}")
    a = pd.read_csv(pa)[["PATIENT_ID", "split"]].sort_values("PATIENT_ID").reset_index(drop=True)
    b = pd.read_csv(pb)[["PATIENT_ID", "split"]].sort_values("PATIENT_ID").reset_index(drop=True)
    if a.equals(b):
        return CheckResult("train/test split", PASS,
                           f"{len(a)} patients, identical id->split mapping")
    # Diagnose the divergence.
    set_a, set_b = set(a["PATIENT_ID"]), set(b["PATIENT_ID"])
    only_a, only_b = set_a - set_b, set_b - set_a
    merged = a.merge(b, on="PATIENT_ID", suffixes=("_a", "_b"))
    reassigned = merged[merged["split_a"] != merged["split_b"]]
    detail = (f"n_a={len(a)} n_b={len(b)} only_in_a={len(only_a)} "
              f"only_in_b={len(only_b)} split_reassigned={len(reassigned)}")
    return CheckResult("train/test split", FAIL, detail)


def check_hyperparams(run_a, run_b, pathway):
    fname = "random_forest_universal_params.json" if pathway == "rf" else "logistic_universal_params.json"
    pat = f"tier3_hyperparameter_optimization/run_*/{fname}"
    pa, err_a = _find_one(run_a, pat, "hyperparams[a]")
    pb, err_b = _find_one(run_b, pat, "hyperparams[b]")
    if pa is None or pb is None:
        return CheckResult("TIER 3 hyperparameters", SKIP, "; ".join(e for e in (err_a, err_b) if e))
    ca = json.load(open(pa)).get("best_params_consensus", {})
    cb = json.load(open(pb)).get("best_params_consensus", {})
    if ca == cb:
        return CheckResult("TIER 3 hyperparameters", PASS, f"consensus identical: {ca}")
    diffs = {k: (ca.get(k), cb.get(k)) for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    return CheckResult("TIER 3 hyperparameters", FAIL, f"consensus differs: {diffs}")


def check_tier5_auc(run_a, run_b):
    pat = "tier5_final_models/run_*/performance_summary.csv"
    pa, err_a = _find_one(run_a, pat, "perf[a]")
    pb, err_b = _find_one(run_b, pat, "perf[b]")
    if pa is None or pb is None:
        return CheckResult("TIER 5 test AUC", SKIP, "; ".join(e for e in (err_a, err_b) if e))
    a = pd.read_csv(pa)[["cancer_type", "test_auc"]]
    b = pd.read_csv(pb)[["cancer_type", "test_auc"]]
    m = a.merge(b, on="cancer_type", suffixes=("_a", "_b"))
    if m.empty:
        return CheckResult("TIER 5 test AUC", FAIL, "no overlapping cancer types")
    m["abs_diff"] = (m["test_auc_a"] - m["test_auc_b"]).abs()
    worst = m.loc[m["abs_diff"].idxmax()]
    n_over = int((m["abs_diff"] > AUC_TOL).sum())
    detail = (f"{len(m)} cancer types; max |ΔAUC|={worst['abs_diff']:.5f} "
              f"({worst['cancer_type']}); {n_over} exceed tol={AUC_TOL}")
    status = PASS if worst["abs_diff"] <= AUC_TOL else FAIL
    return CheckResult("TIER 5 test AUC", status, detail)


def check_shap_ranks(run_a, run_b):
    pat = ["tier6_shap_analysis/primary_values/combined_feature_importance.csv",
           "tier6_shap_analysis/primary_values/run_*/combined_feature_importance.csv"]
    pa, err_a = _find_one(run_a, pat, "shap[a]")
    pb, err_b = _find_one(run_b, pat, "shap[b]")
    if pa is None or pb is None:
        return CheckResult("TIER 6 RF SHAP ranks", SKIP,
                           "; ".join(e for e in (err_a, err_b) if e) or "RF SHAP absent (quick mode?)")
    # Key on feature_name_raw (e.g. cat__x3_Unknown) — the display feature_name is
    # NOT unique within a cancer type (e.g. "Sex: Unknown" appears for two one-hot
    # columns), which would make a feature_name index ambiguous.
    a = pd.read_csv(pa)[["cancer_type", "feature_name_raw", "rank"]]
    b = pd.read_csv(pb)[["cancer_type", "feature_name_raw", "rank"]]
    per_cancer = []
    for ct in sorted(set(a["cancer_type"]) & set(b["cancer_type"])):
        sa = a[a["cancer_type"] == ct].groupby("feature_name_raw")["rank"].min()
        sb = b[b["cancer_type"] == ct].groupby("feature_name_raw")["rank"].min()
        # Features in the top-N of either run, present in both.
        topfeats = set(sa[sa <= SHAP_TOP_N].index) | set(sb[sb <= SHAP_TOP_N].index)
        common = [f for f in topfeats if f in sa.index and f in sb.index]
        if len(common) < 5:
            continue
        rho = pd.Series([sa[f] for f in common]).corr(
            pd.Series([sb[f] for f in common]), method="spearman")
        per_cancer.append((ct, rho, len(common)))
    if not per_cancer:
        return CheckResult("TIER 6 RF SHAP ranks", SKIP, "no cancer type with >=5 common top features")
    rhos = [r for _, r, _ in per_cancer if pd.notna(r)]
    min_rho = min(rhos) if rhos else float("nan")
    worst = min(per_cancer, key=lambda t: (t[1] if pd.notna(t[1]) else 2))
    detail = (f"{len(per_cancer)} cancer types; min Spearman={min_rho:.4f} "
              f"({worst[0]}, n={worst[2]}); threshold={SHAP_SPEARMAN_MIN}")
    status = PASS if (rhos and min_rho >= SHAP_SPEARMAN_MIN) else FAIL
    return CheckResult("TIER 6 RF SHAP ranks", status, detail)


def check_lr_coefficients(run_a, run_b):
    pat = "tier6_shap_analysis/lr_analysis/combined_lr_coefficients.csv"
    pa, pb = os.path.join(run_a, pat), os.path.join(run_b, pat)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return CheckResult("TIER 6 LR coefficients", SKIP,
                           f"missing: {[p for p in (pa, pb) if not os.path.exists(p)]}")
    # Key on feature_name_raw (unique per cancer type); the display feature_name can
    # repeat within a cancer type and would produce a many-to-many merge.
    cols = ["cancer_type", "feature_name_raw", "log_odds"]
    a = pd.read_csv(pa)[cols]
    b = pd.read_csv(pb)[cols]
    m = a.merge(b, on=["cancer_type", "feature_name_raw"], suffixes=("_a", "_b"))
    if m.empty:
        return CheckResult("TIER 6 LR coefficients", FAIL, "no overlapping (cancer, feature) rows")
    m["abs_diff"] = (m["log_odds_a"] - m["log_odds_b"]).abs()
    n_missing = (len(a) - len(m)) + (len(b) - len(m))
    max_diff = float(m["abs_diff"].max())
    detail = (f"{len(m)} coefficients; max |Δlog_odds|={max_diff:.2e} "
              f"(tol={LR_COEF_TOL}); {n_missing} non-overlapping rows")
    status = PASS if (max_diff <= LR_COEF_TOL and n_missing == 0) else FAIL
    return CheckResult("TIER 6 LR coefficients", status, detail)


def check_tier7_hits(run_a, run_b):
    rel = "tier7_posthoc_statistics/gene_stratified/gene_stratified_significant_hits.csv"
    pa, pb = os.path.join(run_a, rel), os.path.join(run_b, rel)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return CheckResult("TIER 7 hit set", SKIP,
                           f"missing: {[p for p in (pa, pb) if not os.path.exists(p)]}")
    key = ["ONCOTREE_CODE", "Hugo_Symbol", "feature_category", "feature"]

    def hitset(path):
        df = pd.read_csv(path)
        if df.empty:
            return set()
        return set(map(tuple, df[key].astype(str).itertuples(index=False, name=None)))

    sa, sb = hitset(pa), hitset(pb)
    if sa == sb:
        return CheckResult("TIER 7 hit set", PASS, f"{len(sa)} hits, identical")
    only_a, only_b = sorted(sa - sb), sorted(sb - sa)
    detail = (f"n_a={len(sa)} n_b={len(sb)} only_in_a={len(only_a)} only_in_b={len(only_b)}"
              + (f"\n    only_in_a: {only_a[:10]}" if only_a else "")
              + (f"\n    only_in_b: {only_b[:10]}" if only_b else ""))
    return CheckResult("TIER 7 hit set", FAIL, detail)


# ---- driver ---------------------------------------------------------------

def run_checks(run_a, run_b, pathway):
    checks = [check_split(run_a, run_b),
              check_hyperparams(run_a, run_b, pathway),
              check_tier5_auc(run_a, run_b)]
    if pathway == "rf":
        checks.append(check_shap_ranks(run_a, run_b))
    else:
        checks.append(check_lr_coefficients(run_a, run_b))
    checks.append(check_tier7_hits(run_a, run_b))
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-a", required=True, help="First run directory (e.g. local).")
    ap.add_argument("--run-b", required=True, help="Second run directory (e.g. HPC).")
    ap.add_argument("--pathway", required=True, choices=["rf", "lr"],
                    help="Which pathway these two runs represent.")
    ap.add_argument("--json", default=None, help="Optional path to write a JSON report.")
    args = ap.parse_args()

    for d in (args.run_a, args.run_b):
        if not os.path.isdir(d):
            ap.error(f"not a directory: {d}")

    results = run_checks(args.run_a, args.run_b, args.pathway)

    width = max(len(r.name) for r in results)
    print(f"\nConcordance: {args.pathway.upper()} pathway")
    print(f"  A: {args.run_a}")
    print(f"  B: {args.run_b}\n")
    icon = {PASS: "✓", FAIL: "✗", SKIP: "·"}
    for r in results:
        print(f"  [{icon[r.status]} {r.status}] {r.name.ljust(width)}  {r.detail}")

    n_fail = sum(1 for r in results if r.status == FAIL)
    n_skip = sum(1 for r in results if r.status == SKIP)
    n_pass = sum(1 for r in results if r.status == PASS)
    verdict = "FAIL" if n_fail else "PASS"
    print(f"\n  Overall: {verdict}  ({n_pass} pass, {n_fail} fail, {n_skip} skip)\n")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"pathway": args.pathway, "run_a": args.run_a, "run_b": args.run_b,
                       "verdict": verdict,
                       "checks": [{"name": r.name, "status": r.status, "detail": r.detail}
                                  for r in results]}, fh, indent=2)
        print(f"  JSON report -> {args.json}\n")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
