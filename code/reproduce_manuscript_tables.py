#!/usr/bin/env python3
"""Reproduce every manuscript results table from the harmonized GENIE database.

Runs the v1 publication-lock recipe end to end and collects the tables into one
directory, with checksums compared against the published values:

  1. setup      TIER 2a cancer-type eligibility + TIER 2b patient-level split (seed 42)
  2. rf         RF pathway, TIER 4-7, pinned Step-2 hyperparameters, ODG excluded
  3. lr         LR pathway, TIER 4-7, module-default hyperparameters, Wald-FDR gate
  4. tier1      Table 1, cohort-flow counts, institutional tables (20-code cohort)
  5. legacy     TIER 7 re-run keeping GENIE's "Unknown" as a category (Table S7)
  6. derived    cross-pathway comparison and robustness analyses
  7. collect    copy tables to <out>/tables, write MANIFEST.csv, verify checksums

Run from the repo root inside balder_conda_env (balder_ml_env must also exist; the
orchestrator uses it for RF SHAP):

    export BALDER_DB_PATH=/path/to/balder-harmonized-biomarker-data-v20250522-ageUpper-patientunit.sqlite
    python code/reproduce_manuscript_tables.py

Hyperparameter search (TIER 3, ~8 h) and the Step-2 depth validation are not re-run;
see README for those commands.
"""

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CODE = REPO / "code"
RUNS = REPO / "runs"
CONFIG = CODE / "config"

EXCLUDE = ["ODG"]               # dropped after eligibility (23 non-actionable patients)
WEIGHT_ALPHA = "0.5"            # institution sample-weight exponent
RF_PARAMS_DIR = CONFIG / "hyperparameters"
LIT_REVIEW = [REPO / "curated" / "lit_review" / "lit_review_age_sex.csv",
              REPO / "curated" / "lit_review" / "lit_review_ancestry_ethnicity.csv"]
CHECKSUMS = CONFIG / "manuscript_table_checksums.csv"
EXPECTED_DB_SHA256 = "9c3d5897564844be7a646534ad658b7e7562a76c41b938dd8790d3db382e850e"

GROUPINGS = Path("tier2_data_setup/cancer_type_groupings/cancer_type_group_definitions.csv")
MAPPING = Path("tier2_data_setup/cancer_type_groupings/patient_to_group_mapping.csv")
SPLIT_DIR = Path("tier2_data_setup/train_test_split")
TIER1 = Path("tier1_independent_analyses")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(script, *args, env=None):
    """Run a repo script with the orchestrator's cwd and PYTHONPATH."""
    cmd = [sys.executable, str(CODE / script), *map(str, args)]
    log("$ " + " ".join(cmd[1:]))
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def script_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(CODE / "modeling"), str(CODE / "tools"), env.get("PYTHONPATH", "")])
    return env


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def find_run(label):
    hits = sorted(RUNS.glob(f"run_*_{label}"))
    if len(hits) != 1:
        raise RuntimeError(f"expected one run dir for label {label!r}, found {hits}")
    return hits[0]


def orchestrate(*args):
    run("run_all_analyses.py", "--tables-only", "--institution-weight-alpha", WEIGHT_ALPHA,
        *args)


def copy_run_without_tier7(src, dst):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("tier7_posthoc_statistics", "*.log"))
    (dst / "tier7_posthoc_statistics").mkdir()


# (output name, path relative to the out dir). Names match the staged supplement.
TABLES = [
    ("Table_1_patient_characteristics.csv", "rf/" + str(TIER1 / "cancer_type_summary_stats/table1_patient_characteristics.csv")),
    ("Table_S1a_threshold_sensitivity.csv", "derived/threshold/threshold_sensitivity.csv"),
    ("Table_S1b_fdr_invariance_raise.csv", "derived/threshold/fdr_invariance_raise.csv"),
    ("Table_S2_feature_definitions.csv", "derived/supplement/supp_table_feature_definitions.csv"),
    ("Table_S3_ml_metrics.csv", "derived/manuscript/supp_table_ml_metrics.csv"),
    ("Table_S4a_full_hit_table_merged.csv", "derived/full_hit_table/full_hit_table_merged.csv"),
    ("Table_S4b_full_hit_table_rf.csv", "derived/full_hit_table/full_hit_table_rf.csv"),
    ("Table_S4c_full_hit_table_lr.csv", "derived/full_hit_table/full_hit_table_lr.csv"),
    ("Table_S5_crude_vs_adjusted_all_features.csv", "derived/center_adjustment/crude_vs_adjusted_all_features.csv"),
    ("Table_S6_sex_enrichment.csv", "derived/manuscript/supp_table_sex_enrichment.csv"),
    ("Table_S7_unknown_sensitivity.csv", "derived/supplement/table_S10_unknown_sensitivity.csv"),
    ("Table_S10a_ancestry.csv", "rf/" + str(TIER1 / "cancer_type_summary_stats/supplement_ancestry.csv")),
    ("Table_S10b_ethnicity.csv", "rf/" + str(TIER1 / "cancer_type_summary_stats/supplement_ethnicity.csv")),
    ("Table_S10c_sample_type.csv", "rf/" + str(TIER1 / "cancer_type_summary_stats/supplement_sample_type.csv")),
    ("Table_S11_hyperparameters.csv", "derived/supplement/table_S4_hyperparameters.csv"),
    ("Table_S12a_global_fdr_summary.csv", "derived/fdr_repool/global_fdr_summary.csv"),
    ("Table_S12b_global_fdr_changed_calls.csv", "derived/fdr_repool/global_fdr_changed_calls.csv"),
    ("core/cohort_flow_counts.csv", "rf/" + str(TIER1 / "strobe_diagram/strobe_filtering_counts.csv")),
    ("core/institutional_cv_by_cancer.csv", "rf/" + str(TIER1 / "cv_forest_plots/institutional_cv_by_cancer.csv")),
    ("core/institutional_stats_by_cancer.csv", "rf/" + str(TIER1 / "institutional/all_cancer_types_institutional_stats.csv")),
    ("core/cancer_type_group_definitions.csv", "rf/" + str(GROUPINGS)),
    ("core/rf_vs_lr_gene_stratified_comparison.csv", "derived/comparison/gene_stratified_comparison.csv"),
]


def collect(out, verify=True):
    """Copy tables into <out>/tables, write MANIFEST.csv, compare to published checksums."""
    tables = out / "tables"
    expected = {}
    if verify and CHECKSUMS.exists():
        with open(CHECKSUMS) as f:
            expected = {r["table"]: r["sha256"] for r in csv.DictReader(f)}
    rows, n_match = [], 0
    for name, rel in TABLES:
        src = out / rel
        dst = tables / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            rows.append((name, rel, "", "MISSING"))
            continue
        shutil.copy2(src, dst)
        digest = sha256(dst)
        status = ("match" if expected.get(name) == digest
                  else "differs" if name in expected else "no reference")
        n_match += status == "match"
        rows.append((name, rel, digest, status))
    with open(tables / "MANIFEST.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "source", "sha256", "vs_published"])
        w.writerows(rows)
    for name, _, _, status in rows:
        log(f"  {status:12s} {name}")
    log(f"{n_match}/{len(rows)} tables match the published checksums -> {tables}")
    return all(r[3] != "MISSING" for r in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: runs/manuscript_tables_<timestamp>")
    p.add_argument("--allow-db-mismatch", action="store_true",
                   help="Continue when the database checksum differs from the published one")
    p.add_argument("--collect-only", action="store_true",
                   help="Skip all analysis; re-collect tables from an existing --output-dir")
    a = p.parse_args()

    tag = "repro" + datetime.now().strftime("%Y%m%d%H%M%S")
    out = (a.output_dir or RUNS / f"manuscript_tables_{tag}").resolve()
    if a.collect_only:
        sys.exit(0 if collect(out) else 1)

    db = os.environ.get("BALDER_DB_PATH")
    if not db or not Path(db).exists():
        sys.exit("Set BALDER_DB_PATH to the harmonized GENIE SQLite database.")
    log(f"Checking database checksum: {db}")
    if sha256(db) != EXPECTED_DB_SHA256:
        msg = "Database checksum differs from the one the manuscript was built on."
        if not a.allow_db_mismatch:
            sys.exit(msg + " Pass --allow-db-mismatch to continue anyway.")
        log("WARNING: " + msg)

    out.mkdir(parents=True, exist_ok=True)
    env = script_env()
    derived = out / "derived"

    log("1/7 setup: eligibility + train/test split")
    setup = RUNS / f"run_{tag}_setup"
    run("analysis/define_cancer_type_groupings.py",
        "--output-dir", setup / GROUPINGS.parent, env=env)
    run("analysis/define_train_test_split.py", "--output-dir", setup / SPLIT_DIR,
        "--split-strategy", "patient", env=env)

    log("2/7 RF pathway")
    orchestrate("--post-model-selection", "--algorithm", "random_forest",
                "--run-label", f"{tag}_rf", "--base-run-dir", setup,
                "--hyperparams-dir", RF_PARAMS_DIR, "--hyperparams-source", "search",
                "--exclude-cancer-types", *EXCLUDE)
    rf = find_run(f"{tag}_rf")
    shutil.copytree(setup / SPLIT_DIR, rf / SPLIT_DIR, dirs_exist_ok=True)

    log("3/7 LR pathway")
    orchestrate("--post-model-selection", "--algorithm", "logistic", "--lr-significant-only",
                "--run-label", f"{tag}_lr", "--base-run-dir", rf,
                "--hyperparams-source", "defaults", "--exclude-cancer-types", *EXCLUDE)
    lr = find_run(f"{tag}_lr")

    log("4/7 TIER 1 tables on the analysed cohort")
    groupings = rf / GROUPINGS
    run("analysis/cancer_type_summary_stats.py", "--format-table1", "--base-run-dir", rf,
        "--output-dir", rf / TIER1 / "cancer_type_summary_stats", env=env)
    run("analysis/figures/generate_strobe_flow_diagram.py", "--counts-only",
        "--base-run-dir", rf, "--output-dir", rf / TIER1 / "strobe_diagram", env=env)
    run("analysis/institutional/plot_cv_and_forest_simple.py", "--table-only",
        "--groupings-file", groupings, "--output-dir", rf / TIER1 / "cv_forest_plots", env=env)
    run("analysis/institutional/analyze_all_cancer_types_institutional_stats.py",
        "--table-only", "--groupings-file", groupings,
        "--output-dir", rf / TIER1 / "institutional", env=env)

    log("5/7 legacy Unknown handling (same models, TIER 7 only)")
    legacy = {}
    for name, src, algo in [("rf", rf, "random_forest"), ("lr", lr, "logistic")]:
        dst = RUNS / f"run_{tag}_legacyUnknown_{name}"
        copy_run_without_tier7(src, dst)
        extra = ["--lr-significant-only"] if algo == "logistic" else []
        orchestrate("--tier7-only", "--algorithm", algo, "--run-dir", dst,
                    "--keep-unknown-category", "--hyperparams-source", "defaults", *extra)
        legacy[name] = dst

    log("6/7 derived tables")
    run("analysis/posthoc/generate_rf_lr_comparison.py", "--rf-run", rf, "--lr-run", lr,
        "--output-dir", derived / "comparison", env=env)
    run("analysis/posthoc/visualize_rf_lr_hit_intersection.py", "--rf-run-dir", rf,
        "--lr-run-dir", lr, "--output-dir", derived / "intersection", env=env)
    run("analysis/posthoc/adjust_hits_for_subject_center.py",
        "--intersection-run", derived / "intersection", "--rf-run", rf, "--lr-run", lr,
        "--mapping-csv", rf / MAPPING, "--output-dir", derived / "center_adjustment", env=env)
    run("analysis/posthoc/repool_global_fdr.py", "--rf-run", rf, "--lr-run", lr,
        "--label", "v1lock", "--output-dir", derived / "fdr_repool", env=env)
    run("analysis/posthoc/characterize_eligibility_threshold.py", "--rf-run", rf,
        "--lr-run", lr, "--output-dir", derived / "threshold",
        "--all-tests-csv", derived / "fdr_repool" / "global_fdr_all_tests.csv",
        "--hits-csv", derived / "center_adjustment" / "crude_vs_adjusted_all_features.csv",
        env=env)
    run("analysis/reporting/build_full_hit_table.py", "--rf-run", rf, "--lr-run", lr,
        "--lit-review-csv", *LIT_REVIEW, "--output-dir", derived / "full_hit_table", env=env)
    run("analysis/reporting/build_feature_table.py", "--groupings-file", groupings,
        "--output-dir", derived / "supplement", env=env)
    run("analysis/reporting/build_supplement_tables.py", "--rf-run", rf, "--lr-run", lr,
        "--rf-legacy-run", legacy["rf"], "--lr-legacy-run", legacy["lr"],
        "--rf-params-json", RF_PARAMS_DIR / "random_forest_universal_params.json",
        "--output-dir", derived / "supplement", env=env)
    run("analysis/posthoc/s3_sex_enrichment.py", "--groupings-file", groupings,
        "--check-table-s4", derived / "full_hit_table" / "full_hit_table_merged.csv",
        "--robustness", "--output-dir", derived / "sex_enrichment", env=env)
    run("analysis/reporting/manuscript_tables.py", "--rf-run", rf, "--lr-run", lr,
        "--s3-dir", derived / "sex_enrichment", "--output-dir", derived / "manuscript", env=env)

    # Run dirs live under runs/; link them so the output dir is self-describing.
    for name, path in [("setup", setup), ("rf", rf), ("lr", lr),
                       ("legacy_rf", legacy["rf"]), ("legacy_lr", legacy["lr"])]:
        (out / name).symlink_to(path, target_is_directory=True)

    log("7/7 collect")
    sys.exit(0 if collect(out) else 1)


if __name__ == "__main__":
    main()
