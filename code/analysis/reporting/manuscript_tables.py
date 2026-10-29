#!/usr/bin/env python3
"""Build the per-model ML metrics and sex-enrichment supplement tables.

Shared by build_results_stats.py, which adds the prose checks. Standalone use writes the two CSVs only:
  python manuscript_tables.py --rf-run runs/<rf_run> --lr-run runs/<lr_run> \
      --s3-dir <sex_enrichment_dir> --output-dir <dir>
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

# Exploratory BH threshold for the ungated sex comparison.
S3_Q_THRESHOLD = 0.05

ML_SIZE_COLS = ["n_train", "n_test", "n_actionable_train", "n_actionable_test"]
ML_METRIC_COLS = ["train_auc", "test_auc", "test_accuracy", "test_precision",
                  "test_recall", "test_f1", "brier_score"]
SEX_TABLE_COLS = ["ONCOTREE_CODE", "actionability_rate", "n_actionable", "n_non_actionable",
                  "odds_ratio", "or_ci_lower", "or_ci_upper", "p_fisher", "q_exploratory",
                  "n3_adj_or", "n3_p", "n4_adj_or", "n4_p", "enrichment", "enrichment_label",
                  "top_gene", "top_gene_n", "top_gene_pct_actionable", "top_gene_pct_patients"]


def find_one(base: Path, pattern: str) -> Path:
    matches = sorted(base.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} under {base}")
    return matches[-1]


def perf_summary(run_dir: Path) -> pd.DataFrame:
    """TIER 5 performance summary, rows with a numeric test AUC only."""
    f = find_one(run_dir / "tier5_final_models", "run_*/performance_summary.csv")
    df = pd.read_csv(f)
    df = df[pd.to_numeric(df["test_auc"], errors="coerce").notna()].copy()
    df["test_auc"] = df["test_auc"].astype(float)
    return df


def delong_ci_by_cancer(run_dir: Path) -> pd.DataFrame:
    """Per-cancer DeLong 95% CI on test AUC; empty if test predictions are absent."""
    try:
        from auc_utils import delong_auc_ci
        f = find_one(run_dir / "tier5_final_models", "run_*/test_predictions.csv")
    except (ImportError, FileNotFoundError):
        return pd.DataFrame()
    preds = pd.read_csv(f)
    rows = []
    for code, g in preds.groupby("cancer_type"):
        y, s = g["y_true"].to_numpy(), g["y_pred_proba"].to_numpy()
        if len(set(y)) < 2:
            continue  # AUC undefined
        try:
            _, lo, hi = delong_auc_ci(y, s)
        except Exception:
            continue  # degenerate covariance
        rows.append({"cancer_type": code,
                     "test_auc_ci_lower": round(float(lo), 4),
                     "test_auc_ci_upper": round(float(hi), 4)})
    return pd.DataFrame(rows).set_index("cancer_type") if rows else pd.DataFrame()


def _ml_arm(df: pd.DataFrame, run_dir) -> pd.DataFrame:
    keep = [c for c in ML_SIZE_COLS + ML_METRIC_COLS if c in df.columns]
    a = df[keep].copy()
    if {"train_auc", "test_auc"} <= set(a.columns):
        a["train_test_gap"] = a["train_auc"] - a["test_auc"]
    if {"n_actionable_test", "n_test"} <= set(a.columns):
        a["test_prevalence"] = a["n_actionable_test"] / a["n_test"]
    if run_dir is not None:
        ci = delong_ci_by_cancer(Path(run_dir))
        if not ci.empty:
            a = a.join(ci, how="left")
    return a


def ml_metrics_table(rf_perf: pd.DataFrame, lr_perf: pd.DataFrame,
                     rf_run=None, lr_run=None) -> pd.DataFrame:
    """ML metrics table: RF and LR side by side, one row per cancer type."""
    rf = rf_perf.set_index("cancer_type")
    lr = lr_perf.set_index("cancer_type")
    merged = (_ml_arm(rf, rf_run).add_prefix("rf_")
              .join(_ml_arm(lr, lr_run).add_prefix("lr_"), how="outer"))
    return merged.round(4)


def top_gene(db_path, codes) -> dict:
    """Most common Level 1 gene per cancer type, as (gene, distinct patients)."""
    conn = sqlite3.connect(db_path)
    try:
        out = {}
        for code in codes:
            rows = conn.execute(
                "SELECT Hugo_Symbol, COUNT(DISTINCT PATIENT_ID) AS n "
                "FROM perPatientOncoKBFilteredForModeling "
                "WHERE ONCOTREE_CODE = ? AND isOncoKbLevel1 = 1 "
                "GROUP BY Hugo_Symbol ORDER BY n DESC, Hugo_Symbol LIMIT 1", (code,)).fetchall()
            if rows:
                out[code] = (rows[0][0], int(rows[0][1]))
    finally:
        conn.close()
    return out


def _enrichment(row) -> str:
    if row["q_exploratory"] >= S3_Q_THRESHOLD:
        return "None"
    return "Female" if row["odds_ratio"] > 1 else "Male"


def load_sex_enrichment(s3_dir: Path) -> pd.DataFrame:
    """Cancer-level sex ORs merged with their institution- and age-adjusted robustness models."""
    cl = pd.read_csv(s3_dir / "cancer_level_or.csv")
    rob = pd.read_csv(s3_dir / "cancer_level_robustness.csv")
    cl = cl.merge(rob[["ONCOTREE_CODE", "n3_adj_or", "n3_p", "n4_adj_or", "n4_p"]],
                  on="ONCOTREE_CODE", how="left", validate="one_to_one")
    cl["enrichment"] = cl.apply(_enrichment, axis=1)
    return cl


def add_top_gene_columns(cl: pd.DataFrame, gd: pd.DataFrame, top: dict) -> pd.DataFrame:
    """Add actionability rate and top-gene share; gd is indexed by ONCOTREE_CODE."""
    cl["actionability_rate"] = cl["ONCOTREE_CODE"].map(gd["actionability_rate"]).astype(float)
    cl["top_gene"] = cl["ONCOTREE_CODE"].map(lambda c: top[c][0])
    cl["top_gene_n"] = cl["ONCOTREE_CODE"].map(lambda c: top[c][1])
    cl["top_gene_pct_actionable"] = (100 * cl["top_gene_n"]
                                     / cl["ONCOTREE_CODE"].map(gd["actionable_patients"]))
    cl["top_gene_pct_patients"] = (100 * cl["top_gene_n"]
                                   / cl["ONCOTREE_CODE"].map(gd["total_patients"]))
    return cl


def _enrichment_label(r) -> str:
    if r["enrichment"] == "None":
        return "None"
    robust = r["n3_p"] < 0.05 and (r["n3_adj_or"] > 1) == (r["odds_ratio"] > 1)
    return r["enrichment"] if robust else f"{r['enrichment']} (not robust to institution)"


def sex_enrichment_table(cl: pd.DataFrame) -> pd.DataFrame:
    """Sex enrichment table, sorted by actionability rate; cl must carry the top-gene columns."""
    cl["enrichment_label"] = cl.apply(_enrichment_label, axis=1)
    return cl.sort_values("actionability_rate", ascending=False)[SEX_TABLE_COLS]


def main():
    from db_config import get_default_db_path

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rf-run", type=Path, required=True)
    p.add_argument("--lr-run", type=Path, required=True)
    p.add_argument("--s3-dir", type=Path, required=True,
                   help="Output of s3_sex_enrichment.py --robustness")
    p.add_argument("--groupings-file", type=Path, default=None,
                   help="Default: the RF run's TIER 2a cancer_type_group_definitions.csv")
    p.add_argument("--db-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()

    db_path = a.db_path or Path(get_default_db_path())
    groupings = a.groupings_file or (a.rf_run / "tier2_data_setup" / "cancer_type_groupings"
                                     / "cancer_type_group_definitions.csv")
    a.output_dir.mkdir(parents=True, exist_ok=True)

    ml = ml_metrics_table(perf_summary(a.rf_run), perf_summary(a.lr_run), a.rf_run, a.lr_run)
    ml.to_csv(a.output_dir / "supp_table_ml_metrics.csv")

    gd = pd.read_csv(groupings).set_index("ONCOTREE_CODE")
    cl = load_sex_enrichment(a.s3_dir)
    outside = set(cl["ONCOTREE_CODE"].astype(str)) - set(gd.index.astype(str))
    if outside:
        sys.exit(f"FAIL: S3 codes outside the run's cohort: {sorted(outside)}")
    cl = add_top_gene_columns(cl, gd, top_gene(db_path, sorted(cl["ONCOTREE_CODE"])))
    sex_enrichment_table(cl).to_csv(a.output_dir / "supp_table_sex_enrichment.csv", index=False)
    print(f"Wrote supp_table_ml_metrics.csv and supp_table_sex_enrichment.csv to {a.output_dir}")


if __name__ == "__main__":
    main()
