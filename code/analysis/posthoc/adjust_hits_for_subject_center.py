#!/usr/bin/env python3
"""SUBJECT_CENTER sensitivity analysis for RF/LR cross-pathway hits.

For each intersected hit (sex, race, age; gene-stratified), fits a logistic
regression with and without C(SUBJECT_CENTER) and reports whether the
association survives institutional adjustment. Outcome: 1 = gene-actionable,
0 = non-actionable; patients with other actionable genes are excluded.
The verdict is the omnibus LR-test of the demographic block in the adjusted
model (versus an institution-only null). Centers with < --min-center-n
patients are pooled; unstable fits are reported crude-only.

Usage:
  python adjust_hits_for_subject_center.py --intersection-run <dir> \
      --rf-run runs/<rf_run> --lr-run runs/<lr_run> --mapping-csv <patient_to_group_mapping.csv>

Output: <output-dir>/{sex,race,age}/ tables and plots, plus summary.md.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))

from db_config import get_default_db_path  # noqa: E402
from figure_style import (  # noqa: E402
    apply_publication_style,
    get_neutral,
    get_palette,
)
from multi_tier_utils import (  # noqa: E402
    get_oncotree_codes_for_group,
    load_patient_to_group_mapping,
)

import statsmodels.api as sm  # noqa: E402
from statsmodels.formula.api import logit  # noqa: E402
from statsmodels.tools.sm_exceptions import PerfectSeparationError  # noqa: E402

PATIENT_TABLE = "perPatientOncoKBFilteredForModeling"

# Colors come from palette.json via figure_style; do not hard-code literals.
_SYSTEMS = get_palette("cancer_systems")
SEX_SYSTEM_COLOR = {k: v for k, v in _SYSTEMS.items() if k != "default"}
SYSTEM_COLOR_DEFAULT = _SYSTEMS["default"]

# Crude vs adjusted is also encoded by marker shape (square = crude, circle = adjusted).
CRUDE_COLOR = get_neutral("muted")
ADJUSTED_COLOR = get_palette("model_pathways")["rf_test"]

# Unencoded ink.
ANNOTATION_COLOR = get_neutral("annotation")
CONNECTOR_COLOR = get_neutral("grid")


# Hit loaders

def load_intersection_summary(intersection_run: Path, feature: str) -> pd.DataFrame:
    csv = intersection_run / feature / "intersection_summary.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No {feature}/intersection_summary.csv at {csv}")
    df = pd.read_csv(csv)
    print(f"  [{feature}] loaded {len(df)} intersected hits from {csv}")
    return df


def load_age_intersection(rf_run: Path, lr_run: Path) -> pd.DataFrame:
    """Inner-join RF/LR age_gene_significant.csv on (ONCOTREE_CODE, Hugo_Symbol)."""
    rel = Path("tier7_posthoc_statistics/gene_stratified/age_by_gene/age_gene_significant.csv")
    rf_csv = rf_run / rel
    lr_csv = lr_run / rel
    if not rf_csv.exists() or not lr_csv.exists():
        raise FileNotFoundError(f"age_gene_significant.csv missing in {rf_run} or {lr_run}")
    rf = pd.read_csv(rf_csv)
    lr = pd.read_csv(lr_csv)
    rf = rf[rf["significant_fdr"] == True]
    lr = lr[lr["significant_fdr"] == True]

    keep = ["ONCOTREE_CODE", "Hugo_Symbol", "gene_rank",
            "n_actionable", "n_non_actionable",
            "actionable_mean_age", "non_actionable_mean_age",
            "cohens_d", "rank_biserial_r", "mann_whitney_p", "p_fdr"]
    rf_sub = rf[keep].rename(columns={
        "gene_rank": "gene_rank_rf", "mann_whitney_p": "p_value_rf",
        "p_fdr": "p_fdr_rf",
    })
    lr_sub = lr[["ONCOTREE_CODE", "Hugo_Symbol", "gene_rank", "mann_whitney_p", "p_fdr"]].rename(columns={
        "gene_rank": "gene_rank_lr", "mann_whitney_p": "p_value_lr",
        "p_fdr": "p_fdr_lr",
    })
    merged = rf_sub.merge(lr_sub, on=["ONCOTREE_CODE", "Hugo_Symbol"], how="inner")
    merged["p_fdr_max"] = merged[["p_fdr_rf", "p_fdr_lr"]].max(axis=1)
    print(f"  [age] RF∩LR intersection: {len(merged)} hits ({len(rf)} RF, {len(lr)} LR)")
    return merged


def _short_label(ot_code: str, gene: str) -> str:
    # Strip a `tier1_` prefix (from rollup-aware runs) so labels are plain OncoTree codes.
    pretty = ot_code[len("tier1_"):] if ot_code.startswith("tier1_") else ot_code
    return f"{pretty} · {gene}"


def _sex_cancer_system(ot_code: str) -> str:
    code = ot_code[len("tier1_"):] if ot_code.startswith("tier1_") else ot_code
    if code in {"LUAD", "NSCLC", "LUNG"}:
        return "lung"
    if code in {"COAD", "COADREAD", "BOWEL", "READ"}:
        return "colorectal"
    if code in {"AML", "MYELOID"}:
        return "myeloid"
    if code in {"IHCH", "BILIARY_TRACT", "CHOL"}:
        return "biliary"
    return "other"


# Cohort builder

def build_cohort_for_hit(
    db_path: str,
    mapping_df: pd.DataFrame,
    tier_prefixed_code: str,
    gene: str,
) -> pd.DataFrame:
    """Build the 2-group cohort for one (cancer, gene) hit, one row per patient.

    Outcome 1 = gene-actionable, 0 = non-actionable; patients with only other
    actionable genes are excluded.
    """
    ot_codes = get_oncotree_codes_for_group(tier_prefixed_code, mapping_df)
    placeholders = ",".join("?" * len(ot_codes))

    conn = sqlite3.connect(db_path)
    try:
        # Numerator: gene-actionable patients
        gene_q = f"""
            SELECT DISTINCT
                PATIENT_ID, AGE_AT_SEQ_REPORT, SEX, RACE, SUBJECT_CENTER
            FROM {PATIENT_TABLE}
            WHERE ONCOTREE_CODE IN ({placeholders})
              AND Hugo_Symbol = ?
              AND isOncoKbLevel1 = 1
        """
        gene_df = pd.read_sql_query(gene_q, conn, params=ot_codes + [gene])

        # Denominator: full cohort with isOncoKbLevel1
        all_q = f"""
            SELECT DISTINCT
                PATIENT_ID, AGE_AT_SEQ_REPORT, SEX, RACE, SUBJECT_CENTER, isOncoKbLevel1
            FROM {PATIENT_TABLE}
            WHERE ONCOTREE_CODE IN ({placeholders})
        """
        all_df = pd.read_sql_query(all_q, conn, params=ot_codes)
    finally:
        conn.close()

    gene_df = gene_df.drop_duplicates(subset=["PATIENT_ID"], keep="first").copy()
    all_df = all_df.drop_duplicates(subset=["PATIENT_ID"], keep="first").copy()

    gene_ids = set(gene_df["PATIENT_ID"])
    non_act = all_df[all_df["isOncoKbLevel1"] == 0].copy()
    non_act = non_act[~non_act["PATIENT_ID"].isin(gene_ids)]
    non_act["outcome"] = 0

    gene_df = gene_df.assign(outcome=1)
    cohort = pd.concat(
        [gene_df[["PATIENT_ID", "AGE_AT_SEQ_REPORT", "SEX", "RACE", "SUBJECT_CENTER", "outcome"]],
         non_act[["PATIENT_ID", "AGE_AT_SEQ_REPORT", "SEX", "RACE", "SUBJECT_CENTER", "outcome"]]],
        ignore_index=True,
    )
    return cohort


# Stability guard + fit helpers

def collapse_small_centers(
    df: pd.DataFrame, min_n: int = 5, label_col: str = "SUBJECT_CENTER"
) -> Tuple[pd.DataFrame, dict]:
    """Collapse centers with < min_n cohort rows into OTHER_SMALL; return df and summary stats."""
    df = df.copy()
    counts = df[label_col].value_counts(dropna=False)
    small = counts[counts < min_n].index.tolist()
    df.loc[df[label_col].isin(small), label_col] = "OTHER_SMALL"

    counts_after = df[label_col].value_counts(dropna=False)
    stats = {
        "n_centers_raw": int(len(counts)),
        "n_centers_used": int(len(counts_after)),
        "n_collapsed_into_other": int(len(small)),
        "largest_center_pct": float(counts_after.iloc[0] / counts_after.sum() * 100),
        "smallest_center_n": int(counts_after.iloc[-1]),
    }
    return df, stats


def drop_separating_levels(
    df: pd.DataFrame, level_col: str, outcome_col: str = "outcome",
    other_label: str = "OTHER_SEPARATED",
) -> Tuple[pd.DataFrame, int]:
    """Pool levels where the outcome is constant into one bucket.

    Quasi-separation in C(LEVEL) breaks MLE convergence; pooling keeps the patients in the model.
    """
    df = df.copy()
    if isinstance(df[level_col].dtype, pd.CategoricalDtype):
        df[level_col] = df[level_col].astype(object)
    var_per_level = df.groupby(level_col)[outcome_col].nunique()
    bad_levels = var_per_level[var_per_level == 1].index.tolist()
    if bad_levels:
        df.loc[df[level_col].isin(bad_levels), level_col] = other_label
    return df, len(bad_levels)


def _wald_or_ci(coef: float, se: float) -> Tuple[float, float, float]:
    or_ = float(np.exp(coef))
    ci_l = float(np.exp(coef - 1.96 * se))
    ci_u = float(np.exp(coef + 1.96 * se))
    return or_, ci_l, ci_u


# Solvers tried in order. Newton inverts the Hessian each step and fails when a nuisance
# C(SUBJECT_CENTER) level is outcome-constant; quasi-Newton solvers reach the same
# unpenalized MLE. This is not a penalized/Firth fallback.
_SOLVER_CHAIN = ("newton", "bfgs", "lbfgs")


def _safe_logit(formula: str, data: pd.DataFrame) -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    """Fit logit, falling back through ``_SOLVER_CHAIN``.

    Returns None only if every solver raises or fails to converge.
    """
    for method in _SOLVER_CHAIN:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = logit(formula, data=data).fit(
                    disp=False, maxiter=500, method=method)
            except (PerfectSeparationError, np.linalg.LinAlgError, ValueError):
                continue
        if not bool(res.mle_retvals.get("converged", True)):
            continue
        res._balder_solver = method
        return res
    return None


def _solver_of(res) -> str:
    return getattr(res, "_balder_solver", "") if res is not None else ""


def _lr_test(full_ll: float, reduced_ll: float, df_diff: int) -> float:
    """Likelihood-ratio chi-square test p-value."""
    from scipy.stats import chi2
    if df_diff <= 0:
        return float("nan")
    stat = 2 * (full_ll - reduced_ll)
    if stat < 0:
        return float("nan")
    return float(chi2.sf(stat, df_diff))


# Per-hit fitters

SEX_REF = "Male"
AGE_PER = 10.0  # report OR per 10-year increment


def fit_sex_hit(cohort: pd.DataFrame, min_n: int = 5) -> dict:
    df = cohort[cohort["SEX"].isin(["Female", "Male"])].dropna(subset=["SEX", "outcome", "SUBJECT_CENTER"]).copy()
    df["SEX"] = pd.Categorical(df["SEX"], categories=[SEX_REF, "Female"])
    df, center_stats = collapse_small_centers(df, min_n=min_n)

    # Stability checks for SEX
    crosstab = pd.crosstab(df["outcome"], df["SEX"])
    smallest_cell = int(crosstab.values.min()) if crosstab.size else 0
    n_centers_used = center_stats["n_centers_used"]
    stable_demographic = smallest_cell >= 5
    stable_centers = n_centers_used >= 2

    out = {
        "n_total": int(len(df)),
        "smallest_demographic_cell": smallest_cell,
        **center_stats,
        "stable_demographic": stable_demographic,
        "stable_centers": stable_centers,
        "stable_fit": stable_demographic and stable_centers,
        "crude_OR": np.nan, "crude_OR_ci_low": np.nan, "crude_OR_ci_high": np.nan,
        "crude_p": np.nan,
        "adj_OR": np.nan, "adj_OR_ci_low": np.nan, "adj_OR_ci_high": np.nan,
        "adj_p": np.nan, "lr_test_p": np.nan,
        "notes": "",
        "adj_solver": "",
    }

    # Crude
    crude = _safe_logit("outcome ~ C(SEX)", df)
    if crude is None:
        out["notes"] = "crude_fit_failed"
        return out
    sex_param = "C(SEX)[T.Female]"
    if sex_param in crude.params.index:
        or_, lo, hi = _wald_or_ci(crude.params[sex_param], crude.bse[sex_param])
        out.update({"crude_OR": or_, "crude_OR_ci_low": lo, "crude_OR_ci_high": hi,
                    "crude_p": float(crude.pvalues[sex_param])})

    if not out["stable_fit"]:
        if not stable_demographic:
            out["notes"] = f"unstable: smallest demographic*outcome cell = {smallest_cell} < 5"
        elif not stable_centers:
            out["notes"] = f"unstable: only {n_centers_used} center(s) after collapse"
        return out

    # Adjusted (with separation fallback)
    df_adj = df
    adj = _safe_logit("outcome ~ C(SEX) + C(SUBJECT_CENTER)", df_adj)
    null_center = _safe_logit("outcome ~ C(SUBJECT_CENTER)", df_adj)
    if adj is None or null_center is None:
        df_adj, n_dropped = drop_separating_levels(df, "SUBJECT_CENTER")
        out["n_centers_pooled_separating"] = n_dropped
        adj = _safe_logit("outcome ~ C(SEX) + C(SUBJECT_CENTER)", df_adj)
        null_center = _safe_logit("outcome ~ C(SUBJECT_CENTER)", df_adj)
        if adj is None or null_center is None:
            out["stable_fit"] = False
            out["notes"] = f"adjusted_fit_failed_after_pooling (pooled {n_dropped} separating centers)"
            return out
        out["notes"] = f"pooled {n_dropped} separating center(s) to converge"
    if sex_param in adj.params.index:
        or_, lo, hi = _wald_or_ci(adj.params[sex_param], adj.bse[sex_param])
        out.update({"adj_OR": or_, "adj_OR_ci_low": lo, "adj_OR_ci_high": hi,
                    "adj_p": float(adj.pvalues[sex_param])})
    out["adj_solver"] = _solver_of(adj)
    out["lr_test_p"] = _lr_test(adj.llf, null_center.llf, df_diff=1)
    return out


def fit_age_hit(cohort: pd.DataFrame, min_n: int = 5) -> dict:
    df = cohort.dropna(subset=["AGE_AT_SEQ_REPORT", "outcome", "SUBJECT_CENTER"]).copy()
    df, center_stats = collapse_small_centers(df, min_n=min_n)
    n_centers_used = center_stats["n_centers_used"]

    # Stability: enough events per center bucket
    stable_centers = n_centers_used >= 2
    # At least 5 cases and 5 non-cases
    n_cases = int((df["outcome"] == 1).sum())
    n_controls = int((df["outcome"] == 0).sum())
    stable_demographic = n_cases >= 5 and n_controls >= 5

    out = {
        "n_total": int(len(df)),
        "n_cases": n_cases,
        "n_controls": n_controls,
        **center_stats,
        "stable_demographic": stable_demographic,
        "stable_centers": stable_centers,
        "stable_fit": stable_demographic and stable_centers,
        "crude_OR_per_decade": np.nan, "crude_OR_ci_low": np.nan, "crude_OR_ci_high": np.nan,
        "crude_p": np.nan,
        "adj_OR_per_decade": np.nan, "adj_OR_ci_low": np.nan, "adj_OR_ci_high": np.nan,
        "adj_p": np.nan, "lr_test_p": np.nan,
        "notes": "",
        "adj_solver": "",
    }

    # AGE/10 so the coefficient is OR per 10 years
    df = df.assign(AGE_DECADE=df["AGE_AT_SEQ_REPORT"].astype(float) / AGE_PER)

    crude = _safe_logit("outcome ~ AGE_DECADE", df)
    if crude is None:
        out["notes"] = "crude_fit_failed"
        return out
    if "AGE_DECADE" in crude.params.index:
        or_, lo, hi = _wald_or_ci(crude.params["AGE_DECADE"], crude.bse["AGE_DECADE"])
        out.update({"crude_OR_per_decade": or_, "crude_OR_ci_low": lo, "crude_OR_ci_high": hi,
                    "crude_p": float(crude.pvalues["AGE_DECADE"])})

    if not out["stable_fit"]:
        out["notes"] = "unstable cohort"
        return out

    df_adj = df
    adj = _safe_logit("outcome ~ AGE_DECADE + C(SUBJECT_CENTER)", df_adj)
    null_center = _safe_logit("outcome ~ C(SUBJECT_CENTER)", df_adj)
    if adj is None or null_center is None:
        df_adj, n_dropped = drop_separating_levels(df, "SUBJECT_CENTER")
        out["n_centers_pooled_separating"] = n_dropped
        adj = _safe_logit("outcome ~ AGE_DECADE + C(SUBJECT_CENTER)", df_adj)
        null_center = _safe_logit("outcome ~ C(SUBJECT_CENTER)", df_adj)
        if adj is None or null_center is None:
            out["stable_fit"] = False
            out["notes"] = f"adjusted_fit_failed_after_pooling (pooled {n_dropped})"
            return out
        out["notes"] = f"pooled {n_dropped} separating center(s) to converge"
    if "AGE_DECADE" in adj.params.index:
        or_, lo, hi = _wald_or_ci(adj.params["AGE_DECADE"], adj.bse["AGE_DECADE"])
        out.update({"adj_OR_per_decade": or_, "adj_OR_ci_low": lo, "adj_OR_ci_high": hi,
                    "adj_p": float(adj.pvalues["AGE_DECADE"])})
    out["adj_solver"] = _solver_of(adj)
    out["lr_test_p"] = _lr_test(adj.llf, null_center.llf, df_diff=1)
    return out


def fit_race_hit(cohort: pd.DataFrame, min_n: int = 5) -> dict:
    """Race is multi-category; headline is the omnibus LR-test of the RACE block.

    Unknown is retained; the reference level is the largest category.
    """
    df = cohort.dropna(subset=["RACE", "outcome", "SUBJECT_CENTER"]).copy()
    df = df[df["RACE"].astype(str).str.len() > 0]
    df, center_stats = collapse_small_centers(df, min_n=min_n)
    n_centers_used = center_stats["n_centers_used"]

    # Collapse rare race levels to OTHER_RACE_SMALL for stability.
    race_counts = df["RACE"].value_counts()
    rare_races = race_counts[race_counts < min_n].index.tolist()
    df.loc[df["RACE"].isin(rare_races), "RACE"] = "OTHER_RACE_SMALL"
    race_counts_after = df["RACE"].value_counts()
    largest_race = race_counts_after.index[0] if len(race_counts_after) else "Unknown"
    df["RACE"] = pd.Categorical(df["RACE"],
                                categories=[largest_race] + [r for r in race_counts_after.index if r != largest_race])

    n_race_levels = int(df["RACE"].cat.categories.size)
    stable_demographic = n_race_levels >= 2 and len(df) >= 50
    stable_centers = n_centers_used >= 2

    out = {
        "n_total": int(len(df)),
        "n_race_levels_used": n_race_levels,
        "race_reference": largest_race,
        "n_rare_races_collapsed": int(len(rare_races)),
        **center_stats,
        "stable_demographic": stable_demographic,
        "stable_centers": stable_centers,
        "stable_fit": stable_demographic and stable_centers,
        "crude_omnibus_p": np.nan, "crude_pseudo_r2": np.nan,
        "adj_omnibus_p": np.nan, "adj_pseudo_r2": np.nan,
        "lr_test_p": np.nan,  # alias of adj_omnibus_p
        "notes": "",
        "adj_solver": "",
    }

    crude_full = _safe_logit("outcome ~ C(RACE)", df)
    crude_null = _safe_logit("outcome ~ 1", df)
    df_for_crude = df
    if crude_full is None or crude_null is None:
        df_for_crude, n_race_dropped = drop_separating_levels(df, "RACE",
                                                               other_label="OTHER_RACE_SEPARATED")
        crude_full = _safe_logit("outcome ~ C(RACE)", df_for_crude)
        crude_null = _safe_logit("outcome ~ 1", df_for_crude)
        out["n_race_levels_pooled_separating"] = n_race_dropped
        if crude_full is None or crude_null is None:
            out["notes"] = "crude_fit_failed_after_pooling"
            out["stable_fit"] = False
            return out
        # Recount race levels post-pooling
        n_race_levels = int(df_for_crude["RACE"].astype("category").cat.categories.size)
    out["crude_omnibus_p"] = _lr_test(crude_full.llf, crude_null.llf, df_diff=n_race_levels - 1)
    out["crude_pseudo_r2"] = float(getattr(crude_full, "prsquared", np.nan))

    if not out["stable_fit"]:
        out["notes"] = f"unstable: levels={n_race_levels}, centers={n_centers_used}"
        return out

    df_adj = df_for_crude
    adj_full = _safe_logit("outcome ~ C(RACE) + C(SUBJECT_CENTER)", df_adj)
    adj_null = _safe_logit("outcome ~ C(SUBJECT_CENTER)", df_adj)
    if adj_full is None or adj_null is None:
        df_adj, n_centers_dropped = drop_separating_levels(df_adj, "SUBJECT_CENTER")
        out["n_centers_pooled_separating"] = n_centers_dropped
        adj_full = _safe_logit("outcome ~ C(RACE) + C(SUBJECT_CENTER)", df_adj)
        adj_null = _safe_logit("outcome ~ C(SUBJECT_CENTER)", df_adj)
        if adj_full is None or adj_null is None:
            out["stable_fit"] = False
            out["notes"] = f"adjusted_fit_failed_after_pooling"
            return out
        out["notes"] = f"pooled {n_centers_dropped} separating center(s) to converge"
    out["adj_solver"] = _solver_of(adj_full)
    out["adj_omnibus_p"] = _lr_test(adj_full.llf, adj_null.llf, df_diff=n_race_levels - 1)
    out["adj_pseudo_r2"] = float(getattr(adj_full, "prsquared", np.nan))
    out["lr_test_p"] = out["adj_omnibus_p"]
    return out


# Per-feature pipelines

def run_sex_pipeline(
    intersection: pd.DataFrame, db_path: str, mapping_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    feature_dir = output_dir / "sex"
    feature_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== SEX adjustment ===")

    rows: List[dict] = []
    for _, h in intersection.iterrows():
        ot, gene = h["ONCOTREE_CODE"], h["Hugo_Symbol"]
        cohort = build_cohort_for_hit(db_path, mapping_df, ot, gene)
        fit = fit_sex_hit(cohort)
        # Reference values from the intersection summary.
        ref_OR = float(h.get("odds_ratio", np.nan))
        ref_lo = float(h.get("or_ci_lower", np.nan))
        ref_hi = float(h.get("or_ci_upper", np.nan))
        ref_p_fdr = float(h.get("p_fdr_max", np.nan))
        rows.append({
            "ONCOTREE_CODE": ot, "Hugo_Symbol": gene,
            "ref_OR_intersection": ref_OR,
            "ref_OR_ci_low": ref_lo, "ref_OR_ci_high": ref_hi,
            "ref_p_fdr_max": ref_p_fdr,
            **fit,
        })
        msg = (f"  {ot}/{gene}: crude_OR={fit['crude_OR']:.3f}  "
               f"adj_OR={fit['adj_OR']:.3f}  LR_p={fit['lr_test_p']:.2e}  "
               f"stable={fit['stable_fit']}  centers={fit['n_centers_used']}")
        print(msg)
    df = pd.DataFrame(rows)
    df.to_csv(feature_dir / "adjusted_results_sex.csv", index=False)
    render_paired_forest_sex(df, feature_dir / "forest_crude_vs_adjusted")
    return df


def run_age_pipeline(
    intersection: pd.DataFrame, db_path: str, mapping_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    feature_dir = output_dir / "age"
    feature_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== AGE adjustment ===")

    rows: List[dict] = []
    for _, h in intersection.iterrows():
        ot, gene = h["ONCOTREE_CODE"], h["Hugo_Symbol"]
        cohort = build_cohort_for_hit(db_path, mapping_df, ot, gene)
        fit = fit_age_hit(cohort)
        rows.append({
            "ONCOTREE_CODE": ot, "Hugo_Symbol": gene,
            "ref_cohens_d": float(h.get("cohens_d", np.nan)),
            "ref_mann_whitney_p": float(h.get("p_value_rf", np.nan)),
            "ref_p_fdr_max": float(h.get("p_fdr_max", np.nan)),
            **fit,
        })
        msg = (f"  {ot}/{gene}: crude_OR/dec={fit['crude_OR_per_decade']:.3f}  "
               f"adj_OR/dec={fit['adj_OR_per_decade']:.3f}  "
               f"LR_p={fit['lr_test_p']:.2e}  stable={fit['stable_fit']}")
        print(msg)
    df = pd.DataFrame(rows)
    df.to_csv(feature_dir / "adjusted_results_age.csv", index=False)
    render_paired_forest_age(df, feature_dir / "forest_crude_vs_adjusted")
    return df


def run_race_pipeline(
    intersection: pd.DataFrame, db_path: str, mapping_df: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    feature_dir = output_dir / "race"
    feature_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== RACE adjustment ===")

    rows: List[dict] = []
    for _, h in intersection.iterrows():
        ot, gene = h["ONCOTREE_CODE"], h["Hugo_Symbol"]
        cohort = build_cohort_for_hit(db_path, mapping_df, ot, gene)
        fit = fit_race_hit(cohort)
        rows.append({
            "ONCOTREE_CODE": ot, "Hugo_Symbol": gene,
            "ref_cramers_v": float(h.get("cramers_v", np.nan)),
            "ref_p_fdr_max": float(h.get("p_fdr_max", np.nan)),
            **fit,
        })
        msg = (f"  {ot}/{gene}: crude_p={fit['crude_omnibus_p']:.2e}  "
               f"adj_p={fit['adj_omnibus_p']:.2e}  "
               f"levels={fit['n_race_levels_used']}  stable={fit['stable_fit']}")
        print(msg)
    df = pd.DataFrame(rows)
    df.to_csv(feature_dir / "adjusted_results_race.csv", index=False)
    render_race_pvalue_dotplot(df, feature_dir / "race_crude_vs_adjusted_pvalues")
    return df


# Plots

def render_paired_forest_sex(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    df = df.sort_values("adj_OR", ascending=True, na_position="first").reset_index(drop=True)
    n = len(df)
    fig_h = max(4.0, 0.55 * n + 1.6)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))

    y = np.arange(n)
    for i, (_, row) in enumerate(df.iterrows()):
        sys_color = SEX_SYSTEM_COLOR.get(_sex_cancer_system(row["ONCOTREE_CODE"]), SYSTEM_COLOR_DEFAULT)
        # Crude marker
        if not np.isnan(row["crude_OR"]):
            ax.hlines(i - 0.18, row["crude_OR_ci_low"], row["crude_OR_ci_high"],
                      color=CRUDE_COLOR, linewidth=1.3, alpha=0.9)
            ax.scatter(row["crude_OR"], i - 0.18, c=CRUDE_COLOR, s=32,
                       edgecolors=get_neutral("ink"), linewidths=0.4, zorder=3, marker="s")
        # Adjusted marker
        if not np.isnan(row["adj_OR"]):
            ax.hlines(i + 0.18, row["adj_OR_ci_low"], row["adj_OR_ci_high"],
                      color=sys_color, linewidth=1.7, alpha=0.95)
            ax.scatter(row["adj_OR"], i + 0.18, c=sys_color, s=42,
                       edgecolors=get_neutral("ink"), linewidths=0.4, zorder=3, marker="o")
        # Annotate with stability + LR p
        note = ""
        if not row["stable_fit"]:
            note = "  (unstable)"
        elif not np.isnan(row["lr_test_p"]):
            note = f"  LR p={row['lr_test_p']:.1e}"
        ax.text(0.05, i, note, va="center", ha="left", fontsize=7,
                color=ANNOTATION_COLOR, transform=ax.get_yaxis_transform())

    ax.axvline(1.0, color=get_neutral("rule"), linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Odds ratio (Female / Male) — 95% Wald CI")
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(r["ONCOTREE_CODE"], r["Hugo_Symbol"])
                        for _, r in df.iterrows()], fontsize=9)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_title("SEX hits — crude vs. SUBJECT_CENTER-adjusted OR\n"
                 "Square = crude logistic, Circle = adjusted (color by cancer system)",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([0], [0], marker="s", color=get_neutral("on_fill"), markerfacecolor=CRUDE_COLOR,
               markeredgecolor=get_neutral("ink"), markersize=8, label="Crude"),
        Line2D([0], [0], marker="o", color=get_neutral("on_fill"), markerfacecolor=ADJUSTED_COLOR,
               markeredgecolor=get_neutral("ink"), markersize=8, label="Adjusted (institution)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_paired_forest_age(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    df = df.sort_values("adj_OR_per_decade", ascending=True, na_position="first").reset_index(drop=True)
    n = len(df)
    fig_h = max(4.0, 0.45 * n + 1.6)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))

    y = np.arange(n)
    for i, (_, row) in enumerate(df.iterrows()):
        if not np.isnan(row["crude_OR_per_decade"]):
            ax.hlines(i - 0.18, row["crude_OR_ci_low"], row["crude_OR_ci_high"],
                      color=CRUDE_COLOR, linewidth=1.3, alpha=0.9)
            ax.scatter(row["crude_OR_per_decade"], i - 0.18, c=CRUDE_COLOR, s=32,
                       edgecolors=get_neutral("ink"), linewidths=0.4, zorder=3, marker="s")
        if not np.isnan(row["adj_OR_per_decade"]):
            ax.hlines(i + 0.18, row["adj_OR_ci_low"], row["adj_OR_ci_high"],
                      color=ADJUSTED_COLOR, linewidth=1.7, alpha=0.95)
            ax.scatter(row["adj_OR_per_decade"], i + 0.18, c=ADJUSTED_COLOR, s=42,
                       edgecolors=get_neutral("ink"), linewidths=0.4, zorder=3, marker="o")
        note = "  (unstable)" if not row["stable_fit"] else (
            f"  LR p={row['lr_test_p']:.1e}" if not np.isnan(row["lr_test_p"]) else ""
        )
        ax.text(0.05, i, note, va="center", ha="left", fontsize=7,
                color=ANNOTATION_COLOR, transform=ax.get_yaxis_transform())

    ax.axvline(1.0, color=get_neutral("rule"), linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio per 10-year age increment — 95% Wald CI")
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(r["ONCOTREE_CODE"], r["Hugo_Symbol"])
                        for _, r in df.iterrows()], fontsize=8)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_title("AGE hits — crude vs. SUBJECT_CENTER-adjusted OR per decade", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([0], [0], marker="s", color=get_neutral("on_fill"), markerfacecolor=CRUDE_COLOR,
               markeredgecolor=get_neutral("ink"), markersize=8, label="Crude"),
        Line2D([0], [0], marker="o", color=get_neutral("on_fill"), markerfacecolor=ADJUSTED_COLOR,
               markeredgecolor=get_neutral("ink"), markersize=8, label="Adjusted"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_race_pvalue_dotplot(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    df = df.copy()
    df["neg_log_crude"] = -np.log10(df["crude_omnibus_p"].clip(lower=1e-300))
    df["neg_log_adj"] = -np.log10(df["adj_omnibus_p"].clip(lower=1e-300))
    df = df.sort_values("neg_log_adj", ascending=True, na_position="first").reset_index(drop=True)
    n = len(df)
    fig_h = max(4.0, 0.45 * n + 1.6)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))

    y = np.arange(n)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([row["neg_log_crude"], row["neg_log_adj"]], [i, i],
                color=CONNECTOR_COLOR, linewidth=1.0, alpha=0.7, zorder=1)
        ax.scatter(row["neg_log_crude"], i, c=CRUDE_COLOR, s=42,
                   edgecolors=get_neutral("ink"), linewidths=0.4, zorder=3, marker="s")
        if not np.isnan(row["neg_log_adj"]):
            ax.scatter(row["neg_log_adj"], i, c=ADJUSTED_COLOR, s=52,
                       edgecolors=get_neutral("ink"), linewidths=0.4, zorder=3, marker="o")
        note = "  (unstable)" if not row["stable_fit"] else ""
        ax.text(0.02, i, note, va="center", ha="left", fontsize=7,
                color=ANNOTATION_COLOR, transform=ax.get_yaxis_transform())

    ax.axvline(-np.log10(0.05), color=get_neutral("rule"), linestyle="--", linewidth=1.0, alpha=0.7,
               label="α = 0.05")
    ax.set_xlabel("−log10(p) — omnibus LR-test for RACE block")
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(r["ONCOTREE_CODE"], r["Hugo_Symbol"])
                        for _, r in df.iterrows()], fontsize=8)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_title("RACE hits — crude vs. SUBJECT_CENTER-adjusted omnibus p-value", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D([0], [0], marker="s", color=get_neutral("on_fill"), markerfacecolor=CRUDE_COLOR,
               markeredgecolor=get_neutral("ink"), markersize=8, label="Crude"),
        Line2D([0], [0], marker="o", color=get_neutral("on_fill"), markerfacecolor=ADJUSTED_COLOR,
               markeredgecolor=get_neutral("ink"), markersize=8, label="Adjusted"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# Verdict + summary.md

def _verdict_sex_or_age(row: pd.Series, alpha: float = 0.05) -> str:
    if not row["stable_fit"]:
        return "unstable"
    if np.isnan(row["lr_test_p"]):
        return "unstable"
    if row["lr_test_p"] < alpha:
        return "survives"
    return "attenuates"


def write_summary(
    output_dir: Path,
    sex_df: Optional[pd.DataFrame],
    race_df: Optional[pd.DataFrame],
    age_df: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# SUBJECT_CENTER sensitivity analysis — summary",
        "",
        f"_Generated: {datetime.now().isoformat(timespec='seconds')}_",
        "",
        f"- Intersection run: `{args.intersection_run}`",
        f"- RF run: `{args.rf_run}`",
        f"- LR run: `{args.lr_run}`",
        f"- Mapping CSV: `{args.mapping_csv}`",
        "",
        "## Method",
        "",
        "For each intersected hit, fit `Logit(outcome) ~ demographic` (crude) and "
        "`Logit(outcome) ~ demographic + C(SUBJECT_CENTER)` (adjusted). Outcome is "
        "1 if patient has gene-actionable status (`isOncoKbLevel1=1` AND `Hugo_Symbol==gene`), "
        "0 if patient is non-actionable in the cancer cohort. Patients with other "
        "actionable genes are excluded. SUBJECT_CENTER levels with < "
        f"{args.min_center_n} patients are collapsed into `OTHER_SMALL`. The "
        "headline is the LR-test p-value of the demographic block in the adjusted "
        "model.",
        "",
        "**Verdict**: `survives` (LR-test p < 0.05), `attenuates` (LR-test p ≥ 0.05), "
        "or `unstable` (could not fit; institution structure too sparse).",
        "",
    ]

    def _section(name: str, df: pd.DataFrame, head_cols: list[str]) -> List[str]:
        if df is None or df.empty:
            return [f"## {name}", "", "_no hits_", ""]
        df = df.copy()
        df["verdict"] = df.apply(_verdict_sex_or_age, axis=1)
        v_counts = df["verdict"].value_counts().to_dict()
        out = [
            f"## {name} ({len(df)} hits)",
            "",
            f"- survives adjustment: **{v_counts.get('survives', 0)}**",
            f"- attenuates: **{v_counts.get('attenuates', 0)}**",
            f"- unstable: **{v_counts.get('unstable', 0)}**",
            "",
            "| " + " | ".join(head_cols + ["verdict"]) + " |",
            "|" + "|".join(["---"] * (len(head_cols) + 1)) + "|",
        ]
        for _, r in df.iterrows():
            cells = []
            for c in head_cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    if abs(v) >= 1e-3 and abs(v) < 1e6:
                        cells.append(f"{v:.3f}")
                    else:
                        cells.append(f"{v:.2e}")
                else:
                    cells.append(str(v))
            cells.append(r["verdict"])
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
        return out

    lines += _section("SEX", sex_df, [
        "ONCOTREE_CODE", "Hugo_Symbol",
        "crude_OR", "adj_OR", "lr_test_p",
        "n_total", "n_centers_used",
    ])
    lines += _section("RACE", race_df, [
        "ONCOTREE_CODE", "Hugo_Symbol",
        "crude_omnibus_p", "adj_omnibus_p",
        "n_total", "n_race_levels_used", "n_centers_used",
    ])
    lines += _section("AGE", age_df, [
        "ONCOTREE_CODE", "Hugo_Symbol",
        "crude_OR_per_decade", "adj_OR_per_decade", "lr_test_p",
        "n_total", "n_centers_used",
    ])

    # Top movers
    if sex_df is not None and not sex_df.empty:
        movers = sex_df.dropna(subset=["crude_OR", "adj_OR"]).copy()
        movers["delta_log_or"] = np.log(movers["adj_OR"]) - np.log(movers["crude_OR"])
        movers = movers.reindex(movers["delta_log_or"].abs().sort_values(ascending=False).index).head(5)
        lines.append("## Top SEX movers (largest |Δlog(OR)| crude → adjusted)")
        lines.append("")
        lines.append("| ONCOTREE_CODE | Hugo_Symbol | crude_OR | adj_OR | Δlog(OR) | LR p |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in movers.iterrows():
            lines.append(
                f"| {r['ONCOTREE_CODE']} | {r['Hugo_Symbol']} | {r['crude_OR']:.3f} | "
                f"{r['adj_OR']:.3f} | {r['delta_log_or']:+.3f} | {r['lr_test_p']:.2e} |"
            )
        lines.append("")

    (output_dir / "summary.md").write_text("\n".join(lines))
    print(f"\n  wrote {output_dir / 'summary.md'}")

    write_crude_vs_adjusted_table(output_dir, sex_df, race_df, age_df)


def write_crude_vs_adjusted_table(
    output_dir: Path,
    sex_df: Optional[pd.DataFrame],
    race_df: Optional[pd.DataFrame],
    age_df: Optional[pd.DataFrame],
) -> None:
    """Write one long-format crude-vs-adjusted table across all features.

    Ancestry has an omnibus p only, so its effect columns are NaN by design.
    """
    frames = []

    if sex_df is not None and not sex_df.empty:
        d = sex_df.copy()
        d["feature"] = "sex"
        d["effect_units"] = "OR (Female vs Male)"
        d = d.rename(columns={"crude_OR": "crude_effect", "adj_OR": "adj_effect"})
        frames.append(d)

    if age_df is not None and not age_df.empty:
        d = age_df.copy()
        d["feature"] = "age"
        d["effect_units"] = "OR per 10 years"
        d = d.rename(columns={"crude_OR_per_decade": "crude_effect",
                              "adj_OR_per_decade": "adj_effect"})
        frames.append(d)

    if race_df is not None and not race_df.empty:
        d = race_df.copy()
        d["feature"] = "ancestry"
        d["effect_units"] = "omnibus LR test only (no per-level OR)"
        d["crude_effect"] = np.nan
        d["adj_effect"] = np.nan
        d = d.rename(columns={"crude_omnibus_p": "crude_p", "adj_omnibus_p": "adj_p"})
        frames.append(d)

    if not frames:
        return

    cols = ["feature", "ONCOTREE_CODE", "Hugo_Symbol", "effect_units",
            "crude_effect", "adj_effect", "delta_log_effect",
            "crude_p", "adj_p", "lr_test_p", "verdict",
            "n_total", "n_centers_used", "adj_solver", "notes"]

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["delta_log_effect"] = np.log(out["adj_effect"]) - np.log(out["crude_effect"])
    out["verdict"] = out.apply(_verdict_sex_or_age, axis=1)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols].sort_values(["feature", "ONCOTREE_CODE", "Hugo_Symbol"])

    dest = output_dir / "crude_vs_adjusted_all_features.csv"
    out.to_csv(dest, index=False)
    print(f"  wrote {dest}")


# Main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intersection-run", type=Path, required=True,
                        help="rf_lr_hit_intersection run dir (must contain sex/, race/)")
    parser.add_argument("--rf-run", type=Path, required=True,
                        help="RF run dir holding age_gene_significant.csv")
    parser.add_argument("--lr-run", type=Path, required=True,
                        help="LR run dir holding age_gene_significant.csv")
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--features", nargs="+", default=["sex", "race", "age"],
                        choices=["sex", "race", "age"])
    parser.add_argument("--hits-limit", type=int, default=None,
                        help="If set, only fit the first N hits per feature (smoke test).")
    parser.add_argument("--min-center-n", type=int, default=5)
    args = parser.parse_args()

    db_path = str(args.db_path) if args.db_path else get_default_db_path()
    print(f"DB: {db_path}")

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = REPO_ROOT / "runs" / "subject_center_adjustment" / f"run_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output_dir}")

    # Shared rcParams (fonts, type scale, vector-safe PDF).
    apply_publication_style()

    mapping_df = load_patient_to_group_mapping(mapping_file=args.mapping_csv, verbose=False)

    sex_df = race_df = age_df = None

    if "sex" in args.features:
        intersection = load_intersection_summary(args.intersection_run, "sex")
        if args.hits_limit:
            intersection = intersection.head(args.hits_limit)
        sex_df = run_sex_pipeline(intersection, db_path, mapping_df, args.output_dir)

    if "race" in args.features:
        intersection = load_intersection_summary(args.intersection_run, "race")
        if args.hits_limit:
            intersection = intersection.head(args.hits_limit)
        race_df = run_race_pipeline(intersection, db_path, mapping_df, args.output_dir)

    if "age" in args.features:
        intersection = load_age_intersection(args.rf_run, args.lr_run)
        if args.hits_limit:
            intersection = intersection.head(args.hits_limit)
        age_df = run_age_pipeline(intersection, db_path, mapping_df, args.output_dir)

    write_summary(args.output_dir, sex_df, race_df, age_df, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
