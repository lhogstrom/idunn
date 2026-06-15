#!/usr/bin/env python3
"""
Gene-Stratified Association Tests for Post-Hoc Analysis

This script runs demographic association tests stratified by both cancer type
AND actionable gene, enabling more granular analysis than cancer-level testing.

IMPORTANT: Tests are only run for features that appear in the top N SHAP rankings
for each cancer type, following the same approach as analyze_shap_feature_associations.py.

For each cancer type and its top 2 genes:
1. Load patients where specific gene is actionable
2. Check SHAP rankings to determine which features to test
3. Run association tests (only for top-ranked features, skipping if n < 50):
   - Age: Mann-Whitney U test, Cohen's d effect size (if AGE in top N)
   - Sex: Fisher's exact / Chi-square test, Odds Ratio (if SEX in top N)
   - Race/Ethnicity: Chi-square test, Cramer's V (if RACE/ETHNICITY in top N)
   - Sample Type Detailed: Chi-square test, Cramer's V (if SAMPLE_TYPE_DETAILED in top N)
   - Technical: Chi-square (categorical) / Mann-Whitney U (numeric) (if in top N)
4. Apply FDR correction per feature category
5. Output results with gene identifier

Key design decisions:
- Multi-gene patients: Included in EACH actionable gene (counted multiple times)
- Gene inclusion: Top 2 genes per cancer type
- Min sample size: n >= 50 to run statistical test
- FDR correction: Per feature category (age, sex, race, technical separately)
- SHAP filtering: Only test features that rank in top N for that cancer type

Usage:
    # With exploration output from Phase 1
    python analyze_gene_stratified_associations.py --input-dir /path/to/exploration_output

    # Standalone (will run exploration automatically)
    python analyze_gene_stratified_associations.py

    # Custom output directory
    python analyze_gene_stratified_associations.py --output-dir /path/to/output

"""

import sqlite3
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Add tools to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path
from shap_utils import (
    get_default_shap_file,
    load_shap_rankings_adaptive,
    identify_top_features,
    feature_in_top_n_for_cancer,
    FEATURE_PATTERNS,
    REPO_ROOT,
    DEFAULT_TOP_N_FEATURES
)
# Multi-tier utilities are no longer needed on the publication branch;
# cancer-type groupings collapse to plain OncoTree codes after the OT-rollup
# strip. The viz that consumes external rollup-aware runs still uses
# multi_tier_utils, but this script reads the new TIER 2a output directly.

from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

# Configuration
DB_PATH = get_default_db_path()
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis_output" / "posthoc_statistics" / "gene_stratified"

# Patient data table - use only this table (has Hugo_Symbol and isOncoKbLevel1)
PATIENT_TABLE = "perPatientOncoKBFilteredForModeling"

# Default parameters
DEFAULT_MIN_N = 50
DEFAULT_TOP_N = 2


def load_top_genes_summary(input_dir: Path, top_n: int = 2) -> pd.DataFrame:
    """
    Load top genes summary from Phase 1 exploration output.

    If not found, runs the exploration to generate it.
    """
    summary_file = input_dir / f"top{top_n}_genes_summary.csv"

    if summary_file.exists():
        print(f"Loading top genes from: {summary_file}")
        return pd.read_csv(summary_file)

    # Try alternative naming
    summary_file_alt = input_dir / "top2_genes_summary.csv"
    if summary_file_alt.exists():
        print(f"Loading top genes from: {summary_file_alt}")
        return pd.read_csv(summary_file_alt)

    print(f"Top genes summary not found at {input_dir}")
    print("Running gene distribution exploration...")

    # Import and run exploration
    from explore_gene_distributions import (
        get_gene_frequencies, identify_top_n_genes, check_sample_size
    )

    freq_df = get_gene_frequencies(DB_PATH)
    top_genes_df = identify_top_n_genes(freq_df, n=top_n)
    top_genes_df = check_sample_size(top_genes_df, min_n=DEFAULT_MIN_N)

    return top_genes_df


def load_gene_patient_data(
    db_path: str,
    oncotree_code: str,
    gene: str,
) -> pd.DataFrame:
    """
    Load patients with specific actionable gene, including demographics.

    Uses only perPatientOncoKBFilteredForModeling table (has Hugo_Symbol column).

    Args:
        db_path: Path to SQLite database
        oncotree_code: Plain OncoTree code (e.g., "LUAD")
        gene: Hugo_Symbol to filter on

    Returns:
        DataFrame with patient demographics for gene-specific actionable patients
    """
    conn = sqlite3.connect(db_path)

    oncotree_codes = [oncotree_code]
    placeholders = ','.join(['?' for _ in oncotree_codes])
    query = f"""
    SELECT DISTINCT
        PATIENT_ID,
        AGE_AT_SEQ_REPORT,
        SEX,
        RACE,
        ETHNICITY,
        SAMPLE_TYPE,
        SAMPLE_TYPE_DETAILED,
        SUBJECT_CENTER,
        number_of_genes,
        preservation_technique,
        calling_strategy
    FROM {PATIENT_TABLE}
    WHERE ONCOTREE_CODE IN ({placeholders})
      AND Hugo_Symbol = ?
      AND isOncoKbLevel1 = 1
    """
    params = oncotree_codes + [gene]
    df = pd.read_sql_query(query, conn, params=params)

    conn.close()

    # Dedup by PATIENT_ID to mirror load_all_cancer_patients (one row per patient).
    # Without this, patients with multiple sequenced samples get counted multiple
    # times in the gene-actionable numerator while the non-actionable denominator
    # is collapsed to one row per patient — an asymmetric count that inflates Ns.
    df = df.drop_duplicates(subset=['PATIENT_ID'], keep='first')

    # DB column is RACE; rename to ANCESTRY at the Python boundary (Feature
    # Review Fix b, 2026-05-21). Matches data_loader.py's rename.
    df = df.rename(columns={'RACE': 'ANCESTRY'})

    return df


def load_all_cancer_patients(
    db_path: str,
    oncotree_code: str,
) -> pd.DataFrame:
    """
    Load all patients for a cancer type (for comparison group).

    Args:
        db_path: Path to SQLite database
        oncotree_code: Plain OncoTree code (e.g., "LUAD")
    """
    conn = sqlite3.connect(db_path)

    oncotree_codes = [oncotree_code]
    placeholders = ','.join(['?' for _ in oncotree_codes])
    query = f"""
    SELECT DISTINCT
        PATIENT_ID,
        AGE_AT_SEQ_REPORT,
        SEX,
        RACE,
        ETHNICITY,
        SAMPLE_TYPE,
        SAMPLE_TYPE_DETAILED,
        SUBJECT_CENTER,
        number_of_genes,
        preservation_technique,
        calling_strategy,
        isOncoKbLevel1
    FROM {PATIENT_TABLE}
    WHERE ONCOTREE_CODE IN ({placeholders})
    """
    df = pd.read_sql_query(query, conn, params=oncotree_codes)

    conn.close()

    # Deduplicate by PATIENT_ID, keeping non-actionable status for comparison
    # (a patient should only appear once in the comparison group)
    df = df.drop_duplicates(subset=['PATIENT_ID'], keep='first')

    # DB column is RACE; rename to ANCESTRY at the Python boundary (Feature
    # Review Fix b, 2026-05-21). Matches data_loader.py's rename.
    df = df.rename(columns={'RACE': 'ANCESTRY'})

    return df


def run_gene_stratified_age_test(
    gene_patients: pd.DataFrame,
    all_cancer_patients: pd.DataFrame
) -> dict:
    """
    Compare age of gene-specific actionable patients vs non-actionable patients.

    Uses Mann-Whitney U test (non-parametric) and calculates Cohen's d effect size.
    """
    # Actionable group: patients with this specific gene
    actionable_ages = gene_patients['AGE_AT_SEQ_REPORT'].dropna()

    # Non-actionable group: patients of same cancer type without ANY actionable event
    non_actionable = all_cancer_patients[all_cancer_patients['isOncoKbLevel1'] == 0]
    non_actionable_ages = non_actionable['AGE_AT_SEQ_REPORT'].dropna()

    n_actionable = len(actionable_ages)
    n_non_actionable = len(non_actionable_ages)

    result = {
        'n_actionable': n_actionable,
        'n_non_actionable': n_non_actionable,
        'actionable_mean_age': np.nan,
        'actionable_median_age': np.nan,
        'actionable_std_age': np.nan,
        'non_actionable_mean_age': np.nan,
        'non_actionable_median_age': np.nan,
        'non_actionable_std_age': np.nan,
        'mean_age_difference': np.nan,
        'median_age_difference': np.nan,
        'cohens_d': np.nan,
        'rank_biserial_r': np.nan,
        'mann_whitney_u': np.nan,
        'mann_whitney_p': np.nan,
        'warning': None
    }

    if n_actionable < 5 or n_non_actionable < 5:
        result['warning'] = f'Insufficient data (n_act={n_actionable}, n_non={n_non_actionable})'
        return result

    # Calculate statistics
    result['actionable_mean_age'] = actionable_ages.mean()
    result['actionable_median_age'] = actionable_ages.median()
    result['actionable_std_age'] = actionable_ages.std()
    result['non_actionable_mean_age'] = non_actionable_ages.mean()
    result['non_actionable_median_age'] = non_actionable_ages.median()
    result['non_actionable_std_age'] = non_actionable_ages.std()

    result['mean_age_difference'] = result['actionable_mean_age'] - result['non_actionable_mean_age']
    result['median_age_difference'] = result['actionable_median_age'] - result['non_actionable_median_age']

    # Cohen's d effect size
    pooled_std = np.sqrt(
        ((n_actionable - 1) * result['actionable_std_age']**2 +
         (n_non_actionable - 1) * result['non_actionable_std_age']**2) /
        (n_actionable + n_non_actionable - 2)
    )
    if pooled_std > 0:
        result['cohens_d'] = result['mean_age_difference'] / pooled_std
    else:
        result['cohens_d'] = np.nan

    # Mann-Whitney U test
    try:
        u_stat, p_value = stats.mannwhitneyu(
            actionable_ages, non_actionable_ages, alternative='two-sided'
        )
        result['mann_whitney_u'] = u_stat
        result['mann_whitney_p'] = p_value
        # Rank-biserial correlation (natural effect size for Mann-Whitney U)
        result['rank_biserial_r'] = 1 - (2 * u_stat) / (n_actionable * n_non_actionable)
    except Exception as e:
        result['warning'] = f'Mann-Whitney test failed: {e}'

    return result


def run_gene_stratified_sex_test(
    gene_patients: pd.DataFrame,
    all_cancer_patients: pd.DataFrame
) -> dict:
    """
    Test sex association for gene-specific actionable patients.

    Compares proportion of females among gene-actionable vs non-actionable patients.
    """
    # Gene-actionable patients
    gene_data = gene_patients[gene_patients['SEX'].isin(['Female', 'Male'])].copy()

    # Non-actionable patients (same cancer type)
    non_actionable = all_cancer_patients[
        (all_cancer_patients['isOncoKbLevel1'] == 0) &
        (all_cancer_patients['SEX'].isin(['Female', 'Male']))
    ].copy()

    result = {
        'n_gene_actionable': len(gene_data),
        'n_non_actionable': len(non_actionable),
        'gene_female': 0,
        'gene_male': 0,
        'gene_female_rate': np.nan,
        'non_actionable_female': 0,
        'non_actionable_male': 0,
        'non_actionable_female_rate': np.nan,
        'odds_ratio': np.nan,
        'or_ci_lower': np.nan,
        'or_ci_upper': np.nan,
        'p_value': np.nan,
        'test_used': 'none',
        'warning': None
    }

    if len(gene_data) < 5 or len(non_actionable) < 5:
        result['warning'] = f'Insufficient data'
        return result

    # Count by sex
    gene_female = (gene_data['SEX'] == 'Female').sum()
    gene_male = (gene_data['SEX'] == 'Male').sum()
    non_act_female = (non_actionable['SEX'] == 'Female').sum()
    non_act_male = (non_actionable['SEX'] == 'Male').sum()

    result['gene_female'] = gene_female
    result['gene_male'] = gene_male
    result['non_actionable_female'] = non_act_female
    result['non_actionable_male'] = non_act_male

    if gene_female + gene_male > 0:
        result['gene_female_rate'] = gene_female / (gene_female + gene_male)
    if non_act_female + non_act_male > 0:
        result['non_actionable_female_rate'] = non_act_female / (non_act_female + non_act_male)

    # Create 2x2 contingency table
    # Rows: gene-actionable vs non-actionable
    # Cols: female vs male
    table = [
        [gene_female, gene_male],
        [non_act_female, non_act_male]
    ]

    # Odds ratio (with Haldane correction for zero cells)
    cells = [gene_female, gene_male, non_act_female, non_act_male]
    if any(c == 0 for c in cells):
        # Haldane correction: add 0.5 to all cells for OR and CI calculation
        ci_gf, ci_gm, ci_nf, ci_nm = [c + 0.5 for c in cells]
        result['odds_ratio'] = (ci_gf * ci_nm) / (ci_gm * ci_nf)
        result['warning'] = 'Zero cell corrected (Haldane +0.5)'
    else:
        ci_gf, ci_gm, ci_nf, ci_nm = cells
        result['odds_ratio'] = (gene_female * non_act_male) / (gene_male * non_act_female)

    # Confidence interval for OR (Woolf's method on corrected cells)
    log_or = np.log(result['odds_ratio'])
    se_log_or = np.sqrt(1/ci_gf + 1/ci_gm + 1/ci_nf + 1/ci_nm)
    result['or_ci_lower'] = np.exp(log_or - 1.96 * se_log_or)
    result['or_ci_upper'] = np.exp(log_or + 1.96 * se_log_or)

    # Compute expected frequencies for Fisher/chi-square decision
    grand_total = sum(cells)
    row1_total = gene_female + gene_male
    row2_total = non_act_female + non_act_male
    col1_total = gene_female + non_act_female
    col2_total = gene_male + non_act_male

    expected_cells = [
        (row1_total * col1_total) / grand_total,
        (row1_total * col2_total) / grand_total,
        (row2_total * col1_total) / grand_total,
        (row2_total * col2_total) / grand_total
    ]
    min_expected = min(expected_cells)

    if min_expected < 5:
        # Fisher's exact test for small expected frequencies
        try:
            _, p_value = fisher_exact(table)
            result['p_value'] = p_value
            result['test_used'] = 'fisher_exact'
        except Exception as e:
            result['warning'] = f'Fisher test failed: {e}'
    else:
        # Chi-square test
        try:
            chi2, p_value, _, _ = chi2_contingency(table)
            result['p_value'] = p_value
            result['test_used'] = 'chi_square'
        except Exception as e:
            result['warning'] = f'Chi-square test failed: {e}'

    return result


def run_gene_stratified_categorical_test(
    gene_patients: pd.DataFrame,
    all_cancer_patients: pd.DataFrame,
    feature: str
) -> dict:
    """
    Chi-square test for categorical feature association.

    Compares distribution of feature among gene-actionable vs non-actionable patients.
    """
    # Clean data
    gene_data = gene_patients[gene_patients[feature].notna()].copy()
    non_actionable = all_cancer_patients[
        (all_cancer_patients['isOncoKbLevel1'] == 0) &
        (all_cancer_patients[feature].notna())
    ].copy()

    result = {
        'n_gene_actionable': len(gene_data),
        'n_non_actionable': len(non_actionable),
        'chi2': np.nan,
        'p_value': np.nan,
        'cramers_v': np.nan,
        'dof': np.nan,
        'warning': None
    }

    if len(gene_data) < 10 or len(non_actionable) < 10:
        result['warning'] = f'Insufficient data'
        return result

    # Add group label
    gene_data['group'] = 'gene_actionable'
    non_actionable['group'] = 'non_actionable'

    combined = pd.concat([
        gene_data[['group', feature]],
        non_actionable[['group', feature]]
    ])

    # Create contingency table
    contingency = pd.crosstab(combined['group'], combined[feature])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        result['warning'] = 'Insufficient categories'
        return result

    # Chi-square test
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        result['chi2'] = chi2
        result['p_value'] = p_value
        result['dof'] = dof

        # Check expected frequency assumption
        min_expected_freq = expected.min()
        pct_below_5 = (expected < 5).sum() / expected.size * 100
        result['min_expected_freq'] = min_expected_freq
        if min_expected_freq < 1:
            result['warning'] = f'Chi-square unreliable: {pct_below_5:.0f}% expected freqs < 5 (min={min_expected_freq:.1f})'
        elif min_expected_freq < 5:
            result['warning'] = f'Chi-square may be unreliable: {pct_below_5:.0f}% expected freqs < 5 (min={min_expected_freq:.1f})'

        # Cramer's V effect size
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        if min_dim > 0:
            result['cramers_v'] = np.sqrt(chi2 / (n * min_dim))

    except Exception as e:
        result['warning'] = f'Chi-square test failed: {e}'

    return result


def run_gene_stratified_numeric_test(
    gene_patients: pd.DataFrame,
    all_cancer_patients: pd.DataFrame,
    feature: str
) -> dict:
    """
    Mann-Whitney U test for numeric feature association.
    """
    gene_data = gene_patients[gene_patients[feature].notna()][feature]
    non_actionable = all_cancer_patients[
        (all_cancer_patients['isOncoKbLevel1'] == 0) &
        (all_cancer_patients[feature].notna())
    ][feature]

    result = {
        'n_gene_actionable': len(gene_data),
        'n_non_actionable': len(non_actionable),
        'gene_mean': np.nan,
        'non_actionable_mean': np.nan,
        'mann_whitney_u': np.nan,
        'p_value': np.nan,
        'effect_size_r': np.nan,
        'warning': None
    }

    if len(gene_data) < 5 or len(non_actionable) < 5:
        result['warning'] = 'Insufficient data'
        return result

    result['gene_mean'] = gene_data.mean()
    result['non_actionable_mean'] = non_actionable.mean()

    try:
        u_stat, p_value = stats.mannwhitneyu(gene_data, non_actionable, alternative='two-sided')
        result['mann_whitney_u'] = u_stat
        result['p_value'] = p_value

        # Effect size: rank-biserial correlation
        n1, n2 = len(gene_data), len(non_actionable)
        result['effect_size_r'] = abs(1 - (2 * u_stat) / (n1 * n2))

    except Exception as e:
        result['warning'] = f'Mann-Whitney test failed: {e}'

    return result


def apply_fdr_correction_to_results(results_df: pd.DataFrame, p_column: str = 'p_value') -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to results."""

    df = results_df.copy()

    valid_mask = ~df[p_column].isna()
    n_valid = valid_mask.sum()

    if n_valid == 0:
        df['p_fdr'] = np.nan
        df['significant_fdr'] = False
        return df

    _, fdr_pvals, _, _ = multipletests(
        df.loc[valid_mask, p_column],
        method='fdr_bh',
        alpha=0.05
    )

    df['p_fdr'] = np.nan
    df.loc[valid_mask, 'p_fdr'] = fdr_pvals
    df['significant_fdr'] = df['p_fdr'] < 0.05

    return df


def analyze_gene_stratified_associations(
    top_genes_df: pd.DataFrame,
    db_path: str,
    shap_df: pd.DataFrame = None,
    min_n: int = DEFAULT_MIN_N,
    top_n_shap: int = DEFAULT_TOP_N_FEATURES,
) -> dict:
    """
    Run gene-stratified association tests, filtered by SHAP rankings.

    Only tests features that appear in top N SHAP rankings for each cancer type.

    Args:
        top_genes_df: DataFrame with top genes per cancer type
        db_path: Path to SQLite database
        shap_df: DataFrame with SHAP rankings (optional, will auto-load if None)
        min_n: Minimum sample size for testing
        top_n_shap: Number of top SHAP features to consider

    Returns:
        dict of DataFrames for each test category.
    """
    results = {
        'age': [],
        'sex': [],
        'race': [],
        'ethnicity': [],
        'sample_type': [],
        'technical': []
    }

    # Filter to genes meeting sample size requirement
    valid_genes = top_genes_df[top_genes_df['meets_min_n']].copy()

    if len(valid_genes) == 0:
        print("WARNING: No gene-cancer combinations meet minimum sample size requirement")
        return {k: pd.DataFrame() for k in results.keys()}

    print(f"\nAnalyzing {len(valid_genes)} gene-cancer combinations...")
    print(f"Testing features ranked in top {top_n_shap} by SHAP importance\n")

    for idx, row in valid_genes.iterrows():
        oncotree_code = row['ONCOTREE_CODE']
        gene = row['Hugo_Symbol']
        n_patients = row['n_patients']

        print(f"\n{oncotree_code} - {gene} (n={n_patients})...")

        gene_patients = load_gene_patient_data(db_path, oncotree_code, gene)
        all_patients = load_all_cancer_patients(db_path, oncotree_code)

        if len(gene_patients) < min_n:
            print(f"  Skipping: only {len(gene_patients)} gene-specific patients loaded")
            continue

        base_info = {
            'ONCOTREE_CODE': oncotree_code,
            'Hugo_Symbol': gene,
            'gene_rank': int(row['rank']),
            'n_gene_patients': len(gene_patients)
        }

        shap_lookup_code = oncotree_code

        # Check SHAP rankings to determine which features to test
        # Age test (if AGE in top N)
        if shap_df is None or feature_in_top_n_for_cancer(shap_df, shap_lookup_code, 'AGE', top_n_shap):
            print(f"  Testing age (in top {top_n_shap} SHAP)...")
            age_result = run_gene_stratified_age_test(gene_patients, all_patients)
            results['age'].append({**base_info, **age_result})
        else:
            print(f"  Skipping age (not in top {top_n_shap} SHAP)")

        # Sex test (if SEX in top N)
        if shap_df is None or feature_in_top_n_for_cancer(shap_df, shap_lookup_code, 'SEX', top_n_shap):
            print(f"  Testing sex (in top {top_n_shap} SHAP)...")
            sex_result = run_gene_stratified_sex_test(gene_patients, all_patients)
            results['sex'].append({**base_info, **sex_result})
        else:
            print(f"  Skipping sex (not in top {top_n_shap} SHAP)")

        # Ancestry test (if ANCESTRY in top N). DB column was RACE; the loader
        # functions above rename it to ANCESTRY at the Python boundary.
        if shap_df is None or feature_in_top_n_for_cancer(shap_df, shap_lookup_code, 'ANCESTRY', top_n_shap):
            print(f"  Testing ancestry (in top {top_n_shap} SHAP)...")
            race_result = run_gene_stratified_categorical_test(gene_patients, all_patients, 'ANCESTRY')
            race_result['feature'] = 'ANCESTRY'
            results['race'].append({**base_info, **race_result})
        else:
            print(f"  Skipping ancestry (not in top {top_n_shap} SHAP)")

        # Ethnicity test (if ETHNICITY in top N)
        if shap_df is None or feature_in_top_n_for_cancer(shap_df, shap_lookup_code, 'ETHNICITY', top_n_shap):
            print(f"  Testing ethnicity (in top {top_n_shap} SHAP)...")
            eth_result = run_gene_stratified_categorical_test(gene_patients, all_patients, 'ETHNICITY')
            eth_result['feature'] = 'ETHNICITY'
            results['ethnicity'].append({**base_info, **eth_result})
        else:
            print(f"  Skipping ethnicity (not in top {top_n_shap} SHAP)")

        # Sample-type-detailed test (if SAMPLE_TYPE_DETAILED in top N).
        # Plain SAMPLE_TYPE was dropped from the modeling feature set 2026-05-21
        # (Feature Review Fix a) — only the DETAILED variant is now ranked.
        if shap_df is None or feature_in_top_n_for_cancer(shap_df, shap_lookup_code, 'SAMPLE_TYPE_DETAILED', top_n_shap):
            print(f"  Testing sample type detailed (in top {top_n_shap} SHAP)...")
            sample_result = run_gene_stratified_categorical_test(gene_patients, all_patients, 'SAMPLE_TYPE_DETAILED')
            sample_result['feature'] = 'SAMPLE_TYPE_DETAILED'
            results['sample_type'].append({**base_info, **sample_result})
        else:
            print(f"  Skipping sample type detailed (not in top {top_n_shap} SHAP)")

        # Technical features (if TECHNICAL in top N)
        if shap_df is None or feature_in_top_n_for_cancer(shap_df, shap_lookup_code, 'TECHNICAL', top_n_shap):
            print(f"  Testing technical features (in top {top_n_shap} SHAP)...")

            # Number of genes (numeric)
            nog_result = run_gene_stratified_numeric_test(gene_patients, all_patients, 'number_of_genes')
            nog_result['feature'] = 'number_of_genes'
            results['technical'].append({**base_info, **nog_result})

            # Calling strategy (categorical)
            cs_result = run_gene_stratified_categorical_test(gene_patients, all_patients, 'calling_strategy')
            cs_result['feature'] = 'calling_strategy'
            results['technical'].append({**base_info, **cs_result})

            # Subject center (categorical)
            sc_result = run_gene_stratified_categorical_test(gene_patients, all_patients, 'SUBJECT_CENTER')
            sc_result['feature'] = 'SUBJECT_CENTER'
            results['technical'].append({**base_info, **sc_result})
        else:
            print(f"  Skipping technical features (not in top {top_n_shap} SHAP)")

    # Convert to DataFrames and apply FDR correction
    result_dfs = {}

    for category, result_list in results.items():
        if len(result_list) == 0:
            result_dfs[category] = pd.DataFrame()
            continue

        df = pd.DataFrame(result_list)

        # Determine p-value column based on category
        if category == 'age':
            p_col = 'mann_whitney_p'
        else:
            p_col = 'p_value'

        if p_col in df.columns:
            df = apply_fdr_correction_to_results(df, p_col)

        result_dfs[category] = df

    return result_dfs


def generate_association_report(
    results: dict,
    output_file: Path,
    min_n: int,
    top_n_shap: int,
    shap_filter_active: bool = True,
    shap_file_path: str = None
):
    """Generate human-readable summary report.

    Args:
        shap_filter_active: True if the top-N SHAP gate was applied at
            runtime; False if --no-shap-filter was set (all features tested).
        shap_file_path: Path of the SHAP/LR rankings file used as the gate
            source (only meaningful when shap_filter_active=True).
    """

    if shap_filter_active:
        cap_desc = (
            f"top {top_n_shap} features per cancer type"
            if top_n_shap is not None
            else "all gate-significant features per cancer type (top-N cap disabled)"
        )
        header_line = f"(Importance-Filtered: {cap_desc}, source: {shap_file_path})"
    else:
        header_line = "(NO SHAP FILTER: --no-shap-filter set, all features tested)"

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Gene-Stratified Association Analysis Report\n")
        f.write(header_line + "\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Minimum sample size: {min_n}\n")
        if not shap_filter_active:
            topn_str = 'DISABLED (--no-shap-filter)'
        elif top_n_shap is None:
            topn_str = 'none (cap disabled; all gate-significant features tested)'
        else:
            topn_str = str(top_n_shap)
        f.write(f"SHAP top N filter: {topn_str}\n")

        for category, df in results.items():
            if len(df) == 0:
                continue

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{category.upper()} ASSOCIATIONS\n")
            f.write("=" * 80 + "\n")

            f.write(f"\nTests performed: {len(df)}\n")

            if 'significant_fdr' in df.columns:
                n_sig = df['significant_fdr'].sum()
                f.write(f"Significant (FDR < 0.05): {n_sig}\n")

                if n_sig > 0:
                    sig_df = df[df['significant_fdr']].sort_values('p_fdr')

                    f.write("\nSignificant associations:\n")
                    f.write("-" * 80 + "\n")

                    for _, row in sig_df.iterrows():
                        f.write(f"\n{row['ONCOTREE_CODE']} - {row['Hugo_Symbol']} (rank {row['gene_rank']}):\n")

                        if category == 'age':
                            f.write(f"  Mean age diff: {row['mean_age_difference']:.1f} years\n")
                            f.write(f"  Cohen's d: {row['cohens_d']:.3f}\n")
                            f.write(f"  p-value: {row['mann_whitney_p']:.4e}\n")
                            f.write(f"  FDR: {row['p_fdr']:.4e}\n")

                        elif category == 'sex':
                            f.write(f"  Gene-actionable female rate: {row['gene_female_rate']:.1%}\n")
                            f.write(f"  Non-actionable female rate: {row['non_actionable_female_rate']:.1%}\n")
                            f.write(f"  Odds ratio: {row['odds_ratio']:.3f}\n")
                            f.write(f"  p-value: {row['p_value']:.4e}\n")
                            f.write(f"  FDR: {row['p_fdr']:.4e}\n")

                        else:
                            if 'cramers_v' in row and not pd.isna(row.get('cramers_v')):
                                f.write(f"  Cramer's V: {row['cramers_v']:.3f}\n")
                            f.write(f"  p-value: {row['p_value']:.4e}\n")
                            f.write(f"  FDR: {row['p_fdr']:.4e}\n")

        # Summary table
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY BY GENE-CANCER COMBINATION\n")
        f.write("=" * 80 + "\n\n")

        # Aggregate significant associations
        all_sig = []
        for category, df in results.items():
            if len(df) > 0 and 'significant_fdr' in df.columns:
                sig = df[df['significant_fdr']].copy()
                sig['category'] = category
                all_sig.append(sig[['ONCOTREE_CODE', 'Hugo_Symbol', 'category']])

        if all_sig:
            combined = pd.concat(all_sig)
            summary = combined.groupby(['ONCOTREE_CODE', 'Hugo_Symbol'])['category'].apply(list).reset_index()
            summary['n_significant'] = summary['category'].apply(len)
            summary = summary.sort_values('n_significant', ascending=False)

            for _, row in summary.iterrows():
                f.write(f"{row['ONCOTREE_CODE']} - {row['Hugo_Symbol']}: {row['n_significant']} significant\n")
                f.write(f"  Categories: {', '.join(row['category'])}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"\nReport saved to: {output_file}")


def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description='Run gene-stratified association tests (SHAP-filtered)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (will auto-run exploration if needed, auto-detect SHAP file)
    python analyze_gene_stratified_associations.py

    # With exploration output from Phase 1
    python analyze_gene_stratified_associations.py --input-dir /path/to/exploration

    # Custom output directory
    python analyze_gene_stratified_associations.py --output-dir /path/to/output

    # Test all features (ignore SHAP rankings)
    python analyze_gene_stratified_associations.py --no-shap-filter
        """
    )

    parser.add_argument('--input-dir', type=str, default=None,
                       help='Input directory with top genes summary from Phase 1')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--shap-file', type=str, default=None,
                       help='Path to SHAP feature rankings CSV')
    parser.add_argument('--min-n', type=int, default=DEFAULT_MIN_N,
                       help=f'Minimum sample size (default: {DEFAULT_MIN_N})')
    parser.add_argument('--top-n', type=int, default=DEFAULT_TOP_N,
                       help=f'Number of top genes per cancer (default: {DEFAULT_TOP_N})')
    parser.add_argument('--top-n-shap', type=int, default=None,
                       help=f'Number of top features to gate on by importance '
                            f'(default: {DEFAULT_TOP_N_FEATURES}). Ignored under '
                            f'--lr-significant-only, where Wald-FDR significance is the gate and '
                            f'all significant features are tested; pass an explicit value to '
                            f're-impose a cap there.')
    parser.add_argument('--no-shap-filter', action='store_true',
                       help='Test all features regardless of SHAP ranking')
    parser.add_argument('--groupings-file', type=str, default=None,
                       help='Path to cancer_type_group_definitions.csv '
                            '(auto-detected from runs/ if omitted; restricts analysis '
                            'to the canonical eligible OncoTree codes)')
    parser.add_argument('--lr-significant-only', action='store_true',
                       help='When using LR rankings, restrict to features with Wald-significant '
                            'coefficients (FDR < 0.05). This significance test IS the gate: the '
                            'top-N importance cap is disabled so all Wald-significant features are '
                            'tested (unless --top-n-shap is passed explicitly).')
    # Legacy flags accepted but ignored after the OT-rollup strip
    parser.add_argument('--tier', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--no-tier-filter', action='store_true', help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Set directories
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = REPO_ROOT / "analysis_output" / "posthoc_statistics" / "gene_stratified" / "gene_distributions"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the importance-gate cap. An explicit --top-n-shap always wins.
    # Otherwise, under --lr-significant-only the Wald-FDR significance test is the
    # principled gate and the top-N cap is a redundant second filter that drops
    # already-significant features (see
    # code_review_and_feedback/fig4b_onehot_collapse_and_tier7_gate_2026-06-15.md §3),
    # so it is disabled (None); the RF/default path keeps the top-5 cap.
    if args.top_n_shap is not None:
        top_n_shap = args.top_n_shap
    elif args.lr_significant_only:
        top_n_shap = None
    else:
        top_n_shap = DEFAULT_TOP_N_FEATURES

    print("=" * 80)
    print("Gene-Stratified Association Analysis (SHAP-Filtered)")
    print("=" * 80)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Minimum sample size: {args.min_n}")
    print(f"Top N genes: {args.top_n}")
    print("Top N SHAP features: "
          + ("ALL gate-significant (top-N cap disabled via --lr-significant-only)"
             if top_n_shap is None else str(top_n_shap)))
    print(f"SHAP filter enabled: {not args.no_shap_filter}")

    # Load SHAP rankings — fail-closed when --shap-file is passed but missing.
    # Only --no-shap-filter is allowed to bypass the gate.
    shap_df = None
    shap_file = None
    if args.no_shap_filter:
        print("\nNOTE: --no-shap-filter set — gate is disabled, all features tested")
    else:
        if args.shap_file:
            shap_file = args.shap_file
            print(f"\nUsing specified SHAP file: {shap_file}")
            if not Path(shap_file).exists():
                print(f"\nERROR: SHAP file not found: {shap_file}")
                print("  Pass --no-shap-filter to deliberately disable the gate, "
                      "or fix the path / re-run TIER 6a.")
                sys.exit(1)
        else:
            try:
                shap_file = get_default_shap_file()
            except FileNotFoundError as e:
                print(f"\nERROR: {e}")
                sys.exit(1)
        shap_df = load_shap_rankings_adaptive(shap_file, lr_significant_only=args.lr_significant_only)

    # Load top genes summary
    top_genes_df = load_top_genes_summary(input_dir, top_n=args.top_n)

    # Filter to canonical cancer types from TIER 2a (if a definitions file is reachable)
    groupings_file = args.groupings_file
    if groupings_file is None:
        candidates = sorted(
            (REPO_ROOT / 'runs').glob(
                'run_*/tier2_data_setup/cancer_type_groupings/cancer_type_group_definitions.csv'
            ),
            reverse=True,
        )
        groupings_file = str(candidates[0]) if candidates else None

    if groupings_file and Path(groupings_file).exists():
        canonical_codes = set(pd.read_csv(groupings_file)['ONCOTREE_CODE'].tolist())
        original_count = top_genes_df['ONCOTREE_CODE'].nunique()
        top_genes_df = top_genes_df[top_genes_df['ONCOTREE_CODE'].isin(canonical_codes)]
        filtered_count = top_genes_df['ONCOTREE_CODE'].nunique()
        print(f"\nFiltered to canonical cancer types from: {groupings_file}")
        print(f"  Original: {original_count} cancer types")
        print(f"  After filter: {filtered_count} cancer types")
    else:
        print("\nNo cancer_type_group_definitions.csv found; testing all cancer types in input")

    results = analyze_gene_stratified_associations(
        top_genes_df, DB_PATH, shap_df=shap_df,
        min_n=args.min_n, top_n_shap=top_n_shap,
    )

    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)

    for category, df in results.items():
        if len(df) > 0:
            # Create subdirectory for category
            category_dir = output_dir / f"{category}_by_gene"
            category_dir.mkdir(parents=True, exist_ok=True)

            # Save full results
            output_file = category_dir / f"{category}_gene_results.csv"
            df.to_csv(output_file, index=False)
            print(f"  {category}: {output_file}")

            # Save significant results
            if 'significant_fdr' in df.columns:
                sig_df = df[df['significant_fdr']]
                if len(sig_df) > 0:
                    sig_file = category_dir / f"{category}_gene_significant.csv"
                    sig_df.to_csv(sig_file, index=False)
                    print(f"  {category} (significant): {sig_file}")

    # Generate report
    report_file = output_dir / "gene_stratified_report.txt"
    generate_association_report(
        results, report_file, args.min_n, top_n_shap,
        shap_filter_active=(shap_df is not None),
        shap_file_path=shap_file
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for category, df in results.items():
        if len(df) > 0 and 'significant_fdr' in df.columns:
            n_sig = df['significant_fdr'].sum()
            print(f"  {category}: {len(df)} tests, {n_sig} significant (FDR < 0.05)")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"Next step: Run visualize_gene_stratified_results.py")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
