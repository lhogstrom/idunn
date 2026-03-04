#!/usr/bin/env python3
"""
Raw Data Plotting Utilities

Shared visualization functions for creating raw data plots that support both:
1. Cancer-level analysis (no gene stratification)
2. Gene-stratified analysis (specific gene within cancer type)

These functions are used by:
- visualize_raw_data.py (cancer-level post-hoc results)
- visualize_gene_stratified_results.py (gene-stratified post-hoc results)

Key Features:
- Optional `gene` parameter for gene-stratified filtering
- Wilson score confidence intervals for binomial proportions
- Violin/box plots for age distributions
- Contingency tables for categorical features
- Consistent styling across all visualizations
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from pathlib import Path
from scipy import stats as sp_stats
from typing import List, Optional, Tuple, Dict, Any

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Repository root and default paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Default table name
PATIENT_TABLE = "perPatientOncoKBFilteredForModeling"
GENE_TABLE = "perPatientLevel1"


def get_default_db_path() -> Optional[str]:
    """Get default database path from db_config."""
    try:
        from db_config import get_default_db_path as db_config_get_path
        return db_config_get_path()
    except ImportError:
        # Fallback if db_config not in path
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from db_config import get_default_db_path as db_config_get_path
        return db_config_get_path()


def resolve_oncotree_codes(cancer_type: str) -> List[str]:
    """
    Resolve a cancer type identifier to plain OncoTree codes for database queries.

    Handles both plain codes ('LUAD') and tier-prefixed codes ('tier1_LUAD',
    'tier3_LUNG'). For tier2/tier3 groups, returns all constituent OncoTree
    codes (e.g., 'tier3_LUNG' -> ['LUAD', 'NSCLC']).

    Parameters
    ----------
    cancer_type : str
        Plain OncoTree code or tier-prefixed code

    Returns
    -------
    List[str]
        List of plain OncoTree codes for use in SQL queries
    """
    if re.match(r'^tier\d+_', cancer_type):
        try:
            from multi_tier_utils import get_oncotree_codes_for_group
            codes = get_oncotree_codes_for_group(cancer_type)
            if codes:
                return codes
        except (ImportError, Exception):
            pass
        # Fallback: strip the tier prefix (works for tier1)
        return [re.sub(r'^tier\d+_', '', cancer_type)]
    return [cancer_type]


def _oncotree_sql_clause(cancer_type: str, table_alias: str = '') -> str:
    """
    Build a SQL WHERE clause fragment for OncoTree code filtering.

    Parameters
    ----------
    cancer_type : str
        Plain or tier-prefixed cancer type code
    table_alias : str
        Table alias prefix (e.g., 'p.' or ''). Include the trailing dot.

    Returns
    -------
    str
        SQL clause like "ONCOTREE_CODE = 'LUAD'" or
        "ONCOTREE_CODE IN ('LUAD', 'NSCLC')"
    """
    codes = resolve_oncotree_codes(cancer_type)
    col = f"{table_alias}ONCOTREE_CODE"
    if len(codes) == 1:
        return f"{col} = '{codes[0]}'"
    else:
        quoted = ', '.join(f"'{c}'" for c in codes)
        return f"{col} IN ({quoted})"


def wilson_ci(n_success: int, n_total: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for binomial proportion.

    Parameters
    ----------
    n_success : int
        Number of successes (e.g., actionable patients)
    n_total : int
        Total number of trials
    alpha : float
        Significance level (default 0.05 for 95% CI)

    Returns
    -------
    Tuple[float, float]
        Lower and upper bounds of confidence interval
    """
    if n_total == 0:
        return 0.0, 0.0

    p = n_success / n_total
    z = sp_stats.norm.ppf(1 - alpha / 2)

    denominator = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denominator

    ci_lower = max(0, centre - margin)
    ci_upper = min(1, centre + margin)

    return ci_lower, ci_upper


def load_age_data(
    cancer_type: str,
    gene: Optional[str] = None,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load age data for a specific cancer type, optionally filtered by gene.

    Parameters
    ----------
    cancer_type : str
        OncoTree cancer type code (e.g., 'LUAD')
    gene : str, optional
        Hugo symbol to filter to patients with this specific actionable gene.
        If None, returns all patients of the cancer type.
    db_path : str, optional
        Path to SQLite database

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: age, actionable, actionable_label

    Notes
    -----
    When gene is provided:
    - Actionable group: patients with the specific gene actionable
    - Non-actionable group: ALL non-actionable patients of the cancer type
      (not just those without any actionable variant)
    """
    db_path = db_path or get_default_db_path()
    conn = sqlite3.connect(db_path)

    # Resolve tier-prefixed codes to plain OncoTree codes
    oncotree_clause = _oncotree_sql_clause(cancer_type)
    oncotree_clause_p = _oncotree_sql_clause(cancer_type, table_alias='p.')

    if gene is None:
        # Cancer-level: all patients of cancer type
        query = f"""
        SELECT
            AGE_AT_SEQ_REPORT as age,
            isOncoKbLevel1 as actionable
        FROM {PATIENT_TABLE}
        WHERE {oncotree_clause}
        AND AGE_AT_SEQ_REPORT IS NOT NULL
        """
    else:
        # Gene-stratified: patients with specific gene actionable vs non-actionable
        # Non-actionable = all non-actionable patients of cancer type
        oncotree_clause_g = _oncotree_sql_clause(cancer_type, table_alias='')
        query = f"""
        SELECT
            p.AGE_AT_SEQ_REPORT as age,
            CASE
                WHEN g.PATIENT_ID IS NOT NULL THEN 1
                ELSE 0
            END as actionable
        FROM {PATIENT_TABLE} p
        LEFT JOIN (
            SELECT DISTINCT PATIENT_ID
            FROM {GENE_TABLE}
            WHERE {oncotree_clause_g}
            AND Hugo_Symbol = '{gene}'
            AND isOncoKbLevel1 = 1
        ) g ON p.PATIENT_ID = g.PATIENT_ID
        WHERE {oncotree_clause_p}
        AND p.AGE_AT_SEQ_REPORT IS NOT NULL
        AND (g.PATIENT_ID IS NOT NULL OR p.isOncoKbLevel1 = 0)
        """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df['actionable_label'] = df['actionable'].map({0: 'Non-Actionable', 1: 'Actionable'})
    return df


def load_categorical_data(
    cancer_type: str,
    column: str,
    gene: Optional[str] = None,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Load categorical data for a specific cancer type, optionally filtered by gene.

    Parameters
    ----------
    cancer_type : str
        OncoTree cancer type code (e.g., 'LUAD')
    column : str
        Database column name (e.g., 'SEX', 'RACE', 'SUBJECT_CENTER')
    gene : str, optional
        Hugo symbol to filter to patients with this specific actionable gene.
        If None, returns all patients of the cancer type.
    db_path : str, optional
        Path to SQLite database

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [column], actionable
    """
    db_path = db_path or get_default_db_path()
    conn = sqlite3.connect(db_path)

    # Resolve tier-prefixed codes to plain OncoTree codes
    oncotree_clause = _oncotree_sql_clause(cancer_type)
    oncotree_clause_p = _oncotree_sql_clause(cancer_type, table_alias='p.')

    if gene is None:
        # Cancer-level: all patients of cancer type
        query = f"""
        SELECT
            {column},
            isOncoKbLevel1 as actionable
        FROM {PATIENT_TABLE}
        WHERE {oncotree_clause}
        AND {column} IS NOT NULL
        """
    else:
        # Gene-stratified: patients with specific gene actionable vs non-actionable
        oncotree_clause_g = _oncotree_sql_clause(cancer_type, table_alias='')
        query = f"""
        SELECT
            p.{column},
            CASE
                WHEN g.PATIENT_ID IS NOT NULL THEN 1
                ELSE 0
            END as actionable
        FROM {PATIENT_TABLE} p
        LEFT JOIN (
            SELECT DISTINCT PATIENT_ID
            FROM {GENE_TABLE}
            WHERE {oncotree_clause_g}
            AND Hugo_Symbol = '{gene}'
            AND isOncoKbLevel1 = 1
        ) g ON p.PATIENT_ID = g.PATIENT_ID
        WHERE {oncotree_clause_p}
        AND p.{column} IS NOT NULL
        AND (g.PATIENT_ID IS NOT NULL OR p.isOncoKbLevel1 = 0)
        """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def plot_age_distribution(
    cancer_type: str,
    cohens_d: float,
    p_val: float,
    mean_diff: float,
    output_dir: Path,
    gene: Optional[str] = None,
    db_path: Optional[str] = None,
    file_prefix: str = "age"
) -> Optional[str]:
    """
    Create age distribution plots (violin + KDE) for a cancer type.

    Parameters
    ----------
    cancer_type : str
        OncoTree cancer type code
    cohens_d : float
        Cohen's d effect size
    p_val : float
        FDR-adjusted p-value
    mean_diff : float
        Mean age difference (actionable - non-actionable)
    output_dir : Path
        Output directory for saving plots
    gene : str, optional
        Hugo symbol for gene-stratified analysis
    db_path : str, optional
        Path to SQLite database
    file_prefix : str
        Prefix for output filenames (default: 'age')

    Returns
    -------
    str or None
        Filename of saved plot, or None if no data
    """
    df = load_age_data(cancer_type, gene=gene, db_path=db_path)

    if len(df) == 0:
        print(f"  Warning: No age data for {cancer_type}" + (f" ({gene})" if gene else ""))
        return None

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Title includes gene if provided
    title_suffix = f" ({gene})" if gene else ""

    # SUBPLOT 1: Violin + box plot
    ax1 = axes[0]

    non_act_ages = df[df['actionable'] == 0]['age'].values
    act_ages = df[df['actionable'] == 1]['age'].values

    if len(non_act_ages) > 0 and len(act_ages) > 0:
        # Violin plot
        parts = ax1.violinplot(
            [non_act_ages, act_ages],
            positions=[0, 1],
            showmeans=True,
            showmedians=True,
            widths=0.7
        )

        # Color the violins
        colors = ['#95a5a6', '#e74c3c']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # Box plot overlay
        ax1.boxplot(
            [non_act_ages, act_ages],
            positions=[0, 1],
            widths=0.3,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor='white', alpha=0.5),
            medianprops=dict(color='black', linewidth=2)
        )

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Non-Actionable', 'Actionable'], fontsize=11)
    ax1.set_ylabel('Age at Sequencing', fontsize=12, fontweight='bold')
    ax1.set_title(f'{cancer_type}{title_suffix}: Age Distribution by Actionability',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # Statistics annotation
    stats_text = f"Cohen's d = {cohens_d:.3f}\n"
    stats_text += f"Mean Δ = {mean_diff:.1f} years\n"
    stats_text += f"FDR p = {p_val:.2e}"
    ax1.text(0.02, 0.98, stats_text,
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # SUBPLOT 2: Histogram + KDE
    ax2 = axes[1]

    ax2.hist(non_act_ages, bins=30, alpha=0.5, color='#95a5a6',
             label='Non-Actionable', density=True)
    ax2.hist(act_ages, bins=30, alpha=0.5, color='#e74c3c',
             label='Actionable', density=True)

    # KDE overlay
    if len(non_act_ages) > 1 and len(act_ages) > 1:
        x_range = np.linspace(df['age'].min(), df['age'].max(), 200)

        kde_non = sp_stats.gaussian_kde(non_act_ages)
        ax2.plot(x_range, kde_non(x_range), color='#7f8c8d', linewidth=2,
                 linestyle='--', alpha=0.8)

        kde_act = sp_stats.gaussian_kde(act_ages)
        ax2.plot(x_range, kde_act(x_range), color='#c0392b', linewidth=2,
                 linestyle='--', alpha=0.8)

    ax2.set_xlabel('Age at Sequencing', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title(f'{cancer_type}{title_suffix}: Age Distribution Overlap',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    # Sample sizes
    ax2.text(0.98, 0.98, f"n(Non-Act) = {len(non_act_ages):,}\nn(Act) = {len(act_ages):,}",
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    # Filename includes gene if provided
    if gene:
        filename = f"{file_prefix}_{cancer_type}_{gene}_distribution"
    else:
        filename = f"{file_prefix}_{cancer_type}_distribution"

    output_dir = Path(output_dir)
    plt.savefig(output_dir / f"{filename}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}.pdf")
    plt.close()

    return f"{filename}.pdf"


def plot_sex_contingency(
    cancer_type: str,
    odds_ratio: float,
    p_val: float,
    output_dir: Path,
    gene: Optional[str] = None,
    db_path: Optional[str] = None,
    file_prefix: str = "sex"
) -> Optional[str]:
    """
    Create sex contingency plots (grouped bars + stacked percentage) for a cancer type.

    Parameters
    ----------
    cancer_type : str
        OncoTree cancer type code
    odds_ratio : float
        Odds ratio (Female/Male for actionability)
    p_val : float
        FDR-adjusted p-value
    output_dir : Path
        Output directory for saving plots
    gene : str, optional
        Hugo symbol for gene-stratified analysis
    db_path : str, optional
        Path to SQLite database
    file_prefix : str
        Prefix for output filenames (default: 'sex')

    Returns
    -------
    str or None
        Filename of saved plot, or None if no data
    """
    df = load_categorical_data(cancer_type, 'SEX', gene=gene, db_path=db_path)
    df = df[df['SEX'].isin(['Female', 'Male'])]

    if len(df) == 0:
        print(f"  Warning: No sex data for {cancer_type}" + (f" ({gene})" if gene else ""))
        return None

    # Create contingency table
    ct = pd.crosstab(df['SEX'], df['actionable'], margins=True)

    if 'Female' not in ct.index or 'Male' not in ct.index:
        print(f"  Warning: Incomplete sex data for {cancer_type}" + (f" ({gene})" if gene else ""))
        return None

    # Title includes gene if provided
    title_suffix = f" ({gene})" if gene else ""

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # SUBPLOT 1: Grouped bar chart (counts)
    ax1 = axes[0]

    sex_labels = ['Female', 'Male']
    non_actionable = [ct.loc['Female', 0], ct.loc['Male', 0]]
    actionable = [ct.loc['Female', 1], ct.loc['Male', 1]]

    x = np.arange(len(sex_labels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, non_actionable, width,
                    label='Non-Actionable', color='#95a5a6', alpha=0.8)
    bars2 = ax1.bar(x + width/2, actionable, width,
                    label='Actionable', color='#9b59b6', alpha=0.8)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    ax1.set_xlabel('Sex', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax1.set_title(f'{cancer_type}{title_suffix}: Actionability by Sex (Counts)',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sex_labels, fontsize=11)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # Statistics
    stats_text = f"OR = {odds_ratio:.3f}\nFDR p = {p_val:.2e}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # SUBPLOT 2: Stacked percentage bars
    ax2 = axes[1]

    female_total = ct.loc['Female', 'All']
    male_total = ct.loc['Male', 'All']

    female_pct = [ct.loc['Female', 0]/female_total*100, ct.loc['Female', 1]/female_total*100]
    male_pct = [ct.loc['Male', 0]/male_total*100, ct.loc['Male', 1]/male_total*100]

    x = np.arange(2)

    ax2.bar(x, [female_pct[0], male_pct[0]], label='Non-Actionable', color='#95a5a6', alpha=0.8)
    ax2.bar(x, [female_pct[1], male_pct[1]], bottom=[female_pct[0], male_pct[0]],
            label='Actionable', color='#9b59b6', alpha=0.8)

    # Percentage labels
    ax2.text(0, female_pct[0]/2, f'{female_pct[0]:.1f}%', ha='center', va='center',
             fontsize=11, fontweight='bold', color='white')
    ax2.text(0, female_pct[0] + female_pct[1]/2, f'{female_pct[1]:.1f}%', ha='center',
             va='center', fontsize=11, fontweight='bold', color='white')
    ax2.text(1, male_pct[0]/2, f'{male_pct[0]:.1f}%', ha='center', va='center',
             fontsize=11, fontweight='bold', color='white')
    ax2.text(1, male_pct[0] + male_pct[1]/2, f'{male_pct[1]:.1f}%', ha='center',
             va='center', fontsize=11, fontweight='bold', color='white')

    ax2.set_xlabel('Sex', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{cancer_type}{title_suffix}: Actionability Rates by Sex',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sex_labels, fontsize=11)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    # Sample sizes
    ax2.text(0.98, 0.02, f"Female n={int(female_total):,}\nMale n={int(male_total):,}",
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    # Filename includes gene if provided
    if gene:
        filename = f"{file_prefix}_{cancer_type}_{gene}_contingency"
    else:
        filename = f"{file_prefix}_{cancer_type}_contingency"

    output_dir = Path(output_dir)
    plt.savefig(output_dir / f"{filename}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}.pdf")
    plt.close()

    return f"{filename}.pdf"


def plot_categorical_contingency(
    cancer_type: str,
    column: str,
    cramers_v: float,
    p_val: float,
    output_dir: Path,
    gene: Optional[str] = None,
    feature_type: str = 'demographic',
    db_path: Optional[str] = None,
    file_prefix: Optional[str] = None
) -> Optional[str]:
    """
    Create contingency plots for categorical features (race/ethnicity/technical).

    Parameters
    ----------
    cancer_type : str
        OncoTree cancer type code
    column : str
        Database column name
    cramers_v : float
        Cramér's V effect size
    p_val : float
        FDR-adjusted p-value
    output_dir : Path
        Output directory for saving plots
    gene : str, optional
        Hugo symbol for gene-stratified analysis
    feature_type : str
        'demographic' or 'technical' (affects color scheme)
    db_path : str, optional
        Path to SQLite database
    file_prefix : str, optional
        Custom prefix for output filenames. If None, uses 'demo' or 'tech' based on feature_type.

    Returns
    -------
    str or None
        Filename of saved plot, or None if no data
    """
    df = load_categorical_data(cancer_type, column, gene=gene, db_path=db_path)

    if len(df) == 0:
        print(f"  Warning: No {column} data for {cancer_type}" + (f" ({gene})" if gene else ""))
        return None

    # Create contingency table
    ct = pd.crosstab(df[column], df['actionable'], margins=True)
    ct_sorted = ct.sort_values('All', ascending=False)

    # Get categories (exclude 'All')
    categories = [cat for cat in ct_sorted.index if cat != 'All']

    if len(categories) < 2:
        print(f"  Warning: Insufficient categories for {column} in {cancer_type}" +
              (f" ({gene})" if gene else ""))
        return None

    # Title includes gene if provided
    title_suffix = f" ({gene})" if gene else ""

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Color scheme based on feature type
    action_color = '#e74c3c' if feature_type == 'demographic' else '#2ecc71'

    # SUBPLOT 1: Actionability rates with CI
    ax1 = axes[0]

    rates = []
    ci_lower = []
    ci_upper = []
    totals = []

    for cat in categories:
        n_total = ct_sorted.loc[cat, 'All']
        n_actionable = ct_sorted.loc[cat, 1] if 1 in ct_sorted.columns else 0
        rate = (n_actionable / n_total * 100) if n_total > 0 else 0

        ci_l, ci_u = wilson_ci(n_actionable, n_total)
        rates.append(rate)
        ci_lower.append(ci_l * 100)
        ci_upper.append(ci_u * 100)
        totals.append(n_total)

    x = np.arange(len(categories))
    # Clip negative error values to 0 (can happen with very small rates)
    yerr_lower = [max(0, rates[i] - ci_lower[i]) for i in range(len(rates))]
    yerr_upper = [max(0, ci_upper[i] - rates[i]) for i in range(len(rates))]

    bars = ax1.bar(x, rates, color=action_color, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.errorbar(x, rates, yerr=[yerr_lower, yerr_upper],
                 fmt='none', ecolor='black', capsize=5, capthick=2, linewidth=2)

    # Labels
    for i, (bar, rate, total) in enumerate(zip(bars, rates, totals)):
        ax1.text(bar.get_x() + bar.get_width()/2., rate + max(yerr_upper) * 0.1,
                f'{rate:.1f}%\n(n={total:,})', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax1.set_xlabel(column, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actionability Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{cancer_type}{title_suffix}: Actionability Rates by {column}',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, rotation=45, ha='right')
    ax1.set_ylim([0, max(rates) * 1.3 if max(rates) > 0 else 10])
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # Statistics
    stats_text = f"Cramér's V = {cramers_v:.3f}\nFDR p = {p_val:.2e}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # SUBPLOT 2: Grouped bar chart (counts)
    ax2 = axes[1]

    non_actionable = [ct_sorted.loc[cat, 0] if 0 in ct_sorted.columns else 0 for cat in categories]
    actionable = [ct_sorted.loc[cat, 1] if 1 in ct_sorted.columns else 0 for cat in categories]

    width = 0.35
    bars1 = ax2.bar(x - width/2, non_actionable, width,
                    label='Non-Actionable', color='#95a5a6', alpha=0.8)
    bars2 = ax2.bar(x + width/2, actionable, width,
                    label='Actionable', color=action_color, alpha=0.8)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom',
                        fontsize=9, rotation=90)

    ax2.set_xlabel(column, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax2.set_title(f'{cancer_type}{title_suffix}: Actionability by {column} (Counts)',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10, rotation=45, ha='right')
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()

    # Safe filename
    safe_column = column.replace('/', '_').replace(' ', '_')

    # Determine prefix
    if file_prefix is None:
        prefix = 'tech' if feature_type == 'technical' else 'demo'
    else:
        prefix = file_prefix

    # Filename includes gene if provided
    if gene:
        filename = f"{prefix}_{cancer_type}_{gene}_{safe_column}_contingency"
    else:
        filename = f"{prefix}_{cancer_type}_{safe_column}_contingency"

    output_dir = Path(output_dir)
    plt.savefig(output_dir / f"{filename}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}.pdf")
    plt.close()

    return f"{filename}.pdf"


def plot_numeric_distribution(
    cancer_type: str,
    column: str,
    effect_size: float,
    p_val: float,
    mean_diff: float,
    output_dir: Path,
    gene: Optional[str] = None,
    feature_type: str = 'technical',
    db_path: Optional[str] = None,
    file_prefix: Optional[str] = None
) -> Optional[str]:
    """
    Create numeric distribution plots for technical features (e.g., number_of_genes).

    Similar to age distribution but for any numeric feature.

    Parameters
    ----------
    cancer_type : str
        OncoTree cancer type code
    column : str
        Database column name (numeric)
    effect_size : float
        Effect size (r from Mann-Whitney U)
    p_val : float
        FDR-adjusted p-value
    mean_diff : float
        Mean difference (actionable - non-actionable)
    output_dir : Path
        Output directory for saving plots
    gene : str, optional
        Hugo symbol for gene-stratified analysis
    feature_type : str
        'demographic' or 'technical' (affects color scheme)
    db_path : str, optional
        Path to SQLite database
    file_prefix : str, optional
        Custom prefix for output filenames

    Returns
    -------
    str or None
        Filename of saved plot, or None if no data
    """
    df = load_categorical_data(cancer_type, column, gene=gene, db_path=db_path)

    if len(df) == 0:
        print(f"  Warning: No {column} data for {cancer_type}" + (f" ({gene})" if gene else ""))
        return None

    # Rename column for consistency
    df['value'] = df[column]
    df['actionable_label'] = df['actionable'].map({0: 'Non-Actionable', 1: 'Actionable'})

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Title includes gene if provided
    title_suffix = f" ({gene})" if gene else ""

    # Color scheme
    action_color = '#e74c3c' if feature_type == 'demographic' else '#2ecc71'

    # SUBPLOT 1: Violin + box plot
    ax1 = axes[0]

    non_act_vals = df[df['actionable'] == 0]['value'].values
    act_vals = df[df['actionable'] == 1]['value'].values

    if len(non_act_vals) > 0 and len(act_vals) > 0:
        # Violin plot
        parts = ax1.violinplot(
            [non_act_vals, act_vals],
            positions=[0, 1],
            showmeans=True,
            showmedians=True,
            widths=0.7
        )

        # Color the violins
        colors = ['#95a5a6', action_color]
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # Box plot overlay
        ax1.boxplot(
            [non_act_vals, act_vals],
            positions=[0, 1],
            widths=0.3,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor='white', alpha=0.5),
            medianprops=dict(color='black', linewidth=2)
        )

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Non-Actionable', 'Actionable'], fontsize=11)
    ax1.set_ylabel(column, fontsize=12, fontweight='bold')
    ax1.set_title(f'{cancer_type}{title_suffix}: {column} by Actionability',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # Statistics annotation
    stats_text = f"Effect size r = {effect_size:.3f}\n"
    stats_text += f"Mean Δ = {mean_diff:.1f}\n"
    stats_text += f"FDR p = {p_val:.2e}"
    ax1.text(0.02, 0.98, stats_text,
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # SUBPLOT 2: Histogram + KDE
    ax2 = axes[1]

    ax2.hist(non_act_vals, bins=30, alpha=0.5, color='#95a5a6',
             label='Non-Actionable', density=True)
    ax2.hist(act_vals, bins=30, alpha=0.5, color=action_color,
             label='Actionable', density=True)

    # KDE overlay
    if len(non_act_vals) > 1 and len(act_vals) > 1:
        try:
            x_range = np.linspace(df['value'].min(), df['value'].max(), 200)

            kde_non = sp_stats.gaussian_kde(non_act_vals)
            ax2.plot(x_range, kde_non(x_range), color='#7f8c8d', linewidth=2,
                     linestyle='--', alpha=0.8)

            kde_act = sp_stats.gaussian_kde(act_vals)
            ax2.plot(x_range, kde_act(x_range), color=action_color, linewidth=2,
                     linestyle='--', alpha=0.8)
        except Exception:
            pass  # KDE can fail with certain data distributions

    ax2.set_xlabel(column, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title(f'{cancer_type}{title_suffix}: {column} Distribution Overlap',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    # Sample sizes
    ax2.text(0.98, 0.98, f"n(Non-Act) = {len(non_act_vals):,}\nn(Act) = {len(act_vals):,}",
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    # Safe filename
    safe_column = column.replace('/', '_').replace(' ', '_')

    # Determine prefix
    if file_prefix is None:
        prefix = 'tech' if feature_type == 'technical' else 'demo'
    else:
        prefix = file_prefix

    # Filename includes gene if provided
    if gene:
        filename = f"{prefix}_{cancer_type}_{gene}_{safe_column}_distribution"
    else:
        filename = f"{prefix}_{cancer_type}_{safe_column}_distribution"

    output_dir = Path(output_dir)
    plt.savefig(output_dir / f"{filename}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}.pdf")
    plt.close()

    return f"{filename}.pdf"


# Convenience functions for common use cases
def plot_gene_stratified_age(
    cancer_type: str,
    gene: str,
    cohens_d: float,
    p_val: float,
    mean_diff: float,
    output_dir: Path,
    db_path: Optional[str] = None
) -> Optional[str]:
    """Convenience wrapper for gene-stratified age plots."""
    return plot_age_distribution(
        cancer_type=cancer_type,
        cohens_d=cohens_d,
        p_val=p_val,
        mean_diff=mean_diff,
        output_dir=output_dir,
        gene=gene,
        db_path=db_path,
        file_prefix="gene_age"
    )


def plot_gene_stratified_sex(
    cancer_type: str,
    gene: str,
    odds_ratio: float,
    p_val: float,
    output_dir: Path,
    db_path: Optional[str] = None
) -> Optional[str]:
    """Convenience wrapper for gene-stratified sex plots."""
    return plot_sex_contingency(
        cancer_type=cancer_type,
        odds_ratio=odds_ratio,
        p_val=p_val,
        output_dir=output_dir,
        gene=gene,
        db_path=db_path,
        file_prefix="gene_sex"
    )


def plot_gene_stratified_categorical(
    cancer_type: str,
    gene: str,
    column: str,
    cramers_v: float,
    p_val: float,
    output_dir: Path,
    feature_type: str = 'demographic',
    db_path: Optional[str] = None
) -> Optional[str]:
    """Convenience wrapper for gene-stratified categorical plots."""
    return plot_categorical_contingency(
        cancer_type=cancer_type,
        column=column,
        cramers_v=cramers_v,
        p_val=p_val,
        output_dir=output_dir,
        gene=gene,
        feature_type=feature_type,
        db_path=db_path,
        file_prefix="gene"
    )
