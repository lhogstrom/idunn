#!/usr/bin/env python3
"""
Visualization Suite for Gene-Stratified Post-Hoc Analysis Results

Creates publication-ready visualizations for gene-stratified association results:
1. Forest plot: Cohen's d by gene within cancer type (grouped bars)
2. Heatmap: Cancer types (rows) x Genes (cols), colored by effect size
3. Comparison plot: Gene-specific vs cancer-wide effect sizes

Usage:
    # Auto-detect input from gene_stratified directory
    python visualize_gene_stratified_results.py

    # With explicit paths
    python visualize_gene_stratified_results.py --input-dir /path/to/gene_stratified --output-dir /path/to/output

"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))

# Import shared raw data plotting utilities for gene-stratified visualizations
from raw_data_plot_utils import (
    plot_gene_stratified_age,
    plot_gene_stratified_sex,
    plot_gene_stratified_categorical,
    plot_numeric_distribution
)
from db_config import get_default_db_path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# Repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "analysis_output" / "posthoc_statistics" / "gene_stratified"


def get_default_input_dir():
    """Find gene-stratified results directory."""
    # Check unified runs directory first
    runs_dir = REPO_ROOT / "runs"
    if runs_dir.exists():
        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        for run_dir in runs:
            gene_dir = run_dir / "tier7_posthoc_statistics" / "gene_stratified"
            if gene_dir.exists():
                return gene_dir

    # Fallback to analysis_output
    if DEFAULT_INPUT_DIR.exists():
        return DEFAULT_INPUT_DIR

    return None


def calculate_cohens_d_ci(d: float, n1: int, n2: int, confidence: float = 0.95):
    """
    Calculate confidence interval for Cohen's d using approximation.
    Based on Hedges & Olkin (1985) variance formula.
    """
    if pd.isna(d) or n1 < 2 or n2 < 2:
        return np.nan, np.nan

    n = n1 + n2
    var_d = ((n1 + n2) / (n1 * n2)) + (d**2 / (2 * (n1 + n2)))
    se_d = np.sqrt(var_d)

    z_critical = 1.96 if confidence == 0.95 else 2.576
    ci_lower = d - z_critical * se_d
    ci_upper = d + z_critical * se_d

    return ci_lower, ci_upper


def pval_to_stars(p):
    """Convert p-value to significance stars."""
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


def load_gene_stratified_data(input_dir: Path) -> dict:
    """
    Load all gene-stratified analysis results.
    Returns dict with available dataframes.
    """
    input_dir = Path(input_dir)
    data = {}

    # Age results
    age_file = input_dir / "age_by_gene" / "age_gene_results.csv"
    if age_file.exists():
        data['age'] = pd.read_csv(age_file)
        print(f"Loaded age results: {len(data['age'])} gene-cancer combinations")

    # Sex results
    sex_file = input_dir / "sex_by_gene" / "sex_gene_results.csv"
    if sex_file.exists():
        data['sex'] = pd.read_csv(sex_file)
        print(f"Loaded sex results: {len(data['sex'])} gene-cancer combinations")

    # Race results
    race_file = input_dir / "race_by_gene" / "race_gene_results.csv"
    if race_file.exists():
        data['race'] = pd.read_csv(race_file)
        print(f"Loaded race results: {len(data['race'])} gene-cancer combinations")

    # Ethnicity results
    eth_file = input_dir / "ethnicity_by_gene" / "ethnicity_gene_results.csv"
    if eth_file.exists():
        data['ethnicity'] = pd.read_csv(eth_file)
        print(f"Loaded ethnicity results: {len(data['ethnicity'])} gene-cancer combinations")

    # Sample type results
    sample_file = input_dir / "sample_type_by_gene" / "sample_type_gene_results.csv"
    if sample_file.exists():
        data['sample_type'] = pd.read_csv(sample_file)
        print(f"Loaded sample type results: {len(data['sample_type'])} gene-cancer combinations")

    # Technical results
    tech_file = input_dir / "technical_by_gene" / "technical_gene_results.csv"
    if tech_file.exists():
        data['technical'] = pd.read_csv(tech_file)
        print(f"Loaded technical results: {len(data['technical'])} tests")

    return data


def plot_gene_age_forest(age_df: pd.DataFrame, output_dir: Path):
    """
    Create forest plot for age differences by gene (Cohen's d with 95% CI).

    Groups genes within cancer type using different markers/colors.
    """
    print("\nCreating Figure 1: Gene-Stratified Age Differences Forest Plot...")

    if 'cohens_d' not in age_df.columns:
        print("  Skipping: no Cohen's d data")
        return

    # Filter to valid data
    plot_df = age_df[~age_df['cohens_d'].isna()].copy()

    if len(plot_df) == 0:
        print("  Skipping: no valid data")
        return

    # Create gene-cancer label
    plot_df['label'] = plot_df['ONCOTREE_CODE'] + ' - ' + plot_df['Hugo_Symbol']

    # Sort by cancer type, then by gene rank
    plot_df = plot_df.sort_values(['ONCOTREE_CODE', 'gene_rank'], ascending=[True, True])

    # Calculate confidence intervals
    ci_data = []
    for _, row in plot_df.iterrows():
        ci_lower, ci_upper = calculate_cohens_d_ci(
            row['cohens_d'],
            row['n_actionable'],
            row['n_non_actionable']
        )
        ci_data.append((ci_lower, ci_upper))

    plot_df['ci_lower'] = [x[0] for x in ci_data]
    plot_df['ci_upper'] = [x[1] for x in ci_data]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.4)))

    # Color by significance, shape by gene rank
    colors = []
    markers = []

    for _, row in plot_df.iterrows():
        if row.get('significant_fdr', False):
            colors.append('#e74c3c')  # Red for significant
        else:
            colors.append('#95a5a6')  # Gray for non-significant

        if row['gene_rank'] == 1:
            markers.append('o')  # Circle for rank 1
        else:
            markers.append('s')  # Square for rank 2

    # Plot error bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ci_lower = row['ci_lower'] if not pd.isna(row['ci_lower']) else row['cohens_d']
        ci_upper = row['ci_upper'] if not pd.isna(row['ci_upper']) else row['cohens_d']

        ax.errorbar(
            x=row['cohens_d'],
            y=i,
            xerr=[[row['cohens_d'] - ci_lower], [ci_upper - row['cohens_d']]],
            fmt=markers[i],
            color=colors[i],
            markersize=8,
            linewidth=2,
            capsize=5,
            capthick=2,
            alpha=0.8
        )

    # Vertical line at zero
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    # Effect size guidelines
    for d, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
        ax.axvline(x=d, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        ax.axvline(x=-d, color='gray', linestyle=':', linewidth=1, alpha=0.3)

    # Styling
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['label'], fontsize=9)
    ax.set_xlabel("Cohen's d (Effect Size)\nNegative = Younger | Positive = Older",
                  fontsize=12, fontweight='bold')
    ax.set_ylabel("Cancer Type - Gene", fontsize=12, fontweight='bold')
    ax.set_title("Age Differences by Gene-Specific Actionability\n(Forest Plot with 95% CI)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Patch(facecolor='#e74c3c', label='Significant (FDR < 0.05)'),
        Patch(facecolor='#95a5a6', label='Not Significant'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Gene Rank 1'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=8, label='Gene Rank 2')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "fig1_gene_age_forest_plot.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig1_gene_age_forest_plot.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: fig1_gene_age_forest_plot.pdf")
    plt.close()


def plot_gene_cancer_effect_heatmap(age_df: pd.DataFrame, output_dir: Path):
    """
    Create heatmap: Cancer types (rows) x Genes (cols), colored by Cohen's d.
    """
    print("\nCreating Figure 2: Gene-Cancer Effect Size Heatmap...")

    if 'cohens_d' not in age_df.columns:
        print("  Skipping: no Cohen's d data")
        return

    # Create pivot table
    pivot_df = age_df.pivot_table(
        index='ONCOTREE_CODE',
        columns='Hugo_Symbol',
        values='cohens_d',
        aggfunc='first'
    )

    if pivot_df.shape[0] == 0 or pivot_df.shape[1] == 0:
        print("  Skipping: insufficient data for heatmap")
        return

    # Create significance annotations
    sig_pivot = age_df.pivot_table(
        index='ONCOTREE_CODE',
        columns='Hugo_Symbol',
        values='p_fdr',
        aggfunc='first'
    )

    # Reindex to match
    sig_pivot = sig_pivot.reindex(index=pivot_df.index, columns=pivot_df.columns)
    annot_matrix = sig_pivot.map(pval_to_stars)

    # Figure size based on data
    n_rows = len(pivot_df)
    n_cols = len(pivot_df.columns)
    fig_width = max(8, n_cols * 0.8)
    fig_height = max(6, n_rows * 0.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Symmetric colormap centered at 0
    vmax = max(abs(pivot_df.min().min()), abs(pivot_df.max().max()))
    vmax = min(vmax, 1.0)  # Cap at 1.0

    sns.heatmap(
        pivot_df,
        annot=annot_matrix,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={'label': "Cohen's d (Effect Size)"},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )

    ax.set_xlabel("Gene", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cancer Type", fontsize=12, fontweight='bold')
    ax.set_title("Age Effect Size by Cancer Type and Gene\n(Cohen's d, * = FDR < 0.05)",
                 fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Significance legend
    ax.text(
        1.15, 0.95,
        "Significance:\n* p < 0.05\n** p < 0.01\n*** p < 0.001",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )

    plt.tight_layout()
    plt.savefig(output_dir / "fig2_gene_cancer_effect_heatmap.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig2_gene_cancer_effect_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: fig2_gene_cancer_effect_heatmap.pdf")
    plt.close()


def plot_gene_sex_forest(sex_df: pd.DataFrame, output_dir: Path):
    """
    Create forest plot for sex associations by gene (Odds Ratios with 95% CI).
    """
    print("\nCreating Figure 3: Gene-Stratified Sex Associations Forest Plot...")

    if 'odds_ratio' not in sex_df.columns:
        print("  Skipping: no odds ratio data")
        return

    # Filter to valid data
    plot_df = sex_df[~sex_df['odds_ratio'].isna()].copy()

    if len(plot_df) == 0:
        print("  Skipping: no valid data")
        return

    # Create gene-cancer label
    plot_df['label'] = plot_df['ONCOTREE_CODE'] + ' - ' + plot_df['Hugo_Symbol']

    # Sort by cancer type, then gene rank
    plot_df = plot_df.sort_values(['ONCOTREE_CODE', 'gene_rank'], ascending=[True, True])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.4)))

    # Colors based on significance
    colors = ['#9b59b6' if sig else '#95a5a6' for sig in plot_df.get('significant_fdr', [False]*len(plot_df))]

    # Plot error bars
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        or_lower = row.get('or_ci_lower', row['odds_ratio'])
        or_upper = row.get('or_ci_upper', row['odds_ratio'])

        if pd.isna(or_lower):
            or_lower = row['odds_ratio']
        if pd.isna(or_upper):
            or_upper = row['odds_ratio']

        ax.errorbar(
            x=row['odds_ratio'],
            y=i,
            xerr=[[row['odds_ratio'] - or_lower], [or_upper - row['odds_ratio']]],
            fmt='o' if row['gene_rank'] == 1 else 's',
            color=colors[i],
            markersize=8,
            linewidth=2,
            capsize=5,
            capthick=2,
            alpha=0.8
        )

    # Vertical line at OR = 1
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    # Styling
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['label'], fontsize=9)
    ax.set_xlabel("Odds Ratio (95% CI)\nOR > 1 = Higher female rate in gene-actionable",
                  fontsize=12, fontweight='bold')
    ax.set_ylabel("Cancer Type - Gene", fontsize=12, fontweight='bold')
    ax.set_title("Sex Associations by Gene-Specific Actionability\n(Forest Plot with 95% CI)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9b59b6', label='Significant (FDR < 0.05)'),
        Patch(facecolor='#95a5a6', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "fig3_gene_sex_forest_plot.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig3_gene_sex_forest_plot.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: fig3_gene_sex_forest_plot.pdf")
    plt.close()


def plot_combined_significance_summary(data: dict, output_dir: Path):
    """
    Create combined significance summary showing number of significant associations
    per gene-cancer combination across all test categories.
    """
    print("\nCreating Figure 4: Combined Significance Summary...")

    # Collect all significant associations
    all_sig = []

    for category, df in data.items():
        if len(df) > 0 and 'significant_fdr' in df.columns:
            sig = df[df['significant_fdr']].copy()
            sig['category'] = category
            if 'ONCOTREE_CODE' in sig.columns and 'Hugo_Symbol' in sig.columns:
                all_sig.append(sig[['ONCOTREE_CODE', 'Hugo_Symbol', 'category']])

    if not all_sig:
        print("  Skipping: no significant associations found")
        return

    combined = pd.concat(all_sig)

    # Count by gene-cancer combination
    summary = combined.groupby(['ONCOTREE_CODE', 'Hugo_Symbol']).agg({
        'category': lambda x: list(x)
    }).reset_index()
    summary['n_significant'] = summary['category'].apply(len)
    summary['label'] = summary['ONCOTREE_CODE'] + ' - ' + summary['Hugo_Symbol']

    # Sort by count
    summary = summary.sort_values('n_significant', ascending=True)

    # Count by category for stacked bars
    category_counts = {}
    for cat in ['age', 'sex', 'race', 'ethnicity', 'sample_type', 'technical']:
        category_counts[cat] = summary['category'].apply(
            lambda x: x.count(cat) if isinstance(x, list) else 0
        )

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, max(8, len(summary) * 0.4)))

    y_pos = np.arange(len(summary))
    bar_width = 0.8

    colors = {
        'age': '#3498db',
        'sex': '#9b59b6',
        'race': '#e74c3c',
        'ethnicity': '#f39c12',
        'sample_type': '#27ae60',
        'technical': '#95a5a6'
    }

    left = np.zeros(len(summary))
    for cat in ['age', 'sex', 'race', 'ethnicity', 'sample_type', 'technical']:
        counts = category_counts[cat].values
        if counts.sum() > 0:
            ax.barh(y_pos, counts, bar_width, left=left,
                   label=cat.replace('_', ' ').title(), color=colors[cat], alpha=0.8)
            left += counts

    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary['label'], fontsize=9)
    ax.set_xlabel("Number of Significant Associations (FDR < 0.05)",
                  fontsize=12, fontweight='bold')
    ax.set_ylabel("Cancer Type - Gene", fontsize=12, fontweight='bold')
    ax.set_title("Significant Gene-Stratified Associations by Category",
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(output_dir / "fig4_combined_significance_summary.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig4_combined_significance_summary.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: fig4_combined_significance_summary.pdf")
    plt.close()


def plot_gene_rank_comparison(age_df: pd.DataFrame, output_dir: Path):
    """
    Create scatter plot comparing effect sizes between gene rank 1 and rank 2.
    """
    print("\nCreating Figure 5: Gene Rank Comparison...")

    if 'cohens_d' not in age_df.columns or 'gene_rank' not in age_df.columns:
        print("  Skipping: required columns not found")
        return

    # Split by gene rank
    rank1 = age_df[age_df['gene_rank'] == 1][['ONCOTREE_CODE', 'cohens_d', 'significant_fdr']].copy()
    rank2 = age_df[age_df['gene_rank'] == 2][['ONCOTREE_CODE', 'cohens_d', 'significant_fdr']].copy()

    rank1.columns = ['ONCOTREE_CODE', 'cohens_d_rank1', 'sig_rank1']
    rank2.columns = ['ONCOTREE_CODE', 'cohens_d_rank2', 'sig_rank2']

    merged = rank1.merge(rank2, on='ONCOTREE_CODE', how='inner')

    if len(merged) == 0:
        print("  Skipping: no cancer types with both rank 1 and rank 2 genes")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Color by whether either is significant
    colors = []
    for _, row in merged.iterrows():
        if row.get('sig_rank1', False) and row.get('sig_rank2', False):
            colors.append('#e74c3c')  # Both significant
        elif row.get('sig_rank1', False) or row.get('sig_rank2', False):
            colors.append('#f39c12')  # One significant
        else:
            colors.append('#95a5a6')  # Neither significant

    ax.scatter(
        merged['cohens_d_rank1'],
        merged['cohens_d_rank2'],
        c=colors,
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )

    # Add labels for significant points
    for _, row in merged.iterrows():
        if row.get('sig_rank1', False) or row.get('sig_rank2', False):
            ax.annotate(
                row['ONCOTREE_CODE'],
                xy=(row['cohens_d_rank1'], row['cohens_d_rank2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8
            )

    # Reference lines
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Diagonal line (equal effect)
    lim_max = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
                  abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
    ax.plot([-lim_max, lim_max], [-lim_max, lim_max],
           'k:', linewidth=1, alpha=0.5, label='Equal effect')

    ax.set_xlabel("Cohen's d (Gene Rank 1 - Top Gene)",
                  fontsize=12, fontweight='bold')
    ax.set_ylabel("Cohen's d (Gene Rank 2 - Second Gene)",
                  fontsize=12, fontweight='bold')
    ax.set_title("Comparison of Age Effects: Top Gene vs Second Gene\nby Cancer Type",
                 fontsize=14, fontweight='bold', pad=20)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Both genes significant'),
        Patch(facecolor='#f39c12', label='One gene significant'),
        Patch(facecolor='#95a5a6', label='Neither significant')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.grid(alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(output_dir / "fig5_gene_rank_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig5_gene_rank_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: fig5_gene_rank_comparison.pdf")
    plt.close()


def plot_gene_raw_data(data: dict, output_dir: Path, db_path: str = None):
    """
    Create raw data plots (violin/contingency) for significant gene-stratified associations.

    Parameters
    ----------
    data : dict
        Dictionary with loaded gene-stratified results
    output_dir : Path
        Output directory for plots
    db_path : str, optional
        Path to database
    """
    print("\n" + "=" * 80)
    print("Creating Raw Data Plots for Significant Gene-Stratified Associations...")
    print("=" * 80)

    db_path = db_path or get_default_db_path()
    raw_data_dir = output_dir / "raw_data_plots"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    total_plots = 0

    # Age raw data plots
    if 'age' in data and len(data['age']) > 0:
        sig_age = data['age'][data['age'].get('significant_fdr', False) == True]
        print(f"\nAge distributions: {len(sig_age)} significant gene-cancer combinations")

        for _, row in sig_age.iterrows():
            cancer = row['ONCOTREE_CODE']
            gene = row['Hugo_Symbol']
            print(f"  Plotting {cancer} - {gene}...")

            result = plot_gene_stratified_age(
                cancer_type=cancer,
                gene=gene,
                cohens_d=row.get('cohens_d', 0),
                p_val=row.get('p_fdr', 1),
                mean_diff=row.get('mean_age_difference', 0),
                output_dir=raw_data_dir,
                db_path=db_path
            )
            if result:
                total_plots += 1

    # Sex raw data plots
    if 'sex' in data and len(data['sex']) > 0:
        sig_sex = data['sex'][data['sex'].get('significant_fdr', False) == True]
        print(f"\nSex contingency: {len(sig_sex)} significant gene-cancer combinations")

        for _, row in sig_sex.iterrows():
            cancer = row['ONCOTREE_CODE']
            gene = row['Hugo_Symbol']
            print(f"  Plotting {cancer} - {gene}...")

            result = plot_gene_stratified_sex(
                cancer_type=cancer,
                gene=gene,
                odds_ratio=row.get('odds_ratio', 1),
                p_val=row.get('p_fdr', 1),
                output_dir=raw_data_dir,
                db_path=db_path
            )
            if result:
                total_plots += 1

    # Race raw data plots
    if 'race' in data and len(data['race']) > 0:
        sig_race = data['race'][data['race'].get('significant_fdr', False) == True]
        print(f"\nRace contingency: {len(sig_race)} significant gene-cancer combinations")

        for _, row in sig_race.iterrows():
            cancer = row['ONCOTREE_CODE']
            gene = row['Hugo_Symbol']
            print(f"  Plotting {cancer} - {gene} (RACE)...")

            result = plot_gene_stratified_categorical(
                cancer_type=cancer,
                gene=gene,
                column='RACE',
                cramers_v=row.get('cramers_v', 0),
                p_val=row.get('p_fdr', 1),
                output_dir=raw_data_dir,
                feature_type='demographic',
                db_path=db_path
            )
            if result:
                total_plots += 1

    # Ethnicity raw data plots
    if 'ethnicity' in data and len(data['ethnicity']) > 0:
        sig_eth = data['ethnicity'][data['ethnicity'].get('significant_fdr', False) == True]
        print(f"\nEthnicity contingency: {len(sig_eth)} significant gene-cancer combinations")

        for _, row in sig_eth.iterrows():
            cancer = row['ONCOTREE_CODE']
            gene = row['Hugo_Symbol']
            print(f"  Plotting {cancer} - {gene} (ETHNICITY)...")

            result = plot_gene_stratified_categorical(
                cancer_type=cancer,
                gene=gene,
                column='ETHNICITY',
                cramers_v=row.get('cramers_v', 0),
                p_val=row.get('p_fdr', 1),
                output_dir=raw_data_dir,
                feature_type='demographic',
                db_path=db_path
            )
            if result:
                total_plots += 1

    # Technical raw data plots (number_of_genes is numeric)
    if 'technical' in data and len(data['technical']) > 0:
        sig_tech = data['technical'][data['technical'].get('significant_fdr', False) == True]
        print(f"\nTechnical feature contingency: {len(sig_tech)} significant tests")

        for _, row in sig_tech.iterrows():
            cancer = row['ONCOTREE_CODE']
            gene = row['Hugo_Symbol']
            feature = row.get('feature', 'unknown')

            if feature == 'number_of_genes':
                # Numeric feature - use distribution plot
                print(f"  Plotting {cancer} - {gene} ({feature}, numeric)...")
                result = plot_numeric_distribution(
                    cancer_type=cancer,
                    column=feature,
                    effect_size=row.get('effect_size_r', 0),
                    p_val=row.get('p_fdr', 1),
                    mean_diff=row.get('gene_mean', 0) - row.get('non_actionable_mean', 0),
                    output_dir=raw_data_dir,
                    gene=gene,
                    feature_type='technical',
                    db_path=db_path
                )
            else:
                # Categorical feature
                print(f"  Plotting {cancer} - {gene} ({feature}, categorical)...")
                result = plot_gene_stratified_categorical(
                    cancer_type=cancer,
                    gene=gene,
                    column=feature,
                    cramers_v=row.get('cramers_v', 0),
                    p_val=row.get('p_fdr', 1),
                    output_dir=raw_data_dir,
                    feature_type='technical',
                    db_path=db_path
                )

            if result:
                total_plots += 1

    print(f"\nTotal raw data plots generated: {total_plots}")
    print(f"Saved to: {raw_data_dir}")


def print_summary_statistics(data: dict):
    """Print summary statistics for all loaded data."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for category, df in data.items():
        if len(df) > 0:
            print(f"\n{category.upper()}:")
            print(f"  Total gene-cancer combinations: {len(df)}")

            if 'significant_fdr' in df.columns:
                n_sig = df['significant_fdr'].sum()
                print(f"  Significant (FDR < 0.05): {n_sig}")

            if category == 'age' and 'cohens_d' in df.columns:
                valid_d = df['cohens_d'].dropna()
                if len(valid_d) > 0:
                    print(f"  Cohen's d range: {valid_d.min():.3f} to {valid_d.max():.3f}")
                    large_effects = (np.abs(valid_d) > 0.8).sum()
                    print(f"  Large effect sizes (|d| > 0.8): {large_effects}")

            if category == 'sex' and 'odds_ratio' in df.columns:
                valid_or = df['odds_ratio'].dropna()
                if len(valid_or) > 0:
                    print(f"  Odds ratio range: {valid_or.min():.3f} to {valid_or.max():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Create visualizations for gene-stratified post-hoc analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect input from latest run
    python visualize_gene_stratified_results.py

    # Specify input and output directories
    python visualize_gene_stratified_results.py --input-dir /path/to/gene_stratified --output-dir /path/to/output

    # Include raw data plots (violin/contingency) for significant associations
    python visualize_gene_stratified_results.py --raw-data-plots
        """
    )

    parser.add_argument('--input-dir', type=str, default=None,
                       help='Directory containing gene-stratified analysis results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for visualizations')
    parser.add_argument('--raw-data-plots', action='store_true',
                       help='Generate raw data plots (violin/contingency) for significant associations')
    parser.add_argument('--db-path', type=str, default=None,
                       help='Path to SQLite database (for raw data plots)')

    args = parser.parse_args()

    # Determine input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = get_default_input_dir()
        if input_dir is None:
            print("ERROR: No gene-stratified results found.")
            print("Run analyze_gene_stratified_associations.py first.")
            return 1

    print(f"Input directory: {input_dir}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / "visualizations"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    print("\n" + "=" * 80)
    print("Loading gene-stratified analysis results...")
    print("=" * 80)

    data = load_gene_stratified_data(input_dir)

    if not data:
        print("ERROR: No gene-stratified results found in input directory")
        return 1

    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)

    # Age plots
    if 'age' in data:
        plot_gene_age_forest(data['age'], output_dir)
        plot_gene_cancer_effect_heatmap(data['age'], output_dir)
        plot_gene_rank_comparison(data['age'], output_dir)

    # Sex plot
    if 'sex' in data:
        plot_gene_sex_forest(data['sex'], output_dir)

    # Combined summary
    plot_combined_significance_summary(data, output_dir)

    # Raw data plots (violin/contingency) for significant associations
    if args.raw_data_plots:
        plot_gene_raw_data(data, output_dir, db_path=args.db_path)

    # Print summary statistics
    print_summary_statistics(data)

    print(f"\n" + "=" * 80)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
