#!/usr/bin/env python3
"""
Visualization Suite for Post-Hoc Statistical Analysis Results

Creates publication-ready visualizations emphasizing effect sizes and significance
for age-actionability, demographic, and technical feature association analyses.

Figures:
1. Age differences: Forest plot with Cohen's d effect sizes
2. Age differences: Volcano plot (effect size vs significance)
3. Sex associations: Forest plot with odds ratios
4. Race/Ethnicity associations: Effect size heatmap
5. Technical feature associations: Effect size heatmap
6. Combined significance summary (demographic + technical)

Usage:
    # With default paths (auto-detect from runs directory)
    python visualize_posthoc_results.py

    # With explicit input/output directories
    python visualize_posthoc_results.py --input-dir /path/to/posthoc_results --output-dir /path/to/output

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

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# Repository root for default paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_default_input_dir():
    """Find latest post-hoc results directory."""
    # Check unified runs directory first
    runs_dir = REPO_ROOT / "runs"
    if runs_dir.exists():
        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        for run_dir in runs:
            posthoc_dir = run_dir / "tier7_posthoc_statistics"
            if posthoc_dir.exists():
                return posthoc_dir

    # Fallback to analysis_output
    legacy_dir = REPO_ROOT / "analysis_output" / "posthoc_statistics"
    if legacy_dir.exists():
        return legacy_dir

    return None


def calculate_cohens_d_ci(d, n1, n2, confidence=0.95):
    """
    Calculate confidence interval for Cohen's d using approximation.

    Based on Hedges & Olkin (1985) variance formula:
    Var(d) = (n1+n2)/(n1*n2) + d^2 / (2*df)
    where df = n1 + n2 - 2.
    """
    n = n1 + n2
    df = n1 + n2 - 2
    # Guard against df <= 0 (degenerate case)
    if df <= 0:
        return np.nan, np.nan
    var_d = ((n1 + n2) / (n1 * n2)) + (d**2 / (2 * df))
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


def load_data(input_dir):
    """
    Load all post-hoc analysis results.

    Returns dict with available dataframes.
    """
    input_dir = Path(input_dir)
    data = {}

    # Try to load from subdirectories (unified run structure)
    age_dir = input_dir / "age_actionability"
    shap_dir = input_dir / "shap_associations"

    # Age results
    age_file = age_dir / "age_actionability_results.csv" if age_dir.exists() else input_dir / "age_actionability_results.csv"
    if age_file.exists():
        data['age'] = pd.read_csv(age_file)
        print(f"Loaded age results: {len(data['age'])} cancer types")

    # Sex associations
    sex_file = shap_dir / "sex_associations.csv" if shap_dir.exists() else input_dir / "sex_associations.csv"
    if sex_file.exists():
        data['sex'] = pd.read_csv(sex_file)
        print(f"Loaded sex associations: {len(data['sex'])} tests")

    # Race/ethnicity associations
    race_file = shap_dir / "race_ethnicity_associations.csv" if shap_dir.exists() else input_dir / "race_ethnicity_associations.csv"
    if race_file.exists():
        data['race_eth'] = pd.read_csv(race_file)
        print(f"Loaded race/ethnicity associations: {len(data['race_eth'])} tests")

    # Sample type associations
    sample_type_file = shap_dir / "sample_type_associations.csv" if shap_dir.exists() else input_dir / "sample_type_associations.csv"
    if sample_type_file.exists():
        data['sample_type'] = pd.read_csv(sample_type_file)
        print(f"Loaded sample type associations: {len(data['sample_type'])} tests")

    # Technical feature associations
    tech_file = shap_dir / "technical_feature_associations.csv" if shap_dir.exists() else input_dir / "technical_feature_associations.csv"
    if tech_file.exists():
        data['technical'] = pd.read_csv(tech_file)
        print(f"Loaded technical feature associations: {len(data['technical'])} tests")

    return data


def plot_age_forest(age_df, output_dir):
    """Create forest plot for age differences (Cohen's d with 95% CI)."""
    print("\nCreating Figure 1: Age Differences Forest Plot...")

    age_plot_df = age_df.copy()
    age_plot_df = age_plot_df.sort_values('cohens_d', ascending=True)

    # Calculate confidence intervals
    ci_data = []
    for _, row in age_plot_df.iterrows():
        ci_lower, ci_upper = calculate_cohens_d_ci(
            row['cohens_d'],
            row['actionable_n'],
            row['non_actionable_n']
        )
        ci_data.append((ci_lower, ci_upper))

    age_plot_df['ci_lower'] = [x[0] for x in ci_data]
    age_plot_df['ci_upper'] = [x[1] for x in ci_data]
    age_plot_df['ci_error_lower'] = age_plot_df['cohens_d'] - age_plot_df['ci_lower']
    age_plot_df['ci_error_upper'] = age_plot_df['ci_upper'] - age_plot_df['cohens_d']

    # Create forest plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(age_plot_df) * 0.4)))

    # Color by significance
    colors = ['#e74c3c' if sig else '#95a5a6'
              for sig in age_plot_df['mann_whitney_significant_fdr']]

    # Plot error bars
    for i, (idx, row) in enumerate(age_plot_df.iterrows()):
        ax.errorbar(
            x=row['cohens_d'],
            y=i,
            xerr=[[row['ci_error_lower']], [row['ci_error_upper']]],
            fmt='o',
            color=colors[i],
            markersize=8,
            linewidth=2,
            capsize=5,
            capthick=2,
            alpha=0.8
        )

    # Vertical line at zero
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    # Styling
    ax.set_yticks(range(len(age_plot_df)))
    ax.set_yticklabels(age_plot_df['ONCOTREE_CODE'], fontsize=10)
    ax.set_xlabel("Cohen's d (Effect Size)\nNegative = Younger | Positive = Older",
                  fontsize=12, fontweight='bold')
    ax.set_ylabel("Cancer Type", fontsize=12, fontweight='bold')
    ax.set_title("Age Differences by Actionability Status\n(Forest Plot with 95% CI)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Significant (FDR < 0.05)'),
        Patch(facecolor='#95a5a6', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "fig1_age_forest_plot.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig1_age_forest_plot.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: fig1_age_forest_plot.pdf")
    plt.close()


def plot_age_volcano(age_df, output_dir):
    """Create volcano plot for age differences."""
    print("\nCreating Figure 2: Age Differences Volcano Plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate -log10(p-value)
    age_df = age_df.copy()
    age_df['neg_log10_p'] = -np.log10(age_df['mann_whitney_p_fdr'].clip(lower=1e-300))

    # Color by significance and direction
    def get_color_volcano(row):
        if row['mann_whitney_significant_fdr']:
            return '#e74c3c' if row['cohens_d'] > 0 else '#3498db'
        return '#95a5a6'

    colors_volcano = age_df.apply(get_color_volcano, axis=1)

    # Scatter plot
    ax.scatter(
        age_df['cohens_d'],
        age_df['neg_log10_p'],
        c=colors_volcano,
        s=150,
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )

    # Add labels for significant results
    for _, row in age_df[age_df['mann_whitney_significant_fdr']].iterrows():
        ax.annotate(
            row['ONCOTREE_CODE'],
            xy=(row['cohens_d'], row['neg_log10_p']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            alpha=0.8
        )

    # Threshold lines
    fdr_threshold = -np.log10(0.05)
    ax.axhline(y=fdr_threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    # Styling
    ax.set_xlabel("Cohen's d (Effect Size)\nNegative = Younger | Positive = Older",
                  fontsize=12, fontweight='bold')
    ax.set_ylabel("-log₁₀(FDR-adjusted p-value)", fontsize=12, fontweight='bold')
    ax.set_title("Age-Actionability Associations\n(Volcano Plot)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, linestyle=':')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Sig: Older when actionable'),
        Patch(facecolor='#3498db', label='Sig: Younger when actionable'),
        Patch(facecolor='#95a5a6', label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "fig2_age_volcano_plot.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig2_age_volcano_plot.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: fig2_age_volcano_plot.pdf")
    plt.close()


def plot_sex_forest(sex_df, output_dir):
    """Create forest plot for sex associations (odds ratios)."""
    print("\nCreating Figure 3: Sex Associations Forest Plot...")

    sex_plot_df = sex_df.copy()
    sex_plot_df = sex_plot_df.sort_values('odds_ratio', ascending=True)

    # Calculate errors for error bars
    sex_plot_df['or_error_lower'] = sex_plot_df['odds_ratio'] - sex_plot_df['or_ci_lower']
    sex_plot_df['or_error_upper'] = sex_plot_df['or_ci_upper'] - sex_plot_df['odds_ratio']

    # Create forest plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(sex_plot_df) * 0.4)))

    # Color by significance
    colors_sex = ['#9b59b6' if sig else '#95a5a6'
                  for sig in sex_plot_df['significant_fdr']]

    # Plot error bars
    for i, (idx, row) in enumerate(sex_plot_df.iterrows()):
        ax.errorbar(
            x=row['odds_ratio'],
            y=i,
            xerr=[[row['or_error_lower']], [row['or_error_upper']]],
            fmt='o',
            color=colors_sex[i],
            markersize=8,
            linewidth=2,
            capsize=5,
            capthick=2,
            alpha=0.8
        )

    # Vertical line at OR = 1
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    # Styling
    ax.set_yticks(range(len(sex_plot_df)))
    ax.set_yticklabels(sex_plot_df['ONCOTREE_CODE'], fontsize=10)
    ax.set_xlabel("Odds Ratio (95% CI)\nOR > 1 = Female more actionable",
                  fontsize=12, fontweight='bold')
    ax.set_ylabel("Cancer Type", fontsize=12, fontweight='bold')
    ax.set_title("Sex-Actionability Associations\n(Forest Plot with 95% CI)",
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
    plt.savefig(output_dir / "fig3_sex_forest_plot.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig3_sex_forest_plot.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: fig3_sex_forest_plot.pdf")
    plt.close()


def plot_categorical_heatmap(df, output_dir, filename_prefix, title, feature_type='demographic'):
    """
    Create heatmap for categorical feature associations (Cramér's V).

    Works for both race/ethnicity and technical features.
    """
    print(f"\nCreating {title}...")

    # Filter to significant results
    sig_df = df[df['significant_fdr']].copy()

    if len(sig_df) == 0:
        print(f"  No significant associations found for {feature_type} features")
        return

    # Create matrix for heatmap
    heatmap_data = sig_df.pivot_table(
        index='ONCOTREE_CODE',
        columns='column',
        values='cramers_v',
        aggfunc='first'
    )

    # Create significance matrix with same structure as heatmap_data
    sig_matrix = sig_df.pivot_table(
        index='ONCOTREE_CODE',
        columns='column',
        values='p_fdr',
        aggfunc='first'
    )

    # Ensure annotation matrix has same shape as heatmap_data
    # Reindex to match heatmap_data structure
    sig_matrix = sig_matrix.reindex(index=heatmap_data.index, columns=heatmap_data.columns)

    # Convert p-values to star annotations
    annot_matrix = sig_matrix.map(pval_to_stars)

    # Determine figure size based on data
    n_rows = len(heatmap_data)
    n_cols = len(heatmap_data.columns)
    fig_width = max(8, n_cols * 2)
    fig_height = max(6, n_rows * 0.5)

    # Create figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Color map based on feature type
    cmap = 'YlOrRd' if feature_type == 'demographic' else 'YlGnBu'

    # Heatmap
    sns.heatmap(
        heatmap_data,
        annot=annot_matrix,
        fmt='',
        cmap=cmap,
        cbar_kws={'label': "Cramér's V (Effect Size)"},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0,
        vmax=0.2
    )

    # Styling
    ax.set_xlabel("Feature", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cancer Type", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
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
    plt.savefig(output_dir / f"{filename_prefix}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"{filename_prefix}.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename_prefix}.pdf")
    plt.close()


def plot_combined_summary(data, output_dir):
    """Create combined significance summary (demographic + technical)."""
    print("\nCreating Figure 7: Combined Significance Summary...")

    # Get all cancer types
    all_cancers = set()
    if 'age' in data:
        all_cancers.update(data['age']['ONCOTREE_CODE'].unique())
    if 'sex' in data:
        all_cancers.update(data['sex']['ONCOTREE_CODE'].unique())
    if 'race_eth' in data:
        all_cancers.update(data['race_eth']['ONCOTREE_CODE'].unique())
    if 'sample_type' in data:
        all_cancers.update(data['sample_type']['ONCOTREE_CODE'].unique())
    if 'technical' in data:
        all_cancers.update(data['technical']['ONCOTREE_CODE'].unique())

    # Count significant associations per cancer type
    sig_counts = []

    for cancer in all_cancers:
        counts = {'ONCOTREE_CODE': cancer}

        # Age
        if 'age' in data:
            age_sig = data['age'][
                (data['age']['ONCOTREE_CODE'] == cancer) &
                (data['age']['mann_whitney_significant_fdr'])
            ].shape[0]
            counts['Age'] = age_sig
        else:
            counts['Age'] = 0

        # Sex
        if 'sex' in data:
            sex_sig = data['sex'][
                (data['sex']['ONCOTREE_CODE'] == cancer) &
                (data['sex']['significant_fdr'])
            ].shape[0]
            counts['Sex'] = sex_sig
        else:
            counts['Sex'] = 0

        # Race/Ethnicity
        if 'race_eth' in data:
            race_sig = data['race_eth'][
                (data['race_eth']['ONCOTREE_CODE'] == cancer) &
                (data['race_eth']['significant_fdr'])
            ].shape[0]
            counts['Race/Ethnicity'] = race_sig
        else:
            counts['Race/Ethnicity'] = 0

        # Sample Type
        if 'sample_type' in data:
            sample_sig = data['sample_type'][
                (data['sample_type']['ONCOTREE_CODE'] == cancer) &
                (data['sample_type']['significant_fdr'])
            ].shape[0]
            counts['Sample Type'] = sample_sig
        else:
            counts['Sample Type'] = 0

        # Technical
        if 'technical' in data:
            tech_sig = data['technical'][
                (data['technical']['ONCOTREE_CODE'] == cancer) &
                (data['technical']['significant_fdr'])
            ].shape[0]
            counts['Technical'] = tech_sig
        else:
            counts['Technical'] = 0

        counts['Total'] = counts['Age'] + counts['Sex'] + counts['Race/Ethnicity'] + counts['Technical']
        sig_counts.append(counts)

    sig_summary = pd.DataFrame(sig_counts).sort_values('Total', ascending=True)

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, max(8, len(sig_summary) * 0.4)))

    bar_width = 0.8
    y_pos = np.arange(len(sig_summary))

    # Colors for different categories
    colors = {
        'Age': '#3498db',
        'Sex': '#9b59b6',
        'Race/Ethnicity': '#e74c3c',
        'Technical': '#2ecc71'
    }

    # Stacked bars
    left = np.zeros(len(sig_summary))
    for category in ['Age', 'Sex', 'Race/Ethnicity', 'Technical']:
        if category in sig_summary.columns:
            ax.barh(y_pos, sig_summary[category], bar_width,
                   left=left, label=category, color=colors[category], alpha=0.8)
            left += sig_summary[category].values

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sig_summary['ONCOTREE_CODE'], fontsize=10)
    ax.set_xlabel("Number of Significant Associations (FDR < 0.05)",
                  fontsize=12, fontweight='bold')
    ax.set_ylabel("Cancer Type", fontsize=12, fontweight='bold')
    ax.set_title("Significant Feature-Actionability Associations\n(Demographic + Technical Features)",
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(axis='x', alpha=0.3, linestyle=':')

    # Add total counts
    for i, (_, row) in enumerate(sig_summary.iterrows()):
        if row['Total'] > 0:
            ax.text(row['Total'] + 0.1, i, f"{int(row['Total'])}",
                   va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "fig7_combined_summary.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig7_combined_summary.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: fig7_combined_summary.pdf")
    plt.close()


def print_summary_statistics(data):
    """Print summary statistics for all loaded data."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if 'age' in data:
        age_df = data['age']
        print(f"\nAge-Actionability Associations:")
        print(f"  Total cancer types tested: {len(age_df)}")
        print(f"  Significant (FDR < 0.05): {age_df['mann_whitney_significant_fdr'].sum()}")
        print(f"  Effect size range (Cohen's d): {age_df['cohens_d'].min():.3f} to {age_df['cohens_d'].max():.3f}")
        print(f"  Large effect sizes (|d| > 0.8): {(np.abs(age_df['cohens_d']) > 0.8).sum()}")

    if 'sex' in data:
        sex_df = data['sex']
        print(f"\nSex-Actionability Associations:")
        print(f"  Total cancer types tested: {len(sex_df)}")
        print(f"  Significant (FDR < 0.05): {sex_df['significant_fdr'].sum()}")
        if 'odds_ratio' in sex_df.columns:
            print(f"  Odds ratio range: {sex_df['odds_ratio'].min():.3f} to {sex_df['odds_ratio'].max():.3f}")

    if 'race_eth' in data:
        race_df = data['race_eth']
        print(f"\nRace/Ethnicity-Actionability Associations:")
        print(f"  Total tests: {len(race_df)}")
        print(f"  Significant (FDR < 0.05): {race_df['significant_fdr'].sum()}")
        if 'cramers_v' in race_df.columns:
            print(f"  Cramér's V range: {race_df['cramers_v'].min():.3f} to {race_df['cramers_v'].max():.3f}")

    if 'sample_type' in data:
        sample_df = data['sample_type']
        print(f"\nSample Type-Actionability Associations:")
        print(f"  Total tests: {len(sample_df)}")
        print(f"  Significant (FDR < 0.05): {sample_df['significant_fdr'].sum()}")
        if 'cramers_v' in sample_df.columns:
            print(f"  Cramér's V range: {sample_df['cramers_v'].min():.3f} to {sample_df['cramers_v'].max():.3f}")

    if 'technical' in data:
        tech_df = data['technical']
        print(f"\nTechnical Feature-Actionability Associations:")
        print(f"  Total tests: {len(tech_df)}")
        print(f"  Significant (FDR < 0.05): {tech_df['significant_fdr'].sum()}")
        if 'cramers_v' in tech_df.columns:
            print(f"  Cramér's V range: {tech_df['cramers_v'].min():.3f} to {tech_df['cramers_v'].max():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Create publication-ready visualizations for post-hoc statistical analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect input from latest run
    python visualize_posthoc_results.py

    # Specify input and output directories
    python visualize_posthoc_results.py --input-dir runs/run_20260127/tier7_posthoc_statistics --output-dir runs/run_20260127/tier7_posthoc_statistics/visualizations
        """
    )

    parser.add_argument('--input-dir', type=str, default=None,
                       help='Directory containing post-hoc analysis results (auto-detects if not specified)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for visualizations (defaults to input-dir/visualizations)')

    args = parser.parse_args()

    # Determine input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = get_default_input_dir()
        if input_dir is None:
            print("ERROR: No post-hoc results found. Run analyze_shap_feature_associations.py first.")
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
    print("Loading post-hoc analysis results...")
    print("=" * 80)

    data = load_data(input_dir)

    if not data:
        print("ERROR: No post-hoc results found in input directory")
        return 1

    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating visualizations...")
    print("=" * 80)

    # Age plots
    if 'age' in data:
        plot_age_forest(data['age'], output_dir)
        plot_age_volcano(data['age'], output_dir)

    # Sex forest plot
    if 'sex' in data:
        plot_sex_forest(data['sex'], output_dir)

    # Race/ethnicity heatmap
    if 'race_eth' in data:
        plot_categorical_heatmap(
            data['race_eth'], output_dir,
            'fig4_race_ethnicity_heatmap',
            "Race/Ethnicity-Actionability Associations\n(Effect Sizes: Cramér's V)",
            feature_type='demographic'
        )

    # Sample type heatmap
    if 'sample_type' in data:
        plot_categorical_heatmap(
            data['sample_type'], output_dir,
            'fig5_sample_type_heatmap',
            "Sample Type-Actionability Associations\n(Effect Sizes: Cramér's V)",
            feature_type='demographic'
        )

    # Technical features heatmap
    if 'technical' in data:
        plot_categorical_heatmap(
            data['technical'], output_dir,
            'fig6_technical_heatmap',
            "Technical Feature-Actionability Associations\n(Effect Sizes: Cramér's V)",
            feature_type='technical'
        )

    # Combined summary
    plot_combined_summary(data, output_dir)

    # Print summary statistics
    print_summary_statistics(data)

    print(f"\n" + "=" * 80)
    print(f"All visualizations saved to: {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
