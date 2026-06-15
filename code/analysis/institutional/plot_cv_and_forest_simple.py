"""
Simple script to generate two key institutional variability plots:
1. CV ranking bar chart (uncolored)
2. Forest plots for all eligible cancer types (split into 2x3 panels)
"""

import argparse
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path
from figure_style import apply_publication_style, get_palette, get_color

DB_PATH = get_default_db_path()
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "analysis_output" / "simple_institutional_plots"

MIN_ACTIONABLE_EVENTS = 250
MIN_INSTITUTION_SAMPLES = 30

# Per-subplot rotation colors from the shared Okabe-Ito palette.
COLORS = get_palette("qualitative")

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_data(db_path, min_events=250):
    """Load eligible cancer types and patient data"""
    conn = sqlite3.connect(db_path)

    # Get eligible cancer types
    query = f"""
    SELECT ONCOTREE_CODE
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE
    HAVING SUM(isOncoKbLevel1) >= {min_events}
    """
    cancer_types = pd.read_sql_query(query, conn)['ONCOTREE_CODE'].tolist()

    # Load patient data for eligible cancer types
    cancer_list = "','".join(cancer_types)
    query = f"""
    SELECT ONCOTREE_CODE, SUBJECT_CENTER, isOncoKbLevel1
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IN ('{cancer_list}')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def wilson_ci(n_success, n_total, alpha=0.05):
    """Calculate Wilson score confidence interval"""
    if n_total == 0:
        return 0, 0

    p = n_success / n_total
    z = stats.norm.ppf(1 - alpha/2)

    denominator = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denominator
    adjustment = z * np.sqrt((p * (1 - p) / n_total) + (z**2 / (4 * n_total**2))) / denominator

    ci_lower = max(0, centre - adjustment)
    ci_upper = min(1, centre + adjustment)

    return ci_lower, ci_upper


def calculate_stats(df, min_samples=30):
    """Calculate institutional statistics and variability metrics"""

    # Group by cancer type and institution
    grouped = df.groupby(['ONCOTREE_CODE', 'SUBJECT_CENTER']).agg({
        'isOncoKbLevel1': ['sum', 'count', 'mean']
    }).reset_index()
    grouped.columns = ['ONCOTREE_CODE', 'SUBJECT_CENTER', 'n_actionable', 'n_total', 'actionable_rate']

    # Filter institutions with minimum samples
    grouped = grouped[grouped['n_total'] >= min_samples].copy()

    # Add confidence intervals
    grouped['ci_lower'] = grouped.apply(
        lambda row: wilson_ci(row['n_actionable'], row['n_total'])[0], axis=1
    )
    grouped['ci_upper'] = grouped.apply(
        lambda row: wilson_ci(row['n_actionable'], row['n_total'])[1], axis=1
    )

    # Calculate variability metrics per cancer type
    metrics = []
    for cancer_type in grouped['ONCOTREE_CODE'].unique():
        subset = grouped[grouped['ONCOTREE_CODE'] == cancer_type]
        rates = subset['actionable_rate'].values
        n_institutions = len(subset)

        if n_institutions < 2:
            continue

        cv = np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else 0

        metrics.append({
            'ONCOTREE_CODE': cancer_type,
            'n_institutions': n_institutions,
            'mean_actionable_rate': np.mean(rates),
            'cv': cv
        })

    variability_metrics = pd.DataFrame(metrics)

    return grouped, variability_metrics


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_cv_ranking_uncolored(variability_metrics, output_dir):
    """Bar chart showing CV ranking (uncolored version)"""

    sorted_metrics = variability_metrics.sort_values('cv', ascending=False)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(4, 10))

    # Bar plot
    bars = ax.barh(
        np.arange(len(sorted_metrics)),
        sorted_metrics['cv'],
        color=get_color("okabe_ito.blue"),
        edgecolor='black',
        linewidth=1.5,
        alpha=0.7
    )

    # Add value labels
    max_cv = sorted_metrics['cv'].max()
    for i, cv in enumerate(sorted_metrics['cv']):
        if cv > 0.08:
            # Inside bar for longer bars
            ax.text(cv - 0.01, i, f'{cv:.3f}', va='center', ha='right',
                   fontsize=9, fontweight='bold', color='white')
        else:
            # Outside bar for short bars
            ax.text(cv + 0.008, i, f'{cv:.3f}', va='center', ha='left',
                   fontsize=9, fontweight='bold', color='black')

    # Formatting
    ax.set_yticks(np.arange(len(sorted_metrics)))
    ax.set_yticklabels(sorted_metrics['ONCOTREE_CODE'], fontsize=11, fontweight='bold')
    ax.set_xlabel('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_xlim(0, max_cv * 1.10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_cv_ranking_uncolored.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_cv_ranking_uncolored.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_forest_all_parts(grouped_stats, variability_metrics, output_dir):
    """Forest plots for ALL cancer types across 4 separate figures (Part 1-4)"""

    # Sort all cancer types by CV (high to low)
    all_cancers = variability_metrics.sort_values('cv', ascending=False)['ONCOTREE_CODE'].tolist()

    # Split into groups of 6 (2x3 grid per figure)
    n_per_fig = 6
    n_figs = int(np.ceil(len(all_cancers) / n_per_fig))

    for fig_num in range(n_figs):
        start_idx = fig_num * n_per_fig
        end_idx = min((fig_num + 1) * n_per_fig, len(all_cancers))
        cancer_subset = all_cancers[start_idx:end_idx]

        # Create 2x3 grid
        apply_publication_style()
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        for idx, cancer_type in enumerate(cancer_subset):
            subset = grouped_stats[grouped_stats['ONCOTREE_CODE'] == cancer_type].copy()
            subset = subset.sort_values('actionable_rate', ascending=True)

            ax = axes[idx]
            y_positions = np.arange(len(subset))

            # Plot confidence intervals
            for i, (_, row) in enumerate(subset.iterrows()):
                ax.plot(
                    [row['ci_lower'], row['ci_upper']],
                    [i, i],
                    'o-',
                    color=COLORS[idx % len(COLORS)],
                    linewidth=2,
                    markersize=8,
                    alpha=0.7
                )

            # Plot point estimates
            ax.scatter(
                subset['actionable_rate'],
                y_positions,
                s=150,
                color=COLORS[idx % len(COLORS)],
                edgecolors='black',
                linewidths=2,
                zorder=3,
                alpha=0.9
            )

            # Add mean line
            overall_mean = subset['actionable_rate'].mean()
            ax.axvline(overall_mean, color='red', linestyle='--', linewidth=2,
                      alpha=0.5, label=f'Mean: {overall_mean:.2f}')

            # Formatting
            ax.set_yticks(y_positions)
            ax.set_yticklabels(subset['SUBJECT_CENTER'].str[:15], fontsize=9)
            ax.set_xlabel('Actionability Rate (95% CI)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Institution', fontsize=11, fontweight='bold')
            ax.set_xlim(-0.05, 1.05)
            ax.grid(axis='x', alpha=0.3)

            # Title with CV
            cv = variability_metrics[variability_metrics['ONCOTREE_CODE'] == cancer_type]['cv'].values[0]
            ax.set_title(f'{cancer_type} (CV={cv:.3f}, n={len(subset)} institutions)',
                        fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)

        # Hide unused subplots if last figure has fewer than 6
        for idx in range(len(cancer_subset), n_per_fig):
            axes[idx].axis('off')

        # In-figure suptitle removed per figures/style/FIGURE_STYLE_GUIDE.md
        # (part number is in the filename; description -> caption).
        plt.tight_layout()
        plt.savefig(output_dir / f'fig4_extended_forest_plots_part{fig_num + 1}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'fig4_extended_forest_plots_part{fig_num + 1}.png', dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for plots (default: {DEFAULT_OUTPUT_DIR})',
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(DB_PATH, MIN_ACTIONABLE_EVENTS)
    grouped_stats, variability_metrics = calculate_stats(df, MIN_INSTITUTION_SAMPLES)

    plot_cv_ranking_uncolored(variability_metrics, args.output_dir)
    plot_forest_all_parts(grouped_stats, variability_metrics, args.output_dir)
    print(f"Plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
