"""Plots of institutional variability in actionability rate.

1. Bar chart ranking cancer types by across-institution coefficient of variation (CV).
2. Forest plots of per-institution actionability rates (representative 2x2 panel and
   all eligible cancer types in 2x3 parts).
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
from figure_style import (
    apply_publication_style, get_palette, get_color, get_neutral,
    panel_size_in, MM_PER_INCH,
)

# Panels are authored at their on-page size so the style's point sizes are the printed sizes.
DEFAULT_CV_WIDTH_MM = 68.0
DEFAULT_FOREST_WIDTH_MM = 108.5

# Content-driven panel height: header + n_rows x row pitch.
CV_ROW_MM = 5.0             # one bar per cancer type
CV_HEADER_MM = 16.0
FOREST_ROW_MM = 3.4         # one row per institution
FOREST_HEADER_MM = 22.0
# The two panels are authored at one shared height so they sit flush side by side.
PANEL_PAD = 0.3

DB_PATH = get_default_db_path()
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "analysis_output" / "simple_institutional_plots"

MIN_ACTIONABLE_EVENTS = 250
MIN_INSTITUTION_SAMPLES = 30

# Qualitative colors from the shared palette.
COLORS = get_palette("qualitative")

# DATA LOADING AND PROCESSING

def load_data(db_path, min_events=250, groupings_file=None):
    """Load eligible cancer types and patient data."""
    conn = sqlite3.connect(db_path)

    # Get eligible cancer types
    query = f"""
    SELECT ONCOTREE_CODE
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE
    HAVING SUM(isOncoKbLevel1) >= {min_events}
    """
    cancer_types = pd.read_sql_query(query, conn)['ONCOTREE_CODE'].tolist()

    if groupings_file:
        gpath = Path(groupings_file)
        if not gpath.exists():
            conn.close()
            raise FileNotFoundError(f"--groupings-file does not exist: {gpath}")
        canonical = pd.read_csv(gpath)['ONCOTREE_CODE'].astype(str).tolist()
        dropped = sorted(set(cancer_types) - set(canonical))
        cancer_types = [c for c in canonical if c in set(cancer_types)]
        missing = sorted(set(canonical) - set(cancer_types))
        print(f"Cohort from {gpath}: {len(cancer_types)} cancer types")
        if dropped:
            print(f"  Dropped by the groupings file: {', '.join(dropped)}")
        if missing:
            # The groupings file names a code the SQL screen did not return.
            print(f"  WARNING: in groupings but not in the SQL screen: {', '.join(missing)}")
        if not cancer_types:
            conn.close()
            raise ValueError("groupings file matched no cancer types from the database.")

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

        # Fold-range: highest / lowest per-institution actionability rate.
        min_rate = float(np.min(rates))
        max_rate = float(np.max(rates))
        fold_range = (max_rate / min_rate) if min_rate > 0 else float('inf')

        metrics.append({
            'ONCOTREE_CODE': cancer_type,
            'n_institutions': n_institutions,
            'mean_actionable_rate': float(np.mean(rates)),
            'cv': cv,
            'min_actionable_rate': min_rate,
            'max_actionable_rate': max_rate,
            'actionability_fold_range': fold_range,
        })

    variability_metrics = pd.DataFrame(metrics)

    return grouped, variability_metrics


def save_cv_metrics_csv(variability_metrics, output_dir):
    """Write per-cancer CV and fold-range to CSV."""
    out = variability_metrics.sort_values('cv', ascending=False)
    path = output_dir / 'institutional_cv_by_cancer.csv'
    out.to_csv(path, index=False)
    print(f"  CV + fold-range table: {path}")
    return path


# PLOTTING FUNCTIONS

def cv_ranking_height_mm(variability_metrics):
    """Content-driven height of the CV-ranking panel: one bar per cancer type."""
    return CV_HEADER_MM + len(variability_metrics) * CV_ROW_MM


def plot_cv_ranking_uncolored(variability_metrics, output_dir,
                              width_mm=DEFAULT_CV_WIDTH_MM, height_mm=None):
    """Bar chart showing CV ranking (uncolored version)."""

    sorted_metrics = variability_metrics.sort_values('cv', ascending=False)

    apply_publication_style()
    if height_mm is None:
        height_mm = cv_ranking_height_mm(variability_metrics)
    fig, ax = plt.subplots(figsize=panel_size_in(width_mm, height_mm=height_mm))

    # Bar plot
    bars = ax.barh(
        np.arange(len(sorted_metrics)),
        sorted_metrics['cv'],
        color=get_color("okabe_ito.blue"),
        edgecolor=get_neutral('ink'),
        linewidth=0.4,
        alpha=0.7
    )

    # Add value labels
    max_cv = sorted_metrics['cv'].max()
    for i, cv in enumerate(sorted_metrics['cv']):
        if cv > 0.08:
            # Inside bar for longer bars
            ax.text(cv - 0.01, i, f'{cv:.3f}', va='center', ha='right',
                   fontsize=5, fontweight='bold', color=get_neutral('on_fill'))
        else:
            # Outside bar for short bars
            ax.text(cv + 0.008, i, f'{cv:.3f}', va='center', ha='left',
                   fontsize=5, fontweight='bold', color=get_neutral('ink'))

    # Formatting
    ax.set_yticks(np.arange(len(sorted_metrics)))
    ax.set_yticklabels(sorted_metrics['ONCOTREE_CODE'], fontweight='bold')
    ax.set_xlabel('Coefficient of Variation (CV)', fontweight='bold')
    ax.set_ylabel('Cancer Type', fontweight='bold')
    ax.set_xlim(0, max_cv * 1.10)
    ax.grid(axis='x', alpha=0.3)

    # Save at the authored size; a tight crop would trim a panel-specific margin.
    plt.tight_layout(pad=PANEL_PAD)
    plt.savefig(output_dir / 'fig5_cv_ranking_uncolored.pdf', dpi=300,
                bbox_inches=fig.bbox_inches)
    plt.savefig(output_dir / 'fig5_cv_ranking_uncolored.png', dpi=300,
                bbox_inches=fig.bbox_inches)
    plt.close()


def _plot_forest_panel(ax, subset, color, cancer_type, cv,
                       show_xlabel=True, show_ylabel=True):
    """Draw one institution forest panel onto ``ax``."""
    subset = subset.sort_values('actionable_rate', ascending=True)
    y_positions = np.arange(len(subset))

    for i, (_, row) in enumerate(subset.iterrows()):
        ax.plot(
            [row['ci_lower'], row['ci_upper']], [i, i], 'o-',
            color=color, linewidth=0.9, markersize=3, alpha=0.7,
        )
    ax.scatter(
        subset['actionable_rate'], y_positions, s=14, color=color,
        edgecolors=get_neutral('ink'), linewidths=0.4, zorder=3, alpha=0.9,
    )

    overall_mean = subset['actionable_rate'].mean()
    # Reference line in neutral ink; it is not a data series.
    ax.axvline(overall_mean, color=get_neutral('rule'), linestyle='--', linewidth=0.8,
               alpha=0.5, label=f'Mean: {overall_mean:.2f}')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(subset['SUBJECT_CENTER'].str[:15])
    if show_xlabel:
        ax.set_xlabel('Actionability Rate (95% CI)', fontweight='bold')
    if show_ylabel:
        ax.set_ylabel('Institution', fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.grid(axis='x', alpha=0.3)
    # Two-part heading: cancer type, then CV.
    ax.set_title(cancer_type, fontweight='bold', pad=8)
    ax.text(0.5, 1.008, f'CV = {cv:.3f}', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=6, color=get_neutral('annotation'))
    ax.legend(loc='lower right')


def plot_forest_all_parts(grouped_stats, variability_metrics, output_dir):
    """Forest plots for ALL cancer types across separate figures (Part 1-N)."""

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
            cv = variability_metrics.loc[
                variability_metrics['ONCOTREE_CODE'] == cancer_type, 'cv'].values[0]
            _plot_forest_panel(axes[idx], subset, COLORS[idx % len(COLORS)],
                               cancer_type, cv)

        # Hide unused subplots if last figure has fewer than 6
        for idx in range(len(cancer_subset), n_per_fig):
            axes[idx].axis('off')

        # The part number is in the filename.
        plt.tight_layout()
        plt.savefig(output_dir / f'fig4_extended_forest_plots_part{fig_num + 1}.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'fig4_extended_forest_plots_part{fig_num + 1}.png', dpi=300, bbox_inches='tight')
        plt.close()


# Expected representative cancer types (two lowest-CV, two highest-CV).
EXPECTED_REPRESENTATIVE = ["ODG", "BRCA", "COAD", "HGSOC"]


def select_representative_cancers(variability_metrics):
    """Return (low_two, high_two): the 2 lowest-CV and 2 highest-CV cancer types."""
    ranked = variability_metrics.sort_values('cv', ascending=True)['ONCOTREE_CODE'].tolist()
    if len(ranked) < 4:
        raise ValueError(f"Need >=4 cancer types with a defined CV; got {len(ranked)}.")
    return ranked[:2], ranked[-2:]


def resolve_representative(variability_metrics, representative=None):
    """Return (top_row, bottom_row) for the representative 2x2 forest panel."""
    if representative is not None:
        if len(representative) != 4:
            raise ValueError("--representative-cancers needs exactly 4 codes "
                             "(2 low-CV for the top row, 2 high-CV for the bottom).")
        top_row, bottom_row = list(representative[:2]), list(representative[2:])
    else:
        low_two, high_two = select_representative_cancers(variability_metrics)
        top_row, bottom_row = list(low_two), list(high_two)
        picked = top_row + bottom_row
        if set(picked) != set(EXPECTED_REPRESENTATIVE):
            absent = [c for c in EXPECTED_REPRESENTATIVE
                      if c not in set(variability_metrics['ONCOTREE_CODE'])]
            print(f"  [note] data-driven CV pick {picked} != expected "
                  f"{EXPECTED_REPRESENTATIVE}; plotting the data-driven pick "
                  f"(pass --representative-cancers to override).")
            if absent:
                # Distinguishes a cohort change from a shift in the CV ranking.
                print(f"  [note] expected code(s) {absent} are not in this cohort at "
                      f"all, so the pick had to change; this is a cohort change, "
                      f"not CV drift.")

    order = top_row + bottom_row
    avail = set(variability_metrics['ONCOTREE_CODE'])
    missing = [c for c in order if c not in avail]
    if missing:
        raise ValueError("representative cancer(s) not in the eligible set "
                         f"(need >=2 institutions with >= {MIN_INSTITUTION_SAMPLES} "
                         f"samples): {missing}")
    return top_row, bottom_row


def representative_forest_height_mm(grouped_stats, top_row, bottom_row):
    """Height of the 2x2 forest panel, driven by the number of institutions per grid row."""
    n_inst = {ct: int((grouped_stats['ONCOTREE_CODE'] == ct).sum())
              for ct in top_row + bottom_row}
    n_rows = max(n_inst[c] for c in top_row) + max(n_inst[c] for c in bottom_row)
    return FOREST_HEADER_MM + n_rows * FOREST_ROW_MM


def plot_representative_cv_2x2(grouped_stats, variability_metrics, output_dir,
                               representative=None,
                               width_mm=DEFAULT_FOREST_WIDTH_MM, height_mm=None):
    """Single 2x2 forest figure for the representative cancer types."""
    top_row, bottom_row = resolve_representative(variability_metrics, representative)
    order = top_row + bottom_row

    apply_publication_style()
    # Authored at its slot width; height defaults to the content-driven rule.
    n_inst = {ct: int((grouped_stats['ONCOTREE_CODE'] == ct).sum()) for ct in order}
    rows_top = max(n_inst[ct] for ct in top_row)
    rows_bottom = max(n_inst[ct] for ct in bottom_row)
    if height_mm is None:
        height_mm = representative_forest_height_mm(grouped_stats, top_row, bottom_row)
    fig, axes = plt.subplots(
        2, 2,
        figsize=panel_size_in(width_mm, height_mm=height_mm),
        gridspec_kw={'height_ratios': [rows_top, rows_bottom]},
    )
    axes = axes.flatten()
    for idx, cancer_type in enumerate(order):
        subset = grouped_stats[grouped_stats['ONCOTREE_CODE'] == cancer_type].copy()
        cv = variability_metrics.loc[
            variability_metrics['ONCOTREE_CODE'] == cancer_type, 'cv'].values[0]
        _plot_forest_panel(
            axes[idx], subset, COLORS[idx % len(COLORS)], cancer_type, cv,
            show_xlabel=(idx >= 2),       # bottom row only
            show_ylabel=(idx % 2 == 0),   # left column only
        )

    plt.tight_layout(pad=PANEL_PAD)
    plt.savefig(output_dir / 'fig4_representative_cv_2x2.pdf', dpi=300,
                bbox_inches=fig.bbox_inches)   # full figure box
    plt.savefig(output_dir / 'fig4_representative_cv_2x2.png', dpi=300,
                bbox_inches=fig.bbox_inches)
    plt.close()
    print(f"  representative CV 2x2: top row (low CV)={top_row}  "
          f"bottom row (high CV)={bottom_row}")


# MAIN EXECUTION

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for plots (default: {DEFAULT_OUTPUT_DIR})',
    )
    parser.add_argument(
        '--representative-cancers',
        nargs=4,
        metavar=('LOW1', 'LOW2', 'HIGH1', 'HIGH2'),
        default=None,
        help='Override the Fig 3C 2x2 panel cancer types: 2 low-CV (top row) '
             'then 2 high-CV (bottom row). Default: data-driven (2 lowest + 2 '
             'highest CV).',
    )
    parser.add_argument(
        '--groupings-file',
        type=str,
        default=None,
        help='Path to a run\'s TIER 2a cancer_type_group_definitions.csv. Restricts '
             'the figure to that run\'s cohort instead of re-deriving eligibility '
             'from the database. Required to reflect --exclude-cancer-types.',
    )
    parser.add_argument(
        '--cv-width-mm',
        type=float,
        default=DEFAULT_CV_WIDTH_MM,
        help=f'On-page width of the Fig 3B CV-ranking panel, mm (default: '
             f'{DEFAULT_CV_WIDTH_MM}). Must match its slot in assemble_figure3.py '
             f'or the type scale stops being literal.',
    )
    parser.add_argument(
        '--forest-width-mm',
        type=float,
        default=DEFAULT_FOREST_WIDTH_MM,
        help=f'On-page width of the Fig 3C 2x2 forest panel, mm (default: '
             f'{DEFAULT_FOREST_WIDTH_MM}).',
    )
    parser.add_argument(
        '--no-match-heights',
        action='store_true',
        help='Author the CV-ranking and 2x2 forest panels at their own content-driven '
             'heights instead of one shared height. The default (shared) keeps the '
             'single-row institutional figure flush top and bottom.',
    )
    parser.add_argument(
        '--table-only',
        action='store_true',
        help='Write institutional_cv_by_cancer.csv and exit; skip all plots.',
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(DB_PATH, MIN_ACTIONABLE_EVENTS, groupings_file=args.groupings_file)
    grouped_stats, variability_metrics = calculate_stats(df, MIN_INSTITUTION_SAMPLES)

    save_cv_metrics_csv(variability_metrics, args.output_dir)
    if args.table_only:
        return
    # Main-text representative 2x2 (2 lowest-CV top, 2 highest-CV bottom).
    top_row, bottom_row = resolve_representative(variability_metrics,
                                                 args.representative_cancers)
    cv_h = cv_ranking_height_mm(variability_metrics)
    forest_h = representative_forest_height_mm(grouped_stats, top_row, bottom_row)
    if args.no_match_heights:
        cv_h_used, forest_h_used = cv_h, forest_h
    else:
        cv_h_used = forest_h_used = max(cv_h, forest_h)
    print(f"  panel heights (mm): content CV={cv_h:.1f} forest={forest_h:.1f} -> "
          f"authored CV={cv_h_used:.1f} forest={forest_h_used:.1f}")
    plot_cv_ranking_uncolored(variability_metrics, args.output_dir,
                              width_mm=args.cv_width_mm, height_mm=cv_h_used)
    plot_representative_cv_2x2(grouped_stats, variability_metrics, args.output_dir,
                               representative=top_row + bottom_row,
                               width_mm=args.forest_width_mm, height_mm=forest_h_used)
    # Supplement: all eligible cancer types, split across 2x3 parts.
    plot_forest_all_parts(grouped_stats, variability_metrics, args.output_dir)
    print(f"Plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
