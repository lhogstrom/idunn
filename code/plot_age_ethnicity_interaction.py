"""
Simple script to explore age-ethnicity interactions across cancer types
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite"
OUTPUT_DIR = Path("experiments/demographic_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ACTIONABLE_EVENTS = 250

# ============================================================================
# DATA LOADING
# ============================================================================

def load_age_ethnicity_data(db_path, min_events=250):
    """Load age and ethnicity data for all eligible cancer types"""

    conn = sqlite3.connect(db_path)

    # Get eligible cancer types
    query_types = f"""
    SELECT ONCOTREE_CODE
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE
    HAVING SUM(isOncoKbLevel1) >= {min_events}
    """
    cancer_types = pd.read_sql_query(query_types, conn)['ONCOTREE_CODE'].tolist()

    # Load demographic data
    cancer_list = "','".join(cancer_types)
    query = f"""
    SELECT
        ONCOTREE_CODE,
        AGE_AT_SEQ_REPORT,
        ETHNICITY,
        RACE,
        isOncoKbLevel1
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IN ('{cancer_list}')
    AND AGE_AT_SEQ_REPORT IS NOT NULL
    AND ETHNICITY IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_age_by_ethnicity_violin(df, output_dir):
    """Violin plot: Age distribution by ethnicity across cancer types"""

    # Get top cancer types by sample size
    top_cancers = df['ONCOTREE_CODE'].value_counts().head(12).index.tolist()
    df_top = df[df['ONCOTREE_CODE'].isin(top_cancers)]

    # Get major ethnicity categories
    major_ethnicities = df['ETHNICITY'].value_counts().head(3).index.tolist()
    df_filtered = df_top[df_top['ETHNICITY'].isin(major_ethnicities)]

    fig, ax = plt.subplots(figsize=(16, 10))

    sns.violinplot(data=df_filtered, x='ONCOTREE_CODE', y='AGE_AT_SEQ_REPORT',
                   hue='ETHNICITY', ax=ax, palette='Set2', split=False)

    ax.set_title('Age Distribution by Ethnicity Across Cancer Types',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Age at Sequencing', fontsize=12, fontweight='bold')
    ax.legend(title='Ethnicity', fontsize=10)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'age_ethnicity_violin_plot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'age_ethnicity_violin_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_age_ethnicity_boxplot_grid(df, output_dir):
    """Faceted boxplots: Age by ethnicity for each cancer type"""

    # Get top 12 cancer types
    top_cancers = df['ONCOTREE_CODE'].value_counts().head(12).index.tolist()
    df_top = df[df['ONCOTREE_CODE'].isin(top_cancers)]

    # Get major ethnicities
    major_ethnicities = df['ETHNICITY'].value_counts().head(4).index.tolist()
    df_filtered = df_top[df_top['ETHNICITY'].isin(major_ethnicities)]

    # Create faceted plot
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    for idx, cancer in enumerate(top_cancers):
        cancer_df = df_filtered[df_filtered['ONCOTREE_CODE'] == cancer]

        ax = axes[idx]
        sns.boxplot(data=cancer_df, x='ETHNICITY', y='AGE_AT_SEQ_REPORT',
                   ax=ax, palette='Set3')

        ax.set_title(f'{cancer} (n={len(cancer_df)})', fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Age' if idx % 4 == 0 else '')
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    plt.suptitle('Age Distribution by Ethnicity - Top 12 Cancer Types',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'age_ethnicity_boxplot_grid.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'age_ethnicity_boxplot_grid.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_mean_age_heatmap(df, output_dir):
    """Heatmap: Mean age by cancer type and ethnicity"""

    # Get major ethnicities
    major_ethnicities = df['ETHNICITY'].value_counts().head(4).index.tolist()
    df_filtered = df[df['ETHNICITY'].isin(major_ethnicities)]

    # Calculate mean age
    age_matrix = df_filtered.groupby(['ONCOTREE_CODE', 'ETHNICITY'])['AGE_AT_SEQ_REPORT'].mean().unstack()

    # Sort by overall mean age
    age_matrix['overall'] = age_matrix.mean(axis=1)
    age_matrix = age_matrix.sort_values('overall', ascending=False).drop('overall', axis=1)

    fig, ax = plt.subplots(figsize=(10, 14))

    sns.heatmap(age_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r',
                cbar_kws={'label': 'Mean Age (years)'},
                linewidths=0.5, ax=ax)

    ax.set_title('Mean Age at Sequencing by Cancer Type and Ethnicity',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Ethnicity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'age_ethnicity_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'age_ethnicity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_age_ethnicity_scatter_actionability(df, output_dir):
    """Scatter plot: Age vs ethnicity diversity, colored by actionability rate"""

    # Calculate statistics per cancer type
    stats = []
    for cancer in df['ONCOTREE_CODE'].unique():
        cancer_df = df[df['ONCOTREE_CODE'] == cancer]

        # Ethnicity diversity (entropy)
        ethnicity_counts = cancer_df['ETHNICITY'].value_counts()
        ethnicity_entropy = -sum((ethnicity_counts / len(cancer_df)) *
                                np.log(ethnicity_counts / len(cancer_df)))

        stats.append({
            'cancer_type': cancer,
            'mean_age': cancer_df['AGE_AT_SEQ_REPORT'].mean(),
            'ethnicity_entropy': ethnicity_entropy,
            'actionable_rate': cancer_df['isOncoKbLevel1'].mean(),
            'n_patients': len(cancer_df)
        })

    stats_df = pd.DataFrame(stats)

    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(stats_df['mean_age'], stats_df['ethnicity_entropy'],
                        s=stats_df['n_patients']/50,
                        c=stats_df['actionable_rate'],
                        cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)

    # Add cancer type labels
    for _, row in stats_df.iterrows():
        ax.annotate(row['cancer_type'],
                   (row['mean_age'], row['ethnicity_entropy']),
                   fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('Mean Age at Sequencing', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ethnicity Diversity (Entropy)', fontsize=12, fontweight='bold')
    ax.set_title('Age vs Ethnicity Diversity by Cancer Type\n(Point size = sample size, color = actionability rate)',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Actionability Rate', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'age_ethnicity_diversity_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'age_ethnicity_diversity_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_age_distribution_by_ethnicity_kde(df, output_dir):
    """KDE plot: Age distribution for each ethnicity category"""

    # Get major ethnicities
    major_ethnicities = df['ETHNICITY'].value_counts().head(4).index.tolist()
    df_filtered = df[df['ETHNICITY'].isin(major_ethnicities)]

    fig, ax = plt.subplots(figsize=(14, 8))

    for ethnicity in major_ethnicities:
        ethnicity_df = df_filtered[df_filtered['ETHNICITY'] == ethnicity]
        ethnicity_df['AGE_AT_SEQ_REPORT'].plot.kde(ax=ax, linewidth=2, label=ethnicity)

    ax.set_xlabel('Age at Sequencing', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Age Distribution by Ethnicity (All Cancer Types)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Ethnicity', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'age_ethnicity_kde.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'age_ethnicity_kde.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_age_ethnicity_kde_faceted(df, output_dir):
    """Faceted KDE plot: Age distribution by ethnicity for each cancer type"""

    # Get major ethnicities
    major_ethnicities = df['ETHNICITY'].value_counts().head(3).index.tolist()
    df_filtered = df[df['ETHNICITY'].isin(major_ethnicities)]

    # Get all cancer types
    cancer_types = sorted(df['ONCOTREE_CODE'].unique())
    n_cancers = len(cancer_types)

    # Create faceted plot (7 rows x 3 columns for 21 cancer types)
    fig, axes = plt.subplots(7, 3, figsize=(18, 24))
    axes = axes.flatten()

    for idx, cancer in enumerate(cancer_types):
        cancer_df = df_filtered[df_filtered['ONCOTREE_CODE'] == cancer]

        ax = axes[idx]

        # Plot KDE for each ethnicity
        for ethnicity in major_ethnicities:
            ethnicity_data = cancer_df[cancer_df['ETHNICITY'] == ethnicity]['AGE_AT_SEQ_REPORT']
            if len(ethnicity_data) > 5:  # Need enough data for KDE
                ethnicity_data.plot.kde(ax=ax, linewidth=2, label=ethnicity)

        ax.set_title(f'{cancer} (n={len(cancer_df)})', fontweight='bold', fontsize=10)
        ax.set_xlabel('Age' if idx >= 18 else '', fontsize=9)
        ax.set_ylabel('Density' if idx % 3 == 0 else '', fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)

        # Only show legend for first plot
        if idx == 0:
            ax.legend(title='Ethnicity', fontsize=8, title_fontsize=9)
        else:
            ax.legend().set_visible(False)

    # Hide unused subplots
    for idx in range(n_cancers, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Age Distribution by Ethnicity - All Cancer Types',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'age_ethnicity_kde_faceted.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'age_ethnicity_kde_faceted.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hispanic_age_difference(df, output_dir):
    """Forest plot: Mean age difference between Hispanic and Non-Hispanic patients"""

    # Calculate statistics for each cancer type
    results = []

    for cancer in sorted(df['ONCOTREE_CODE'].unique()):
        cancer_df = df[df['ONCOTREE_CODE'] == cancer]

        # Split by Hispanic vs Non-Hispanic (using actual database labels)
        hispanic = cancer_df[cancer_df['ETHNICITY'] == 'Spanish/Hispanic']['AGE_AT_SEQ_REPORT'].dropna()
        non_hispanic = cancer_df[cancer_df['ETHNICITY'] == 'Non-Spanish/non-Hispanic']['AGE_AT_SEQ_REPORT'].dropna()

        # Only include if both groups have sufficient data
        if len(hispanic) >= 10 and len(non_hispanic) >= 10:
            # Calculate means
            mean_hispanic = hispanic.mean()
            mean_non_hispanic = non_hispanic.mean()
            diff = mean_hispanic - mean_non_hispanic

            # Welch's t-test (unequal variances)
            t_stat, p_value = stats.ttest_ind(hispanic, non_hispanic, equal_var=False)

            # Standard error of difference
            se_diff = np.sqrt(hispanic.var()/len(hispanic) + non_hispanic.var()/len(non_hispanic))

            # 95% CI for difference
            ci_lower = diff - 1.96 * se_diff
            ci_upper = diff + 1.96 * se_diff

            results.append({
                'cancer_type': cancer,
                'diff': diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                'n_hispanic': len(hispanic),
                'n_non_hispanic': len(non_hispanic),
                'mean_hispanic': mean_hispanic,
                'mean_non_hispanic': mean_non_hispanic
            })

    # Handle case with no results
    if len(results) == 0:
        return

    results_df = pd.DataFrame(results).sort_values('diff')

    # Apply Benjamini-Hochberg FDR correction
    reject, q_values, _, _ = multipletests(results_df['p_value'].values,
                                           method='fdr_bh', alpha=0.05)
    results_df['q_value'] = q_values
    results_df['significant'] = reject

    # Create significance markers
    def sig_marker(q):
        if q < 0.001:
            return '***'
        elif q < 0.01:
            return '**'
        elif q < 0.05:
            return '*'
        else:
            return 'ns'

    results_df['sig_marker'] = results_df['q_value'].apply(sig_marker)

    # Create forest plot
    fig, ax = plt.subplots(figsize=(10, 12))

    y_positions = np.arange(len(results_df))

    # Plot confidence intervals
    for i, row in results_df.iterrows():
        y_idx = np.where(y_positions == list(results_df.index).index(i))[0][0]
        ax.plot([row['ci_lower'], row['ci_upper']], [y_idx, y_idx],
               'o-', color='steelblue', linewidth=2, markersize=6, alpha=0.7)

    # Plot point estimates
    ax.scatter(results_df['diff'], y_positions, s=100, color='steelblue',
              edgecolors='black', linewidths=2, zorder=3, alpha=0.9)

    # Add reference line at zero
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5,
              label='No difference')

    # VERSION 1: WITHOUT significance markers (original style)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(results_df['cancer_type'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Mean Age Difference (Hispanic - Non-Hispanic) in years',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_title('Age Difference: Hispanic vs Non-Hispanic Patients\n(with 95% Confidence Intervals)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    # Add sample size annotations (no significance markers)
    for i, row in results_df.iterrows():
        y_idx = list(results_df.index).index(i)
        ax.text(ax.get_xlim()[1] * 0.95, y_idx,
               f"n={row['n_hispanic']}/{row['n_non_hispanic']}",
               va='center', ha='right', fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'hispanic_age_difference_forest_plot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'hispanic_age_difference_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # VERSION 2: WITH significance markers on the right side
    fig, ax = plt.subplots(figsize=(10, 12))

    y_positions = np.arange(len(results_df))

    # Plot confidence intervals
    for i, row in results_df.iterrows():
        y_idx = np.where(y_positions == list(results_df.index).index(i))[0][0]
        ax.plot([row['ci_lower'], row['ci_upper']], [y_idx, y_idx],
               'o-', color='steelblue', linewidth=2, markersize=6, alpha=0.7)

    # Plot point estimates
    ax.scatter(results_df['diff'], y_positions, s=100, color='steelblue',
              edgecolors='black', linewidths=2, zorder=3, alpha=0.9)

    # Add reference line at zero
    ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5,
              label='No difference')

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(results_df['cancer_type'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Mean Age Difference (Hispanic - Non-Hispanic) in years',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_title('Age Difference: Hispanic vs Non-Hispanic Patients\n(with 95% Confidence Intervals)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    # Add legend for significance levels
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='No difference'),
        Line2D([0], [0], marker='', color='w', label='Significance (FDR):'),
        Line2D([0], [0], marker='', color='w', label='*** q < 0.001'),
        Line2D([0], [0], marker='', color='w', label='** q < 0.01'),
        Line2D([0], [0], marker='', color='w', label='* q < 0.05'),
        Line2D([0], [0], marker='', color='w', label='ns q ≥ 0.05')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Add sample size annotations WITH significance markers
    for i, row in results_df.iterrows():
        y_idx = list(results_df.index).index(i)
        ax.text(ax.get_xlim()[1] * 0.95, y_idx,
               f"{row['sig_marker']}  n={row['n_hispanic']}/{row['n_non_hispanic']}",
               va='center', ha='right', fontsize=8, style='italic', fontweight='bold')

    # Add footnote
    fig.text(0.5, 0.02,
             f"P-values from Welch's t-test with Benjamini-Hochberg FDR correction ({len(results_df)} comparisons)",
             ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / 'hispanic_age_difference_forest_plot_with_sig.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'hispanic_age_difference_forest_plot_with_sig.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hispanic_age_boxplot(df, output_dir):
    """Boxplot comparison: Hispanic vs Non-Hispanic age distributions"""

    # Prepare data for boxplot
    plot_data = []
    cancer_types = []

    for cancer in sorted(df['ONCOTREE_CODE'].unique()):
        cancer_df = df[df['ONCOTREE_CODE'] == cancer]

        hispanic = cancer_df[cancer_df['ETHNICITY'] == 'Spanish/Hispanic']['AGE_AT_SEQ_REPORT'].dropna()
        non_hispanic = cancer_df[cancer_df['ETHNICITY'] == 'Non-Spanish/non-Hispanic']['AGE_AT_SEQ_REPORT'].dropna()

        if len(hispanic) >= 10 and len(non_hispanic) >= 10:
            cancer_types.append(cancer)
            for age in hispanic:
                plot_data.append({'cancer': cancer, 'age': age, 'group': 'Hispanic'})
            for age in non_hispanic:
                plot_data.append({'cancer': cancer, 'age': age, 'group': 'Non-Hispanic'})

    if len(plot_data) == 0:
        return

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(18, 10))

    # Create grouped boxplot
    sns.boxplot(data=plot_df, x='cancer', y='age', hue='group',
               palette=['#ff7f0e', '#1f77b4'], ax=ax)

    ax.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Age at Sequencing (years)', fontsize=12, fontweight='bold')
    ax.set_title('Age Distribution: Hispanic vs Non-Hispanic Patients',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Ethnicity', fontsize=11, title_fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'hispanic_age_boxplot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'hispanic_age_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hispanic_age_bar_chart(df, output_dir):
    """Bar chart: Mean age with error bars for Hispanic vs Non-Hispanic"""

    # Calculate statistics
    results = []

    for cancer in sorted(df['ONCOTREE_CODE'].unique()):
        cancer_df = df[df['ONCOTREE_CODE'] == cancer]

        hispanic = cancer_df[cancer_df['ETHNICITY'] == 'Spanish/Hispanic']['AGE_AT_SEQ_REPORT'].dropna()
        non_hispanic = cancer_df[cancer_df['ETHNICITY'] == 'Non-Spanish/non-Hispanic']['AGE_AT_SEQ_REPORT'].dropna()

        if len(hispanic) >= 10 and len(non_hispanic) >= 10:
            results.append({
                'cancer': cancer,
                'Hispanic': hispanic.mean(),
                'Hispanic_se': hispanic.std() / np.sqrt(len(hispanic)),
                'Non-Hispanic': non_hispanic.mean(),
                'Non-Hispanic_se': non_hispanic.std() / np.sqrt(len(non_hispanic))
            })

    if len(results) == 0:
        return

    results_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(18, 10))

    x = np.arange(len(results_df))
    width = 0.35

    # Plot bars with error bars
    ax.bar(x - width/2, results_df['Hispanic'], width,
          yerr=results_df['Hispanic_se'], capsize=4,
          label='Hispanic', color='#ff7f0e', alpha=0.8,
          error_kw={'linewidth': 2})

    ax.bar(x + width/2, results_df['Non-Hispanic'], width,
          yerr=results_df['Non-Hispanic_se'], capsize=4,
          label='Non-Hispanic', color='#1f77b4', alpha=0.8,
          error_kw={'linewidth': 2})

    ax.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Age at Sequencing (years)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Age Comparison: Hispanic vs Non-Hispanic Patients\n(Error bars = SEM)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['cancer'], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'hispanic_age_bar_chart.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'hispanic_age_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hispanic_age_heatmap(df, output_dir):
    """Heatmap: Mean age comparison for Hispanic vs Non-Hispanic"""

    # Calculate mean ages
    age_data = []

    for cancer in sorted(df['ONCOTREE_CODE'].unique()):
        cancer_df = df[df['ONCOTREE_CODE'] == cancer]

        hispanic = cancer_df[cancer_df['ETHNICITY'] == 'Spanish/Hispanic']['AGE_AT_SEQ_REPORT'].dropna()
        non_hispanic = cancer_df[cancer_df['ETHNICITY'] == 'Non-Spanish/non-Hispanic']['AGE_AT_SEQ_REPORT'].dropna()

        if len(hispanic) >= 10 and len(non_hispanic) >= 10:
            age_data.append({
                'Cancer Type': cancer,
                'Hispanic': hispanic.mean(),
                'Non-Hispanic': non_hispanic.mean()
            })

    if len(age_data) == 0:
        return

    age_df = pd.DataFrame(age_data).set_index('Cancer Type')

    fig, ax = plt.subplots(figsize=(6, 12))

    sns.heatmap(age_df, annot=True, fmt='.1f', cmap='RdYlBu_r',
               cbar_kws={'label': 'Mean Age (years)'},
               linewidths=0.5, ax=ax, vmin=40, vmax=70)

    ax.set_title('Mean Age Heatmap: Hispanic vs Non-Hispanic',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_xlabel('Ethnicity', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'hispanic_age_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'hispanic_age_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hispanic_age_violin(df, output_dir):
    """Violin plot: Age distribution comparison for top cancer types"""

    # Get top 12 cancer types with sufficient Hispanic representation
    top_cancers = []
    for cancer in df['ONCOTREE_CODE'].value_counts().head(15).index:
        cancer_df = df[df['ONCOTREE_CODE'] == cancer]
        hispanic = cancer_df[cancer_df['ETHNICITY'] == 'Spanish/Hispanic']
        if len(hispanic) >= 20:
            top_cancers.append(cancer)
            if len(top_cancers) >= 12:
                break

    # Prepare data
    plot_data = []
    for cancer in top_cancers:
        cancer_df = df[df['ONCOTREE_CODE'] == cancer]
        for _, row in cancer_df[cancer_df['ETHNICITY'].isin(['Spanish/Hispanic', 'Non-Spanish/non-Hispanic'])].iterrows():
            ethnicity_label = 'Hispanic' if row['ETHNICITY'] == 'Spanish/Hispanic' else 'Non-Hispanic'
            plot_data.append({
                'cancer': cancer,
                'age': row['AGE_AT_SEQ_REPORT'],
                'group': ethnicity_label
            })

    if len(plot_data) == 0:
        return

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(18, 10))

    sns.violinplot(data=plot_df, x='cancer', y='age', hue='group',
                  palette=['#ff7f0e', '#1f77b4'], ax=ax, split=False)

    ax.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Age at Sequencing (years)', fontsize=12, fontweight='bold')
    ax.set_title('Age Distribution: Hispanic vs Non-Hispanic Patients (Top Cancer Types)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Ethnicity', fontsize=11, title_fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'hispanic_age_violin.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'hispanic_age_violin.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load data
    df = load_age_ethnicity_data(DB_PATH, MIN_ACTIONABLE_EVENTS)

    # Generate plots
    plot_age_by_ethnicity_violin(df, OUTPUT_DIR)
    plot_age_ethnicity_boxplot_grid(df, OUTPUT_DIR)
    plot_mean_age_heatmap(df, OUTPUT_DIR)
    plot_age_ethnicity_scatter_actionability(df, OUTPUT_DIR)
    plot_age_distribution_by_ethnicity_kde(df, OUTPUT_DIR)
    plot_age_ethnicity_kde_faceted(df, OUTPUT_DIR)

    # Hispanic vs Non-Hispanic comparison plots
    plot_hispanic_age_difference(df, OUTPUT_DIR)
    plot_hispanic_age_boxplot(df, OUTPUT_DIR)
    plot_hispanic_age_bar_chart(df, OUTPUT_DIR)
    plot_hispanic_age_heatmap(df, OUTPUT_DIR)
    plot_hispanic_age_violin(df, OUTPUT_DIR)
