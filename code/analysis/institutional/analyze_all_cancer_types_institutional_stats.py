"""
Comprehensive Institutional Distribution Analysis for All Eligible Cancer Types

Analyzes all cancer types with ≥250 actionable events to understand:
- Institutional concentration patterns
- Actionability variance across institutions
- Sample size and distribution characteristics

This helps identify which cancer types may show institution-dependent patterns
similar to the GIST anomaly discovered in GroupKFold CV experiments.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path
from scipy import stats

# Configuration
DB_PATH = get_default_db_path()
DEFAULT_OUTPUT_DIR = Path("../../../analysis_output/all_cancer_types_institutional_analysis")
OUTPUT_DIR = None  # Set by main()

MIN_ACTIONABLE_EVENTS = 250


def query_eligible_cancer_types(db_path, min_events=250):
    """Query all cancer types meeting minimum actionable event threshold"""
    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT
        ONCOTREE_CODE,
        COUNT(*) as total_patients,
        SUM(isOncoKbLevel1) as n_actionable,
        AVG(isOncoKbLevel1) as actionable_rate
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE
    HAVING SUM(isOncoKbLevel1) >= {min_events}
    ORDER BY n_actionable DESC
    """

    eligible = pd.read_sql_query(query, conn)
    conn.close()

    return eligible


def load_cancer_type_data(db_path, cancer_type):
    """Load patient-level data for a specific cancer type"""
    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT *
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE = '{cancer_type}'
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def analyze_institutional_distribution(df, cancer_type):
    """Analyze how samples are distributed across institutions"""

    # Basic statistics
    n_patients = len(df)
    n_institutions = df['SUBJECT_CENTER'].nunique()

    # Samples per institution
    inst_counts = df['SUBJECT_CENTER'].value_counts()

    # Actionability by institution
    actionable_by_inst = df.groupby('SUBJECT_CENTER')['isOncoKbLevel1'].agg([
        'sum', 'count', 'mean'
    ]).reset_index()
    actionable_by_inst.columns = ['SUBJECT_CENTER', 'n_actionable', 'n_total', 'actionable_rate']
    actionable_by_inst = actionable_by_inst.sort_values('n_total', ascending=False)

    # Calculate concentration metrics
    # Herfindahl index (0 = perfectly distributed, 1 = all at one institution)
    proportions = inst_counts / inst_counts.sum()
    herfindahl = (proportions ** 2).sum()

    # Gini coefficient (0 = perfectly equal, 1 = perfectly unequal)
    sorted_counts = np.sort(inst_counts.values)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

    # Top institution concentration
    top1_pct = inst_counts.iloc[0] / n_patients if len(inst_counts) > 0 else 0
    top3_pct = inst_counts.iloc[:3].sum() / n_patients if len(inst_counts) >= 3 else top1_pct
    top5_pct = inst_counts.iloc[:5].sum() / n_patients if len(inst_counts) >= 5 else top3_pct

    # Actionability variance metrics
    actionable_rate_std = actionable_by_inst['actionable_rate'].std()
    actionable_rate_range = actionable_by_inst['actionable_rate'].max() - actionable_by_inst['actionable_rate'].min()
    actionable_rate_cv = actionable_rate_std / actionable_by_inst['actionable_rate'].mean() if actionable_by_inst['actionable_rate'].mean() > 0 else 0

    stats_dict = {
        'cancer_type': cancer_type,
        'n_patients': n_patients,
        'n_actionable': df['isOncoKbLevel1'].sum(),
        'n_institutions': n_institutions,
        'patients_per_institution_mean': inst_counts.mean(),
        'patients_per_institution_median': inst_counts.median(),
        'patients_per_institution_std': inst_counts.std(),
        'herfindahl_index': herfindahl,
        'gini_coefficient': gini,
        'top1_institution_pct': top1_pct,
        'top3_institutions_pct': top3_pct,
        'top5_institutions_pct': top5_pct,
        'overall_actionable_rate': df['isOncoKbLevel1'].mean(),
        'actionable_rate_std_across_institutions': actionable_rate_std,
        'actionable_rate_range_across_institutions': actionable_rate_range,
        'actionable_rate_cv_across_institutions': actionable_rate_cv,
        'groupkfold_institutions_per_fold': n_institutions // 5,
        'groupkfold_patients_per_fold': n_patients // 5
    }

    return stats_dict, inst_counts, actionable_by_inst


def create_summary_table(all_stats_df, output_dir):
    """Create formatted summary table"""

    # Sort by number of actionable events
    summary = all_stats_df.sort_values('n_actionable', ascending=False).copy()

    # Format for display
    display_cols = [
        'cancer_type',
        'n_patients',
        'n_actionable',
        'overall_actionable_rate',
        'n_institutions',
        'patients_per_institution_median',
        'herfindahl_index',
        'gini_coefficient',
        'top1_institution_pct',
        'top3_institutions_pct',
        'top5_institutions_pct',
        'actionable_rate_std_across_institutions',
        'actionable_rate_range_across_institutions'
    ]

    summary_display = summary[display_cols].copy()

    # Save detailed CSV
    summary.to_csv(output_dir / 'all_cancer_types_institutional_stats.csv', index=False)
    print(f"Saved: {output_dir / 'all_cancer_types_institutional_stats.csv'}")

    return summary


def create_concentration_heatmap(all_stats_df, output_dir):
    """Create heatmap of concentration metrics across cancer types"""

    # Select key metrics for heatmap
    heatmap_cols = [
        'herfindahl_index',
        'gini_coefficient',
        'top3_institutions_pct',
        'overall_actionable_rate',
        'actionable_rate_std_across_institutions',
        'actionable_rate_cv_across_institutions'
    ]

    # Prepare data
    heatmap_data = all_stats_df.set_index('cancer_type')[heatmap_cols].copy()

    # Sort by Herfindahl index
    heatmap_data = heatmap_data.sort_values('herfindahl_index', ascending=False)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(heatmap_data) * 0.4)))

    # Normalize each column to 0-1 for heatmap
    heatmap_normalized = heatmap_data.copy()
    for col in heatmap_cols:
        heatmap_normalized[col] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min())

    # Create heatmap
    sns.heatmap(heatmap_normalized, annot=heatmap_data, fmt='.3f',
                cmap='RdYlGn_r', cbar_kws={'label': 'Normalized Value'},
                linewidths=0.5, ax=ax)

    # Format
    ax.set_xlabel('Metric', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=11, fontweight='bold')
    ax.set_title('Institutional Concentration & Actionability Metrics\n(All Cancer Types with ≥250 Actionable Events)',
                 fontsize=12, fontweight='bold', pad=15)

    # Rename columns for display
    col_labels = [
        'Herfindahl\nIndex',
        'Gini\nCoefficient',
        'Top 3\nInst. %',
        'Overall\nActionable\nRate',
        'Actionable\nRate Std\n(across inst.)',
        'Actionable\nRate CV\n(across inst.)'
    ]
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'institutional_metrics_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'institutional_metrics_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'institutional_metrics_heatmap.pdf'}")
    plt.close()


def create_scatter_plots(all_stats_df, output_dir):
    """Create scatter plots showing relationships between metrics"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Concentration vs Actionability Variance
    ax = axes[0, 0]
    scatter = ax.scatter(all_stats_df['herfindahl_index'],
                        all_stats_df['actionable_rate_std_across_institutions'],
                        s=all_stats_df['n_patients'] / 50,
                        c=all_stats_df['overall_actionable_rate'],
                        cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5)

    # Label interesting points
    for idx, row in all_stats_df.iterrows():
        if row['actionable_rate_std_across_institutions'] > 0.15 or row['herfindahl_index'] > 0.3:
            ax.annotate(row['cancer_type'],
                       (row['herfindahl_index'], row['actionable_rate_std_across_institutions']),
                       fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('Herfindahl Index (Institutional Concentration)', fontsize=10)
    ax.set_ylabel('Actionability Rate Std (across institutions)', fontsize=10)
    ax.set_title('Concentration vs Actionability Variance', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Overall Actionable Rate', fontsize=9)

    # Plot 2: Sample Size vs Concentration
    ax = axes[0, 1]
    scatter = ax.scatter(all_stats_df['n_patients'],
                        all_stats_df['herfindahl_index'],
                        s=all_stats_df['n_actionable'] / 10,
                        c=all_stats_df['actionable_rate_std_across_institutions'],
                        cmap='Reds', alpha=0.6, edgecolors='black', linewidth=0.5)

    for idx, row in all_stats_df.iterrows():
        if row['n_patients'] < 3000 and row['herfindahl_index'] > 0.25:
            ax.annotate(row['cancer_type'],
                       (row['n_patients'], row['herfindahl_index']),
                       fontsize=8, ha='left', va='bottom')

    ax.set_xlabel('Total Patients', fontsize=10)
    ax.set_ylabel('Herfindahl Index', fontsize=10)
    ax.set_title('Sample Size vs Institutional Concentration', fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Actionability Variance', fontsize=9)

    # Plot 3: Number of Institutions vs Patients
    ax = axes[1, 0]
    scatter = ax.scatter(all_stats_df['n_institutions'],
                        all_stats_df['n_patients'],
                        s=all_stats_df['n_actionable'] / 10,
                        c=all_stats_df['herfindahl_index'],
                        cmap='plasma', alpha=0.6, edgecolors='black', linewidth=0.5)

    for idx, row in all_stats_df.iterrows():
        if row['n_patients'] < 3000:
            ax.annotate(row['cancer_type'],
                       (row['n_institutions'], row['n_patients']),
                       fontsize=7, ha='left', va='bottom')

    ax.set_xlabel('Number of Institutions', fontsize=10)
    ax.set_ylabel('Total Patients', fontsize=10)
    ax.set_title('Institutional Diversity vs Sample Size', fontsize=11, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Herfindahl Index', fontsize=9)

    # Plot 4: Actionable Rate vs Variance
    ax = axes[1, 1]
    scatter = ax.scatter(all_stats_df['overall_actionable_rate'],
                        all_stats_df['actionable_rate_std_across_institutions'],
                        s=all_stats_df['n_patients'] / 50,
                        c=all_stats_df['herfindahl_index'],
                        cmap='coolwarm', alpha=0.6, edgecolors='black', linewidth=0.5)

    # Label outliers
    for idx, row in all_stats_df.iterrows():
        if row['actionable_rate_std_across_institutions'] > 0.15:
            ax.annotate(row['cancer_type'],
                       (row['overall_actionable_rate'], row['actionable_rate_std_across_institutions']),
                       fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('Overall Actionable Rate', fontsize=10)
    ax.set_ylabel('Actionability Rate Std (across institutions)', fontsize=10)
    ax.set_title('Mean Actionability vs Variance Across Institutions', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Herfindahl Index', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'institutional_metrics_scatter_plots.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'institutional_metrics_scatter_plots.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'institutional_metrics_scatter_plots.pdf'}")
    plt.close()


def create_additional_scatter_plots(all_stats_df, output_dir):
    """Create additional scatter plot variations focusing on key institutional metrics"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Number of Patients vs Number of Institutions
    ax = axes[0, 0]
    scatter = ax.scatter(all_stats_df['n_patients'],
                        all_stats_df['n_institutions'],
                        s=all_stats_df['n_actionable'] / 10,
                        c=all_stats_df['top3_institutions_pct'],
                        cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=0.5)

    # Label all points
    for idx, row in all_stats_df.iterrows():
        ax.annotate(row['cancer_type'],
                   (row['n_patients'], row['n_institutions']),
                   fontsize=7, ha='center', va='bottom')

    ax.set_xlabel('Total Patients (log scale)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Institutions', fontsize=11, fontweight='bold')
    ax.set_title('Patient Volume vs Institutional Diversity', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Top 3 Institutions %', fontsize=10)

    # Plot 2: Top 3 Institutions % vs Overall Actionable Rate
    ax = axes[0, 1]
    scatter = ax.scatter(all_stats_df['top3_institutions_pct'],
                        all_stats_df['overall_actionable_rate'],
                        s=all_stats_df['n_patients'] / 50,
                        c=all_stats_df['n_institutions'],
                        cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)

    # Label interesting points
    for idx, row in all_stats_df.iterrows():
        if row['top3_institutions_pct'] > 0.85 or row['overall_actionable_rate'] > 0.7:
            ax.annotate(row['cancer_type'],
                       (row['top3_institutions_pct'], row['overall_actionable_rate']),
                       fontsize=7, ha='center', va='bottom')

    ax.set_xlabel('Top 3 Institutions % (Concentration)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Overall Actionable Rate', fontsize=11, fontweight='bold')
    ax.set_title('Institutional Concentration vs Actionability', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Institutions', fontsize=10)

    # Plot 3: Number of Patients vs Overall Actionable Rate
    ax = axes[0, 2]
    scatter = ax.scatter(all_stats_df['n_patients'],
                        all_stats_df['overall_actionable_rate'],
                        s=all_stats_df['n_actionable'] / 10,
                        c=all_stats_df['top3_institutions_pct'],
                        cmap='plasma', alpha=0.7, edgecolors='black', linewidth=0.5)

    # Label outliers
    for idx, row in all_stats_df.iterrows():
        if row['overall_actionable_rate'] > 0.7 or row['n_patients'] > 20000:
            ax.annotate(row['cancer_type'],
                       (row['n_patients'], row['overall_actionable_rate']),
                       fontsize=7, ha='left', va='bottom')

    ax.set_xlabel('Total Patients (log scale)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Overall Actionable Rate', fontsize=11, fontweight='bold')
    ax.set_title('Sample Size vs Actionability Rate', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Top 3 Institutions %', fontsize=10)

    # Plot 4: Number of Institutions vs Top 3 Institutions %
    ax = axes[1, 0]
    scatter = ax.scatter(all_stats_df['n_institutions'],
                        all_stats_df['top3_institutions_pct'],
                        s=all_stats_df['n_patients'] / 50,
                        c=all_stats_df['overall_actionable_rate'],
                        cmap='coolwarm', alpha=0.7, edgecolors='black', linewidth=0.5)

    # Label high concentration cancer types
    for idx, row in all_stats_df.iterrows():
        if row['top3_institutions_pct'] > 0.85:
            ax.annotate(row['cancer_type'],
                       (row['n_institutions'], row['top3_institutions_pct']),
                       fontsize=7, ha='left', va='bottom')

    ax.set_xlabel('Number of Institutions', fontsize=11, fontweight='bold')
    ax.set_ylabel('Top 3 Institutions % (Concentration)', fontsize=11, fontweight='bold')
    ax.set_title('Institutional Diversity vs Concentration', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Overall Actionable Rate', fontsize=10)

    # Plot 5: Number of Patients vs Top 3 Institutions %
    ax = axes[1, 1]
    scatter = ax.scatter(all_stats_df['n_patients'],
                        all_stats_df['top3_institutions_pct'],
                        s=all_stats_df['n_actionable'] / 10,
                        c=all_stats_df['overall_actionable_rate'],
                        cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=0.5)

    # Label all cancer types
    for idx, row in all_stats_df.iterrows():
        ax.annotate(row['cancer_type'],
                   (row['n_patients'], row['top3_institutions_pct']),
                   fontsize=6, ha='center', va='bottom')

    ax.set_xlabel('Total Patients (log scale)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Top 3 Institutions % (Concentration)', fontsize=11, fontweight='bold')
    ax.set_title('Sample Size vs Institutional Concentration', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Overall Actionable Rate', fontsize=10)

    # Plot 6: Overall Actionable Rate vs Number of Institutions
    ax = axes[1, 2]
    scatter = ax.scatter(all_stats_df['overall_actionable_rate'],
                        all_stats_df['n_institutions'],
                        s=all_stats_df['n_patients'] / 50,
                        c=all_stats_df['top3_institutions_pct'],
                        cmap='Spectral_r', alpha=0.7, edgecolors='black', linewidth=0.5)

    # Label outliers
    for idx, row in all_stats_df.iterrows():
        if row['overall_actionable_rate'] > 0.7 or row['n_institutions'] < 12:
            ax.annotate(row['cancer_type'],
                       (row['overall_actionable_rate'], row['n_institutions']),
                       fontsize=7, ha='left', va='bottom')

    ax.set_xlabel('Overall Actionable Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Institutions', fontsize=11, fontweight='bold')
    ax.set_title('Actionability vs Institutional Diversity', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Top 3 Institutions %', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'institutional_metrics_additional_scatter_plots.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'institutional_metrics_additional_scatter_plots.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'institutional_metrics_additional_scatter_plots.pdf'}")
    plt.close()


def create_standalone_patient_institution_plot(all_stats_df, output_dir):
    """Create standalone version of Patient Volume vs Institutional Diversity plot"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Scatter plot: Number of Patients vs Number of Institutions
    scatter = ax.scatter(all_stats_df['n_patients'],
                        all_stats_df['n_institutions'],
                        s=all_stats_df['n_actionable'] / 10,
                        c=all_stats_df['top3_institutions_pct'],
                        cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=1.0)

    # Label all points with improved positioning
    for idx, row in all_stats_df.iterrows():
        ax.annotate(row['cancer_type'],
                   (row['n_patients'], row['n_institutions']),
                   fontsize=9, ha='center', va='bottom',
                   fontweight='bold')

    ax.set_xlabel('Total Patients (log scale)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Institutions', fontsize=14, fontweight='bold')
    ax.set_title('Patient Volume vs Institutional Diversity\nAcross 21 Cancer Types',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax.tick_params(labelsize=11)

    # Add colorbar with better formatting
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Top 3 Institutions %\n(Concentration)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add size legend
    # Create dummy points for size legend
    size_labels = [1000, 5000, 10000]
    legend_elements = []
    for size_val in size_labels:
        legend_elements.append(plt.scatter([], [], s=size_val/10, c='gray',
                                          alpha=0.6, edgecolors='black', linewidth=0.5,
                                          label=f'{size_val:,} actionable'))

    legend = ax.legend(handles=legend_elements,
                      title='Actionable Events\n(point size)',
                      loc='upper left',
                      fontsize=10,
                      title_fontsize=11,
                      framealpha=0.9,
                      edgecolor='black')

    plt.tight_layout()
    plt.savefig(output_dir / 'patient_volume_vs_institutional_diversity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'patient_volume_vs_institutional_diversity.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'patient_volume_vs_institutional_diversity.pdf'}")
    plt.close()


def create_standalone_patient_institution_plot_narrow(all_stats_df, output_dir):
    """Create narrower version of Patient Volume vs Institutional Diversity plot"""

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    # Scatter plot: Number of Patients vs Number of Institutions
    scatter = ax.scatter(all_stats_df['n_patients'],
                        all_stats_df['n_institutions'],
                        s=all_stats_df['n_actionable'] / 10,
                        c=all_stats_df['top3_institutions_pct'],
                        cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=1.0)

    # Label all points with improved positioning
    for idx, row in all_stats_df.iterrows():
        ax.annotate(row['cancer_type'],
                   (row['n_patients'], row['n_institutions']),
                   fontsize=9, ha='center', va='bottom',
                   fontweight='bold')

    ax.set_xlabel('Total Patients (log scale)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Institutions', fontsize=14, fontweight='bold')
    ax.set_title('Patient Volume vs\nInstitutional Diversity\nAcross 21 Cancer Types',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax.tick_params(labelsize=11)

    # Add colorbar with better formatting
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Top 3 Institutions %\n(Concentration)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add size legend
    # Create dummy points for size legend
    size_labels = [1000, 5000, 10000]
    legend_elements = []
    for size_val in size_labels:
        legend_elements.append(plt.scatter([], [], s=size_val/10, c='gray',
                                          alpha=0.6, edgecolors='black', linewidth=0.5,
                                          label=f'{size_val:,} actionable'))

    legend = ax.legend(handles=legend_elements,
                      title='Actionable Events\n(point size)',
                      loc='upper left',
                      fontsize=10,
                      title_fontsize=11,
                      framealpha=0.9,
                      edgecolor='black')

    plt.tight_layout()
    plt.savefig(output_dir / 'patient_volume_vs_institutional_diversity_narrow.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'patient_volume_vs_institutional_diversity_narrow.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'patient_volume_vs_institutional_diversity_narrow.pdf'}")
    plt.close()


def create_ranked_bar_charts(all_stats_df, output_dir):
    """Create ranked bar charts for key metrics"""

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    metrics = [
        ('n_actionable', 'Number of Actionable Events', False),
        ('overall_actionable_rate', 'Overall Actionable Rate', True),
        ('herfindahl_index', 'Herfindahl Index (Concentration)', False),
        ('actionable_rate_std_across_institutions', 'Actionability Std Across Institutions', False),
        ('top3_institutions_pct', 'Top 3 Institutions %', True),
        ('gini_coefficient', 'Gini Coefficient (Inequality)', False)
    ]

    for idx, (metric, title, is_pct) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Sort and plot
        sorted_df = all_stats_df.sort_values(metric, ascending=True)

        # Color code: highlight high variance cancer types
        colors = ['#d62728' if std > 0.15 else '#1f77b4'
                 for std in sorted_df['actionable_rate_std_across_institutions']]

        y_pos = np.arange(len(sorted_df))
        bars = ax.barh(y_pos, sorted_df[metric], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_df['cancer_type'], fontsize=8)
        ax.set_xlabel(title, fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sorted_df[metric])):
            width = bar.get_width()
            if is_pct:
                label = f'{val:.1%}'
            elif metric == 'n_actionable':
                label = f'{int(val):,}'
            else:
                label = f'{val:.3f}'

            ax.text(width + (ax.get_xlim()[1] * 0.01), bar.get_y() + bar.get_height()/2.,
                   label, ha='left', va='center', fontsize=7)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.7, label='High actionability variance (std > 0.15)'),
        Patch(facecolor='#1f77b4', alpha=0.7, label='Low actionability variance (std ≤ 0.15)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
              bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(output_dir / 'ranked_metrics_bar_charts.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ranked_metrics_bar_charts.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'ranked_metrics_bar_charts.pdf'}")
    plt.close()


def generate_text_report(all_stats_df, output_dir):
    """Generate comprehensive text report"""

    report = []
    report.append("=" * 100)
    report.append("COMPREHENSIVE INSTITUTIONAL DISTRIBUTION ANALYSIS")
    report.append("All Cancer Types with ≥250 Actionable Events")
    report.append("=" * 100)
    report.append("")
    report.append(f"Total cancer types analyzed: {len(all_stats_df)}")
    report.append(f"Total patients: {all_stats_df['n_patients'].sum():,}")
    report.append(f"Total actionable events: {all_stats_df['n_actionable'].sum():,}")
    report.append("")

    # Sort by actionable events
    sorted_df = all_stats_df.sort_values('n_actionable', ascending=False)

    report.append("=" * 100)
    report.append("DETAILED STATISTICS BY CANCER TYPE")
    report.append("=" * 100)

    for idx, row in sorted_df.iterrows():
        report.append("")
        report.append(f"{row['cancer_type']}:")
        report.append("-" * 50)
        report.append(f"  Total patients: {int(row['n_patients']):,}")
        report.append(f"  Actionable events: {int(row['n_actionable']):,}")
        report.append(f"  Number of institutions: {int(row['n_institutions'])}")
        report.append(f"  Patients per institution (mean): {row['patients_per_institution_mean']:.1f}")
        report.append(f"  Patients per institution (median): {row['patients_per_institution_median']:.1f}")
        report.append(f"  Patients per institution (std): {row['patients_per_institution_std']:.1f}")
        report.append(f"  Herfindahl index: {row['herfindahl_index']:.3f}")
        report.append(f"  Gini coefficient: {row['gini_coefficient']:.3f}")
        report.append(f"  Top 1 institution: {row['top1_institution_pct']:.1%}")
        report.append(f"  Top 3 institutions: {row['top3_institutions_pct']:.1%}")
        report.append(f"  Top 5 institutions: {row['top5_institutions_pct']:.1%}")
        report.append(f"  Overall actionable rate: {row['overall_actionable_rate']:.1%}")
        report.append(f"  Actionable rate std (across institutions): {row['actionable_rate_std_across_institutions']:.3f}")
        report.append(f"  Actionable rate range (across institutions): {row['actionable_rate_range_across_institutions']:.3f}")
        report.append(f"  GroupKFold: ~{int(row['groupkfold_institutions_per_fold'])} institutions per fold, ~{int(row['groupkfold_patients_per_fold']):,} patients per fold")

    report.append("")
    report.append("=" * 100)
    report.append("KEY INSIGHTS")
    report.append("=" * 100)
    report.append("")

    # High variance cancer types (like GIST)
    high_variance = sorted_df[sorted_df['actionable_rate_std_across_institutions'] > 0.15]
    report.append(f"CANCER TYPES WITH HIGH ACTIONABILITY VARIANCE (std > 0.15):")
    report.append(f"  Count: {len(high_variance)}")
    if len(high_variance) > 0:
        report.append("  Types:")
        for idx, row in high_variance.iterrows():
            report.append(f"    - {row['cancer_type']}: std={row['actionable_rate_std_across_institutions']:.3f}, "
                         f"range={row['actionable_rate_range_across_institutions']:.3f}, "
                         f"mean rate={row['overall_actionable_rate']:.1%}")
        report.append("")
        report.append("  → These cancer types may show institution-dependent actionability patterns")
        report.append("  → Technical features (institution, panel) may have legitimate predictive value")
        report.append("  → GroupKFold CV likely more appropriate than standard CV")

    report.append("")

    # Highly concentrated cancer types
    high_conc = sorted_df[sorted_df['herfindahl_index'] > 0.3]
    report.append(f"HIGHLY CONCENTRATED CANCER TYPES (Herfindahl > 0.3):")
    report.append(f"  Count: {len(high_conc)}")
    if len(high_conc) > 0:
        report.append("  Types:")
        for idx, row in high_conc.iterrows():
            report.append(f"    - {row['cancer_type']}: Herfindahl={row['herfindahl_index']:.3f}, "
                         f"top 3 inst.={row['top3_institutions_pct']:.1%}, "
                         f"n_patients={int(row['n_patients']):,}")
        report.append("")
        report.append("  → Samples heavily concentrated in few institutions")
        report.append("  → May have small effective sample size for GroupKFold")

    report.append("")

    # Small sample size with GroupKFold concerns
    small_sample = sorted_df[sorted_df['groupkfold_institutions_per_fold'] < 5]
    report.append(f"CANCER TYPES WITH POTENTIAL GROUPKFOLD INSTABILITY (<5 institutions per fold):")
    report.append(f"  Count: {len(small_sample)}")
    if len(small_sample) > 0:
        report.append("  Types:")
        for idx, row in small_sample.iterrows():
            report.append(f"    - {row['cancer_type']}: {int(row['n_institutions'])} institutions, "
                         f"~{int(row['groupkfold_institutions_per_fold'])} per fold, "
                         f"{int(row['n_patients']):,} patients")
        report.append("")
        report.append("  → GroupKFold may create non-representative splits")
        report.append("  → Consider alternative CV strategies or collect more data")

    report.append("")
    report.append("=" * 100)
    report.append("")

    # Save report
    report_path = output_dir / 'comprehensive_institutional_analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Saved: {report_path}")
    return '\n'.join(report)


def main():
    """Main analysis pipeline"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Institutional distribution analysis')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    args = parser.parse_args()

    # Set output directory
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("COMPREHENSIVE INSTITUTIONAL DISTRIBUTION ANALYSIS")
    print("All Cancer Types with ≥250 Actionable Events")
    print("=" * 100)
    print()

    # Query eligible cancer types
    print(f"Querying cancer types with ≥{MIN_ACTIONABLE_EVENTS} actionable events...")
    eligible = query_eligible_cancer_types(DB_PATH, MIN_ACTIONABLE_EVENTS)
    print(f"Found {len(eligible)} eligible cancer types")
    print()

    # Analyze each cancer type
    all_stats = []

    for idx, row in eligible.iterrows():
        cancer_type = row['ONCOTREE_CODE']
        print(f"Analyzing {cancer_type} ({idx+1}/{len(eligible)})...")

        df = load_cancer_type_data(DB_PATH, cancer_type)
        stats, inst_counts, actionable_by_inst = analyze_institutional_distribution(df, cancer_type)
        all_stats.append(stats)

    # Create summary dataframe
    all_stats_df = pd.DataFrame(all_stats)

    print()
    print("=" * 100)
    print("GENERATING OUTPUTS")
    print("=" * 100)
    print()

    # Create summary table
    print("Creating summary table...")
    summary_df = create_summary_table(all_stats_df, OUTPUT_DIR)

    # Create visualizations
    print("Creating heatmap...")
    create_concentration_heatmap(all_stats_df, OUTPUT_DIR)

    print("Creating scatter plots...")
    create_scatter_plots(all_stats_df, OUTPUT_DIR)

    print("Creating additional scatter plots...")
    create_additional_scatter_plots(all_stats_df, OUTPUT_DIR)

    print("Creating standalone patient vs institution plot...")
    create_standalone_patient_institution_plot(all_stats_df, OUTPUT_DIR)

    print("Creating narrow patient vs institution plot...")
    create_standalone_patient_institution_plot_narrow(all_stats_df, OUTPUT_DIR)

    print("Creating ranked bar charts...")
    create_ranked_bar_charts(all_stats_df, OUTPUT_DIR)

    # Generate text report
    print("Generating comprehensive report...")
    report_text = generate_text_report(all_stats_df, OUTPUT_DIR)

    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print()
    print(f"Results saved to: {OUTPUT_DIR}")
    print()
    print("Files generated:")
    print("  - all_cancer_types_institutional_stats.csv")
    print("  - institutional_metrics_heatmap.pdf/png")
    print("  - institutional_metrics_scatter_plots.pdf/png")
    print("  - institutional_metrics_additional_scatter_plots.pdf/png")
    print("  - patient_volume_vs_institutional_diversity.pdf/png")
    print("  - patient_volume_vs_institutional_diversity_narrow.pdf/png")
    print("  - ranked_metrics_bar_charts.pdf/png")
    print("  - comprehensive_institutional_analysis_report.txt")
    print()

    # Quick summary
    print("=" * 100)
    print("QUICK SUMMARY")
    print("=" * 100)
    print()
    print(f"Total cancer types: {len(all_stats_df)}")
    print(f"Total patients: {all_stats_df['n_patients'].sum():,}")
    print(f"Total actionable events: {all_stats_df['n_actionable'].sum():,}")
    print()
    print("Top 5 by actionable events:")
    top5 = all_stats_df.nlargest(5, 'n_actionable')
    for idx, row in top5.iterrows():
        print(f"  {row['cancer_type']}: {int(row['n_actionable']):,} actionable events, "
              f"{int(row['n_patients']):,} patients, "
              f"{row['overall_actionable_rate']:.1%} rate")
    print()

    print("High actionability variance (std > 0.15):")
    high_var = all_stats_df[all_stats_df['actionable_rate_std_across_institutions'] > 0.15]
    if len(high_var) > 0:
        for idx, row in high_var.sort_values('actionable_rate_std_across_institutions', ascending=False).iterrows():
            print(f"  {row['cancer_type']}: std={row['actionable_rate_std_across_institutions']:.3f}")
    else:
        print("  None found")
    print()


if __name__ == "__main__":
    main()
