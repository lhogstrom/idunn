"""
Simple script to generate patient volume vs institutional diversity scatter plot
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite"
OUTPUT_DIR = Path("experiments/simple_institutional_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_ACTIONABLE_EVENTS = 250

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def query_eligible_cancer_types(db_path, min_events=250):
    """Query all cancer types meeting minimum actionable event threshold"""
    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT ONCOTREE_CODE
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE
    HAVING SUM(isOncoKbLevel1) >= {min_events}
    """

    cancer_types = pd.read_sql_query(query, conn)['ONCOTREE_CODE'].tolist()
    conn.close()

    return cancer_types


def load_cancer_type_data(db_path, cancer_type):
    """Load patient-level data for a specific cancer type"""
    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT SUBJECT_CENTER, isOncoKbLevel1
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE = '{cancer_type}'
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def analyze_cancer_type(df, cancer_type):
    """Calculate institutional statistics for a cancer type"""

    n_patients = len(df)
    n_institutions = df['SUBJECT_CENTER'].nunique()
    n_actionable = df['isOncoKbLevel1'].sum()

    # Calculate top 3 institutions concentration
    inst_counts = df['SUBJECT_CENTER'].value_counts()
    top3_pct = (inst_counts.iloc[:3].sum() / n_patients * 100) if len(inst_counts) >= 3 else 100.0

    return {
        'cancer_type': cancer_type,
        'n_patients': n_patients,
        'n_institutions': n_institutions,
        'n_actionable': n_actionable,
        'top3_institutions_pct': top3_pct
    }


def gather_all_stats(db_path, min_events=250):
    """Gather statistics for all eligible cancer types"""

    cancer_types = query_eligible_cancer_types(db_path, min_events)

    all_stats = []
    for cancer_type in cancer_types:
        df = load_cancer_type_data(db_path, cancer_type)
        stats = analyze_cancer_type(df, cancer_type)
        all_stats.append(stats)

    return pd.DataFrame(all_stats)


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_patient_volume_vs_diversity(all_stats_df, output_dir):
    """Create scatter plot: Patient Volume vs Institutional Diversity"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Scatter plot with size = actionable events, color = concentration
    scatter = ax.scatter(all_stats_df['n_patients'],
                        all_stats_df['n_institutions'],
                        s=all_stats_df['n_actionable'] / 10,
                        c=all_stats_df['top3_institutions_pct'],
                        cmap='RdYlBu_r', alpha=0.7, edgecolors='black', linewidth=1.0)

    # Label all points
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

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Top 3 Institutions %\n(Concentration)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add size legend
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
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Gather statistics for all cancer types
    all_stats_df = gather_all_stats(DB_PATH, MIN_ACTIONABLE_EVENTS)

    # Generate plot
    plot_patient_volume_vs_diversity(all_stats_df, OUTPUT_DIR)
