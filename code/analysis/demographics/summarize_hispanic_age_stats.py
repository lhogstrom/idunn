"""
Generate statistical summary table for Hispanic vs Non-Hispanic age differences
"""

import sqlite3
import pandas as pd
import numpy as np
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
# ANALYSIS
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
        ETHNICITY
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IN ('{cancer_list}')
    AND AGE_AT_SEQ_REPORT IS NOT NULL
    AND ETHNICITY IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def calculate_hispanic_age_stats(df, output_dir):
    """Calculate statistics for Hispanic vs Non-Hispanic age differences"""

    results = []

    for cancer in sorted(df['ONCOTREE_CODE'].unique()):
        cancer_df = df[df['ONCOTREE_CODE'] == cancer]

        # Split by Hispanic vs Non-Hispanic
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
                'Cancer_Type': cancer,
                'Mean_Hispanic': mean_hispanic,
                'Mean_Non_Hispanic': mean_non_hispanic,
                'Age_Difference': diff,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'P_Value': p_value,
                'N_Hispanic': len(hispanic),
                'N_Non_Hispanic': len(non_hispanic)
            })

    # Handle case with no results
    if len(results) == 0:
        return

    results_df = pd.DataFrame(results)

    # Apply Benjamini-Hochberg FDR correction
    reject, q_values, _, _ = multipletests(results_df['P_Value'].values,
                                           method='fdr_bh', alpha=0.05)
    results_df['Q_Value_FDR'] = q_values
    results_df['Significant_FDR'] = reject

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

    results_df['Sig_Marker'] = results_df['Q_Value_FDR'].apply(sig_marker)

    # Sort by age difference
    results_df = results_df.sort_values('Age_Difference')

    # Round numeric columns for readability
    results_df['Mean_Hispanic'] = results_df['Mean_Hispanic'].round(1)
    results_df['Mean_Non_Hispanic'] = results_df['Mean_Non_Hispanic'].round(1)
    results_df['Age_Difference'] = results_df['Age_Difference'].round(2)
    results_df['CI_Lower'] = results_df['CI_Lower'].round(2)
    results_df['CI_Upper'] = results_df['CI_Upper'].round(2)
    results_df['P_Value'] = results_df['P_Value'].apply(lambda x: f"{x:.3e}")
    results_df['Q_Value_FDR'] = results_df['Q_Value_FDR'].apply(lambda x: f"{x:.3e}")

    # Save to CSV
    results_df.to_csv(output_dir / 'hispanic_age_difference_statistics.csv', index=False)

    # Create formatted text report
    with open(output_dir / 'hispanic_age_difference_statistics.txt', 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("Statistical Analysis: Age Difference Between Hispanic and Non-Hispanic Patients\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"Test: Welch's two-sample t-test (unequal variances)\n")
        f.write(f"Multiple testing correction: Benjamini-Hochberg FDR\n")
        f.write(f"Number of comparisons: {len(results_df)}\n")
        f.write(f"Significance threshold: q < 0.05\n\n")

        f.write("Significance markers:\n")
        f.write("  *** q < 0.001 (highly significant)\n")
        f.write("  **  q < 0.01  (very significant)\n")
        f.write("  *   q < 0.05  (significant)\n")
        f.write("  ns  q ≥ 0.05  (not significant)\n\n")

        f.write("=" * 120 + "\n")
        f.write(f"{'Cancer':<8} {'Mean_Hisp':<11} {'Mean_NonH':<11} {'Diff':<8} "
                f"{'95% CI':<20} {'P-value':<12} {'Q-value':<12} {'Sig':<5} "
                f"{'N_Hisp':<8} {'N_NonH':<8}\n")
        f.write("=" * 120 + "\n")

        for _, row in results_df.iterrows():
            ci_str = f"[{row['CI_Lower']}, {row['CI_Upper']}]"
            f.write(f"{row['Cancer_Type']:<8} "
                   f"{row['Mean_Hispanic']:<11} "
                   f"{row['Mean_Non_Hispanic']:<11} "
                   f"{row['Age_Difference']:<8} "
                   f"{ci_str:<20} "
                   f"{row['P_Value']:<12} "
                   f"{row['Q_Value_FDR']:<12} "
                   f"{row['Sig_Marker']:<5} "
                   f"{row['N_Hispanic']:<8} "
                   f"{row['N_Non_Hispanic']:<8}\n")

        f.write("=" * 120 + "\n\n")

        # Summary statistics
        sig_count = results_df['Significant_FDR'].sum()
        f.write(f"SUMMARY:\n")
        f.write(f"  Total comparisons: {len(results_df)}\n")
        f.write(f"  Significant after FDR correction: {sig_count} ({sig_count/len(results_df)*100:.1f}%)\n")
        f.write(f"  Not significant: {len(results_df) - sig_count} ({(len(results_df) - sig_count)/len(results_df)*100:.1f}%)\n\n")

        # Largest differences
        f.write("Largest age differences:\n")
        top_diffs = results_df.nlargest(3, 'Age_Difference')
        for _, row in top_diffs.iterrows():
            f.write(f"  {row['Cancer_Type']}: {row['Age_Difference']} years older in Hispanic patients ({row['Sig_Marker']})\n")

        f.write("\nLargest negative differences:\n")
        bottom_diffs = results_df.nsmallest(3, 'Age_Difference')
        for _, row in bottom_diffs.iterrows():
            f.write(f"  {row['Cancer_Type']}: {abs(row['Age_Difference'])} years younger in Hispanic patients ({row['Sig_Marker']})\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    df = load_age_ethnicity_data(DB_PATH, MIN_ACTIONABLE_EVENTS)
    calculate_hispanic_age_stats(df, OUTPUT_DIR)
