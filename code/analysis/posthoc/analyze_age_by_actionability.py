"""
Post-Hoc Statistical Testing: Age Differences by Actionability Status

This script analyzes age differences between patients with and without
clinically actionable events across all cancer types.

For each cancer type:
1. Calculates mean and median age for isOncoKbLevel1=TRUE vs FALSE
2. Performs Mann-Whitney U test (non-parametric)
3. Performs t-test (parametric, for comparison)
4. Applies Benjamini-Hochberg FDR correction across all tests

"""

import sqlite3
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Configuration
DB_PATH = get_default_db_path()
DEFAULT_OUTPUT_DIR = Path("../../../analysis_output/posthoc_statistics")
OUTPUT_DIR = None  # Set by main()

# Table to use
TABLE_NAME = "perPatientOncoKBFilteredForModeling"


def load_data(db_path, table_name):
    """Load patient data from database"""

    print(f"Loading data from {db_path}...")
    print(f"Table: {table_name}")

    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT
        ONCOTREE_CODE,
        AGE_AT_SEQ_REPORT,
        isOncoKbLevel1
    FROM {table_name}
    WHERE ONCOTREE_CODE IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(df):,} patients")
    print(f"Cancer types: {df['ONCOTREE_CODE'].nunique()}")

    return df


def calculate_age_statistics(df_group):
    """Calculate age statistics for a group"""

    age_clean = df_group['AGE_AT_SEQ_REPORT'].dropna()

    if len(age_clean) == 0:
        return {
            'n': 0,
            'n_missing': len(df_group) - len(age_clean),
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'q25': np.nan,
            'q75': np.nan
        }

    return {
        'n': len(age_clean),
        'n_missing': len(df_group) - len(age_clean),
        'mean': age_clean.mean(),
        'median': age_clean.median(),
        'std': age_clean.std(),
        'min': age_clean.min(),
        'max': age_clean.max(),
        'q25': age_clean.quantile(0.25),
        'q75': age_clean.quantile(0.75)
    }


def test_age_differences(df_cancer_type):
    """Test age differences between actionable and non-actionable patients"""

    # Split by actionability status
    actionable = df_cancer_type[df_cancer_type['isOncoKbLevel1'] == 1]['AGE_AT_SEQ_REPORT'].dropna()
    non_actionable = df_cancer_type[df_cancer_type['isOncoKbLevel1'] == 0]['AGE_AT_SEQ_REPORT'].dropna()

    # Check if we have enough data
    if len(actionable) < 5 or len(non_actionable) < 5:
        return {
            'mann_whitney_u': np.nan,
            'mann_whitney_p': np.nan,
            'rank_biserial_r': np.nan,
            't_statistic': np.nan,
            't_test_p': np.nan,
            'mean_diff': np.nan,
            'median_diff': np.nan,
            'cohens_d': np.nan,
            'warning': f'Insufficient data (n_actionable={len(actionable)}, n_non_actionable={len(non_actionable)})'
        }

    # Mann-Whitney U test (non-parametric)
    try:
        u_stat, p_mann = stats.mannwhitneyu(actionable, non_actionable, alternative='two-sided')
        # Rank-biserial correlation (natural effect size for Mann-Whitney U)
        n1, n2 = len(actionable), len(non_actionable)
        rank_biserial_r = 1 - (2 * u_stat) / (n1 * n2)
    except Exception as e:
        u_stat, p_mann = np.nan, np.nan
        rank_biserial_r = np.nan
        print(f"  Warning: Mann-Whitney U test failed - {e}")

    # T-test (parametric)
    try:
        t_stat, p_ttest = stats.ttest_ind(actionable, non_actionable, equal_var=False)
    except Exception as e:
        t_stat, p_ttest = np.nan, np.nan
        print(f"  Warning: t-test failed - {e}")

    # Calculate effect size (Cohen's d)
    mean_diff = actionable.mean() - non_actionable.mean()
    median_diff = actionable.median() - non_actionable.median()

    # Cohen's d
    pooled_std = np.sqrt(((len(actionable) - 1) * actionable.std()**2 +
                          (len(non_actionable) - 1) * non_actionable.std()**2) /
                         (len(actionable) + len(non_actionable) - 2))

    cohens_d = mean_diff / pooled_std if pooled_std > 0 else np.nan

    return {
        'mann_whitney_u': u_stat,
        'mann_whitney_p': p_mann,
        'rank_biserial_r': rank_biserial_r,
        't_statistic': t_stat,
        't_test_p': p_ttest,
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'cohens_d': cohens_d,
        'warning': None
    }


def analyze_all_cancer_types(df):
    """Analyze age differences for all cancer types"""

    print("\nAnalyzing age differences by cancer type...")

    results = []

    cancer_types = sorted(df['ONCOTREE_CODE'].unique())
    print(f"Testing {len(cancer_types)} cancer types")

    for cancer_type in cancer_types:
        print(f"\nProcessing {cancer_type}...")

        df_cancer = df[df['ONCOTREE_CODE'] == cancer_type].copy()

        # Overall statistics
        n_total = len(df_cancer)
        n_actionable = (df_cancer['isOncoKbLevel1'] == 1).sum()
        n_non_actionable = (df_cancer['isOncoKbLevel1'] == 0).sum()

        print(f"  Total: {n_total}, Actionable: {n_actionable}, Non-actionable: {n_non_actionable}")

        # Calculate statistics for each group
        actionable_stats = calculate_age_statistics(
            df_cancer[df_cancer['isOncoKbLevel1'] == 1]
        )
        non_actionable_stats = calculate_age_statistics(
            df_cancer[df_cancer['isOncoKbLevel1'] == 0]
        )

        # Statistical tests
        test_results = test_age_differences(df_cancer)

        # Compile results
        result = {
            'ONCOTREE_CODE': cancer_type,
            'n_total': n_total,
            'n_actionable': n_actionable,
            'n_non_actionable': n_non_actionable,
            'actionable_rate': n_actionable / n_total if n_total > 0 else np.nan,

            # Actionable group
            'actionable_n': actionable_stats['n'],
            'actionable_n_missing': actionable_stats['n_missing'],
            'actionable_mean_age': actionable_stats['mean'],
            'actionable_median_age': actionable_stats['median'],
            'actionable_std_age': actionable_stats['std'],
            'actionable_min_age': actionable_stats['min'],
            'actionable_max_age': actionable_stats['max'],
            'actionable_q25_age': actionable_stats['q25'],
            'actionable_q75_age': actionable_stats['q75'],

            # Non-actionable group
            'non_actionable_n': non_actionable_stats['n'],
            'non_actionable_n_missing': non_actionable_stats['n_missing'],
            'non_actionable_mean_age': non_actionable_stats['mean'],
            'non_actionable_median_age': non_actionable_stats['median'],
            'non_actionable_std_age': non_actionable_stats['std'],
            'non_actionable_min_age': non_actionable_stats['min'],
            'non_actionable_max_age': non_actionable_stats['max'],
            'non_actionable_q25_age': non_actionable_stats['q25'],
            'non_actionable_q75_age': non_actionable_stats['q75'],

            # Test results
            'mean_age_difference': test_results['mean_diff'],
            'median_age_difference': test_results['median_diff'],
            'cohens_d': test_results['cohens_d'],
            'rank_biserial_r': test_results['rank_biserial_r'],
            'mann_whitney_u': test_results['mann_whitney_u'],
            'mann_whitney_p': test_results['mann_whitney_p'],
            't_statistic': test_results['t_statistic'],
            't_test_p': test_results['t_test_p'],
            'warning': test_results['warning']
        }

        results.append(result)

        if not pd.isna(test_results['mann_whitney_p']):
            print(f"  Mean age diff: {test_results['mean_diff']:.2f} years")
            print(f"  Mann-Whitney p: {test_results['mann_whitney_p']:.4f}")
            print(f"  Cohen's d: {test_results['cohens_d']:.3f}")
            print(f"  Rank-biserial r: {test_results['rank_biserial_r']:.3f}")

    results_df = pd.DataFrame(results)

    return results_df


def apply_fdr_correction(results_df):
    """Apply FDR correction using Benjamini-Hochberg procedure"""

    print("\nApplying FDR correction (Benjamini-Hochberg)...")

    # Get valid p-values (not NaN)
    valid_mask = ~results_df['mann_whitney_p'].isna()
    n_valid = valid_mask.sum()

    print(f"Valid tests: {n_valid} / {len(results_df)}")

    if n_valid == 0:
        print("Warning: No valid p-values for FDR correction")
        results_df['mann_whitney_p_fdr'] = np.nan
        results_df['t_test_p_fdr'] = np.nan
        results_df['mann_whitney_significant_fdr'] = False
        results_df['t_test_significant_fdr'] = False
        return results_df

    # FDR correction for Mann-Whitney test
    mann_whitney_fdr = np.full(len(results_df), np.nan)
    if valid_mask.sum() > 0:
        _, fdr_pvals, _, _ = multipletests(
            results_df.loc[valid_mask, 'mann_whitney_p'],
            method='fdr_bh',
            alpha=0.05
        )
        mann_whitney_fdr[valid_mask] = fdr_pvals

    results_df['mann_whitney_p_fdr'] = mann_whitney_fdr
    results_df['mann_whitney_significant_fdr'] = mann_whitney_fdr < 0.05

    # FDR correction for t-test
    valid_mask_t = ~results_df['t_test_p'].isna()
    t_test_fdr = np.full(len(results_df), np.nan)
    if valid_mask_t.sum() > 0:
        _, fdr_pvals, _, _ = multipletests(
            results_df.loc[valid_mask_t, 't_test_p'],
            method='fdr_bh',
            alpha=0.05
        )
        t_test_fdr[valid_mask_t] = fdr_pvals

    results_df['t_test_p_fdr'] = t_test_fdr
    results_df['t_test_significant_fdr'] = t_test_fdr < 0.05

    # Count significant results
    n_sig_mann = results_df['mann_whitney_significant_fdr'].sum()
    n_sig_t = results_df['t_test_significant_fdr'].sum()

    print(f"Significant (FDR < 0.05):")
    print(f"  Mann-Whitney: {n_sig_mann} / {n_valid}")
    print(f"  t-test: {n_sig_t} / {valid_mask_t.sum()}")

    return results_df


def generate_summary_report(results_df, output_file):
    """Generate human-readable summary report"""

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Age Differences by Actionability Status - Statistical Analysis\n")
        f.write("=" * 80 + "\n\n")

        f.write("Analysis Summary\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total cancer types tested: {len(results_df)}\n")

        valid_tests = (~results_df['mann_whitney_p'].isna()).sum()
        f.write(f"Valid statistical tests: {valid_tests}\n")

        sig_mann = results_df['mann_whitney_significant_fdr'].sum()
        sig_t = results_df['t_test_significant_fdr'].sum()

        f.write(f"Significant results (Mann-Whitney, FDR < 0.05): {sig_mann}\n")
        f.write(f"Significant results (t-test, FDR < 0.05): {sig_t}\n\n")

        # Significant results details
        if sig_mann > 0:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Significant Age Differences (Mann-Whitney, FDR < 0.05)\n")
            f.write("=" * 80 + "\n\n")

            sig_results = results_df[results_df['mann_whitney_significant_fdr'] == True].copy()
            sig_results = sig_results.sort_values('mann_whitney_p_fdr')

            for _, row in sig_results.iterrows():
                f.write(f"Cancer Type: {row['ONCOTREE_CODE']}\n")
                f.write(f"  Total patients: {row['n_total']:,}\n")
                f.write(f"  Actionable: {row['n_actionable']:,} ({row['actionable_rate']:.1%})\n")
                f.write(f"  \n")
                f.write(f"  Age (Actionable):     Mean={row['actionable_mean_age']:.1f}, Median={row['actionable_median_age']:.1f}\n")
                f.write(f"  Age (Non-actionable): Mean={row['non_actionable_mean_age']:.1f}, Median={row['non_actionable_median_age']:.1f}\n")
                f.write(f"  \n")
                f.write(f"  Mean difference: {row['mean_age_difference']:.2f} years\n")
                f.write(f"  Median difference: {row['median_age_difference']:.2f} years\n")
                f.write(f"  Cohen's d: {row['cohens_d']:.3f}\n")
                f.write(f"  \n")
                f.write(f"  Mann-Whitney p-value: {row['mann_whitney_p']:.4e}\n")
                f.write(f"  Mann-Whitney FDR: {row['mann_whitney_p_fdr']:.4e}\n")
                f.write(f"  t-test p-value: {row['t_test_p']:.4e}\n")
                f.write(f"  t-test FDR: {row['t_test_p_fdr']:.4e}\n")
                f.write("\n" + "-" * 80 + "\n\n")

        # All results summary table
        f.write("\n" + "=" * 80 + "\n")
        f.write("All Cancer Types - Summary Table\n")
        f.write("=" * 80 + "\n\n")

        summary_cols = [
            'ONCOTREE_CODE', 'n_total', 'n_actionable',
            'actionable_mean_age', 'non_actionable_mean_age',
            'mean_age_difference', 'cohens_d',
            'mann_whitney_p', 'mann_whitney_p_fdr'
        ]

        summary = results_df[summary_cols].copy()
        summary = summary.sort_values('mann_whitney_p_fdr')

        f.write(summary.to_string(index=False))
        f.write("\n\n")

        # Statistical interpretation guide
        f.write("\n" + "=" * 80 + "\n")
        f.write("Statistical Interpretation Guide\n")
        f.write("=" * 80 + "\n\n")

        f.write("Test Methods:\n")
        f.write("  - Mann-Whitney U: Non-parametric test for age differences (primary test)\n")
        f.write("  - t-test: Parametric test for comparison (assumes normality)\n")
        f.write("  - FDR correction: Benjamini-Hochberg procedure to control false discovery rate\n\n")

        f.write("Effect Size (Cohen's d):\n")
        f.write("  - Small: |d| = 0.2\n")
        f.write("  - Medium: |d| = 0.5\n")
        f.write("  - Large: |d| = 0.8\n\n")

        f.write("Interpretation:\n")
        f.write("  - Positive mean difference: Actionable patients are older\n")
        f.write("  - Negative mean difference: Actionable patients are younger\n")
        f.write("  - FDR < 0.05: Statistically significant after multiple testing correction\n\n")

    print(f"Summary report saved to: {output_file}")


def main():
    """Main execution function"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Age by actionability statistical testing')
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

    print("=" * 80)
    print("Post-Hoc Statistical Testing: Age by Actionability Status")
    print("=" * 80)

    # Load data
    df = load_data(DB_PATH, TABLE_NAME)

    # Analyze all cancer types
    results_df = analyze_all_cancer_types(df)

    # Apply FDR correction
    results_df = apply_fdr_correction(results_df)

    # Save results
    output_csv = OUTPUT_DIR / "age_actionability_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nFull results saved to: {output_csv}")

    # Save significant results only
    sig_results = results_df[results_df['mann_whitney_significant_fdr'] == True].copy()
    if len(sig_results) > 0:
        sig_csv = OUTPUT_DIR / "age_actionability_significant.csv"
        sig_results.to_csv(sig_csv, index=False)
        print(f"Significant results saved to: {sig_csv}")
    else:
        print("No significant results to save")

    # Generate summary report
    report_file = OUTPUT_DIR / "age_actionability_report.txt"
    generate_summary_report(results_df, report_file)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
