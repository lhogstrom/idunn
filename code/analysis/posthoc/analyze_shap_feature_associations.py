"""
Post-Hoc Statistical Testing: SHAP Feature Associations with Actionability

This script analyzes demographic and technical feature associations with
actionability based on SHAP feature importance rankings.

For features ranking in top 5:
1. SEX: Odds ratios and Fisher's exact test / chi-square
2. RACE/ETHNICITY: Chi-square tests across groups
3. Technical features (library_selection, calling_strategy): Chi-square tests
4. Applies Benjamini-Hochberg FDR correction for each category

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
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

# Configuration
DB_PATH = get_default_db_path()

# Use absolute path to ensure output goes to repo directory
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Legacy SHAP file path (for backward compatibility)
LEGACY_SHAP_FILE = "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522b/demographic_technical_ml_modeling/shap_feature_ranks.csv"

DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis_output" / "posthoc_statistics"
OUTPUT_DIR = None  # Set by main()

# Feature patterns for pattern matching in both old and new formats
# Old format uses sklearn-style names: "cat__SEX_Male"
# New format uses readable names: "Sex: Male"
FEATURE_PATTERNS = {
    'SEX': ['SEX_', 'Sex:'],
    'RACE_ETHNICITY': ['RACE_', 'ETHNICITY_', 'Race:', 'Ethnicity:'],
    'TECHNICAL': ['library_selection_', 'calling_strategy_', 'Library Selection:', 'Calling Strategy:']
}

TABLE_NAME = "perPatientOncoKBFilteredForModeling"
TOP_N_FEATURES = 5


def get_default_shap_file():
    """
    Get default SHAP file, preferring new format (TIER 6a) over legacy.

    Searches for SHAP primary values output from compute_shap_primary_values.py
    in the standard output locations.

    Returns:
        str: Path to SHAP rankings file, or None if not found
    """
    # Check for new format output in analysis_output
    new_output = REPO_ROOT / "analysis_output" / "shap_primary_values"
    if new_output.exists():
        # Find latest run
        runs = sorted(new_output.glob("run_*"), reverse=True)
        for run_dir in runs:
            combined_file = run_dir / "combined_feature_importance.csv"
            if combined_file.exists():
                print(f"Found SHAP file (new format): {combined_file}")
                return str(combined_file)

    # Check for new format in runs directory (unified output structure)
    runs_dir = REPO_ROOT / "runs"
    if runs_dir.exists():
        # Find latest run with SHAP output
        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        for run_dir in runs:
            shap_dir = run_dir / "tier6_shap_analysis" / "primary_values"
            combined_file = shap_dir / "combined_feature_importance.csv"
            if combined_file.exists():
                print(f"Found SHAP file (new format, unified runs): {combined_file}")
                return str(combined_file)

    # Fall back to legacy file
    if Path(LEGACY_SHAP_FILE).exists():
        print(f"Using legacy SHAP file: {LEGACY_SHAP_FILE}")
        print("  WARNING: Consider running TIER 6a (compute_shap_primary_values.py) for updated SHAP rankings")
        return LEGACY_SHAP_FILE

    return None


def load_shap_rankings_adaptive(shap_file):
    """
    Load SHAP rankings, auto-detecting format (new vs legacy).

    New format (from shap_primary_analyzer.py):
        - cancer_type, feature_name, feature_name_raw, mean_abs_shap, rank

    Legacy format:
        - ONCOTREE_CODE, feature, mean_abs_shap, feature_rank

    Args:
        shap_file: Path to SHAP rankings CSV file

    Returns:
        pd.DataFrame with standardized columns:
            - ONCOTREE_CODE: cancer type
            - feature: feature name (human-readable preferred)
            - mean_abs_shap: mean absolute SHAP value
            - feature_rank: 1-based rank

    Raises:
        ValueError: If file format is not recognized
    """
    print(f"Loading SHAP rankings from {shap_file}...")

    df = pd.read_csv(shap_file)

    # Detect format by checking column names
    if 'cancer_type' in df.columns and 'feature_name' in df.columns:
        # New format from shap_primary_analyzer.py
        print("  Detected NEW format (TIER 6a output)")

        # Rename columns to legacy format for compatibility
        df = df.rename(columns={
            'cancer_type': 'ONCOTREE_CODE',
            'feature_name': 'feature',  # Use readable name for pattern matching
            'rank': 'feature_rank'
        })

        # Keep feature_name_raw as additional column for reference
        if 'feature_name_raw' in df.columns:
            df['feature_raw'] = df['feature_name_raw']

    elif 'ONCOTREE_CODE' in df.columns and 'feature' in df.columns:
        # Legacy format
        print("  Detected LEGACY format")

    else:
        raise ValueError(
            f"Unrecognized SHAP file format. "
            f"Expected columns 'cancer_type'/'feature_name' (new) or 'ONCOTREE_CODE'/'feature' (legacy). "
            f"Found: {df.columns.tolist()}"
        )

    # Ensure feature_rank exists
    if 'feature_rank' not in df.columns and 'rank' in df.columns:
        df['feature_rank'] = df['rank']

    print(f"Loaded {len(df):,} feature rankings")
    print(f"Cancer types: {df['ONCOTREE_CODE'].nunique()}")
    print(f"Unique features: {df['feature'].nunique()}")

    return df


def load_shap_rankings(shap_file):
    """Load SHAP feature rankings (wrapper for backward compatibility)"""
    return load_shap_rankings_adaptive(shap_file)


def identify_top_features(shap_df, feature_patterns, top_n=5):
    """Identify cancer types where specific features appear in top N"""

    results = []

    for pattern in feature_patterns:
        # Filter for features matching pattern
        mask = shap_df['feature'].str.contains(pattern, case=False, na=False)
        matching = shap_df[mask & (shap_df['feature_rank'] <= top_n)].copy()

        if len(matching) > 0:
            print(f"\nFeature pattern '{pattern}': {len(matching)} top-{top_n} occurrences")
            for _, row in matching.iterrows():
                print(f"  {row['ONCOTREE_CODE']}: {row['feature']} (rank {row['feature_rank']})")
                results.append({
                    'pattern': pattern,
                    'ONCOTREE_CODE': row['ONCOTREE_CODE'],
                    'feature': row['feature'],
                    'feature_rank': row['feature_rank'],  # Standardized column name
                    'mean_abs_shap': row['mean_abs_shap']
                })

    results_df = pd.DataFrame(results)
    return results_df


def load_patient_data(db_path, cancer_types, columns):
    """Load patient data for specified cancer types and columns"""

    conn = sqlite3.connect(db_path)

    # Build column list
    col_str = ', '.join(columns)

    # Build cancer type filter
    cancer_type_str = "', '".join(cancer_types)

    query = f"""
    SELECT {col_str}
    FROM {TABLE_NAME}
    WHERE ONCOTREE_CODE IN ('{cancer_type_str}')
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def calculate_odds_ratio_sex(df_cancer):
    """Calculate odds ratio for sex association with actionability"""

    # Create contingency table: rows=sex, columns=actionability
    # Use Female as reference (Female vs Male)

    df = df_cancer[df_cancer['SEX'].isin(['Female', 'Male'])].copy()

    if len(df) == 0:
        return {
            'n_total': 0,
            'n_female': 0,
            'n_male': 0,
            'female_actionable': 0,
            'female_total': 0,
            'male_actionable': 0,
            'male_total': 0,
            'female_rate': np.nan,
            'male_rate': np.nan,
            'odds_ratio': np.nan,
            'or_ci_lower': np.nan,
            'or_ci_upper': np.nan,
            'p_value': np.nan,
            'test_used': 'none',
            'warning': 'No valid sex data'
        }

    # Contingency table
    contingency = pd.crosstab(
        df['SEX'],
        df['isOncoKbLevel1'],
        margins=False
    )

    # Ensure we have both sexes and both outcomes
    if contingency.shape != (2, 2):
        return {
            'n_total': len(df),
            'n_female': (df['SEX'] == 'Female').sum(),
            'n_male': (df['SEX'] == 'Male').sum(),
            'female_actionable': np.nan,
            'female_total': np.nan,
            'male_actionable': np.nan,
            'male_total': np.nan,
            'female_rate': np.nan,
            'male_rate': np.nan,
            'odds_ratio': np.nan,
            'or_ci_lower': np.nan,
            'or_ci_upper': np.nan,
            'p_value': np.nan,
            'test_used': 'none',
            'warning': 'Incomplete contingency table'
        }

    # Extract counts (Female first, then Male)
    try:
        female_non_act = contingency.loc['Female', 0]
        female_act = contingency.loc['Female', 1]
        male_non_act = contingency.loc['Male', 0]
        male_act = contingency.loc['Male', 1]
    except KeyError:
        return {
            'n_total': len(df),
            'n_female': (df['SEX'] == 'Female').sum(),
            'n_male': (df['SEX'] == 'Male').sum(),
            'female_actionable': np.nan,
            'female_total': np.nan,
            'male_actionable': np.nan,
            'male_total': np.nan,
            'female_rate': np.nan,
            'male_rate': np.nan,
            'odds_ratio': np.nan,
            'or_ci_lower': np.nan,
            'or_ci_upper': np.nan,
            'p_value': np.nan,
            'test_used': 'none',
            'warning': 'Missing sex categories'
        }

    female_total = female_non_act + female_act
    male_total = male_non_act + male_act

    female_rate = female_act / female_total if female_total > 0 else np.nan
    male_rate = male_act / male_total if male_total > 0 else np.nan

    # Calculate odds ratio (Female vs Male)
    # OR = (female_act / female_non_act) / (male_act / male_non_act)
    if female_non_act > 0 and male_non_act > 0:
        odds_ratio = (female_act * male_non_act) / (male_act * female_non_act)
    else:
        odds_ratio = np.nan

    # Decide on test: Fisher's exact if small cells, chi-square otherwise
    min_expected = min(female_act, female_non_act, male_act, male_non_act)

    if min_expected < 5:
        # Use Fisher's exact test
        try:
            # Fisher's exact requires 2x2 table: [[a, b], [c, d]]
            # Female: [non_act, act], Male: [non_act, act]
            table = [[female_non_act, female_act], [male_non_act, male_act]]
            odds_ratio_fisher, p_value = fisher_exact(table)
            test_used = 'fisher_exact'

            # Calculate confidence interval for odds ratio (log scale)
            # Using Woolf's method
            if odds_ratio > 0:
                log_or = np.log(odds_ratio)
                se_log_or = np.sqrt(1/female_act + 1/female_non_act +
                                    1/male_act + 1/male_non_act)
                ci_lower = np.exp(log_or - 1.96 * se_log_or)
                ci_upper = np.exp(log_or + 1.96 * se_log_or)
            else:
                ci_lower, ci_upper = np.nan, np.nan

        except Exception as e:
            print(f"    Warning: Fisher's exact test failed - {e}")
            p_value = np.nan
            ci_lower, ci_upper = np.nan, np.nan
            test_used = 'failed'
    else:
        # Use chi-square test
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            test_used = 'chi_square'

            # Calculate CI for odds ratio
            if odds_ratio > 0:
                log_or = np.log(odds_ratio)
                se_log_or = np.sqrt(1/female_act + 1/female_non_act +
                                    1/male_act + 1/male_non_act)
                ci_lower = np.exp(log_or - 1.96 * se_log_or)
                ci_upper = np.exp(log_or + 1.96 * se_log_or)
            else:
                ci_lower, ci_upper = np.nan, np.nan

        except Exception as e:
            print(f"    Warning: Chi-square test failed - {e}")
            p_value = np.nan
            ci_lower, ci_upper = np.nan, np.nan
            test_used = 'failed'

    return {
        'n_total': len(df),
        'n_female': female_total,
        'n_male': male_total,
        'female_actionable': female_act,
        'female_total': female_total,
        'male_actionable': male_act,
        'male_total': male_total,
        'female_rate': female_rate,
        'male_rate': male_rate,
        'odds_ratio': odds_ratio,
        'or_ci_lower': ci_lower,
        'or_ci_upper': ci_upper,
        'p_value': p_value,
        'test_used': test_used,
        'warning': None
    }


def test_race_ethnicity_association(df_cancer, feature_name):
    """Test race or ethnicity association with actionability using chi-square"""

    # Clean data
    df = df_cancer[df_cancer[feature_name].notna()].copy()

    if len(df) < 20:
        return {
            'n_total': len(df),
            'chi2': np.nan,
            'p_value': np.nan,
            'cramers_v': np.nan,
            'dof': np.nan,
            'warning': f'Insufficient data (n={len(df)})'
        }

    # Create contingency table
    contingency = pd.crosstab(df[feature_name], df['isOncoKbLevel1'])

    # Need at least 2 categories and both outcomes
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {
            'n_total': len(df),
            'chi2': np.nan,
            'p_value': np.nan,
            'cramers_v': np.nan,
            'dof': np.nan,
            'warning': 'Insufficient categories'
        }

    # Chi-square test
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        # Calculate Cramér's V (effect size)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else np.nan

        return {
            'n_total': len(df),
            'chi2': chi2,
            'p_value': p_value,
            'cramers_v': cramers_v,
            'dof': dof,
            'contingency_table': contingency.to_dict(),
            'warning': None
        }

    except Exception as e:
        return {
            'n_total': len(df),
            'chi2': np.nan,
            'p_value': np.nan,
            'cramers_v': np.nan,
            'dof': np.nan,
            'warning': f'Chi-square test failed: {e}'
        }


def test_technical_feature_association(df_cancer, feature_name):
    """Test technical feature association with actionability using chi-square"""

    # Same as race/ethnicity test
    return test_race_ethnicity_association(df_cancer, feature_name)


def analyze_sex_associations(shap_df, db_path):
    """Analyze sex associations for cancer types where SEX ranks in top 5"""

    print("\n" + "=" * 80)
    print("Analyzing SEX associations")
    print("=" * 80)

    # Identify cancer types with sex features in top 5
    # Use both old (SEX_) and new (Sex:) patterns for compatibility
    sex_features = identify_top_features(shap_df, FEATURE_PATTERNS['SEX'], TOP_N_FEATURES)

    if len(sex_features) == 0:
        print("No cancer types with SEX features in top 5")
        return pd.DataFrame()

    # Get unique cancer types
    cancer_types = sex_features['ONCOTREE_CODE'].unique()
    print(f"\nTesting {len(cancer_types)} cancer types")

    # Load patient data
    df = load_patient_data(db_path, cancer_types, ['ONCOTREE_CODE', 'SEX', 'isOncoKbLevel1'])

    results = []

    for cancer_type in cancer_types:
        print(f"\nTesting {cancer_type}...")

        df_cancer = df[df['ONCOTREE_CODE'] == cancer_type].copy()

        # Get SHAP rank info
        shap_info = sex_features[sex_features['ONCOTREE_CODE'] == cancer_type].iloc[0]

        # Calculate odds ratio and test
        test_result = calculate_odds_ratio_sex(df_cancer)

        result = {
            'ONCOTREE_CODE': cancer_type,
            'feature': shap_info['feature'],
            'shap_rank': shap_info['feature_rank'],
            'shap_importance': shap_info['mean_abs_shap'],
            **test_result
        }

        results.append(result)

        if not pd.isna(test_result['p_value']):
            print(f"  Female rate: {test_result['female_rate']:.1%}")
            print(f"  Male rate: {test_result['male_rate']:.1%}")
            print(f"  Odds ratio: {test_result['odds_ratio']:.3f}")
            print(f"  p-value: {test_result['p_value']:.4e}")

    results_df = pd.DataFrame(results)

    # Apply FDR correction
    valid_mask = ~results_df['p_value'].isna()
    if valid_mask.sum() > 0:
        _, fdr_pvals, _, _ = multipletests(
            results_df.loc[valid_mask, 'p_value'],
            method='fdr_bh',
            alpha=0.05
        )
        results_df.loc[valid_mask, 'p_fdr'] = fdr_pvals
        results_df['significant_fdr'] = results_df['p_fdr'] < 0.05
    else:
        results_df['p_fdr'] = np.nan
        results_df['significant_fdr'] = False

    print(f"\nSignificant (FDR < 0.05): {results_df['significant_fdr'].sum()}")

    return results_df


def analyze_race_ethnicity_associations(shap_df, db_path):
    """Analyze race/ethnicity associations for top-ranked features"""

    print("\n" + "=" * 80)
    print("Analyzing RACE and ETHNICITY associations")
    print("=" * 80)

    # Identify cancer types with race/ethnicity features in top 5
    # Use both old (RACE_, ETHNICITY_) and new (Race:, Ethnicity:) patterns
    race_eth_features = identify_top_features(
        shap_df,
        FEATURE_PATTERNS['RACE_ETHNICITY'],
        TOP_N_FEATURES
    )

    if len(race_eth_features) == 0:
        print("No cancer types with RACE/ETHNICITY features in top 5")
        return pd.DataFrame()

    # Get unique cancer types
    cancer_types = race_eth_features['ONCOTREE_CODE'].unique()
    print(f"\nTesting {len(cancer_types)} cancer types")

    # Load patient data
    df = load_patient_data(
        db_path,
        cancer_types,
        ['ONCOTREE_CODE', 'RACE', 'ETHNICITY', 'isOncoKbLevel1']
    )

    results = []

    for cancer_type in cancer_types:
        print(f"\nTesting {cancer_type}...")

        df_cancer = df[df['ONCOTREE_CODE'] == cancer_type].copy()

        # Get features for this cancer type
        features_ct = race_eth_features[race_eth_features['ONCOTREE_CODE'] == cancer_type]

        for _, feat_info in features_ct.iterrows():
            feature_name = feat_info['feature']

            # Determine column name (RACE or ETHNICITY)
            # Support both old format (RACE_, ETHNICITY_) and new format (Race:, Ethnicity:)
            if 'RACE_' in feature_name or 'Race:' in feature_name:
                column = 'RACE'
            elif 'ETHNICITY_' in feature_name or 'Ethnicity:' in feature_name:
                column = 'ETHNICITY'
            else:
                continue

            print(f"  Testing {column}...")

            # Test association
            test_result = test_race_ethnicity_association(df_cancer, column)

            result = {
                'ONCOTREE_CODE': cancer_type,
                'feature': feature_name,
                'column': column,
                'shap_rank': feat_info['feature_rank'],
                'shap_importance': feat_info['mean_abs_shap'],
                **test_result
            }

            results.append(result)

            if not pd.isna(test_result['p_value']):
                print(f"    Chi-square: {test_result['chi2']:.2f}")
                print(f"    p-value: {test_result['p_value']:.4e}")
                print(f"    Cramér's V: {test_result['cramers_v']:.3f}")

    results_df = pd.DataFrame(results)

    # Apply FDR correction
    valid_mask = ~results_df['p_value'].isna()
    if valid_mask.sum() > 0:
        _, fdr_pvals, _, _ = multipletests(
            results_df.loc[valid_mask, 'p_value'],
            method='fdr_bh',
            alpha=0.05
        )
        results_df.loc[valid_mask, 'p_fdr'] = fdr_pvals
        results_df['significant_fdr'] = results_df['p_fdr'] < 0.05
    else:
        results_df['p_fdr'] = np.nan
        results_df['significant_fdr'] = False

    print(f"\nSignificant (FDR < 0.05): {results_df['significant_fdr'].sum()}")

    return results_df


def analyze_technical_feature_associations(shap_df, db_path):
    """Analyze technical feature associations for top-ranked features"""

    print("\n" + "=" * 80)
    print("Analyzing TECHNICAL feature associations")
    print("=" * 80)

    # Identify cancer types with technical features in top 5
    # Use both old and new format patterns
    tech_features = identify_top_features(
        shap_df,
        FEATURE_PATTERNS['TECHNICAL'],
        TOP_N_FEATURES
    )

    if len(tech_features) == 0:
        print("No cancer types with technical features in top 5")
        return pd.DataFrame()

    # Get unique cancer types
    cancer_types = tech_features['ONCOTREE_CODE'].unique()
    print(f"\nTesting {len(cancer_types)} cancer types")

    # Load patient data - need to load all columns to find the right ones
    conn = sqlite3.connect(db_path)

    # Get column names first
    query = f"PRAGMA table_info({TABLE_NAME})"
    col_info = pd.read_sql_query(query, conn)
    all_columns = col_info['name'].tolist()

    # Find library_selection and calling_strategy columns
    lib_sel_cols = [c for c in all_columns if 'library_selection' in c.lower()]
    call_strat_cols = [c for c in all_columns if 'calling_strategy' in c.lower()]

    print(f"Found {len(lib_sel_cols)} library_selection columns")
    print(f"Found {len(call_strat_cols)} calling_strategy columns")

    # Use first match (should be the base column, not one-hot encoded)
    columns_to_load = ['ONCOTREE_CODE', 'isOncoKbLevel1']
    if len(lib_sel_cols) > 0:
        columns_to_load.append(lib_sel_cols[0])
    if len(call_strat_cols) > 0:
        columns_to_load.append(call_strat_cols[0])

    cancer_type_str = "', '".join(cancer_types)
    col_str = ', '.join(columns_to_load)

    query = f"""
    SELECT {col_str}
    FROM {TABLE_NAME}
    WHERE ONCOTREE_CODE IN ('{cancer_type_str}')
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    results = []

    for cancer_type in cancer_types:
        print(f"\nTesting {cancer_type}...")

        df_cancer = df[df['ONCOTREE_CODE'] == cancer_type].copy()

        # Get features for this cancer type
        features_ct = tech_features[tech_features['ONCOTREE_CODE'] == cancer_type]

        for _, feat_info in features_ct.iterrows():
            feature_name = feat_info['feature']

            # Determine column name
            # Support both old format (library_selection_, calling_strategy_)
            # and new format (Library Selection:, Calling Strategy:)
            if 'library_selection_' in feature_name.lower() or 'Library Selection:' in feature_name:
                column = lib_sel_cols[0] if len(lib_sel_cols) > 0 else None
            elif 'calling_strategy_' in feature_name.lower() or 'Calling Strategy:' in feature_name:
                column = call_strat_cols[0] if len(call_strat_cols) > 0 else None
            else:
                continue

            if column is None or column not in df_cancer.columns:
                print(f"  Skipping {feature_name} - column not found")
                continue

            print(f"  Testing {column}...")

            # Test association
            test_result = test_technical_feature_association(df_cancer, column)

            result = {
                'ONCOTREE_CODE': cancer_type,
                'feature': feature_name,
                'column': column,
                'shap_rank': feat_info['feature_rank'],
                'shap_importance': feat_info['mean_abs_shap'],
                **test_result
            }

            results.append(result)

            if not pd.isna(test_result.get('p_value')):
                print(f"    Chi-square: {test_result['chi2']:.2f}")
                print(f"    p-value: {test_result['p_value']:.4e}")
                print(f"    Cramér's V: {test_result['cramers_v']:.3f}")

    results_df = pd.DataFrame(results)

    # Apply FDR correction
    if len(results_df) > 0:
        valid_mask = ~results_df['p_value'].isna()
        if valid_mask.sum() > 0:
            _, fdr_pvals, _, _ = multipletests(
                results_df.loc[valid_mask, 'p_value'],
                method='fdr_bh',
                alpha=0.05
            )
            results_df.loc[valid_mask, 'p_fdr'] = fdr_pvals
            results_df['significant_fdr'] = results_df['p_fdr'] < 0.05
        else:
            results_df['p_fdr'] = np.nan
            results_df['significant_fdr'] = False

        print(f"\nSignificant (FDR < 0.05): {results_df['significant_fdr'].sum()}")

    return results_df


def generate_summary_report(sex_results, race_eth_results, tech_results, output_file):
    """Generate comprehensive summary report"""

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SHAP Feature Associations with Actionability - Statistical Analysis\n")
        f.write("=" * 80 + "\n\n")

        # SEX associations
        if len(sex_results) > 0:
            f.write("\n" + "=" * 80 + "\n")
            f.write("SEX Associations\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Cancer types tested: {len(sex_results)}\n")
            sig_count = sex_results['significant_fdr'].sum()
            f.write(f"Significant (FDR < 0.05): {sig_count}\n\n")

            if sig_count > 0:
                sig = sex_results[sex_results['significant_fdr'] == True].copy()
                sig = sig.sort_values('p_fdr')

                for _, row in sig.iterrows():
                    f.write(f"Cancer Type: {row['ONCOTREE_CODE']}\n")
                    f.write(f"  SHAP Feature: {row['feature']} (rank {row['shap_rank']})\n")
                    f.write(f"  Female: {row['female_actionable']}/{row['female_total']} ({row['female_rate']:.1%})\n")
                    f.write(f"  Male: {row['male_actionable']}/{row['male_total']} ({row['male_rate']:.1%})\n")
                    f.write(f"  Odds Ratio: {row['odds_ratio']:.3f} ({row['or_ci_lower']:.3f}-{row['or_ci_upper']:.3f})\n")
                    f.write(f"  p-value: {row['p_value']:.4e}, FDR: {row['p_fdr']:.4e}\n")
                    f.write(f"  Test: {row['test_used']}\n\n")

        # RACE/ETHNICITY associations
        if len(race_eth_results) > 0:
            f.write("\n" + "=" * 80 + "\n")
            f.write("RACE / ETHNICITY Associations\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Tests performed: {len(race_eth_results)}\n")
            sig_count = race_eth_results['significant_fdr'].sum()
            f.write(f"Significant (FDR < 0.05): {sig_count}\n\n")

            if sig_count > 0:
                sig = race_eth_results[race_eth_results['significant_fdr'] == True].copy()
                sig = sig.sort_values('p_fdr')

                for _, row in sig.iterrows():
                    f.write(f"Cancer Type: {row['ONCOTREE_CODE']}\n")
                    f.write(f"  SHAP Feature: {row['feature']} (rank {row['shap_rank']})\n")
                    f.write(f"  Column tested: {row['column']}\n")
                    f.write(f"  Chi-square: {row['chi2']:.2f}, df={row['dof']}\n")
                    f.write(f"  Cramér's V: {row['cramers_v']:.3f}\n")
                    f.write(f"  p-value: {row['p_value']:.4e}, FDR: {row['p_fdr']:.4e}\n\n")

        # TECHNICAL feature associations
        if len(tech_results) > 0:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TECHNICAL Feature Associations\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Tests performed: {len(tech_results)}\n")
            sig_count = tech_results['significant_fdr'].sum()
            f.write(f"Significant (FDR < 0.05): {sig_count}\n\n")

            if sig_count > 0:
                sig = tech_results[tech_results['significant_fdr'] == True].copy()
                sig = sig.sort_values('p_fdr')

                for _, row in sig.iterrows():
                    f.write(f"Cancer Type: {row['ONCOTREE_CODE']}\n")
                    f.write(f"  SHAP Feature: {row['feature']} (rank {row['shap_rank']})\n")
                    f.write(f"  Column tested: {row['column']}\n")
                    f.write(f"  Chi-square: {row['chi2']:.2f}, df={row['dof']}\n")
                    f.write(f"  Cramér's V: {row['cramers_v']:.3f}\n")
                    f.write(f"  p-value: {row['p_value']:.4e}, FDR: {row['p_fdr']:.4e}\n\n")

        # Interpretation guide
        f.write("\n" + "=" * 80 + "\n")
        f.write("Statistical Interpretation Guide\n")
        f.write("=" * 80 + "\n\n")

        f.write("Test Methods:\n")
        f.write("  - Fisher's exact test: Used for 2x2 tables with small expected counts (<5)\n")
        f.write("  - Chi-square test: Used for larger sample sizes\n")
        f.write("  - FDR correction: Benjamini-Hochberg procedure\n\n")

        f.write("Effect Sizes:\n")
        f.write("  - Odds Ratio (OR): Risk ratio for binary outcomes\n")
        f.write("    - OR > 1: First group has higher odds of actionability\n")
        f.write("    - OR < 1: First group has lower odds of actionability\n")
        f.write("  - Cramér's V: Association strength for categorical variables\n")
        f.write("    - Small: 0.1, Medium: 0.3, Large: 0.5\n\n")

    print(f"Summary report saved to: {output_file}")


def main():
    """Main execution function"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='SHAP feature association testing',
        epilog="""
Examples:
  # Use auto-detected SHAP file (new format preferred)
  python analyze_shap_feature_associations.py

  # Use specific SHAP file (new TIER 6a format)
  python analyze_shap_feature_associations.py --shap-file ../../../analysis_output/shap_primary_values/run_*/combined_feature_importance.csv

  # Use legacy SHAP file
  python analyze_shap_feature_associations.py --shap-file /path/to/legacy/shap_feature_ranks.csv
        """
    )
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--shap-file', type=str, default=None,
                       help='Path to SHAP feature rankings CSV (new TIER 6a format or legacy format)')
    args = parser.parse_args()

    # Set output directory
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Post-Hoc Statistical Testing: SHAP Feature Associations")
    print("=" * 80)

    # Determine SHAP file to use
    if args.shap_file:
        shap_file = args.shap_file
        print(f"\nUsing specified SHAP file: {shap_file}")
    else:
        shap_file = get_default_shap_file()
        if shap_file is None:
            print("\nERROR: No SHAP file found!")
            print("  Please run TIER 6a (compute_shap_primary_values.py) first, or")
            print("  specify a SHAP file using --shap-file argument")
            return 1

    # Verify SHAP file exists
    if not Path(shap_file).exists():
        print(f"\nERROR: SHAP file not found: {shap_file}")
        return 1

    # Load SHAP rankings (auto-detects format)
    shap_df = load_shap_rankings_adaptive(shap_file)

    # Save top 5 features for all cancer types
    top5_all = shap_df[shap_df['feature_rank'] <= TOP_N_FEATURES].copy()
    top5_file = OUTPUT_DIR / "shap_top5_features.csv"
    top5_all.to_csv(top5_file, index=False)
    print(f"\nTop-5 features saved to: {top5_file}")

    # Analyze SEX associations
    sex_results = analyze_sex_associations(shap_df, DB_PATH)
    if len(sex_results) > 0:
        sex_file = OUTPUT_DIR / "sex_associations.csv"
        sex_results.to_csv(sex_file, index=False)
        print(f"SEX results saved to: {sex_file}")

    # Analyze RACE/ETHNICITY associations
    race_eth_results = analyze_race_ethnicity_associations(shap_df, DB_PATH)
    if len(race_eth_results) > 0:
        race_eth_file = OUTPUT_DIR / "race_ethnicity_associations.csv"
        race_eth_results.to_csv(race_eth_file, index=False)
        print(f"RACE/ETHNICITY results saved to: {race_eth_file}")

    # Analyze technical feature associations
    tech_results = analyze_technical_feature_associations(shap_df, DB_PATH)
    if len(tech_results) > 0:
        tech_file = OUTPUT_DIR / "technical_feature_associations.csv"
        tech_results.to_csv(tech_file, index=False)
        print(f"Technical feature results saved to: {tech_file}")

    # Generate summary report
    report_file = OUTPUT_DIR / "shap_associations_report.txt"
    generate_summary_report(sex_results, race_eth_results, tech_results, report_file)

    # Generate presentation tables
    print("\n" + "=" * 80)
    print("Creating presentation-friendly tables...")
    print("=" * 80)

    from create_presentation_tables import (
        create_sex_presentation_table,
        create_race_presentation_table,
        create_tech_presentation_table,
        generate_sex_legend,
        generate_race_legend,
        generate_tech_legend,
        load_oncotree_mapping
    )

    # Load OncoTree mapping
    oncotree_mapping = load_oncotree_mapping(DB_PATH)

    # Create presentation tables
    if len(sex_results) > 0:
        sex_pres = create_sex_presentation_table(sex_results, oncotree_mapping)
        if len(sex_pres) > 0:
            sex_pres_file = OUTPUT_DIR / "sex_associations_PRESENTATION.csv"
            sex_pres.to_csv(sex_pres_file, index=False)
            print(f"Sex presentation table: {sex_pres_file}")

            # Generate legend
            sex_legend_file = OUTPUT_DIR / "sex_associations_LEGEND.md"
            generate_sex_legend(sex_legend_file)

    if len(race_eth_results) > 0:
        race_pres = create_race_presentation_table(race_eth_results, oncotree_mapping)
        if len(race_pres) > 0:
            race_pres_file = OUTPUT_DIR / "race_ethnicity_associations_PRESENTATION.csv"
            race_pres.to_csv(race_pres_file, index=False)
            print(f"Race/Ethnicity presentation table: {race_pres_file}")

            # Generate legend
            race_legend_file = OUTPUT_DIR / "race_ethnicity_associations_LEGEND.md"
            generate_race_legend(race_legend_file)

    if len(tech_results) > 0:
        tech_pres = create_tech_presentation_table(tech_results, oncotree_mapping)
        if len(tech_pres) > 0:
            tech_pres_file = OUTPUT_DIR / "technical_feature_associations_PRESENTATION.csv"
            tech_pres.to_csv(tech_pres_file, index=False)
            print(f"Technical features presentation table: {tech_pres_file}")

            # Generate legend
            tech_legend_file = OUTPUT_DIR / "technical_feature_associations_LEGEND.md"
            generate_tech_legend(tech_legend_file)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
