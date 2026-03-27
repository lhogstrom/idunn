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
from shap_utils import (
    get_default_shap_file,
    load_shap_rankings_adaptive,
    identify_top_features,
    load_multi_tier_cancer_types,
    FEATURE_PATTERNS,
    REPO_ROOT
)
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

# Configuration
DB_PATH = get_default_db_path()

DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis_output" / "posthoc_statistics"
OUTPUT_DIR = None  # Set by main()

TABLE_NAME = "perPatientOncoKBFilteredForModeling"
TOP_N_FEATURES = 5


def load_shap_rankings(shap_file):
    """Load SHAP feature rankings (wrapper for backward compatibility)"""
    return load_shap_rankings_adaptive(shap_file)


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
    # Apply Haldane correction (+0.5 to all cells) when any cell is zero
    cells = [female_act, female_non_act, male_act, male_non_act]
    if any(c == 0 for c in cells):
        ha_fa, ha_fn, ha_ma, ha_mn = [c + 0.5 for c in cells]
        odds_ratio = (ha_fa * ha_mn) / (ha_ma * ha_fn)
    elif female_non_act > 0 and male_non_act > 0 and male_act > 0:
        odds_ratio = (female_act * male_non_act) / (male_act * female_non_act)
    else:
        odds_ratio = np.nan

    # Decide on test: Fisher's exact if expected frequencies < 5, chi-square otherwise
    # Compute expected frequencies for the 2x2 table
    grand_total = female_total + male_total
    total_act = female_act + male_act
    total_non_act = female_non_act + male_non_act

    if grand_total > 0:
        expected_cells = [
            (female_total * total_non_act) / grand_total,  # female, non-actionable
            (female_total * total_act) / grand_total,      # female, actionable
            (male_total * total_non_act) / grand_total,    # male, non-actionable
            (male_total * total_act) / grand_total         # male, actionable
        ]
        min_expected = min(expected_cells)
    else:
        min_expected = 0

    # Prepare contingency table for tests
    table = [[female_non_act, female_act], [male_non_act, male_act]]

    # Calculate CI for odds ratio using Woolf's method (with Haldane correction if needed)
    # Haldane correction: add 0.5 to all cells when any cell is zero
    cells_for_ci = [female_act, female_non_act, male_act, male_non_act]
    if any(c == 0 for c in cells_for_ci):
        ci_fa, ci_fn, ci_ma, ci_mn = [c + 0.5 for c in cells_for_ci]
        or_for_ci = (ci_fa * ci_mn) / (ci_ma * ci_fn)
    else:
        ci_fa, ci_fn, ci_ma, ci_mn = cells_for_ci
        or_for_ci = odds_ratio

    if or_for_ci > 0:
        log_or = np.log(or_for_ci)
        se_log_or = np.sqrt(1/ci_fa + 1/ci_fn + 1/ci_ma + 1/ci_mn)
        ci_lower = np.exp(log_or - 1.96 * se_log_or)
        ci_upper = np.exp(log_or + 1.96 * se_log_or)
    else:
        ci_lower, ci_upper = np.nan, np.nan

    if min_expected < 5:
        # Use Fisher's exact test for small expected frequencies
        try:
            odds_ratio_fisher, p_value = fisher_exact(table)
            test_used = 'fisher_exact'
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

        # Check expected frequency assumption
        min_expected_freq = expected.min()
        pct_below_5 = (expected < 5).sum() / expected.size * 100
        warning = None
        if min_expected_freq < 1:
            warning = f'Chi-square unreliable: {pct_below_5:.0f}% of expected frequencies < 5 (min={min_expected_freq:.1f})'
        elif min_expected_freq < 5:
            warning = f'Chi-square may be unreliable: {pct_below_5:.0f}% of expected frequencies < 5 (min={min_expected_freq:.1f})'

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
            'min_expected_freq': min_expected_freq,
            'contingency_table': contingency.to_dict(),
            'warning': warning
        }

    except Exception as e:
        return {
            'n_total': len(df),
            'chi2': np.nan,
            'p_value': np.nan,
            'cramers_v': np.nan,
            'dof': np.nan,
            'min_expected_freq': np.nan,
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

    if len(results_df) == 0:
        print("\nNo cancer types with SEX in top features — skipping FDR correction")
        return results_df

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

    if len(results_df) == 0:
        print("\nNo cancer types with RACE/ETHNICITY in top features — skipping FDR correction")
        return results_df

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


def analyze_sample_type_associations(shap_df, db_path):
    """Analyze SAMPLE_TYPE and SAMPLE_TYPE_DETAILED associations for top-ranked features.

    Tests whether sample type (Primary, Metastasis, etc.) is associated with
    actionability using Chi-square tests.
    """

    print("\n" + "=" * 80)
    print("Analyzing SAMPLE_TYPE associations")
    print("=" * 80)

    # Identify cancer types with sample type features in top 5
    sample_type_features = identify_top_features(
        shap_df,
        FEATURE_PATTERNS['SAMPLE_TYPE'],
        TOP_N_FEATURES
    )

    if len(sample_type_features) == 0:
        print("No cancer types with SAMPLE_TYPE features in top 5")
        return pd.DataFrame()

    # Get unique cancer types
    cancer_types = sample_type_features['ONCOTREE_CODE'].unique()
    print(f"\nTesting {len(cancer_types)} cancer types")

    # Load patient data
    df = load_patient_data(
        db_path,
        cancer_types,
        ['ONCOTREE_CODE', 'SAMPLE_TYPE', 'SAMPLE_TYPE_DETAILED', 'isOncoKbLevel1']
    )

    results = []

    for cancer_type in cancer_types:
        print(f"\nTesting {cancer_type}...")

        df_cancer = df[df['ONCOTREE_CODE'] == cancer_type].copy()

        # Get features for this cancer type
        features_ct = sample_type_features[sample_type_features['ONCOTREE_CODE'] == cancer_type]

        for _, feat_info in features_ct.iterrows():
            feature_name = feat_info['feature']

            # Determine column name (SAMPLE_TYPE or SAMPLE_TYPE_DETAILED)
            # Support both old format (SAMPLE_TYPE_) and new format (Sample Type)
            if 'SAMPLE_TYPE_DETAILED' in feature_name or 'Sample Type (Detailed)' in feature_name or 'Sample Type Detailed' in feature_name:
                column = 'SAMPLE_TYPE_DETAILED'
            elif 'SAMPLE_TYPE' in feature_name or 'Sample Type' in feature_name:
                column = 'SAMPLE_TYPE'
            else:
                continue

            print(f"  Testing {column}...")

            # Test association using chi-square (same as race/ethnicity)
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


def analyze_technical_feature_associations(shap_df, db_path):
    """Analyze technical feature associations for top-ranked features.

    Handles all technical features from features.yaml:
    - SUBJECT_CENTER (categorical): Chi-square test
    - number_of_genes (numeric): Mann-Whitney U test
    - preservation_technique (categorical): Chi-square test
    - calling_strategy (categorical): Chi-square test
    """

    print("\n" + "=" * 80)
    print("Analyzing TECHNICAL feature associations")
    print("=" * 80)

    # Identify cancer types with technical features in top 5
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

    # Map SHAP feature names to database column names
    # Key: patterns to match in SHAP feature name, Value: (db_column, is_numeric)
    TECH_FEATURE_MAPPING = {
        ('SUBJECT_CENTER', 'Subject Center'): ('SUBJECT_CENTER', False),
        ('number_of_genes', 'Number of Genes'): ('number_of_genes', True),
        ('preservation_technique', 'Preservation Technique', 'Preservation'): ('preservation_technique', False),
        ('calling_strategy', 'Calling Strategy', 'Variant Calling Strategy'): ('calling_strategy', False),
        ('library_selection', 'Library Selection'): ('library_selection', False),
    }

    # Load patient data
    conn = sqlite3.connect(db_path)

    # Get available columns
    query = f"PRAGMA table_info({TABLE_NAME})"
    col_info = pd.read_sql_query(query, conn)
    all_columns = col_info['name'].tolist()

    # Determine which columns to load
    columns_to_load = ['ONCOTREE_CODE', 'isOncoKbLevel1']
    for patterns, (db_col, _) in TECH_FEATURE_MAPPING.items():
        # Find the actual column name (may be case-insensitive match)
        matching_cols = [c for c in all_columns if c.lower() == db_col.lower()]
        if matching_cols:
            columns_to_load.append(matching_cols[0])

    columns_to_load = list(set(columns_to_load))  # Remove duplicates

    print(f"Loading columns: {columns_to_load}")

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

            # Find matching database column
            column = None
            is_numeric = False

            for patterns, (db_col, numeric) in TECH_FEATURE_MAPPING.items():
                if any(p.lower() in feature_name.lower() for p in patterns):
                    # Find actual column name in loaded data
                    matching_cols = [c for c in df_cancer.columns if c.lower() == db_col.lower()]
                    if matching_cols:
                        column = matching_cols[0]
                        is_numeric = numeric
                    break

            if column is None:
                print(f"  Skipping {feature_name} - no matching column")
                continue

            print(f"  Testing {column} ({'numeric' if is_numeric else 'categorical'})...")

            # Test association based on feature type
            if is_numeric:
                test_result = test_numeric_technical_feature(df_cancer, column)
            else:
                test_result = test_technical_feature_association(df_cancer, column)

            result = {
                'ONCOTREE_CODE': cancer_type,
                'feature': feature_name,
                'column': column,
                'feature_type': 'numeric' if is_numeric else 'categorical',
                'shap_rank': feat_info['feature_rank'],
                'shap_importance': feat_info['mean_abs_shap'],
                **test_result
            }

            results.append(result)

            if not pd.isna(test_result.get('p_value')):
                if is_numeric:
                    print(f"    Mann-Whitney U: {test_result.get('statistic', 'N/A')}")
                    print(f"    p-value: {test_result['p_value']:.4e}")
                    print(f"    Effect size (r): {test_result.get('effect_size', 'N/A'):.3f}" if test_result.get('effect_size') else "")
                else:
                    print(f"    Chi-square: {test_result.get('chi2', 'N/A'):.2f}" if test_result.get('chi2') else "")
                    print(f"    p-value: {test_result['p_value']:.4e}")
                    print(f"    Cramér's V: {test_result.get('cramers_v', 'N/A'):.3f}" if test_result.get('cramers_v') else "")

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


def test_numeric_technical_feature(df, column):
    """
    Test association between a numeric technical feature and actionability.

    Uses Mann-Whitney U test (non-parametric comparison of distributions).
    """
    try:
        df_clean = df.dropna(subset=[column, 'isOncoKbLevel1'])

        actionable = df_clean[df_clean['isOncoKbLevel1'] == 1][column]
        non_actionable = df_clean[df_clean['isOncoKbLevel1'] == 0][column]

        if len(actionable) < 5 or len(non_actionable) < 5:
            return {'p_value': np.nan, 'statistic': np.nan, 'effect_size': np.nan}

        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(actionable, non_actionable, alternative='two-sided')

        # Effect size: rank-biserial correlation (r)
        n1, n2 = len(actionable), len(non_actionable)
        effect_size = 1 - (2 * statistic) / (n1 * n2)

        return {
            'p_value': p_value,
            'statistic': statistic,
            'effect_size': abs(effect_size),
            'mean_actionable': actionable.mean(),
            'mean_non_actionable': non_actionable.mean(),
            'n_actionable': n1,
            'n_non_actionable': n2
        }
    except Exception as e:
        print(f"    Error testing {column}: {e}")
        return {'p_value': np.nan, 'statistic': np.nan, 'effect_size': np.nan}


def generate_summary_report(sex_results, race_eth_results, sample_type_results, tech_results, output_file):
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

        # SAMPLE_TYPE associations
        if len(sample_type_results) > 0:
            f.write("\n" + "=" * 80 + "\n")
            f.write("SAMPLE TYPE Associations\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Tests performed: {len(sample_type_results)}\n")
            sig_count = sample_type_results['significant_fdr'].sum()
            f.write(f"Significant (FDR < 0.05): {sig_count}\n\n")

            if sig_count > 0:
                sig = sample_type_results[sample_type_results['significant_fdr'] == True].copy()
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

  # Use all tiers (not just tier1)
  python analyze_shap_feature_associations.py --tier all

  # Disable multi-tier filtering
  python analyze_shap_feature_associations.py --no-tier-filter
        """
    )
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--shap-file', type=str, default=None,
                       help='Path to SHAP feature rankings CSV (new TIER 6a format or legacy format)')
    parser.add_argument('--tier', type=str, default='tier1',
                       help='Cancer type tier to use: tier1, tier2, tier3, or all (default: tier1)')
    parser.add_argument('--no-tier-filter', action='store_true',
                       help='Disable multi-tier filtering (use all cancer types in SHAP file)')
    parser.add_argument('--groupings-file', type=str, default=None,
                       help='Path to multi_tier_group_definitions.csv (auto-detected if not specified)')
    parser.add_argument('--lr-significant-only', action='store_true',
                       help='When using LR rankings, only include features with Wald-significant '
                            'coefficients (FDR < 0.05). Features without significant associations '
                            'in the LR model are excluded from post-hoc testing.')
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
    shap_df = load_shap_rankings_adaptive(shap_file, lr_significant_only=args.lr_significant_only)

    # Filter to canonical cancer types from multi-tier groupings
    if not args.no_tier_filter:
        try:
            groupings_file = Path(args.groupings_file) if args.groupings_file else None
            tier_filter = None if args.tier == 'all' else args.tier

            canonical_types = load_multi_tier_cancer_types(
                tier=tier_filter,
                definitions_file=groupings_file
            )
            original_count = shap_df['ONCOTREE_CODE'].nunique()
            shap_df = shap_df[shap_df['ONCOTREE_CODE'].isin(canonical_types)]
            filtered_count = shap_df['ONCOTREE_CODE'].nunique()

            print(f"\nFiltered to canonical cancer types (tier={args.tier}):")
            print(f"  Original: {original_count} cancer types")
            print(f"  After filter: {filtered_count} cancer types")
            print(f"  Cancer types: {', '.join(sorted(shap_df['ONCOTREE_CODE'].unique())[:10])}{'...' if filtered_count > 10 else ''}")
        except FileNotFoundError as e:
            print(f"\nWARNING: {e}")
            print("  Using all cancer types from SHAP file")
    else:
        print("\nUsing all cancer types from SHAP file (--no-tier-filter)")

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

    # Analyze SAMPLE_TYPE associations
    sample_type_results = analyze_sample_type_associations(shap_df, DB_PATH)
    if len(sample_type_results) > 0:
        sample_type_file = OUTPUT_DIR / "sample_type_associations.csv"
        sample_type_results.to_csv(sample_type_file, index=False)
        print(f"SAMPLE_TYPE results saved to: {sample_type_file}")

    # Analyze technical feature associations
    tech_results = analyze_technical_feature_associations(shap_df, DB_PATH)
    if len(tech_results) > 0:
        tech_file = OUTPUT_DIR / "technical_feature_associations.csv"
        tech_results.to_csv(tech_file, index=False)
        print(f"Technical feature results saved to: {tech_file}")

    # Generate summary report
    report_file = OUTPUT_DIR / "shap_associations_report.txt"
    generate_summary_report(sex_results, race_eth_results, sample_type_results, tech_results, report_file)

    # Generate presentation tables
    print("\n" + "=" * 80)
    print("Creating presentation-friendly tables...")
    print("=" * 80)

    # Add staged directory to path for presentation table imports
    staged_dir = str(Path(__file__).parent.parent.parent / 'staged')
    if staged_dir not in sys.path:
        sys.path.insert(0, staged_dir)

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
