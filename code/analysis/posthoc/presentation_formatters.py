"""Formatting utilities for presentation-ready post-hoc analysis tables."""

import pandas as pd
import numpy as np
import sqlite3
import re


def format_scientific(value, precision=2):
    """Format a float, using scientific notation for very small/large values ('NA' for NaN)."""
    if pd.isna(value):
        return "NA"

    if abs(value) < 0.001 or abs(value) > 10000:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"


def format_ci(estimate, ci_lower, ci_upper, precision=2):
    """Format an estimate with CI as "estimate [lower-upper]" ("NA" if missing)."""
    if any(pd.isna([estimate, ci_lower, ci_upper])):
        return "NA"

    return f"{estimate:.{precision}f} [{ci_lower:.{precision}f}-{ci_upper:.{precision}f}]"


def create_association_label(cancer_name, feature_name, feature_type=None):
    """Create a readable association label, e.g. "SEX_Female" -> "Actionability vs. Female sex".

    Handles sklearn-style prefixes such as "cat__".
    """
    # Strip sklearn pipeline prefixes (e.g. 'cat__SEX_Female' -> 'SEX_Female').
    clean_feature = feature_name
    if '__' in feature_name:
        clean_feature = feature_name.split('__')[-1]

    # Extract the value from names like SEX_Female.
    if '_' in clean_feature:
        parts = clean_feature.split('_')
        prefix = parts[0]
        value = '_'.join(parts[1:])
    else:
        prefix = clean_feature
        value = ""

    if feature_type is None:
        prefix_upper = prefix.upper()
        if prefix_upper == 'SEX':
            feature_type = 'sex'
        elif prefix_upper == 'RACE':
            feature_type = 'ancestry'
        elif prefix_upper == 'ETHNICITY':
            feature_type = 'ethnicity'
        elif 'LIBRARY' in prefix_upper or 'CALLING' in prefix_upper or 'CENTER' in prefix_upper:
            feature_type = 'technical factor'
        else:
            feature_type = 'feature'

    if value:
        return f"Actionability vs. {value} {feature_type}"
    else:
        return f"Actionability vs. {clean_feature}"


def load_oncotree_mapping(db_path):
    """Load the OncoTree code -> name mapping from the database."""
    conn = sqlite3.connect(db_path)

    query = """
    SELECT DISTINCT ot_code, ot_name
    FROM OncoTreeCodeHierarchy
    WHERE ot_code IS NOT NULL AND ot_name IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    mapping = dict(zip(df['ot_code'], df['ot_name']))

    return mapping


def filter_unknown_other(df, feature_column='feature'):
    """Drop rows whose feature value is "Other" or "Unknown" (uninterpretable associations)."""
    exclude_patterns = [
        'unknown',
        'other',
        'not reported',
        'not available',
        'missing',
        'unspecified',
        'not specified',
        'nr'
    ]

    pattern = '|'.join(exclude_patterns)

    mask = ~df[feature_column].str.contains(pattern, case=False, na=False, regex=True)

    filtered_df = df[mask].copy()

    n_filtered = len(df) - len(filtered_df)
    if n_filtered > 0:
        print(f"  Filtered out {n_filtered} 'Other/Unknown' associations")

    return filtered_df


def format_shap_importance(value, precision=3):
    """Format mean absolute SHAP importance with fixed decimals."""
    if pd.isna(value):
        return "NA"

    return f"{value:.{precision}f}"


def get_feature_type_from_name(feature_name):
    """Infer feature type ('sex', 'ancestry', 'ethnicity', 'technical', 'other') from the feature name."""
    feature_upper = feature_name.upper()

    if feature_upper.startswith('SEX_'):
        return 'sex'
    elif feature_upper.startswith('RACE_'):
        return 'ancestry'
    elif feature_upper.startswith('ETHNICITY_'):
        return 'ethnicity'
    elif any(x in feature_upper for x in ['LIBRARY', 'CALLING', 'CENTER', 'COVERAGE', 'PANEL']):
        return 'technical factor'
    else:
        return 'feature'


def add_cancer_type_names(df, oncotree_mapping, code_column='ONCOTREE_CODE'):
    """Add a 'cancer_type' name column from OncoTree codes (falls back to the code)."""
    df = df.copy()

    df['cancer_type'] = df[code_column].map(oncotree_mapping)

    # Unmapped codes keep the code itself.
    df['cancer_type'] = df['cancer_type'].fillna(df[code_column])

    return df


def sort_by_significance(df, p_column='p_fdr', effect_column=None, ascending_effect=False):
    """Sort by p-value ascending, then by effect size."""
    df = df.copy()

    if effect_column and effect_column in df.columns:
        df = df.sort_values([p_column, effect_column],
                           ascending=[True, ascending_effect])
    else:
        df = df.sort_values(p_column, ascending=True)

    return df


def create_presentation_metadata(n_original, n_filtered, n_significant,
                                 filter_criteria, significance_threshold=0.05):
    """Create the metadata dictionary for a presentation table."""
    metadata = {
        'n_original': n_original,
        'n_filtered': n_filtered,
        'n_removed': n_original - n_filtered,
        'n_significant': n_significant,
        'filter_criteria': filter_criteria,
        'significance_threshold': significance_threshold,
        'sort_order': ['p_fdr (ascending)', 'effect_size (descending)']
    }

    return metadata
