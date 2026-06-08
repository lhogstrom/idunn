"""
Formatting utilities for creating presentation-ready post-hoc analysis tables.

This module provides functions to transform raw statistical analysis output
into publication-ready tables with human-readable labels and appropriate
formatting.

"""

import pandas as pd
import numpy as np
import sqlite3
import re


def format_scientific(value, precision=2):
    """
    Format float with appropriate precision.

    Uses scientific notation for very small/large values, fixed decimal otherwise.

    Parameters
    ----------
    value : float
        Value to format
    precision : int, default=2
        Number of decimal places

    Returns
    -------
    str
        Formatted string

    Examples
    --------
    >>> format_scientific(0.00000123, 2)
    '1.23e-06'
    >>> format_scientific(1.234567, 2)
    '1.23'
    >>> format_scientific(np.nan, 2)
    'NA'
    """
    if pd.isna(value):
        return "NA"

    # Use scientific notation for very small or very large values
    if abs(value) < 0.001 or abs(value) > 10000:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"


def format_ci(estimate, ci_lower, ci_upper, precision=2):
    """
    Format estimate with confidence interval.

    Parameters
    ----------
    estimate : float
        Point estimate (e.g., odds ratio)
    ci_lower : float
        Lower confidence interval bound
    ci_upper : float
        Upper confidence interval bound
    precision : int, default=2
        Number of decimal places

    Returns
    -------
    str
        Formatted as "estimate [lower-upper]"

    Examples
    --------
    >>> format_ci(1.71, 1.62, 1.81, 2)
    '1.71 [1.62-1.81]'
    >>> format_ci(np.nan, 1.0, 2.0, 2)
    'NA'
    """
    if any(pd.isna([estimate, ci_lower, ci_upper])):
        return "NA"

    return f"{estimate:.{precision}f} [{ci_lower:.{precision}f}-{ci_upper:.{precision}f}]"


def create_association_label(cancer_name, feature_name, feature_type=None):
    """
    Create human-readable association label.

    Transforms feature names like "SEX_Female" or "RACE_Asian" into
    readable labels like "Actionability vs. Female sex" or
    "Actionability vs. Asian ancestry".

    Parameters
    ----------
    cancer_name : str
        Human-readable cancer type name
    feature_name : str
        Raw feature name (e.g., "SEX_Female", "RACE_Asian", "cat__SEX_Female")
    feature_type : str, optional
        Override feature type detection

    Returns
    -------
    str
        Association label

    Examples
    --------
    >>> create_association_label("Lung Adenocarcinoma", "SEX_Female")
    'Actionability vs. Female sex'
    >>> create_association_label("Melanoma", "RACE_Asian")
    'Actionability vs. Asian ancestry'
    >>> create_association_label("Breast Cancer", "ETHNICITY_Hispanic")
    'Actionability vs. Hispanic ethnicity'
    """
    # Remove common prefixes from sklearn pipeline naming
    # e.g., "cat__SEX_Female" → "SEX_Female"
    clean_feature = feature_name
    if '__' in feature_name:
        clean_feature = feature_name.split('__')[-1]

    # Extract the value from the feature name
    # Feature names are like: SEX_Female, RACE_Asian, ETHNICITY_Hispanic, etc.
    if '_' in clean_feature:
        parts = clean_feature.split('_')
        prefix = parts[0]
        value = '_'.join(parts[1:])  # Handle values with underscores
    else:
        # Fallback if no underscore
        prefix = clean_feature
        value = ""

    # Determine feature type if not provided
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

    # Create label
    if value:
        return f"Actionability vs. {value} {feature_type}"
    else:
        return f"Actionability vs. {clean_feature}"


def load_oncotree_mapping(db_path):
    """
    Load OncoTree code to name mapping from database.

    Parameters
    ----------
    db_path : str
        Path to SQLite database

    Returns
    -------
    dict
        Mapping of ONCOTREE_CODE to human-readable name

    Examples
    --------
    >>> mapping = load_oncotree_mapping("path/to/db.sqlite")
    >>> mapping['LUAD']
    'Lung Adenocarcinoma'
    """
    conn = sqlite3.connect(db_path)

    query = """
    SELECT DISTINCT ot_code, ot_name
    FROM OncoTreeCodeHierarchy
    WHERE ot_code IS NOT NULL AND ot_name IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Create mapping dictionary
    mapping = dict(zip(df['ot_code'], df['ot_name']))

    return mapping


def filter_unknown_other(df, feature_column='feature'):
    """
    Filter out rows with "Other" or "Unknown" feature values.

    These associations are often uninterpretable and should be excluded
    from presentation tables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_column : str, default='feature'
        Name of column containing feature names

    Returns
    -------
    pd.DataFrame
        Filtered dataframe

    Examples
    --------
    >>> df = pd.DataFrame({'feature': ['SEX_Female', 'RACE_Other', 'SEX_Male']})
    >>> filtered = filter_unknown_other(df)
    >>> len(filtered)
    2
    >>> 'RACE_Other' in filtered['feature'].values
    False
    """
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

    # Create regex pattern (case-insensitive)
    pattern = '|'.join(exclude_patterns)

    # Filter out rows matching any exclude pattern
    mask = ~df[feature_column].str.contains(pattern, case=False, na=False, regex=True)

    filtered_df = df[mask].copy()

    n_filtered = len(df) - len(filtered_df)
    if n_filtered > 0:
        print(f"  Filtered out {n_filtered} 'Other/Unknown' associations")

    return filtered_df


def format_shap_importance(value, precision=3):
    """
    Format SHAP importance value.

    Parameters
    ----------
    value : float
        SHAP importance (mean absolute SHAP value)
    precision : int, default=3
        Number of decimal places

    Returns
    -------
    str
        Formatted value

    Examples
    --------
    >>> format_shap_importance(0.01734)
    '0.017'
    >>> format_shap_importance(0.0001234)
    '0.000'
    """
    if pd.isna(value):
        return "NA"

    # Always use fixed decimal for SHAP (typically small values 0.001-0.10)
    return f"{value:.{precision}f}"


def get_feature_type_from_name(feature_name):
    """
    Determine feature type from feature name.

    Parameters
    ----------
    feature_name : str
        Feature name (e.g., "SEX_Female", "RACE_Asian")

    Returns
    -------
    str
        Feature type ('sex', 'ancestry', 'ethnicity', 'technical', 'other')
    """
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
    """
    Add human-readable cancer type names to dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with OncoTree codes
    oncotree_mapping : dict
        Mapping from code to name
    code_column : str, default='ONCOTREE_CODE'
        Name of column containing OncoTree codes

    Returns
    -------
    pd.DataFrame
        Dataframe with added 'cancer_type' column
    """
    df = df.copy()

    # Map codes to names
    df['cancer_type'] = df[code_column].map(oncotree_mapping)

    # For any unmapped codes, use the code itself
    df['cancer_type'] = df['cancer_type'].fillna(df[code_column])

    return df


def sort_by_significance(df, p_column='p_fdr', effect_column=None, ascending_effect=False):
    """
    Sort dataframe by significance and effect size.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    p_column : str, default='p_fdr'
        Column name for p-values
    effect_column : str, optional
        Column name for effect size (odds_ratio or cramers_v)
    ascending_effect : bool, default=False
        Sort effect size ascending (True) or descending (False)

    Returns
    -------
    pd.DataFrame
        Sorted dataframe
    """
    df = df.copy()

    # Sort by p-value (ascending - most significant first)
    # Then by effect size (descending by default - largest effects first)
    if effect_column and effect_column in df.columns:
        df = df.sort_values([p_column, effect_column],
                           ascending=[True, ascending_effect])
    else:
        df = df.sort_values(p_column, ascending=True)

    return df


def create_presentation_metadata(n_original, n_filtered, n_significant,
                                 filter_criteria, significance_threshold=0.05):
    """
    Create metadata dictionary for presentation table.

    Parameters
    ----------
    n_original : int
        Number of rows in original table
    n_filtered : int
        Number of rows after filtering
    n_significant : int
        Number of significant associations
    filter_criteria : list
        List of filter criteria applied
    significance_threshold : float, default=0.05
        FDR significance threshold

    Returns
    -------
    dict
        Metadata dictionary
    """
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
