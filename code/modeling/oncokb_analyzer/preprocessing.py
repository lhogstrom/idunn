"""
Preprocessing Pipeline for OncoKB Analyzer

Shared preprocessing logic used by both trainer.py and hyperparameter_optimizer.py.
Handles imputation and encoding of demographic and technical features.

Feature definitions are centralized in code/config/features.yaml and loaded
via code/tools/feature_config.py. This module re-exports constants for
backwards compatibility with existing imports.

"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import List, Optional
import sys
from pathlib import Path

# Add tools to path for feature_config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from feature_config import (
    get_feature_list as _get_feature_list,
    get_numeric_features as _get_numeric_features,
    get_categorical_features as _get_categorical_features,
)

from .missingness_indicator import MissingnessIndicatorTransformer

# Columns where the string "Unknown" should be split into a binary indicator
# and mode-imputed (Feature Review Fix b, 2026-05-21). NKI and CRUK are 100%
# Unknown for both — treating Unknown as a one-hot category turned those
# institutions into a covert demographic signal.
MISSINGNESS_INDICATOR_COLS = ['ANCESTRY', 'ETHNICITY']
MISSINGNESS_INDICATOR_SUFFIX = '_missing'


# ============================================================================
# Feature Definitions (loaded from centralized config)
# ============================================================================
# These are re-exported for backwards compatibility with existing imports
# SINGLE SOURCE OF TRUTH: code/config/features.yaml

# Demographic features (no data leakage)
DEMOGRAPHIC_FEATURES = _get_feature_list('demographic')

# Technical features (may cause data leakage in cross-institution prediction)
TECHNICAL_FEATURES = _get_feature_list('technical')

# Numeric features (subset of all features)
NUMERIC_FEATURES = _get_numeric_features()

# Categorical features (all others are categorical)
CATEGORICAL_FEATURES = _get_categorical_features()


def get_feature_list(feature_set: str = 'demographic') -> List[str]:
    """
    Get list of features for specified feature set.

    Parameters
    ----------
    feature_set : str, default='demographic'
        Which features to include:
        - 'demographic': Only demographic features (no data leakage)
        - 'technical': Only technical/sequencing features (may leak)
        - 'both': All features

    Returns
    -------
    feature_list : list of str
        List of feature names

    Raises
    ------
    ValueError
        If feature_set not in ['demographic', 'technical', 'both']

    Examples
    --------
    >>> get_feature_list('demographic')
    ['SAMPLE_TYPE_DETAILED', 'SEX', 'ANCESTRY', 'ETHNICITY', 'AGE_AT_SEQ_REPORT', 'ANCESTRY_missing', 'ETHNICITY_missing']
    """
    valid_feature_sets = ['demographic', 'technical', 'both']
    if feature_set not in valid_feature_sets:
        raise ValueError(
            f"feature_set must be one of {valid_feature_sets}, got: {feature_set}"
        )

    if feature_set == 'demographic':
        return DEMOGRAPHIC_FEATURES.copy()
    elif feature_set == 'technical':
        return TECHNICAL_FEATURES.copy()
    else:  # 'both'
        return DEMOGRAPHIC_FEATURES + TECHNICAL_FEATURES


def _expand_for_missingness(
    available_columns: Optional[List[str]],
) -> Optional[List[str]]:
    """
    Append `<col>_missing` indicator columns to `available_columns` for each
    source column in MISSINGNESS_INDICATOR_COLS that's present.

    The MissingnessIndicatorTransformer prepended to the preprocessor will
    create these columns at fit/transform time; advertising them in
    `available_columns` lets the downstream ColumnTransformer route them as
    numeric features.
    """
    if available_columns is None:
        return None
    expanded = list(available_columns)
    for src in MISSINGNESS_INDICATOR_COLS:
        if src in available_columns:
            indicator = f"{src}{MISSINGNESS_INDICATOR_SUFFIX}"
            if indicator not in expanded:
                expanded.append(indicator)
    return expanded


def create_preprocessor(available_columns: Optional[List[str]] = None):
    """
    Create preprocessing pipeline for OncoKB data.

    Pipeline structure (Feature Review Fix b, 2026-05-21):
      1. MissingnessIndicatorTransformer — for each col in
         MISSINGNESS_INDICATOR_COLS that's present, emits a binary indicator
         column and mode-imputes the "Unknown" sentinel using the training-set
         mode.
      2. ColumnTransformer — numeric features get median imputation;
         categorical features get most-frequent imputation + one-hot encoding.

    When none of the missingness source columns are present, step 1 is
    skipped and the returned value is the bare ColumnTransformer (same
    contract as before).

    Parameters
    ----------
    available_columns : list of str, optional
        List of columns actually present in the input DataFrame at fit time
        (typically `X.columns.tolist()`). If provided, the returned pipeline
        is configured to handle exactly those columns plus any indicator
        columns the missingness step will add.

    Returns
    -------
    preprocessor : Pipeline or ColumnTransformer
        Scikit-learn preprocessor. Always returns a Pipeline when any
        MISSINGNESS_INDICATOR_COLS source is present; otherwise returns the
        bare ColumnTransformer for backwards compatibility.
    """
    # Determine which missingness source columns apply, if any.
    #
    # When `available_columns is None`, we have no way to know whether the
    # caller's DataFrame actually contains ANCESTRY/ETHNICITY. Configuring the
    # missingness step unconditionally would crash any caller whose data lacks
    # those columns — including the legacy numpy-array entrypoint in
    # hyperparameter_optimizer.py that wraps arrays as DataFrames with synthetic
    # integer column names (pre-lock review #8, 2026-05-22). Skip the step in
    # that case; callers that actually have ANCESTRY/ETHNICITY should pass
    # `available_columns` explicitly so the step gets configured.
    if available_columns is None:
        missingness_cols = []
    else:
        missingness_cols = [
            c for c in MISSINGNESS_INDICATOR_COLS if c in available_columns
        ]

    # Expand the column manifest with indicator names so the ColumnTransformer
    # below routes them as numeric features.
    available_columns = _expand_for_missingness(available_columns)

    # Start with all possible numeric and categorical columns (NUMERIC_FEATURES
    # already contains ANCESTRY_missing / ETHNICITY_missing via features.yaml).
    numeric_cols = NUMERIC_FEATURES.copy()
    categorical_cols = CATEGORICAL_FEATURES.copy()

    if available_columns is not None:
        numeric_cols = [col for col in numeric_cols if col in available_columns]
        categorical_cols = [col for col in categorical_cols if col in available_columns]

    # Use column NAMES (not indices). The MissingnessIndicatorTransformer
    # produces a DataFrame, so the ColumnTransformer can address columns by
    # name. This also keeps the contract uniform whether or not the
    # missingness step is present.
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ]), categorical_cols)
        ],
        remainder='drop',
    )

    if not missingness_cols:
        return column_transformer

    return Pipeline([
        ('missingness', MissingnessIndicatorTransformer(
            columns=missingness_cols,
            missing_value='Unknown',
            indicator_suffix=MISSINGNESS_INDICATOR_SUFFIX,
        )),
        ('column_transformer', column_transformer),
    ])


def create_pipeline(estimator, available_columns: Optional[List[str]] = None) -> Pipeline:
    """
    Create full preprocessing + model pipeline.

    Convenience function that combines preprocessing with a classifier.

    Parameters
    ----------
    estimator : sklearn estimator
        The classification model (e.g., RandomForestClassifier, LogisticRegression)
    available_columns : list of str, optional
        List of columns actually present in the data

    Returns
    -------
    pipeline : Pipeline
        Complete pipeline: preprocessing → classifier

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> rf = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> pipeline = create_pipeline(rf, available_columns=['SEX', 'ANCESTRY', 'AGE_AT_SEQ_REPORT'])
    >>> pipeline.fit(X_train, y_train)
    >>> predictions = pipeline.predict(X_test)
    """
    preprocessor = create_preprocessor(available_columns=available_columns)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])

    return pipeline


def get_column_types(columns: List[str]) -> dict:
    """
    Classify columns as numeric or categorical.

    Useful for understanding which columns will be processed how.

    Parameters
    ----------
    columns : list of str
        List of column names

    Returns
    -------
    column_types : dict
        Dictionary with keys 'numeric' and 'categorical', each containing
        a list of column names

    Examples
    --------
    >>> cols = ['SEX', 'AGE_AT_SEQ_REPORT', 'ANCESTRY']
    >>> types = get_column_types(cols)
    >>> print(types['numeric'])
    ['AGE_AT_SEQ_REPORT']
    >>> print(types['categorical'])
    ['SEX', 'ANCESTRY']
    """
    numeric = [col for col in columns if col in NUMERIC_FEATURES]
    categorical = [col for col in columns if col in CATEGORICAL_FEATURES]

    return {
        'numeric': numeric,
        'categorical': categorical
    }


# ============================================================================
# Utility Functions
# ============================================================================

def validate_feature_set(feature_set: str):
    """
    Validate that feature_set is valid.

    Parameters
    ----------
    feature_set : str
        Feature set name to validate

    Raises
    ------
    ValueError
        If feature_set not in ['demographic', 'technical', 'both']
    """
    valid_feature_sets = ['demographic', 'technical', 'both']
    if feature_set not in valid_feature_sets:
        raise ValueError(
            f"feature_set must be one of {valid_feature_sets}, got: {feature_set}"
        )


if __name__ == '__main__':
    # Print feature information
    print("="*80)
    print("PREPROCESSING PIPELINE CONFIGURATION")
    print("="*80)

    print("\nDemographic Features:")
    print(f"  Count: {len(DEMOGRAPHIC_FEATURES)}")
    print(f"  Features: {DEMOGRAPHIC_FEATURES}")

    print("\nTechnical Features:")
    print(f"  Count: {len(TECHNICAL_FEATURES)}")
    print(f"  Features: {TECHNICAL_FEATURES}")

    print("\nNumeric Features:")
    print(f"  Count: {len(NUMERIC_FEATURES)}")
    print(f"  Features: {NUMERIC_FEATURES}")

    print("\nCategorical Features:")
    print(f"  Count: {len(CATEGORICAL_FEATURES)}")
    print(f"  Features: {CATEGORICAL_FEATURES}")

    print("\n" + "="*80)
    print("EXAMPLE USAGE")
    print("="*80)

    # Example 1: Get feature list
    demo_features = get_feature_list('demographic')
    print(f"\nDemographic feature list: {demo_features}")

    # Example 2: Create preprocessor
    preprocessor = create_preprocessor(available_columns=demo_features)
    print(f"\nPreprocessor created with {len(demo_features)} available columns")

    # Example 3: Classify columns
    types = get_column_types(demo_features)
    print(f"\nColumn types:")
    print(f"  Numeric: {types['numeric']}")
    print(f"  Categorical: {types['categorical']}")
