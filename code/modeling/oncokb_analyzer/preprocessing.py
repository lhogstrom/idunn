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
    ['SAMPLE_TYPE', 'SAMPLE_TYPE_DETAILED', 'SEX', 'RACE', 'ETHNICITY', 'AGE_AT_SEQ_REPORT']
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


def create_preprocessor(available_columns: Optional[List[str]] = None) -> ColumnTransformer:
    """
    Create preprocessing pipeline for OncoKB data.

    This pipeline handles:
    - Numeric features: Median imputation
    - Categorical features: Most frequent imputation + one-hot encoding

    Parameters
    ----------
    available_columns : list of str, optional
        List of columns actually present in the data.
        If provided, filters numeric_cols and categorical_cols to only
        include columns in this list. This handles cases where some
        features are missing from the dataset.

    Returns
    -------
    preprocessor : ColumnTransformer
        Scikit-learn preprocessing pipeline

    Notes
    -----
    Uses integer indices instead of column names to ensure compatibility
    with numpy arrays (which scikit-learn's cross-validation produces).

    Examples
    --------
    >>> preprocessor = create_preprocessor()
    >>> # Use in a full pipeline:
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.pipeline import Pipeline
    >>> model = Pipeline([
    ...     ('preprocessor', preprocessor),
    ...     ('classifier', RandomForestClassifier())
    ... ])
    >>> model.fit(X, y)
    """
    # Start with all possible numeric and categorical columns
    numeric_cols = NUMERIC_FEATURES.copy()
    categorical_cols = CATEGORICAL_FEATURES.copy()

    # Filter to only available columns if specified
    if available_columns is not None:
        numeric_cols = [col for col in numeric_cols if col in available_columns]
        categorical_cols = [col for col in categorical_cols if col in available_columns]

        # Convert column names to indices (for numpy array compatibility)
        numeric_indices = [available_columns.index(col) for col in numeric_cols]
        categorical_indices = [available_columns.index(col) for col in categorical_cols]
    else:
        # If no available columns specified, use column names
        # (assumes input will be DataFrame)
        numeric_indices = numeric_cols
        categorical_indices = categorical_cols

    # Create preprocessing transformers
    preprocessor = ColumnTransformer(
        transformers=[
            # Numeric features: median imputation
            ('num', SimpleImputer(strategy='median'), numeric_indices),

            # Categorical features: most frequent imputation + one-hot encoding
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ]), categorical_indices)
        ],
        remainder='drop'  # Drop any columns not explicitly handled
    )

    return preprocessor


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
    >>> pipeline = create_pipeline(rf, available_columns=['SEX', 'RACE', 'AGE_AT_SEQ_REPORT'])
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
    >>> cols = ['SEX', 'AGE_AT_SEQ_REPORT', 'RACE']
    >>> types = get_column_types(cols)
    >>> print(types['numeric'])
    ['AGE_AT_SEQ_REPORT']
    >>> print(types['categorical'])
    ['SEX', 'RACE']
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
