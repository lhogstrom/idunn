"""Shared preprocessing (imputation and encoding) used by the trainer and optimizer.

Feature definitions are loaded from code/config/features.yaml via feature_config.
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

# Columns where the string "Unknown" should be split into a binary indicator and mode-imputed.
MISSINGNESS_INDICATOR_COLS = ['ANCESTRY', 'ETHNICITY']
MISSINGNESS_INDICATOR_SUFFIX = '_missing'


# Feature definitions (re-exported from the central config).

# Demographic features (no data leakage)
DEMOGRAPHIC_FEATURES = _get_feature_list('demographic')

# Technical features (may cause data leakage in cross-institution prediction)
TECHNICAL_FEATURES = _get_feature_list('technical')

# Numeric features (subset of all features)
NUMERIC_FEATURES = _get_numeric_features()

# Categorical features (all others are categorical)
CATEGORICAL_FEATURES = _get_categorical_features()


def get_feature_list(feature_set: str = 'demographic') -> List[str]:
    """Get list of features for specified feature set."""
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
    """Append `<col>_missing` names to `available_columns` for each present source column."""
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
    """Create preprocessing pipeline for OncoKB data."""
    # Determine which missingness source columns apply, if any.
    if available_columns is None:
        missingness_cols = []
    else:
        missingness_cols = [
            c for c in MISSINGNESS_INDICATOR_COLS if c in available_columns
        ]

    # Add indicator names so the ColumnTransformer routes them as numeric.
    available_columns = _expand_for_missingness(available_columns)

    # Start from all possible numeric and categorical columns.
    numeric_cols = NUMERIC_FEATURES.copy()
    categorical_cols = CATEGORICAL_FEATURES.copy()

    if available_columns is not None:
        numeric_cols = [col for col in numeric_cols if col in available_columns]
        categorical_cols = [col for col in categorical_cols if col in available_columns]

    # Use column NAMES (not indices).
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
    """Create full preprocessing + model pipeline."""
    preprocessor = create_preprocessor(available_columns=available_columns)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', estimator)
    ])

    return pipeline


def get_column_types(columns: List[str]) -> dict:
    """Classify columns as numeric or categorical."""
    numeric = [col for col in columns if col in NUMERIC_FEATURES]
    categorical = [col for col in columns if col in CATEGORICAL_FEATURES]

    return {
        'numeric': numeric,
        'categorical': categorical
    }


# Utility Functions

def validate_feature_set(feature_set: str):
    """Validate that feature_set is valid."""
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
