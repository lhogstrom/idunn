"""
Unit tests for preprocessing.py

Tests feature list retrieval, preprocessor creation, and column type classification.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncokb_analyzer.preprocessing import (
    get_feature_list,
    create_preprocessor,
    create_pipeline,
    get_column_types,
    validate_feature_set,
    DEMOGRAPHIC_FEATURES,
    TECHNICAL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)


class TestFeatureConstants:
    """Tests for feature constant definitions."""

    def test_demographic_features_not_empty(self):
        """Test that demographic features list is not empty."""
        assert len(DEMOGRAPHIC_FEATURES) > 0

    def test_technical_features_not_empty(self):
        """Test that technical features list is not empty."""
        assert len(TECHNICAL_FEATURES) > 0

    def test_numeric_features_not_empty(self):
        """Test that numeric features list is not empty."""
        assert len(NUMERIC_FEATURES) > 0

    def test_categorical_features_not_empty(self):
        """Test that categorical features list is not empty."""
        assert len(CATEGORICAL_FEATURES) > 0

    def test_no_overlap_demographic_technical(self):
        """Test that demographic and technical features don't overlap."""
        overlap = set(DEMOGRAPHIC_FEATURES) & set(TECHNICAL_FEATURES)
        assert len(overlap) == 0, f"Unexpected overlap: {overlap}"

    def test_all_features_classified(self):
        """Test that all features are classified as numeric or categorical."""
        all_features = set(DEMOGRAPHIC_FEATURES + TECHNICAL_FEATURES)
        classified = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

        # All features should be in either numeric or categorical
        for feature in all_features:
            assert feature in classified, f"Feature '{feature}' not classified"


class TestGetFeatureList:
    """Tests for get_feature_list() function."""

    def test_get_demographic_features(self):
        """Test getting demographic feature list."""
        features = get_feature_list('demographic')

        assert isinstance(features, list)
        assert len(features) > 0
        assert features == DEMOGRAPHIC_FEATURES

    def test_get_technical_features(self):
        """Test getting technical feature list."""
        features = get_feature_list('technical')

        assert isinstance(features, list)
        assert len(features) > 0
        assert features == TECHNICAL_FEATURES

    def test_get_both_features(self):
        """Test getting combined feature list."""
        features = get_feature_list('both')

        assert isinstance(features, list)
        assert len(features) == len(DEMOGRAPHIC_FEATURES) + len(TECHNICAL_FEATURES)

        # Should contain all demographic and technical features
        for f in DEMOGRAPHIC_FEATURES:
            assert f in features
        for f in TECHNICAL_FEATURES:
            assert f in features

    def test_invalid_feature_set_raises_error(self):
        """Test that invalid feature_set raises ValueError."""
        with pytest.raises(ValueError, match="feature_set must be one of"):
            get_feature_list('invalid')

    def test_returns_copy_not_reference(self):
        """Test that get_feature_list returns a copy."""
        features1 = get_feature_list('demographic')
        features2 = get_feature_list('demographic')

        # Modifying one should not affect the other
        features1.append('NEW_FEATURE')
        assert 'NEW_FEATURE' not in features2


class TestValidateFeatureSet:
    """Tests for validate_feature_set() function."""

    def test_valid_feature_sets_pass(self):
        """Test that valid feature sets don't raise errors."""
        validate_feature_set('demographic')  # Should not raise
        validate_feature_set('technical')    # Should not raise
        validate_feature_set('both')         # Should not raise

    def test_invalid_feature_set_raises_error(self):
        """Test that invalid feature set raises ValueError."""
        with pytest.raises(ValueError):
            validate_feature_set('invalid_set')

        with pytest.raises(ValueError):
            validate_feature_set('')


class TestCreatePreprocessor:
    """Tests for create_preprocessor() function."""

    def test_create_preprocessor_default(self):
        """Test creating preprocessor with default settings."""
        preprocessor = create_preprocessor()

        assert preprocessor is not None
        # Check that it has transformers
        assert hasattr(preprocessor, 'transformers')
        assert len(preprocessor.transformers) == 2  # num and cat

    def test_create_preprocessor_with_columns(self):
        """Test creating preprocessor with specific columns."""
        columns = ['SEX', 'AGE_AT_SEQ_REPORT', 'RACE']
        preprocessor = create_preprocessor(available_columns=columns)

        assert preprocessor is not None

    def test_preprocessor_fits_data(self, toy_dataframe):
        """Test that preprocessor can fit data."""
        # Use only demographic features
        feature_cols = [c for c in toy_dataframe.columns
                        if c not in ['isOncoKbLevel1', 'PATIENT_ID']]
        X = toy_dataframe[feature_cols]

        preprocessor = create_preprocessor(available_columns=feature_cols)
        preprocessor.fit(X)

        # Should be able to transform after fitting
        X_transformed = preprocessor.transform(X)
        assert X_transformed is not None
        assert X_transformed.shape[0] == len(X)


class TestCreatePipeline:
    """Tests for create_pipeline() function."""

    def test_create_pipeline_random_forest(self):
        """Test creating pipeline with RandomForest."""
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        pipeline = create_pipeline(rf)

        assert pipeline is not None
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == 'preprocessor'
        assert pipeline.steps[1][0] == 'classifier'

    def test_create_pipeline_logistic_regression(self):
        """Test creating pipeline with LogisticRegression."""
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(random_state=42)
        pipeline = create_pipeline(lr)

        assert pipeline is not None
        assert len(pipeline.steps) == 2

    def test_pipeline_fits_and_predicts(self, toy_dataframe):
        """Test that pipeline can fit and predict."""
        from sklearn.ensemble import RandomForestClassifier

        feature_cols = ['SEX', 'AGE_AT_SEQ_REPORT', 'RACE', 'SUBJECT_CENTER']
        X = toy_dataframe[feature_cols]
        y = toy_dataframe['isOncoKbLevel1']

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        pipeline = create_pipeline(rf, available_columns=feature_cols)

        # Fit
        pipeline.fit(X, y)

        # Predict
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)

        # Predict probabilities
        proba = pipeline.predict_proba(X)
        assert proba.shape[0] == len(X)
        assert proba.shape[1] == 2  # Binary classification


class TestGetColumnTypes:
    """Tests for get_column_types() function."""

    def test_classify_known_columns(self):
        """Test classifying known columns."""
        columns = ['SEX', 'AGE_AT_SEQ_REPORT', 'RACE']
        types = get_column_types(columns)

        assert 'numeric' in types
        assert 'categorical' in types

        # AGE_AT_SEQ_REPORT should be numeric
        assert 'AGE_AT_SEQ_REPORT' in types['numeric']

        # SEX and RACE should be categorical
        assert 'SEX' in types['categorical']
        assert 'RACE' in types['categorical']

    def test_classify_empty_list(self):
        """Test classifying empty column list."""
        types = get_column_types([])

        assert types['numeric'] == []
        assert types['categorical'] == []

    def test_classify_unknown_columns(self):
        """Test that unknown columns are not classified."""
        columns = ['UNKNOWN_COLUMN_1', 'UNKNOWN_COLUMN_2']
        types = get_column_types(columns)

        assert types['numeric'] == []
        assert types['categorical'] == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
