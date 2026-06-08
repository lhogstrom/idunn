"""
Unit tests for trainer.py

Tests ModelTrainer initialization, feature preparation, and model training.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncokb_analyzer.trainer import ModelTrainer


class TestModelTrainerInitialization:
    """Tests for ModelTrainer initialization."""

    def test_default_initialization(self):
        """Test default ModelTrainer initialization."""
        trainer = ModelTrainer()

        assert trainer.algorithm == "random_forest"
        assert trainer.cv_folds == 5
        assert trainer.random_state == 42
        assert trainer.cv_strategy == "stratified_group"
        assert trainer.group_column == "SUBJECT_CENTER"

    def test_custom_initialization(self):
        """Test ModelTrainer with custom parameters."""
        trainer = ModelTrainer(
            algorithm="logistic",
            cv_folds=3,
            random_state=123,
            cv_strategy="standard",
            n_jobs=1
        )

        assert trainer.algorithm == "logistic"
        assert trainer.cv_folds == 3
        assert trainer.random_state == 123
        assert trainer.cv_strategy == "standard"
        assert trainer.n_jobs == 1

    def test_invalid_algorithm_raises_error(self):
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="Algorithm must be"):
            ModelTrainer(algorithm="invalid")

    def test_invalid_cv_strategy_raises_error(self):
        """Test that invalid cv_strategy raises ValueError."""
        with pytest.raises(ValueError, match="cv_strategy must be"):
            ModelTrainer(cv_strategy="invalid")


class TestPrepareFeatures:
    """Tests for ModelTrainer.prepare_features() method."""

    def test_prepare_demographic_features(self, toy_dataframe):
        """Test preparing demographic features."""
        trainer = ModelTrainer(cv_strategy="standard")

        X, y, groups = trainer.prepare_features(
            toy_dataframe,
            outcome_col='isOncoKbLevel1',
            feature_set='demographic'
        )

        assert X is not None
        assert y is not None
        assert len(X) == len(toy_dataframe)
        assert len(y) == len(toy_dataframe)

        # Check that only demographic columns are included
        assert 'SEX' in X.columns
        assert 'ANCESTRY' in X.columns
        assert 'AGE_AT_SEQ_REPORT' in X.columns

        # Technical columns should not be included
        assert 'SUBJECT_CENTER' not in X.columns
        assert 'number_of_genes' not in X.columns

    def test_prepare_technical_features(self, toy_dataframe):
        """Test preparing technical features."""
        trainer = ModelTrainer(cv_strategy="standard")

        X, y, groups = trainer.prepare_features(
            toy_dataframe,
            outcome_col='isOncoKbLevel1',
            feature_set='technical'
        )

        assert X is not None

        # Check that only technical columns are included
        assert 'SUBJECT_CENTER' in X.columns
        assert 'number_of_genes' in X.columns

        # Demographic columns should not be included
        assert 'SEX' not in X.columns
        assert 'ANCESTRY' not in X.columns

    def test_prepare_both_features(self, toy_dataframe):
        """Test preparing both feature sets."""
        trainer = ModelTrainer(cv_strategy="standard")

        X, y, groups = trainer.prepare_features(
            toy_dataframe,
            outcome_col='isOncoKbLevel1',
            feature_set='both'
        )

        # Both demographic and technical should be included
        assert 'SEX' in X.columns
        assert 'SUBJECT_CENTER' in X.columns

    def test_missing_outcome_column_raises_error(self, toy_dataframe):
        """Test that missing outcome column raises ValueError."""
        trainer = ModelTrainer(cv_strategy="standard")

        with pytest.raises(ValueError, match="Outcome column .* not found"):
            trainer.prepare_features(
                toy_dataframe,
                outcome_col='nonexistent_column',
                feature_set='demographic'
            )

    def test_invalid_feature_set_raises_error(self, toy_dataframe):
        """Test that invalid feature_set raises ValueError."""
        trainer = ModelTrainer(cv_strategy="standard")

        with pytest.raises(ValueError, match="feature_set must be one of"):
            trainer.prepare_features(
                toy_dataframe,
                outcome_col='isOncoKbLevel1',
                feature_set='invalid'
            )

    def test_groups_extracted_for_group_cv(self, toy_dataframe):
        """Test that groups are extracted for group CV strategies."""
        trainer = ModelTrainer(cv_strategy="group", group_column="SUBJECT_CENTER")

        X, y, groups = trainer.prepare_features(
            toy_dataframe,
            outcome_col='isOncoKbLevel1',
            feature_set='demographic'
        )

        assert groups is not None
        assert len(groups) == len(X)


class TestCreatePipeline:
    """Tests for ModelTrainer.create_pipeline() method."""

    def test_create_random_forest_pipeline(self):
        """Test creating Random Forest pipeline."""
        trainer = ModelTrainer(algorithm="random_forest")
        pipeline = trainer.create_pipeline()

        assert pipeline is not None
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == 'preprocessor'
        assert pipeline.steps[1][0] == 'classifier'

    def test_create_logistic_pipeline(self):
        """Test creating Logistic Regression pipeline."""
        trainer = ModelTrainer(algorithm="logistic")
        pipeline = trainer.create_pipeline()

        assert pipeline is not None
        assert len(pipeline.steps) == 2

    def test_pipeline_with_available_columns(self):
        """Test creating pipeline with specific columns."""
        trainer = ModelTrainer()
        columns = ['SEX', 'AGE_AT_SEQ_REPORT', 'ANCESTRY']
        pipeline = trainer.create_pipeline(available_columns=columns)

        assert pipeline is not None


class TestTrainAndEvaluate:
    """Tests for ModelTrainer.train_and_evaluate() method."""

    def test_train_and_evaluate_basic(self, toy_dataframe):
        """Test basic training and evaluation."""
        trainer = ModelTrainer(
            algorithm="random_forest",
            cv_strategy="standard",
            n_jobs=1
        )

        X, y, groups = trainer.prepare_features(
            toy_dataframe,
            outcome_col='isOncoKbLevel1',
            feature_set='demographic'
        )

        results = trainer.train_and_evaluate(X, y, groups)

        # Check required keys
        assert 'algorithm' in results
        assert 'cv_auc_mean' in results
        assert 'cv_auc_std' in results
        assert 'test_auc' in results
        assert 'test_accuracy' in results
        assert 'model' in results

        # Check reasonable values
        assert 0.0 <= results['cv_auc_mean'] <= 1.0
        assert 0.0 <= results['test_auc'] <= 1.0
        assert 0.0 <= results['test_accuracy'] <= 1.0

    def test_train_and_evaluate_logistic(self, toy_dataframe):
        """Test training with Logistic Regression."""
        trainer = ModelTrainer(
            algorithm="logistic",
            cv_strategy="standard",
            n_jobs=1
        )

        X, y, groups = trainer.prepare_features(
            toy_dataframe,
            outcome_col='isOncoKbLevel1',
            feature_set='demographic'
        )

        results = trainer.train_and_evaluate(X, y, groups)

        assert results['algorithm'] == 'logistic'
        assert 0.0 <= results['cv_auc_mean'] <= 1.0

    def test_insufficient_data_raises_error(self):
        """Test that insufficient data raises ValueError."""
        trainer = ModelTrainer(cv_strategy="standard")

        # Create very small dataset
        X = pd.DataFrame({'SEX': ['Male', 'Female'], 'AGE': [30, 40]})
        y = pd.Series([0, 1])

        with pytest.raises(ValueError, match="Insufficient data"):
            trainer.train_and_evaluate(X, y)

    def test_missing_groups_for_group_cv_raises_error(self, toy_dataframe):
        """Test that missing groups raises error for group CV."""
        trainer = ModelTrainer(cv_strategy="group")

        X, y, _ = trainer.prepare_features(
            toy_dataframe,
            outcome_col='isOncoKbLevel1',
            feature_set='demographic'
        )

        # Pass groups=None when group CV is expected
        with pytest.raises(ValueError, match="groups parameter is required"):
            trainer.train_and_evaluate(X, y, groups=None)

    def test_results_contain_trained_model(self, toy_dataframe):
        """Test that results contain a fitted model."""
        trainer = ModelTrainer(cv_strategy="standard", n_jobs=1)

        X, y, groups = trainer.prepare_features(
            toy_dataframe,
            outcome_col='isOncoKbLevel1',
            feature_set='demographic'
        )

        results = trainer.train_and_evaluate(X, y, groups)

        # Check model can make predictions
        model = results['model']
        predictions = model.predict(X)
        assert len(predictions) == len(X)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
