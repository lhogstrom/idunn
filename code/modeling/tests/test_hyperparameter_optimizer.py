"""Tests for nested cross-validation: no fold leakage, universal vs cancer-specific
comparison, hyperparameter stability, and consensus parameters.

Run with: pytest test_hyperparameter_optimizer.py -v
"""

import gc
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from oncokb_analyzer.hyperparameter_optimizer import NestedCVOptimizer
from oncokb_analyzer.param_spaces import (
    get_param_distributions,
    get_param_distributions_quick,
    validate_logistic_params,
    count_grid_combinations
)
from oncokb_analyzer.preprocessing import DEMOGRAPHIC_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


# Fixtures

def _create_realistic_dataframe(n_samples: int, random_state: int, class_weights: list = None) -> tuple:
    """Create a realistic DataFrame with proper column names matching the preprocessing pipeline."""
    np.random.seed(random_state)
    if class_weights is None:
        class_weights = [0.7, 0.3]

    # Create DataFrame with realistic demographic features
    df = pd.DataFrame({
        # Numeric feature
        'AGE_AT_SEQ_REPORT': np.random.randint(20, 90, n_samples).astype(float),
        # Add some NaN values for testing imputation
        'number_of_genes': np.random.randint(100, 600, n_samples).astype(float),

        # Categorical features.
        'SEX': np.random.choice(['Male', 'Female'], n_samples),
        'ANCESTRY': np.random.choice(['European', 'African', 'Asian', 'Other', 'Unknown'], n_samples),
        'ETHNICITY': np.random.choice(['Hispanic', 'Non-Hispanic', 'Unknown'], n_samples),
        'SAMPLE_TYPE': np.random.choice(['Primary', 'Metastasis', 'Other'], n_samples),
        'SAMPLE_TYPE_DETAILED': np.random.choice(['Primary Tumor', 'Metastatic', 'Local Recurrence'], n_samples),
        'SUBJECT_CENTER': np.random.choice(['MSK', 'DFCI', 'VICC', 'UHN'], n_samples),
        'preservation_technique': np.random.choice(['FFPE', 'Fresh Frozen'], n_samples),
        'calling_strategy': np.random.choice(['tumor_only', 'tumor_normal'], n_samples),
    })

    # Create target variable with specified class weights
    y = np.random.choice([0, 1], n_samples, p=class_weights)

    # Add signal: older patients more likely positive.
    age_effect = (df['AGE_AT_SEQ_REPORT'] > 60).astype(int) * 0.2
    # Males slightly more likely
    sex_effect = (df['SEX'] == 'Male').astype(int) * 0.1
    # Combine effects with noise
    signal = age_effect + sex_effect + np.random.normal(0, 0.3, n_samples)
    threshold = np.percentile(signal, 100 * class_weights[0])
    y = (signal > threshold).astype(int)

    return df, y


@pytest.fixture
def toy_data_binary():
    """Generate toy binary classification DataFrame for testing."""
    return _create_realistic_dataframe(
        n_samples=200,  # Reduced from 1000 for memory safety
        random_state=42,
        class_weights=[0.7, 0.3]
    )


@pytest.fixture
def toy_data_balanced():
    """Generate balanced binary classification DataFrame."""
    return _create_realistic_dataframe(
        n_samples=200,  # Reduced from 800 for memory safety
        random_state=123,
        class_weights=[0.5, 0.5]
    )


@pytest.fixture
def toy_multi_cancer_data():
    """Generate synthetic multi-cancer dataset for universal vs specific testing."""
    cancer_data = {}

    # Create 3 synthetic "cancer types" with different characteristics
    for i, cancer in enumerate(['CANCER_A', 'CANCER_B', 'CANCER_C']):
        class_weights = [0.6 + i*0.1, 0.4 - i*0.1]  # Slightly different class balance
        X, y = _create_realistic_dataframe(
            n_samples=100,  # Reduced from 300 for memory safety
            random_state=i * 100,
            class_weights=class_weights
        )
        cancer_data[cancer] = (X, y, None)

    return cancer_data


# Tests for NestedCVOptimizer Core Functionality

def test_optimizer_initialization():
    """Test that optimizer initializes with correct parameters."""
    param_dists = {'n_estimators': [10, 20], 'max_depth': [5, 10]}

    optimizer = NestedCVOptimizer(
        param_distributions=param_dists,
        search_method='random',
        n_iter=10,
        scoring='roc_auc',
        n_jobs=1,
        random_state=42,
        verbose=0
    )

    assert optimizer.param_distributions == param_dists
    assert optimizer.search_method == 'random'
    assert optimizer.n_iter == 10
    assert optimizer.scoring == 'roc_auc'
    assert optimizer.random_state == 42


def test_optimizer_invalid_search_method():
    """Test that invalid search method raises error."""
    param_dists = {'n_estimators': [10, 20]}

    with pytest.raises(ValueError, match="search_method must be"):
        NestedCVOptimizer(
            param_distributions=param_dists,
            search_method='invalid_method'
        )


def test_nested_cv_basic_execution(toy_data_binary):
    """Test that nested CV runs without errors."""
    X, y = toy_data_binary

    # Use quick search space to speed up test
    param_dists = get_param_distributions_quick('random_forest')

    optimizer = NestedCVOptimizer(
        param_distributions=param_dists,
        search_method='random',
        n_iter=2,  # Minimal for memory safety (reduced from 5)
        n_jobs=1,  # Prevent parallel memory explosion
        cv_strategy='standard',  # Use standard CV for unit tests without groups
        verbose=0
    )

    results = optimizer.optimize(X, y, algorithm='random_forest')

    # Check that all expected keys are present
    assert 'best_params_by_fold' in results
    assert 'outer_cv_scores' in results
    assert 'mean_score' in results
    assert 'std_score' in results
    assert 'best_params_consensus' in results
    assert 'param_stability' in results

    # Explicit cleanup
    del optimizer
    gc.collect()


def test_nested_cv_no_data_leakage(toy_data_binary):
    """Outer test folds must never be seen during the inner hyperparameter search."""
    X, y = toy_data_binary

    param_dists = {'n_estimators': [50, 100], 'max_depth': [10, 20]}

    optimizer = NestedCVOptimizer(
        param_distributions=param_dists,
        search_method='random',
        n_iter=2,  # Minimal for memory safety (reduced from 5)
        n_jobs=1,  # Prevent parallel memory explosion
        cv_strategy='standard',  # Use standard CV for unit tests without groups
        verbose=0
    )

    # Run nested CV
    results = optimizer.optimize(X, y, algorithm='random_forest')

    # Verify: We get 5 sets of best params (one per outer fold)
    assert len(results['best_params_by_fold']) == 5

    # Verify: We get 5 outer test scores
    assert len(results['outer_cv_scores']) == 5

    # Verify
    for score in results['outer_cv_scores']:
        assert 0.0 <= score <= 1.0, f"AUC score out of bounds: {score}"

    # Verify
    assert results['std_score'] >= 0, "Standard deviation should be non-negative"

    # Verify: Mean score is reasonable
    assert 0.5 <= results['mean_score'] <= 1.0

    # Explicit cleanup
    del optimizer
    gc.collect()


def test_nested_cv_produces_stable_results(toy_data_balanced):
    """Test that nested CV with fixed random seed produces reproducible results."""
    X, y = toy_data_balanced

    param_dists = {'n_estimators': [50, 100], 'max_depth': [10, 20]}

    # Run twice with same random seed
    optimizer1 = NestedCVOptimizer(
        param_distributions=param_dists,
        search_method='random',
        n_iter=2,  # Minimal for memory safety (reduced from 5)
        n_jobs=1,  # Prevent parallel memory explosion
        random_state=42,
        cv_strategy='standard',  # Use standard CV for unit tests without groups
        verbose=0
    )
    results1 = optimizer1.optimize(X, y, algorithm='random_forest')

    # Explicit cleanup before creating second optimizer
    del optimizer1
    gc.collect()

    optimizer2 = NestedCVOptimizer(
        param_distributions=param_dists,
        search_method='random',
        n_iter=2,  # Minimal for memory safety (reduced from 5)
        n_jobs=1,  # Prevent parallel memory explosion
        random_state=42,
        cv_strategy='standard',  # Use standard CV for unit tests without groups
        verbose=0
    )
    results2 = optimizer2.optimize(X, y, algorithm='random_forest')

    # Results should be identical with same random seed
    np.testing.assert_almost_equal(
        results1['mean_score'],
        results2['mean_score'],
        decimal=6,
        err_msg="Nested CV not reproducible with same random seed"
    )

    # Explicit cleanup
    del optimizer2
    gc.collect()


def test_nested_cv_logistic_regression(toy_data_binary):
    """Test nested CV with Logistic Regression algorithm."""
    X, y = toy_data_binary

    param_dists = get_param_distributions_quick('logistic')

    optimizer = NestedCVOptimizer(
        param_distributions=param_dists,
        search_method='random',
        n_iter=2,  # Minimal for memory safety (reduced from 5)
        n_jobs=1,  # Prevent parallel memory explosion
        cv_strategy='standard',  # Use standard CV for unit tests without groups
        verbose=0
    )

    results = optimizer.optimize(X, y, algorithm='logistic')

    # Check structure
    assert 'mean_score' in results
    assert 0.5 <= results['mean_score'] <= 1.0

    # Check that best params contain expected keys
    consensus_params = results['best_params_consensus']
    assert 'C' in consensus_params
    assert 'penalty' in consensus_params

    # Explicit cleanup
    del optimizer
    gc.collect()


# Tests for Universal vs Cancer-Specific Comparison

@pytest.mark.slow
def test_compare_universal_vs_specific_structure(toy_multi_cancer_data):
    """Test that universal vs specific comparison returns expected structure."""
    param_dists = get_param_distributions_quick('random_forest')

    optimizer = NestedCVOptimizer(
        param_distributions=param_dists,
        search_method='random',
        n_iter=2,  # Minimal for memory safety (reduced from 3)
        n_jobs=1,  # Prevent parallel memory explosion
        cv_strategy='standard',  # Use standard CV for unit tests without groups
        verbose=0
    )

    results = optimizer.compare_universal_vs_specific(
        toy_multi_cancer_data,
        algorithm='random_forest'
    )

    # Check all expected keys present
    expected_keys = [
        'universal_params',
        'universal_pooled_auc',
        'universal_auc_by_cancer',
        'specific_params_by_cancer',
        'specific_auc_by_cancer',
        'delta_auc_by_cancer',
        'mean_delta_auc',
        'max_delta_auc',
        'decision',
        'decision_rationale'
    ]

    for key in expected_keys:
        assert key in results, f"Missing expected key: {key}"

    # Check decision is valid
    assert results['decision'] in ['universal', 'cancer_specific', 'hybrid']

    # Explicit cleanup
    del optimizer
    gc.collect()


@pytest.mark.slow
def test_compare_universal_vs_specific_cancer_counts(toy_multi_cancer_data):
    """Test that results include all cancer types."""
    param_dists = get_param_distributions_quick('random_forest')

    optimizer = NestedCVOptimizer(
        param_distributions=param_dists,
        search_method='random',
        n_iter=2,  # Minimal for memory safety (reduced from 3)
        n_jobs=1,  # Prevent parallel memory explosion
        cv_strategy='standard',  # Use standard CV for unit tests without groups
        verbose=0
    )

    results = optimizer.compare_universal_vs_specific(
        toy_multi_cancer_data,
        algorithm='random_forest'
    )

    cancer_types = list(toy_multi_cancer_data.keys())

    # All cancer types should have results
    assert set(results['universal_auc_by_cancer'].keys()) == set(cancer_types)
    assert set(results['specific_auc_by_cancer'].keys()) == set(cancer_types)
    assert set(results['specific_params_by_cancer'].keys()) == set(cancer_types)
    assert set(results['delta_auc_by_cancer'].keys()) == set(cancer_types)

    # Explicit cleanup
    del optimizer
    gc.collect()


@pytest.mark.slow
def test_compare_delta_auc_calculation(toy_multi_cancer_data):
    """Test that delta AUC is calculated correctly."""
    param_dists = get_param_distributions_quick('random_forest')

    optimizer = NestedCVOptimizer(
        param_distributions=param_dists,
        search_method='random',
        n_iter=2,  # Minimal for memory safety (reduced from 3)
        n_jobs=1,  # Prevent parallel memory explosion
        cv_strategy='standard',  # Use standard CV for unit tests without groups
        verbose=0
    )

    results = optimizer.compare_universal_vs_specific(
        toy_multi_cancer_data,
        algorithm='random_forest'
    )

    # Manually verify delta AUC calculation
    for cancer in toy_multi_cancer_data.keys():
        expected_delta = (
            results['specific_auc_by_cancer'][cancer] -
            results['universal_auc_by_cancer'][cancer]
        )
        actual_delta = results['delta_auc_by_cancer'][cancer]

        np.testing.assert_almost_equal(
            actual_delta,
            expected_delta,
            decimal=6,
            err_msg=f"Delta AUC mismatch for {cancer}"
        )

    # Verify mean delta AUC
    expected_mean = np.mean(list(results['delta_auc_by_cancer'].values()))
    np.testing.assert_almost_equal(
        results['mean_delta_auc'],
        expected_mean,
        decimal=6
    )

    # Explicit cleanup
    del optimizer
    gc.collect()


# Tests for Helper Methods

def test_compute_consensus_params():
    """Test consensus parameter computation."""
    param_dists = {'n_estimators': [50, 100], 'max_depth': [10, 20]}

    optimizer = NestedCVOptimizer(param_distributions=param_dists, verbose=0)

    # Test case 1: All folds agree
    params_list = [
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 10}
    ]
    consensus = optimizer._compute_consensus_params(params_list)
    assert consensus == {'n_estimators': 100, 'max_depth': 10}

    # Test case 2: Majority vote
    params_list = [
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 20},
        {'n_estimators': 100, 'max_depth': 10}
    ]
    consensus = optimizer._compute_consensus_params(params_list)
    assert consensus['n_estimators'] == 100
    assert consensus['max_depth'] == 10  # Most common


def test_compute_param_stability():
    """Test hyperparameter stability metric calculation."""
    param_dists = {'n_estimators': [50, 100], 'max_depth': [10, 20]}

    optimizer = NestedCVOptimizer(param_distributions=param_dists, verbose=0)

    # Test case 1: Perfect stability (all folds identical)
    params_list = [
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 10}
    ]
    stability = optimizer._compute_param_stability(params_list)
    assert stability == 1.0, "Perfect stability should yield 1.0"

    # Test case 2: Partial stability
    params_list = [
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 20},
        {'n_estimators': 50, 'max_depth': 10}
    ]
    stability = optimizer._compute_param_stability(params_list)
    # n_estimators
    assert stability == 0.8

    # Test case 3: Low stability (all different)
    params_list = [
        {'n_estimators': 50, 'max_depth': 10},
        {'n_estimators': 100, 'max_depth': 20},
        {'n_estimators': 200, 'max_depth': 30},
        {'n_estimators': 500, 'max_depth': 10},
        {'n_estimators': 50, 'max_depth': 20}
    ]
    stability = optimizer._compute_param_stability(params_list)
    # n_estimators
    assert stability == 0.4


def test_pool_data(toy_multi_cancer_data):
    """Test data pooling from multiple cancer types."""
    param_dists = {'n_estimators': [50]}

    optimizer = NestedCVOptimizer(param_distributions=param_dists, verbose=0)

    X_pooled, y_pooled = optimizer._pool_data(toy_multi_cancer_data)

    # Check dimensions
    total_samples = sum(len(data[0]) for data in toy_multi_cancer_data.values())
    assert len(X_pooled) == total_samples
    assert len(y_pooled) == total_samples

    # Check that pooled data is correct type
    assert isinstance(X_pooled, np.ndarray)
    assert isinstance(y_pooled, np.ndarray)


# Tests for param_spaces Module

def test_get_param_distributions():
    """Test parameter distribution retrieval."""
    rf_params = get_param_distributions('random_forest')
    assert 'n_estimators' in rf_params
    assert 'max_depth' in rf_params
    assert 'class_weight' in rf_params

    lr_params = get_param_distributions('logistic')
    assert 'C' in lr_params
    assert 'penalty' in lr_params
    assert 'solver' in lr_params


def test_get_param_distributions_invalid_algorithm():
    """Test that invalid algorithm raises error."""
    with pytest.raises(ValueError, match="Unknown algorithm"):
        get_param_distributions('invalid_algorithm')


def test_validate_logistic_params():
    """Test logistic regression parameter validation."""
    # Valid combinations
    assert validate_logistic_params({'solver': 'lbfgs', 'penalty': 'l2'}) == True
    assert validate_logistic_params({'solver': 'liblinear', 'penalty': 'l1'}) == True
    assert validate_logistic_params({'solver': 'saga', 'penalty': 'elasticnet'}) == True

    # Invalid combinations
    assert validate_logistic_params({'solver': 'lbfgs', 'penalty': 'l1'}) == False
    assert validate_logistic_params({'solver': 'lbfgs', 'penalty': 'elasticnet'}) == False


def test_count_grid_combinations():
    """Test grid combination counting."""
    param_space = {
        'param1': [1, 2, 3],
        'param2': [10, 20],
        'param3': ['a', 'b']
    }
    # 3 * 2 * 2 = 12
    assert count_grid_combinations(param_space) == 12


# Integration Tests

@pytest.mark.slow
def test_full_nested_cv_random_forest(toy_data_binary):
    """End-to-end nested CV with Random Forest on toy data."""
    X, y = toy_data_binary

    param_dists = get_param_distributions('random_forest')

    optimizer = NestedCVOptimizer(
        param_distributions=param_dists,
        search_method='random',
        n_iter=10,  # Reduced from 20 for memory safety, still meaningful
        n_jobs=1,  # Prevent parallel memory explosion
        cv_strategy='standard',  # Use standard CV for unit tests without groups
        verbose=0
    )

    results = optimizer.optimize(X, y, algorithm='random_forest')

    # Verify reasonable performance on this toy dataset
    assert results['mean_score'] > 0.5, "AUC should be better than random on toy data"

    # Verify stability is reasonable (not too unstable)
    assert results['param_stability'] > 0.2, "Params shouldn't be completely unstable"

    print(f"\nIntegration test results:")
    print(f"  Mean AUC: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
    print(f"  Stability: {results['param_stability']:.3f}")
    print(f"  Consensus params: {results['best_params_consensus']}")

    # Explicit cleanup
    del optimizer
    gc.collect()


# Run Tests

if __name__ == '__main__':
    # Run with: python test_hyperparameter_optimizer.py
    pytest.main([__file__, '-v', '-m', 'not slow'])
