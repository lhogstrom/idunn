"""
Hyperparameter Search Spaces for ML Algorithms

Defines search spaces for hyperparameter optimization with RandomizedSearchCV
and GridSearchCV. Based on best practices from scikit-learn documentation and
cancer genomics ML literature.

Author: Claude Code
Date: December 2025
"""

# ============================================================================
# Random Forest Search Space
# ============================================================================

RF_PARAM_DISTRIBUTIONS = {
    # Number of trees in the forest
    # More trees = better performance but diminishing returns after ~200
    # Trade-off: Computation time vs stability
    'n_estimators': [50, 100, 200, 500],

    # Maximum depth of each tree
    # None = grow until pure (may overfit)
    # 10-30 = reasonable range for medical data
    # Shallower = more regularization
    'max_depth': [10, 20, 30, None],

    # Minimum samples required to split an internal node
    # Higher values = more regularization (prevent overfitting)
    # Especially important for small datasets
    'min_samples_split': [2, 5, 10, 20],

    # Minimum samples required to be at a leaf node
    # Higher values = smoother decision boundaries
    # Prevents creating leaves with very few samples
    'min_samples_leaf': [1, 2, 4, 8],

    # Number of features to consider when looking for best split
    # 'sqrt' = sqrt(n_features) - standard for classification
    # 'log2' = log2(n_features) - more aggressive feature subsampling
    # 0.3, 0.5 = fixed fractions (useful when n_features varies)
    'max_features': ['sqrt', 'log2', 0.3, 0.5],

    # Weighting strategy for imbalanced classes
    # 'balanced' = weights inversely proportional to class frequencies
    # 'balanced_subsample' = same as balanced but computed per tree
    #   (recommended for Random Forest to increase diversity)
    'class_weight': ['balanced', 'balanced_subsample']
}


# ============================================================================
# Logistic Regression Search Space
# ============================================================================

LR_PARAM_DISTRIBUTIONS = {
    # Inverse of regularization strength
    # Small C = strong regularization (simpler model)
    # Large C = weak regularization (more complex model)
    # Range covers 6 orders of magnitude
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],

    # Type of regularization penalty
    # Only using 'l2' and None to ensure compatibility with all solvers
    # 'l2' = Ridge (shrinkage, all features)
    # None = no penalty
    'penalty': ['l2', None],

    # Optimization algorithm
    # 'lbfgs' = good for large datasets, supports l2 and None
    # 'saga' = supports all penalties, good for large datasets
    # Note: Using only lbfgs and saga to avoid liblinear (deprecated for large datasets)
    'solver': ['lbfgs', 'saga'],

    # Class weighting for imbalanced datasets
    # 'balanced' = weights inversely proportional to class frequencies
    # None = all classes have weight 1
    'class_weight': ['balanced', None],

    # Maximum number of iterations for convergence
    # Increase if getting convergence warnings
    'max_iter': [100, 200, 500, 1000]
}


# ============================================================================
# XGBoost Search Space
# ============================================================================

XGBOOST_PARAM_DISTRIBUTIONS = {
    # Number of boosting rounds (trees)
    # More trees = better performance but risk overfitting
    # Range similar to RF n_estimators
    'n_estimators': [50, 100, 200, 500],

    # Maximum depth of trees
    # Controls model complexity
    # Smaller values = more regularization
    'max_depth': [3, 5, 7, 10],

    # Learning rate (eta)
    # Lower values need more trees but generalize better
    # Range covers typical values for classification
    'learning_rate': [0.01, 0.05, 0.1, 0.3],

    # Minimum sum of instance weight needed in a child
    # Similar to min_samples_leaf in RF
    # Higher values = more regularization
    'min_child_weight': [1, 3, 5, 10],

    # Subsample ratio of training instances
    # Controls stochasticity and overfitting
    # 0.8-1.0 typical for classification
    'subsample': [0.6, 0.8, 1.0],

    # Subsample ratio of features when constructing each tree
    # Similar to max_features in RF
    # Lower values = more regularization
    'colsample_bytree': [0.6, 0.8, 1.0],

    # L2 regularization term on weights
    # Helps prevent overfitting
    'reg_alpha': [0, 0.1, 1.0, 10.0],

    # L1 regularization term on weights
    # Can produce sparser models
    'reg_lambda': [0, 0.1, 1.0, 10.0],

    # Balancing of positive and negative weights
    # Automatically computed from class distribution
    # 'balanced' computed as: sum(negative) / sum(positive)
    'scale_pos_weight': [1.0, 3.0, 5.0]  # Will compute balanced ratio if needed
}


# ============================================================================
# Solver-Penalty Compatibility Rules
# ============================================================================

# Not all solver-penalty combinations are valid in scikit-learn
# Use this dictionary to filter invalid combinations during search

SOLVER_PENALTY_COMPATIBILITY = {
    'liblinear': ['l1', 'l2'],
    'lbfgs': ['l2'],  # Only supports l2
    'saga': ['l1', 'l2', 'elasticnet']  # Supports all penalties
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_param_distributions(algorithm: str) -> dict:
    """
    Get hyperparameter distributions for a specific algorithm.

    Parameters
    ----------
    algorithm : str
        Algorithm name: 'random_forest', 'logistic', or 'xgboost'

    Returns
    -------
    param_distributions : dict
        Dictionary of hyperparameter distributions suitable for
        RandomizedSearchCV or GridSearchCV

    Examples
    --------
    >>> from oncokb_analyzer.param_spaces import get_param_distributions
    >>> rf_params = get_param_distributions('random_forest')
    >>> print(rf_params.keys())
    dict_keys(['n_estimators', 'max_depth', 'min_samples_split', ...])
    """
    if algorithm == 'random_forest':
        return RF_PARAM_DISTRIBUTIONS
    elif algorithm == 'logistic':
        return LR_PARAM_DISTRIBUTIONS
    elif algorithm == 'xgboost':
        return XGBOOST_PARAM_DISTRIBUTIONS
    else:
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. "
            f"Supported: 'random_forest', 'logistic', 'xgboost'"
        )


def validate_logistic_params(params: dict) -> bool:
    """
    Validate that Logistic Regression hyperparameters are compatible.

    Checks that solver-penalty combination is valid according to
    SOLVER_PENALTY_COMPATIBILITY rules.

    Parameters
    ----------
    params : dict
        Dictionary of hyperparameters with 'solver' and 'penalty' keys

    Returns
    -------
    is_valid : bool
        True if parameters are compatible, False otherwise

    Examples
    --------
    >>> validate_logistic_params({'solver': 'lbfgs', 'penalty': 'l2'})
    True
    >>> validate_logistic_params({'solver': 'lbfgs', 'penalty': 'l1'})
    False
    """
    if 'solver' not in params or 'penalty' not in params:
        # If either is missing, can't validate
        return True

    solver = params['solver']
    penalty = params['penalty']

    if solver in SOLVER_PENALTY_COMPATIBILITY:
        return penalty in SOLVER_PENALTY_COMPATIBILITY[solver]
    else:
        # Unknown solver, assume valid
        return True


def filter_invalid_logistic_params(param_list: list) -> list:
    """
    Filter out invalid solver-penalty combinations from parameter list.

    Useful when manually generating parameter combinations for GridSearchCV.

    Parameters
    ----------
    param_list : list of dict
        List of parameter dictionaries

    Returns
    -------
    filtered_list : list of dict
        List with invalid combinations removed

    Examples
    --------
    >>> params = [
    ...     {'solver': 'lbfgs', 'penalty': 'l2', 'C': 1.0},
    ...     {'solver': 'lbfgs', 'penalty': 'l1', 'C': 1.0},  # Invalid
    ... ]
    >>> filtered = filter_invalid_logistic_params(params)
    >>> len(filtered)
    1
    """
    return [p for p in param_list if validate_logistic_params(p)]


# ============================================================================
# Reduced Search Spaces (for quick tests)
# ============================================================================

# Smaller search spaces for rapid testing and debugging

RF_PARAM_DISTRIBUTIONS_QUICK = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4],
    'max_features': ['sqrt', 0.5],
    'class_weight': ['balanced']
}

LR_PARAM_DISTRIBUTIONS_QUICK = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l2'],  # Only l2 to avoid solver compatibility issues
    'solver': ['lbfgs'],
    'class_weight': ['balanced', None],
    'max_iter': [200, 500]
}

XGBOOST_PARAM_DISTRIBUTIONS_QUICK = {
    'n_estimators': [50, 100],
    'max_depth': [3, 7],
    'learning_rate': [0.05, 0.1],
    'min_child_weight': [1, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 1.0],
    'reg_lambda': [0, 1.0],
    'scale_pos_weight': [1.0, 3.0]
}


def get_param_distributions_quick(algorithm: str) -> dict:
    """
    Get reduced hyperparameter search spaces for quick testing.

    Useful for:
    - Debugging nested CV implementation
    - Quick pilot runs to estimate runtime
    - Integration tests

    Parameters
    ----------
    algorithm : str
        Algorithm name: 'random_forest', 'logistic', or 'xgboost'

    Returns
    -------
    param_distributions : dict
        Reduced search space (fewer combinations)
    """
    if algorithm == 'random_forest':
        return RF_PARAM_DISTRIBUTIONS_QUICK
    elif algorithm == 'logistic':
        return LR_PARAM_DISTRIBUTIONS_QUICK
    elif algorithm == 'xgboost':
        return XGBOOST_PARAM_DISTRIBUTIONS_QUICK
    else:
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. "
            f"Supported: 'random_forest', 'logistic', 'xgboost'"
        )


# ============================================================================
# Search Space Statistics
# ============================================================================

def count_grid_combinations(param_distributions: dict) -> int:
    """
    Count total number of hyperparameter combinations in grid.

    This is the number of models that would be evaluated by GridSearchCV
    (exhaustive search). For RandomizedSearchCV, you can sample a subset.

    Parameters
    ----------
    param_distributions : dict
        Hyperparameter search space

    Returns
    -------
    n_combinations : int
        Total number of unique parameter combinations

    Examples
    --------
    >>> from oncokb_analyzer.param_spaces import RF_PARAM_DISTRIBUTIONS
    >>> n_combos = count_grid_combinations(RF_PARAM_DISTRIBUTIONS)
    >>> print(f"Random Forest grid has {n_combos} combinations")
    Random Forest grid has 2048 combinations
    """
    n_combinations = 1
    for param_values in param_distributions.values():
        n_combinations *= len(param_values)
    return n_combinations


if __name__ == '__main__':
    # Print search space statistics
    print("="*80)
    print("HYPERPARAMETER SEARCH SPACE STATISTICS")
    print("="*80)

    print("\nRandom Forest:")
    print(f"  Parameters: {list(RF_PARAM_DISTRIBUTIONS.keys())}")
    print(f"  Grid combinations: {count_grid_combinations(RF_PARAM_DISTRIBUTIONS):,}")
    print(f"  Quick grid combinations: {count_grid_combinations(RF_PARAM_DISTRIBUTIONS_QUICK):,}")

    print("\nLogistic Regression:")
    print(f"  Parameters: {list(LR_PARAM_DISTRIBUTIONS.keys())}")
    print(f"  Grid combinations (before filtering): {count_grid_combinations(LR_PARAM_DISTRIBUTIONS):,}")
    print(f"  Quick grid combinations: {count_grid_combinations(LR_PARAM_DISTRIBUTIONS_QUICK):,}")
    print(f"  Note: Some solver-penalty combinations invalid, actual grid smaller")

    print("\nXGBoost:")
    print(f"  Parameters: {list(XGBOOST_PARAM_DISTRIBUTIONS.keys())}")
    print(f"  Grid combinations: {count_grid_combinations(XGBOOST_PARAM_DISTRIBUTIONS):,}")
    print(f"  Quick grid combinations: {count_grid_combinations(XGBOOST_PARAM_DISTRIBUTIONS_QUICK):,}")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("- Use RandomizedSearchCV with n_iter=100 for Phase 2b")
    print("- Full grid search would require 100,000+ model fits (XGBoost)")
    print("- Quick search spaces reduce combinations by ~10-100x for testing")
