"""Hyperparameter search spaces for RandomizedSearchCV / GridSearchCV."""

# Random Forest Search Space

RF_PARAM_DISTRIBUTIONS = {
    # Number of trees
    'n_estimators': [100, 200, 500],

    # Maximum tree depth (None = grow until pure)
    'max_depth': [3, 5, 7, 10, 15],

    # Minimum samples to split a node
    'min_samples_split': [10, 20, 50, 100],

    # Minimum samples per leaf
    'min_samples_leaf': [5, 10, 20, 30],

    # Features considered per split
    'max_features': ['sqrt', 'log2', 0.3],

    # Class weighting for imbalance
    'class_weight': ['balanced', 'balanced_subsample']
}


# Logistic Regression Search Space

LR_PARAM_DISTRIBUTIONS = {
    # Inverse regularization strength
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],

    # Penalty type (l2 or None, for solver compatibility)
    'penalty': ['l2', None],

    # Solver
    'solver': ['lbfgs', 'saga'],

    # Class weighting for imbalance
    'class_weight': ['balanced', None],

    # Maximum solver iterations
    'max_iter': [100, 200, 500, 1000]
}


# XGBoost Search Space

XGBOOST_PARAM_DISTRIBUTIONS = {
    # Number of boosting rounds
    'n_estimators': [50, 100, 200, 500],

    # Maximum tree depth
    'max_depth': [3, 5, 7, 10],

    # Learning rate
    'learning_rate': [0.01, 0.05, 0.1, 0.3],

    # Minimum instance weight in a child
    'min_child_weight': [1, 3, 5, 10],

    # Row subsample ratio
    'subsample': [0.6, 0.8, 1.0],

    # Column subsample ratio per tree
    'colsample_bytree': [0.6, 0.8, 1.0],

    # L1 regularization
    'reg_alpha': [0, 0.1, 1.0, 10.0],

    # L2 regularization
    'reg_lambda': [0, 0.1, 1.0, 10.0],

    # Positive/negative class weight balance
    'scale_pos_weight': [1.0, 3.0, 5.0]
}


# Solver-Penalty Compatibility Rules

# Valid solver-penalty combinations in scikit-learn.

SOLVER_PENALTY_COMPATIBILITY = {
    'liblinear': ['l1', 'l2'],
    'lbfgs': ['l2'],  # Only supports l2
    'saga': ['l1', 'l2', 'elasticnet']  # Supports all penalties
}


# Helper Functions

def get_param_distributions(algorithm: str) -> dict:
    """Get hyperparameter distributions for a specific algorithm."""
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
    """Validate that Logistic Regression hyperparameters are compatible."""
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
    """Filter out invalid solver-penalty combinations from parameter list."""
    return [p for p in param_list if validate_logistic_params(p)]


# Reduced Search Spaces (for quick tests)

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
    """Get reduced hyperparameter search spaces for quick testing."""
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


# Search Space Statistics

def count_grid_combinations(param_distributions: dict) -> int:
    """Count total number of hyperparameter combinations in grid."""
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
