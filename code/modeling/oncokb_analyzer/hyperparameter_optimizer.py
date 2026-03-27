"""
Hyperparameter Optimizer with Nested Cross-Validation

This module implements nested cross-validation for unbiased hyperparameter optimization.
Supports both universal (pooled) and cancer-type-specific optimization strategies.

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import warnings

from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold,
    GroupKFold,
    cross_val_score
)

# Import StratifiedGroupKFold
from .stratified_group_kfold import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# Import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: conda install -c conda-forge xgboost")

# Import shared preprocessing
from .preprocessing import create_preprocessor


class NestedCVOptimizer:
    """
    Performs nested cross-validation for hyperparameter optimization.

    Nested CV prevents data leakage by using:
    - Outer CV: Provides unbiased performance estimate
    - Inner CV: Selects best hyperparameters (only on training data)

    Parameters
    ----------
    param_distributions : dict
        Hyperparameter search space for the algorithm.
        Example: {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]}

    search_method : str, default='random'
        Search strategy: 'random' (RandomizedSearchCV) or 'grid' (GridSearchCV)

    n_iter : int, default=50
        Number of iterations for random search (ignored for grid search)

    scoring : str, default='roc_auc'
        Metric for optimization (must be sklearn scoring string)

    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all CPU cores)

    random_state : int, default=42
        Random seed for reproducibility

    verbose : int, default=1
        Verbosity level (0=silent, 1=progress, 2=detailed)

    cv_strategy : str, default='group'
        Cross-validation strategy: 'standard' (StratifiedKFold), 'group' (GroupKFold),
        or 'stratified_group' (StratifiedGroupKFold for balanced institution-aware folds)

    Examples
    --------
    >>> from oncokb_analyzer.hyperparameter_optimizer import NestedCVOptimizer
    >>> from oncokb_analyzer.param_spaces import RF_PARAM_DISTRIBUTIONS
    >>>
    >>> optimizer = NestedCVOptimizer(
    ...     param_distributions=RF_PARAM_DISTRIBUTIONS,
    ...     search_method='random',
    ...     n_iter=50
    ... )
    >>>
    >>> results = optimizer.optimize(X, y, algorithm='random_forest')
    >>> print(f"Unbiased AUC: {results['mean_score']:.4f}")
    >>> print(f"Best params: {results['best_params_consensus']}")
    """

    def __init__(
        self,
        param_distributions: Dict,
        search_method: str = 'random',
        n_iter: int = 50,
        scoring: str = 'roc_auc',
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
        cv_strategy: str = 'group'
    ):
        self.param_distributions = param_distributions
        self.search_method = search_method
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.cv_strategy = cv_strategy

        # Validate inputs
        if search_method not in ['random', 'grid']:
            raise ValueError(f"search_method must be 'random' or 'grid', got '{search_method}'")
        if cv_strategy not in ['standard', 'group', 'stratified_group']:
            raise ValueError(f"cv_strategy must be 'standard', 'group', or 'stratified_group', got '{cv_strategy}'")

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        outer_cv: Optional = None,
        inner_cv: Optional = None,
        algorithm: str = 'random_forest',
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Perform nested cross-validation for hyperparameter optimization.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix

        y : np.ndarray, shape (n_samples,)
            Target labels (binary classification)

        groups : np.ndarray, shape (n_samples,), optional
            Group labels for GroupKFold (e.g., institution IDs)
            If provided, uses GroupKFold instead of StratifiedKFold

        outer_cv : cross-validation generator, optional
            Outer CV strategy for performance estimation
            Default: StratifiedKFold(n_splits=5) or GroupKFold(n_splits=5)

        inner_cv : cross-validation generator, optional
            Inner CV strategy for hyperparameter selection
            Default: StratifiedKFold(n_splits=4) or GroupKFold(n_splits=4)

        algorithm : str, default='random_forest'
            Algorithm to optimize: 'random_forest' or 'logistic'

        sample_weight : np.ndarray, shape (n_samples,), optional
            Sample weights for training (e.g., institution frequency weighting).
            Passed to classifier via fit_params during inner CV and final model fitting.

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'best_params_by_fold': List of best params per outer fold
            - 'outer_cv_scores': List of AUC scores per outer fold
            - 'mean_score': Mean AUC across outer folds (unbiased estimate)
            - 'std_score': Std dev of AUC across outer folds
            - 'best_params_consensus': Most common params across folds
            - 'param_stability': Stability metric (0-1, higher=more stable)

        Notes
        -----
        The test set from each outer fold is NEVER used for hyperparameter selection.
        This ensures the performance estimate is unbiased.
        """
        # Store column names if DataFrame, then convert to numpy for CV compatibility
        column_names = X.columns.tolist() if hasattr(X, 'columns') else None
        X = np.asarray(X)
        y = np.asarray(y)
        if groups is not None:
            groups = np.asarray(groups)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)

        if self.verbose >= 1:
            print(f"\n{'='*80}")
            print(f"NESTED CROSS-VALIDATION: {algorithm.upper()}")
            print(f"{'='*80}")
            print(f"Samples: {len(X)}, Features: {X.shape[1]}")
            print(f"Positive class: {np.sum(y)} ({100*np.mean(y):.1f}%)")
            print(f"Search method: {self.search_method}, n_iter: {self.n_iter}")

        # Default CV strategies
        use_groups = groups is not None

        # Outer CV strategy
        if outer_cv is None:
            if self.cv_strategy == 'stratified_group':
                if not use_groups:
                    raise ValueError("cv_strategy='stratified_group' requires groups to be provided")
                outer_cv = StratifiedGroupKFold(n_splits=5, random_state=self.random_state)
                if self.verbose >= 1:
                    print(f"Outer CV: StratifiedGroupKFold(n_splits=5)")
            elif self.cv_strategy == 'group':
                if not use_groups:
                    raise ValueError("cv_strategy='group' requires groups to be provided")
                outer_cv = GroupKFold(n_splits=5)
                if self.verbose >= 1:
                    print(f"Outer CV: GroupKFold(n_splits=5)")
            else:  # standard
                outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                if self.verbose >= 1:
                    print(f"Outer CV: StratifiedKFold(n_splits=5)")

        # Inner CV strategy
        if inner_cv is None:
            if self.cv_strategy == 'stratified_group':
                if not use_groups:
                    raise ValueError("cv_strategy='stratified_group' requires groups to be provided")
                inner_cv = StratifiedGroupKFold(n_splits=4, random_state=self.random_state + 1)
                if self.verbose >= 1:
                    print(f"Inner CV: StratifiedGroupKFold(n_splits=4)")
            elif self.cv_strategy == 'group':
                if not use_groups:
                    raise ValueError("cv_strategy='group' requires groups to be provided")
                inner_cv = GroupKFold(n_splits=4)
                if self.verbose >= 1:
                    print(f"Inner CV: GroupKFold(n_splits=4)")
            else:  # standard
                inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=self.random_state + 1)
                if self.verbose >= 1:
                    print(f"Inner CV: StratifiedKFold(n_splits=4)")

        # Storage for results
        best_params_by_fold = []
        outer_scores = []

        # Outer CV loop (performance estimation)
        if self.verbose >= 1:
            print(f"\n{'='*80}")
            print("OUTER CV LOOP (Performance Estimation)")
            print(f"{'='*80}")

        split_args = [X, y, groups] if use_groups else [X, y]
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(*split_args)):
            if self.verbose >= 1:
                print(f"\n--- Outer Fold {fold_idx + 1}/{outer_cv.n_splits} ---")

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx] if use_groups else None

            if self.verbose >= 2:
                print(f"Train: {len(X_train)} samples ({np.sum(y_train)} positive, {100*np.mean(y_train):.1f}%)")
                print(f"Test:  {len(X_test)} samples ({np.sum(y_test)} positive, {100*np.mean(y_test):.1f}%)")

            # Inner CV: Hyperparameter search (ONLY on training data)
            estimator = self._get_estimator(algorithm, X=X_train, column_names=column_names)

            # Prefix hyperparameters with 'classifier__' for pipeline
            prefixed_params = {f'classifier__{k}': v for k, v in self.param_distributions.items()}

            if self.search_method == 'random':
                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=prefixed_params,
                    n_iter=self.n_iter,
                    cv=inner_cv,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    verbose=max(0, self.verbose - 1),
                    error_score='raise'
                )
            else:  # grid
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=prefixed_params,
                    cv=inner_cv,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    verbose=max(0, self.verbose - 1),
                    error_score='raise'
                )

            # Prepare fit params for sample weights
            fit_params = {}
            weights_train = None
            if sample_weight is not None:
                weights_train = sample_weight[train_idx]
                fit_params['classifier__sample_weight'] = weights_train

            # Fit inner CV
            if self.verbose >= 1:
                print(f"Running inner CV hyperparameter search...")
                if weights_train is not None:
                    print(f"Using sample weights (range: [{weights_train.min():.3f}, {weights_train.max():.3f}])")

            if use_groups:
                search.fit(X_train, y_train, groups=groups_train, **fit_params)
            else:
                search.fit(X_train, y_train, **fit_params)

            # Store best params from this fold (remove 'classifier__' prefix for storage)
            best_params_unprefixed = {
                k.replace('classifier__', ''): v
                for k, v in search.best_params_.items()
            }
            best_params_by_fold.append(best_params_unprefixed)

            if self.verbose >= 1:
                print(f"Best params: {best_params_unprefixed}")
                print(f"Best inner CV score: {search.best_score_:.4f}")

            # Train final model with best params on FULL training data
            final_model = self._get_estimator(algorithm, X=X_train, column_names=column_names)
            final_model.set_params(**search.best_params_)  # Use prefixed params for pipeline
            if weights_train is not None:
                final_model.fit(X_train, y_train, classifier__sample_weight=weights_train)
            else:
                final_model.fit(X_train, y_train)

            # Evaluate on outer test fold (NEVER seen during hyperparameter search)
            y_pred_proba = final_model.predict_proba(X_test)[:, 1]
            fold_score = roc_auc_score(y_test, y_pred_proba)
            outer_scores.append(fold_score)

            if self.verbose >= 1:
                print(f"Outer test AUC: {fold_score:.4f}")

        # Aggregate results
        if self.verbose >= 1:
            print(f"\n{'='*80}")
            print("NESTED CV RESULTS")
            print(f"{'='*80}")

        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        consensus_params = self._compute_consensus_params(best_params_by_fold)
        param_stability = self._compute_param_stability(best_params_by_fold)

        if self.verbose >= 1:
            print(f"Unbiased AUC: {mean_score:.4f} ± {std_score:.4f}")
            print(f"AUC range: [{min(outer_scores):.4f}, {max(outer_scores):.4f}]")
            print(f"Consensus params: {consensus_params}")
            print(f"Param stability: {param_stability:.3f} (1.0 = all folds agree)")

        results = {
            'best_params_by_fold': best_params_by_fold,
            'outer_cv_scores': outer_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'best_params_consensus': consensus_params,
            'param_stability': param_stability
        }

        return results

    def compare_universal_vs_specific(
        self,
        cancer_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        algorithm: str = 'random_forest'
    ) -> Dict:
        """
        Compare universal vs cancer-type-specific hyperparameter optimization.

        This is the key analysis for Phase 2a: determining whether universal
        hyperparameters are sufficient or if cancer-specific optimization
        provides meaningful performance gains.

        Parameters
        ----------
        cancer_data_dict : dict
            Dictionary mapping cancer type to (X, y, groups) tuples.
            Example: {'LUAD': (X_luad, y_luad, groups_luad), 'BRCA': ...}

        algorithm : str, default='random_forest'
            Algorithm to optimize: 'random_forest' or 'logistic'

        Returns
        -------
        results : dict
            Dictionary containing:
            - 'universal_params': Best universal hyperparameters
            - 'universal_pooled_auc': AUC on pooled dataset
            - 'universal_auc_by_cancer': {cancer: auc, ...} with universal params
            - 'specific_params_by_cancer': {cancer: best_params, ...}
            - 'specific_auc_by_cancer': {cancer: auc, ...} with specific params
            - 'delta_auc_by_cancer': {cancer: delta, ...} (specific - universal)
            - 'mean_delta_auc': Average improvement from cancer-specific
            - 'decision': 'universal' | 'cancer_specific' | 'hybrid'
            - 'decision_rationale': Explanation of decision

        Examples
        --------
        >>> cancer_data = {
        ...     'LUAD': (X_luad, y_luad, None),
        ...     'BRCA': (X_brca, y_brca, None),
        ...     'COAD': (X_coad, y_coad, None),
        ... }
        >>> results = optimizer.compare_universal_vs_specific(cancer_data)
        >>> print(results['decision'])  # 'universal', 'cancer_specific', or 'hybrid'
        >>> print(f"Mean Δ AUC: {results['mean_delta_auc']:.4f}")
        """
        if self.verbose >= 1:
            print("\n" + "="*80)
            print("UNIVERSAL vs CANCER-SPECIFIC OPTIMIZATION COMPARISON")
            print("="*80)
            print(f"Cancer types: {list(cancer_data_dict.keys())}")
            print(f"Total cancers: {len(cancer_data_dict)}")

        # ====================================================================
        # STEP 1: UNIVERSAL OPTIMIZATION (pooled data)
        # ====================================================================
        if self.verbose >= 1:
            print("\n" + "="*80)
            print("STEP 1: UNIVERSAL OPTIMIZATION (pooled data)")
            print("="*80)

        # Get column names from first cancer type (all should have same columns)
        first_cancer_X = list(cancer_data_dict.values())[0][0]
        column_names = first_cancer_X.columns.tolist() if hasattr(first_cancer_X, 'columns') else None

        X_pooled, y_pooled = self._pool_data(cancer_data_dict)

        if self.verbose >= 1:
            print(f"Pooled dataset: {len(X_pooled)} samples, {np.sum(y_pooled)} positive ({100*np.mean(y_pooled):.1f}%)")

        # Pass column_names to optimize (X_pooled is numpy array from _pool_data)
        # We need to wrap it as a DataFrame-like object or pass column_names separately
        # Since optimize() now extracts column names, we need to ensure it gets them
        # The easiest way is to convert X_pooled back to DataFrame temporarily
        import pandas as pd
        if column_names is not None:
            X_pooled_df = pd.DataFrame(X_pooled, columns=column_names)
        else:
            X_pooled_df = X_pooled

        universal_results = self.optimize(X_pooled_df, y_pooled, algorithm=algorithm)
        universal_params = universal_results['best_params_consensus']
        universal_pooled_auc = universal_results['mean_score']

        if self.verbose >= 1:
            print(f"\nUniversal hyperparameters: {universal_params}")
            print(f"Pooled dataset CV AUC: {universal_pooled_auc:.4f} ± {universal_results['std_score']:.4f}")

        # ====================================================================
        # STEP 2: Evaluate universal params on each cancer separately
        # ====================================================================
        if self.verbose >= 1:
            print("\n" + "="*80)
            print("STEP 2: Evaluating universal params on individual cancers")
            print("="*80)

        universal_auc_by_cancer = {}
        for cancer, (X, y, groups) in cancer_data_dict.items():
            if self.verbose >= 1:
                print(f"\n{cancer}:")

            # Get column names and convert to numpy
            column_names = X.columns.tolist() if hasattr(X, 'columns') else None
            X_array = np.asarray(X)
            y_array = np.asarray(y)

            model = self._get_estimator(algorithm, X=X_array, column_names=column_names)
            # Prefix universal params with 'classifier__' for pipeline
            prefixed_universal_params = {f'classifier__{k}': v for k, v in universal_params.items()}
            model.set_params(**prefixed_universal_params)

            # 5-fold CV on this cancer with universal params
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X_array, y_array, cv=cv, scoring='roc_auc', n_jobs=self.n_jobs)
            auc = np.mean(scores)
            universal_auc_by_cancer[cancer] = auc

            if self.verbose >= 1:
                print(f"  Universal params AUC: {auc:.4f} ± {np.std(scores):.4f}")

        # ====================================================================
        # STEP 3: CANCER-SPECIFIC OPTIMIZATION
        # ====================================================================
        if self.verbose >= 1:
            print("\n" + "="*80)
            print("STEP 3: CANCER-SPECIFIC OPTIMIZATION")
            print("="*80)

        specific_params_by_cancer = {}
        specific_auc_by_cancer = {}

        for cancer, (X, y, groups) in cancer_data_dict.items():
            if self.verbose >= 1:
                print(f"\n{'-'*80}")
                print(f"Optimizing {cancer}...")
                print(f"{'-'*80}")

            results = self.optimize(X, y, groups=groups, algorithm=algorithm)
            specific_params_by_cancer[cancer] = results['best_params_consensus']
            specific_auc_by_cancer[cancer] = results['mean_score']

            if self.verbose >= 1:
                print(f"Cancer-specific params: {results['best_params_consensus']}")
                print(f"Cancer-specific AUC: {results['mean_score']:.4f} ± {results['std_score']:.4f}")

        # ====================================================================
        # STEP 4: COMPARISON & DECISION
        # ====================================================================
        if self.verbose >= 1:
            print("\n" + "="*80)
            print("STEP 4: COMPARISON & DECISION")
            print("="*80)

        delta_auc_by_cancer = {}
        for cancer in cancer_data_dict.keys():
            delta = specific_auc_by_cancer[cancer] - universal_auc_by_cancer[cancer]
            delta_auc_by_cancer[cancer] = delta

            if self.verbose >= 1:
                print(f"\n{cancer}:")
                print(f"  Universal:       {universal_auc_by_cancer[cancer]:.4f}")
                print(f"  Cancer-specific: {specific_auc_by_cancer[cancer]:.4f}")
                print(f"  Δ AUC:           {delta:+.4f}")

        mean_delta = np.mean(list(delta_auc_by_cancer.values()))
        max_delta = max(delta_auc_by_cancer.values())
        max_cancer = max(delta_auc_by_cancer, key=delta_auc_by_cancer.get)

        if self.verbose >= 1:
            print(f"\n{'='*80}")
            print(f"Mean Δ AUC: {mean_delta:+.4f}")
            print(f"Max Δ AUC:  {max_delta:+.4f} ({max_cancer})")
            print(f"Min Δ AUC:  {min(delta_auc_by_cancer.values()):+.4f} ({min(delta_auc_by_cancer, key=delta_auc_by_cancer.get)})")

        # Decision logic (from report Section 4.6)
        if mean_delta < 0.01:
            decision = 'universal'
            rationale = (
                f"Cancer-specific optimization yields minimal benefit (mean Δ AUC = {mean_delta:+.4f} < 0.01). "
                f"Universal hyperparameters are sufficient for consistency and simplicity."
            )
        elif mean_delta >= 0.02:
            decision = 'cancer_specific'
            rationale = (
                f"Cancer-specific optimization provides substantial benefit (mean Δ AUC = {mean_delta:+.4f} ≥ 0.02). "
                f"Recommend cancer-type-specific hyperparameters for Phase 2b."
            )
        else:  # 0.01 <= mean_delta < 0.02
            decision = 'hybrid'
            rationale = (
                f"Cancer-specific optimization shows borderline benefit (0.01 ≤ mean Δ AUC = {mean_delta:+.4f} < 0.02). "
                f"Consider hybrid approach: cancer-specific for large cancers, universal for small."
            )

        if self.verbose >= 1:
            print(f"\n{'='*80}")
            print(f"DECISION: {decision.upper()}")
            print(f"{'='*80}")
            print(f"{rationale}")
            print(f"{'='*80}")

        return {
            'universal_params': universal_params,
            'universal_pooled_auc': universal_pooled_auc,
            'universal_auc_by_cancer': universal_auc_by_cancer,
            'specific_params_by_cancer': specific_params_by_cancer,
            'specific_auc_by_cancer': specific_auc_by_cancer,
            'delta_auc_by_cancer': delta_auc_by_cancer,
            'mean_delta_auc': mean_delta,
            'max_delta_auc': max_delta,
            'decision': decision,
            'decision_rationale': rationale
        }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_estimator(self, algorithm: str, X=None, column_names=None):
        """
        Get estimator pipeline for algorithm.

        Returns a full pipeline: preprocessing → classifier.
        This ensures categorical features are properly encoded.

        Parameters
        ----------
        algorithm : str
            'random_forest' or 'logistic'
        X : array-like, optional
            Feature matrix (not used, kept for backwards compatibility)
        column_names : list of str, optional
            Column names for feature matrix (used to create preprocessing pipeline)

        Returns
        -------
        pipeline : Pipeline
            Full preprocessing + model pipeline
        """
        # Get base estimator
        if algorithm == 'random_forest':
            base_estimator = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=1  # Set to 1 when using parallel search
            )
        elif algorithm == 'logistic':
            base_estimator = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='saga'  # Supports all penalties
            )
        elif algorithm == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError(
                    "XGBoost not available. Install with: conda install -c conda-forge xgboost"
                )
            base_estimator = XGBClassifier(
                random_state=self.random_state,
                n_jobs=1,  # Set to 1 when using parallel search
                eval_metric='logloss',  # Suppress warning
                use_label_encoder=False  # Suppress deprecation warning
            )
        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Supported: 'random_forest', 'logistic', 'xgboost'"
            )

        # Create preprocessing pipeline
        preprocessor = create_preprocessor(available_columns=column_names)

        # Return full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', base_estimator)
        ])

        return pipeline

    def _compute_consensus_params(self, params_list: List[Dict]) -> Dict:
        """
        Find most common hyperparameters across folds.

        For each hyperparameter, returns the mode (most frequently selected value).
        This provides a single "consensus" set of hyperparameters that represents
        the most stable choice across all outer CV folds.
        """
        if not params_list:
            return {}

        consensus = {}
        param_names = params_list[0].keys()

        for param_name in param_names:
            values = [p[param_name] for p in params_list]
            # Find most common value (mode)
            most_common = Counter(values).most_common(1)[0][0]
            consensus[param_name] = most_common

        return consensus

    def _compute_param_stability(self, params_list: List[Dict]) -> float:
        """
        Measure hyperparameter stability across folds.

        Returns a score between 0 and 1, where:
        - 1.0 = All folds selected identical hyperparameters (perfect stability)
        - 0.2 = Each fold selected different hyperparameters (unstable)

        Calculated as the average fraction of folds that selected the most
        common value for each hyperparameter.
        """
        if not params_list or len(params_list) < 2:
            return 1.0

        stabilities = []
        param_names = params_list[0].keys()

        for param_name in param_names:
            values = [p[param_name] for p in params_list]
            counts = Counter(values)
            most_common_count = counts.most_common(1)[0][1]
            stability = most_common_count / len(params_list)
            stabilities.append(stability)

        return np.mean(stabilities)

    def _pool_data(
        self,
        cancer_data_dict: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pool data from multiple cancer types.

        Concatenates feature matrices and target vectors from all cancer types
        into single arrays for universal optimization.

        Note: Group information is discarded in pooled optimization.
        """
        X_list = []
        y_list = []

        for cancer, (X, y, groups) in cancer_data_dict.items():
            X_list.append(X)
            y_list.append(y)

        X_pooled = np.vstack(X_list)
        y_pooled = np.concatenate(y_list)

        return X_pooled, y_pooled
