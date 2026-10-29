"""Train and cross-validate Random Forest or Logistic Regression models.

Feature lists come from preprocessing.py.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging

from .stratified_group_kfold import StratifiedGroupKFold
from .preprocessing import (
    DEMOGRAPHIC_FEATURES, TECHNICAL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    create_preprocessor,
)


class ModelTrainer:
    """Simple model trainer with basic preprocessing."""

    def __init__(self, algorithm: str = "random_forest", cv_folds: int = 5, random_state: int = 42,
                 cv_strategy: str = "stratified_group", group_column: str = "SUBJECT_CENTER",
                 institution_aware_test_split: bool = False, test_split_group_column: str = "SUBJECT_CENTER",
                 n_jobs: int = -1):
        """Initialize model trainer."""
        self.algorithm = algorithm
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv_strategy = cv_strategy
        self.group_column = group_column
        self.institution_aware_test_split = institution_aware_test_split
        self.test_split_group_column = test_split_group_column
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(self.__class__.__name__)

        # Validate algorithm choice
        if algorithm not in ["random_forest", "logistic"]:
            raise ValueError(f"Algorithm must be 'random_forest' or 'logistic', got: {algorithm}")

        # Validate cv_strategy choice
        if cv_strategy not in ["standard", "group", "stratified_group"]:
            raise ValueError(f"cv_strategy must be 'standard', 'group', or 'stratified_group', got: {cv_strategy}")

    def prepare_features(self, data: pd.DataFrame, outcome_col: str, feature_set: str = "both"):
        """Prepare features for modeling with flexible feature selection."""
        if outcome_col not in data.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found in data")

        # Validate feature_set parameter
        valid_feature_sets = ["demographic", "technical", "both"]
        if feature_set not in valid_feature_sets:
            raise ValueError(f"feature_set must be one of {valid_feature_sets}, got: {feature_set}")

        # Feature lists come from preprocessing.py.
        if feature_set == "demographic":
            feature_list = DEMOGRAPHIC_FEATURES.copy()
        elif feature_set == "technical":
            feature_list = TECHNICAL_FEATURES.copy()
        else:  # feature_set == "both"
            feature_list = DEMOGRAPHIC_FEATURES + TECHNICAL_FEATURES

        # Filter to columns that actually exist in the data
        available_cols = [col for col in feature_list if col in data.columns]

        if not available_cols:
            raise ValueError(f"No feature columns found in data. Available columns: {data.columns.tolist()}")

        self.logger.info(f"Using {len(available_cols)} {feature_set} features: {available_cols}")

        X = data[available_cols].copy()
        y = data[outcome_col].copy()

        # Extract groups for GroupKFold/StratifiedGroupKFold or institution-aware test splits if needed
        groups = None
        if self.cv_strategy in ["group", "stratified_group"] or self.institution_aware_test_split:
            # Determine which column to use for grouping
            group_col = self.group_column if self.cv_strategy in ["group", "stratified_group"] else self.test_split_group_column

            if group_col not in data.columns:
                raise ValueError(f"Group column '{group_col}' not found in data. Available columns: {data.columns.tolist()}")

            groups = data[group_col].copy()

            if self.cv_strategy == "group":
                self.logger.info(f"Extracted {groups.nunique()} unique groups from '{group_col}' for GroupKFold CV")
            elif self.cv_strategy == "stratified_group":
                self.logger.info(f"Extracted {groups.nunique()} unique groups from '{group_col}' for StratifiedGroupKFold CV")
            if self.institution_aware_test_split:
                self.logger.info(f"Extracted {groups.nunique()} unique groups from '{group_col}' for institution-aware test split")

        # Basic validation
        if len(X) == 0:
            raise ValueError("No data available for modeling")

        if y.nunique() < 2:
            raise ValueError(f"Outcome variable has only {y.nunique()} unique values - need at least 2 for classification")

        self.logger.info(f"Prepared features: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y, groups

    def create_pipeline(self, available_columns=None):
        """Create preprocessing + model pipeline."""
        # Delegate to the shared preprocessor factory in preprocessing.py.
        preprocessor = create_preprocessor(available_columns=available_columns)

        # Select model
        if self.algorithm == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=self.n_jobs,
                oob_score=True,   # Enable out-of-bag score for overfitting monitoring
                bootstrap=True    # Required for oob_score
            )
        else:  # logistic regression
            model = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            )

        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        return pipeline

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series = None):
        """Train model and get basic evaluation metrics."""
        if len(X) < 20:
            raise ValueError(f"Insufficient data for training: {len(X)} samples (minimum 20 required)")

        # Check class balance
        class_counts = y.value_counts()
        if class_counts.min() < 5:
            raise ValueError(f"Insufficient samples in minority class: {class_counts.to_dict()}")

        # Validate groups if using GroupKFold or StratifiedGroupKFold
        if self.cv_strategy in ["group", "stratified_group"]:
            if groups is None:
                raise ValueError(f"groups parameter is required when cv_strategy is '{self.cv_strategy}'")
            if len(groups) != len(X):
                raise ValueError(f"groups length ({len(groups)}) must match X length ({len(X)})")

        self.logger.info(f"Training {self.algorithm} model with {len(X)} samples using {self.cv_strategy} CV")

        # Create pipeline with available columns
        pipeline = self.create_pipeline(available_columns=X.columns.tolist())

        # Cross-validation with appropriate strategy
        if self.cv_strategy == "group":
            cv_splitter = GroupKFold(n_splits=self.cv_folds)
            self.logger.info(f"Performing {self.cv_folds}-fold GroupKFold cross-validation on '{self.group_column}'...")
            cv_scores = cross_val_score(
                pipeline, X, y,
                cv=cv_splitter,
                groups=groups,
                scoring='roc_auc',
                n_jobs=self.n_jobs
            )
        elif self.cv_strategy == "stratified_group":
            cv_splitter = StratifiedGroupKFold(n_splits=self.cv_folds, random_state=self.random_state)
            self.logger.info(f"Performing {self.cv_folds}-fold StratifiedGroupKFold cross-validation on '{self.group_column}'...")
            cv_scores = cross_val_score(
                pipeline, X, y,
                cv=cv_splitter,
                groups=groups,
                scoring='roc_auc',
                n_jobs=self.n_jobs
            )
        else:  # standard
            self.logger.info(f"Performing {self.cv_folds}-fold standard cross-validation...")
            cv_scores = cross_val_score(
                pipeline, X, y,
                cv=self.cv_folds,
                scoring='roc_auc',
                n_jobs=self.n_jobs
            )

        # Train-test split for additional evaluation
        if self.institution_aware_test_split and groups is not None:
            # Group shuffle split keeps institutions from overlapping between train and test.
            self.logger.info(f"Using institution-aware test split with GroupShuffleSplit on '{self.group_column}'")

            splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=self.random_state)

            try:
                # Get train/test indices ensuring group separation
                train_idx, test_idx = next(splitter.split(X, y, groups=groups))
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Log group separation
                train_groups = groups.iloc[train_idx].unique()
                test_groups = groups.iloc[test_idx].unique()
                overlap = set(train_groups) & set(test_groups)

                self.logger.info(f"Train set: {len(train_groups)} unique groups, Test set: {len(test_groups)} unique groups")
                if len(overlap) > 0:
                    self.logger.warning(f"WARNING: {len(overlap)} groups overlap between train and test!")
                else:
                    self.logger.info("No group overlap between train and test sets (institution-aware split successful)")

            except Exception as e:
                self.logger.error(f"Institution-aware split failed: {e}. Falling back to standard split.")
                # Fallback to standard split
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=0.3,
                        random_state=self.random_state,
                        stratify=y
                    )
                except ValueError as e2:
                    self.logger.warning(f"Stratified split failed: {e2}. Using random split.")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=0.3,
                        random_state=self.random_state
                    )
        else:
            # Standard train-test split
            if self.institution_aware_test_split:
                self.logger.warning(
                    "Institution-aware test split requested but groups parameter is None. "
                    "Falling back to standard split."
                )

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.3,
                    random_state=self.random_state,
                    stratify=y
                )
            except ValueError as e:
                # If stratification fails due to class imbalance, try without stratification
                self.logger.warning(f"Stratified split failed: {e}. Using random split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.3,
                    random_state=self.random_state
                )

        # Fit pipeline and evaluate
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        results = {
            'algorithm': self.algorithm,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'test_auc': test_auc,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'model': pipeline  # Store trained model
        }

        self.logger.info(f"Training completed - CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        self.logger.info(f"Test AUC: {test_auc:.3f}, Test Accuracy: {test_accuracy:.3f}")

        return results
