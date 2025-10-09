"""
Basic Model Trainer for OncoKB Analyzer

Simple trainer with demographic features only (no technical features to avoid data leakage).
Supports Random Forest or Logistic Regression with basic preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging


class ModelTrainer:
    """Simple model trainer with basic preprocessing"""

    def __init__(self, algorithm: str = "random_forest", cv_folds: int = 5, random_state: int = 42,
                 cv_strategy: str = "standard", group_column: str = "SUBJECT_CENTER"):
        """
        Initialize model trainer.

        Args:
            algorithm: Either "random_forest" or "logistic"
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            cv_strategy: Either "standard" (KFold) or "group" (GroupKFold)
            group_column: Column name for grouping in GroupKFold (e.g., "SUBJECT_CENTER", "SEQ_ASSAY_ID")
        """
        self.algorithm = algorithm
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv_strategy = cv_strategy
        self.group_column = group_column
        self.logger = logging.getLogger(self.__class__.__name__)

        # Validate algorithm choice
        if algorithm not in ["random_forest", "logistic"]:
            raise ValueError(f"Algorithm must be 'random_forest' or 'logistic', got: {algorithm}")

        # Validate cv_strategy choice
        if cv_strategy not in ["standard", "group"]:
            raise ValueError(f"cv_strategy must be 'standard' or 'group', got: {cv_strategy}")

    def prepare_features(self, data: pd.DataFrame, outcome_col: str, feature_set: str = "both"):
        """
        Prepare features for modeling with flexible feature selection.

        Args:
            data: Patient data
            outcome_col: Name of outcome column
            feature_set: Which features to use - "demographic", "technical", or "both" (default)

        Returns:
            Tuple of (X, y, groups) for modeling. groups is None if cv_strategy is "standard",
            otherwise contains group identifiers for GroupKFold

        Raises:
            ValueError: If insufficient data, missing outcome column, or invalid feature_set
        """
        if outcome_col not in data.columns:
            raise ValueError(f"Outcome column '{outcome_col}' not found in data")

        # Validate feature_set parameter
        valid_feature_sets = ["demographic", "technical", "both"]
        if feature_set not in valid_feature_sets:
            raise ValueError(f"feature_set must be one of {valid_feature_sets}, got: {feature_set}")

        # Demographic columns
        demographic_list = [
            'SAMPLE_TYPE',
            'SAMPLE_TYPE_DETAILED',
            'SEX',
            'RACE',
            'ETHNICITY',
            'AGE_AT_SEQ_REPORT',
        ]

        ###excluded columns:
        #'PATIENT_ID',
        #'CANCER_TYPE',
        #'CANCER_TYPE_DETAILED',
        #'ONCOTREE_CODE',
        #'INT_Biopsy_To_Death',
        #'DEAD',
        #'INT_SEQ_TO_DEATH',
        #'INT_SEQ_TO_CONTACT',
        #'INT_Seq_or_Biopsy_To_Death',
        # 'TUMOR_PURITY',
        # 'TUMOR_PLOIDY'
        #'STAGE',

        technical_list = [
            'SUBJECT_CENTER',
            'SEQ_ASSAY_ID',
            'is_paired_end',
            'library_selection',
            'library_strategy',
            'platform',
            'read_length',
            'target_capture_kit',
            'instrument_model',
            'gene_padding',
            'number_of_genes',
            'variant_classifications',
            'alteration_types',
            'preservation_technique',
            'specimen_tumor_cellularity',
            'calling_strategy',
            'coverage',
            'SEQ_PIPELINE_ID'
        ]

        # Select features based on feature_set parameter
        if feature_set == "demographic":
            feature_list = demographic_list
        elif feature_set == "technical":
            feature_list = technical_list
        else:  # feature_set == "both"
            feature_list = demographic_list + technical_list

        # Filter to columns that actually exist in the data
        available_cols = [col for col in feature_list if col in data.columns]

        if not available_cols:
            raise ValueError(f"No feature columns found in data. Available columns: {data.columns.tolist()}")

        self.logger.info(f"Using {len(available_cols)} {feature_set} features: {available_cols}")

        X = data[available_cols].copy()
        y = data[outcome_col].copy()

        # Extract groups for GroupKFold if needed
        groups = None
        if self.cv_strategy == "group":
            if self.group_column not in data.columns:
                raise ValueError(f"Group column '{self.group_column}' not found in data. Available columns: {data.columns.tolist()}")

            groups = data[self.group_column].copy()
            self.logger.info(f"Extracted {groups.nunique()} unique groups from '{self.group_column}' for GroupKFold")

        # Basic validation
        if len(X) == 0:
            raise ValueError("No data available for modeling")

        if y.nunique() < 2:
            raise ValueError(f"Outcome variable has only {y.nunique()} unique values - need at least 2 for classification")

        self.logger.info(f"Prepared features: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y, groups

    def create_pipeline(self, available_columns=None):
        """
        Create preprocessing + model pipeline.

        Args:
            available_columns: List of available columns to filter preprocessing

        Returns:
            Scikit-learn Pipeline object
        """
        # Define all possible column types for preprocessing
        all_numeric_cols = [
            'AGE_AT_SEQ_REPORT',
            'read_length',
            'gene_padding',
            'number_of_genes'
        ]

        all_categorical_cols = [
            # Demographic
            'SAMPLE_TYPE', 'SAMPLE_TYPE_DETAILED', 'SEX', 'RACE', 'ETHNICITY',
            # Technical
            'SUBJECT_CENTER', 'SEQ_ASSAY_ID', 'is_paired_end', 'library_selection',
            'library_strategy', 'platform', 'target_capture_kit', 'instrument_model',
            'variant_classifications', 'alteration_types', 'preservation_technique',
            'calling_strategy', 'SEQ_PIPELINE_ID', 'specimen_tumor_cellularity', 'coverage'
        ]

        # Filter to only available columns if specified
        if available_columns is not None:
            numeric_cols = [col for col in all_numeric_cols if col in available_columns]
            categorical_cols = [col for col in all_categorical_cols if col in available_columns]
        else:
            numeric_cols = all_numeric_cols
            categorical_cols = all_categorical_cols

        # Create preprocessor
        preprocessor = ColumnTransformer([
            ('num', SimpleImputer(strategy='median'), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ]), categorical_cols)
        ], remainder='drop')  # Drop any columns not explicitly handled

        # Select model
        if self.algorithm == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1  # Use all available cores
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
        """
        Train model and get basic evaluation metrics.

        Args:
            X: Feature matrix
            y: Target vector
            groups: Group labels for GroupKFold (required if cv_strategy is "group")

        Returns:
            Dictionary with evaluation results

        Raises:
            ValueError: If insufficient data for training or groups missing for GroupKFold
        """
        if len(X) < 20:
            raise ValueError(f"Insufficient data for training: {len(X)} samples (minimum 20 required)")

        # Check class balance
        class_counts = y.value_counts()
        if class_counts.min() < 5:
            raise ValueError(f"Insufficient samples in minority class: {class_counts.to_dict()}")

        # Validate groups if using GroupKFold
        if self.cv_strategy == "group":
            if groups is None:
                raise ValueError("groups parameter is required when cv_strategy is 'group'")
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
                n_jobs=-1
            )
        else:  # standard
            self.logger.info(f"Performing {self.cv_folds}-fold standard cross-validation...")
            cv_scores = cross_val_score(
                pipeline, X, y,
                cv=self.cv_folds,
                scoring='roc_auc',
                n_jobs=-1
            )

        # Train-test split for additional evaluation
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
