"""
SHAP Primary Feature Value Analyzer

Core analysis engine for computing primary SHAP values (mean absolute SHAP per feature)
to determine feature importance rankings for actionability prediction.

This module provides the SHAPPrimaryAnalyzer class which:
- Calculates SHAP values using TreeExplainer or LinearExplainer with background data optimization
- Ranks features by mean absolute SHAP value
- Generates beeswarm summary plots
- Handles pre-trained models from TIER 5

Supported model families:
- Tree-based: RandomForest, GradientBoosting, XGBoost, LightGBM (uses shap.TreeExplainer)
- Linear: LogisticRegression (uses shap.LinearExplainer)

Different from interaction_analyzer.py which computes pairwise interactions.
This module focuses on individual feature importance.

"""

import numpy as np
import pandas as pd
import shap
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from feature_name_mapper import FeatureNameMapper, create_mapper_from_data_loader


class SHAPPrimaryAnalyzer:
    """
    SHAP primary value analyzer for pre-trained models.

    Computes standard SHAP values (not interaction values) to determine
    individual feature importance for actionability prediction.

    Key Concepts:
    - SHAP value: Contribution of a single feature to the prediction
    - Mean |SHAP|: Average absolute contribution across all samples
    - Feature importance ranking: Features sorted by mean |SHAP|

    Example Usage:
        >>> analyzer = SHAPPrimaryAnalyzer(model, preprocessor, feature_columns)
        >>> shap_values, expected_value = analyzer.calculate_shap_values(X_test)
        >>> importance_df = analyzer.get_feature_importance(shap_values)
        >>> analyzer.plot_summary(shap_values, X_test, 'output/shap_summary.pdf')
    """

    def __init__(
        self,
        model,
        preprocessor,
        feature_columns: List[str],
        feature_name_mapper: Optional[FeatureNameMapper] = None,
        background_sample_size: int = 500,
        random_seed: int = 42
    ):
        """
        Initialize analyzer with pre-trained model and preprocessor.

        Args:
            model: Trained sklearn classifier (e.g., RandomForestClassifier)
            preprocessor: Fitted sklearn ColumnTransformer
            feature_columns: Original feature column names (before transformation)
            feature_name_mapper: Optional FeatureNameMapper for human-readable names
            background_sample_size: Number of samples for TreeExplainer background
            random_seed: Random seed for reproducibility

        Raises:
            ValueError: If model type is not supported by TreeExplainer
        """
        self.logger = logging.getLogger(__name__)
        self.model = model
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns
        self.background_sample_size = background_sample_size
        self.random_seed = random_seed

        # Validate model type and determine explainer family
        model_type = type(self.model).__name__
        supported_tree_models = ['RandomForest', 'GradientBoosting', 'XGB', 'LightGBM']
        supported_linear_models = ['LogisticRegression']

        if any(s in model_type for s in supported_tree_models):
            self.model_family = 'tree'
        elif any(s in model_type for s in supported_linear_models):
            self.model_family = 'linear'
        else:
            raise ValueError(
                f"Model type {model_type} not supported. "
                f"Use tree-based models: {supported_tree_models} "
                f"or linear models: {supported_linear_models}"
            )

        # Get feature names after transformation
        try:
            self.transformed_feature_names = list(self.preprocessor.get_feature_names_out())
        except AttributeError:
            # Fallback if get_feature_names_out not available
            self.transformed_feature_names = None
            self.logger.warning("Could not extract transformed feature names")

        # Create or use provided feature name mapper
        if feature_name_mapper is not None:
            self.feature_name_mapper = feature_name_mapper
        else:
            self.feature_name_mapper = create_mapper_from_data_loader(feature_columns)

        # Map transformed names to human-readable
        if self.transformed_feature_names is not None:
            self.feature_names_readable = self.feature_name_mapper.map_all_feature_names(
                self.transformed_feature_names, include_value=True, short_form=False
            )
        else:
            self.feature_names_readable = None

        self.explainer = None  # Created lazily with background data
        self.expected_value = None

        self.logger.info(f"SHAPPrimaryAnalyzer initialized for {model_type}")
        self.logger.info(f"  Original features: {len(feature_columns)}")
        if self.transformed_feature_names:
            self.logger.info(f"  Transformed features: {len(self.transformed_feature_names)}")

    def _create_explainer(self, X_train: pd.DataFrame):
        """
        Create SHAP explainer with background data sample.

        Uses TreeExplainer for tree-based models and LinearExplainer for linear models.
        A random sample of training data is used as background for faster computation
        and reduced memory usage.

        Args:
            X_train: Training data (DataFrame with original feature columns)
        """
        explainer_name = 'TreeExplainer' if self.model_family == 'tree' else 'LinearExplainer'
        self.logger.info(f"Creating {explainer_name} with background data...")

        # Transform training data
        X_train_transformed_raw = self.preprocessor.transform(X_train[self.feature_columns])

        # Convert sparse to dense if needed
        if hasattr(X_train_transformed_raw, 'toarray'):
            X_train_transformed = X_train_transformed_raw.toarray()
        else:
            X_train_transformed = X_train_transformed_raw

        # Sample background data for memory efficiency
        n_samples = X_train_transformed.shape[0]
        if n_samples > self.background_sample_size:
            np.random.seed(self.random_seed)
            indices = np.random.choice(n_samples, self.background_sample_size, replace=False)
            background_data = X_train_transformed[indices]
            self.logger.info(
                f"  Using {self.background_sample_size} background samples "
                f"(from {n_samples} total)"
            )
        else:
            background_data = X_train_transformed
            self.logger.info(f"  Using all {n_samples} samples as background")

        # Create explainer based on model family
        if self.model_family == 'tree':
            self.explainer = shap.TreeExplainer(
                self.model,
                data=background_data,
                feature_perturbation='interventional'
            )
        else:  # linear
            self.explainer = shap.LinearExplainer(
                self.model,
                background_data
            )

        # Store expected value (base prediction)
        self.expected_value = self.explainer.expected_value
        if isinstance(self.expected_value, np.ndarray):
            # For binary classification, use positive class
            self.expected_value = self.expected_value[1] if len(self.expected_value) > 1 else self.expected_value[0]

        self.logger.info(f"  Expected value (base): {self.expected_value:.4f}")
        self.logger.info(f"{explainer_name} created successfully")

    def calculate_shap_values(
        self,
        X_test: pd.DataFrame,
        X_train: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Compute SHAP values for test set.

        Args:
            X_test: Test features (DataFrame with original feature columns)
            X_train: Training data for background (required on first call)

        Returns:
            Tuple of (shap_values, expected_value)
            - shap_values: 2D array (n_samples, n_features)
            - expected_value: Base prediction value

        Raises:
            ValueError: If explainer not initialized and X_train not provided
        """
        if self.explainer is None:
            if X_train is None:
                raise ValueError("X_train required to create explainer on first call")
            self._create_explainer(X_train)

        self.logger.info(f"Calculating SHAP values for {len(X_test)} samples...")

        # Transform test data
        X_test_transformed_raw = self.preprocessor.transform(X_test[self.feature_columns])

        # Convert sparse to dense if needed
        if hasattr(X_test_transformed_raw, 'toarray'):
            X_test_transformed = X_test_transformed_raw.toarray()
        else:
            X_test_transformed = X_test_transformed_raw

        # Calculate SHAP values
        shap_values_raw = self.explainer.shap_values(X_test_transformed)

        # Handle binary classification output format
        if isinstance(shap_values_raw, list):
            # Old SHAP format: list of arrays per class - use positive class (index 1)
            shap_values = shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0]
            self.logger.info("  Extracted SHAP values for positive class (list format)")
        elif len(shap_values_raw.shape) == 3:
            # New SHAP format: 3D array (samples, features, classes) - use positive class
            shap_values = shap_values_raw[:, :, 1] if shap_values_raw.shape[2] > 1 else shap_values_raw[:, :, 0]
            self.logger.info("  Extracted SHAP values for positive class (3D format)")
        else:
            shap_values = shap_values_raw
            self.logger.info("  Using SHAP values as-is (2D format)")

        # Validate shape
        expected_features = len(self.transformed_feature_names) if self.transformed_feature_names else shap_values.shape[1]
        if shap_values.shape[1] != expected_features:
            self.logger.warning(
                f"SHAP values shape {shap_values.shape} doesn't match "
                f"expected features {expected_features}"
            )

        self.logger.info(f"  SHAP values calculated: shape {shap_values.shape}")

        # Debug statistics
        mean_abs = np.mean(np.abs(shap_values))
        self.logger.info(f"  Mean |SHAP|: {mean_abs:.6f}")

        return shap_values, self.expected_value

    def get_feature_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        Calculate feature importance rankings from SHAP values.

        Features are ranked by mean absolute SHAP value across all samples.

        Args:
            shap_values: 2D array (n_samples, n_features) from calculate_shap_values()

        Returns:
            pd.DataFrame with columns:
                - feature_name: Human-readable feature name
                - feature_name_raw: sklearn transformed feature name
                - mean_abs_shap: Mean |SHAP| across samples
                - mean_shap: Mean signed SHAP value
                - std_shap: Standard deviation
                - rank: 1-based rank (1 = most important)
        """
        self.logger.info("Calculating feature importance rankings...")

        n_features = shap_values.shape[1]

        # Calculate statistics for each feature
        importance_data = []
        for i in range(n_features):
            feature_shap = shap_values[:, i]

            feature_name_raw = (
                self.transformed_feature_names[i]
                if self.transformed_feature_names and i < len(self.transformed_feature_names)
                else f"feature_{i}"
            )
            feature_name_readable = (
                self.feature_names_readable[i]
                if self.feature_names_readable and i < len(self.feature_names_readable)
                else feature_name_raw
            )

            importance_data.append({
                'feature_name': feature_name_readable,
                'feature_name_raw': feature_name_raw,
                'mean_abs_shap': np.abs(feature_shap).mean(),
                'mean_shap': feature_shap.mean(),
                'std_shap': feature_shap.std(),
                'median_abs_shap': np.median(np.abs(feature_shap)),
                'max_abs_shap': np.abs(feature_shap).max()
            })

        # Create DataFrame and sort by importance
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values(
            'mean_abs_shap', ascending=False
        ).reset_index(drop=True)

        # Add rank column
        importance_df['rank'] = range(1, len(importance_df) + 1)

        # Log top features
        self.logger.info("Top 10 features by mean |SHAP|:")
        for _, row in importance_df.head(10).iterrows():
            self.logger.info(
                f"  {row['rank']:2d}. {row['feature_name'][:50]:50s} "
                f"mean_abs={row['mean_abs_shap']:.4f}"
            )

        return importance_df

    def get_grouped_feature_importance(
        self,
        shap_values: np.ndarray,
        X_test_transformed: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Calculate feature importance with one-hot encoded features grouped.

        Sums absolute SHAP values across all one-hot encoded columns for each
        original categorical feature.

        Args:
            shap_values: 2D array from calculate_shap_values()
            X_test_transformed: Transformed test data (optional, for validation)

        Returns:
            pd.DataFrame with grouped feature importance
        """
        self.logger.info("Calculating grouped feature importance...")

        if self.transformed_feature_names is None:
            self.logger.warning("No transformed feature names available, returning ungrouped")
            return self.get_feature_importance(shap_values)

        # Group transformed features by original feature
        feature_groups = {}
        for i, feat_name in enumerate(self.transformed_feature_names):
            # Get base feature name (group) using mapper
            base_name = self.feature_name_mapper.get_feature_group(feat_name)
            if base_name not in feature_groups:
                feature_groups[base_name] = []
            feature_groups[base_name].append(i)

        # Calculate grouped importance
        grouped_data = []
        for base_name, indices in feature_groups.items():
            # Sum absolute SHAP values across all columns in group
            group_shap = np.sum(np.abs(shap_values[:, indices]), axis=1)

            # Get display name
            display_name = self.feature_name_mapper.DISPLAY_NAMES.get(base_name, base_name)

            grouped_data.append({
                'feature_name': display_name,
                'original_feature': base_name,
                'n_columns': len(indices),
                'mean_abs_shap': group_shap.mean(),
                'std_shap': group_shap.std(),
                'median_abs_shap': np.median(group_shap),
                'max_abs_shap': group_shap.max()
            })

        # Create DataFrame and sort
        grouped_df = pd.DataFrame(grouped_data)
        grouped_df = grouped_df.sort_values(
            'mean_abs_shap', ascending=False
        ).reset_index(drop=True)
        grouped_df['rank'] = range(1, len(grouped_df) + 1)

        self.logger.info(f"Grouped {len(self.transformed_feature_names)} columns into {len(grouped_df)} features")

        return grouped_df

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X_test: pd.DataFrame,
        output_path: str,
        max_display: int = 20,
        plot_type: str = 'beeswarm'
    ):
        """
        Generate SHAP summary plot (beeswarm or bar).

        Args:
            shap_values: 2D array from calculate_shap_values()
            X_test: Test features (for feature values in beeswarm)
            output_path: Path to save PDF
            max_display: Maximum number of features to display
            plot_type: 'beeswarm' (default) or 'bar'
        """
        self.logger.info(f"Generating {plot_type} summary plot...")

        # Transform test data for plotting
        X_test_transformed_raw = self.preprocessor.transform(X_test[self.feature_columns])
        if hasattr(X_test_transformed_raw, 'toarray'):
            X_test_transformed = X_test_transformed_raw.toarray()
        else:
            X_test_transformed = X_test_transformed_raw

        # Create figure
        plt.figure(figsize=(12, 10))

        # Create feature names for display
        if self.feature_names_readable:
            feature_names = self.feature_names_readable
        elif self.transformed_feature_names:
            feature_names = self.transformed_feature_names
        else:
            feature_names = [f"Feature {i}" for i in range(shap_values.shape[1])]

        # Truncate long names for display
        display_names = [
            (name[:40] + '...' if len(name) > 43 else name)
            for name in feature_names
        ]

        if plot_type == 'beeswarm':
            # Create SHAP explanation object for plotting
            explanation = shap.Explanation(
                values=shap_values,
                data=X_test_transformed,
                feature_names=display_names
            )
            shap.plots.beeswarm(explanation, max_display=max_display, show=False)
        else:
            # Bar plot of mean |SHAP|
            shap.summary_plot(
                shap_values,
                features=X_test_transformed,
                feature_names=display_names,
                max_display=max_display,
                plot_type='bar',
                show=False
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  Saved: {output_path}")

    def plot_feature_importance_bar(
        self,
        importance_df: pd.DataFrame,
        output_path: str,
        top_n: int = 20,
        title: Optional[str] = None
    ):
        """
        Generate bar chart of feature importance rankings.

        Args:
            importance_df: DataFrame from get_feature_importance()
            output_path: Path to save PDF
            top_n: Number of top features to display
            title: Optional plot title
        """
        self.logger.info(f"Generating feature importance bar chart (top {top_n})...")

        # Get top N features
        plot_df = importance_df.head(top_n).copy()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Truncate long names
        plot_df['display_name'] = plot_df['feature_name'].apply(
            lambda x: x[:35] + '...' if len(x) > 38 else x
        )

        # Horizontal bar chart
        y_pos = range(len(plot_df))
        ax.barh(y_pos, plot_df['mean_abs_shap'], color='steelblue', alpha=0.8)

        # Add error bars if std available
        if 'std_shap' in plot_df.columns:
            ax.errorbar(
                plot_df['mean_abs_shap'], y_pos,
                xerr=plot_df['std_shap'],
                fmt='none', color='black', alpha=0.5, capsize=3
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['display_name'], fontsize=9)
        ax.invert_yaxis()  # Top feature at top

        ax.set_xlabel('Mean |SHAP value|', fontsize=11)
        ax.set_ylabel('')

        if title:
            ax.set_title(title, fontsize=12)
        else:
            ax.set_title(f'Top {top_n} Features by SHAP Importance', fontsize=12)

        # Add rank labels
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.text(
                row['mean_abs_shap'] + ax.get_xlim()[1] * 0.01,
                i, f"#{row['rank']}",
                va='center', fontsize=8, color='gray'
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"  Saved: {output_path}")


def create_analyzer_from_tier5_model(
    model_file: str,
    feature_name_mapper: Optional[FeatureNameMapper] = None,
    background_sample_size: int = 500
) -> SHAPPrimaryAnalyzer:
    """
    Convenience function to create analyzer from TIER 5 saved model.

    Supports both tree-based models (RandomForest, etc.) and linear models
    (LogisticRegression). The appropriate SHAP explainer (TreeExplainer or
    LinearExplainer) is selected automatically based on the model type.

    Args:
        model_file: Path to joblib model file from TIER 5
        feature_name_mapper: Optional custom feature name mapper
        background_sample_size: Number of background samples for SHAP explainer

    Returns:
        Configured SHAPPrimaryAnalyzer instance

    Example:
        >>> analyzer = create_analyzer_from_tier5_model('LUAD_random_forest_model.joblib')
        >>> analyzer = create_analyzer_from_tier5_model('LUAD_logistic_model.joblib')
    """
    import joblib

    # Load model dict
    model_dict = joblib.load(model_file)

    # Extract components
    model = model_dict['model']
    preprocessor = model_dict['preprocessor']
    feature_columns = model_dict['feature_columns']

    return SHAPPrimaryAnalyzer(
        model=model,
        preprocessor=preprocessor,
        feature_columns=feature_columns,
        feature_name_mapper=feature_name_mapper,
        background_sample_size=background_sample_size
    )
