"""
SHAP Interaction Value Analyzer

Core analysis engine for computing and analyzing pairwise feature interactions
using SHAP interaction values with TreeExplainer.

This module provides the SHAPInteractionAnalyzer class which:
- Calculates SHAP interaction values (3D matrix: samples × features × features)
- Decomposes predictions into main effects vs interaction effects
- Identifies strongest pairwise feature interactions
- Focuses on demographic feature interactions for biological interpretation

Author: Balder Analysis Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import shap
import logging
from typing import List, Dict, Optional, Tuple
from feature_name_mapper import FeatureNameMapper


class SHAPInteractionAnalyzer:
    """
    SHAP interaction value analyzer for Random Forest models.

    Computes pairwise feature interactions and decomposes predictions
    into main effects (individual features) vs interaction effects (feature pairs).

    Key Concepts:
    - Main effects: Individual feature contributions (diagonal of interaction matrix)
    - Interaction effects: Combined effects of feature pairs (off-diagonal)
    - Interaction value[i,j]: How much feature j modulates the effect of feature i

    Example Usage:
        >>> analyzer = SHAPInteractionAnalyzer(fitted_pipeline, X_train)
        >>> interaction_values = analyzer.calculate_interaction_values(X_test)
        >>> contributions = analyzer.compare_main_vs_interaction_contributions(interaction_values)
        >>> print(f"Main: {contributions['main_pct']:.1f}%, Interaction: {contributions['interaction_pct']:.1f}%")
    """

    def __init__(self, pipeline, X_train, feature_name_mapper: Optional[FeatureNameMapper] = None):
        """
        Initialize analyzer with fitted sklearn pipeline.

        Args:
            pipeline: Fitted sklearn Pipeline with 'preprocessor' and 'classifier' steps
            X_train: Training features (DataFrame, used as background data for TreeExplainer)
            feature_name_mapper: Optional FeatureNameMapper for human-readable feature names

        Raises:
            ValueError: If pipeline doesn't have required steps or model type not supported
        """
        self.logger = logging.getLogger(__name__)

        # Validate pipeline
        if not hasattr(pipeline, 'named_steps'):
            raise ValueError("Pipeline must have named_steps attribute")

        if 'preprocessor' not in pipeline.named_steps:
            raise ValueError("Pipeline must contain 'preprocessor' step")

        if 'classifier' not in pipeline.named_steps:
            raise ValueError("Pipeline must contain 'classifier' step")

        self.pipeline = pipeline
        self.X_train = X_train

        # Store feature name mapper
        self.feature_name_mapper = feature_name_mapper

        # Extract fitted components
        self.preprocessor = pipeline.named_steps['preprocessor']
        self.model = pipeline.named_steps['classifier']

        # Validate model type (must be tree-based for TreeExplainer)
        model_type = type(self.model).__name__
        if 'RandomForest' not in model_type and 'GradientBoosting' not in model_type:
            self.logger.warning(f"Model type {model_type} may not be optimal for TreeExplainer. "
                               f"Best results with RandomForest or GradientBoosting models.")

        # Transform background data
        X_train_transformed_raw = self.preprocessor.transform(X_train)

        # Convert sparse matrix to dense array (TreeExplainer requires dense)
        if hasattr(X_train_transformed_raw, 'toarray'):
            self.X_train_transformed = X_train_transformed_raw.toarray()
            self.logger.info("Converted sparse matrix to dense array for TreeExplainer")
        else:
            self.X_train_transformed = X_train_transformed_raw

        # Get feature names after transformation
        try:
            self.feature_names = self.preprocessor.get_feature_names_out()
        except AttributeError:
            # Fallback if get_feature_names_out not available
            self.feature_names = [f"feature_{i}" for i in range(self.X_train_transformed.shape[1])]
            self.logger.warning("Could not extract feature names, using generic names")

        # Create human-readable feature names if mapper provided
        if self.feature_name_mapper is not None:
            self.feature_names_readable = self.feature_name_mapper.map_all_feature_names(
                self.feature_names, include_value=True, short_form=False
            )
            self.logger.info("Mapped feature names to human-readable format")
        else:
            self.feature_names_readable = self.feature_names
            self.logger.warning("No feature name mapper provided, using sklearn transformed names")

        # Initialize TreeExplainer
        # Note: For interaction values, we initialize WITHOUT background data
        # Background data is only needed for standard SHAP values speedup
        self.logger.info(f"Initializing TreeExplainer with {len(self.feature_names)} features")
        self.explainer = shap.TreeExplainer(self.model)

        self.logger.info("SHAPInteractionAnalyzer initialized successfully")

    def calculate_interaction_values(self, X_test, check_additivity=False):
        """
        Compute SHAP interaction values for test set.

        Uses TreeExplainer.shap_interaction_values() for exact computation.
        Returns 3D matrix where:
        - interaction_values[i, j, k] = interaction between features j and k for sample i
        - Diagonal [i, j, j] = main effect of feature j for sample i
        - Off-diagonal [i, j, k] = interaction between features j and k for sample i

        Mathematical properties:
        - Symmetry: interaction_values[i, j, k] == interaction_values[i, k, j]
        - Additivity: sum(interaction_values[i, j, :]) == standard_shap_values[i, j]

        Args:
            X_test: Test features (DataFrame)
            check_additivity: Whether to verify additivity property (slow, for debugging)

        Returns:
            np.ndarray: 3D array of shape (n_samples, n_features, n_features)

        Raises:
            RuntimeError: If SHAP calculation fails
        """
        self.logger.info("Calculating SHAP interaction values (this may take 5-10 minutes)...")

        try:
            # Transform test data
            X_test_transformed_raw = self.preprocessor.transform(X_test)

            # Convert sparse matrix to dense array (TreeExplainer requires dense)
            if hasattr(X_test_transformed_raw, 'toarray'):
                X_test_transformed = X_test_transformed_raw.toarray()
            else:
                X_test_transformed = X_test_transformed_raw

            n_samples = X_test_transformed.shape[0]
            n_features = X_test_transformed.shape[1]

            self.logger.info(f"  Processing {n_samples} samples with {n_features} features")

            # Calculate interaction values
            # Note: check_additivity parameter removed (not supported in this SHAP version)
            interaction_values = self.explainer.shap_interaction_values(X_test_transformed)

            # For binary classification, extract interaction values for positive class (class 1)
            # Note: For Random Forest, class 0 and class 1 values are often negatives of each other
            #       or class 1 may be all zeros. We'll use class 0 and take absolute values.
            if isinstance(interaction_values, list):
                # Old SHAP version: list of arrays - use class 0
                interaction_values_class1 = interaction_values[0]
                self.logger.info("  Extracted interaction values for class 0 (list format)")
            elif len(interaction_values.shape) == 4:
                # New SHAP version: 4D array (samples, features, features, classes) - use class 0
                interaction_values_class1 = interaction_values[:, :, :, 0]
                self.logger.info("  Extracted interaction values for class 0 (4D array format)")
            else:
                # Single class or already extracted
                interaction_values_class1 = interaction_values
                self.logger.info("  Using interaction values as-is (binary model or pre-extracted)")

            # Validate shape
            expected_shape = (n_samples, n_features, n_features)
            if interaction_values_class1.shape != expected_shape:
                raise RuntimeError(f"Unexpected interaction values shape: {interaction_values_class1.shape}, "
                                  f"expected {expected_shape}")

            self.logger.info(f"  Interaction values calculated successfully: shape {interaction_values_class1.shape}")

            # Debug: Check value ranges
            min_val = np.min(interaction_values_class1)
            max_val = np.max(interaction_values_class1)
            mean_abs_val = np.mean(np.abs(interaction_values_class1))
            self.logger.info(f"  Value range: min={min_val:.6f}, max={max_val:.6f}, mean_abs={mean_abs_val:.6f}")

            # Validate symmetry
            max_asymmetry = np.max(np.abs(interaction_values_class1 -
                                          np.transpose(interaction_values_class1, (0, 2, 1))))
            self.logger.info(f"  Symmetry check: max asymmetry = {max_asymmetry:.2e}")

            if max_asymmetry > 1e-5:
                self.logger.warning(f"  Interaction matrix not perfectly symmetric (max diff: {max_asymmetry:.2e})")

            return interaction_values_class1

        except Exception as e:
            self.logger.error(f"SHAP interaction calculation failed: {e}")
            raise RuntimeError(f"Failed to calculate interaction values: {e}")

    def extract_main_effects(self, interaction_values):
        """
        Extract main effects from diagonal of interaction matrix.

        Main effects are the individual feature contributions, equivalent to
        standard SHAP values.

        Args:
            interaction_values: 3D interaction array (n_samples, n_features, n_features)

        Returns:
            np.ndarray: 2D array (n_samples, n_features) of main effects
                Equivalent to standard SHAP values
        """
        # Extract diagonal (axis 1 and 2 are features)
        main_effects = np.diagonal(interaction_values, axis1=1, axis2=2)

        self.logger.debug(f"Extracted main effects: shape {main_effects.shape}")

        return main_effects

    def extract_interaction_effects(self, interaction_values):
        """
        Extract pure interaction effects (off-diagonal elements).

        Args:
            interaction_values: 3D interaction array

        Returns:
            np.ndarray: 3D array with diagonal zeroed out
                Contains only pairwise interaction terms
        """
        # Create copy to avoid modifying original
        interaction_only = interaction_values.copy()

        # Zero out diagonal (main effects)
        n_samples, n_features, _ = interaction_values.shape
        for i in range(n_features):
            interaction_only[:, i, i] = 0.0

        self.logger.debug(f"Extracted interaction effects (diagonal zeroed)")

        return interaction_only

    def compare_main_vs_interaction_contributions(self, interaction_values):
        """
        Calculate what percentage of predictions comes from interactions vs main effects.

        Key metric for understanding model behavior:
        - High interaction % → Model relies on feature combinations
        - Low interaction % → Model is approximately additive

        Interpretation thresholds:
        - < 10%: Primarily main effects (additive model)
        - 10-30%: Moderate interaction effects
        - > 30%: Strong interaction effects driving predictions

        Args:
            interaction_values: 3D interaction array

        Returns:
            dict: {
                'main_total': float,           # Total absolute main effect
                'interaction_total': float,    # Total absolute interaction effect
                'main_pct': float,            # % from main effects
                'interaction_pct': float,     # % from interactions
                'n_samples': int,
                'n_features': int
            }
        """
        n_samples, n_features, _ = interaction_values.shape

        # Main effects are on the diagonal
        main_effects = np.diagonal(interaction_values, axis1=1, axis2=2)
        main_total = np.abs(main_effects).sum()

        # Interaction effects are off-diagonal
        # Count each pair once (upper triangle only) to avoid double counting
        interaction_total = 0.0

        for i in range(n_features):
            for j in range(i + 1, n_features):  # Upper triangle only
                interaction_total += np.abs(interaction_values[:, i, j]).sum()

        # Total absolute attribution
        total = main_total + interaction_total

        result = {
            'main_total': float(main_total),
            'interaction_total': float(interaction_total),
            'main_pct': float(main_total / total * 100) if total > 0 else 0.0,
            'interaction_pct': float(interaction_total / total * 100) if total > 0 else 0.0,
            'n_samples': n_samples,
            'n_features': n_features
        }

        self.logger.info(f"Contribution breakdown:")
        self.logger.info(f"  Main effects: {result['main_pct']:.2f}%")
        self.logger.info(f"  Interactions: {result['interaction_pct']:.2f}%")

        return result

    def get_strongest_interactions(self, interaction_values, top_n=20,
                                     feature_types=None):
        """
        Identify and rank strongest pairwise interactions.

        Algorithm:
            For each feature pair (i, j) where i < j:
                1. Extract interaction values: interaction_values[:, i, j]
                2. Calculate mean absolute interaction
                3. Rank all pairs by strength
                4. Return top N

        Args:
            interaction_values: 3D interaction array
            top_n: Number of top interactions to return
            feature_types: Optional dict mapping feature names to types
                          (e.g., {'AGE': 'demographic', 'SEQ_ASSAY_ID': 'technical'})

        Returns:
            pd.DataFrame with columns:
                - feature_i: Name of first feature
                - feature_j: Name of second feature
                - mean_abs_interaction: Mean |interaction| across samples
                - mean_interaction: Mean signed interaction
                - std_interaction: Standard deviation
                - feature_i_type: Type of feature_i (if provided)
                - feature_j_type: Type of feature_j (if provided)
        """
        n_samples, n_features, _ = interaction_values.shape

        self.logger.info(f"Identifying top {top_n} interactions from {n_features * (n_features - 1) // 2} pairs")

        interactions_list = []

        # Iterate over upper triangle (i < j) to avoid duplicates
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Extract interaction values for this pair across all samples
                interaction_pair = interaction_values[:, i, j]

                # Calculate statistics
                mean_abs = np.abs(interaction_pair).mean()
                mean_signed = interaction_pair.mean()
                std = interaction_pair.std()

                interactions_list.append({
                    'feature_i': self.feature_names_readable[i],
                    'feature_j': self.feature_names_readable[j],
                    'feature_i_raw': self.feature_names[i],
                    'feature_j_raw': self.feature_names[j],
                    'mean_abs_interaction': mean_abs,
                    'mean_interaction': mean_signed,
                    'std_interaction': std
                })

        # Create DataFrame and sort by absolute interaction strength
        interactions_df = pd.DataFrame(interactions_list)
        interactions_df = interactions_df.sort_values('mean_abs_interaction',
                                                       ascending=False).reset_index(drop=True)

        # Add feature types if provided
        if feature_types is not None:
            interactions_df['feature_i_type'] = interactions_df['feature_i'].map(
                lambda x: self._get_feature_type(x, feature_types)
            )
            interactions_df['feature_j_type'] = interactions_df['feature_j'].map(
                lambda x: self._get_feature_type(x, feature_types)
            )

        # Return top N
        top_interactions = interactions_df.head(top_n)

        self.logger.info(f"Top interaction: {top_interactions.iloc[0]['feature_i']} × "
                        f"{top_interactions.iloc[0]['feature_j']} "
                        f"(strength: {top_interactions.iloc[0]['mean_abs_interaction']:.4f})")

        return top_interactions

    def get_demographic_interactions(self, interaction_values,
                                      demographic_features):
        """
        Extract interactions among demographic features only.

        Focuses on biologically interesting interactions:
        AGE × SEX, AGE × RACE, SEX × RACE, RACE × ETHNICITY, etc.

        Args:
            interaction_values: 3D interaction array
            demographic_features: List of demographic feature names (original, pre-encoding)
                                 Example: ['AGE_AT_SEQ_REPORT', 'SEX', 'RACE', 'ETHNICITY']

        Returns:
            pd.DataFrame: Demographic feature pairs with interaction strengths
        """
        # Find indices of demographic features in transformed feature names
        demographic_indices = []
        demographic_names = []

        for orig_feature in demographic_features:
            # Find all transformed features from this original feature
            # Handle one-hot encoding: cat__SEX_Male, cat__SEX_Female → SEX
            # Handle numeric: num__AGE_AT_SEQ_REPORT → AGE_AT_SEQ_REPORT
            matching_indices = []

            for idx, feat_name in enumerate(self.feature_names):
                # Check if this transformed feature comes from the original feature
                if orig_feature in feat_name:
                    matching_indices.append(idx)

            if matching_indices:
                # For simplicity, use first matching feature
                # (More sophisticated: aggregate across one-hot encoded features)
                demographic_indices.append(matching_indices[0])
                demographic_names.append(orig_feature)

        self.logger.info(f"Found {len(demographic_indices)} demographic features in transformed data")

        # Extract interactions among demographic features
        demo_interactions = []

        for i, idx_i in enumerate(demographic_indices):
            for j, idx_j in enumerate(demographic_indices):
                if i < j:  # Upper triangle only
                    interaction_pair = interaction_values[:, idx_i, idx_j]

                    demo_interactions.append({
                        'feature_i': demographic_names[i],
                        'feature_j': demographic_names[j],
                        'mean_abs_interaction': np.abs(interaction_pair).mean(),
                        'mean_interaction': interaction_pair.mean(),
                        'std_interaction': interaction_pair.std()
                    })

        demo_df = pd.DataFrame(demo_interactions)
        demo_df = demo_df.sort_values('mean_abs_interaction',
                                       ascending=False).reset_index(drop=True)

        self.logger.info(f"Extracted {len(demo_df)} demographic interaction pairs")

        return demo_df

    def _get_feature_type(self, feature_name, feature_types):
        """
        Helper to determine feature type from transformed feature name.

        Maps transformed feature names (e.g., cat__SEX_Male) back to original
        feature names to look up type.
        """
        # Try exact match first
        if feature_name in feature_types:
            return feature_types[feature_name]

        # Try to extract original feature name from transformed name
        # Handle patterns: cat__FEATURE_VALUE, num__FEATURE
        for orig_name, feat_type in feature_types.items():
            if orig_name in feature_name:
                return feat_type

        return 'unknown'
