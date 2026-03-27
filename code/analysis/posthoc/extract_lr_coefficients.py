#!/usr/bin/env python3
"""
Extract Logistic Regression Coefficients, Odds Ratios, and Wald CIs

Extracts and analyzes logistic regression model coefficients from TIER 5
trained models. For each cancer type:
1. Loads LR model from joblib
2. Extracts log-odds coefficients and computes odds ratios
3. Computes Wald-based standard errors via Fisher information (Hessian)
4. Performs Wald z-tests with FDR correction (Benjamini-Hochberg)
5. Maps feature names to human-readable labels
6. Aggregates one-hot encoded features to parent variable level
7. Documents reference categories for each categorical feature

Outputs per cancer type:
    - lr_coefficients.csv: Per-feature coefficients, ORs, CIs, p-values, standardized coefs
    - lr_odds_ratio_forest_plot.pdf: Forest plot of odds ratios with CIs
    - lr_grouped_importance.csv: Aggregated to parent variable level

Combined outputs:
    - combined_lr_coefficients.csv: All cancer types combined
    - combined_lr_grouped_importance.csv: Grouped importance across types
    - lr_or_heatmap.pdf: Raw log-odds heatmap across cancer types
    - standardized_lr_or_heatmap.pdf: Standardized log-odds heatmap (scale-corrected)

Usage:
    # Quick test (LUAD only)
    python extract_lr_coefficients.py --quick-test

    # Full run
    python extract_lr_coefficients.py \\
        --model-dir /path/to/tier5_models \\
        --train-test-split-file /path/to/split.csv \\
        --db-path /path/to/database.sqlite

    # Specific cancer types
    python extract_lr_coefficients.py --cancer-types LUAD BRCA MEL
"""

import argparse
import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path

# Add modeling directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'modeling'))
from oncokb_analyzer.data_loader import DataLoader

# Add shap_interactions directory to path for feature name mapper
sys.path.insert(0, str(Path(__file__).parent.parent / 'shap_interactions'))
from feature_name_mapper import FeatureNameMapper, create_mapper_from_data_loader


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to both file and console."""
    log_level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger('lr_coefficients')
    logger.setLevel(log_level)
    logger.handlers = []

    # File handler
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    log_file = output_dir / 'lr_coefficients_analysis.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def find_model_files(
    model_dir: Path,
    cancer_types: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Find TIER 5 logistic regression model files in directory.

    Args:
        model_dir: Path to TIER 5 models directory
        cancer_types: Optional list of cancer types to filter

    Returns:
        Dict mapping cancer type to model file path
    """
    model_files = {}

    search_dirs = [model_dir, model_dir / 'models']

    # Add nested run directories
    for run_dir in model_dir.glob('run_*'):
        if run_dir.is_dir():
            search_dirs.append(run_dir)
            search_dirs.append(run_dir / 'models')

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for f in search_dir.glob('*_logistic_model.joblib'):
            cancer_type = f.stem.replace('_logistic_model', '')

            if cancer_types is None or cancer_type in cancer_types:
                model_files[cancer_type] = f

    return model_files


def compute_wald_standard_errors(
    model,
    preprocessor,
    feature_columns: List[str],
    train_data: pd.DataFrame,
    logger: logging.Logger
) -> Tuple[np.ndarray, bool]:
    """
    Compute standard errors for LR coefficients via the Fisher information matrix.

    Uses the Hessian-based Wald approach:
        Var(beta) = inv(X^T W X)
    where W = diag(p * (1 - p)) and p = predicted probabilities.

    Args:
        model: Trained LogisticRegression model
        preprocessor: Fitted ColumnTransformer
        feature_columns: Original feature column names
        train_data: Training data DataFrame
        logger: Logger instance

    Returns:
        Tuple of (standard_errors, se_reliable)
        - standard_errors: array of SEs for each coefficient
        - se_reliable: True if Fisher info matrix was invertible
    """
    # Transform training data
    X_train_raw = preprocessor.transform(train_data[feature_columns])

    # Convert sparse to dense
    if hasattr(X_train_raw, 'toarray'):
        X_train = X_train_raw.toarray()
    else:
        X_train = np.asarray(X_train_raw)

    n_samples, n_features = X_train.shape

    # Compute predicted probabilities
    y_proba = model.predict_proba(X_train)[:, 1]

    # Diagonal weight matrix: W = p * (1 - p)
    W = y_proba * (1 - y_proba)

    # Clip to avoid numerical issues with near-0 or near-1 probabilities
    W = np.clip(W, 1e-10, 1 - 1e-10)

    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(n_samples), X_train])

    # Fisher information matrix: X^T W X
    # Compute as X^T @ diag(W) @ X efficiently
    X_weighted = X_with_intercept * W[:, np.newaxis]
    fisher_info = X_with_intercept.T @ X_weighted

    # Invert Fisher information matrix
    se_reliable = True
    try:
        cov_matrix = np.linalg.inv(fisher_info)
    except np.linalg.LinAlgError:
        logger.warning("Fisher information matrix is singular, using pseudo-inverse")
        cov_matrix = np.linalg.pinv(fisher_info)
        se_reliable = False

    # Check for negative diagonal elements (numerical instability)
    diag_values = np.diag(cov_matrix)
    if np.any(diag_values < 0):
        logger.warning(
            f"Negative variance estimates detected ({(diag_values < 0).sum()} features). "
            "Using pseudo-inverse."
        )
        cov_matrix = np.linalg.pinv(fisher_info)
        diag_values = np.diag(cov_matrix)
        se_reliable = False

    # Standard errors = sqrt(diag(cov_matrix)), skip intercept [0]
    standard_errors = np.sqrt(np.maximum(diag_values[1:], 0))

    logger.info(f"  Computed Wald SEs for {n_features} features (reliable={se_reliable})")

    return standard_errors, se_reliable


def get_reference_categories(preprocessor, feature_columns: List[str]) -> Dict[str, str]:
    """
    Extract reference (dropped) categories from the preprocessor's OneHotEncoder.

    Args:
        preprocessor: Fitted ColumnTransformer
        feature_columns: Original feature column names

    Returns:
        Dict mapping feature name to its reference category
    """
    reference_cats = {}

    for name, transformer, columns in preprocessor.transformers_:
        if name == 'cat' and hasattr(transformer, 'named_steps'):
            encoder = transformer.named_steps.get('encoder') or transformer.named_steps.get('onehot')
            if encoder is not None and hasattr(encoder, 'categories_') and hasattr(encoder, 'drop_idx_'):
                # Get the categorical feature names
                from oncokb_analyzer.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
                cat_features_used = [f for f in feature_columns if f not in set(NUMERIC_FEATURES)]

                if encoder.drop_idx_ is not None:
                    for i, (cats, drop_idx) in enumerate(zip(encoder.categories_, encoder.drop_idx_)):
                        if i < len(cat_features_used) and drop_idx is not None:
                            reference_cats[cat_features_used[i]] = str(cats[int(drop_idx)])

    return reference_cats


def compute_feature_stds(
    preprocessor,
    feature_columns: List[str],
    train_data: pd.DataFrame,
    logger: logging.Logger
) -> np.ndarray:
    """
    Compute standard deviations of transformed features on training data.

    Used for standardizing LR coefficients: standardized_beta = beta * SD(x).
    This makes coefficients comparable across features with different scales
    (e.g., age in years vs binary one-hot indicators).

    Args:
        preprocessor: Fitted ColumnTransformer
        feature_columns: Original feature column names
        train_data: Training data DataFrame
        logger: Logger instance

    Returns:
        Array of standard deviations, one per transformed feature
    """
    X_train_raw = preprocessor.transform(train_data[feature_columns])

    if hasattr(X_train_raw, 'toarray'):
        X_train = X_train_raw.toarray()
    else:
        X_train = np.asarray(X_train_raw)

    feature_stds = np.std(X_train, axis=0)
    logger.info(f"  Feature SDs: min={feature_stds.min():.4f}, max={feature_stds.max():.4f}, "
                f"zero_count={np.sum(feature_stds == 0)}")

    return feature_stds


def extract_coefficients_single_cancer_type(
    cancer_type: str,
    model_file: Path,
    train_data: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> Optional[Dict]:
    """
    Extract LR coefficients for a single cancer type.

    Args:
        cancer_type: Cancer type code
        model_file: Path to TIER 5 logistic model file
        train_data: Training data for this cancer type
        output_dir: Output directory for this cancer type
        logger: Logger instance

    Returns:
        Result dict with status and metadata, or None if failed
    """
    logger.info("=" * 70)
    logger.info(f"EXTRACTING COEFFICIENTS: {cancer_type}")
    logger.info("=" * 70)

    try:
        # Load model
        logger.info(f"Loading model from {model_file}")
        model_dict = joblib.load(model_file)

        model = model_dict['model']
        preprocessor = model_dict['preprocessor']
        feature_columns = model_dict['feature_columns']
        algorithm = model_dict.get('algorithm', 'logistic')

        if algorithm != 'logistic':
            logger.warning(f"Model algorithm is '{algorithm}', expected 'logistic'. Skipping.")
            return {'cancer_type': cancer_type, 'status': 'skipped', 'error': 'Not a logistic model'}

        model_type = type(model).__name__
        logger.info(f"  Model type: {model_type}")
        logger.info(f"  Features: {len(feature_columns)}")
        logger.info(f"  Train samples: {len(train_data)}")

        # Extract coefficients
        log_odds = model.coef_[0]
        intercept = model.intercept_[0]

        # Get transformed feature names
        transformed_names = list(preprocessor.get_feature_names_out())
        logger.info(f"  Transformed features: {len(transformed_names)}")

        if len(log_odds) != len(transformed_names):
            logger.error(
                f"Coefficient count ({len(log_odds)}) != transformed feature count "
                f"({len(transformed_names)})"
            )
            return {'cancer_type': cancer_type, 'status': 'failed', 'error': 'Dimension mismatch'}

        # Compute Wald standard errors
        standard_errors, se_reliable = compute_wald_standard_errors(
            model, preprocessor, feature_columns, train_data, logger
        )

        # Wald z-test: z = beta / se(beta)
        z_scores = np.where(standard_errors > 0, log_odds / standard_errors, 0.0)
        p_values = 2 * stats.norm.sf(np.abs(z_scores))

        # Odds ratios with 95% CI
        odds_ratios = np.exp(log_odds)
        or_ci_lower = np.exp(log_odds - 1.96 * standard_errors)
        or_ci_upper = np.exp(log_odds + 1.96 * standard_errors)

        # FDR correction (Benjamini-Hochberg)
        valid_mask = ~np.isnan(p_values) & (p_values > 0)
        fdr_pvalues = np.full_like(p_values, np.nan)
        significant_fdr = np.full(len(p_values), False)

        if valid_mask.sum() > 0:
            _, fdr_vals, _, _ = multipletests(
                p_values[valid_mask], method='fdr_bh', alpha=0.05
            )
            fdr_pvalues[valid_mask] = fdr_vals
            significant_fdr[valid_mask] = fdr_vals < 0.05

        # Compute standardized coefficients: standardized_beta = beta * SD(x)
        feature_stds = compute_feature_stds(
            preprocessor, feature_columns, train_data, logger
        )
        standardized_log_odds = log_odds * feature_stds

        # Map feature names
        mapper = create_mapper_from_data_loader(feature_columns)
        readable_names = mapper.map_all_feature_names(
            transformed_names, include_value=True, short_form=False
        )
        feature_groups = [mapper.get_feature_group(name) for name in transformed_names]

        # Get reference categories
        reference_cats = get_reference_categories(preprocessor, feature_columns)

        # Build per-feature DataFrame
        coef_df = pd.DataFrame({
            'feature_name': readable_names,
            'feature_name_raw': transformed_names,
            'feature_group': feature_groups,
            'log_odds': log_odds,
            'standardized_log_odds': standardized_log_odds,
            'feature_std': feature_stds,
            'odds_ratio': odds_ratios,
            'or_ci_lower': or_ci_lower,
            'or_ci_upper': or_ci_upper,
            'standard_error': standard_errors,
            'z_score': z_scores,
            'p_value': p_values,
            'p_fdr': fdr_pvalues,
            'significant_fdr': significant_fdr,
            'se_reliable': se_reliable,
            'cancer_type': cancer_type
        })

        # Sort by absolute standardized log-odds (scale-corrected importance)
        coef_df = coef_df.sort_values('standardized_log_odds', key=abs, ascending=False).reset_index(drop=True)
        coef_df['rank'] = range(1, len(coef_df) + 1)

        # Create output directory
        cancer_output_dir = output_dir / cancer_type
        cancer_output_dir.mkdir(parents=True, exist_ok=True)

        # Save coefficients CSV
        coef_file = cancer_output_dir / 'lr_coefficients.csv'
        coef_df.to_csv(coef_file, index=False)
        logger.info(f"  Saved: {coef_file}")

        # Save reference categories
        if reference_cats:
            ref_df = pd.DataFrame([
                {'feature': k, 'reference_category': v, 'cancer_type': cancer_type}
                for k, v in reference_cats.items()
            ])
            ref_file = cancer_output_dir / 'lr_reference_categories.csv'
            ref_df.to_csv(ref_file, index=False)
            logger.info(f"  Saved: {ref_file}")

        # Aggregate to parent variable level
        grouped_df = _aggregate_to_parent_variable(coef_df, cancer_type)
        grouped_file = cancer_output_dir / 'lr_grouped_importance.csv'
        grouped_df.to_csv(grouped_file, index=False)
        logger.info(f"  Saved: {grouped_file}")

        # Generate forest plot
        plot_file = cancer_output_dir / 'lr_odds_ratio_forest_plot.pdf'
        _plot_odds_ratio_forest(coef_df, cancer_type, str(plot_file))
        logger.info(f"  Saved: {plot_file}")

        # Log top features
        n_sig = coef_df['significant_fdr'].sum()
        logger.info(f"  Intercept: {intercept:.4f}")
        logger.info(f"  Significant features (FDR < 0.05): {n_sig}/{len(coef_df)}")
        logger.info("  Top 5 by |standardized log-odds|:")
        for _, row in coef_df.head(5).iterrows():
            sig_mark = '*' if row['significant_fdr'] else ''
            logger.info(
                f"    {row['rank']:2d}. {row['feature_name'][:40]:40s} "
                f"std_coef={row['standardized_log_odds']:+.4f} "
                f"OR={row['odds_ratio']:.3f} p={row['p_value']:.2e}{sig_mark}"
            )

        return {
            'cancer_type': cancer_type,
            'n_train': len(train_data),
            'n_features': len(feature_columns),
            'n_transformed_features': len(transformed_names),
            'intercept': float(intercept),
            'n_significant_fdr': int(n_sig),
            'se_reliable': se_reliable,
            'top_feature': coef_df.iloc[0]['feature_name'],
            'top_feature_or': float(coef_df.iloc[0]['odds_ratio']),
            'top_feature_std_coef': float(coef_df.iloc[0]['standardized_log_odds']),
            'status': 'success',
            'output_dir': str(cancer_output_dir)
        }

    except Exception as e:
        logger.error(f"Extraction failed for {cancer_type}: {str(e)}")
        logger.exception("Full traceback:")
        return {
            'cancer_type': cancer_type,
            'status': 'failed',
            'error': str(e)
        }


def _aggregate_to_parent_variable(coef_df: pd.DataFrame, cancer_type: str) -> pd.DataFrame:
    """
    Aggregate one-hot encoded feature coefficients to parent variable level.

    For each original feature (e.g., SEX, RACE), computes:
    - max_abs_log_odds: Maximum absolute log-odds across categories
    - sum_abs_log_odds: Sum of absolute log-odds (total effect size)
    - n_categories: Number of encoded categories
    - min_p_value: Minimum p-value across categories
    - any_significant_fdr: Whether any category is FDR-significant

    Args:
        coef_df: Per-feature coefficient DataFrame
        cancer_type: Cancer type code

    Returns:
        Grouped importance DataFrame
    """
    grouped = coef_df.groupby('feature_group').agg(
        max_abs_log_odds=('log_odds', lambda x: x.abs().max()),
        sum_abs_log_odds=('log_odds', lambda x: x.abs().sum()),
        mean_abs_log_odds=('log_odds', lambda x: x.abs().mean()),
        max_abs_std_log_odds=('standardized_log_odds', lambda x: x.abs().max()),
        sum_abs_std_log_odds=('standardized_log_odds', lambda x: x.abs().sum()),
        mean_abs_std_log_odds=('standardized_log_odds', lambda x: x.abs().mean()),
        n_categories=('log_odds', 'count'),
        min_p_value=('p_value', 'min'),
        any_significant_fdr=('significant_fdr', 'any'),
        max_odds_ratio=('odds_ratio', 'max'),
        min_odds_ratio=('odds_ratio', 'min'),
    ).reset_index()

    grouped = grouped.sort_values('max_abs_std_log_odds', ascending=False).reset_index(drop=True)
    grouped['rank'] = range(1, len(grouped) + 1)
    grouped['cancer_type'] = cancer_type

    return grouped


def _plot_odds_ratio_forest(
    coef_df: pd.DataFrame,
    cancer_type: str,
    output_path: str,
    max_display: int = 25
):
    """
    Generate odds ratio forest plot for a single cancer type.

    Horizontal bars with CI whiskers, reference line at OR=1,
    FDR significance markers.

    Args:
        coef_df: Per-feature coefficient DataFrame
        cancer_type: Cancer type code
        output_path: Path to save PDF
        max_display: Maximum number of features to display
    """
    # Select top features by absolute log-odds
    plot_df = coef_df.head(max_display).copy()
    plot_df = plot_df.sort_values('odds_ratio', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))

    y_pos = np.arange(len(plot_df))

    # Color by significance
    colors = ['#c0392b' if sig else '#7f8c8d' for sig in plot_df['significant_fdr']]

    # Plot horizontal error bars (CI whiskers)
    ax.errorbar(
        plot_df['odds_ratio'], y_pos,
        xerr=[
            plot_df['odds_ratio'] - plot_df['or_ci_lower'],
            plot_df['or_ci_upper'] - plot_df['odds_ratio']
        ],
        fmt='o', color='black', markersize=5, capsize=3, linewidth=1, zorder=5
    )

    # Color the markers
    for i, (or_val, sig) in enumerate(zip(plot_df['odds_ratio'], plot_df['significant_fdr'])):
        color = '#c0392b' if sig else '#7f8c8d'
        ax.plot(or_val, i, 'o', color=color, markersize=7, zorder=6)

    # Reference line at OR = 1
    ax.axvline(x=1.0, color='navy', linestyle='--', linewidth=1.5, alpha=0.7, label='OR = 1')

    # Truncate feature names for display
    display_names = [
        (name[:40] + '...' if len(name) > 43 else name)
        for name in plot_df['feature_name']
    ]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=8)
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=11)
    ax.set_title(f'{cancer_type} - Logistic Regression Odds Ratios', fontsize=13, fontweight='bold')

    # Use log scale for x-axis if range is large
    or_range = plot_df['or_ci_upper'].max() / max(plot_df['or_ci_lower'].min(), 0.01)
    if or_range > 10:
        ax.set_xscale('log')

    # Add significance markers
    n_sig = plot_df['significant_fdr'].sum()
    ax.text(
        0.98, 0.02,
        f'Red = FDR < 0.05 ({n_sig}/{len(plot_df)})',
        transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
    )

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_combined_outputs(
    results: List[Dict],
    output_dir: Path,
    logger: logging.Logger
):
    """
    Generate combined outputs across all cancer types.

    Args:
        results: List of result dicts from extract_coefficients_single_cancer_type()
        output_dir: Root output directory
        logger: Logger instance
    """
    logger.info("=" * 70)
    logger.info("GENERATING COMBINED OUTPUTS")
    logger.info("=" * 70)

    successful = [r for r in results if r.get('status') == 'success']

    if not successful:
        logger.warning("No successful extractions to combine")
        return

    # Combine per-feature coefficients
    all_coefs = []
    all_grouped = []

    for result in successful:
        cancer_dir = Path(result['output_dir'])

        coef_file = cancer_dir / 'lr_coefficients.csv'
        if coef_file.exists():
            df = pd.read_csv(coef_file)
            all_coefs.append(df)

        grouped_file = cancer_dir / 'lr_grouped_importance.csv'
        if grouped_file.exists():
            df = pd.read_csv(grouped_file)
            all_grouped.append(df)

    if all_coefs:
        combined_coefs = pd.concat(all_coefs, ignore_index=True)
        combined_file = output_dir / 'combined_lr_coefficients.csv'
        combined_coefs.to_csv(combined_file, index=False)
        logger.info(f"  Saved: {combined_file}")

    if all_grouped:
        combined_grouped = pd.concat(all_grouped, ignore_index=True)
        grouped_file = output_dir / 'combined_lr_grouped_importance.csv'
        combined_grouped.to_csv(grouped_file, index=False)
        logger.info(f"  Saved: {grouped_file}")

        # Create OR heatmaps across cancer types
        _plot_or_heatmap(combined_coefs, output_dir, logger)
        _plot_standardized_or_heatmap(combined_coefs, output_dir, logger)

    # Generate summary report
    _generate_summary_report(results, output_dir, logger)


def _plot_or_heatmap(
    combined_coefs: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Generate heatmap of log-odds ratios across cancer types (grouped by parent feature).

    Args:
        combined_coefs: Combined coefficient DataFrame
        output_dir: Output directory
        logger: Logger instance
    """
    # Pivot by feature group and cancer type, using max absolute log-odds
    pivot_data = combined_coefs.groupby(['feature_group', 'cancer_type']).agg(
        max_abs_log_odds=('log_odds', lambda x: x.abs().max())
    ).reset_index()

    pivot = pivot_data.pivot(
        index='feature_group',
        columns='cancer_type',
        values='max_abs_log_odds'
    )

    if pivot.empty:
        logger.warning("No data for heatmap")
        return

    # Sort rows by mean importance
    pivot['mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean', ascending=False).drop(columns='mean')

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.8), max(5, len(pivot) * 0.5)))

    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='Max |log-odds|')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    ax.set_title('LR Feature Importance (Max |log-odds|) Across Cancer Types',
                  fontsize=12, fontweight='bold')

    plt.tight_layout()
    heatmap_file = output_dir / 'lr_or_heatmap.pdf'
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {heatmap_file}")


def _plot_standardized_or_heatmap(
    combined_coefs: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Generate heatmap of standardized log-odds across cancer types (grouped by parent feature).

    Standardized log-odds = log_odds * SD(feature), making coefficients comparable
    across features with different scales (e.g., age in years vs binary indicators).

    Args:
        combined_coefs: Combined coefficient DataFrame (must contain 'standardized_log_odds')
        output_dir: Output directory
        logger: Logger instance
    """
    if 'standardized_log_odds' not in combined_coefs.columns:
        logger.warning("No standardized_log_odds column, skipping standardized heatmap")
        return

    # Pivot by feature group and cancer type, using max absolute standardized log-odds
    pivot_data = combined_coefs.groupby(['feature_group', 'cancer_type']).agg(
        max_abs_std_log_odds=('standardized_log_odds', lambda x: x.abs().max())
    ).reset_index()

    pivot = pivot_data.pivot(
        index='feature_group',
        columns='cancer_type',
        values='max_abs_std_log_odds'
    )

    if pivot.empty:
        logger.warning("No data for standardized heatmap")
        return

    # Sort rows by mean importance
    pivot['mean'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('mean', ascending=False).drop(columns='mean')

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.8), max(5, len(pivot) * 0.5)))

    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='Max |standardized log-odds|')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    ax.set_title('LR Feature Importance (Standardized |log-odds|) Across Cancer Types',
                  fontsize=12, fontweight='bold')

    plt.tight_layout()
    heatmap_file = output_dir / 'standardized_lr_or_heatmap.pdf'
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {heatmap_file}")


def _generate_summary_report(
    results: List[Dict],
    output_dir: Path,
    logger: logging.Logger
):
    """Generate text summary report."""
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'failed']

    lines = [
        "=" * 70,
        "LOGISTIC REGRESSION COEFFICIENT EXTRACTION REPORT",
        "=" * 70,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Cancer types analyzed: {len(successful)}",
        f"Failed: {len(failed)}",
        "",
        "SUMMARY BY CANCER TYPE",
        "-" * 70,
    ]

    for r in sorted(successful, key=lambda x: x['cancer_type']):
        lines.extend([
            f"\n{r['cancer_type']}:",
            f"  Train samples: {r['n_train']}",
            f"  Features: {r['n_features']} (original), {r['n_transformed_features']} (after encoding)",
            f"  Intercept: {r['intercept']:.4f}",
            f"  Significant (FDR < 0.05): {r['n_significant_fdr']}",
            f"  SE reliable: {r['se_reliable']}",
            f"  Top feature (by standardized coef): {r['top_feature']} (OR={r['top_feature_or']:.3f}, std_coef={r['top_feature_std_coef']:.4f})"
        ])

    if failed:
        lines.extend([
            "",
            "FAILED EXTRACTIONS",
            "-" * 70
        ])
        for r in failed:
            lines.append(f"  {r['cancer_type']}: {r.get('error', 'Unknown')}")

    lines.extend([
        "",
        "=" * 70,
        "OUTPUT FILES",
        "=" * 70,
        f"  {output_dir}/",
        "    combined_lr_coefficients.csv",
        "    combined_lr_grouped_importance.csv",
        "    lr_or_heatmap.pdf",
        "    standardized_lr_or_heatmap.pdf",
        "    {CANCER_TYPE}/lr_coefficients.csv",
        "    {CANCER_TYPE}/lr_grouped_importance.csv",
        "    {CANCER_TYPE}/lr_odds_ratio_forest_plot.pdf",
        "    {CANCER_TYPE}/lr_reference_categories.csv",
    ])

    report_file = output_dir / 'summary_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"  Saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract logistic regression coefficients and odds ratios from TIER 5 models'
    )

    parser.add_argument(
        '--cancer-types', nargs='+',
        help='Cancer types to analyze. If not specified, uses all available LR models.'
    )
    parser.add_argument(
        '--model-dir', type=str,
        help='Path to TIER 5 models directory (contains *_logistic_model.joblib files)'
    )
    parser.add_argument(
        '--train-test-split-file', type=str, default=None,
        help='Path to train/test split CSV (default: auto-detect)'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='../../../analysis_output/lr_coefficients',
        help='Output directory for results'
    )
    parser.add_argument(
        '--db-path', type=str, default=None,
        help='Path to SQLite database (default: from BALDER_DB_PATH env var)'
    )
    parser.add_argument(
        '--quick-test', action='store_true',
        help='Quick test mode: analyze LUAD only'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--no-timestamp-subdir', action='store_true',
        help='Write directly to output-dir without creating timestamp subdirectory (for pipeline integration)'
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent.parent

    # Output directory
    if args.output_dir.startswith('/'):
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / args.output_dir

    # Add timestamp subdirectory unless --no-timestamp-subdir is set
    if not args.no_timestamp_subdir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, args.verbose)

    logger.info("=" * 70)
    logger.info("LOGISTIC REGRESSION COEFFICIENT EXTRACTION")
    logger.info("=" * 70)

    # Database path
    db_path = args.db_path or get_default_db_path()
    logger.info(f"Database: {db_path}")

    # Find train/test split file
    if args.train_test_split_file:
        split_file = Path(args.train_test_split_file)
    else:
        possible_paths = [
            repo_root / 'analysis_output' / 'train_test_split' / 'train_test_split.csv',
            script_dir.parent.parent.parent / 'analysis_output' / 'train_test_split' / 'train_test_split.csv'
        ]
        split_file = None
        for p in possible_paths:
            if p.exists():
                split_file = p
                break

        if split_file is None:
            logger.error("Could not find train_test_split.csv. Specify with --train-test-split-file")
            sys.exit(1)

    logger.info(f"Train/test split: {split_file}")

    # Find model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        possible_dirs = [
            repo_root / 'ml_experiments' / 'final_models',
            repo_root / 'runs'
        ]

        model_dir = None
        for base_dir in possible_dirs:
            if not base_dir.exists():
                continue

            for run_dir in sorted(base_dir.glob('run_*'), reverse=True):
                models_subdir = run_dir / 'models'
                if models_subdir.exists() and list(models_subdir.glob('*_logistic_model.joblib')):
                    model_dir = run_dir
                    break
                if list(run_dir.glob('*_logistic_model.joblib')):
                    model_dir = run_dir
                    break

            if model_dir:
                break

        if model_dir is None:
            logger.error("Could not find LR model directory. Specify with --model-dir")
            sys.exit(1)

    logger.info(f"Model directory: {model_dir}")

    # Find model files
    cancer_types = args.cancer_types
    if args.quick_test:
        cancer_types = ['LUAD']
        logger.info("Quick test mode: analyzing LUAD only")

    model_files = find_model_files(model_dir, cancer_types)

    if not model_files:
        logger.error(f"No logistic model files found in {model_dir}")
        sys.exit(1)

    logger.info(f"Found {len(model_files)} LR model files")
    for ct, mf in sorted(model_files.items()):
        logger.info(f"  {ct}: {mf.name}")

    # Load train/test split
    logger.info("Loading train/test split...")
    split_df = pd.read_csv(split_file)
    train_patient_ids = set(split_df[split_df['split'] == 'train']['PATIENT_ID'])
    logger.info(f"  Train patients: {len(train_patient_ids)}")

    # Load full dataset
    logger.info("Loading data from database...")
    loader = DataLoader(db_path)
    full_data = loader.load_patient_data()
    logger.info(f"  Total patients: {len(full_data)}")

    # Process each cancer type
    results = []

    for cancer_type, model_file in sorted(model_files.items()):
        # Filter data for this cancer type
        cancer_data = loader.filter_for_cancer_type(full_data, cancer_type)

        if len(cancer_data) == 0:
            logger.warning(f"No data for {cancer_type}, skipping")
            continue

        # Get training data only
        train_data = cancer_data[cancer_data['PATIENT_ID'].isin(train_patient_ids)]

        if len(train_data) == 0:
            logger.warning(f"No training data for {cancer_type}, skipping")
            continue

        result = extract_coefficients_single_cancer_type(
            cancer_type=cancer_type,
            model_file=model_file,
            train_data=train_data,
            output_dir=output_dir,
            logger=logger
        )

        if result:
            results.append(result)

    # Generate combined outputs
    generate_combined_outputs(results, output_dir, logger)

    # Save results JSON
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Final summary
    successful = sum(1 for r in results if r.get('status') == 'success')
    failed = len(results) - successful

    logger.info("")
    logger.info("=" * 70)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Output directory: {output_dir}")

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
