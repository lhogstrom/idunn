#!/usr/bin/env python3
"""
Compute SHAP Primary Feature Values

Entry point script for TIER 6a SHAP analysis. Calculates primary SHAP values
(mean absolute SHAP per feature) to determine feature importance rankings.

This script:
- Loads pre-trained models from TIER 5
- Uses test data (20% held-out) for SHAP calculations
- Generates feature importance rankings
- Creates beeswarm summary plots
- Outputs combined cross-cancer-type summaries

Prerequisites:
- TIER 5 must be completed (trained models available)
- Train/test split file must exist

Usage:
    # Quick test (LUAD only)
    python compute_shap_primary_values.py --quick-test

    # Full run (all cancer types from TIER 5)
    python compute_shap_primary_values.py --model-dir ../../ml_experiments/final_models/run_YYYYMMDD_HHMMSS

    # Custom cancer types
    python compute_shap_primary_values.py --cancer-types LUAD BRCA MEL --model-dir /path/to/models

Author: Balder Analysis Team
Date: January 2026
"""

import argparse
import logging
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import joblib

# Add tools directory to path for db_config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path

# Add modeling directory to path for DataLoader
sys.path.append(str(Path(__file__).parent.parent.parent / 'modeling'))
from oncokb_analyzer.data_loader import DataLoader

# Local imports
from shap_primary_analyzer import SHAPPrimaryAnalyzer, create_analyzer_from_tier5_model
from feature_name_mapper import create_mapper_from_data_loader


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to both file and console."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create logger
    logger = logging.getLogger('shap_primary')
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers

    # File handler
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    log_file = output_dir / 'shap_primary_analysis.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (shorter format)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def find_model_files(model_dir: Path, cancer_types: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    Find TIER 5 model files in directory.

    Args:
        model_dir: Path to TIER 5 models directory
        cancer_types: Optional list of cancer types to filter

    Returns:
        Dict mapping cancer type to model file path
    """
    model_files = {}

    # Look for model files in both root and models/ subdirectory
    search_dirs = [model_dir, model_dir / 'models']

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for f in search_dir.glob('*_random_forest_model.joblib'):
            # Extract cancer type from filename
            cancer_type = f.stem.replace('_random_forest_model', '')

            if cancer_types is None or cancer_type in cancer_types:
                model_files[cancer_type] = f

    return model_files


def analyze_cancer_type(
    cancer_type: str,
    model_file: Path,
    test_data: pd.DataFrame,
    train_data: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger
) -> Optional[Dict]:
    """
    Analyze single cancer type with SHAP primary values.

    Args:
        cancer_type: Cancer type code
        model_file: Path to TIER 5 model file
        test_data: Test set data for this cancer type
        train_data: Training set data (for explainer background)
        output_dir: Output directory for this cancer type
        logger: Logger instance

    Returns:
        Result dict with metrics and status, or None if failed
    """
    logger.info("=" * 70)
    logger.info(f"ANALYZING {cancer_type}")
    logger.info("=" * 70)

    try:
        # Load model
        logger.info(f"Loading model from {model_file}")
        model_dict = joblib.load(model_file)

        model = model_dict['model']
        preprocessor = model_dict['preprocessor']
        feature_columns = model_dict['feature_columns']

        logger.info(f"  Model type: {type(model).__name__}")
        logger.info(f"  Features: {len(feature_columns)}")
        logger.info(f"  Test samples: {len(test_data)}")
        logger.info(f"  Train samples for background: {len(train_data)}")

        # Create analyzer
        analyzer = SHAPPrimaryAnalyzer(
            model=model,
            preprocessor=preprocessor,
            feature_columns=feature_columns,
            background_sample_size=500
        )

        # Calculate SHAP values
        shap_values, expected_value = analyzer.calculate_shap_values(
            X_test=test_data,
            X_train=train_data
        )

        # Get feature importance rankings
        importance_df = analyzer.get_feature_importance(shap_values)

        # Get grouped importance (combine one-hot encoded columns)
        grouped_df = analyzer.get_grouped_feature_importance(shap_values)

        # Create output directory for this cancer type
        cancer_output_dir = output_dir / cancer_type
        cancer_output_dir.mkdir(parents=True, exist_ok=True)

        # Save feature importance CSV
        importance_file = cancer_output_dir / f'{cancer_type}_feature_importance.csv'
        importance_df.to_csv(importance_file, index=False)
        logger.info(f"  Saved: {importance_file}")

        # Save grouped importance CSV
        grouped_file = cancer_output_dir / f'{cancer_type}_grouped_feature_importance.csv'
        grouped_df.to_csv(grouped_file, index=False)
        logger.info(f"  Saved: {grouped_file}")

        # Generate beeswarm plot
        beeswarm_file = cancer_output_dir / f'{cancer_type}_shap_summary.pdf'
        analyzer.plot_summary(
            shap_values=shap_values,
            X_test=test_data,
            output_path=str(beeswarm_file),
            max_display=20,
            plot_type='beeswarm'
        )

        # Generate bar chart
        bar_file = cancer_output_dir / f'{cancer_type}_feature_importance_bar.pdf'
        analyzer.plot_feature_importance_bar(
            importance_df=importance_df,
            output_path=str(bar_file),
            top_n=20,
            title=f'{cancer_type} Feature Importance (SHAP)'
        )

        # Prepare result
        result = {
            'cancer_type': cancer_type,
            'n_test_samples': len(test_data),
            'n_train_samples': len(train_data),
            'n_features': len(feature_columns),
            'n_transformed_features': shap_values.shape[1],
            'expected_value': float(expected_value),
            'top_feature': importance_df.iloc[0]['feature_name'],
            'top_feature_shap': float(importance_df.iloc[0]['mean_abs_shap']),
            'status': 'success',
            'output_dir': str(cancer_output_dir)
        }

        logger.info(f"  Analysis complete for {cancer_type}")
        logger.info(f"  Top feature: {result['top_feature']} (mean |SHAP|={result['top_feature_shap']:.4f})")

        return result

    except Exception as e:
        logger.error(f"Analysis failed for {cancer_type}: {str(e)}")
        logger.exception("Full traceback:")
        return {
            'cancer_type': cancer_type,
            'status': 'failed',
            'error': str(e)
        }


def generate_combined_outputs(
    results: List[Dict],
    output_dir: Path,
    logger: logging.Logger
):
    """
    Generate combined outputs across all cancer types.

    Args:
        results: List of result dicts from analyze_cancer_type()
        output_dir: Root output directory
        logger: Logger instance
    """
    logger.info("=" * 70)
    logger.info("GENERATING COMBINED OUTPUTS")
    logger.info("=" * 70)

    # Filter successful results
    successful = [r for r in results if r.get('status') == 'success']

    if not successful:
        logger.warning("No successful analyses to combine")
        return

    # Load and combine feature importance from all cancer types
    all_importance = []

    for result in successful:
        cancer_type = result['cancer_type']
        cancer_dir = Path(result['output_dir'])

        importance_file = cancer_dir / f'{cancer_type}_feature_importance.csv'
        if importance_file.exists():
            df = pd.read_csv(importance_file)
            df['cancer_type'] = cancer_type
            all_importance.append(df)

    if all_importance:
        combined_df = pd.concat(all_importance, ignore_index=True)

        # Save combined importance
        combined_file = output_dir / 'combined_feature_importance.csv'
        combined_df.to_csv(combined_file, index=False)
        logger.info(f"  Saved combined importance: {combined_file}")

        # Create pivot table of ranks
        rank_pivot = combined_df.pivot_table(
            index='feature_name',
            columns='cancer_type',
            values='rank',
            aggfunc='first'
        )

        # Sort by mean rank across cancer types
        rank_pivot['mean_rank'] = rank_pivot.mean(axis=1)
        rank_pivot = rank_pivot.sort_values('mean_rank')

        rank_file = output_dir / 'feature_rank_matrix.csv'
        rank_pivot.to_csv(rank_file)
        logger.info(f"  Saved rank matrix: {rank_file}")

    # Generate summary report
    report_lines = [
        "=" * 70,
        "SHAP PRIMARY FEATURE VALUES ANALYSIS REPORT",
        "=" * 70,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Cancer types analyzed: {len(successful)}",
        "",
        "SUMMARY BY CANCER TYPE",
        "-" * 70
    ]

    for r in sorted(successful, key=lambda x: x['cancer_type']):
        report_lines.extend([
            f"\n{r['cancer_type']}:",
            f"  Test samples: {r['n_test_samples']}",
            f"  Features: {r['n_features']} (original), {r['n_transformed_features']} (after encoding)",
            f"  Expected value: {r['expected_value']:.4f}",
            f"  Top feature: {r['top_feature']} (mean |SHAP|={r['top_feature_shap']:.4f})"
        ])

    report_lines.extend([
        "",
        "-" * 70,
        "ANALYSIS STATUS",
        "-" * 70,
        f"  Successful: {len(successful)}",
        f"  Failed: {len(results) - len(successful)}"
    ])

    failed = [r for r in results if r.get('status') == 'failed']
    if failed:
        report_lines.append("\nFailed analyses:")
        for r in failed:
            report_lines.append(f"  {r['cancer_type']}: {r.get('error', 'Unknown error')}")

    report_lines.extend([
        "",
        "=" * 70,
        "OUTPUT FILES",
        "=" * 70,
        f"  {output_dir}/",
        "    combined_feature_importance.csv - All cancer types combined",
        "    feature_rank_matrix.csv - Feature ranks across cancer types",
        "    {CANCER_TYPE}/",
        "      {CANCER_TYPE}_feature_importance.csv - Per-feature SHAP rankings",
        "      {CANCER_TYPE}_grouped_feature_importance.csv - Grouped by original feature",
        "      {CANCER_TYPE}_shap_summary.pdf - Beeswarm plot",
        "      {CANCER_TYPE}_feature_importance_bar.pdf - Bar chart"
    ])

    report_file = output_dir / 'summary_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"  Saved summary report: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute SHAP primary feature values for actionability prediction'
    )

    parser.add_argument(
        '--cancer-types', nargs='+',
        help='Cancer types to analyze. If not specified, uses all available in model directory'
    )

    parser.add_argument(
        '--model-dir', type=str,
        help='Path to TIER 5 models directory (contains *.joblib files)'
    )

    parser.add_argument(
        '--train-test-split-file', type=str,
        default=None,
        help='Path to train/test split CSV (default: auto-detect from analysis_output)'
    )

    parser.add_argument(
        '--output-dir', type=str,
        default='../../../analysis_output/shap_primary_values',
        help='Output directory for results'
    )

    parser.add_argument(
        '--db-path', type=str,
        default=None,
        help='Path to SQLite database (default: from BALDER_DB_PATH env var)'
    )

    parser.add_argument(
        '--background-sample-size', type=int, default=500,
        help='Number of background samples for TreeExplainer (default: 500)'
    )

    parser.add_argument(
        '--quick-test', action='store_true',
        help='Quick test mode: analyze LUAD only'
    )

    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
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

    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, args.verbose)

    logger.info("=" * 70)
    logger.info("SHAP PRIMARY FEATURE VALUES ANALYSIS")
    logger.info("=" * 70)

    # Database path
    db_path = args.db_path or get_default_db_path()
    logger.info(f"Database: {db_path}")

    # Find train/test split file
    if args.train_test_split_file:
        split_file = Path(args.train_test_split_file)
    else:
        # Auto-detect from common locations
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
            logger.error("Could not find train_test_split.csv. Please specify with --train-test-split-file")
            sys.exit(1)

    logger.info(f"Train/test split: {split_file}")

    # Find model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        # Auto-detect: look for most recent TIER 5 run
        possible_dirs = [
            repo_root / 'ml_experiments' / 'final_models',
            repo_root / 'runs'
        ]

        model_dir = None
        for base_dir in possible_dirs:
            if not base_dir.exists():
                continue

            # Find most recent run with models
            for run_dir in sorted(base_dir.glob('run_*'), reverse=True):
                models_subdir = run_dir / 'models'
                if models_subdir.exists() and list(models_subdir.glob('*.joblib')):
                    model_dir = run_dir
                    break
                # Also check for joblib files directly in run_dir
                if list(run_dir.glob('*.joblib')):
                    model_dir = run_dir
                    break

            if model_dir:
                break

        if model_dir is None:
            logger.error("Could not find TIER 5 model directory. Please specify with --model-dir")
            sys.exit(1)

    logger.info(f"Model directory: {model_dir}")

    # Find model files
    cancer_types = args.cancer_types
    if args.quick_test:
        cancer_types = ['LUAD']
        logger.info("Quick test mode: analyzing LUAD only")

    model_files = find_model_files(model_dir, cancer_types)

    if not model_files:
        logger.error(f"No model files found in {model_dir}")
        sys.exit(1)

    logger.info(f"Found {len(model_files)} model files")
    for ct, mf in sorted(model_files.items()):
        logger.info(f"  {ct}: {mf.name}")

    # Load train/test split
    logger.info("Loading train/test split...")
    split_df = pd.read_csv(split_file)
    train_patient_ids = set(split_df[split_df['split'] == 'train']['PATIENT_ID'])
    test_patient_ids = set(split_df[split_df['split'] == 'test']['PATIENT_ID'])
    logger.info(f"  Train patients: {len(train_patient_ids)}")
    logger.info(f"  Test patients: {len(test_patient_ids)}")

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

        # Split into train/test
        train_data = cancer_data[cancer_data['PATIENT_ID'].isin(train_patient_ids)]
        test_data = cancer_data[cancer_data['PATIENT_ID'].isin(test_patient_ids)]

        if len(test_data) == 0:
            logger.warning(f"No test data for {cancer_type}, skipping")
            continue

        if len(train_data) == 0:
            logger.warning(f"No training data for {cancer_type}, skipping")
            continue

        # Analyze
        result = analyze_cancer_type(
            cancer_type=cancer_type,
            model_file=model_file,
            test_data=test_data,
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
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Output directory: {output_dir}")

    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
