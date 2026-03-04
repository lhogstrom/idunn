"""
Validation Script: Compare Standard GroupKFold vs Stratified GroupKFold

Tests the new StratifiedGroupKFold implementation on real cancer data.
Compares fold imbalance metrics and model performance between the two approaches.

"""

import sys
import os

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.model_selection import GroupKFold

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oncokb_analyzer.config import Config
from oncokb_analyzer.data_loader import DataLoader
from oncokb_analyzer.trainer import ModelTrainer
from oncokb_analyzer.stratified_group_kfold import (
    StratifiedGroupKFold,
    calculate_fold_imbalance_metrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_cancer_type(cancer_type: str, config: Config, output_dir: str):
    """
    Validate StratifiedGroupKFold vs standard GroupKFold for a single cancer type.

    Parameters:
    ----------
    cancer_type : str
        OncoTree cancer type code
    config : Config
        Configuration object
    output_dir : str
        Output directory for results
    """
    logger.info("="*80)
    logger.info(f"VALIDATING: {cancer_type}")
    logger.info("="*80)

    # Load data
    loader = DataLoader(config.db_path)
    data = loader.load_cancer_type_data(cancer_type)

    if data is None or len(data) < 1000:
        logger.warning(f"Skipping {cancer_type}: insufficient data ({len(data) if data is not None else 0} samples)")
        return None

    logger.info(f"Loaded {len(data)} samples for {cancer_type}")

    # Prepare features (demographic only)
    trainer_standard = ModelTrainer(
        algorithm="random_forest",
        cv_strategy="group",
        group_column="SUBJECT_CENTER",
        random_state=42
    )

    X, y, groups = trainer_standard.prepare_features(
        data,
        outcome_col='isOncoKbLevel1',
        feature_set='demographic'
    )

    logger.info(f"Features: {X.shape[1]} features, {len(X)} samples")
    logger.info(f"Groups: {groups.nunique()} unique institutions")
    logger.info(f"Class balance: {y.value_counts().to_dict()}")

    # Calculate fold imbalance metrics for both strategies
    logger.info("\n" + "="*80)
    logger.info("STANDARD GROUPKFOLD")
    logger.info("="*80)

    standard_gkf = GroupKFold(n_splits=5)
    standard_metrics = calculate_fold_imbalance_metrics(X, y, groups, standard_gkf)

    logger.info(f"Fold sizes: {standard_metrics['fold_sizes']}")
    logger.info(f"Fold group counts: {standard_metrics['fold_group_counts']}")
    logger.info(f"Patients CV: {standard_metrics['patients_cv']:.3f}")
    logger.info(f"Groups CV: {standard_metrics['groups_cv']:.3f}")
    logger.info(f"Size ratio: {standard_metrics['size_ratio']:.2f}×")
    logger.info(f"Actionable rates per fold: {[f'{r:.3f}' for r in standard_metrics['actionable_rates']]}")

    logger.info("\n" + "="*80)
    logger.info("STRATIFIED GROUPKFOLD")
    logger.info("="*80)

    stratified_gkf = StratifiedGroupKFold(n_splits=5, random_state=42)
    stratified_metrics = calculate_fold_imbalance_metrics(X, y, groups, stratified_gkf)

    logger.info(f"Fold sizes: {stratified_metrics['fold_sizes']}")
    logger.info(f"Fold group counts: {stratified_metrics['fold_group_counts']}")
    logger.info(f"Patients CV: {stratified_metrics['patients_cv']:.3f}")
    logger.info(f"Groups CV: {stratified_metrics['groups_cv']:.3f}")
    logger.info(f"Size ratio: {stratified_metrics['size_ratio']:.2f}×")
    logger.info(f"Actionable rates per fold: {[f'{r:.3f}' for r in stratified_metrics['actionable_rates']]}")

    # Calculate improvement
    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT")
    logger.info("="*80)

    cv_improvement = (standard_metrics['patients_cv'] - stratified_metrics['patients_cv']) / standard_metrics['patients_cv'] * 100
    ratio_improvement = (standard_metrics['size_ratio'] - stratified_metrics['size_ratio']) / standard_metrics['size_ratio'] * 100

    logger.info(f"Patients CV: {standard_metrics['patients_cv']:.3f} → {stratified_metrics['patients_cv']:.3f} ({cv_improvement:+.1f}%)")
    logger.info(f"Size ratio: {standard_metrics['size_ratio']:.2f}× → {stratified_metrics['size_ratio']:.2f}× ({ratio_improvement:+.1f}%)")

    # Train models and compare performance
    logger.info("\n" + "="*80)
    logger.info("MODEL PERFORMANCE COMPARISON")
    logger.info("="*80)

    # Standard GroupKFold
    trainer_standard = ModelTrainer(
        algorithm="random_forest",
        cv_strategy="group",
        group_column="SUBJECT_CENTER",
        random_state=42
    )

    logger.info("\nTraining with standard GroupKFold...")
    results_standard = trainer_standard.train_and_evaluate(X, y, groups)

    # Stratified GroupKFold
    trainer_stratified = ModelTrainer(
        algorithm="random_forest",
        cv_strategy="stratified_group",
        group_column="SUBJECT_CENTER",
        random_state=42
    )

    logger.info("\nTraining with StratifiedGroupKFold...")
    results_stratified = trainer_stratified.train_and_evaluate(X, y, groups)

    # Compare results
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"Standard GroupKFold CV AUC:    {results_standard['cv_auc_mean']:.4f} ± {results_standard['cv_auc_std']:.4f}")
    logger.info(f"Stratified GroupKFold CV AUC:  {results_stratified['cv_auc_mean']:.4f} ± {results_stratified['cv_auc_std']:.4f}")
    logger.info(f"Difference:                     {results_stratified['cv_auc_mean'] - results_standard['cv_auc_mean']:.4f}")

    # Compile results
    result_dict = {
        'cancer_type': cancer_type,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_groups': groups.nunique(),
        'actionable_rate': y.mean(),
        # Standard GroupKFold metrics
        'standard_patients_cv': standard_metrics['patients_cv'],
        'standard_groups_cv': standard_metrics['groups_cv'],
        'standard_size_ratio': standard_metrics['size_ratio'],
        'standard_cv_auc_mean': results_standard['cv_auc_mean'],
        'standard_cv_auc_std': results_standard['cv_auc_std'],
        # Stratified GroupKFold metrics
        'stratified_patients_cv': stratified_metrics['patients_cv'],
        'stratified_groups_cv': stratified_metrics['groups_cv'],
        'stratified_size_ratio': stratified_metrics['size_ratio'],
        'stratified_cv_auc_mean': results_stratified['cv_auc_mean'],
        'stratified_cv_auc_std': results_stratified['cv_auc_std'],
        # Improvement metrics
        'cv_improvement_pct': cv_improvement,
        'ratio_improvement_pct': ratio_improvement,
        'auc_difference': results_stratified['cv_auc_mean'] - results_standard['cv_auc_mean']
    }

    return result_dict


def main():
    """Run validation on multiple cancer types."""
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"../../analysis_output/stratified_groupkfold_validation/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging to file
    file_handler = logging.FileHandler(f"{output_dir}/validation.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logger.info("="*80)
    logger.info("STRATIFIED GROUPKFOLD VALIDATION")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")

    # Database path (check multiple possible locations)
    db_path = get_default_db_path()
    if not os.path.exists(db_path):
        db_path = get_default_db_path()

    if not os.path.exists(db_path):
        logger.error(f"Database not found at: {db_path}")
        return

    logger.info(f"Database: {db_path}")

    # Cancer types to validate (known to have fold imbalance)
    cancer_types = ['BRCA', 'PRAD', 'LUAD']
    logger.info(f"Cancer types: {cancer_types}")

    # Create config
    config = Config(
        db_path=db_path,
        cancer_types=cancer_types
    )

    # Run validation for each cancer type
    results = []
    for cancer_type in cancer_types:
        try:
            result = validate_cancer_type(cancer_type, config, output_dir)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.error(f"Error validating {cancer_type}: {e}", exc_info=True)
            continue

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{output_dir}/validation_results.csv", index=False)
        logger.info(f"\nResults saved to: {output_dir}/validation_results.csv")

        # Print summary table
        logger.info("\n" + "="*80)
        logger.info("SUMMARY TABLE")
        logger.info("="*80)

        summary_cols = [
            'cancer_type',
            'n_samples',
            'n_groups',
            'standard_patients_cv',
            'stratified_patients_cv',
            'cv_improvement_pct',
            'standard_size_ratio',
            'stratified_size_ratio',
            'ratio_improvement_pct',
            'standard_cv_auc_mean',
            'stratified_cv_auc_mean',
            'auc_difference'
        ]
        print(results_df[summary_cols].to_string(index=False))

        # Generate comparison report
        generate_report(results_df, output_dir)

    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*80)


def generate_report(results_df: pd.DataFrame, output_dir: str):
    """Generate markdown report summarizing validation results."""
    report_path = f"{output_dir}/validation_report.md"

    with open(report_path, 'w') as f:
        f.write("# Stratified GroupKFold Validation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Objective\n\n")
        f.write("Validate that StratifiedGroupKFold improves fold balance compared to standard GroupKFold ")
        f.write("without compromising model performance.\n\n")

        f.write("## Cancer Types Tested\n\n")
        for _, row in results_df.iterrows():
            f.write(f"- **{row['cancer_type']}**: {row['n_samples']:,} samples, ")
            f.write(f"{row['n_groups']} institutions, ")
            f.write(f"{row['actionable_rate']:.1%} actionable\n")

        f.write("\n## Fold Balance Improvement\n\n")
        f.write("| Cancer Type | Standard CV | Stratified CV | Improvement |\n")
        f.write("|-------------|-------------|---------------|-------------|\n")
        for _, row in results_df.iterrows():
            f.write(f"| {row['cancer_type']} | ")
            f.write(f"{row['standard_patients_cv']:.3f} | ")
            f.write(f"{row['stratified_patients_cv']:.3f} | ")
            f.write(f"{row['cv_improvement_pct']:+.1f}% |\n")

        f.write("\n## Size Ratio Improvement\n\n")
        f.write("| Cancer Type | Standard Ratio | Stratified Ratio | Improvement |\n")
        f.write("|-------------|----------------|------------------|-------------|\n")
        for _, row in results_df.iterrows():
            f.write(f"| {row['cancer_type']} | ")
            f.write(f"{row['standard_size_ratio']:.2f}× | ")
            f.write(f"{row['stratified_size_ratio']:.2f}× | ")
            f.write(f"{row['ratio_improvement_pct']:+.1f}% |\n")

        f.write("\n## Model Performance\n\n")
        f.write("| Cancer Type | Standard AUC | Stratified AUC | Difference |\n")
        f.write("|-------------|--------------|----------------|------------|\n")
        for _, row in results_df.iterrows():
            f.write(f"| {row['cancer_type']} | ")
            f.write(f"{row['standard_cv_auc_mean']:.4f} ± {row['standard_cv_auc_std']:.4f} | ")
            f.write(f"{row['stratified_cv_auc_mean']:.4f} ± {row['stratified_cv_auc_std']:.4f} | ")
            f.write(f"{row['auc_difference']:+.4f} |\n")

        f.write("\n## Key Findings\n\n")

        # Calculate average improvements
        avg_cv_improvement = results_df['cv_improvement_pct'].mean()
        avg_ratio_improvement = results_df['ratio_improvement_pct'].mean()
        avg_auc_difference = results_df['auc_difference'].mean()

        f.write(f"1. **Average fold balance improvement**: {avg_cv_improvement:+.1f}% (Patients CV)\n")
        f.write(f"2. **Average size ratio improvement**: {avg_ratio_improvement:+.1f}%\n")
        f.write(f"3. **Average AUC difference**: {avg_auc_difference:+.4f}\n\n")

        # Determine if stratified is better
        if avg_cv_improvement > 0:
            f.write("✅ **CONCLUSION**: StratifiedGroupKFold improves fold balance ")
        else:
            f.write("⚠️ **CONCLUSION**: StratifiedGroupKFold does NOT improve fold balance ")

        if abs(avg_auc_difference) < 0.01:
            f.write("without meaningful impact on model performance.\n\n")
        elif avg_auc_difference > 0:
            f.write("AND improves model performance.\n\n")
        else:
            f.write("but may slightly reduce model performance.\n\n")

        f.write("## Recommendation\n\n")
        if avg_cv_improvement > 10 and abs(avg_auc_difference) < 0.02:
            f.write("**ADOPT StratifiedGroupKFold** as the default CV strategy for cancer types with ")
            f.write("severe fold imbalance (CV > 0.5).\n")
        else:
            f.write("Continue investigating. Consider cancer-type-specific CV strategy selection.\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == '__main__':
    main()
