"""
Phase 2b Step 1: Universal Hyperparameter Optimization

This script implements Phase 2b Step 1, which optimizes hyperparameters using a
universal approach (pooled data from all 21 cancer types) for three algorithms:
1. Random Forest
2. Logistic Regression
3. XGBoost (NEW)

Based on Phase 2a findings:
- Universal hyperparameters perform equivalently to cancer-specific (Δ AUC = -0.0094)
- Parameter stability across cancer types = 1.000
- Therefore, use single universal parameter set for all cancer types

Objective:
- Pool all 21 eligible cancer types (~83,000 patients, ~30,000 actionable)
- Run RandomizedSearchCV with 100 iterations for each algorithm
- Use nested CV (5 outer × 4 inner folds) to prevent data leakage
- Identify optimal hyperparameters for each algorithm

Output:
- Optimal hyperparameters for RF, LR, and XGBoost
- Nested CV performance estimates
- Comparison of three algorithms on pooled data
- JSON results for reproducibility

Cancer Types (21 eligible types, ≥250 actionable events):
LUAD, IDC, BRCA, MEL, GIST, COAD, THPA, ILC, PRAD, AML,
BLCA, SKCM, NSCLC, ODG, HGSOC, ASTR, MDLC, BRCNOS, BRCANOS, COADREAD, IHCH

Usage Examples:
    # Quick test with limited parallelism (recommended for testing)
    python run_ml_model_selection_full.py --algorithm logistic --quick-test --n-jobs 2

    # Full run with memory-safe configuration
    python run_ml_model_selection_full.py --n-jobs 2

    # Use 50% of data to reduce memory (recommended for systems with <16 GB RAM)
    python run_ml_model_selection_full.py --sample-fraction 0.5 --n-jobs 1

    # Use demographic features only (low memory)
    python run_ml_model_selection_full.py --feature-set demographic

    # Full run with all cores (REQUIRES 100+ GB RAM with feature_set='both')
    python run_ml_model_selection_full.py

IMPORTANT - Memory Usage:
    Default configuration (--feature-set both --n-jobs -1) can consume 50-100+ GB RAM
    due to one-hot encoding of high-cardinality features (SEQ_ASSAY_ID, SUBJECT_CENTER).

    Memory reduction strategies:
    - Use --sample-fraction 0.5 to use 50% of data (~42k samples instead of ~85k)
    - Use --n-jobs 1 or --n-jobs 2 for sequential/limited parallel processing
    - Use --feature-set demographic (6 features instead of 24)

"""

import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

from oncokb_analyzer.data_loader import DataLoader
from db_config import get_default_db_path
from oncokb_analyzer.hyperparameter_optimizer import NestedCVOptimizer
from oncokb_analyzer.param_spaces import (
    get_param_distributions,
    get_param_distributions_quick,
    count_grid_combinations
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase2bUniversal:
    """
    Phase 2b Step 1: Universal hyperparameter optimization for 3 algorithms.
    """

    def __init__(self, db_path, output_dir='../../ml_experiments/phase2b_universal', quick_mode=False, cv_strategy='group', n_jobs=-1, feature_set='both', sample_fraction=1.0, train_test_split_file=None, n_iter=None, institution_weight_alpha=0.0, institution_column='SUBJECT_CENTER'):
        """
        Initialize Phase 2b universal optimization.

        Parameters
        ----------
        db_path : str
            Path to SQLite database
        output_dir : str
            Directory for output files
        quick_mode : bool
            If True, use reduced search space and fewer iterations for testing
        cv_strategy : str
            CV strategy: 'standard', 'group', or 'stratified_group' (default: 'group')
        n_jobs : int
            Number of parallel jobs for RandomizedSearchCV (-1 uses all cores, 1 for sequential)
            IMPORTANT: Use n_jobs=1 or 2 with feature_set='both' to avoid memory issues
        feature_set : str
            Feature set to use: 'demographic', 'technical', or 'both' (default: 'both')
        sample_fraction : float
            Fraction of data to use for optimization (0.0 to 1.0, default: 1.0)
            Use < 1.0 to reduce memory usage (e.g., 0.5 for 50% of data)
        train_test_split_file : str, optional
            Path to train/test split CSV file. If provided, only training data will be used.
            File should have columns: PATIENT_ID, split (with 'train'/'test' values)
        n_iter : int, optional
            Number of RandomizedSearchCV iterations. Overrides the default (10 for quick, 200 for full).
        institution_weight_alpha : float
            Power parameter for institutional sample weighting (0.0 to 1.0).
            0.0 = no weighting (default), 0.5 = moderate correction, 1.0 = full inverse frequency.
        institution_column : str
            Column name for institution grouping (default: 'SUBJECT_CENTER')
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.quick_mode = quick_mode
        self.cv_strategy = cv_strategy
        self.n_jobs = n_jobs
        self.feature_set = feature_set
        self.sample_fraction = sample_fraction
        self.train_test_split_file = train_test_split_file
        self.institution_weight_alpha = institution_weight_alpha
        self.institution_column = institution_column
        self.train_patient_ids = None

        # Load train/test split if provided
        if train_test_split_file is not None:
            split_path = Path(train_test_split_file)
            if split_path.exists():
                split_df = pd.read_csv(split_path)
                self.train_patient_ids = split_df[split_df['split'] == 'train']['PATIENT_ID'].tolist()
                logger.info(f"Loaded train/test split: {len(self.train_patient_ids):,} training patients")
            else:
                logger.warning(f"Train/test split file not found: {train_test_split_file}")
                logger.warning("Using ALL data (no train/test split filtering)")

        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = self.output_dir / f'run_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Phase 2b Universal Optimization initialized")
        logger.info(f"Database: {db_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Quick mode: {quick_mode}")
        logger.info(f"Feature set: {feature_set}")
        logger.info(f"Parallel jobs (n_jobs): {n_jobs}")
        logger.info(f"Sample fraction: {sample_fraction:.1%}")
        if self.institution_weight_alpha > 0:
            logger.info(f"Institution weighting: alpha={institution_weight_alpha} on '{institution_column}'")
        if self.train_patient_ids:
            logger.info(f"Train/test split: TRAINING DATA ONLY ({len(self.train_patient_ids):,} patients)")

        # Memory warning for high-memory configurations
        if feature_set == 'both' and n_jobs == -1 and sample_fraction >= 0.9:
            logger.warning("⚠️  WARNING: Using feature_set='both' with n_jobs=-1 (all cores)")
            logger.warning("   This can consume 50-100+ GB of RAM due to one-hot encoding of")
            logger.warning("   high-cardinality features (SEQ_ASSAY_ID, SUBJECT_CENTER, etc.)")
            logger.warning("   If you experience memory issues, use: --n-jobs 1 or --n-jobs 2")
            logger.warning("   OR: Use --feature-set demographic (6 features vs 24)")
            logger.warning("   OR: Use --sample-fraction 0.5 (use 50% of data)")

        # Phase 2b configuration - ALL 21 eligible cancer types
        self.cancer_types = [
            'LUAD', 'IDC', 'BRCA', 'MEL', 'GIST', 'COAD', 'THPA', 'ILC', 'PRAD', 'AML',
            'BLCA', 'SKCM', 'NSCLC', 'ODG', 'HGSOC', 'ASTR', 'MDLC', 'BRCNOS',
            'BRCANOS', 'COADREAD', 'IHCH'
        ]

        # Three algorithms: RF, LR, and XGBoost (NEW)
        self.algorithms = ['random_forest', 'logistic', 'xgboost']

        # CV configuration
        if n_iter is not None:
            self.n_iter = n_iter
        elif quick_mode:
            self.n_iter = 10  # Reduced for testing
        else:
            self.n_iter = 200  # Full Phase 2b search (increased from 100 for thorough exploration)

        logger.info(f"Cancer types: {len(self.cancer_types)} types (pooled)")
        logger.info(f"Algorithms: {self.algorithms}")
        logger.info(f"Feature set: {self.feature_set}")
        logger.info(f"Nested CV: 5 outer folds × 4 inner folds (hardcoded in optimizer)")
        logger.info(f"Search iterations: {self.n_iter}")

    def load_pooled_data(self):
        """
        Load and pool data from all 21 cancer types.

        If train_test_split_file was provided, only training data is loaded.

        Returns
        -------
        X_pooled : pd.DataFrame
            Pooled feature matrix
        y_pooled : pd.Series
            Pooled outcome vector
        groups_pooled : pd.Series
            Pooled institution groups
        cancer_type_pooled : pd.Series
            Cancer type labels for each sample
        """
        if self.train_patient_ids:
            logger.info("Loading and pooling TRAINING DATA from 21 cancer types...")
            logger.info(f"Filtering to {len(self.train_patient_ids):,} training patients")
        else:
            logger.info("Loading and pooling data from 21 cancer types...")

        loader = DataLoader(self.db_path)

        # Load data for all cancer types (with optional patient ID filter)
        cancer_data_dict = loader.prepare_cancer_data_dict(
            cancer_types=self.cancer_types,
            feature_set=self.feature_set,
            outcome_col='isOncoKbLevel1',
            group_column='SUBJECT_CENTER',
            patient_ids_filter=self.train_patient_ids
        )

        # Pool data from all cancer types
        X_list = []
        y_list = []
        groups_list = []
        cancer_type_list = []

        logger.info("\nData Summary (by cancer type):")
        logger.info("=" * 90)
        logger.info(f"{'Cancer':12s} {'Samples':>8s} {'Features':>8s} {'Actionable':>10s} "
                   f"{'Rate':>8s} {'Institutions':>12s}")
        logger.info("-" * 90)

        total_samples = 0
        total_actionable = 0
        total_institutions = set()

        for cancer_type, (X, y, groups) in cancer_data_dict.items():
            X_list.append(X)
            y_list.append(y)
            groups_list.append(groups)
            cancer_type_list.extend([cancer_type] * len(X))

            n_samples = X.shape[0]
            n_actionable = y.sum()
            rate = y.mean()
            n_institutions = groups.nunique()

            logger.info(
                f"{cancer_type:12s} {n_samples:8d} {X.shape[1]:8d} {n_actionable:10d} "
                f"{rate:7.1%} {n_institutions:12d}"
            )

            total_samples += n_samples
            total_actionable += n_actionable
            total_institutions.update(groups.unique())

        logger.info("=" * 90)
        logger.info(
            f"{'POOLED':12s} {total_samples:8d} {X_list[0].shape[1]:8d} "
            f"{total_actionable:10d} {total_actionable/total_samples:7.1%} "
            f"{len(total_institutions):12d}"
        )
        logger.info("=" * 90)

        # Concatenate all data
        X_pooled = pd.concat(X_list, axis=0, ignore_index=True)
        y_pooled = pd.concat(y_list, axis=0, ignore_index=True)
        groups_pooled = pd.concat(groups_list, axis=0, ignore_index=True)
        cancer_type_pooled = pd.Series(cancer_type_list)

        logger.info(f"\nPooled dataset: {X_pooled.shape[0]:,} samples, {X_pooled.shape[1]} features")

        # Sample data if sample_fraction < 1.0
        if self.sample_fraction < 1.0:
            n_samples = len(X_pooled)
            n_sampled = int(n_samples * self.sample_fraction)

            logger.info(f"Sampling {self.sample_fraction:.1%} of data ({n_sampled:,} / {n_samples:,} samples)")

            # Random sampling with fixed seed for reproducibility
            np.random.seed(42)
            sample_idx = np.random.choice(n_samples, size=n_sampled, replace=False)
            sample_idx = np.sort(sample_idx)  # Sort to maintain order

            X_pooled = X_pooled.iloc[sample_idx].reset_index(drop=True)
            y_pooled = y_pooled.iloc[sample_idx].reset_index(drop=True)
            groups_pooled = groups_pooled.iloc[sample_idx].reset_index(drop=True)
            cancer_type_pooled = cancer_type_pooled.iloc[sample_idx].reset_index(drop=True)

            logger.info(f"Sampled dataset: {X_pooled.shape[0]:,} samples, {y_pooled.sum():,} actionable ({y_pooled.mean():.1%})")

        return X_pooled, y_pooled, groups_pooled, cancer_type_pooled

    def _compute_institution_weights(self, groups: pd.Series) -> np.ndarray:
        """
        Compute sample weights based on institution frequency using power weighting.

        Weights are computed as: weight_i = (1 / count_i) ^ alpha
        where count_i is the number of samples from that institution.

        Parameters
        ----------
        groups : pd.Series
            Institution group labels for each sample

        Returns
        -------
        weights : np.ndarray or None
            Normalized sample weights (sum to N, mean of 1.0), or None if alpha <= 0
        """
        if self.institution_weight_alpha <= 0:
            return None

        institution_counts = groups.value_counts()

        # weight = (1 / count) ^ alpha
        raw_weights = (1.0 / groups.map(institution_counts)) ** self.institution_weight_alpha

        # Normalize so weights sum to N (mean weight = 1.0)
        normalized_weights = raw_weights / raw_weights.sum() * len(raw_weights)

        # Log weight statistics
        weight_by_inst = pd.DataFrame({'group': groups, 'weight': normalized_weights.values}).groupby('group')['weight'].mean().sort_values(ascending=False)

        logger.info(f"  Institution weights (alpha={self.institution_weight_alpha}):")
        logger.info(f"    Weight range: [{normalized_weights.min():.3f}, {normalized_weights.max():.3f}]")
        logger.info(f"    Top 3 weighted: {dict(weight_by_inst.head(3).round(3))}")
        logger.info(f"    Bottom 3 weighted: {dict(weight_by_inst.tail(3).round(3))}")

        return normalized_weights.values

    def optimize_universal_hyperparameters(self, algorithm='random_forest'):
        """
        Run universal hyperparameter optimization for one algorithm.

        Uses nested cross-validation on pooled data from all 21 cancer types.

        Parameters
        ----------
        algorithm : str
            'random_forest', 'logistic', or 'xgboost'

        Returns
        -------
        results : dict
            Optimization results including best parameters and CV score
        """
        logger.info(f"\n{'='*90}")
        logger.info(f"UNIVERSAL HYPERPARAMETER OPTIMIZATION: {algorithm.upper()}")
        logger.info(f"{'='*90}\n")

        # Load pooled data
        X_pooled, y_pooled, groups_pooled, cancer_type_pooled = self.load_pooled_data()

        # Get hyperparameter search space
        if self.quick_mode:
            param_distributions = get_param_distributions_quick(algorithm)
            logger.info(f"Using quick search space for testing")
        else:
            param_distributions = get_param_distributions(algorithm)
            logger.info(f"Using full search space")

        # Log search space statistics
        n_combos = count_grid_combinations(param_distributions)
        logger.info(f"Search space: {len(param_distributions)} parameters")
        logger.info(f"Grid combinations: {n_combos:,}")
        logger.info(f"RandomizedSearchCV iterations: {self.n_iter}")
        logger.info(f"Total models to fit (nested CV): ~{self.n_iter * 5 * 4:,}")

        # Create optimizer
        optimizer = NestedCVOptimizer(
            param_distributions=param_distributions,
            search_method='random',
            n_iter=self.n_iter,
            scoring='roc_auc',
            n_jobs=self.n_jobs,  # Use configured n_jobs
            random_state=42,
            verbose=2,  # More verbose for long runs
            cv_strategy=self.cv_strategy  # Use configured CV strategy
        )

        # Compute institution sample weights if alpha > 0
        sample_weight = self._compute_institution_weights(groups_pooled)

        # Run nested CV optimization on pooled data
        logger.info(f"\nRunning nested cross-validation on pooled dataset...")
        logger.info(f"This will take approximately {self._estimate_runtime(algorithm, n_combos)} minutes")

        results = optimizer.optimize(
            X=X_pooled,
            y=y_pooled,
            groups=groups_pooled,
            algorithm=algorithm,
            sample_weight=sample_weight
        )

        # Add metadata
        results['algorithm'] = algorithm
        results['n_cancer_types'] = len(self.cancer_types)
        results['cancer_types'] = self.cancer_types
        results['n_samples'] = X_pooled.shape[0]
        results['n_features'] = X_pooled.shape[1]
        results['n_actionable'] = int(y_pooled.sum())
        results['actionable_rate'] = float(y_pooled.mean())
        results['n_institutions'] = int(groups_pooled.nunique())

        # Save results
        self._save_results(results, algorithm)

        logger.info(f"\n{'='*90}")
        logger.info(f"OPTIMIZATION COMPLETE: {algorithm.upper()}")
        logger.info(f"Best CV AUC: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
        logger.info(f"Best parameters (consensus):")
        for param, value in results['best_params_consensus'].items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Parameter stability: {results['param_stability']:.3f}")
        logger.info(f"{'='*90}\n")

        return results

    def run_all_algorithms(self):
        """
        Run universal hyperparameter optimization for all three algorithms.

        Returns
        -------
        all_results : dict
            {algorithm: results, ...}
        """
        logger.info(f"\n{'#'*90}")
        logger.info(f"PHASE 2B STEP 1: UNIVERSAL HYPERPARAMETER OPTIMIZATION")
        logger.info(f"Optimizing hyperparameters for {len(self.algorithms)} algorithms")
        logger.info(f"{'#'*90}\n")

        all_results = {}

        for algorithm in self.algorithms:
            try:
                results = self.optimize_universal_hyperparameters(algorithm)
                all_results[algorithm] = results
            except Exception as e:
                logger.error(f"Error optimizing {algorithm}: {str(e)}")
                logger.exception(e)
                continue

        # Generate comparison report
        self._generate_comparison_report(all_results)

        logger.info(f"\n{'#'*90}")
        logger.info(f"PHASE 2B STEP 1 COMPLETE")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"{'#'*90}\n")

        return all_results

    def _save_results(self, results, algorithm):
        """Save results for one algorithm."""
        # Save JSON
        json_file = self.output_dir / f'{algorithm}_universal_params.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results: {json_file}")

        # Save best params as CSV for easy reference
        params_df = pd.DataFrame([results['best_params_consensus']])
        params_df.insert(0, 'algorithm', algorithm)
        params_df.insert(1, 'cv_auc', results['mean_score'])
        params_df.insert(2, 'cv_auc_std', results['std_score'])
        params_df.insert(3, 'param_stability', results['param_stability'])

        csv_file = self.output_dir / f'{algorithm}_best_params.csv'
        params_df.to_csv(csv_file, index=False)
        logger.info(f"Saved best params: {csv_file}")

    def _generate_comparison_report(self, all_results):
        """Generate comparison report across all algorithms."""
        logger.info("\nGenerating algorithm comparison report...")

        # Create summary table
        summary_data = []
        for algorithm, results in all_results.items():
            summary_data.append({
                'algorithm': algorithm,
                'cv_auc': results['mean_score'],
                'cv_auc_std': results['std_score'],
                'param_stability': results['param_stability'],
                'n_samples': results['n_samples'],
                'n_actionable': results['n_actionable'],
                'actionable_rate': results['actionable_rate'],
                'n_institutions': results['n_institutions']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('cv_auc', ascending=False)

        # Save summary table
        summary_file = self.output_dir / 'algorithm_comparison.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved comparison table: {summary_file}")

        # Generate text report
        report_lines = []
        report_lines.append("=" * 90)
        report_lines.append("PHASE 2B STEP 1: ALGORITHM COMPARISON")
        report_lines.append("=" * 90)
        report_lines.append("")
        report_lines.append(f"Pooled data: {summary_df.iloc[0]['n_samples']:,} samples, "
                          f"{summary_df.iloc[0]['n_actionable']:,} actionable "
                          f"({summary_df.iloc[0]['actionable_rate']:.1%})")
        report_lines.append(f"Institutions: {summary_df.iloc[0]['n_institutions']}")
        report_lines.append(f"Cancer types: {len(self.cancer_types)} types pooled")
        report_lines.append("")
        report_lines.append("ALGORITHM PERFORMANCE (Nested CV AUC):")
        report_lines.append("-" * 90)

        for _, row in summary_df.iterrows():
            report_lines.append(
                f"{row['algorithm']:20s} {row['cv_auc']:.4f} ± {row['cv_auc_std']:.4f}"
            )

        report_lines.append("-" * 90)
        report_lines.append("")

        # Add best parameters for each algorithm
        for algorithm in self.algorithms:
            if algorithm in all_results:
                report_lines.append(f"\nBest Parameters (Consensus) - {algorithm.upper()}:")
                report_lines.append("-" * 90)
                for param, value in all_results[algorithm]['best_params_consensus'].items():
                    report_lines.append(f"  {param:25s}: {value}")
                report_lines.append(f"  {'param_stability':25s}: {all_results[algorithm]['param_stability']:.3f}")

        report_lines.append("")
        report_lines.append("=" * 90)
        report_lines.append("NEXT STEPS (Phase 2b Steps 2-5):")
        report_lines.append("=" * 90)
        report_lines.append("Step 2: Validate universal parameters on each of 21 cancer types individually")
        report_lines.append("Step 3: Select single best algorithm based on AUC, calibration, interpretability")
        report_lines.append("Step 4: Train final models for all 21 cancer types using selected algorithm")
        report_lines.append("Step 5: Generate comprehensive performance documentation")
        report_lines.append("=" * 90)

        report_text = "\n".join(report_lines)

        # Save text report
        report_file = self.output_dir / 'comparison_report.txt'
        with open(report_file, 'w') as f:
            f.write(report_text)
        logger.info(f"Saved comparison report: {report_file}")

        # Print to console
        print("\n" + report_text)

    def _estimate_runtime(self, algorithm, n_combos):
        """Estimate runtime in minutes."""
        # Rough estimates based on Phase 2a experience
        # Nested CV: 5 outer × 4 inner = 20 fits per iteration
        # Total fits: n_iter * 20

        if self.quick_mode:
            base_time = 0.5  # minutes per iteration (quick mode)
        else:
            if algorithm == 'random_forest':
                base_time = 2.0  # minutes per iteration
            elif algorithm == 'logistic':
                base_time = 0.5  # faster
            elif algorithm == 'xgboost':
                base_time = 3.0  # slower than RF
            else:
                base_time = 2.0

        # Adjust for large pooled dataset (~83k samples vs Phase 2a's ~20k per cancer type)
        pooled_multiplier = 2.0

        estimated_minutes = self.n_iter * base_time * pooled_multiplier
        return int(estimated_minutes)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 2b Step 1: Universal hyperparameter optimization'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=get_default_db_path(),
        help='Path to SQLite database (default: from BALDER_DB_PATH env var or ../../../output/...)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../../ml_experiments/phase2b_universal',
        help='Output directory for results (default: repo root ml_experiments/)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with reduced search space and iterations'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['random_forest', 'logistic', 'xgboost', 'all'],
        default='all',
        help='Algorithm to optimize (default: all)'
    )
    parser.add_argument(
        '--cv-strategy',
        type=str,
        choices=['standard', 'group', 'stratified_group'],
        default='group',
        help='Cross-validation strategy: standard (StratifiedKFold), group (GroupKFold), '
             'or stratified_group (StratifiedGroupKFold for balanced institution-aware folds). '
             'Default: group'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel jobs for RandomizedSearchCV. -1 uses all CPU cores. '
             'IMPORTANT: Use --n-jobs 1 or --n-jobs 2 with --feature-set both to avoid '
             'memory issues (50-100+ GB RAM with all cores). Default: -1'
    )
    parser.add_argument(
        '--feature-set',
        type=str,
        choices=['demographic', 'technical', 'both'],
        default='both',
        help='Feature set to use: demographic (6 features), technical (18 features), '
             'or both (24 features). Default: both. Note: "both" with n_jobs=-1 '
             'can consume 50-100+ GB RAM due to high-cardinality feature encoding.'
    )
    parser.add_argument(
        '--sample-fraction',
        type=float,
        default=1.0,
        help='Fraction of pooled data to use for optimization (0.0 to 1.0). '
             'Use < 1.0 to reduce memory usage. For example, --sample-fraction 0.5 '
             'uses 50%% of data (~42k samples instead of ~85k). Default: 1.0 (use all data)'
    )
    parser.add_argument(
        '--train-test-split-file',
        type=str,
        default=None,
        help='Path to train/test split CSV file. If provided, only training data '
             'will be used for hyperparameter optimization. File should have columns: '
             'PATIENT_ID, split (with "train"/"test" values). '
             'Default location: ../../analysis_output/train_test_split/train_test_split.csv'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=None,
        help='Number of RandomizedSearchCV iterations. Overrides the default '
             '(10 for --quick-test, 200 for full runs). Higher values explore '
             'more of the hyperparameter space but take longer.'
    )
    parser.add_argument(
        '--institution-weight-alpha',
        type=float,
        default=0.0,
        help='Power parameter for institutional sample weighting (0.0 to 1.0). '
             '0.0 = no weighting (default), 0.5 = moderate correction, 1.0 = full inverse frequency. '
             'Higher values reduce the influence of dominant institutions (MSK, DFCI) during training.'
    )

    args = parser.parse_args()

    # Validate sample_fraction
    if not (0.0 < args.sample_fraction <= 1.0):
        logger.error(f"Invalid --sample-fraction: {args.sample_fraction}. Must be between 0.0 and 1.0")
        sys.exit(1)

    # Check database exists
    if not Path(args.db_path).exists():
        # Try alternative path
        alt_path = '../../../output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite'
        if Path(alt_path).exists():
            args.db_path = alt_path
            logger.info(f"Using alternative database path: {alt_path}")
        else:
            logger.error(f"Database not found: {args.db_path}")
            logger.error(f"Also tried: {alt_path}")
            logger.error(f"Set BALDER_DB_PATH environment variable or check database location")
            sys.exit(1)

    # Initialize Phase 2b
    phase2b = Phase2bUniversal(
        db_path=args.db_path,
        output_dir=args.output_dir,
        quick_mode=args.quick_test,
        cv_strategy=args.cv_strategy,
        n_jobs=args.n_jobs,
        feature_set=args.feature_set,
        sample_fraction=args.sample_fraction,
        train_test_split_file=args.train_test_split_file,
        n_iter=args.n_iter,
        institution_weight_alpha=args.institution_weight_alpha
    )

    # Run optimization
    if args.algorithm == 'all':
        results = phase2b.run_all_algorithms()
    else:
        results = {args.algorithm: phase2b.optimize_universal_hyperparameters(args.algorithm)}

    logger.info("\n✓ Phase 2b Step 1 complete!")
    logger.info(f"Results saved to: {phase2b.output_dir}")


if __name__ == '__main__':
    main()
