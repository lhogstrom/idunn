"""
Phase 2a Pilot: Universal vs Cancer-Type-Specific Hyperparameter Optimization

This script implements the Phase 2a pilot study to determine whether:
1. Universal hyperparameters (same for all cancer types) are sufficient, OR
2. Cancer-type-specific hyperparameters provide significant improvement

Decision Rule (from hyperparameter_optimization_plan.md Section 4.6):
- If mean(delta_auc) < 0.01: Universal sufficient → Phase 2b uses universal
- If mean(delta_auc) ≥ 0.02: Cancer-specific valuable → Phase 2b uses specific
- If 0.01 ≤ mean(delta_auc) < 0.02: Borderline → Consider hybrid

Cancer Types (5 representative types):
- LUAD (Lung Adenocarcinoma) - Large, common
- BRCA (Breast Cancer) - Large, common
- COAD (Colon Adenocarcinoma) - Large, common
- MEL (Melanoma) - Medium, specific biology
- GIST (Gastrointestinal Stromal Tumor) - Small, specialized centers

Algorithms:
- Random Forest (primary)
- Logistic Regression (secondary)
- XGBoost (gradient boosting)

Outputs:
- Nested CV results for universal approach
- Nested CV results for cancer-specific approach
- Comparison table with delta_auc
- Decision recommendation
- Detailed JSON results

"""

import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

from oncokb_analyzer.data_loader import DataLoader
from db_config import get_default_db_path
from oncokb_analyzer.hyperparameter_optimizer import NestedCVOptimizer
from oncokb_analyzer.param_spaces import (
    RF_PARAM_DISTRIBUTIONS,
    LR_PARAM_DISTRIBUTIONS,
    get_param_distributions
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase2aPilot:
    """
    Phase 2a pilot experiment comparing universal vs cancer-specific hyperparameters.
    """

    def __init__(self, db_path, output_dir='../../ml_experiments/phase2a_pilot', quick_mode=False):
        """
        Initialize Phase 2a pilot experiment.

        Parameters
        ----------
        db_path : str
            Path to SQLite database
        output_dir : str
            Directory for output files
        quick_mode : bool
            If True, use reduced search space and fewer CV folds for testing
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.quick_mode = quick_mode

        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = self.output_dir / f'run_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Phase 2a Pilot initialized")
        logger.info(f"Database: {db_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Quick mode: {quick_mode}")

        # Phase 2a configuration
        self.cancer_types = ['LUAD', 'BRCA', 'COAD', 'MEL', 'GIST']
        self.algorithms = ['random_forest', 'logistic', 'xgboost']
        self.feature_set = 'both'  # Combined demographic + technical features

        # CV configuration
        # Note: compare_universal_vs_specific uses fixed 5-fold CV (hardcoded in optimizer)
        # Only n_iter is configurable via optimizer constructor
        if quick_mode:
            self.n_iter = 10  # Reduced search iterations
        else:
            self.n_iter = 50  # Full search

        logger.info(f"Cancer types: {self.cancer_types}")
        logger.info(f"Algorithms: {self.algorithms}")
        logger.info(f"CV folds: 5 (fixed in optimizer)")
        logger.info(f"Search iterations: {self.n_iter}")

    def load_data(self):
        """
        Load data for all 5 cancer types.

        Returns
        -------
        cancer_data_dict : dict
            {cancer_type: (X, y, groups), ...}
        """
        logger.info("Loading data for Phase 2a pilot...")

        loader = DataLoader(self.db_path)
        cancer_data_dict = loader.prepare_cancer_data_dict(
            cancer_types=self.cancer_types,
            feature_set=self.feature_set,
            outcome_col='isOncoKbLevel1',
            group_column='SUBJECT_CENTER'
        )

        # Log summary statistics
        logger.info("\nData Summary:")
        logger.info("=" * 80)
        for cancer_type, (X, y, groups) in cancer_data_dict.items():
            logger.info(
                f"{cancer_type:6s}: {X.shape[0]:6d} samples, "
                f"{X.shape[1]:2d} features, "
                f"{y.sum():5d} actionable ({y.mean():5.1%}), "
                f"{groups.nunique():3d} institutions"
            )
        logger.info("=" * 80)

        return cancer_data_dict

    def run_comparison(self, algorithm='random_forest'):
        """
        Run universal vs cancer-specific comparison for one algorithm.

        Parameters
        ----------
        algorithm : str
            'random_forest' or 'logistic'

        Returns
        -------
        results : dict
            Comparison results from NestedCVOptimizer
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING COMPARISON: {algorithm.upper()}")
        logger.info(f"{'='*80}\n")

        # Load data
        cancer_data_dict = self.load_data()

        # Get hyperparameter search space
        param_distributions = get_param_distributions(algorithm)

        if self.quick_mode:
            # Use smaller search space for quick testing
            from oncokb_analyzer.param_spaces import get_param_distributions_quick
            param_distributions = get_param_distributions_quick(algorithm)
            logger.info("Using quick search space for testing")

        # Create optimizer
        optimizer = NestedCVOptimizer(
            param_distributions=param_distributions,
            search_method='random',
            n_iter=self.n_iter,
            scoring='roc_auc',
            n_jobs=-1,  # Use all cores
            random_state=42,
            cv_strategy='standard',  # Use standard for Phase 2a (pooled data doesn't have groups)
            verbose=1
        )

        # Run comparison
        logger.info(f"Running universal vs cancer-specific comparison...")
        logger.info(f"This will perform nested CV for each cancer type and pooled data")
        logger.info(f"Estimated time: {self._estimate_runtime(algorithm)} minutes")

        # Note: CV configuration is handled internally by compare_universal_vs_specific
        # It uses the outer_cv_folds and inner_cv_folds from self
        results = optimizer.compare_universal_vs_specific(
            cancer_data_dict=cancer_data_dict,
            algorithm=algorithm
        )

        # Save results
        self._save_results(results, algorithm)

        # Print summary
        self._print_summary(results, algorithm)

        return results

    def run_all_algorithms(self):
        """
        Run comparison for all algorithms (Random Forest and Logistic Regression).

        Returns
        -------
        all_results : dict
            {algorithm: results, ...}
        """
        logger.info("\n" + "="*80)
        logger.info("PHASE 2A PILOT: UNIVERSAL VS CANCER-SPECIFIC HYPERPARAMETERS")
        logger.info("="*80)

        all_results = {}

        for algorithm in self.algorithms:
            results = self.run_comparison(algorithm)
            all_results[algorithm] = results

        # Generate combined summary
        self._generate_combined_summary(all_results)

        logger.info("\n" + "="*80)
        logger.info("PHASE 2A PILOT COMPLETE")
        logger.info("="*80)
        logger.info(f"Results saved to: {self.output_dir}")

        return all_results

    def _estimate_runtime(self, algorithm):
        """Estimate runtime in minutes"""
        # Rough estimates based on algorithm and CV configuration
        # Random Forest is slower than Logistic Regression
        base_time = 30 if algorithm == 'random_forest' else 15

        if self.quick_mode:
            return base_time * 0.3  # ~9 min for RF, ~4.5 min for LR
        else:
            return base_time  # ~30 min for RF, ~15 min for LR

    def _save_results(self, results, algorithm):
        """Save results to JSON and CSV"""
        # Save full results as JSON
        json_path = self.output_dir / f'{algorithm}_results.json'
        with open(json_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            results_serializable = self._make_serializable(results)
            json.dump(results_serializable, f, indent=2)
        logger.info(f"Saved JSON results to: {json_path}")

        # Save comparison table as CSV
        comparison_df = self._create_comparison_table(results)
        csv_path = self.output_dir / f'{algorithm}_comparison.csv'
        comparison_df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison table to: {csv_path}")

    def _make_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def _create_comparison_table(self, results):
        """Create comparison table DataFrame"""
        rows = []

        for cancer_type in results['delta_auc_by_cancer'].keys():
            rows.append({
                'cancer_type': cancer_type,
                'universal_auc': results['universal_auc_by_cancer'][cancer_type],
                'specific_auc': results['specific_auc_by_cancer'][cancer_type],
                'delta_auc': results['delta_auc_by_cancer'][cancer_type],
                'improvement_pct': results['delta_auc_by_cancer'][cancer_type] * 100
            })

        df = pd.DataFrame(rows)

        # Add summary row
        summary_row = {
            'cancer_type': 'MEAN',
            'universal_auc': df['universal_auc'].mean(),
            'specific_auc': df['specific_auc'].mean(),
            'delta_auc': results['mean_delta_auc'],
            'improvement_pct': results['mean_delta_auc'] * 100
        }
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

        return df

    def _print_summary(self, results, algorithm):
        """Print summary of results"""
        logger.info(f"\n{'='*80}")
        logger.info(f"SUMMARY: {algorithm.upper()}")
        logger.info(f"{'='*80}\n")

        # Print comparison table
        df = self._create_comparison_table(results)
        logger.info("\nComparison Table:")
        logger.info(df.to_string(index=False))

        # Print decision
        logger.info(f"\n{'='*80}")
        logger.info(f"DECISION: {results['decision'].upper()}")
        logger.info(f"{'='*80}")
        logger.info(f"Mean delta AUC: {results['mean_delta_auc']:.4f}")
        logger.info(f"\nRationale:")
        logger.info(results['decision_rationale'])

        # Print universal parameters
        logger.info(f"\n{'='*80}")
        logger.info("UNIVERSAL PARAMETERS:")
        logger.info(f"{'='*80}")
        for param, value in results['universal_params'].items():
            logger.info(f"  {param}: {value}")

    def _generate_combined_summary(self, all_results):
        """Generate combined summary across all algorithms"""
        summary_path = self.output_dir / 'combined_summary.txt'

        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PHASE 2A PILOT: COMBINED SUMMARY\n")
            f.write("="*80 + "\n\n")

            for algorithm, results in all_results.items():
                f.write(f"\n{algorithm.upper()}\n")
                f.write("-"*80 + "\n")
                f.write(f"Decision: {results['decision']}\n")
                f.write(f"Mean delta AUC: {results['mean_delta_auc']:.4f}\n")
                f.write(f"Rationale: {results['decision_rationale']}\n")

                # Write comparison table
                f.write("\nComparison Table:\n")
                df = self._create_comparison_table(results)
                f.write(df.to_string(index=False))
                f.write("\n\n")

            # Overall recommendation
            f.write("="*80 + "\n")
            f.write("OVERALL RECOMMENDATION FOR PHASE 2B\n")
            f.write("="*80 + "\n")

            # Count decisions
            decisions = [results['decision'] for results in all_results.values()]
            if all(d == 'universal' for d in decisions):
                f.write("\nRecommendation: UNIVERSAL HYPERPARAMETERS\n")
                f.write("Both algorithms show that universal hyperparameters are sufficient.\n")
                f.write("Proceed with Phase 2b using universal hyperparameters for all 21 cancer types.\n")
            elif all(d == 'cancer_specific' for d in decisions):
                f.write("\nRecommendation: CANCER-SPECIFIC HYPERPARAMETERS\n")
                f.write("Both algorithms show significant benefit from cancer-specific tuning.\n")
                f.write("Proceed with Phase 2b optimizing hyperparameters separately for each cancer type.\n")
            else:
                f.write("\nRecommendation: MIXED RESULTS - CONSIDER HYBRID\n")
                f.write("Results differ by algorithm. Consider:\n")
                f.write("1. Use cancer-specific for algorithms showing benefit\n")
                f.write("2. Use universal for algorithms showing no benefit\n")
                f.write("3. Further investigation on borderline cases\n")

        logger.info(f"Saved combined summary to: {summary_path}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Phase 2a Pilot: Universal vs Cancer-Specific Hyperparameters'
    )
    parser.add_argument(
        '--db-path',
        default=get_default_db_path(),
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--output-dir',
        default='../../ml_experiments/phase2a_pilot',
        help='Output directory for results (default: repo root ml_experiments/)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (reduced search space and CV folds)'
    )
    parser.add_argument(
        '--algorithm',
        choices=['random_forest', 'logistic', 'xgboost', 'both'],
        default='both',
        help='Which algorithm(s) to run'
    )

    args = parser.parse_args()

    # Create pilot experiment
    pilot = Phase2aPilot(
        db_path=args.db_path,
        output_dir=args.output_dir,
        quick_mode=args.quick_test
    )

    # Run experiments
    if args.algorithm == 'both':
        pilot.run_all_algorithms()
    else:
        pilot.run_comparison(args.algorithm)


if __name__ == '__main__':
    main()
