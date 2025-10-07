"""
Experiment Framework for Feature Set Comparison

Systematic comparison of demographic vs. technical features across cancer types and algorithms.
Provides statistical rigor and comprehensive reporting for experimental analysis.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from .config import Config
from .analyzer import OncoKBAnalyzer


@dataclass
class ExperimentConfig:
    """Configuration for feature set comparison experiments"""

    # Base settings
    db_path: str
    cancer_types: List[str]
    algorithms: List[str] = field(default_factory=lambda: ["random_forest", "logistic"])
    feature_sets: List[str] = field(default_factory=lambda: ["demographic", "technical", "both"])

    # Cross-validation settings
    cv_folds: int = 5
    n_random_seeds: int = 5  # For statistical robustness
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])

    # Output settings
    output_dir: str = "experiments/"
    experiment_name: str = ""  # Will be auto-generated if empty

    # Statistical testing
    statistical_tests: bool = True
    alpha: float = 0.05  # Significance level

    def __post_init__(self):
        """Post-initialization setup"""
        # Auto-generate experiment name if not provided
        if not self.experiment_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_name = f"feature_comparison_{timestamp}"

        # Ensure we have the right number of random seeds
        if len(self.random_seeds) != self.n_random_seeds:
            # Generate additional seeds if needed
            if len(self.random_seeds) < self.n_random_seeds:
                additional_seeds = np.random.RandomState(42).randint(
                    1000, 9999,
                    self.n_random_seeds - len(self.random_seeds)
                ).tolist()
                self.random_seeds.extend(additional_seeds)
            else:
                # Truncate if too many
                self.random_seeds = self.random_seeds[:self.n_random_seeds]

    def validate(self) -> bool:
        """Validate experiment configuration"""
        # Check required fields
        if not self.db_path:
            raise ValueError("db_path is required")
        if not self.cancer_types:
            raise ValueError("cancer_types cannot be empty")

        # Validate algorithms
        valid_algorithms = ["random_forest", "logistic"]
        for algo in self.algorithms:
            if algo not in valid_algorithms:
                raise ValueError(f"Algorithm '{algo}' not supported. Valid: {valid_algorithms}")

        # Validate feature sets
        valid_feature_sets = ["demographic", "technical", "both"]
        for fs in self.feature_sets:
            if fs not in valid_feature_sets:
                raise ValueError(f"Feature set '{fs}' not supported. Valid: {valid_feature_sets}")

        # Validate CV settings
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.n_random_seeds < 1:
            raise ValueError("n_random_seeds must be at least 1")

        # Validate statistical settings
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError("alpha must be between 0 and 1")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'db_path': self.db_path,
            'cancer_types': self.cancer_types,
            'algorithms': self.algorithms,
            'feature_sets': self.feature_sets,
            'cv_folds': self.cv_folds,
            'n_random_seeds': self.n_random_seeds,
            'random_seeds': self.random_seeds,
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name,
            'statistical_tests': self.statistical_tests,
            'alpha': self.alpha
        }


class FeatureSetExperiment:
    """Systematic comparison of feature sets across cancer types and algorithms"""

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment with configuration"""
        self.config = config
        self.config.validate()  # Validate configuration

        self.results = {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self.logger.info(f"Initialized experiment: {self.config.experiment_name}")
        self.logger.info(f"Cancer types: {self.config.cancer_types}")
        self.logger.info(f"Algorithms: {self.config.algorithms}")
        self.logger.info(f"Feature sets: {self.config.feature_sets}")
        self.logger.info(f"Random seeds: {self.config.n_random_seeds}")

    def run_full_experiment(self) -> Dict[str, Any]:
        """Run complete feature set comparison experiment"""

        self.logger.info("Starting full experiment...")

        # Calculate total number of experiments
        total_experiments = (
            len(self.config.cancer_types) *
            len(self.config.algorithms) *
            len(self.config.feature_sets) *
            self.config.n_random_seeds
        )

        self.logger.info(f"Total experiments to run: {total_experiments}")

        experiment_count = 0

        # Nested loop structure: Cancer Type -> Algorithm -> Feature Set -> Random Seed
        for cancer_type in self.config.cancer_types:
            self.logger.info(f"Starting cancer type: {cancer_type}")
            self.results[cancer_type] = {}

            for algorithm in self.config.algorithms:
                self.logger.info(f"  Starting algorithm: {algorithm}")
                self.results[cancer_type][algorithm] = {}

                for feature_set in self.config.feature_sets:
                    self.logger.info(f"    Starting feature set: {feature_set}")
                    self.results[cancer_type][algorithm][feature_set] = []

                    # Multiple runs with different seeds for robust statistics
                    for seed in self.config.random_seeds:
                        experiment_count += 1
                        progress = (experiment_count / total_experiments) * 100

                        self.logger.info(
                            f"      Running experiment {experiment_count}/{total_experiments} "
                            f"({progress:.1f}%) - seed {seed}"
                        )

                        try:
                            result = self._run_single_experiment(
                                cancer_type, algorithm, feature_set, seed
                            )
                            self.results[cancer_type][algorithm][feature_set].append(result)

                        except Exception as e:
                            self.logger.error(
                                f"      Failed experiment: {cancer_type}-{algorithm}-{feature_set}-{seed}: {e}"
                            )
                            # Store error information
                            error_result = {
                                'error': str(e),
                                'cancer_type': cancer_type,
                                'algorithm': algorithm,
                                'feature_set': feature_set,
                                'random_seed': seed
                            }
                            self.results[cancer_type][algorithm][feature_set].append(error_result)

        self.logger.info("All experiments completed. Processing results...")

        # Post-process results for statistical analysis
        self._compute_summary_statistics()

        if self.config.statistical_tests:
            self._perform_statistical_tests()

        # Generate comprehensive report
        self._generate_experiment_report()

        self.logger.info(f"Experiment completed! Results saved to: {self._get_output_dir()}")

        return self.results

    def _run_single_experiment(self, cancer_type: str, algorithm: str,
                              feature_set: str, random_seed: int) -> Dict[str, Any]:
        """Run a single experimental condition"""

        # Create specific config for this run
        run_config = Config(
            db_path=self.config.db_path,
            cancer_types=[cancer_type],
            algorithm=algorithm,
            feature_set=feature_set,
            random_state=random_seed,
            cv_folds=self.config.cv_folds,
            min_actionable_events=250,  # Use default threshold
            outcome_column="isOncoKbLevel1"
        )

        # Run analysis
        analyzer = OncoKBAnalyzer(run_config)
        results = analyzer.run_analysis()

        # Extract key metrics
        cancer_result = results['cancer_type_results'][cancer_type]

        # Check if analysis succeeded
        if cancer_result.get('status') != 'completed':
            raise ValueError(f"Analysis failed: {cancer_result.get('error', 'Unknown error')}")

        return {
            'cv_auc_mean': cancer_result['cv_auc_mean'],
            'cv_auc_std': cancer_result['cv_auc_std'],
            'cv_scores': cancer_result['cv_scores'],
            'test_auc': cancer_result['test_auc'],
            'test_accuracy': cancer_result['test_accuracy'],
            'test_precision': cancer_result['test_precision'],
            'test_recall': cancer_result['test_recall'],
            'test_f1': cancer_result['test_f1'],
            'n_features': cancer_result['n_features'],
            'n_patients': cancer_result['n_patients'],
            'n_actionable': cancer_result['n_actionable'],
            'actionable_rate': cancer_result['actionable_rate'],
            'random_seed': random_seed,
            'duration': results['analysis_metadata']['duration_seconds']
        }

    def _get_output_dir(self) -> Path:
        """Get output directory path"""
        return Path(self.config.output_dir) / self.config.experiment_name

    def _compute_summary_statistics(self):
        """Compute summary statistics across multiple runs"""
        self.logger.info("Computing summary statistics...")

        for cancer_type in self.results:
            for algorithm in self.results[cancer_type]:
                for feature_set in self.results[cancer_type][algorithm]:
                    runs = self.results[cancer_type][algorithm][feature_set]

                    # Filter out failed runs
                    successful_runs = [run for run in runs if 'error' not in run]
                    failed_runs = [run for run in runs if 'error' in run]

                    if not successful_runs:
                        self.logger.warning(
                            f"No successful runs for {cancer_type}-{algorithm}-{feature_set}"
                        )
                        continue

                    # Aggregate CV AUC scores across all runs
                    all_cv_aucs = [run['cv_auc_mean'] for run in successful_runs]
                    all_test_aucs = [run['test_auc'] for run in successful_runs]

                    # Summary statistics
                    summary = {
                        'mean_cv_auc': np.mean(all_cv_aucs),
                        'std_cv_auc': np.std(all_cv_aucs),
                        'median_cv_auc': np.median(all_cv_aucs),
                        'min_cv_auc': np.min(all_cv_aucs),
                        'max_cv_auc': np.max(all_cv_aucs),
                        'mean_test_auc': np.mean(all_test_aucs),
                        'std_test_auc': np.std(all_test_aucs),
                        'mean_n_features': np.mean([run['n_features'] for run in successful_runs]),
                        'mean_n_patients': np.mean([run['n_patients'] for run in successful_runs]),
                        'mean_actionable_rate': np.mean([run['actionable_rate'] for run in successful_runs]),
                        'successful_runs': len(successful_runs),
                        'failed_runs': len(failed_runs),
                        'total_runs': len(runs)
                    }

                    # Store results in new structure
                    self.results[cancer_type][algorithm][feature_set] = {
                        'individual_runs': successful_runs,
                        'failed_runs': failed_runs,
                        'summary': summary
                    }

    def _perform_statistical_tests(self):
        """Perform statistical tests between feature sets"""
        self.logger.info("Performing statistical tests...")

        # Import here to avoid dependency issues if not available
        try:
            from scipy.stats import ttest_rel
        except ImportError:
            self.logger.warning("scipy not available - skipping statistical tests")
            return

        # For each cancer type and algorithm, compare feature sets
        for cancer_type in self.results:
            for algorithm in self.results[cancer_type]:

                # Extract AUC values for each feature set
                feature_set_aucs = {}
                for feature_set in self.config.feature_sets:
                    if feature_set in self.results[cancer_type][algorithm]:
                        runs = self.results[cancer_type][algorithm][feature_set]['individual_runs']
                        if runs:  # Only if we have successful runs
                            aucs = [run['cv_auc_mean'] for run in runs]
                            feature_set_aucs[feature_set] = aucs

                # Only proceed if we have at least 2 feature sets with data
                if len(feature_set_aucs) < 2:
                    continue

                # Pairwise comparisons using paired t-tests
                comparisons = {}
                feature_set_list = list(feature_set_aucs.keys())

                for i, fs1 in enumerate(feature_set_list):
                    for fs2 in feature_set_list[i+1:]:

                        # Ensure equal sample sizes for paired test
                        min_samples = min(len(feature_set_aucs[fs1]), len(feature_set_aucs[fs2]))
                        aucs1 = feature_set_aucs[fs1][:min_samples]
                        aucs2 = feature_set_aucs[fs2][:min_samples]

                        if min_samples >= 2:  # Need at least 2 pairs for t-test
                            t_stat, p_value = ttest_rel(aucs1, aucs2)

                            comparisons[f"{fs1}_vs_{fs2}"] = {
                                't_statistic': float(t_stat),
                                'p_value': float(p_value),
                                'significant': p_value < self.config.alpha,
                                'effect_size': np.mean(aucs1) - np.mean(aucs2),
                                'n_pairs': min_samples,
                                'fs1_mean': np.mean(aucs1),
                                'fs2_mean': np.mean(aucs2)
                            }

                # Store statistical results
                if comparisons:
                    self.results[cancer_type][algorithm]['statistical_tests'] = comparisons

    def _generate_experiment_report(self):
        """Generate comprehensive experiment report"""
        self.logger.info("Generating experiment report...")

        # Create output directory
        report_dir = self._get_output_dir()
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(report_dir / "experiment_config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save raw results
        with open(report_dir / "raw_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Generate summary table
        self._create_summary_table(report_dir)

        # Generate text report
        self._create_text_report(report_dir)

        self.logger.info(f"Report generated in: {report_dir}")

    def _create_summary_table(self, report_dir: Path):
        """Create summary table of results"""

        rows = []
        for cancer_type in self.results:
            for algorithm in self.results[cancer_type]:
                for feature_set in self.config.feature_sets:
                    if (feature_set in self.results[cancer_type][algorithm] and
                        'summary' in self.results[cancer_type][algorithm][feature_set]):

                        summary = self.results[cancer_type][algorithm][feature_set]['summary']
                        rows.append({
                            'Cancer_Type': cancer_type,
                            'Algorithm': algorithm,
                            'Feature_Set': feature_set,
                            'Mean_CV_AUC': f"{summary['mean_cv_auc']:.3f}",
                            'Std_CV_AUC': f"{summary['std_cv_auc']:.3f}",
                            'Mean_Test_AUC': f"{summary['mean_test_auc']:.3f}",
                            'Std_Test_AUC': f"{summary['std_test_auc']:.3f}",
                            'N_Features': int(summary['mean_n_features']),
                            'N_Patients': int(summary['mean_n_patients']),
                            'Actionable_Rate': f"{summary['mean_actionable_rate']:.3f}",
                            'Successful_Runs': summary['successful_runs'],
                            'Failed_Runs': summary['failed_runs']
                        })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(report_dir / "summary_table.csv", index=False)

            # Create performance matrix
            pivot_table = df.pivot_table(
                values='Mean_CV_AUC',
                index=['Cancer_Type', 'Algorithm'],
                columns='Feature_Set',
                aggfunc='first'
            )
            pivot_table.to_csv(report_dir / "performance_matrix.csv")

    def _create_text_report(self, report_dir: Path):
        """Create detailed text report"""

        with open(report_dir / "detailed_report.txt", 'w') as f:
            f.write("Feature Set Comparison Experiment Report\n")
            f.write("=" * 50 + "\n\n")

            # Experiment configuration
            f.write("Experiment Configuration:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Experiment Name: {self.config.experiment_name}\n")
            f.write(f"Database: {self.config.db_path}\n")
            f.write(f"Cancer Types: {', '.join(self.config.cancer_types)}\n")
            f.write(f"Algorithms: {', '.join(self.config.algorithms)}\n")
            f.write(f"Feature Sets: {', '.join(self.config.feature_sets)}\n")
            f.write(f"Random Seeds: {self.config.n_random_seeds}\n")
            f.write(f"CV Folds: {self.config.cv_folds}\n\n")

            # Results summary
            f.write("Results Summary:\n")
            f.write("-" * 15 + "\n")

            for cancer_type in self.results:
                f.write(f"\n{cancer_type}:\n")
                for algorithm in self.results[cancer_type]:
                    f.write(f"  {algorithm}:\n")
                    for feature_set in self.config.feature_sets:
                        if (feature_set in self.results[cancer_type][algorithm] and
                            'summary' in self.results[cancer_type][algorithm][feature_set]):

                            summary = self.results[cancer_type][algorithm][feature_set]['summary']
                            f.write(f"    {feature_set}: ")
                            f.write(f"AUC {summary['mean_cv_auc']:.3f} ± {summary['std_cv_auc']:.3f} ")
                            f.write(f"({summary['mean_n_features']:.0f} features)\n")

            # Statistical tests
            f.write("\nStatistical Comparisons:\n")
            f.write("-" * 23 + "\n")

            for cancer_type in self.results:
                for algorithm in self.results[cancer_type]:
                    if 'statistical_tests' in self.results[cancer_type][algorithm]:
                        f.write(f"\n{cancer_type} - {algorithm}:\n")
                        tests = self.results[cancer_type][algorithm]['statistical_tests']
                        for comparison, result in tests.items():
                            significance = "***" if result['significant'] else "n.s."
                            f.write(f"  {comparison}: ")
                            f.write(f"p={result['p_value']:.3f} {significance} ")
                            f.write(f"(effect size: {result['effect_size']:.3f})\n")