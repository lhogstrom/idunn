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
import matplotlib.pyplot as plt
import seaborn as sns

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
    cv_strategies: List[str] = field(default_factory=lambda: ["standard", "group"])
    group_column: str = "SUBJECT_CENTER"  # Column for GroupKFold
    n_random_seeds: int = 5  # For statistical robustness
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])

    # Train-test split settings
    institution_aware_test_split: bool = False  # Use GroupShuffleSplit for test splits
    test_split_group_column: str = "SUBJECT_CENTER"  # Column for institution-aware test splits
    train_test_split_file: str = None  # Path to train/test split CSV file (PATIENT_ID, split columns)

    # Parallel processing
    n_jobs: int = -1  # Number of parallel jobs (-1 uses all cores, 1 for single-threaded to save memory)

    # Output settings
    output_dir: str = "../../ml_experiments/"
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

        # Validate CV strategies
        valid_cv_strategies = ["standard", "group", "stratified_group"]
        for cv_strat in self.cv_strategies:
            if cv_strat not in valid_cv_strategies:
                raise ValueError(f"CV strategy '{cv_strat}' not supported. Valid: {valid_cv_strategies}")

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
            'cv_strategies': self.cv_strategies,
            'group_column': self.group_column,
            'n_random_seeds': self.n_random_seeds,
            'random_seeds': self.random_seeds,
            'institution_aware_test_split': self.institution_aware_test_split,
            'test_split_group_column': self.test_split_group_column,
            'train_test_split_file': self.train_test_split_file,
            'n_jobs': self.n_jobs,
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
        self.logger.info(f"CV strategies: {self.config.cv_strategies}")
        self.logger.info(f"Random seeds: {self.config.n_random_seeds}")

    def run_full_experiment(self) -> Dict[str, Any]:
        """Run complete feature set comparison experiment"""

        self.logger.info("Starting full experiment...")

        # Calculate total number of experiments
        total_experiments = (
            len(self.config.cancer_types) *
            len(self.config.algorithms) *
            len(self.config.cv_strategies) *
            len(self.config.feature_sets) *
            self.config.n_random_seeds
        )

        self.logger.info(f"Total experiments to run: {total_experiments}")

        experiment_count = 0

        # Nested loop structure: Cancer Type -> Algorithm -> CV Strategy -> Feature Set -> Random Seed
        for cancer_type in self.config.cancer_types:
            self.logger.info(f"Starting cancer type: {cancer_type}")
            self.results[cancer_type] = {}

            for algorithm in self.config.algorithms:
                self.logger.info(f"  Starting algorithm: {algorithm}")
                self.results[cancer_type][algorithm] = {}

                for cv_strategy in self.config.cv_strategies:
                    self.logger.info(f"    Starting CV strategy: {cv_strategy}")
                    self.results[cancer_type][algorithm][cv_strategy] = {}

                    for feature_set in self.config.feature_sets:
                        self.logger.info(f"      Starting feature set: {feature_set}")
                        self.results[cancer_type][algorithm][cv_strategy][feature_set] = []

                        # Multiple runs with different seeds for robust statistics
                        for seed in self.config.random_seeds:
                            experiment_count += 1
                            progress = (experiment_count / total_experiments) * 100

                            self.logger.info(
                                f"        Running experiment {experiment_count}/{total_experiments} "
                                f"({progress:.1f}%) - seed {seed}"
                            )

                            try:
                                result = self._run_single_experiment(
                                    cancer_type, algorithm, cv_strategy, feature_set, seed
                                )
                                self.results[cancer_type][algorithm][cv_strategy][feature_set].append(result)

                            except Exception as e:
                                self.logger.error(
                                    f"        Failed experiment: {cancer_type}-{algorithm}-{cv_strategy}-{feature_set}-{seed}: {e}"
                                )
                                # Store error information
                                error_result = {
                                    'error': str(e),
                                    'cancer_type': cancer_type,
                                    'algorithm': algorithm,
                                    'cv_strategy': cv_strategy,
                                    'feature_set': feature_set,
                                    'random_seed': seed
                                }
                                self.results[cancer_type][algorithm][cv_strategy][feature_set].append(error_result)

        self.logger.info("All experiments completed. Processing results...")

        # Post-process results for statistical analysis
        self._compute_summary_statistics()

        if self.config.statistical_tests:
            self._perform_statistical_tests()

        # Generate comprehensive report
        self._generate_experiment_report()

        self.logger.info(f"Experiment completed! Results saved to: {self._get_output_dir()}")

        return self.results

    def _run_single_experiment(self, cancer_type: str, algorithm: str, cv_strategy: str,
                              feature_set: str, random_seed: int) -> Dict[str, Any]:
        """Run a single experimental condition"""

        # Create specific config for this run
        run_config = Config(
            db_path=self.config.db_path,
            cancer_types=[cancer_type],
            algorithm=algorithm,
            feature_set=feature_set,
            cv_strategy=cv_strategy,
            group_column=self.config.group_column,
            institution_aware_test_split=self.config.institution_aware_test_split,
            test_split_group_column=self.config.test_split_group_column,
            train_test_split_file=self.config.train_test_split_file,
            random_state=random_seed,
            cv_folds=self.config.cv_folds,
            min_actionable_events=250,  # Use default threshold
            outcome_column="isOncoKbLevel1",
            n_jobs=self.config.n_jobs
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
                for cv_strategy in self.results[cancer_type][algorithm]:
                    for feature_set in self.results[cancer_type][algorithm][cv_strategy]:
                        runs = self.results[cancer_type][algorithm][cv_strategy][feature_set]

                        # Filter out failed runs
                        successful_runs = [run for run in runs if 'error' not in run]
                        failed_runs = [run for run in runs if 'error' in run]

                        if not successful_runs:
                            self.logger.warning(
                                f"No successful runs for {cancer_type}-{algorithm}-{cv_strategy}-{feature_set}"
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
                        self.results[cancer_type][algorithm][cv_strategy][feature_set] = {
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

        # For each cancer type, algorithm, and cv_strategy, compare feature sets
        for cancer_type in self.results:
            for algorithm in self.results[cancer_type]:
                for cv_strategy in self.results[cancer_type][algorithm]:

                    # Extract AUC values for each feature set
                    feature_set_aucs = {}
                    for feature_set in self.config.feature_sets:
                        if feature_set in self.results[cancer_type][algorithm][cv_strategy]:
                            runs = self.results[cancer_type][algorithm][cv_strategy][feature_set]['individual_runs']
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
                        self.results[cancer_type][algorithm][cv_strategy]['statistical_tests'] = comparisons

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

        # Generate visualizations
        self._create_roc_curves(report_dir)
        self._create_performance_barchart(report_dir)
        self._create_class_imbalance_plots(report_dir)

        self.logger.info(f"Report generated in: {report_dir}")

    def _create_summary_table(self, report_dir: Path):
        """Create summary table of results"""

        rows = []
        for cancer_type in self.results:
            for algorithm in self.results[cancer_type]:
                for cv_strategy in self.results[cancer_type][algorithm]:
                    for feature_set in self.config.feature_sets:
                        if (feature_set in self.results[cancer_type][algorithm][cv_strategy] and
                            'summary' in self.results[cancer_type][algorithm][cv_strategy][feature_set]):

                            summary = self.results[cancer_type][algorithm][cv_strategy][feature_set]['summary']
                            rows.append({
                                'Cancer_Type': cancer_type,
                                'Algorithm': algorithm,
                                'CV_Strategy': cv_strategy,
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
                index=['Cancer_Type', 'Algorithm', 'CV_Strategy'],
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
                    for cv_strategy in self.results[cancer_type][algorithm]:
                        f.write(f"    {cv_strategy} CV:\n")
                        for feature_set in self.config.feature_sets:
                            if (feature_set in self.results[cancer_type][algorithm][cv_strategy] and
                                'summary' in self.results[cancer_type][algorithm][cv_strategy][feature_set]):

                                summary = self.results[cancer_type][algorithm][cv_strategy][feature_set]['summary']
                                f.write(f"      {feature_set}: ")
                                f.write(f"AUC {summary['mean_cv_auc']:.3f} ± {summary['std_cv_auc']:.3f} ")
                                f.write(f"({summary['mean_n_features']:.0f} features)\n")

            # Statistical tests
            f.write("\nStatistical Comparisons:\n")
            f.write("-" * 23 + "\n")

            for cancer_type in self.results:
                for algorithm in self.results[cancer_type]:
                    for cv_strategy in self.results[cancer_type][algorithm]:
                        if 'statistical_tests' in self.results[cancer_type][algorithm][cv_strategy]:
                            f.write(f"\n{cancer_type} - {algorithm} - {cv_strategy} CV:\n")
                            tests = self.results[cancer_type][algorithm][cv_strategy]['statistical_tests']
                            for comparison, result in tests.items():
                                significance = "***" if result['significant'] else "n.s."
                                f.write(f"  {comparison}: ")
                                f.write(f"p={result['p_value']:.3f} {significance} ")
                                f.write(f"(effect size: {result['effect_size']:.3f})\n")

    def _create_roc_curves(self, report_dir: Path):
        """
        Create ROC curves for each cancer type comparing feature sets.

        Args:
            report_dir: Directory to save plots
        """
        self.logger.info("Generating ROC curves...")

        # Need to store predict_proba results - for now, skip if not available
        # This would require modifying analyzer to return probability predictions
        self.logger.warning("ROC curves require probability predictions - not yet implemented")
        self.logger.warning("Will be added in future update after modifying trainer.py")

    def _create_performance_barchart(self, report_dir: Path):
        """
        Create multiple performance visualization plots stratified by different factors.

        Args:
            report_dir: Directory to save plots
        """
        self.logger.info("Generating performance visualizations...")

        # Collect data for plotting
        plot_data = []
        for cancer_type in self.results:
            for algorithm in self.results[cancer_type]:
                for cv_strategy in self.results[cancer_type][algorithm]:
                    for feature_set in self.config.feature_sets:
                        if (feature_set in self.results[cancer_type][algorithm][cv_strategy] and
                            'summary' in self.results[cancer_type][algorithm][cv_strategy][feature_set]):

                            summary = self.results[cancer_type][algorithm][cv_strategy][feature_set]['summary']
                            plot_data.append({
                                'Cancer_Type': cancer_type,
                                'Algorithm': algorithm,
                                'CV_Strategy': cv_strategy,
                                'Feature_Set': feature_set,
                                'AUC': summary['mean_cv_auc'],
                                'AUC_Std': summary['std_cv_auc']
                            })

        if not plot_data:
            self.logger.warning("No data available for performance visualizations")
            return

        df = pd.DataFrame(plot_data)

        # Create multiple visualizations
        self._plot_feature_comparison_by_cv(df, report_dir)
        self._plot_cv_comparison_by_feature(df, report_dir)
        self._plot_heatmap_by_cancer_type(df, report_dir)
        self._plot_algorithm_panels(df, report_dir)

    def _plot_feature_comparison_by_cv(self, df: pd.DataFrame, report_dir: Path):
        """Figure 1: Compare feature sets, grouped by CV strategy"""

        # Color palette for feature sets
        feature_colors = {
            'demographic': '#2E86AB',  # Blue
            'technical': '#A23B72',    # Purple
            'both': '#F18F01'          # Orange
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for idx, cv_strategy in enumerate(['standard', 'group']):
            ax = axes[idx]
            df_cv = df[df['CV_Strategy'] == cv_strategy]

            if df_cv.empty:
                continue

            # Group by cancer type and algorithm, then plot feature sets
            cancer_alg_combos = df_cv.groupby(['Cancer_Type', 'Algorithm']).size().index

            x_pos = np.arange(len(cancer_alg_combos))
            width = 0.25

            for i, feature_set in enumerate(['demographic', 'technical', 'both']):
                df_feature = df_cv[df_cv['Feature_Set'] == feature_set]
                aucs = []
                errors = []

                for cancer, alg in cancer_alg_combos:
                    row = df_feature[(df_feature['Cancer_Type'] == cancer) &
                                    (df_feature['Algorithm'] == alg)]
                    if not row.empty:
                        aucs.append(row['AUC'].values[0])
                        errors.append(row['AUC_Std'].values[0])
                    else:
                        aucs.append(0)
                        errors.append(0)

                ax.bar(x_pos + i * width, aucs, width,
                      label=feature_set.capitalize(),
                      color=feature_colors.get(feature_set, '#999999'),
                      yerr=errors, capsize=3, alpha=0.8)

            # Formatting
            ax.set_xlabel('Cancer Type & Algorithm', fontsize=11, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Cross-Validation AUC', fontsize=11, fontweight='bold')

            cv_label = 'Standard K-Fold' if cv_strategy == 'standard' else 'GroupKFold (Institution-Aware)'
            ax.set_title(f'{cv_label}', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels([f'{c}\n{a[:3]}' for c, a in cancer_alg_combos],
                              rotation=0, ha='center', fontsize=9)
            ax.set_ylim(0.3, 0.7)
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.grid(axis='y', alpha=0.3, linestyle=':')
            ax.legend(title='Feature Set', fontsize=9, title_fontsize=10)

        plt.suptitle('Feature Set Comparison Across CV Strategies',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        outfile = report_dir / "fig1_feature_comparison_by_cv.pdf"
        plt.savefig(outfile, format="pdf", bbox_inches='tight', dpi=300)
        plt.close()
        self.logger.info(f"Figure 1 saved to: {outfile}")

    def _plot_cv_comparison_by_feature(self, df: pd.DataFrame, report_dir: Path):
        """Figure 2: Compare CV strategies, grouped by feature set"""

        # Color palette for CV strategies
        cv_colors = {
            'standard': '#06A77D',     # Green
            'group': '#D1495B'         # Red
        }

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

        for idx, feature_set in enumerate(['demographic', 'technical', 'both']):
            ax = axes[idx]
            df_feature = df[df['Feature_Set'] == feature_set]

            if df_feature.empty:
                continue

            # Group by cancer type and algorithm
            cancer_alg_combos = df_feature.groupby(['Cancer_Type', 'Algorithm']).size().index

            x_pos = np.arange(len(cancer_alg_combos))
            width = 0.35

            for i, cv_strategy in enumerate(['standard', 'group']):
                df_cv = df_feature[df_feature['CV_Strategy'] == cv_strategy]
                aucs = []
                errors = []

                for cancer, alg in cancer_alg_combos:
                    row = df_cv[(df_cv['Cancer_Type'] == cancer) &
                               (df_cv['Algorithm'] == alg)]
                    if not row.empty:
                        aucs.append(row['AUC'].values[0])
                        errors.append(row['AUC_Std'].values[0])
                    else:
                        aucs.append(0)
                        errors.append(0)

                cv_label = 'Standard' if cv_strategy == 'standard' else 'GroupKFold'
                ax.bar(x_pos + i * width, aucs, width,
                      label=cv_label,
                      color=cv_colors.get(cv_strategy, '#999999'),
                      yerr=errors, capsize=3, alpha=0.8)

            # Formatting
            ax.set_xlabel('Cancer Type & Algorithm', fontsize=10, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Cross-Validation AUC', fontsize=11, fontweight='bold')

            ax.set_title(f'{feature_set.capitalize()} Features', fontsize=12, fontweight='bold')
            ax.set_xticks(x_pos + width / 2)
            ax.set_xticklabels([f'{c}\n{a[:3]}' for c, a in cancer_alg_combos],
                              rotation=0, ha='center', fontsize=9)
            ax.set_ylim(0.3, 0.7)
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.grid(axis='y', alpha=0.3, linestyle=':')
            ax.legend(title='CV Strategy', fontsize=9, title_fontsize=10)

        plt.suptitle('CV Strategy Comparison Across Feature Sets',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        outfile = report_dir / "fig2_cv_comparison_by_feature.pdf"
        plt.savefig(outfile, format="pdf", bbox_inches='tight', dpi=300)
        plt.close()
        self.logger.info(f"Figure 2 saved to: {outfile}")

    def _plot_heatmap_by_cancer_type(self, df: pd.DataFrame, report_dir: Path):
        """Figure 3: Heatmap showing AUC across conditions"""

        # Create separate heatmaps for each algorithm
        algorithms = df['Algorithm'].unique()

        for algorithm in algorithms:
            df_alg = df[df['Algorithm'] == algorithm]

            if df_alg.empty:
                continue

            # Create pivot table: rows = cancer types, columns = feature_cv combinations
            pivot_data = []
            cancer_types = df_alg['Cancer_Type'].unique()

            for cancer_type in cancer_types:
                row_data = {'Cancer_Type': cancer_type}
                for feature_set in ['demographic', 'technical', 'both']:
                    for cv_strategy in ['standard', 'group']:
                        subset = df_alg[(df_alg['Cancer_Type'] == cancer_type) &
                                       (df_alg['Feature_Set'] == feature_set) &
                                       (df_alg['CV_Strategy'] == cv_strategy)]
                        if not subset.empty:
                            cv_label = 'Std' if cv_strategy == 'standard' else 'Grp'
                            col_name = f'{feature_set[:4].capitalize()}\n{cv_label}'
                            row_data[col_name] = subset['AUC'].values[0]
                        else:
                            row_data[col_name] = np.nan

                pivot_data.append(row_data)

            pivot_df = pd.DataFrame(pivot_data)
            pivot_df.set_index('Cancer_Type', inplace=True)

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, max(4, len(cancer_types) * 0.6)))

            # Use diverging colormap centered at 0.5 (random performance)
            im = ax.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto',
                          vmin=0.3, vmax=0.7, interpolation='nearest')

            # Set ticks and labels
            ax.set_xticks(np.arange(len(pivot_df.columns)))
            ax.set_yticks(np.arange(len(pivot_df.index)))
            ax.set_xticklabels(pivot_df.columns, fontsize=9)
            ax.set_yticklabels(pivot_df.index, fontsize=10)

            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Add values in cells
            for i in range(len(pivot_df.index)):
                for j in range(len(pivot_df.columns)):
                    val = pivot_df.values[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val < 0.45 or val > 0.6 else 'black'
                        ax.text(j, i, f'{val:.3f}', ha="center", va="center",
                               color=text_color, fontsize=8, fontweight='bold')

            ax.set_title(f'AUC Performance Heatmap - {algorithm.replace("_", " ").title()}',
                        fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Feature Set & CV Strategy', fontsize=10, fontweight='bold')
            ax.set_ylabel('Cancer Type', fontsize=10, fontweight='bold')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('AUC', rotation=270, labelpad=15, fontsize=10, fontweight='bold')

            plt.tight_layout()

            outfile = report_dir / f"fig3_heatmap_{algorithm}.pdf"
            plt.savefig(outfile, format="pdf", bbox_inches='tight', dpi=300)
            plt.close()
            self.logger.info(f"Figure 3 ({algorithm}) saved to: {outfile}")

    def _plot_algorithm_panels(self, df: pd.DataFrame, report_dir: Path):
        """Figure 4: Faceted plot stratified by algorithm"""

        algorithms = df['Algorithm'].unique()

        if len(algorithms) < 2:
            self.logger.info("Skipping algorithm panel plot (only one algorithm)")
            return

        fig, axes = plt.subplots(len(algorithms), 2, figsize=(14, 5 * len(algorithms)),
                                sharey=True, sharex=True)

        if len(algorithms) == 1:
            axes = axes.reshape(1, -1)

        for alg_idx, algorithm in enumerate(algorithms):
            df_alg = df[df['Algorithm'] == algorithm]

            for cv_idx, cv_strategy in enumerate(['standard', 'group']):
                ax = axes[alg_idx, cv_idx]
                df_subset = df_alg[df_alg['CV_Strategy'] == cv_strategy]

                if df_subset.empty:
                    continue

                # Group by cancer type
                cancer_types = df_subset['Cancer_Type'].unique()
                x_pos = np.arange(len(cancer_types))
                width = 0.25

                feature_colors = {
                    'demographic': '#2E86AB',
                    'technical': '#A23B72',
                    'both': '#F18F01'
                }

                for i, feature_set in enumerate(['demographic', 'technical', 'both']):
                    df_feature = df_subset[df_subset['Feature_Set'] == feature_set]
                    aucs = []
                    errors = []

                    for cancer in cancer_types:
                        row = df_feature[df_feature['Cancer_Type'] == cancer]
                        if not row.empty:
                            aucs.append(row['AUC'].values[0])
                            errors.append(row['AUC_Std'].values[0])
                        else:
                            aucs.append(0)
                            errors.append(0)

                    ax.bar(x_pos + i * width, aucs, width,
                          label=feature_set.capitalize(),
                          color=feature_colors.get(feature_set, '#999999'),
                          yerr=errors, capsize=3, alpha=0.8)

                # Formatting
                cv_label = 'Standard K-Fold' if cv_strategy == 'standard' else 'GroupKFold'
                ax.set_title(f'{algorithm.replace("_", " ").title()} - {cv_label}',
                           fontsize=11, fontweight='bold')

                if alg_idx == len(algorithms) - 1:
                    ax.set_xlabel('Cancer Type', fontsize=10, fontweight='bold')
                    ax.set_xticks(x_pos + width)
                    ax.set_xticklabels(cancer_types, rotation=45, ha='right', fontsize=9)

                if cv_idx == 0:
                    ax.set_ylabel('Cross-Validation AUC', fontsize=10, fontweight='bold')

                ax.set_ylim(0.3, 0.7)
                ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                ax.grid(axis='y', alpha=0.3, linestyle=':')

                if alg_idx == 0 and cv_idx == 1:
                    ax.legend(title='Feature Set', fontsize=9, title_fontsize=10,
                            loc='upper right')

        plt.suptitle('Algorithm-Stratified Performance Comparison',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        outfile = report_dir / "fig4_algorithm_panels.pdf"
        plt.savefig(outfile, format="pdf", bbox_inches='tight', dpi=300)
        plt.close()
        self.logger.info(f"Figure 4 saved to: {outfile}")

    def _create_class_imbalance_plots(self, report_dir: Path):
        """
        Create scatter plots showing relationship between class imbalance and performance.

        Args:
            report_dir: Directory to save plots
        """
        self.logger.info("Generating class imbalance plots...")

        # Collect data
        plot_data = []
        for cancer_type in self.results:
            for algorithm in self.results[cancer_type]:
                for cv_strategy in self.results[cancer_type][algorithm]:
                    for feature_set in self.config.feature_sets:
                        if (feature_set in self.results[cancer_type][algorithm][cv_strategy] and
                            'summary' in self.results[cancer_type][algorithm][cv_strategy][feature_set]):

                            summary = self.results[cancer_type][algorithm][cv_strategy][feature_set]['summary']
                            plot_data.append({
                                'Cancer_Type': cancer_type,
                                'Algorithm': algorithm,
                                'CV_Strategy': cv_strategy,
                                'Feature_Set': feature_set,
                                'AUC': summary['mean_cv_auc'],
                                'F1': summary.get('mean_test_f1', 0),
                                'Actionable_Rate': summary['mean_actionable_rate']
                            })

        if not plot_data:
            self.logger.warning("No data available for class imbalance plots")
            return

        df = pd.DataFrame(plot_data)

        # Plot 1: Actionable Rate vs AUC
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(df['Actionable_Rate'], df['AUC'], s=100, alpha=0.6)

        # Add labels
        for _, row in df.iterrows():
            label = f"{row['Cancer_Type']}\n{row['Feature_Set'][:4]}"
            ax.text(row['Actionable_Rate'] + 0.01, row['AUC'], label, fontsize=7)

        ax.set_xlabel('Proportion Actionable', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('Class Imbalance vs AUC Performance', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        outfile = report_dir / "class_imbalance_vs_auc.pdf"
        plt.savefig(outfile, format="pdf", bbox_inches='tight')
        plt.close()

        self.logger.info(f"Class imbalance vs AUC plot saved to: {outfile}")

        # Plot 2: Actionable Rate vs F1
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(df['Actionable_Rate'], df['F1'], s=100, alpha=0.6)

        # Add labels
        for _, row in df.iterrows():
            label = f"{row['Cancer_Type']}\n{row['Feature_Set'][:4]}"
            ax.text(row['Actionable_Rate'] + 0.01, row['F1'], label, fontsize=7)

        ax.set_xlabel('Proportion Actionable', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('Class Imbalance vs F1 Performance', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        outfile = report_dir / "class_imbalance_vs_f1.pdf"
        plt.savefig(outfile, format="pdf", bbox_inches='tight')
        plt.close()

        self.logger.info(f"Class imbalance vs F1 plot saved to: {outfile}")