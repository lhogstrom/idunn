#!/usr/bin/env python3
"""Compare demographic vs. technical feature sets across cancer types and algorithms.

Usage: python run_feature_set_experiment.py [options]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

from oncokb_analyzer.experiments import ExperimentConfig, FeatureSetExperiment
from db_config import get_default_db_path
from feature_config import load_feature_config, copy_config_to_run_dir, get_config_path


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run feature set comparison experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default=get_default_db_path(),
        help="Path to SQLite database file (default: from BALDER_DB_PATH env var or ../../../output/...)"
    )

    parser.add_argument(
        "--cancer-types",
        type=str,
        nargs="+",
        default=["LUAD", "BRCA"],
        help="Cancer types to analyze (OncoTree codes)"
    )

    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["random_forest", "logistic"],
        choices=["random_forest", "logistic"],
        help="Machine learning algorithms to compare"
    )

    parser.add_argument(
        "--feature-sets",
        type=str,
        nargs="+",
        default=["demographic", "technical", "both"],
        choices=["demographic", "technical", "both"],
        help="Feature sets to compare"
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )

    parser.add_argument(
        "--cv-strategies",
        type=str,
        nargs="+",
        default=["standard", "group"],
        choices=["standard", "group", "stratified_group"],
        help="Cross-validation strategies to compare"
    )

    parser.add_argument(
        "--group-column",
        type=str,
        default="SUBJECT_CENTER",
        help="Column name for GroupKFold grouping (e.g., SUBJECT_CENTER, SEQ_ASSAY_ID)"
    )

    parser.add_argument(
        "--institution-aware-test-split",
        type=str,
        nargs='?',
        const="SUBJECT_CENTER",
        default=None,
        metavar="GROUP_COLUMN",
        help=(
            "Enable institution-aware test splits using GroupShuffleSplit. "
            "Optionally specify the grouping column (default: SUBJECT_CENTER). "
            "This prevents institutional overlap between train and test sets, "
            "providing more realistic generalization estimates."
        )
    )

    parser.add_argument(
        "--train-test-split-file",
        type=str,
        default=None,
        help=(
            "Path to train/test split CSV file. If provided, only training data "
            "will be used for feature set comparison. File should have columns: "
            "PATIENT_ID, split (with 'train'/'test' values). "
            "Default location: ../../analysis_output/train_test_split/train_test_split.csv"
        )
    )

    parser.add_argument(
        "--n-random-seeds",
        type=int,
        default=5,
        help="Number of random seeds for robustness testing"
    )

    parser.add_argument(
        "--random-seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 999],
        help="Specific random seeds to use"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../ml_experiments/",
        help="Output directory for experiment results"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="",
        help="Custom experiment name (auto-generated if not provided)"
    )

    parser.add_argument(
        "--no-statistical-tests",
        action="store_true",
        help="Skip statistical testing between feature sets"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests"
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for cross-validation (-1 uses all cores, 1 for single-threaded to save memory)"
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal settings (1 cancer type, 1 algorithm, 2 seeds)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show experiment configuration without running"
    )

    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt and run immediately"
    )

    parser.add_argument(
        "--feature-config",
        type=str,
        default=None,
        help=(
            "Path to custom feature configuration YAML file. "
            "Default: code/config/features.yaml. "
            "Can also be set via BALDER_FEATURES_CONFIG environment variable."
        )
    )

    return parser.parse_args()


def create_experiment_config(args) -> ExperimentConfig:
    """Build an ExperimentConfig from parsed arguments."""

    institution_aware = args.institution_aware_test_split is not None
    test_split_group_col = args.institution_aware_test_split if institution_aware else "SUBJECT_CENTER"

    if args.quick_test:
        config = ExperimentConfig(
            db_path=args.db_path,
            cancer_types=[args.cancer_types[0]],
            algorithms=[args.algorithms[0]],
            feature_sets=args.feature_sets,
            cv_folds=3,
            cv_strategies=args.cv_strategies,
            group_column=args.group_column,
            institution_aware_test_split=institution_aware,
            test_split_group_column=test_split_group_col,
            train_test_split_file=args.train_test_split_file,
            n_random_seeds=2,
            random_seeds=[42, 123],
            n_jobs=args.n_jobs,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name or "",
            statistical_tests=not args.no_statistical_tests,
            alpha=args.alpha
        )
    else:
        config = ExperimentConfig(
            db_path=args.db_path,
            cancer_types=args.cancer_types,
            algorithms=args.algorithms,
            feature_sets=args.feature_sets,
            cv_folds=args.cv_folds,
            cv_strategies=args.cv_strategies,
            group_column=args.group_column,
            institution_aware_test_split=institution_aware,
            test_split_group_column=test_split_group_col,
            train_test_split_file=args.train_test_split_file,
            n_random_seeds=args.n_random_seeds,
            random_seeds=args.random_seeds,
            n_jobs=args.n_jobs,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            statistical_tests=not args.no_statistical_tests,
            alpha=args.alpha
        )

    return config


def print_experiment_summary(config: ExperimentConfig):
    """Print the experiment design summary."""
    print("Feature Set Comparison Experiment")
    print("=" * 50)
    print(f"Experiment Name: {config.experiment_name}")
    print(f"Database: {config.db_path}")
    print(f"Feature Config: {get_config_path()}")
    print(f"Cancer Types: {', '.join(config.cancer_types)}")
    print(f"Algorithms: {', '.join(config.algorithms)}")
    print(f"Feature Sets: {', '.join(config.feature_sets)}")
    print(f"CV Strategies: {', '.join(config.cv_strategies)}")
    print(f"Group Column (CV): {config.group_column}")
    print(f"Institution-Aware Test Split: {'Yes (' + config.test_split_group_column + ')' if config.institution_aware_test_split else 'No'}")
    print(f"Train/Test Split File: {config.train_test_split_file or 'None (using all data)'}")
    print(f"CV Folds: {config.cv_folds}")
    print(f"Random Seeds: {config.n_random_seeds} seeds")
    print(f"Parallel Jobs: {config.n_jobs} (-1 = all cores)")
    print(f"Statistical Tests: {'Yes' if config.statistical_tests else 'No'}")
    print(f"Output Directory: {config.output_dir}")

    total_experiments = (
        len(config.cancer_types) *
        len(config.algorithms) *
        len(config.cv_strategies) *
        len(config.feature_sets) *
        config.n_random_seeds
    )
    print(f"Total Experiments: {total_experiments}")

    # rough estimate: ~7 s per experiment
    estimated_minutes = (total_experiments * 7) / 60
    print(f"Estimated Time: ~{estimated_minutes:.1f} minutes")
    print()


def main():
    """Entry point."""

    args = parse_arguments()

    if args.feature_config:
        print(f"Loading custom feature config: {args.feature_config}")
        load_feature_config(args.feature_config)
    else:
        load_feature_config()
    print(f"Feature config loaded from: {get_config_path()}")

    try:
        config = create_experiment_config(args)
        config.validate()
    except Exception as e:
        print(f"ERROR: Invalid configuration: {e}")
        return 1

    print_experiment_summary(config)

    if args.dry_run:
        print("Dry run mode - configuration validated successfully!")
        print("Use --no-dry-run or remove --dry-run to execute the experiment.")
        return 0

    if not args.quick_test:
        total_experiments = (
            len(config.cancer_types) *
            len(config.algorithms) *
            len(config.cv_strategies) *
            len(config.feature_sets) *
            config.n_random_seeds
        )

        if total_experiments > 20:
            if not args.yes:
                response = input(f"This will run {total_experiments} experiments. Continue? [y/N]: ")
                if response.lower() not in ['y', 'yes']:
                    print("Experiment cancelled.")
                    return 0
            else:
                print(f"Running {total_experiments} experiments (confirmation skipped with --yes flag)...")

    try:
        print("Starting experiment...")
        experiment = FeatureSetExperiment(config)
        results = experiment.run_full_experiment()

        output_path = Path(config.output_dir) / config.experiment_name
        copied_config = copy_config_to_run_dir(str(output_path), args.feature_config)
        print(f"Feature config copied to: {copied_config}")

        print("\nExperiment completed successfully!")
        print(f"Results saved to: {config.output_dir}/{config.experiment_name}/")

        print("\nQuick Summary:")
        print("-" * 15)
        for cancer_type in results:
            print(f"{cancer_type}:")
            for algorithm in results[cancer_type]:
                print(f"  {algorithm}:")
                for cv_strategy in results[cancer_type][algorithm]:
                    print(f"    {cv_strategy} CV:")
                    for feature_set in config.feature_sets:
                        if (feature_set in results[cancer_type][algorithm][cv_strategy] and
                            'summary' in results[cancer_type][algorithm][cv_strategy][feature_set]):
                            summary = results[cancer_type][algorithm][cv_strategy][feature_set]['summary']
                            print(f"      {feature_set}: AUC {summary['mean_cv_auc']:.3f} ± {summary['std_cv_auc']:.3f}")

        return 0

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        return 1

    except Exception as e:
        print(f"ERROR: Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
