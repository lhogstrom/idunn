#!/usr/bin/env python3
"""
Master Analysis Script for Balder Project

Orchestrates execution of all analysis scripts and ML experiments in scientifically
sound order. Handles environment switching, dependency management, and error reporting.

Scientific Workflow Order (January 2026):
    TIER 1: Independent Analyses (demographics, institutional, feature engineering)
    TIER 2: Data Setup (multi-tier groupings + train/test split)
    TIER 3: Hyperparameter Optimization (training data only)
    TIER 4: Feature Set Comparison (training data only)
    TIER 5: Final Model Training & Test Set Evaluation
    TIER 6: SHAP Analysis
    TIER 7: Post-Hoc Statistical Analysis (based on SHAP features)

    IMPORTANT: Test set (20%) is held out BEFORE hyperparameter optimization.
    All model development uses ONLY the 80% training data.

Usage:
    python run_all_analyses.py --full              # Run everything (~5 hours)
    python run_all_analyses.py --quick             # Quick test run (~30 min, skips SHAP)
    python run_all_analyses.py --ml-only           # Only ML experiments (TIER 2-5)
    python run_all_analyses.py --analysis-only     # Only analysis scripts (TIER 1, 7)
    python run_all_analyses.py --post-model-selection      # Only downstream of Phase 2b (TIER 4-7, includes SHAP primary)
    python run_all_analyses.py --dry-run           # Show what would run
    python run_all_analyses.py --list              # List all scripts

Note: --quick mode automatically skips ALL SHAP analysis (TIER 6) to prevent memory issues.
Note: --post-model-selection runs SHAP primary values (6a) but skips SHAP interactions (6b).
"""

import argparse
import subprocess
import sys
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'tools'))
from db_config import get_default_db_path
from feature_config import load_feature_config, copy_config_to_run_dir, get_config_path

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ScriptConfig:
    """Configuration for a single analysis script"""

    def __init__(self, path: str, env: str, runtime: str,
                 tier: int, args: List[str] = None, depends_on: List[str] = None):
        self.path = path
        self.env = env  # 'primary' or 'ml'
        self.runtime = runtime  # 'quick', 'medium', 'long'
        self.tier = tier  # 1, 2, 3, 4
        self.args = args or []
        self.depends_on = depends_on or []
        self.status = 'pending'  # 'pending', 'running', 'success', 'failed', 'skipped'
        self.duration = None
        self.error_message = None

class MasterAnalysis:
    """Master orchestrator for all Balder analyses"""

    def __init__(self, mode: str = 'full', dry_run: bool = False,
                 continue_on_error: bool = True, skip_shap: bool = False,
                 skip_shap_interactions: bool = False, parallel: bool = False,
                 feature_config_path: str = None, institution_weight_alpha: float = 0.0,
                 base_run_dir: str = None, algorithm: str = 'random_forest',
                 lr_significant_only: bool = False):
        self.mode = mode
        self.dry_run = dry_run
        self.continue_on_error = continue_on_error
        self.skip_shap = skip_shap  # Skip ALL SHAP (both 6a and 6b)
        self.skip_shap_interactions = skip_shap_interactions  # Skip only 6b (interactions)
        self.parallel = parallel
        self.feature_config_path = feature_config_path  # Custom feature config path
        self.institution_weight_alpha = institution_weight_alpha  # Power weighting for TIER 5
        self.base_run_dir = Path(base_run_dir).resolve() if base_run_dir else None  # Existing run to draw from
        self.algorithm = algorithm  # ML algorithm: 'random_forest' or 'logistic'
        self.lr_significant_only = lr_significant_only  # Filter LR rankings to Wald-significant only

        # Get repository root
        self.repo_root = Path(__file__).parent.parent

        # Create unified timestamped run directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.repo_root / "runs" / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create tier subdirectories
        self.tier_dirs = {
            1: self.run_dir / "tier1_independent_analyses",
            2: self.run_dir / "tier2_data_setup",
            3: self.run_dir / "tier3_hyperparameter_optimization",
            4: self.run_dir / "tier4_feature_comparison",
            5: self.run_dir / "tier5_final_models",
            6: self.run_dir / "tier6_shap_analysis",
            7: self.run_dir / "tier7_posthoc_statistics"
        }
        for tier_dir in self.tier_dirs.values():
            tier_dir.mkdir(parents=True, exist_ok=True)

        # Log file in run directory
        self.log_file = self.run_dir / "run.log"

        # Database path (from BALDER_DB_PATH env var or default)
        self.db_path = get_default_db_path()

        # Load feature configuration
        load_feature_config(self.feature_config_path)
        self.feature_config_loaded = get_config_path()

        # Copy feature config to run directory for reproducibility
        copy_config_to_run_dir(str(self.run_dir), self.feature_config_path)

        # Validate base_run_dir if specified
        if self.base_run_dir:
            if not self.base_run_dir.exists():
                raise ValueError(f"Base run directory not found: {self.base_run_dir}")
            print(f"{Colors.OKCYAN}Using base run directory: {self.base_run_dir}{Colors.ENDC}")

        # Conda environments
        self.primary_env = "balder_conda_env"
        self.ml_env = "balder_ml_env"

        # Script registry
        self.scripts = self._define_scripts()

        # Results tracking
        self.results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }

    def _get_train_test_split_file(self) -> str:
        """Get train-test split file path, using base_run_dir if specified."""
        if self.base_run_dir:
            split_file = self.base_run_dir / 'tier2_data_setup' / 'train_test_split' / 'train_test_split.csv'
            if split_file.exists():
                return str(split_file)
            else:
                print(f"{Colors.WARNING}Warning: Train-test split not found in base run: {split_file}{Colors.ENDC}")
                print(f"{Colors.WARNING}Falling back to default location{Colors.ENDC}")

        # Default for post-model-selection mode
        if self.mode == 'post-model-selection':
            return '../../analysis_output/train_test_split/train_test_split.csv'

        # Default: use this run's tier2 directory
        return str(self.tier_dirs[2] / 'train_test_split' / 'train_test_split.csv')

    def _get_hyperparams_file(self) -> Optional[str]:
        """Get hyperparameters file from base_run_dir if available."""
        if self.base_run_dir:
            # Look for hyperparameter results in tier3
            tier3_dir = self.base_run_dir / 'tier3_hyperparameter_optimization'
            if tier3_dir.exists():
                # Find the most recent run subdirectory
                run_dirs = sorted([d for d in tier3_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
                if run_dirs:
                    latest_run = run_dirs[-1]
                    # Look for algorithm-specific universal params
                    params_filename = f'{self.algorithm}_universal_params.json'
                    params_file = latest_run / params_filename
                    if params_file.exists():
                        return str(params_file)
        return None

    def _get_tier5_model_dir(self) -> str:
        """Get TIER 5 model directory, using base_run_dir if specified for SHAP."""
        # For SHAP analysis, we need the models from TIER 5
        # If running fresh TIER 5, use this run's directory
        # If base_run_dir specified and we're not running TIER 5, use base models
        return str(self.tier_dirs[5])

    def _define_scripts(self) -> List[ScriptConfig]:
        """Define all analysis scripts with their configurations.

        Scientific Workflow Order (January 2026 Reorganization):
        - TIER 1: Independent Analyses (demographics, institutional, feature engineering)
        - TIER 2a: Multi-Tier Grouping Definition (OncoTree hierarchy, patient groupings)
        - TIER 2b: Train/Test Split Definition (80/20 stratified by cancer type)
        - TIER 3: Hyperparameter Optimization (training data only)
        - TIER 4: Feature Set Comparison (training data only)
        - TIER 5: Final Model Training & Test Set Evaluation
        - TIER 6: SHAP Analysis
        - TIER 7: Post-Hoc Statistical Analysis (based on SHAP features)

        IMPORTANT: Test set is held out BEFORE hyperparameter optimization to prevent
        data leakage. All model development (TIER 3-4) uses only training data.
        """

        scripts = []

        # Helper to get relative output dir path from script location
        def get_output_dir_for_tier(tier: int, script_path: str) -> str:
            """Calculate relative path to tier directory from script location."""
            tier_dir = self.tier_dirs[tier]
            # Scripts run from repo root, so use path relative to repo root
            return str(tier_dir.relative_to(self.repo_root))

        # TIER 1: Independent analysis scripts (can run in parallel)
        if self.mode in ['full', 'quick', 'analysis-only']:
            tier1_dir = get_output_dir_for_tier(1, "code/analysis/")
            scripts.extend([
                ScriptConfig(
                    path="code/analysis/cancer_type_summary_stats.py",
                    env="primary",
                    runtime="quick",
                    tier=1,
                    args=['--output-dir', tier1_dir + '/cancer_type_summary_stats']
                ),
                ScriptConfig(
                    path="code/analysis/demographics/analyze_demographics_by_cancer_type.py",
                    env="primary",
                    runtime="quick",
                    tier=1,
                    args=['--output-dir', tier1_dir + '/demographics']
                ),
                ScriptConfig(
                    path="code/analysis/institutional/analyze_all_cancer_types_institutional_stats.py",
                    env="primary",
                    runtime="quick",
                    tier=1,
                    args=['--output-dir', tier1_dir + '/institutional']
                ),
                ScriptConfig(
                    path="code/analysis/feature_engineering_analysis.py",
                    env="primary",
                    runtime="medium",
                    tier=1,
                    args=['--output-dir', tier1_dir + '/feature_engineering']
                ),
                ScriptConfig(
                    path="code/analysis/variants/plot_variant_demographic_stratification.py",
                    env="primary",
                    runtime="medium",
                    tier=1,
                    args=['--output-dir', tier1_dir + '/variants']
                ),
            ])

        # TIER 2a: Multi-Tier Grouping Definition
        if self.mode in ['full', 'quick', 'ml-only']:
            tier2_dir = get_output_dir_for_tier(2, "code/analysis/")
            scripts.extend([
                ScriptConfig(
                    path="code/analysis/oncotree_grouping_analysis.py",
                    env="primary",
                    runtime="quick",
                    tier=2,
                    args=['--output-dir', tier2_dir + '/oncotree_grouping']
                ),
                ScriptConfig(
                    path="code/analysis/define_multi_tier_groupings_v3.py",
                    env="primary",
                    runtime="quick",
                    tier=2,
                    args=['--input-dir', tier2_dir + '/oncotree_grouping',
                          '--output-dir', tier2_dir + '/multi_tier_groupings']
                ),
            ])

        # TIER 2b: Train/Test Split Definition (MUST run before any modeling)
        if self.mode in ['full', 'quick', 'ml-only']:
            tier2_dir = get_output_dir_for_tier(2, "code/analysis/")
            scripts.append(
                ScriptConfig(
                    path="code/analysis/define_train_test_split.py",
                    env="primary",
                    runtime="quick",
                    tier=2,
                    args=['--output-dir', tier2_dir + '/train_test_split']
                )
            )

        # TIER 3: Hyperparameter Optimization & Model Selection (TRAINING DATA ONLY)
        if self.mode in ['full', 'quick', 'ml-only']:
            tier3_dir = get_output_dir_for_tier(3, "code/modeling/")
            # Use train_test_split from tier2 directory within this run (or base_run_dir if specified)
            train_split_file = self._get_train_test_split_file()
            phase2b_args = [
                '--n-jobs', '1',
                '--sample-fraction', '0.5',
                '--train-test-split-file', train_split_file,
                '--output-dir', tier3_dir
            ]
            if self.mode == 'quick':
                phase2b_args.append('--quick-test')

            scripts.append(
                ScriptConfig(
                    path="code/modeling/run_ml_model_selection_full.py",
                    env="primary",
                    runtime="long" if self.mode == 'full' else "medium",
                    tier=3,
                    args=phase2b_args,
                    depends_on=["define_train_test_split.py"] if not self.base_run_dir else []
                )
            )

        # TIER 4: Feature Set Comparison (training data only)
        # NOTE: Using demographic features only by default to avoid memory explosion
        # from one-hot encoding of high-cardinality categoricals (SEQ_ASSAY_ID: 98 values,
        # SEQ_PIPELINE_ID: 67 values). The "both" feature set expands to 300-400 columns
        # and can require 40-100+ GB RAM. Run manually with --feature-sets both if needed.
        if self.mode in ['full', 'quick', 'ml-only', 'post-model-selection']:
            tier4_dir = get_output_dir_for_tier(4, "code/modeling/")
            # Use helper method to get train_split_file (respects base_run_dir)
            train_split_file = self._get_train_test_split_file()

            # Use selected algorithm, demographic features only (memory-safe)
            ml_args = [
                '--cancer-types', 'LUAD', 'BRCA', 'MEL', 'GIST', 'COAD',
                '--algorithms', self.algorithm,
                '--feature-sets', 'demographic',  # Memory-safe: avoids one-hot explosion
                '--institution-aware-test-split',
                '--n-random-seeds', '3',
                '--n-jobs', '1',  # Limit parallelism to prevent memory issues
                '--train-test-split-file', train_split_file,
                '--output-dir', tier4_dir,
                '--yes'  # Skip confirmation prompt when running via subprocess
            ]
            # Add custom feature config if specified
            if self.feature_config_path:
                ml_args.extend(['--feature-config', self.feature_config_path])

            if self.mode == 'quick':
                ml_args = [
                    '--quick-test',
                    '--n-jobs', '1',  # Limit parallelism to prevent memory issues
                    '--train-test-split-file', train_split_file,
                    '--output-dir', tier4_dir
                ]
                if self.feature_config_path:
                    ml_args.extend(['--feature-config', self.feature_config_path])

            scripts.append(
                ScriptConfig(
                    path="code/modeling/run_feature_set_experiment.py",
                    env="primary",
                    runtime="long" if self.mode == 'full' else "medium",
                    tier=4,
                    args=ml_args,
                    depends_on=["run_ml_model_selection_full.py"] if self.mode != 'post-model-selection' else []
                )
            )

        # TIER 5: Final Model Training & Test Set Evaluation
        # Trains final Random Forest models on training data, evaluates on held-out test set (20%)
        # This is the FIRST and ONLY time the test set is used
        if self.mode in ['full', 'quick', 'ml-only', 'post-model-selection']:
            tier5_dir = get_output_dir_for_tier(5, "code/modeling/")
            # Use helper method to get train_split_file (respects base_run_dir)
            train_split_file = self._get_train_test_split_file()

            tier5_args = [
                '--train-test-split-file', train_split_file,
                '--feature-set', 'both',
                '--output-dir', tier5_dir,
                '--algorithm', self.algorithm
            ]
            # Add institution weight alpha if specified (for overfitting reduction)
            if self.institution_weight_alpha > 0:
                tier5_args.extend(['--institution-weight-alpha', str(self.institution_weight_alpha)])
            # Add custom feature config if specified
            if self.feature_config_path:
                tier5_args.extend(['--feature-config', self.feature_config_path])
            if self.mode == 'quick':
                tier5_args.append('--quick-test')

            scripts.append(
                ScriptConfig(
                    path="code/modeling/run_final_model_training.py",
                    env="primary",
                    runtime="long" if self.mode == 'full' else "medium",
                    tier=5,
                    args=tier5_args,
                    depends_on=["run_feature_set_experiment.py"]
                )
            )

        # TIER 6: Feature Importance Analysis
        # RF pathway: SHAP analysis (TIER 6a primary values, 6b interactions)
        # LR pathway: Coefficient extraction (replaces SHAP)
        # Include 'quick' mode for LR pathway (fast extraction, no memory concerns)
        tier6_modes = ['full', 'post-model-selection']
        if self.algorithm == 'logistic':
            tier6_modes.append('quick')
        if self.mode in tier6_modes and not self.skip_shap:
            # Use absolute paths for TIER 6 (scripts resolve relative paths from script directory)
            tier6_abs_dir = str(self.tier_dirs[6])
            tier5_model_dir = str(self.tier_dirs[5])
            # Use helper method to get train_split_file (respects base_run_dir)
            # Need absolute path for scripts
            train_split_file = self._get_train_test_split_file()
            if not train_split_file.startswith('/'):
                train_split_file = str(self.repo_root / train_split_file)

            if self.algorithm == 'random_forest':
                # RF pathway: SHAP analysis
                # TIER 6a: Primary SHAP values (feature importance rankings)
                shap_primary_args = [
                    '--model-dir', tier5_model_dir,
                    '--train-test-split-file', train_split_file,
                    '--output-dir', tier6_abs_dir + '/primary_values'
                ]

                scripts.append(
                    ScriptConfig(
                        path="code/analysis/shap_interactions/compute_shap_primary_values.py",
                        env="ml",  # Requires SHAP library (available in balder_ml_env)
                        runtime="medium",
                        tier=6,
                        args=shap_primary_args,
                        depends_on=["run_final_model_training.py"]
                    )
                )

                # TIER 6b: SHAP interactions (pairwise feature interactions)
                if not self.skip_shap_interactions:
                    shap_interaction_args = [
                        '--cancer-types', 'LUAD', 'BRCA', 'MEL', 'GIST',
                        '--output-dir', tier6_abs_dir + '/interactions'
                    ]

                    scripts.append(
                        ScriptConfig(
                            path="code/analysis/shap_interactions/compute_shap_interactions.py",
                            env="ml",
                            runtime="long",
                            tier=6,
                            args=shap_interaction_args
                        )
                    )

            elif self.algorithm == 'logistic':
                # LR pathway: Coefficient extraction (replaces SHAP)
                lr_analysis_dir = tier6_abs_dir + '/lr_analysis'
                lr_args = [
                    '--model-dir', tier5_model_dir,
                    '--train-test-split-file', train_split_file,
                    '--output-dir', lr_analysis_dir,
                    '--no-timestamp-subdir'
                ]

                scripts.append(
                    ScriptConfig(
                        path="code/analysis/posthoc/extract_lr_coefficients.py",
                        env="primary",  # No SHAP dependency — runs in primary env
                        runtime="quick",  # LR extraction is fast
                        tier=6,
                        args=lr_args,
                        depends_on=["run_final_model_training.py"]
                    )
                )
                # No SHAP interactions for LR pathway

        # TIER 7: Post-Hoc Statistical Analysis (based on feature importance rankings)
        if self.mode in ['full', 'quick', 'analysis-only', 'post-model-selection']:
            tier7_dir = get_output_dir_for_tier(7, "code/analysis/")
            tier6_dir = get_output_dir_for_tier(6, "code/analysis/")

            # Determine feature rankings file and dependencies based on algorithm
            if self.algorithm == 'random_forest':
                rankings_file = tier6_dir + '/primary_values/combined_feature_importance.csv'
                rankings_depends = ["compute_shap_primary_values.py"]
            elif self.algorithm == 'logistic':
                rankings_file = str(self.tier_dirs[6]) + '/lr_analysis/combined_lr_grouped_importance.csv'
                rankings_depends = ["extract_lr_coefficients.py"]
            else:
                rankings_file = None
                rankings_depends = []

            # Age analysis doesn't depend on feature rankings format
            tier6_dep_script = rankings_depends[0] if rankings_depends else None
            scripts.append(
                ScriptConfig(
                    path="code/analysis/posthoc/analyze_age_by_actionability.py",
                    env="primary",
                    runtime="quick",
                    tier=7,
                    args=['--output-dir', tier7_dir + '/age_actionability'],
                    depends_on=rankings_depends if not self.skip_shap else []
                )
            )

            # Feature associations - pass rankings file from TIER 6 if available
            shap_assoc_args = ['--output-dir', tier7_dir + '/shap_associations']

            # If TIER 6 ran, pass the output file explicitly
            if self.mode in tier6_modes and not self.skip_shap and rankings_file:
                shap_assoc_args.extend(['--shap-file', rankings_file])
                shap_depends = rankings_depends
            else:
                # In quick/analysis-only modes, let the script find existing files
                shap_depends = []

            # Pass --lr-significant-only to post-hoc scripts if set
            if self.lr_significant_only:
                shap_assoc_args.append('--lr-significant-only')

            scripts.append(
                ScriptConfig(
                    path="code/analysis/posthoc/analyze_shap_feature_associations.py",
                    env="primary",
                    runtime="quick",
                    tier=7,
                    args=shap_assoc_args,
                    depends_on=shap_depends
                )
            )

            # SHAP feature rank visualization - only for RF pathway (SHAP-specific plots)
            if self.mode in tier6_modes and not self.skip_shap and self.algorithm == 'random_forest':
                tier6_abs_dir = str(self.tier_dirs[6])
                shap_ranks_args = [
                    '--shap-dir', tier6_abs_dir + '/primary_values',
                    '--output-dir', str(self.tier_dirs[7]) + '/shap_ranks'
                ]

                scripts.append(
                    ScriptConfig(
                        path="code/analysis/posthoc/plot_shap_feature_ranks.py",
                        env="primary",
                        runtime="quick",
                        tier=7,
                        args=shap_ranks_args,
                        depends_on=["compute_shap_primary_values.py"]
                    )
                )

            # Post-hoc visualizations - publication-ready figures
            scripts.append(
                ScriptConfig(
                    path="code/analysis/posthoc/visualize_posthoc_results.py",
                    env="primary",
                    runtime="quick",
                    tier=7,
                    args=[
                        '--input-dir', str(self.tier_dirs[7]),
                        '--output-dir', str(self.tier_dirs[7]) + '/visualizations'
                    ],
                    depends_on=["analyze_shap_feature_associations.py"]
                )
            )

            # Raw data visualizations - detailed plots for significant associations
            scripts.append(
                ScriptConfig(
                    path="code/analysis/posthoc/visualize_raw_data.py",
                    env="primary",
                    runtime="medium",  # Generates many per-cancer-type plots
                    tier=7,
                    args=[
                        '--input-dir', str(self.tier_dirs[7]),
                        '--output-dir', str(self.tier_dirs[7]) + '/raw_data_plots'
                    ],
                    depends_on=["analyze_shap_feature_associations.py"]
                )
            )

            # Gene-Stratified Post-Hoc Analysis (TIER 7 extension)
            # Phase 1: Explore gene distributions per cancer type
            gene_strat_dir = tier7_dir + '/gene_stratified'
            scripts.append(
                ScriptConfig(
                    path="code/analysis/posthoc/explore_gene_distributions.py",
                    env="primary",
                    runtime="quick",
                    tier=7,
                    args=['--output-dir', gene_strat_dir + '/gene_distributions', '--tier', 'all'],
                    depends_on=[]  # Independent - queries database directly
                )
            )

            # Phase 2: Gene-stratified association tests
            gene_strat_args = [
                '--input-dir', gene_strat_dir + '/gene_distributions',
                '--output-dir', gene_strat_dir,
                '--tier', 'all'
            ]
            # Pass rankings file for feature filtering
            if self.mode in tier6_modes and not self.skip_shap and rankings_file:
                gene_strat_args.extend(['--shap-file', rankings_file])
            if self.lr_significant_only:
                gene_strat_args.append('--lr-significant-only')
            gene_strat_depends = ["explore_gene_distributions.py"]
            if shap_depends:
                gene_strat_depends.extend(shap_depends)

            scripts.append(
                ScriptConfig(
                    path="code/analysis/posthoc/analyze_gene_stratified_associations.py",
                    env="primary",
                    runtime="medium",
                    tier=7,
                    args=gene_strat_args,
                    depends_on=gene_strat_depends
                )
            )

            # Phase 3: Gene-stratified visualizations
            scripts.append(
                ScriptConfig(
                    path="code/analysis/posthoc/visualize_gene_stratified_results.py",
                    env="primary",
                    runtime="quick",
                    tier=7,
                    args=[
                        '--input-dir', gene_strat_dir,
                        '--output-dir', gene_strat_dir + '/visualizations',
                        '--raw-data-plots'
                    ],
                    depends_on=["analyze_gene_stratified_associations.py"]
                )
            )

            # Phase 4: Gene-stratified summary tables
            scripts.append(
                ScriptConfig(
                    path="code/analysis/posthoc/summarize_gene_stratified_results.py",
                    env="primary",
                    runtime="quick",
                    tier=7,
                    args=[
                        '--input-dir', gene_strat_dir,
                        '--output-dir', gene_strat_dir
                    ],
                    depends_on=["analyze_gene_stratified_associations.py"]
                )
            )

        return scripts

    def _log(self, message: str, color: str = ''):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"

        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')

        # Print to console with color
        if color:
            print(f"{color}{log_line}{Colors.ENDC}")
        else:
            print(log_line)

    def _check_environment(self) -> bool:
        """Validate environment and dependencies"""
        self._log("Validating environment...", Colors.OKCYAN)

        # Check database exists
        if not Path(self.db_path).exists():
            self._log(f"ERROR: Database not found at {self.db_path}", Colors.FAIL)
            return False

        self._log(f"✓ Database found: {self.db_path}", Colors.OKGREEN)

        # Check conda environments
        try:
            result = subprocess.run(['conda', 'env', 'list'],
                                  capture_output=True, text=True, check=True)

            if self.primary_env not in result.stdout:
                self._log(f"ERROR: Conda environment '{self.primary_env}' not found", Colors.FAIL)
                return False
            self._log(f"✓ Found conda environment: {self.primary_env}", Colors.OKGREEN)

            if not self.skip_shap and self.ml_env not in result.stdout:
                self._log(f"WARNING: Conda environment '{self.ml_env}' not found (needed for SHAP)", Colors.WARNING)
                self._log(f"  Consider running with --skip-shap-interactions", Colors.WARNING)
            elif not self.skip_shap:
                self._log(f"✓ Found conda environment: {self.ml_env}", Colors.OKGREEN)

        except (subprocess.CalledProcessError, FileNotFoundError):
            self._log("ERROR: Conda not found or not accessible", Colors.FAIL)
            return False

        return True

    def _run_script(self, script: ScriptConfig) -> bool:
        """
        Execute a single script.

        Returns:
            True if successful, False otherwise
        """
        script.status = 'running'

        # Build command
        env_name = self.primary_env if script.env == 'primary' else self.ml_env
        script_path = self.repo_root / script.path

        cmd = ['conda', 'run', '-n', env_name, 'python', str(script_path)] + script.args

        self._log(f"\n{'='*80}", Colors.HEADER)
        self._log(f"Running: {script.path}", Colors.HEADER)
        self._log(f"Environment: {env_name}", Colors.OKBLUE)
        self._log(f"Tier: {script.tier} | Runtime: {script.runtime}", Colors.OKBLUE)
        if script.args:
            self._log(f"Arguments: {' '.join(script.args)}", Colors.OKBLUE)
        self._log(f"{'='*80}\n", Colors.HEADER)

        if self.dry_run:
            self._log(f"[DRY RUN] Would execute: {' '.join(cmd)}", Colors.WARNING)
            script.status = 'skipped'
            script.duration = 0
            return True

        # Execute
        start_time = time.time()

        # Set PYTHONPATH to include modeling directory for oncokb_analyzer imports
        import os
        env = os.environ.copy()
        modeling_dir = str(self.repo_root / "code" / "modeling")
        tools_dir = str(self.repo_root / "code" / "tools")
        env['PYTHONPATH'] = modeling_dir + os.pathsep + tools_dir + os.pathsep + env.get('PYTHONPATH', '')

        # Per-tier timeouts (seconds) to accommodate HPC runtimes
        tier_timeouts = {
            1: 1800,    # 30 min  - independent analyses
            2: 1800,    # 30 min  - data setup
            3: 21600,   # 6 hours - hyperparameter optimization
            4: 14400,   # 4 hours - feature set comparison
            5: 7200,    # 2 hours - final model training
            6: 14400,   # 4 hours - SHAP analysis
            7: 3600,    # 1 hour  - post-hoc statistics
        }
        timeout_secs = tier_timeouts.get(script.tier, 7200)

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout_secs,
                env=env
            )

            duration = time.time() - start_time
            script.duration = duration

            # Log output
            if result.stdout:
                self._log("STDOUT:")
                for line in result.stdout.split('\n'):
                    self._log(f"  {line}")

            if result.stderr:
                self._log("STDERR:")
                for line in result.stderr.split('\n'):
                    self._log(f"  {line}")

            # Check return code
            if result.returncode == 0:
                script.status = 'success'
                self._log(f"✓ SUCCESS ({duration:.1f}s)", Colors.OKGREEN)
                return True
            else:
                script.status = 'failed'
                script.error_message = f"Exit code {result.returncode}"
                self._log(f"✗ FAILED (exit code {result.returncode})", Colors.FAIL)
                return False

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            script.duration = duration
            script.status = 'failed'
            script.error_message = f"Timeout (>{timeout_secs}s)"
            self._log(f"✗ FAILED (timeout after {duration:.1f}s, limit {timeout_secs}s)", Colors.FAIL)
            return False

        except Exception as e:
            duration = time.time() - start_time
            script.duration = duration
            script.status = 'failed'
            script.error_message = str(e)
            self._log(f"✗ FAILED (exception: {e})", Colors.FAIL)
            return False

    def _estimate_runtime(self) -> str:
        """Estimate total runtime based on scripts to run"""
        runtime_estimates = {
            'quick': 5,    # 5 minutes
            'medium': 20,  # 20 minutes
            'long': 120    # 2 hours
        }

        total_minutes = sum(runtime_estimates.get(s.runtime, 10) for s in self.scripts)
        hours = total_minutes // 60
        minutes = total_minutes % 60

        if hours > 0:
            return f"~{hours}h {minutes}m"
        else:
            return f"~{minutes}m"

    def list_scripts(self):
        """Print list of all scripts that would be run"""
        print(f"\n{Colors.BOLD}Scripts to be executed in {self.mode.upper()} mode:{Colors.ENDC}\n")

        tier_names = {
            1: "TIER 1: Independent Analyses",
            2: "TIER 2: Data Setup (Groupings + Train/Test Split)",
            3: "TIER 3: Hyperparameter Optimization (Training Data Only)",
            4: "TIER 4: Feature Set Comparison (Training Data Only)",
            5: "TIER 5: Final Model Training & Test Evaluation",
            6: "TIER 6: SHAP Analysis",
            7: "TIER 7: Post-Hoc Statistics"
        }

        for tier in range(1, 8):
            tier_scripts = [s for s in self.scripts if s.tier == tier]
            if tier_scripts:
                print(f"{Colors.HEADER}{tier_names[tier]}{Colors.ENDC}")
                for script in tier_scripts:
                    env = script.env.upper()
                    runtime = script.runtime.upper()
                    print(f"  [{env:8s}] [{runtime:6s}] {script.path}")
                    if script.args:
                        print(f"               Args: {' '.join(script.args)}")
                print()

        print(f"Total scripts: {len(self.scripts)}")
        print(f"Estimated runtime: {self._estimate_runtime()}\n")

    def run(self):
        """Execute all analyses"""

        # Header
        print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}BALDER MASTER ANALYSIS RUNNER{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

        print(f"Mode: {Colors.OKBLUE}{self.mode.upper()}{Colors.ENDC}")
        print(f"Scripts to run: {Colors.OKBLUE}{len(self.scripts)}{Colors.ENDC}")
        print(f"Estimated runtime: {Colors.OKBLUE}{self._estimate_runtime()}{Colors.ENDC}")
        print(f"Run directory: {Colors.OKBLUE}{self.run_dir}{Colors.ENDC}")
        print(f"Log file: {Colors.OKBLUE}{self.log_file}{Colors.ENDC}")
        if self.dry_run:
            print(f"{Colors.WARNING}DRY RUN MODE - No scripts will actually execute{Colors.ENDC}")
        print()

        # Validate environment
        if not self._check_environment():
            self._log("Environment validation failed. Exiting.", Colors.FAIL)
            sys.exit(1)

        print()

        # Start timer
        self.results['start_time'] = datetime.now()
        self.results['total'] = len(self.scripts)

        # Execute by tier
        tier_names = {
            1: "Independent Analyses",
            2: "Data Setup (Groupings + Train/Test Split)",
            3: "Hyperparameter Optimization (Training Data Only)",
            4: "Feature Set Comparison (Training Data Only)",
            5: "Final Model Training & Test Evaluation",
            6: "SHAP Analysis",
            7: "Post-Hoc Statistics"
        }

        for tier in range(1, 8):
            tier_scripts = [s for s in self.scripts if s.tier == tier]
            if not tier_scripts:
                continue

            self._log(f"\n\n{'#'*80}", Colors.BOLD)
            self._log(f"# TIER {tier}: {tier_names[tier]}", Colors.BOLD)
            self._log(f"{'#'*80}\n", Colors.BOLD)

            for script in tier_scripts:
                success = self._run_script(script)

                if success:
                    self.results['success'] += 1
                else:
                    self.results['failed'] += 1
                    if not self.continue_on_error:
                        self._log("\nAborting due to failure (--continue-on-error not set)", Colors.FAIL)
                        break

            # If we aborted, stop processing tiers
            if self.results['failed'] > 0 and not self.continue_on_error:
                break

        # End timer
        self.results['end_time'] = datetime.now()

        # Generate summary
        self._generate_summary()

    def _generate_summary(self):
        """Generate and display summary report"""

        duration = self.results['end_time'] - self.results['start_time']

        print(f"\n\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}SUMMARY{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

        # Overall stats
        print(f"Total scripts: {self.results['total']}")
        print(f"{Colors.OKGREEN}Successful: {self.results['success']}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed: {self.results['failed']}{Colors.ENDC}")
        if self.dry_run:
            print(f"{Colors.WARNING}Skipped (dry run): {len(self.scripts)}{Colors.ENDC}")
        print(f"Total runtime: {duration}")
        print()

        # Failed scripts details
        if self.results['failed'] > 0:
            print(f"{Colors.FAIL}Failed Scripts:{Colors.ENDC}")
            for script in self.scripts:
                if script.status == 'failed':
                    print(f"  ✗ {script.path}")
                    if script.error_message:
                        print(f"    Error: {script.error_message}")
                    if script.duration:
                        print(f"    Duration: {script.duration:.1f}s")
            print()

        # Successful scripts
        if self.results['success'] > 0:
            print(f"{Colors.OKGREEN}Successful Scripts:{Colors.ENDC}")
            for script in self.scripts:
                if script.status == 'success':
                    duration_str = f"{script.duration:.1f}s" if script.duration else "N/A"
                    print(f"  ✓ {script.path} ({duration_str})")
            print()

        # Log file location
        print(f"Complete log: {Colors.OKBLUE}{self.log_file}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

        # Save JSON summary
        summary_file = self.run_dir / "run_summary.json"
        summary_data = {
            'mode': self.mode,
            'run_directory': str(self.run_dir),
            'timestamp': self.timestamp,
            'start_time': self.results['start_time'].isoformat(),
            'end_time': self.results['end_time'].isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_scripts': self.results['total'],
            'successful': self.results['success'],
            'failed': self.results['failed'],
            'tier_directories': {
                f'tier{k}': str(v) for k, v in self.tier_dirs.items()
            },
            'scripts': [
                {
                    'path': s.path,
                    'tier': s.tier,
                    'env': s.env,
                    'status': s.status,
                    'duration': s.duration,
                    'error': s.error_message,
                    'output_dir': str(self.tier_dirs.get(s.tier, 'N/A'))
                }
                for s in self.scripts
            ]
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        self._log(f"Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Master script to run all Balder analyses and ML experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_analyses.py --full                    # Run everything
  python run_all_analyses.py --quick                   # Quick test run
  python run_all_analyses.py --analysis-only           # Skip ML experiments
  python run_all_analyses.py --ml-only                 # Only ML experiments
  python run_all_analyses.py --post-model-selection            # Downstream of Phase 2b (TIER 4-7)
  python run_all_analyses.py --skip-shap-interactions  # Skip SHAP (no env switch)
  python run_all_analyses.py --dry-run --list          # Preview without running
        """
    )

    # Execution modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--full', action='store_const', const='full', dest='mode',
                           help='Run all analyses (hours)')
    mode_group.add_argument('--quick', action='store_const', const='quick', dest='mode',
                           help='Quick test run (~30 min, skips SHAP to save memory)')
    mode_group.add_argument('--ml-only', action='store_const', const='ml-only', dest='mode',
                           help='Only ML experiments')
    mode_group.add_argument('--analysis-only', action='store_const', const='analysis-only', dest='mode',
                           help='Only analysis scripts')
    mode_group.add_argument('--post-model-selection', action='store_const', const='post-model-selection', dest='mode',
                           help='Only downstream of Phase 2b (TIER 4-7, includes SHAP primary)')

    # Options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would run without executing')
    parser.add_argument('--list', action='store_true',
                       help='List all scripts and exit')
    parser.add_argument('--skip-shap-interactions', action='store_true',
                       help='Skip SHAP interactions (avoid env switch)')
    parser.add_argument('--stop-on-error', action='store_true',
                       help='Stop execution if any script fails (default: continue)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run Tier 1 scripts in parallel (experimental)')
    parser.add_argument('--feature-config', type=str, default=None,
                       help=(
                           'Path to custom feature configuration YAML file. '
                           'Default: code/config/features.yaml. '
                           'Can also be set via BALDER_FEATURES_CONFIG environment variable.'
                       ))
    parser.add_argument('--institution-weight-alpha', type=float, default=0.0,
                       help=(
                           'Power parameter for institutional sample weighting in TIER 5 (0.0 to 1.0). '
                           '0.0 = no weighting (default), 0.5 = moderate correction, 1.0 = full inverse frequency. '
                           'Higher values reduce the influence of dominant institutions (MSK, DFCI) during training.'
                       ))
    parser.add_argument('--base-run-dir', type=str, default=None,
                       help=(
                           'Path to an existing run directory to use for train-test split and hyperparameter results. '
                           'Useful for re-running downstream tiers (e.g., TIER 4-7) with settings from a previous run. '
                           'Example: --base-run-dir runs/run_20260121_215129'
                       ))
    parser.add_argument('--algorithm', type=str, default='random_forest',
                       choices=['random_forest', 'logistic'],
                       help=(
                           'ML algorithm for TIER 3-5 (default: random_forest). '
                           'logistic uses LR coefficient analysis instead of SHAP for feature rankings.'
                       ))
    parser.add_argument('--lr-significant-only', action='store_true',
                       help=(
                           'When --algorithm logistic, restrict TIER 7 post-hoc testing to only '
                           'features with Wald-significant LR coefficients (FDR < 0.05). '
                           'Without this flag, all features are ranked by effect size regardless of significance.'
                       ))

    parser.set_defaults(mode='full')

    args = parser.parse_args()

    # Determine SHAP skip flags:
    # - skip_shap: Skip ALL SHAP (both 6a primary values and 6b interactions)
    # - skip_shap_interactions: Skip only 6b (interactions), but run 6a (primary values)
    skip_shap = False
    skip_shap_interactions = args.skip_shap_interactions

    if args.mode == 'quick':
        if args.algorithm == 'logistic':
            # LR extraction is fast, no need to skip TIER 6
            skip_shap = False
            skip_shap_interactions = True  # No interactions for LR
            print(f"{Colors.WARNING}Note: LR coefficient extraction will run in quick mode (fast).{Colors.ENDC}")
        else:
            # Quick mode with RF: skip ALL SHAP (both 6a and 6b)
            skip_shap = True
            print(f"{Colors.WARNING}Note: All SHAP analysis skipped in quick mode (memory-intensive).{Colors.ENDC}")
            print(f"{Colors.WARNING}      Use --full mode or run SHAP separately if needed.{Colors.ENDC}")
    elif args.mode == 'post-model-selection':
        # Post-model-selection mode: run SHAP primary (6a), skip interactions (6b)
        skip_shap_interactions = True
        print(f"{Colors.WARNING}Note: SHAP interactions (6b) skipped in post-model-selection mode.{Colors.ENDC}")
        print(f"{Colors.WARNING}      Running TIER 4, 5, 6a (SHAP Primary), and 7.{Colors.ENDC}")

    # Create runner
    runner = MasterAnalysis(
        mode=args.mode,
        dry_run=args.dry_run,
        continue_on_error=not args.stop_on_error,
        skip_shap=skip_shap,
        skip_shap_interactions=skip_shap_interactions,
        parallel=args.parallel,
        feature_config_path=args.feature_config,
        institution_weight_alpha=args.institution_weight_alpha,
        base_run_dir=args.base_run_dir,
        algorithm=args.algorithm,
        lr_significant_only=args.lr_significant_only
    )

    # If --list, just show scripts and exit
    if args.list:
        runner.list_scripts()
        sys.exit(0)

    # Run
    runner.run()

if __name__ == "__main__":
    main()
