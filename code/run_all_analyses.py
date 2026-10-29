#!/usr/bin/env python3
"""Master orchestrator for the analysis pipeline (TIER 1-7).

Runs the analysis scripts and ML experiments in order, handling environment switching,
dependencies and error reporting. Test data are held out before any model development.

  TIER 1 independent analyses · TIER 2 cohort groupings + train/test split · TIER 3 hyperparameter
  search · TIER 4 feature-set comparison · TIER 5 final models + test evaluation · TIER 6 feature
  importance (RF: SHAP; LR: coefficients) · TIER 7 post-hoc statistics

By default both RF and LR pathways run: RF first, then LR with --base-run-dir set to the RF run.

Usage:
    python run_all_analyses.py --full                 # RF + LR, all tiers
    python run_all_analyses.py --quick                # smoke test
    python run_all_analyses.py --algorithm logistic   # single pathway
    python run_all_analyses.py --dry-run --list       # preview without running
"""

import argparse
import shutil
import subprocess
import sys
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

import pandas as pd

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

# Scripts that write no manuscript table (figure-only / exploratory); dropped by --tables-only
NON_TABLE_SCRIPTS = frozenset({
    'analyze_demographics_by_cancer_type.py',
    'feature_engineering_analysis.py',
    'plot_variant_demographic_stratification.py',
    'plot_shap_feature_ranks.py',
    'plot_rf_shap_lr_or_feature_ranks.py',
    'visualize_posthoc_results.py',
    'visualize_raw_data.py',
    'visualize_gene_stratified_results.py',
    'create_hit_list_composite_figures.py',
    'visualize_rf_lr_hit_venn.py',
    'plot_rf_vs_lr_auc_per_cancer.py',
})

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
                 parallel: bool = False,
                 feature_config_path: str = None, institution_weight_alpha: float = 0.0,
                 base_run_dir: str = None, algorithm: str = 'random_forest',
                 lr_significant_only: bool = False, shap_file_override: str = None,
                 tier3_algorithms: str = 'rf,lr', run_dir_suffix: str = '',
                 run_dir_override: str = None,
                 split_strategy: str = 'patient', fresh_split: bool = False,
                 hyperparams_source: str = 'search',
                 hyperparams_dir_override: Optional[str] = None,
                 exclude_cancer_types: Optional[List[str]] = None,
                 keep_unknown_category: bool = False,
                 tables_only: bool = False):
        self.mode = mode
        # Skip NON_TABLE_SCRIPTS and ask mixed table+figure scripts for tables only.
        self.tables_only = tables_only
        # 'search': TIER 5 uses TIER 3's result; 'defaults': the hand-tuned constants
        self.hyperparams_source = hyperparams_source
        self.hyperparams_dir_override = hyperparams_dir_override
        self.split_strategy = split_strategy  # TIER 2 split: 'patient' (default) or legacy 'row'
        # Build a new TIER 2b split while still reusing base_run_dir's hyperparameters
        self.fresh_split = fresh_split
        self.dry_run = dry_run
        self.continue_on_error = continue_on_error
        self.skip_shap = skip_shap  # Skip TIER 6 entirely (RF SHAP primary or LR coefficients)
        self.parallel = parallel
        self.tier3_algorithms = tier3_algorithms  # Comma list for TIER 3 search
        self.run_dir_suffix = run_dir_suffix  # Suffix appended to run_<TS>; used by dual-pathway loop
        self.run_dir_override = run_dir_override  # --tier7-only: write into an existing run dir
        self.feature_config_path = feature_config_path  # Custom feature config path
        self.institution_weight_alpha = institution_weight_alpha  # Power weighting for TIER 5
        self.base_run_dir = Path(base_run_dir).resolve() if base_run_dir else None  # Existing run to draw from
        self.algorithm = algorithm  # ML algorithm: 'random_forest' or 'logistic'
        self.lr_significant_only = lr_significant_only  # Filter LR rankings to Wald-significant only
        # Legacy TIER 7 behaviour: GENIE's "Unknown" sentinel is its own category level (default: excluded)
        self.keep_unknown_category = keep_unknown_category
        self.shap_file_override = shap_file_override  # Optional explicit TIER 6 rankings path for TIER 7
        # Codes dropped from the cohort after TIER 2a eligibility (upper-cased to match ONCOTREE_CODE)
        self.exclude_cancer_types = sorted({c.strip().upper()
                                            for c in (exclude_cancer_types or [])
                                            if c.strip()})

        # Invocation context recorded in run_summary.json (sys.argv as typed, not the internal LR re-invocation)
        self.invocation = {
            'argv': list(sys.argv),
            'cwd': os.getcwd(),
            'algorithm': algorithm,
            'base_run_dir': str(base_run_dir) if base_run_dir else None,
            # Which hyperparameters TIER 5 used (recorded for reproducibility)
            'hyperparams_source': hyperparams_source,
            'hyperparams_dir_override': hyperparams_dir_override,
            # Cancer types removed after TIER 2a eligibility; recorded because it changes the analysed cohort
            'exclude_cancer_types': list(exclude_cancer_types or []),
            'tables_only': tables_only,
        }

        self.repo_root = Path(__file__).parent.parent

        # Unified timestamped run directory; --run-dir (with --tier7-only) reuses an existing one
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.run_dir_override:
            if self.mode != 'tier7-only':
                raise ValueError(
                    "--run-dir is only valid with --tier7-only. Every other mode "
                    "creates its own timestamped run directory so that runs cannot "
                    "overwrite each other."
                )
            self.run_dir = Path(self.run_dir_override).resolve()
            if not self.run_dir.exists():
                raise FileNotFoundError(
                    f"--run-dir does not exist: {self.run_dir}\n"
                    f"  --tier7-only writes into a run dir that already carries "
                    f"tiers 1-6; create it by copying them from the base run."
                )
        else:
            suffix = f"_{self.run_dir_suffix}" if self.run_dir_suffix else ""
            self.run_dir = self.repo_root / "runs" / f"run_{self.timestamp}{suffix}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

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

        self.log_file = self.run_dir / "run.log"

        self.db_path = get_default_db_path()

        load_feature_config(self.feature_config_path)
        self.feature_config_loaded = get_config_path()

        copy_config_to_run_dir(str(self.run_dir), self.feature_config_path)

        if self.base_run_dir:
            if not self.base_run_dir.exists():
                raise ValueError(f"Base run directory not found: {self.base_run_dir}")
            print(f"{Colors.OKCYAN}Using base run directory: {self.base_run_dir}{Colors.ENDC}")

        self.primary_env = "balder_conda_env"
        self.ml_env = "balder_ml_env"

        # Resolve the cohort before building the script registry (TIER 5 and TIER 7 take it as an argument)
        self.groupings_file, self.cancer_types = self._resolve_cancer_type_cohort()

        self.scripts = self._define_scripts()
        if self.tables_only:
            self.scripts = [s for s in self.scripts
                            if Path(s.path).name not in NON_TABLE_SCRIPTS]

        self.results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }

    # --- Cancer-type cohort resolution ---

    def _find_source_groupings_file(self) -> Optional[Path]:
        """Locate the TIER 2a eligibility CSV that defines this run's cohort (None if not yet reachable)."""
        own = (self.tier_dirs[2] / "cancer_type_groupings"
               / "cancer_type_group_definitions.csv")
        if own.exists():
            return own
        if self.base_run_dir:
            base = (self.base_run_dir / "tier2_data_setup" / "cancer_type_groupings"
                    / "cancer_type_group_definitions.csv")
            if base.exists():
                return base
        # TIER 2a runs later in this invocation; its output path is deterministic
        if self.mode in ['full', 'quick', 'analysis-only']:
            return own
        return None

    def _resolve_cancer_type_cohort(self):
        """Return (groupings_file, cancer_types) for this run.

        Pins the groupings file and applies --exclude-cancer-types by writing a filtered copy
        of the eligibility CSV into the run dir. Fails closed on a code not in the eligibility file.
        """
        source = self._find_source_groupings_file()

        if not self.exclude_cancer_types:
            return (str(source) if source else None, None)

        # An exclusion was requested, so the file must be readable now
        if source is None or not source.exists():
            raise FileNotFoundError(
                "--exclude-cancer-types needs TIER 2a's cancer_type_group_definitions.csv "
                "to validate against, but none was found.\n"
                f"  Looked in: {self.tier_dirs[2] / 'cancer_type_groupings'}"
                + (f"\n  and:       {self.base_run_dir / 'tier2_data_setup' / 'cancer_type_groupings'}"
                   if self.base_run_dir else "")
                + "\n  Point --base-run-dir at a run that carries TIER 2a outputs."
            )

        defs = pd.read_csv(source)
        eligible = set(defs['ONCOTREE_CODE'].astype(str))

        # A code already excluded by the source run is idempotent; the marker file distinguishes it from a typo (which fails closed)
        already = set()
        marker = source.parent / "EXCLUDED_CANCER_TYPES.txt"
        if marker.exists():
            text = marker.read_text()
            for line in text.splitlines():
                if line.lower().startswith('excluded:'):
                    already = {c.strip().upper()
                               for c in line.split(':', 1)[1].split(',') if c.strip()}

        unknown = [c for c in self.exclude_cancer_types
                   if c not in eligible and c not in already]
        if unknown:
            raise ValueError(
                f"--exclude-cancer-types names {unknown}, which are not in the "
                f"eligibility file {source}.\n"
                f"  Eligible codes ({len(eligible)}): {', '.join(sorted(eligible))}"
                + (f"\n  Already excluded upstream: {', '.join(sorted(already))}"
                   if already else "")
            )

        redundant = [c for c in self.exclude_cancer_types if c in already
                     and c not in eligible]
        if redundant:
            print(f"{Colors.OKCYAN}  Already excluded by the base run "
                  f"({', '.join(redundant)}); carrying that exclusion forward."
                  f"{Colors.ENDC}")

        kept = defs[~defs['ONCOTREE_CODE'].astype(str).isin(self.exclude_cancer_types)].copy()

        out_dir = self.tier_dirs[2] / "cancer_type_groupings"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "cancer_type_group_definitions.csv"

        # Keep the pre-exclusion file: it records what the eligibility screen returned
        if out_file.exists():
            backup = out_dir / "cancer_type_group_definitions.pre_exclusion.csv"
            if not backup.exists():
                shutil.copy2(out_file, backup)
        kept.to_csv(out_file, index=False)

        # Filter the second TIER 2a file the same way so both describe one cohort
        src_map = source.parent / "patient_to_group_mapping.csv"
        if src_map.exists():
            out_map = out_dir / "patient_to_group_mapping.csv"
            if out_map.exists() and not (out_dir / "patient_to_group_mapping.pre_exclusion.csv").exists():
                shutil.copy2(out_map, out_dir / "patient_to_group_mapping.pre_exclusion.csv")
            mapping = pd.read_csv(src_map)
            mapping = mapping[~mapping['ONCOTREE_CODE'].astype(str)
                              .isin(self.exclude_cancer_types)]
            mapping.to_csv(out_map, index=False)

        # Cumulative, so a chained run records the full exclusion
        all_excluded = sorted(set(self.exclude_cancer_types) | already)
        (out_dir / "EXCLUDED_CANCER_TYPES.txt").write_text(
            "Cancer types excluded from this run downstream of TIER 2a eligibility\n"
            "(--exclude-cancer-types). The DB and the train/test split are unchanged;\n"
            "these codes are simply never fitted or tested.\n\n"
            f"excluded: {', '.join(all_excluded)}\n"
            f"retained: {len(kept)} of {len(defs) + len(already - eligible)}\n"
            f"source:   {source}\n"
        )

        codes = sorted(kept['ONCOTREE_CODE'].astype(str))
        newly = [c for c in self.exclude_cancer_types if c in eligible]
        if newly:
            print(f"{Colors.WARNING}Cohort restricted: excluding "
                  f"{', '.join(newly)} -> {len(codes)} cancer types "
                  f"(was {len(defs)}){Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}Cohort: {len(codes)} cancer types "
                  f"(excluding {', '.join(all_excluded)}){Colors.ENDC}")
        print(f"{Colors.OKCYAN}  Filtered eligibility file: {out_file}{Colors.ENDC}")
        return (str(out_file), codes)

    def _get_train_test_split_file(self) -> str:
        """Train/test split path, using base_run_dir if set. Fails closed if no split is found."""
        # --fresh-split: read THIS run's split (written by TIER 2b before any consumer runs)
        if self.fresh_split:
            return str(self.run_dir / 'tier2_data_setup' / 'train_test_split' / 'train_test_split.csv')

        if self.base_run_dir:
            split_file = self.base_run_dir / 'tier2_data_setup' / 'train_test_split' / 'train_test_split.csv'
            if split_file.exists():
                return str(split_file)
            raise FileNotFoundError(
                f"base_run_dir is set but no train_test_split.csv found at:\n"
                f"  {split_file}\n"
                f"This typically happens with --post-model-selection + dual-pathway, "
                f"where the RF re-run skipped TIER 2 and the LR pass points base_run_dir "
                f"at the RF re-run's dir. Pass --base-run-dir explicitly to the dir that "
                f"contains TIER 2 outputs, or run without --post-model-selection."
            )

        if self.mode == 'post-model-selection':
            raise FileNotFoundError(
                "post-model-selection mode requires --base-run-dir to point at a run "
                "containing tier2_data_setup/train_test_split/train_test_split.csv."
            )

        # Default: use this run's tier2 directory (will be created by TIER 2b)
        return str(self.tier_dirs[2] / 'train_test_split' / 'train_test_split.csv')

    def _get_hyperparams_dir(self) -> Optional[str]:
        """TIER 3 output directory whose search result TIER 5 should consume.

        Returns a directory (it may not exist yet: args are built before TIER 3 runs); TIER 5
        resolves the params file at runtime. Precedence: --hyperparams-dir, this run's TIER 3,
        base_run_dir's. None leaves TIER 5 on the hand-tuned constants.
        """
        if self.hyperparams_dir_override:
            d = Path(self.hyperparams_dir_override).resolve()
            if not d.is_dir():
                raise FileNotFoundError(
                    f"--hyperparams-dir does not exist or is not a directory: {d}")
            if not (list(d.glob(f'run_*/{self.algorithm}_universal_params.json'))
                    or (d / f'{self.algorithm}_universal_params.json').exists()):
                raise FileNotFoundError(
                    f"--hyperparams-dir has no {self.algorithm}_universal_params.json "
                    f"(directly or under run_*/): {d}")
            return str(d)

        if self._tier3_runs_in_this_run():
            return str(self.tier_dirs[3].resolve())
        if self.base_run_dir:
            base_tier3 = (self.base_run_dir / 'tier3_hyperparameter_optimization').resolve()
            if base_tier3.exists() and any(base_tier3.glob(
                    f'run_*/{self.algorithm}_universal_params.json')):
                return str(base_tier3)
        return None

    def _tier3_runs_in_this_run(self) -> bool:
        """Whether TIER 3 will execute in this invocation (mirrors the TIER 3 gate)."""
        return self.mode in ('full', 'quick', 'ml-only') and not self.base_run_dir

    def _get_tier5_model_dir(self) -> str:
        """Get TIER 5 model directory, using base_run_dir if specified for SHAP."""
        # Models come from this run's TIER 5, or from base_run_dir when TIER 5 is not run
        return str(self.tier_dirs[5])

    def _define_scripts(self) -> List[ScriptConfig]:
        """Define all analysis scripts and their configurations, in tier order."""

        scripts = []

        def get_output_dir_for_tier(tier: int, script_path: str) -> str:
            """Calculate relative path to tier directory from script location."""
            tier_dir = self.tier_dirs[tier]
            return str(tier_dir.relative_to(self.repo_root))

        # TIER 1: independent analyses; skipped when reusing a base run
        if self.mode in ['full', 'quick', 'analysis-only'] and not self.base_run_dir:
            tier1_dir = get_output_dir_for_tier(1, "code/analysis/")
            scripts.extend([
                ScriptConfig(
                    path="code/analysis/figures/generate_strobe_flow_diagram.py",
                    env="primary",
                    runtime="quick",
                    tier=1,
                    args=['--output-dir', tier1_dir + '/strobe_diagram',
                          '--base-run-dir', str(self.run_dir.relative_to(self.repo_root))]
                         + (['--counts-only'] if self.tables_only else [])
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
                         + (['--table-only'] if self.tables_only else [])
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
                ScriptConfig(
                    path="code/analysis/institutional/plot_cv_and_forest_simple.py",
                    env="primary",
                    runtime="quick",
                    tier=1,
                    args=['--output-dir', tier1_dir + '/cv_forest_plots']
                         + (['--table-only'] if self.tables_only else [])
                ),
            ])

        # TIER 2a: cancer-type grouping; skipped when reusing a base run
        if self.mode in ['full', 'quick', 'ml-only'] and not self.base_run_dir:
            tier2_dir = get_output_dir_for_tier(2, "code/analysis/")
            scripts.append(
                ScriptConfig(
                    path="code/analysis/define_cancer_type_groupings.py",
                    env="primary",
                    runtime="quick",
                    tier=2,
                    args=['--output-dir', tier2_dir + '/cancer_type_groupings']
                ),
            )

        # Table 1 runs after TIER 2a: it reads cancer_type_group_definitions.csv for OncoTree names
        if self.mode in ['full', 'quick', 'analysis-only'] and not self.base_run_dir:
            table1_dir = get_output_dir_for_tier(1, "code/analysis/")
            scripts.append(
                ScriptConfig(
                    path="code/analysis/cancer_type_summary_stats.py",
                    env="primary",
                    runtime="quick",
                    tier=2,
                    args=['--output-dir', table1_dir + '/cancer_type_summary_stats',
                          '--format-table1',
                          '--base-run-dir', str(self.run_dir.relative_to(self.repo_root))]
                         + ([] if self.tables_only else ['--markdown', '--latex'])
                )
            )

        # TIER 2b: train/test split (before any modeling); skipped with a base run unless --fresh-split
        if ((self.mode in ['full', 'quick', 'ml-only'] and not self.base_run_dir)
                or (self.fresh_split and self.mode in ['full', 'quick', 'ml-only',
                                                       'post-model-selection'])):
            tier2_dir = get_output_dir_for_tier(2, "code/analysis/")
            scripts.append(
                ScriptConfig(
                    path="code/analysis/define_train_test_split.py",
                    env="primary",
                    runtime="quick",
                    tier=2,
                    args=['--output-dir', tier2_dir + '/train_test_split',
                          '--split-strategy', self.split_strategy]
                )
            )

        # TIER 3: hyperparameter search (training data only); skipped when reusing a base run
        if self.mode in ['full', 'quick', 'ml-only'] and not self.base_run_dir:
            tier3_dir = get_output_dir_for_tier(3, "code/modeling/")
            train_split_file = self._get_train_test_split_file()
            phase2b_args = [
                '--algorithm', self.tier3_algorithms,
                '--n-jobs', '1',
                '--n-iter', '500',
                '--sample-fraction', '1.0',
                '--feature-set', 'both',
                '--train-test-split-file', train_split_file,
                '--output-dir', tier3_dir
            ]
            if self.institution_weight_alpha > 0:
                phase2b_args.extend(['--institution-weight-alpha', str(self.institution_weight_alpha)])
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

        # TIER 4: feature-set comparison (training data only). Demographic only by default: the 'both' set one-hot encodes high-cardinality columns and can need 40-100+ GB RAM
        if self.mode in ['full', 'quick', 'ml-only', 'post-model-selection']:
            tier4_dir = get_output_dir_for_tier(4, "code/modeling/")
            train_split_file = self._get_train_test_split_file()

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

        # TIER 5: final models on training data; the test set is used once, for evaluation
        if self.mode in ['full', 'quick', 'ml-only', 'post-model-selection']:
            tier5_dir = get_output_dir_for_tier(5, "code/modeling/")
            train_split_file = self._get_train_test_split_file()

            tier5_args = [
                '--train-test-split-file', train_split_file,
                '--feature-set', 'both',
                '--output-dir', tier5_dir,
                '--algorithm', self.algorithm
            ]
            # Cohort restriction (TIER 6 follows it: it globs the .joblib models TIER 5 wrote)
            if self.cancer_types:
                tier5_args.extend(['--cancer-types'] + self.cancer_types)
            # TIER 3 -> TIER 5: pass a directory (args are built before TIER 3 runs); TIER 5 resolves it and fails closed
            hp_dir = self._get_hyperparams_dir()
            if self.hyperparams_source == 'defaults':
                tier5_args.extend(['--hyperparams-source', 'defaults'])
            elif hp_dir:
                tier5_args.extend(['--hyperparams-dir', hp_dir])
            else:
                raise RuntimeError(
                    "TIER 5 would fall back to the hand-tuned hyperparameter constants: no "
                    "TIER 3 search runs in this invocation and --base-run-dir has no "
                    f"tier3_hyperparameter_optimization/run_*/{self.algorithm}_universal_params.json. "
                    "Those constants are not search-derived and were selected with reference to "
                    "held-out performance. Point --base-run-dir at a run "
                    "that has a TIER 3 search, or pass --hyperparams-source defaults to accept "
                    "them deliberately.")
            if self.institution_weight_alpha > 0:
                tier5_args.extend(['--institution-weight-alpha', str(self.institution_weight_alpha)])
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

        # TIER 6: RF -> SHAP primary values; LR -> coefficient extraction
        tier6_modes = ['full', 'post-model-selection']
        if self.algorithm == 'logistic':
            tier6_modes.append('quick')
        if self.mode in tier6_modes and not self.skip_shap:
            # Absolute paths: scripts resolve relative paths from their own directory
            tier6_abs_dir = str(self.tier_dirs[6])
            tier5_model_dir = str(self.tier_dirs[5])
            train_split_file = self._get_train_test_split_file()
            if not train_split_file.startswith('/'):
                train_split_file = str(self.repo_root / train_split_file)

            if self.algorithm == 'random_forest':
                # TIER 6a: primary SHAP values; --no-timestamp-subdir keeps a stable path for TIER 7
                shap_primary_args = [
                    '--model-dir', tier5_model_dir,
                    '--train-test-split-file', train_split_file,
                    '--output-dir', tier6_abs_dir + '/primary_values',
                    '--no-timestamp-subdir'
                ]

                scripts.append(
                    ScriptConfig(
                        path="code/analysis/shap_interactions/compute_shap_primary_values.py",
                        env="ml",  # Requires shap (balder_ml_env)
                        runtime="medium",
                        tier=6,
                        args=shap_primary_args,
                        depends_on=["run_final_model_training.py"]
                    )
                )

            elif self.algorithm == 'logistic':
                # LR: coefficient extraction (replaces SHAP)
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
                        runtime="quick",
                        tier=6,
                        args=lr_args,
                        depends_on=["run_final_model_training.py"]
                    )
                )

        # TIER 7: post-hoc statistics, gated on TIER 6 feature importance
        if self.mode in ['full', 'quick', 'analysis-only', 'post-model-selection',
                         'tier7-only']:
            tier7_dir = get_output_dir_for_tier(7, "code/analysis/")
            tier6_dir = get_output_dir_for_tier(6, "code/analysis/")

            if self.algorithm == 'random_forest':
                rankings_file = tier6_dir + '/primary_values/combined_feature_importance.csv'
                rankings_depends = ["compute_shap_primary_values.py"]
            elif self.algorithm == 'logistic':
                rankings_file = str(self.tier_dirs[6]) + '/lr_analysis/combined_lr_grouped_importance.csv'
                rankings_depends = ["extract_lr_coefficients.py"]
            else:
                rankings_file = None
                rankings_depends = []

            # Optional explicit TIER 6 rankings path (to consume an existing TIER 6 output)
            if self.shap_file_override:
                override_path = Path(self.shap_file_override)
                if not override_path.is_absolute():
                    override_path = (self.repo_root / override_path).resolve()
                if not override_path.exists():
                    raise FileNotFoundError(
                        f"--shap-file override does not exist: {override_path}"
                    )
                print(f"{Colors.OKBLUE}TIER 7 rankings_file overridden via --shap-file: {override_path}{Colors.ENDC}")
                rankings_file = str(override_path)
                # Existing file: no dependency on a TIER 6 script in this session
                rankings_depends = []

            # Whether a rankings file exists to gate on. Tested on the artifact, not the mode, so a reused TIER 6 (--tier7-only) gates like a fresh one
            tier6_will_run = self.mode in tier6_modes
            tier5_will_run = self.mode in ['full', 'quick', 'ml-only',
                                           'post-model-selection']
            rankings_exist = bool(rankings_file) and Path(
                rankings_file if str(rankings_file).startswith('/')
                else self.repo_root / rankings_file).exists()
            have_rankings = (bool(rankings_file) and not self.skip_shap
                             and (tier6_will_run or rankings_exist))
            # TIER 6 is not running here: declare no dependency on its scripts (it would never be satisfied)
            if not tier6_will_run:
                rankings_depends = []
            cross_pathway_depends = (["extract_lr_coefficients.py"]
                                     if tier6_will_run else [])

            if self.mode == 'tier7-only' and not have_rankings:
                raise FileNotFoundError(
                    f"--tier7-only needs an existing TIER 6 rankings file to gate on, "
                    f"but none was found at: {rankings_file}\n"
                    f"  Copy the base run's tier6_shap_analysis/ into this run dir, or "
                    f"pass --shap-file explicitly."
                )

            # Pin the cancer-type cohort explicitly rather than letting scripts auto-glob a run dir
            groupings_args = (['--groupings-file', self.groupings_file]
                              if self.groupings_file else [])

            tier6_dep_script = rankings_depends[0] if rankings_depends else None
            scripts.append(
                ScriptConfig(
                    path="code/analysis/posthoc/analyze_age_by_actionability.py",
                    env="primary",
                    runtime="quick",
                    tier=7,
                    args=['--output-dir', tier7_dir + '/age_actionability'] + groupings_args,
                    depends_on=rankings_depends if not self.skip_shap else []
                )
            )

            shap_assoc_args = (['--output-dir', tier7_dir + '/shap_associations']
                               + groupings_args)

            if have_rankings:
                shap_assoc_args.extend(['--shap-file', rankings_file])
                shap_depends = rankings_depends
            else:
                # In quick/analysis-only modes the script finds existing files
                shap_depends = []

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

            # RF only (SHAP-specific plots)
            if have_rankings and self.algorithm == 'random_forest':
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

            # Cross-pathway feature-rank comparison: LR run against an RF base run with SHAP outputs
            if (have_rankings
                    and self.algorithm == 'logistic'
                    and self.base_run_dir is not None):
                base_shap_dir = self.base_run_dir / 'tier6_shap_analysis' / 'primary_values'
                if base_shap_dir.exists() and any(
                    base_shap_dir.rglob('combined_feature_importance.csv')
                ):
                    scripts.append(
                        ScriptConfig(
                            path="code/analysis/posthoc/plot_rf_shap_lr_or_feature_ranks.py",
                            env="primary",
                            runtime="quick",
                            tier=7,
                            args=[
                                '--shap-run-dir', str(self.base_run_dir),
                                '--lr-run-dir',   str(self.run_dir),
                                '--output-dir',   str(self.tier_dirs[7]) + '/cross_pathway',
                            ],
                            depends_on=cross_pathway_depends,
                        )
                    )
                else:
                    # Fails open: the plot is dropped (not an error) when the base run lacks these outputs; verify cross_pathway/ after an LR re-run
                    print(
                        Colors.WARNING
                        + 'WARNING: skipping plot_rf_shap_lr_or_feature_ranks.py -- base run '
                        'has no RF SHAP primary values at '
                        + str(base_shap_dir)
                        + '. The cross-pathway output will be MISSING from '
                          'tier7_posthoc_statistics/cross_pathway/.'
                        + Colors.ENDC
                    )

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

            # Phase 1: Explore gene distributions per cancer type
            gene_strat_dir = tier7_dir + '/gene_stratified'
            scripts.append(
                ScriptConfig(
                    path="code/analysis/posthoc/explore_gene_distributions.py",
                    env="primary",
                    runtime="quick",
                    tier=7,
                    args=(['--output-dir', gene_strat_dir + '/gene_distributions',
                           '--tier', 'all'] + groupings_args),
                    depends_on=[]  # Independent - queries database directly
                )
            )

            # Phase 2: Gene-stratified association tests
            gene_strat_args = [
                '--input-dir', gene_strat_dir + '/gene_distributions',
                '--output-dir', gene_strat_dir,
                '--tier', 'all'
            ] + groupings_args
            # TIER 7 gene-stratified importance gate. RF gates on the grouped SHAP ranking (one row per feature, the same unit as LR),
            # so a multi-level feature is not penalized for splitting importance across dummies. The gate enriches for
            # conditionally important associations; it is not a conditional test. LR instead gates on Wald significance (FDR < 0.05).
            if have_rankings:
                # Both pathways gate on their own TIER 6 rankings (grouping is a no-op for LR)
                gene_strat_args.extend(['--shap-file', rankings_file])
            else:
                # No rankings file will be produced (e.g. --quick with RF): nothing to gate on, so opt out
                gene_strat_args.append('--no-shap-filter')
            if self.lr_significant_only and self.algorithm != 'random_forest':
                gene_strat_args.append('--lr-significant-only')
            # Ancestry/ethnicity tests exclude GENIE's "Unknown" sentinel by default, matching the model's missingness handling; this flag restores it
            if self.keep_unknown_category:
                gene_strat_args.append('--keep-unknown-category')
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

            # Phase 5a: Cancer-level hit list composite figures
            scripts.append(
                ScriptConfig(
                    path="code/analysis/posthoc/create_hit_list_composite_figures.py",
                    env="primary",
                    runtime="quick",
                    tier=7,
                    args=[
                        '--input-dir', str(self.tier_dirs[7]),
                        '--output-dir', str(self.tier_dirs[7]) + '/hit_list_figures',
                        '--level', 'cancer'
                    ],
                    depends_on=["analyze_shap_feature_associations.py"]
                )
            )

            # Phase 5b: Gene-level hit list composite figures
            scripts.append(
                ScriptConfig(
                    path="code/analysis/posthoc/create_hit_list_composite_figures.py",
                    env="primary",
                    runtime="quick",
                    tier=7,
                    args=[
                        '--input-dir', str(self.tier_dirs[7]),
                        '--output-dir', str(self.tier_dirs[7]) + '/hit_list_figures',
                        '--level', 'gene'
                    ],
                    depends_on=["analyze_gene_stratified_associations.py"]
                )
            )

            # Cross-pathway hit-set Venn: LR run against an RF base run with gene-stratified hits
            if (have_rankings
                    and self.algorithm == 'logistic'
                    and self.base_run_dir is not None):
                base_gene_strat = (
                    self.base_run_dir / 'tier7_posthoc_statistics' / 'gene_stratified'
                )
                if base_gene_strat.exists() and any(
                    base_gene_strat.rglob('*_gene_significant.csv')
                ):
                    scripts.append(
                        ScriptConfig(
                            path="code/analysis/posthoc/visualize_rf_lr_hit_venn.py",
                            env="primary",
                            runtime="quick",
                            tier=7,
                            args=[
                                '--rf-run-dir', str(self.base_run_dir),
                                '--lr-run-dir', str(self.run_dir),
                                '--output-dir', str(self.tier_dirs[7]) + '/cross_pathway',
                            ],
                            depends_on=["analyze_gene_stratified_associations.py"],
                        )
                    )
                else:
                    # Fails open: the plot is dropped (not an error) when the base run lacks these outputs; verify cross_pathway/ after an LR re-run
                    print(
                        Colors.WARNING
                        + 'WARNING: skipping visualize_rf_lr_hit_venn.py -- base run '
                          'has no RF gene-stratified hits at '
                        + str(base_gene_strat)
                        + '. The cross-pathway output will be MISSING from '
                          'tier7_posthoc_statistics/cross_pathway/.'
                        + Colors.ENDC
                    )

            # Cross-pathway per-cancer AUC plot; DeLong CIs when both runs have test_predictions.csv
            if (have_rankings
                    and self.algorithm == 'logistic'
                    and self.base_run_dir is not None):
                base_tier5 = self.base_run_dir / 'tier5_final_models'
                if base_tier5.exists() and any(
                    base_tier5.rglob('performance_summary.csv')
                ):
                    scripts.append(
                        ScriptConfig(
                            path="code/analysis/posthoc/plot_rf_vs_lr_auc_per_cancer.py",
                            env="primary",
                            runtime="quick",
                            tier=7,
                            args=[
                                '--rf-run-dir', str(self.base_run_dir),
                                '--lr-run-dir', str(self.run_dir),
                                '--output-dir', str(self.tier_dirs[7]) + '/cross_pathway',
                            ],
                            depends_on=(["run_final_model_training.py"]
                                         if tier5_will_run else []),
                        )
                    )
                else:
                    # Fails open: the plot is dropped (not an error) when the base run lacks these outputs; verify cross_pathway/ after an LR re-run
                    print(
                        Colors.WARNING
                        + 'WARNING: skipping plot_rf_vs_lr_auc_per_cancer.py -- base run '
                        'has no RF TIER 5 performance_summary.csv at '
                        + str(base_tier5)
                        + '. The cross-pathway output will be MISSING from '
                          'tier7_posthoc_statistics/cross_pathway/.'
                        + Colors.ENDC
                    )

        return scripts

    def _log(self, message: str, color: str = ''):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"

        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')

        if color:
            print(f"{color}{log_line}{Colors.ENDC}")
        else:
            print(log_line)

    def _check_environment(self) -> bool:
        """Validate environment and dependencies"""
        self._log("Validating environment...", Colors.OKCYAN)

        if not Path(self.db_path).exists():
            self._log(f"ERROR: Database not found at {self.db_path}", Colors.FAIL)
            return False

        self._log(f"✓ Database found: {self.db_path}", Colors.OKGREEN)

        try:
            result = subprocess.run(['conda', 'env', 'list'],
                                  capture_output=True, text=True, check=True)

            if self.primary_env not in result.stdout:
                self._log(f"ERROR: Conda environment '{self.primary_env}' not found", Colors.FAIL)
                return False
            self._log(f"✓ Found conda environment: {self.primary_env}", Colors.OKGREEN)

            if not self.skip_shap and self.algorithm == 'random_forest' and self.ml_env not in result.stdout:
                self._log(f"WARNING: Conda environment '{self.ml_env}' not found (needed for SHAP)", Colors.WARNING)
                self._log(f"  Run TIER 6 SHAP separately or use --algorithm logistic", Colors.WARNING)
            elif not self.skip_shap and self.algorithm == 'random_forest':
                self._log(f"✓ Found conda environment: {self.ml_env}", Colors.OKGREEN)

        except (subprocess.CalledProcessError, FileNotFoundError):
            self._log("ERROR: Conda not found or not accessible", Colors.FAIL)
            return False

        return True

    def _run_script(self, script: ScriptConfig) -> bool:
        """Execute a single script; return True on success."""
        script.status = 'running'

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

        start_time = time.time()

        # Set PYTHONPATH to include modeling directory for oncokb_analyzer imports
        import os
        env = os.environ.copy()
        modeling_dir = str(self.repo_root / "code" / "modeling")
        tools_dir = str(self.repo_root / "code" / "tools")
        env['PYTHONPATH'] = modeling_dir + os.pathsep + tools_dir + os.pathsep + env.get('PYTHONPATH', '')

        # Per-tier timeouts (seconds)
        tier_timeouts = {
            1: 1800,    # 30 min  - independent analyses
            2: 1800,    # 30 min  - data setup
            3: 259200,  # 72 hours - hyperparameter search
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

            if result.stdout:
                self._log("STDOUT:")
                for line in result.stdout.split('\n'):
                    self._log(f"  {line}")

            if result.stderr:
                self._log("STDERR:")
                for line in result.stderr.split('\n'):
                    self._log(f"  {line}")

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

        # A registered script that is not on disk would fail mid-run; catch it up front.
        missing = sorted({s.path for s in self.scripts
                          if not (self.repo_root / s.path).exists()})
        if missing and not self.dry_run:
            hint = ("" if self.tables_only else
                    "\n  This checkout may ship tables-only code; pass --tables-only.")
            raise FileNotFoundError("Registered scripts not found:\n  "
                                    + "\n  ".join(missing) + hint)

        if not self._check_environment():
            self._log("Environment validation failed. Exiting.", Colors.FAIL)
            sys.exit(1)

        print()

        self.results['start_time'] = datetime.now()
        self.results['total'] = len(self.scripts)

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

            if self.results['failed'] > 0 and not self.continue_on_error:
                break

        self.results['end_time'] = datetime.now()

        self._generate_summary()

    def _generate_summary(self):
        """Generate and display summary report"""

        duration = self.results['end_time'] - self.results['start_time']

        print(f"\n\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}SUMMARY{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

        print(f"Total scripts: {self.results['total']}")
        print(f"{Colors.OKGREEN}Successful: {self.results['success']}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed: {self.results['failed']}{Colors.ENDC}")
        if self.dry_run:
            print(f"{Colors.WARNING}Skipped (dry run): {len(self.scripts)}{Colors.ENDC}")
        print(f"Total runtime: {duration}")
        print()

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

        if self.results['success'] > 0:
            print(f"{Colors.OKGREEN}Successful Scripts:{Colors.ENDC}")
            for script in self.scripts:
                if script.status == 'success':
                    duration_str = f"{script.duration:.1f}s" if script.duration else "N/A"
                    print(f"  ✓ {script.path} ({duration_str})")
            print()

        print(f"Complete log: {Colors.OKBLUE}{self.log_file}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

        summary_file = self.run_dir / "run_summary.json"
        summary_data = {
            'mode': self.mode,
            'run_directory': str(self.run_dir),
            'timestamp': self.timestamp,
            'invocation': self.invocation,
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

def _skip_shap_for_quick(algorithm: str) -> bool:
    """Quick mode: RF SHAP is memory-heavy → skip; LR coefficient extraction is fast → keep."""
    if algorithm == 'random_forest':
        print(f"{Colors.WARNING}Note: RF SHAP analysis skipped in quick mode (memory-intensive).{Colors.ENDC}")
        return True
    print(f"{Colors.WARNING}Note: LR coefficient extraction will run in quick mode (fast).{Colors.ENDC}")
    return False


def _build_runner(args, algorithm: str, base_run_dir, run_dir_suffix: str,
                  fresh_split: bool = None) -> 'MasterAnalysis':
    """Construct a MasterAnalysis runner for a single pathway.

    `fresh_split` defaults to args.fresh_split; the dual-pathway loop sets it False for the LR pass
    so LR consumes the split the RF pass built.
    """
    skip_shap = args.mode == 'quick' and _skip_shap_for_quick(algorithm)
    tier3_algorithms = 'rf,lr,xgb' if args.include_xgboost else args.tier3_algorithms
    if fresh_split is None:
        fresh_split = args.fresh_split

    return MasterAnalysis(
        mode=args.mode,
        dry_run=args.dry_run,
        continue_on_error=not args.stop_on_error,
        skip_shap=skip_shap,
        parallel=args.parallel,
        feature_config_path=args.feature_config,
        institution_weight_alpha=args.institution_weight_alpha,
        base_run_dir=str(base_run_dir) if base_run_dir else None,
        algorithm=algorithm,
        lr_significant_only=args.lr_significant_only,
        keep_unknown_category=args.keep_unknown_category,
        shap_file_override=args.shap_file,
        tier3_algorithms=tier3_algorithms,
        split_strategy=args.split_strategy,
        fresh_split=fresh_split,
        run_dir_suffix=run_dir_suffix,
        run_dir_override=getattr(args, 'run_dir', None),
        hyperparams_source=args.hyperparams_source,
        hyperparams_dir_override=args.hyperparams_dir,
        exclude_cancer_types=getattr(args, 'exclude_cancer_types', None),
        tables_only=args.tables_only,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Master script to run all Balder analyses and ML experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_analyses.py --full                       # RF+LR, all tiers (~5 hours)
  python run_all_analyses.py --quick                      # RF+LR smoke test (~30 min)
  python run_all_analyses.py --algorithm logistic         # Single-pathway run (LR only)
  python run_all_analyses.py --post-model-selection \\
      --algorithm logistic --base-run-dir runs/<rf>       # Downstream-only LR re-run
  python run_all_analyses.py --include-xgboost            # Add XGBoost to TIER 3 search
  python run_all_analyses.py --analysis-only              # Skip ML experiments
  python run_all_analyses.py --ml-only                    # Only ML (TIER 2-5)
  python run_all_analyses.py --dry-run --list             # Preview without running
        """
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--full', action='store_const', const='full', dest='mode',
                           help='Run all analyses (hours)')
    mode_group.add_argument('--quick', action='store_const', const='quick', dest='mode',
                           help='Quick test run (~30 min; skips RF SHAP to save memory)')
    mode_group.add_argument('--ml-only', action='store_const', const='ml-only', dest='mode',
                           help='Only ML experiments')
    mode_group.add_argument('--analysis-only', action='store_const', const='analysis-only', dest='mode',
                           help='Only analysis scripts')
    mode_group.add_argument('--post-model-selection', action='store_const', const='post-model-selection', dest='mode',
                           help='Downstream-only re-run (TIER 4-7); requires --base-run-dir')
    mode_group.add_argument('--tier7-only', action='store_const', const='tier7-only', dest='mode',
                           help='Re-run TIER 7 ONLY against an existing TIER 6 ranking. For '
                                'changes to the post-hoc gate or tests that do not touch the '
                                'model: nothing is re-fitted, so SHAP values and AUCs are '
                                'carried over unchanged. The run dir must already contain '
                                'tier6_shap_analysis/ (copy it from the base run) or you must '
                                'pass --shap-file; it fails closed otherwise.')
    parser.add_argument('--run-dir', type=str, default=None,
                       help='Write into this EXISTING run directory instead of creating a new\n'
                            'timestamped one. Only valid with --tier7-only.')

    parser.add_argument('--tables-only', action='store_true',
                        help='Run only scripts that write manuscript tables: skip '
                             'figure-only and exploratory scripts, and run mixed '
                             'scripts in their table-only mode.')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would run without executing')
    parser.add_argument('--list', action='store_true',
                       help='List all scripts and exit')
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
                           'Power parameter for institutional sample weighting (0.0 to 1.0). '
                           '0.0 = no weighting (default), 0.5 = moderate correction, 1.0 = full inverse frequency.'
                       ))
    parser.add_argument('--split-strategy', type=str, choices=['patient', 'row'], default='patient',
                       help=(
                           "TIER 2 train/test split unit. 'patient' (default, since 2026-08-13) "
                           "assigns each PATIENT_ID to one split so no patient appears in both. "
                           "'row' is the legacy row shuffle over the sample-level modeling table, "
                           "which places 2,409 patients in both splits (15.3%% of test rows); use "
                           "it only to reproduce runs made before 2026-08-13."
                       ))
    parser.add_argument('--exclude-cancer-types', nargs='+', default=None, metavar='CODE',
                       help=(
                           'OncoTree codes to drop from the cohort AFTER TIER 2a eligibility '
                           '(e.g. --exclude-cancer-types ODG). Writes a filtered '
                           'cancer_type_group_definitions.csv into this run and passes it to '
                           'TIER 5 and TIER 7; TIER 6 follows automatically because it '
                           'enumerates cancer types from the models TIER 5 wrote. The database '
                           'and the train/test split are NOT modified -- the excluded codes are '
                           'simply never fitted or tested, so every retained cancer type keeps '
                           'the models and AUCs it would have had. Fails closed if a named code '
                           'is not in the eligibility file. Requires a reachable TIER 2a '
                           'cancer_type_group_definitions.csv, i.e. --base-run-dir on a re-run.'
                       ))
    parser.add_argument('--run-label', type=str, default=None,
                       help=(
                           'Label embedded in the run directory name, before the pathway suffix: '
                           'runs/run_<ts>_<label>_rf and _lr. Use to make arm membership legible on '
                           'disk (e.g. --run-label sampleunit). Renaming run dirs after the fact '
                           'breaks the LR run\'s recorded base_run_dir pointer, so set it up front.'
                       ))
    parser.add_argument('--fresh-split', action='store_true',
                       help=(
                           'Build a NEW TIER 2b train/test split in this run even when '
                           '--base-run-dir is set, while still reusing the base run\'s '
                           'hyperparameters (TIER 3 stays skipped). Without this, TIER 2b and '
                           'TIER 3 are both gated on the absence of --base-run-dir, so '
                           're-splitting forces a full ~27h hyperparameter search. Downstream '
                           'tiers read this run\'s split, not the base run\'s.'
                       ))
    parser.add_argument('--hyperparams-dir', type=str, default=None,
                       help=(
                           'Explicit TIER 3 parameter directory for TIER 5, overriding the '
                           'auto-resolved one. Must contain <algorithm>_universal_params.json '
                           'directly or under run_*/. Intended for Step-2-validated parameter '
                           'sets. Validated at '
                           'startup, not four tiers in. Ignored when --hyperparams-source '
                           'defaults is passed.'
                       ))
    parser.add_argument('--hyperparams-source', type=str, choices=['search', 'defaults'],
                       default='search',
                       help=(
                           "Where TIER 5's final-model hyperparameters come from. 'search' "
                           "(default) wires TIER 3's result into TIER 5 -- this wiring did not "
                           "exist before 2026-08-26, so every earlier run silently used the "
                           "hand-tuned module constants. 'defaults' keeps those constants; they "
                           'are NOT search-derived and were selected with reference to held-out '
                           'performance, so choosing this is a disclosable decision. '
                           "With 'search', a run that cannot resolve a TIER 3 result aborts "
                           'rather than falling back.'
                       ))
    parser.add_argument('--base-run-dir', type=str, default=None,
                       help=(
                           'Path to an existing run directory to reuse train-test split and hyperparameter results. '
                           'Skips TIER 1, 2, and 3 in the new run. '
                           'Example: --base-run-dir runs/<run>'
                       ))
    parser.add_argument('--algorithm', type=str, default=None,
                       choices=['random_forest', 'logistic'],
                       help=(
                           'ML algorithm for TIER 4-7. Default (omitted) runs BOTH RF and LR pathways '
                           'in a single invocation: RF first, then LR with --base-run-dir set to RF, '
                           'which auto-fires the cross-pathway plots in TIER 7.'
                       ))
    parser.add_argument('--tier3-algorithms', type=str, default='rf,lr',
                       help=(
                           'Comma-separated algorithm shorthand for TIER 3 hyperparameter search '
                           '(default: rf,lr). Use --include-xgboost as a convenient override.'
                       ))
    parser.add_argument('--include-xgboost', action='store_true',
                       help='Include XGBoost in TIER 3 search (overrides --tier3-algorithms to rf,lr,xgb)')
    parser.add_argument('--lr-significant-only', action='store_true',
                       help=(
                           'When the LR pathway runs, restrict TIER 7 post-hoc testing to features '
                           'with Wald-significant LR coefficients (FDR < 0.05).'
                       ))
    parser.add_argument('--keep-unknown-category', action='store_true',
                       help=(
                           'Retain GENIE\'s "Unknown" sentinel as a category level in the TIER 7 '
                           'ancestry/ethnicity chi-square tests. Default (since 2026-09-18) is '
                           'to EXCLUDE it, so the post-hoc estimand matches the modeling pipeline\'s '
                           'treatment of missingness. Pass this ONLY to reproduce earlier runs.'
                       ))
    parser.add_argument('--shap-file', type=str, default=None,
                       help=(
                           'Override the auto-constructed TIER 6 rankings file passed to TIER 7 '
                           'post-hoc scripts. Pass an absolute path or a path relative to the repo root.'
                       ))

    parser.set_defaults(mode='full')

    args = parser.parse_args()

    if args.mode == 'post-model-selection' and not args.base_run_dir:
        parser.error('--post-model-selection requires --base-run-dir')

    # Single pathway: --algorithm set, or analysis-only (TIER 4-7 do not run; pinned to RF)
    single_pathway = args.algorithm is not None or args.mode == 'analysis-only'

    if single_pathway:
        algorithm = args.algorithm or 'random_forest'
        runner = _build_runner(
            args,
            algorithm=algorithm,
            base_run_dir=args.base_run_dir,
            run_dir_suffix=(args.run_label or ''),
        )
        if args.list:
            runner.list_scripts()
            sys.exit(0)
        runner.run()
        # Exit non-zero on any tier failure so CI/HPC callers see it
        sys.exit(1 if runner.results['failed'] > 0 else 0)

    # Dual-pathway: RF then LR, with LR reusing RF's run_dir as --base-run-dir.
    print(f"{Colors.HEADER}{Colors.BOLD}Dual-pathway run: RF then LR{Colors.ENDC}")

    _label = f"{args.run_label}_" if args.run_label else ""
    rf_runner = _build_runner(
        args,
        algorithm='random_forest',
        base_run_dir=args.base_run_dir,
        run_dir_suffix=f'{_label}rf',
    )
    if args.list:
        print(f"{Colors.OKCYAN}--- RF pathway ---{Colors.ENDC}")
        rf_runner.list_scripts()
        print(f"{Colors.OKCYAN}--- LR pathway ---{Colors.ENDC}")
        try:
            lr_preview = _build_runner(
                args, algorithm='logistic', base_run_dir=rf_runner.run_dir,
                run_dir_suffix=f'{_label}lr', fresh_split=False,
            )
            lr_preview.list_scripts()
        except FileNotFoundError:
            # Under --list the RF pass has not written the split yet, so skip the fail-closed lookup (preview only)
            print("  (LR pathway preview unavailable: it consumes the split the RF pathway "
                  "produces, which does not exist until the RF pathway runs. The real run "
                  "resolves this — RF executes before the LR runner is built.)")
        sys.exit(0)

    rf_runner.run()

    lr_runner = _build_runner(
        args,
        algorithm='logistic',
        base_run_dir=rf_runner.run_dir,
        run_dir_suffix=f'{_label}lr',
        fresh_split=False,  # consume the RF pass's split; never build a second one
    )
    lr_runner.run()

    print(f"\n{Colors.OKGREEN}{Colors.BOLD}Dual-pathway run complete.{Colors.ENDC}")
    print(f"  RF run: {rf_runner.run_dir}")
    print(f"  LR run: {lr_runner.run_dir}")

    # Propagate non-zero exit if either pathway had any tier failure.
    total_failed = rf_runner.results['failed'] + lr_runner.results['failed']
    if total_failed > 0:
        print(f"{Colors.FAIL}{total_failed} tier script(s) failed across the two pathways.{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
