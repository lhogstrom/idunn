# Master Analysis Runner - Documentation

## Overview

`run_all_analyses.py` is a master orchestration script that executes all Balder analysis scripts and ML experiments in the correct order with proper dependency management and environment handling.

## Quick Start

```bash
# Preview what would run
python run_all_analyses.py --list

# Quick test run (~1.5-2 hours)
python run_all_analyses.py --quick

# Full analysis (5-6 hours)
python run_all_analyses.py --full

# Dry run (no execution)
python run_all_analyses.py --quick --dry-run
```

## Execution Modes

### `--full` (default)
- Runs **all** analysis scripts and ML experiments
- Includes full ML training (5 random seeds)
- Includes SHAP interaction analysis
- **Estimated runtime**: ~5-6 hours

### `--quick`
- Same scripts as `--full` but with quick-test flags
- ML uses `--quick-test` mode (reduced iterations)
- SHAP interactions on 4 cancer types only
- **Estimated runtime**: ~1.5-2 hours

### `--analysis-only`
- Runs only analysis scripts (no ML experiments)
- Skips: `run_feature_set_experiment.py`
- **Estimated runtime**: ~1 hour

### `--ml-only`
- Runs only ML experiments
- Skips all analysis scripts
- **Estimated runtime**: ~3-4 hours (full) or ~30 min (quick)

## Options

### `--skip-shap-interactions`
Skip SHAP interaction analysis (avoids environment switching to `balder_ml_env`)

### `--stop-on-error`
Abort execution if any script fails (default: continue)

### `--dry-run`
Preview what would run without actually executing scripts

### `--list`
Show all scripts that would be executed and exit

### `--parallel` *(experimental)*
Run Tier 1 (independent) scripts in parallel (not yet implemented)

## Execution Tiers

Scripts are executed in 7 tiers to respect dependencies (scientific workflow):

### Tier 1: Independent Analyses
Can run in any order (or parallel). Includes:
- `cancer_type_summary_stats.py`
- `analyze_demographics_by_cancer_type.py`
- `analyze_all_cancer_types_institutional_stats.py`
- `feature_engineering_analysis.py`
- `plot_variant_demographic_stratification.py`

### Tier 2: Data Setup
Cancer-type groupings and train/test split definition. Includes:
- `define_cancer_type_groupings.py` (21 OncoTree codes ≥250 actionable)
- `define_train_test_split.py` (80/20 stratified split)

### Tier 3: Hyperparameter Optimization
Universal hyperparameter tuning (training data only). Includes:
- `run_ml_model_selection_full.py`

### Tier 4: Feature Set Comparison
Compare demographic vs technical vs both (training data only). Includes:
- `run_feature_set_experiment.py`

### Tier 5: Final Model Training
Train final models and evaluate on held-out test set. Includes:
- `run_final_model_training.py`

### Tier 6: SHAP Analysis
Requires `balder_ml_env` and pre-trained TIER 5 models. Includes:
- `compute_shap_primary_values.py` (TIER 6a - feature importance)
- `compute_shap_interactions.py` (TIER 6b - pairwise interactions)

### Tier 7: Post-Hoc Statistics
Statistical testing based on SHAP feature rankings. Includes:
- `analyze_age_by_actionability.py`
- `analyze_shap_feature_associations.py`
  - **TIER 6→7 Integration**: In `--full` mode, SHAP file path is automatically passed from TIER 6a
  - In `--quick` mode, auto-detects existing SHAP files
- `plot_shap_feature_ranks.py` (RF pathway only) — boxplots of SHAP feature ranks across cancer types; output at `tier7_posthoc_statistics/shap_ranks/`
- `plot_rf_shap_lr_or_feature_ranks.py` (LR pathway with `--base-run-dir <rf_run>` only) — 2-row figure of RF SHAP feature ranks (top, from base run) vs. LR odds-ratio feature ranks (bottom, from current run) across cancer types; output at `tier7_posthoc_statistics/cross_pathway/`. Also runnable standalone (auto-detects newest RF and LR run dirs, writes to `runs/shap_vs_lr_comparison/`).
- `visualize_rf_lr_hit_venn.py` (LR pathway with `--base-run-dir <rf_run>` only) — Venn diagrams of FDR-significant gene-stratified hits in RF vs. LR: 2×3 per-feature panel (age, sex, ancestry, ethnicity, sample_type, technical) plus an all-features pooled Venn, with companion membership and summary CSVs; output at `tier7_posthoc_statistics/cross_pathway/`. Also runnable standalone (auto-detects newest RF and LR run dirs, writes to `runs/shap_vs_lr_comparison/`).
- `plot_rf_vs_lr_auc_per_cancer.py` (LR pathway with `--base-run-dir <rf_run>` only) — 2-panel bar chart of test AUC per cancer type (RF top, LR bottom; shared x-axis ordered by RF test AUC). Emits both variants per invocation: paired train+test bars and held-out-test-only. DeLong analytic 95% CI error bars render automatically when `test_predictions.csv` (saved in TIER 5) is present in both runs. Output at `tier7_posthoc_statistics/cross_pathway/`. Also runnable standalone (writes to `runs/shap_vs_lr_comparison/`).

## Environment Requirements

### Primary Environment: `balder_conda_env`
Used for most scripts (Tier 1, 2, 3)

Required packages:
- Python 3.9+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- sqlalchemy

### ML Environment: `balder_ml_env`
Used only for SHAP interactions (Tier 4)

Required packages:
- All packages from `balder_conda_env`
- shap
- Additional ML libraries

## Output

### Unified Run Structure (January 2026)
Each run creates a timestamped directory organizing all outputs by tier:
```
runs/run_YYYYMMDD_HHMMSS/
├── run.log                           # Execution log
├── run_summary.json                  # Run metadata + invocation context (argv, cwd,
│                                     # algorithm, base_run_dir); per-script status
├── tier1_independent_analyses/       # Demographics, institutional, feature analysis
├── tier2_data_setup/                 # Groupings and train/test split
├── tier3_hyperparameter_optimization/
├── tier4_feature_comparison/
├── tier5_final_models/
├── tier6_shap_analysis/
│   ├── primary_values/               # TIER 6a output
│   │   └── combined_feature_importance.csv  # ← Used by TIER 7
│   └── interactions/                 # TIER 6b output
└── tier7_posthoc_statistics/
    ├── age_actionability/
    ├── shap_associations/            # Uses SHAP file from TIER 6a
    ├── shap_ranks/                   # RF pathway: SHAP rank boxplots
    └── cross_pathway/                # LR-with-base-run: feature-rank comparison + hit-set Venns
```

### Exit codes (added 2026-05-23)

The orchestrator propagates a non-zero exit code when any tier script fails:

- `0` — every tier script succeeded (and, for dual-pathway, both pathways succeeded).
- `1` — at least one tier script failed (`runner.results['failed'] > 0`), OR the smoke regen surfaced cascading failures. CI / HPC schedulers can rely on this signal.
- `2` — argparse / invocation error (e.g., `--post-model-selection` without `--base-run-dir`).

Failures in one tier do NOT abort the orchestrator by default (`--continue-on-error` is the default, so the run proceeds and you get a complete picture in `run_summary.json`). Pass `--stop-on-error` to abort on the first failure.

Previously (pre-2026-05-23) the orchestrator always exited 0, masking cascading failures — a CI system would mark a 0/22-successful run as green. Always check `run_summary.json`'s `failed` count and the exit code together.

### Legacy Output Locations
Standalone script runs still output to:
- `analysis_output/cancer_type_summary_stats/`
- `analysis_output/demographic_analysis/`
- `analysis_output/institutional_actionability_patterns/`
- `analysis_output/feature_engineering_analysis/`
- `analysis_output/posthoc_statistics/`
- `analysis_output/level1_variant_analysis/`
- `analysis_output/shap_primary_values/`
- `ml_experiments/feature_comparison_*/`

## Examples

### Quick validation run before full analysis
```bash
python run_all_analyses.py --quick --dry-run
python run_all_analyses.py --quick
```

### Run only new analyses (skip ML)
```bash
python run_all_analyses.py --analysis-only
```

### Full production run
```bash
# Preview
python run_all_analyses.py --full --list

# Execute
nohup python run_all_analyses.py --full > master_run.out 2>&1 &
```

### Run without SHAP interactions
```bash
python run_all_analyses.py --quick --skip-shap-interactions
```

## Monitoring Progress

### During execution
```bash
# Watch log file
tail -f analysis_output/master_runs/master_run_*.log

# Check which scripts are running
ps aux | grep python | grep -E "(analysis|modeling)"
```

### After completion
Check the summary at the end of the log file or view the JSON summary:
```bash
cat analysis_output/master_runs/master_run_YYYYMMDD_HHMMSS.json
```

## Troubleshooting

### "Database not found"
Ensure database exists at expected path:
```bash
ls /Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/*.sqlite
```

### "Conda environment not found"
```bash
conda env list | grep balder
# Should show: balder_conda_env and balder_ml_env
```

Create missing environment:
```bash
conda env create -f code/dependencies/balder_conda.yml
conda env create -f code/dependencies/balder_ml_env.yml
```

### Script fails but execution continues
This is expected behavior (default: `--continue-on-error`)

To stop on first failure:
```bash
python run_all_analyses.py --quick --stop-on-error
```

### Environment activation fails
Make sure conda is initialized in your shell:
```bash
conda init bash  # or zsh, etc.
# Restart shell
```

### Long-running timeout
Scripts have a 2-hour timeout. If legitimate analysis needs longer:
1. Run that script separately
2. Modify timeout in `run_all_analyses.py` (line ~225)

## Estimated Runtimes

Based on current hardware (approximate):

| Mode | Duration | Notes |
|------|----------|-------|
| `--quick` | ~1.5-2h | Recommended for testing |
| `--full` | ~5-6h | Production run |
| `--analysis-only` | ~1h | No ML experiments |
| `--ml-only --quick` | ~30min | ML only, quick test |
| `--ml-only --full` | ~3-4h | ML only, full training |

Individual script runtimes (full mode):
- Tier 1 analyses: 5-20 min each
- ML experiment: 2-3 hours
- SHAP interactions: 1-2 hours

## Integration with Workflow

### Before running:
1. Ensure database is up-to-date (`workflow/main.nf`)
2. Activate base conda environment
3. Check available disk space (outputs can be several GB)

### After running:
1. Review summary in log file
2. Check for failed scripts
3. Validate key outputs exist
4. Consider committing result summaries to git

## Future Enhancements

- [ ] Parallel execution for Tier 1 scripts
- [ ] Email notification on completion
- [ ] Slack/webhook integration for long runs
- [ ] Automatic result validation checks
- [ ] Resume capability (skip completed scripts)
- [ ] YAML configuration file support
