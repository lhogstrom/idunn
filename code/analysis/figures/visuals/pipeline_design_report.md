# Python Design Considerations: ML and Post-Hoc Pipeline

## Architecture Overview

The Balder pipeline is a 7-tier scientific workflow orchestrated by `run_all_analyses.py`, which dispatches ~30 scripts as isolated subprocesses across two conda environments. Three layers collaborate: (1) a master orchestrator managing execution order and environment routing, (2) an ML framework package (`oncokb_analyzer/`) encapsulating model training and evaluation, and (3) shared utility modules (`code/tools/`) providing data access, feature configuration, and ranking normalization. The design prioritizes reproducibility and statistical rigor (nested CV, strict train/test separation) over maximal code reuse.

## Orchestration

The orchestrator uses a **subprocess-dispatch model**: each analysis script runs in its own `conda run` invocation with an injected `PYTHONPATH`. Scripts are represented as `ScriptConfig` dataclass objects carrying metadata (tier, conda environment, timeout, CLI arguments, `depends_on` references). This isolation prevents import conflicts — particularly between `balder_conda_env` (primary) and `balder_ml_env` (SHAP requires a different scikit-learn version) — at the cost of repeated DB loads across scripts. Mode flags (`--full`, `--quick`, `--post-model-selection`) filter which tiers run; `--algorithm` controls conditional branching at TIER 6 (SHAP analysis vs. LR coefficient extraction). Per-tier timeouts range from 30 minutes (setup) to 72 hours (hyperparameter search). The `depends_on` field exists but is not validated at runtime; correct ordering relies on sequential tier dispatch.

## ML Framework (`oncokb_analyzer/`)

The framework uses **dataclass-driven configuration**: `Config` and `ExperimentConfig` carry all experiment parameters, validated at construction. `ModelTrainer` implements a **strategy pattern** for algorithm selection (RF, LR, XGBoost) and CV strategy (`StratifiedKFold`, `GroupKFold`, `StratifiedGroupKFold`). `DataLoader` provides a repository-pattern interface to SQLite, loading the full patient table once and filtering by cancer type in memory. The `ColumnTransformer` preprocessor (numeric imputation + one-hot encoding) is wrapped in a `Pipeline` to ensure transforms fit only on training folds. `NestedCVOptimizer` separates performance estimation (outer folds) from hyperparameter tuning (inner folds via `RandomizedSearchCV`), returning a consensus parameter set selected by majority vote with a stability score.

## Configuration Propagation

All feature definitions flow from a single source: `code/config/features.yaml`. The `feature_config.py` module provides a cached loader with a context manager for temporary config overrides during experiments. At run start, the orchestrator copies the YAML into the timestamped run directory for reproducibility. Downstream modules (`preprocessing.py`, `feature_name_mapper.py`, post-hoc scripts) all resolve features through this facade. Environment variables (`BALDER_DB_PATH`, `BALDER_FEATURES_CONFIG`, `BALDER_OUTPUT_ROOT`) provide deployment-time overrides. Configuration management spans four mechanisms (argparse, dataclass, YAML, environment variables) without a unifying hierarchy — a pragmatic but fragile arrangement.

## Dual-Algorithm Abstraction

The pipeline supports two analysis pathways that converge at TIER 7. The RF pathway produces SHAP importance rankings via `SHAPPrimaryAnalyzer` (which selects `TreeExplainer` or `LinearExplainer` based on model type). The LR pathway extracts standardized coefficients, odds ratios, and Wald statistics via `extract_lr_coefficients.py`. The key integration point is `shap_utils.py`: its `load_shap_rankings_adaptive()` function auto-detects the input format and normalizes both to a common DataFrame schema (`ONCOTREE_CODE`, `feature`, `mean_abs_shap`, `feature_rank`). This allows all 13 post-hoc scripts to operate identically regardless of upstream algorithm — adding full LR support required modifying only the ranking-loading layer, not TIER 7 code.

## Post-Hoc Scripts

The ~13 scripts in `code/analysis/posthoc/` follow a **standalone template pattern**: each is an independent CLI tool with its own `argparse`, DB connection, logging, and output routing via `--output-dir`. They share no base class. The common workflow is: load patient data → filter by feature importance rankings (top-5 per cancer type via `shap_utils.feature_in_top_n_for_cancer()`) → run statistical tests → apply Benjamini-Hochberg FDR correction per feature category. Test selection is adaptive (Fisher's exact when expected cell frequencies < 5, chi-square otherwise; Haldane +0.5 correction for zero cells). This loose coupling makes individual scripts easy to run and debug in isolation, but leads to duplicated boilerplate and independent database queries across scripts.

## Key Trade-Offs

| Decision | Benefit | Cost |
|----------|---------|------|
| Subprocess isolation per script | No import conflicts; independent timeouts | Repeated DB loads; no shared in-memory state |
| Loose coupling in post-hoc scripts | Easy to add/debug individual analyses | Boilerplate duplication across ~13 scripts |
| Centralized YAML feature config | Single source of truth; reproducible | Extra indirection; cache invalidation edge cases |
| Algorithm-agnostic TIER 7 interface | One set of post-hoc scripts serves both RF and LR | Format-detection complexity in `shap_utils.py` |
| Filesystem-based inter-tier communication | Scripts are independently runnable; outputs inspectable | Path resolution fragility; working directory sensitivity |
| `sys.path.insert()` for imports | No package installation required | Fragile; depends on working directory conventions |
