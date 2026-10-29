# Balder/Idunn: ML Analysis of Clinically Actionable Cancer Biomarkers

A cancer genomics research platform that harmonizes tumor profile data from AACR GENIE and analyzes demographic associations with clinically actionable biomarkers using machine learning.

## Overview

This project investigates whether patient demographics (age, sex, ancestry, ethnicity) are associated with the presence of clinically actionable (OncoKB Level 1) genomic alterations, per cancer type, in AACR GENIE. Access to the required MAF and clinical data must requested here: https://aacrprojectgenie.org/. 

## Quick Start

```bash
# 1. Environments (the second is used for RF SHAP values)
conda env create -f code/dependencies/balder_conda.yml
conda env create -f code/dependencies/balder_ml_env.yml
conda activate balder_conda_env

# 2. Point at the harmonized database (see "Building the database")
export BALDER_DB_PATH="/path/to/balder-harmonized-biomarker-data-v20250522-ageUpper-patientunit.sqlite"

# 3. Rebuild every table (from the repo root)
python code/reproduce_manuscript_tables.py
```

| Output | Content |
|---|---|
| `Table_1_patient_characteristics.csv` | Table 1 |
| `Table_S1a/b_*.csv` | Eligibility-threshold sensitivity |
| `Table_S2_feature_definitions.csv` | Model features |
| `Table_S3_ml_metrics.csv` | Per-cancer RF and LR performance (DeLong 95% CI) |
| `Table_S4a-c_full_hit_table_*.csv` | All gene-stratified association tests |
| `Table_S5_crude_vs_adjusted_all_features.csv` | Institution-adjusted associations |
| `Table_S6_sex_enrichment.csv` | Ungated per-cancer sex comparison |
| `Table_S7_unknown_sensitivity.csv` | Ancestry/ethnicity tests with and without `Unknown` |
| `Table_S10a-c_*.csv`, `Table_S11_*`, `Table_S12a/b_*` | Cohort breakdowns, hyperparameters, FDR-scope sensitivity |
| `core/` | Cohort-flow counts, cancer-type definitions, institutional variation, RF vs LR comparison |

Per-pathway outputs (models, test predictions, feature importance, every post-hoc test) stay in the run directories under `runs/`.

### Hyperparameters

The random forest uses `code/config/hyperparameters/random_forest_universal_params.json`, selected on training data only in two steps: nested cross-validation over a pooled hyperparameter search (TIER 3), then a per-cancer-type validation of tree depth. Logistic regression uses the module defaults. To repeat the selection (~8 h):

```bash
cd code && python run_all_analyses.py --ml-only --algorithm random_forest \
    --institution-weight-alpha 0.5                                  # TIER 2-3 search
python code/analysis/posthoc/diagnose_rf_depth_capacity.py --candidates --extend \
    --split-file runs/<run>/tier2_data_setup/train_test_split/train_test_split.csv
```

## Building the database

1. `workflow/main.nf` (Nextflow) runs the R scripts in `code/pipeline/r_pipeline_scripts/` on the GENIE release and OncoKB annotations to produce the harmonized SQLite database.
2. `code/tools/patch_age_redaction.py` recodes GENIE's redacted ages (`<18`, `>89`) on a copy of the database.
3. `code/tools/collapse_modeling_table_to_patient_unit.r` collapses the modeling table to one record per patient and cancer type.

## Scientific Workflow

| Tier | Step | Description |
|------|------|-------------|
| 1 | Cohort description | Table 1, cohort flow counts, institutional variation |
| 2a | Cancer Type Eligibility | OncoTree codes with >= 250 actionable patients |
| 2b | **Train/Test Split** | 80/20 split by patient, BEFORE any modeling |
| 3 | Hyperparameter Optimization | Nested CV on training data only |
| 4 | Feature Set Comparison | Demographic vs technical features |
| 5 | **Final Evaluation** | Per-cancer models, evaluated on the held-out test set |
| 6 | Feature Importance | RF: SHAP values; LR: standardized coefficients |
| 7 | Post-Hoc Statistics | Cancer- and gene-level association tests, BH-FDR per feature category |

## Project Structure

```
idunn/
├── code/
│   ├── reproduce_manuscript_tables.py  # One-command reproduction of all tables
│   ├── run_all_analyses.py             # Tier orchestrator (use --tables-only in this repo)
│   ├── config/                         # Features, pinned hyperparameters, table checksums
│   ├── modeling/                       # ML framework (oncokb_analyzer) + unit tests
│   ├── analysis/                       # Cohort, post-hoc, robustness and reporting scripts
│   ├── pipeline/r_pipeline_scripts/    # R scripts driven by workflow/main.nf
│   └── tools/                          # Shared utilities and database patch steps
├── curated/lit_review/                 # Curated literature status for Table S4
├── workflow/main.nf                    # Nextflow pipeline
└── runs/                               # Run outputs (gitignored)
```

## Tests

```bash
cd code/modeling && python -m pytest tests/ -m "not slow and not integration"
```

## Configuration on Cancell server
(Utilize screen if running a larger job)
module purge
module load Miniconda3/23.5.2-0
* Clone the repo into user dir such as: /data/nobackup/user

### Activate env
conda env create -f code/dependencies/balder_ml_env.yml
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/user/miniconda3/envs/balder_conda_env

### sym link for conda location
ln -s /home/user/miniconda3/envs/balder_conda_env /home/user/.conda/envs/balder_conda_env
