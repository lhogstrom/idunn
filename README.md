# Balder/Idunn: ML Analysis of Clinically Actionable Cancer Biomarkers

A cancer genomics research platform that harmonizes tumor profile data from AACR GENIE and analyzes demographic associations with clinically actionable biomarkers using machine learning.

## Overview

This project investigates whether patient demographics (age, sex, race, ethnicity) are associated with the presence of clinically actionable genomic alterations in cancer patients. It combines:

## Quick Start

```bash
# 1. Clone and setup environment
conda env create -f code/dependencies/balder_conda.yml
conda activate balder_conda_env

# 2. Set database path (one-time setup)
export BALDER_DB_PATH="/path/to/balder-harmonized-biomarker-data-v*.sqlite"

# 3. Run quick test (~30 min)
cd code && python run_all_analyses.py --quick

# 4. Or run full analysis (~5 hours)
cd code && python run_all_analyses.py --full
```

## Scientific Workflow

The analysis follows a rigorous 7-tier workflow designed to prevent data leakage:

| Tier | Step | Description |
|------|------|-------------|
| 1 | Independent Analyses | Demographics, institutional patterns, feature EDA |
| 2a | Cancer Type Grouping | Define 34 hierarchical cancer type groups |
| 2b | **Train/Test Split** | 80/20 stratified split (BEFORE any modeling) |
| 3 | Hyperparameter Optimization | Tune models using ONLY training data |
| 4 | Feature Set Comparison | Compare demographic vs technical features |
| 5 | **Final Evaluation** | Train final models, evaluate on held-out test set |
| 6 | SHAP Analysis | Feature importance and interactions |
| 7a | Post-Hoc Statistics | Cancer-level hypothesis testing with FDR correction |
| 7b | Gene-Stratified Analysis | Per-gene associations (top 2 genes per cancer type) |

## Project Structure

```
idunn/
├── code/
│   ├── modeling/           # ML framework (oncokb_analyzer package)
│   │   └── tests/          # Unit and integration tests
│   ├── analysis/           # Active analysis scripts (demographics, SHAP, post-hoc)
│   ├── pipeline/           # Nextflow pipeline R scripts
│   ├── tools/              # Shared utilities (db_config, feature_config, multi_tier_utils, shap_utils)
│   ├── deprecated/         # Superseded scripts (historical reference)
│   └── run_all_analyses.py # Master orchestration script
├── workflow/
│   └── main.nf             # Nextflow pipeline definition
├── runs/                   # Unified timestamped run outputs (gitignored)
│   ├── motivation_for_runs.log  # Human-readable run history (tracked)
│   └── run_YYYYMMDD_HHMMSS/     # Each run's outputs organized by tier
```

**Note**: Output directories (`runs/`, `ml_experiments/`) are excluded from git to prevent repo bloat.

## Installation

### Prerequisites
- Conda/Miniconda
- Nextflow (for data pipeline)

### Environment Setup

**Primary environment** (pipeline + ML):
```bash
conda env create -f code/dependencies/balder_conda.yml
conda activate balder_conda_env
```

**ML environment** (advanced features: SHAP interactions, LIME):
```bash
conda env create -f code/dependencies/balder_ml_env.yml
conda activate balder_ml_env
```

### Database Configuration

Set the `BALDER_DB_PATH` environment variable:
```bash
# Add to ~/.bashrc or ~/.zshrc
export BALDER_DB_PATH="/path/to/balder-harmonized-biomarker-data-v*.sqlite"
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

## Usage

### Master Analysis Script

```bash
cd code

# Quick test (~30 min, skips memory-intensive SHAP)
python run_all_analyses.py --quick

# Full production run (~5 hours)
python run_all_analyses.py --full

```
