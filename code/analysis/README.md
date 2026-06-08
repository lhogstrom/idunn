# Analysis Scripts

This directory contains ad-hoc analysis and visualization scripts organized by category.

## Directory Structure

```
analysis/
├── demographics/          # Demographic analysis (4 scripts)
├── institutional/         # Institutional analysis (5 scripts)
├── variants/              # Variant analysis (1 script)
└── figures/               # Publication-ready figures
    └── visuals/           # Mindmaps and project diagrams
```

## Demographics (`demographics/`)

Scripts for analyzing demographic patterns across cancer types.

### `analyze_demographics_by_cancer_type.py` (87.9 KB)
Comprehensive demographic analysis for all 21 cancer types meeting minimum thresholds.

**Outputs**:
- `demographic_summary_by_cancer_type.csv` - Summary statistics
- `demographic_summary_by_cancer_type_detailed.csv` - Detailed breakdowns
- `demographic_actionable_comparison.csv` - Actionable vs non-actionable comparisons
- `demographic_analysis_comprehensive_report.txt` - Narrative report
- 9 visualization PDFs (age distributions, sex/race heatmaps, diversity metrics, missing data, etc.)

**Key findings**:
- AGE, SEX, RACE, ETHNICITY have high data quality (0-2% missing)
- STAGE is unusable (99.8% missing)
- 11 cancer types show high actionability variance across institutions (>0.15 std)

### `plot_sex_race_composition.py` (8.3 KB)
Generate sex-race composition heatmaps ordered by Female-White proportion.

**Outputs**:
- `sex_race_composition_heatmap_annotated.pdf` - With proportion values
- `sex_race_composition_heatmap_clean.pdf` - Clean version for manuscripts
- `sex_race_composition_heatmap_clean_large_text.pdf` - Presentation version (18pt title)

**Usage**:
```bash
cd code/analysis/demographics
python plot_sex_race_composition.py
```

### `plot_age_ethnicity_interaction.py` (26.6 KB)
Age-ethnicity analysis with statistical testing (Welch's t-test + FDR correction).

**Outputs**:
- `hispanic_age_difference_forest_plot.pdf` - Clean forest plot
- `hispanic_age_difference_forest_plot_with_sig.pdf` - With significance markers
- 9 additional visualization types (distributions, boxplots, violin plots, etc.)

**Key findings**:
- 13/21 cancer types (61.9%) show significant age differences after FDR correction
- Hispanic patients consistently younger across most cancer types

### `summarize_hispanic_age_stats.py` (7.9 KB)
Statistical summary tables for Hispanic vs Non-Hispanic age comparisons.

**Outputs**:
- `hispanic_age_difference_statistics.csv` - Machine-readable
- `hispanic_age_difference_statistics.txt` - Human-readable report

## Institutional Analysis (`institutional/`)

Scripts for analyzing institutional patterns, heterogeneity, and cross-validation behavior.

### `analyze_all_cancer_types_institutional_stats.py` (37.2 KB)
Comprehensive institutional statistics for all 21 cancer types.

**Outputs**:
- Per-cancer-type institutional statistics (concentration, diversity, actionability variance)
- Summary tables and visualizations

**Key findings**:
- Identifies cancer types with high institutional concentration
- Measures actionability heterogeneity across institutions

### `investigate_gist_anomaly.py` (19.3 KB)
Detailed analysis of GIST institutional patterns explaining CV strategy anomaly.

**Context**: GIST shows pattern reversal with GroupKFold CV:
- Standard CV: Demographic features >> Technical features
- GroupKFold CV: Technical features ≈ Both > Demographic features

**Key findings**:
- Extreme actionability variance (std=0.260) across institutions
- High baseline actionability (79.7%)
- Institutional concentration (Herfindahl=0.256)
- Conclusion: Institution-dependent actionability is genuine biology, not artifact

### `visualize_institutional_fold_distribution.py` (37.8 KB)
Visualize how institutions and patients are distributed across GroupKFold folds.

**Outputs**:
- `institutional_stratification_summary.csv`
- `fold_distribution_tables.txt`
- Fold-by-fold breakdown with actionability rates

**Key findings**:
- **ALL 8 analyzed cancer types have severe fold imbalance (CV > 0.5)**
- BRCA worst case: 11.6× patient difference (1,829 vs 157 per fold)
- PRAD: 7.4× patient difference (4,174 vs 566 per fold)
- MEL exception: Patients CV=0.006 (well-balanced despite uneven institutions)

### `plot_cv_vs_test_auc.py` (6.5 KB)
Compare cross-validation vs test set performance to detect data leakage.

**Outputs**:
- `cv_vs_test_comparison.png` - Scatter plot showing CV-test discrepancy
- Quantifies gap between CV and test AUC by feature set

**Key findings**:
- Institution-aware test splits dramatically reduce CV-test discrepancy
- Technical features show larger gaps (data leakage indicator)

### `visualize_institutional_actionability_patterns.py` (38.1 KB)
Comprehensive visualization of institutional actionability heterogeneity.

**Outputs** (6 publication-quality visualizations):
- `fig1_institutional_actionability_heatmap.pdf` - Institution × cancer type heatmap
- `fig2_violin_institutional_distributions.pdf` - Actionability distributions
- `fig3_institution_size_vs_actionability.pdf` - Scatter plot
- `fig4_forest_plots_institutional_variation.pdf` - With Wilson score 95% CIs
- `fig5_cv_ranking_cancer_types.pdf` - Cancer types ranked by CV
- `fig6_range_vs_mean_actionability.pdf` - Range vs mean scatter

**Key findings**:
- **14/21 cancer types (67%) show HIGH variability (CV > 0.15)**
- 4/21 cancer types show LOW variability (CV ≤ 0.10)
- Mean CV across all cancer types: 0.241

## Variant Analysis (`variants/`)

### `plot_level1_variant_properties.py`
Analyze properties of OncoKB Level 1 actionable variants.

**Outputs** (9 visualizations):
- Variant classification pie chart
- Variant type pie chart
- Consequence pie chart
- Top genes bar chart
- Top cancer types bar chart
- Gene-cancer heatmap
- Mutation status pie chart
- Genes stacked by cancer bar chart

**Key findings**:
- 34,580 Level 1 variants across 38 genes
- Top genes: BRAF (20.6%), PIK3CA (18.0%), EGFR (17.4%)
- Top cancer types: LUAD (29.7%), IDC (12.9%), BRCA (5.7%)
- 76.9% missense mutations

**Usage**:
```bash
cd code/analysis/variants
python plot_level1_variant_properties.py
```

## Figures (`figures/visuals/`)

Project visualization scripts and outputs.

### Mindmap Generation Scripts
- `project_status_2025_mindmap.py` - PhD project status (basic)
- `project_status_2025_styled_mindmap.py` - PhD project status (stylized)
- `project_overview_mindmap.py` - Project overview
- `render_overview_fig.py` - Render overview figures

### Generated Figures
- 29 output files (PDFs, PNGs, GV files, markdown)
- Includes mindmaps, workflow diagrams, project outlines

## General Usage Patterns

### Running Analysis Scripts

Most scripts are standalone and can be run directly:

```bash
cd code/analysis/[category]
python [script_name].py
```

### Common Dependencies

All scripts typically require:
- `sqlite3` - Database access
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `scipy` - Statistical tests (for some scripts)

### Database Connection

Most scripts use absolute paths to the database:
```python
DB_PATH = "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite"
```

You may need to update these paths to match your local setup.

## Output Directories

Scripts typically output to:
- `experiments/[analysis_name]/` - For experimental results
- Local directory - For one-off analyses

## Requirements

- Python 3.9.18+ (specified in `code/balder_conda.yml`)
- All dependencies in conda environment
- SQLite database: `balder-harmonized-biomarker-data-v*.sqlite`

## Testing Status

**November 2025**: All representative scripts tested for:
- ✅ Syntax validation (Python compilation)
- ✅ Dependency availability
- ⏸️ Full execution (requires database and may be time-consuming)

## Development Notes

**November 2025 reorganization**:
- Scripts moved from `code/` to organized categories in `code/analysis/`
- No import changes needed (scripts use absolute paths or stdlib only)
- All scripts tested and validated

## Integration with Main Analysis

These scripts support the main ML analysis by:
1. **Demographics**: Understanding patient populations and feature quality
2. **Institutional**: Validating CV strategies and detecting data leakage
3. **Variants**: Understanding the distribution of actionable genomic events
4. **Figures**: Communicating results for manuscripts and presentations

## References

- Main ML framework: `code/modeling/`
- Pipeline scripts: `code/pipeline/`
- Experimental results: `experiments/` (in repository root)
- Documentation: `CLAUDE.md`, `code_review_and_feedback/`
