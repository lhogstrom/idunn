"""
Create presentation-ready tables from post-hoc statistical analysis results.

This module transforms raw analysis output into publication-ready tables with:
- Human-readable cancer type names
- Clear association labels
- Formatted statistics and confidence intervals
- Filtered to exclude "Other/Unknown" values
- Comprehensive legends explaining all columns

"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path
from datetime import datetime

from presentation_formatters import (
    format_scientific,
    format_ci,
    format_shap_importance,
    create_association_label,
    load_oncotree_mapping,
    filter_unknown_other,
    add_cancer_type_names,
    sort_by_significance,
    create_presentation_metadata
)


def create_sex_presentation_table(sex_df, oncotree_mapping):
    """
    Transform sex associations to presentation format.

    Parameters
    ----------
    sex_df : pd.DataFrame
        Raw sex associations dataframe
    oncotree_mapping : dict
        OncoTree code to name mapping

    Returns
    -------
    pd.DataFrame
        Presentation-ready dataframe
    """
    if len(sex_df) == 0:
        return pd.DataFrame()

    print("\nCreating SEX presentation table...")
    df = sex_df.copy()

    # Add cancer type names
    df = add_cancer_type_names(df, oncotree_mapping)

    # Filter Other/Unknown
    n_before = len(df)
    df = filter_unknown_other(df, 'feature')
    n_after = len(df)

    # Create association labels
    df['association'] = df.apply(
        lambda row: create_association_label(row['cancer_type'], row['feature'], 'sex'),
        axis=1
    )

    # Format SHAP importance
    df['shap_importance_fmt'] = df['shap_importance'].apply(format_shap_importance)

    # Format odds ratio with CI
    df['odds_ratio_ci'] = df.apply(
        lambda row: format_ci(row['odds_ratio'], row['or_ci_lower'], row['or_ci_upper'], precision=2),
        axis=1
    )

    # Format p_fdr
    df['p_fdr_fmt'] = df['p_fdr'].apply(lambda x: format_scientific(x, precision=2))

    # Select columns for presentation
    presentation_df = df[[
        'cancer_type',
        'association',
        'shap_importance_fmt',
        'n_total',
        'odds_ratio_ci',
        'p_fdr_fmt',
        'test_used'
    ]].copy()

    # Rename columns
    presentation_df.columns = [
        'cancer_type',
        'association',
        'shap_importance',
        'n',
        'odds_ratio_ci',
        'p_fdr',
        'test'
    ]

    # Sort by significance
    # Note: Need to sort by original p_fdr values, not formatted strings
    df['_p_fdr_for_sort'] = df['p_fdr']
    df['_or_for_sort'] = df['odds_ratio']
    presentation_df['_p_fdr_for_sort'] = df['_p_fdr_for_sort'].values
    presentation_df['_or_for_sort'] = df['_or_for_sort'].values

    presentation_df = sort_by_significance(
        presentation_df,
        p_column='_p_fdr_for_sort',
        effect_column='_or_for_sort',
        ascending_effect=False
    )

    # Drop sort columns
    presentation_df = presentation_df.drop(columns=['_p_fdr_for_sort', '_or_for_sort'])

    print(f"  Rows: {n_before} → {n_after} (removed {n_before - n_after} Other/Unknown)")
    print(f"  Significant (p_fdr < 0.05): {(df['p_fdr'] < 0.05).sum()}")

    return presentation_df


def create_race_presentation_table(race_df, oncotree_mapping):
    """
    Transform race/ethnicity associations to presentation format.

    Parameters
    ----------
    race_df : pd.DataFrame
        Raw race/ethnicity associations dataframe
    oncotree_mapping : dict
        OncoTree code to name mapping

    Returns
    -------
    pd.DataFrame
        Presentation-ready dataframe
    """
    if len(race_df) == 0:
        return pd.DataFrame()

    print("\nCreating RACE/ETHNICITY presentation table...")
    df = race_df.copy()

    # Add cancer type names
    df = add_cancer_type_names(df, oncotree_mapping)

    # Filter Other/Unknown
    n_before = len(df)
    df = filter_unknown_other(df, 'feature')
    n_after = len(df)

    # Create association labels
    df['association'] = df.apply(
        lambda row: create_association_label(row['cancer_type'], row['feature']),
        axis=1
    )

    # Format SHAP importance
    df['shap_importance_fmt'] = df['shap_importance'].apply(format_shap_importance)

    # Format Cramér's V
    df['cramers_v_fmt'] = df['cramers_v'].apply(lambda x: format_scientific(x, precision=3))

    # Format p_fdr
    df['p_fdr_fmt'] = df['p_fdr'].apply(lambda x: format_scientific(x, precision=2))

    # Select columns for presentation
    presentation_df = df[[
        'cancer_type',
        'association',
        'shap_importance_fmt',
        'n_total',
        'cramers_v_fmt',
        'p_fdr_fmt'
    ]].copy()

    # Rename columns
    presentation_df.columns = [
        'cancer_type',
        'association',
        'shap_importance',
        'n',
        'cramers_v',
        'p_fdr'
    ]

    # Sort by significance
    df['_p_fdr_for_sort'] = df['p_fdr']
    df['_v_for_sort'] = df['cramers_v']
    presentation_df['_p_fdr_for_sort'] = df['_p_fdr_for_sort'].values
    presentation_df['_v_for_sort'] = df['_v_for_sort'].values

    presentation_df = sort_by_significance(
        presentation_df,
        p_column='_p_fdr_for_sort',
        effect_column='_v_for_sort',
        ascending_effect=False
    )

    # Drop sort columns
    presentation_df = presentation_df.drop(columns=['_p_fdr_for_sort', '_v_for_sort'])

    print(f"  Rows: {n_before} → {n_after} (removed {n_before - n_after} Other/Unknown)")
    print(f"  Significant (p_fdr < 0.05): {(df['p_fdr'] < 0.05).sum()}")

    return presentation_df


def create_tech_presentation_table(tech_df, oncotree_mapping):
    """
    Transform technical feature associations to presentation format.

    Parameters
    ----------
    tech_df : pd.DataFrame
        Raw technical feature associations dataframe
    oncotree_mapping : dict
        OncoTree code to name mapping

    Returns
    -------
    pd.DataFrame
        Presentation-ready dataframe
    """
    # Technical features use same format as race/ethnicity (chi-square, Cramér's V)
    return create_race_presentation_table(tech_df, oncotree_mapping)


def generate_sex_legend(output_path, db_version="v20250522"):
    """Generate legend file for sex associations table"""

    legend_text = f"""# Sex Associations with Actionability - Table Legend

**Date Generated**: {datetime.now().strftime('%Y-%m-%d')}
**Database**: balder-harmonized-biomarker-data-{db_version}.sqlite
**Analysis Script**: analyze_shap_feature_associations.py

---

## Column Definitions

### cancer_type
Full OncoTree cancer type name (human-readable)
- Example: "Lung Adenocarcinoma" (not "LUAD")

### association
Human-readable description of the tested association
- Format: "Actionability vs. [sex] sex"
- Example: "Actionability vs. Female sex"
- Interpretation: Tests whether actionability rates differ between Female and Male patients

### shap_importance
Mean absolute SHAP value for this feature in the Random Forest model
- **What it measures**: Average magnitude of this feature's contribution to predictions (log-odds scale)
- **Range**: Typically 0.001 to 0.100 for top features
- **Interpretation**:
  - 0.017 = This feature shifts predictions by ~0.017 log-odds on average
  - Higher values = More important for model predictions
- **Context**: Top demographic features typically range from 0.01 to 0.05

### n
Total sample size for this cancer type (patients with non-missing sex data)
- Excludes patients with missing or "Other" sex values
- Minimum threshold: n ≥ 20 for analysis inclusion

### odds_ratio_ci
Odds ratio comparing Female vs. Male actionability rates, with 95% confidence interval
- **Format**: "OR [95% CI lower-upper]"
- **Interpretation**:
  - OR > 1.0: Females have higher odds of actionable variants than males
  - OR < 1.0: Males have higher odds of actionable variants than females
  - OR = 1.0: No difference between sexes
  - CI excludes 1.0: Statistically significant difference
- **Example**: "1.71 [1.62-1.81]"
  - Females have 1.71× higher odds than males
  - 95% confident true OR is between 1.62 and 1.81
  - Significant (CI doesn't include 1.0)

### p_fdr
False Discovery Rate (FDR) adjusted p-value
- **Method**: Benjamini-Hochberg correction applied across all sex associations tested
- **Threshold**: p_fdr < 0.05 considered statistically significant
- **Interpretation**: Expected proportion of false positives among all results called significant at this threshold
- **Format**: Scientific notation used for very small values (e.g., 6.3e-104)

### test
Statistical test used for this comparison
- **fisher_exact**: Used for 2×2 contingency tables with expected cell counts < 5 in any cell
- **chi_square**: Used for larger sample sizes (all expected counts ≥ 5)
- Both tests are valid; choice based on statistical assumptions

---

## Table Organization

### Ranking
Rows sorted by:
1. **Primary**: p_fdr (ascending) - Most significant associations first
2. **Secondary**: odds_ratio (descending) - Largest effect sizes first

### Filtering Applied
This table **EXCLUDES** associations involving:
- "Unknown" sex values
- "Other" sex values
- "Not reported" values

**Rationale**: These categories are ambiguous and difficult to interpret clinically.

**Note**: Complete unfiltered results available in `sex_associations.csv` (without PRESENTATION suffix)

---

## Statistical Methods

### Multiple Testing Correction
- **Method**: Benjamini-Hochberg False Discovery Rate (FDR) correction
- **Applied to**: All sex associations tested (across all cancer types)
- **Controls**: Expected proportion of false discoveries among significant results

### Significance Threshold
- **p_fdr < 0.05** (5% false discovery rate)
- Interpretation: Among all associations called significant, we expect ≤5% to be false positives

### Confidence Intervals
- **Level**: 95% confidence intervals for odds ratios
- **Method**: Woolf's method (log scale, then exponentiated)
- **Interpretation**: 95% probability that true OR falls within this range

### Sample Size Requirements
- **Minimum n**: 20 patients with non-missing sex data
- Analyses with n < 20 excluded (insufficient statistical power)

---

## Interpretation Example

**Sample Row**:
```
Lung Adenocarcinoma | Actionability vs. Female sex | 0.017 | 10,286 | 1.71 [1.62-1.81] | 6.3e-104 | chi_square
```

**Full Interpretation**:
- **Cancer type**: Lung Adenocarcinoma (LUAD)
- **Association tested**: Do Female and Male patients differ in actionability rates?
- **SHAP importance**: Sex has mean absolute SHAP value of 0.017 in the Random Forest model
  - This indicates a moderate contribution to predictions
  - Among demographic features, this is a relatively important predictor
- **Sample size**: 10,286 patients with known sex
- **Odds ratio**: 1.71 [1.62-1.81]
  - Females have 71% higher odds of having actionable variants than males
  - 95% confident the true odds ratio is between 1.62 and 1.81
  - Highly significant (CI doesn't include 1.0)
- **Statistical significance**: p_fdr = 6.3 × 10^-104
  - Extremely strong evidence of association
  - FDR-adjusted, so robust to multiple testing
- **Test used**: Chi-square test (large sample size allows this)

**Clinical interpretation**: In Lung Adenocarcinoma, female sex is associated with substantially higher likelihood of having clinically actionable genomic variants. This is consistent with known biology (e.g., higher EGFR mutation rates in female NSCLC patients).

---

## Related Files

- **Complete results**: `sex_associations.csv` (unfiltered, all columns)
- **Race/Ethnicity**: `race_ethnicity_associations_PRESENTATION.csv`
- **Technical features**: `technical_feature_associations_PRESENTATION.csv`
- **Summary report**: `shap_associations_report.txt`

---

**Questions or issues?** See analysis script: `code/analysis/posthoc/analyze_shap_feature_associations.py`
"""

    with open(output_path, 'w') as f:
        f.write(legend_text)

    print(f"  Legend saved: {output_path.name}")


def generate_race_legend(output_path, db_version="v20250522"):
    """Generate legend file for race/ethnicity associations table"""

    legend_text = f"""# Race/Ethnicity Associations with Actionability - Table Legend

**Date Generated**: {datetime.now().strftime('%Y-%m-%d')}
**Database**: balder-harmonized-biomarker-data-{db_version}.sqlite
**Analysis Script**: analyze_shap_feature_associations.py

---

## Column Definitions

### cancer_type
Full OncoTree cancer type name (human-readable)
- Example: "Lung Adenocarcinoma" (not "LUAD")

### association
Human-readable description of the tested association
- Format: "Actionability vs. [group] [ancestry/ethnicity]"
- Examples:
  - "Actionability vs. Asian ancestry"
  - "Actionability vs. Hispanic ethnicity"
- Interpretation: Tests whether actionability rates differ across racial/ethnic groups

### shap_importance
Mean absolute SHAP value for this feature in the Random Forest model
- **What it measures**: Average magnitude of this feature's contribution to predictions (log-odds scale)
- **Range**: Typically 0.001 to 0.100 for top features
- **Interpretation**:
  - 0.025 = This feature shifts predictions by ~0.025 log-odds on average
  - Higher values = More important for model predictions
- **Context**: Top demographic features typically range from 0.01 to 0.05

### n
Total sample size for this cancer type (patients with non-missing race/ethnicity data)
- Excludes patients with missing, "Unknown", or "Other" values
- Minimum threshold: n ≥ 20 for analysis inclusion

### cramers_v
Cramér's V effect size for chi-square test
- **What it measures**: Strength of association between race/ethnicity and actionability
- **Range**: 0 to 1
  - 0 = No association
  - 1 = Perfect association
- **Interpretation guidelines (for genomics research)**:
  - **0.05-0.10**: Small effect
  - **0.10-0.20**: Medium effect
  - **>0.20**: Large effect
- **Example**: V = 0.152 represents a medium-to-large effect
  - Race explains ~15% of variance in actionability (on phi-based scale)

### p_fdr
False Discovery Rate (FDR) adjusted p-value
- **Method**: Benjamini-Hochberg correction applied across all race/ethnicity associations tested
- **Threshold**: p_fdr < 0.05 considered statistically significant
- **Interpretation**: Expected proportion of false positives among all results called significant at this threshold
- **Format**: Scientific notation used for very small values (e.g., 2.1e-89)

---

## Table Organization

### Ranking
Rows sorted by:
1. **Primary**: p_fdr (ascending) - Most significant associations first
2. **Secondary**: cramers_v (descending) - Largest effect sizes first

### Filtering Applied
This table **EXCLUDES** associations involving:
- "Unknown" race/ethnicity values
- "Other" race/ethnicity values
- "Not reported" values
- "Not available" values

**Rationale**: These categories are heterogeneous and difficult to interpret clinically.

**Note**: Complete unfiltered results available in `race_ethnicity_associations.csv`

---

## Statistical Methods

### Chi-Square Test
- **Purpose**: Test for association between categorical variables (race/ethnicity × actionability)
- **Null hypothesis**: Race/ethnicity and actionability are independent
- **Assumptions**: Expected cell counts ≥ 5 (or Fisher's exact test used)

### Cramér's V Effect Size
- **Formula**: V = √(χ² / (n × min(r-1, c-1)))
  - Where r = number of racial/ethnic groups, c = 2 (actionable vs not)
- **Advantage**: Normalized effect size, comparable across studies
- **Interpretation**: Proportion of variance explained (on phi-based scale)

### Multiple Testing Correction
- **Method**: Benjamini-Hochberg False Discovery Rate (FDR) correction
- **Applied to**: All race/ethnicity associations tested (across all cancer types)
- **Controls**: Expected proportion of false discoveries among significant results

### Significance Threshold
- **p_fdr < 0.05** (5% false discovery rate)
- Among all associations called significant, we expect ≤5% to be false positives

### Sample Size Requirements
- **Minimum n**: 20 patients with non-missing race/ethnicity data
- **Minimum groups**: At least 2 racial/ethnic groups with data
- Analyses not meeting these criteria are excluded

---

## Interpretation Example

**Sample Row**:
```
Lung Adenocarcinoma | Actionability vs. Asian ancestry | 0.025 | 9,847 | 0.152 | 2.1e-89
```

**Full Interpretation**:
- **Cancer type**: Lung Adenocarcinoma (LUAD)
- **Association tested**: Do actionability rates differ across racial groups?
- **SHAP importance**: Race has mean absolute SHAP value of 0.025 in the Random Forest model
  - This indicates a moderate-to-strong contribution to predictions
  - Among demographic features, this is an important predictor
- **Sample size**: 9,847 patients with known race
- **Cramér's V**: 0.152
  - Medium-to-large effect size
  - Race explains ~15% of variance in actionability (substantial)
- **Statistical significance**: p_fdr = 2.1 × 10^-89
  - Extremely strong evidence of association
  - FDR-adjusted, so robust to multiple testing

**Clinical interpretation**: In Lung Adenocarcinoma, there is a strong association between race and actionability. Specifically, Asian ancestry is associated with higher actionability rates. This is consistent with known biology (e.g., higher EGFR mutation rates in Asian LUAD patients). The medium-to-large effect size (V=0.152) indicates this is a clinically meaningful association.

**Note**: This test shows an **overall association** across all racial groups. To understand which specific groups drive the association, examine the underlying contingency table or perform post-hoc pairwise comparisons.

---

## Understanding Cramér's V in Context

### Why different thresholds than Cohen's guidelines?
Traditional Cohen guidelines (0.1=small, 0.3=medium, 0.5=large) were developed for psychology research. In genomics:
- Genetic associations are often subtle but real
- Large sample sizes detect smaller but meaningful effects
- Effect sizes of 0.10-0.20 can have important clinical implications

### Genomics-specific interpretation:
- **V = 0.05-0.10**: Small but potentially meaningful (warrants investigation)
- **V = 0.10-0.20**: Moderate association (clinically relevant)
- **V > 0.20**: Strong association (major clinical factor)

For comparison:
- Our LUAD-Race finding (V=0.152) is similar in magnitude to known genetic associations
- This is larger than typical demographic effects but smaller than single-gene effects

---

## Related Files

- **Complete results**: `race_ethnicity_associations.csv` (unfiltered, all columns)
- **Sex associations**: `sex_associations_PRESENTATION.csv`
- **Technical features**: `technical_feature_associations_PRESENTATION.csv`
- **Cramér's V guide**: `cramers_v_interpretation_guide.md`
- **Summary report**: `shap_associations_report.txt`

---

**Questions or issues?** See analysis script: `code/analysis/posthoc/analyze_shap_feature_associations.py`
"""

    with open(output_path, 'w') as f:
        f.write(legend_text)

    print(f"  Legend saved: {output_path.name}")


def generate_tech_legend(output_path, db_version="v20250522"):
    """Generate legend file for technical feature associations table"""

    legend_text = f"""# Technical Feature Associations with Actionability - Table Legend

**Date Generated**: {datetime.now().strftime('%Y-%m-%d')}
**Database**: balder-harmonized-biomarker-data-{db_version}.sqlite
**Analysis Script**: analyze_shap_feature_associations.py

---

## Column Definitions

### cancer_type
Full OncoTree cancer type name (human-readable)

### association
Human-readable description of the tested association
- Format: "Actionability vs. [value] technical factor"
- Examples:
  - "Actionability vs. WXS library_selection"
  - "Actionability vs. MSK-IMPACT calling_strategy"

### shap_importance
Mean absolute SHAP value for this feature in the Random Forest model
- Higher values indicate greater importance in predictions

### n
Total sample size for this cancer type

### cramers_v
Effect size measuring strength of association (0 to 1)
- 0.05-0.10: Small effect
- 0.10-0.20: Medium effect
- >0.20: Large effect

### p_fdr
FDR-adjusted p-value (significance threshold: p_fdr < 0.05)

---

## Important Note on Technical Features

**Potential Data Leakage**: Technical features (sequencing platform, calling strategy, library selection, sequencing center) may introduce data leakage in predictive models. These features can predict actionability because:

1. **Detection capability**: Different sequencing platforms/panels have different detection capabilities
2. **Institutional variation**: Different centers may have different patient populations or testing practices
3. **Temporal confounding**: Sequencing technologies changed over time, coinciding with changes in actionable variant prevalence

**Interpretation**: Associations with technical features should be interpreted cautiously:
- Strong associations may indicate technical artifacts rather than biological signals
- These features may predict **which variants can be detected** rather than **which variants exist**
- Use for quality control and study design, not clinical prediction

**See also**: Our data leakage validation analysis showing technical features predict detection, not biology.

---

## Statistical Methods

Same as Race/Ethnicity table:
- Chi-square test for association
- Cramér's V effect size
- FDR correction for multiple testing
- Significance threshold: p_fdr < 0.05

---

## Filtering Applied

This table **EXCLUDES** associations involving:
- "Unknown" values
- "Other" values
- "Not reported/available" values

Complete results: `technical_feature_associations.csv`

---

**Questions or issues?** See analysis script: `code/analysis/posthoc/analyze_shap_feature_associations.py`
"""

    with open(output_path, 'w') as f:
        f.write(legend_text)

    print(f"  Legend saved: {output_path.name}")
