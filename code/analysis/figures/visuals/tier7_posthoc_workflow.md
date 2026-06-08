# Post-Hoc Statistical Analysis

## Input

- SHAP Feature Rankings

## Cancer-Level Associations

- Age x Actionability: Mann-Whitney U | Cohen's d
- Feature x Actionability: Sex, Race, Sample Type, Technical -- Chi-square | Fisher's exact
- FDR Correction: per feature category
- ~77 tests | 54 significant (70%)

## Gene-Stratified Associations

- Phase 1: Identify top 2 actionable genes per cancer type (33 types)
- Phase 2: Gene-Stratified Tests -- Age, Sex, Race, Ethnicity, Sample Type, Technical -- 48 gene-cancer combos (n >= 50)
- FDR Correction: per feature category
- 193 tests | 134 significant (69%)

## Connections

- top-5 filter (TIER 6a to Feature x Actionability)
- top-5 filter (TIER 6a to Phase 2: Gene-Stratified Tests)
