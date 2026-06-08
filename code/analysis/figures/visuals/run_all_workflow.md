# ML Analysis Pipeline (run_all_analyses.py)

## TIER 1: Independent Analyses
### Demographics analysis
### Institutional statistics
### Feature engineering
### Cancer type summary
### Variant analysis

## TIER 2: Data Setup
### 2a: Multi-tier groupings
OncoTree hierarchy
### 2b: Train/Test split
80/20 stratified by cancer type

## TIER 3: Hyperparameter Optimization
### Training data only (80%)
### RandomizedSearchCV
### Result: Random Forest selected

## TIER 4: Feature Set Comparison
### Training data only (80%)
### Compare demographic vs technical vs both

## TIER 5: Final Model Training
### Train on training data (80%)
### Evaluate on test data (20%)
First and only use of test set
### Save models (.joblib)

## TIER 6: SHAP Analysis
### 6a: Primary feature values
Mean |SHAP| rankings
Beeswarm plots
### 6b: Feature interactions
Pairwise interactions

## TIER 7: Post-Hoc Statistics
### Age by actionability
Mann-Whitney U, FDR correction
### SHAP feature associations
Fisher's exact, chi-square
