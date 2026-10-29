"""Shared utilities for loading and filtering SHAP / LR feature rankings.

Used by the TIER 7 post-hoc association scripts to select cancer types in which a
feature category ranks in the top N. Main functions: get_default_shap_file,
load_shap_rankings_adaptive, identify_top_features, feature_in_top_n_for_cancer.
"""

import pandas as pd
from pathlib import Path

# repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Patterns matched against feature names in sklearn-style ("cat__SEX_Male") and readable ("Sex: Male") formats
FEATURE_PATTERNS = {
    'SEX': ['SEX_', 'Sex:', 'SEX'],
    # Formerly RACE; legacy 'RACE' patterns are kept so older rankings still match.
    'ANCESTRY': [
        'ANCESTRY_', 'Ancestry:', 'Ancestry', 'ANCESTRY',
        'RACE_', 'Race:', 'RACE',
    ],
    'ETHNICITY': ['ETHNICITY_', 'Ethnicity:', 'ETHNICITY'],
    'ANCESTRY_ETHNICITY': [
        'ANCESTRY_', 'ETHNICITY_', 'Ancestry:', 'Ethnicity:',
        'ANCESTRY', 'ETHNICITY', 'Ancestry',
        # legacy RACE patterns, for pre-rename rankings
        'RACE_', 'Race:', 'RACE',
    ],
    # Missingness indicators (ANCESTRY_missing, ETHNICITY_missing) have no pattern keys;
    # identify_top_features excludes them from the ancestry/ethnicity categories.
    'SAMPLE_TYPE_DETAILED': [
        'SAMPLE_TYPE_DETAILED_', 'Sample Type (Detailed)', 'Sample Type Detailed',
        'SAMPLE_TYPE_DETAILED',
    ],
    'AGE': ['AGE_AT_SEQ_REPORT', 'Age at Sequencing', 'Age'],
    'TECHNICAL': [
        # Subject Center
        'SUBJECT_CENTER', 'Subject Center',
        # Number of genes (numeric)
        'number_of_genes', 'Number of Genes',
        # Preservation technique (categorical)
        'preservation_technique', 'Preservation Technique', 'Preservation',
        # Calling strategy (categorical)
        'calling_strategy', 'Calling Strategy', 'Variant Calling Strategy',
    ]
}

# RF one-hot level grouping. LR importance is already one row per feature, while RF
# importance is one row per one-hot dummy, which dilutes multi-level categoricals in a
# rank cut. Grouping dummies by parent feature makes the RF gate rank the same unit.
# Group names are the raw feature names so FEATURE_PATTERNS matching still works.
# Ordered: first match wins, and `_missing` indicators must precede their parent.
RF_LEVEL_GROUPS = [
    ('ANCESTRY_missing',       ['ANCESTRY_missing', 'Ancestry Missing (Indicator)']),
    ('ETHNICITY_missing',      ['ETHNICITY_missing', 'Ethnicity Missing (Indicator)']),
    ('AGE_AT_SEQ_REPORT',      ['AGE_AT_SEQ_REPORT', 'Age at Sequencing']),
    ('SEX',                    ['SEX_', 'Sex:', 'SEX']),
    ('SAMPLE_TYPE_DETAILED',   ['SAMPLE_TYPE_DETAILED', 'Sample Type (Detailed)',
                                'Sample Type Detailed']),
    ('ANCESTRY',               ['ANCESTRY', 'Ancestry', 'RACE', 'Race']),
    ('ETHNICITY',              ['ETHNICITY', 'Ethnicity']),
    ('SUBJECT_CENTER',         ['SUBJECT_CENTER', 'Subject Center']),
    ('number_of_genes',        ['number_of_genes', 'Number of Genes']),
    ('preservation_technique', ['preservation_technique', 'Preservation Technique',
                                'Preservation']),
    ('calling_strategy',       ['calling_strategy', 'Calling Strategy',
                                'Variant Calling Strategy']),
]

# Default number of top features
DEFAULT_TOP_N_FEATURES = 5


def _rf_level_group_for(feature_name):
    """Map a raw RF feature/dummy name to its parent feature group, or None."""
    name = str(feature_name)
    lowered = name.lower()
    for group, patterns in RF_LEVEL_GROUPS:
        for pattern in patterns:
            if str(pattern).lower() in lowered:
                return group
    return None


# where the missingness indicators go under missingness='fold'
RF_MISSINGNESS_PARENT = {
    'ANCESTRY_missing': 'ANCESTRY',
    'ETHNICITY_missing': 'ETHNICITY',
}


def group_rf_shap_levels(df, agg='sum', missingness='keep', verbose=True):
    """Collapse one-hot dummy rows of an RF SHAP ranking to one row per feature.

    agg: 'sum' (additive decomposition; an upper bound on grouped importance) or 'max'.
    missingness: 'keep' (own groups), 'drop', or 'fold' (into the parent ANCESTRY/ETHNICITY
    group, as in the grouped importance files written by the SHAP step).
    Returns one row per (ONCOTREE_CODE, feature) with feature_rank recomputed.
    """
    if missingness not in ('keep', 'drop', 'fold'):
        raise ValueError(
            f"missingness must be 'keep', 'drop' or 'fold', got {missingness!r}")

    work = df.copy()
    work['_group'] = work['feature'].map(_rf_level_group_for)

    unmatched = work['_group'].isna()
    if unmatched.any():
        # keep unmatched rows as their own group so no feature silently drops out of the gate
        names = sorted(work.loc[unmatched, 'feature'].unique())
        print(f"  WARNING: {unmatched.sum()} rows matched no RF_LEVEL_GROUPS "
              f"entry; each kept as its own group: {names[:10]}"
              + (" ..." if len(names) > 10 else ""))
        work.loc[unmatched, '_group'] = work.loc[unmatched, 'feature']

    if missingness == 'drop':
        before = len(work)
        work = work[~work['_group'].isin(RF_MISSINGNESS_PARENT)]
        if verbose:
            print(f"  Dropped {before - len(work)} missingness-indicator rows "
                  f"(missingness='drop')")
    elif missingness == 'fold':
        n_folded = int(work['_group'].isin(RF_MISSINGNESS_PARENT).sum())
        work['_group'] = work['_group'].map(
            lambda g: RF_MISSINGNESS_PARENT.get(g, g))
        if verbose:
            print(f"  Folded {n_folded} missingness-indicator rows into their "
                  f"parent feature (missingness='fold')")

    if agg not in ('sum', 'max'):
        raise ValueError(f"agg must be 'sum' or 'max', got {agg!r}")

    grouped = (
        work.groupby(['ONCOTREE_CODE', '_group'], as_index=False)['mean_abs_shap']
        .agg(agg)
        .rename(columns={'_group': 'feature'})
    )
    grouped['feature_rank'] = (
        grouped.groupby('ONCOTREE_CODE')['mean_abs_shap']
        .rank(ascending=False, method='min')
        .astype(int)
    )
    grouped = grouped.sort_values(
        ['ONCOTREE_CODE', 'feature_rank']).reset_index(drop=True)

    if verbose:
        print(f"  Grouped RF one-hot levels: {len(df):,} dummy rows -> "
              f"{len(grouped):,} feature rows "
              f"({grouped['feature'].nunique()} distinct features, agg={agg!r})")

    return grouped


def get_default_shap_file(repo_root=None):
    """Find the default SHAP file from TIER 6a output paths.

    Raises FileNotFoundError if none is found; pass --shap-file explicitly in that case.
    """
    if repo_root is None:
        repo_root = REPO_ROOT

    repo_root = Path(repo_root)
    searched = []

    # Check for new format output in analysis_output
    new_output = repo_root / "analysis_output" / "shap_primary_values"
    searched.append(str(new_output / "run_*/combined_feature_importance.csv"))
    if new_output.exists():
        # Find latest run
        runs = sorted(new_output.glob("run_*"), reverse=True)
        for run_dir in runs:
            combined_file = run_dir / "combined_feature_importance.csv"
            if combined_file.exists():
                print(f"Found SHAP file (new format): {combined_file}")
                return str(combined_file)

    # Check for new format in runs directory (unified output structure)
    runs_dir = repo_root / "runs"
    searched.append(str(runs_dir / "run_*/tier6_shap_analysis/primary_values/combined_feature_importance.csv"))
    if runs_dir.exists():
        # Find latest run with SHAP output
        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        for run_dir in runs:
            shap_dir = run_dir / "tier6_shap_analysis" / "primary_values"
            combined_file = shap_dir / "combined_feature_importance.csv"
            if combined_file.exists():
                print(f"Found SHAP file (new format, unified runs): {combined_file}")
                return str(combined_file)

    raise FileNotFoundError(
        "No v2.0 SHAP file found. Searched:\n  - "
        + "\n  - ".join(searched)
        + "\nFix: pass --shap-file <path-to-combined_feature_importance.csv> "
        "explicitly, or re-run TIER 6a (compute_shap_primary_values.py) "
        "for the current run."
    )


def get_default_feature_rankings_file(algorithm='random_forest', repo_root=None):
    """Return the default rankings file for an algorithm (random_forest or logistic), or None."""
    if algorithm == 'logistic':
        return get_default_lr_rankings_file(repo_root)
    else:
        return get_default_shap_file(repo_root)


def get_default_lr_rankings_file(repo_root=None):
    """Find the LR grouped importance file in the standard locations, or None."""
    if repo_root is None:
        repo_root = REPO_ROOT
    repo_root = Path(repo_root)

    # Check runs directory for LR analysis output
    runs_dir = repo_root / "runs"
    if runs_dir.exists():
        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        for run_dir in runs:
            # Check tier6 LR analysis subdirectory
            lr_file = run_dir / "tier6_shap_analysis" / "lr_analysis" / "combined_lr_grouped_importance.csv"
            if lr_file.exists():
                print(f"Found LR rankings file: {lr_file}")
                return str(lr_file)

    return None


def load_shap_rankings_adaptive(shap_file, lr_significant_only=False,
                                group_rf_levels=False, rf_group_agg='sum',
                                rf_group_missingness='keep'):
    """Load a ranking file, auto-detecting RF SHAP, LR grouped-importance or legacy format.

    lr_significant_only keeps only Wald-FDR-significant LR features and re-ranks.
    Returns columns ONCOTREE_CODE, feature, mean_abs_shap, feature_rank.
    """
    print(f"Loading SHAP rankings from {shap_file}...")

    df = pd.read_csv(shap_file)

    # set by the LR branch; group_rf_levels is a no-op on LR rankings
    is_lr_format = False

    # Detect format by checking column names
    if 'cancer_type' in df.columns and 'feature_name' in df.columns:
        # New format from shap_primary_analyzer.py
        print("  Detected NEW format (TIER 6a output)")

        # Rename columns to legacy format for compatibility
        df = df.rename(columns={
            'cancer_type': 'ONCOTREE_CODE',
            'feature_name': 'feature',  # Use readable name for pattern matching
            'rank': 'feature_rank'
        })

        # Keep feature_name_raw as additional column for reference
        if 'feature_name_raw' in df.columns:
            df['feature_raw'] = df['feature_name_raw']

    elif 'feature_group' in df.columns and 'cancer_type' in df.columns:
        # LR grouped importance format (from extract_lr_coefficients.py)
        print("  Detected LR GROUPED IMPORTANCE format")
        is_lr_format = True

        # Filter to Wald-significant features if requested
        if lr_significant_only and 'any_significant_fdr' in df.columns:
            before_count = len(df)
            df = df[df['any_significant_fdr'] == True].copy()
            after_count = len(df)
            print(f"  Filtered to Wald-significant only: {after_count}/{before_count} feature-cancer pairs retained")

            if len(df) == 0:
                print("  WARNING: No Wald-significant features found in any cancer type")
            else:
                # Re-rank within each cancer type after filtering
                df['rank'] = df.groupby('cancer_type')['max_abs_std_log_odds' if 'max_abs_std_log_odds' in df.columns else 'sum_abs_std_log_odds'].rank(ascending=False, method='min').astype(int)

        # max_abs_std_log_odds is the LR analogue of mean_abs_shap
        importance_col = 'max_abs_std_log_odds' if 'max_abs_std_log_odds' in df.columns else 'sum_abs_std_log_odds'

        df = df.rename(columns={
            'cancer_type': 'ONCOTREE_CODE',
            'feature_group': 'feature',
            importance_col: 'mean_abs_shap',
            'rank': 'feature_rank'
        })

    elif 'ONCOTREE_CODE' in df.columns and 'feature' in df.columns:
        # Legacy format
        print("  Detected LEGACY format")

    else:
        raise ValueError(
            f"Unrecognized SHAP file format. "
            f"Expected columns 'cancer_type'/'feature_name' (new), "
            f"'feature_group'/'cancer_type' (LR), or 'ONCOTREE_CODE'/'feature' (legacy). "
            f"Found: {df.columns.tolist()}"
        )

    # ensure feature_rank exists
    if 'feature_rank' not in df.columns and 'rank' in df.columns:
        df['feature_rank'] = df['rank']

    # optionally collapse RF one-hot dummies to one row per feature (LR is already grouped)
    if group_rf_levels:
        if is_lr_format:
            print("  NOTE: group_rf_levels=True ignored — LR rankings are "
                  "already grouped by feature upstream (feature_group).")
        else:
            df = group_rf_shap_levels(
                df,
                agg=rf_group_agg,
                missingness=rf_group_missingness,
            )

    print(f"Loaded {len(df):,} feature rankings")
    print(f"Cancer types: {df['ONCOTREE_CODE'].nunique()}")
    print(f"Unique features: {df['feature'].nunique()}")

    return df


def identify_top_features(shap_df, feature_patterns, top_n=DEFAULT_TOP_N_FEATURES, verbose=True):
    """Find cancer types where features matching `feature_patterns` rank in the top N.

    top_n=None disables the rank cap. Returns columns pattern, ONCOTREE_CODE, feature,
    feature_rank, mean_abs_shap. Missingness indicators are excluded unless a pattern names them.
    """
    results = []

    # unless the patterns mention "missing", exclude missingness indicators so they do not leak into ANCESTRY/ETHNICITY
    exclude_missingness_indicators = not any(
        'missing' in str(p).lower() for p in feature_patterns
    )

    for pattern in feature_patterns:
        # regex=False: parentheses in names like 'Sample Type (Detailed)' must match literally
        mask = shap_df['feature'].str.contains(pattern, case=False, na=False, regex=False)
        if exclude_missingness_indicators:
            # exclude both the raw ('ANCESTRY_missing') and display ('Ancestry Missing (Indicator)') forms
            missingness_re = r'_missing|Missing \(Indicator\)'
            mask = mask & ~shap_df['feature'].str.contains(
                missingness_re, case=False, na=False, regex=True
            )
        # top_n=None disables the rank cap (used when Wald-FDR significance is the gate)
        if top_n is None:
            rank_ok = pd.Series(True, index=shap_df.index)
        else:
            rank_ok = shap_df['feature_rank'] <= top_n
        matching = shap_df[mask & rank_ok].copy()

        if len(matching) > 0:
            if verbose:
                print(f"\nFeature pattern '{pattern}': {len(matching)} top-{top_n} occurrences")
            for _, row in matching.iterrows():
                if verbose:
                    print(f"  {row['ONCOTREE_CODE']}: {row['feature']} (rank {row['feature_rank']})")
                results.append({
                    'pattern': pattern,
                    'ONCOTREE_CODE': row['ONCOTREE_CODE'],
                    'feature': row['feature'],
                    'feature_rank': row['feature_rank'],
                    'mean_abs_shap': row['mean_abs_shap']
                })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # one feature can match several patterns; dedup so the FDR denominator is not inflated
        results_df = results_df.drop_duplicates(
            subset=['ONCOTREE_CODE', 'feature'], keep='first'
        ).reset_index(drop=True)
    return results_df


def get_cancer_types_with_top_feature(shap_df, feature_category, top_n=DEFAULT_TOP_N_FEATURES):
    """Return the cancer types where a feature category (a FEATURE_PATTERNS key) is in the top N."""
    if feature_category not in FEATURE_PATTERNS:
        raise ValueError(f"Unknown feature category: {feature_category}. "
                        f"Valid categories: {list(FEATURE_PATTERNS.keys())}")

    patterns = FEATURE_PATTERNS[feature_category]
    top_features = identify_top_features(shap_df, patterns, top_n=top_n, verbose=False)

    if len(top_features) == 0:
        return []

    return top_features['ONCOTREE_CODE'].unique().tolist()


def feature_in_top_n_for_cancer(shap_df, cancer_type, feature_category, top_n=DEFAULT_TOP_N_FEATURES):
    """Return True if a feature category is in the top-N for a cancer type (plain OncoTree code)."""
    cancer_types_with_feature = get_cancer_types_with_top_feature(
        shap_df, feature_category, top_n
    )
    return cancer_type in cancer_types_with_feature
