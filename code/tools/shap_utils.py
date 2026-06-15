"""
Shared utilities for SHAP-based feature analysis.

This module provides common functions for loading and processing SHAP feature
rankings used by post-hoc analysis scripts.

Functions:
    - get_default_shap_file: Find SHAP rankings file in standard locations
    - load_shap_rankings_adaptive: Load SHAP rankings with auto-format detection
    - identify_top_features: Find cancer types with specific features in top N
    - feature_in_top_n_for_cancer: Test whether a feature category is in top N
      for a given cancer type

Used by:
    - analyze_shap_feature_associations.py (cancer-level post-hoc)
    - analyze_gene_stratified_associations.py (gene-stratified post-hoc)
"""

import pandas as pd
from pathlib import Path

# Get repository root (relative to this file in code/tools/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Feature patterns for pattern matching in both old and new formats
# Old format uses sklearn-style names: "cat__SEX_Male"
# New format uses readable names: "Sex: Male"
FEATURE_PATTERNS = {
    'SEX': ['SEX_', 'Sex:', 'SEX'],
    # ANCESTRY key (was 'RACE'; renamed 2026-05-21 Feature Review Fix b). The
    # underlying feature is now named ANCESTRY in code and YAML; the DB column
    # stays RACE and data_loader.py renames it on load. Patterns keep the
    # legacy 'RACE_' / 'Race:' / 'RACE' forms so historical SHAP rankings
    # (e.g. runs/run_20260505_164207/) still match — see Bug C history in
    # CLAUDE.md and the 2026-05-07 audit.
    'ANCESTRY': [
        'ANCESTRY_', 'Ancestry:', 'Ancestry', 'ANCESTRY',
        'RACE_', 'Race:', 'RACE',
    ],
    'ETHNICITY': ['ETHNICITY_', 'Ethnicity:', 'ETHNICITY'],
    'ANCESTRY_ETHNICITY': [
        'ANCESTRY_', 'ETHNICITY_', 'Ancestry:', 'Ethnicity:',
        'ANCESTRY', 'ETHNICITY', 'Ancestry',
        # Legacy patterns kept for back-compat with pre-rename SHAP rankings:
        'RACE_', 'Race:', 'RACE',
    ],
    # Indicator columns (ANCESTRY_missing, ETHNICITY_missing) are not given
    # dedicated pattern keys. identify_top_features auto-excludes _missing-
    # suffixed features from the ANCESTRY/ETHNICITY/ANCESTRY_ETHNICITY
    # categories (2026-05-22 pre-lock review #4 + #9), so the institutional-
    # missingness signal does not double-count into the raw-ancestry gate.
    # If a future analysis needs to test missingness indicators specifically,
    # query shap_df with `feature.str.contains('_missing')` directly.
    # SAMPLE_TYPE removed 2026-05-21 (Feature Review Fix a; see features.yaml).
    # SAMPLE_TYPE_DETAILED is the surviving sample-type feature in the model.
    'SAMPLE_TYPE_DETAILED': [
        'SAMPLE_TYPE_DETAILED_', 'Sample Type (Detailed)', 'Sample Type Detailed',
        'SAMPLE_TYPE_DETAILED',
    ],
    'AGE': ['AGE_AT_SEQ_REPORT', 'Age at Sequencing', 'Age'],
    'TECHNICAL': [
        # Subject Center (categorical)
        'SUBJECT_CENTER', 'Subject Center',
        # Number of genes (numeric)
        'number_of_genes', 'Number of Genes',
        # Preservation technique (categorical)
        'preservation_technique', 'Preservation Technique', 'Preservation',
        # Calling strategy (categorical)
        'calling_strategy', 'Calling Strategy', 'Variant Calling Strategy',
    ]
}

# Default number of top features to consider
DEFAULT_TOP_N_FEATURES = 5


def get_default_shap_file(repo_root=None):
    """
    Get default SHAP file from current TIER 6a output paths.

    Searches for SHAP primary values output from compute_shap_primary_values.py
    in the standard output locations. Raises FileNotFoundError if no v2.0
    output is found — callers must pass --shap-file explicitly or run TIER 6a.

    Args:
        repo_root: Repository root path (optional, uses default if not provided)

    Returns:
        str: Path to SHAP rankings file

    Raises:
        FileNotFoundError: If no v2.0 SHAP file is found in the standard
            locations. The pre-2026-05 silent fallback to a legacy SHAP file
            with the obsolete v1 feature space has been removed; see
            ROADMAP.md "ACTIVE DATA-QUALITY ISSUE: Post-hoc SHAP-gate
            provenance audit" for context.
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
    """Get default feature rankings file based on algorithm.

    Args:
        algorithm: 'random_forest' (uses SHAP) or 'logistic' (uses LR coefficients)
        repo_root: Repository root path (optional, uses default if not provided)

    Returns:
        str: Path to rankings file, or None if not found
    """
    if algorithm == 'logistic':
        return get_default_lr_rankings_file(repo_root)
    else:
        return get_default_shap_file(repo_root)


def get_default_lr_rankings_file(repo_root=None):
    """Find LR grouped importance file in standard locations.

    Args:
        repo_root: Repository root path (optional, uses default if not provided)

    Returns:
        str: Path to LR rankings file, or None if not found
    """
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


def load_shap_rankings_adaptive(shap_file, lr_significant_only=False):
    """
    Load SHAP rankings, auto-detecting format (new vs legacy vs LR).

    New format (from shap_primary_analyzer.py):
        - cancer_type, feature_name, feature_name_raw, mean_abs_shap, rank

    LR grouped importance format (from extract_lr_coefficients.py):
        - feature_group, cancer_type, max_abs_std_log_odds, rank, any_significant_fdr

    Legacy format:
        - ONCOTREE_CODE, feature, mean_abs_shap, feature_rank

    Args:
        shap_file: Path to SHAP rankings CSV file
        lr_significant_only: If True and LR format detected, filter to only
            features with any_significant_fdr=True (Wald test, FDR-corrected),
            then re-rank within each cancer type.

    Returns:
        pd.DataFrame with standardized columns:
            - ONCOTREE_CODE: cancer type
            - feature: feature name (human-readable preferred)
            - mean_abs_shap: mean absolute SHAP value
            - feature_rank: 1-based rank

    Raises:
        ValueError: If file format is not recognized
    """
    print(f"Loading SHAP rankings from {shap_file}...")

    df = pd.read_csv(shap_file)

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

        # Use max_abs_std_log_odds as the importance metric (analogous to mean_abs_shap)
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

    # Ensure feature_rank exists
    if 'feature_rank' not in df.columns and 'rank' in df.columns:
        df['feature_rank'] = df['rank']

    print(f"Loaded {len(df):,} feature rankings")
    print(f"Cancer types: {df['ONCOTREE_CODE'].nunique()}")
    print(f"Unique features: {df['feature'].nunique()}")

    return df


def identify_top_features(shap_df, feature_patterns, top_n=DEFAULT_TOP_N_FEATURES, verbose=True):
    """
    Identify cancer types where specific features appear in top N SHAP rankings.

    Args:
        shap_df: DataFrame with SHAP rankings (from load_shap_rankings_adaptive)
        feature_patterns: List of patterns to match in feature names
        top_n: Number of top features to consider (default: 5). Pass None to
            disable the rank cap entirely (every pattern-matched feature is
            returned) — used by the LR --lr-significant-only gate.
        verbose: Whether to print progress (default: True)

    Returns:
        pd.DataFrame with columns:
            - pattern: matched pattern
            - ONCOTREE_CODE: cancer type
            - feature: feature name
            - feature_rank: rank (1-based)
            - mean_abs_shap: SHAP importance value

    Notes:
        Substring matching means a bare 'ANCESTRY' or 'ETHNICITY' pattern would
        otherwise also match the new ANCESTRY_missing / ETHNICITY_missing
        indicator features (added 2026-05-21 by Feature Review Fix b). To keep
        the missingness indicator from being double-counted in the raw-ancestry
        gate, _missing-suffixed features are excluded from category patterns
        that don't themselves reference 'missing'. This auto-disambiguation
        replaces the originally-defined-but-unused ANCESTRY_MISSING /
        ETHNICITY_MISSING pattern keys (removed 2026-05-22).
    """
    results = []

    # Auto-detect: if none of this category's patterns reference 'missing',
    # exclude `_missing`-suffixed indicator features from the matches. The
    # missingness indicator is a distinct semantic category — including it
    # under ANCESTRY/ETHNICITY would confound the raw-ancestry chi-square gate.
    exclude_missingness_indicators = not any(
        'missing' in str(p).lower() for p in feature_patterns
    )

    for pattern in feature_patterns:
        # regex=False — patterns like 'Sample Type (Detailed)' contain unescaped
        # regex metachars (parens are group syntax); without this they would
        # match 'Sample Type Detailed' (no parens) instead of the literal string.
        mask = shap_df['feature'].str.contains(pattern, case=False, na=False, regex=False)
        if exclude_missingness_indicators:
            # Catch both the raw form ('ANCESTRY_missing') and the display form
            # ('Ancestry Missing (Indicator)' from features.yaml display_names)
            # of the missingness indicator features. The display form is what
            # production SHAP CSVs carry today, so the underscore-only check
            # would silently fail to exclude — letting ANCESTRY_missing leak
            # into the bare ANCESTRY substring-match. regex=True for the OR.
            missingness_re = r'_missing|Missing \(Indicator\)'
            mask = mask & ~shap_df['feature'].str.contains(
                missingness_re, case=False, na=False, regex=True
            )
        # top_n=None disables the rank cap entirely (every matched feature is
        # eligible). Used by the LR `--lr-significant-only` path, where Wald-FDR
        # significance is the gate and the top-N cap would redundantly drop
        # already-significant features (see the 2026-06-15 gate-sensitivity note).
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
        # A single feature can match multiple patterns within the same category
        # (e.g., 'SAMPLE_TYPE_DETAILED_' and 'SAMPLE_TYPE_DETAILED' both match
        # 'SAMPLE_TYPE_DETAILED_Primary tumor'). Without dedup, downstream
        # chi-square consumers see the same (cancer, feature) pair multiple
        # times and inflate the FDR denominator at multipletests.
        results_df = results_df.drop_duplicates(
            subset=['ONCOTREE_CODE', 'feature'], keep='first'
        ).reset_index(drop=True)
    return results_df


def get_cancer_types_with_top_feature(shap_df, feature_category, top_n=DEFAULT_TOP_N_FEATURES):
    """
    Get list of cancer types where a feature category appears in top N.

    Args:
        shap_df: DataFrame with SHAP rankings
        feature_category: Key from FEATURE_PATTERNS (e.g., 'SEX', 'AGE', 'TECHNICAL')
        top_n: Number of top features to consider

    Returns:
        List of ONCOTREE_CODE values
    """
    if feature_category not in FEATURE_PATTERNS:
        raise ValueError(f"Unknown feature category: {feature_category}. "
                        f"Valid categories: {list(FEATURE_PATTERNS.keys())}")

    patterns = FEATURE_PATTERNS[feature_category]
    top_features = identify_top_features(shap_df, patterns, top_n=top_n, verbose=False)

    if len(top_features) == 0:
        return []

    return top_features['ONCOTREE_CODE'].unique().tolist()


def feature_in_top_n_for_cancer(shap_df, cancer_type, feature_category, top_n=DEFAULT_TOP_N_FEATURES):
    """
    Check whether a feature category appears in top-N for the given cancer type.

    Both `cancer_type` and the SHAP file's ONCOTREE_CODE column use plain
    OncoTree codes after the publication-branch OT-rollup strip.

    Args:
        shap_df: DataFrame with SHAP rankings
        cancer_type: Plain ONCOTREE_CODE (e.g., "LUAD")
        feature_category: Key from FEATURE_PATTERNS
        top_n: Number of top features to consider

    Returns:
        bool
    """
    cancer_types_with_feature = get_cancer_types_with_top_feature(
        shap_df, feature_category, top_n
    )
    return cancer_type in cancer_types_with_feature
