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

# ---------------------------------------------------------------------------
# RF one-hot level grouping — ADOPTED 2026-08-31 (E2o, "Arm C")
# ---------------------------------------------------------------------------
# STATUS: this is IN PRODUCTION. Both TIER 7 scripts
# (analyze_gene_stratified_associations.py, analyze_shap_feature_associations.py)
# default to group_rf_levels=True with agg='sum', missingness='fold'; their
# --no-group-rf-levels flag restores the pre-2026-08-31 column-level gate for
# reproducing historical runs. The LIBRARY default here stays False so that any
# other caller keeps its old behaviour unless it opts in -- the decision is
# expressed at the analysis layer, not here.
#
# BEFORE 2026-08-31 the two importance gates ranked DIFFERENT UNITS:
#
#   LR  — combined_lr_grouped_importance.csv carries `feature_group`, which is
#         already one row per FEATURE (9 groups: AGE_AT_SEQ_REPORT, ANCESTRY,
#         ETHNICITY, SAMPLE_TYPE_DETAILED, SEX, SUBJECT_CENTER,
#         calling_strategy, number_of_genes, preservation_technique).
#   RF  — combined_feature_importance.csv is one row per ONE-HOT DUMMY
#         (cat__ANCESTRY_Asian, cat__SUBJECT_CENTER_DFCI, …), so a rank cut
#         penalises multi-level categoricals purely for their encoding: a
#         5-level ANCESTRY has its SHAP split five ways while single-column
#         numerics (AGE_AT_SEQ_REPORT, number_of_genes) concentrate theirs in
#         one row. Measured dilution ratio for ANCESTRY in IDC: 2.18.
#
# Consequence (C5 lock SHAP, top-5 cut): ANCESTRY clears the gate in 6/21 cancer
# types when ranked as dummies vs 15/21 when ranked as features. (The 16/21
# figure quoted here before 2026-08-31 was a prototype estimate; 15/21 is the
# verified value from code/tools/report_gate_composition.py.)
#
# This grouping collapses dummy rows to one row per feature so the RF gate is
# structurally symmetric with LR. Enable via
# load_shap_rankings_adaptive(..., group_rf_levels=True, rf_group_agg='sum',
# rf_group_missingness='fold') -- the configuration TIER 7 uses.
#
# ⚠️ Any caller that reports on "what the TIER 7 gate did" MUST pass the same
# three arguments, or it silently describes a gate that was never applied. This
# already bit code/tools/report_gate_composition.py, whose whole claim is that
# its numbers match TIER 7's; it now groups by default too.
#
# The C5 publication lock (2026-08-27) predates this and ran with the RF gate
# disabled entirely, so its 114 hits are NOT comparable to a grouped run's 93
# without saying which unit was ranked.
#
# Group names are deliberately the RAW FEATURE NAMES so the existing
# FEATURE_PATTERNS matching in identify_top_features() keeps working unchanged
# on the grouped frame.
#
# Ordered — FIRST MATCH WINS. The `_missing` indicators MUST precede their
# parent feature, or 'ANCESTRY' would swallow 'ANCESTRY_missing'.
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

# Default number of top features to consider
DEFAULT_TOP_N_FEATURES = 5


def _rf_level_group_for(feature_name):
    """Map one raw RF feature/dummy name to its parent feature group.

    Substring match, case-insensitive, literal (regex=False semantics) — the
    same matching discipline identify_top_features() uses. Returns None when
    nothing matches, so the caller can decide whether to keep the row as its
    own group or raise.
    """
    name = str(feature_name)
    lowered = name.lower()
    for group, patterns in RF_LEVEL_GROUPS:
        for pattern in patterns:
            if str(pattern).lower() in lowered:
                return group
    return None


# Where the ANCESTRY_missing / ETHNICITY_missing indicators go under
# missingness='fold'. This is the mapping TIER 6a's own
# feature_name_mapper.get_feature_group() already applies when it writes
# <CODE>_grouped_feature_importance.csv.
RF_MISSINGNESS_PARENT = {
    'ANCESTRY_missing': 'ANCESTRY',
    'ETHNICITY_missing': 'ETHNICITY',
}


def group_rf_shap_levels(df, agg='sum', missingness='keep', verbose=True):
    """Collapse one-hot dummy rows in an RF SHAP ranking to one row per feature.

    IN PRODUCTION as of 2026-08-31 (E2o, "Arm C"): both TIER 7 scripts call this
    via load_shap_rankings_adaptive(group_rf_levels=True, rf_group_agg='sum',
    rf_group_missingness='fold'). The library default stays off; see the module
    header for why.

    Args:
        df: Normalized ranking frame (ONCOTREE_CODE, feature, mean_abs_shap,
            feature_rank), as produced by load_shap_rankings_adaptive's
            NEW-format branch. Must already be RF-format (one row per dummy).
        agg: How to combine a feature's dummy rows. 'sum' (default) treats the
            dummies as an additive decomposition of the feature's contribution,
            which is what SHAP's additivity implies. 'max' mirrors LR's
            `max_abs_std_log_odds`. See the caveat below.
        missingness: What to do with ANCESTRY_missing / ETHNICITY_missing.
            'keep'  — each is its own feature group (11 groups); they compete
                      for a top-N slot on their own.
            'drop'  — removed entirely before ranking (9 groups).
            'fold'  — merged into their parent ANCESTRY / ETHNICITY group
                      (9 groups). ⭐ This is what TIER 6a ALREADY does when it
                      writes <CODE>_grouped_feature_importance.csv, and it
                      yields exactly the 9 groups the LR file carries. Verified
                      2026-08-31: fold+sum reproduces those files to 8.5e-16
                      with 0/189 rank disagreements.
        verbose: Print a grouping summary.

    Returns:
        DataFrame with one row per (ONCOTREE_CODE, feature), `mean_abs_shap`
        aggregated and `feature_rank` recomputed within each cancer type.

    ⚠️ CAVEAT on agg='sum': summing per-dummy mean|SHAP| is an UPPER BOUND on
    the feature's grouped importance, because it cannot see per-sample
    cancellation across a feature's mutually-exclusive levels. The exact
    quantity is mean|Σ_levels SHAP| computed from per-sample SHAP values.
    TIER 6a does NOT retain per-sample SHAP (only these summary CSVs and PDFs),
    and its own grouped file uses the same sum — so the upper bound is the
    best available without re-running TIER 6a.
    """
    if missingness not in ('keep', 'drop', 'fold'):
        raise ValueError(
            f"missingness must be 'keep', 'drop' or 'fold', got {missingness!r}")

    work = df.copy()
    work['_group'] = work['feature'].map(_rf_level_group_for)

    unmatched = work['_group'].isna()
    if unmatched.any():
        # Keep the row as its own group rather than dropping it — silently
        # losing a feature here would silently shrink the gate.
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
            code_review_and_feedback/shap_gate_provenance_audit_2026-05-07.md
            for the full audit, and BACKLOG.md "SHAP-gate audit — outstanding
            cleanup" for what remains.
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


def load_shap_rankings_adaptive(shap_file, lr_significant_only=False,
                                group_rf_levels=False, rf_group_agg='sum',
                                rf_group_missingness='keep'):
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

    # Set by the LR branch below; consumed by the group_rf_levels block, which
    # is a no-op on LR rankings (already grouped upstream by feature_group).
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

    # PROTOTYPE (2026-08-30, opt-in, default OFF): collapse RF one-hot dummy
    # rows to one row per feature so the RF rank cut ranks the same unit the LR
    # gate already does. Only meaningful for the RF/new format — the LR branch
    # is grouped upstream and the legacy branch is frozen historical output.
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
