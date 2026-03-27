"""
Shared utilities for SHAP-based feature analysis.

This module provides common functions for loading and processing SHAP feature
rankings used by multiple post-hoc analysis scripts.

Functions:
    - get_default_shap_file: Find SHAP rankings file in standard locations
    - load_shap_rankings_adaptive: Load SHAP rankings with auto-format detection
    - identify_top_features: Find cancer types with specific features in top N
    - load_multi_tier_cancer_types: Load canonical cancer type list from multi-tier groupings

Used by:
    - analyze_shap_feature_associations.py (cancer-level post-hoc)
    - analyze_gene_stratified_associations.py (gene-stratified post-hoc)
    - explore_gene_distributions.py (gene distribution exploration)
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional

# Get repository root (relative to this file in code/tools/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Legacy SHAP file path (for backward compatibility)
LEGACY_SHAP_FILE = "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522b/demographic_technical_ml_modeling/shap_feature_ranks.csv"

# Feature patterns for pattern matching in both old and new formats
# Old format uses sklearn-style names: "cat__SEX_Male"
# New format uses readable names: "Sex: Male"
FEATURE_PATTERNS = {
    'SEX': ['SEX_', 'Sex:', 'SEX'],
    'RACE': ['RACE_', 'Race:', 'RACE'],
    'ETHNICITY': ['ETHNICITY_', 'Ethnicity:', 'ETHNICITY'],
    'RACE_ETHNICITY': ['RACE_', 'ETHNICITY_', 'Race:', 'Ethnicity:', 'RACE', 'ETHNICITY'],
    'SAMPLE_TYPE': ['SAMPLE_TYPE_', 'Sample Type', 'SAMPLE_TYPE'],
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
        # Legacy patterns (for backwards compatibility)
        'library_selection_', 'Library Selection:'
    ]
}

# Default number of top features to consider
DEFAULT_TOP_N_FEATURES = 5


def get_default_shap_file(repo_root=None):
    """
    Get default SHAP file, preferring new format (TIER 6a) over legacy.

    Searches for SHAP primary values output from compute_shap_primary_values.py
    in the standard output locations.

    Args:
        repo_root: Repository root path (optional, uses default if not provided)

    Returns:
        str: Path to SHAP rankings file, or None if not found
    """
    if repo_root is None:
        repo_root = REPO_ROOT

    repo_root = Path(repo_root)

    # Check for new format output in analysis_output
    new_output = repo_root / "analysis_output" / "shap_primary_values"
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
    if runs_dir.exists():
        # Find latest run with SHAP output
        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        for run_dir in runs:
            shap_dir = run_dir / "tier6_shap_analysis" / "primary_values"
            combined_file = shap_dir / "combined_feature_importance.csv"
            if combined_file.exists():
                print(f"Found SHAP file (new format, unified runs): {combined_file}")
                return str(combined_file)

    # Fall back to legacy file
    if Path(LEGACY_SHAP_FILE).exists():
        print(f"Using legacy SHAP file: {LEGACY_SHAP_FILE}")
        print("  WARNING: Consider running TIER 6a (compute_shap_primary_values.py) for updated SHAP rankings")
        return LEGACY_SHAP_FILE

    return None


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
        top_n: Number of top features to consider (default: 5)
        verbose: Whether to print progress (default: True)

    Returns:
        pd.DataFrame with columns:
            - pattern: matched pattern
            - ONCOTREE_CODE: cancer type
            - feature: feature name
            - feature_rank: rank (1-based)
            - mean_abs_shap: SHAP importance value
    """
    results = []

    for pattern in feature_patterns:
        # Filter for features matching pattern
        mask = shap_df['feature'].str.contains(pattern, case=False, na=False)
        matching = shap_df[mask & (shap_df['feature_rank'] <= top_n)].copy()

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


def _strip_tier_prefix(cancer_type):
    """
    Strip tier prefix from a tier_prefixed_code if present.

    Handles both tier_prefixed_code (e.g., "tier1_AML") and plain ONCOTREE_CODE (e.g., "AML").
    This allows SHAP lookups to work regardless of whether the caller passes
    a tier_prefixed_code or a plain ONCOTREE_CODE.

    Args:
        cancer_type: Either "tier1_AML" or "AML"

    Returns:
        str: Plain ONCOTREE_CODE (e.g., "AML")
    """
    import re
    match = re.match(r'^tier\d+_(.+)$', cancer_type)
    if match:
        return match.group(1)
    return cancer_type


def feature_in_top_n_for_cancer(shap_df, cancer_type, feature_category, top_n=DEFAULT_TOP_N_FEATURES):
    """
    Check if a feature category appears in top N for a specific cancer type.

    Handles tier_prefixed_code (e.g., "tier1_AML", "tier3_LUNG") and plain ONCOTREE_CODE
    by resolving to the constituent OncoTree codes used in SHAP data.

    For tier1 groups: checks the single OncoTree code (e.g., tier1_AML → AML)
    For tier2/tier3 groups: checks ALL constituent OncoTree codes and returns True
        if the feature is in top N for ANY of them (e.g., tier3_LUNG → LUAD, NSCLC)

    Args:
        shap_df: DataFrame with SHAP rankings
        cancer_type: ONCOTREE_CODE or tier_prefixed_code to check
        feature_category: Key from FEATURE_PATTERNS
        top_n: Number of top features to consider

    Returns:
        bool: True if feature category is in top N for this cancer type
    """
    cancer_types_with_feature = get_cancer_types_with_top_feature(shap_df, feature_category, top_n)

    # Try direct match first
    plain_code = _strip_tier_prefix(cancer_type)
    if cancer_type in cancer_types_with_feature or plain_code in cancer_types_with_feature:
        return True

    # For tier2/tier3 groups, check constituent OncoTree codes
    import re
    if re.match(r'^tier[23]_', cancer_type):
        try:
            constituent_codes = _get_oncotree_codes_for_group(cancer_type)
            for code in constituent_codes:
                if code in cancer_types_with_feature:
                    return True
        except Exception:
            pass  # Fall through to return False

    return False


# =============================================================================
# Multi-Tier Cancer Type Groupings
# =============================================================================
# NOTE: These functions are now wrappers around the canonical implementations
# in multi_tier_utils.py for backward compatibility. New code should import
# directly from multi_tier_utils.py.

# Import from sibling module (same directory)
from multi_tier_utils import (
    get_group_definitions_file as get_multi_tier_definitions_file,
    load_group_definitions as load_multi_tier_definitions,
    get_all_tier_prefixed_codes,
    get_patient_ids_for_group,
    get_oncotree_codes_for_group as _get_oncotree_codes_for_group,
    load_patient_to_group_mapping,
    get_patient_to_group_mapping_file,
    build_oncotree_to_group_mapping
)


def load_multi_tier_cancer_types(
    tier: Optional[str] = None,
    repo_root: Optional[Path] = None,
    definitions_file: Optional[Path] = None
) -> List[str]:
    """
    Load canonical cancer type list from multi-tier groupings output.

    DEPRECATED: Use get_all_tier_prefixed_codes() from multi_tier_utils.py instead.
    This function returns group_code (e.g., "LUAD") not tier_prefixed_code (e.g., "tier1_LUAD").

    For new code, prefer:
        from tools.multi_tier_utils import get_all_tier_prefixed_codes
        tier_codes = get_all_tier_prefixed_codes(tier='tier1')

    Args:
        tier: Filter to specific tier ('tier1', 'tier2', 'tier3', or None for all)
        repo_root: Repository root path (optional)
        definitions_file: Direct path to definitions CSV (overrides search)

    Returns:
        List of group_code values (e.g., ["LUAD", "BRCA", ...])
    """
    defs = load_multi_tier_definitions(repo_root, definitions_file, verbose=False)

    if tier is not None:
        if tier not in defs['tier'].unique():
            raise ValueError(f"Unknown tier: {tier}. Available tiers: {defs['tier'].unique().tolist()}")
        defs = defs[defs['tier'] == tier]

    return defs['group_code'].tolist()


def load_group_to_oncotree_mapping(
    repo_root: Optional[Path] = None,
    mapping_file: Optional[Path] = None
) -> dict:
    """
    Load mapping from group_code to list of constituent OncoTree codes.

    DEPRECATED: Use get_oncotree_codes_for_group() from multi_tier_utils.py instead,
    which accepts tier_prefixed_code for consistency.

    Returns:
        Dict mapping group_code -> list of original OncoTree codes
    """
    mapping_df = load_patient_to_group_mapping(repo_root, mapping_file, verbose=True)

    # Create mapping: group_code -> unique list of original_oncotree_code
    mapping = {}
    for group_code in mapping_df['group_code'].unique():
        oncotree_codes = mapping_df[
            mapping_df['group_code'] == group_code
        ]['original_oncotree_code'].unique().tolist()
        mapping[group_code] = oncotree_codes

    return mapping


def get_oncotree_codes_for_group(
    group_code: str,
    group_mapping: dict = None,
    repo_root: Optional[Path] = None
) -> List[str]:
    """
    Get the list of OncoTree codes that belong to a group.

    DEPRECATED: Use get_oncotree_codes_for_group() from multi_tier_utils.py instead,
    which accepts tier_prefixed_code for consistency.

    Args:
        group_code: The group code (e.g., "LUAD", "LUNG", "BREAST")
        group_mapping: Pre-loaded mapping dict (optional, loads if not provided)
        repo_root: Repository root path (optional)

    Returns:
        List of OncoTree codes belonging to this group
    """
    if group_mapping is None:
        group_mapping = load_group_to_oncotree_mapping(repo_root)

    if group_code in group_mapping:
        return group_mapping[group_code]
    else:
        # Fallback: assume 1:1 mapping (group_code is the OncoTree code)
        print(f"  Warning: Group code '{group_code}' not in mapping, using as-is")
        return [group_code]


def get_tier_for_group(
    group_code: str,
    repo_root: Optional[Path] = None,
    definitions_file: Optional[Path] = None
) -> Optional[str]:
    """
    Get the tier (tier1, tier2, tier3) for a group code.

    Args:
        group_code: The group code to look up
        repo_root: Repository root path (optional)
        definitions_file: Direct path to definitions CSV (optional)

    Returns:
        Tier string ('tier1', 'tier2', or 'tier3'), or None if not found
    """
    try:
        defs = load_multi_tier_definitions(repo_root, definitions_file, verbose=False)
        match = defs[defs['group_code'] == group_code]
        if len(match) > 0:
            return match['tier'].iloc[0]
    except FileNotFoundError:
        pass

    return None
