#!/usr/bin/env python3
"""
Shared utilities for multi-tier cancer type groupings.

This module provides common functions for loading and using the multi-tier
cancer type grouping system across both ML modeling and post-hoc analysis.

The 3-tier system:
- Tier 1 (21 groups): Specific OncoTree codes with ≥250 actionable events
- Tier 2 (3 groups): Rescue roll-ups for rare types (THYROID, ENCG, BLADDER)
- Tier 3 (10 groups): Anatomical aggregations (LUNG, BREAST, SKIN, etc.)

Key concepts:
- tier_prefixed_code: Full group name with tier prefix (e.g., "tier1_LUAD", "tier3_LUNG")
- group_code: Short group name without prefix (e.g., "LUAD", "LUNG")
- original_oncotree_code: Database ONCOTREE_CODE values (e.g., "LUAD", "NSCLC")

Usage:
    from tools.multi_tier_utils import (
        load_patient_to_group_mapping,
        get_patient_ids_for_group,
        get_oncotree_codes_for_group,
        get_all_tier_prefixed_codes
    )

    # Get patient IDs for a tier group (recommended approach)
    patient_ids = get_patient_ids_for_group("tier3_LUNG")

    # Get OncoTree codes for gene-level queries
    oncotree_codes = get_oncotree_codes_for_group("tier3_LUNG")
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Set

# Repository root (relative to this file in code/tools/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# =============================================================================
# File Location Functions
# =============================================================================

def get_multi_tier_groupings_dir(repo_root: Optional[Path] = None) -> Optional[Path]:
    """
    Find the multi-tier groupings directory.

    Searches in standard locations:
    1. analysis_output/multi_tier_groupings/
    2. runs/run_*/tier2_data_setup/multi_tier_groupings/

    Args:
        repo_root: Repository root path (optional, uses default if not provided)

    Returns:
        Path to the groupings directory, or None if not found
    """
    if repo_root is None:
        repo_root = REPO_ROOT

    repo_root = Path(repo_root)

    # Check analysis_output first (default location)
    default_dir = repo_root / "analysis_output" / "multi_tier_groupings"
    if default_dir.exists():
        return default_dir

    # Check runs directory for any run with multi-tier output
    runs_dir = repo_root / "runs"
    if runs_dir.exists():
        # Find latest run with multi-tier groupings
        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        for run_dir in runs:
            groupings_dir = run_dir / "tier2_data_setup" / "multi_tier_groupings"
            if groupings_dir.exists():
                return groupings_dir

    return None


def get_patient_to_group_mapping_file(repo_root: Optional[Path] = None) -> Optional[Path]:
    """
    Find the patient_to_group_mapping.csv file.

    This file maps each patient to their tier group(s).

    Returns:
        Path to the mapping file, or None if not found
    """
    groupings_dir = get_multi_tier_groupings_dir(repo_root)
    if groupings_dir is None:
        return None

    mapping_file = groupings_dir / "patient_to_group_mapping.csv"
    if mapping_file.exists():
        return mapping_file

    return None


def get_group_definitions_file(repo_root: Optional[Path] = None) -> Optional[Path]:
    """
    Find the multi_tier_group_definitions.csv file.

    This file contains summary statistics for each tier group.

    Returns:
        Path to the definitions file, or None if not found
    """
    groupings_dir = get_multi_tier_groupings_dir(repo_root)
    if groupings_dir is None:
        return None

    defs_file = groupings_dir / "multi_tier_group_definitions.csv"
    if defs_file.exists():
        return defs_file

    return None


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_patient_to_group_mapping(
    repo_root: Optional[Path] = None,
    mapping_file: Optional[Path] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load the patient-to-group mapping DataFrame.

    Columns in the mapping file:
    - PATIENT_ID: Patient identifier
    - original_oncotree_code: Database ONCOTREE_CODE value
    - is_actionable: 1 if patient has actionable event
    - tier: "tier1", "tier2", or "tier3"
    - group_code: Short group name (e.g., "LUAD", "LUNG")
    - tier_prefixed_code: Full group name (e.g., "tier1_LUAD", "tier3_LUNG")
    - group_name: Human-readable name

    Args:
        repo_root: Repository root path (optional)
        mapping_file: Direct path to mapping CSV (overrides search)
        verbose: Whether to print loading info

    Returns:
        DataFrame with patient-to-group mapping

    Raises:
        FileNotFoundError: If no mapping file is found
    """
    if mapping_file is not None:
        map_file = Path(mapping_file)
    else:
        map_file = get_patient_to_group_mapping_file(repo_root)

    if map_file is None or not map_file.exists():
        raise FileNotFoundError(
            "Patient-to-group mapping not found. "
            "Run define_multi_tier_groupings_v3.py first, or specify mapping_file path."
        )

    if verbose:
        print(f"Loading patient-to-group mapping from: {map_file}")

    df = pd.read_csv(map_file)

    if verbose:
        n_patients = df['PATIENT_ID'].nunique()
        n_groups = df['tier_prefixed_code'].nunique()
        print(f"  Loaded {len(df):,} patient-group assignments")
        print(f"  Unique patients: {n_patients:,}")
        print(f"  Unique groups: {n_groups}")

    return df


def load_group_definitions(
    repo_root: Optional[Path] = None,
    definitions_file: Optional[Path] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load the group definitions DataFrame.

    Columns:
    - tier: "tier1", "tier2", or "tier3"
    - tier_prefixed_code: Full group name (e.g., "tier1_LUAD")
    - group_code: Short group name (e.g., "LUAD")
    - group_name: Human-readable name
    - num_original_codes: Number of OncoTree codes in this group
    - total_patients: Total patient count
    - actionable_patients: Actionable patient count
    - actionability_rate: Percentage actionable

    Args:
        repo_root: Repository root path (optional)
        definitions_file: Direct path to definitions CSV (overrides search)
        verbose: Whether to print loading info

    Returns:
        DataFrame with group definitions

    Raises:
        FileNotFoundError: If no definitions file is found
    """
    if definitions_file is not None:
        defs_file = Path(definitions_file)
    else:
        defs_file = get_group_definitions_file(repo_root)

    if defs_file is None or not defs_file.exists():
        raise FileNotFoundError(
            "Group definitions not found. "
            "Run define_multi_tier_groupings_v3.py first, or specify definitions_file path."
        )

    if verbose:
        print(f"Loading group definitions from: {defs_file}")

    df = pd.read_csv(defs_file)

    if verbose:
        for tier in sorted(df['tier'].unique()):
            count = len(df[df['tier'] == tier])
            print(f"  {tier}: {count} groups")

    return df


# =============================================================================
# Tier Group Query Functions
# =============================================================================

def get_all_tier_prefixed_codes(
    tier: Optional[str] = None,
    repo_root: Optional[Path] = None
) -> List[str]:
    """
    Get list of all tier_prefixed_codes.

    Args:
        tier: Filter to specific tier ('tier1', 'tier2', 'tier3', or None for all)
        repo_root: Repository root path (optional)

    Returns:
        List of tier_prefixed_code values (e.g., ["tier1_LUAD", "tier3_LUNG", ...])

    Example:
        >>> get_all_tier_prefixed_codes(tier='tier1')
        ['tier1_AML', 'tier1_ASTR', 'tier1_BLCA', ...]
    """
    defs = load_group_definitions(repo_root, verbose=False)

    if tier is not None:
        if tier not in defs['tier'].unique():
            raise ValueError(f"Unknown tier: {tier}. Valid tiers: {defs['tier'].unique().tolist()}")
        defs = defs[defs['tier'] == tier]

    return sorted(defs['tier_prefixed_code'].tolist())


def get_patient_ids_for_group(
    tier_prefixed_code: str,
    mapping_df: pd.DataFrame = None,
    repo_root: Optional[Path] = None
) -> List[str]:
    """
    Get list of patient IDs belonging to a tier group.

    This is the RECOMMENDED approach for filtering data by tier group,
    as it uses the exact patient assignments from the mapping file.

    Args:
        tier_prefixed_code: Full group name (e.g., "tier1_LUAD", "tier3_LUNG")
        mapping_df: Pre-loaded mapping DataFrame (optional, loads if not provided)
        repo_root: Repository root path (optional)

    Returns:
        List of PATIENT_ID values

    Example:
        >>> patient_ids = get_patient_ids_for_group("tier3_LUNG")
        >>> df_lung = full_data[full_data['PATIENT_ID'].isin(patient_ids)]
    """
    if mapping_df is None:
        mapping_df = load_patient_to_group_mapping(repo_root, verbose=False)

    patient_ids = mapping_df[
        mapping_df['tier_prefixed_code'] == tier_prefixed_code
    ]['PATIENT_ID'].unique().tolist()

    return patient_ids


def get_oncotree_codes_for_group(
    tier_prefixed_code: str,
    mapping_df: pd.DataFrame = None,
    repo_root: Optional[Path] = None
) -> List[str]:
    """
    Get list of OncoTree codes that belong to a tier group.

    Use this for queries against tables that have ONCOTREE_CODE column
    (e.g., perPatientOncoKBLevelSorted for gene-level queries).

    For tier1 groups, returns the single OncoTree code (1:1 mapping).
    For tier2/tier3 groups, returns all constituent OncoTree codes.

    Args:
        tier_prefixed_code: Full group name (e.g., "tier1_LUAD", "tier3_LUNG")
        mapping_df: Pre-loaded mapping DataFrame (optional, loads if not provided)
        repo_root: Repository root path (optional)

    Returns:
        List of OncoTree code values

    Example:
        >>> get_oncotree_codes_for_group("tier1_LUAD")
        ['LUAD']
        >>> get_oncotree_codes_for_group("tier3_LUNG")
        ['LUAD', 'NSCLC']
        >>> get_oncotree_codes_for_group("tier3_BREAST")
        ['IDC', 'ILC', 'BRCA', 'BRCANOS', 'BRCNOS', 'MDLC']
    """
    if mapping_df is None:
        mapping_df = load_patient_to_group_mapping(repo_root, verbose=False)

    oncotree_codes = mapping_df[
        mapping_df['tier_prefixed_code'] == tier_prefixed_code
    ]['original_oncotree_code'].unique().tolist()

    return sorted(oncotree_codes)


def get_group_info(
    tier_prefixed_code: str,
    repo_root: Optional[Path] = None
) -> Dict:
    """
    Get information about a tier group.

    Args:
        tier_prefixed_code: Full group name (e.g., "tier1_LUAD", "tier3_LUNG")
        repo_root: Repository root path (optional)

    Returns:
        Dict with group information:
        - tier: "tier1", "tier2", or "tier3"
        - group_code: Short name (e.g., "LUAD", "LUNG")
        - group_name: Human-readable name
        - oncotree_codes: List of constituent OncoTree codes
        - total_patients: Total patient count
        - actionable_patients: Actionable patient count
    """
    defs = load_group_definitions(repo_root, verbose=False)
    mapping = load_patient_to_group_mapping(repo_root, verbose=False)

    group_row = defs[defs['tier_prefixed_code'] == tier_prefixed_code]
    if len(group_row) == 0:
        raise ValueError(f"Unknown tier_prefixed_code: {tier_prefixed_code}")

    row = group_row.iloc[0]
    oncotree_codes = get_oncotree_codes_for_group(tier_prefixed_code, mapping)

    return {
        'tier': row['tier'],
        'tier_prefixed_code': tier_prefixed_code,
        'group_code': row['group_code'],
        'group_name': row['group_name'],
        'oncotree_codes': oncotree_codes,
        'total_patients': int(row['total_patients']),
        'actionable_patients': int(row['actionable_patients']),
        'actionability_rate': float(row['actionability_rate'])
    }


# =============================================================================
# Convenience Functions for Common Patterns
# =============================================================================

def build_oncotree_to_group_mapping(
    tier: Optional[str] = None,
    repo_root: Optional[Path] = None
) -> Dict[str, str]:
    """
    Build mapping from OncoTree code to tier_prefixed_code.

    Useful for aggregating results from OncoTree-code-level queries
    back to tier groups.

    Args:
        tier: Filter to specific tier (optional)
        repo_root: Repository root path (optional)

    Returns:
        Dict mapping OncoTree code -> tier_prefixed_code
        e.g., {"LUAD": "tier3_LUNG", "NSCLC": "tier3_LUNG", ...}

    Note:
        For tier3, OncoTree codes may map to multiple groups.
        This function returns the tier3 group when available.
    """
    mapping_df = load_patient_to_group_mapping(repo_root, verbose=False)

    if tier is not None:
        mapping_df = mapping_df[mapping_df['tier'] == tier]

    # Build mapping (later tiers override earlier for same code)
    oncotree_to_group = {}
    for _, row in mapping_df.iterrows():
        oncotree_to_group[row['original_oncotree_code']] = row['tier_prefixed_code']

    return oncotree_to_group


def filter_dataframe_by_group(
    df: pd.DataFrame,
    tier_prefixed_code: str,
    patient_id_column: str = 'PATIENT_ID',
    mapping_df: pd.DataFrame = None,
    repo_root: Optional[Path] = None
) -> pd.DataFrame:
    """
    Filter a DataFrame to patients belonging to a tier group.

    This is the recommended way to filter data for a specific tier group.

    Args:
        df: DataFrame to filter (must have patient_id_column)
        tier_prefixed_code: Group to filter to (e.g., "tier3_LUNG")
        patient_id_column: Column name containing patient IDs
        mapping_df: Pre-loaded mapping DataFrame (optional)
        repo_root: Repository root path (optional)

    Returns:
        Filtered DataFrame containing only patients in the group
    """
    patient_ids = get_patient_ids_for_group(
        tier_prefixed_code, mapping_df, repo_root
    )
    return df[df[patient_id_column].isin(patient_ids)].copy()


# =============================================================================
# Validation Functions
# =============================================================================

def validate_tier_prefixed_code(tier_prefixed_code: str, repo_root: Optional[Path] = None) -> bool:
    """
    Check if a tier_prefixed_code is valid.

    Args:
        tier_prefixed_code: Code to validate
        repo_root: Repository root path (optional)

    Returns:
        True if valid, False otherwise
    """
    try:
        valid_codes = get_all_tier_prefixed_codes(repo_root=repo_root)
        return tier_prefixed_code in valid_codes
    except FileNotFoundError:
        return False


def parse_tier_prefixed_code(tier_prefixed_code: str) -> Dict[str, str]:
    """
    Parse a tier_prefixed_code into its components.

    Args:
        tier_prefixed_code: Code to parse (e.g., "tier1_LUAD", "tier3_LUNG")

    Returns:
        Dict with 'tier' and 'group_code' keys

    Raises:
        ValueError: If code doesn't match expected format
    """
    if not tier_prefixed_code.startswith('tier'):
        raise ValueError(
            f"Invalid tier_prefixed_code format: {tier_prefixed_code}. "
            f"Expected format: tier[1-3]_<GROUP_CODE>"
        )

    parts = tier_prefixed_code.split('_', 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid tier_prefixed_code format: {tier_prefixed_code}. "
            f"Expected format: tier[1-3]_<GROUP_CODE>"
        )

    tier = parts[0]
    group_code = parts[1]

    if tier not in ['tier1', 'tier2', 'tier3']:
        raise ValueError(
            f"Invalid tier in {tier_prefixed_code}. "
            f"Expected tier1, tier2, or tier3"
        )

    return {'tier': tier, 'group_code': group_code}


# =============================================================================
# Main (for testing/debugging)
# =============================================================================

if __name__ == "__main__":
    print("Multi-Tier Utilities - Testing")
    print("=" * 60)

    try:
        # Test loading
        defs = load_group_definitions()
        mapping = load_patient_to_group_mapping()

        # Test getting codes
        print("\nTier 1 groups:")
        tier1_codes = get_all_tier_prefixed_codes(tier='tier1')
        print(f"  {len(tier1_codes)} groups: {tier1_codes[:5]}...")

        print("\nTier 3 groups:")
        tier3_codes = get_all_tier_prefixed_codes(tier='tier3')
        print(f"  {len(tier3_codes)} groups: {tier3_codes}")

        # Test OncoTree code lookup
        print("\nOncoTree codes for tier3_LUNG:")
        lung_codes = get_oncotree_codes_for_group("tier3_LUNG", mapping)
        print(f"  {lung_codes}")

        print("\nOncoTree codes for tier3_BREAST:")
        breast_codes = get_oncotree_codes_for_group("tier3_BREAST", mapping)
        print(f"  {breast_codes}")

        # Test patient ID lookup
        print("\nPatient IDs for tier3_LUNG:")
        lung_patients = get_patient_ids_for_group("tier3_LUNG", mapping)
        print(f"  {len(lung_patients)} patients")

        print("\n" + "=" * 60)
        print("All tests passed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
