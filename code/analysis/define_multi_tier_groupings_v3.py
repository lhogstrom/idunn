#!/usr/bin/env python3
"""
Generate Multi-Tier Cancer Type Groupings (v2 - with OncoTree hierarchy traversal)

This script defines three tiers of cancer type groupings:
- Tier 1: 21 existing OncoTree codes (≥250 actionable, "as-is")
- Tier 2: +3 rescue roll-ups (THYROID, ENCG, BLADDER) with ALL descendants
- Tier 3: +5 Level 3 OncoTree aggregations with ALL descendants

Generates patient-to-group mapping table with tier-prefixed group names
(e.g., 'tier1_LUAD', 'tier2_THYROID', 'tier3_DIFG').

Patient overlap across tiers is intentional and allowed.

**Version 2 Changes**:
- Added get_descendant_codes() function to traverse OncoTree hierarchy
- Tier 2 now finds ALL codes that roll up to THYROID, ENCG, BLADDER
- Tier 3 now finds ALL codes that roll up to Level 3 parent codes

Output:
- multi_tier_group_definitions.csv: Group definitions with metadata
- patient_to_group_mapping.csv: Patient-level mapping (patient_id, tier, group_code, etc.)
- group_summary_stats.csv: Summary statistics per group

Author: Balder Analysis Pipeline
Date: January 14, 2026
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Database path
DB_PATH = "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522/balder-harmonized-biomarker-data-v20250522.sqlite"

# Input CSV paths
GROUPING_DIR = Path(__file__).parent.parent.parent / "analysis_output" / "oncotree_grouping_analysis"
B1_UNDER250_CSV = GROUPING_DIR / "oncotree_grouping_B1_Under250.csv"
A2_LEVEL3TO2_CSV = GROUPING_DIR / "oncotree_grouping_A2_Level3to2.csv"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "analysis_output" / "multi_tier_groupings"

# Minimum actionable threshold
MIN_ACTIONABLE = 250


def get_descendant_codes(db_path, parent_codes):
    """
    Get all OncoTree codes that roll up to any of the parent codes (descendants).

    Uses OncoTreeCodeHierarchy table to traverse the hierarchy.
    For each parent code, finds all codes where the parent appears anywhere
    in the hierarchy path (ot_code_level_1, ot_code_level_2, ot_code_level_3, ...).

    Args:
        db_path: Path to SQLite database
        parent_codes: List of parent OncoTree codes (e.g., ['THYROID', 'ENCG', 'BLADDER'])

    Returns:
        Dictionary: {parent_code: set of descendant codes (including parent)}

    Example:
        >>> descendants = get_descendant_codes(db_path, ['THYROID'])
        >>> # Returns: {'THYROID': {'THYROID', 'THPA', 'WDTC', ...}}
    """
    print(f"\n  Querying OncoTree hierarchy for descendants of: {parent_codes}")

    conn = sqlite3.connect(db_path)

    # Load OncoTree hierarchy table
    hierarchy_df = pd.read_sql_query("SELECT * FROM OncoTreeCodeHierarchy", conn)
    conn.close()

    print(f"  Loaded {len(hierarchy_df):,} OncoTree hierarchy entries")

    descendant_dict = {}

    for parent_code in parent_codes:
        # Include parent itself
        descendants = set([parent_code])

        # Find all rows where parent_code appears in any hierarchy level column
        # Hierarchy columns are like: ot_code_level_1, ot_code_level_2, ot_code_level_3, etc.
        level_columns = [col for col in hierarchy_df.columns if col.startswith('ot_code_level_')]

        for idx, row in hierarchy_df.iterrows():
            # Check if parent_code appears in any level of this row's hierarchy
            for col in level_columns:
                if pd.notna(row[col]) and row[col] == parent_code:
                    # This row's ot_code descends from parent_code
                    descendants.add(row['ot_code'])
                    break  # Don't need to check other columns for this row

        descendant_dict[parent_code] = descendants
        print(f"    {parent_code}: Found {len(descendants)} descendant codes (including parent)")

    return descendant_dict


def load_existing_grouping_files():
    """Load existing B1_Under250 and A2_Level3to2 grouping files."""
    print("\n" + "="*80)
    print("STEP 1: Loading existing grouping files")
    print("="*80)

    if not B1_UNDER250_CSV.exists():
        raise FileNotFoundError(f"B1_Under250 file not found: {B1_UNDER250_CSV}")
    if not A2_LEVEL3TO2_CSV.exists():
        raise FileNotFoundError(f"A2_Level3to2 file not found: {A2_LEVEL3TO2_CSV}")

    b1_df = pd.read_csv(B1_UNDER250_CSV)
    a2_df = pd.read_csv(A2_LEVEL3TO2_CSV)

    print(f"\nLoaded B1_Under250: {len(b1_df)} groups")
    print(f"Loaded A2_Level3to2: {len(a2_df)} groups")

    return b1_df, a2_df


def define_tier1_groups(b1_df):
    """
    Define Tier 1 groups: Top 21 groups by actionable patients.

    These are the existing groups from the original 21-cancer-type analysis.
    Some may have rolled-up children (e.g., BRCA, MEL, AML, NSCLC, COADREAD).
    """
    print("\n" + "="*80)
    print("STEP 2: Defining Tier 1 Groups")
    print("="*80)

    # Filter: meets_threshold=True, exclude Tier 2 rescue groups
    tier2_codes = ['THYROID', 'ENCG', 'BLADDER']

    tier1 = b1_df[
        (b1_df['meets_threshold'] == True) &
        (~b1_df['group_code'].isin(tier2_codes))
    ].copy()

    tier1['tier'] = 'tier1'
    tier1['tier_prefixed_code'] = 'tier1_' + tier1['group_code']

    # Select and rename columns
    tier1 = tier1[[
        'tier', 'tier_prefixed_code', 'group_code', 'group_name',
        'num_original_codes', 'num_rolled_up', 'total_patients',
        'actionable_patients', 'actionability_rate'
    ]]

    print(f"\nTier 1 groups defined: {len(tier1)}")
    print("\nTier 1 group codes:")
    for code in sorted(tier1['group_code'].tolist()):
        print(f"  - {code}")

    return tier1


def define_tier2_groups(b1_df):
    """
    Define Tier 2 groups: Rescue roll-ups (THYROID, ENCG, BLADDER).

    These are codes with ≥250 actionable after rolling up under-threshold children.
    """
    print("\n" + "="*80)
    print("STEP 3: Defining Tier 2 Groups")
    print("="*80)

    # Rescue groups from B1_Under250
    rescue_codes = ['THYROID', 'ENCG', 'BLADDER']

    tier2 = b1_df[
        (b1_df['meets_threshold'] == True) &
        (b1_df['num_rolled_up'] > 0) &
        (b1_df['group_code'].isin(rescue_codes))
    ].copy()

    tier2['tier'] = 'tier2'
    tier2['tier_prefixed_code'] = 'tier2_' + tier2['group_code']

    # Select and rename columns
    tier2 = tier2[[
        'tier', 'tier_prefixed_code', 'group_code', 'group_name',
        'num_original_codes', 'num_rolled_up', 'total_patients',
        'actionable_patients', 'actionability_rate'
    ]]

    print(f"\nTier 2 rescue groups defined: {len(tier2)}")
    print("\nTier 2 group codes:")
    for code in sorted(tier2['group_code'].tolist()):
        actionable = tier2[tier2['group_code']==code]['actionable_patients'].values[0]
        print(f"  - {code} ({actionable} actionable)")

    if len(tier2) != 3:
        print(f"\nWARNING: Expected 3 Tier 2 groups, found {len(tier2)}")

    return tier2


def define_tier3_groups_from_level1(tier1_codes, tier2_codes):
    """
    Define Tier 3 groups: Level 1 OncoTree aggregations with ≥250 actionable.
    
    Queries database directly for Level 1 patient counts.
    Excludes codes already in Tier 1 or Tier 2 to avoid duplication.
    """
    print("\n" + "="*80)
    print("STEP 4: Defining Tier 3 Groups")
    print("="*80)
    
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    
    # Query Level 1 patient counts
    query = """
    SELECT
        h.ot_code_level_1 as group_code,
        COUNT(DISTINCT p.PATIENT_ID) as total_patients,
        SUM(p.isOncoKbLevel1) as actionable_patients
    FROM perPatientOncoKBFilteredForModeling p
    JOIN OncoTreeCodeHierarchy h ON p.ONCOTREE_CODE = h.ot_code
    WHERE h.ot_code_level_1 IS NOT NULL
    GROUP BY h.ot_code_level_1
    """
    
    tier3 = pd.read_sql_query(query, conn)
    conn.close()
    
    # Filter: ≥250 actionable AND not in tier1/tier2
    # Convert tier2_codes list to simple codes (remove tier prefix if present)
    tier2_simple_codes = [c.replace('tier2_', '') if c.startswith('tier2_') else c for c in tier2_codes]
    
    # Exclude Tier 1 and Tier 2 codes
    excluded_codes = tier1_codes + tier2_simple_codes
    tier3 = tier3[
        (tier3['actionable_patients'] >= 250) &
        (~tier3['group_code'].isin(excluded_codes))
    ].copy()
    
    # Calculate actionability rate
    tier3['actionability_rate'] = (tier3['actionable_patients'] / tier3['total_patients'] * 100)
    
    # Add tier columns
    tier3['tier'] = 'tier3'
    tier3['tier_prefixed_code'] = 'tier3_' + tier3['group_code']
    
    # Add group name (use group_code as name for now, can be enhanced)
    tier3['group_name'] = tier3['group_code'].str.replace('_', ' ').str.title()
    
    # Add num_original_codes (1 for Level 1 codes)
    tier3['num_original_codes'] = 1
    
    # Add num_rolled_up (will be calculated by hierarchy)
    tier3['num_rolled_up'] = 0  # Will be updated after hierarchy traversal
    
    # Select consistent columns
    tier3 = tier3[[
        'tier', 'tier_prefixed_code', 'group_code', 'group_name',
        'num_original_codes', 'num_rolled_up', 'total_patients',
        'actionable_patients', 'actionability_rate'
    ]]
    
    # Sort by actionable patients descending
    tier3 = tier3.sort_values('actionable_patients', ascending=False).reset_index(drop=True)
    
    print(f"\nTier 3 groups defined: {len(tier3)}")
    print(f"\nTier 3 Level 1 aggregations (all by actionable patients):")
    for _, row in tier3.iterrows():
        print(f"  - {row['group_code']}: {int(row['actionable_patients'])} actionable, "
              f"{int(row['total_patients'])} total ({row['actionability_rate']:.1f}%)")
    
    return tier3


def combine_tiers(tier1, tier2, tier3):
    """Combine all three tiers into a single DataFrame."""
    print("\n" + "="*80)
    print("STEP 5: Combining All Tiers")
    print("="*80)

    # Reset index to avoid InvalidIndexError
    tier1 = tier1.reset_index(drop=True).copy()
    tier2 = tier2.reset_index(drop=True).copy()
    tier3 = tier3.reset_index(drop=True).copy()

    all_groups = pd.concat([tier1, tier2, tier3], ignore_index=True)

    print(f"\nTotal groups across all tiers: {len(all_groups)}")
    print(f"  - Tier 1: {len(tier1)} groups")
    print(f"  - Tier 2: {len(tier2)} groups")
    print(f"  - Tier 3: {len(tier3)} groups")

    # Validation
    duplicates = all_groups['tier_prefixed_code'].duplicated()
    if duplicates.any():
        print(f"\nWARNING: Found {duplicates.sum()} duplicate tier_prefixed_codes:")
        print(all_groups[duplicates]['tier_prefixed_code'].tolist())

    return all_groups


def generate_patient_mapping(all_groups):
    """
    Generate patient-to-group mapping table using OncoTree hierarchy traversal.

    **Key change**: Uses get_descendant_codes() for Tier 2 and Tier 3 to find
    ALL codes that roll up to parent codes, not just direct children.

    Returns DataFrame with columns:
    - PATIENT_ID
    - tier
    - group_code
    - tier_prefixed_code
    - group_name
    - original_oncotree_code
    - is_actionable
    """
    print("\n" + "="*80)
    print("STEP 6: Generating Patient-to-Group Mapping (with hierarchy traversal)")
    print("="*80)

    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # Pre-compute descendant mappings for Tier 2 and Tier 3
    print("\nPre-computing OncoTree descendants...")

    # Tier 2: THYROID, ENCG, BLADDER
    tier2_groups = all_groups[all_groups['tier'] == 'tier2']
    if len(tier2_groups) > 0:
        tier2_descendants = get_descendant_codes(
            DB_PATH,
            tier2_groups['group_code'].tolist()
        )
    else:
        tier2_descendants = {}

    # Tier 3: Level 3 codes
    tier3_groups = all_groups[all_groups['tier'] == 'tier3']
    if len(tier3_groups) > 0:
        tier3_descendants = get_descendant_codes(
            DB_PATH,
            tier3_groups['group_code'].tolist()
        )
    else:
        tier3_descendants = {}

    patient_mappings = []

    for idx, row in all_groups.iterrows():
        tier = row['tier']
        group_code = row['group_code']
        tier_prefixed_code = row['tier_prefixed_code']
        group_name = row['group_name']

        print(f"\nProcessing {tier_prefixed_code} ({group_name})...")

        # Determine which OncoTree codes to query
        if tier == 'tier1':
            # Tier 1: Only the group_code itself (as-is)
            oncotree_codes = [group_code]
        elif tier == 'tier2':
            # Tier 2: Use pre-computed descendants
            oncotree_codes = list(tier2_descendants.get(group_code, {group_code}))
        elif tier == 'tier3':
            # Tier 3: Use pre-computed descendants
            oncotree_codes = list(tier3_descendants.get(group_code, {group_code}))
        else:
            raise ValueError(f"Unknown tier: {tier}")

        print(f"  Querying {len(oncotree_codes)} OncoTree codes")

        # Query database for patients
        placeholders = ','.join(['?' for _ in oncotree_codes])
        query = f"""
        SELECT DISTINCT
            PATIENT_ID,
            ONCOTREE_CODE,
            isOncoKbLevel1
        FROM perPatientOncoKBFilteredForModeling
        WHERE ONCOTREE_CODE IN ({placeholders})
        """

        try:
            patients_df = pd.read_sql_query(query, conn, params=oncotree_codes)

            if len(patients_df) == 0:
                print(f"  WARNING: No patients found for {tier_prefixed_code}")
                continue

            # Add group information
            patients_df['tier'] = tier
            patients_df['group_code'] = group_code
            patients_df['tier_prefixed_code'] = tier_prefixed_code
            patients_df['group_name'] = group_name
            patients_df.rename(columns={
                'ONCOTREE_CODE': 'original_oncotree_code',
                'isOncoKbLevel1': 'is_actionable'
            }, inplace=True)

            n_total = len(patients_df)
            n_actionable = patients_df['is_actionable'].sum()
            actionability_rate = (n_actionable / n_total * 100) if n_total > 0 else 0

            print(f"  Found {n_total} patients ({n_actionable} actionable, {actionability_rate:.1f}%)")

            patient_mappings.append(patients_df)

        except Exception as e:
            print(f"  ERROR querying database for {tier_prefixed_code}: {e}")
            continue

    conn.close()

    if len(patient_mappings) == 0:
        raise ValueError("No patient mappings generated - check database queries")

    # Combine all patient mappings
    all_patient_mappings = pd.concat(patient_mappings, ignore_index=True)

    # Remove duplicates (patient can appear multiple times per group if they have multiple tumors with descendant OncoTree codes)
    all_patient_mappings = all_patient_mappings.drop_duplicates(
        subset=['PATIENT_ID', 'tier_prefixed_code'],
        keep='first'
    )

    print(f"\n" + "="*80)
    print(f"Total patient-group mappings: {len(all_patient_mappings):,}")
    print(f"Unique patients: {all_patient_mappings['PATIENT_ID'].nunique():,}")
    print(f"Unique groups: {all_patient_mappings['tier_prefixed_code'].nunique()}")
    print("="*80)

    return all_patient_mappings


def generate_group_summary_stats(patient_mapping_df):
    """Generate summary statistics per group from patient mapping."""
    print("\n" + "="*80)
    print("STEP 7: Generating Group Summary Statistics")
    print("="*80)

    summary = patient_mapping_df.groupby([
        'tier', 'tier_prefixed_code', 'group_code', 'group_name'
    ]).agg(
        total_patients=('PATIENT_ID', 'count'),
        actionable_patients=('is_actionable', 'sum'),
        unique_oncotree_codes=('original_oncotree_code', 'nunique')
    ).reset_index()

    summary['actionability_rate'] = (
        summary['actionable_patients'] / summary['total_patients'] * 100
    )

    # Sort by tier and actionable_patients descending
    summary = summary.sort_values(['tier', 'actionable_patients'], ascending=[True, False])

    print("\nGroup Summary Statistics:")
    print(f"Total groups: {len(summary)}")
    print(f"\nBy tier:")
    for tier in ['tier1', 'tier2', 'tier3']:
        tier_summary = summary[summary['tier'] == tier]
        print(f"  - {tier}: {len(tier_summary)} groups, "
              f"{tier_summary['total_patients'].sum():,} total patients, "
              f"{tier_summary['actionable_patients'].sum():,} actionable")

    # Check for groups below threshold
    below_threshold = summary[summary['actionable_patients'] < MIN_ACTIONABLE]
    if len(below_threshold) > 0:
        print(f"\nWARNING: {len(below_threshold)} groups below {MIN_ACTIONABLE} actionable threshold:")
        for _, row in below_threshold.iterrows():
            print(f"  - {row['tier_prefixed_code']}: {row['actionable_patients']} actionable")

    return summary


def save_outputs(all_groups, patient_mapping, summary_stats):
    """Save all output files."""
    print("\n" + "="*80)
    print("STEP 8: Saving Output Files")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define output paths
    group_defs_path = OUTPUT_DIR / f"multi_tier_group_definitions_{timestamp}.csv"
    patient_map_path = OUTPUT_DIR / f"patient_to_group_mapping_{timestamp}.csv"
    summary_path = OUTPUT_DIR / f"group_summary_stats_{timestamp}.csv"

    # Also create latest symlinks
    group_defs_latest = OUTPUT_DIR / "multi_tier_group_definitions.csv"
    patient_map_latest = OUTPUT_DIR / "patient_to_group_mapping.csv"
    summary_latest = OUTPUT_DIR / "group_summary_stats.csv"

    # Save files
    all_groups.to_csv(group_defs_path, index=False)
    patient_mapping.to_csv(patient_map_path, index=False)
    summary_stats.to_csv(summary_path, index=False)

    print(f"\nSaved output files:")
    print(f"  - Group definitions: {group_defs_path}")
    print(f"  - Patient mapping: {patient_map_path}")
    print(f"  - Summary stats: {summary_path}")

    # Create latest symlinks
    for src, dst in [
        (group_defs_path, group_defs_latest),
        (patient_map_path, patient_map_latest),
        (summary_path, summary_latest)
    ]:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.name)

    print(f"\nCreated latest symlinks:")
    print(f"  - {group_defs_latest}")
    print(f"  - {patient_map_latest}")
    print(f"  - {summary_latest}")

    return group_defs_path, patient_map_path, summary_path


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("MULTI-TIER CANCER TYPE GROUPING DEFINITION (v3 - Level 1 Tier 3)")
    print("="*80)
    print(f"Minimum actionable threshold: {MIN_ACTIONABLE}")
    print(f"Database: {DB_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    try:
        # Step 1: Load existing grouping files
        b1_df, a2_df = load_existing_grouping_files()

        # Step 2-4: Define three tiers
        tier1 = define_tier1_groups(b1_df)
        tier2 = define_tier2_groups(b1_df)

        # Get tier1 and tier2 codes for exclusion from tier3
        tier1_codes = tier1['group_code'].tolist()
        tier2_codes = tier2['group_code'].tolist()

        tier3 = define_tier3_groups_from_level1(tier1_codes, tier2_codes)

        # Step 5: Combine tiers
        all_groups = combine_tiers(tier1, tier2, tier3)

        # Step 6: Generate patient-to-group mapping (WITH HIERARCHY TRAVERSAL)
        patient_mapping = generate_patient_mapping(all_groups)

        # Step 7: Generate summary statistics
        summary_stats = generate_group_summary_stats(patient_mapping)

        # Step 8: Save outputs
        group_defs_path, patient_map_path, summary_path = save_outputs(
            all_groups, patient_mapping, summary_stats
        )

        print("\n" + "="*80)
        print("MULTI-TIER GROUPING DEFINITION COMPLETE")
        print("="*80)
        print(f"\nTotal groups defined: {len(all_groups)}")
        print(f"  - Tier 1: {len(tier1)} groups")
        print(f"  - Tier 2: {len(tier2)} groups")
        print(f"  - Tier 3: {len(tier3)} groups")
        print(f"\nTotal patient-group mappings: {len(patient_mapping):,}")
        print(f"Unique patients: {patient_mapping['PATIENT_ID'].nunique():,}")
        print(f"\nOutput files saved to: {OUTPUT_DIR}")

        return 0

    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR")
        print("="*80)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
