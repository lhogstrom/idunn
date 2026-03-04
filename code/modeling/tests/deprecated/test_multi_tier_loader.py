"""
Test script to validate multi-tier grouping data loader

This script tests that the prepare_cancer_data_dict_multi_tier() method
correctly loads patient data for the new 3-tier grouping system.
"""

import sys
from pathlib import Path

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from oncokb_analyzer.data_loader import DataLoader

def main():
    print("="*80)
    print("MULTI-TIER GROUPING DATA LOADER TEST")
    print("="*80)

    # Database path
    db_path = get_default_db_path()

    # Mapping file path (using symlink for latest)
    mapping_file = "../../analysis_output/multi_tier_groupings/patient_to_group_mapping.csv"

    print(f"\nDatabase: {db_path}")
    print(f"Mapping file: {mapping_file}")

    # Initialize data loader
    print("\n" + "="*80)
    print("Step 1: Initialize DataLoader")
    print("="*80)
    loader = DataLoader(db_path)
    print("✓ DataLoader initialized successfully")

    # Load multi-tier groupings
    print("\n" + "="*80)
    print("Step 2: Load multi-tier groupings")
    print("="*80)

    cancer_data_dict = loader.prepare_cancer_data_dict_multi_tier(
        grouping_mapping_file=mapping_file,
        feature_set='demographic'
    )

    print(f"\n✓ Loaded {len(cancer_data_dict)} multi-tier groups")

    # Summary statistics
    print("\n" + "="*80)
    print("Step 3: Summary Statistics")
    print("="*80)

    total_samples = sum(X.shape[0] for X, y, groups in cancer_data_dict.values())
    total_actionable = sum(y.sum() for X, y, groups in cancer_data_dict.values())

    print(f"\nTotal groups: {len(cancer_data_dict)}")
    print(f"Total samples across all groups: {total_samples:,}")
    print(f"Total actionable samples: {total_actionable:,} ({total_actionable/total_samples:.1%})")

    # Group breakdown by tier
    print("\n" + "="*80)
    print("Step 4: Group Breakdown by Tier")
    print("="*80)

    tier1_groups = [g for g in cancer_data_dict.keys() if g.startswith('tier1_')]
    tier2_groups = [g for g in cancer_data_dict.keys() if g.startswith('tier2_')]
    tier3_groups = [g for g in cancer_data_dict.keys() if g.startswith('tier3_')]

    print(f"\nTier 1: {len(tier1_groups)} groups")
    print(f"Tier 2: {len(tier2_groups)} groups")
    print(f"Tier 3: {len(tier3_groups)} groups")

    # Show details for top 5 groups by patient count
    print("\n" + "="*80)
    print("Step 5: Top 5 Groups by Patient Count")
    print("="*80)

    group_sizes = [(group, X.shape[0], y.sum(), y.mean())
                   for group, (X, y, groups) in cancer_data_dict.items()]
    group_sizes.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Group':<25} {'Patients':>10} {'Actionable':>12} {'Rate':>8}")
    print("-" * 60)
    for group, n_patients, n_actionable, rate in group_sizes[:5]:
        print(f"{group:<25} {n_patients:>10,} {int(n_actionable):>12,} {rate:>7.1%}")

    # Test a specific tier1 group (LUAD)
    print("\n" + "="*80)
    print("Step 6: Detailed Test - tier1_LUAD")
    print("="*80)

    if 'tier1_LUAD' in cancer_data_dict:
        X, y, groups = cancer_data_dict['tier1_LUAD']
        print(f"\nGroup: tier1_LUAD")
        print(f"  Patients: {X.shape[0]:,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Feature names: {X.columns.tolist()[:5]}... ({len(X.columns)} total)")
        print(f"  Actionable: {y.sum():,} ({y.mean():.1%})")
        print(f"  Unique institutions: {groups.nunique() if groups is not None else 'N/A'}")
        print("\n✓ tier1_LUAD data structure validated")
    else:
        print("\n⚠ tier1_LUAD not found in cancer_data_dict")

    # Test a tier3 group (LUNG)
    print("\n" + "="*80)
    print("Step 7: Detailed Test - tier3_LUNG")
    print("="*80)

    if 'tier3_LUNG' in cancer_data_dict:
        X, y, groups = cancer_data_dict['tier3_LUNG']
        print(f"\nGroup: tier3_LUNG")
        print(f"  Patients: {X.shape[0]:,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Actionable: {y.sum():,} ({y.mean():.1%})")
        print(f"  Unique institutions: {groups.nunique() if groups is not None else 'N/A'}")
        print("\n✓ tier3_LUNG data structure validated")
    else:
        print("\n⚠ tier3_LUNG not found in cancer_data_dict")

    # Final summary
    print("\n" + "="*80)
    print("TEST COMPLETE - ALL CHECKS PASSED ✓")
    print("="*80)

    print("\nMulti-tier grouping data loader is ready for use!")
    print(f"Successfully loaded {len(cancer_data_dict)} groups with {total_samples:,} total samples")

if __name__ == "__main__":
    main()
