#!/usr/bin/env python3
"""
Minimal utilities for consuming an external rollup-aware run's
patient_to_group_mapping.csv.

After the OT-rollup strip on the publication branch, the publication-branch
pipeline emits a plain 21-row mapping (one row per OncoTree code, no tier
prefix). This module is reduced to just the two helpers that survive:

  * `load_patient_to_group_mapping()` — load a mapping CSV (auto-locates one
    when none is passed).
  * `get_oncotree_codes_for_group()` — expand a group identifier to the
    constituent OncoTree codes recorded in the mapping.

Both are used exclusively by the standalone scripts that read an
**external** rollup-aware run dir (e.g., `visualize_rf_lr_hit_intersection.py`,
`adjust_hits_for_subject_center.py`). External mappings carry the legacy
`tier_prefixed_code` + `original_oncotree_code` schema, so both helpers
preserve that contract verbatim.

Older helpers (group definitions, patient-id lookup, tier-prefix parsing,
DataFrame filtering) were removed along with the rollup-mode plumbing.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def get_multi_tier_groupings_dir(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Locate a `multi_tier_groupings/` directory under runs/.

    External rollup-aware runs continue to use this layout. The publication
    branch's own TIER 2a now writes to `cancer_type_groupings/` instead.
    """
    if repo_root is None:
        repo_root = REPO_ROOT
    repo_root = Path(repo_root)

    default_dir = repo_root / "analysis_output" / "multi_tier_groupings"
    if default_dir.exists():
        return default_dir

    runs_dir = repo_root / "runs"
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.glob("run_*"), reverse=True):
            groupings_dir = run_dir / "tier2_data_setup" / "multi_tier_groupings"
            if groupings_dir.exists():
                return groupings_dir

    return None


def get_patient_to_group_mapping_file(repo_root: Optional[Path] = None) -> Optional[Path]:
    """Return the path to a `patient_to_group_mapping.csv` under an external
    rollup-aware run, or None if none is found.
    """
    groupings_dir = get_multi_tier_groupings_dir(repo_root)
    if groupings_dir is None:
        return None
    mapping_file = groupings_dir / "patient_to_group_mapping.csv"
    return mapping_file if mapping_file.exists() else None


def load_patient_to_group_mapping(
    repo_root: Optional[Path] = None,
    mapping_file: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load a `patient_to_group_mapping.csv` produced by an external
    rollup-aware run.

    Expected columns (from the legacy rollup-aware schema):
      PATIENT_ID, original_oncotree_code, is_actionable, tier,
      group_code, tier_prefixed_code, group_name
    """
    if mapping_file is not None:
        map_file = Path(mapping_file)
    else:
        map_file = get_patient_to_group_mapping_file(repo_root)

    if map_file is None or not map_file.exists():
        raise FileNotFoundError(
            "patient_to_group_mapping.csv not found. Pass mapping_file= "
            "explicitly when consuming an external rollup-aware run."
        )

    if verbose:
        print(f"Loading patient-to-group mapping from: {map_file}")

    df = pd.read_csv(map_file)

    if verbose:
        n_patients = df['PATIENT_ID'].nunique()
        n_groups = (
            df['tier_prefixed_code'].nunique()
            if 'tier_prefixed_code' in df.columns
            else df['ONCOTREE_CODE'].nunique()
        )
        print(f"  Loaded {len(df):,} patient-group assignments")
        print(f"  Unique patients: {n_patients:,}")
        print(f"  Unique groups: {n_groups}")

    return df


def get_oncotree_codes_for_group(
    tier_prefixed_code: str,
    mapping_df: Optional[pd.DataFrame] = None,
    repo_root: Optional[Path] = None,
) -> List[str]:
    """Return the OncoTree codes constituting a group from an external
    rollup-aware mapping.

    For Tier 1 groups, the result is the single OncoTree code (1:1).
    For Tier 2/Tier 3 groups, the result expands to all rolled-up children.

    When the supplied identifier lacks a tier prefix (i.e., a publication-
    branch plain code), the function returns it as-is.
    """
    if mapping_df is None:
        mapping_df = load_patient_to_group_mapping(repo_root, verbose=False)

    if 'tier_prefixed_code' in mapping_df.columns:
        sub = mapping_df[mapping_df['tier_prefixed_code'] == tier_prefixed_code]
        if not sub.empty:
            col = (
                'original_oncotree_code'
                if 'original_oncotree_code' in sub.columns
                else 'ONCOTREE_CODE'
            )
            return sorted(sub[col].unique().tolist())

    return [tier_prefixed_code]
