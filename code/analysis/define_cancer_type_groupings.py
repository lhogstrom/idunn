#!/usr/bin/env python3
"""Select cancer types with at least 250 actionable patients and write their group definitions."""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from db_config import get_default_db_path  # noqa: E402

DB_PATH = get_default_db_path()
MIN_ACTIONABLE = 250

# Rollup parent codes that are excluded from the eligible set.
EXCLUDED_RESCUE_CODES = ('THYROID', 'ENCG', 'BLADDER')


def query_eligible_codes(db_path):
    """Return OncoTree codes with >=MIN_ACTIONABLE actionable patients."""
    conn = sqlite3.connect(db_path)
    placeholders = ','.join('?' for _ in EXCLUDED_RESCUE_CODES)
    query = f"""
    SELECT
        ONCOTREE_CODE,
        COUNT(DISTINCT PATIENT_ID) AS total_patients,
        COUNT(DISTINCT CASE WHEN isOncoKbLevel1 = 1 THEN PATIENT_ID END)
            AS actionable_patients
    FROM perPatientOncoKBFilteredForModeling
    WHERE ONCOTREE_CODE NOT IN ({placeholders})
    GROUP BY ONCOTREE_CODE
    HAVING actionable_patients >= ?
    ORDER BY actionable_patients DESC
    """
    params = list(EXCLUDED_RESCUE_CODES) + [MIN_ACTIONABLE]
    df = pd.read_sql_query(query, conn, params=params)

    try:
        hierarchy = pd.read_sql_query(
            "SELECT ot_code, ot_name FROM OncoTreeCodeHierarchy", conn
        )
        df = df.merge(
            hierarchy, left_on='ONCOTREE_CODE', right_on='ot_code', how='left'
        ).drop(columns=['ot_code'])
        df['ot_name'] = df['ot_name'].fillna(df['ONCOTREE_CODE'])
    except Exception:
        df['ot_name'] = df['ONCOTREE_CODE']

    conn.close()

    df['actionability_rate'] = (
        df['actionable_patients'] / df['total_patients'] * 100
    ).round(2)
    return df[[
        'ONCOTREE_CODE', 'ot_name', 'total_patients',
        'actionable_patients', 'actionability_rate',
    ]]


def build_patient_mapping(db_path, eligible_codes):
    """One row per patient whose ONCOTREE_CODE is in `eligible_codes`."""
    conn = sqlite3.connect(db_path)
    placeholders = ','.join('?' for _ in eligible_codes)
    query = f"""
    SELECT
        PATIENT_ID,
        ONCOTREE_CODE,
        isOncoKbLevel1 AS is_actionable
    FROM perPatientOncoKBFilteredForModeling
    WHERE ONCOTREE_CODE IN ({placeholders})
    """
    df = pd.read_sql_query(query, conn, params=list(eligible_codes))
    conn.close()
    return df.drop_duplicates(subset=['PATIENT_ID', 'ONCOTREE_CODE'])


def main():
    parser = argparse.ArgumentParser(
        description='Cancer-type grouping definition (OncoTree codes with '
                    f'>={MIN_ACTIONABLE} actionable patients)'
    )
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for the CSV files')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CANCER-TYPE GROUPING DEFINITION")
    print("=" * 80)
    print(f"Database: {DB_PATH}")
    print(f"Output directory: {output_dir}")
    print(f"Minimum actionable threshold: {MIN_ACTIONABLE}")
    print(f"Excluded rescue codes: {', '.join(EXCLUDED_RESCUE_CODES)}")

    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    print("\nQuerying eligible OncoTree codes...")
    group_defs = query_eligible_codes(DB_PATH)
    print(f"  -> {len(group_defs)} eligible codes")

    print("\nBuilding patient-to-group mapping...")
    mapping = build_patient_mapping(
        DB_PATH, group_defs['ONCOTREE_CODE'].tolist()
    )
    print(f"  -> {len(mapping):,} patient rows "
          f"({mapping['PATIENT_ID'].nunique():,} unique patients)")

    group_defs_path = output_dir / "cancer_type_group_definitions.csv"
    mapping_path = output_dir / "patient_to_group_mapping.csv"
    group_defs.to_csv(group_defs_path, index=False)
    mapping.to_csv(mapping_path, index=False)
    print(f"\nWrote: {group_defs_path}")
    print(f"Wrote: {mapping_path}")

    print("\nGroup definitions:")
    print(group_defs.to_string(index=False))

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
