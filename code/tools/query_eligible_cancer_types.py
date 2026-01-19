#!/usr/bin/env python3
"""
Query Eligible Cancer Types

Queries the harmonized database for cancer types with sufficient actionable events
for machine learning modeling (≥250 actionable events).

Usage:
    python query_eligible_cancer_types.py [--db-path DB_PATH] [--min-actionable MIN]
"""

import pandas as pd
from sqlalchemy import create_engine
import argparse

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path
import sys
from pathlib import Path


def query_eligible_cancer_types(db_path: str, min_actionable: int = 250) -> pd.DataFrame:
    """
    Query database for cancer types with sufficient actionable events.

    Args:
        db_path: Path to SQLite database
        min_actionable: Minimum number of actionable events required

    Returns:
        DataFrame with eligible cancer types and statistics
    """
    engine = create_engine(f"sqlite:///{db_path}")

    # Query to get patient counts and actionable event counts per cancer type
    query = """
    SELECT
        ONCOTREE_CODE,
        CANCER_TYPE_DETAILED,
        COUNT(DISTINCT PATIENT_ID) as n_patients,
        SUM(CASE WHEN isOncoKbLevel1 = 1 THEN 1 ELSE 0 END) as n_actionable,
        CAST(SUM(CASE WHEN isOncoKbLevel1 = 1 THEN 1 ELSE 0 END) AS FLOAT) /
            COUNT(DISTINCT PATIENT_ID) as proportion_actionable
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE, CANCER_TYPE_DETAILED
    HAVING n_actionable >= ?
    ORDER BY n_actionable DESC
    """

    df = pd.read_sql_query(query, engine, params=(min_actionable,))

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Query eligible cancer types from harmonized database"
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default=get_default_db_path(),
        help="Path to SQLite database"
    )

    parser.add_argument(
        "--min-actionable",
        type=int,
        default=250,
        help="Minimum number of actionable events required (default: 250)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="eligible_cancer_types.csv",
        help="Output CSV file path"
    )

    args = parser.parse_args()

    # Check if database exists
    if not Path(args.db_path).exists():
        print(f"Error: Database not found at {args.db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Querying database: {args.db_path}")
    print(f"Minimum actionable events: {args.min_actionable}")

    # Query database
    df = query_eligible_cancer_types(args.db_path, args.min_actionable)

    # Display results
    print(f"\n{'='*70}")
    print(f"Found {len(df)} eligible cancer types")
    print(f"{'='*70}\n")
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to: {args.output}")

    # Summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics:")
    print(f"{'='*70}")
    print(f"Total cancer types: {len(df)}")
    print(f"Total patients: {df['n_patients'].sum():,}")
    print(f"Total actionable events: {df['n_actionable'].sum():,}")
    print(f"Mean actionable rate: {df['proportion_actionable'].mean():.3f}")
    print(f"Median actionable rate: {df['proportion_actionable'].median():.3f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
