#!/usr/bin/env python3
"""Count actionable patients per gene within each cancer type.

Identifies the top-N genes per cancer type for gene-stratified testing.
Writes gene_frequency_by_cancer.csv, top<N>_genes_summary.csv,
multi_gene_patient_stats.csv and exploration_report.txt.
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path  # noqa: E402

DB_PATH = get_default_db_path()
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis_output" / "posthoc_statistics" / "gene_stratified" / "gene_distributions"

DEFAULT_TOP_N = 2
DEFAULT_MIN_SAMPLE_SIZE = 50


def load_eligible_codes(groupings_file=None):
    """Return the eligible OncoTree codes from TIER 2a's group-definitions CSV."""
    if groupings_file is None:
        candidates = sorted(
            (REPO_ROOT / 'runs').glob(
                'run_*/tier2_data_setup/cancer_type_groupings/cancer_type_group_definitions.csv'
            ),
            reverse=True,
        )
        groupings_file = candidates[0] if candidates else None

    if groupings_file is None or not Path(groupings_file).exists():
        print("  NOTE: No cancer_type_group_definitions.csv found; using all OncoTree codes")
        return None

    defs = pd.read_csv(groupings_file)
    codes = defs['ONCOTREE_CODE'].tolist()
    print(f"  Using {len(codes)} eligible OncoTree codes from: {groupings_file}")
    return codes


def get_gene_frequencies(db_path, oncotree_codes=None):
    """Count distinct patients per (ONCOTREE_CODE, Hugo_Symbol) for actionable variants."""
    conn = sqlite3.connect(db_path)
    if oncotree_codes:
        placeholders = ','.join('?' for _ in oncotree_codes)
        query = f"""
        SELECT ONCOTREE_CODE, Hugo_Symbol,
               COUNT(DISTINCT PATIENT_ID) AS n_patients
        FROM perPatientOncoKBLevelSorted
        WHERE isOncoKbLevel1 = 1
          AND ONCOTREE_CODE IS NOT NULL
          AND Hugo_Symbol IS NOT NULL
          AND ONCOTREE_CODE IN ({placeholders})
        GROUP BY ONCOTREE_CODE, Hugo_Symbol
        ORDER BY ONCOTREE_CODE, n_patients DESC
        """
        df = pd.read_sql_query(query, conn, params=oncotree_codes)
    else:
        df = pd.read_sql_query(
            """
            SELECT ONCOTREE_CODE, Hugo_Symbol,
                   COUNT(DISTINCT PATIENT_ID) AS n_patients
            FROM perPatientOncoKBLevelSorted
            WHERE isOncoKbLevel1 = 1
              AND ONCOTREE_CODE IS NOT NULL
              AND Hugo_Symbol IS NOT NULL
            GROUP BY ONCOTREE_CODE, Hugo_Symbol
            ORDER BY ONCOTREE_CODE, n_patients DESC
            """,
            conn,
        )
    conn.close()
    df['constituent_codes'] = df['ONCOTREE_CODE']
    print(f"Loaded {len(df):,} gene-cancer combinations across "
          f"{df['ONCOTREE_CODE'].nunique()} cancer types")
    return df


def get_multi_gene_patient_stats(db_path, oncotree_codes=None):
    """Patients-with-multiple-actionable-genes stats per cancer type."""
    conn = sqlite3.connect(db_path)
    if oncotree_codes:
        placeholders = ','.join('?' for _ in oncotree_codes)
        query = f"""
        SELECT ONCOTREE_CODE, PATIENT_ID,
               COUNT(DISTINCT Hugo_Symbol) AS n_actionable_genes
        FROM perPatientOncoKBLevelSorted
        WHERE isOncoKbLevel1 = 1
          AND ONCOTREE_CODE IS NOT NULL
          AND Hugo_Symbol IS NOT NULL
          AND ONCOTREE_CODE IN ({placeholders})
        GROUP BY ONCOTREE_CODE, PATIENT_ID
        """
        df = pd.read_sql_query(query, conn, params=oncotree_codes)
    else:
        df = pd.read_sql_query(
            """
            SELECT ONCOTREE_CODE, PATIENT_ID,
                   COUNT(DISTINCT Hugo_Symbol) AS n_actionable_genes
            FROM perPatientOncoKBLevelSorted
            WHERE isOncoKbLevel1 = 1
              AND ONCOTREE_CODE IS NOT NULL
              AND Hugo_Symbol IS NOT NULL
            GROUP BY ONCOTREE_CODE, PATIENT_ID
            """,
            conn,
        )
    conn.close()

    stats = df.groupby('ONCOTREE_CODE').agg(
        total_patients=('PATIENT_ID', 'count'),
        mean_genes_per_patient=('n_actionable_genes', 'mean'),
        max_genes_per_patient=('n_actionable_genes', 'max'),
    ).reset_index()

    multi_counts = df[df['n_actionable_genes'] > 1].groupby('ONCOTREE_CODE').size()
    stats['patients_with_multiple_genes'] = (
        stats['ONCOTREE_CODE'].map(multi_counts).fillna(0).astype(int)
    )
    stats['pct_multi_gene'] = (
        stats['patients_with_multiple_genes'] / stats['total_patients'] * 100
    )
    return stats


def identify_top_n_genes(freq_df, n=DEFAULT_TOP_N):
    freq_df = freq_df.copy()
    freq_df['rank'] = freq_df.groupby('ONCOTREE_CODE')['n_patients'].rank(
        method='first', ascending=False
    )
    top = freq_df[freq_df['rank'] <= n].copy()

    totals = freq_df.groupby('ONCOTREE_CODE')['n_patients'].sum().reset_index()
    totals.columns = ['ONCOTREE_CODE', 'total_gene_patient_events']
    top = top.merge(totals, on='ONCOTREE_CODE')
    top['pct_of_events'] = top['n_patients'] / top['total_gene_patient_events'] * 100
    return top.sort_values(['ONCOTREE_CODE', 'rank'])


def check_sample_size(top_df, min_n=DEFAULT_MIN_SAMPLE_SIZE):
    top_df = top_df.copy()
    top_df['meets_min_n'] = top_df['n_patients'] >= min_n
    return top_df


def write_report(freq_df, top_df, multi_stats, out_file, top_n, min_n):
    with open(out_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Gene Distribution Exploration Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Top N genes: {top_n}\n")
        f.write(f"Minimum sample size: {min_n}\n\n")

        f.write(f"Total cancer types: {freq_df['ONCOTREE_CODE'].nunique()}\n")
        f.write(f"Unique actionable genes: {freq_df['Hugo_Symbol'].nunique()}\n")
        f.write(f"Gene-cancer combinations: {len(freq_df):,}\n")

        total_multi = multi_stats['patients_with_multiple_genes'].sum()
        total_patients = multi_stats['total_patients'].sum()
        pct = (total_multi / total_patients * 100) if total_patients else 0
        f.write(
            f"\nPatients with >1 actionable gene: {total_multi:,} ({pct:.1f}%)\n"
        )

        f.write(f"\nTop {top_n} genes per cancer type:\n")
        for cancer_type in sorted(top_df['ONCOTREE_CODE'].unique()):
            ct = top_df[top_df['ONCOTREE_CODE'] == cancer_type]
            f.write(f"\n{cancer_type}:\n")
            for _, row in ct.iterrows():
                status = "OK" if row['meets_min_n'] else "LOW N"
                f.write(
                    f"  #{int(row['rank'])}: {row['Hugo_Symbol']:12s} "
                    f"n={row['n_patients']:5d} ({row['pct_of_events']:5.1f}%) [{status}]\n"
                )
    print(f"Exploration report saved to: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Explore gene distributions for gene-stratified post-hoc analysis',
    )
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--top-n', type=int, default=DEFAULT_TOP_N,
                        help=f'Top N genes per cancer type (default: {DEFAULT_TOP_N})')
    parser.add_argument('--min-n', type=int, default=DEFAULT_MIN_SAMPLE_SIZE,
                        help=f'Minimum sample size (default: {DEFAULT_MIN_SAMPLE_SIZE})')
    parser.add_argument('--groupings-file', type=str, default=None,
                        help='Path to cancer_type_group_definitions.csv '
                             '(auto-detected from runs/ if omitted)')
    parser.add_argument('--tier', type=str, default=None,
                        help=argparse.SUPPRESS)  # legacy flag accepted but ignored

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Gene Distribution Exploration")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Top N genes: {args.top_n}")
    print(f"Minimum sample size: {args.min_n}")

    oncotree_codes = load_eligible_codes(args.groupings_file)

    freq_df = get_gene_frequencies(DB_PATH, oncotree_codes)
    multi_stats = get_multi_gene_patient_stats(DB_PATH, oncotree_codes)
    top_df = identify_top_n_genes(freq_df, n=args.top_n)
    top_df = check_sample_size(top_df, min_n=args.min_n)

    freq_df.to_csv(output_dir / "gene_frequency_by_cancer.csv", index=False)
    top_df.to_csv(output_dir / f"top{args.top_n}_genes_summary.csv", index=False)
    multi_stats.to_csv(output_dir / "multi_gene_patient_stats.csv", index=False)
    write_report(freq_df, top_df, multi_stats,
                 output_dir / "exploration_report.txt",
                 args.top_n, args.min_n)

    print("\nSUMMARY")
    print(f"  Cancer types analyzed:    {top_df['ONCOTREE_CODE'].nunique()}")
    print(f"  Top-{args.top_n} combinations: {len(top_df)}")
    print(f"  Meets n>={args.min_n}:     {int(top_df['meets_min_n'].sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
