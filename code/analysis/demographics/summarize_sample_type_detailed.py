"""Summarize SAMPLE_TYPE_DETAILED counts and percentages across cancer types.

Uses the canonical 21-row cancer-type grouping from TIER 2a
(`cancer_type_group_definitions.csv`). One row per patient.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path  # noqa: E402


REPO_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_OUTDIR = REPO_ROOT / 'runs/sample_type_detailed_summary'


def _auto_groupings_file():
    candidates = sorted(
        (REPO_ROOT / 'runs').glob(
            'run_*/tier2_data_setup/cancer_type_groupings/cancer_type_group_definitions.csv'
        ),
        reverse=True,
    )
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groupings-file', default=None,
                    help='Path to cancer_type_group_definitions.csv '
                         '(auto-detects the latest run if omitted)')
    ap.add_argument('--output-dir', default=str(DEFAULT_OUTDIR))
    ap.add_argument('--db-path', default=None)
    args = ap.parse_args()

    db = args.db_path or get_default_db_path()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    groupings_file = args.groupings_file or _auto_groupings_file()
    if groupings_file is None or not Path(groupings_file).exists():
        raise FileNotFoundError(
            "cancer_type_group_definitions.csv not found. "
            "Run TIER 2a or pass --groupings-file."
        )
    eligible_codes = set(pd.read_csv(groupings_file)['ONCOTREE_CODE'].tolist())

    conn = sqlite3.connect(db)
    pat = pd.read_sql(
        "SELECT DISTINCT PATIENT_ID, ONCOTREE_CODE, SAMPLE_TYPE_DETAILED "
        "FROM perPatientOncoKBFilteredForModeling",
        conn,
    )
    conn.close()
    pat['SAMPLE_TYPE_DETAILED'] = pat['SAMPLE_TYPE_DETAILED'].fillna('Missing')
    merged = pat[pat['ONCOTREE_CODE'].isin(eligible_codes)].copy()

    counts = (
        merged.groupby(['ONCOTREE_CODE', 'SAMPLE_TYPE_DETAILED'])
        .size()
        .rename('n')
        .reset_index()
    )
    totals = merged.groupby('ONCOTREE_CODE').size().rename('total').reset_index()
    counts = counts.merge(totals, on='ONCOTREE_CODE')
    counts['pct'] = (counts['n'] / counts['total'] * 100).round(2)

    n_wide = counts.pivot(index='ONCOTREE_CODE', columns='SAMPLE_TYPE_DETAILED',
                          values='n').fillna(0).astype(int)
    pct_wide = counts.pivot(index='ONCOTREE_CODE', columns='SAMPLE_TYPE_DETAILED',
                            values='pct').fillna(0.0)
    n_wide['Total'] = n_wide.sum(axis=1)

    combined = n_wide.copy().astype(object)
    for col in pct_wide.columns:
        combined[col] = n_wide[col].astype(int).astype(str) + ' (' + \
                        pct_wide[col].map(lambda x: f"{x:.1f}").astype(str) + '%)'
    combined = combined[list(pct_wide.columns) + ['Total']]
    combined = combined.sort_index()

    # Overall row across all eligible patients
    overall_n = merged['SAMPLE_TYPE_DETAILED'].value_counts()
    overall_total = len(merged)
    overall_row = {}
    for col in pct_wide.columns:
        n = int(overall_n.get(col, 0))
        pct = n / overall_total * 100 if overall_total else 0.0
        overall_row[col] = f"{n} ({pct:.1f}%)"
    overall_row['Total'] = overall_total
    combined.loc['OVERALL_unique_patients'] = overall_row

    counts.to_csv(out / 'sample_type_detailed_long.csv', index=False)
    n_wide.to_csv(out / 'sample_type_detailed_counts_wide.csv')
    pct_wide.round(2).to_csv(out / 'sample_type_detailed_percent_wide.csv')
    combined.to_csv(out / 'sample_type_detailed_combined.csv')

    with open(out / 'sample_type_detailed_combined.md', 'w') as f:
        f.write("# SAMPLE_TYPE_DETAILED by cancer type\n\n")
        f.write(f"Source DB: `{db}`\n\n")
        f.write(f"Groupings: `{groupings_file}`\n\n")
        try:
            f.write(combined.to_markdown())
        except ImportError:
            f.write("```\n" + combined.to_string() + "\n```")
        f.write("\n")

    print(f"Wrote outputs to {out}")
    print(combined.to_string())


if __name__ == '__main__':
    main()
