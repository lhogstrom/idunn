#!/usr/bin/env python3
"""Build the supplement table that defines every model feature and its values in the analysed cohort.

A read-only emit (one read of the modeling table, restricted to the run's cohort codes).
Rows follow demographic / specimen / technical groups plus the outcome; detailed sample
type is listed under specimen to match how the Results count it. The feature list, the
fields carrying GENIE's `Unknown` sentinel as "not recorded", and all footnote numbers
are derived from the config and data, not hard-coded. Fails closed if the row or
actionable counts differ from the run's cancer_type_group_definitions.csv, if
(PATIENT_ID, ONCOTREE_CODE) is not unique, or if a raw level has no print label.

Usage:
  BALDER_DB_PATH=<patched db> python build_feature_table.py \
      --groupings-file <cancer_type_group_definitions.csv> --output-dir <dir>

Outputs: supp_table_feature_definitions.csv (one row per feature), ..._levels.csv
(feature x level with n and %), and ..._definitions.md (table, caption, footnotes).
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'code' / 'tools'))
sys.path.insert(0, str(REPO_ROOT / 'code' / 'modeling'))

from feature_config import get_feature_list, get_value_display_name  # noqa: E402
from oncokb_analyzer.missingness_indicator import DEFAULT_MISSING_SENTINEL  # noqa: E402
from oncokb_analyzer.preprocessing import MISSINGNESS_INDICATOR_COLS  # noqa: E402

DEFAULT_GROUPINGS = (REPO_ROOT / 'runs/run_20260918_125112_v1lock/tier2_data_setup'
                     '/cancer_type_groupings/cancer_type_group_definitions.csv')
DEFAULT_OUT = REPO_ROOT / 'runs/supplement_tables_20260918_v1lock'
STEM = 'supp_table_feature_definitions'

# Panels above this many genes are whole-exome / whole-genome assays (footnote only).
EXOME_GENE_FLOOR = 1000

# Print labels for raw DB levels. A raw level missing from a map aborts the build.
SAMPLE_TYPE_LABELS = {
    'Primary tumor': 'Primary tumor',
    'Metastasis site unspecified': 'Metastasis, site unspecified',
    'Not Collected': 'Not collected',
    'Distant organ metastasis': 'Distant organ metastasis',
    'Not otherwise specified': 'Not otherwise specified',
    'Lymph node metastasis': 'Lymph node metastasis',
    'Not applicable or hematologic malignancy': 'Not applicable (hematologic malignancy)',
    'Local recurrence': 'Local recurrence',
}
PRESERVATION_LABELS = {
    'FFPE': 'FFPE',
    'fresh_frozen': 'Fresh-frozen',
    'FFPE;fresh_frozen': 'FFPE or fresh-frozen',
}
CALLING_LABELS = {
    'tumor_only': 'Tumor-only',
    'tumor_normal': 'Tumor–matched normal',
}

# Row spec, in print order. `col` is the DB column; ANCESTRY is stored as RACE.
# kind: numeric | categorical | indicator (derived from `parent`).
SPEC = [
    ('AGE_AT_SEQ_REPORT', 'Demographic', 'Age at sequencing report', 'numeric',
     dict(col='AGE_AT_SEQ_REPORT', unit='years', footnote='a')),
    ('SEX', 'Demographic', 'Sex', 'categorical', dict(col='SEX')),
    ('ANCESTRY', 'Demographic', 'Self-reported ancestry', 'categorical',
     dict(col='RACE', footnote='b')),
    ('ANCESTRY_missing', 'Demographic', 'Ancestry recorded', 'indicator',
     dict(parent='ANCESTRY')),
    ('ETHNICITY', 'Demographic', 'Ethnicity', 'categorical',
     dict(col='ETHNICITY', labels='config')),
    ('ETHNICITY_missing', 'Demographic', 'Ethnicity recorded', 'indicator',
     dict(parent='ETHNICITY')),
    ('SAMPLE_TYPE_DETAILED', 'Specimen', 'Sample type (detailed)', 'categorical',
     dict(col='SAMPLE_TYPE_DETAILED', labels=SAMPLE_TYPE_LABELS)),
    ('SUBJECT_CENTER', 'Technical', 'Contributing institution', 'categorical',
     dict(col='SUBJECT_CENTER', footnote='d')),
    ('number_of_genes', 'Technical', 'Panel gene count', 'numeric',
     dict(col='number_of_genes', unit='genes', footnote='c,e')),
    ('preservation_technique', 'Technical', 'Preservation technique', 'categorical',
     dict(col='preservation_technique', labels=PRESERVATION_LABELS, footnote='c')),
    ('calling_strategy', 'Technical', 'Variant-calling strategy', 'categorical',
     dict(col='calling_strategy', labels=CALLING_LABELS, footnote='c')),
]
DB_COL = {k: o['col'] for k, _, _, kind, o in SPEC if kind != 'indicator'}


def fmt_pct(n: int, d: int) -> str:
    p = 100 * n / d
    return '<0.1%' if 0 < p < 0.05 else f'{p:.1f}%'


def fmt_n(n: int, d: int) -> str:
    return f'{n:,} ({fmt_pct(n, d)})'


def fmt_num(x: float) -> str:
    return f'{x:,.0f}' if float(x).is_integer() else f'{x:,.1f}'


def load_cohort(db: Path, codes: list) -> pd.DataFrame:
    cols = sorted(set(DB_COL.values()) | {'PATIENT_ID', 'ONCOTREE_CODE', 'isOncoKbLevel1'})
    q = (f"SELECT {', '.join(cols)} FROM perPatientOncoKBFilteredForModeling "
         f"WHERE ONCOTREE_CODE IN ({','.join('?' * len(codes))})")
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as con:
        return pd.read_sql(q, con, params=codes)


def is_missing(feature: str, s: pd.Series) -> pd.Series:
    """Not recorded = NULL, plus the GENIE sentinel for the indicator-paired fields only."""
    miss = s.isna()
    if feature in MISSINGNESS_INDICATOR_COLS:
        miss |= s.astype(str).eq(DEFAULT_MISSING_SENTINEL)
    return miss


def label_for(feature: str, raw: str, labels) -> str:
    if labels is None:
        return raw
    if labels == 'config':
        return get_value_display_name(raw)
    if raw not in labels:
        raise SystemExit(f'ERROR: {feature} level {raw!r} has no print label. Add it to the '
                         f'label map in {Path(__file__).name} rather than shipping a raw code.')
    return labels[raw]


def build(df: pd.DataFrame):
    N = len(df)
    rows, levels = [], []

    def add_level(feature, level, n):
        levels.append({'feature': feature, 'level': level, 'n': int(n),
                       'pct': round(100 * n / N, 3)})

    y = df['isOncoKbLevel1'].astype(int)
    rows.append({'group': 'Outcome', 'feature': 'Actionable mutation (OncoKB Level 1)',
                 'model_column': 'isOncoKbLevel1', 'type': 'Binary',
                 'values': f'Present {fmt_n(y.sum(), N)}; absent {fmt_n(N - y.sum(), N)}',
                 'footnote': ''})
    add_level('isOncoKbLevel1', 'Present', y.sum())
    add_level('isOncoKbLevel1', 'Absent', N - y.sum())

    for key, group, name, kind, o in SPEC:
        if kind == 'indicator':
            parent = o['parent']
            miss = is_missing(parent, df[DB_COL[parent]])
            values = f'Recorded {fmt_n((~miss).sum(), N)}; not recorded {fmt_n(miss.sum(), N)}'
            ftype = 'Binary indicator'
            add_level(key, 'Recorded', (~miss).sum())
            add_level(key, 'Not recorded', miss.sum())
        elif kind == 'numeric':
            s = pd.to_numeric(df[o['col']], errors='coerce')
            miss = s.isna()
            q1, med, q3 = s.quantile([.25, .5, .75])
            values = (f'Median {fmt_num(med)} (IQR {fmt_num(q1)}–{fmt_num(q3)}); '
                      f'range {fmt_num(s.min())}–{fmt_num(s.max())}')
            if miss.any():
                values += f'; not recorded {fmt_n(miss.sum(), N)}'
            ftype = f'Numeric ({o["unit"]})'
            for stat, v in [('median', med), ('q1', q1), ('q3', q3),
                            ('min', s.min()), ('max', s.max())]:
                levels.append({'feature': key, 'level': stat, 'n': np.nan, 'pct': np.nan,
                               'value': float(v)})
            add_level(key, 'Not recorded', miss.sum())
        else:
            s = df[o['col']]
            miss = is_missing(key, s)
            vc = s[~miss].astype(str).value_counts()
            parts = []
            for raw, n in vc.items():
                lab = label_for(key, raw, o.get('labels'))
                parts.append(f'{lab} {fmt_n(n, N)}')
                add_level(key, lab, n)
            if miss.any():
                parts.append(f'not recorded {fmt_n(miss.sum(), N)}')
                add_level(key, 'Not recorded', miss.sum())
            values = '; '.join(parts)
            ftype = f'Categorical, {len(vc)} levels'
        rows.append({'group': group, 'feature': name, 'model_column': key, 'type': ftype,
                     'values': values, 'footnote': o.get('footnote', '')})

    return pd.DataFrame(rows), pd.DataFrame(levels)


def footnotes(df: pd.DataFrame) -> dict:
    age = pd.to_numeric(df['AGE_AT_SEQ_REPORT'], errors='coerce')
    genes = pd.to_numeric(df['number_of_genes'], errors='coerce')
    big = genes[genes > EXOME_GENE_FLOOR]

    assay_cols = ['number_of_genes', 'preservation_technique', 'calling_strategy']
    null_sets = {c: frozenset(df.index[df[c].isna()]) for c in assay_cols}
    shared = len(set(null_sets.values())) == 1
    n_assay_missing = len(next(iter(null_sets.values()))) if shared else None

    unrec = []
    for center, g in df.groupby('SUBJECT_CENTER'):
        if all(is_missing(f, g[DB_COL[f]]).all() for f in MISSINGNESS_INDICATOR_COLS):
            unrec.append(center)

    return {
        'a': (f'GENIE reports ages below 18 and above 89 only as `<18` and `>89`; these were '
              f'coded as 17 (n = {(age == 17).sum():,}) and 90 (n = {(age == 90).sum():,}).'),
        'b': ('GENIE field `PRIMARY_RACE`, recoded (White → European, Black → African). '
              '`Unknown` is counted as not recorded; see the paired indicator row.'),
        'c': ('From GENIE assay metadata (`assay_information.txt`, joined on the sequencing '
              'assay), so these describe the assay rather than the individual specimen.'
              + (f' The same {n_assay_missing:,} records lack all three assay fields.'
                 if shared and n_assay_missing else '')),
        'd': (f'{", ".join(sorted(unrec))} record no ancestry or ethnicity for any patient '
              f'in the cohort.'
              if unrec else ''),
        'e': (f'Values above {EXOME_GENE_FLOOR:,} genes are whole-exome or whole-genome assays '
              f'({", ".join(f"{int(v):,}" for v in sorted(big.unique()))} genes; '
              f'n = {len(big):,}).' if len(big) else ''),
    }


def write_md(table: pd.DataFrame, notes: dict, N: int, n_codes: int, path: Path) -> None:
    lines = [
        '**Supplementary Table. Model features and their recorded values.** '
        f'Counts are patient × cancer-type records in the analysed cohort (N = {N:,}; '
        f'{n_codes} OncoTree codes); percentages use N as the denominator. Every cancer-type '
        'model used all features listed. Categorical features entered the models as indicator '
        'columns against a reference level.', '',
        '| Group | Feature | Type | Values, n (%) |', '|---|---|---|---|',
    ]
    last = None
    for r in table.itertuples():
        grp = f'**{r.group}**' if r.group != last else ''
        last = r.group
        sup = f'<sup>{r.footnote}</sup>' if r.footnote else ''
        lines.append(f'| {grp} | {r.feature}{sup} | {r.type} | {r.values} |')
    lines.append('')
    lines += [f'<sup>{k}</sup> {v}  ' for k, v in notes.items() if v]
    path.write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--db-path', type=Path, default=os.environ.get('BALDER_DB_PATH'),
                   help='Patient-unit, age-patched DB. Defaults to $BALDER_DB_PATH; there is '
                        'deliberately no further fallback (it resolves to the unpatched DB).')
    p.add_argument('--groupings-file', type=Path, default=DEFAULT_GROUPINGS,
                   help="The run's cancer_type_group_definitions.csv (the cohort authority).")
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    a = p.parse_args()

    if not a.db_path:
        raise SystemExit('ERROR: pass --db-path or set BALDER_DB_PATH.')
    for f in (a.db_path, a.groupings_file):
        if not Path(f).exists():
            raise SystemExit(f'ERROR: not found: {f}')

    configured = set(get_feature_list('both'))
    specced = {k for k, *_ in SPEC}
    if configured != specced:
        raise SystemExit('ERROR: features.yaml `both` and this table disagree.\n'
                         f'  in config only: {sorted(configured - specced)}\n'
                         f'  in table only:  {sorted(specced - configured)}')

    groups = pd.read_csv(a.groupings_file)
    codes = groups['ONCOTREE_CODE'].tolist()
    df = load_cohort(Path(a.db_path), codes)

    exp_n, exp_act = int(groups['total_patients'].sum()), int(groups['actionable_patients'].sum())
    got_n, got_act = len(df), int(df['isOncoKbLevel1'].astype(int).sum())
    if (got_n, got_act) != (exp_n, exp_act):
        raise SystemExit(f'ERROR: cohort mismatch -- DB gives {got_n:,} records / {got_act:,} '
                         f'actionable, groupings file {exp_n:,} / {exp_act:,}. Wrong DB '
                         f'(sample-unit or unpatched?) or wrong groupings file.')
    if df.duplicated(['PATIENT_ID', 'ONCOTREE_CODE']).any():
        raise SystemExit('ERROR: duplicate (PATIENT_ID, ONCOTREE_CODE) rows -- not a '
                         'patient-unit modeling table.')

    table, levels = build(df)
    notes = footnotes(df)

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f'{STEM}.csv', index=False)
    levels.to_csv(out / f'{STEM}_levels.csv', index=False)
    write_md(table, notes, got_n, len(codes), out / f'{STEM}.md')

    print(f'Cohort: {got_n:,} records, {got_act:,} actionable, {len(codes)} codes '
          f'(matches {a.groupings_file.name})')
    print(f'Features: {len(specced)} from features.yaml + outcome -> {len(table)} rows')
    print(f'Sentinel-as-missing fields: {MISSINGNESS_INDICATOR_COLS} '
          f'(sentinel {DEFAULT_MISSING_SENTINEL!r})')
    for suffix in ('.csv', '_levels.csv', '.md'):
        print(f'  wrote {out / (STEM + suffix)}')


if __name__ == '__main__':
    main()
