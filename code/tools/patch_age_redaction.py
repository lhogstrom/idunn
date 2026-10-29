#!/usr/bin/env python3
"""Restore GENIE's redacted ages into a copy of the harmonized DB.

GENIE de-identifies age as the strings "<18" / ">89", which a numeric coercion
turns into NULL. This script rewrites AGE_AT_SEQ_REPORT on a copy of the DB from
the raw GENIE clinical file, touching only redacted samples (cohort membership
and the outcome are unchanged).

Variants: upper (>89 -> 90, <18 -> 17); conservative (>89 -> 90, <18 -> NULL).

Usage:
    python code/tools/patch_age_redaction.py --clinical-sample data_clinical_sample.txt \
        --variant upper --out-db <patched>.sqlite
    python code/tools/patch_age_redaction.py --clinical-sample ... --dry-run
"""

import argparse
import csv
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db_config import get_default_db_path

# GENIE redaction tokens and the value each variant substitutes (None = leave NULL).
VARIANTS = {
    'upper':        {'>89': 90.0, '<18': 17.0},
    'conservative': {'>89': 90.0, '<18': None},
}

# Every table carrying AGE_AT_SEQ_REPORT.
MANUSCRIPT_TABLES = [
    'perPatientOncoKBFilteredForModeling',
    'perPatientOncoKBLevelSorted',
]
OTHER_TABLES = [
    'harmonizedSampleInfo',
    'perPatientLevel1',
    'perPatientLevel2orLevel1',
    'patientObservedVariantTableWithSampleInfoOncokb',
    'exhaustiveClinicalAnnotatedPatientVariants',
    'representativeClinicalAnnotatedPatientVariants',
    'representativeClinicalAnnotatedPatientVariantsWithOncoKB',
]

CLINICAL_SAMPLE_HEADER_ROWS = 4  # GENIE ships 4 "#" comment lines before the header


def load_raw_ages(path):
    """SAMPLE_ID -> raw AGE_AT_SEQ_REPORT string, straight from the GENIE release."""
    ages = {}
    with open(path, newline='') as fh:
        for _ in range(CLINICAL_SAMPLE_HEADER_ROWS):
            next(fh)
        reader = csv.DictReader(fh, delimiter='\t')
        for row in reader:
            ages[row['SAMPLE_ID']] = row['AGE_AT_SEQ_REPORT']
    return ages


def key_column(conn, table):
    """Tumor_Sample_Barcode where present, else SAMPLE_ID."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info('{table}')")]
    for candidate in ('Tumor_Sample_Barcode', 'SAMPLE_ID'):
        if candidate in cols:
            return candidate
    raise SystemExit(f"FATAL: {table} has no Tumor_Sample_Barcode or SAMPLE_ID column")


def preflight(conn, raw_ages, tables):
    """Fail closed unless the release matches the DB.

    Every numeric age in the DB must equal the raw file, and every barcode must resolve.
    """
    print("== pre-flight ==")
    ok = True
    for table in tables:
        key = key_column(conn, table)
        rows = conn.execute(
            f"SELECT \"{key}\", AGE_AT_SEQ_REPORT FROM \"{table}\""
        ).fetchall()

        absent = compared = disagreed = 0
        for bc, age in rows:
            raw = raw_ages.get(bc)
            if raw is None:
                # non-GENIE sources legitimately are not in the GENIE release
                absent += 1
                continue
            if age is not None and _is_number(raw):
                compared += 1
                if abs(float(raw) - float(age)) > 1e-9:
                    disagreed += 1

        flag = ''
        if disagreed:
            flag = '  <-- RELEASE MISMATCH'
            ok = False
        print(f"   {table:56s} rows={len(rows):8d} compared={compared:8d} "
              f"disagree={disagreed:6d} not-in-release={absent:7d}{flag}")

    if not ok:
        raise SystemExit(
            "\nFATAL: numeric ages disagree with the GENIE release.\n"
            "This is not the release that built the DB -- refusing to patch."
        )
    print("   -> release concordance exact; safe to patch\n")


def _is_number(s):
    return s.replace('.', '', 1).lstrip('-').isdigit()


def build_recode_map(raw_ages, variant):
    """barcode -> new age, for samples whose raw value is a redaction token only."""
    subs = VARIANTS[variant]
    return {bc: subs[raw] for bc, raw in raw_ages.items() if raw in subs}


def apply_patch(conn, recode, tables):
    """One set-based UPDATE per table joined to a TEMP recode table (tables are unindexed, so per-barcode loops would be slow)."""
    print("== applying patch ==")
    conn.execute("CREATE TEMP TABLE _age_recode (key TEXT PRIMARY KEY, new_age REAL)")
    conn.executemany(
        "INSERT OR REPLACE INTO _age_recode VALUES (?, ?)",
        [(bc, age) for bc, age in recode.items() if age is not None],
    )
    n_map, = conn.execute("SELECT COUNT(*) FROM _age_recode").fetchone()
    print(f"   recode table: {n_map} barcodes with a value to write")

    total = 0
    for table in tables:
        key = key_column(conn, table)
        cur = conn.execute(f'''
            UPDATE "{table}"
               SET AGE_AT_SEQ_REPORT = (
                   SELECT new_age FROM _age_recode WHERE key = "{table}"."{key}")
             WHERE AGE_AT_SEQ_REPORT IS NULL
               AND "{key}" IN (SELECT key FROM _age_recode)
        ''')
        changed = cur.rowcount
        conn.commit()
        total += changed
        print(f"   {table:56s} rows updated: {changed}")
    print(f"   -> {total} row-updates total\n")
    return total


def verify(conn, source_db, tables):
    """Post-patch invariants vs. the source DB: cohort membership and outcome must not move."""
    print("== verification (patched vs source) ==")
    conn.execute("ATTACH DATABASE ? AS src", (str(source_db),))
    ok = True

    for table in tables:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info('{table}')")]
        n_new, = conn.execute(f'SELECT COUNT(*) FROM main."{table}"').fetchone()
        n_old, = conn.execute(f'SELECT COUNT(*) FROM src."{table}"').fetchone()

        problems = []
        if n_new != n_old:
            problems.append(f"row count {n_old} -> {n_new}")

        if 'isOncoKbLevel1' in cols:
            o_new, = conn.execute(
                f'SELECT SUM(isOncoKbLevel1) FROM main."{table}"').fetchone()
            o_old, = conn.execute(
                f'SELECT SUM(isOncoKbLevel1) FROM src."{table}"').fetchone()
            if o_new != o_old:
                problems.append(f"isOncoKbLevel1 {o_old} -> {o_new}")

        # Compare on rowid, not barcode: variant tables have many rows per sample, so a
        # barcode join would fan out. rowid is stable (byte-for-byte copy, UPDATE keeps it).
        clobbered, = conn.execute(f'''
            SELECT COUNT(*) FROM main."{table}" a
            JOIN src."{table}" b ON a.rowid = b.rowid
            WHERE b.AGE_AT_SEQ_REPORT IS NOT NULL
              AND (a.AGE_AT_SEQ_REPORT IS NULL
                   OR a.AGE_AT_SEQ_REPORT != b.AGE_AT_SEQ_REPORT)
        ''').fetchone()
        if clobbered:
            problems.append(f"{clobbered} pre-existing ages overwritten")

        recovered, = conn.execute(f'''
            SELECT COUNT(*) FROM main."{table}" a
            JOIN src."{table}" b ON a.rowid = b.rowid
            WHERE b.AGE_AT_SEQ_REPORT IS NULL
              AND a.AGE_AT_SEQ_REPORT IS NOT NULL
        ''').fetchone()

        status = 'FAIL: ' + '; '.join(problems) if problems else 'PASS'
        if problems:
            ok = False
        print(f"   {table:56s} n={n_new:8d} recovered={recovered:7d}  {status}")

    conn.execute("DETACH DATABASE src")
    print()
    if not ok:
        raise SystemExit("FATAL: verification failed -- do not use this DB.")
    print("   -> cohort membership and isOncoKbLevel1 unchanged; "
          "only NULL ages were filled\n")


def write_provenance(conn, args, recode, n_updates):
    """Standalone provenance table; leaves the patched tables' schemas untouched."""
    conn.execute("DROP TABLE IF EXISTS age_patch_provenance")
    conn.execute("""
        CREATE TABLE age_patch_provenance (
            key   TEXT,
            value TEXT
        )
    """)
    counts = {'>89': 0, '<18': 0}
    for raw in (args._raw_ages[bc] for bc in recode):
        counts[raw] = counts.get(raw, 0) + 1
    # recode covers every redacted sample; under "conservative" the <18 entries are never written
    n_written = sum(1 for v in recode.values() if v is not None)
    rows = [
        ('patched_utc', datetime.now(timezone.utc).isoformat(timespec='seconds')),
        ('variant', args.variant),
        ('rule', '>89 -> 90; <18 -> ' + ('17' if args.variant == 'upper' else 'NULL')),
        ('source_db', str(args.source_db)),
        ('clinical_sample', str(args.clinical_sample)),
        ('release_redacted_samples', str(len(recode))),
        ('release_samples_gt89', str(counts.get('>89', 0))),
        ('release_samples_lt18', str(counts.get('<18', 0))),
        ('samples_written_by_variant', str(n_written)),
        ('row_updates', str(n_updates)),
        ('tables_scope', args.tables),
        ('note', 'Path B / A1. Column patch only -- no OncoKB re-annotation. '
                 'Cohort membership and isOncoKbLevel1 verified unchanged.'),
    ]
    conn.executemany("INSERT INTO age_patch_provenance VALUES (?, ?)", rows)
    conn.commit()
    print("== provenance ==")
    for k, v in rows:
        print(f"   {k:20s} {v}")
    print()


def main():
    p = argparse.ArgumentParser(
        description='Restore GENIE redacted ages into a copy of the harmonized DB.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--clinical-sample', required=True, type=Path,
                   help='GENIE data_clinical_sample.txt (Release 17p0)')
    p.add_argument('--source-db', type=Path, default=None,
                   help='Harmonized SQLite to copy from (default: BALDER_DB_PATH)')
    p.add_argument('--out-db', type=Path, default=None,
                   help='Destination for the patched copy (required unless --dry-run)')
    p.add_argument('--variant', choices=sorted(VARIANTS), default='upper',
                   help='Recode rule for <18 (default: upper). Run both -- see D-e.')
    p.add_argument('--tables', choices=['manuscript', 'all'], default='all',
                   help="'manuscript' patches only the 2 tables the paper reads; "
                        "'all' patches all 9 carrying the column (default)")
    p.add_argument('--dry-run', action='store_true',
                   help='Pre-flight against the source DB only; write nothing')
    p.add_argument('--verify-only', action='store_true',
                   help='Re-run verification against an already-built --out-db '
                        '(no 6.7 GB re-copy, no re-patch)')
    p.add_argument('--force', action='store_true',
                   help='Overwrite --out-db if it already exists')
    args = p.parse_args()

    args.source_db = args.source_db or Path(get_default_db_path())
    if not args.source_db.exists():
        raise SystemExit(f"FATAL: source DB not found: {args.source_db}")
    if not args.clinical_sample.exists():
        raise SystemExit(f"FATAL: clinical sample file not found: {args.clinical_sample}")

    tables = MANUSCRIPT_TABLES + (OTHER_TABLES if args.tables == 'all' else [])

    print(f"\nsource DB : {args.source_db}")
    print(f"release   : {args.clinical_sample}")
    print(f"variant   : {args.variant}  ({'>89 -> 90; <18 -> '}"
          f"{'17' if args.variant == 'upper' else 'NULL'})")
    print(f"tables    : {args.tables} ({len(tables)})\n")

    raw_ages = load_raw_ages(args.clinical_sample)
    args._raw_ages = raw_ages
    print(f"loaded {len(raw_ages)} sample ages from the GENIE release")
    tokens = {t: sum(1 for v in raw_ages.values() if v == t) for t in ('>89', '<18')}
    print(f"   redaction tokens: >89={tokens['>89']}  <18={tokens['<18']}\n")

    with sqlite3.connect(f"file:{args.source_db}?mode=ro", uri=True) as src:
        preflight(src, raw_ages, tables)

    recode = build_recode_map(raw_ages, args.variant)
    n_fill = sum(1 for v in recode.values() if v is not None)
    print(f"recode map: {len(recode)} redacted samples, {n_fill} will be written "
          f"({len(recode) - n_fill} left NULL by this variant)\n")

    if args.dry_run:
        print("--dry-run: pre-flight passed, nothing written.")
        return 0

    if args.verify_only:
        if args.out_db is None or not args.out_db.exists():
            raise SystemExit("FATAL: --verify-only requires an existing --out-db")
        conn = sqlite3.connect(f"file:{args.out_db}?mode=ro", uri=True)
        try:
            verify(conn, args.source_db, tables)
        finally:
            conn.close()
        print(f"OK -- verification re-run against {args.out_db}")
        return 0

    if args.out_db is None:
        raise SystemExit("FATAL: --out-db is required unless --dry-run")
    if args.out_db.exists() and not args.force:
        raise SystemExit(f"FATAL: {args.out_db} exists (use --force to overwrite)")

    print(f"copying {args.source_db.stat().st_size / 1e9:.1f} GB -> {args.out_db} ...")
    args.out_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source_db, args.out_db)
    print("   done\n")

    conn = sqlite3.connect(args.out_db)
    try:
        n_updates = apply_patch(conn, recode, tables)
        verify(conn, args.source_db, tables)
        write_provenance(conn, args, recode, n_updates)
    finally:
        conn.close()

    print(f"OK -- patched DB written to {args.out_db}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
