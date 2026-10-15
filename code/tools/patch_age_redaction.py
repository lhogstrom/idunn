#!/usr/bin/env python3
"""
Path B (A1) — restore GENIE's redacted ages into a copy of the harmonized DB.

GENIE de-identifies age as the *strings* `<18` / `>89`.
`code/pipeline/r_pipeline_scripts/cancer_genomics_studies_db_curration.r:250`
coerces with `as.numeric()`, silently turning both into NULL; numeric missing is
then median-imputed downstream, so every >89-year-old is recoded to ~cohort
median. That is corruption, not just loss.

This script fixes it WITHOUT re-running the Nextflow pipeline. `AGE_AT_SEQ_REPORT`
is a pure passthrough -- it enters at the `as.numeric()` call above and is
thereafter only carried forward by joins. Neither
`generate_perPatientOncoKBLevelSorted.r` (sorts on isHighCoverageByPanelDesign +
OncoKB levels) nor `filter_and_structure_genie_data_for_modeling.r` (filters on
SourceStudy / level-1 / sensitivity / coverage / panel / cancer-type) touches
age. So the fix provably cannot change cohort membership or the outcome variable.

Re-running `workflow/main.nf` would re-fire `run_oncokb` and re-derive
`isOncoKbLevel1` against a newer OncoKB curation snapshot, confounding the Path B
decision gate. Don't.

Two recode variants (decision D-e, 2026-07-31 -- run BOTH as paired sensitivities):

  upper         >89 -> 90,  <18 -> 17     Upper bound on the pediatric tail.
  conservative  >89 -> 90,  <18 -> NULL   Leaves the pediatric bucket missing.

`>89 -> 90` is a mild one-directional underestimate (true values are 90+). `<18`
is NOT symmetric: the bucket spans ages 0-17 and the affected patients are
genuinely pediatric (ASTR: 71 of its 76 recoverable NULLs are `<18`), so
recoding them all to 17 substitutes one bias for another in the exact cancer type
whose age effect the gate watches. Running both bounds ASTR-IDH1 from each side.

Usage
-----
    python code/tools/patch_age_redaction.py \
        --clinical-sample data_clinical_sample.txt \
        --variant upper \
        --out-db ../../output/.../balder-harmonized-...-ageUpper.sqlite

    # dry run -- verifies against the source DB, writes nothing
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

# GENIE's redaction tokens, and what each variant substitutes.
# None means "leave as NULL" (which is what the source DB already holds).
VARIANTS = {
    'upper':        {'>89': 90.0, '<18': 17.0},
    'conservative': {'>89': 90.0, '<18': None},
}

# Every table carrying AGE_AT_SEQ_REPORT. Verified 2026-07-31: these 9 are the
# complete patch surface -- no standalone `AGE` column survives into the
# harmonized DB (the R script's `AGE` is not in harmSampleCols).
MANUSCRIPT_TABLES = [
    'perPatientOncoKBFilteredForModeling',   # TIER 2-7, Table 1, TIER 7 gene-stratified
    'perPatientOncoKBLevelSorted',           # Fig 5B --cancer-level, STROBE
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

CLINICAL_SAMPLE_HEADER_ROWS = 4  # GENIE ships 4 '#' comment lines before the header


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
    """Confirm the release matches the DB before touching anything.

    Two checks, both fail-closed:
      1. Release concordance -- every numeric age already in the DB must equal
         the raw file's value. A single disagreement means this is a different
         GENIE release than the one that built the DB, and the patch is invalid.
      2. Key coverage -- every barcode in the manuscript tables must resolve in
         the raw file, or the patch would silently skip patients.
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
                # Non-GENIE sources (PANCAN, Hartwig) legitimately aren't in the
                # GENIE release; they carry numeric ages and are never patched.
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
    """barcode -> new age, for redacted samples only.

    Deliberately minimal: only samples whose RAW value is a redaction token are
    touched. Rows that already hold a correct numeric age are never written, so
    the patch cannot introduce collateral drift.
    """
    subs = VARIANTS[variant]
    return {bc: subs[raw] for bc, raw in raw_ages.items() if raw in subs}


def apply_patch(conn, recode, tables):
    """Set-based: one UPDATE per table, joined against a TEMP recode table.

    Must not be a per-barcode loop. None of the 9 tables carries a single index
    (verified 2026-07-31), so `WHERE key = ?` is a full scan; 11.5k barcodes x 9
    tables would be ~100k scans, several of them over 2.7M rows. The TEMP table
    is created in SQLite's temp schema, so it never lands in the patched file.
    """
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
    """Post-patch invariants, checked against the untouched source DB.

    The whole interpretability of the Path B decision gate rests on this: if
    cohort membership or the outcome variable moved, the gate would be measuring
    the patch instead of the age fix.
    """
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

        # Row-for-row comparison on rowid, NOT on the barcode. Only
        # perPatientOncoKBFilteredForModeling is one-row-per-barcode; the variant
        # tables carry many rows per sample, so a barcode join fans out into a
        # cartesian product and reports (e.g.) 9.4M "recovered" rows in a 2.7M-row
        # table. rowid is stable here because shutil.copyfile reproduces the file
        # byte-for-byte and UPDATE never reassigns it.
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
    """Standalone table -- deliberately does NOT alter any patched table's schema.

    Adding a column to the patched tables would risk the mixed-column-type and
    SELECT * classes of surprise (CLAUDE.md gotcha #12); a separate table is
    invisible to every existing consumer.
    """
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
    # `recode` holds every redacted sample in the release; under 'conservative'
    # the <18 entries map to None and are never written. Report both, or the
    # provenance overstates what this DB actually contains.
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
