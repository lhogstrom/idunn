"""Generate the STROBE cohort selection flow diagram.

Replays the cohort filtering chain against the database, writes the step counts
(CSV/JSON), and renders the diagram with graphviz.

Usage:
    cd code/analysis/figures
    python generate_strobe_flow_diagram.py [--output-dir PATH]
"""

import sqlite3
import argparse
import csv
import json
import numpy as np
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path
from figure_style import get_graphviz_theme, relative_luminance

# Theme
_GV = get_graphviz_theme()


def _tint(hex_color: str, t: float) -> str:
    """Blend ``hex_color`` toward white by fraction ``t`` (0 = unchanged, 1 = white)."""
    s = hex_color.lstrip('#')
    rgb = tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    blended = tuple(round(c + (255 - c) * t) for c in rgb)
    return '#{:02x}{:02x}{:02x}'.format(*blended)


def _on(hex_color: str) -> str:
    """Return whichever of ink / on_fill is legible on ``hex_color``."""
    return _GV['node_text'] if relative_luminance(hex_color) > 0.35 else _GV['chain_text']


# Flattened into module constants because the DOT f-strings cannot nest the same quote character.
GV_FONT = _GV['font']
GV_FONT_BOLD = _GV['font_bold']
GV_BG = _GV['background']
GV_NODE_FILL = _GV['node_fill']
GV_NODE_LINE = _GV['node_line']
GV_NODE_TEXT = _GV['node_text']
GV_EDGE = _GV['edge']
GV_CHAIN = _GV['chain_fill']
GV_CHAIN_TXT = _GV['chain_text']
GV_TERM = _GV['terminal_fill']
GV_TERM_TXT = _GV['terminal_text']
GV_EXCL = _GV['exclusion_text']
GV_EXCL_EDGE = _GV['exclusion_edge']

DB_PATH = get_default_db_path()
DEFAULT_OUTPUT_DIR = Path("../../../analysis_output/strobe_diagram")

# Fallback when the cancer-type group-definitions CSV cannot be located.
DEFAULT_GROUP_COUNT = 21


def replay_filtering_steps(db_path, canonical_codes=None):
    """Replay the R pipeline filtering steps against the database."""
    conn = sqlite3.connect(db_path)

    # --- Precompute shared filter sets ---

    # Genomic position subset (panel coverage >= 90%)
    gp_df = conn.execute(
        "SELECT positionCtypeID FROM genomicPositionLookup "
        "WHERE perc_of_patients_with_coverage > 90"
    ).fetchall()
    genomic_position_subset = {r[0] for r in gp_df}

    # Median coverage cutoff: 1st percentile of GENIE samples
    coverage_rows = conn.execute(
        "SELECT medianCoverage FROM medianCoveragePerSample "
        "WHERE SourceStudy = 'AACR-GENIE-data' AND medianCoverage IS NOT NULL"
    ).fetchall()
    coverage_values = np.array([r[0] for r in coverage_rows])
    bottom_1_perc = float(np.percentile(coverage_values, 1, method='linear'))

    # Coverage-passing sample barcodes
    coverage_pass_rows = conn.execute(
        "SELECT Tumor_Sample_Barcode FROM medianCoveragePerSample "
        "WHERE medianCoverage IS NOT NULL AND medianCoverage >= ?",
        (bottom_1_perc,)
    ).fetchall()
    coverage_pass_barcodes = {r[0] for r in coverage_pass_rows}

    # --- Load full variant table ---
    print("Loading perPatientOncoKBLevelSorted...")
    all_rows = conn.execute(
        "SELECT SourceStudy, isOncoKbLevel1, ONCOKB_HAS_SENSITIVE_ENTRY, "
        "Tumor_Sample_Barcode, ONCOTREE_CODE, Hugo_Symbol, Start_Position "
        "FROM perPatientOncoKBLevelSorted"
    ).fetchall()

    columns = ['SourceStudy', 'isOncoKbLevel1', 'ONCOKB_HAS_SENSITIVE_ENTRY',
               'Tumor_Sample_Barcode', 'ONCOTREE_CODE', 'Hugo_Symbol', 'Start_Position']

    # Convert to list of dicts for step-by-step filtering
    data = [dict(zip(columns, row)) for row in all_rows]
    print(f"  Loaded {len(data):,} rows")

    # --- Actionable pathway ---
    actionable_steps = []

    # Step 0: Start
    df = data
    actionable_steps.append({
        'step': 0, 'description': 'All sequenced samples in database',
        'n_rows': len(df), 'n_excluded': 0
    })

    # Step 1: GENIE only
    prev = len(df)
    df = [r for r in df if r['SourceStudy'] == 'AACR-GENIE-data']
    actionable_steps.append({
        'step': 1, 'description': 'GENIE data only',
        'n_rows': len(df), 'n_excluded': prev - len(df),
        'exclusion_reason': 'Non-GENIE sources (PANCAN, Hartwig)'
    })

    # Step 2: Level 1 actionable
    prev = len(df)
    df = [r for r in df if r['isOncoKbLevel1'] == 1]
    actionable_steps.append({
        'step': 2, 'description': 'OncoKB Level 1 actionable',
        'n_rows': len(df), 'n_excluded': prev - len(df),
        'exclusion_reason': 'No Level 1 actionable variant'
    })

    # Step 3: Drug sensitivity
    prev = len(df)
    df = [r for r in df if r['ONCOKB_HAS_SENSITIVE_ENTRY'] == 1]
    actionable_steps.append({
        'step': 3, 'description': 'Drug sensitivity confirmed',
        'n_rows': len(df), 'n_excluded': prev - len(df),
        'exclusion_reason': 'No drug sensitivity entry'
    })

    # Step 4: Median coverage >= 1st percentile
    prev = len(df)
    df = [r for r in df if r['Tumor_Sample_Barcode'] in coverage_pass_barcodes]
    actionable_steps.append({
        'step': 4, 'description': f'Sequencing depth filter (>= {bottom_1_perc:.1f}x)',
        'n_rows': len(df), 'n_excluded': prev - len(df),
        'exclusion_reason': 'Below 1st percentile median coverage'
    })

    # Step 5: Panel coverage >= 90%
    prev = len(df)
    df = [r for r in df
          if f"{r['ONCOTREE_CODE']}-{r['Hugo_Symbol']}-{r['Start_Position']}"
          in genomic_position_subset]
    actionable_steps.append({
        'step': 5, 'description': 'Panel coverage >= 90%',
        'n_rows': len(df), 'n_excluded': prev - len(df),
        'exclusion_reason': 'Genomic position covered in < 90% of samples'
    })

    # Step 6: Cancer type >= 250 actionable patients
    from collections import Counter
    ctype_counts = Counter(r['ONCOTREE_CODE'] for r in df)
    eligible_ctypes = sorted([ct for ct, n in ctype_counts.items() if n >= 250])
    prev = len(df)
    df_actionable_final = [r for r in df if r['ONCOTREE_CODE'] in eligible_ctypes]
    actionable_steps.append({
        'step': 6, 'description': f'Cancer type >= 250 events ({len(eligible_ctypes)} types)',
        'n_rows': len(df_actionable_final), 'n_excluded': prev - len(df_actionable_final),
        'exclusion_reason': 'Cancer type with < 250 actionable events'
    })

    # --- Non-actionable pathway ---
    non_actionable_steps = []

    # Step 0: Start (same starting data)
    df = data
    non_actionable_steps.append({
        'step': 0, 'description': 'All sequenced samples in database',
        'n_rows': len(df), 'n_excluded': 0
    })

    # Step 1: GENIE only
    prev = len(df)
    df = [r for r in df if r['SourceStudy'] == 'AACR-GENIE-data']
    non_actionable_steps.append({
        'step': 1, 'description': 'GENIE data only',
        'n_rows': len(df), 'n_excluded': prev - len(df),
        'exclusion_reason': 'Non-GENIE sources'
    })

    # Step 2: NOT Level 1
    prev = len(df)
    df = [r for r in df if r['isOncoKbLevel1'] == 0]
    non_actionable_steps.append({
        'step': 2, 'description': 'No Level 1 actionable variant',
        'n_rows': len(df), 'n_excluded': prev - len(df),
        'exclusion_reason': 'Has Level 1 actionable variant'
    })

    # Step 3: Median coverage >= 1st percentile
    prev = len(df)
    df = [r for r in df if r['Tumor_Sample_Barcode'] in coverage_pass_barcodes]
    non_actionable_steps.append({
        'step': 3, 'description': f'Sequencing depth filter (>= {bottom_1_perc:.1f}x)',
        'n_rows': len(df), 'n_excluded': prev - len(df),
        'exclusion_reason': 'Below 1st percentile median coverage'
    })

    # Step 4: Same eligible cancer types
    prev = len(df)
    df_non_actionable_final = [r for r in df if r['ONCOTREE_CODE'] in eligible_ctypes]
    non_actionable_steps.append({
        'step': 4, 'description': f'Eligible cancer types ({len(eligible_ctypes)} types)',
        'n_rows': len(df_non_actionable_final), 'n_excluded': prev - len(df_non_actionable_final),
        'exclusion_reason': 'Cancer type not in eligible set'
    })

    # --- Final cohort ---
    n_actionable = len(df_actionable_final)
    n_non_actionable = len(df_non_actionable_final)
    final_cohort = n_actionable + n_non_actionable

    # Cross-check against the modeling table.
    db_count = conn.execute(
        "SELECT COUNT(*) FROM perPatientOncoKBFilteredForModeling"
    ).fetchone()[0]

    print(f"\nFiltering complete:")
    print(f"  Actionable: {n_actionable:,}")
    print(f"  Non-actionable: {n_non_actionable:,}")
    print(f"  Total (replayed): {final_cohort:,}")
    print(f"  Total (DB table): {db_count:,}")

    # The replay is sample-level; a patient-unit database has an extra collapse step that must be drawn.
    collapse = None
    if final_cohort != db_count:
        is_collapsed = False
        if db_count < final_cohort:
            dup = conn.execute(
                "SELECT COUNT(*) FROM (SELECT PATIENT_ID, ONCOTREE_CODE "
                "FROM perPatientOncoKBFilteredForModeling "
                "GROUP BY PATIENT_ID, ONCOTREE_CODE HAVING COUNT(*) > 1)"
            ).fetchone()[0]
            is_collapsed = (dup == 0)
        if not is_collapsed:
            conn.close()
            raise SystemExit(
                f"FATAL: replayed cohort ({final_cohort:,}) != "
                f"perPatientOncoKBFilteredForModeling ({db_count:,}), and the "
                f"difference is NOT explained by a patient x cancer-type collapse "
                f"(the table still has duplicate PATIENT_ID x ONCOTREE_CODE rows). "
                f"Refusing to render a STROBE diagram whose terminal node would "
                f"disagree with the analysed cohort.")
        n_act_db = conn.execute(
            "SELECT COUNT(*) FROM perPatientOncoKBFilteredForModeling "
            "WHERE isOncoKbLevel1 = 1").fetchone()[0]
        collapse = {
            'n_before': final_cohort,
            'n_after': db_count,
            'n_removed': final_cohort - db_count,
            'n_actionable': n_act_db,
            'n_non_actionable': db_count - n_act_db,
            'n_patients': conn.execute(
                "SELECT COUNT(DISTINCT PATIENT_ID) FROM "
                "perPatientOncoKBFilteredForModeling").fetchone()[0],
        }
        print(f"  Patient-unit collapse detected: {final_cohort:,} samples -> "
              f"{db_count:,} patient x cancer-type records "
              f"({collapse['n_patients']:,} distinct patients)")
        # The terminal node must be the analysed cohort.
        n_actionable, n_non_actionable = collapse['n_actionable'], collapse['n_non_actionable']
        final_cohort = db_count

    # A cancer type excluded downstream of the database is invisible to the replay, so draw an explicit node for it.
    exclusion = None
    excluded_codes = (sorted(set(eligible_ctypes) - set(canonical_codes))
                      if canonical_codes else None)
    if excluded_codes:
        codes = sorted(set(excluded_codes))
        placeholders = ",".join("?" * len(codes))
        row = conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(isOncoKbLevel1), 0) "
            f"FROM perPatientOncoKBFilteredForModeling "
            f"WHERE ONCOTREE_CODE IN ({placeholders})", codes).fetchone()
        n_removed, n_removed_act = int(row[0]), int(row[1])
        if n_removed == 0:
            conn.close()
            raise SystemExit(
                f"FATAL: --exclude-cancer-types named {codes}, but none of those "
                f"codes has any row in perPatientOncoKBFilteredForModeling. "
                f"Refusing to draw an exclusion node that removes nothing.")
        exclusion = {
            'codes': codes,
            'n_before': final_cohort,
            'n_after': final_cohort - n_removed,
            'n_removed': n_removed,
            'n_removed_actionable': n_removed_act,
            'n_actionable': n_actionable - n_removed_act,
            'n_non_actionable': n_non_actionable - (n_removed - n_removed_act),
        }
        print(f"  Cancer-type exclusion: {final_cohort:,} -> "
              f"{exclusion['n_after']:,} records "
              f"(removed {n_removed:,} in {', '.join(codes)})")
        n_actionable = exclusion['n_actionable']
        n_non_actionable = exclusion['n_non_actionable']
        final_cohort = exclusion['n_after']
        # eligible_ctypes is deliberately not narrowed: it labels upstream nodes whose counts still include the excluded codes.

    conn.close()

    return {
        'actionable_steps': actionable_steps,
        'non_actionable_steps': non_actionable_steps,
        'final_cohort': final_cohort,
        'db_table_count': db_count,
        'n_actionable_final': n_actionable,
        'n_non_actionable_final': n_non_actionable,
        'eligible_cancer_types': eligible_ctypes,
        'coverage_cutoff': bottom_1_perc,
        'collapse': collapse,
        'exclusion': exclusion,
    }


def load_group_count(base_run_dir=None):
    """Load the number of eligible cancer types from the group-definitions CSV."""
    if base_run_dir:
        defs_path = (
            Path(base_run_dir)
            / "tier2_data_setup"
            / "cancer_type_groupings"
            / "cancer_type_group_definitions.csv"
        )
        if defs_path.exists():
            import pandas as pd
            defs = pd.read_csv(defs_path)
            return {'count': int(len(defs)), 'source': str(defs_path)}

    print("  NOTE: Using fallback cancer-type group count (definitions file not found)")
    return {'count': DEFAULT_GROUP_COUNT, 'source': 'fallback (default)'}


def _collapse_dot(counts, color=False, side_boxes=True):
    """DOT fragment for the steps between the cancer-type filter and the final cohort (collapse, exclusion)."""
    c = counts.get('collapse')
    x = counts.get('exclusion')
    if not c and not x:
        return '', 'final'

    f = lambda n: f"{n:,}"
    frag = ''
    # Build the chain back-to-front so each node points at its successor.
    successor = 'final'

    if x:
        style = f', fillcolor="{GV_TERM}", fontcolor="{GV_TERM_TXT}"' if color else ''
        label = (f"Excluded cancer type{'s' if len(x['codes']) > 1 else ''}: "
                 f"{', '.join(x['codes'])}\\n"
                 f"{f(x['n_after'])} records")
        excl_note = ''
        if side_boxes:
            excl_note = (f'    excl_ctype_drop [label="Excluded: {f(x["n_removed"])}'
                         f'\\nInsufficient non-actionable patients", shape=plaintext, '
                         f'fillcolor="{GV_NODE_FILL}", fontsize=9, fontcolor="{GV_EXCL}"]\n'
                         f'    exclusion -> excl_ctype_drop [style=dashed, color="{GV_EXCL_EDGE}"]\n')
        frag = (f'    exclusion [label="{label}"{style}]\n'
                f'    exclusion -> {successor}\n' + excl_note) + frag
        successor = 'exclusion'

    if c:
        style = f', fillcolor="{GV_TERM}", fontcolor="{GV_TERM_TXT}"' if color else ''
        label = (f"Patient-level collapse\\n"
                 f"{f(c['n_after'])} records across "
                 f"{f(c['n_patients'])} distinct patients")
        frag = (f'    collapse [label="{label}"{style}]\n'
                f'    collapse -> {successor}\n') + frag
        successor = 'collapse'

    return frag, successor


def generate_strobe_dot(counts, group_count):
    """Generate Graphviz DOT source for the STROBE cohort flow diagram."""

    act = counts['actionable_steps']
    nact = counts['non_actionable_steps']
    final = counts['final_cohort']
    n_act = counts['n_actionable_final']
    n_nact = counts['n_non_actionable_final']
    n_ctypes = len(counts['eligible_cancer_types'])
    n_groups = group_count['count']

    # Format numbers with commas
    def fmt(n):
        return f"{n:,}"

    collapse_dot, pre_final = _collapse_dot(counts, color=True)

    # Graded fill ramps for the two arms, tinted toward white down the chain.
    act_ramp = [_tint(_GV['actionable_fill'], t) for t in (0.00, 0.12, 0.24, 0.36, 0.48)]
    nact_ramp = [_tint(_GV['non_actionable_fill'], t) for t in (0.00, 0.16, 0.32)]
    chain, chain_txt = _GV['chain_fill'], _GV['chain_text']
    term, term_txt = _GV['terminal_fill'], _GV['terminal_text']
    train_fill = _tint(_GV['split_fill'], 0.25)
    test_fill = _tint(_GV['split_fill'], 0.70)

    dot = f'''digraph STROBE {{
    rankdir=TB
    size="10,16"
    bgcolor="{GV_BG}"
    node [shape=box, style="filled,rounded", fontname="{GV_FONT}", color="{GV_NODE_LINE}", fontsize=10]
    edge [color="{GV_EDGE}", arrowsize=0.7, fontname="{GV_FONT}", fontsize=9]

    labelloc="t"
    label="STROBE Cohort Selection Flow Diagram"
    fontsize=16
    fontname="{GV_FONT_BOLD}"

    // ===== INITIAL POPULATION =====
    start [label="All sequenced samples in database\\n(perPatientOncoKBLevelSorted;\\none representative variant per sample)\\nN = {fmt(act[0]['n_rows'])}", fillcolor="{chain}", fontcolor="{chain_txt}"]

    // GENIE filter
    genie [label="GENIE data only\\nN = {fmt(act[1]['n_rows'])}", fillcolor="{_tint(chain, 0.15)}", fontcolor="{chain_txt}"]
    excl_genie [label="Excluded: {fmt(act[1]['n_excluded'])}\\n{act[1].get('exclusion_reason', '')}", shape=plaintext, fillcolor="{GV_NODE_FILL}", fontsize=9, fontcolor="{GV_EXCL}"]

    start -> genie
    start -> excl_genie [style=dashed, color="{GV_EXCL_EDGE}"]

    // ===== SPLIT INTO ACTIONABLE / NON-ACTIONABLE =====
    {{rank=same; act_branch; nact_branch}}

    act_branch [label="Actionable (Level 1)\\nN = {fmt(act[2]['n_rows'])}", fillcolor="{act_ramp[0]}", fontcolor="{_on(act_ramp[0])}"]
    nact_branch [label="Non-actionable (controls)\\nN = {fmt(nact[2]['n_rows'])}", fillcolor="{nact_ramp[0]}", fontcolor="{_on(nact_ramp[0])}"]

    genie -> act_branch [label="  isOncoKbLevel1 = 1"]
    genie -> nact_branch [label="  isOncoKbLevel1 = 0"]

    // ===== ACTIONABLE PATHWAY =====

    // Drug sensitivity
    act_drug [label="Drug sensitivity confirmed\\nN = {fmt(act[3]['n_rows'])}", fillcolor="{act_ramp[1]}", fontcolor="{_on(act_ramp[1])}"]
    excl_drug [label="Excluded: {fmt(act[3]['n_excluded'])}\\n{act[3].get('exclusion_reason', '')}", shape=plaintext, fillcolor="{GV_NODE_FILL}", fontsize=9, fontcolor="{GV_EXCL}"]
    act_branch -> act_drug
    act_branch -> excl_drug [style=dashed, color="{GV_EXCL_EDGE}"]

    // Coverage filter (actionable)
    act_cov [label="Sequencing depth filter\\nN = {fmt(act[4]['n_rows'])}", fillcolor="{act_ramp[2]}", fontcolor="{_on(act_ramp[2])}"]
    excl_act_cov [label="Excluded: {fmt(act[4]['n_excluded'])}\\nBelow 1st percentile coverage", shape=plaintext, fillcolor="{GV_NODE_FILL}", fontsize=9, fontcolor="{GV_EXCL}"]
    act_drug -> act_cov
    act_drug -> excl_act_cov [style=dashed, color="{GV_EXCL_EDGE}"]

    // Panel coverage
    act_panel [label="Panel coverage >= 90%\\nN = {fmt(act[5]['n_rows'])}", fillcolor="{act_ramp[3]}", fontcolor="{_on(act_ramp[3])}"]
    excl_panel [label="Excluded: {fmt(act[5]['n_excluded'])}\\nInsufficient panel coverage", shape=plaintext, fillcolor="{GV_NODE_FILL}", fontsize=9, fontcolor="{GV_EXCL}"]
    act_cov -> act_panel
    act_cov -> excl_panel [style=dashed, color="{GV_EXCL_EDGE}"]

    // Cancer type selection (actionable)
    act_ctype [label="Cancer type >= 250 events\\nN = {fmt(act[6]['n_rows'])}\\n({n_ctypes} cancer types)", fillcolor="{act_ramp[4]}", fontcolor="{_on(act_ramp[4])}"]
    excl_act_ctype [label="Excluded: {fmt(act[6]['n_excluded'])}\\nRare cancer types", shape=plaintext, fillcolor="{GV_NODE_FILL}", fontsize=9, fontcolor="{GV_EXCL}"]
    act_panel -> act_ctype
    act_panel -> excl_act_ctype [style=dashed, color="{GV_EXCL_EDGE}"]

    // ===== NON-ACTIONABLE PATHWAY =====

    // Coverage filter (non-actionable)
    nact_cov [label="Sequencing depth filter\\nN = {fmt(nact[3]['n_rows'])}", fillcolor="{nact_ramp[1]}", fontcolor="{_on(nact_ramp[1])}"]
    excl_nact_cov [label="Excluded: {fmt(nact[3]['n_excluded'])}\\nBelow 1st percentile coverage", shape=plaintext, fillcolor="{GV_NODE_FILL}", fontsize=9, fontcolor="{GV_EXCL}"]
    nact_branch -> nact_cov
    nact_branch -> excl_nact_cov [style=dashed, color="{GV_EXCL_EDGE}"]

    // Cancer type selection (non-actionable)
    nact_ctype [label="Eligible cancer types\\nN = {fmt(nact[4]['n_rows'])}\\n({n_ctypes} cancer types)", fillcolor="{nact_ramp[2]}", fontcolor="{_on(nact_ramp[2])}"]
    excl_nact_ctype [label="Excluded: {fmt(nact[4]['n_excluded'])}\\nCancer type not eligible", shape=plaintext, fillcolor="{GV_NODE_FILL}", fontsize=9, fontcolor="{GV_EXCL}"]
    nact_cov -> nact_ctype
    nact_cov -> excl_nact_ctype [style=dashed, color="{GV_EXCL_EDGE}"]

    // ===== MERGE =====
    {{rank=same; act_ctype; nact_ctype}}

    final [label="Final analytic cohort\\n(perPatientOncoKBFilteredForModeling)\\nN = {fmt(final)}\\n({fmt(n_act)} actionable + {fmt(n_nact)} non-actionable)", fillcolor="{term}", fontcolor="{term_txt}"]

{collapse_dot}    act_ctype -> {pre_final}
    nact_ctype -> {pre_final}

    // ===== CANCER-TYPE GROUPINGS =====
    groupings [label="Cancer-type groupings\\n({n_groups} OncoTree codes)", fillcolor="{term}", fontcolor="{term_txt}"]

    final -> groupings

    // ===== TRAIN/TEST SPLIT =====
    subgraph cluster_split {{
        label=""
        style="invis"
        rank=same

        train [label="Train set (80%)\\nStratified by cancer type", shape=folder, fillcolor="{train_fill}", fontcolor="{_on(train_fill)}", style="filled", fontname="{GV_FONT}", fontsize=10]
        test [label="Test set (20%)\\nHeld out until TIER 5", shape=folder, fillcolor="{test_fill}", fontcolor="{_on(test_fill)}", style="filled", fontname="{GV_FONT}", fontsize=10]
    }}

    groupings -> train [label="  80/20 stratified split (seed=42)"]
    groupings -> test
}}
'''
    return dot


def generate_strobe_dot_simple(counts, group_count):
    """Generate a minimal black-and-white STROBE diagram with square boxes and orthogonal arrows."""

    act = counts['actionable_steps']
    nact = counts['non_actionable_steps']
    final = counts['final_cohort']
    n_act = counts['n_actionable_final']
    n_nact = counts['n_non_actionable_final']
    n_ctypes = len(counts['eligible_cancer_types'])
    n_groups = group_count['count']

    def fmt(n):
        return f"{n:,}"

    collapse_dot, pre_final = _collapse_dot(counts, color=False)

    dot = f'''digraph STROBE {{
    rankdir=TB
    size="10,16"
    bgcolor=white
    splines=ortho
    nodesep=0.5
    ranksep=0.6
    node [shape=box, style="", fontname="Arial", fontsize=10, color=black, fillcolor=white, penwidth=1.0]
    edge [color=black, arrowsize=0.7, fontname="Arial", fontsize=9]

    // ===== STARTING POPULATION (GENIE only) =====
    genie [label="AACR GENIE data\\nN = {fmt(act[1]['n_rows'])}"]

    // ===== ROW 1: SPLIT =====
    {{rank=same; act_branch; nact_branch}}
    act_branch [label="Actionable Drug Sensitivity\\n(OncoKB Level 1)\\nN = {fmt(act[3]['n_rows'])}"]
    nact_branch [label="Non-actionable (controls)\\nN = {fmt(nact[2]['n_rows'])}"]

    genie -> act_branch [label="  isOncoKbLevel1 = 1"]
    genie -> nact_branch [label="  isOncoKbLevel1 = 0"]

    // ===== ROW 2: Seq depth filter (both pathways) =====
    {{rank=same; excl_act_cov; act_cov; nact_cov; excl_nact_cov}}
    excl_act_cov [label="Excluded: {fmt(act[4]['n_excluded'])}\\nBelow 1st percentile coverage"]
    act_cov [label="Sequencing depth filter\\nN = {fmt(act[4]['n_rows'])}"]
    nact_cov [label="Sequencing depth filter\\nN = {fmt(nact[3]['n_rows'])}"]
    excl_nact_cov [label="Excluded: {fmt(nact[3]['n_excluded'])}\\nBelow 1st percentile coverage"]

    act_branch -> act_cov
    act_cov -> excl_act_cov [style=dashed]
    nact_branch -> nact_cov
    nact_cov -> excl_nact_cov [style=dashed]

    // ===== ROW 3: Panel coverage =====
    {{rank=same; excl_panel; act_panel; nact_spacer1}}
    excl_panel [label="Excluded: {fmt(act[5]['n_excluded'])}\\nInsufficient panel coverage"]
    act_panel [label="Panel coverage >= 90%\\nN = {fmt(act[5]['n_rows'])}"]
    nact_spacer1 [style=invis, width=0, height=0, label=""]

    act_cov -> act_panel
    act_panel -> excl_panel [style=dashed]
    nact_cov -> nact_spacer1 [style=invis]

    // ===== ROW 4: Eligible cancer types (both pathways, aligned) =====
    {{rank=same; excl_act_ctype; act_ctype; nact_ctype; excl_nact_ctype}}
    excl_act_ctype [label="Excluded: {fmt(act[6]['n_excluded'])}\\nRare cancer types"]
    act_ctype [label="Eligible cancer types >= 250 events\\nN = {fmt(act[6]['n_rows'])}\\n({n_ctypes} cancer types)"]
    nact_ctype [label="Eligible cancer types\\nN = {fmt(nact[4]['n_rows'])}\\n({n_ctypes} cancer types)"]
    excl_nact_ctype [label="Excluded: {fmt(nact[4]['n_excluded'])}\\nCancer type not eligible"]

    act_panel -> act_ctype
    act_ctype -> excl_act_ctype [style=dashed]
    nact_spacer1 -> nact_ctype [style=invis]
    nact_cov -> nact_ctype
    nact_ctype -> excl_nact_ctype [style=dashed]

    // ===== MERGE =====
    final [label="Final analytic cohort\\nN = {fmt(final)}"]

{collapse_dot}    act_ctype -> {pre_final}
    nact_ctype -> {pre_final}

    // ===== CANCER-TYPE GROUPINGS =====
    groupings [label="Cancer-type groupings\\n({n_groups} OncoTree codes)"]

    final -> groupings

    // ===== TRAIN/TEST SPLIT =====
    {{rank=same; train; test}}
    train [label="Train set (80%)\\nStratified by cancer type"]
    test [label="Test set (20%)\\nHeld out until TIER 5"]

    groupings -> train [label="  80/20 stratified split (seed=42)"]
    groupings -> test
}}
'''
    return dot


def generate_strobe_dot_minimal(counts, group_count):
    """Generate a minimal black-and-white STROBE diagram with no exclusion boxes or dashed edges."""

    act = counts['actionable_steps']
    nact = counts['non_actionable_steps']
    final = counts['final_cohort']
    n_ctypes = len(counts['eligible_cancer_types'])
    n_groups = group_count['count']

    def fmt(n):
        return f"{n:,}"

    collapse_dot, pre_final = _collapse_dot(counts, color=False, side_boxes=False)

    dot = f'''digraph STROBE {{
    rankdir=TB
    size="8,12"
    bgcolor=white
    splines=ortho
    nodesep=0.5
    ranksep=0.6
    node [shape=box, style="", fontname="Arial", fontsize=10, color=black, fillcolor=white, penwidth=1.0]
    edge [color=black, arrowsize=0.7, fontname="Arial", fontsize=9]

    // ===== STARTING POPULATION (GENIE only) =====
    genie [label="AACR GENIE data\\nN = {fmt(act[1]['n_rows'])}"]

    // ===== ROW 1: SPLIT =====
    {{rank=same; act_branch; nact_branch}}
    act_branch [label="Actionable Drug Sensitivity\\n(OncoKB Level 1)\\nN = {fmt(act[3]['n_rows'])}"]
    nact_branch [label="Non-actionable (controls)\\nN = {fmt(nact[2]['n_rows'])}"]

    genie -> act_branch [label="  isOncoKbLevel1 = 1"]
    genie -> nact_branch [label="  isOncoKbLevel1 = 0"]

    // ===== ROW 2: Seq depth filter (both pathways) =====
    {{rank=same; act_cov; nact_cov}}
    act_cov [label="Sequencing depth filter\\nN = {fmt(act[4]['n_rows'])}"]
    nact_cov [label="Sequencing depth filter\\nN = {fmt(nact[3]['n_rows'])}"]

    act_branch -> act_cov
    nact_branch -> nact_cov

    // ===== ROW 3: Panel coverage (actionable only) =====
    act_panel [label="Panel coverage >= 90%\\nN = {fmt(act[5]['n_rows'])}"]
    act_cov -> act_panel

    // ===== ROW 4: Eligible cancer types (both pathways, aligned) =====
    {{rank=same; act_ctype; nact_ctype}}
    act_ctype [label="Eligible cancer types >= 250 events\\nN = {fmt(act[6]['n_rows'])}\\n({n_ctypes} cancer types)"]
    nact_ctype [label="Eligible cancer types\\nN = {fmt(nact[4]['n_rows'])}\\n({n_ctypes} cancer types)"]

    act_panel -> act_ctype
    nact_cov -> nact_ctype

    // ===== MERGE =====
    final [label="Final analytic cohort\\nN = {fmt(final)}"]

{collapse_dot}    act_ctype -> {pre_final}
    nact_ctype -> {pre_final}

    // ===== CANCER-TYPE GROUPINGS =====
    groupings [label="Cancer-type groupings\\n({n_groups} OncoTree codes)"]

    final -> groupings

    // ===== TRAIN/TEST SPLIT =====
    {{rank=same; train; test}}
    train [label="Train set (80%)\\nStratified by cancer type"]
    test [label="Test set (20%)\\nHeld out until TIER 5"]

    groupings -> train [label="  80/20 stratified split (seed=42)"]
    groupings -> test
}}
'''
    return dot


def render_diagram(dot_source, output_dir, prefix="strobe_cohort_flow_diagram"):
    """Render DOT source to PDF, PNG, and save .gv.txt source."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Graphviz source
    gv_path = output_dir / f"{prefix}.gv.txt"
    gv_path.write_text(dot_source)
    print(f"  Graphviz source: {gv_path}")

    # Render to PDF and PNG
    for fmt in ['pdf', 'png']:
        out_path = output_dir / f"{prefix}.{fmt}"
        try:
            result = subprocess.run(
                ['dot', f'-T{fmt}', '-o', str(out_path)],
                input=dot_source, capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  Rendered {fmt.upper()}: {out_path}")
            else:
                print(f"  WARNING: dot failed for {fmt}: {result.stderr}")
        except FileNotFoundError:
            print(f"  WARNING: 'dot' binary not found, skipping {fmt} rendering")
            print(f"  Install graphviz: conda install graphviz or brew install graphviz")


def save_counts_csv(counts, output_dir):
    """Save filtering counts as a CSV with one row per filter step."""
    output_dir = Path(output_dir)
    csv_path = output_dir / "strobe_filtering_counts.csv"

    fieldnames = ['pathway', 'step', 'description', 'n_rows', 'n_excluded',
                  'exclusion_reason']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for step in counts['actionable_steps']:
            writer.writerow({
                'pathway': 'actionable',
                'step': step['step'],
                'description': step['description'],
                'n_rows': step['n_rows'],
                'n_excluded': step['n_excluded'],
                'exclusion_reason': step.get('exclusion_reason', ''),
            })

        for step in counts['non_actionable_steps']:
            writer.writerow({
                'pathway': 'non_actionable',
                'step': step['step'],
                'description': step['description'],
                'n_rows': step['n_rows'],
                'n_excluded': step['n_excluded'],
                'exclusion_reason': step.get('exclusion_reason', ''),
            })

        # Final merged cohort row
        writer.writerow({
            'pathway': 'merged',
            'step': 'final',
            'description': 'Final analytic cohort (perPatientOncoKBFilteredForModeling)',
            'n_rows': counts['final_cohort'],
            'n_excluded': '',
            'exclusion_reason': '',
        })

        gc = counts.get('group_count')
        if gc:
            writer.writerow({
                'pathway': 'groupings',
                'step': 'cancer_type_groupings',
                'description': f"Cancer-type groupings ({gc['count']} OncoTree codes)",
                'n_rows': gc['count'],
                'n_excluded': '',
                'exclusion_reason': '',
            })

    print(f"  Counts CSV: {csv_path}")


def save_counts_json(counts, output_dir):
    """Save filtering counts as machine-readable JSON."""
    output_dir = Path(output_dir)
    json_path = output_dir / "strobe_filtering_counts.json"

    # Make JSON-serializable
    output = {
        'actionable_steps': counts['actionable_steps'],
        'non_actionable_steps': counts['non_actionable_steps'],
        'final_cohort': counts['final_cohort'],
        'db_table_count': counts['db_table_count'],
        'n_actionable_final': counts['n_actionable_final'],
        'n_non_actionable_final': counts['n_non_actionable_final'],
        'eligible_cancer_types': counts['eligible_cancer_types'],
        'coverage_cutoff': counts['coverage_cutoff'],
        'group_count': counts.get('group_count'),
        'collapse': counts.get('collapse'),
        # Record the exclusion so the audit trail matches the figure.
        'exclusion': counts.get('exclusion'),
    }

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Counts JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate STROBE cohort selection flow diagram'
    )
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: analysis_output/strobe_diagram)')
    parser.add_argument('--db-path', type=str, default=DB_PATH,
                        help='Path to SQLite database')
    parser.add_argument('--base-run-dir', type=str, default=None,
                        help='Run directory whose TIER 2a cancer_type_group_definitions.csv '
                             'supplies the eligible cancer-type count for the diagram '
                             '(optional; falls back to a default count when absent)')
    parser.add_argument('--counts-only', action='store_true',
                        help='Write the filtering counts CSV/JSON only; skip rendering '
                             '(no graphviz needed)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("STROBE Cohort Selection Flow Diagram Generator")
    print("=" * 60)

    if not Path(args.db_path).exists():
        print(f"ERROR: Database not found at {args.db_path}")
        return 1

    # Read the run's cohort definition before the replay so excluded cancer types get an explicit node.
    canonical_codes = None
    if args.base_run_dir:
        defs_path = (Path(args.base_run_dir) / "tier2_data_setup"
                     / "cancer_type_groupings" / "cancer_type_group_definitions.csv")
        if defs_path.exists():
            import pandas as pd
            canonical_codes = pd.read_csv(defs_path)['ONCOTREE_CODE'].astype(str).tolist()
            print(f"\nCohort source: {defs_path} ({len(canonical_codes)} codes)")

    # Step 1: Replay filtering
    print("\n[1/4] Replaying filtering steps...")
    counts = replay_filtering_steps(args.db_path, canonical_codes=canonical_codes)

    # Step 2: Load cancer-type group count
    print("\n[2/6] Loading cancer-type group count...")
    group_count = load_group_count(args.base_run_dir)
    print(f"  Cancer-type groups: {group_count['count']}")
    print(f"  Source: {group_count['source']}")
    counts['group_count'] = group_count

    # Step 3: Generate diagrams
    if not args.counts_only:
        print("\n[3/6] Generating STROBE diagram (color)...")
        dot_source = generate_strobe_dot(counts, group_count)
        render_diagram(dot_source, output_dir)

        print("\n[4/6] Generating STROBE diagram (simple B&W)...")
        dot_simple = generate_strobe_dot_simple(counts, group_count)
        render_diagram(dot_simple, output_dir, prefix="strobe_cohort_flow_diagram_simple")

        print("\n[5/6] Generating STROBE diagram (minimal B&W, no exclusion boxes)...")
        dot_minimal = generate_strobe_dot_minimal(counts, group_count)
        render_diagram(dot_minimal, output_dir, prefix="strobe_cohort_flow_diagram_minimal")

    # Step 6: Save counts
    print("\n[6/6] Saving audit data...")
    save_counts_json(counts, output_dir)
    save_counts_csv(counts, output_dir)

    print("\n" + "=" * 60)
    print("STROBE DIAGRAM GENERATION COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
