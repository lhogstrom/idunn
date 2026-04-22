"""
Generate STROBE Cohort Selection Flow Diagram

Replays the R pipeline filtering steps (filter_and_structure_genie_data_for_modeling.r)
against the SQLite database to obtain exact patient counts at each step, then renders
a STROBE-compliant flow diagram using Graphviz.

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

DB_PATH = get_default_db_path()
DEFAULT_OUTPUT_DIR = Path("../../../analysis_output/strobe_diagram")

# Fallback tier structure (used when multi-tier mapping is unavailable).
# Source of truth: runs/run_*/tier2_data_setup/multi_tier_groupings/multi_tier_group_definitions.csv
DEFAULT_TIER_COUNTS = {'tier1': 21, 'tier2': 3, 'tier3': 10}


def replay_filtering_steps(db_path):
    """
    Replay the R pipeline filtering steps against the database.

    Returns dict with keys:
        'actionable_steps': list of {step, description, n_rows, n_excluded}
        'non_actionable_steps': list of {step, description, n_rows, n_excluded}
        'final_cohort': int (total rows in perPatientOncoKBFilteredForModeling)
        'eligible_cancer_types': list of str
        'n_actionable_final': int
        'n_non_actionable_final': int
    """
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
        'step': 0, 'description': 'All patients in database',
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
        'exclusion_reason': 'Genomic position covered in < 90% of patients'
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
        'step': 0, 'description': 'All patients in database',
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

    # Cross-check against perPatientOncoKBFilteredForModeling
    db_count = conn.execute(
        "SELECT COUNT(*) FROM perPatientOncoKBFilteredForModeling"
    ).fetchone()[0]

    conn.close()

    print(f"\nFiltering complete:")
    print(f"  Actionable: {n_actionable:,}")
    print(f"  Non-actionable: {n_non_actionable:,}")
    print(f"  Total (replayed): {final_cohort:,}")
    print(f"  Total (DB table): {db_count:,}")
    if final_cohort != db_count:
        print(f"  WARNING: Count mismatch ({final_cohort} vs {db_count})")

    return {
        'actionable_steps': actionable_steps,
        'non_actionable_steps': non_actionable_steps,
        'final_cohort': final_cohort,
        'db_table_count': db_count,
        'n_actionable_final': n_actionable,
        'n_non_actionable_final': n_non_actionable,
        'eligible_cancer_types': eligible_ctypes,
        'coverage_cutoff': bottom_1_perc,
    }


def load_multi_tier_counts(base_run_dir=None):
    """
    Load actual multi-tier group counts if available; otherwise return fallback.

    Returns dict with keys: 'tier1', 'tier2', 'tier3', 'source', 'groups_by_tier' (list per tier)
    """
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))

    # Try explicit base-run-dir first
    if base_run_dir:
        defs_path = Path(base_run_dir) / "tier2_data_setup" / "multi_tier_groupings" / "multi_tier_group_definitions.csv"
        if defs_path.exists():
            import pandas as pd
            defs = pd.read_csv(defs_path)
            return {
                'tier1': int((defs['tier'] == 'tier1').sum()),
                'tier2': int((defs['tier'] == 'tier2').sum()),
                'tier3': int((defs['tier'] == 'tier3').sum()),
                'source': str(defs_path),
            }

    # Auto-search via multi_tier_utils
    try:
        from multi_tier_utils import get_group_definitions_file
        defs_file = get_group_definitions_file()
        if defs_file and defs_file.exists():
            import pandas as pd
            defs = pd.read_csv(defs_file)
            return {
                'tier1': int((defs['tier'] == 'tier1').sum()),
                'tier2': int((defs['tier'] == 'tier2').sum()),
                'tier3': int((defs['tier'] == 'tier3').sum()),
                'source': str(defs_file),
            }
    except Exception as e:
        print(f"  NOTE: Could not load multi-tier definitions ({e})")

    # Fallback: known structure
    print(f"  NOTE: Using fallback tier structure (mapping file not found)")
    return {
        'tier1': DEFAULT_TIER_COUNTS['tier1'],
        'tier2': DEFAULT_TIER_COUNTS['tier2'],
        'tier3': DEFAULT_TIER_COUNTS['tier3'],
        'source': 'fallback (default)',
    }


def generate_strobe_dot(counts, tier_counts):
    """Generate Graphviz DOT source for the STROBE cohort flow diagram."""

    act = counts['actionable_steps']
    nact = counts['non_actionable_steps']
    final = counts['final_cohort']
    n_act = counts['n_actionable_final']
    n_nact = counts['n_non_actionable_final']
    n_ctypes = len(counts['eligible_cancer_types'])
    n_tier1 = tier_counts['tier1']
    n_tier2 = tier_counts['tier2']
    n_tier3 = tier_counts['tier3']
    n_total_groups = n_tier1 + n_tier2 + n_tier3

    # Format numbers with commas
    def fmt(n):
        return f"{n:,}"

    dot = f'''digraph STROBE {{
    rankdir=TB
    size="10,16"
    bgcolor="#ffffff"
    node [shape=box, style="filled,rounded", fontname="Helvetica", color=black, fontsize=10]
    edge [color=gray50, arrowsize=0.7, fontname="Helvetica", fontsize=9]

    labelloc="t"
    label="STROBE Cohort Selection Flow Diagram"
    fontsize=16
    fontname="Helvetica-Bold"

    // ===== INITIAL POPULATION =====
    start [label="All patients in database\\n(perPatientOncoKBLevelSorted)\\nN = {fmt(act[0]['n_rows'])}", fillcolor="#1565c0", fontcolor=white]

    // GENIE filter
    genie [label="GENIE data only\\nN = {fmt(act[1]['n_rows'])}", fillcolor="#1976d2", fontcolor=white]
    excl_genie [label="Excluded: {fmt(act[1]['n_excluded'])}\\n{act[1].get('exclusion_reason', '')}", shape=plaintext, fillcolor=white, fontsize=9, fontcolor="#b71c1c"]

    start -> genie
    start -> excl_genie [style=dashed, color="#b71c1c"]

    // ===== SPLIT INTO ACTIONABLE / NON-ACTIONABLE =====
    {{rank=same; act_branch; nact_branch}}

    act_branch [label="Actionable (Level 1)\\nN = {fmt(act[2]['n_rows'])}", fillcolor="#2e7d32", fontcolor=white]
    nact_branch [label="Non-actionable (controls)\\nN = {fmt(nact[2]['n_rows'])}", fillcolor="#e65100", fontcolor=white]

    genie -> act_branch [label="  isOncoKbLevel1 = 1"]
    genie -> nact_branch [label="  isOncoKbLevel1 = 0"]

    // ===== ACTIONABLE PATHWAY =====

    // Drug sensitivity
    act_drug [label="Drug sensitivity confirmed\\nN = {fmt(act[3]['n_rows'])}", fillcolor="#388e3c", fontcolor=white]
    excl_drug [label="Excluded: {fmt(act[3]['n_excluded'])}\\n{act[3].get('exclusion_reason', '')}", shape=plaintext, fillcolor=white, fontsize=9, fontcolor="#b71c1c"]
    act_branch -> act_drug
    act_branch -> excl_drug [style=dashed, color="#b71c1c"]

    // Coverage filter (actionable)
    act_cov [label="Sequencing depth filter\\nN = {fmt(act[4]['n_rows'])}", fillcolor="#43a047", fontcolor=white]
    excl_act_cov [label="Excluded: {fmt(act[4]['n_excluded'])}\\nBelow 1st percentile coverage", shape=plaintext, fillcolor=white, fontsize=9, fontcolor="#b71c1c"]
    act_drug -> act_cov
    act_drug -> excl_act_cov [style=dashed, color="#b71c1c"]

    // Panel coverage
    act_panel [label="Panel coverage >= 90%\\nN = {fmt(act[5]['n_rows'])}", fillcolor="#4caf50", fontcolor=white]
    excl_panel [label="Excluded: {fmt(act[5]['n_excluded'])}\\nInsufficient panel coverage", shape=plaintext, fillcolor=white, fontsize=9, fontcolor="#b71c1c"]
    act_cov -> act_panel
    act_cov -> excl_panel [style=dashed, color="#b71c1c"]

    // Cancer type selection (actionable)
    act_ctype [label="Cancer type >= 250 events\\nN = {fmt(act[6]['n_rows'])}\\n({n_ctypes} cancer types)", fillcolor="#66bb6a"]
    excl_act_ctype [label="Excluded: {fmt(act[6]['n_excluded'])}\\nRare cancer types", shape=plaintext, fillcolor=white, fontsize=9, fontcolor="#b71c1c"]
    act_panel -> act_ctype
    act_panel -> excl_act_ctype [style=dashed, color="#b71c1c"]

    // ===== NON-ACTIONABLE PATHWAY =====

    // Coverage filter (non-actionable)
    nact_cov [label="Sequencing depth filter\\nN = {fmt(nact[3]['n_rows'])}", fillcolor="#ef6c00", fontcolor=white]
    excl_nact_cov [label="Excluded: {fmt(nact[3]['n_excluded'])}\\nBelow 1st percentile coverage", shape=plaintext, fillcolor=white, fontsize=9, fontcolor="#b71c1c"]
    nact_branch -> nact_cov
    nact_branch -> excl_nact_cov [style=dashed, color="#b71c1c"]

    // Cancer type selection (non-actionable)
    nact_ctype [label="Eligible cancer types\\nN = {fmt(nact[4]['n_rows'])}\\n({n_ctypes} cancer types)", fillcolor="#ff9800"]
    excl_nact_ctype [label="Excluded: {fmt(nact[4]['n_excluded'])}\\nCancer type not eligible", shape=plaintext, fillcolor=white, fontsize=9, fontcolor="#b71c1c"]
    nact_cov -> nact_ctype
    nact_cov -> excl_nact_ctype [style=dashed, color="#b71c1c"]

    // ===== MERGE =====
    {{rank=same; act_ctype; nact_ctype}}

    final [label="Final analytic cohort\\n(perPatientOncoKBFilteredForModeling)\\nN = {fmt(final)}\\n({fmt(n_act)} actionable + {fmt(n_nact)} non-actionable)", fillcolor="#1565c0", fontcolor=white]

    act_ctype -> final
    nact_ctype -> final

    // ===== MULTI-TIER GROUPINGS =====
    multitier [label="Multi-tier cancer groupings\\n{n_total_groups} analysis groups\\n({n_tier1} Tier 1 + {n_tier2} Tier 2 + {n_tier3} Tier 3)", fillcolor="#0d47a1", fontcolor=white]

    final -> multitier [label="  OncoTree-based roll-up"]

    // ===== TRAIN/TEST SPLIT =====
    subgraph cluster_split {{
        label=""
        style="invis"
        rank=same

        train [label="Train set (80%)\\nStratified by cancer type", shape=folder, fillcolor="#a5d6a7", style="filled", fontname="Helvetica", fontsize=10]
        test [label="Test set (20%)\\nHeld out until TIER 5", shape=folder, fillcolor="#e8f5e9", style="filled", fontname="Helvetica", fontsize=10]
    }}

    multitier -> train [label="  80/20 stratified split (seed=42)"]
    multitier -> test
}}
'''
    return dot


def generate_strobe_dot_simple(counts, tier_counts):
    """Generate a minimal black-and-white STROBE diagram with square boxes and orthogonal arrows."""

    act = counts['actionable_steps']
    nact = counts['non_actionable_steps']
    final = counts['final_cohort']
    n_act = counts['n_actionable_final']
    n_nact = counts['n_non_actionable_final']
    n_ctypes = len(counts['eligible_cancer_types'])
    n_tier1 = tier_counts['tier1']
    n_tier2 = tier_counts['tier2']
    n_tier3 = tier_counts['tier3']
    n_total_groups = n_tier1 + n_tier2 + n_tier3

    def fmt(n):
        return f"{n:,}"

    dot = f'''digraph STROBE {{
    rankdir=TB
    size="10,16"
    bgcolor=white
    splines=ortho
    nodesep=0.5
    ranksep=0.6
    node [shape=box, style="", fontname="Helvetica", fontsize=10, color=black, fillcolor=white, penwidth=1.0]
    edge [color=black, arrowsize=0.7, fontname="Helvetica", fontsize=9]

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

    act_ctype -> final
    nact_ctype -> final

    // ===== MULTI-TIER GROUPINGS =====
    multitier [label="Multi-tier cancer groupings\\n{n_total_groups} analysis groups\\n({n_tier1} Tier 1 + {n_tier2} Tier 2 + {n_tier3} Tier 3)"]

    final -> multitier [label="  OncoTree-based roll-up"]

    // ===== TRAIN/TEST SPLIT =====
    {{rank=same; train; test}}
    train [label="Train set (80%)\\nStratified by cancer type"]
    test [label="Test set (20%)\\nHeld out until TIER 5"]

    multitier -> train [label="  80/20 stratified split (seed=42)"]
    multitier -> test
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

        # Multi-tier grouping rows (post-cohort analysis structure)
        tc = counts.get('tier_counts')
        if tc:
            total_groups = tc['tier1'] + tc['tier2'] + tc['tier3']
            writer.writerow({
                'pathway': 'multitier',
                'step': 'grouping',
                'description': f"Multi-tier analysis groups ({total_groups} total)",
                'n_rows': counts['final_cohort'],
                'n_excluded': '',
                'exclusion_reason': '',
            })
            for tier_name in ('tier1', 'tier2', 'tier3'):
                writer.writerow({
                    'pathway': 'multitier',
                    'step': tier_name,
                    'description': f"{tier_name} analysis groups",
                    'n_rows': tc[tier_name],
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
        'tier_counts': counts.get('tier_counts'),
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
                        help='Run directory containing multi-tier group definitions '
                             '(optional; falls back to auto-search, then default structure)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("STROBE Cohort Selection Flow Diagram Generator")
    print("=" * 60)

    if not Path(args.db_path).exists():
        print(f"ERROR: Database not found at {args.db_path}")
        return 1

    # Step 1: Replay filtering
    print("\n[1/4] Replaying filtering steps...")
    counts = replay_filtering_steps(args.db_path)

    # Step 2: Load multi-tier group counts
    print("\n[2/4] Loading multi-tier group counts...")
    tier_counts = load_multi_tier_counts(args.base_run_dir)
    print(f"  Tier 1: {tier_counts['tier1']} groups")
    print(f"  Tier 2: {tier_counts['tier2']} groups")
    print(f"  Tier 3: {tier_counts['tier3']} groups")
    print(f"  Source: {tier_counts['source']}")
    counts['tier_counts'] = tier_counts

    # Step 3: Generate diagrams
    print("\n[3/5] Generating STROBE diagram (color)...")
    dot_source = generate_strobe_dot(counts, tier_counts)
    render_diagram(dot_source, output_dir)

    print("\n[4/5] Generating STROBE diagram (simple B&W)...")
    dot_simple = generate_strobe_dot_simple(counts, tier_counts)
    render_diagram(dot_simple, output_dir, prefix="strobe_cohort_flow_diagram_simple")

    # Step 5: Save counts
    print("\n[5/5] Saving audit data...")
    save_counts_json(counts, output_dir)
    save_counts_csv(counts, output_dir)

    print("\n" + "=" * 60)
    print("STROBE DIAGRAM GENERATION COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
