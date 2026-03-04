"""
OncoTree Grouping Analysis for Actionable Variants

This script analyzes parent-child relationships in OncoTree to identify optimal
cancer type groupings that maximize the number of cancer types with ≥250 actionable
patients for ML modeling.

Usage: python oncotree_grouping_analysis.py
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from db_config import get_default_db_path
from collections import defaultdict

# Configuration
DB_PATH = get_default_db_path()
DEFAULT_OUTPUT_DIR = Path("../../analysis_output/oncotree_grouping_analysis")
OUTPUT_DIR = None  # Set by main()

# Threshold for modeling eligibility
ACTIONABLE_THRESHOLD = 250


def load_patient_data(db_path):
    """
    Phase 1.1-1.2: Load patient data and calculate actionability metrics
    """
    print("="*80)
    print("PHASE 1: DATA LOADING AND INITIAL ASSESSMENT")
    print("="*80)
    print(f"\nStep 1.1: Loading patient data from {db_path}")

    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        ONCOTREE_CODE,
        isOncoKbLevel1
    FROM perPatientOncoKBLevelSorted
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(df):,} patient records")

    # Calculate metrics per cancer type
    print("\nStep 1.2: Calculating actionability metrics per cancer type")

    metrics = df.groupby('ONCOTREE_CODE').agg(
        total_patients=('isOncoKbLevel1', 'count'),
        actionable_patients=('isOncoKbLevel1', 'sum')
    ).reset_index()

    metrics['non_actionable_patients'] = metrics['total_patients'] - metrics['actionable_patients']
    metrics['actionability_rate'] = (metrics['actionable_patients'] / metrics['total_patients'] * 100).round(2)
    metrics['meets_threshold'] = metrics['actionable_patients'] >= ACTIONABLE_THRESHOLD

    # Sort by actionable patients descending
    metrics = metrics.sort_values('actionable_patients', ascending=False)

    print(f"\nTotal unique cancer types: {len(metrics)}")
    print(f"Cancer types with ≥{ACTIONABLE_THRESHOLD} actionable patients: {metrics['meets_threshold'].sum()}")
    print(f"Total patients: {metrics['total_patients'].sum():,}")
    print(f"Total actionable patients: {metrics['actionable_patients'].sum():,}")

    return metrics


def load_oncotree_hierarchy(db_path):
    """
    Phase 1.3: Load OncoTree hierarchy
    """
    print("\nStep 1.3: Loading OncoTree hierarchy")

    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM OncoTreeCodeHierarchy"
    hierarchy_df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"Loaded {len(hierarchy_df):,} OncoTree hierarchy entries")

    return hierarchy_df


def merge_patient_hierarchy(metrics, hierarchy_df):
    """
    Phase 1.4: Merge patient counts with hierarchy
    """
    print("\nStep 1.4: Merging patient data with OncoTree hierarchy")

    merged = metrics.merge(
        hierarchy_df,
        left_on='ONCOTREE_CODE',
        right_on='ot_code',
        how='left'
    )

    print(f"Merged data contains {len(merged)} cancer types")
    print(f"Cancer types without hierarchy info: {merged['ot_name'].isna().sum()}")

    return merged


def build_parent_child_maps(merged_df):
    """
    Phase 2: Build parent-child relationship dictionaries
    """
    print("\n" + "="*80)
    print("PHASE 2: PARENT-CHILD RELATIONSHIP MAPPING")
    print("="*80)

    print("\nStep 2.1-2.3: Building parent-child maps at each level")

    parent_child_maps = {}
    child_parent_maps = {}

    # Build maps for each level
    for level in range(1, 6):  # Levels 1-5 can be parents
        parent_col = f'ot_code_level_{level}'
        child_col = f'ot_code_level_{level + 1}'

        # Create parent → children mapping
        parent_child = defaultdict(list)
        child_parent = {}

        for _, row in merged_df.iterrows():
            if pd.notna(row.get(parent_col)) and pd.notna(row.get(child_col)):
                parent = row[parent_col]
                child = row[child_col]

                if child not in [x[0] for x in parent_child[parent]]:
                    parent_child[parent].append((
                        child,
                        row['total_patients'],
                        row['actionable_patients']
                    ))

                if child not in child_parent:
                    child_parent[child] = parent

        parent_child_maps[level] = dict(parent_child)
        child_parent_maps[level] = child_parent

        print(f"  Level {level} → {level+1}: {len(parent_child)} parents, {len(child_parent)} children")

    return parent_child_maps, child_parent_maps


def analyze_current_codes(merged_df):
    """
    Analyze distribution of codes across levels
    """
    print("\nCurrent code distribution by OncoTree level:")
    print("-" * 40)

    level_dist = merged_df.groupby('oncotree_level').agg(
        num_codes=('ONCOTREE_CODE', 'nunique'),
        total_patients=('total_patients', 'sum'),
        actionable_patients=('actionable_patients', 'sum'),
        meets_threshold=('meets_threshold', 'sum')
    ).reset_index()

    print(level_dist.to_string(index=False))

    return level_dist


def scenario_a_level_rollup(merged_df, parent_child_maps):
    """
    Phase 3.1: Scenario A - Level-based rollup
    """
    print("\n" + "="*80)
    print("PHASE 3: GROUPING SCENARIO GENERATION")
    print("="*80)
    print("\nScenario A: Level-Based Rollup")
    print("-" * 40)

    scenarios = {}

    # A1: Roll Level 4 → Level 3
    print("\nA1: Roll all Level 4 codes to their Level 3 parents")
    a1_groups = rollup_to_level(merged_df, target_level=3)
    scenarios['A1_Level4to3'] = a1_groups
    print_scenario_summary(a1_groups, "A1")

    # A2: Roll Level 3 → Level 2
    print("\nA2: Roll all Level 3 codes to their Level 2 parents")
    a2_groups = rollup_to_level(merged_df, target_level=2)
    scenarios['A2_Level3to2'] = a2_groups
    print_scenario_summary(a2_groups, "A2")

    # A3: Roll Level 2 → Level 1
    print("\nA3: Roll all Level 2 codes to their Level 1 parents")
    a3_groups = rollup_to_level(merged_df, target_level=1)
    scenarios['A3_Level2to1'] = a3_groups
    print_scenario_summary(a3_groups, "A3")

    return scenarios


def rollup_to_level(merged_df, target_level):
    """
    Roll up all codes to their parent at target_level
    """
    grouped_data = []

    for _, row in merged_df.iterrows():
        current_level = row['oncotree_level']

        # Handle missing hierarchy info
        if pd.isna(current_level):
            group_code = row['ONCOTREE_CODE']
            group_name = row['ot_name']
        # Determine which code to use
        elif current_level <= target_level:
            # Already at or above target level, keep as is
            group_code = row['ONCOTREE_CODE']
            group_name = row['ot_name']
        else:
            # Roll up to parent at target level
            parent_col = f'ot_code_level_{target_level}'
            parent_name_col = f'level_{target_level}_disease'
            group_code = row[parent_col] if pd.notna(row.get(parent_col)) else row['ONCOTREE_CODE']
            group_name = row[parent_name_col] if pd.notna(row.get(parent_name_col)) else row['ot_name']

        grouped_data.append({
            'group_code': group_code,
            'group_name': group_name,
            'original_code': row['ONCOTREE_CODE'],
            'original_name': row['ot_name'],
            'original_level': current_level,
            'total_patients': row['total_patients'],
            'actionable_patients': row['actionable_patients']
        })

    grouped_df = pd.DataFrame(grouped_data)

    # Aggregate by group
    result = grouped_df.groupby(['group_code', 'group_name']).agg(
        num_original_codes=('original_code', 'nunique'),
        total_patients=('total_patients', 'sum'),
        actionable_patients=('actionable_patients', 'sum')
    ).reset_index()

    result['actionability_rate'] = (result['actionable_patients'] / result['total_patients'] * 100).round(2)
    result['meets_threshold'] = result['actionable_patients'] >= ACTIONABLE_THRESHOLD
    result = result.sort_values('actionable_patients', ascending=False)

    # Add rollup details
    rollup_details = []
    for group in result['group_code'].unique():
        children = grouped_df[grouped_df['group_code'] == group]['original_code'].tolist()
        rollup_details.append({
            'group_code': group,
            'children_codes': '|'.join(children),
            'num_children': len(children)
        })

    rollup_df = pd.DataFrame(rollup_details)
    result = result.merge(rollup_df, on='group_code', how='left')

    return result


def scenario_b_threshold_rollup(merged_df):
    """
    Phase 3.2: Scenario B - Threshold-based selective rollup
    """
    print("\nScenario B: Threshold-Based Selective Rollup")
    print("-" * 40)

    scenarios = {}

    # B1: Roll up codes with <250 actionable to immediate parent
    print("\nB1: Roll up codes with <250 actionable to immediate parent")
    b1_groups = threshold_rollup(merged_df, actionable_threshold=250)
    scenarios['B1_Under250'] = b1_groups
    print_scenario_summary(b1_groups, "B1")

    # B2: Roll up codes with <100 total patients to parent
    print("\nB2: Roll up codes with <100 total patients to parent")
    b2_groups = threshold_rollup(merged_df, total_threshold=100)
    scenarios['B2_Under100Total'] = b2_groups
    print_scenario_summary(b2_groups, "B2")

    # B3: Iterative rollup until all meet threshold
    print("\nB3: Iterative rollup until all groups have ≥250 actionable")
    b3_groups = iterative_rollup(merged_df, target_actionable=250)
    scenarios['B3_Iterative250'] = b3_groups
    print_scenario_summary(b3_groups, "B3")

    return scenarios


def threshold_rollup(merged_df, actionable_threshold=None, total_threshold=None):
    """
    Roll up codes below threshold to their immediate parent
    """
    result_data = []

    for _, row in merged_df.iterrows():
        # Check if should be rolled up
        should_rollup = False

        if actionable_threshold and row['actionable_patients'] < actionable_threshold:
            should_rollup = True
        elif total_threshold and row['total_patients'] < total_threshold:
            should_rollup = True

        if should_rollup:
            # Find immediate parent (one level up)
            if pd.isna(row['oncotree_level']):
                # No hierarchy info, keep as is
                group_code = row['ONCOTREE_CODE']
                group_name = row['ot_name']
                rolled_up = False
            else:
                current_level = int(row['oncotree_level'])
                if current_level > 1:
                    parent_col = f'ot_code_level_{current_level - 1}'
                    parent_name_col = f'level_{current_level - 1}_disease'
                    group_code = row[parent_col] if pd.notna(row.get(parent_col)) else row['ONCOTREE_CODE']
                    group_name = row[parent_name_col] if pd.notna(row.get(parent_name_col)) else row['ot_name']
                    rolled_up = True
                else:
                    # Already at level 1, can't roll up
                    group_code = row['ONCOTREE_CODE']
                    group_name = row['ot_name']
                    rolled_up = False
        else:
            # Keep as is
            group_code = row['ONCOTREE_CODE']
            group_name = row['ot_name']
            rolled_up = False

        result_data.append({
            'group_code': group_code,
            'group_name': group_name,
            'original_code': row['ONCOTREE_CODE'],
            'original_name': row['ot_name'],
            'was_rolled_up': rolled_up,
            'total_patients': row['total_patients'],
            'actionable_patients': row['actionable_patients']
        })

    result_df = pd.DataFrame(result_data)

    # Aggregate
    result = result_df.groupby(['group_code', 'group_name']).agg(
        num_original_codes=('original_code', 'nunique'),
        num_rolled_up=('was_rolled_up', 'sum'),
        total_patients=('total_patients', 'sum'),
        actionable_patients=('actionable_patients', 'sum')
    ).reset_index()

    result['actionability_rate'] = (result['actionable_patients'] / result['total_patients'] * 100).round(2)
    result['meets_threshold'] = result['actionable_patients'] >= ACTIONABLE_THRESHOLD
    result = result.sort_values('actionable_patients', ascending=False)

    return result


def iterative_rollup(merged_df, target_actionable=250, max_iterations=10):
    """
    Iteratively roll up smallest groups until all meet threshold

    Note: Due to complexity of tracking hierarchy through iterations,
    we'll just do one round of selective rollup for now.
    """
    print(f"  Performing single-pass selective rollup (iterative not yet implemented)")

    # Just do threshold rollup once
    result = threshold_rollup(merged_df, actionable_threshold=target_actionable)

    return result


def scenario_c_tissue_grouping(merged_df):
    """
    Phase 3.3: Scenario C - Tissue/organ based grouping
    """
    print("\nScenario C: Tissue/Organ Grouping")
    print("-" * 40)

    scenarios = {}

    # C1: Group by Level 2 (tissue)
    print("\nC1: Group all codes by Level 2 tissue categories")
    c1_groups = rollup_to_level(merged_df, target_level=2)
    scenarios['C1_ByTissue'] = c1_groups
    print_scenario_summary(c1_groups, "C1")

    # C2: Group by Level 1 (major organ systems)
    print("\nC2: Group all codes by Level 1 organ systems")
    c2_groups = rollup_to_level(merged_df, target_level=1)
    scenarios['C2_ByOrgan'] = c2_groups
    print_scenario_summary(c2_groups, "C2")

    return scenarios


def scenario_d_custom_grouping(merged_df):
    """
    Phase 3.4: Scenario D - Custom hybrid groupings
    """
    print("\nScenario D: Custom Hybrid Grouping")
    print("-" * 40)

    scenarios = {}

    # D1: Keep high-frequency subtypes (≥1000 patients), roll up others
    print("\nD1: Keep high-frequency subtypes (≥1000 total), roll up rare ones")
    d1_groups = hybrid_rollup(merged_df, keep_threshold_total=1000)
    scenarios['D1_KeepFrequent'] = d1_groups
    print_scenario_summary(d1_groups, "D1")

    # D2: Keep subtypes with ≥500 actionable, roll up others
    print("\nD2: Keep subtypes with ≥500 actionable, roll up others")
    d2_groups = hybrid_rollup(merged_df, keep_threshold_actionable=500)
    scenarios['D2_KeepActionable'] = d2_groups
    print_scenario_summary(d2_groups, "D2")

    return scenarios


def hybrid_rollup(merged_df, keep_threshold_total=None, keep_threshold_actionable=None):
    """
    Hybrid approach: keep high-frequency codes, roll up others
    """
    result_data = []

    for _, row in merged_df.iterrows():
        # Decide whether to keep or roll up
        should_keep = False

        if keep_threshold_total and row['total_patients'] >= keep_threshold_total:
            should_keep = True
        elif keep_threshold_actionable and row['actionable_patients'] >= keep_threshold_actionable:
            should_keep = True

        if should_keep:
            group_code = row['ONCOTREE_CODE']
            group_name = row['ot_name']
        else:
            # Roll up to immediate parent
            if pd.isna(row['oncotree_level']):
                group_code = row['ONCOTREE_CODE']
                group_name = row['ot_name']
            else:
                current_level = int(row['oncotree_level'])
                if current_level > 1:
                    parent_col = f'ot_code_level_{current_level - 1}'
                    parent_name_col = f'level_{current_level - 1}_disease'
                    group_code = row[parent_col] if pd.notna(row.get(parent_col)) else row['ONCOTREE_CODE']
                    group_name = row[parent_name_col] if pd.notna(row.get(parent_name_col)) else row['ot_name']
                else:
                    group_code = row['ONCOTREE_CODE']
                    group_name = row['ot_name']

        result_data.append({
            'group_code': group_code,
            'group_name': group_name,
            'original_code': row['ONCOTREE_CODE'],
            'total_patients': row['total_patients'],
            'actionable_patients': row['actionable_patients']
        })

    result_df = pd.DataFrame(result_data)

    result = result_df.groupby(['group_code', 'group_name']).agg(
        num_original_codes=('original_code', 'nunique'),
        total_patients=('total_patients', 'sum'),
        actionable_patients=('actionable_patients', 'sum')
    ).reset_index()

    result['actionability_rate'] = (result['actionable_patients'] / result['total_patients'] * 100).round(2)
    result['meets_threshold'] = result['actionable_patients'] >= ACTIONABLE_THRESHOLD
    result = result.sort_values('actionable_patients', ascending=False)

    return result


def print_scenario_summary(scenario_df, scenario_name):
    """
    Print summary statistics for a scenario
    """
    total_groups = len(scenario_df)
    eligible_groups = scenario_df['meets_threshold'].sum()
    total_patients = scenario_df['total_patients'].sum()
    actionable_patients = scenario_df['actionable_patients'].sum()

    print(f"  Total groups: {total_groups}")
    print(f"  Groups with ≥{ACTIONABLE_THRESHOLD} actionable: {eligible_groups}")
    print(f"  Total patients: {total_patients:,}")
    print(f"  Total actionable: {actionable_patients:,}")


def compare_scenarios(baseline_df, all_scenarios):
    """
    Phase 4: Compare all scenarios
    """
    print("\n" + "="*80)
    print("PHASE 4: SCENARIO COMPARISON")
    print("="*80)

    baseline_eligible = baseline_df['meets_threshold'].sum()
    baseline_patients = baseline_df[baseline_df['meets_threshold']]['total_patients'].sum()

    comparison_data = []

    # Add baseline
    comparison_data.append({
        'Scenario': 'Baseline (Current)',
        'Total_Groups': len(baseline_df),
        'Eligible_Groups': baseline_eligible,
        'Eligible_Patients': baseline_patients,
        'Eligible_Actionable': baseline_df[baseline_df['meets_threshold']]['actionable_patients'].sum(),
        'Gain_vs_Baseline': 0
    })

    # Add all scenarios
    for scenario_name, scenario_df in all_scenarios.items():
        eligible = scenario_df['meets_threshold'].sum()
        eligible_patients = scenario_df[scenario_df['meets_threshold']]['total_patients'].sum()
        eligible_actionable = scenario_df[scenario_df['meets_threshold']]['actionable_patients'].sum()

        comparison_data.append({
            'Scenario': scenario_name,
            'Total_Groups': len(scenario_df),
            'Eligible_Groups': eligible,
            'Eligible_Patients': eligible_patients,
            'Eligible_Actionable': eligible_actionable,
            'Gain_vs_Baseline': eligible - baseline_eligible
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Eligible_Groups', ascending=False)

    print("\nScenario Comparison Table:")
    print("="*80)
    print(comparison_df.to_string(index=False))

    return comparison_df


def generate_detailed_mappings(merged_df, scenario_df, scenario_name):
    """
    Phase 5: Generate detailed mapping of which children go to which parents
    """
    mapping_data = []

    # For scenarios with rollup info
    if 'children_codes' in scenario_df.columns:
        for _, row in scenario_df.iterrows():
            if row['num_original_codes'] > 1:
                # This is a rolled-up group
                children = row['children_codes'].split('|')
                for child in children:
                    # Get child details from original data
                    child_info = merged_df[merged_df['ONCOTREE_CODE'] == child].iloc[0]
                    mapping_data.append({
                        'scenario': scenario_name,
                        'parent_code': row['group_code'],
                        'parent_name': row['group_name'],
                        'child_code': child,
                        'child_name': child_info['ot_name'],
                        'child_level': child_info['oncotree_level'],
                        'child_total_patients': child_info['total_patients'],
                        'child_actionable': child_info['actionable_patients'],
                        'parent_total_patients': row['total_patients'],
                        'parent_actionable': row['actionable_patients']
                    })

    return pd.DataFrame(mapping_data) if mapping_data else None


def create_visualizations(baseline_df, comparison_df, all_scenarios):
    """
    Phase 6: Create visualizations
    """
    print("\n" + "="*80)
    print("PHASE 6: VISUALIZATION GENERATION")
    print("="*80)

    # Figure 1: Scenario comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1a: Eligible groups comparison
    ax1 = axes[0, 0]
    scenarios = comparison_df['Scenario'].tolist()
    eligible = comparison_df['Eligible_Groups'].tolist()
    colors = ['red' if s == 'Baseline (Current)' else 'steelblue' for s in scenarios]

    ax1.barh(scenarios, eligible, color=colors)
    ax1.set_xlabel('Number of Eligible Cancer Types', fontsize=12)
    ax1.set_title(f'Cancer Types with ≥{ACTIONABLE_THRESHOLD} Actionable Patients', fontsize=14, fontweight='bold')
    ax1.axvline(x=comparison_df[comparison_df['Scenario'] == 'Baseline (Current)']['Eligible_Groups'].iloc[0],
                color='red', linestyle='--', linewidth=2, label='Baseline')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # 1b: Gain vs baseline
    ax2 = axes[0, 1]
    gains = comparison_df[comparison_df['Scenario'] != 'Baseline (Current)']['Gain_vs_Baseline'].tolist()
    scenarios_no_baseline = comparison_df[comparison_df['Scenario'] != 'Baseline (Current)']['Scenario'].tolist()
    colors_gain = ['green' if g > 0 else 'gray' for g in gains]

    ax2.barh(scenarios_no_baseline, gains, color=colors_gain)
    ax2.set_xlabel('Gain in Eligible Cancer Types', fontsize=12)
    ax2.set_title('Net Gain vs. Baseline', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

    # 1c: Total eligible patients
    ax3 = axes[1, 0]
    eligible_patients = comparison_df['Eligible_Patients'].tolist()
    colors = ['red' if s == 'Baseline (Current)' else 'seagreen' for s in scenarios]

    ax3.barh(scenarios, eligible_patients, color=colors)
    ax3.set_xlabel('Total Eligible Patients', fontsize=12)
    ax3.set_title('Total Patients in Eligible Cancer Types', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # Format x-axis with commas
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # 1d: Top 15 cancer types from best scenario
    ax4 = axes[1, 1]
    best_scenario_name = comparison_df.sort_values('Eligible_Groups', ascending=False).iloc[0]['Scenario']
    if best_scenario_name != 'Baseline (Current)':
        best_scenario = all_scenarios[best_scenario_name]
        top_15 = best_scenario[best_scenario['meets_threshold']].head(15).sort_values('actionable_patients')

        ax4.barh(range(len(top_15)), top_15['actionable_patients'], color='coral')
        ax4.set_yticks(range(len(top_15)))
        ax4.set_yticklabels(top_15['group_code'], fontsize=9)
        ax4.set_xlabel('Actionable Patients', fontsize=12)
        ax4.set_title(f'Top 15 Eligible Types ({best_scenario_name})', fontsize=14, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)

        # Add counts
        for i, v in enumerate(top_15['actionable_patients']):
            ax4.text(v + 20, i, f'{int(v):,}', va='center', fontsize=8)

    plt.tight_layout()

    fig_file = OUTPUT_DIR / "oncotree_grouping_comparison.pdf"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.savefig(fig_file.with_suffix('.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison visualization saved: {fig_file}")
    plt.close()


def generate_recommendations(baseline_df, all_scenarios, comparison_df):
    """
    Generate final recommendations
    """
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Find best scenario
    best_scenario_name = comparison_df.sort_values('Eligible_Groups', ascending=False).iloc[0]['Scenario']
    best_scenario = all_scenarios.get(best_scenario_name)

    baseline_eligible = baseline_df['meets_threshold'].sum()
    best_eligible = comparison_df[comparison_df['Scenario'] == best_scenario_name]['Eligible_Groups'].iloc[0]
    gain = best_eligible - baseline_eligible

    print(f"\nBest Grouping Strategy: {best_scenario_name}")
    print(f"  Current eligible cancer types: {baseline_eligible}")
    print(f"  Projected eligible cancer types: {best_eligible}")
    print(f"  Net gain: +{gain} cancer types ({100*gain/baseline_eligible:.1f}% increase)")

    if best_scenario is not None:
        print(f"\n  Newly eligible cancer types:")
        newly_eligible = best_scenario[best_scenario['meets_threshold']].copy()

        # Check which are new vs. already eligible
        baseline_eligible_codes = baseline_df[baseline_df['meets_threshold']]['ONCOTREE_CODE'].tolist()

        for _, row in newly_eligible.iterrows():
            if row['group_code'] not in baseline_eligible_codes:
                print(f"    - {row['group_code']}: {row['group_name']}")
                print(f"      {int(row['actionable_patients'])} actionable / {int(row['total_patients'])} total ({row['actionability_rate']:.1f}%)")
                if row.get('num_original_codes', 1) > 1:
                    print(f"      (Rolled up from {int(row['num_original_codes'])} subtypes)")

    recommendations = {
        'best_scenario': best_scenario_name,
        'baseline_eligible': baseline_eligible,
        'projected_eligible': best_eligible,
        'net_gain': gain,
        'percent_increase': 100 * gain / baseline_eligible
    }

    return recommendations


def save_outputs(baseline_df, all_scenarios, comparison_df, merged_df):
    """
    Phase 7: Save all output files
    """
    print("\n" + "="*80)
    print("PHASE 7: SAVING OUTPUT FILES")
    print("="*80)

    # Save baseline
    baseline_file = OUTPUT_DIR / "oncotree_patient_counts_current.csv"
    baseline_df.to_csv(baseline_file, index=False)
    print(f"\n✓ Baseline counts: {baseline_file}")

    # Save each scenario
    for scenario_name, scenario_df in all_scenarios.items():
        scenario_file = OUTPUT_DIR / f"oncotree_grouping_{scenario_name}.csv"
        scenario_df.to_csv(scenario_file, index=False)
        print(f"✓ Scenario {scenario_name}: {scenario_file}")

    # Save comparison
    comparison_file = OUTPUT_DIR / "oncotree_grouping_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"✓ Comparison table: {comparison_file}")

    # Save detailed mappings for key scenarios
    for scenario_name in ['A2_Level3to2', 'B1_Under250', 'D1_KeepFrequent']:
        if scenario_name in all_scenarios:
            mapping_df = generate_detailed_mappings(merged_df, all_scenarios[scenario_name], scenario_name)
            if mapping_df is not None and len(mapping_df) > 0:
                mapping_file = OUTPUT_DIR / f"oncotree_mapping_details_{scenario_name}.csv"
                mapping_df.to_csv(mapping_file, index=False)
                print(f"✓ Mapping details {scenario_name}: {mapping_file}")


def main():
    """Main execution"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='OncoTree grouping analysis')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    args = parser.parse_args()

    # Set output directory
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("OncoTree Grouping Analysis for Actionable Variants")
    print("="*80)

    # Check database
    if not Path(DB_PATH).exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return 1

    # Phase 1: Load data
    metrics = load_patient_data(DB_PATH)
    hierarchy_df = load_oncotree_hierarchy(DB_PATH)
    merged_df = merge_patient_hierarchy(metrics, hierarchy_df)

    # Phase 2: Build parent-child maps
    parent_child_maps, child_parent_maps = build_parent_child_maps(merged_df)
    level_dist = analyze_current_codes(merged_df)

    # Phase 3: Generate scenarios
    all_scenarios = {}

    scenarios_a = scenario_a_level_rollup(merged_df, parent_child_maps)
    all_scenarios.update(scenarios_a)

    scenarios_b = scenario_b_threshold_rollup(merged_df)
    all_scenarios.update(scenarios_b)

    scenarios_c = scenario_c_tissue_grouping(merged_df)
    all_scenarios.update(scenarios_c)

    scenarios_d = scenario_d_custom_grouping(merged_df)
    all_scenarios.update(scenarios_d)

    # Phase 4: Compare scenarios
    comparison_df = compare_scenarios(metrics, all_scenarios)

    # Phase 5-6: Visualizations
    create_visualizations(metrics, comparison_df, all_scenarios)

    # Generate recommendations
    recommendations = generate_recommendations(metrics, all_scenarios, comparison_df)

    # Phase 7: Save outputs
    save_outputs(metrics, all_scenarios, comparison_df, merged_df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
