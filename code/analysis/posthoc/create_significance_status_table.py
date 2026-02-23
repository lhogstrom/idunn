#!/usr/bin/env python3
"""
Create Gene-Stratified Significance Status Summary Table and Heatmap

Produces:
1. gene_stratified_summary_significant_status.csv - Compact table with significance,
   directionality, and effect size for each association
2. fig6_significance_status_heatmap.pdf - Visual matrix of significant associations

Usage:
    python create_significance_status_table.py \
        --input-dir /path/to/gene_stratified \
        --output-dir /path/to/gene_stratified
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_default_input_dir():
    """Find gene-stratified results directory from latest run."""
    runs_dir = REPO_ROOT / "runs"
    if runs_dir.exists():
        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        for run_dir in runs:
            gene_dir = run_dir / "tier7_posthoc_statistics" / "gene_stratified"
            if gene_dir.exists() and (gene_dir / "gene_stratified_significant_hits.csv").exists():
                return gene_dir
    return None


def load_all_results(input_dir: Path) -> dict:
    """Load per-feature result files."""
    data = {}
    for feature in ['age', 'sex', 'race', 'ethnicity', 'sample_type', 'technical']:
        subdir = f"{feature}_by_gene"
        results_file = input_dir / subdir / f"{feature}_gene_results.csv"
        if results_file.exists():
            data[feature] = pd.read_csv(results_file)
    return data


def get_direction_label(row, feature):
    """Determine direction label for a significant result."""
    if feature == 'age':
        d = row.get('cohens_d', 0)
        if d < 0:
            return 'younger'
        else:
            return 'older'
    elif feature == 'sex':
        or_val = row.get('odds_ratio', 1)
        if or_val > 1:
            return 'higher_female'
        else:
            return 'higher_male'
    else:
        # Categorical features - no clear direction
        return 'association'


def get_effect_size(row, feature):
    """Extract the primary effect size measure."""
    if feature == 'age':
        return row.get('cohens_d', np.nan)
    elif feature == 'sex':
        return row.get('odds_ratio', np.nan)
    elif feature in ['race', 'ethnicity', 'sample_type']:
        return row.get('cramers_v', np.nan)
    elif feature == 'technical':
        # Technical has multiple sub-features
        tech_feature = row.get('feature', '')
        if tech_feature == 'number_of_genes':
            return row.get('effect_size_r', np.nan)
        else:
            return row.get('cramers_v', np.nan)
    return np.nan


def get_effect_type(feature, tech_feature=None):
    """Get the effect size type label."""
    if feature == 'age':
        return "Cohen's d"
    elif feature == 'sex':
        return "Odds Ratio"
    elif feature == 'technical' and tech_feature == 'number_of_genes':
        return "Effect size r"
    else:
        return "Cramer's V"


def create_status_table(data: dict) -> pd.DataFrame:
    """
    Create the significance status table.

    Each row = cancer_type + gene combination.
    Columns: significance (bool), direction, effect size for each feature.
    """
    rows = []

    # Collect all cancer-gene combinations from age (most complete)
    if 'age' not in data:
        print("ERROR: No age data")
        return pd.DataFrame()

    # Get unique cancer-gene pairs across all features
    all_pairs = set()
    for feature, df in data.items():
        if feature == 'technical':
            for _, r in df.iterrows():
                all_pairs.add((r['ONCOTREE_CODE'], r['Hugo_Symbol'], int(r['gene_rank'])))
        else:
            for _, r in df.iterrows():
                all_pairs.add((r['ONCOTREE_CODE'], r['Hugo_Symbol'], int(r['gene_rank'])))

    # Sort pairs
    all_pairs = sorted(all_pairs, key=lambda x: (x[0], x[2], x[1]))

    # Demographic features (one column set each)
    demo_features = ['age', 'sex', 'race', 'ethnicity', 'sample_type']

    # Technical sub-features
    tech_subfeatures = ['calling_strategy', 'SUBJECT_CENTER', 'number_of_genes']

    for cancer, gene, rank in all_pairs:
        row = {
            'cancer_type': cancer,
            'gene': gene,
            'gene_rank': rank,
        }

        # Add n_patients from age data if available
        if 'age' in data:
            age_match = data['age'][
                (data['age']['ONCOTREE_CODE'] == cancer) &
                (data['age']['Hugo_Symbol'] == gene)
            ]
            if len(age_match) > 0:
                row['n_gene_patients'] = age_match.iloc[0].get('n_gene_patients',
                                         age_match.iloc[0].get('n_actionable', np.nan))

        # Demographic features
        for feature in demo_features:
            if feature not in data:
                row[f'{feature}_sig'] = np.nan
                row[f'{feature}_direction'] = ''
                row[f'{feature}_effect_size'] = np.nan
                row[f'{feature}_effect_type'] = ''
                continue

            df = data[feature]
            match = df[(df['ONCOTREE_CODE'] == cancer) & (df['Hugo_Symbol'] == gene)]

            if len(match) == 0:
                # Not tested (SHAP-filtered out)
                row[f'{feature}_sig'] = 'not_tested'
                row[f'{feature}_direction'] = ''
                row[f'{feature}_effect_size'] = np.nan
                row[f'{feature}_effect_type'] = ''
            else:
                r = match.iloc[0]
                is_sig = r.get('significant_fdr', False)
                row[f'{feature}_sig'] = is_sig
                row[f'{feature}_direction'] = get_direction_label(r, feature) if is_sig else 'ns'
                row[f'{feature}_effect_size'] = get_effect_size(r, feature)
                row[f'{feature}_effect_type'] = get_effect_type(feature)

        # Technical sub-features
        if 'technical' in data:
            tech_df = data['technical']
            tech_match = tech_df[
                (tech_df['ONCOTREE_CODE'] == cancer) & (tech_df['Hugo_Symbol'] == gene)
            ]

            for tech_feat in tech_subfeatures:
                tech_row = tech_match[tech_match['feature'] == tech_feat] if len(tech_match) > 0 else pd.DataFrame()

                if len(tech_row) == 0:
                    row[f'tech_{tech_feat}_sig'] = 'not_tested'
                    row[f'tech_{tech_feat}_direction'] = ''
                    row[f'tech_{tech_feat}_effect_size'] = np.nan
                    row[f'tech_{tech_feat}_effect_type'] = ''
                else:
                    r = tech_row.iloc[0]
                    is_sig = r.get('significant_fdr', False)
                    row[f'tech_{tech_feat}_sig'] = is_sig
                    row[f'tech_{tech_feat}_direction'] = 'association' if is_sig else 'ns'
                    row[f'tech_{tech_feat}_effect_size'] = get_effect_size(r, 'technical')
                    row[f'tech_{tech_feat}_effect_type'] = get_effect_type('technical', tech_feat)
        else:
            for tech_feat in tech_subfeatures:
                row[f'tech_{tech_feat}_sig'] = 'not_tested'
                row[f'tech_{tech_feat}_direction'] = ''
                row[f'tech_{tech_feat}_effect_size'] = np.nan
                row[f'tech_{tech_feat}_effect_type'] = ''

        rows.append(row)

    return pd.DataFrame(rows)


def create_heatmap(status_df: pd.DataFrame, output_dir: Path):
    """
    Create a significance status heatmap.

    Rows: cancer_type (gene)
    Columns: feature categories
    Color: direction (blue=younger/higher_male, red=older/higher_female, gray=association)
    Markers: significant vs not significant
    """
    # Build row labels
    status_df = status_df.copy()
    status_df['row_label'] = status_df['cancer_type'] + '\n' + status_df['gene']

    # Define columns to show
    feature_cols = [
        ('age_sig', 'age_direction', 'age_effect_size', 'Age'),
        ('sex_sig', 'sex_direction', 'sex_effect_size', 'Sex'),
        ('race_sig', 'race_direction', 'race_effect_size', 'Race'),
        ('ethnicity_sig', 'ethnicity_direction', 'ethnicity_effect_size', 'Ethnicity'),
        ('sample_type_sig', 'sample_type_direction', 'sample_type_effect_size', 'Sample\nType'),
        ('tech_calling_strategy_sig', 'tech_calling_strategy_direction', 'tech_calling_strategy_effect_size', 'Calling\nStrategy'),
        ('tech_SUBJECT_CENTER_sig', 'tech_SUBJECT_CENTER_direction', 'tech_SUBJECT_CENTER_effect_size', 'Subject\nCenter'),
        ('tech_number_of_genes_sig', 'tech_number_of_genes_direction', 'tech_number_of_genes_effect_size', '#Genes'),
    ]

    n_rows = len(status_df)
    n_cols = len(feature_cols)

    # Create matrix for plotting
    # Encode: 0=not tested, 1=not significant, 2=sig positive/older/higher_female/assoc,
    #         -2=sig negative/younger/higher_male
    matrix = np.zeros((n_rows, n_cols))
    effect_matrix = np.full((n_rows, n_cols), np.nan)

    for j, (sig_col, dir_col, eff_col, _) in enumerate(feature_cols):
        for i in range(n_rows):
            sig_val = status_df.iloc[i].get(sig_col, 'not_tested')
            direction = status_df.iloc[i].get(dir_col, '')
            effect = status_df.iloc[i].get(eff_col, np.nan)

            if sig_val == 'not_tested' or (isinstance(sig_val, float) and np.isnan(sig_val)):
                matrix[i, j] = 0  # Not tested
            elif sig_val == True or sig_val == 'True':
                effect_matrix[i, j] = effect if not (isinstance(effect, float) and np.isnan(effect)) else 0

                if direction in ['younger', 'higher_male']:
                    matrix[i, j] = -2  # Negative direction
                elif direction in ['older', 'higher_female']:
                    matrix[i, j] = 2   # Positive direction
                else:
                    matrix[i, j] = 1.5   # Association (no direction)
            else:
                matrix[i, j] = 0.5  # Tested but not significant

    # Create figure
    fig_height = max(6, n_rows * 0.45 + 2)
    fig_width = max(8, n_cols * 1.2 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Custom colormap
    # not_tested=white, ns=light gray, sig_neg=blue, sig_assoc=orange, sig_pos=red
    from matplotlib.colors import ListedColormap, BoundaryNorm

    colors = ['#FFFFFF',    # 0: not tested (white)
              '#E8E8E8',    # 0.5: tested, not significant (light gray)
              '#FFA500',    # 1.5: significant, association/no direction (orange)
              '#4393C3',    # -2: significant, younger/higher_male (blue)
              '#D6604D',    # 2: significant, older/higher_female (red)
              ]

    # Plot each cell individually for precise control
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if val == 0:
                color = '#FFFFFF'
                edge = '#CCCCCC'
            elif val == 0.5:
                color = '#E8E8E8'
                edge = '#AAAAAA'
            elif val == -2:
                color = '#4393C3'
                edge = '#2166AC'
            elif val == 2:
                color = '#D6604D'
                edge = '#B2182B'
            elif val == 1.5:
                color = '#FFA500'
                edge = '#CC8400'
            else:
                color = '#FFFFFF'
                edge = '#CCCCCC'

            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                facecolor=color, edgecolor=edge, linewidth=0.5)
            ax.add_patch(rect)

            # Add effect size text for significant results
            if abs(val) >= 1.5:
                eff = effect_matrix[i, j]
                if not np.isnan(eff):
                    # Format effect size
                    if abs(eff) >= 10:
                        txt = f'{eff:.1f}'
                    elif abs(eff) >= 1:
                        txt = f'{eff:.2f}'
                    else:
                        txt = f'{eff:.3f}'
                    ax.text(j, i, txt, ha='center', va='center',
                           fontsize=6, fontweight='bold', color='white')

    # Set axis properties
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    # Row labels
    row_labels = status_df['row_label'].tolist()
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Column labels
    col_labels = [fc[3] for fc in feature_cols]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8, ha='center')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Add vertical line to separate demographic from technical
    ax.axvline(x=4.5, color='black', linewidth=1.5)
    ax.text(2, -1.2, 'Demographic Features', ha='center', fontsize=9, fontweight='bold')
    ax.text(6, -1.2, 'Technical Features', ha='center', fontsize=9, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#D6604D', edgecolor='#B2182B', label='Sig: older / higher female'),
        mpatches.Patch(facecolor='#4393C3', edgecolor='#2166AC', label='Sig: younger / higher male'),
        mpatches.Patch(facecolor='#FFA500', edgecolor='#CC8400', label='Sig: association (categorical)'),
        mpatches.Patch(facecolor='#E8E8E8', edgecolor='#AAAAAA', label='Tested, not significant'),
        mpatches.Patch(facecolor='#FFFFFF', edgecolor='#CCCCCC', label='Not tested (SHAP-filtered)'),
    ]
    ax.legend(handles=legend_elements, loc='lower center',
             bbox_to_anchor=(0.5, -0.02 - 0.15 * (n_rows > 15)),
             ncol=2, fontsize=7, frameon=True)

    ax.set_title('Gene-Stratified Association Significance Status\n(FDR < 0.05, effect sizes shown)',
                fontsize=11, fontweight='bold', pad=30)

    plt.tight_layout()

    # Save
    for ext in ['pdf', 'png']:
        outfile = output_dir / f'fig6_significance_status_heatmap.{ext}'
        fig.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Saved: fig6_significance_status_heatmap.pdf")


def main():
    parser = argparse.ArgumentParser(
        description='Create significance status summary table and heatmap')
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Directory containing gene-stratified results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as input)')
    args = parser.parse_args()

    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = get_default_input_dir()
        if input_dir is None:
            print("ERROR: No gene-stratified results found.")
            return 1

    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load results
    print("\nLoading gene-stratified results...")
    data = load_all_results(input_dir)
    print(f"  Loaded features: {list(data.keys())}")

    # Create status table
    print("\nCreating significance status table...")
    status_df = create_status_table(data)

    # Save CSV
    csv_file = output_dir / 'gene_stratified_summary_significant_status.csv'
    status_df.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file}")
    print(f"  Rows: {len(status_df)}, Columns: {len(status_df.columns)}")

    # Create heatmap
    print("\nCreating significance status heatmap...")
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    create_heatmap(status_df, viz_dir)

    print(f"\nDone. All output saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
