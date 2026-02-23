#!/usr/bin/env python3
"""
Summarize Gene-Stratified Association Results

Creates a summary CSV table consolidating all gene-stratified association results.
Each cancer type is one row, with association details as columns.

Output columns include:
- Cancer type info (ONCOTREE_CODE, gene1_name, gene2_name, sample sizes)
- Age associations (Cohen's d, p-value, significance for each gene)
- Sex associations (OR, p-value, significance for each gene)
- Race associations (Cramer's V, p-value, significance)
- Ethnicity associations
- Sample type associations
- Technical feature associations (calling_strategy, SUBJECT_CENTER, number_of_genes)

Usage:
    # Auto-detect input from latest run
    python summarize_gene_stratified_results.py

    # Specify input directory
    python summarize_gene_stratified_results.py --input-dir /path/to/gene_stratified

    # Specify output file
    python summarize_gene_stratified_results.py --output gene_stratified_summary.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_default_input_dir():
    """Find gene-stratified results directory."""
    runs_dir = REPO_ROOT / "runs"
    if runs_dir.exists():
        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        for run_dir in runs:
            gene_dir = run_dir / "tier7_posthoc_statistics" / "gene_stratified"
            if gene_dir.exists():
                return gene_dir
    return None


def load_results(input_dir: Path) -> dict:
    """Load all gene-stratified result files."""
    data = {}

    # Age results
    age_file = input_dir / "age_by_gene" / "age_gene_results.csv"
    if age_file.exists():
        data['age'] = pd.read_csv(age_file)
        print(f"Loaded age: {len(data['age'])} rows")

    # Sex results
    sex_file = input_dir / "sex_by_gene" / "sex_gene_results.csv"
    if sex_file.exists():
        data['sex'] = pd.read_csv(sex_file)
        print(f"Loaded sex: {len(data['sex'])} rows")

    # Race results
    race_file = input_dir / "race_by_gene" / "race_gene_results.csv"
    if race_file.exists():
        data['race'] = pd.read_csv(race_file)
        print(f"Loaded race: {len(data['race'])} rows")

    # Ethnicity results
    eth_file = input_dir / "ethnicity_by_gene" / "ethnicity_gene_results.csv"
    if eth_file.exists():
        data['ethnicity'] = pd.read_csv(eth_file)
        print(f"Loaded ethnicity: {len(data['ethnicity'])} rows")

    # Sample type results
    sample_file = input_dir / "sample_type_by_gene" / "sample_type_gene_results.csv"
    if sample_file.exists():
        data['sample_type'] = pd.read_csv(sample_file)
        print(f"Loaded sample_type: {len(data['sample_type'])} rows")

    # Technical results
    tech_file = input_dir / "technical_by_gene" / "technical_gene_results.csv"
    if tech_file.exists():
        data['technical'] = pd.read_csv(tech_file)
        print(f"Loaded technical: {len(data['technical'])} rows")

    # Top 2 genes summary
    top2_file = input_dir / "gene_distributions" / "top2_genes_summary.csv"
    if top2_file.exists():
        data['top2'] = pd.read_csv(top2_file)
        print(f"Loaded top2 genes: {len(data['top2'])} rows")

    return data


def create_summary_table(data: dict) -> pd.DataFrame:
    """
    Create wide-format summary table with cancer types as rows.

    Returns DataFrame with one row per cancer type and columns for all associations.
    """
    # Get list of cancer types from age data (most complete)
    if 'age' not in data:
        print("ERROR: No age data found")
        return pd.DataFrame()

    # Get unique cancer types
    cancer_types = data['age']['ONCOTREE_CODE'].unique()

    # Initialize summary dict
    summary_rows = []

    for cancer in cancer_types:
        row = {'ONCOTREE_CODE': cancer}

        # === Age associations ===
        age_df = data['age'][data['age']['ONCOTREE_CODE'] == cancer]
        for _, age_row in age_df.iterrows():
            gene = age_row['Hugo_Symbol']
            rank = int(age_row['gene_rank'])
            prefix = f"gene{rank}"

            row[f'{prefix}_name'] = gene
            row[f'{prefix}_n_patients'] = age_row.get('n_gene_patients', np.nan)
            row[f'{prefix}_n_actionable'] = age_row.get('n_actionable', np.nan)
            row[f'{prefix}_age_cohens_d'] = age_row.get('cohens_d', np.nan)
            row[f'{prefix}_age_mean_diff'] = age_row.get('mean_age_difference', np.nan)
            row[f'{prefix}_age_p_fdr'] = age_row.get('p_fdr', np.nan)
            row[f'{prefix}_age_sig'] = age_row.get('significant_fdr', False)

        # === Sex associations ===
        if 'sex' in data:
            sex_df = data['sex'][data['sex']['ONCOTREE_CODE'] == cancer]
            for _, sex_row in sex_df.iterrows():
                gene = sex_row['Hugo_Symbol']
                rank = int(sex_row['gene_rank'])
                prefix = f"gene{rank}"

                row[f'{prefix}_sex_OR'] = sex_row.get('odds_ratio', np.nan)
                row[f'{prefix}_sex_OR_ci_lower'] = sex_row.get('or_ci_lower', np.nan)
                row[f'{prefix}_sex_OR_ci_upper'] = sex_row.get('or_ci_upper', np.nan)
                row[f'{prefix}_sex_p_fdr'] = sex_row.get('p_fdr', np.nan)
                row[f'{prefix}_sex_sig'] = sex_row.get('significant_fdr', False)

        # === Race associations ===
        if 'race' in data:
            race_df = data['race'][data['race']['ONCOTREE_CODE'] == cancer]
            for _, race_row in race_df.iterrows():
                gene = race_row['Hugo_Symbol']
                rank = int(race_row['gene_rank'])
                prefix = f"gene{rank}"

                row[f'{prefix}_race_cramers_v'] = race_row.get('cramers_v', np.nan)
                row[f'{prefix}_race_p_fdr'] = race_row.get('p_fdr', np.nan)
                row[f'{prefix}_race_sig'] = race_row.get('significant_fdr', False)

        # === Ethnicity associations ===
        if 'ethnicity' in data:
            eth_df = data['ethnicity'][data['ethnicity']['ONCOTREE_CODE'] == cancer]
            for _, eth_row in eth_df.iterrows():
                gene = eth_row['Hugo_Symbol']
                rank = int(eth_row['gene_rank'])
                prefix = f"gene{rank}"

                row[f'{prefix}_ethnicity_cramers_v'] = eth_row.get('cramers_v', np.nan)
                row[f'{prefix}_ethnicity_p_fdr'] = eth_row.get('p_fdr', np.nan)
                row[f'{prefix}_ethnicity_sig'] = eth_row.get('significant_fdr', False)

        # === Sample type associations ===
        if 'sample_type' in data:
            sample_df = data['sample_type'][data['sample_type']['ONCOTREE_CODE'] == cancer]
            for _, sample_row in sample_df.iterrows():
                gene = sample_row['Hugo_Symbol']
                rank = int(sample_row['gene_rank'])
                prefix = f"gene{rank}"

                row[f'{prefix}_sample_type_cramers_v'] = sample_row.get('cramers_v', np.nan)
                row[f'{prefix}_sample_type_p_fdr'] = sample_row.get('p_fdr', np.nan)
                row[f'{prefix}_sample_type_sig'] = sample_row.get('significant_fdr', False)

        # === Technical associations ===
        if 'technical' in data:
            tech_df = data['technical'][data['technical']['ONCOTREE_CODE'] == cancer]
            for _, tech_row in tech_df.iterrows():
                gene = tech_row['Hugo_Symbol']
                rank = int(tech_row['gene_rank'])
                feature = tech_row.get('feature', 'unknown')
                prefix = f"gene{rank}"

                # Create column name based on feature
                feat_short = feature.lower().replace('subject_center', 'center')

                if feature == 'number_of_genes':
                    # Numeric feature - use effect_size_r
                    row[f'{prefix}_tech_{feat_short}_effect_r'] = tech_row.get('effect_size_r', np.nan)
                else:
                    # Categorical feature - use cramers_v
                    row[f'{prefix}_tech_{feat_short}_cramers_v'] = tech_row.get('cramers_v', np.nan)

                row[f'{prefix}_tech_{feat_short}_p_fdr'] = tech_row.get('p_fdr', np.nan)
                row[f'{prefix}_tech_{feat_short}_sig'] = tech_row.get('significant_fdr', False)

        summary_rows.append(row)

    # Create DataFrame
    summary_df = pd.DataFrame(summary_rows)

    # Reorder columns for readability
    # First: cancer type and gene info
    # Then: demographic associations
    # Finally: technical associations

    base_cols = ['ONCOTREE_CODE']
    gene1_info = ['gene1_name', 'gene1_n_patients', 'gene1_n_actionable']
    gene2_info = ['gene2_name', 'gene2_n_patients', 'gene2_n_actionable']

    # Demographic columns for gene1
    gene1_demo = [c for c in summary_df.columns if c.startswith('gene1_') and
                  any(x in c for x in ['age', 'sex', 'race', 'ethnicity', 'sample_type'])
                  and 'tech' not in c]
    gene1_demo = sorted(gene1_demo)

    # Demographic columns for gene2
    gene2_demo = [c for c in summary_df.columns if c.startswith('gene2_') and
                  any(x in c for x in ['age', 'sex', 'race', 'ethnicity', 'sample_type'])
                  and 'tech' not in c]
    gene2_demo = sorted(gene2_demo)

    # Technical columns
    gene1_tech = [c for c in summary_df.columns if c.startswith('gene1_tech')]
    gene1_tech = sorted(gene1_tech)
    gene2_tech = [c for c in summary_df.columns if c.startswith('gene2_tech')]
    gene2_tech = sorted(gene2_tech)

    # Build ordered column list
    ordered_cols = base_cols
    ordered_cols.extend([c for c in gene1_info if c in summary_df.columns])
    ordered_cols.extend([c for c in gene1_demo if c in summary_df.columns])
    ordered_cols.extend([c for c in gene1_tech if c in summary_df.columns])
    ordered_cols.extend([c for c in gene2_info if c in summary_df.columns])
    ordered_cols.extend([c for c in gene2_demo if c in summary_df.columns])
    ordered_cols.extend([c for c in gene2_tech if c in summary_df.columns])

    # Add any remaining columns
    remaining = [c for c in summary_df.columns if c not in ordered_cols]
    ordered_cols.extend(remaining)

    # Reorder
    summary_df = summary_df[[c for c in ordered_cols if c in summary_df.columns]]

    # Sort by cancer type
    summary_df = summary_df.sort_values('ONCOTREE_CODE').reset_index(drop=True)

    return summary_df


def create_significant_hits_table(data: dict) -> pd.DataFrame:
    """
    Create a table of only significant associations (long format).

    Each row is one significant hit with details.
    """
    sig_rows = []

    # Age
    if 'age' in data:
        sig_age = data['age'][data['age']['significant_fdr'] == True]
        for _, row in sig_age.iterrows():
            sig_rows.append({
                'ONCOTREE_CODE': row['ONCOTREE_CODE'],
                'Hugo_Symbol': row['Hugo_Symbol'],
                'gene_rank': row['gene_rank'],
                'feature_category': 'age',
                'feature': 'AGE_AT_SEQ_REPORT',
                'effect_size': row.get('cohens_d', np.nan),
                'effect_size_type': "Cohen's d",
                'direction': 'younger' if row.get('cohens_d', 0) < 0 else 'older',
                'p_fdr': row.get('p_fdr', np.nan),
                'n_actionable': row.get('n_actionable', np.nan),
                'n_non_actionable': row.get('n_non_actionable', np.nan),
            })

    # Sex
    if 'sex' in data:
        sig_sex = data['sex'][data['sex']['significant_fdr'] == True]
        for _, row in sig_sex.iterrows():
            or_val = row.get('odds_ratio', 1)
            sig_rows.append({
                'ONCOTREE_CODE': row['ONCOTREE_CODE'],
                'Hugo_Symbol': row['Hugo_Symbol'],
                'gene_rank': row['gene_rank'],
                'feature_category': 'sex',
                'feature': 'SEX',
                'effect_size': or_val,
                'effect_size_type': 'Odds Ratio (F/M)',
                'direction': 'higher_female' if or_val > 1 else 'higher_male',
                'p_fdr': row.get('p_fdr', np.nan),
                'n_actionable': row.get('n_gene_actionable', np.nan),
                'n_non_actionable': row.get('n_non_actionable', np.nan),
            })

    # Race
    if 'race' in data:
        sig_race = data['race'][data['race']['significant_fdr'] == True]
        for _, row in sig_race.iterrows():
            sig_rows.append({
                'ONCOTREE_CODE': row['ONCOTREE_CODE'],
                'Hugo_Symbol': row['Hugo_Symbol'],
                'gene_rank': row['gene_rank'],
                'feature_category': 'race',
                'feature': 'RACE',
                'effect_size': row.get('cramers_v', np.nan),
                'effect_size_type': "Cramer's V",
                'direction': 'association',
                'p_fdr': row.get('p_fdr', np.nan),
                'n_actionable': row.get('n_gene_actionable', np.nan),
                'n_non_actionable': row.get('n_non_actionable', np.nan),
            })

    # Ethnicity
    if 'ethnicity' in data:
        sig_eth = data['ethnicity'][data['ethnicity']['significant_fdr'] == True]
        for _, row in sig_eth.iterrows():
            sig_rows.append({
                'ONCOTREE_CODE': row['ONCOTREE_CODE'],
                'Hugo_Symbol': row['Hugo_Symbol'],
                'gene_rank': row['gene_rank'],
                'feature_category': 'ethnicity',
                'feature': 'ETHNICITY',
                'effect_size': row.get('cramers_v', np.nan),
                'effect_size_type': "Cramer's V",
                'direction': 'association',
                'p_fdr': row.get('p_fdr', np.nan),
                'n_actionable': row.get('n_gene_actionable', np.nan),
                'n_non_actionable': row.get('n_non_actionable', np.nan),
            })

    # Sample type
    if 'sample_type' in data:
        sig_sample = data['sample_type'][data['sample_type']['significant_fdr'] == True]
        for _, row in sig_sample.iterrows():
            sig_rows.append({
                'ONCOTREE_CODE': row['ONCOTREE_CODE'],
                'Hugo_Symbol': row['Hugo_Symbol'],
                'gene_rank': row['gene_rank'],
                'feature_category': 'sample_type',
                'feature': 'SAMPLE_TYPE',
                'effect_size': row.get('cramers_v', np.nan),
                'effect_size_type': "Cramer's V",
                'direction': 'association',
                'p_fdr': row.get('p_fdr', np.nan),
                'n_actionable': row.get('n_gene_actionable', np.nan),
                'n_non_actionable': row.get('n_non_actionable', np.nan),
            })

    # Technical
    if 'technical' in data:
        sig_tech = data['technical'][data['technical']['significant_fdr'] == True]
        for _, row in sig_tech.iterrows():
            feature = row.get('feature', 'unknown')
            if feature == 'number_of_genes':
                effect = row.get('effect_size_r', np.nan)
                effect_type = 'Effect size r'
            else:
                effect = row.get('cramers_v', np.nan)
                effect_type = "Cramer's V"

            sig_rows.append({
                'ONCOTREE_CODE': row['ONCOTREE_CODE'],
                'Hugo_Symbol': row['Hugo_Symbol'],
                'gene_rank': row['gene_rank'],
                'feature_category': 'technical',
                'feature': feature,
                'effect_size': effect,
                'effect_size_type': effect_type,
                'direction': 'association',
                'p_fdr': row.get('p_fdr', np.nan),
                'n_actionable': row.get('n_gene_actionable', np.nan),
                'n_non_actionable': row.get('n_non_actionable', np.nan),
            })

    sig_df = pd.DataFrame(sig_rows)

    if len(sig_df) > 0:
        # Sort by cancer type, gene, then feature category
        sig_df = sig_df.sort_values(
            ['ONCOTREE_CODE', 'Hugo_Symbol', 'feature_category']
        ).reset_index(drop=True)

    return sig_df


def print_summary_stats(summary_df: pd.DataFrame, sig_df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nCancer types analyzed: {len(summary_df)}")

    if len(sig_df) > 0:
        print(f"\nTotal significant associations: {len(sig_df)}")

        # By category
        cat_counts = sig_df['feature_category'].value_counts()
        print("\nBy feature category:")
        for cat, count in cat_counts.items():
            print(f"  {cat}: {count}")

        # By cancer type
        print("\nSignificant hits per cancer type:")
        cancer_counts = sig_df['ONCOTREE_CODE'].value_counts()
        for cancer, count in cancer_counts.head(10).items():
            print(f"  {cancer}: {count}")

        # Genes with most significant associations
        print("\nGenes with most significant associations:")
        gene_counts = sig_df.groupby(['ONCOTREE_CODE', 'Hugo_Symbol']).size()
        gene_counts = gene_counts.sort_values(ascending=False)
        for (cancer, gene), count in gene_counts.head(10).items():
            print(f"  {cancer} - {gene}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description='Create summary table of gene-stratified association results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect input from latest run
    python summarize_gene_stratified_results.py

    # Specify input directory
    python summarize_gene_stratified_results.py --input-dir /path/to/gene_stratified

    # Custom output filename
    python summarize_gene_stratified_results.py --output my_summary.csv
        """
    )

    parser.add_argument('--input-dir', type=str, default=None,
                       help='Directory containing gene-stratified results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as input)')
    parser.add_argument('--output', type=str, default='gene_stratified_summary.csv',
                       help='Output filename for wide-format summary (default: gene_stratified_summary.csv)')

    args = parser.parse_args()

    # Determine input directory
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = get_default_input_dir()
        if input_dir is None:
            print("ERROR: No gene-stratified results found.")
            return 1

    print(f"Input directory: {input_dir}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    print("\n" + "=" * 80)
    print("Loading gene-stratified results...")
    print("=" * 80)

    data = load_results(input_dir)

    if not data:
        print("ERROR: No results found")
        return 1

    # Create wide-format summary (one row per cancer type)
    print("\n" + "=" * 80)
    print("Creating wide-format summary table...")
    print("=" * 80)

    summary_df = create_summary_table(data)

    # Save wide-format summary
    summary_file = output_dir / args.output
    summary_df.to_csv(summary_file, index=False)
    print(f"\nWide-format summary saved to: {summary_file}")
    print(f"  Rows (cancer types): {len(summary_df)}")
    print(f"  Columns: {len(summary_df.columns)}")

    # Create long-format significant hits table
    print("\n" + "=" * 80)
    print("Creating significant hits table...")
    print("=" * 80)

    sig_df = create_significant_hits_table(data)

    # Save significant hits
    sig_file = output_dir / 'gene_stratified_significant_hits.csv'
    sig_df.to_csv(sig_file, index=False)
    print(f"\nSignificant hits saved to: {sig_file}")
    print(f"  Total significant associations: {len(sig_df)}")

    # Print summary statistics
    print_summary_stats(summary_df, sig_df)

    print(f"\n" + "=" * 80)
    print(f"All output saved to: {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
