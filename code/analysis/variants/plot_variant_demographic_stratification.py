"""
Variant-Level and Demographic Stratification Visualizations

This script extends the variant analysis with:
1. Variant-level granularity: Specific mutations (e.g., BRAF V600E) by cancer type
2. Demographic stratification: Gene/variant patterns stratified by demographics

Uses perPatientOncoKBFilteredForModeling table to align with ML modeling dataset.

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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path

# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = get_default_db_path()
DEFAULT_OUTPUT_DIR = Path("../../../analysis_output/level1_variant_analysis")
OUTPUT_DIR = None  # Set by main()

MIN_MUTATION_COUNT = 10  # Minimum patients per mutation to display
MIN_GENE_COUNT = 15      # Minimum patients per gene for demographic stratification

# ============================================================================
# DATA LOADING
# ============================================================================

def load_modeling_data_with_variants(db_path):
    """
    Load patient data from modeling table

    Note: perPatientOncoKBFilteredForModeling is patient-level, so we'll
    need to join with variant-level data if available, or use an alternative
    table that has both patient demographics and variant details.
    """

    conn = sqlite3.connect(db_path)

    # First, check what tables are available
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Available tables: {tables}")

    # Query from perPatientOncoKBLevelSorted which has both patient and variant info
    # Filter for actionable patients only (isOncoKbLevel1 = 1)
    query = """
    SELECT
        PATIENT_ID,
        ONCOTREE_CODE,
        Hugo_Symbol,
        HGVSp_Short,
        Variant_Classification,
        Variant_Type,
        AGE_AT_SEQ_REPORT,
        SEX,
        RACE,
        ETHNICITY,
        isOncoKbLevel1
    FROM perPatientOncoKBLevelSorted
    WHERE isOncoKbLevel1 = 1
        AND Hugo_Symbol IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"\nLoaded {len(df):,} actionable patient records")
    print(f"Unique patients: {df['PATIENT_ID'].nunique():,}")
    print(f"Unique genes: {df['Hugo_Symbol'].nunique()}")
    print(f"Unique cancer types: {df['ONCOTREE_CODE'].nunique()}")
    print(f"Records with HGVSp_Short: {df['HGVSp_Short'].notna().sum():,}")

    return df


def create_age_groups(age_series):
    """Create age groups for stratification"""

    def categorize_age(age):
        if pd.isna(age):
            return 'Unknown'
        elif age < 50:
            return '<50'
        elif age < 65:
            return '50-64'
        else:
            return '65+'

    return age_series.apply(categorize_age)


# ============================================================================
# VARIANT-LEVEL GRANULARITY PLOTS
# ============================================================================

def plot_specific_mutations_by_cancer(data, output_dir, top_mutations=25, top_cancers=10):
    """
    Stacked bar chart showing specific mutations (HGVSp) colored by cancer type

    Similar to gene-level plot but with mutation-level detail
    """

    # Filter to records with HGVSp annotation
    df_variants = data[data['HGVSp_Short'].notna()].copy()

    if len(df_variants) == 0:
        print("No HGVSp_Short data available, skipping mutation-level plot")
        return

    # Create gene + mutation label
    df_variants['Gene_Mutation'] = df_variants['Hugo_Symbol'] + ' ' + df_variants['HGVSp_Short']

    # Get top mutations by patient count
    mutation_counts = df_variants['Gene_Mutation'].value_counts()
    top_mutation_list = mutation_counts[mutation_counts >= MIN_MUTATION_COUNT].head(top_mutations).index.tolist()

    if len(top_mutation_list) == 0:
        print(f"No mutations with >={MIN_MUTATION_COUNT} patients, skipping plot")
        return

    print(f"\nPlotting top {len(top_mutation_list)} specific mutations")

    # Filter to top mutations
    filtered_data = df_variants[df_variants['Gene_Mutation'].isin(top_mutation_list)]

    # Get top cancer types overall
    top_cancer_list = data['ONCOTREE_CODE'].value_counts().head(top_cancers).index.tolist()

    # Create crosstab: mutations × cancer types
    crosstab_df = pd.crosstab(
        filtered_data['Gene_Mutation'],
        filtered_data['ONCOTREE_CODE']
    )

    # Sort by total count (descending)
    mutation_totals = crosstab_df.sum(axis=1).sort_values(ascending=True)
    crosstab_df = crosstab_df.reindex(mutation_totals.index)

    # Keep only top cancer types as individual colors, group rest as "Other"
    other_cancers = [col for col in crosstab_df.columns if col not in top_cancer_list]
    if other_cancers:
        crosstab_df['Other'] = crosstab_df[other_cancers].sum(axis=1)
        crosstab_df = crosstab_df.drop(columns=other_cancers)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(10, len(crosstab_df) * 0.4)))

    # Create color palette
    colors = sns.color_palette('tab20', len(crosstab_df.columns))

    # Create stacked bar chart
    crosstab_df.plot(
        kind='barh',
        stacked=True,
        ax=ax,
        color=colors,
        width=0.8,
        edgecolor='black',
        linewidth=0.5
    )

    # Formatting
    ax.set_xlabel('Number of Patients', fontsize=12, fontweight='bold')
    ax.set_ylabel('Specific Mutation', fontsize=12, fontweight='bold')
    ax.set_title('Actionable Mutations (Variant-Level) by Cancer Type',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)

    # Legend
    ax.legend(
        title='Cancer Type',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        edgecolor='black'
    )

    plt.tight_layout()
    plt.savefig(output_dir / 'mutations_by_cancer_stacked_bar.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'mutations_by_cancer_stacked_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: mutations_by_cancer_stacked_bar.pdf")


def plot_top_mutations_per_gene(data, output_dir, genes_to_plot=6, top_mutations=10):
    """
    For top genes, show the specific mutation distribution
    Faceted plot showing mutation diversity within each gene
    """

    df_variants = data[data['HGVSp_Short'].notna()].copy()

    if len(df_variants) == 0:
        print("No HGVSp_Short data available, skipping per-gene mutation plot")
        return

    # Get top genes
    top_genes = df_variants['Hugo_Symbol'].value_counts().head(genes_to_plot).index.tolist()

    # Create subplot grid
    n_cols = 2
    n_rows = (len(top_genes) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten() if len(top_genes) > 1 else [axes]

    for idx, gene in enumerate(top_genes):
        ax = axes[idx]

        # Get mutations for this gene
        gene_data = df_variants[df_variants['Hugo_Symbol'] == gene]
        mutation_counts = gene_data['HGVSp_Short'].value_counts().head(top_mutations)

        # Create bar chart
        colors = sns.color_palette('viridis', len(mutation_counts))
        bars = ax.barh(range(len(mutation_counts)), mutation_counts.values,
                       color=colors, edgecolor='black', linewidth=1)

        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, mutation_counts.values)):
            ax.text(count + max(mutation_counts.values) * 0.01, i, f'{count:,}',
                   va='center', ha='left', fontsize=9, fontweight='bold')

        # Formatting
        ax.set_yticks(range(len(mutation_counts)))
        ax.set_yticklabels(mutation_counts.index, fontsize=9)
        ax.set_xlabel('Number of Patients', fontsize=10)
        ax.set_title(f'{gene}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    # Hide unused subplots
    for idx in range(len(top_genes), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Top Mutations Within Frequently Mutated Genes',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'mutations_per_gene_faceted.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'mutations_per_gene_faceted.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: mutations_per_gene_faceted.pdf")


# ============================================================================
# DEMOGRAPHIC STRATIFICATION PLOTS
# ============================================================================

def plot_genes_by_sex_stratified(data, output_dir, top_genes=15):
    """
    Stacked bar chart of genes by cancer type, faceted by sex
    Shows sex-specific patterns in actionable mutations
    """

    # Filter to records with sex data
    df_sex = data[data['SEX'].isin(['Male', 'Female'])].copy()

    if len(df_sex) == 0:
        print("No sex data available, skipping sex-stratified plot")
        return

    # Get top genes overall
    top_gene_list = data['Hugo_Symbol'].value_counts().head(top_genes).index.tolist()

    # Filter to top genes
    df_sex = df_sex[df_sex['Hugo_Symbol'].isin(top_gene_list)]

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(top_gene_list) * 0.35)))

    for idx, sex in enumerate(['Male', 'Female']):
        ax = axes[idx]

        # Filter for this sex
        sex_data = df_sex[df_sex['SEX'] == sex]

        if len(sex_data) == 0:
            ax.text(0.5, 0.5, f'No data for {sex}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Get top cancer types for this sex
        top_cancers = sex_data['ONCOTREE_CODE'].value_counts().head(8).index.tolist()

        # Create crosstab
        crosstab_df = pd.crosstab(sex_data['Hugo_Symbol'], sex_data['ONCOTREE_CODE'])

        # Sort by total count
        gene_totals = crosstab_df.sum(axis=1).sort_values(ascending=True)
        crosstab_df = crosstab_df.reindex(gene_totals.index)

        # Filter to top genes that appear in top_gene_list
        crosstab_df = crosstab_df.reindex(
            [g for g in gene_totals.index if g in top_gene_list],
            fill_value=0
        )

        # Group non-top cancers as "Other"
        other_cancers = [col for col in crosstab_df.columns if col not in top_cancers]
        if other_cancers:
            crosstab_df['Other'] = crosstab_df[other_cancers].sum(axis=1)
            crosstab_df = crosstab_df.drop(columns=other_cancers)

        # Create color palette
        colors = sns.color_palette('tab20', len(crosstab_df.columns))

        # Create stacked bar chart
        crosstab_df.plot(
            kind='barh',
            stacked=True,
            ax=ax,
            color=colors,
            width=0.8,
            edgecolor='black',
            linewidth=0.5,
            legend=(idx == 1)  # Only show legend on right panel
        )

        # Formatting
        ax.set_xlabel('Number of Patients', fontsize=11, fontweight='bold')
        ax.set_ylabel('Gene', fontsize=11, fontweight='bold')
        ax.set_title(f'{sex} (N={len(sex_data):,})', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Legend only on right panel
        if idx == 1:
            ax.legend(
                title='Cancer Type',
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                fontsize=9,
                title_fontsize=10,
                frameon=True,
                edgecolor='black'
            )

    plt.suptitle('Actionable Genes by Cancer Type - Stratified by Sex',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'genes_by_cancer_stratified_by_sex.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'genes_by_cancer_stratified_by_sex.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: genes_by_cancer_stratified_by_sex.pdf")


def plot_genes_by_age_stratified(data, output_dir, top_genes=15):
    """
    Stacked bar chart of genes by cancer type, faceted by age group
    Shows age-specific patterns in actionable mutations
    """

    # Create age groups
    data = data.copy()
    data['AGE_GROUP'] = create_age_groups(data['AGE_AT_SEQ_REPORT'])

    # Filter to records with age data
    df_age = data[data['AGE_GROUP'] != 'Unknown'].copy()

    if len(df_age) == 0:
        print("No age data available, skipping age-stratified plot")
        return

    # Get top genes overall
    top_gene_list = data['Hugo_Symbol'].value_counts().head(top_genes).index.tolist()

    # Filter to top genes
    df_age = df_age[df_age['Hugo_Symbol'].isin(top_gene_list)]

    # Age groups to plot
    age_groups = ['<50', '50-64', '65+']

    # Create figure with three panels
    fig, axes = plt.subplots(1, 3, figsize=(22, max(8, len(top_gene_list) * 0.35)))

    for idx, age_group in enumerate(age_groups):
        ax = axes[idx]

        # Filter for this age group
        age_data = df_age[df_age['AGE_GROUP'] == age_group]

        if len(age_data) == 0:
            ax.text(0.5, 0.5, f'No data for {age_group}',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Get top cancer types for this age group
        top_cancers = age_data['ONCOTREE_CODE'].value_counts().head(8).index.tolist()

        # Create crosstab
        crosstab_df = pd.crosstab(age_data['Hugo_Symbol'], age_data['ONCOTREE_CODE'])

        # Sort by total count
        gene_totals = crosstab_df.sum(axis=1).sort_values(ascending=True)
        crosstab_df = crosstab_df.reindex(gene_totals.index)

        # Filter to top genes
        crosstab_df = crosstab_df.reindex(
            [g for g in gene_totals.index if g in top_gene_list],
            fill_value=0
        )

        # Group non-top cancers as "Other"
        other_cancers = [col for col in crosstab_df.columns if col not in top_cancers]
        if other_cancers:
            crosstab_df['Other'] = crosstab_df[other_cancers].sum(axis=1)
            crosstab_df = crosstab_df.drop(columns=other_cancers)

        # Create color palette
        colors = sns.color_palette('tab20', len(crosstab_df.columns))

        # Create stacked bar chart
        crosstab_df.plot(
            kind='barh',
            stacked=True,
            ax=ax,
            color=colors,
            width=0.8,
            edgecolor='black',
            linewidth=0.5,
            legend=(idx == 2)  # Only show legend on right panel
        )

        # Formatting
        ax.set_xlabel('Number of Patients', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Gene', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel('')
        ax.set_title(f'Age {age_group} (N={len(age_data):,})', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Legend only on right panel
        if idx == 2:
            ax.legend(
                title='Cancer Type',
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                fontsize=9,
                title_fontsize=10,
                frameon=True,
                edgecolor='black'
            )

    plt.suptitle('Actionable Genes by Cancer Type - Stratified by Age Group',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'genes_by_cancer_stratified_by_age.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'genes_by_cancer_stratified_by_age.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: genes_by_cancer_stratified_by_age.pdf")


def plot_genes_by_race_stratified(data, output_dir, top_genes=15):
    """
    Stacked bar chart of genes by cancer type, faceted by race
    Shows race-specific patterns in actionable mutations

    Note: Gene order is consistent across all race panels (based on overall totals)
    """

    # Include all race categories
    race_categories = ['White', 'Black', 'Asian', 'Other', 'Unknown']
    df_race = data[data['RACE'].isin(race_categories)].copy()

    if len(df_race) == 0:
        print("No race data available, skipping race-stratified plot")
        return

    # Get top genes overall (across all races) and establish consistent ordering
    top_gene_list = data['Hugo_Symbol'].value_counts().head(top_genes).index.tolist()

    # Calculate overall gene totals to establish consistent ordering
    overall_gene_totals = data[data['Hugo_Symbol'].isin(top_gene_list)]['Hugo_Symbol'].value_counts()
    gene_order = overall_gene_totals.index.tolist()  # Most common to least common

    # Filter to top genes
    df_race = df_race[df_race['Hugo_Symbol'].isin(top_gene_list)]

    # Create figure with five panels (one per race category)
    fig, axes = plt.subplots(1, 5, figsize=(28, max(8, len(top_gene_list) * 0.35)))

    for idx, race in enumerate(race_categories):
        ax = axes[idx]

        # Filter for this race
        race_data = df_race[df_race['RACE'] == race]

        if len(race_data) == 0:
            ax.text(0.5, 0.5, f'No data for {race}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_xlabel('Number of Patients', fontsize=11, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Gene', fontsize=11, fontweight='bold')
            ax.set_title(f'{race} (N=0)', fontsize=12, fontweight='bold')
            continue

        # Get top cancer types for this race
        top_cancers = race_data['ONCOTREE_CODE'].value_counts().head(8).index.tolist()

        # Create crosstab
        crosstab_df = pd.crosstab(race_data['Hugo_Symbol'], race_data['ONCOTREE_CODE'])

        # *** KEY CHANGE: Use consistent gene ordering across all panels ***
        # Reindex to match overall gene order (reversed for horizontal bars - least to most)
        crosstab_df = crosstab_df.reindex(gene_order[::-1], fill_value=0)

        # Group non-top cancers as "Other"
        other_cancers = [col for col in crosstab_df.columns if col not in top_cancers]
        if other_cancers:
            crosstab_df['Other_Cancer'] = crosstab_df[other_cancers].sum(axis=1)
            crosstab_df = crosstab_df.drop(columns=other_cancers)

        # Create color palette
        colors = sns.color_palette('tab20', len(crosstab_df.columns))

        # Create stacked bar chart
        crosstab_df.plot(
            kind='barh',
            stacked=True,
            ax=ax,
            color=colors,
            width=0.8,
            edgecolor='black',
            linewidth=0.5,
            legend=(idx == 4)  # Only show legend on rightmost panel
        )

        # Formatting
        ax.set_xlabel('Number of Patients', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Gene', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel('')
        ax.set_title(f'{race} (N={len(race_data):,})', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Legend only on rightmost panel
        if idx == 4:
            ax.legend(
                title='Cancer Type',
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                fontsize=9,
                title_fontsize=10,
                frameon=True,
                edgecolor='black'
            )

    plt.suptitle('Actionable Genes by Cancer Type - Stratified by Race (Consistent Gene Order)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'genes_by_cancer_stratified_by_race.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'genes_by_cancer_stratified_by_race.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: genes_by_cancer_stratified_by_race.pdf")


def generate_summary_report(data, output_dir):
    """Generate text summary of the analysis"""

    with open(output_dir / 'variant_demographic_stratification_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Variant-Level and Demographic Stratification Analysis\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total actionable patient records: {len(data):,}\n")
        f.write(f"Unique patients: {data['PATIENT_ID'].nunique():,}\n")
        f.write(f"Unique genes: {data['Hugo_Symbol'].nunique()}\n")
        f.write(f"Cancer types: {data['ONCOTREE_CODE'].nunique()}\n\n")

        # Variant-level info
        f.write("-" * 80 + "\n")
        f.write("VARIANT-LEVEL INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Records with HGVSp annotation: {data['HGVSp_Short'].notna().sum():,} ")
        f.write(f"({data['HGVSp_Short'].notna().sum()/len(data)*100:.1f}%)\n")
        f.write(f"Unique mutations (HGVSp): {data['HGVSp_Short'].nunique()}\n\n")

        # Demographic coverage
        f.write("-" * 80 + "\n")
        f.write("DEMOGRAPHIC DATA COVERAGE\n")
        f.write("-" * 80 + "\n")

        for col in ['SEX', 'AGE_AT_SEQ_REPORT', 'RACE', 'ETHNICITY']:
            if col in data.columns:
                missing = data[col].isna().sum()
                pct_missing = missing / len(data) * 100
                f.write(f"{col:<25} Missing: {missing:>8,} ({pct_missing:>5.1f}%)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("SEX DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        sex_counts = data['SEX'].value_counts(dropna=False)
        for sex, count in sex_counts.items():
            pct = count / len(data) * 100
            f.write(f"{str(sex):<15} {count:>8,} ({pct:>5.1f}%)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("AGE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        data_with_age = data.copy()
        data_with_age['AGE_GROUP'] = create_age_groups(data_with_age['AGE_AT_SEQ_REPORT'])
        age_counts = data_with_age['AGE_GROUP'].value_counts()
        for age_group, count in age_counts.items():
            pct = count / len(data) * 100
            f.write(f"{age_group:<15} {count:>8,} ({pct:>5.1f}%)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("RACE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        race_counts = data['RACE'].value_counts(dropna=False).head(10)
        for race, count in race_counts.items():
            pct = count / len(data) * 100
            f.write(f"{str(race):<25} {count:>8,} ({pct:>5.1f}%)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("TOP 20 GENES\n")
        f.write("-" * 80 + "\n")
        top_genes = data['Hugo_Symbol'].value_counts().head(20)
        for gene, count in top_genes.items():
            pct = count / len(data) * 100
            f.write(f"{gene:<20} {count:>8,} ({pct:>5.1f}%)\n")

        if data['HGVSp_Short'].notna().any():
            f.write("\n" + "-" * 80 + "\n")
            f.write("TOP 20 SPECIFIC MUTATIONS\n")
            f.write("-" * 80 + "\n")
            data['Gene_Mutation'] = data['Hugo_Symbol'] + ' ' + data['HGVSp_Short'].fillna('?')
            top_mutations = data[data['HGVSp_Short'].notna()]['Gene_Mutation'].value_counts().head(20)
            for mutation, count in top_mutations.items():
                pct = count / len(data) * 100
                f.write(f"{mutation:<35} {count:>8,} ({pct:>5.1f}%)\n")

    print(f"Saved: variant_demographic_stratification_summary.txt")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Variant-level demographic stratification')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Variant-Level and Demographic Stratification Analysis")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = load_modeling_data_with_variants(DB_PATH)

    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, OUTPUT_DIR)

    # VARIANT-LEVEL GRANULARITY PLOTS
    print("\n" + "=" * 80)
    print("VARIANT-LEVEL GRANULARITY VISUALIZATIONS")
    print("=" * 80)

    print("\n1. Specific mutations by cancer type (stacked bar)...")
    plot_specific_mutations_by_cancer(df, OUTPUT_DIR, top_mutations=25, top_cancers=10)

    print("\n2. Mutation diversity within top genes (faceted)...")
    plot_top_mutations_per_gene(df, OUTPUT_DIR, genes_to_plot=6, top_mutations=10)

    # DEMOGRAPHIC STRATIFICATION PLOTS
    print("\n" + "=" * 80)
    print("DEMOGRAPHIC STRATIFICATION VISUALIZATIONS")
    print("=" * 80)

    print("\n3. Genes by cancer type, stratified by sex...")
    plot_genes_by_sex_stratified(df, OUTPUT_DIR, top_genes=15)

    print("\n4. Genes by cancer type, stratified by age group...")
    plot_genes_by_age_stratified(df, OUTPUT_DIR, top_genes=15)

    print("\n5. Genes by cancer type, stratified by race...")
    plot_genes_by_race_stratified(df, OUTPUT_DIR, top_genes=15)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
    print("\nGenerated visualizations:")
    print("  - mutations_by_cancer_stacked_bar.pdf")
    print("  - mutations_per_gene_faceted.pdf")
    print("  - genes_by_cancer_stratified_by_sex.pdf")
    print("  - genes_by_cancer_stratified_by_age.pdf")
    print("  - genes_by_cancer_stratified_by_race.pdf")
    print("  - variant_demographic_stratification_summary.txt")
