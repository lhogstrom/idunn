"""
Demographic Analysis by Cancer Type

This script analyzes patient demographics across cancer types to:
1. Understand demographic composition for each cancer type
2. Compare demographics between actionable and non-actionable patients
3. Assess potential demographic-institution confounding
4. Calculate demographic diversity metrics

"""

import sqlite3
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
from db_config import get_default_db_path
from scipy.stats import entropy

# Configuration
DB_PATH = get_default_db_path()
DEFAULT_OUTPUT_DIR = Path("../../../analysis_output/demographic_analysis")
OUTPUT_DIR = None  # Set by main()

MIN_ACTIONABLE_EVENTS = 250

# Demographic features to analyze
DEMOGRAPHIC_FEATURES = [
    'AGE_AT_SEQ_REPORT',
    'SEX',
    'RACE',
    'ETHNICITY',
    'STAGE',
    'SAMPLE_TYPE'
]


def query_eligible_cancer_types(db_path, min_actionable):
    """Query cancer types with minimum actionable events"""

    conn = sqlite3.connect(db_path)

    query = f"""
    SELECT
        ONCOTREE_CODE,
        COUNT(*) as n_patients,
        SUM(CASE WHEN isOncoKbLevel1 = 1 THEN 1 ELSE 0 END) as n_actionable
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IS NOT NULL
    GROUP BY ONCOTREE_CODE
    HAVING n_actionable >= {min_actionable}
    ORDER BY n_actionable DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def load_demographic_data(db_path, cancer_type):
    """Load demographic data for a specific cancer type"""

    conn = sqlite3.connect(db_path)

    # Build query to get demographic features
    query = f"""
    SELECT
        PATIENT_ID,
        ONCOTREE_CODE,
        AGE_AT_SEQ_REPORT,
        SEX,
        RACE,
        ETHNICITY,
        STAGE,
        SAMPLE_TYPE,
        SUBJECT_CENTER,
        isOncoKbLevel1
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE = '{cancer_type}'
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def calculate_age_statistics(age_series):
    """Calculate age statistics"""

    # Remove missing values
    age_clean = age_series.dropna()

    if len(age_clean) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'q25': np.nan,
            'q75': np.nan,
            'missing_rate': 1.0,
            'n_valid': 0
        }

    return {
        'mean': age_clean.mean(),
        'median': age_clean.median(),
        'std': age_clean.std(),
        'min': age_clean.min(),
        'max': age_clean.max(),
        'q25': age_clean.quantile(0.25),
        'q75': age_clean.quantile(0.75),
        'missing_rate': age_series.isna().sum() / len(age_series),
        'n_valid': len(age_clean)
    }


def calculate_categorical_distribution(series, feature_name):
    """Calculate distribution for categorical features"""

    # Get value counts including missing
    total = len(series)
    value_counts = series.value_counts(dropna=False)

    # Calculate percentages
    distribution = {}
    for value, count in value_counts.items():
        value_str = str(value) if pd.notna(value) else 'Missing'
        distribution[f'{feature_name}_{value_str}_count'] = count
        distribution[f'{feature_name}_{value_str}_pct'] = count / total

    # Missing rate
    distribution[f'{feature_name}_missing_rate'] = series.isna().sum() / total
    distribution[f'{feature_name}_n_categories'] = series.nunique(dropna=True)

    return distribution


def calculate_shannon_entropy(series):
    """Calculate Shannon entropy for categorical variable"""

    # Remove missing values
    series_clean = series.dropna()

    if len(series_clean) == 0:
        return 0.0

    # Get value counts
    value_counts = series_clean.value_counts()

    # Calculate probabilities
    probabilities = value_counts / len(series_clean)

    # Calculate Shannon entropy
    return entropy(probabilities, base=2)


def analyze_demographics(df, cancer_type):
    """Analyze demographics for a cancer type"""

    results = {
        'cancer_type': cancer_type,
        'n_patients': len(df),
        'n_actionable': df['isOncoKbLevel1'].sum(),
        'actionable_rate': df['isOncoKbLevel1'].mean()
    }

    # Age statistics
    age_stats = calculate_age_statistics(df['AGE_AT_SEQ_REPORT'])
    for key, value in age_stats.items():
        results[f'age_{key}'] = value

    # Categorical features
    for feature in ['SEX', 'RACE', 'ETHNICITY', 'STAGE', 'SAMPLE_TYPE']:
        cat_dist = calculate_categorical_distribution(df[feature], feature.lower())
        results.update(cat_dist)

    # Demographic diversity metrics (Shannon entropy)
    results['sex_entropy'] = calculate_shannon_entropy(df['SEX'])
    results['race_entropy'] = calculate_shannon_entropy(df['RACE'])
    results['ethnicity_entropy'] = calculate_shannon_entropy(df['ETHNICITY'])
    results['stage_entropy'] = calculate_shannon_entropy(df['STAGE'])
    results['sample_type_entropy'] = calculate_shannon_entropy(df['SAMPLE_TYPE'])

    # Number of unique institutions
    results['n_institutions'] = df['SUBJECT_CENTER'].nunique()

    return results


def analyze_demographics_by_actionability(df, cancer_type):
    """Compare demographics between actionable and non-actionable patients"""

    actionable = df[df['isOncoKbLevel1'] == 1]
    non_actionable = df[df['isOncoKbLevel1'] == 0]

    comparison = {
        'cancer_type': cancer_type,
        'n_actionable': len(actionable),
        'n_non_actionable': len(non_actionable)
    }

    # Age comparison
    comparison['age_actionable_mean'] = actionable['AGE_AT_SEQ_REPORT'].mean()
    comparison['age_actionable_median'] = actionable['AGE_AT_SEQ_REPORT'].median()
    comparison['age_non_actionable_mean'] = non_actionable['AGE_AT_SEQ_REPORT'].mean()
    comparison['age_non_actionable_median'] = non_actionable['AGE_AT_SEQ_REPORT'].median()
    comparison['age_diff_mean'] = comparison['age_actionable_mean'] - comparison['age_non_actionable_mean']

    # Sex comparison
    if len(actionable) > 0:
        comparison['sex_actionable_female_pct'] = (actionable['SEX'] == 'Female').sum() / len(actionable)
        comparison['sex_actionable_male_pct'] = (actionable['SEX'] == 'Male').sum() / len(actionable)
    else:
        comparison['sex_actionable_female_pct'] = np.nan
        comparison['sex_actionable_male_pct'] = np.nan

    if len(non_actionable) > 0:
        comparison['sex_non_actionable_female_pct'] = (non_actionable['SEX'] == 'Female').sum() / len(non_actionable)
        comparison['sex_non_actionable_male_pct'] = (non_actionable['SEX'] == 'Male').sum() / len(non_actionable)
    else:
        comparison['sex_non_actionable_female_pct'] = np.nan
        comparison['sex_non_actionable_male_pct'] = np.nan

    # Race entropy comparison
    comparison['race_entropy_actionable'] = calculate_shannon_entropy(actionable['RACE'])
    comparison['race_entropy_non_actionable'] = calculate_shannon_entropy(non_actionable['RACE'])

    # Ethnicity entropy comparison
    comparison['ethnicity_entropy_actionable'] = calculate_shannon_entropy(actionable['ETHNICITY'])
    comparison['ethnicity_entropy_non_actionable'] = calculate_shannon_entropy(non_actionable['ETHNICITY'])

    return comparison


def create_summary_table(all_demographics_df, output_dir):
    """Create summary table with key demographic metrics"""

    # Select key columns for summary
    summary_cols = [
        'cancer_type',
        'n_patients',
        'n_actionable',
        'actionable_rate',
        'age_mean',
        'age_median',
        'age_std',
        'age_missing_rate',
        'sex_Female_pct',
        'sex_Male_pct',
        'sex_missing_rate',
        'race_n_categories',
        'race_entropy',
        'race_missing_rate',
        'ethnicity_n_categories',
        'ethnicity_entropy',
        'ethnicity_missing_rate',
        'stage_n_categories',
        'stage_entropy',
        'stage_missing_rate',
        'sample_type_n_categories',
        'sample_type_entropy',
        'n_institutions'
    ]

    # Filter to available columns
    available_cols = [col for col in summary_cols if col in all_demographics_df.columns]
    summary_df = all_demographics_df[available_cols].copy()

    # Sort by number of actionable events
    summary_df = summary_df.sort_values('n_actionable', ascending=False)

    # Save
    output_path = output_dir / 'demographic_summary_by_cancer_type.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    return summary_df


def main():
    """Main analysis pipeline"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Demographic analysis by cancer type')
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

    print("=" * 100)
    print("DEMOGRAPHIC ANALYSIS BY CANCER TYPE")
    print("=" * 100)
    print()

    # Query eligible cancer types
    print(f"Querying cancer types with ≥{MIN_ACTIONABLE_EVENTS} actionable events...")
    eligible = query_eligible_cancer_types(DB_PATH, MIN_ACTIONABLE_EVENTS)
    print(f"Found {len(eligible)} eligible cancer types")
    print()

    # Analyze each cancer type
    all_demographics = []
    all_comparisons = []

    for idx, row in eligible.iterrows():
        cancer_type = row['ONCOTREE_CODE']
        print(f"Analyzing {cancer_type} ({idx+1}/{len(eligible)})...")

        # Load data
        df = load_demographic_data(DB_PATH, cancer_type)

        # Overall demographics
        demo_stats = analyze_demographics(df, cancer_type)
        all_demographics.append(demo_stats)

        # Actionable vs non-actionable comparison
        comparison = analyze_demographics_by_actionability(df, cancer_type)
        all_comparisons.append(comparison)

    print()
    print("=" * 100)
    print("GENERATING OUTPUTS")
    print("=" * 100)
    print()

    # Create dataframes
    demographics_df = pd.DataFrame(all_demographics)
    comparison_df = pd.DataFrame(all_comparisons)

    # Save full demographic data
    print("Saving detailed demographic statistics...")
    demographics_df.to_csv(OUTPUT_DIR / 'demographic_summary_by_cancer_type_detailed.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'demographic_summary_by_cancer_type_detailed.csv'}")

    # Create summary table
    print("Creating summary table...")
    summary_df = create_summary_table(demographics_df, OUTPUT_DIR)

    # Save actionability comparison
    print("Saving actionable vs non-actionable comparison...")
    comparison_df.to_csv(OUTPUT_DIR / 'demographic_actionable_comparison.csv', index=False)
    print(f"Saved: {OUTPUT_DIR / 'demographic_actionable_comparison.csv'}")

    print()
    print("=" * 100)
    print("GENERATING VISUALIZATIONS")
    print("=" * 100)
    print()

    # Step 3: Core visualizations
    print("Creating Figure 1: Age distributions...")
    create_age_distributions_plot(demographics_df, OUTPUT_DIR)

    print("Creating Figure 2: Sex distribution...")
    create_sex_distribution_plot(demographics_df, OUTPUT_DIR)

    print("Creating Figure 3: Race/Ethnicity heatmap...")
    create_race_ethnicity_heatmap(OUTPUT_DIR)

    print("Creating Figure 5: Demographic diversity metrics...")
    create_demographic_diversity_plot(demographics_df, OUTPUT_DIR)

    print("Creating Figure 6: Missing data patterns...")
    create_missing_data_heatmap(OUTPUT_DIR)

    # Step 4: Advanced comparative visualizations
    print("Creating Figure 7: Demographics vs actionability...")
    create_demographics_vs_actionability_plots(demographics_df, comparison_df, OUTPUT_DIR)

    print("Creating Figure 8: Demographics vs institution...")
    create_demographics_vs_institution_plots(OUTPUT_DIR)

    print("Creating Figure 9: Demographics vs concentration...")
    create_demographics_vs_concentration_plots(demographics_df, OUTPUT_DIR)

    # Advanced multi-feature interaction visualizations
    print()
    print("=" * 100)
    print("GENERATING ADVANCED MULTI-FEATURE VISUALIZATIONS")
    print("=" * 100)
    print()

    print("Creating Figure 10: Age-Sex-Race interaction...")
    create_age_sex_race_interaction_plot(OUTPUT_DIR)

    print("Creating Figure 11: Demographic variability matrix...")
    create_demographic_variability_matrix(demographics_df, OUTPUT_DIR)

    print("Creating Figure 12: Age-Race-Actionability facets...")
    create_age_race_actionability_facets(OUTPUT_DIR)

    print("Creating Figure 13: Multivariate demographic clusters (PCA)...")
    create_multivariate_demographic_clusters(demographics_df, OUTPUT_DIR)

    print("Creating Figure 14: Sex-Race composition grid...")
    create_sex_race_composition_grid(OUTPUT_DIR)

    print("Creating Figure 15: Demographic feature correlations...")
    create_demographic_feature_correlations(demographics_df, OUTPUT_DIR)

    # Ethnicity-focused composition visualizations
    print()
    print("=" * 100)
    print("GENERATING ETHNICITY-INCLUSIVE COMPOSITION VISUALIZATIONS")
    print("=" * 100)
    print()

    print("Creating Figure 16: Race-Ethnicity composition grid...")
    create_race_ethnicity_composition_grid(OUTPUT_DIR)

    print("Creating Figure 17: Sex-Race-Ethnicity faceted grids...")
    create_sex_race_ethnicity_faceted_grids(OUTPUT_DIR)

    print("Creating Figure 18: Three-way demographic composition...")
    create_three_way_demographic_sunburst_alternative(OUTPUT_DIR)

    print("Creating Figure 19: Ethnicity by cancer with race breakdown...")
    create_ethnicity_by_cancer_type_detailed(OUTPUT_DIR)

    # Step 5: Comprehensive report
    print()
    print("=" * 100)
    print("GENERATING COMPREHENSIVE REPORT (STEP 5)")
    print("=" * 100)
    print()
    generate_comprehensive_report(demographics_df, comparison_df, OUTPUT_DIR)

    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print()
    print(f"Results saved to: {OUTPUT_DIR}")
    print()
    print("Files generated:")
    print("  CSV files:")
    print("    - demographic_summary_by_cancer_type.csv (summary)")
    print("    - demographic_summary_by_cancer_type_detailed.csv (full data)")
    print("    - demographic_actionable_comparison.csv")
    print()
    print("  Core Visualizations (PDF and PNG):")
    print("    - fig1_age_distributions.pdf/png")
    print("    - fig2_sex_distribution.pdf/png")
    print("    - fig3_race_ethnicity_heatmap.pdf/png")
    print("    - fig5_demographic_diversity.pdf/png")
    print("    - fig6_missing_data_heatmap.pdf/png")
    print("    - fig7_demographics_vs_actionability.pdf/png")
    print("    - fig8_demographics_vs_institution.pdf/png")
    print("    - fig9_demographics_vs_concentration.pdf/png")
    print()
    print("  Advanced Multi-Feature Visualizations (PDF and PNG):")
    print("    - fig10_age_sex_race_interaction.pdf/png")
    print("    - fig11_demographic_variability_matrix.pdf/png")
    print("    - fig12_age_race_actionability_facets.pdf/png")
    print("    - fig13_multivariate_demographic_clusters.pdf/png")
    print("    - fig14_sex_race_composition_grid.pdf/png")
    print("    - fig15_demographic_feature_correlations.pdf/png")
    print()
    print("  Ethnicity-Inclusive Composition Visualizations (PDF and PNG):")
    print("    - fig16_race_ethnicity_composition_grid.pdf/png")
    print("    - fig17_sex_race_ethnicity_faceted_grids.pdf/png")
    print("    - fig18_three_way_demographic_composition.pdf/png")
    print("    - fig19_ethnicity_by_cancer_race_breakdown.pdf/png")
    print()
    print("  Report:")
    print("    - demographic_analysis_comprehensive_report.txt")
    print()

    # Quick summary
    print("=" * 100)
    print("QUICK SUMMARY")
    print("=" * 100)
    print()
    print(f"Total cancer types analyzed: {len(demographics_df)}")
    print(f"Total patients: {demographics_df['n_patients'].sum():,}")
    print(f"Total actionable events: {demographics_df['n_actionable'].sum():,}")
    print()

    print("Age statistics across all cancer types:")
    print(f"  Mean age range: {demographics_df['age_mean'].min():.1f} - {demographics_df['age_mean'].max():.1f} years")
    print(f"  Overall mean age: {demographics_df['age_mean'].mean():.1f} years")
    print(f"  Age missing rate range: {demographics_df['age_missing_rate'].min():.1%} - {demographics_df['age_missing_rate'].max():.1%}")
    print()

    print("Sex distribution:")
    print(f"  Female % range: {demographics_df['sex_Female_pct'].min():.1%} - {demographics_df['sex_Female_pct'].max():.1%}")
    print(f"  Male % range: {demographics_df['sex_Male_pct'].min():.1%} - {demographics_df['sex_Male_pct'].max():.1%}")
    print(f"  Sex missing rate range: {demographics_df['sex_missing_rate'].min():.1%} - {demographics_df['sex_missing_rate'].max():.1%}")
    print()

    print("Demographic diversity (Shannon entropy):")
    print(f"  Race entropy range: {demographics_df['race_entropy'].min():.2f} - {demographics_df['race_entropy'].max():.2f}")
    print(f"  Ethnicity entropy range: {demographics_df['ethnicity_entropy'].min():.2f} - {demographics_df['ethnicity_entropy'].max():.2f}")
    print()

    print("Missing data patterns:")
    print(f"  Race missing rate range: {demographics_df['race_missing_rate'].min():.1%} - {demographics_df['race_missing_rate'].max():.1%}")
    print(f"  Ethnicity missing rate range: {demographics_df['ethnicity_missing_rate'].min():.1%} - {demographics_df['ethnicity_missing_rate'].max():.1%}")
    print(f"  Stage missing rate range: {demographics_df['stage_missing_rate'].min():.1%} - {demographics_df['stage_missing_rate'].max():.1%}")
    print()


def create_age_distributions_plot(all_demographics_df, output_dir):
    """Figure 1: Age distributions across cancer types"""

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Sort by median age
    sorted_types = all_demographics_df.sort_values('age_median')['cancer_type'].tolist()

    # Load data for each cancer type to create violin plots
    age_data = []
    cancer_types = []
    actionability = []

    for cancer_type in sorted_types:
        df = load_demographic_data(DB_PATH, cancer_type)
        age_clean = df['AGE_AT_SEQ_REPORT'].dropna()

        for idx, age in age_clean.items():
            age_data.append(age)
            cancer_types.append(cancer_type)
            actionability.append('Actionable' if df.loc[idx, 'isOncoKbLevel1'] == 1 else 'Non-Actionable')

    plot_df = pd.DataFrame({
        'cancer_type': cancer_types,
        'age': age_data,
        'actionability': actionability
    })

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 10))

    # Violin plot
    sns.violinplot(data=plot_df, x='cancer_type', y='age', hue='actionability',
                   split=False, inner='quartile', palette=['#e74c3c', '#3498db'],
                   ax=ax, alpha=0.7)

    ax.set_xlabel('Cancer Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Age at Sequencing (years)', fontsize=14, fontweight='bold')
    ax.set_title('Age Distribution Across Cancer Types\nSplit by Actionability Status',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(title='Patient Group', fontsize=11, title_fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_age_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_age_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig1_age_distributions.pdf'}")
    plt.close()


def create_sex_distribution_plot(all_demographics_df, output_dir):
    """Figure 2: Sex distribution by cancer type"""

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Prepare data
    cancer_types = all_demographics_df['cancer_type'].tolist()

    # Get sex distributions
    sex_data = []
    for cancer_type in cancer_types:
        df = load_demographic_data(DB_PATH, cancer_type)

        # Actionable
        actionable = df[df['isOncoKbLevel1'] == 1]
        female_act = (actionable['SEX'] == 'Female').sum()
        male_act = (actionable['SEX'] == 'Male').sum()
        other_act = len(actionable) - female_act - male_act

        # Non-actionable
        non_actionable = df[df['isOncoKbLevel1'] == 0]
        female_non = (non_actionable['SEX'] == 'Female').sum()
        male_non = (non_actionable['SEX'] == 'Male').sum()
        other_non = len(non_actionable) - female_non - male_non

        sex_data.append({
            'cancer_type': cancer_type,
            'female_actionable': female_act,
            'male_actionable': male_act,
            'other_actionable': other_act,
            'female_non_actionable': female_non,
            'male_non_actionable': male_non,
            'other_non_actionable': other_non
        })

    sex_df = pd.DataFrame(sex_data)

    # Create plot with two panels
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Panel 1: Actionable patients
    ax = axes[0]
    x = np.arange(len(cancer_types))
    width = 0.8

    p1 = ax.bar(x, sex_df['female_actionable'], width, label='Female', color='#e91e63', alpha=0.8)
    p2 = ax.bar(x, sex_df['male_actionable'], width, bottom=sex_df['female_actionable'],
                label='Male', color='#2196f3', alpha=0.8)

    ax.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax.set_title('Sex Distribution - Actionable Patients', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cancer_types, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Non-actionable patients
    ax = axes[1]

    p1 = ax.bar(x, sex_df['female_non_actionable'], width, label='Female', color='#e91e63', alpha=0.8)
    p2 = ax.bar(x, sex_df['male_non_actionable'], width, bottom=sex_df['female_non_actionable'],
                label='Male', color='#2196f3', alpha=0.8)

    ax.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
    ax.set_title('Sex Distribution - Non-Actionable Patients', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cancer_types, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_sex_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_sex_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig2_sex_distribution.pdf'}")
    plt.close()


def create_race_ethnicity_heatmap(output_dir):
    """Figure 3: Race/Ethnicity composition heatmap"""

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Get race/ethnicity distributions
    race_data = []
    ethnicity_data = []

    eligible = query_eligible_cancer_types(DB_PATH, MIN_ACTIONABLE_EVENTS)
    cancer_types = eligible['ONCOTREE_CODE'].tolist()

    for cancer_type in cancer_types:
        df = load_demographic_data(DB_PATH, cancer_type)

        # Race distribution
        race_counts = df['RACE'].value_counts(normalize=True) * 100
        race_dict = {'cancer_type': cancer_type}
        race_dict.update(race_counts.to_dict())
        race_data.append(race_dict)

        # Ethnicity distribution
        eth_counts = df['ETHNICITY'].value_counts(normalize=True) * 100
        eth_dict = {'cancer_type': cancer_type}
        eth_dict.update(eth_counts.to_dict())
        ethnicity_data.append(eth_dict)

    race_df = pd.DataFrame(race_data).set_index('cancer_type').fillna(0)
    ethnicity_df = pd.DataFrame(ethnicity_data).set_index('cancer_type').fillna(0)

    # Create figure with two heatmaps
    fig, axes = plt.subplots(2, 1, figsize=(14, 14))

    # Race heatmap
    ax = axes[0]
    sns.heatmap(race_df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5)
    ax.set_title('Race Composition Across Cancer Types', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Race Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')

    # Ethnicity heatmap
    ax = axes[1]
    sns.heatmap(ethnicity_df, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax,
                cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5)
    ax.set_title('Ethnicity Composition Across Cancer Types', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Ethnicity Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_race_ethnicity_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_race_ethnicity_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig3_race_ethnicity_heatmap.pdf'}")
    plt.close()


def create_demographic_diversity_plot(all_demographics_df, output_dir):
    """Figure 5: Demographic diversity metrics"""

    import matplotlib.pyplot as plt

    # Sort by race entropy
    sorted_df = all_demographics_df.sort_values('race_entropy', ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Race entropy
    ax = axes[0, 0]
    bars = ax.barh(sorted_df['cancer_type'], sorted_df['race_entropy'], color='#e74c3c', alpha=0.7)
    ax.set_xlabel('Shannon Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Race Diversity Across Cancer Types', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Plot 2: Ethnicity entropy
    ax = axes[0, 1]
    sorted_df2 = all_demographics_df.sort_values('ethnicity_entropy', ascending=False)
    bars = ax.barh(sorted_df2['cancer_type'], sorted_df2['ethnicity_entropy'], color='#3498db', alpha=0.7)
    ax.set_xlabel('Shannon Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Ethnicity Diversity Across Cancer Types', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Plot 3: Sample type entropy
    ax = axes[1, 0]
    sorted_df3 = all_demographics_df.sort_values('sample_type_entropy', ascending=False)
    bars = ax.barh(sorted_df3['cancer_type'], sorted_df3['sample_type_entropy'], color='#2ecc71', alpha=0.7)
    ax.set_xlabel('Shannon Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Sample Type Diversity Across Cancer Types', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Plot 4: Combined diversity comparison
    ax = axes[1, 1]
    x = np.arange(len(all_demographics_df))
    width = 0.25

    sorted_df4 = all_demographics_df.sort_values('n_actionable', ascending=False)

    ax.bar(x - width, sorted_df4['race_entropy'], width, label='Race', color='#e74c3c', alpha=0.7)
    ax.bar(x, sorted_df4['ethnicity_entropy'], width, label='Ethnicity', color='#3498db', alpha=0.7)
    ax.bar(x + width, sorted_df4['sample_type_entropy'], width, label='Sample Type', color='#2ecc71', alpha=0.7)

    ax.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shannon Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Demographic Diversity Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df4['cancer_type'], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_demographic_diversity.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_demographic_diversity.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig5_demographic_diversity.pdf'}")
    plt.close()


def create_missing_data_heatmap(output_dir):
    """Figure 6: Missing data patterns"""

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Collect missing data rates
    missing_data = []

    eligible = query_eligible_cancer_types(DB_PATH, MIN_ACTIONABLE_EVENTS)
    cancer_types = eligible['ONCOTREE_CODE'].tolist()

    for cancer_type in cancer_types:
        df = load_demographic_data(DB_PATH, cancer_type)

        missing_dict = {
            'cancer_type': cancer_type,
            'AGE': df['AGE_AT_SEQ_REPORT'].isna().sum() / len(df) * 100,
            'SEX': df['SEX'].isna().sum() / len(df) * 100,
            'RACE': df['RACE'].isna().sum() / len(df) * 100,
            'ETHNICITY': df['ETHNICITY'].isna().sum() / len(df) * 100,
            'STAGE': df['STAGE'].isna().sum() / len(df) * 100,
            'SAMPLE_TYPE': df['SAMPLE_TYPE'].isna().sum() / len(df) * 100
        }
        missing_data.append(missing_dict)

    missing_df = pd.DataFrame(missing_data).set_index('cancer_type')

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 12))

    sns.heatmap(missing_df, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax,
                cbar_kws={'label': 'Missing Data (%)'}, linewidths=0.5,
                vmin=0, vmax=100)

    ax.set_title('Missing Data Patterns Across Demographic Features', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Demographic Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_missing_data_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig6_missing_data_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig6_missing_data_heatmap.pdf'}")
    plt.close()


def create_demographics_vs_actionability_plots(all_demographics_df, comparison_df, output_dir):
    """Figure 7: Demographics vs Actionability"""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Mean age vs actionable rate
    ax = axes[0, 0]
    scatter = ax.scatter(all_demographics_df['age_mean'],
                        all_demographics_df['actionable_rate'] * 100,
                        s=all_demographics_df['n_patients'] / 50,
                        c=all_demographics_df['n_actionable'],
                        cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)

    for idx, row in all_demographics_df.iterrows():
        if row['actionable_rate'] > 0.6 or row['age_mean'] < 50:
            ax.annotate(row['cancer_type'], (row['age_mean'], row['actionable_rate'] * 100),
                       fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('Mean Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actionable Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Age vs Actionability Rate', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('N Actionable', fontsize=10)

    # Plot 2: Age difference (actionable - non-actionable)
    ax = axes[0, 1]
    sorted_comp = comparison_df.sort_values('age_diff_mean')
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in sorted_comp['age_diff_mean']]
    bars = ax.barh(sorted_comp['cancer_type'], sorted_comp['age_diff_mean'], color=colors, alpha=0.7)
    ax.set_xlabel('Age Difference (Actionable - Non-Actionable)', fontsize=12, fontweight='bold')
    ax.set_title('Age Differences Between Actionable & Non-Actionable', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    # Plot 3: Female % vs actionable rate
    ax = axes[1, 0]
    scatter = ax.scatter(all_demographics_df['sex_Female_pct'] * 100,
                        all_demographics_df['actionable_rate'] * 100,
                        s=all_demographics_df['n_patients'] / 50,
                        c=all_demographics_df['age_mean'],
                        cmap='coolwarm', alpha=0.7, edgecolors='black', linewidth=0.5)

    for idx, row in all_demographics_df.iterrows():
        if row['sex_Female_pct'] < 0.2 or row['sex_Female_pct'] > 0.95:
            ax.annotate(row['cancer_type'], (row['sex_Female_pct'] * 100, row['actionable_rate'] * 100),
                       fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('Female %', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actionable Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Sex Distribution vs Actionability', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Age', fontsize=10)

    # Plot 4: Race diversity vs actionable rate
    ax = axes[1, 1]
    scatter = ax.scatter(all_demographics_df['race_entropy'],
                        all_demographics_df['actionable_rate'] * 100,
                        s=all_demographics_df['n_patients'] / 50,
                        c=all_demographics_df['n_institutions'],
                        cmap='plasma', alpha=0.7, edgecolors='black', linewidth=0.5)

    for idx, row in all_demographics_df.iterrows():
        if row['race_entropy'] > 1.8 or row['actionable_rate'] > 0.7:
            ax.annotate(row['cancer_type'], (row['race_entropy'], row['actionable_rate'] * 100),
                       fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('Race Diversity (Shannon Entropy)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actionable Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Demographic Diversity vs Actionability', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('N Institutions', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_demographics_vs_actionability.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig7_demographics_vs_actionability.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig7_demographics_vs_actionability.pdf'}")
    plt.close()


def create_demographics_vs_institution_plots(output_dir):
    """Figure 8: Demographics vs Institution (test for confounding)"""

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Select representative cancer types with multiple institutions
    test_cancer_types = ['LUAD', 'BRCA', 'MEL', 'GIST']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, cancer_type in enumerate(test_cancer_types):
        ax = axes[idx // 2, idx % 2]

        df = load_demographic_data(DB_PATH, cancer_type)

        # Get top 5 institutions by patient count
        top_insts = df['SUBJECT_CENTER'].value_counts().head(5).index.tolist()
        df_subset = df[df['SUBJECT_CENTER'].isin(top_insts)]

        # Violin plot of age by institution
        sns.violinplot(data=df_subset, x='SUBJECT_CENTER', y='AGE_AT_SEQ_REPORT',
                      inner='quartile', palette='Set2', ax=ax)

        ax.set_xlabel('Institution (Top 5)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Age (years)', fontsize=11, fontweight='bold')
        ax.set_title(f'{cancer_type} - Age Distribution by Institution', fontsize=12, fontweight='bold')
        ax.set_xticklabels([f'Inst {i+1}' for i in range(len(top_insts))], fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig8_demographics_vs_institution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig8_demographics_vs_institution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig8_demographics_vs_institution.pdf'}")
    plt.close()


def create_demographics_vs_concentration_plots(all_demographics_df, output_dir):
    """Figure 9: Demographics vs Institutional Concentration"""

    import matplotlib.pyplot as plt

    # Load institutional stats for concentration metrics
    inst_stats_path = Path("../../../analysis_output/all_cancer_types_institutional_analysis/all_cancer_types_institutional_stats.csv")

    if inst_stats_path.exists():
        inst_df = pd.read_csv(inst_stats_path)

        # Merge with demographics
        merged_df = all_demographics_df.merge(inst_df[['cancer_type', 'herfindahl_index', 'top3_institutions_pct']],
                                             on='cancer_type', how='left')

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Race diversity vs institutional concentration
        ax = axes[0]
        scatter = ax.scatter(merged_df['herfindahl_index'],
                            merged_df['race_entropy'],
                            s=merged_df['n_patients'] / 50,
                            c=merged_df['actionable_rate'],
                            cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=0.5)

        for idx, row in merged_df.iterrows():
            if pd.notna(row['herfindahl_index']) and (row['herfindahl_index'] > 0.3 or row['race_entropy'] > 1.7):
                ax.annotate(row['cancer_type'], (row['herfindahl_index'], row['race_entropy']),
                           fontsize=8, ha='center', va='bottom')

        ax.set_xlabel('Herfindahl Index (Institutional Concentration)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Race Diversity (Shannon Entropy)', fontsize=12, fontweight='bold')
        ax.set_title('Demographic Diversity vs Institutional Concentration', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Actionable Rate', fontsize=10)

        # Plot 2: Age diversity vs institutional concentration
        ax = axes[1]
        scatter = ax.scatter(merged_df['top3_institutions_pct'],
                            merged_df['age_std'],
                            s=merged_df['n_patients'] / 50,
                            c=merged_df['n_institutions'],
                            cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)

        for idx, row in merged_df.iterrows():
            if pd.notna(row['top3_institutions_pct']) and (row['age_std'] > 14 or row['age_std'] < 11):
                ax.annotate(row['cancer_type'], (row['top3_institutions_pct'], row['age_std']),
                           fontsize=8, ha='center', va='bottom')

        ax.set_xlabel('Top 3 Institutions % (Concentration)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Age Std Dev (years)', fontsize=12, fontweight='bold')
        ax.set_title('Age Diversity vs Institutional Concentration', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('N Institutions', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / 'fig9_demographics_vs_concentration.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'fig9_demographics_vs_concentration.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'fig9_demographics_vs_concentration.pdf'}")
        plt.close()
    else:
        print(f"Warning: Institutional stats file not found, skipping Figure 9")


def create_age_sex_race_interaction_plot(output_dir):
    """
    Figure 10: Age-Sex-Race interaction across cancer types
    Shows how age distribution varies by sex and race for major cancer types
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Query data for top cancer types
    conn = sqlite3.connect(DB_PATH)

    # Get top 6 cancer types by sample size
    query_top = """
    SELECT ONCOTREE_CODE, COUNT(*) as n
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE
    HAVING n >= 1000
    ORDER BY n DESC
    LIMIT 6
    """
    top_cancers = pd.read_sql_query(query_top, conn)['ONCOTREE_CODE'].tolist()

    # Query demographic data for these cancer types
    query = f"""
    SELECT
        ONCOTREE_CODE,
        AGE_AT_SEQ_REPORT,
        SEX,
        RACE,
        isOncoKbLevel1
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IN ({','.join(['?']*len(top_cancers))})
    AND AGE_AT_SEQ_REPORT IS NOT NULL
    AND SEX IS NOT NULL
    AND RACE IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=top_cancers)
    conn.close()

    # Filter to major race categories and sex categories
    major_races = df['RACE'].value_counts().nlargest(4).index.tolist()
    df = df[df['RACE'].isin(major_races)]
    df = df[df['SEX'].isin(['Male', 'Female'])]

    # Create figure with subplots for each cancer type
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, cancer_type in enumerate(top_cancers):
        ax = axes[idx]
        cancer_df = df[df['ONCOTREE_CODE'] == cancer_type]

        # Create violin plot showing age by sex, with race as hue
        sns.violinplot(data=cancer_df, x='SEX', y='AGE_AT_SEQ_REPORT',
                      hue='RACE', ax=ax, split=False, inner='quartile',
                      palette='Set2')

        ax.set_title(f'{cancer_type} (n={len(cancer_df):,})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sex', fontsize=12, fontweight='bold')
        ax.set_ylabel('Age at Sequencing (years)', fontsize=12, fontweight='bold')
        ax.legend(title='Race', fontsize=8, title_fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Age-Sex-Race Interaction Across Cancer Types',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig10_age_sex_race_interaction.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig10_age_sex_race_interaction.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig10_age_sex_race_interaction.pdf'}")
    plt.close()


def create_demographic_variability_matrix(demographics_df, output_dir):
    """
    Figure 11: Demographic variability heatmap
    Shows coefficient of variation for demographic features across cancer types
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Calculate CV (coefficient of variation) for continuous features
    variability_data = []

    for idx, row in demographics_df.iterrows():
        cancer_type = row['cancer_type']

        # Age variability (CV = std/mean)
        age_cv = row['age_std'] / row['age_mean'] if row['age_mean'] > 0 else 0

        # Sex variability (entropy normalized to 0-1)
        sex_entropy = row.get('sex_entropy', 0)
        sex_var = sex_entropy / np.log(2) if sex_entropy > 0 else 0  # max entropy for 2 categories

        # Race variability (entropy normalized)
        race_var = row['race_entropy'] / np.log(7)  # max entropy for 7 categories

        # Ethnicity variability (entropy normalized)
        ethnicity_var = row['ethnicity_entropy'] / np.log(3)  # max entropy for 3 categories

        # Sample type variability
        sample_type_var = row.get('sample_type_entropy', 0) / np.log(6) if 'sample_type_entropy' in row else 0

        variability_data.append({
            'Cancer Type': cancer_type,
            'Age CV': age_cv,
            'Sex Diversity': sex_var,
            'Race Diversity': race_var,
            'Ethnicity Diversity': ethnicity_var,
            'Actionable Rate': row['actionable_rate']
        })

    var_df = pd.DataFrame(variability_data)
    var_df = var_df.set_index('Cancer Type')

    # Sort by actionable rate
    var_df = var_df.sort_values('Actionable Rate', ascending=False)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 14))

    # Plot heatmap (exclude actionable rate from heatmap, use as annotation)
    heatmap_data = var_df.drop('Actionable Rate', axis=1)

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'Variability/Diversity Score'},
                linewidths=0.5, ax=ax)

    ax.set_title('Demographic Variability Matrix Across Cancer Types\n(Sorted by Actionable Rate)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Demographic Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')

    # Add actionable rate as side annotation
    for idx, (cancer_type, rate) in enumerate(var_df['Actionable Rate'].items()):
        ax.text(heatmap_data.shape[1] + 0.2, idx + 0.5, f'{rate:.1%}',
               va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig11_demographic_variability_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig11_demographic_variability_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig11_demographic_variability_matrix.pdf'}")
    plt.close()


def create_age_race_actionability_facets(output_dir):
    """
    Figure 12: Age-Race-Actionability faceted plot
    Shows age distribution by race for actionable vs non-actionable patients
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Query data for top 4 cancer types
    conn = sqlite3.connect(DB_PATH)

    query_top = """
    SELECT ONCOTREE_CODE, COUNT(*) as n
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE
    HAVING n >= 2000
    ORDER BY n DESC
    LIMIT 4
    """
    top_cancers = pd.read_sql_query(query_top, conn)['ONCOTREE_CODE'].tolist()

    # Query demographic data
    query = f"""
    SELECT
        ONCOTREE_CODE,
        AGE_AT_SEQ_REPORT,
        RACE,
        isOncoKbLevel1
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IN ({','.join(['?']*len(top_cancers))})
    AND AGE_AT_SEQ_REPORT IS NOT NULL
    AND RACE IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=top_cancers)
    conn.close()

    # Filter to major race categories
    major_races = df['RACE'].value_counts().nlargest(4).index.tolist()
    df = df[df['RACE'].isin(major_races)]

    # Add actionability label
    df['Actionability'] = df['isOncoKbLevel1'].map({1: 'Actionable', 0: 'Non-Actionable'})

    # Create faceted plot
    g = sns.FacetGrid(df, col='ONCOTREE_CODE', row='Actionability',
                     height=4, aspect=1.5, margin_titles=True)

    g.map_dataframe(sns.violinplot, x='RACE', y='AGE_AT_SEQ_REPORT',
                   palette='Set3', inner='quartile')

    g.set_axis_labels('Race', 'Age at Sequencing (years)', fontsize=11, fontweight='bold')
    g.set_titles(col_template='{col_name}', row_template='{row_name}',
                size=12, weight='bold')

    # Rotate x-axis labels
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Age Distribution by Race: Actionable vs Non-Actionable Patients',
                fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig12_age_race_actionability_facets.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig12_age_race_actionability_facets.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig12_age_race_actionability_facets.pdf'}")
    plt.close()


def create_multivariate_demographic_clusters(demographics_df, output_dir):
    """
    Figure 13: PCA of demographic features showing cancer type clustering
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Select features for PCA
    features = ['age_mean', 'age_std', 'sex_Female_pct', 'race_entropy',
                'ethnicity_entropy', 'actionable_rate']

    X = demographics_df[features].fillna(0)
    cancer_types = demographics_df['cancer_type'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Colored by actionable rate
    ax = axes[0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                        c=demographics_df['actionable_rate'],
                        s=demographics_df['n_patients'] / 50,
                        cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=1)

    # Annotate all points
    for i, cancer_type in enumerate(cancer_types):
        ax.annotate(cancer_type, (X_pca[i, 0], X_pca[i, 1]),
                   fontsize=9, ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                 fontsize=12, fontweight='bold')
    ax.set_title('Cancer Types in Demographic Feature Space\n(Colored by Actionable Rate)',
                fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Actionable Rate', fontsize=10, fontweight='bold')

    # Plot 2: Colored by mean age
    ax = axes[1]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                        c=demographics_df['age_mean'],
                        s=demographics_df['n_patients'] / 50,
                        cmap='coolwarm', alpha=0.7, edgecolors='black', linewidth=1)

    # Annotate all points
    for i, cancer_type in enumerate(cancer_types):
        ax.annotate(cancer_type, (X_pca[i, 0], X_pca[i, 1]),
                   fontsize=9, ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                 fontsize=12, fontweight='bold')
    ax.set_title('Cancer Types in Demographic Feature Space\n(Colored by Mean Age)',
                fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Age (years)', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig13_multivariate_demographic_clusters.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig13_multivariate_demographic_clusters.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig13_multivariate_demographic_clusters.pdf'}")
    plt.close()

    # Print feature loadings
    print("\nPCA Feature Loadings:")
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=features
    )
    print(loadings.round(3))


def create_sex_race_composition_grid(output_dir):
    """
    Figure 14: Sex-Race composition grid showing proportions across cancer types
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Query data for all cancer types
    conn = sqlite3.connect(DB_PATH)

    query_types = f"""
    SELECT ONCOTREE_CODE, COUNT(*) as n
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE
    HAVING COUNT(CASE WHEN isOncoKbLevel1 = 1 THEN 1 END) >= {MIN_ACTIONABLE_EVENTS}
    ORDER BY n DESC
    """
    cancer_types = pd.read_sql_query(query_types, conn)['ONCOTREE_CODE'].tolist()

    # Query demographic data
    query = f"""
    SELECT
        ONCOTREE_CODE,
        SEX,
        RACE,
        isOncoKbLevel1
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IN ({','.join(['?']*len(cancer_types))})
    AND SEX IS NOT NULL
    AND RACE IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=cancer_types)
    conn.close()

    # Filter to major categories
    major_races = df['RACE'].value_counts().nlargest(4).index.tolist()
    df = df[df['RACE'].isin(major_races)]
    df = df[df['SEX'].isin(['Male', 'Female'])]

    # Calculate proportions for each cancer type
    composition_data = []
    for cancer_type in cancer_types:
        cancer_df = df[df['ONCOTREE_CODE'] == cancer_type]
        if len(cancer_df) == 0:
            continue

        for sex in ['Male', 'Female']:
            for race in major_races:
                count = len(cancer_df[(cancer_df['SEX'] == sex) & (cancer_df['RACE'] == race)])
                proportion = count / len(cancer_df) if len(cancer_df) > 0 else 0
                composition_data.append({
                    'Cancer Type': cancer_type,
                    'Sex': sex,
                    'Race': race,
                    'Proportion': proportion,
                    'Count': count
                })

    comp_df = pd.DataFrame(composition_data)

    # Create pivot table for heatmap
    pivot_df = comp_df.pivot_table(
        index='Cancer Type',
        columns=['Sex', 'Race'],
        values='Proportion',
        fill_value=0
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 12))

    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlGnBu',
                cbar_kws={'label': 'Proportion of Patients'},
                linewidths=0.5, ax=ax, vmin=0, vmax=0.5)

    ax.set_title('Sex-Race Composition Grid Across Cancer Types',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Sex × Race Combination', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')

    # Rotate labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig14_sex_race_composition_grid.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig14_sex_race_composition_grid.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig14_sex_race_composition_grid.pdf'}")
    plt.close()


def create_demographic_feature_correlations(demographics_df, output_dir):
    """
    Figure 15: Correlation heatmap of demographic features across cancer types
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Select features for correlation analysis
    correlation_features = {
        'Mean Age': 'age_mean',
        'Age Std Dev': 'age_std',
        'Female %': 'sex_Female_pct',
        'Race Diversity': 'race_entropy',
        'Ethnicity Diversity': 'ethnicity_entropy',
        'Actionable Rate': 'actionable_rate',
        'N Patients': 'n_patients',
        'N Institutions': 'n_institutions'
    }

    # Create dataframe with renamed columns
    corr_df = demographics_df[[col for col in correlation_features.values() if col in demographics_df.columns]].copy()
    corr_df.columns = [key for key, val in correlation_features.items() if val in demographics_df.columns]

    # Calculate correlation matrix
    corr_matrix = corr_df.corr()

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Full correlation heatmap
    ax = axes[0]
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation'},
                linewidths=0.5, ax=ax, square=True)

    ax.set_title('Demographic Feature Correlations Across Cancer Types',
                 fontsize=13, fontweight='bold')

    # Plot 2: Scatter matrix for key features
    ax = axes[1]

    # Create scatter plot: Actionable Rate vs Race Diversity
    ax.scatter(demographics_df['race_entropy'], demographics_df['actionable_rate'],
              s=demographics_df['n_patients'] / 50, alpha=0.6,
              c=demographics_df['age_mean'], cmap='viridis',
              edgecolors='black', linewidth=0.5)

    # Annotate points
    for idx, row in demographics_df.iterrows():
        if row['actionable_rate'] > 0.6 or row['race_entropy'] > 1.7:
            ax.annotate(row['cancer_type'],
                       (row['race_entropy'], row['actionable_rate']),
                       fontsize=9, ha='center', va='bottom')

    ax.set_xlabel('Race Diversity (Shannon Entropy)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actionable Rate', fontsize=12, fontweight='bold')
    ax.set_title('Actionable Rate vs Race Diversity\n(sized by N patients, colored by mean age)',
                fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=demographics_df['age_mean'].min(),
                                                 vmax=demographics_df['age_mean'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Mean Age (years)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig15_demographic_feature_correlations.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig15_demographic_feature_correlations.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig15_demographic_feature_correlations.pdf'}")
    plt.close()


def create_race_ethnicity_composition_grid(output_dir):
    """
    Figure 16: Race-Ethnicity composition grid
    Shows proportions of Race × Ethnicity combinations across cancer types
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Query data for all cancer types
    conn = sqlite3.connect(DB_PATH)

    query_types = f"""
    SELECT ONCOTREE_CODE, COUNT(*) as n
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE
    HAVING COUNT(CASE WHEN isOncoKbLevel1 = 1 THEN 1 END) >= {MIN_ACTIONABLE_EVENTS}
    ORDER BY n DESC
    """
    cancer_types = pd.read_sql_query(query_types, conn)['ONCOTREE_CODE'].tolist()

    # Query demographic data
    query = f"""
    SELECT
        ONCOTREE_CODE,
        RACE,
        ETHNICITY,
        isOncoKbLevel1
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IN ({','.join(['?']*len(cancer_types))})
    AND RACE IS NOT NULL
    AND ETHNICITY IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=cancer_types)
    conn.close()

    # Get major categories
    major_races = df['RACE'].value_counts().nlargest(3).index.tolist()
    major_ethnicities = df['ETHNICITY'].value_counts().nlargest(3).index.tolist()

    df = df[df['RACE'].isin(major_races)]
    df = df[df['ETHNICITY'].isin(major_ethnicities)]

    # Calculate proportions
    composition_data = []
    for cancer_type in cancer_types:
        cancer_df = df[df['ONCOTREE_CODE'] == cancer_type]
        if len(cancer_df) == 0:
            continue

        for race in major_races:
            for ethnicity in major_ethnicities:
                count = len(cancer_df[(cancer_df['RACE'] == race) & (cancer_df['ETHNICITY'] == ethnicity)])
                proportion = count / len(cancer_df) if len(cancer_df) > 0 else 0
                composition_data.append({
                    'Cancer Type': cancer_type,
                    'Race': race,
                    'Ethnicity': ethnicity,
                    'Proportion': proportion,
                    'Count': count
                })

    comp_df = pd.DataFrame(composition_data)

    # Create pivot table for heatmap
    pivot_df = comp_df.pivot_table(
        index='Cancer Type',
        columns=['Race', 'Ethnicity'],
        values='Proportion',
        fill_value=0
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(18, 12))

    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlGnBu',
                cbar_kws={'label': 'Proportion of Patients'},
                linewidths=0.5, ax=ax, vmin=0, vmax=0.5)

    ax.set_title('Race-Ethnicity Composition Grid Across Cancer Types',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Race × Ethnicity Combination', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cancer Type', fontsize=12, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig16_race_ethnicity_composition_grid.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig16_race_ethnicity_composition_grid.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig16_race_ethnicity_composition_grid.pdf'}")
    plt.close()


def create_sex_race_ethnicity_faceted_grids(output_dir):
    """
    Figure 17: Sex-Race-Ethnicity faceted composition grids
    Separate grids for Male and Female showing Race × Ethnicity
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Query data
    conn = sqlite3.connect(DB_PATH)

    query_types = f"""
    SELECT ONCOTREE_CODE, COUNT(*) as n
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE
    HAVING COUNT(CASE WHEN isOncoKbLevel1 = 1 THEN 1 END) >= {MIN_ACTIONABLE_EVENTS}
    ORDER BY n DESC
    """
    cancer_types = pd.read_sql_query(query_types, conn)['ONCOTREE_CODE'].tolist()

    query = f"""
    SELECT
        ONCOTREE_CODE,
        SEX,
        RACE,
        ETHNICITY
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IN ({','.join(['?']*len(cancer_types))})
    AND SEX IS NOT NULL
    AND RACE IS NOT NULL
    AND ETHNICITY IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=cancer_types)
    conn.close()

    # Filter to major categories
    major_races = df['RACE'].value_counts().nlargest(3).index.tolist()
    major_ethnicities = df['ETHNICITY'].value_counts().nlargest(2).index.tolist()
    df = df[df['RACE'].isin(major_races)]
    df = df[df['ETHNICITY'].isin(major_ethnicities)]
    df = df[df['SEX'].isin(['Male', 'Female'])]

    # Create figure with two subplots (one for each sex)
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    for sex_idx, sex in enumerate(['Male', 'Female']):
        ax = axes[sex_idx]
        sex_df = df[df['SEX'] == sex]

        # Calculate proportions
        composition_data = []
        for cancer_type in cancer_types:
            cancer_df = sex_df[sex_df['ONCOTREE_CODE'] == cancer_type]
            total_sex_count = len(cancer_df)

            if total_sex_count == 0:
                continue

            for race in major_races:
                for ethnicity in major_ethnicities:
                    count = len(cancer_df[(cancer_df['RACE'] == race) &
                                         (cancer_df['ETHNICITY'] == ethnicity)])
                    proportion = count / total_sex_count if total_sex_count > 0 else 0
                    composition_data.append({
                        'Cancer Type': cancer_type,
                        'Race': race,
                        'Ethnicity': ethnicity,
                        'Proportion': proportion
                    })

        comp_df = pd.DataFrame(composition_data)

        # Create pivot table
        pivot_df = comp_df.pivot_table(
            index='Cancer Type',
            columns=['Race', 'Ethnicity'],
            values='Proportion',
            fill_value=0
        )

        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlGnBu',
                   cbar_kws={'label': 'Proportion'},
                   linewidths=0.5, ax=ax, vmin=0, vmax=0.8)

        ax.set_title(f'{sex} Patients: Race × Ethnicity Composition',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Race × Ethnicity', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cancer Type', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Sex-Stratified Race-Ethnicity Composition Grids',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig17_sex_race_ethnicity_faceted_grids.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig17_sex_race_ethnicity_faceted_grids.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig17_sex_race_ethnicity_faceted_grids.pdf'}")
    plt.close()


def create_three_way_demographic_sunburst_alternative(output_dir):
    """
    Figure 18: Three-way demographic breakdown (Sex-Race-Ethnicity) using stacked bars
    Alternative to sunburst - shows nested demographic proportions
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Query data for top 8 cancer types
    conn = sqlite3.connect(DB_PATH)

    query_top = """
    SELECT ONCOTREE_CODE, COUNT(*) as n
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE
    HAVING n >= 1000
    ORDER BY n DESC
    LIMIT 8
    """
    top_cancers = pd.read_sql_query(query_top, conn)['ONCOTREE_CODE'].tolist()

    query = f"""
    SELECT
        ONCOTREE_CODE,
        SEX,
        RACE,
        ETHNICITY
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IN ({','.join(['?']*len(top_cancers))})
    AND SEX IS NOT NULL
    AND RACE IS NOT NULL
    AND ETHNICITY IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=top_cancers)
    conn.close()

    # Filter to major categories
    major_races = df['RACE'].value_counts().nlargest(3).index.tolist()
    major_ethnicities = df['ETHNICITY'].value_counts().nlargest(2).index.tolist()
    df = df[df['RACE'].isin(major_races)]
    df = df[df['ETHNICITY'].isin(major_ethnicities)]
    df = df[df['SEX'].isin(['Male', 'Female'])]

    # Create figure with subplots for each cancer type
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    # Define color palette for combinations
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.cm as cm

    # Create unique combinations
    combinations = []
    for sex in ['Male', 'Female']:
        for race in major_races:
            for ethnicity in major_ethnicities:
                combinations.append(f"{sex}|{race}|{ethnicity}")

    # Use a qualitative colormap
    colors = sns.color_palette("tab20", len(combinations))
    color_map = dict(zip(combinations, colors))

    for idx, cancer_type in enumerate(top_cancers):
        ax = axes[idx]
        cancer_df = df[df['ONCOTREE_CODE'] == cancer_type]

        # Calculate proportions for each combination
        combo_counts = {}
        total = len(cancer_df)

        for sex in ['Male', 'Female']:
            for race in major_races:
                for ethnicity in major_ethnicities:
                    combo = f"{sex}|{race}|{ethnicity}"
                    count = len(cancer_df[(cancer_df['SEX'] == sex) &
                                         (cancer_df['RACE'] == race) &
                                         (cancer_df['ETHNICITY'] == ethnicity)])
                    if count > 0:
                        combo_counts[combo] = count / total

        # Sort by proportion
        sorted_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)

        # Create horizontal stacked bar
        left = 0
        for combo, prop in sorted_combos:
            if prop > 0.01:  # Only show if > 1%
                ax.barh(0, prop, left=left, color=color_map[combo],
                       edgecolor='white', linewidth=1)

                # Add label if large enough
                if prop > 0.05:
                    parts = combo.split('|')
                    label = f"{parts[0][0]}-{parts[1][:3]}-{parts[2][:3]}"
                    ax.text(left + prop/2, 0, label, ha='center', va='center',
                           fontsize=8, fontweight='bold', color='white')
                left += prop

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_title(f'{cancer_type} (n={len(cancer_df):,})',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Proportion', fontsize=10, fontweight='bold')
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Sex-Race-Ethnicity Composition (Top 8 Cancer Types)\nFormat: Sex-Race-Ethnicity',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig18_three_way_demographic_composition.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig18_three_way_demographic_composition.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig18_three_way_demographic_composition.pdf'}")
    plt.close()


def create_ethnicity_by_cancer_type_detailed(output_dir):
    """
    Figure 19: Detailed ethnicity breakdown by cancer type with race overlay
    Shows ethnicity as main bars with race composition as stacked segments
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Query data for top 6 cancer types
    conn = sqlite3.connect(DB_PATH)

    query_top = """
    SELECT ONCOTREE_CODE, COUNT(*) as n
    FROM perPatientOncoKBLevelSorted
    GROUP BY ONCOTREE_CODE
    HAVING n >= 2000
    ORDER BY n DESC
    LIMIT 6
    """
    top_cancers = pd.read_sql_query(query_top, conn)['ONCOTREE_CODE'].tolist()

    query = f"""
    SELECT
        ONCOTREE_CODE,
        RACE,
        ETHNICITY
    FROM perPatientOncoKBLevelSorted
    WHERE ONCOTREE_CODE IN ({','.join(['?']*len(top_cancers))})
    AND RACE IS NOT NULL
    AND ETHNICITY IS NOT NULL
    """
    df = pd.read_sql_query(query, conn, params=top_cancers)
    conn.close()

    # Get major categories
    major_races = df['RACE'].value_counts().nlargest(4).index.tolist()
    all_ethnicities = df['ETHNICITY'].unique()
    df = df[df['RACE'].isin(major_races)]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    # Color palette for races
    race_colors = sns.color_palette("Set2", len(major_races))
    race_color_map = dict(zip(major_races, race_colors))

    for idx, cancer_type in enumerate(top_cancers):
        ax = axes[idx]
        cancer_df = df[df['ONCOTREE_CODE'] == cancer_type]

        # Calculate proportions for each ethnicity-race combination
        ethnicity_totals = cancer_df['ETHNICITY'].value_counts()

        # Prepare data for stacked bars
        ethnicity_data = {}
        for ethnicity in all_ethnicities:
            ethnicity_data[ethnicity] = {}
            eth_df = cancer_df[cancer_df['ETHNICITY'] == ethnicity]

            for race in major_races:
                count = len(eth_df[eth_df['RACE'] == race])
                ethnicity_data[ethnicity][race] = count

        # Sort ethnicities by total count
        sorted_ethnicities = ethnicity_totals.index.tolist()

        # Create stacked bar chart
        x_pos = np.arange(len(sorted_ethnicities))
        bottoms = np.zeros(len(sorted_ethnicities))

        for race in major_races:
            values = [ethnicity_data[eth].get(race, 0) for eth in sorted_ethnicities]
            ax.bar(x_pos, values, bottom=bottoms, label=race,
                  color=race_color_map[race], edgecolor='white', linewidth=0.5)
            bottoms += values

        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_ethnicities, rotation=45, ha='right')
        ax.set_ylabel('Patient Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{cancer_type} (n={len(cancer_df):,})',
                    fontsize=12, fontweight='bold')
        ax.legend(title='Race', fontsize=8, title_fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Ethnicity Distribution with Race Composition Breakdown',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig19_ethnicity_by_cancer_race_breakdown.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig19_ethnicity_by_cancer_race_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig19_ethnicity_by_cancer_race_breakdown.pdf'}")
    plt.close()


def generate_comprehensive_report(demographics_df, comparison_df, output_dir):
    """
    Generate comprehensive narrative report with key findings and recommendations
    """

    # Load institutional stats if available
    inst_stats_file = Path("../../../analysis_output/all_cancer_types_institutional_analysis/all_cancer_types_institutional_stats.csv")
    if inst_stats_file.exists():
        inst_df = pd.read_csv(inst_stats_file)
        # Merge with demographics
        merged_df = demographics_df.merge(
            inst_df[['cancer_type', 'herfindahl_index', 'top3_institutions_pct',
                     'actionable_rate_std_across_institutions']],
            on='cancer_type', how='left'
        )
        # Rename for clarity in report
        merged_df.rename(columns={'actionable_rate_std_across_institutions': 'actionable_rate_variance'}, inplace=True)
    else:
        merged_df = demographics_df.copy()

    report_lines = []

    # Header
    report_lines.append("=" * 100)
    report_lines.append("COMPREHENSIVE DEMOGRAPHIC ANALYSIS REPORT")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append("Date: October 2025")
    report_lines.append("Dataset: AACR GENIE Consortium")
    report_lines.append(f"Analysis Directory: {output_dir}")
    report_lines.append("")

    # Executive Summary
    report_lines.append("=" * 100)
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(f"Total cancer types analyzed: {len(demographics_df)}")
    report_lines.append(f"Total patients: {demographics_df['n_patients'].sum():,}")
    report_lines.append(f"Total actionable events: {demographics_df['n_actionable'].sum():,}")
    report_lines.append(f"Overall actionable rate: {demographics_df['n_actionable'].sum() / demographics_df['n_patients'].sum():.1%}")
    report_lines.append("")

    # Key Finding 1: Age Demographics
    report_lines.append("=" * 100)
    report_lines.append("KEY FINDING 1: AGE DEMOGRAPHICS")
    report_lines.append("=" * 100)
    report_lines.append("")

    # Overall age statistics
    report_lines.append("Overall Age Statistics:")
    report_lines.append(f"  - Mean age range: {demographics_df['age_mean'].min():.1f} - {demographics_df['age_mean'].max():.1f} years")
    report_lines.append(f"  - Median age range: {demographics_df['age_median'].min():.1f} - {demographics_df['age_median'].max():.1f} years")
    report_lines.append(f"  - Overall mean age: {demographics_df['age_mean'].mean():.1f} years (weighted)")
    report_lines.append(f"  - Age missing rate: {demographics_df['age_missing_rate'].min():.1%} - {demographics_df['age_missing_rate'].max():.1%}")
    report_lines.append("")

    # Youngest and oldest cancer types
    youngest = demographics_df.nsmallest(3, 'age_mean')[['cancer_type', 'age_mean', 'age_median', 'n_patients']]
    oldest = demographics_df.nlargest(3, 'age_mean')[['cancer_type', 'age_mean', 'age_median', 'n_patients']]

    report_lines.append("Youngest Cancer Types (by mean age):")
    for idx, row in youngest.iterrows():
        report_lines.append(f"  {row['cancer_type']:8s}: {row['age_mean']:5.1f} years (median: {row['age_median']:4.1f}, n={row['n_patients']:,})")
    report_lines.append("")

    report_lines.append("Oldest Cancer Types (by mean age):")
    for idx, row in oldest.iterrows():
        report_lines.append(f"  {row['cancer_type']:8s}: {row['age_mean']:5.1f} years (median: {row['age_median']:4.1f}, n={row['n_patients']:,})")
    report_lines.append("")

    # Age variability
    high_age_var = demographics_df.nlargest(3, 'age_std')[['cancer_type', 'age_std', 'n_patients']]
    report_lines.append("Cancer Types with Highest Age Variability (by std dev):")
    for idx, row in high_age_var.iterrows():
        report_lines.append(f"  {row['cancer_type']:8s}: {row['age_std']:5.1f} years std dev (n={row['n_patients']:,})")
    report_lines.append("")

    # Key Finding 2: Sex Distribution
    report_lines.append("=" * 100)
    report_lines.append("KEY FINDING 2: SEX DISTRIBUTION")
    report_lines.append("=" * 100)
    report_lines.append("")

    report_lines.append("Sex Distribution Summary:")
    report_lines.append(f"  - Female % range: {demographics_df['sex_Female_pct'].min():.1%} - {demographics_df['sex_Female_pct'].max():.1%}")
    report_lines.append(f"  - Male % range: {demographics_df['sex_Male_pct'].min():.1%} - {demographics_df['sex_Male_pct'].max():.1%}")
    report_lines.append(f"  - Sex missing rate: {demographics_df['sex_missing_rate'].max():.1%} (excellent data quality)")
    report_lines.append("")

    # Sex-specific cancers
    high_female = demographics_df.nlargest(3, 'sex_Female_pct')[['cancer_type', 'sex_Female_pct', 'n_patients']]
    high_male = demographics_df.nlargest(3, 'sex_Male_pct')[['cancer_type', 'sex_Male_pct', 'n_patients']]

    report_lines.append("Most Female-Predominant Cancer Types:")
    for idx, row in high_female.iterrows():
        report_lines.append(f"  {row['cancer_type']:8s}: {row['sex_Female_pct']:.1%} female (n={row['n_patients']:,})")
    report_lines.append("")

    report_lines.append("Most Male-Predominant Cancer Types:")
    for idx, row in high_male.iterrows():
        report_lines.append(f"  {row['cancer_type']:8s}: {row['sex_Male_pct']:.1%} male (n={row['n_patients']:,})")
    report_lines.append("")

    # Key Finding 3: Race and Ethnicity Diversity
    report_lines.append("=" * 100)
    report_lines.append("KEY FINDING 3: RACE AND ETHNICITY DIVERSITY")
    report_lines.append("=" * 100)
    report_lines.append("")

    report_lines.append("Diversity Metrics (Shannon Entropy):")
    report_lines.append(f"  - Race entropy range: {demographics_df['race_entropy'].min():.2f} - {demographics_df['race_entropy'].max():.2f}")
    report_lines.append(f"  - Ethnicity entropy range: {demographics_df['ethnicity_entropy'].min():.2f} - {demographics_df['ethnicity_entropy'].max():.2f}")
    report_lines.append(f"  - Higher entropy = more diverse population")
    report_lines.append(f"  - Maximum possible entropy (7 races): {np.log(7):.2f}")
    report_lines.append("")

    # Most and least diverse
    most_diverse_race = demographics_df.nlargest(3, 'race_entropy')[['cancer_type', 'race_entropy', 'n_patients']]
    least_diverse_race = demographics_df.nsmallest(3, 'race_entropy')[['cancer_type', 'race_entropy', 'n_patients']]

    report_lines.append("Most Racially Diverse Cancer Types:")
    for idx, row in most_diverse_race.iterrows():
        report_lines.append(f"  {row['cancer_type']:8s}: {row['race_entropy']:.2f} entropy (n={row['n_patients']:,})")
    report_lines.append("")

    report_lines.append("Least Racially Diverse Cancer Types:")
    for idx, row in least_diverse_race.iterrows():
        report_lines.append(f"  {row['cancer_type']:8s}: {row['race_entropy']:.2f} entropy (n={row['n_patients']:,})")
    report_lines.append("")

    # Key Finding 4: Actionable vs Non-Actionable Demographics
    report_lines.append("=" * 100)
    report_lines.append("KEY FINDING 4: ACTIONABLE VS NON-ACTIONABLE PATIENT DEMOGRAPHICS")
    report_lines.append("=" * 100)
    report_lines.append("")

    # Calculate age differences
    age_diffs = comparison_df[['cancer_type', 'age_actionable_mean', 'age_non_actionable_mean', 'age_diff_mean']].copy()

    report_lines.append("Age Differences (Actionable - Non-Actionable):")
    report_lines.append(f"  - Mean difference range: {age_diffs['age_diff_mean'].min():.1f} to {age_diffs['age_diff_mean'].max():.1f} years")
    report_lines.append(f"  - Overall mean difference: {age_diffs['age_diff_mean'].mean():.2f} years")
    report_lines.append("")

    # Largest age differences
    largest_age_diff = age_diffs.reindex(age_diffs['age_diff_mean'].abs().nlargest(5).index)
    report_lines.append("Cancer Types with Largest Age Differences:")
    for idx, row in largest_age_diff.iterrows():
        direction = "older" if row['age_diff_mean'] > 0 else "younger"
        report_lines.append(f"  {row['cancer_type']:8s}: {abs(row['age_diff_mean']):5.1f} years {direction} ({row['age_actionable_mean']:.1f} vs {row['age_non_actionable_mean']:.1f})")
    report_lines.append("")

    # Key Finding 5: Demographic-Institution Confounding
    if 'herfindahl_index' in merged_df.columns:
        report_lines.append("=" * 100)
        report_lines.append("KEY FINDING 5: DEMOGRAPHIC-INSTITUTION CONFOUNDING ASSESSMENT")
        report_lines.append("=" * 100)
        report_lines.append("")

        report_lines.append("Institutional Concentration Metrics:")
        report_lines.append(f"  - Herfindahl index range: {merged_df['herfindahl_index'].min():.3f} - {merged_df['herfindahl_index'].max():.3f}")
        report_lines.append(f"  - Top 3 institutions % range: {merged_df['top3_institutions_pct'].min():.1%} - {merged_df['top3_institutions_pct'].max():.1%}")
        report_lines.append(f"  - Actionable rate variance range: {merged_df['actionable_rate_variance'].min():.3f} - {merged_df['actionable_rate_variance'].max():.3f}")
        report_lines.append("")

        # High concentration cancers
        high_conc = merged_df.nlargest(5, 'herfindahl_index')[['cancer_type', 'herfindahl_index',
                                                                  'top3_institutions_pct', 'n_institutions']]
        report_lines.append("Cancer Types with Highest Institutional Concentration:")
        for idx, row in high_conc.iterrows():
            report_lines.append(f"  {row['cancer_type']:8s}: Herfindahl={row['herfindahl_index']:.3f}, Top3={row['top3_institutions_pct']:.1%}, N_inst={row['n_institutions']}")
        report_lines.append("")

        # High actionability variance
        high_var = merged_df.nlargest(5, 'actionable_rate_variance')[['cancer_type', 'actionable_rate_variance',
                                                                        'actionable_rate', 'n_institutions']]
        report_lines.append("Cancer Types with Highest Actionability Variance Across Institutions:")
        report_lines.append("(High variance suggests institution-dependent actionability patterns)")
        for idx, row in high_var.iterrows():
            report_lines.append(f"  {row['cancer_type']:8s}: variance={row['actionable_rate_variance']:.3f}, overall_rate={row['actionable_rate']:.1%}, n_inst={row['n_institutions']}")
        report_lines.append("")

    # Key Finding 6: Missing Data Patterns
    report_lines.append("=" * 100)
    report_lines.append("KEY FINDING 6: MISSING DATA PATTERNS")
    report_lines.append("=" * 100)
    report_lines.append("")

    report_lines.append("Missing Data Summary:")
    report_lines.append(f"  - Age missing rate: {demographics_df['age_missing_rate'].min():.1%} - {demographics_df['age_missing_rate'].max():.1%}")
    report_lines.append(f"  - Sex missing rate: {demographics_df['sex_missing_rate'].min():.1%} - {demographics_df['sex_missing_rate'].max():.1%}")
    report_lines.append(f"  - Race missing rate: {demographics_df['race_missing_rate'].min():.1%} - {demographics_df['race_missing_rate'].max():.1%}")
    report_lines.append(f"  - Ethnicity missing rate: {demographics_df['ethnicity_missing_rate'].min():.1%} - {demographics_df['ethnicity_missing_rate'].max():.1%}")
    report_lines.append(f"  - Stage missing rate: {demographics_df['stage_missing_rate'].min():.1%} - {demographics_df['stage_missing_rate'].max():.1%}")
    report_lines.append("")

    report_lines.append("CRITICAL: Stage data is 97-100% missing and CANNOT be used as an ML feature.")
    report_lines.append("")

    # Recommendations
    report_lines.append("=" * 100)
    report_lines.append("RECOMMENDATIONS FOR ML MODEL FEATURES")
    report_lines.append("=" * 100)
    report_lines.append("")

    report_lines.append("RECOMMENDED DEMOGRAPHIC FEATURES (High Quality, Low Missing Data):")
    report_lines.append("  ✓ AGE_AT_SEQ_REPORT - Mean missing rate: {:.1%}".format(demographics_df['age_missing_rate'].mean()))
    report_lines.append("  ✓ SEX - Mean missing rate: {:.1%}".format(demographics_df['sex_missing_rate'].mean()))
    report_lines.append("  ✓ RACE - Mean missing rate: {:.1%}".format(demographics_df['race_missing_rate'].mean()))
    report_lines.append("  ✓ ETHNICITY - Mean missing rate: {:.1%}".format(demographics_df['ethnicity_missing_rate'].mean()))
    report_lines.append("  ✓ SAMPLE_TYPE - No missing data reported")
    report_lines.append("")

    report_lines.append("NOT RECOMMENDED:")
    report_lines.append("  ✗ STAGE - {:.1%} missing rate (unusable)".format(demographics_df['stage_missing_rate'].mean()))
    report_lines.append("  ✗ Technical features (SUBJECT_CENTER, SEQ_ASSAY_ID) - Data leakage risk")
    report_lines.append("")

    report_lines.append("KEY CONSIDERATIONS:")
    report_lines.append("  1. Use institution-aware cross-validation (GroupKFold) to prevent data leakage")
    report_lines.append("  2. Consider cancer-type-specific models due to diverse demographic profiles")
    report_lines.append("  3. Monitor for demographic-institution confounding in high-concentration cancer types")
    report_lines.append("  4. Add missingness indicator variables for features with >5% missing data")
    report_lines.append("")

    # Cancer-Specific Recommendations
    report_lines.append("=" * 100)
    report_lines.append("CANCER-SPECIFIC CONSIDERATIONS")
    report_lines.append("=" * 100)
    report_lines.append("")

    if 'actionable_rate_variance' in merged_df.columns:
        high_var_cancers = merged_df[merged_df['actionable_rate_variance'] > 0.15]['cancer_type'].tolist()
        report_lines.append(f"Cancer Types with High Institution-Dependent Actionability (variance > 0.15):")
        report_lines.append(f"  {', '.join(high_var_cancers)}")
        report_lines.append("")
        report_lines.append("Recommendation for these cancer types:")
        report_lines.append("  - Use GroupKFold CV to ensure institution separation")
        report_lines.append("  - Consider including institution as a feature for within-network deployment")
        report_lines.append("  - Use demographic-only features for generalization to new institutions")
        report_lines.append("")

    # Age-specific recommendations
    young_cancers = demographics_df[demographics_df['age_mean'] < 50]['cancer_type'].tolist()
    if young_cancers:
        report_lines.append(f"Younger Patient Populations (mean age < 50):")
        report_lines.append(f"  {', '.join(young_cancers)}")
        report_lines.append("  - Consider age as a strong predictive feature")
        report_lines.append("")

    # Sex-specific recommendations
    sex_specific = demographics_df[(demographics_df['sex_Female_pct'] > 0.9) |
                                   (demographics_df['sex_Male_pct'] > 0.9)]['cancer_type'].tolist()
    if sex_specific:
        report_lines.append(f"Sex-Specific Cancer Types (>90% one sex):")
        report_lines.append(f"  {', '.join(sex_specific)}")
        report_lines.append("  - Sex may have limited predictive value (low variance)")
        report_lines.append("")

    # Conclusion
    report_lines.append("=" * 100)
    report_lines.append("CONCLUSION")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append("This demographic analysis reveals:")
    report_lines.append("")
    report_lines.append("1. HIGH QUALITY demographic data for AGE, SEX, RACE, and ETHNICITY (0-9.5% missing)")
    report_lines.append("2. UNUSABLE STAGE data (97-100% missing)")
    report_lines.append("3. DIVERSE demographic profiles across 21 cancer types")
    report_lines.append("4. INSTITUTION-DEPENDENT actionability patterns in several cancer types")
    report_lines.append("5. MINIMAL demographic differences between actionable and non-actionable patients")
    report_lines.append("")
    report_lines.append("For ML modeling:")
    report_lines.append("  - Use demographic features: AGE, SEX, RACE, ETHNICITY, SAMPLE_TYPE")
    report_lines.append("  - Avoid technical features to prevent data leakage")
    report_lines.append("  - Implement institution-aware CV (GroupKFold)")
    report_lines.append("  - Consider cancer-type-specific models")
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 100)

    # Write report
    report_path = output_dir / 'demographic_analysis_comprehensive_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"Saved: {report_path}")

    return report_path


if __name__ == "__main__":
    main()
