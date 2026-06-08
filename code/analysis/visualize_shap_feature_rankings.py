"""
Visualize SHAP Feature Rankings Across Cancer Types

Creates comprehensive heatmap visualizations of ML feature importance rankings:
1. Rank-based heatmaps (with and without clustering)
2. SHAP value-based heatmaps (with and without clustering)
3. Top features comparison across cancer types
4. Feature category analysis

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import warnings

# Add tools to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools''))
from db_config import get_default_db_path
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = "/Users/larsonhogstrom/Documents/balder_analysis/output/actionability_db_curration_20250522b/demographic_technical_ml_modeling/shap_feature_ranks.csv"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "analysis_output" / "shap_feature_rankings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visualization parameters
TOP_N_FEATURES = 20  # Number of top features to display
FIGSIZE_LARGE = (16, 12)
FIGSIZE_MEDIUM = (14, 10)
FIGSIZE_SMALL = (12, 8)


def load_and_prepare_data(data_file):
    """Load SHAP feature rankings and prepare for visualization"""
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)

    print(f"  Loaded {len(df):,} feature rankings")
    print(f"  Cancer types: {df['ONCOTREE_CODE'].nunique()}")
    print(f"  Unique features: {df['feature'].nunique()}")

    return df


def extract_feature_category(feature_name):
    """Extract feature category from feature name

    Examples:
    - cat__SEX_Female → SEX
    - cat__RACE_Asian → RACE
    - num__AGE_AT_SEQ_REPORT → AGE
    - cat__library_selection_PCR → library_selection
    """
    # Remove pipeline prefix
    if '__' in feature_name:
        clean_feature = feature_name.split('__', 1)[1]
    else:
        clean_feature = feature_name

    # Extract category (first part before underscore)
    if '_' in clean_feature:
        category = clean_feature.split('_')[0]
    else:
        category = clean_feature

    return category


def create_rank_heatmap(df, top_n=20, cluster=False, output_file=None):
    """
    Create heatmap of feature ranks across cancer types

    Parameters
    ----------
    df : pd.DataFrame
        SHAP feature rankings
    top_n : int
        Number of top features to include
    cluster : bool
        Whether to apply hierarchical clustering
    output_file : Path
        Output file path
    """
    print(f"\nCreating rank heatmap (top {top_n} features, cluster={cluster})...")

    # Get top features (by minimum rank across all cancer types)
    feature_ranks = df.groupby('feature')['feature_rank'].min().sort_values()
    top_features = feature_ranks.head(top_n).index.tolist()

    # Create pivot table: features × cancer types
    rank_matrix = df[df['feature'].isin(top_features)].pivot(
        index='feature',
        columns='ONCOTREE_CODE',
        values='feature_rank'
    )

    # Fill NaN with max_rank + 1 (feature not in top N for that cancer type)
    max_rank = rank_matrix.max().max()
    rank_matrix_filled = rank_matrix.fillna(max_rank + 10)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    # Cluster if requested
    if cluster:
        # Cluster rows (features)
        row_linkage = linkage(pdist(rank_matrix_filled.fillna(max_rank + 10), metric='euclidean'), method='ward')
        row_order = dendrogram(row_linkage, no_plot=True)['leaves']

        # Cluster columns (cancer types)
        col_linkage = linkage(pdist(rank_matrix_filled.T.fillna(max_rank + 10), metric='euclidean'), method='ward')
        col_order = dendrogram(col_linkage, no_plot=True)['leaves']

        # Reorder matrix
        rank_matrix_ordered = rank_matrix_filled.iloc[row_order, col_order]
    else:
        rank_matrix_ordered = rank_matrix_filled

    # Create heatmap
    # Reverse colormap so darker = more important (lower rank number)
    sns.heatmap(
        rank_matrix_ordered,
        cmap='YlOrRd_r',  # Reversed: darker = lower rank = more important
        cbar_kws={'label': 'Feature Rank (darker = more important)'},
        linewidths=0.5,
        linecolor='lightgray',
        ax=ax,
        vmin=1,
        vmax=top_n,
        mask=rank_matrix_ordered > max_rank  # Mask features not in top N
    )

    # Customize
    ax.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')

    cluster_text = " (Hierarchical Clustering)" if cluster else ""
    ax.set_title(f'Feature Importance Rankings Across Cancer Types{cluster_text}\n' +
                 f'Top {top_n} Features • Darker = More Important',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file.name}")

    plt.close()

    return rank_matrix_ordered


def create_shap_value_heatmap(df, top_n=20, cluster=False, output_file=None):
    """
    Create heatmap of SHAP values across cancer types

    Parameters
    ----------
    df : pd.DataFrame
        SHAP feature rankings
    top_n : int
        Number of top features to include
    cluster : bool
        Whether to apply hierarchical clustering
    output_file : Path
        Output file path
    """
    print(f"\nCreating SHAP value heatmap (top {top_n} features, cluster={cluster})...")

    # Get top features (by maximum SHAP value across all cancer types)
    feature_shaps = df.groupby('feature')['mean_abs_shap'].max().sort_values(ascending=False)
    top_features = feature_shaps.head(top_n).index.tolist()

    # Create pivot table: features × cancer types
    shap_matrix = df[df['feature'].isin(top_features)].pivot(
        index='feature',
        columns='ONCOTREE_CODE',
        values='mean_abs_shap'
    )

    # Fill NaN with 0 (feature not important for that cancer type)
    shap_matrix_filled = shap_matrix.fillna(0)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)

    # Cluster if requested
    if cluster:
        # Cluster rows (features)
        row_linkage = linkage(pdist(shap_matrix_filled, metric='euclidean'), method='ward')
        row_order = dendrogram(row_linkage, no_plot=True)['leaves']

        # Cluster columns (cancer types)
        col_linkage = linkage(pdist(shap_matrix_filled.T, metric='euclidean'), method='ward')
        col_order = dendrogram(col_linkage, no_plot=True)['leaves']

        # Reorder matrix
        shap_matrix_ordered = shap_matrix_filled.iloc[row_order, col_order]
    else:
        # Order by overall importance
        shap_matrix_ordered = shap_matrix_filled.loc[
            shap_matrix_filled.mean(axis=1).sort_values(ascending=False).index
        ]

    # Create heatmap
    sns.heatmap(
        shap_matrix_ordered,
        cmap='YlOrRd',  # Darker = higher SHAP value = more important
        cbar_kws={'label': 'Mean Absolute SHAP Value'},
        linewidths=0.5,
        linecolor='lightgray',
        ax=ax,
        vmin=0
    )

    # Customize
    ax.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')

    cluster_text = " (Hierarchical Clustering)" if cluster else ""
    ax.set_title(f'SHAP Feature Importance Values Across Cancer Types{cluster_text}\n' +
                 f'Top {top_n} Features by Maximum SHAP Value',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file.name}")

    plt.close()

    return shap_matrix_ordered


def create_top_features_comparison(df, top_n=10, output_file=None):
    """
    Create barplot comparing top N features across cancer types

    Shows which features are consistently important
    """
    print(f"\nCreating top {top_n} features comparison...")

    # Get top N features for each cancer type
    top_features_per_cancer = df[df['feature_rank'] <= top_n].copy()

    # Count how many cancer types each feature appears in
    feature_prevalence = top_features_per_cancer['feature'].value_counts()

    # Get top features by prevalence
    top_prevalent_features = feature_prevalence.head(20)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    # Plot 1: Feature prevalence (how many cancer types)
    top_prevalent_features.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Number of Cancer Types', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Feature', fontsize=11, fontweight='bold')
    ax1.set_title(f'Most Prevalent Features in Top {top_n}\n(Across Cancer Types)',
                  fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # Plot 2: Average rank of top features
    avg_rank = top_features_per_cancer.groupby('feature')['feature_rank'].mean().sort_values()
    top_avg_rank = avg_rank.head(20)

    top_avg_rank.plot(kind='barh', ax=ax2, color='coral')
    ax2.set_xlabel('Average Rank', fontsize=11, fontweight='bold')
    ax2.set_ylabel('')  # Remove duplicate ylabel
    ax2.set_title(f'Top Features by Average Rank\n(When in Top {top_n})',
                  fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.invert_xaxis()  # Lower rank = better
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file.name}")

    plt.close()


def create_feature_category_heatmap(df, top_categories=15, output_file=None):
    """
    Create heatmap of feature categories across cancer types

    Groups features by category (SEX, RACE, AGE, etc.)
    """
    print(f"\nCreating feature category heatmap (top {top_categories} categories)...")

    # Extract categories
    df['category'] = df['feature'].apply(extract_feature_category)

    # Calculate average rank per category per cancer type
    category_ranks = df.groupby(['ONCOTREE_CODE', 'category'])['feature_rank'].mean().reset_index()

    # Create pivot table
    category_matrix = category_ranks.pivot(
        index='category',
        columns='ONCOTREE_CODE',
        values='feature_rank'
    )

    # Get top categories (by average rank across all cancer types)
    category_avg_rank = category_matrix.mean(axis=1).sort_values()
    top_cats = category_avg_rank.head(top_categories).index.tolist()

    category_matrix_top = category_matrix.loc[top_cats]

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)

    # Create heatmap (reversed colormap: darker = more important)
    sns.heatmap(
        category_matrix_top,
        cmap='YlOrRd_r',
        cbar_kws={'label': 'Average Feature Rank'},
        linewidths=0.5,
        linecolor='lightgray',
        ax=ax,
        annot=False
    )

    # Customize
    ax.set_xlabel('Cancer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Category', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Category Importance Across Cancer Types\n' +
                 f'Top {top_categories} Categories by Average Rank',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file.name}")

    plt.close()


def create_demographic_vs_technical_comparison(df, output_file=None):
    """
    Compare demographic vs technical feature importance
    """
    print("\nCreating demographic vs technical comparison...")

    # Define categories. 'ANCESTRY' added 2026-05-22 (pre-lock review #6); the
    # Python identifier was renamed RACE → ANCESTRY by Feature Review Fix b
    # (2026-05-21), so post-rename SHAP feature names use 'Ancestry: ...' /
    # 'ANCESTRY_...' forms. RACE is kept for back-compat with pre-rename
    # rankings. _MISSING is added so the new ANCESTRY_missing / ETHNICITY_missing
    # indicator features also bucket as Demographic (their institutional-signal
    # interpretation is still a demographic-pipeline concern).
    demographic_keywords = ['SEX', 'RACE', 'ANCESTRY', 'ETHNICITY', 'AGE', '_MISSING']
    technical_keywords = ['library', 'calling', 'coverage', 'instrument', 'platform',
                         'target', 'CENTER', 'SUBJECT', 'pipeline', 'assay']

    def categorize_feature(feature):
        feature_upper = feature.upper()
        if any(kw in feature_upper for kw in demographic_keywords):
            return 'Demographic'
        elif any(kw in feature_upper for kw in technical_keywords):
            return 'Technical'
        else:
            return 'Other'

    df['feature_type'] = df['feature'].apply(categorize_feature)

    # Calculate average rank by feature type and cancer type
    avg_ranks = df.groupby(['ONCOTREE_CODE', 'feature_type'])['feature_rank'].mean().reset_index()

    # Create pivot for plotting
    pivot_data = avg_ranks.pivot(
        index='ONCOTREE_CODE',
        columns='feature_type',
        values='feature_rank'
    )

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_MEDIUM)

    # Plot 1: Grouped bar chart
    pivot_data[['Demographic', 'Technical']].plot(
        kind='bar',
        ax=ax1,
        color=['steelblue', 'coral']
    )
    ax1.set_xlabel('Cancer Type', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Feature Rank', fontsize=11, fontweight='bold')
    ax1.set_title('Demographic vs Technical Features\nAverage Rank by Cancer Type',
                  fontsize=12, fontweight='bold')
    ax1.legend(title='Feature Type', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.invert_yaxis()  # Lower rank = better

    # Plot 2: Scatter plot
    ax2.scatter(pivot_data['Demographic'], pivot_data['Technical'],
                s=100, alpha=0.6, c='steelblue', edgecolors='black')

    # Add diagonal line
    min_val = min(pivot_data['Demographic'].min(), pivot_data['Technical'].min())
    max_val = max(pivot_data['Demographic'].max(), pivot_data['Technical'].max())
    ax2.plot([min_val, max_val], [min_val, max_val],
             'r--', alpha=0.5, label='Equal importance')

    # Annotate points
    for idx, row in pivot_data.iterrows():
        ax2.annotate(idx, (row['Demographic'], row['Technical']),
                    fontsize=8, ha='center', va='bottom')

    ax2.set_xlabel('Demographic Features Avg Rank', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Technical Features Avg Rank', fontsize=11, fontweight='bold')
    ax2.set_title('Demographic vs Technical Feature Importance\nLower = More Important',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.invert_xaxis()
    ax2.invert_yaxis()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file.name}")

    plt.close()


def create_summary_statistics(df, output_file=None):
    """Create summary statistics table"""
    print("\nCreating summary statistics...")

    # Overall statistics
    stats = {
        'Total Features': df['feature'].nunique(),
        'Cancer Types': df['ONCOTREE_CODE'].nunique(),
        'Total Feature Rankings': len(df),
        'Avg Features per Cancer Type': len(df) / df['ONCOTREE_CODE'].nunique(),
        'Top Feature (Most Prevalent)': df[df['feature_rank'] == 1]['feature'].mode()[0] if len(df[df['feature_rank'] == 1]) > 0 else 'N/A',
        'Median SHAP Value': df['mean_abs_shap'].median(),
        'Max SHAP Value': df['mean_abs_shap'].max()
    }

    # Create summary dataframe
    summary_df = pd.DataFrame([stats]).T
    summary_df.columns = ['Value']

    if output_file:
        summary_df.to_csv(output_file)
        print(f"  Saved: {output_file.name}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:.<50} {value:.4f}")
        else:
            print(f"{key:.<50} {value}")
    print("="*60)


def main():
    """Main execution function"""

    print("="*80)
    print("SHAP FEATURE RANKINGS VISUALIZATION")
    print("="*80)

    # Load data
    df = load_and_prepare_data(DATA_FILE)

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # 1. Rank-based heatmaps
    create_rank_heatmap(
        df,
        top_n=TOP_N_FEATURES,
        cluster=False,
        output_file=OUTPUT_DIR / "heatmap_ranks_no_clustering.pdf"
    )

    create_rank_heatmap(
        df,
        top_n=TOP_N_FEATURES,
        cluster=True,
        output_file=OUTPUT_DIR / "heatmap_ranks_with_clustering.pdf"
    )

    # 2. SHAP value-based heatmaps
    create_shap_value_heatmap(
        df,
        top_n=TOP_N_FEATURES,
        cluster=False,
        output_file=OUTPUT_DIR / "heatmap_shap_values_no_clustering.pdf"
    )

    create_shap_value_heatmap(
        df,
        top_n=TOP_N_FEATURES,
        cluster=True,
        output_file=OUTPUT_DIR / "heatmap_shap_values_with_clustering.pdf"
    )

    # 3. Top features comparison
    create_top_features_comparison(
        df,
        top_n=10,
        output_file=OUTPUT_DIR / "top_features_comparison.pdf"
    )

    # 4. Feature category heatmap
    create_feature_category_heatmap(
        df,
        top_categories=15,
        output_file=OUTPUT_DIR / "heatmap_feature_categories.pdf"
    )

    # 5. Demographic vs technical comparison
    create_demographic_vs_technical_comparison(
        df,
        output_file=OUTPUT_DIR / "demographic_vs_technical_comparison.pdf"
    )

    # 6. Summary statistics
    create_summary_statistics(
        df,
        output_file=OUTPUT_DIR / "summary_statistics.csv"
    )

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
