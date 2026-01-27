#!/usr/bin/env python3
"""
Plot SHAP Feature Ranks Across Cancer Types

Creates visualizations showing how features rank (by mean |SHAP|) across different
cancer types, similar to the R script inspect_rf_shap_values.r.

Plots generated:
1. Top features rank plot - Line/point plot showing rank of top features across cancer types
2. Feature rank boxplot - Boxplot showing rank distribution for each feature, faceted by type
3. Feature rank heatmap - Heatmap showing ranks across cancer types

Usage:
    # Auto-detect latest SHAP output
    python plot_shap_feature_ranks.py

    # Specify SHAP directory
    python plot_shap_feature_ranks.py --shap-dir ../../../analysis_output/shap_primary_values/run_20260119_085841

    # Custom output directory
    python plot_shap_feature_ranks.py --output-dir /path/to/output

"""

import sys
import os
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'shap_interactions'))

# Import feature type classifier from feature_name_mapper
from feature_name_mapper import classify_feature_type, DEMOGRAPHIC_FEATURES, TECHNICAL_FEATURES

# Repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_default_shap_dir():
    """Find latest SHAP primary values output directory."""
    # Check analysis_output
    shap_output = REPO_ROOT / "analysis_output" / "shap_primary_values"
    if shap_output.exists():
        runs = sorted(shap_output.glob("run_*"), reverse=True)
        for run_dir in runs:
            if (run_dir / "combined_feature_importance.csv").exists():
                return run_dir

    # Check unified runs directory
    runs_dir = REPO_ROOT / "runs"
    if runs_dir.exists():
        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        for run_dir in runs:
            shap_dir = run_dir / "tier6_shap_analysis" / "primary_values"
            if shap_dir.exists() and (shap_dir / "combined_feature_importance.csv").exists():
                return shap_dir

    return None


def load_grouped_shap_data(shap_dir):
    """
    Load grouped feature importance data from all cancer types.

    Parameters
    ----------
    shap_dir : Path
        Directory containing SHAP primary values output

    Returns
    -------
    pd.DataFrame
        Combined dataframe with columns: feature_name, mean_abs_shap, rank, cancer_type, feature_type
    """
    shap_dir = Path(shap_dir)

    all_data = []

    # Determine the actual data directory
    # Check if there's a nested run_* directory (from run_all_analyses.py output structure)
    run_dirs = sorted(shap_dir.glob('run_*'), reverse=True)
    if run_dirs:
        # Use the latest run directory
        data_dir = run_dirs[0]
        print(f"Using SHAP run directory: {data_dir.name}")
    else:
        # Files directly in shap_dir
        data_dir = shap_dir

    # Find all cancer type subdirectories
    for subdir in data_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.') and not subdir.name.startswith('run_'):
            grouped_file = subdir / f"{subdir.name}_grouped_feature_importance.csv"
            if grouped_file.exists():
                df = pd.read_csv(grouped_file)
                df['cancer_type'] = subdir.name
                all_data.append(df)

    if not all_data:
        raise FileNotFoundError(f"No grouped feature importance files found in {shap_dir} (checked {data_dir})")

    combined = pd.concat(all_data, ignore_index=True)

    # Classify features using the imported classifier
    combined['feature_type'] = combined['feature_name'].apply(classify_feature_type)

    print(f"Loaded {len(combined)} feature rankings across {combined['cancer_type'].nunique()} cancer types")
    print(f"Feature types: {combined['feature_type'].value_counts().to_dict()}")

    return combined


def plot_top5_features_rank(df, output_dir):
    """
    Create line/point plot showing rank of exactly TOP 5 features across cancer types.

    Shows the 5 features that most frequently appear in the top 5 rankings.
    Each feature gets a distinct color.
    Similar to shap_feature_rank_plot.pdf from R script.
    """
    # Find features that appear in top 5 across cancer types
    # Count how many cancer types each feature appears in top 5
    top5_counts = df[df['rank'] <= 5].groupby('feature_name').size().sort_values(ascending=False)

    # Select exactly the top 5 most frequent features
    selected_features = top5_counts.head(5).index.tolist()

    # Filter to selected features (show all their ranks across all cancer types)
    plot_df = df[df['feature_name'].isin(selected_features)].copy()

    if len(plot_df) == 0:
        print("Warning: No data for top 5 features rank plot")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Use 5 distinct, easily distinguishable colors
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']  # Red, Blue, Green, Purple, Orange
    markers = ['o', 's', '^', 'D', 'v']  # Circle, Square, Triangle, Diamond, Inverted Triangle
    n_features = len(selected_features)

    # Sort cancer types alphabetically for consistent x-axis
    cancer_types = sorted(df['cancer_type'].unique())

    for i, feature in enumerate(selected_features):
        feat_data = plot_df[plot_df['feature_name'] == feature].copy()
        feat_data = feat_data.set_index('cancer_type').reindex(cancer_types).reset_index()

        # Plot line connecting points
        ax.plot(feat_data['cancer_type'], feat_data['rank'],
                marker=markers[i % len(markers)], markersize=10, linewidth=2, alpha=0.7,
                color=colors[i % len(colors)], label=feature)

    ax.set_ylabel('Feature Rank', fontsize=12)
    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_title('Top 5 SHAP Features: Rank Across Cancer Types', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Rank 1 at top

    # Set y-axis to show reasonable range
    max_rank = plot_df['rank'].max()
    ax.set_ylim(min(max_rank + 2, 25), 0)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Top 5 threshold')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    output_file = output_dir / 'shap_top5_feature_rank_plot.pdf'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

    # Also save PNG for quick viewing
    png_file = output_dir / 'shap_top5_feature_rank_plot.png'
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, feature in enumerate(selected_features):
        feat_data = plot_df[plot_df['feature_name'] == feature].copy()
        feat_data = feat_data.set_index('cancer_type').reindex(cancer_types).reset_index()
        ax.plot(feat_data['cancer_type'], feat_data['rank'],
                marker=markers[i % len(markers)], markersize=10, linewidth=2, alpha=0.7,
                color=colors[i % len(colors)], label=feature)
    ax.set_ylabel('Feature Rank', fontsize=12)
    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_title('Top 5 SHAP Features: Rank Across Cancer Types', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_ylim(min(max_rank + 2, 25), 0)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_file}")


def plot_feature_rank_boxplot_all(df, output_dir):
    """
    Create boxplot showing rank distribution for ALL features, faceted by feature type.

    Similar to technical_demographic_feature_shap_rank.pdf from R script.
    Shows all modeled features, not just top N.
    Only shows demographic (left panel) and technical (right panel) features.
    """
    # Calculate mean rank per feature to order them
    mean_ranks = df.groupby('feature_name')['rank'].mean().sort_values()

    # Use ALL features
    plot_df = df.copy()

    # Order features by mean rank
    plot_df['feature_name'] = pd.Categorical(
        plot_df['feature_name'],
        categories=mean_ranks.index.tolist(),
        ordered=True
    )

    # Separate by feature type - ONLY demographic and technical
    # IMPORTANT: Convert feature_name back to string to avoid categorical issues
    # where subsets inherit ALL categories from the parent dataframe
    demo_df = plot_df[plot_df['feature_type'] == 'demographic'].copy()
    demo_df['feature_name'] = demo_df['feature_name'].astype(str)

    tech_df = plot_df[plot_df['feature_type'] == 'technical'].copy()
    tech_df['feature_name'] = tech_df['feature_name'].astype(str)

    # Determine subplot layout - only demographic and technical panels
    has_demo = len(demo_df) > 0
    has_tech = len(tech_df) > 0
    n_panels = sum([has_demo, has_tech])

    if n_panels == 0:
        print("Warning: No demographic or technical features found for boxplot")
        return

    # Calculate figure width based on number of features
    n_demo = demo_df['feature_name'].nunique() if has_demo else 0
    n_tech = tech_df['feature_name'].nunique() if has_tech else 0

    # Width per feature (minimum 0.5 inch)
    width_per_feature = 0.6
    total_width = max(12, (n_demo + n_tech) * width_per_feature + 2)

    # Create figure with two subplots: demographic (left), technical (right)
    width_ratios = []
    if has_demo:
        width_ratios.append(max(n_demo, 1))
    if has_tech:
        width_ratios.append(max(n_tech, 1))

    fig, axes = plt.subplots(1, n_panels, figsize=(total_width, 10),
                              gridspec_kw={'width_ratios': width_ratios})

    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    # Demographic features - LEFT panel
    if has_demo:
        ax = axes[ax_idx]
        demo_order = demo_df.groupby('feature_name')['rank'].mean().sort_values().index
        sns.boxplot(data=demo_df, x='feature_name', y='rank', ax=ax,
                   order=demo_order, color='#3498db', width=0.6)
        sns.stripplot(data=demo_df, x='feature_name', y='rank', ax=ax,
                     order=demo_order, color='black', alpha=0.5, size=4)
        ax.invert_yaxis()
        ax.set_title('Demographic Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Feature Rank')
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha('right')
        ax_idx += 1

    # Technical features - RIGHT panel
    if has_tech:
        ax = axes[ax_idx]
        tech_order = tech_df.groupby('feature_name')['rank'].mean().sort_values().index
        sns.boxplot(data=tech_df, x='feature_name', y='rank', ax=ax,
                   order=tech_order, color='#e74c3c', width=0.6)
        sns.stripplot(data=tech_df, x='feature_name', y='rank', ax=ax,
                     order=tech_order, color='black', alpha=0.5, size=4)
        ax.invert_yaxis()
        ax.set_title('Technical Features', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha('right')

    fig.suptitle('SHAP Feature Rank Distribution Across Cancer Types\n(All Modeled Features)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save PDF
    output_file = output_dir / 'technical_demographic_feature_shap_rank_all.pdf'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")

    # Also save PNG
    png_file = output_dir / 'technical_demographic_feature_shap_rank_all.png'
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {png_file}")

    plt.close()


def plot_feature_rank_heatmap(df, output_dir, top_n=10):
    """
    Create heatmap showing feature ranks across cancer types.
    """
    # Get top features by mean rank
    mean_ranks = df.groupby('feature_name')['rank'].mean().sort_values()
    top_features = mean_ranks.head(top_n).index.tolist()

    # Create pivot table
    plot_df = df[df['feature_name'].isin(top_features)].copy()
    pivot = plot_df.pivot_table(index='feature_name', columns='cancer_type', values='rank')

    # Order by mean rank
    pivot = pivot.reindex(mean_ranks[top_features].index)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'Feature Rank'},
                vmin=1, vmax=pivot.max().max())

    ax.set_title('SHAP Feature Rank by Cancer Type\n(Lower = More Important)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)

    plt.tight_layout()

    output_file = output_dir / 'shap_feature_rank_heatmap.pdf'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")


def plot_top5_rank_summary(df, output_dir):
    """
    Create summary showing which features appear in top 5 most frequently.
    """
    # Count how often each feature appears in top 5
    top5_counts = df[df['rank'] <= 5].groupby(['feature_name', 'feature_type']).size().reset_index(name='count')
    top5_counts = top5_counts.sort_values('count', ascending=True)

    # Filter to features appearing in top 5 at least once
    top5_counts = top5_counts[top5_counts['count'] > 0]

    if len(top5_counts) == 0:
        print("Warning: No features in top 5")
        return

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(top5_counts) * 0.4)))

    colors = top5_counts['feature_type'].map({
        'demographic': '#3498db',
        'technical': '#e74c3c',
        'other': '#95a5a6'
    })

    bars = ax.barh(top5_counts['feature_name'], top5_counts['count'], color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, count in zip(bars, top5_counts['count']):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
               f'{count}', va='center', ha='left', fontsize=9)

    ax.set_xlabel('Number of Cancer Types (in Top 5)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Features Appearing in Top 5 SHAP Rankings', fontsize=14, fontweight='bold')
    ax.set_xlim(0, top5_counts['count'].max() + 2)

    # Add legend (only demographic and technical)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Demographic'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Technical'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()

    output_file = output_dir / 'shap_top5_feature_frequency.pdf'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}")

    # Also save as CSV
    csv_file = output_dir / 'shap_top5_features.csv'
    top5_counts.to_csv(csv_file, index=False)
    print(f"Saved: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot SHAP feature ranks across cancer types',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect latest SHAP output
    python plot_shap_feature_ranks.py

    # Specify SHAP directory
    python plot_shap_feature_ranks.py --shap-dir ../../../analysis_output/shap_primary_values/run_20260119_085841
        """
    )

    parser.add_argument('--shap-dir', type=str, default=None,
                       help='Directory containing SHAP primary values output')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots')

    args = parser.parse_args()

    # Find SHAP directory
    if args.shap_dir:
        shap_dir = Path(args.shap_dir)
    else:
        shap_dir = get_default_shap_dir()
        if shap_dir is None:
            print("ERROR: No SHAP output found. Run compute_shap_primary_values.py first.")
            return 1

    print(f"Using SHAP directory: {shap_dir}")

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPO_ROOT / "analysis_output" / "shap_feature_rank_plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    print("\nLoading grouped SHAP data...")
    df = load_grouped_shap_data(shap_dir)

    # Generate plots
    print("\nGenerating plots...")

    print("\n1. Top 5 features rank plot (line plot across cancer types)...")
    plot_top5_features_rank(df, output_dir)

    print("\n2. Feature rank boxplot for ALL features (demographic vs technical)...")
    plot_feature_rank_boxplot_all(df, output_dir)

    print("\n3. Feature rank heatmap (top 10 features)...")
    plot_feature_rank_heatmap(df, output_dir)

    print("\n4. Top 5 feature frequency bar chart...")
    plot_top5_rank_summary(df, output_dir)

    # Save combined data
    combined_file = output_dir / 'combined_grouped_shap_ranks.csv'
    df.to_csv(combined_file, index=False)
    print(f"\nSaved combined data: {combined_file}")

    print(f"\n{'='*60}")
    print("SHAP Feature Rank Visualization Complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
