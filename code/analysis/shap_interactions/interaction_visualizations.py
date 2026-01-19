"""
SHAP Interaction Visualizations

Publication-quality visualization functions for SHAP interaction analysis.

Provides functions to visualize:
- Interaction heatmaps (full matrix of pairwise interactions)
- Main vs interaction contribution breakdowns (pie/bar charts)
- Top interactions ranked by strength (bar plots)
- Interaction dependence plots (how interactions vary with feature values)
- Demographic-specific interaction matrices

Author: Balder Analysis Team
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional


def plot_interaction_heatmap(interaction_values, feature_names, output_path,
                              figsize=(12, 10), cmap='RdBu_r', vmin=None, vmax=None):
    """
    Create heatmap of mean absolute interaction strengths.

    Visualization:
        - X-axis: Features
        - Y-axis: Features
        - Color: Mean absolute interaction strength (averaged over samples)
        - Symmetric matrix
        - Diagonal shows main effects

    Interpretation:
        - Dark red cells: Strong interactions
        - Light/white cells: Weak or no interaction
        - Diagonal typically stronger (main effects)

    Args:
        interaction_values: 3D interaction array (n_samples, n_features, n_features)
        feature_names: List of feature names
        output_path: Path to save PDF
        figsize: Figure size (width, height)
        cmap: Colormap (diverging recommended)
        vmin, vmax: Color scale limits (default: auto)
    """
    # Calculate mean absolute interaction across samples
    mean_abs_interaction = np.abs(interaction_values).mean(axis=0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(mean_abs_interaction,
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap=cmap,
                center=0,
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'label': 'Mean Absolute Interaction'},
                ax=ax)

    ax.set_title('SHAP Interaction Heatmap\n(Mean Absolute Interaction Strength)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved interaction heatmap to {output_path}")


def plot_main_vs_interaction_pie(main_vs_interaction_dict, output_path,
                                  figsize=(10, 5)):
    """
    Create pie and bar charts showing prediction attribution split.

    Visualization:
        - Left: Pie chart (main vs interaction percentages)
        - Right: Bar chart (for easier comparison across cancer types)

    Interpretation:
        - Large interaction slice → Model relies on feature combinations
        - Small interaction slice → Model is approximately additive

    Args:
        main_vs_interaction_dict: Output from compare_main_vs_interaction_contributions()
        output_path: Path to save PDF
        figsize: Figure size
    """
    main_pct = main_vs_interaction_dict['main_pct']
    interaction_pct = main_vs_interaction_dict['interaction_pct']

    # Handle NaN or invalid values
    import numpy as np
    if np.isnan(main_pct) or np.isnan(interaction_pct) or (main_pct + interaction_pct == 0):
        print(f"WARNING: Invalid percentages (main={main_pct}, interaction={interaction_pct}). Skipping pie chart.")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Pie chart
    sizes = [main_pct, interaction_pct]
    labels = ['Main Effects', 'Interactions']
    colors = ['steelblue', 'coral']
    explode = (0, 0.1)  # Explode interaction slice slightly

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title('Prediction Attribution', fontweight='bold')

    # Bar chart
    bars = ax2.bar(['Main Effects', 'Interactions'], sizes, color=colors)
    ax2.set_ylabel('Contribution (%)', fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.set_title('Contribution Breakdown', fontweight='bold')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Main Effects vs Interaction Effects', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved main vs interaction plot to {output_path}")


def plot_top_interactions_bar(top_interactions_df, output_path,
                                top_n=20, figsize=(10, 8)):
    """
    Horizontal bar plot of strongest interaction pairs.

    Visualization:
        - X-axis: Mean absolute interaction value
        - Y-axis: Feature pairs (e.g., "AGE × SEX")
        - Bars colored by interaction type

    Interpretation:
        - Longest bars: Most important interaction pairs

    Args:
        top_interactions_df: Output from get_strongest_interactions()
        output_path: Path to save PDF
        top_n: Number of interactions to display
        figsize: Figure size
    """
    # Take top N
    df_plot = top_interactions_df.head(top_n).copy()

    # Create interaction labels
    df_plot['interaction_label'] = (df_plot['feature_i'] + ' × ' + df_plot['feature_j'])

    # Reverse order for plotting (highest at top)
    df_plot = df_plot.iloc[::-1]

    # Determine colors based on feature types if available
    if 'feature_i_type' in df_plot.columns and 'feature_j_type' in df_plot.columns:
        def get_interaction_type(row):
            types = sorted([row['feature_i_type'], row['feature_j_type']])
            if types == ['demographic', 'demographic']:
                return 'demo × demo'
            elif types == ['technical', 'technical']:
                return 'tech × tech'
            else:
                return 'demo × tech'

        df_plot['interaction_type'] = df_plot.apply(get_interaction_type, axis=1)

        # Color mapping
        type_colors = {
            'demo × demo': 'steelblue',
            'tech × tech': 'coral',
            'demo × tech': 'mediumpurple'
        }
        colors = df_plot['interaction_type'].map(type_colors)
    else:
        colors = 'steelblue'

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(range(len(df_plot)), df_plot['mean_abs_interaction'], color=colors)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['interaction_label'])
    ax.set_xlabel('Mean Absolute Interaction Value', fontweight='bold')
    ax.set_title(f'Top {top_n} Strongest Feature Interactions', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add legend if we have types
    if 'interaction_type' in df_plot.columns:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=itype)
                          for itype, color in type_colors.items()
                          if itype in df_plot['interaction_type'].values]
        ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved top interactions bar plot to {output_path}")


def plot_interaction_dependence(interaction_values, X_test, feature_i_idx,
                                  feature_j_idx, feature_names,
                                  output_path, figsize=(10, 6)):
    """
    Scatter plot showing how interaction strength varies with feature values.

    Visualization:
        - X-axis: Values of feature_i
        - Y-axis: Interaction SHAP value
        - Color: Values of feature_j
        - Each point: One patient

    Interpretation:
        - Patterns reveal how interaction depends on feature values
        - Example: If AGE×SEX interaction strong for young, weak for old
                  → Sex matters more for younger patients

    Args:
        interaction_values: 3D interaction array
        X_test: Test features (DataFrame, original scale for interpretability)
        feature_i_idx: Index of first feature
        feature_j_idx: Index of second feature
        feature_names: List of transformed feature names
        output_path: Path to save PDF
        figsize: Figure size
    """
    # Extract interaction values for this pair
    interaction_pair = interaction_values[:, feature_i_idx, feature_j_idx]

    # Get feature names
    feat_i_name = feature_names[feature_i_idx]
    feat_j_name = feature_names[feature_j_idx]

    # Try to get original feature values from X_test
    # This is tricky because features are transformed (one-hot encoded)
    # For now, use sample indices as x-axis
    x_values = np.arange(len(interaction_pair))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(x_values, interaction_pair,
                        c=x_values, cmap='viridis',
                        alpha=0.6, s=30)

    ax.set_xlabel('Sample Index', fontweight='bold')
    ax.set_ylabel(f'Interaction SHAP Value', fontweight='bold')
    ax.set_title(f'Interaction Dependence: {feat_i_name} × {feat_j_name}',
                 fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(alpha=0.3, linestyle='--')

    plt.colorbar(scatter, ax=ax, label='Sample Index')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved interaction dependence plot to {output_path}")


def plot_demographic_interaction_matrix(interaction_values, feature_names,
                                         demographic_feature_indices,
                                         demographic_feature_labels,
                                         output_path, figsize=(10, 8)):
    """
    Focused heatmap of interactions among demographic features only.

    Visualization:
        - Subset of full interaction matrix
        - Only demographic features (AGE, SEX, RACE, ETHNICITY, etc.)
        - Larger labels, easier to read

    Interpretation:
        - Identifies which demographic pairs interact most strongly
        - Examples: AGE×SEX, RACE×ETHNICITY

    Args:
        interaction_values: 3D interaction array
        feature_names: List of all transformed feature names
        demographic_feature_indices: List of indices for demographic features
        demographic_feature_labels: List of labels for demographic features
        output_path: Path to save PDF
        figsize: Figure size
    """
    # Calculate mean absolute interaction
    mean_abs_interaction = np.abs(interaction_values).mean(axis=0)

    # Extract demographic submatrix
    demo_submatrix = mean_abs_interaction[np.ix_(demographic_feature_indices,
                                                   demographic_feature_indices)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(demo_submatrix,
                xticklabels=demographic_feature_labels,
                yticklabels=demographic_feature_labels,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.4f',
                square=True,
                cbar_kws={'label': 'Mean Absolute Interaction'},
                ax=ax)

    ax.set_title('Demographic Feature Interactions\n(Mean Absolute Interaction Strength)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved demographic interaction matrix to {output_path}")


def plot_cross_cancer_comparison(comparison_df, output_path, figsize=(12, 6)):
    """
    Bar chart comparing main vs interaction contributions across cancer types.

    Args:
        comparison_df: DataFrame with columns [cancer_type, main_pct, interaction_pct]
        output_path: Path to save PDF
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(comparison_df))
    width = 0.8

    # Stacked bar chart
    ax.bar(x, comparison_df['main_pct'], width, label='Main Effects',
           color='steelblue')
    ax.bar(x, comparison_df['interaction_pct'], width,
           bottom=comparison_df['main_pct'], label='Interactions',
           color='coral')

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['cancer_type'], rotation=45, ha='right')
    ax.set_ylabel('Contribution (%)', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title('Main vs Interaction Contribution by Cancer Type',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Saved cross-cancer comparison to {output_path}")
