#!/usr/bin/env python3
"""
Compare RF SHAP Values vs LR Standardized Coefficients

One-off analysis script that correlates SHAP feature importance (from RF models)
with standardized logistic regression coefficients across cancer types.

Comparison at two levels:
1. Per-feature level: mean |SHAP| vs |standardized_log_odds| for each transformed feature
2. Grouped level: grouped SHAP importance vs grouped LR standardized importance per parent variable

Outputs:
    - shap_vs_lr_per_feature_scatter.pdf: Scatter plot per cancer type (feature level)
    - shap_vs_lr_grouped_scatter.pdf: Scatter plot per cancer type (grouped level)
    - shap_vs_lr_rank_comparison.pdf: Rank comparison bar chart (grouped level)
    - shap_vs_lr_correlations.csv: Spearman and Pearson correlations per cancer type
    - shap_vs_lr_combined.csv: Merged data for further analysis

Usage:
    python compare_shap_vs_lr_coefficients.py \
        --shap-dir /path/to/tier6_shap_analysis/primary_values/run_*/ \
        --lr-dir /path/to/lr_coefficients_full/run_*/ \
        --output-dir /path/to/output \
        --lr-metric sum_abs          # (default) sum of |standardized betas| per group
        --lr-metric max_abs          # max |standardized beta| per group
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_shap_grouped(shap_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load grouped SHAP importance for each cancer type."""
    results = {}
    for cancer_dir in sorted(shap_dir.iterdir()):
        if not cancer_dir.is_dir():
            continue
        cancer_type = cancer_dir.name
        grouped_file = cancer_dir / f'{cancer_type}_grouped_feature_importance.csv'
        if grouped_file.exists():
            df = pd.read_csv(grouped_file)
            results[cancer_type] = df
    return results


def load_shap_per_feature(shap_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load per-feature SHAP importance for each cancer type."""
    results = {}
    for cancer_dir in sorted(shap_dir.iterdir()):
        if not cancer_dir.is_dir():
            continue
        cancer_type = cancer_dir.name
        feat_file = cancer_dir / f'{cancer_type}_feature_importance.csv'
        if feat_file.exists():
            df = pd.read_csv(feat_file)
            results[cancer_type] = df
    return results


def load_lr_grouped(lr_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load grouped LR importance for each cancer type."""
    results = {}
    for cancer_dir in sorted(lr_dir.iterdir()):
        if not cancer_dir.is_dir():
            continue
        cancer_type = cancer_dir.name
        grouped_file = cancer_dir / 'lr_grouped_importance.csv'
        if grouped_file.exists():
            df = pd.read_csv(grouped_file)
            results[cancer_type] = df
    return results


def load_lr_per_feature(lr_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load per-feature LR coefficients for each cancer type."""
    results = {}
    for cancer_dir in sorted(lr_dir.iterdir()):
        if not cancer_dir.is_dir():
            continue
        cancer_type = cancer_dir.name
        coef_file = cancer_dir / 'lr_coefficients.csv'
        if coef_file.exists():
            df = pd.read_csv(coef_file)
            results[cancer_type] = df
    return results


# Mapping from CLI metric name to LR grouped CSV column name
LR_METRIC_COLUMNS = {
    'sum_abs': 'sum_abs_std_log_odds',
    'max_abs': 'max_abs_std_log_odds',
}

LR_METRIC_LABELS = {
    'sum_abs': 'LR sum |std log-odds|',
    'max_abs': 'LR max |std log-odds|',
}


def merge_grouped(shap_grouped: Dict, lr_grouped: Dict,
                  lr_metric: str = 'sum_abs') -> pd.DataFrame:
    """Merge grouped SHAP and LR data on parent feature name.

    Args:
        lr_metric: Which LR grouped column to use.
            'sum_abs' (default) - sum of |standardized betas| per group
            'max_abs' - max |standardized beta| per group
    """
    lr_col = LR_METRIC_COLUMNS[lr_metric]
    all_merged = []

    for cancer_type in sorted(set(shap_grouped.keys()) & set(lr_grouped.keys())):
        shap_df = shap_grouped[cancer_type][['original_feature', 'mean_abs_shap', 'rank']].copy()
        shap_df = shap_df.rename(columns={
            'original_feature': 'feature_group',
            'mean_abs_shap': 'shap_mean_abs',
            'rank': 'shap_rank'
        })

        lr_df = lr_grouped[cancer_type][['feature_group', lr_col, 'rank']].copy()
        lr_df = lr_df.rename(columns={
            lr_col: 'lr_std_importance',
            'rank': 'lr_rank'
        })

        merged = shap_df.merge(lr_df, on='feature_group', how='inner')
        merged['cancer_type'] = cancer_type
        all_merged.append(merged)

    if all_merged:
        return pd.concat(all_merged, ignore_index=True)
    return pd.DataFrame()


def merge_per_feature(shap_feat: Dict, lr_feat: Dict) -> pd.DataFrame:
    """Merge per-feature SHAP and LR data on raw feature name."""
    all_merged = []

    for cancer_type in sorted(set(shap_feat.keys()) & set(lr_feat.keys())):
        shap_df = shap_feat[cancer_type][['feature_name_raw', 'mean_abs_shap', 'rank']].copy()
        shap_df = shap_df.rename(columns={
            'mean_abs_shap': 'shap_mean_abs',
            'rank': 'shap_rank'
        })

        lr_df = lr_feat[cancer_type][['feature_name_raw', 'feature_name', 'feature_group',
                                       'standardized_log_odds', 'log_odds', 'rank']].copy()
        lr_df = lr_df.rename(columns={'rank': 'lr_rank'})
        lr_df['lr_abs_std'] = lr_df['standardized_log_odds'].abs()

        merged = shap_df.merge(lr_df, on='feature_name_raw', how='inner')
        merged['cancer_type'] = cancer_type
        all_merged.append(merged)

    if all_merged:
        return pd.concat(all_merged, ignore_index=True)
    return pd.DataFrame()


def compute_correlations(merged_grouped: pd.DataFrame, merged_features: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman and Pearson correlations per cancer type at both levels."""
    rows = []

    for cancer_type in sorted(merged_grouped['cancer_type'].unique()):
        # Grouped level
        g = merged_grouped[merged_grouped['cancer_type'] == cancer_type]
        if len(g) >= 3:
            sp_r, sp_p = stats.spearmanr(g['shap_mean_abs'], g['lr_std_importance'])
            pe_r, pe_p = stats.pearsonr(g['shap_mean_abs'], g['lr_std_importance'])
            rank_sp_r, rank_sp_p = stats.spearmanr(g['shap_rank'], g['lr_rank'])
        else:
            sp_r = sp_p = pe_r = pe_p = rank_sp_r = rank_sp_p = np.nan

        # Per-feature level
        f = merged_features[merged_features['cancer_type'] == cancer_type]
        if len(f) >= 3:
            feat_sp_r, feat_sp_p = stats.spearmanr(f['shap_mean_abs'], f['lr_abs_std'])
            feat_pe_r, feat_pe_p = stats.pearsonr(f['shap_mean_abs'], f['lr_abs_std'])
        else:
            feat_sp_r = feat_sp_p = feat_pe_r = feat_pe_p = np.nan

        rows.append({
            'cancer_type': cancer_type,
            'n_groups': len(g),
            'n_features': len(f),
            'grouped_spearman_r': sp_r,
            'grouped_spearman_p': sp_p,
            'grouped_pearson_r': pe_r,
            'grouped_pearson_p': pe_p,
            'grouped_rank_spearman_r': rank_sp_r,
            'grouped_rank_spearman_p': rank_sp_p,
            'feature_spearman_r': feat_sp_r,
            'feature_spearman_p': feat_sp_p,
            'feature_pearson_r': feat_pe_r,
            'feature_pearson_p': feat_pe_p,
        })

    return pd.DataFrame(rows)


def plot_grouped_scatter(merged_grouped: pd.DataFrame, output_dir: Path,
                         lr_metric: str = 'sum_abs'):
    """Scatter plot of grouped SHAP vs LR standardized importance per cancer type."""
    lr_label = LR_METRIC_LABELS[lr_metric]
    cancer_types = sorted(merged_grouped['cancer_type'].unique())
    n = len(cancer_types)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    for idx, cancer_type in enumerate(cancer_types):
        ax = axes[idx // ncols][idx % ncols]
        g = merged_grouped[merged_grouped['cancer_type'] == cancer_type]

        ax.scatter(g['lr_std_importance'], g['shap_mean_abs'], s=60, alpha=0.8, edgecolors='black', linewidth=0.5)

        # Label each point
        for _, row in g.iterrows():
            label = row['feature_group']
            if len(label) > 15:
                label = label[:13] + '..'
            ax.annotate(label, (row['lr_std_importance'], row['shap_mean_abs']),
                        fontsize=6, alpha=0.7, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')

        # Correlation
        if len(g) >= 3:
            r, p = stats.spearmanr(g['shap_mean_abs'], g['lr_std_importance'])
            ax.set_title(f'{cancer_type}\nSpearman r={r:.2f} (p={p:.3f})', fontsize=10)
        else:
            ax.set_title(cancer_type, fontsize=10)

        ax.set_xlabel(lr_label, fontsize=9)
        ax.set_ylabel('RF mean |SHAP|', fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f'RF SHAP vs LR Standardized Coefficients (Grouped, {lr_metric})',
                  fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_vs_lr_grouped_scatter.pdf', dpi=200, bbox_inches='tight')
    plt.close()


def plot_per_feature_scatter(merged_features: pd.DataFrame, output_dir: Path):
    """Scatter plot of per-feature SHAP vs LR standardized importance per cancer type."""
    cancer_types = sorted(merged_features['cancer_type'].unique())
    n = len(cancer_types)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)

    for idx, cancer_type in enumerate(cancer_types):
        ax = axes[idx // ncols][idx % ncols]
        f = merged_features[merged_features['cancer_type'] == cancer_type]

        ax.scatter(f['lr_abs_std'], f['shap_mean_abs'], s=20, alpha=0.6, edgecolors='black', linewidth=0.3)

        # Label top 5 features
        top5 = f.nlargest(5, 'shap_mean_abs')
        for _, row in top5.iterrows():
            label = row['feature_name']
            if len(label) > 20:
                label = label[:18] + '..'
            ax.annotate(label, (row['lr_abs_std'], row['shap_mean_abs']),
                        fontsize=5.5, alpha=0.7, ha='left', va='bottom',
                        xytext=(2, 2), textcoords='offset points')

        if len(f) >= 3:
            r, p = stats.spearmanr(f['shap_mean_abs'], f['lr_abs_std'])
            ax.set_title(f'{cancer_type} (n={len(f)})\nSpearman r={r:.2f} (p={p:.3f})', fontsize=9)
        else:
            ax.set_title(cancer_type, fontsize=9)

        ax.set_xlabel('LR |std log-odds|', fontsize=8)
        ax.set_ylabel('RF mean |SHAP|', fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('RF SHAP vs LR Standardized Coefficients (Per Transformed Feature)',
                  fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_vs_lr_per_feature_scatter.pdf', dpi=200, bbox_inches='tight')
    plt.close()


def plot_rank_comparison(merged_grouped: pd.DataFrame, output_dir: Path):
    """Side-by-side rank comparison bar chart across cancer types."""
    cancer_types = sorted(merged_grouped['cancer_type'].unique())
    n = len(cancer_types)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, cancer_type in enumerate(cancer_types):
        ax = axes[idx // ncols][idx % ncols]
        g = merged_grouped[merged_grouped['cancer_type'] == cancer_type].sort_values('shap_rank')

        y = np.arange(len(g))
        bar_height = 0.35

        ax.barh(y - bar_height / 2, g['shap_rank'], bar_height, label='SHAP rank', color='#3498db', alpha=0.8)
        ax.barh(y + bar_height / 2, g['lr_rank'], bar_height, label='LR rank', color='#e74c3c', alpha=0.8)

        labels = [name[:18] for name in g['feature_group']]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Rank', fontsize=8)
        ax.set_title(cancer_type, fontsize=10, fontweight='bold')
        ax.invert_xaxis()
        if idx == 0:
            ax.legend(fontsize=7, loc='lower left')

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('Feature Rank Comparison: RF SHAP vs LR Standardized',
                  fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_vs_lr_rank_comparison.pdf', dpi=200, bbox_inches='tight')
    plt.close()


def plot_correlation_summary(corr_df: pd.DataFrame, output_dir: Path,
                             lr_metric: str = 'sum_abs'):
    """Bar chart of correlation coefficients across cancer types."""
    lr_label = LR_METRIC_LABELS[lr_metric]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Grouped level
    ax = axes[0]
    corr_sorted = corr_df.sort_values('grouped_spearman_r', ascending=True)
    colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in corr_sorted['grouped_spearman_p']]
    ax.barh(range(len(corr_sorted)), corr_sorted['grouped_spearman_r'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(corr_sorted)))
    ax.set_yticklabels(corr_sorted['cancer_type'], fontsize=8)
    ax.set_xlabel('Spearman r', fontsize=11)
    ax.set_title(f'Grouped Level ({lr_label})', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    mean_r = corr_sorted['grouped_spearman_r'].mean()
    ax.axvline(x=mean_r, color='navy', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.text(mean_r + 0.02, len(corr_sorted) - 1, f'mean={mean_r:.2f}', fontsize=9, color='navy')

    # Per-feature level
    ax = axes[1]
    corr_sorted = corr_df.sort_values('feature_spearman_r', ascending=True)
    colors = ['#2ecc71' if p < 0.05 else '#95a5a6' for p in corr_sorted['feature_spearman_p']]
    ax.barh(range(len(corr_sorted)), corr_sorted['feature_spearman_r'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(corr_sorted)))
    ax.set_yticklabels(corr_sorted['cancer_type'], fontsize=8)
    ax.set_xlabel('Spearman r', fontsize=11)
    ax.set_title('Per-Feature Level (Transformed Features)', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    mean_r = corr_sorted['feature_spearman_r'].mean()
    ax.axvline(x=mean_r, color='navy', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.text(mean_r + 0.02, len(corr_sorted) - 1, f'mean={mean_r:.2f}', fontsize=9, color='navy')

    fig.suptitle('RF SHAP vs LR Standardized Coefficients: Correlation by Cancer Type\n(green = p < 0.05)',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_vs_lr_correlation_summary.pdf', dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compare RF SHAP values vs LR standardized coefficients'
    )
    parser.add_argument(
        '--shap-dir', type=str, required=True,
        help='Path to SHAP primary values directory (contains cancer type subdirectories)'
    )
    parser.add_argument(
        '--lr-dir', type=str, required=True,
        help='Path to LR coefficients directory (contains cancer type subdirectories)'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--lr-metric', type=str, default='sum_abs',
        choices=['sum_abs', 'max_abs'],
        help='LR grouped aggregation metric: '
             'sum_abs (default) = sum of |standardized betas| per group, '
             'max_abs = max |standardized beta| per group'
    )

    args = parser.parse_args()

    shap_dir = Path(args.shap_dir)
    lr_dir = Path(args.lr_dir)
    output_dir = Path(args.output_dir)
    lr_metric = args.lr_metric
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"LR grouped metric: {lr_metric} ({LR_METRIC_LABELS[lr_metric]})")
    print("Loading SHAP data...")
    shap_grouped = load_shap_grouped(shap_dir)
    shap_feat = load_shap_per_feature(shap_dir)
    print(f"  Loaded {len(shap_grouped)} cancer types")

    print("Loading LR data...")
    lr_grouped = load_lr_grouped(lr_dir)
    lr_feat = load_lr_per_feature(lr_dir)
    print(f"  Loaded {len(lr_grouped)} cancer types")

    common = sorted(set(shap_grouped.keys()) & set(lr_grouped.keys()))
    print(f"  Common cancer types: {len(common)}")

    if not common:
        print("ERROR: No common cancer types between SHAP and LR data")
        sys.exit(1)

    # Merge
    print("Merging data...")
    merged_grouped = merge_grouped(shap_grouped, lr_grouped, lr_metric=lr_metric)
    merged_features = merge_per_feature(shap_feat, lr_feat)
    print(f"  Grouped: {len(merged_grouped)} rows")
    print(f"  Per-feature: {len(merged_features)} rows")

    # Compute correlations
    print("Computing correlations...")
    corr_df = compute_correlations(merged_grouped, merged_features)

    # Print summary
    print()
    print("=" * 80)
    print("CORRELATION SUMMARY")
    print("=" * 80)
    print(f"{'Cancer':<10} {'Grp Spearman':>14} {'Grp p':>10} {'Feat Spearman':>14} {'Feat p':>10}")
    print("-" * 80)
    for _, r in corr_df.sort_values('grouped_spearman_r', ascending=False).iterrows():
        g_sig = '*' if r['grouped_spearman_p'] < 0.05 else ''
        f_sig = '*' if r['feature_spearman_p'] < 0.05 else ''
        print(f"{r['cancer_type']:<10} {r['grouped_spearman_r']:+14.3f}{g_sig:1s} {r['grouped_spearman_p']:10.4f} "
              f"{r['feature_spearman_r']:+14.3f}{f_sig:1s} {r['feature_spearman_p']:10.4f}")
    print("-" * 80)
    print(f"{'MEAN':<10} {corr_df['grouped_spearman_r'].mean():+14.3f}            "
          f"{corr_df['feature_spearman_r'].mean():+14.3f}")
    print(f"{'MEDIAN':<10} {corr_df['grouped_spearman_r'].median():+14.3f}            "
          f"{corr_df['feature_spearman_r'].median():+14.3f}")

    # Save outputs
    print()
    print("Generating plots...")
    plot_grouped_scatter(merged_grouped, output_dir, lr_metric=lr_metric)
    print(f"  Saved: shap_vs_lr_grouped_scatter.pdf")

    plot_per_feature_scatter(merged_features, output_dir)
    print(f"  Saved: shap_vs_lr_per_feature_scatter.pdf")

    plot_rank_comparison(merged_grouped, output_dir)
    print(f"  Saved: shap_vs_lr_rank_comparison.pdf")

    plot_correlation_summary(corr_df, output_dir, lr_metric=lr_metric)
    print(f"  Saved: shap_vs_lr_correlation_summary.pdf")

    # Save CSVs
    corr_df.to_csv(output_dir / 'shap_vs_lr_correlations.csv', index=False)
    print(f"  Saved: shap_vs_lr_correlations.csv")

    merged_grouped.to_csv(output_dir / 'shap_vs_lr_grouped_merged.csv', index=False)
    print(f"  Saved: shap_vs_lr_grouped_merged.csv")

    merged_features.to_csv(output_dir / 'shap_vs_lr_per_feature_merged.csv', index=False)
    print(f"  Saved: shap_vs_lr_per_feature_merged.csv")

    print()
    print(f"All outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
