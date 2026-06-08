#!/usr/bin/env python3
"""
Hit List Composite Figures for Post-Hoc Statistical Analysis

Creates publication-ready 3-panel composite figures showing effect size,
raw actionability difference, and FDR significance for each demographic feature
(Age, Sex, Race, Ethnicity) at both cancer-type and gene-by-cancer-type levels.

Layout per feature:
  Left (wide):  Forest plot of effect size with 95% CI
  Center:       Lollipop of raw actionability difference
  Right:        Dot plot of -log10(p_fdr) with significance threshold

Usage:
    # Cancer-level figures from latest run
    python create_hit_list_composite_figures.py --level cancer

    # Gene-level figures from specific run (post-SHAP-gate-audit; pre-fix
    # `run_20260212_190358/tier7_posthoc_statistics/` is now
    # `.DEPRECATED_20260507/` — use the regenerated subdir):
    python create_hit_list_composite_figures.py --level gene \
        --input-dir ../../../runs/run_20260505_164207/tier7_posthoc_statistics_REGENERATED_20260507

    # Single feature, PDF only
    python create_hit_list_composite_figures.py --level cancer --features age --format pdf
"""

import argparse
import ast
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools'))

# Repository root for default paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Display label (forward-compatible with race/ancestry renaming)
RACE_DISPLAY_LABEL = "Race/Ancestry"

# Colors
COLOR_SIG = '#C0392B'      # Red for significant
COLOR_NONSIG = '#95A5A6'   # Gray for non-significant
COLOR_POS = '#2980B9'      # Blue for positive direction
COLOR_NEG = '#E67E22'      # Orange for negative direction
THRESHOLD_COLOR = '#7F8C8D'

# Max rows per page for gene-level figures
MAX_ROWS_PER_PAGE = 40


def get_default_input_dir():
    """Find latest post-hoc results directory."""
    runs_dir = REPO_ROOT / "runs"
    if runs_dir.exists():
        runs = sorted(runs_dir.glob("run_*"), reverse=True)
        for run_dir in runs:
            posthoc_dir = run_dir / "tier7_posthoc_statistics"
            if posthoc_dir.exists():
                return posthoc_dir

    legacy_dir = REPO_ROOT / "analysis_output" / "posthoc_statistics"
    if legacy_dir.exists():
        return legacy_dir

    return None


def calculate_cohens_d_ci(d, n1, n2, confidence=0.95):
    """Calculate confidence interval for Cohen's d (Hedges & Olkin 1985)."""
    df = n1 + n2 - 2
    if df <= 0:
        return np.nan, np.nan
    var_d = ((n1 + n2) / (n1 * n2)) + (d**2 / (2 * df))
    se_d = np.sqrt(var_d)
    z_critical = 1.96 if confidence == 0.95 else 2.576
    return d - z_critical * se_d, d + z_critical * se_d


def compute_max_rate_spread(contingency_str):
    """Parse contingency_table string and compute max actionability rate spread.

    contingency_table is a nested dict: {"0": {group: count}, "1": {group: count}}
    where "0" = non-actionable, "1" = actionable.
    Returns max(rate) - min(rate) across groups with sufficient data.
    """
    try:
        if pd.isna(contingency_str):
            return np.nan
        # Try JSON first, then ast.literal_eval
        try:
            table = json.loads(contingency_str)
        except (json.JSONDecodeError, TypeError):
            table = ast.literal_eval(contingency_str)

        non_act = table.get('0', table.get(0, {}))
        act = table.get('1', table.get(1, {}))

        rates = []
        for group in set(list(non_act.keys()) + list(act.keys())):
            n0 = non_act.get(group, 0)
            n1 = act.get(group, 0)
            total = n0 + n1
            if total >= 10 and group not in ('Unknown', 'Other'):
                rates.append(n1 / total)

        if len(rates) >= 2:
            return max(rates) - min(rates)
        return np.nan
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_cancer_level_data(input_dir):
    """Load cancer-level posthoc CSVs into a dict keyed by feature name."""
    input_dir = Path(input_dir)
    data = {}

    # Age
    age_path = input_dir / 'age_actionability' / 'age_actionability_results.csv'
    if age_path.exists():
        data['age'] = pd.read_csv(age_path)
        print(f"  Loaded age data: {len(data['age'])} cancer types")
    else:
        warnings.warn(f"Age CSV not found: {age_path}")

    # Sex
    sex_path = input_dir / 'shap_associations' / 'sex_associations.csv'
    if sex_path.exists():
        data['sex'] = pd.read_csv(sex_path)
        print(f"  Loaded sex data: {len(data['sex'])} cancer types")
    else:
        warnings.warn(f"Sex CSV not found: {sex_path}")

    # Race/Ethnicity (combined file)
    race_path = input_dir / 'shap_associations' / 'race_ethnicity_associations.csv'
    if race_path.exists():
        df_re = pd.read_csv(race_path)
        # Detect column name for race vs ethnicity
        if 'column' in df_re.columns:
            race_col = df_re['column'].unique()
            for col_name in race_col:
                subset = df_re[df_re['column'] == col_name].copy()
                if 'race' in col_name.lower():
                    data['race'] = subset
                    print(f"  Loaded race data: {len(subset)} cancer types")
                elif 'ethnicit' in col_name.lower():
                    data['ethnicity'] = subset
                    print(f"  Loaded ethnicity data: {len(subset)} cancer types")
        elif 'feature' in df_re.columns:
            for feat in df_re['feature'].unique():
                subset = df_re[df_re['feature'] == feat].copy()
                key = 'race' if 'race' in feat.lower() else 'ethnicity'
                data[key] = subset
                print(f"  Loaded {key} data: {len(subset)} cancer types")
        else:
            # Single feature file — assume race
            data['race'] = df_re
            print(f"  Loaded race data: {len(df_re)} cancer types")
    else:
        warnings.warn(f"Race/ethnicity CSV not found: {race_path}")

    return data


def load_gene_level_data(input_dir):
    """Load gene-stratified posthoc CSVs into a dict keyed by feature name."""
    input_dir = Path(input_dir)
    gene_dir = input_dir / 'gene_stratified'
    data = {}

    paths = {
        'age': gene_dir / 'age_by_gene' / 'age_gene_results.csv',
        'sex': gene_dir / 'sex_by_gene' / 'sex_gene_results.csv',
        'race': gene_dir / 'race_by_gene' / 'race_gene_results.csv',
        'ethnicity': gene_dir / 'ethnicity_by_gene' / 'ethnicity_gene_results.csv',
    }

    for feat, path in paths.items():
        if path.exists():
            data[feat] = pd.read_csv(path)
            print(f"  Loaded gene-level {feat} data: {len(data[feat])} rows")
        else:
            warnings.warn(f"Gene-level {feat} CSV not found: {path}")

    return data


# ---------------------------------------------------------------------------
# Core figure function
# ---------------------------------------------------------------------------

def _create_three_panel_figure(
    labels, effect_sizes, effect_ci_low, effect_ci_high,
    raw_diffs, pvals_fdr, significant,
    effect_label, raw_label, title,
    effect_center=0.0,
    gene_ranks=None,
    figsize_per_row=0.35,
    min_fig_height=4.0
):
    """Create a 3-panel horizontally-stacked composite figure.

    Parameters
    ----------
    labels : array-like of str
    effect_sizes : array-like of float
    effect_ci_low, effect_ci_high : array-like of float (can contain NaN)
    raw_diffs : array-like of float (can contain NaN)
    pvals_fdr : array-like of float
    significant : array-like of bool
    effect_label : str — x-axis label for left panel
    raw_label : str — x-axis label for center panel
    title : str — suptitle
    effect_center : float — reference line in left panel (0 for d, 0 for logOR)
    gene_ranks : array-like of int or None — if provided, marker shape varies
    figsize_per_row : float — inches per row
    min_fig_height : float

    Returns
    -------
    list of (fig, page_number) tuples
    """
    n = len(labels)
    if n == 0:
        return []

    # Paginate if needed
    pages = []
    for start in range(0, n, MAX_ROWS_PER_PAGE):
        end = min(start + MAX_ROWS_PER_PAGE, n)
        page_slice = slice(start, end)
        n_rows = end - start

        fig_height = max(min_fig_height, n_rows * figsize_per_row + 1.5)
        fig = plt.figure(figsize=(14, fig_height))

        gs = gridspec.GridSpec(1, 3, width_ratios=[3, 2, 1.5],
                               wspace=0.08, left=0.22, right=0.95,
                               top=0.92, bottom=0.08)

        ax_effect = fig.add_subplot(gs[0])
        ax_raw = fig.add_subplot(gs[1])
        ax_sig = fig.add_subplot(gs[2])

        y_pos = np.arange(n_rows)
        p_labels = np.asarray(labels)[page_slice]
        p_effect = np.asarray(effect_sizes, dtype=float)[page_slice]
        p_ci_lo = np.asarray(effect_ci_low, dtype=float)[page_slice]
        p_ci_hi = np.asarray(effect_ci_high, dtype=float)[page_slice]
        p_raw = np.asarray(raw_diffs, dtype=float)[page_slice]
        p_pvals = np.asarray(pvals_fdr, dtype=float)[page_slice]
        p_sig = np.asarray(significant, dtype=bool)[page_slice]
        p_ranks = np.asarray(gene_ranks)[page_slice] if gene_ranks is not None else None

        colors = [COLOR_SIG if s else COLOR_NONSIG for s in p_sig]

        # --- Left panel: effect size forest plot ---
        for i in range(n_rows):
            ci_lo_i = p_ci_lo[i] if not np.isnan(p_ci_lo[i]) else p_effect[i]
            ci_hi_i = p_ci_hi[i] if not np.isnan(p_ci_hi[i]) else p_effect[i]
            ax_effect.plot([ci_lo_i, ci_hi_i], [y_pos[i], y_pos[i]],
                           color=colors[i], linewidth=1.5, solid_capstyle='round')

            marker = 'o'
            if p_ranks is not None:
                try:
                    marker = 's' if int(p_ranks[i]) == 2 else 'o'
                except (ValueError, TypeError):
                    pass

            ax_effect.plot(p_effect[i], y_pos[i], marker=marker,
                           color=colors[i], markersize=7, zorder=3)

        ax_effect.axvline(effect_center, color=THRESHOLD_COLOR, linestyle='--',
                          linewidth=0.8, alpha=0.7)
        ax_effect.set_yticks(y_pos)
        ax_effect.set_yticklabels(p_labels, fontsize=8)
        ax_effect.set_xlabel(effect_label, fontsize=9)
        ax_effect.invert_yaxis()
        ax_effect.set_title("Effect Size", fontsize=10, fontweight='bold')
        ax_effect.grid(axis='x', alpha=0.3)

        # --- Center panel: raw difference lollipop ---
        has_raw = ~np.isnan(p_raw)
        for i in range(n_rows):
            if has_raw[i]:
                lollipop_color = COLOR_POS if p_raw[i] >= 0 else COLOR_NEG
                if not p_sig[i]:
                    lollipop_color = COLOR_NONSIG
                ax_raw.plot([0, p_raw[i]], [y_pos[i], y_pos[i]],
                            color=lollipop_color, linewidth=1.5)
                ax_raw.plot(p_raw[i], y_pos[i], 'o', color=lollipop_color,
                            markersize=5, zorder=3)

        ax_raw.axvline(0, color=THRESHOLD_COLOR, linestyle='--',
                        linewidth=0.8, alpha=0.7)
        ax_raw.set_yticks(y_pos)
        ax_raw.set_yticklabels([])
        ax_raw.set_xlabel(raw_label, fontsize=9)
        ax_raw.invert_yaxis()
        ax_raw.set_title("Raw Difference", fontsize=10, fontweight='bold')
        ax_raw.grid(axis='x', alpha=0.3)

        # --- Right panel: significance dot plot ---
        neg_log_p = -np.log10(np.clip(p_pvals, 1e-300, 1.0))
        threshold = -np.log10(0.05)

        for i in range(n_rows):
            marker = 'o'
            if p_ranks is not None:
                try:
                    marker = 's' if int(p_ranks[i]) == 2 else 'o'
                except (ValueError, TypeError):
                    pass
            ax_sig.plot(neg_log_p[i], y_pos[i], marker=marker,
                        color=colors[i], markersize=7, zorder=3)

        ax_sig.axvline(threshold, color=COLOR_SIG, linestyle='--',
                        linewidth=0.8, alpha=0.7, label='FDR = 0.05')
        ax_sig.set_yticks(y_pos)
        ax_sig.set_yticklabels([])
        ax_sig.set_xlabel('-log₁₀(p_FDR)', fontsize=9)
        ax_sig.invert_yaxis()
        ax_sig.set_title("Significance", fontsize=10, fontweight='bold')
        ax_sig.grid(axis='x', alpha=0.3)
        ax_sig.set_xlim(left=0)

        # Legend for gene ranks
        if p_ranks is not None:
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='gray', label='Gene rank 1',
                       markersize=7, linestyle='None'),
                Line2D([0], [0], marker='s', color='gray', label='Gene rank 2',
                       markersize=7, linestyle='None'),
            ]
            ax_sig.legend(handles=legend_elements, fontsize=7, loc='lower right')

        page_num = start // MAX_ROWS_PER_PAGE + 1
        total_pages = (n + MAX_ROWS_PER_PAGE - 1) // MAX_ROWS_PER_PAGE
        page_suffix = f" (page {page_num}/{total_pages})" if total_pages > 1 else ""
        fig.suptitle(title + page_suffix, fontsize=12, fontweight='bold', y=0.97)

        pages.append((fig, page_num))

    return pages


# ---------------------------------------------------------------------------
# Feature-specific composites
# ---------------------------------------------------------------------------

def _sort_df(df, sort_col, ascending=False):
    """Sort dataframe by absolute value of sort_col."""
    df = df.copy()
    df['_abs_sort'] = df[sort_col].abs()
    df = df.sort_values('_abs_sort', ascending=ascending).drop(columns=['_abs_sort'])
    return df


def create_age_composite(df, output_dir, level='cancer', sort_by='effect_size'):
    """Create age hit-list composite figure."""
    if df is None or df.empty:
        print("  Skipping age composite: no data")
        return []

    sort_col = 'cohens_d' if sort_by == 'effect_size' else 'mann_whitney_p_fdr'
    asc = sort_by == 'p_value'
    df = _sort_df(df, sort_col, ascending=asc)

    if level == 'gene':
        labels = (df['ONCOTREE_CODE'] + ' - ' + df['Hugo_Symbol']).tolist()
        n_act = df.get('n_actionable', df.get('n_gene_actionable', pd.Series([50]*len(df))))
        n_nonact = df.get('n_non_actionable', pd.Series([50]*len(df)))
        gene_ranks = df.get('gene_rank', None)
        if gene_ranks is not None:
            gene_ranks = gene_ranks.tolist()
    else:
        labels = df['ONCOTREE_CODE'].tolist()
        n_act = df['actionable_n']
        n_nonact = df['non_actionable_n']
        gene_ranks = None

    # Compute CIs for Cohen's d
    ci_low, ci_high = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        n1 = n_act.iloc[i] if isinstance(n_act, pd.Series) else 50
        n2 = n_nonact.iloc[i] if isinstance(n_nonact, pd.Series) else 50
        lo, hi = calculate_cohens_d_ci(row['cohens_d'], n1, n2)
        ci_low.append(lo)
        ci_high.append(hi)

    p_col = 'mann_whitney_p_fdr' if 'mann_whitney_p_fdr' in df.columns else 'p_fdr'
    sig_col = 'mann_whitney_significant_fdr' if 'mann_whitney_significant_fdr' in df.columns else 'significant_fdr'
    significant = df[sig_col].astype(bool) if sig_col in df.columns else df[p_col] < 0.05

    title = "Age × Actionability" + (" (Gene-Stratified)" if level == 'gene' else "")
    return _create_three_panel_figure(
        labels=labels,
        effect_sizes=df['cohens_d'].tolist(),
        effect_ci_low=ci_low,
        effect_ci_high=ci_high,
        raw_diffs=df['mean_age_difference'].tolist(),
        pvals_fdr=df[p_col].tolist(),
        significant=significant.tolist(),
        effect_label="Cohen's d",
        raw_label="Mean age diff (years)",
        title=title,
        effect_center=0.0,
        gene_ranks=gene_ranks,
    )


def create_sex_composite(df, output_dir, level='cancer', sort_by='effect_size'):
    """Create sex hit-list composite figure."""
    if df is None or df.empty:
        print("  Skipping sex composite: no data")
        return []

    # Compute log2 OR
    df = df.copy()
    df['log2_or'] = np.log2(df['odds_ratio'].clip(lower=1e-10))

    sort_col = 'log2_or' if sort_by == 'effect_size' else 'p_fdr'
    asc = sort_by == 'p_value'
    df = _sort_df(df, sort_col, ascending=asc)

    if level == 'gene':
        labels = (df['ONCOTREE_CODE'] + ' - ' + df['Hugo_Symbol']).tolist()
        gene_ranks = df.get('gene_rank', None)
        if gene_ranks is not None:
            gene_ranks = gene_ranks.tolist()
        # Compute rate difference
        if 'gene_female_rate' in df.columns and 'non_actionable_female_rate' in df.columns:
            raw_diffs = ((df['gene_female_rate'] - df['non_actionable_female_rate']) * 100).tolist()
        else:
            raw_diffs = [np.nan] * len(df)
    else:
        labels = df['ONCOTREE_CODE'].tolist()
        gene_ranks = None
        raw_diffs = ((df['female_rate'] - df['male_rate']) * 100).tolist()

    # CI for log2 OR
    ci_low = np.log2(df['or_ci_lower'].clip(lower=1e-10)).tolist()
    ci_high = np.log2(df['or_ci_upper'].clip(lower=1e-10)).tolist()

    significant = df['significant_fdr'].astype(bool) if 'significant_fdr' in df.columns else df['p_fdr'] < 0.05

    title = "Sex × Actionability" + (" (Gene-Stratified)" if level == 'gene' else "")
    return _create_three_panel_figure(
        labels=labels,
        effect_sizes=df['log2_or'].tolist(),
        effect_ci_low=ci_low,
        effect_ci_high=ci_high,
        raw_diffs=raw_diffs,
        pvals_fdr=df['p_fdr'].tolist(),
        significant=significant.tolist(),
        effect_label="log₂(Odds Ratio)",
        raw_label="Rate diff (F−M, %)",
        title=title,
        effect_center=0.0,
        gene_ranks=gene_ranks,
    )


def create_race_composite(df, output_dir, level='cancer', sort_by='effect_size'):
    """Create race hit-list composite figure."""
    if df is None or df.empty:
        print("  Skipping race composite: no data")
        return []

    sort_col = 'cramers_v' if sort_by == 'effect_size' else 'p_fdr'
    asc = sort_by == 'p_value'
    df = _sort_df(df, sort_col, ascending=asc)

    if level == 'gene':
        labels = (df['ONCOTREE_CODE'] + ' - ' + df['Hugo_Symbol']).tolist()
        gene_ranks = df.get('gene_rank', None)
        if gene_ranks is not None:
            gene_ranks = gene_ranks.tolist()
    else:
        labels = df['ONCOTREE_CODE'].tolist()
        gene_ranks = None

    # Raw difference: max rate spread from contingency table (cancer level only)
    if 'contingency_table' in df.columns:
        raw_diffs = (df['contingency_table'].apply(compute_max_rate_spread) * 100).tolist()
    else:
        raw_diffs = [np.nan] * len(df)

    # Cramer's V has no CI — use NaN
    ci_low = [np.nan] * len(df)
    ci_high = [np.nan] * len(df)

    significant = df['significant_fdr'].astype(bool) if 'significant_fdr' in df.columns else df['p_fdr'] < 0.05

    title = f"{RACE_DISPLAY_LABEL} × Actionability" + (" (Gene-Stratified)" if level == 'gene' else "")
    return _create_three_panel_figure(
        labels=labels,
        effect_sizes=df['cramers_v'].tolist(),
        effect_ci_low=ci_low,
        effect_ci_high=ci_high,
        raw_diffs=raw_diffs,
        pvals_fdr=df['p_fdr'].tolist(),
        significant=significant.tolist(),
        effect_label="Cramér's V",
        raw_label="Max rate spread (%)",
        title=title,
        effect_center=0.0,
        gene_ranks=gene_ranks,
    )


def create_ethnicity_composite(df, output_dir, level='cancer', sort_by='effect_size'):
    """Create ethnicity hit-list composite figure."""
    if df is None or df.empty:
        print("  Skipping ethnicity composite: no data")
        return []

    sort_col = 'cramers_v' if sort_by == 'effect_size' else 'p_fdr'
    asc = sort_by == 'p_value'
    df = _sort_df(df, sort_col, ascending=asc)

    if level == 'gene':
        labels = (df['ONCOTREE_CODE'] + ' - ' + df['Hugo_Symbol']).tolist()
        gene_ranks = df.get('gene_rank', None)
        if gene_ranks is not None:
            gene_ranks = gene_ranks.tolist()
    else:
        labels = df['ONCOTREE_CODE'].tolist()
        gene_ranks = None

    if 'contingency_table' in df.columns:
        raw_diffs = (df['contingency_table'].apply(compute_max_rate_spread) * 100).tolist()
    else:
        raw_diffs = [np.nan] * len(df)

    ci_low = [np.nan] * len(df)
    ci_high = [np.nan] * len(df)

    significant = df['significant_fdr'].astype(bool) if 'significant_fdr' in df.columns else df['p_fdr'] < 0.05

    title = "Ethnicity × Actionability" + (" (Gene-Stratified)" if level == 'gene' else "")
    return _create_three_panel_figure(
        labels=labels,
        effect_sizes=df['cramers_v'].tolist(),
        effect_ci_low=ci_low,
        effect_ci_high=ci_high,
        raw_diffs=raw_diffs,
        pvals_fdr=df['p_fdr'].tolist(),
        significant=significant.tolist(),
        effect_label="Cramér's V",
        raw_label="Max rate spread (%)",
        title=title,
        effect_center=0.0,
        gene_ranks=gene_ranks,
    )


# ---------------------------------------------------------------------------
# Summary dashboard
# ---------------------------------------------------------------------------

def create_summary_dashboard(data, output_dir, level='cancer', sort_by='effect_size'):
    """Create a 2×2 mini-grid dashboard with all four feature composites."""
    feature_order = ['age', 'sex', 'race', 'ethnicity']
    available = [f for f in feature_order if f in data and data[f] is not None and not data[f].empty]

    if len(available) == 0:
        print("  Skipping summary dashboard: no data")
        return None

    n_panels = len(available)
    ncols = 2
    nrows = (n_panels + 1) // 2

    fig = plt.figure(figsize=(18, 5 * nrows))
    outer_gs = gridspec.GridSpec(nrows, ncols, wspace=0.35, hspace=0.35,
                                  left=0.06, right=0.97, top=0.93, bottom=0.05)

    titles = {
        'age': 'Age', 'sex': 'Sex',
        'race': RACE_DISPLAY_LABEL, 'ethnicity': 'Ethnicity'
    }
    effect_labels = {
        'age': "Cohen's d", 'sex': 'log₂(OR)',
        'race': "Cramér's V", 'ethnicity': "Cramér's V"
    }

    for idx, feat in enumerate(available):
        row, col = divmod(idx, ncols)
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer_gs[row, col],
            width_ratios=[3, 2, 1.5], wspace=0.08
        )

        df = data[feat].copy()
        # Prep mini data (top 15 by effect size)
        if feat == 'age':
            sort_col_name = 'cohens_d'
        elif feat == 'sex':
            df['log2_or'] = np.log2(df['odds_ratio'].clip(lower=1e-10))
            sort_col_name = 'log2_or'
        else:
            sort_col_name = 'cramers_v'

        df['_abs_sort'] = df[sort_col_name].abs()
        df = df.sort_values('_abs_sort', ascending=False).head(15).drop(columns=['_abs_sort'])

        if level == 'gene' and 'Hugo_Symbol' in df.columns:
            labels = (df['ONCOTREE_CODE'] + '-' + df['Hugo_Symbol']).tolist()
        else:
            labels = df['ONCOTREE_CODE'].tolist()

        effect = df[sort_col_name].tolist()

        # p values
        if 'mann_whitney_p_fdr' in df.columns:
            pvals = df['mann_whitney_p_fdr'].tolist()
        else:
            pvals = df['p_fdr'].tolist()

        sig_col = None
        for c in ['mann_whitney_significant_fdr', 'significant_fdr']:
            if c in df.columns:
                sig_col = c
                break
        sig = df[sig_col].astype(bool).tolist() if sig_col else [p < 0.05 for p in pvals]

        n_rows = len(labels)
        y_pos = np.arange(n_rows)
        colors = [COLOR_SIG if s else COLOR_NONSIG for s in sig]

        # Mini effect panel
        ax_e = fig.add_subplot(inner_gs[0])
        for i in range(n_rows):
            ax_e.plot(effect[i], y_pos[i], 'o', color=colors[i], markersize=5)
        ax_e.axvline(0, color=THRESHOLD_COLOR, linestyle='--', linewidth=0.6, alpha=0.6)
        ax_e.set_yticks(y_pos)
        ax_e.set_yticklabels(labels, fontsize=6)
        ax_e.invert_yaxis()
        ax_e.set_xlabel(effect_labels[feat], fontsize=7)
        ax_e.tick_params(axis='x', labelsize=6)
        ax_e.set_title(titles[feat], fontsize=9, fontweight='bold')
        ax_e.grid(axis='x', alpha=0.2)

        # Mini raw diff panel
        ax_r = fig.add_subplot(inner_gs[1])
        if feat == 'age':
            raw = df['mean_age_difference'].tolist()
        elif feat == 'sex':
            if 'female_rate' in df.columns and 'male_rate' in df.columns:
                raw = ((df['female_rate'] - df['male_rate']) * 100).tolist()
            else:
                raw = [np.nan] * n_rows
        else:
            if 'contingency_table' in df.columns:
                raw = (df['contingency_table'].apply(compute_max_rate_spread) * 100).tolist()
            else:
                raw = [np.nan] * n_rows

        for i in range(n_rows):
            if not np.isnan(raw[i]):
                c = COLOR_POS if raw[i] >= 0 else COLOR_NEG
                if not sig[i]:
                    c = COLOR_NONSIG
                ax_r.plot([0, raw[i]], [y_pos[i], y_pos[i]], color=c, linewidth=1)
                ax_r.plot(raw[i], y_pos[i], 'o', color=c, markersize=4)
        ax_r.axvline(0, color=THRESHOLD_COLOR, linestyle='--', linewidth=0.6, alpha=0.6)
        ax_r.set_yticks(y_pos)
        ax_r.set_yticklabels([])
        ax_r.invert_yaxis()
        ax_r.tick_params(axis='x', labelsize=6)
        ax_r.grid(axis='x', alpha=0.2)

        # Mini significance panel
        ax_s = fig.add_subplot(inner_gs[2])
        neg_log_p = -np.log10(np.clip(pvals, 1e-300, 1.0))
        for i in range(n_rows):
            ax_s.plot(neg_log_p[i], y_pos[i], 'o', color=colors[i], markersize=5)
        ax_s.axvline(-np.log10(0.05), color=COLOR_SIG, linestyle='--',
                      linewidth=0.6, alpha=0.6)
        ax_s.set_yticks(y_pos)
        ax_s.set_yticklabels([])
        ax_s.invert_yaxis()
        ax_s.set_xlabel('-log₁₀(p)', fontsize=7)
        ax_s.tick_params(axis='x', labelsize=6)
        ax_s.set_xlim(left=0)
        ax_s.grid(axis='x', alpha=0.2)

    level_label = "Gene-Stratified" if level == 'gene' else "Cancer-Type"
    fig.suptitle(f"Hit List Summary Dashboard ({level_label})",
                 fontsize=14, fontweight='bold', y=0.98)
    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_figure(fig, output_dir, filename, fmt):
    """Save figure in requested format(s)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = ['pdf', 'png'] if fmt == 'both' else [fmt]
    for f in formats:
        path = output_dir / f"{filename}.{f}"
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Create hit-list composite figures for post-hoc results'
    )
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Path to tier7_posthoc_statistics directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: input_dir/hit_list_figures/)')
    parser.add_argument('--level', type=str, required=True,
                        choices=['cancer', 'gene'],
                        help='Analysis level: cancer-type or gene-by-cancer-type')
    parser.add_argument('--features', type=str, nargs='+',
                        default=['all'],
                        choices=['age', 'sex', 'race', 'ethnicity', 'all'],
                        help='Features to plot (default: all)')
    parser.add_argument('--sort-by', type=str, default='effect_size',
                        choices=['effect_size', 'p_value'],
                        help='Sort rows by effect size or p-value')
    parser.add_argument('--format', type=str, default='both',
                        choices=['pdf', 'png', 'both'],
                        help='Output format (default: both)')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Skip combined summary dashboard figure')

    args = parser.parse_args()

    # Resolve input directory
    input_dir = Path(args.input_dir) if args.input_dir else get_default_input_dir()
    if input_dir is None or not Path(input_dir).exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        print("Specify --input-dir or ensure a run exists in runs/")
        sys.exit(1)
    input_dir = Path(input_dir)
    print(f"Input directory: {input_dir}")

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'hit_list_figures'

    level_subdir = 'cancer_level' if args.level == 'cancer' else 'gene_level'
    output_dir = output_dir / level_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine which features to plot
    if 'all' in args.features:
        features = ['age', 'sex', 'race', 'ethnicity']
    else:
        features = args.features

    # Load data
    print(f"\nLoading {args.level}-level data...")
    if args.level == 'cancer':
        data = load_cancer_level_data(input_dir)
    else:
        data = load_gene_level_data(input_dir)

    if not data:
        print("ERROR: No data loaded. Check input directory and CSV files.")
        sys.exit(1)

    # Create composite figures
    creators = {
        'age': create_age_composite,
        'sex': create_sex_composite,
        'race': create_race_composite,
        'ethnicity': create_ethnicity_composite,
    }

    prefix = 'hit_list_'
    suffix = '_gene_composite' if args.level == 'gene' else '_composite'
    n_saved = 0

    for feat in features:
        if feat not in data:
            print(f"\n  No {feat} data available, skipping.")
            continue

        print(f"\nCreating {feat} composite figure...")
        creator = creators[feat]
        pages = creator(data[feat], output_dir, level=args.level, sort_by=args.sort_by)

        for fig, page_num in pages:
            page_suffix = f"_p{page_num}" if len(pages) > 1 else ""
            filename = f"{prefix}{feat}{suffix}{page_suffix}"
            _save_figure(fig, output_dir, filename, args.format)
            n_saved += 1

    # Summary dashboard
    if not args.no_dashboard and len([f for f in features if f in data]) > 0:
        print("\nCreating summary dashboard...")
        fig = create_summary_dashboard(data, output_dir, level=args.level,
                                        sort_by=args.sort_by)
        if fig is not None:
            dash_name = 'hit_list_gene_summary_dashboard' if args.level == 'gene' else 'hit_list_summary_dashboard'
            _save_figure(fig, output_dir, dash_name, args.format)
            n_saved += 1

    print(f"\nDone. {n_saved} figure(s) saved to {output_dir}")


if __name__ == '__main__':
    main()
