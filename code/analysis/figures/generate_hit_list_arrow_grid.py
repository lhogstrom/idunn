#!/usr/bin/env python3
"""Hit-list arrow-grid visualization from gene-stratified lit review CSV.

Produces a Tier 1 figure summarizing age/sex hits as direction arrows
colored by Lit status (novel vs. confirmed). The companion Tier 2+3 panel
that previously rendered here was removed alongside the OncoTree rollup
strip on the publication branch.

Usage:
    cd code/analysis/figures
    python generate_hit_list_arrow_grid.py
    python generate_hit_list_arrow_grid.py --csv-path PATH --output-dir DIR
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))
from figure_style import apply_publication_style, get_color  # noqa: E402

# Hit-status colors from the shared palette (code/config/palette.json).
NOVEL_COLOR = get_color("hit_status.novel")
CONFIRMED_COLOR = get_color("hit_status.confirmed")

DIRECTION_TO_ARROW = {
    'older': '↑',
    'higher_female': '↑',
    'younger': '↓',
    'higher_male': '↓',
}

DEFAULT_CSV = REPO_ROOT / 'code_review_and_feedback' / 'lit_review' / 'gene_stratified_comparison_lit_review_hit_subset.csv'
DEFAULT_OUTDIR = REPO_ROOT / 'code_review_and_feedback' / 'lit_review' / 'figures'


def load_hits(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['direction', 'Lit status'])
    df['Lit status'] = df['Lit status'].astype(str).str.strip().str.lower()
    df['direction'] = df['direction'].astype(str).str.strip()
    df['feature_category'] = df['feature_category'].astype(str).str.strip().str.lower()
    return df


def build_name_map() -> dict:
    """Build {tier1_ONCOTREE_CODE: ot_name} from the most recent TIER 2a
    cancer_type_group_definitions.csv.

    The curated lit-review CSV stores hits with `tier1_` prefixed
    ONCOTREE_CODE values (an artifact of the rollup-aware run that produced
    them). We prefix the plain OncoTree codes from the new TIER 2a mapping
    to keep the lookup working without restoring the multi-tier utilities.
    """
    runs_root = REPO_ROOT / 'runs'
    candidates = sorted(
        runs_root.glob('run_*/tier2_data_setup/cancer_type_groupings/cancer_type_group_definitions.csv'),
        reverse=True,
    )
    if not candidates:
        return {}
    defs = pd.read_csv(candidates[0])
    return {f"tier1_{code}": name
            for code, name in zip(defs['ONCOTREE_CODE'], defs['ot_name'])}


def _color_for_status(status: str) -> str:
    return NOVEL_COLOR if status == 'novel' else CONFIRMED_COLOR


def plot_arrow_grid(df: pd.DataFrame, name_map: dict, tiers: list, title: str, out_path: Path) -> int:
    """Render one arrow-grid plot. Returns row count rendered."""
    tier_set = set(tiers)
    sub = df[df['ONCOTREE_CODE'].str.split('_').str[0].isin(tier_set)].copy()

    cells: dict = {}
    for _, row in sub.iterrows():
        key = (row['ONCOTREE_CODE'], row['feature_category'])
        cells.setdefault(key, []).append((row['Hugo_Symbol'], row['direction'], row['Lit status']))

    codes = sorted(set(sub['ONCOTREE_CODE']), key=lambda c: name_map.get(c, c).lower())
    n_rows = len(codes)

    col_x = {'code': 0.5, 'name': 2.4, 'age': 6.2, 'sex': 8.6}
    row_height = 0.7

    fig_height = max(5.0, row_height * n_rows + 2.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    header_y = n_rows + 0.6
    ax.text(col_x['code'], header_y, 'OncoTree code', fontsize=11, fontweight='bold', ha='left', va='center')
    ax.text(col_x['name'], header_y, 'Cancer type', fontsize=11, fontweight='bold', ha='left', va='center')
    ax.text(col_x['age'], header_y, 'Age', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(col_x['sex'], header_y, 'Sex', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.plot([0, 10.5], [n_rows + 0.05, n_rows + 0.05], color='gray', linewidth=0.6)

    for i, code in enumerate(codes):
        y = n_rows - i - 0.5
        display_code = code.split('_', 1)[1] if '_' in code else code
        ax.text(col_x['code'], y, display_code, fontsize=9, family='monospace', ha='left', va='center')
        ax.text(col_x['name'], y, name_map.get(code, code), fontsize=10, ha='left', va='center')

        for feat, x_center in [('age', col_x['age']), ('sex', col_x['sex'])]:
            entries = cells.get((code, feat), [])
            if not entries:
                ax.text(x_center, y, '–', fontsize=16, color='lightgray', ha='center', va='center')
                continue
            n = len(entries)
            spacing = 0.55
            start_x = x_center - spacing * (n - 1) / 2.0
            for j, (gene, direction, status) in enumerate(entries):
                gx = start_x + j * spacing
                arrow = DIRECTION_TO_ARROW.get(direction, '?')
                color = _color_for_status(status)
                ax.text(gx, y - 0.06, arrow, fontsize=22, fontweight='bold', color=color,
                        ha='center', va='center')
                ax.text(gx, y + 0.30, gene, fontsize=8, ha='center', va='center')

        if i < n_rows - 1:
            ax.plot([0, 10.5], [y - 0.5, y - 0.5], color='lightgray', linewidth=0.3)

    legend_y = -0.7
    legend_items = [
        ('↑', NOVEL_COLOR, 'Novel; older / female'),
        ('↑', CONFIRMED_COLOR, 'Confirmed; older / female'),
        ('↓', NOVEL_COLOR, 'Novel; younger / male'),
        ('↓', CONFIRMED_COLOR, 'Confirmed; younger / male'),
    ]
    legend_x_starts = [0.3, 3.0, 5.7, 8.4]
    for (arrow, color, label), lx in zip(legend_items, legend_x_starts):
        ax.text(lx, legend_y, arrow, fontsize=18, fontweight='bold', color=color, ha='center', va='center')
        ax.text(lx + 0.25, legend_y, label, fontsize=9, ha='left', va='center')

    ax.set_xlim(0, 10.7)
    ax.set_ylim(legend_y - 0.6, header_y + 0.6)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.axis('off')

    fig.savefig(str(out_path) + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(str(out_path) + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return n_rows


TIER_BADGE_COLORS = {
    'tier1': get_color("okabe_ito.blue"),
    'tier2': get_color("okabe_ito.bluish_green"),
    'tier3': get_color("okabe_ito.gray"),
}


def _draw_arrow(ax, x: float, y_center: float, direction: str, color: str,
                length: float = 0.36, lw: float = 2.4, head_scale: float = 12.0) -> None:
    """Draw a real arrow patch (FancyArrowPatch) so glyph rendering is font-independent."""
    is_up = direction in ('older', 'higher_female')
    if is_up:
        y0, y1 = y_center - length / 2.0, y_center + length / 2.0
    else:
        y0, y1 = y_center + length / 2.0, y_center - length / 2.0
    arrow = FancyArrowPatch(
        (x, y0), (x, y1),
        arrowstyle='-|>',
        mutation_scale=head_scale,
        color=color,
        linewidth=lw,
        capstyle='round',
        joinstyle='miter',
        zorder=3,
    )
    ax.add_patch(arrow)


def plot_arrow_grid_polished(df: pd.DataFrame, name_map: dict, tiers: list,
                             title: str, out_path: Path,
                             show_tier_badge: bool = False) -> int:
    """Polished version: real arrow patches, row banding, tier badges, compact legend."""
    tier_set = set(tiers)
    sub = df[df['ONCOTREE_CODE'].str.split('_').str[0].isin(tier_set)].copy()

    cells: dict = {}
    for _, row in sub.iterrows():
        key = (row['ONCOTREE_CODE'], row['feature_category'])
        cells.setdefault(key, []).append((row['Hugo_Symbol'], row['direction'], row['Lit status']))

    codes = sorted(set(sub['ONCOTREE_CODE']), key=lambda c: name_map.get(c, c).lower())
    n_rows = len(codes)
    n_arrows = len(sub)

    badge_x = 0.35
    code_x = 0.95 if show_tier_badge else 0.5
    name_x = 2.4 if show_tier_badge else 2.4
    age_x = 6.2
    sex_x = 8.6
    x_left, x_right = 0.0, 10.7
    row_h = 1.0

    fig_height = max(5.0, 0.7 * n_rows + 2.6)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    band_color = '#f5f5f5'
    for i in range(n_rows):
        if i % 2 == 0:
            y_top = n_rows - i
            rect = patches.Rectangle((x_left, y_top - row_h), x_right - x_left, row_h,
                                     facecolor=band_color, edgecolor='none', zorder=0)
            ax.add_patch(rect)

    header_y = n_rows + 0.55
    ax.text(code_x, header_y, 'OncoTree code', fontsize=11, fontweight='bold', ha='left', va='center')
    ax.text(name_x, header_y, 'Cancer type', fontsize=11, fontweight='bold', ha='left', va='center')
    ax.text(age_x, header_y, 'Age', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(sex_x, header_y, 'Sex', fontsize=12, fontweight='bold', ha='center', va='center')
    ax.plot([x_left, x_right], [n_rows + 0.04, n_rows + 0.04],
            color='#555555', linewidth=0.8, zorder=2)

    for div_x in [name_x - 0.25, age_x - 1.2, sex_x - 1.2]:
        ax.plot([div_x, div_x], [0, n_rows], color='#dcdcdc', linewidth=0.5, zorder=1)

    for i, code in enumerate(codes):
        y = n_rows - i - 0.5
        display_code = code.split('_', 1)[1] if '_' in code else code
        tier_key = code.split('_', 1)[0]

        if show_tier_badge:
            badge_color = TIER_BADGE_COLORS.get(tier_key, '#888888')
            ax.text(badge_x, y, f'T{tier_key[-1]}',
                    fontsize=7.5, fontweight='bold', color='white',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.28', facecolor=badge_color,
                              edgecolor='none'),
                    zorder=4)

        ax.text(code_x, y, display_code, fontsize=9, family='monospace',
                ha='left', va='center', zorder=4)
        ax.text(name_x, y, name_map.get(code, code), fontsize=10,
                ha='left', va='center', zorder=4)

        for feat, x_center in [('age', age_x), ('sex', sex_x)]:
            entries = cells.get((code, feat), [])
            if not entries:
                continue
            n = len(entries)
            spacing = 0.55
            start_x = x_center - spacing * (n - 1) / 2.0
            for j, (gene, direction, status) in enumerate(entries):
                gx = start_x + j * spacing
                color = _color_for_status(status)
                _draw_arrow(ax, gx, y - 0.04, direction, color)
                ax.text(gx, y + 0.32, gene, fontsize=8, ha='center', va='center', zorder=4)

    legend_y_color = -0.65
    legend_y_dir = -1.15

    ax.text(0.3, legend_y_color, 'Color:', fontsize=9, fontweight='bold', ha='left', va='center')
    cx = 1.2
    ax.add_patch(patches.Circle((cx, legend_y_color), 0.10, color=NOVEL_COLOR, zorder=4))
    ax.text(cx + 0.22, legend_y_color, 'Novel', fontsize=9, ha='left', va='center')
    cx2 = 3.3
    ax.add_patch(patches.Circle((cx2, legend_y_color), 0.10, color=CONFIRMED_COLOR, zorder=4))
    ax.text(cx2 + 0.22, legend_y_color, 'Confirmed', fontsize=9, ha='left', va='center')

    ax.text(0.3, legend_y_dir, 'Direction:', fontsize=9, fontweight='bold', ha='left', va='center')
    _draw_arrow(ax, 1.4, legend_y_dir, 'older', '#444444', length=0.32, lw=2.0, head_scale=14)
    ax.text(1.62, legend_y_dir, 'older / female', fontsize=9, ha='left', va='center')
    _draw_arrow(ax, 4.4, legend_y_dir, 'younger', '#444444', length=0.32, lw=2.0, head_scale=14)
    ax.text(4.62, legend_y_dir, 'younger / male', fontsize=9, ha='left', va='center')

    if show_tier_badge:
        bx = 7.0
        for label, tier_key in [('T2', 'tier2'), ('T3', 'tier3')]:
            ax.text(bx, legend_y_dir, label, fontsize=7.5, fontweight='bold', color='white',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.28',
                              facecolor=TIER_BADGE_COLORS[tier_key], edgecolor='none'))
            ax.text(bx + 0.25, legend_y_dir,
                    'Tier 2 (rescue)' if tier_key == 'tier2' else 'Tier 3 (anatomical)',
                    fontsize=9, ha='left', va='center')
            bx += 1.85

    subtitle = f'{n_rows} cancer types · {n_arrows} demographic associations'
    ax.text((x_left + x_right) / 2, header_y + 1.0, title,
            fontsize=14, fontweight='bold', ha='center', va='center')
    ax.text((x_left + x_right) / 2, header_y + 0.55, subtitle,
            fontsize=10, color='#555555', ha='center', va='center', style='italic')

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(legend_y_dir - 0.6, header_y + 1.4)
    ax.axis('off')

    fig.savefig(str(out_path) + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(str(out_path) + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return n_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-path', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    apply_publication_style()

    df = load_hits(args.csv_path)
    name_map = build_name_map()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_tier1 = plot_arrow_grid(df, name_map, ['tier1'],
                              'Lit-Review Hits: Tier 1 OncoTree Codes',
                              args.output_dir / 'tier1_hits_arrows')
    n_tier1_p = plot_arrow_grid_polished(df, name_map, ['tier1'],
                                         'Lit-Review Hits: Tier 1 OncoTree Codes',
                                         args.output_dir / 'tier1_hits_arrows_polished',
                                         show_tier_badge=False)

    print(f"Tier 1 plot: {n_tier1} rows (basic) / {n_tier1_p} rows (polished)")
    print(f"Total arrows: {len(df)}")
    print(f"Outputs written to: {args.output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
