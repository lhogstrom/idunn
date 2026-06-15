#!/usr/bin/env python3
"""
Plot RF SHAP vs. LR Odds-Ratio Feature Ranks (stacked panels).

Top panel: same content as ``technical_demographic_feature_shap_rank_all.pdf``
(RF SHAP rank distribution across cancer types, demographic | technical).

Bottom panel: equivalent rank distribution, but ranks are derived from the LR
pathway -- by default per-cancer-type rank by max |log(OR)| of the
*standardized* coefficients (the canonical LR ranking metric in this
codebase). Pass ``--lr-rank-by log_odds`` to rank by raw |log(OR)| instead.

Default output: ``runs/shap_vs_lr_comparison/`` (matches the precedent set by
``compare_shap_vs_lr_coefficients.py``). The orchestrator override
``--output-dir`` points at ``tier7_posthoc_statistics/cross_pathway/`` of the
new LR run when wired into ``run_all_analyses.py``.

By default both ``--shap-run-dir`` and ``--lr-run-dir`` are auto-detected as
the latest run dirs containing the relevant artifacts.

Usage::

    # Auto-detect both pathways
    python plot_rf_shap_lr_or_feature_ranks.py

    # Pin specific runs
    python plot_rf_shap_lr_or_feature_ranks.py \
        --shap-run-dir runs/run_20260123_075513 \
        --lr-run-dir   runs/run_20260313_111146
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))
sys.path.insert(0, str(REPO_ROOT / "code" / "analysis" / "shap_interactions"))

from feature_name_mapper import (  # noqa: E402
    DEMOGRAPHIC_FEATURES,
    TECHNICAL_FEATURES,
    FeatureNameMapper,
    classify_feature_type,
    create_mapper_from_data_loader,
)
from feature_config import get_feature_list  # noqa: E402
from figure_style import (  # noqa: E402
    apply_publication_style,
    get_feature_category_colors,
)

DEFAULT_OUTPUT = REPO_ROOT / "runs" / "shap_vs_lr_comparison"


def autodetect_shap_run_dir() -> Path | None:
    """Return the newest ``runs/run_*`` containing RF SHAP outputs, or None."""
    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.exists():
        return None
    for run_dir in sorted(runs_dir.glob("run_*"), reverse=True):
        primary = run_dir / "tier6_shap_analysis" / "primary_values"
        if not primary.exists():
            continue
        if any(primary.rglob("combined_feature_importance.csv")):
            return run_dir
    return None


def autodetect_lr_run_dir() -> Path | None:
    """Return the newest ``runs/run_*`` containing LR coefficient outputs, or None."""
    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.exists():
        return None
    for run_dir in sorted(runs_dir.glob("run_*"), reverse=True):
        candidate = (
            run_dir
            / "tier6_shap_analysis"
            / "lr_analysis"
            / "combined_lr_grouped_importance.csv"
        )
        if candidate.exists():
            return run_dir
    return None

# Category colors come from the shared palette (code/config/palette.json) via
# figure_style — keep figures coordinated instead of hard-coding hex here.
CATEGORY_COLORS = get_feature_category_colors()
SAMPLE_TYPE_DISPLAY = {"Sample Type", "Sample Type (Detailed)"}

# Normalize older SHAP display names to current features.yaml labels so the
# top and bottom panels share an x-axis.
DISPLAY_NAME_ALIASES = {"Race": "Ancestry"}


def categorize_feature(display_name: str) -> str:
    if display_name in SAMPLE_TYPE_DISPLAY:
        return "sample_type"
    return classify_feature_type(display_name)


def normalize_display(name: str) -> str:
    return DISPLAY_NAME_ALIASES.get(name, name)


def build_feature_mapper():
    """Mapper over the full modeled feature set (``both``), used to collapse
    one-hot transformed feature names back to their parent input variable."""
    return create_mapper_from_data_loader(get_feature_list("both"))


def collapse_to_parent(raw_name: str, mapper) -> str:
    """Map a (possibly one-hot-encoded) transformed feature name to its parent
    input-variable display name (e.g. ``cat__SUBJECT_CENTER_MSK`` -> "Subject
    Center", ``cat__ANCESTRY_Asian`` -> "Ancestry").

    Why this lives here: the upstream ``*_grouped_feature_importance.csv`` (RF)
    and ``combined_lr_grouped_importance.csv`` (LR) are *meant* to be collapsed
    to input variables, but the collapse silently degraded to per-one-hot rows
    once the preprocessor began emitting verbose transformed names that the old
    ``get_feature_group`` regex could not parse. Re-collapsing at load time keeps
    this panel at input-variable granularity regardless of whether the upstream
    CSVs were (re)generated with the repaired grouping. Idempotent on names that
    are already collapsed display names.
    """
    base = mapper.get_feature_group(raw_name)
    display = mapper.DISPLAY_NAMES.get(base, base)
    return normalize_display(display)


def resolve_shap_dir(run_dir: Path) -> Path:
    """Locate the directory containing per-cancer-type SHAP grouped CSVs."""
    candidates = [
        run_dir / "tier6_shap_analysis" / "primary_values",
        run_dir,
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        nested = sorted(cand.glob("run_*"), reverse=True)
        if nested:
            return nested[0]
        if any(cand.glob("*/*_grouped_feature_importance.csv")):
            return cand
    raise FileNotFoundError(f"Could not find SHAP grouped CSVs under {run_dir}")


def resolve_lr_dir(run_dir: Path) -> Path:
    """Locate the directory containing combined_lr_grouped_importance.csv."""
    candidates = [
        run_dir / "tier6_shap_analysis" / "lr_analysis",
        run_dir,
    ]
    for cand in candidates:
        if cand.exists() and (cand / "combined_lr_grouped_importance.csv").exists():
            return cand
    raise FileNotFoundError(
        f"Could not find combined_lr_grouped_importance.csv under {run_dir}"
    )


def load_shap_ranks(shap_dir: Path, mapper) -> pd.DataFrame:
    """Load RF SHAP importance per cancer type and collapse to input variables.

    Reads each cancer's ``*_grouped_feature_importance.csv``, maps every
    (possibly one-hot) feature to its parent input variable, sums ``mean_abs_shap``
    within each parent per cancer type, then re-ranks. Summing per-column means is
    exact for the intended group importance because the mean over samples is
    linear: ``E[sum_c |shap_c|] = sum_c E[|shap_c|]``.
    """
    frames = []
    for sub in sorted(shap_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        f = sub / f"{sub.name}_grouped_feature_importance.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["cancer_type"] = sub.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No SHAP grouped CSVs under {shap_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["display_name"] = out["feature_name"].map(
        lambda n: collapse_to_parent(n, mapper)
    )
    grouped = out.groupby(["cancer_type", "display_name"], as_index=False)[
        "mean_abs_shap"
    ].sum()
    grouped["rank"] = (
        grouped.groupby("cancer_type")["mean_abs_shap"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    grouped["feature_type"] = grouped["display_name"].apply(classify_feature_type)
    grouped["category"] = grouped["display_name"].apply(categorize_feature)
    return grouped[["cancer_type", "display_name", "rank", "feature_type", "category"]]


def load_lr_ranks(lr_dir: Path, rank_by: str, mapper) -> pd.DataFrame:
    """
    Load combined LR grouped importance, collapse to input variables, and rank
    features within each cancer type by the requested odds-ratio-derived metric.

    rank_by:
        "std_log_odds" -> max |log(OR)| of standardized coefficients
                          (canonical LR rank used elsewhere in codebase)
        "log_odds"     -> max |log(OR)| of raw coefficients
                          (direct OR magnitude)

    The metric is a max over a feature's encoded categories, so collapsing a
    per-one-hot CSV to parent variables takes the max of the per-level maxes —
    identical to what a properly-grouped upstream CSV would carry.
    """
    metric_col = {
        "std_log_odds": "max_abs_std_log_odds",
        "log_odds": "max_abs_log_odds",
    }[rank_by]

    df = pd.read_csv(lr_dir / "combined_lr_grouped_importance.csv")
    df["display_name"] = df["feature_group"].map(
        lambda n: collapse_to_parent(n, mapper)
    )
    grouped = df.groupby(["cancer_type", "display_name"], as_index=False)[
        metric_col
    ].max()
    grouped["rank"] = (
        grouped.groupby("cancer_type")[metric_col]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    grouped["feature_type"] = grouped["display_name"].apply(classify_feature_type)
    grouped["category"] = grouped["display_name"].apply(categorize_feature)
    return grouped[["cancer_type", "display_name", "rank", "feature_type", "category"]]


def _draw_boxstrip(ax, data: pd.DataFrame, order: list, palette: dict,
                   title: str, ylabel: str, max_rank: int):
    if data.empty:
        ax.set_visible(False)
        return
    sns.boxplot(
        data=data,
        x="display_name",
        y="rank",
        ax=ax,
        order=order,
        palette=palette,
        width=0.6,
        fliersize=0,
    )
    sns.stripplot(
        data=data,
        x="display_name",
        y="rank",
        ax=ax,
        order=order,
        color="black",
        alpha=0.5,
        size=4,
    )
    ax.set_ylim(max_rank + 0.5, 0.5)  # invert with shared range
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def make_figure(
    shap_df: pd.DataFrame,
    lr_df: pd.DataFrame,
    lr_rank_by: str,
    output_dir: Path,
    filename_stem: str = "rf_shap_vs_lr_or_feature_rank_combined",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lock x-axis order to RF SHAP mean rank ascending (rank 1 = most important
    # = leftmost). Append any LR-only features at the end.
    shap_order = (
        shap_df.groupby("display_name")["rank"].mean().sort_values().index.tolist()
    )
    extra_lr = [f for f in lr_df["display_name"].unique() if f not in shap_order]
    feature_order = shap_order + sorted(extra_lr)

    palette = {f: CATEGORY_COLORS[categorize_feature(f)] for f in feature_order}

    max_rank = int(max(shap_df["rank"].max(), lr_df["rank"].max()))
    n_features = len(feature_order)
    width = max(12, n_features * 1.0 + 2)

    apply_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=(width, 12), sharex=True)

    _draw_boxstrip(
        axes[0],
        shap_df,
        order=feature_order,
        palette=palette,
        title="RF SHAP feature rank across cancer types",
        ylabel="SHAP rank (1 = most important)",
        max_rank=max_rank,
    )

    lr_rank_label = {
        "std_log_odds": "LR rank (max |log OR|, standardized)",
        "log_odds":     "LR rank (max |log OR|)",
    }[lr_rank_by]
    _draw_boxstrip(
        axes[1],
        lr_df,
        order=feature_order,
        palette=palette,
        title="LR odds-ratio feature rank across cancer types",
        ylabel=lr_rank_label,
        max_rank=max_rank,
    )

    # Shared legend for the 3 categories
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=CATEGORY_COLORS["demographic"], edgecolor="black",
              label="Demographic"),
        Patch(facecolor=CATEGORY_COLORS["sample_type"], edgecolor="black",
              label="Sample type"),
        Patch(facecolor=CATEGORY_COLORS["technical"], edgecolor="black",
              label="Technical"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right",
                   fontsize=10, framealpha=0.9)

    # In-figure title removed per figures/style/FIGURE_STYLE_GUIDE.md
    # (descriptive text moves to the figure caption).
    plt.tight_layout()

    pdf = output_dir / f"{filename_stem}.pdf"
    png = output_dir / f"{filename_stem}.png"
    fig.savefig(pdf, dpi=150, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shap-run-dir", type=Path, default=None,
        help="Run dir holding RF SHAP outputs. "
        "Default: newest runs/run_* with tier6_shap_analysis/primary_values/.",
    )
    p.add_argument(
        "--lr-run-dir", type=Path, default=None,
        help="Run dir holding LR coefficient outputs. "
        "Default: newest runs/run_* with tier6_shap_analysis/lr_analysis/.",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--lr-rank-by",
        choices=["std_log_odds", "log_odds"],
        default="std_log_odds",
        help="LR ranking metric within each cancer type "
        "(default: standardized log-odds, the codebase canonical).",
    )
    args = p.parse_args()

    shap_run_dir = args.shap_run_dir or autodetect_shap_run_dir()
    if shap_run_dir is None:
        raise SystemExit(
            "ERROR: could not auto-detect a SHAP run dir. "
            "Pass --shap-run-dir explicitly."
        )
    lr_run_dir = args.lr_run_dir or autodetect_lr_run_dir()
    if lr_run_dir is None:
        raise SystemExit(
            "ERROR: could not auto-detect an LR run dir. "
            "Pass --lr-run-dir explicitly."
        )

    shap_dir = resolve_shap_dir(shap_run_dir)
    lr_dir = resolve_lr_dir(lr_run_dir)
    print(f"SHAP run: {shap_run_dir}")
    print(f"LR run:   {lr_run_dir}")
    print(f"SHAP dir: {shap_dir}")
    print(f"LR dir:   {lr_dir}")

    mapper = build_feature_mapper()
    shap_df = load_shap_ranks(shap_dir, mapper)
    lr_df = load_lr_ranks(lr_dir, args.lr_rank_by, mapper)
    print(
        f"SHAP: {len(shap_df)} rows / {shap_df['cancer_type'].nunique()} cancers"
        f" / {shap_df['display_name'].nunique()} features"
    )
    print(
        f"LR:   {len(lr_df)} rows / {lr_df['cancer_type'].nunique()} cancers"
        f" / {lr_df['display_name'].nunique()} features"
    )

    make_figure(shap_df, lr_df, args.lr_rank_by, args.output_dir)


if __name__ == "__main__":
    main()
