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

CATEGORY_COLORS = {
    "demographic": "#3498db",   # blue
    "sample_type": "#9b59b6",   # purple - Sample Type / Sample Type (Detailed)
    "technical":   "#e74c3c",   # red
}
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


def load_shap_ranks(shap_dir: Path) -> pd.DataFrame:
    """Load SHAP grouped feature importance for every cancer type subdir."""
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
    out = out.rename(columns={"feature_name": "display_name"})
    out["display_name"] = out["display_name"].map(normalize_display)
    out["feature_type"] = out["display_name"].apply(classify_feature_type)
    out["category"] = out["display_name"].apply(categorize_feature)
    return out[["cancer_type", "display_name", "rank", "feature_type", "category"]]


def load_lr_ranks(lr_dir: Path, rank_by: str) -> pd.DataFrame:
    """
    Load combined LR grouped importance and rank features within each cancer
    type by the requested odds-ratio-derived metric.

    rank_by:
        "std_log_odds" -> max |log(OR)| of standardized coefficients
                          (canonical LR rank used elsewhere in codebase)
        "log_odds"     -> max |log(OR)| of raw coefficients
                          (direct OR magnitude)
    """
    metric_col = {
        "std_log_odds": "max_abs_std_log_odds",
        "log_odds": "max_abs_log_odds",
    }[rank_by]

    df = pd.read_csv(lr_dir / "combined_lr_grouped_importance.csv")
    df["rank"] = (
        df.groupby("cancer_type")[metric_col]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    display_map = FeatureNameMapper.DISPLAY_NAMES
    df["display_name"] = (
        df["feature_group"]
        .map(lambda x: display_map.get(x, x))
        .map(normalize_display)
    )
    df["feature_type"] = df["display_name"].apply(classify_feature_type)
    df["category"] = df["display_name"].apply(categorize_feature)
    return df[["cancer_type", "display_name", "rank", "feature_type", "category"]]


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

    n_cancers = max(shap_df["cancer_type"].nunique(), lr_df["cancer_type"].nunique())
    fig.suptitle(
        f"Feature rank distribution across cancer types (n={n_cancers}); "
        f"x-axis ordered by RF SHAP mean rank",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
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

    shap_df = load_shap_ranks(shap_dir)
    lr_df = load_lr_ranks(lr_dir, args.lr_rank_by)
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
