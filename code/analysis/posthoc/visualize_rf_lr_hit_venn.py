#!/usr/bin/env python3
"""
RF vs LR Significant Gene-Stratified Hit Venn Diagrams.

Two figures:

1. ``venn_rf_vs_lr_hits.{pdf,png}`` -- 2x3 panel of Venn diagrams, one per
   feature (age, sex, ancestry, ethnicity, sample_type, technical), showing
   FDR-significant gene-stratified hits in the RF pathway vs. the LR pathway.
2. ``venn_rf_vs_lr_hits_pooled.{pdf,png}`` -- single Venn pooled across all
   selected features.

A "hit" is keyed on ``(ONCOTREE_CODE, Hugo_Symbol, feature)`` and appears when
that row is FDR-significant in
``tier7_posthoc_statistics/gene_stratified/<feature>_by_gene/<feature>_gene_significant.csv``.
The patient-level test statistics agree across pathways for any pair both
pathways test (see CLAUDE.md, "RF/LR cross-pathway hit intersection"); set
differences come from per-pathway top-N feature filters and BH denominators.
The ``sample_type`` and ``technical`` panels reflect pathway-level differences
in feature ranking (RF SHAP weights confounders more heavily than LR
coefficients), not biological signal -- the demographic panels are the ones
to interpret biologically.

Default output: ``runs/shap_vs_lr_comparison/`` (matches the precedent set by
``compare_shap_vs_lr_coefficients.py`` and ``plot_rf_shap_lr_or_feature_ranks.py``).
The orchestrator override ``--output-dir`` points at
``tier7_posthoc_statistics/cross_pathway/`` of the new LR run when wired into
``run_all_analyses.py``.

By default both ``--rf-run-dir`` and ``--lr-run-dir`` are auto-detected as
the latest run dirs containing gene-stratified significant CSVs.

Usage::

    # Auto-detect both pathways
    python visualize_rf_lr_hit_venn.py

    # Pin specific runs (post-SHAP-gate-audit; see code_review_and_feedback/shap_gate_provenance_audit_2026-05-07.md)
    python visualize_rf_lr_hit_venn.py \\
        --rf-run-dir runs/run_20260505_164207 \\
        --lr-run-dir runs/run_20260313_111146
    # Note: `tier7_posthoc_statistics/` must resolve to the post-audit regenerated
    # outputs (symlink `tier7_posthoc_statistics → tier7_posthoc_statistics_REGENERATED_20260507`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))

from feature_config import get_display_name  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "runs" / "shap_vs_lr_comparison"

FEATURES = ["age", "sex", "race", "ethnicity", "sample_type", "technical"]

# Map our feature-category key (the subdirectory / CSV-prefix used in TIER 7)
# to a YAML-driven display label. "technical" has no single YAML entry so we
# label it explicitly.
FEATURE_DISPLAY = {
    "age": get_display_name("AGE_AT_SEQ_REPORT"),
    "sex": get_display_name("SEX"),
    # Display label sourced from features.yaml under the post-rename key
    # ANCESTRY (Feature Review Fix b, 2026-05-21). The dict key stays "race"
    # because the TIER 7 bucket name in current output runs is still "race"
    # (see ROADMAP "ANCESTRY-rename loose ends" — bucket rename deferred to
    # lock-run time so locked outputs are uniformly post-rename).
    "race": get_display_name("ANCESTRY"),
    "ethnicity": get_display_name("ETHNICITY"),
    "sample_type": get_display_name("SAMPLE_TYPE"),
    "technical": "Technical",
}

RF_COLOR = "#1f77b4"
LR_COLOR = "#d62728"


def autodetect_rf_run_dir() -> Path | None:
    """Return the newest ``runs/run_*`` containing TIER 7 gene-stratified hits."""
    return _autodetect_gene_strat_run_dir()


def autodetect_lr_run_dir() -> Path | None:
    """Return the newest ``runs/run_*`` containing TIER 7 gene-stratified hits."""
    return _autodetect_gene_strat_run_dir()


def _autodetect_gene_strat_run_dir() -> Path | None:
    runs_dir = REPO_ROOT / "runs"
    if not runs_dir.exists():
        return None
    for run_dir in sorted(runs_dir.glob("run_*"), reverse=True):
        gene_strat = run_dir / "tier7_posthoc_statistics" / "gene_stratified"
        if not gene_strat.exists():
            continue
        if any(gene_strat.rglob("*_gene_significant.csv")):
            return run_dir
    return None


def load_hits(run_dir: Path, feature: str) -> set[tuple]:
    """Return FDR-significant hits keyed on (ONCOTREE_CODE, Hugo_Symbol, feature).

    Some categories (race, ethnicity, sample_type, technical) carry a
    ``feature`` sub-column with multiple sub-features per cancer-gene; we
    include it in the key so they're not silently deduped.
    """
    path = (
        run_dir
        / "tier7_posthoc_statistics"
        / "gene_stratified"
        / f"{feature}_by_gene"
        / f"{feature}_gene_significant.csv"
    )
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    if "feature" in df.columns:
        return set(zip(df["ONCOTREE_CODE"], df["Hugo_Symbol"], df["feature"]))
    return set(zip(df["ONCOTREE_CODE"], df["Hugo_Symbol"], [feature] * len(df)))


def draw_venn(ax, rf: set, lr: set, title: str) -> None:
    only_rf = len(rf - lr)
    only_lr = len(lr - rf)
    both = len(rf & lr)

    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")

    ax.add_patch(Circle((-0.55, 0), 1.0, alpha=0.45, facecolor=RF_COLOR,
                        edgecolor="black", linewidth=1.0))
    ax.add_patch(Circle(( 0.55, 0), 1.0, alpha=0.45, facecolor=LR_COLOR,
                        edgecolor="black", linewidth=1.0))

    ax.text(-1.15, 0, str(only_rf), ha="center", va="center",
            fontsize=14, fontweight="bold")
    ax.text( 1.15, 0, str(only_lr), ha="center", va="center",
            fontsize=14, fontweight="bold")
    ax.text( 0.00, 0, str(both), ha="center", va="center",
            fontsize=14, fontweight="bold")

    ax.text(-1.15,  1.15, "RF only", ha="center", va="bottom",
            fontsize=10, color=RF_COLOR, fontweight="bold")
    ax.text( 1.15,  1.15, "LR only", ha="center", va="bottom",
            fontsize=10, color=LR_COLOR, fontweight="bold")

    ax.set_title(
        f"{title}\nRF total={len(rf)}  LR total={len(lr)}  shared={both}",
        fontsize=11,
    )


def membership_rows(rf: set, lr: set, feature: str) -> list[dict]:
    rows = []
    for key in sorted(rf | lr):
        ot, gene, sub = key
        in_rf = key in rf
        in_lr = key in lr
        if in_rf and in_lr:
            membership = "both"
        elif in_rf:
            membership = "rf_only"
        else:
            membership = "lr_only"
        rows.append(
            dict(
                feature=feature,
                sub_feature=sub,
                ONCOTREE_CODE=ot,
                Hugo_Symbol=gene,
                membership=membership,
            )
        )
    return rows


def make_panel_figure(
    rf_run_dir: Path,
    lr_run_dir: Path,
    features: list[str],
    output_dir: Path,
) -> tuple[list[dict], set, set]:
    """Build the 2x3 per-feature panel and return rows + pooled sets."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    all_rows: list[dict] = []
    rf_pooled: set = set()
    lr_pooled: set = set()

    for ax, feature in zip(axes, features):
        rf = load_hits(rf_run_dir, feature)
        lr = load_hits(lr_run_dir, feature)
        draw_venn(ax, rf, lr, title=FEATURE_DISPLAY.get(feature, feature))
        all_rows.extend(membership_rows(rf, lr, feature))
        rf_pooled |= {(feature,) + key for key in rf}
        lr_pooled |= {(feature,) + key for key in lr}

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle(
        "RF vs LR gene-stratified significant hits (FDR < 0.05)\n"
        f"RF: {rf_run_dir.name}   LR: {lr_run_dir.name}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    pdf = output_dir / "venn_rf_vs_lr_hits.pdf"
    png = output_dir / "venn_rf_vs_lr_hits.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")

    return all_rows, rf_pooled, lr_pooled


def make_pooled_figure(
    rf_pooled: set,
    lr_pooled: set,
    rf_run_dir: Path,
    lr_run_dir: Path,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    draw_venn(ax, rf_pooled, lr_pooled, title="All features pooled")
    fig.suptitle(
        "RF vs LR gene-stratified significant hits (FDR < 0.05) — pooled across features\n"
        f"RF: {rf_run_dir.name}   LR: {lr_run_dir.name}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    pdf = output_dir / "venn_rf_vs_lr_hits_pooled.pdf"
    png = output_dir / "venn_rf_vs_lr_hits_pooled.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rf-run-dir", type=Path, default=None,
        help="Run dir holding RF TIER 7 gene-stratified hits. "
        "Default: newest runs/run_* with tier7_posthoc_statistics/gene_stratified/.",
    )
    p.add_argument(
        "--lr-run-dir", type=Path, default=None,
        help="Run dir holding LR TIER 7 gene-stratified hits. "
        "Default: newest runs/run_* with tier7_posthoc_statistics/gene_stratified/.",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--features", nargs="+", default=FEATURES,
                   help=f"Feature categories to include (default: {FEATURES}).")
    args = p.parse_args()

    rf_run_dir = args.rf_run_dir or autodetect_rf_run_dir()
    if rf_run_dir is None:
        raise SystemExit(
            "ERROR: could not auto-detect an RF run dir. "
            "Pass --rf-run-dir explicitly."
        )
    lr_run_dir = args.lr_run_dir or autodetect_lr_run_dir()
    if lr_run_dir is None:
        raise SystemExit(
            "ERROR: could not auto-detect an LR run dir. "
            "Pass --lr-run-dir explicitly."
        )

    print(f"RF run: {rf_run_dir}")
    print(f"LR run: {lr_run_dir}")
    print(f"Features: {args.features}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, rf_pooled, lr_pooled = make_panel_figure(
        rf_run_dir, lr_run_dir, args.features, args.output_dir
    )
    make_pooled_figure(rf_pooled, lr_pooled, rf_run_dir, lr_run_dir, args.output_dir)

    membership_df = pd.DataFrame(rows)
    membership_csv = args.output_dir / "venn_rf_vs_lr_hits_membership.csv"
    membership_df.to_csv(membership_csv, index=False)
    print(f"Saved: {membership_csv}")

    summary = (
        membership_df.groupby(["feature", "membership"]).size().unstack(fill_value=0)
    )
    summary_csv = args.output_dir / "venn_rf_vs_lr_hits_summary.csv"
    summary.to_csv(summary_csv)
    print(f"Saved: {summary_csv}")
    print(summary)
    print(
        f"\nPooled across all features: "
        f"both={len(rf_pooled & lr_pooled)}  "
        f"rf_only={len(rf_pooled - lr_pooled)}  "
        f"lr_only={len(lr_pooled - rf_pooled)}"
    )


if __name__ == "__main__":
    main()
