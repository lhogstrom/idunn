#!/usr/bin/env python3
"""Cross-pathway visualization of gene-stratified hits significant in BOTH RF and LR pathways.

Reads the per-feature `*_gene_significant.csv` from one RF run and one LR run,
computes the inner-join intersection on (ONCOTREE_CODE, Hugo_Symbol), and emits:

  RACE (column 'RACE')
    - race/intersection_summary.csv         one row per intersected hit, both p_fdr's
    - race/per_hit/<gene_race_*>            existing-style contingency plot per hit
    - race/fig1_lollipop_effect_sizes_{tier1,tier2_tier3}.{pdf,png}
    - race/fig2_diverging_dpp_small_multiples_{tier1,tier2_tier3}.{pdf,png}

  SEX (column 'SEX' — Female / Male odds ratio)
    - sex/intersection_summary.csv          OR + 95% CI + both p_fdr + F/M counts
    - sex/per_hit_counts/<gene_sex_*>.csv   per-hit F/M counts
    - sex/sex_or_forest_{all,tier1,tier2_tier3}.{pdf,png}

Usage:
    # Post-SHAP-gate-audit (2026-05-07); see code_review_and_feedback/shap_gate_provenance_audit_2026-05-07.md
    python visualize_rf_lr_hit_intersection.py \\
        --rf-run-dir runs/run_20260505_164207 \\
        --lr-run-dir runs/run_20260313_111146 \\
        [--features race sex]
        [--output-dir ...]
    # Note: `tier7_posthoc_statistics/` must resolve to the post-audit regenerated
    # outputs (symlink `tier7_posthoc_statistics → tier7_posthoc_statistics_REGENERATED_20260507`).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "code" / "tools"))

from db_config import get_default_db_path  # noqa: E402
from multi_tier_utils import (  # noqa: E402
    get_oncotree_codes_for_group,
    load_patient_to_group_mapping,
)
from raw_data_plot_utils import plot_categorical_contingency  # noqa: E402
from figure_style import apply_publication_style, get_palette, get_color  # noqa: E402

PATIENT_TABLE = "perPatientOncoKBFilteredForModeling"
MAPPING_REL_PATH = Path("tier2_data_setup/multi_tier_groupings/patient_to_group_mapping.csv")

RACE_ORDER = [
    "European", "Asian", "African", "Other", "Unknown",
    "Native American", "Pacific Islander",
]

# Colors sourced from the shared palette (code/config/palette.json).
SYSTEM_COLOR_DEFAULT = get_color("cancer_systems.default")
SYSTEM_PALETTE = get_palette("qualitative")

# Cancer-system color map for SEX forest plots (root cancer-area buckets)
SEX_SYSTEM_COLOR = {
    "lung":       get_color("cancer_systems.lung"),
    "colorectal": get_color("cancer_systems.colorectal"),
    "myeloid":    get_color("cancer_systems.myeloid"),
    "biliary":    get_color("cancer_systems.biliary"),
}


# -----------------------------------------------------------------------------
# Feature spec
# -----------------------------------------------------------------------------

FEATURE_SPEC: Dict[str, dict] = {
    "race": {
        "csv_rel_path": Path(
            "tier7_posthoc_statistics/gene_stratified/race_by_gene/race_gene_significant.csv"
        ),
        "model_agnostic_numeric_cols": ("cramers_v",),
        "model_agnostic_count_cols": ("n_gene_actionable", "n_non_actionable"),
        "subdir": "race",
    },
    "sex": {
        "csv_rel_path": Path(
            "tier7_posthoc_statistics/gene_stratified/sex_by_gene/sex_gene_significant.csv"
        ),
        "model_agnostic_numeric_cols": ("odds_ratio", "or_ci_lower", "or_ci_upper"),
        "model_agnostic_count_cols": (
            "n_gene_actionable", "n_non_actionable",
            "gene_female", "gene_male",
            "non_actionable_female", "non_actionable_male",
        ),
        "subdir": "sex",
    },
}


# -----------------------------------------------------------------------------
# Loading and intersection (feature-aware)
# -----------------------------------------------------------------------------

def load_significant_hits(run_dir: Path, feature: str, pathway_label: str) -> pd.DataFrame:
    spec = FEATURE_SPEC[feature]
    csv_path = run_dir / spec["csv_rel_path"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"[{pathway_label}/{feature}] {spec['csv_rel_path'].name} not found at {csv_path}"
        )
    df = pd.read_csv(csv_path)
    required = {"ONCOTREE_CODE", "Hugo_Symbol", "gene_rank",
                "p_fdr", "significant_fdr"}
    required |= set(spec["model_agnostic_numeric_cols"])
    required |= set(spec["model_agnostic_count_cols"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{pathway_label}/{feature}] missing columns in {csv_path}: {missing}"
        )
    df = df[df["significant_fdr"] == True].copy()
    print(f"  [{pathway_label}/{feature}] loaded {len(df)} significant hits from {csv_path}")
    return df


def intersect_hits(rf: pd.DataFrame, lr: pd.DataFrame, feature: str) -> pd.DataFrame:
    spec = FEATURE_SPEC[feature]
    numeric_cols = list(spec["model_agnostic_numeric_cols"])
    count_cols = list(spec["model_agnostic_count_cols"])
    model_agnostic_cols = numeric_cols + count_cols

    # Per-pathway columns to keep + suffix
    base_cols = ["ONCOTREE_CODE", "Hugo_Symbol", "gene_rank",
                 "p_value", "p_fdr"] + model_agnostic_cols
    base_cols = [c for c in base_cols if c in rf.columns]

    rename_map_rf = {
        "gene_rank": "gene_rank_rf",
        "p_value": "p_value_rf",
        "p_fdr": "p_fdr_rf",
    }
    rename_map_lr = {
        "gene_rank": "gene_rank_lr",
        "p_value": "p_value_lr",
        "p_fdr": "p_fdr_lr",
    }
    for c in model_agnostic_cols:
        rename_map_rf[c] = f"{c}_rf"
        rename_map_lr[c] = f"{c}_lr"

    # Carry test_used through if present (feature='sex')
    extra_cols = [c for c in ("test_used", "warning") if c in rf.columns]
    rf_sub = rf[base_cols + extra_cols].rename(columns=rename_map_rf)
    lr_sub = lr[base_cols + extra_cols].rename(columns=rename_map_lr)
    if "test_used" in extra_cols:
        rf_sub = rf_sub.rename(columns={"test_used": "test_used_rf",
                                        "warning": "warning_rf"} if "warning" in extra_cols
                                        else {"test_used": "test_used_rf"})
        lr_sub = lr_sub.rename(columns={"test_used": "test_used_lr",
                                        "warning": "warning_lr"} if "warning" in extra_cols
                                        else {"test_used": "test_used_lr"})

    merged = rf_sub.merge(lr_sub, on=["ONCOTREE_CODE", "Hugo_Symbol"], how="inner")

    # Numeric model-agnostic columns: assert near-equal across pathways and collapse
    for col in numeric_cols:
        rf_vals = merged[f"{col}_rf"].to_numpy()
        lr_vals = merged[f"{col}_lr"].to_numpy()
        mismatched = ~np.isclose(rf_vals, lr_vals, rtol=1e-9, atol=1e-12)
        if mismatched.any():
            bad = merged[mismatched][["ONCOTREE_CODE", "Hugo_Symbol",
                                      f"{col}_rf", f"{col}_lr"]]
            raise AssertionError(
                f"[{feature}] {col} differs between RF and LR pathways for "
                f"{int(mismatched.sum())} hit(s) — likely a mapping mismatch:\n"
                f"{bad.to_string(index=False)}"
            )
        merged[col] = rf_vals

    # Count columns: assert exact equality and collapse
    for col in count_cols:
        rf_vals = merged[f"{col}_rf"].to_numpy()
        lr_vals = merged[f"{col}_lr"].to_numpy()
        mismatched = rf_vals != lr_vals
        if mismatched.any():
            bad = merged[mismatched][["ONCOTREE_CODE", "Hugo_Symbol",
                                      f"{col}_rf", f"{col}_lr"]]
            raise AssertionError(
                f"[{feature}] {col} differs between RF and LR pathways for "
                f"{int(mismatched.sum())} hit(s) — likely a mapping mismatch:\n"
                f"{bad.to_string(index=False)}"
            )
        merged[col] = rf_vals

    drop_cols = [f"{c}_rf" for c in model_agnostic_cols] + [f"{c}_lr" for c in model_agnostic_cols]
    merged = merged.drop(columns=drop_cols)
    merged["p_fdr_max"] = merged[["p_fdr_rf", "p_fdr_lr"]].max(axis=1)

    sort_col = numeric_cols[0]  # 'cramers_v' for race, 'odds_ratio' for sex
    merged = merged.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return merged


def split_by_tier(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Plain (non-prefixed) OncoTree codes — what the tier1-only publication pipeline
    # emits — count as tier1. Only explicit tier2_/tier3_ rollup codes (from the
    # external rollup-aware runs) go to the tier2_tier3 panel.
    is_t23 = df["ONCOTREE_CODE"].str.match(r"^tier[23]_")
    return {
        "tier1": df[~is_t23].copy(),
        "tier2_tier3": df[is_t23].copy(),
    }


def _tier_subset_label(ot_code: str) -> str:
    # Plain (non-prefixed) codes from the tier1-only publication pipeline are tier1.
    return "tier2_tier3" if ot_code.startswith(("tier2_", "tier3_")) else "tier1"


def _short_label(tier_prefixed_code: str, gene: str) -> str:
    pretty = (tier_prefixed_code
              .replace("tier1_", "T1·")
              .replace("tier2_", "T2·")
              .replace("tier3_", "T3·"))
    return f"{pretty} · {gene}"


# -----------------------------------------------------------------------------
# RACE: Contingency reconstruction (for Δpp small multiples)
# -----------------------------------------------------------------------------

def build_contingency(
    db_path: str,
    mapping_df: pd.DataFrame,
    tier_prefixed_code: str,
    gene: str,
) -> pd.DataFrame:
    if tier_prefixed_code.startswith(("tier1_", "tier2_", "tier3_")):
        # Rollup-aware run: expand the group code to its constituent OncoTree codes.
        ot_codes = get_oncotree_codes_for_group(tier_prefixed_code, mapping_df)
    else:
        # Plain OncoTree code (tier1-only publication run): the code IS the OncoTree
        # code, so no rollup mapping is needed (mapping_df may be None here).
        ot_codes = [tier_prefixed_code]
    if not ot_codes:
        raise ValueError(f"No OncoTree codes resolved for {tier_prefixed_code}")
    placeholders = ",".join("?" * len(ot_codes))

    conn = sqlite3.connect(db_path)
    try:
        gene_q = f"""
            SELECT DISTINCT PATIENT_ID, RACE
            FROM {PATIENT_TABLE}
            WHERE ONCOTREE_CODE IN ({placeholders})
              AND Hugo_Symbol = ?
              AND isOncoKbLevel1 = 1
        """
        gene_df = pd.read_sql_query(gene_q, conn, params=ot_codes + [gene])

        all_q = f"""
            SELECT PATIENT_ID, RACE, isOncoKbLevel1
            FROM {PATIENT_TABLE}
            WHERE ONCOTREE_CODE IN ({placeholders})
        """
        all_df = pd.read_sql_query(all_q, conn, params=ot_codes)
    finally:
        conn.close()

    all_df = all_df.drop_duplicates(subset=["PATIENT_ID"], keep="first")
    nonact_df = all_df[all_df["isOncoKbLevel1"] == 0]

    gene_df = gene_df[gene_df["RACE"].notna()]
    nonact_df = nonact_df[nonact_df["RACE"].notna()]

    g = gene_df["RACE"].value_counts().rename("gene_actionable")
    n = nonact_df["RACE"].value_counts().rename("non_actionable")
    tab = pd.concat([g, n], axis=1).fillna(0).astype(int)
    tab.index.name = "RACE"
    g_sum = tab["gene_actionable"].sum() or 1
    n_sum = tab["non_actionable"].sum() or 1
    tab["gene_actionable_pct"] = tab["gene_actionable"] / g_sum * 100
    tab["non_actionable_pct"] = tab["non_actionable"] / n_sum * 100
    tab["delta_pp"] = tab["gene_actionable_pct"] - tab["non_actionable_pct"]
    return tab.sort_values("non_actionable", ascending=False)


def build_long_format(
    intersection: pd.DataFrame,
    db_path: str,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[dict] = []
    for _, h in intersection.iterrows():
        tier = h["ONCOTREE_CODE"]
        gene = h["Hugo_Symbol"]
        tab = build_contingency(db_path, mapping_df, tier, gene)
        n_gene = int(tab["gene_actionable"].sum())
        n_non = int(tab["non_actionable"].sum())
        for race, r in tab.iterrows():
            rows.append({
                "ONCOTREE_CODE": tier,
                "Hugo_Symbol": gene,
                "label": _short_label(tier, gene),
                "race": race,
                "n_gene_actionable": int(r["gene_actionable"]),
                "n_non_actionable": int(r["non_actionable"]),
                "pct_gene_actionable": float(r["gene_actionable_pct"]),
                "pct_non_actionable": float(r["non_actionable_pct"]),
                "delta_pp": float(r["delta_pp"]),
                "n_gene_total": n_gene,
                "n_non_total": n_non,
                "cramers_v": float(h["cramers_v"]),
                "p_fdr_max": float(h["p_fdr_max"]),
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# RACE: Per-hit and summary figures
# -----------------------------------------------------------------------------

def render_race_per_hit(
    intersection: pd.DataFrame,
    output_dir: Path,
    db_path: str,
    long_format: pd.DataFrame,
) -> None:
    per_hit_dir = output_dir / "per_hit"
    per_hit_dir.mkdir(parents=True, exist_ok=True)

    for _, h in intersection.iterrows():
        tier = h["ONCOTREE_CODE"]
        gene = h["Hugo_Symbol"]
        cramers_v = float(h["cramers_v"])
        p_val = float(h["p_fdr_max"])

        result = plot_categorical_contingency(
            cancer_type=tier,
            column="RACE",
            cramers_v=cramers_v,
            p_val=p_val,
            output_dir=per_hit_dir,
            gene=gene,
            feature_type="demographic",
            db_path=db_path,
            file_prefix="gene_race_intersection",
        )

        d = long_format[
            (long_format["ONCOTREE_CODE"] == tier) &
            (long_format["Hugo_Symbol"] == gene)
        ]
        csv_path = per_hit_dir / f"gene_race_intersection_{tier}_{gene}_RACE_counts.csv"
        d_out = d[["race", "n_gene_actionable", "n_non_actionable",
                   "pct_gene_actionable", "pct_non_actionable", "delta_pp"]]
        d_out.to_csv(csv_path, index=False)

        if result is None:
            print(f"  [warn] no plot produced for {tier} / {gene}")


def _race_system_palette(intersection: pd.DataFrame) -> Dict[str, str]:
    keys = (intersection["ONCOTREE_CODE"]
            .str.replace(r"^tier\d+_", "", regex=True)
            .unique())
    palette = {}
    for i, k in enumerate(sorted(keys)):
        palette[k] = SYSTEM_PALETTE[i % len(SYSTEM_PALETTE)]
    return palette


def fig1_lollipop(intersection: pd.DataFrame, output_path: Path, title_suffix: str) -> None:
    if intersection.empty:
        print(f"  [skip] fig1 {title_suffix}: no hits")
        return
    df = intersection.sort_values("cramers_v", ascending=True).reset_index(drop=True)
    palette = _race_system_palette(df)
    df["system_key"] = df["ONCOTREE_CODE"].str.replace(r"^tier\d+_", "", regex=True)

    y = np.arange(len(df))
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.5 * len(df) + 1.5)),
                           constrained_layout=True)

    for i, row in df.iterrows():
        color = palette.get(row["system_key"], SYSTEM_COLOR_DEFAULT)
        ax.hlines(y=i, xmin=0, xmax=row["cramers_v"],
                  color=color, linewidth=2, alpha=0.7)
        ax.plot(row["cramers_v"], i, "o", color=color, markersize=8,
                markeredgecolor="white", markeredgewidth=0.8)
        neg_log_p = -np.log10(max(row["p_fdr_max"], 1e-300))
        n_total = int(row["n_gene_actionable"]) + int(row["n_non_actionable"])
        ax.text(row["cramers_v"] + 0.005, i,
                f"  -log10(p)={neg_log_p:5.1f}   N={n_total:,}",
                va="center", ha="left", fontsize=8, color="#444")

    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(r["ONCOTREE_CODE"], r["Hugo_Symbol"]) for _, r in df.iterrows()])
    ax.set_xlabel("Cramer's V")
    xmax = df["cramers_v"].max() * 1.45
    ax.set_xlim(0, xmax if xmax > 0 else 0.1)
    # In-figure title removed per figures/style/FIGURE_STYLE_GUIDE.md (-> caption).
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=c, markersize=9, label=k)
                      for k, c in palette.items()]
    ax.legend(handles=legend_handles, loc="lower right", title="OT group code",
              frameon=False, fontsize=8)

    for ext in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)


def fig2_diverging_dpp(
    intersection: pd.DataFrame,
    long_format: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
) -> None:
    if intersection.empty:
        print(f"  [skip] fig2 {title_suffix}: no hits")
        return
    hits = intersection.sort_values("cramers_v", ascending=False).reset_index(drop=True)
    sub = long_format[long_format["ONCOTREE_CODE"].isin(hits["ONCOTREE_CODE"]) &
                      long_format["Hugo_Symbol"].isin(hits["Hugo_Symbol"])]

    n_panels = len(hits)
    ncols = 3 if n_panels >= 3 else max(1, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    apply_publication_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_2d(axes).reshape(-1)

    xmax = max(sub["delta_pp"].abs().max() * 1.15, 1.0)
    pos_color = get_color("direction.positive")
    neg_color = get_color("direction.negative")

    for ax, (_, h) in zip(axes, hits.iterrows()):
        d = sub[(sub["ONCOTREE_CODE"] == h["ONCOTREE_CODE"]) &
                (sub["Hugo_Symbol"] == h["Hugo_Symbol"])].copy()
        d["race"] = pd.Categorical(d["race"], categories=RACE_ORDER, ordered=True)
        d = d.sort_values("race", ascending=False).reset_index(drop=True)

        colors = [pos_color if v >= 0 else neg_color for v in d["delta_pp"]]
        ax.barh(d["race"].astype(str), d["delta_pp"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-xmax, xmax)
        n_g = int(d["n_gene_total"].iloc[0]) if len(d) else 0
        n_n = int(d["n_non_total"].iloc[0]) if len(d) else 0
        ax.set_title(
            f"{_short_label(h['ONCOTREE_CODE'], h['Hugo_Symbol'])}    "
            f"V={h['cramers_v']:.3f}\n"
            f"n_actionable={n_g:,}   n_control={n_n:,}",
            fontsize=10,
        )
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        for j, v in enumerate(d["delta_pp"]):
            xtxt = v + (0.4 if v >= 0 else -0.4)
            ha = "left" if v >= 0 else "right"
            ax.text(xtxt, j, f"{v:+.1f}", va="center", ha=ha, fontsize=8, color="#222")

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    # In-figure suptitle removed per figures/style/FIGURE_STYLE_GUIDE.md (-> caption).
    fig.supxlabel("Δ percentage points", fontsize=11)

    for ext in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# SEX: per-hit counts + forest plot
# -----------------------------------------------------------------------------

def _sex_cancer_system(ot_code: str) -> str:
    code = ot_code.replace("tier1_", "").replace("tier2_", "").replace("tier3_", "")
    if code in {"LUAD", "NSCLC", "LUNG"}:
        return "lung"
    if code in {"COAD", "COADREAD", "BOWEL", "READ"}:
        return "colorectal"
    if code in {"AML", "MYELOID"}:
        return "myeloid"
    if code in {"IHCH", "BILIARY_TRACT", "CHOL"}:
        return "biliary"
    return "other"


def render_sex_per_hit_counts(intersection: pd.DataFrame, output_dir: Path) -> None:
    per_hit_dir = output_dir / "per_hit_counts"
    per_hit_dir.mkdir(parents=True, exist_ok=True)
    for _, h in intersection.iterrows():
        tier = h["ONCOTREE_CODE"]
        gene = h["Hugo_Symbol"]
        gf = int(h["gene_female"])
        gm = int(h["gene_male"])
        nf = int(h["non_actionable_female"])
        nm = int(h["non_actionable_male"])
        gf_pct = gf / (gf + gm) * 100 if (gf + gm) else float("nan")
        nf_pct = nf / (nf + nm) * 100 if (nf + nm) else float("nan")
        d = pd.DataFrame({
            "group": ["gene_actionable", "non_actionable"],
            "n_female": [gf, nf],
            "n_male": [gm, nm],
            "n_total": [gf + gm, nf + nm],
            "pct_female": [gf_pct, nf_pct],
            "delta_pp_female": [gf_pct - nf_pct, 0.0],
        })
        d.to_csv(per_hit_dir / f"gene_sex_intersection_{tier}_{gene}_counts.csv", index=False)


def render_sex_forest(
    intersection: pd.DataFrame,
    output_path: Path,
    title_suffix: str,
) -> None:
    if intersection.empty:
        print(f"  [skip] sex forest {title_suffix}: no hits")
        return

    df = intersection.copy()
    df["label"] = df.apply(lambda r: _short_label(r["ONCOTREE_CODE"], r["Hugo_Symbol"])
                                      + f"  (rk {int(r['gene_rank_rf'])})", axis=1)
    df["system"] = df["ONCOTREE_CODE"].apply(_sex_cancer_system)
    df["color"] = df["system"].map(SEX_SYSTEM_COLOR).fillna(SYSTEM_COLOR_DEFAULT)
    df = df.sort_values("odds_ratio", ascending=True).reset_index(drop=True)

    n = len(df)
    fig_h = max(3.5, 0.42 * n + 1.6)
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    y = np.arange(n)
    ors = df["odds_ratio"].to_numpy()
    lo = df["or_ci_lower"].to_numpy()
    hi = df["or_ci_upper"].to_numpy()
    colors = df["color"].to_numpy()

    for yi, lo_i, hi_i, c in zip(y, lo, hi, colors):
        ax.hlines(yi, lo_i, hi_i, color=c, linewidth=1.8, alpha=0.85)
    ax.scatter(ors, y, c=colors, s=42, zorder=3, edgecolors="black", linewidths=0.4)

    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xscale("log")
    ax.set_xticks([0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Odds ratio (Female / Male) for gene-actionable status — 95% Wald CI on log(OR)")

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_ylim(-0.7, n - 0.3)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markeredgecolor="black", markersize=8, label=label.title())
        for label, c in SEX_SYSTEM_COLOR.items() if (df["system"] == label).any()
    ]
    if handles:
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)

    # In-figure title removed per figures/style/FIGURE_STYLE_GUIDE.md (-> caption).
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Per-feature pipelines
# -----------------------------------------------------------------------------

def run_race_pipeline(
    rf_run: Path,
    lr_run: Path,
    output_dir: Path,
    db_path: str,
) -> Dict[str, int]:
    feature = "race"
    feature_dir = output_dir / FEATURE_SPEC[feature]["subdir"]
    feature_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== RACE intersection ===")
    rf_hits = load_significant_hits(rf_run, feature, "RF")
    lr_hits = load_significant_hits(lr_run, feature, "LR")
    intersection = intersect_hits(rf_hits, lr_hits, feature)
    print(f"  Intersection size: {len(intersection)} race hits")

    if intersection.empty:
        intersection.to_csv(feature_dir / "intersection_summary.csv", index=False)
        return {"intersection": 0, "tier1": 0, "tier2_tier3": 0}

    by_tier = split_by_tier(intersection)
    intersection_with_tier = intersection.copy()
    intersection_with_tier["tier_subset"] = intersection_with_tier["ONCOTREE_CODE"].apply(_tier_subset_label)
    intersection_with_tier.to_csv(feature_dir / "intersection_summary.csv", index=False)

    mapping_path = resolve_mapping(rf_run, lr_run)
    mapping_df = (load_patient_to_group_mapping(mapping_file=mapping_path, verbose=False)
                  if mapping_path is not None else None)

    long_format = build_long_format(intersection, db_path, mapping_df)
    long_format.to_csv(feature_dir / "intersection_long_format.csv", index=False)

    print("  Rendering per-hit contingency plots...")
    render_race_per_hit(intersection, feature_dir, db_path, long_format)

    print("  Rendering summary figures...")
    for tier_subset, sub in by_tier.items():
        print(f"    tier_subset={tier_subset}  n={len(sub)}")
        fig1_lollipop(sub,
                      feature_dir / f"fig1_lollipop_effect_sizes_{tier_subset}",
                      title_suffix=tier_subset)
        sub_long = long_format[long_format["ONCOTREE_CODE"].isin(sub["ONCOTREE_CODE"])]
        fig2_diverging_dpp(sub, sub_long,
                           feature_dir / f"fig2_diverging_dpp_small_multiples_{tier_subset}",
                           title_suffix=tier_subset)

    return {
        "intersection": len(intersection),
        "tier1": len(by_tier["tier1"]),
        "tier2_tier3": len(by_tier["tier2_tier3"]),
    }


def run_sex_pipeline(
    rf_run: Path,
    lr_run: Path,
    output_dir: Path,
) -> Dict[str, int]:
    feature = "sex"
    feature_dir = output_dir / FEATURE_SPEC[feature]["subdir"]
    feature_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== SEX intersection ===")
    rf_hits = load_significant_hits(rf_run, feature, "RF")
    lr_hits = load_significant_hits(lr_run, feature, "LR")
    intersection = intersect_hits(rf_hits, lr_hits, feature)
    print(f"  Intersection size: {len(intersection)} sex hits")

    if intersection.empty:
        intersection.to_csv(feature_dir / "intersection_summary.csv", index=False)
        return {"intersection": 0, "tier1": 0, "tier2_tier3": 0}

    by_tier = split_by_tier(intersection)
    intersection_with_tier = intersection.copy()
    intersection_with_tier["tier_subset"] = intersection_with_tier["ONCOTREE_CODE"].apply(_tier_subset_label)
    intersection_with_tier.to_csv(feature_dir / "intersection_summary.csv", index=False)

    print("  Writing per-hit counts CSVs...")
    render_sex_per_hit_counts(intersection, feature_dir)

    print("  Rendering forest figures...")
    render_sex_forest(intersection, feature_dir / "sex_or_forest_all", title_suffix="all")
    for tier_subset, sub in by_tier.items():
        print(f"    tier_subset={tier_subset}  n={len(sub)}")
        render_sex_forest(sub, feature_dir / f"sex_or_forest_{tier_subset}",
                          title_suffix=tier_subset)

    return {
        "intersection": len(intersection),
        "tier1": len(by_tier["tier1"]),
        "tier2_tier3": len(by_tier["tier2_tier3"]),
    }


# -----------------------------------------------------------------------------
# Mapping resolution + README
# -----------------------------------------------------------------------------

def resolve_mapping(rf_run_dir: Path, lr_run_dir: Path) -> Optional[Path]:
    for d in (rf_run_dir, lr_run_dir):
        candidate = d / MAPPING_REL_PATH
        if candidate.exists():
            return candidate

    from multi_tier_utils import get_patient_to_group_mapping_file
    fallback = get_patient_to_group_mapping_file()
    if fallback is not None and fallback.exists():
        print(f"  [info] mapping not in run dirs; falling back to {fallback}")
        return fallback

    runs_dir = REPO_ROOT / "runs"
    candidates = sorted(
        runs_dir.glob("*/tier2_data_setup/multi_tier_groupings/patient_to_group_mapping.csv")
    )
    if candidates:
        chosen = candidates[-1]
        print(f"  [info] mapping not in run dirs; using most-recent {chosen}")
        return chosen

    # No multi_tier_groupings mapping anywhere — expected for a tier1-only publication
    # run, whose plain OncoTree codes need no rollup expansion. The race contingency
    # handles plain codes directly (build_contingency), so return None rather than fail.
    print("  [info] no multi_tier_groupings mapping found; assuming plain tier1-only "
          "codes (race contingency queries OncoTree codes directly).")
    return None


def write_readme(
    output_dir: Path,
    rf_run_dir: Path,
    lr_run_dir: Path,
    db_path: str,
    feature_counts: Dict[str, Dict[str, int]],
) -> None:
    lines = [
        "# RF / LR cross-pathway hit intersection",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        f"- RF run dir: `{rf_run_dir}`",
        f"- LR run dir: `{lr_run_dir}`",
        f"- DB (race only): `{db_path}`",
        "",
        "## Intersection sizes",
    ]
    for feature, counts in feature_counts.items():
        lines.append(
            f"- **{feature}**: {counts['intersection']} hits "
            f"(tier1={counts['tier1']}, tier2_tier3={counts['tier2_tier3']})"
        )

    lines += [
        "",
        "## Files",
        "### race/",
        "- `intersection_summary.csv` — one row per intersected RACE hit, both pathways' Cramer's V and FDR p",
        "- `intersection_long_format.csv` — long-format counts per (hit, race) for plotting",
        "- `per_hit/` — one contingency plot per hit (existing TIER 7 style) plus a counts CSV",
        "- `fig1_lollipop_effect_sizes_<tier>.{pdf,png}` — effect-size ranking, split by tier",
        "- `fig2_diverging_dpp_small_multiples_<tier>.{pdf,png}` — Δpp small multiples, split by tier",
        "",
        "### sex/",
        "- `intersection_summary.csv` — one row per intersected SEX hit (OR + 95% CI + both p_fdr + F/M counts)",
        "- `per_hit_counts/` — one F/M counts CSV per hit",
        "- `sex_or_forest_{all,tier1,tier2_tier3}.{pdf,png}` — Female/Male OR forest plot with 95% Wald CI on log(OR)",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rf-run-dir", type=Path, required=True,
                        help="RF pipeline run directory containing TIER 7 outputs")
    parser.add_argument("--lr-run-dir", type=Path, required=True,
                        help="LR pipeline run directory containing TIER 7 outputs")
    parser.add_argument("--features", nargs="+",
                        choices=list(FEATURE_SPEC.keys()),
                        default=list(FEATURE_SPEC.keys()),
                        help="Which feature(s) to process (default: all)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (defaults to runs/rf_lr_hit_intersection/run_<ts>/)")
    parser.add_argument("--db-path", type=str, default=None,
                        help="SQLite DB path (defaults to BALDER_DB_PATH or db_config default). "
                             "Required only when --features includes 'race'.")
    args = parser.parse_args()

    rf_run = args.rf_run_dir.resolve()
    lr_run = args.lr_run_dir.resolve()
    features = list(args.features)

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = REPO_ROOT / "runs" / "rf_lr_hit_intersection" / f"run_{ts}"
    else:
        out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    db_path = None
    if "race" in features:
        db_path = args.db_path or get_default_db_path()
        if not db_path or not Path(db_path).exists():
            print(f"[error] SQLite DB not found: {db_path}", file=sys.stderr)
            return 1

    print(f"RF run:     {rf_run}")
    print(f"LR run:     {lr_run}")
    print(f"Features:   {features}")
    print(f"Output dir: {out}")
    if db_path:
        print(f"DB:         {db_path}")

    feature_counts: Dict[str, Dict[str, int]] = {}
    if "race" in features:
        feature_counts["race"] = run_race_pipeline(rf_run, lr_run, out, db_path)
    if "sex" in features:
        feature_counts["sex"] = run_sex_pipeline(rf_run, lr_run, out)

    write_readme(out, rf_run, lr_run, db_path or "n/a", feature_counts)
    print(f"\nDone. Outputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
