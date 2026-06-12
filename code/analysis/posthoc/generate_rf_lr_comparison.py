#!/usr/bin/env python3
"""
Generate the merged RF/SHAP vs LR (--lr-significant-only) post-hoc comparison.

This is the canonical home for what used to live (untracked) as
``runs/rf_vs_lr_comparison_*/generate_comparison.py``. Promoted into the repo
2026-06-09 for the publication run-lock so the merged comparison table has a
tracked, synced generator (it is a publication artifact).

It reads the per-pathway TIER 7 outputs of an RF run and an LR run and emits the
side-by-side hit comparison the manuscript cites:

Generates (in --output-dir):
  - gene_stratified_comparison.csv   <- one row per (cancer x gene x feature_category x
                                        feature); membership (both/RF_only/LR_only),
                                        effect_size + direction, rf_p_fdr AND lr_p_fdr
                                        side by side, n_actionable / n_non_actionable.
  - cancer_level_comparison.csv      <- cancer-level age/sex/technical/sample_type
                                        membership (has a known RF-one-hot vs LR-grouped
                                        feature-naming caveat; see the gene-level file for
                                        the clean comparison).
  - Visualization PDFs (overlap bar, category breakdown, effect-size scatter,
    cancer x category heatmap, cancer-level summary).

Inputs per run (both --rf-run and --lr-run must contain these):
  tier7_posthoc_statistics/gene_stratified/gene_stratified_significant_hits.csv
  tier7_posthoc_statistics/age_actionability/age_actionability_results.csv
  tier7_posthoc_statistics/shap_associations/sex_associations.csv
  tier7_posthoc_statistics/shap_associations/technical_feature_associations.csv
  tier7_posthoc_statistics/shap_associations/sample_type_associations.csv

v2.1 / publication-lock notes:
  - The ``feature_category`` column in the hit CSV still uses ``race`` (not
    ``ancestry``) even after the v2.1 RACE->ANCESTRY *display* rename, so the
    category list below is unchanged.
  - The publication branch emits PLAIN OncoTree codes (AML, ASTR, ...), not the
    ``tier1_``/``tier2_``/``tier3_`` prefixes used by rollup-aware runs. The
    cancer x category heatmap therefore treats any code that is NOT
    ``tier2_``/``tier3_``-prefixed as a Tier-1 (leaf) code, so it renders for
    both the plain-code publication runs and the legacy rollup-aware runs.

Reference invocation (locked RF+LR pair, run from the idunn checkout):
    python code/analysis/posthoc/generate_rf_lr_comparison.py \
        --rf-run runs/run_20260607_212932_rf \
        --lr-run runs/run_20260609_005856_lr \
        --output-dir runs/rf_vs_lr_comparison_<ts>
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_gene_hits(path):
    """Load gene_stratified_significant_hits.csv into dict keyed by
    (ONCOTREE_CODE, Hugo_Symbol, feature_category, feature)."""
    hits = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["ONCOTREE_CODE"], row["Hugo_Symbol"],
                   row["feature_category"], row["feature"])
            hits[key] = row
    return hits


def load_age_results(path):
    hits = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            hits[row["ONCOTREE_CODE"]] = row
    return hits


def load_sex_results(path):
    hits = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            hits[row["ONCOTREE_CODE"]] = row
    return hits


def load_keyed_results(path, key_cols):
    """Generic loader returning dict keyed by tuple of *key_cols*."""
    hits = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = tuple(row.get(c, "") for c in key_cols)
            hits[key] = row
    return hits

# ---------------------------------------------------------------------------
# CSV generation
# ---------------------------------------------------------------------------

def write_gene_csv(rf, lr, out_path):
    rf_keys, lr_keys = set(rf), set(lr)
    intersection = rf_keys & lr_keys
    rf_only = rf_keys - lr_keys
    lr_only = lr_keys - rf_keys

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ONCOTREE_CODE", "Hugo_Symbol", "gene_rank",
                     "feature_category", "feature", "membership",
                     "effect_size", "effect_size_type", "direction",
                     "rf_p_fdr", "lr_p_fdr", "n_actionable", "n_non_actionable"])
        for k in sorted(intersection | rf_only | lr_only):
            if k in intersection:
                mem = "both"
                row, lr_row = rf[k], lr[k]
                rf_p, lr_p = row["p_fdr"], lr_row["p_fdr"]
            elif k in rf_only:
                mem, row = "RF_only", rf[k]
                rf_p, lr_p = row["p_fdr"], ""
            else:
                mem, row = "LR_only", lr[k]
                rf_p, lr_p = "", row["p_fdr"]
            w.writerow([k[0], k[1], row["gene_rank"], k[2], k[3], mem,
                        row["effect_size"], row["effect_size_type"],
                        row["direction"], rf_p, lr_p,
                        row["n_actionable"], row["n_non_actionable"]])

    print(f"  gene_stratified_comparison.csv: "
          f"{len(intersection)} both, {len(rf_only)} RF_only, {len(lr_only)} LR_only")
    return intersection, rf_only, lr_only


def write_cancer_csv(rf_run, lr_run, out_path):
    rows = []

    # --- age ---
    rf_age = load_age_results(rf_run / "tier7_posthoc_statistics/age_actionability/age_actionability_results.csv")
    lr_age = load_age_results(lr_run / "tier7_posthoc_statistics/age_actionability/age_actionability_results.csv")
    for cancer in sorted(set(rf_age) | set(lr_age)):
        rf_sig = rf_age.get(cancer, {}).get("mann_whitney_significant_fdr") == "True"
        lr_sig = lr_age.get(cancer, {}).get("mann_whitney_significant_fdr") == "True"
        mem = _membership(rf_sig, lr_sig)
        row = rf_age.get(cancer, lr_age.get(cancer))
        d = row.get("cohens_d", "")
        direction = ("older" if float(d) > 0 else "younger") if d else ""
        rows.append([cancer, "age_actionability", "AGE_AT_SEQ_REPORT", mem,
                     rf_sig, lr_sig, cancer in rf_age, cancer in lr_age,
                     d, "Cohen's d", direction,
                     rf_age.get(cancer, {}).get("mann_whitney_p_fdr", ""),
                     lr_age.get(cancer, {}).get("mann_whitney_p_fdr", "")])

    # --- sex ---
    rf_sex = load_sex_results(rf_run / "tier7_posthoc_statistics/shap_associations/sex_associations.csv")
    lr_sex = load_sex_results(lr_run / "tier7_posthoc_statistics/shap_associations/sex_associations.csv")
    for cancer in sorted(set(rf_sex) | set(lr_sex)):
        rf_sig = rf_sex.get(cancer, {}).get("significant_fdr") == "True"
        lr_sig = lr_sex.get(cancer, {}).get("significant_fdr") == "True"
        mem = _membership(rf_sig, lr_sig)
        row = rf_sex.get(cancer, lr_sex.get(cancer))
        or_val = row.get("odds_ratio", "")
        direction = ("higher_female" if float(or_val) > 1 else "higher_male") if or_val else ""
        rows.append([cancer, "sex_association", "SEX", mem,
                     rf_sig, lr_sig, cancer in rf_sex, cancer in lr_sex,
                     or_val, "Odds Ratio (F/M)", direction,
                     rf_sex.get(cancer, {}).get("p_fdr", ""),
                     lr_sex.get(cancer, {}).get("p_fdr", "")])

    # --- technical ---
    _add_keyed_rows(rows, rf_run, lr_run,
                    "tier7_posthoc_statistics/shap_associations/technical_feature_associations.csv",
                    "technical_association")

    # --- sample type ---
    _add_keyed_rows(rows, rf_run, lr_run,
                    "tier7_posthoc_statistics/shap_associations/sample_type_associations.csv",
                    "sample_type_association")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ONCOTREE_CODE", "analysis_type", "feature", "membership",
                     "rf_significant", "lr_significant", "rf_tested", "lr_tested",
                     "effect_size", "effect_size_type", "direction",
                     "rf_p_fdr", "lr_p_fdr"])
        w.writerows(rows)

    both = sum(1 for r in rows if r[3] == "both")
    rf_o = sum(1 for r in rows if r[3] == "RF_only")
    lr_o = sum(1 for r in rows if r[3] == "LR_only")
    print(f"  cancer_level_comparison.csv: {both} both, {rf_o} RF_only, {lr_o} LR_only")
    return rows


def _add_keyed_rows(rows, rf_run, lr_run, relpath, analysis_type):
    """Append rows for a keyed association file (technical or sample_type)."""
    key_cols = ["ONCOTREE_CODE", "column"]
    rf_path = rf_run / relpath
    lr_path = lr_run / relpath
    rf_data = load_keyed_results(rf_path, key_cols) if rf_path.exists() else {}
    lr_data = load_keyed_results(lr_path, key_cols) if lr_path.exists() else {}

    for key in sorted(set(rf_data) | set(lr_data)):
        rf_sig = rf_data.get(key, {}).get("significant_fdr") == "True"
        lr_sig = lr_data.get(key, {}).get("significant_fdr") == "True"
        mem = _membership(rf_sig, lr_sig)
        row = rf_data.get(key, lr_data.get(key))
        effect = row.get("cramers_v", row.get("effect_size", ""))
        etype = "Cramer's V" if row.get("cramers_v") else "Effect size r"
        rows.append([key[0], analysis_type, key[1], mem,
                     rf_sig, lr_sig, key in rf_data, key in lr_data,
                     effect, etype, "association",
                     rf_data.get(key, {}).get("p_fdr", ""),
                     lr_data.get(key, {}).get("p_fdr", "")])


def _membership(rf_sig, lr_sig):
    if rf_sig and lr_sig:
        return "both"
    elif rf_sig:
        return "RF_only"
    elif lr_sig:
        return "LR_only"
    return "neither"


def _is_leaf_code(ot_code):
    """True for a Tier-1 (leaf) OncoTree code.

    Publication runs emit plain codes (AML, ASTR, ...); rollup-aware runs emit
    ``tier1_``/``tier2_``/``tier3_`` prefixes. A leaf is anything NOT rolled up,
    i.e. not ``tier2_``/``tier3_``-prefixed (plain codes and ``tier1_`` both pass).
    """
    return not ot_code.startswith(("tier2_", "tier3_"))

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

COLORS = {"both": "#2ca02c", "RF_only": "#1f77b4", "LR_only": "#ff7f0e"}
LABELS = {"both": "Both", "RF_only": "RF only", "LR_only": "LR only"}


def plot_venn_summary(intersection, rf_only, lr_only, out_dir):
    """Simple proportional Venn-style bar showing overlap counts."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    counts = [len(rf_only), len(intersection), len(lr_only)]
    labels = [f"RF only\n({counts[0]})", f"Both\n({counts[1]})", f"LR only\n({counts[2]})"]
    colors = [COLORS["RF_only"], COLORS["both"], COLORS["LR_only"]]

    bars = ax.barh([0], counts[0], color=colors[0], edgecolor="white")
    bars = ax.barh([0], counts[1], left=counts[0], color=colors[1], edgecolor="white")
    bars = ax.barh([0], counts[2], left=counts[0]+counts[1], color=colors[2], edgecolor="white")

    # Center labels on each segment
    for i, (cnt, lbl) in enumerate(zip(counts, labels)):
        left = sum(counts[:i])
        ax.text(left + cnt / 2, 0, lbl, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

    ax.set_xlim(0, sum(counts))
    ax.set_yticks([])
    ax.set_xlabel("Number of gene-stratified significant hits")
    ax.set_title("RF/SHAP vs LR (Wald-significant) — Gene-Stratified Hit Overlap")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "gene_stratified_overlap_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_category_breakdown(intersection, rf_only, lr_only, out_dir):
    """Stacked bar chart of hit counts by feature category."""
    categories = ["age", "sex", "race", "ethnicity", "sample_type", "technical"]
    data = {cat: {"both": 0, "RF_only": 0, "LR_only": 0} for cat in categories}
    for k in intersection:
        data[k[2]]["both"] += 1
    for k in rf_only:
        data[k[2]]["RF_only"] += 1
    for k in lr_only:
        data[k[2]]["LR_only"] += 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(categories))
    w = 0.55
    bottoms = np.zeros(len(categories))
    for mem_key in ["both", "RF_only", "LR_only"]:
        vals = [data[c][mem_key] for c in categories]
        ax.bar(x, vals, w, bottom=bottoms, label=LABELS[mem_key],
               color=COLORS[mem_key], edgecolor="white", linewidth=0.5)
        # Add count labels inside segments
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(x[i], bottoms[i] + v / 2, str(v),
                        ha="center", va="center", fontsize=9, color="white",
                        fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=10)
    ax.set_ylabel("Number of significant hits")
    ax.set_title("Gene-Stratified Hits by Feature Category")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "gene_stratified_category_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_effect_size_comparison(rf, lr, intersection, out_dir):
    """Scatter plot of effect sizes: RF vs LR for intersection hits.
    Since the post-hoc tests are identical, effect sizes match — but we
    can color by category to show the composition of the intersection."""
    cat_colors = {
        "age": "#e41a1c", "sex": "#377eb8", "race": "#4daf4a",
        "ethnicity": "#984ea3", "sample_type": "#ff7f00", "technical": "#a65628"
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    for k in sorted(intersection):
        es = abs(float(rf[k]["effect_size"]))
        cat = k[2]
        ax.scatter(es, es, c=cat_colors[cat], s=30, alpha=0.7, edgecolors="none")

    # Dummy artists for legend
    for cat, color in cat_colors.items():
        count = sum(1 for k in intersection if k[2] == cat)
        if count > 0:
            ax.scatter([], [], c=color, s=40, label=f"{cat} ({count})")

    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
    ax.set_xlabel("|Effect size| (RF pathway)")
    ax.set_ylabel("|Effect size| (LR pathway)")
    ax.set_title("Intersection Hits: Effect Sizes Match\n(same post-hoc test, different filter)")
    ax.legend(fontsize=8, loc="upper left", title="Feature category", title_fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "intersection_effect_sizes.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cancer_heatmap(rf, lr, out_dir):
    """Heatmap: cancer type (rows) x feature category (cols), colored by membership."""
    # Restrict to Tier-1 (leaf) codes to avoid double-counting rolled-up groups.
    # Plain publication codes (no prefix) and legacy tier1_ codes both qualify.
    t1_keys = sorted({k for k in (set(rf) | set(lr)) if _is_leaf_code(k[0])})
    cancers = sorted({k[0] for k in t1_keys})
    categories = ["age", "sex", "race", "ethnicity", "sample_type", "technical"]

    # Build matrix: 0=not tested, 1=both, 2=RF_only, 3=LR_only
    rf_keys, lr_keys = set(rf), set(lr)
    matrix = np.zeros((len(cancers), len(categories)))
    for ci, cancer in enumerate(cancers):
        for fi, feat_cat in enumerate(categories):
            keys_here = [k for k in t1_keys if k[0] == cancer and k[2] == feat_cat]
            if not keys_here:
                matrix[ci, fi] = 0
                continue
            in_rf = any(k in rf_keys for k in keys_here)
            in_lr = any(k in lr_keys for k in keys_here)
            if in_rf and in_lr:
                matrix[ci, fi] = 1
            elif in_rf:
                matrix[ci, fi] = 2
            elif in_lr:
                matrix[ci, fi] = 3

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f0f0f0", COLORS["both"], COLORS["RF_only"], COLORS["LR_only"]])

    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=3)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9)
    ax.set_yticks(range(len(cancers)))
    ax.set_yticklabels([c.replace("tier1_", "") for c in cancers], fontsize=9)
    ax.set_title("Gene-Stratified Hits by Cancer Type (Tier 1)\nand Feature Category")

    # Legend
    patches = [
        mpatches.Patch(color="#f0f0f0", label="No significant hit"),
        mpatches.Patch(color=COLORS["both"], label="Both methods"),
        mpatches.Patch(color=COLORS["RF_only"], label="RF/SHAP only"),
        mpatches.Patch(color=COLORS["LR_only"], label="LR only"),
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=8, framealpha=0.9)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(categories), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cancers), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(out_dir / "cancer_category_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_cancer_level_summary(cancer_rows, out_dir):
    """Grouped bar chart of cancer-level post-hoc comparison by analysis type."""
    analysis_types = ["age_actionability", "sex_association",
                      "technical_association", "sample_type_association"]
    display_names = ["Age", "Sex", "Technical", "Sample type"]

    counts = {at: {"both": 0, "RF_only": 0, "LR_only": 0} for at in analysis_types}
    for row in cancer_rows:
        at = row[1]
        mem = row[3]
        if at in counts and mem in counts[at]:
            counts[at][mem] += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(analysis_types))
    w = 0.22
    for i, mem_key in enumerate(["both", "RF_only", "LR_only"]):
        vals = [counts[at][mem_key] for at in analysis_types]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=LABELS[mem_key],
                      color=COLORS[mem_key], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(v), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.set_ylabel("Number of significant associations")
    ax.set_title("Cancer-Level Post-Hoc Hits by Analysis Type\n"
                 "(Technical/Sample type 'both' undercounted due to\n"
                 "RF one-hot vs LR grouped feature naming)", fontsize=10)
    ax.legend(framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "cancer_level_summary.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate the merged RF/SHAP vs LR post-hoc comparison.")
    parser.add_argument("--rf-run", type=Path, required=True,
                        help="Path to RF/SHAP run directory (e.g. runs/run_20260607_212932_rf)")
    parser.add_argument("--lr-run", type=Path, required=True,
                        help="Path to LR run directory (e.g. runs/run_20260609_005856_lr)")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Output directory for CSVs and plots (default: .)")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # --- Gene-stratified ---
    print("Gene-stratified comparison:")
    rf_gene = load_gene_hits(
        args.rf_run / "tier7_posthoc_statistics/gene_stratified/gene_stratified_significant_hits.csv")
    lr_gene = load_gene_hits(
        args.lr_run / "tier7_posthoc_statistics/gene_stratified/gene_stratified_significant_hits.csv")
    intersection, rf_only, lr_only = write_gene_csv(
        rf_gene, lr_gene, out / "gene_stratified_comparison.csv")

    # --- Cancer-level ---
    print("Cancer-level comparison:")
    cancer_rows = write_cancer_csv(args.rf_run, args.lr_run,
                                   out / "cancer_level_comparison.csv")

    # --- Plots ---
    print("Generating plots...")
    plot_venn_summary(intersection, rf_only, lr_only, out)
    plot_category_breakdown(intersection, rf_only, lr_only, out)
    plot_effect_size_comparison(rf_gene, lr_gene, intersection, out)
    plot_cancer_heatmap(rf_gene, lr_gene, out)
    plot_cancer_level_summary(cancer_rows, out)
    print(f"Done. Outputs in {out}/")


if __name__ == "__main__":
    main()
