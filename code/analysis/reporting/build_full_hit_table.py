#!/usr/bin/env python3
"""Build the full gene-stratified hit table for the supplement (all tested associations).

The published hit list carries only rows that cleared FDR, so a reader has no denominator.
This emits every tested cancer x gene x feature row from both pathways, with effect size,
raw p, BH q and a significance flag, merged side by side. The RF and LR denominators are not
comparable: RF is gated on the top-5 grouped-SHAP ranking, LR on Wald-FDR significance, so
LR's higher hit rate reflects its gate. The significant subset is checked against the run's
own `gene_stratified_significant_hits.csv`; any disagreement exits 1.

Usage:
  python build_full_hit_table.py --rf-run runs/<rf_run> --lr-run runs/<lr_run> \
      --output-dir <dir> [--lit-review-csv FILE [FILE ...]]

`--lit-review-csv` adds `lit_status` and `lit_references` for cross-pathway demographic hits
(other rows are blank). A curated row that is not a both-pathway hit, or a both-pathway
demographic hit without curation, exits 1.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# One row per feature family: (subdir, filename, category, implicit feature name).
# `feature_col=None` means the family tests a single feature and does not name it.
FAMILIES = [
    ("age_by_gene", "age_gene_results.csv", "age", "AGE_AT_SEQ_REPORT"),
    ("sex_by_gene", "sex_gene_results.csv", "sex", "SEX"),
    ("race_by_gene", "race_gene_results.csv", "race", None),
    ("ethnicity_by_gene", "ethnicity_gene_results.csv", "ethnicity", None),
    ("sample_type_by_gene", "sample_type_gene_results.csv", "sample_type", None),
    ("technical_by_gene", "technical_gene_results.csv", "technical", None),
]

OUT_COLS = [
    "ONCOTREE_CODE", "Hugo_Symbol", "gene_rank", "feature_category", "feature",
    "effect_size", "effect_size_type", "direction", "p_value", "p_fdr", "significant_fdr",
    "n_actionable", "n_non_actionable", "warning",
]


def _normalize(df: pd.DataFrame, category: str, implicit_feature) -> pd.DataFrame:
    """Map one family's native schema onto the shared hit-row convention."""
    out = pd.DataFrame(index=df.index)
    out["ONCOTREE_CODE"] = df["ONCOTREE_CODE"]
    out["Hugo_Symbol"] = df["Hugo_Symbol"]
    out["gene_rank"] = df.get("gene_rank", np.nan)
    out["feature_category"] = category
    out["feature"] = implicit_feature if implicit_feature else df.get("feature", "unknown")

    if category == "age":
        d = df.get("cohens_d", pd.Series(np.nan, index=df.index))
        out["effect_size"] = d
        out["effect_size_type"] = "Cohen's d"
        out["direction"] = np.where(d.fillna(0) < 0, "younger", "older")
        out["p_value"] = df.get("mann_whitney_p", np.nan)
        out["n_actionable"] = df.get("n_actionable", np.nan)
    elif category == "sex":
        orv = df.get("odds_ratio", pd.Series(np.nan, index=df.index))
        out["effect_size"] = orv
        out["effect_size_type"] = "Odds Ratio (F/M)"
        out["direction"] = np.where(orv.fillna(1) > 1, "higher_female", "higher_male")
        out["p_value"] = df.get("p_value", np.nan)
        out["n_actionable"] = df.get("n_gene_actionable", np.nan)
    elif category == "technical":
        # number_of_genes is numeric (Mann-Whitney -> r); the rest are categorical (chi2 -> V)
        is_numeric = df.get("feature", "") == "number_of_genes"
        out["effect_size"] = np.where(is_numeric, df.get("effect_size_r", np.nan),
                                      df.get("cramers_v", np.nan))
        out["effect_size_type"] = np.where(is_numeric, "Effect size r", "Cramer's V")
        out["direction"] = "association"
        out["p_value"] = df.get("p_value", np.nan)
        out["n_actionable"] = df.get("n_gene_actionable", np.nan)
    else:  # race / ethnicity / sample_type: all chi2
        out["effect_size"] = df.get("cramers_v", np.nan)
        out["effect_size_type"] = "Cramer's V"
        out["direction"] = "association"
        out["p_value"] = df.get("p_value", np.nan)
        out["n_actionable"] = df.get("n_gene_actionable", np.nan)

    out["p_fdr"] = df.get("p_fdr", np.nan)
    out["significant_fdr"] = df.get("significant_fdr", False).astype(bool)
    out["n_non_actionable"] = df.get("n_non_actionable", np.nan)
    out["warning"] = df.get("warning", "")
    return out[OUT_COLS]


def load_all_tested(run_dir: Path) -> pd.DataFrame:
    base = run_dir / "tier7_posthoc_statistics" / "gene_stratified"
    if not base.is_dir():
        raise FileNotFoundError(f"No tier7_posthoc_statistics/gene_stratified under {run_dir}")
    frames = []
    for subdir, fname, category, implicit in FAMILIES:
        f = base / subdir / fname
        if not f.exists():
            print(f"  [warn] {category}: no {f.relative_to(run_dir)} — family absent from this run")
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        frames.append(_normalize(df, category, implicit))
    if not frames:
        raise RuntimeError(f"No gene-stratified result files found under {base}")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ONCOTREE_CODE", "Hugo_Symbol", "feature_category", "feature"]) \
              .reset_index(drop=True)


KEY = ["ONCOTREE_CODE", "Hugo_Symbol", "feature_category", "feature"]


def verify_against_published(tested: pd.DataFrame, run_dir: Path, label: str) -> None:
    """Fail closed: our significant subset must equal the run's own published hit list."""
    published = run_dir / "tier7_posthoc_statistics" / "gene_stratified" / \
        "gene_stratified_significant_hits.csv"
    if not published.exists():
        print(f"  [warn] {label}: no gene_stratified_significant_hits.csv — check skipped")
        return
    pub = pd.read_csv(published)
    ours = tested[tested["significant_fdr"]]
    pub_keys = set(map(tuple, pub[KEY].astype(str).to_numpy()))
    our_keys = set(map(tuple, ours[KEY].astype(str).to_numpy()))
    if pub_keys != our_keys:
        only_pub = sorted(pub_keys - our_keys)
        only_ours = sorted(our_keys - pub_keys)
        raise SystemExit(
            f"FAIL [{label}]: significant subset disagrees with the published hit list.\n"
            f"  published {len(pub_keys)}, reconstructed {len(our_keys)}\n"
            f"  only in published:      {only_pub[:10]}\n"
            f"  only in reconstruction: {only_ours[:10]}\n"
            "  The normalization here has drifted from summarize_gene_stratified_results.py."
        )
    print(f"  [ok]   {label}: {len(our_keys)} significant rows reproduce the published hit list")


# Lit-review curation covers exactly these categories, and only both-pathway hits in them.
LIT_CATEGORIES = ("age", "sex", "race", "ethnicity")
LIT_STATUS_COL, LIT_REFS_COL = "Lit status", "PMID or Lit review studies"
PMID_RE = re.compile(r"\b\d{7,8}\b")
URL_RE = re.compile(r"https?://\S+")


# Curated status -> the label shipped in the supplement table; a file with any other status fails closed.
LIT_STATUS_LABELS = {"confirmed": "previously reported", "unresolved": "unresolved"}

# Role words curators write beside a PMID; matched in order, so 'not supporting' is found before 'supporting'.
ROLE_WORDS = [("not supporting", "not supporting"), ("contradict", "contradicting"),
              ("indirect", "indirect"), ("supporting", "supporting")]


def _role(segment: str) -> str:
    low = segment.lower()
    for needle, tag in ROLE_WORDS:
        if needle in low:
            return tag
    return ""


def _format_refs(raw) -> str:
    """Format PMIDs as 'PMID 123; PMID 456', keeping any curator-written role.

    Bibliographic annotations are dropped; a role beside a PMID is kept, e.g.
    '37682528 (...) -- supporting' -> 'PMID 37682528 (supporting)'. A URL with no PMID is kept.
    """
    if pd.isna(raw):
        return ""
    text = str(raw)
    # A PMID's role is read from the text up to the next PMID, not a ';'-split (citations contain semicolons).
    hits = list(PMID_RE.finditer(text))
    refs, seen = [], set()
    for i, m in enumerate(hits):
        p = m.group(0)
        if p in seen:
            continue
        seen.add(p)
        role = _role(text[m.end():hits[i + 1].start() if i + 1 < len(hits) else len(text)])
        refs.append(f"PMID {p} ({role})" if role else f"PMID {p}")
    if not refs:
        refs += URL_RE.findall(text)
    return "; ".join(refs)


def load_lit_review(paths) -> pd.DataFrame:
    """Stack the curated files onto the table's key, dropping stale rollup rows (tier2_/tier3_)."""
    frames = []
    for path in paths:
        d = pd.read_csv(path)
        for col in ("ONCOTREE_CODE", "Hugo_Symbol", "feature_category",
                    LIT_STATUS_COL, LIT_REFS_COL):
            if col not in d.columns:
                raise SystemExit(f"FAIL: {path} has no {col!r} column")
        code = d["ONCOTREE_CODE"].astype(str)
        d = d[~code.str.match(r"tier[23]_")].copy()
        d["ONCOTREE_CODE"] = d["ONCOTREE_CODE"].astype(str).str.replace(r"^tier1_", "",
                                                                         regex=True)
        d["lit_status"] = d[LIT_STATUS_COL].astype(str).str.strip()
        d["lit_references"] = d[LIT_REFS_COL].map(_format_refs)
        d["lit_source"] = Path(path).name
        frames.append(d[["ONCOTREE_CODE", "Hugo_Symbol", "feature_category",
                         "lit_status", "lit_references", "lit_source"]])
    lit = pd.concat(frames, ignore_index=True)
    lk = ["ONCOTREE_CODE", "Hugo_Symbol", "feature_category"]
    dup = lit[lit.duplicated(lk, keep=False)]
    if not dup.empty:
        raise SystemExit("FAIL: the same association is curated in more than one file:\n"
                         + dup.to_string(index=False))
    return lit


def attach_lit_review(merged: pd.DataFrame, lit: pd.DataFrame) -> pd.DataFrame:
    """Add lit_status / lit_references to the both-pathway demographic rows.

    Keyed on cancer x gene x category (the curation's grain); the join must not fan out.
    """
    lk = ["ONCOTREE_CODE", "Hugo_Symbol", "feature_category"]
    target = merged[merged["feature_category"].isin(LIT_CATEGORIES)
                    & (merged["membership"] == "both")]
    if target.duplicated(lk).any():
        raise SystemExit("FAIL: more than one both-pathway row per cancer x gene x category; "
                         "the lit-review key is too coarse for this run")

    m = target[lk].merge(lit, on=lk, how="outer", indicator=True)
    stale = m[m["_merge"] == "right_only"]
    if not stale.empty:
        raise SystemExit("FAIL: curated rows that are NOT both-pathway hits in this run "
                         "(stale curation -- re-curate against this run):\n"
                         + stale[lk + ["lit_source"]].to_string(index=False))
    uncurated = m[(m["_merge"] == "left_only") | (m["lit_references"].fillna("") == "")]
    if not uncurated.empty:
        raise SystemExit("FAIL: both-pathway demographic hits with no curation or no "
                         "reference:\n" + uncurated[lk].to_string(index=False))
    bad = sorted(set(lit["lit_status"]) - set(LIT_STATUS_LABELS))
    if bad:
        raise SystemExit(f"FAIL: unexpected lit_status value(s): {bad}; expected one of "
                         f"{sorted(LIT_STATUS_LABELS)}")
    lit = lit.assign(lit_status=lit["lit_status"].map(LIT_STATUS_LABELS))

    out = merged.merge(lit.drop(columns="lit_source"), on=lk, how="left")
    out.loc[out["membership"] != "both", ["lit_status", "lit_references"]] = np.nan
    return out


def merge_pathways(rf: pd.DataFrame, lr: pd.DataFrame) -> pd.DataFrame:
    shared = ["gene_rank", "effect_size", "effect_size_type", "direction",
              "p_value", "p_fdr", "significant_fdr", "n_actionable",
              "n_non_actionable", "warning"]
    r = rf.set_index(KEY)[shared].add_prefix("rf_")
    l = lr.set_index(KEY)[shared].add_prefix("lr_")
    m = r.join(l, how="outer").reset_index()
    m["tested_rf"] = m["rf_p_fdr"].notna()
    m["tested_lr"] = m["lr_p_fdr"].notna()
    sig_rf = m["rf_significant_fdr"].fillna(False).astype(bool)
    sig_lr = m["lr_significant_fdr"].fillna(False).astype(bool)
    m["membership"] = np.select(
        [sig_rf & sig_lr, sig_rf & ~sig_lr, ~sig_rf & sig_lr],
        ["both", "RF_only", "LR_only"], default="neither")
    return m.sort_values(["ONCOTREE_CODE", "Hugo_Symbol", "feature_category", "feature"]) \
            .reset_index(drop=True)


README = """\
Full gene-stratified association table — all tested associations
================================================================

One row per cancer type x gene x feature that entered TIER 7 post-hoc testing in either
pathway, with effect size, raw p, BH-corrected q, and a significance flag for each.

Files
-----
  full_hit_table_merged.csv   RF and LR side by side (the supplement table)
  full_hit_table_rf.csv       RF only, long format
  full_hit_table_lr.csv       LR only, long format

READ THIS BEFORE COMPARING THE TWO PATHWAYS
-------------------------------------------
The RF and LR denominators are NOT comparable. The two pathways select which features
enter post-hoc testing by different rules, in kind and not merely in strictness:

  RF  gates on the top-5 GROUPED SHAP importance ranking (one row per feature, missingness
      indicators folded into their parent) for that cancer type.
  LR  gates on Wald-FDR significance of the fitted coefficient (--lr-significant-only);
      the top-N importance cap is disabled, so every Wald-significant feature is tested.

LR therefore tests far fewer associations and clears a higher FRACTION of them. That is a
property of its gate, not evidence that logistic regression is better calibrated.

The RF gate is noisy in both directions — roughly 1 hit in 5 in each direction — so the
defensible claim is that it ENRICHES FOR conditionally-important associations, NOT that a
surviving hit passed a conditional test. That stronger statement describes the LR Wald gate
only.

FDR correction
--------------
BH is applied PER FEATURE CATEGORY within each pathway (the pre-specified primary
correction). `p_fdr` is that q-value. Global re-pooling across categories and pathways is
reported separately as a sensitivity bound and is NOT a primary correction: the pathways
test overlapping hypotheses on the same patients, so BH's independence assumption is not
met, and pooling PROMOTES borderline calls as well as demoting them.

Columns
-------
  membership           both / RF_only / LR_only / neither (by FDR significance)
  tested_rf, tested_lr whether the association entered that pathway's testing at all
  *_effect_size        units vary by family; see *_effect_size_type
                         age          Cohen's d          (negative = younger)
                         sex          Odds ratio (F/M)
                         ancestry/ethnicity/sample type   Cramer's V
                         technical    Cramer's V, or effect size r for number_of_genes
  *_warning            non-empty where the chi-square expected-frequency assumption is strained
"""

README_LIT = """\
  lit_status           literature verdict for the association: "previously reported" (a published
                       study reports the same association in the same direction) or
                       "unresolved" (no consistent prior report was found -- untested, or
                       supported only indirectly). Populated ONLY for associations found
                       by both pathways in the age, sex, ancestry and ethnicity families, which
                       are the only ones that were reviewed against the literature. Blank
                       elsewhere by design: blank means not reviewed, not "no literature".
  lit_references       the studies that call rests on, as PubMed IDs, each tagged with its role
                       where the curator recorded one: (supporting), (contradicting), (indirect)
                       or (not supporting). A previously reported association can also cite a
                       contradicting study where the literature is split. For an unresolved
                       association these are the closest related studies (e.g. those reporting
                       the gene's prevalence without testing age), not a prior report.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rf-run", required=True, type=Path)
    p.add_argument("--lr-run", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the check against the published significant-hits file")
    p.add_argument("--lit-review-csv", nargs="+", type=Path, default=None,
                   help="Hand-curated lit-review CSV(s); adds lit_status + lit_references "
                        "to the merged table (both-pathway demographic hits only)")
    args = p.parse_args()

    print("Loading all tested associations")
    rf = load_all_tested(args.rf_run)
    lr = load_all_tested(args.lr_run)
    if not args.no_verify:
        verify_against_published(rf, args.rf_run, "RF")
        verify_against_published(lr, args.lr_run, "LR")

    merged = merge_pathways(rf, lr)
    readme = README
    if args.lit_review_csv:
        merged = attach_lit_review(merged, load_lit_review(args.lit_review_csv))
        readme += README_LIT
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rf.to_csv(args.output_dir / "full_hit_table_rf.csv", index=False)
    lr.to_csv(args.output_dir / "full_hit_table_lr.csv", index=False)
    merged.to_csv(args.output_dir / "full_hit_table_merged.csv", index=False)
    (args.output_dir / "full_hit_table_README.txt").write_text(readme)

    print(f"\nRF: {len(rf):4d} tested -> {int(rf['significant_fdr'].sum()):3d} significant")
    print(f"LR: {len(lr):4d} tested -> {int(lr['significant_fdr'].sum()):3d} significant")
    print(f"Merged rows (union of tested): {len(merged)}")
    print("\nBy feature category (tested / significant):")
    for cat in merged["feature_category"].unique():
        s = merged[merged.feature_category == cat]
        print(f"  {cat:12s} RF {int(s.tested_rf.sum()):3d}/"
              f"{int(s.rf_significant_fdr.fillna(False).sum()):3d}   "
              f"LR {int(s.tested_lr.sum()):3d}/"
              f"{int(s.lr_significant_fdr.fillna(False).sum()):3d}")
    print("\nMembership:")
    print(merged["membership"].value_counts().to_string())
    if args.lit_review_csv:
        lit = merged[merged["lit_status"].notna()]
        print(f"\nLiterature columns: {len(lit)} rows annotated "
              f"({lit['lit_status'].value_counts().to_dict()}); "
              f"{lit['lit_references'].str.count('PMID ').sum():.0f} PMID citations, "
              f"{lit['lit_references'].str.split('; ').explode().nunique()} distinct references")
    print(f"\nWrote -> {args.output_dir}")


if __name__ == "__main__":
    main()
