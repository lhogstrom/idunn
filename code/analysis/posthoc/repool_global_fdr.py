#!/usr/bin/env python3
"""Global FDR re-pooling for TIER 7 post-hoc tests.

TIER 7 applies Benjamini-Hochberg within each results file (pathway x analysis level x
feature category). This re-applies BH at wider scopes from the stored raw p-values, as a
sensitivity analysis; no model is refit and no TIER 7 output is modified.

Scopes: per_family (status quo, recomputed as a self-check); per_pathway_global (one BH per
pathway); pooled_rf_lr (RF and LR together; a sensitivity bound only, since the pathways test
overlapping hypotheses on the same patients and BH's independence/PRDS assumption is not met).

Usage:
  python repool_global_fdr.py --rf-run runs/<rf_run> --lr-run runs/<lr_run> --output-dir <dir>

Outputs: global_fdr_all_tests.csv, global_fdr_summary.csv, global_fdr_changed_calls.csv,
global_fdr_report.md
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALPHA = 0.05

# Family definitions. `pcol` differs across files (gene-level age uses `mann_whitney_p`);
# `id_cols` make a test uniquely addressable within its family (e.g. `feature` for technical_by_gene).
FAMILIES = [
    # gene-stratified
    dict(level='gene_stratified', category='age',
         rel='tier7_posthoc_statistics/gene_stratified/age_by_gene/age_gene_results.csv',
         pcol='mann_whitney_p', id_cols=['ONCOTREE_CODE', 'Hugo_Symbol']),
    dict(level='gene_stratified', category='ethnicity',
         rel='tier7_posthoc_statistics/gene_stratified/ethnicity_by_gene/ethnicity_gene_results.csv',
         pcol='p_value', id_cols=['ONCOTREE_CODE', 'Hugo_Symbol', 'feature']),
    dict(level='gene_stratified', category='race',
         rel='tier7_posthoc_statistics/gene_stratified/race_by_gene/race_gene_results.csv',
         pcol='p_value', id_cols=['ONCOTREE_CODE', 'Hugo_Symbol', 'feature']),
    dict(level='gene_stratified', category='sample_type',
         rel='tier7_posthoc_statistics/gene_stratified/sample_type_by_gene/sample_type_gene_results.csv',
         pcol='p_value', id_cols=['ONCOTREE_CODE', 'Hugo_Symbol', 'feature']),
    dict(level='gene_stratified', category='sex',
         rel='tier7_posthoc_statistics/gene_stratified/sex_by_gene/sex_gene_results.csv',
         pcol='p_value', id_cols=['ONCOTREE_CODE', 'Hugo_Symbol']),
    dict(level='gene_stratified', category='technical',
         rel='tier7_posthoc_statistics/gene_stratified/technical_by_gene/technical_gene_results.csv',
         pcol='p_value', id_cols=['ONCOTREE_CODE', 'Hugo_Symbol', 'feature']),
    # cancer-level
    dict(level='cancer_level', category='race_ethnicity',
         rel='tier7_posthoc_statistics/shap_associations/race_ethnicity_associations.csv',
         pcol='p_value', id_cols=['ONCOTREE_CODE', 'feature', 'column']),
    dict(level='cancer_level', category='sex',
         rel='tier7_posthoc_statistics/shap_associations/sex_associations.csv',
         pcol='p_value', id_cols=['ONCOTREE_CODE', 'feature']),
    dict(level='cancer_level', category='sample_type',
         rel='tier7_posthoc_statistics/shap_associations/sample_type_associations.csv',
         pcol='p_value', id_cols=['ONCOTREE_CODE', 'feature', 'column']),
    dict(level='cancer_level', category='technical',
         rel='tier7_posthoc_statistics/shap_associations/technical_feature_associations.csv',
         pcol='p_value', id_cols=['ONCOTREE_CODE', 'feature', 'column']),
]

# Cancer-level age carries two p-columns for the same hypothesis (Mann-Whitney and t-test);
# only the non-parametric one is pooled, matching the gene-level age choice.
AGE_ACTIONABILITY = dict(
    level='cancer_level', category='age',
    rel='tier7_posthoc_statistics/age_actionability/age_actionability_results.csv',
    pcol='mann_whitney_p', id_cols=['ONCOTREE_CODE'], stored_fdr_col='mann_whitney_p_fdr',
)


def load_family(run_dir: Path, pathway: str, fam: dict) -> pd.DataFrame:
    """Read one family's results file into long form; an absent file gives an empty frame (e.g. LR has no race_ethnicity file)."""
    path = run_dir / fam['rel']
    if not path.exists():
        logger.warning(f"  [{pathway}] absent, skipping: {fam['level']}/{fam['category']}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    pcol = fam['pcol']
    if pcol not in df.columns:
        raise ValueError(
            f"{path} lacks expected p-value column '{pcol}'. "
            f"Columns present: {list(df.columns)}. Family definitions in "
            f"repool_global_fdr.py are stale relative to this run."
        )

    stored_fdr_col = fam.get('stored_fdr_col', 'p_fdr')
    id_cols = [c for c in fam['id_cols'] if c in df.columns]
    missing_ids = set(fam['id_cols']) - set(id_cols)
    if missing_ids:
        logger.warning(f"  [{pathway}] {fam['category']}: id columns absent {sorted(missing_ids)}")

    out = pd.DataFrame({
        'pathway': pathway,
        'level': fam['level'],
        'category': fam['category'],
        'test_key': df[id_cols].astype(str).agg('|'.join, axis=1) if id_cols else df.index.astype(str),
        'p_raw': df[pcol],
        'p_fdr_stored': df[stored_fdr_col] if stored_fdr_col in df.columns else np.nan,
        'source_file': str(fam['rel']),
    })
    for c in id_cols:
        out[c] = df[c]

    n_drop = int(out['p_raw'].isna().sum())
    if n_drop:
        logger.info(f"  [{pathway}] {fam['level']}/{fam['category']}: dropping {n_drop} rows with null p")
    out = out[out['p_raw'].notna()].copy()
    logger.info(f"  [{pathway}] {fam['level']:15s} {fam['category']:15s} n={len(out)}")
    return out


def bh(pvals: pd.Series) -> np.ndarray:
    """BH-adjusted q-values; empty-safe."""
    if len(pvals) == 0:
        return np.array([])
    return multipletests(pvals.values, alpha=ALPHA, method='fdr_bh')[1]


def main():
    ap = argparse.ArgumentParser(
        description='Re-apply BH correction to TIER 7 tests at wider scopes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE ON `pooled_rf_lr`
    RF and LR test overlapping hypotheses on the same patients. Their p-values
    are therefore strongly positively dependent, and BH controls FDR under
    independence or PRDS -- an assumption this pooling does not credibly meet.
    Report it as a conservative upper bound on the hit count, never as the
    primary correction.
        """)
    ap.add_argument('--rf-run', type=str, required=True, help='RF run directory')
    ap.add_argument('--lr-run', type=str, default=None, help='LR run directory (optional)')
    ap.add_argument('--label', type=str, default='unlabeled',
                    help="Arm label recorded in outputs (e.g. L_leaky, S_sampleunit, P_patientunit)")
    ap.add_argument('--output-dir', type=str, required=True, help='Directory for outputs')
    ap.add_argument('--alpha', type=float, default=ALPHA, help='FDR threshold (default 0.05)')
    ap.add_argument('--no-cancer-level-age', action='store_true',
                    help='Exclude age_actionability from the pool (default: include, Mann-Whitney only)')
    args = ap.parse_args()

    rf_run = Path(args.rf_run)
    lr_run = Path(args.lr_run) if args.lr_run else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rf_run.exists():
        logger.error(f"RF run directory not found: {rf_run}")
        sys.exit(1)
    if lr_run is not None and not lr_run.exists():
        logger.error(f"LR run directory not found: {lr_run}")
        sys.exit(1)

    families = list(FAMILIES)
    if not args.no_cancer_level_age:
        families.append(AGE_ACTIONABILITY)

    # collect
    frames = []
    for pathway, run_dir in [('RF', rf_run), ('LR', lr_run)]:
        if run_dir is None:
            continue
        logger.info(f"Loading {pathway} from {run_dir}")
        for fam in families:
            frames.append(load_family(run_dir, pathway, fam))

    tests = pd.concat([f for f in frames if len(f)], ignore_index=True)
    if tests.empty:
        logger.error("No tests loaded -- check run directories.")
        sys.exit(1)

    tests['family'] = tests['pathway'] + ':' + tests['level'] + ':' + tests['category']
    logger.info(f"Loaded {len(tests)} tests across {tests['family'].nunique()} families")

    # scope 1: per family (self-check)
    tests['q_per_family'] = np.nan
    for fam_key, idx in tests.groupby('family').groups.items():
        tests.loc[idx, 'q_per_family'] = bh(tests.loc[idx, 'p_raw'])

    has_stored = tests['p_fdr_stored'].notna()
    if has_stored.any():
        drift = ~np.isclose(tests.loc[has_stored, 'q_per_family'],
                            tests.loc[has_stored, 'p_fdr_stored'], rtol=1e-6, atol=1e-9)
        n_drift = int(drift.sum())
        if n_drift:
            logger.error(
                f"SELF-CHECK FAILED: {n_drift} tests where recomputed per-family BH != stored p_fdr. "
                f"The family definitions no longer describe what TIER 7 did; fix before trusting "
                f"the pooled scopes."
            )
            tests.loc[has_stored][drift].to_csv(out_dir / 'global_fdr_SELFCHECK_FAILURES.csv', index=False)
            sys.exit(2)
        logger.info(f"Self-check PASSED: per-family BH reproduces stored p_fdr for all "
                    f"{int(has_stored.sum())} tests with a stored value")

    # scope 2: per pathway, global
    tests['q_pathway_global'] = np.nan
    for pathway, idx in tests.groupby('pathway').groups.items():
        tests.loc[idx, 'q_pathway_global'] = bh(tests.loc[idx, 'p_raw'])

    # scope 3: fully pooled (sensitivity bound only)
    tests['q_pooled_rf_lr'] = bh(tests['p_raw'])

    scopes = ['q_per_family', 'q_pathway_global', 'q_pooled_rf_lr']
    for s in scopes:
        tests[s.replace('q_', 'sig_')] = tests[s] < args.alpha

    tests.insert(0, 'arm_label', args.label)

    # outputs
    all_path = out_dir / 'global_fdr_all_tests.csv'
    tests.to_csv(all_path, index=False)

    summary = (tests.groupby(['pathway', 'level', 'category'])
               .agg(n_tests=('p_raw', 'size'),
                    sig_per_family=('sig_per_family', 'sum'),
                    sig_pathway_global=('sig_pathway_global', 'sum'),
                    sig_pooled_rf_lr=('sig_pooled_rf_lr', 'sum'))
               .reset_index())
    summary.insert(0, 'arm_label', args.label)
    summary.to_csv(out_dir / 'global_fdr_summary.csv', index=False)

    changed = tests[tests['sig_per_family'] != tests['sig_pathway_global']].copy()
    changed['direction'] = np.where(changed['sig_per_family'], 'lost_under_global', 'gained_under_global')
    changed.to_csv(out_dir / 'global_fdr_changed_calls.csv', index=False)

    # report
    tot = len(tests)
    lines = [
        f"# Global FDR re-pooling — arm `{args.label}`",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} · alpha = {args.alpha}",
        "",
        f"- RF run: `{rf_run}`",
        f"- LR run: `{lr_run}`" if lr_run else "- LR run: _(none)_",
        f"- Cancer-level age included: {'no' if args.no_cancer_level_age else 'yes (Mann-Whitney only)'}",
        "",
        "## Totals",
        "",
        "| Scope | Corrections applied | Significant | of |",
        "|---|---|---|---|",
        f"| Per family (status quo) | {tests['family'].nunique()} | "
        f"{int(tests['sig_per_family'].sum())} | {tot} |",
        f"| Per pathway, global | {tests['pathway'].nunique()} | "
        f"{int(tests['sig_pathway_global'].sum())} | {tot} |",
        f"| Pooled RF+LR (upper bound) | 1 | {int(tests['sig_pooled_rf_lr'].sum())} | {tot} |",
        "",
        f"Tests changing call between per-family and per-pathway-global: "
        f"**{len(changed)}** "
        f"({int((changed['direction'] == 'lost_under_global').sum())} lost, "
        f"{int((changed['direction'] == 'gained_under_global').sum())} gained).",
        "",
        "## Per family",
        "",
        summary.drop(columns=['arm_label']).to_markdown(index=False),
        "",
        "## Caveat on the pooled column",
        "",
        "`pooled_rf_lr` corrects RF and LR together. The two pathways test overlapping",
        "hypotheses on the same patients, so their p-values are strongly dependent and BH's",
        "independence/PRDS assumption is not credibly met. Treat that column as a",
        "conservative upper bound on the multiple-comparison burden, not as a primary result.",
        "",
    ]
    (out_dir / 'global_fdr_report.md').write_text('\n'.join(lines))

    print(f"\n{'='*64}")
    print(f"GLOBAL FDR RE-POOLING — arm {args.label}")
    print(f"{'='*64}")
    print(f"Tests pooled:              {tot}")
    print(f"Families (status quo):     {tests['family'].nunique()}")
    print(f"Significant, per family:   {int(tests['sig_per_family'].sum())}")
    print(f"Significant, pathway-wide: {int(tests['sig_pathway_global'].sum())}")
    print(f"Significant, pooled RF+LR: {int(tests['sig_pooled_rf_lr'].sum())}  (upper bound only)")
    print(f"Calls changed:             {len(changed)}")
    print(f"\nOutputs -> {out_dir}")
    print(f"{'='*64}\n")


if __name__ == '__main__':
    main()
