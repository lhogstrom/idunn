"""Ungated sex enrichment of Level 1 actionability, plus its age-band modification.

Exploratory diagnostic (not part of TIER 7). Asks, across every mixed-sex cancer type,
whether Level 1 actionability is female-enriched, and whether the female/male OR differs
before vs after age 50. Strata come from the TIER 7 loaders and are tested with TIER 7's
`run_gene_stratified_sex_test`, so gated strata reproduce the published p-values
(`--check-table-s4` asserts this). BH q-values here are a separate exploratory family.

Usage:
  python s3_sex_enrichment.py --groupings-file <cancer_type_group_definitions.csv> \
      --check-table-s4 <full_hit_table.csv> --robustness --output-dir <dir>
  (BALDER_DB_PATH or --db-path must point at the patched patient-unit DB.)

Outputs (in --output-dir):
  stratum_or.csv           cancer x gene strata with >= 50 actionable patients; only mixed-sex
                           strata enter the BH family (minority sex >= 5%)
  cancer_level_or.csv      any Level 1 vs none, per mixed-sex cancer type (headline unit)
  sex_or_by_age_band.csv   sex OR at < 50 and >= 50, Woolf CIs, Breslow-Day test
  cancer_level_robustness.csv  (--robustness) OR adjusted for institution, for age + institution,
                           excluding MSK, and excluding the dominant gene
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))
from analyze_gene_stratified_associations import (  # noqa: E402
    PATIENT_TABLE,
    load_all_cancer_patients,
    load_gene_patient_data,
    run_gene_stratified_sex_test,
)
from adjust_hits_for_subject_center import (  # noqa: E402
    _safe_logit,
    _wald_or_ci,
    collapse_small_centers,
    drop_separating_levels,
    fit_sex_hit,
)

MIN_GENE_PATIENTS = 50  # TIER 7's minimum stratum size
MIN_MINORITY_SEX_SHARE = 0.05  # below this a cancer type is treated as single-sex
AGE_SPLIT = 50  # age bands: < 50 vs >= 50
MIN_BAND_CELL = 5  # 'powered': every cell of both band tables >= this
EXPOSURE_ASSOCIATED = {'LUAD', 'NSCLC', 'MEL', 'SKCM', 'BLCA'}  # smoking / UV-associated types
SEXES = ['Female', 'Male']
EXCLUDED_CENTER = 'MSK'  # excluded in the MSK-excluded sensitivity
MIN_CENTER_N = 5  # adjust_hits_for_subject_center.py's --min-center-n default
MIN_REMAINING_ACTIONABLE = 50  # eligibility / reporting floor after exclusions


def minority_sex_share(all_patients):
    s = all_patients[all_patients.SEX.isin(SEXES)].SEX
    return min((s == 'Female').mean(), (s == 'Male').mean()) if len(s) else 0.0


def actionable_genes(db_path, code):
    con = sqlite3.connect(db_path)
    try:
        genes = pd.read_sql_query(f"""
            SELECT Hugo_Symbol, COUNT(DISTINCT PATIENT_ID) AS n
            FROM {PATIENT_TABLE}
            WHERE ONCOTREE_CODE = ? AND isOncoKbLevel1 = 1
            GROUP BY Hugo_Symbol HAVING n >= ?
        """, con, params=[code, MIN_GENE_PATIENTS])
    finally:
        con.close()
    return genes.Hugo_Symbol.tolist()


def woolf(table):
    """OR and 95% CI for [[a, b], [c, d]], Haldane-corrected when any cell is zero."""
    cells = np.asarray(table, dtype=float).ravel()
    if (cells == 0).any():
        cells = cells + 0.5
    a, b, c, d = cells
    log_or = np.log(a * d / (b * c))
    se = np.sqrt((1 / cells).sum())
    return np.exp(log_or), np.exp(log_or - 1.96 * se), np.exp(log_or + 1.96 * se)


def sex_table(gene_patients, all_patients):
    """[[gene F, gene M], [non-actionable F, non-actionable M]] (TIER 7's orientation)."""
    g = gene_patients[gene_patients.SEX.isin(SEXES)].SEX
    n = all_patients[(all_patients.isOncoKbLevel1 == 0) & all_patients.SEX.isin(SEXES)].SEX
    return [[(g == 'Female').sum(), (g == 'Male').sum()],
            [(n == 'Female').sum(), (n == 'Male').sum()]]


def scan(db_path, codes):
    strata, cancers, cache = [], [], {}
    for code in codes:
        allp = load_all_cancer_patients(db_path, code)
        share = minority_sex_share(allp)
        mixed = share >= MIN_MINORITY_SEX_SHARE
        group = 'exposure_associated' if code in EXPOSURE_ASSOCIATED else 'other'

        cache[(code, None)] = allp
        if mixed:
            t = sex_table(allp[allp.isOncoKbLevel1 == 1], allp)
            or_, lo, hi = woolf(t)
            cancers.append(dict(ONCOTREE_CODE=code, exposure_group=group,
                                n_actionable=sum(t[0]), n_non_actionable=sum(t[1]),
                                female_rate_actionable=t[0][0] / sum(t[0]),
                                female_rate_non_actionable=t[1][0] / sum(t[1]),
                                odds_ratio=or_, or_ci_lower=lo, or_ci_upper=hi,
                                p_fisher=fisher_exact(t)[1]))

        for gene in actionable_genes(db_path, code):
            gp = load_gene_patient_data(db_path, code, gene)
            row = dict(ONCOTREE_CODE=code, Hugo_Symbol=gene, exposure_group=group,
                       minority_sex_share=share, mixed_sex=mixed)
            if mixed:
                res = run_gene_stratified_sex_test(gp, allp)
                row.update({k: res[k] for k in (
                    'n_gene_actionable', 'n_non_actionable', 'gene_female_rate',
                    'non_actionable_female_rate', 'odds_ratio', 'or_ci_lower', 'or_ci_upper',
                    'p_value', 'test_used', 'warning')})
                row['p_fisher'] = fisher_exact(sex_table(gp, allp))[1]
                cache[(code, gene)] = (gp, allp)
            strata.append(row)

    strata, cancers = pd.DataFrame(strata), pd.DataFrame(cancers)
    tested = strata.mixed_sex & strata.p_value.notna()
    strata['q_exploratory'] = np.nan
    strata.loc[tested, 'q_exploratory'] = multipletests(strata.loc[tested, 'p_value'],
                                                        method='fdr_bh')[1]
    cancers['q_exploratory'] = multipletests(cancers.p_fisher, method='fdr_bh')[1]
    return strata, cancers, cache


def age_bands(keys, cache):
    rows = []
    for code, gene in keys:
        gp, allp = cache[(code, gene)]
        gp, allp = gp.dropna(subset=['AGE_AT_SEQ_REPORT']), allp.dropna(subset=['AGE_AT_SEQ_REPORT'])
        tables, row = [], dict(ONCOTREE_CODE=code, Hugo_Symbol=gene)
        for label, mask_fn in (('lt50', lambda d: d.AGE_AT_SEQ_REPORT < AGE_SPLIT),
                               ('ge50', lambda d: d.AGE_AT_SEQ_REPORT >= AGE_SPLIT)):
            t = sex_table(gp[mask_fn(gp)], allp[mask_fn(allp)])
            tables.append(t)
            or_, lo, hi = woolf(t)
            row.update({f'n_gene_{label}': sum(t[0]), f'n_non_{label}': sum(t[1]),
                        f'min_cell_{label}': int(np.min(t)),
                        f'or_{label}': or_, f'or_ci_lower_{label}': lo, f'or_ci_upper_{label}': hi})
        row['powered'] = min(row['min_cell_lt50'], row['min_cell_ge50']) >= MIN_BAND_CELL
        row['delta_log_or'] = np.log(row['or_lt50']) - np.log(row['or_ge50'])
        try:
            st = StratifiedTable([np.asarray(t, dtype=float) for t in tables])
            row['p_breslow_day'] = st.test_equal_odds(adjust=False).pvalue
        except Exception as e:  # zero margins in a band
            row['p_breslow_day'], row['warning'] = np.nan, f'Breslow-Day failed: {e}'
        rows.append(row)
    return pd.DataFrame(rows)


def d_rule(bands):
    """Age-band rule, evaluated mechanically. Returns printable lines."""
    p = bands[bands.powered]
    if p.empty:
        return ['S3-D: no powered strata -> KILL (uninformative)']
    pos = int((p.delta_log_or > 0).sum())
    major = 1 if pos >= len(p) - pos else -1
    share = max(pos, len(p) - pos) / len(p)
    sign_p = binomtest(pos, len(p)).pvalue
    in_dir = p[(np.sign(p.delta_log_or) == major) & (p.p_breslow_day < 0.05)]
    lines = [f'S3-D powered strata: {len(p)} of {len(bands)}',
             f'  same direction ({"stronger <50" if major > 0 else "stronger >=50"}): '
             f'{max(pos, len(p) - pos)}/{len(p)} = {share:.0%}; sign test p = {sign_p:.3g}',
             f'  Breslow-Day p < 0.05 in that direction: {len(in_dir)} '
             f'({", ".join(in_dir.ONCOTREE_CODE + "·" + in_dir.Hugo_Symbol) or "none"})']
    go = share >= 0.70 and len(in_dir) >= 2
    if go:  # "a single stratum drives it": drop the largest |delta| and re-check
        loo = p.drop(p.delta_log_or.abs().idxmax())
        n_major = int((np.sign(loo.delta_log_or) == major).sum())
        loo_in = loo[(np.sign(loo.delta_log_or) == major) & (loo.p_breslow_day < 0.05)]
        go = n_major / len(loo) >= 0.70 and len(loo_in) >= 2
        lines.append(f'  leave-largest-out: {n_major}/{len(loo)} same direction, '
                     f'{len(loo_in)} nominal -> {"holds" if go else "FAILS"}')
    # Kill if directions are mixed (sign test p > 0.2), even when the share criterion is met.
    if sign_p > 0.2:
        verdict = 'KILL (directions mixed, sign test p > 0.2)'
    else:
        verdict = 'GO (exploratory)' if go else 'KILL'
    lines.append(f'S3-D rule: {verdict}')
    return lines


def fit_sex_age_center(cohort):
    """Outcome ~ SEX + age (per 10 y) + C(SUBJECT_CENTER), using fit_sex_hit's machinery."""
    df = cohort[cohort.SEX.isin(SEXES)].dropna(
        subset=['SEX', 'outcome', 'SUBJECT_CENTER', 'AGE_AT_SEQ_REPORT']).copy()
    df['SEX'] = pd.Categorical(df.SEX, categories=['Male', 'Female'])
    df['AGE10'] = df.AGE_AT_SEQ_REPORT / 10.0
    df, _ = collapse_small_centers(df, min_n=MIN_CENTER_N)
    formula = 'outcome ~ C(SEX) + AGE10 + C(SUBJECT_CENTER)'
    res, note = _safe_logit(formula, df), ''
    if res is None:
        df, n = drop_separating_levels(df, 'SUBJECT_CENTER')
        res, note = _safe_logit(formula, df), f'pooled {n} separating center(s)'
    if res is None:
        return dict(n=len(df), OR=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan,
                    note='fit failed')
    param = 'C(SEX)[T.Female]'
    or_, lo, hi = _wald_or_ci(res.params[param], res.bse[param])
    return dict(n=len(df), OR=or_, ci_low=lo, ci_high=hi, p=float(res.pvalues[param]), note=note)


def robustness(cancers, cache):
    """Robustness fits for every mixed-sex cancer type. One row per cancer type."""
    rows = []
    for code in cancers.ONCOTREE_CODE:
        allp = cache[(code, None)]
        cohort = allp.assign(outcome=allp.isOncoKbLevel1.astype(int))
        row = dict(ONCOTREE_CODE=code)

        n3 = fit_sex_hit(cohort, min_n=MIN_CENTER_N)
        row.update(n3_adj_or=n3['adj_OR'], n3_ci_low=n3['adj_OR_ci_low'],
                   n3_ci_high=n3['adj_OR_ci_high'], n3_p=n3['adj_p'], n3_note=n3['notes'])

        n4 = fit_sex_age_center(cohort)
        row.update(n4_n=n4['n'], n4_adj_or=n4['OR'], n4_ci_low=n4['ci_low'],
                   n4_ci_high=n4['ci_high'], n4_p=n4['p'], n4_note=n4['note'])

        rest = allp[allp.SUBJECT_CENTER != EXCLUDED_CENTER]
        t = sex_table(rest[rest.isOncoKbLevel1 == 1], rest)
        or_, lo, hi = woolf(t)
        row.update(n5_n_actionable=sum(t[0]), n5_n_non_actionable=sum(t[1]), n5_or=or_,
                   n5_ci_low=lo, n5_ci_high=hi, n5_p=fisher_exact(t)[1])

        genes = {g: v[0] for (c, g), v in cache.items() if c == code and g is not None}
        if genes:
            dom = max(genes, key=lambda g: len(genes[g]))
            act = allp[allp.isOncoKbLevel1 == 1]
            kept = act[~act.PATIENT_ID.isin(genes[dom].PATIENT_ID)]
            t = sex_table(kept, allp)
            or_, lo, hi = woolf(t) if sum(t[0]) else (np.nan,) * 3
            row.update(n6_dominant_gene=dom, n6_dominant_share=1 - len(kept) / len(act),
                       n6_n_actionable_remaining=sum(t[0]), n6_or=or_, n6_ci_low=lo,
                       n6_ci_high=hi)
        rows.append(row)
    return cancers[['ONCOTREE_CODE', 'odds_ratio', 'q_exploratory']].merge(
        pd.DataFrame(rows), on='ONCOTREE_CODE')


def robustness_rules(rob):
    """Robustness pass rules, evaluated mechanically. Returns printable lines."""
    fem = rob[(rob.odds_ratio > 1) & (rob.q_exploratory < 0.05)]
    male = rob[(rob.odds_ratio < 1) & (rob.q_exploratory < 0.05)]
    lines = [f'Robustness: {len(fem)} female-enriched types '
             f'({", ".join(fem.ONCOTREE_CODE)}); male: {", ".join(male.ONCOTREE_CODE) or "none"}']
    verdicts = {}
    for step, col_or, col_p in (('N3 institution', 'n3_adj_or', 'n3_p'),
                                ('N4 age + institution', 'n4_adj_or', 'n4_p')):
        held = fem[(fem[col_or] > 1) & (fem[col_p] < 0.05)]
        male_ok = bool((male[col_or] < 1).all())
        ok = len(held) >= len(fem) - 1 and male_ok  # ">= 8 of 9" when 9 female types
        verdicts[step] = ok
        lost = sorted(set(fem.ONCOTREE_CODE) - set(held.ONCOTREE_CODE))
        lines.append(f'  {step}: {len(held)}/{len(fem)} female types hold (lost: '
                     f'{", ".join(lost) or "none"}); male stays < 1: {male_ok} -> '
                     f'{"PASS" if ok else "FAIL"}')
    elig = fem[fem.n5_n_actionable >= MIN_REMAINING_ACTIONABLE]
    kept = elig[elig.n5_or > 1]
    ok = len(elig) > 0 and len(kept) / len(elig) >= 7 / 9
    verdicts['N5 non-MSK'] = ok
    lines.append(f'  N5 non-MSK: direction kept in {len(kept)}/{len(elig)} eligible female types '
                 f'(>= {MIN_REMAINING_ACTIONABLE} non-MSK actionable; rule >= 7/9) -> '
                 f'{"PASS" if ok else "FAIL"}; male types: '
                 + ', '.join(f'{c} {o:.2f}' for c, o in zip(male.ONCOTREE_CODE, male.n5_or)))
    n6 = rob.dropna(subset=['n6_or'])
    n6 = n6[n6.n6_n_actionable_remaining >= MIN_REMAINING_ACTIONABLE]
    lines.append('  N6 leave-dominant-gene-out (descriptive): ' + '; '.join(
        f'{r.ONCOTREE_CODE} w/o {r.n6_dominant_gene} ({r.n6_dominant_share:.0%} of actionable) '
        f'OR {r.odds_ratio:.2f} -> {r.n6_or:.2f}' for r in n6.itertuples()))
    lines.append(f'Robustness overall: {"PASS" if all(verdicts.values()) else "FAIL"} '
                 f'({", ".join(k for k, v in verdicts.items() if not v) or "all steps pass"})')
    return lines


def check_table_s4(strata, path):
    s4 = pd.read_csv(path)
    s4 = s4[s4.feature_category.str.lower().str.contains('sex')]
    merged = s4.merge(strata, on=['ONCOTREE_CODE', 'Hugo_Symbol'], how='left')
    if merged.p_value.isna().any():
        missing = merged[merged.p_value.isna()][['ONCOTREE_CODE', 'Hugo_Symbol']]
        sys.exit(f'ERROR: Table S4 sex strata missing from the ungated scan:\n{missing}')
    ref = merged.rf_p_value.fillna(merged.lr_p_value)
    if not np.allclose(ref, merged.p_value, rtol=1e-9, atol=0):
        bad = merged[~np.isclose(ref, merged.p_value, rtol=1e-9, atol=0)]
        sys.exit(f'ERROR: p-values differ from Table S4:\n'
                 f'{bad[["ONCOTREE_CODE", "Hugo_Symbol", "rf_p_value", "lr_p_value", "p_value"]]}')
    print(f'Table S4 check: all {len(merged)} tested sex strata reproduce exactly')
    return s4[s4.rf_significant_fdr.fillna(False).astype(bool)
              | s4.lr_significant_fdr.fillna(False).astype(bool)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--db-path', default=os.environ.get('BALDER_DB_PATH'),
                        help='Patched patient-unit DB (default: $BALDER_DB_PATH)')
    parser.add_argument('--groupings-file', type=Path, required=True,
                        help="The run's cancer_type_group_definitions.csv (the cohort)")
    parser.add_argument('--check-table-s4', type=Path, default=None,
                        help='Assert the gated sex strata reproduce this Table S4 CSV exactly')
    parser.add_argument('--robustness', action='store_true',
                        help='Also run S3-A robustness steps N3-N6 (cancer_level_robustness.csv)')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()

    # Fail closed: an unset path would silently resolve to the unpatched DB.
    if not args.db_path or not Path(args.db_path).exists():
        sys.exit('ERROR: pass --db-path or set BALDER_DB_PATH to the patched patient-unit DB')
    if not args.groupings_file.exists():
        sys.exit(f'ERROR: groupings file not found: {args.groupings_file}')

    codes = pd.read_csv(args.groupings_file).ONCOTREE_CODE.tolist()
    strata, cancers, cache = scan(args.db_path, codes)

    t = strata[strata.q_exploratory.notna()]
    n_f = int(((t.q_exploratory < 0.05) & (t.odds_ratio > 1)).sum())
    n_m = int(((t.q_exploratory < 0.05) & (t.odds_ratio < 1)).sum())
    print(f'Cohort: {len(codes)} cancer types; {len(cancers)} mixed-sex')
    print(f'Cancer level (headline): OR > 1 in {(cancers.odds_ratio > 1).sum()}/{len(cancers)}; '
          f'q < 0.05 female {((cancers.q_exploratory < .05) & (cancers.odds_ratio > 1)).sum()} '
          f'/ male {((cancers.q_exploratory < .05) & (cancers.odds_ratio < 1)).sum()}')
    for g, sub in cancers.groupby('exposure_group'):
        print(f'  {g:20s} OR > 1 in {(sub.odds_ratio > 1).sum()}/{len(sub)}: '
              + ', '.join(f'{c} {o:.2f}' for c, o in zip(sub.ONCOTREE_CODE, sub.odds_ratio)))
    print(f'Strata: {len(strata)} with >= {MIN_GENE_PATIENTS} actionable; '
          f'{(~strata.mixed_sex).sum()} sex-skewed (not tested); {len(t)} mixed-sex tested; '
          f'OR > 1 in {(t.odds_ratio > 1).sum()}/{len(t)} (descriptive, correlated within cancer)')
    print(f'  q_exploratory < 0.05: {n_f} female-enriched, {n_m} male-enriched')

    d_keys = set(map(tuple, t[t.q_exploratory < 0.05][['ONCOTREE_CODE', 'Hugo_Symbol']].values))
    if args.check_table_s4:
        sig = check_table_s4(strata, args.check_table_s4)
        d_keys |= set(map(tuple, sig[['ONCOTREE_CODE', 'Hugo_Symbol']].values))
    bands = age_bands(sorted(d_keys), cache)
    print('\n'.join(d_rule(bands)))

    rob = None
    if args.robustness:
        rob = robustness(cancers, cache)
        print('\n'.join(robustness_rules(rob)))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        strata.to_csv(args.output_dir / 'stratum_or.csv', index=False)
        cancers.to_csv(args.output_dir / 'cancer_level_or.csv', index=False)
        bands.to_csv(args.output_dir / 'sex_or_by_age_band.csv', index=False)
        if rob is not None:
            rob.to_csv(args.output_dir / 'cancer_level_robustness.csv', index=False)
        print(f'wrote {3 + (rob is not None)} CSVs to {args.output_dir}')


if __name__ == '__main__':
    main()
