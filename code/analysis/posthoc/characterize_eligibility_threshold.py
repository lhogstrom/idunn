#!/usr/bin/env python3
"""Characterize the >=250-actionable cancer-type eligibility threshold.

The cut is applied inside the R pipeline, so downstream tables hold only the surviving
codes. This script replays the upstream filter chain against `perPatientOncoKBLevelSorted`
with the cancer-type cut omitted, to show where every OncoTree code sits relative to the
threshold (sample and patient units), which codes enter or leave at other thresholds, and
(optionally, with --rf-run/--lr-run) per-cancer test AUC with DeLong CIs for near-floor
codes, and (with --all-tests-csv + --hits-csv) whether hits survive FDR at other thresholds.
Fails closed (exit 2) if the replicated cohort does not match the documented totals.

Usage:
  python characterize_eligibility_threshold.py --output-dir <dir> \
      --rf-run runs/<rf_run> --lr-run runs/<lr_run>
"""

import argparse
import glob
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from db_config import get_default_db_path  # noqa: E402

# Documented cohort invariants (sample unit).
EXPECTED_ELIGIBLE_SAMPLES = 85419
EXPECTED_ELIGIBLE_ACTIONABLE = 27661
DEFAULT_THRESHOLD = 250
DEFAULT_CANDIDATES = [100, 150, 175, 200, 225, 250, 275, 300, 350, 400, 500]


def load_streams(db_path):
    """Replay the R filter chain per OncoTree code, cancer-type cut omitted."""
    con = sqlite3.connect(db_path)
    cov = pd.read_sql_query(
        'SELECT SourceStudy, Tumor_Sample_Barcode, medianCoverage FROM medianCoveragePerSample', con)
    gpl = pd.read_sql_query(
        'SELECT positionCtypeID, perc_of_patients_with_coverage FROM genomicPositionLookup', con)
    pps = pd.read_sql_query(
        'SELECT ONCOTREE_CODE, CANCER_TYPE_DETAILED, PATIENT_ID, Tumor_Sample_Barcode, SourceStudy, '
        'isOncoKbLevel1, ONCOKB_HAS_SENSITIVE_ENTRY, Hugo_Symbol, Start_Position '
        'FROM perPatientOncoKBLevelSorted', con)
    con.close()

    position_subset = set(gpl.loc[gpl['perc_of_patients_with_coverage'] > 90, 'positionCtypeID'])
    genie_coverage = cov.loc[cov['SourceStudy'] == 'AACR-GENIE-data', 'medianCoverage'].astype(float)
    cutoff = np.nanquantile(genie_coverage.values, 0.01)  # R quantile type 7 == numpy 'linear'
    coverage_subset = set(cov.loc[cov['medianCoverage'].notna() & (cov['medianCoverage'] >= cutoff),
                                  'Tumor_Sample_Barcode'])

    genie = pps[pps['SourceStudy'] == 'AACR-GENIE-data']

    actionable = genie[genie['isOncoKbLevel1'].astype(bool)
                       & genie['ONCOKB_HAS_SENSITIVE_ENTRY'].astype(bool)]
    actionable = actionable[actionable['Tumor_Sample_Barcode'].isin(coverage_subset)]
    position_id = (actionable['ONCOTREE_CODE'].astype(str) + '-'
                   + actionable['Hugo_Symbol'].astype(str) + '-'
                   + actionable['Start_Position'].astype('Int64').astype(str))
    actionable = actionable[position_id.isin(position_subset).values]

    comparison = genie[~genie['isOncoKbLevel1'].astype(bool)]
    comparison = comparison[comparison['Tumor_Sample_Barcode'].isin(coverage_subset)]

    return actionable, comparison, pps, cutoff


def build_table(actionable, comparison, pps, threshold):
    t = pd.DataFrame({
        'actionable_samples': actionable.groupby('ONCOTREE_CODE').size(),
        'actionable_patients': actionable.groupby('ONCOTREE_CODE')['PATIENT_ID'].nunique(),
        'comparison_samples': comparison.groupby('ONCOTREE_CODE').size(),
    }).fillna(0).astype(int)
    t['total_samples'] = t['actionable_samples'] + t['comparison_samples']
    t['actionability_rate'] = 100 * t['actionable_samples'] / t['total_samples']
    t['margin_samples'] = t['actionable_samples'] - threshold
    t['margin_patients'] = t['actionable_patients'] - threshold
    t['eligible'] = t['actionable_samples'] >= threshold
    names = pps.drop_duplicates('ONCOTREE_CODE').set_index('ONCOTREE_CODE')['CANCER_TYPE_DETAILED']
    t['ot_name'] = t.index.map(names)
    return t.sort_values('actionable_samples', ascending=False).reset_index()


def threshold_sensitivity(t, candidates, reference):
    baseline = set(t.loc[t['actionable_samples'] >= reference, 'ONCOTREE_CODE'])
    rows = []
    for c in candidates:
        by_sample = set(t.loc[t['actionable_samples'] >= c, 'ONCOTREE_CODE'])
        by_patient = set(t.loc[t['actionable_patients'] >= c, 'ONCOTREE_CODE'])
        rows.append({
            'threshold': c,
            'n_eligible_by_samples': len(by_sample),
            'n_eligible_by_patients': len(by_patient),
            'units_agree': sorted(by_sample) == sorted(by_patient),
            'enters_vs_reference': ','.join(sorted(by_sample - baseline)),
            'leaves_vs_reference': ','.join(sorted(baseline - by_sample)),
        })
    return pd.DataFrame(rows)


def fdr_invariance(all_tests_csv, hits_csv, drop_sets, out_dir, q=0.05):
    """Would moving the eligibility threshold change which hits survive BH correction?

    Raising the threshold is settled by arithmetic: surviving codes' models, ranks and gates
    are unchanged and only the FDR family shrinks (which can tighten the BH threshold, so it
    is measured). Lowering is settled only for multiplicity: this computes the largest number
    of added tests (worst case, p=1.0) a family can absorb before a hit fails BH.
    """
    d = pd.read_csv(all_tests_csv)
    hits = pd.read_csv(hits_csv)
    cat = {'age': 'age', 'sex': 'sex', 'ancestry': 'race'}
    keys = {(cat[r['feature']], r['ONCOTREE_CODE'], r['Hugo_Symbol'])
            for _, r in hits.iterrows() if r['feature'] in cat}

    def bh(p):
        p = np.asarray(p, float)
        m = len(p)
        o = np.argsort(p, kind='stable')
        ok = p[o] <= (np.arange(1, m + 1) / m) * q
        k = int(np.max(np.nonzero(ok)[0]) + 1) if ok.any() else 0
        rej = np.zeros(m, bool)
        rej[o[:k]] = True
        return rej

    # Fail closed: BH here must reproduce the stored per-family calls.
    for fam, sub in d.groupby('family'):
        if not (bh(sub['p_raw'].values) == sub['sig_per_family'].values).all():
            print(f'FAIL — BH does not reproduce sig_per_family in {fam}', file=sys.stderr)
            return None, None

    raise_rows = []
    for label, drop in sorted(drop_sets.items()):
        kept = total = 0
        lost = []
        for fam, sub in d.groupby('family'):
            sub = sub[~sub['ONCOTREE_CODE'].isin(drop)]
            if not len(sub):
                continue
            for (_, r), rej in zip(sub.iterrows(), bh(sub['p_raw'].values)):
                if (r['category'], r['ONCOTREE_CODE'], r['Hugo_Symbol']) in keys:
                    total += 1
                    kept += bool(rej)
                    if not rej:
                        lost.append(f"{r['pathway']}:{r['ONCOTREE_CODE']}·{r['Hugo_Symbol']}·{r['category']}")
        raise_rows.append({'threshold': label, 'codes_removed': ','.join(sorted(drop)),
                           'hits_remaining': len({k for k in keys if k[1] not in drop}),
                           'hit_rows_tested': total, 'hit_rows_still_significant': kept,
                           'lost': ','.join(lost)})
    raise_df = pd.DataFrame(raise_rows)

    head_rows = []
    for fam, sub in d.groupby('family'):
        p = np.sort(sub['p_raw'].values)
        m = len(p)
        j = np.arange(1, m + 1)
        # a hit at rank i survives a family of size M iff M <= max_{k>=i} (k*q / p_(k))
        bound = np.maximum.accumulate((j * q / p)[::-1])[::-1]
        for _, r in sub.iterrows():
            if (r['category'], r['ONCOTREE_CODE'], r['Hugo_Symbol']) not in keys:
                continue
            i = int(np.searchsorted(p, r['p_raw'], side='left'))
            cap = float(np.floor(bound[i]))
            head_rows.append({'family': fam, 'family_size': m,
                              'hit': f"{r['pathway']}:{r['ONCOTREE_CODE']}·{r['Hugo_Symbol']}·{r['category']}",
                              'p_raw': r['p_raw'], 'max_family_size': cap,
                              'headroom_tests': cap - m})
    head_df = pd.DataFrame(head_rows).sort_values('headroom_tests')

    raise_df.to_csv(Path(out_dir) / 'fdr_invariance_raise.csv', index=False)
    head_df.to_csv(Path(out_dir) / 'fdr_invariance_headroom.csv', index=False)
    print('\nraising the threshold (split held fixed, per-cancer models unchanged):')
    print(raise_df.to_string(index=False))
    print(f'\nlowering the threshold — multiplicity headroom over {len(head_df)} hit-rows: '
          f'min {head_df["headroom_tests"].min():.0f} extra tests, '
          f'median {head_df["headroom_tests"].median():.0f}')
    print(head_df.head(6).to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    return raise_df, head_df


def delong_by_cancer(run_dir, label):
    from auc_utils import delong_auc_ci
    hits = glob.glob(f'{run_dir}/tier5_final_models/*/test_predictions.csv')
    if not hits:
        print(f'  ! no test_predictions.csv under {run_dir}/tier5_final_models — skipping {label}')
        return None
    d = pd.read_csv(hits[0])
    rows = []
    for ct, sub in d.groupby('cancer_type'):
        auc, lo, hi = delong_auc_ci(sub['y_true'].values, sub['y_pred_proba'].values)
        rows.append({'ONCOTREE_CODE': ct, f'{label}_n_test': len(sub),
                     f'{label}_minority_test': int(min(sub['y_true'].sum(),
                                                       (1 - sub['y_true']).sum())),
                     f'{label}_auc': auc, f'{label}_ci_lo': lo, f'{label}_ci_hi': hi,
                     f'{label}_ci_includes_0.5': bool(lo <= 0.5)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--db-path', default=None,
                    help='SQLite DB (default: db_config.get_default_db_path()). Any Path B / Path C '
                         'copy gives identical answers — this reads only upstream tables.')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--threshold', type=int, default=DEFAULT_THRESHOLD)
    ap.add_argument('--candidates', type=int, nargs='+', default=DEFAULT_CANDIDATES)
    ap.add_argument('--rf-run', default=None, help='RF run dir, for DeLong CIs on per-cancer AUC')
    ap.add_argument('--lr-run', default=None, help='LR run dir, for DeLong CIs on per-cancer AUC')
    ap.add_argument('--all-tests-csv', default=None,
                    help="repool_global_fdr.py's global_fdr_all_tests.csv — enables the FDR-invariance "
                         'check (needs --hits-csv too)')
    ap.add_argument('--hits-csv', default=None,
                    help="adjust_hits_for_subject_center.py's crude_vs_adjusted_all_features.csv — "
                         'the both-pathway hit list to test invariance for')
    ap.add_argument('--skip-invariant-check', action='store_true',
                    help='Do not fail on cohort-invariant drift. Only for a deliberately different '
                         'cohort definition; never to push past an unexplained mismatch.')
    args = ap.parse_args()

    db_path = args.db_path or get_default_db_path()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f'DB: {db_path}')

    actionable, comparison, pps, cutoff = load_streams(db_path)
    print(f'GENIE median-coverage bottom-1% cutoff = {cutoff}')
    t = build_table(actionable, comparison, pps, args.threshold)

    elig = t[t['eligible']]
    n_samples = int(elig['total_samples'].sum())
    n_actionable = int(elig['actionable_samples'].sum())
    print(f'eligible codes: {len(elig)}  ({n_actionable:,} actionable / {n_samples:,} samples)')

    if not args.skip_invariant_check:
        drift = []
        if n_samples != EXPECTED_ELIGIBLE_SAMPLES:
            drift.append(f'total samples {n_samples:,} != {EXPECTED_ELIGIBLE_SAMPLES:,}')
        if n_actionable != EXPECTED_ELIGIBLE_ACTIONABLE:
            drift.append(f'actionable {n_actionable:,} != {EXPECTED_ELIGIBLE_ACTIONABLE:,}')
        if drift:
            print('FAIL — the replayed filter chain no longer reproduces the cohort:', file=sys.stderr)
            for d in drift:
                print(f'  - {d}', file=sys.stderr)
            print('Every excluded-code count below would be wrong in the same direction. '
                  'Re-check the filter chain against filter_and_structure_genie_data_for_modeling.r '
                  'before trusting any output.', file=sys.stderr)
            return 2
        print('✓ invariant check passed — replication reproduces the documented cohort exactly')

    t.to_csv(out / 'eligibility_all_codes.csv', index=False)
    sens = threshold_sensitivity(t, args.candidates, args.threshold)
    sens.to_csv(out / 'threshold_sensitivity.csv', index=False)

    excluded = t[~t['eligible']]
    print(f'\nnearest excluded codes (of {len(excluded)}):')
    print(excluded.head(5)[['ONCOTREE_CODE', 'ot_name', 'actionable_samples',
                            'actionable_patients', 'actionability_rate']].to_string(index=False))
    print('\nthreshold sensitivity:')
    print(sens.to_string(index=False))

    inc_rate = 100 * elig['actionable_samples'].sum() / elig['total_samples'].sum()
    exc_rate = 100 * excluded['actionable_samples'].sum() / excluded['total_samples'].sum()
    all_rate = 100 * t['actionable_samples'].sum() / t['total_samples'].sum()
    print(f'\nactionability — included {inc_rate:.2f}%  excluded {exc_rate:.2f}%  all GENIE {all_rate:.2f}%')

    if args.rf_run or args.lr_run:
        merged = t[t['eligible']][['ONCOTREE_CODE', 'actionable_samples', 'margin_samples',
                                   'total_samples', 'actionability_rate']]
        for run, label in [(args.rf_run, 'rf'), (args.lr_run, 'lr')]:
            if not run:
                continue
            d = delong_by_cancer(run, label)
            if d is not None:
                merged = merged.merge(d, on='ONCOTREE_CODE', how='left')
        merged = merged.sort_values('actionable_samples')
        merged.to_csv(out / 'model_discrimination_vs_margin.csv', index=False)
        print('\nper-cancer discrimination vs. distance above the threshold:')
        print(merged.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    if args.all_tests_csv and args.hits_csv:
        # thresholds that would drop the near-boundary codes
        drop_sets = {}
        for _, r in sens.iterrows():
            leaves = r['leaves_vs_reference']
            if isinstance(leaves, str) and leaves:
                drop_sets[int(r['threshold'])] = set(leaves.split(','))
        if fdr_invariance(args.all_tests_csv, args.hits_csv, drop_sets, out)[0] is None:
            return 2

    print(f'\nwrote -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
